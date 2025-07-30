#!/usr/bin/env python3
"""
Dataset Serialization Script for CasualConversations - Keyword-Based

This script processes the CasualConversations dataset by:
1. Reading the keyword search results from out.json
2. Finding available videos that match the keyword criteria
3. Extracting face sequences at keyword timestamps
4. Creating clips based on expression timestamps from transcriptions
"""

import json
import os
import random
import glob
from typing import List, Dict, Tuple
from mp4_utils import extract_face_sequence, get_video_info
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_keyword_results(json_path: str) -> Dict:
    """
    Load the keyword search results from out.json.
    
    Args:
        json_path (str): Path to the keyword results JSON file
    
    Returns:
        Dict: Loaded keyword results data
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Keyword results file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        keyword_results = json.load(f)
    
    logger.info(f"Loaded keyword results with {keyword_results.get('total_videos', 0)} videos")
    return keyword_results


def find_video_path(video_id: str, base_path: str, dataset_folder: str) -> str:
    """
    Find the full path to a video file.
    
    Args:
        video_id (str): Video ID from keyword results (e.g., "CasualConversationsA/1226/1226_14.MP4")
        base_path (str): Base path to the dataset
        dataset_folder (str): Dataset folder name
    
    Returns:
        str: Full path to the video file, or None if not found
    """
    # Construct the full path
    full_path = os.path.join(base_path, video_id)
    
    if os.path.exists(full_path):
        logger.info(f"Found video: {full_path}")
        return full_path
    else:
        logger.warning(f"Video not found: {full_path}")
        return None


def extract_keyword_timestamps(video_data: Dict, keywords: List[str]) -> List[float]:
    """
    Extract all keyword timestamps from video data.
    
    Args:
        video_data (Dict): Video data from keyword results
        keywords (List[str]): List of keywords to extract timestamps for
    
    Returns:
        List[float]: List of all keyword timestamps
    """
    timestamps = []
    
    for keyword in keywords:
        if keyword in video_data['keywords']:
            keyword_timestamps = video_data['keywords'][keyword]
            if keyword_timestamps:  # Check if not empty
                timestamps.extend(keyword_timestamps)
    
    # Remove duplicates and sort
    unique_timestamps = sorted(list(set(timestamps)))
    logger.info(f"Extracted {len(unique_timestamps)} unique timestamps from {len(timestamps)} keyword occurrences")
    
    return unique_timestamps


def process_video_clips(video_path: str, timestamps: List[float], output_base: str, subject_id: str = None, subject_label: Dict = None) -> List[Dict]:
    """
    Process video clips by extracting face sequences at given timestamps.
    
    Args:
        video_path (str): Path to the video file
        timestamps (List[float]): List of timestamps to extract
        output_base (str): Base output directory
        subject_id (str): Subject ID from JSON annotations
        subject_label (Dict): Subject label information (age, gender, skin-type)
    
    Returns:
        List[Dict]: List of extraction results
    """
    results = []
    
    for i, timestamp in enumerate(timestamps):
        try:
            logger.info(f"Processing clip {i+1}/{len(timestamps)} at {timestamp:.2f}s")
            
            # Extract face sequence
            result = extract_face_sequence(
                video_path=video_path,
                timestamp=timestamp,
                output_folder=output_base,  # Use the same base folder for all files
                face_size=(518, 518),
                num_frames=30,  # Extract 30 frames (one per frame for 30fps)
                frame_skip=0,  # No frame skipping for better quality
                confidence_threshold=0.5,
                subject_id=subject_id,
                subject_label=subject_label
            )
            
            # Check if extraction was successful (not None)
            if result is not None:
                result['clip_index'] = i
                result['timestamp'] = timestamp
                results.append(result)
            else:
                logger.warning(f"Skipped clip {i} at {timestamp}s due to insufficient faces")
            
        except Exception as e:
            logger.error(f"Error processing clip {i} at {timestamp}s: {str(e)}")
    
    return results


def serialize_dataset_from_keywords(
    keyword_results_path: str,
    dataset_folder: str,
    base_path: str,
    output_base: str = "serialized_dataset_keywords",
    keywords: List[str] = None
) -> Dict:
    """
    Serialize the CasualConversations dataset using keyword timestamps.
    
    Args:
        keyword_results_path (str): Path to the keyword results JSON file (out.json)
        dataset_folder (str): Dataset folder name (e.g., 'CasualConversationsA')
        base_path (str): Base path to the dataset
        output_base (str): Base output directory for serialized data
        keywords (List[str]): List of keywords to extract timestamps for
    
    Returns:
        Dict: Summary of the serialization process
    """
    logger.info(f"Starting keyword-based dataset serialization for {dataset_folder}")
    
    # Create output directory
    os.makedirs(output_base, exist_ok=True)
    
    # Load keyword results
    keyword_results = load_keyword_results(keyword_results_path)
    
    # Use default keywords if not specified
    if keywords is None:
        keywords = ['smile', 'natural', 'neutral', 'sad', 'surprised', 'expressions', 'happy', 'angry']
    
    logger.info(f"Using keywords: {keywords}")
    
    # Get videos from keyword results
    videos = keyword_results.get('videos', [])
    logger.info(f"Found {len(videos)} videos with keyword matches")
    
    if not videos:
        logger.error(f"No videos found in keyword results")
        return {'error': 'No videos found in keyword results'}
    
    # Process each video
    total_clips = 0
    successful_videos = 0
    failed_videos = 0
    subject_stats = {}  # Track statistics per subject
    
    for video_data in videos:
        video_path = video_data['video_path']
        subject_id = video_data['subject_id']
        
        logger.info(f"Processing video: {video_path}")
        logger.info(f"Subject ID: {subject_id}")
        
        # Create subject label from video data
        subject_label = {
            'age': video_data.get('age', 'unknown'),
            'gender': video_data.get('gender', 'unknown'),
            'skin-type': video_data.get('skin_type', 'unknown')
        }
        
        # Find the actual video file
        full_video_path = find_video_path(video_path, base_path, dataset_folder)
        
        if not full_video_path:
            logger.warning(f"Video file not found: {video_path}")
            failed_videos += 1
            continue
        
        # Extract keyword timestamps
        timestamps = extract_keyword_timestamps(video_data, keywords)
        
        if not timestamps:
            logger.warning(f"No keyword timestamps found for {video_path}")
            failed_videos += 1
            continue
        
        try:
            # Process clips at keyword timestamps
            clip_results = process_video_clips(full_video_path, timestamps, output_base, subject_id, subject_label)
            
            total_clips += len(clip_results)
            successful_videos += 1
            
            # Track subject statistics
            if subject_id not in subject_stats:
                subject_stats[subject_id] = {
                    'subject_id': subject_id,
                    'label': subject_label,
                    'total_clips': 0,
                    'video_files': [],
                    'clip_files': [],
                    'keyword_timestamps': {}
                }
            
            subject_stats[subject_id]['total_clips'] += len(clip_results)
            subject_stats[subject_id]['video_files'].append(full_video_path)
            
            # Store keyword timestamps for this video
            for keyword in keywords:
                if keyword in video_data['keywords']:
                    subject_stats[subject_id]['keyword_timestamps'][keyword] = video_data['keywords'][keyword]
            
            # Add clip file information
            for result in clip_results:
                if result is not None:
                    subject_stats[subject_id]['clip_files'].append({
                        'hdf5_file': result['hdf5_path'],
                        'mp4_file': result['mp4_path'],
                        'timestamp': result['timestamp'],
                        'num_frames': result['num_frames'],
                        'total_faces': result['total_faces']
                    })
            
            logger.info(f"Successfully processed {len(clip_results)} clips for {video_path}")
            
        except Exception as e:
            logger.error(f"Error processing {video_path}: {str(e)}")
            failed_videos += 1
    
    # Create detailed summary with subject information
    summary = {
        'dataset_folder': dataset_folder,
        'keyword_results_file': keyword_results_path,
        'keywords_used': keywords,
        'total_videos_processed': successful_videos,
        'failed_videos': failed_videos,
        'total_clips_extracted': total_clips,
        'output_directory': output_base,
        'dataset_stats': {
            'num_subjects': len(subject_stats),
            'subjects': subject_stats
        }
    }
    
    logger.info("=== Keyword-Based Serialization Summary ===")
    logger.info(f"Dataset: {dataset_folder}")
    logger.info(f"Keyword results: {keyword_results_path}")
    logger.info(f"Keywords used: {keywords}")
    logger.info(f"Successful videos: {successful_videos}")
    logger.info(f"Failed videos: {failed_videos}")
    logger.info(f"Total clips extracted: {total_clips}")
    logger.info(f"Output directory: {output_base}")
    
    # Log subject statistics
    logger.info(f"Dataset contains {len(subject_stats)} subjects:")
    for subject_id, stats in subject_stats.items():
        logger.info(f"  Subject {subject_id}: {stats['total_clips']} clips")
        if stats['label']:
            logger.info(f"    Age: {stats['label'].get('age', 'N/A')}, Gender: {stats['label'].get('gender', 'N/A')}")
    
    # Save metadata JSON file
    metadata_file = os.path.join(output_base, "dataset_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Dataset metadata saved to: {metadata_file}")
    
    return summary


def main():
    """Main function to run the keyword-based dataset serialization."""
    
    import sys
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Serialize dataset using keyword timestamps from out.json')
    parser.add_argument('--keyword_results', type=str, 
                       default='out.json',
                       help='Path to keyword results JSON file (default: out.json)')
    parser.add_argument('--dataset_folder', type=str, 
                       default='CasualConversationsA',
                       help='Dataset folder name (default: CasualConversationsA)')
    parser.add_argument('--output_base', type=str, 
                       default='/Users/ozgewhiting/Documents/EQLabs/datasets_serial/CCA_train_db3_expr',
                       help='Output directory (default: serialized_dataset_keywords)')
    parser.add_argument('--base_path', type=str, 
                       default='/Users/ozgewhiting/Documents/EQLabs/datasets/CasualConversations',
                       help='Base path to dataset (default: /Users/ozgewhiting/Documents/EQLabs/datasets/CasualConversations)')
    parser.add_argument('--keywords', nargs='+', 
                       default=['smile', 'natural', 'neutral', 'sad', 'surprised', 'expressions', 'happy', 'angry'],
                       help='Keywords to extract timestamps for')
    
    args = parser.parse_args()
    
    print("=== Keyword-Based CasualConversations Dataset Serialization ===")
    print(f"Keyword results: {args.keyword_results}")
    print(f"Base path: {args.base_path}")
    print(f"Dataset folder: {args.dataset_folder}")
    print(f"Output directory: {args.output_base}")
    print(f"Keywords: {args.keywords}")
    print()
    
    try:
        # Run serialization
        summary = serialize_dataset_from_keywords(
            keyword_results_path=args.keyword_results,
            dataset_folder=args.dataset_folder,
            base_path=args.base_path,
            output_base=args.output_base,
            keywords=args.keywords
        )
        
        print("=== Final Summary ===")
        for key, value in summary.items():
            if key != 'dataset_stats':  # Skip detailed stats in console output
                print(f"{key}: {value}")
        
    except Exception as e:
        logger.error(f"Serialization failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 