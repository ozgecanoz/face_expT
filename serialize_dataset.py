#!/usr/bin/env python3
"""
Dataset Serialization Script for CasualConversations

This script processes the CasualConversations dataset by:
1. Reading the JSON annotations file
2. Finding available videos (excluding dark_files)
3. Sampling 5 random clips per video
4. Extracting face sequences for training
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


def load_annotations(json_path: str) -> Dict:
    """
    Load the CasualConversations annotations JSON file.
    
    Args:
        json_path (str): Path to the annotations JSON file
    
    Returns:
        Dict: Loaded JSON data
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Annotations file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        annotations = json.load(f)
    
    logger.info(f"Loaded annotations with {len(annotations)} entries")
    return annotations


def find_available_videos(annotations: Dict, dataset_folder: str, base_path: str) -> List[Dict]:
    """
    Find available videos that are not in dark_files.
    
    Args:
        annotations (Dict): Loaded annotations data
        dataset_folder (str): Dataset folder name (e.g., 'CasualConversationsA')
        base_path (str): Base path to the dataset
    
    Returns:
        List[Dict]: List of available video information
    """
    available_videos = []
    
    for subject_id, video_data in annotations.items():
        # check each video file in video_data.get('files')
        for video_file in video_data.get('files', []):
            # create the full path to the video file
            full_path = os.path.join(base_path, video_file)
            if os.path.exists(full_path):
                # exclude the video if it is in dark_files:
                dark_files = video_data.get('dark_files', [])
                if video_file not in dark_files:
                    available_videos.append({
                        'video_id': video_file,
                        'data': video_data
                    })
                else:
                    logger.info(f"Excluding dark file: {video_file}")
            else:
                logger.info(f"Video file not found: {full_path}")
    
    logger.info(f"Found {len(available_videos)} available videos in {dataset_folder}")
    return available_videos


def find_video_files(video_data: Dict, base_path: str) -> List[str]:
    """
    Find actual video files using the files segment from JSON data.
    
    Args:
        video_data (Dict): Video data from JSON annotations
        base_path (str): Base path to the dataset
    
    Returns:
        List[str]: List of found video file paths
    """
    # VS Code breakpoint - click in the left margin next to this line
    video_files = []
    
    # Debug: Print video data structure
    logger.info(f"Video data keys: {list(video_data.keys())}")
    
    # Get the files list from video data
    files = video_data.get('files', [])
    logger.info(f"Files found in JSON: {files}")
    
    for relative_path in files:
        # Construct full path
        full_path = os.path.join(base_path, relative_path)
        logger.info(f"Checking file: {full_path}")
        
        # Check if file exists
        if os.path.exists(full_path):
            logger.info(f"✓ File exists: {full_path}")
            video_files.append(full_path)
        else:
            logger.warning(f"✗ File not found: {full_path}")
    
    logger.info(f"Total video files found: {len(video_files)}")
    return sorted(video_files)


def sample_random_clips(video_path: str, num_clips: int = 5, min_duration: float = 5.0) -> List[float]:
    """
    Sample random timestamps for clips from a video.
    
    Args:
        video_path (str): Path to the video file
        num_clips (int): Number of clips to sample
        min_duration (float): Minimum duration in seconds to ensure enough video length
    
    Returns:
        List[float]: List of random timestamps
    """
    try:
        # Get video information
        video_info = get_video_info(video_path)
        duration = video_info['duration_seconds']
        
        if duration < min_duration:
            logger.warning(f"Video {video_path} too short ({duration}s), skipping")
            return []
        
        # Calculate safe range for sampling (leave some buffer)
        safe_duration = duration - min_duration
        if safe_duration <= 0:
            return []
        
        # Sample random timestamps
        timestamps = []
        for _ in range(num_clips):
            timestamp = random.uniform(0, safe_duration)
            timestamps.append(timestamp)
        
        return sorted(timestamps)
        
    except Exception as e:
        logger.error(f"Error sampling clips from {video_path}: {str(e)}")
        return []


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


def serialize_dataset(
    json_path: str,
    dataset_folder: str,
    base_path: str,
    output_base: str = "serialized_dataset",
    clips_per_video: int = 5
) -> Dict:
    """
    Serialize the CasualConversations dataset.
    
    Args:
        json_path (str): Path to the annotations JSON file
        dataset_folder (str): Dataset folder name (e.g., 'CasualConversationsA')
        output_base (str): Base output directory for serialized data
        clips_per_video (int): Number of clips to sample per video
    
    Returns:
        Dict: Summary of the serialization process
    """
    logger.info(f"Starting dataset serialization for {dataset_folder}")
    
    # Create output directory
    os.makedirs(output_base, exist_ok=True)
    
    # Load annotations
    annotations = load_annotations(json_path)
    
    # Find available videos
    available_videos = find_available_videos(annotations, dataset_folder, base_path)
    logger.info(f"Available videos found: {len(available_videos)}")
    
    if not available_videos:
        logger.error(f"No available videos found in {dataset_folder}")
        return {'error': 'No available videos found'}
    
    # Process each video
    total_clips = 0
    successful_videos = 0
    failed_videos = 0
    subject_stats = {}  # Track statistics per subject
    
    for video_info in available_videos:
        video_id = video_info['video_id']
        video_data = video_info['data']
        logger.info(f"Processing video: {video_id}")
        logger.info(f"Video data: {video_data}")
        
        # Extract subject information from video_id
        # video_id format: "CasualConversationsA/1140/1140_02.MP4"
        # We need to find the subject_id from annotations
        subject_id = None
        subject_label = None
        
        # Find the subject_id by matching the video_id pattern
        for subj_id, subj_data in annotations.items():
            if 'files' in subj_data and video_id in subj_data['files']:
                subject_id = subj_id
                subject_label = subj_data.get('label', {})
                logger.info(f"Found subject_id: {subject_id}, label: {subject_label}")
                break
        
        if subject_id is None:
            logger.warning(f"Could not find subject_id for video: {video_id}")
        
        # Find video files using the files segment from JSON
        video_files = find_video_files(video_data, base_path)
        
        if not video_files:
            logger.warning(f"No video files found for {video_id}")
            failed_videos += 1
            continue
        
        # Process each video file
        for video_path in video_files:
            try:
                # Sample random clips
                timestamps = sample_random_clips(video_path, clips_per_video)
                
                if not timestamps:
                    logger.warning(f"No valid clips sampled for {video_path}")
                    continue
                
                # Process clips - all files go to the same output base directory
                clip_results = process_video_clips(video_path, timestamps, output_base, subject_id, subject_label)
                
                total_clips += len(clip_results)
                successful_videos += 1
                
                # Track subject statistics
                if subject_id not in subject_stats:
                    subject_stats[subject_id] = {
                        'subject_id': subject_id,
                        'label': subject_label,
                        'total_clips': 0,
                        'video_files': [],
                        'clip_files': []
                    }
                
                subject_stats[subject_id]['total_clips'] += len(clip_results)
                subject_stats[subject_id]['video_files'].append(video_path)
                
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
                
                logger.info(f"Successfully processed {len(clip_results)} clips for {video_name}")
                
            except Exception as e:
                logger.error(f"Error processing {video_path}: {str(e)}")
                failed_videos += 1
    
    # Create detailed summary with subject information
    summary = {
        'dataset_folder': dataset_folder,
        'total_videos_processed': successful_videos,
        'failed_videos': failed_videos,
        'total_clips_extracted': total_clips,
        'output_directory': output_base,
        'clips_per_video': clips_per_video,
        'dataset_stats': {
            'num_subjects': len(subject_stats),
            'subjects': subject_stats
        }
    }
    
    logger.info("=== Serialization Summary ===")
    logger.info(f"Dataset: {dataset_folder}")
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
    """Main function to run the dataset serialization."""
    
    import sys
    
    # Check if command line arguments are provided
    if len(sys.argv) >= 2:
        dataset_folder = sys.argv[1]
    else:
        dataset_folder = "CasualConversationsA"  # Default
    
    if len(sys.argv) >= 3:
        output_base = sys.argv[2]
    else:
        output_base = "serialized_dataset_small"  # Default
    
    if len(sys.argv) >= 4:
        clips_per_video = int(sys.argv[3])
    else:
        clips_per_video = 2  # Default
    
    # Configuration
    json_path = "/Users/ozgewhiting/Documents/EQLabs/datasets/CasualConversations/CC_annotations/CasualConversations.json"
    base_path = "/Users/ozgewhiting/Documents/EQLabs/datasets/CasualConversations"
    
    print("=== CasualConversations Dataset Serialization ===")
    print(f"Annotations file: {json_path}")
    print(f"Base path: {base_path}")
    print(f"Dataset folder: {dataset_folder}")
    print(f"Output directory: {output_base}")
    print(f"Clips per video: {clips_per_video}")
    print()
    print("Usage: python serialize_dataset.py [dataset_folder] [output_dir] [clips_per_video]")
    print("Example: python serialize_dataset.py CasualConversationsA my_output 5")
    print()
    
    try:
        # Run serialization
        summary = serialize_dataset(
            json_path=json_path,
            dataset_folder=dataset_folder,
            base_path=base_path,
            output_base=output_base,
            clips_per_video=clips_per_video
        )
        
        print("=== Final Summary ===")
        for key, value in summary.items():
            print(f"{key}: {value}")
        
    except Exception as e:
        logger.error(f"Serialization failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 