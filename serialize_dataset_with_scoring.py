#!/usr/bin/env python3
"""
Dataset Serialization Script with Expression-Based Scoring

This script processes the CasualConversations dataset by:
1. Reading the JSON annotations file
2. Finding available videos (excluding dark_files)
3. Using expression-based scoring to select the best clips
4. Extracting face sequences for training using mp4_utils
"""

import json
import os
import random
import glob
from typing import List, Dict, Tuple
import logging
import argparse
from tqdm import tqdm
import numpy as np

# Import functions from original serialize_dataset.py and mp4_utils.py
from serialize_dataset import load_annotations, find_available_videos, sample_random_clips
from mp4_utils import extract_face_sequence

# Import clip generation components
from clip_generation.video_expression_analyzer import VideoExpressionAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_video_with_scoring(
    video_path: str,
    video_id: str,
    subject_id: str,
    subject_label: Dict,
    output_base: str,
    analyzer: VideoExpressionAnalyzer,
    K: int = 5,
    N: int = 50,
    sequence_length: int = 30,
    expression_weight: float = 1.0,
    position_weight: float = 0.0
) -> List[Dict]:
    """
    Process a video using scoring-based clip selection.
    
    Args:
        video_path (str): Path to video file
        video_id (str): Video identifier
        subject_id (str): Subject identifier
        subject_label (Dict): Subject label information
        output_base (str): Base output directory
        analyzer (VideoExpressionAnalyzer): Video analyzer instance
        K (int): Number of top clips to select
        N (int): Number of sequences to sample
        sequence_length (int): Length of each sequence
        expression_weight (float): Weight for expression variation scoring
        position_weight (float): Weight for position stability scoring
        
    Returns:
        List[Dict]: List of clip results with metadata
    """
    logger.info(f"Processing video: {video_id}")
    
    try:
        # Step 1: Sample N random timestamps from the video
        timestamps = sample_random_clips(video_path, N, min_duration=sequence_length/30.0)
        
        if not timestamps:
            logger.warning(f"No valid timestamps could be sampled from {video_id}")
            return []
        
        # Step 2: Extract face sequences for all timestamps using mp4_utils
        extracted_clips = []
        for i, timestamp in enumerate(timestamps):
            try:
                logger.info(f"Extracting face sequence {i+1}/{len(timestamps)} at {timestamp:.2f}s")
                
                # Extract face sequence using mp4_utils
                result = extract_face_sequence(
                    video_path=video_path,
                    timestamp=timestamp,
                    output_folder=output_base,
                    face_size=(518, 518),
                    num_frames=sequence_length,
                    frame_skip=0,  # No frame skipping for better quality
                    confidence_threshold=0.5,
                    subject_id=subject_id,
                    subject_label=subject_label
                )
                
                if result is not None:
                    # Add timestamp and index information
                    result['timestamp'] = timestamp
                    result['clip_index'] = i
                    extracted_clips.append(result)
                    logger.info(f"✅ Extracted clip {i+1}: {result['hdf5_path']}")
                else:
                    logger.warning(f"❌ Failed to extract clip {i+1} at {timestamp:.2f}s")
                    
            except Exception as e:
                logger.error(f"Error extracting clip {i+1} at {timestamp:.2f}s: {e}")
                continue
        
        if not extracted_clips:
            logger.warning(f"No valid clips extracted from {video_id}")
            return []
        
        # Step 3: Score the extracted clips using the analyzer
        scored_clips = []
        for clip in extracted_clips:
            try:
                # Load face data from HDF5 for scoring
                import h5py
                with h5py.File(clip['hdf5_path'], 'r') as f:
                    face_data = []
                    for i in range(clip['num_frames']):
                        frame_key = f'frame_{i:03d}/face_000/data'
                        if frame_key in f:
                            face_array = f[frame_key][:]
                            # Convert to RGB format expected by analyzer
                            face_data.append(face_array)
                        else:
                            logger.warning(f"Missing face data for frame {i} in {clip['hdf5_path']}")
                            face_data = None
                            break
                
                if face_data is None or len(face_data) != sequence_length:
                    logger.warning(f"Incomplete face data for clip {clip['clip_index']}")
                    continue
                
                # Convert to numpy array and normalize to 0-1 range
                face_data = np.array(face_data) / 255.0  # Normalize to 0-1
                
                # Score the clip using analyzer's expression variation calculation
                # Convert subject_id to int for analyzer
                analyzer_subject_id = int(subject_id) if subject_id.isdigit() else 0
                
                # Calculate expression variation using the analyzer's method
                expression_variation = analyzer._calculate_expression_variation_from_frames(
                    face_data, analyzer_subject_id
                )
                
                # Calculate position stability (simplified - assume stable if all frames have faces)
                position_stability = 1.0 if len(face_data) == sequence_length else 0.0
                
                # Calculate combined score
                combined_score = (expression_weight * expression_variation + 
                                position_weight * position_stability)
                
                # Add scoring information to clip
                clip['expression_variation'] = expression_variation
                clip['position_stability'] = position_stability
                clip['combined_score'] = combined_score
                clip['expression_weight'] = expression_weight
                clip['position_weight'] = position_weight
                
                scored_clips.append(clip)
                logger.info(f"Scored clip {clip['clip_index']}: expression={expression_variation:.8f}, "
                          f"position={position_stability:.8f}, combined={combined_score:.8f}")
                
            except Exception as e:
                logger.error(f"Error scoring clip {clip['clip_index']}: {e}")
                continue
        
        if not scored_clips:
            logger.warning(f"No clips could be scored for {video_id}")
            return []
        
        # Step 4: Select top K clips based on scores
        scored_clips.sort(key=lambda x: x['combined_score'], reverse=True)
        top_clips = scored_clips[:K]
        
        logger.info(f"Selected {len(top_clips)} top clips from {len(scored_clips)} scored clips")
        for i, clip in enumerate(top_clips):
            logger.info(f"Top clip {i+1}: score={clip['combined_score']:.8f}, "
                      f"expression={clip['expression_variation']:.8f}")
        
        return top_clips
        
    except Exception as e:
        logger.error(f"Error processing video {video_id}: {e}")
        return []


def serialize_dataset_with_scoring(
    json_path: str,
    dataset_folder: str,
    base_path: str,
    output_base: str = "serialized_dataset_with_scoring",
    expression_transformer_checkpoint_path: str = None,
    K: int = 5,
    N: int = 50,
    sequence_length: int = 30,
    expression_weight: float = 1.0,
    position_weight: float = 0.0,
    device: str = "cpu"
) -> Dict:
    """
    Serialize dataset with expression-based scoring.
    
    Args:
        json_path (str): Path to JSON annotations file
        dataset_folder (str): Dataset folder name (e.g., 'CasualConversationsA')
        base_path (str): Base path to video files
        output_base (str): Output directory for serialized data
        expression_transformer_checkpoint_path (str): Path to expression transformer checkpoint
        K (int): Number of top clips to select per video
        N (int): Number of sequences to sample per video
        sequence_length (int): Length of each sequence in frames
        expression_weight (float): Weight for expression variation scoring
        position_weight (float): Weight for position stability scoring
        device (str): Device to run models on
        
    Returns:
        Dict: Summary of serialization results
    """
    logger.info(f"🚀 Starting dataset serialization with scoring")
    logger.info(f"📁 Dataset folder: {dataset_folder}")
    logger.info(f"📂 Base path: {base_path}")
    logger.info(f"📄 Annotations: {json_path}")
    logger.info(f"📤 Output: {output_base}")
    
    # Create output directory
    os.makedirs(output_base, exist_ok=True)
    
    # Load annotations
    logger.info(f"Loading annotations from: {json_path}")
    annotations = load_annotations(json_path)
    
    # Find available videos
    logger.info(f"Finding available videos in {dataset_folder}...")
    available_videos = find_available_videos(annotations, dataset_folder, base_path)
    
    if not available_videos:
        logger.error(f"No available videos found in {dataset_folder}")
        return {}
    
    # Initialize analyzer
    logger.info("Initializing video expression analyzer...")
    analyzer = VideoExpressionAnalyzer(
        expression_transformer_checkpoint_path=expression_transformer_checkpoint_path,
        device=device
    )
    
    # Process videos
    total_clips = 0
    successful_videos = 0
    
    for video_info in tqdm(available_videos, desc="Processing videos"):
        try:
            video_id = video_info['video_id']
            video_data = video_info['data']
            
            # Construct video path
            video_path = os.path.join(base_path, video_id)
            
            if not os.path.exists(video_path):
                logger.warning(f"Video file not found: {video_path}")
                continue
            
            # Find subject_id by matching video_id to annotations
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
                logger.warning(f"Could not find subject_id for video {video_id}")
                continue
            
            # Process video with scoring
            clip_results = process_video_with_scoring(
                video_path=video_path,
                video_id=video_id,
                subject_id=subject_id,
                subject_label=subject_label,
                output_base=output_base,
                analyzer=analyzer,
                K=K,
                N=N,
                sequence_length=sequence_length,
                expression_weight=expression_weight,
                position_weight=position_weight
            )
            
            if clip_results:
                total_clips += len(clip_results)
                successful_videos += 1
                logger.info(f"✅ Processed {video_id}: {len(clip_results)} clips")
            else:
                logger.warning(f"⚠️ No clips extracted from {video_id}")
                
        except Exception as e:
            logger.error(f"❌ Error processing {video_id}: {e}")
            continue
    
    # Create metadata summary
    metadata = {
        'dataset_folder': dataset_folder,
        'total_videos_processed': successful_videos,
        'total_clips_extracted': total_clips,
        'parameters': {
            'K': K,
            'N': N,
            'sequence_length': sequence_length,
            'expression_weight': expression_weight,
            'position_weight': position_weight
        },
        'output_directory': output_base
    }
    
    metadata_path = os.path.join(output_base, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"🎉 Dataset serialization completed!")
    logger.info(f"📊 Summary:")
    logger.info(f"   - Dataset folder: {dataset_folder}")
    logger.info(f"   - Videos processed: {successful_videos}")
    logger.info(f"   - Total clips extracted: {total_clips}")
    logger.info(f"   - Output directory: {output_base}")
    logger.info(f"   - Metadata saved to: {metadata_path}")
    
    return metadata


def main():
    """Main function for dataset serialization with scoring"""
    parser = argparse.ArgumentParser(description="Serialize dataset with expression-based scoring")
    parser.add_argument("--json_path", required=True, help="Path to JSON annotations file")
    parser.add_argument("--base_path", required=True, help="Base path to video files")
    parser.add_argument("--dataset_folder", required=True, help="Dataset folder name (e.g., 'CasualConversationsA')")
    parser.add_argument("--output_path", required=True, help="Output directory for serialized data")
    parser.add_argument("--expression_transformer_checkpoint", required=True, help="Path to expression transformer checkpoint")
    parser.add_argument("--K", type=int, default=5, help="Number of top clips to select per video")
    parser.add_argument("--N", type=int, default=50, help="Number of sequences to sample per video")
    parser.add_argument("--sequence_length", type=int, default=30, help="Length of each sequence in frames")
    parser.add_argument("--expression_weight", type=float, default=1.0, help="Weight for expression variation scoring")
    parser.add_argument("--position_weight", type=float, default=0.0, help="Weight for position stability scoring")
    parser.add_argument("--max_videos", type=int, default=None, help="Maximum number of videos to process")
    parser.add_argument("--device", default="cpu", help="Device to run models on")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Load annotations
    logger.info(f"Loading annotations from: {args.json_path}")
    with open(args.json_path, 'r') as f:
        annotations = json.load(f)
    
    # Find available videos
    logger.info(f"Finding available videos in {args.dataset_folder}...")
    available_videos = find_available_videos(annotations, args.dataset_folder, args.base_path)
    
    if not available_videos:
        logger.error(f"No available videos found in {args.dataset_folder}")
        return
    
    # Limit videos if max_videos is specified
    if args.max_videos:
        available_videos = available_videos[:args.max_videos]
        logger.info(f"Limited to {args.max_videos} videos")
    
    # Run serialization
    serialize_dataset_with_scoring(
        json_path=args.json_path,
        dataset_folder=args.dataset_folder,
        base_path=args.base_path,
        output_base=args.output_path,
        expression_transformer_checkpoint_path=args.expression_transformer_checkpoint,
        K=args.K,
        N=args.N,
        sequence_length=args.sequence_length,
        expression_weight=args.expression_weight,
        position_weight=args.position_weight,
        device=args.device
    )


if __name__ == "__main__":
    # Default parameters for easy script execution
    import sys
    
    # If no arguments provided, use defaults
    if len(sys.argv) == 1:
        # Set default values for quick testing
        sys.argv.extend([
            "--json_path", "/Users/ozgewhiting/Documents/EQLabs/datasets/CasualConversations/CC_annotations/CasualConversations.json",
            "--base_path", "/Users/ozgewhiting/Documents/EQLabs/datasets/CasualConversations",
            "--dataset_folder", "CasualConversationsA",
            "--output_path", "/Users/ozgewhiting/Documents/EQLabs/datasets_serial/CCA_train_db3/",
            "--expression_transformer_checkpoint", "/Users/ozgewhiting/Documents/projects/cloud_checkpoints_with_subject_ids/expression_transformer_epoch_5.pt",
            "--K", "3",
            "--N", "10",
            "--expression_weight", "1.0",
            "--position_weight", "0.0",
            "--max_videos", "2"
        ])
        print("🚀 Using default parameters for quick testing")
        print("   To use custom parameters, provide them as command line arguments")
    
    main() 