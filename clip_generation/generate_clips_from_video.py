#!/usr/bin/env python3
"""
Generate Clips from Video
Main script for analyzing videos and extracting expressive face sequences
"""

import os
import logging
import argparse
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clip_generation.video_expression_analyzer import VideoExpressionAnalyzer
from clip_generation.clip_extractor import extract_clip_frames, extract_cropped_face_frames, save_clip_as_h5, save_clip_as_mp4

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main function for generating clips from a single video"""
    parser = argparse.ArgumentParser(description="Generate expressive clips from a video")
    parser.add_argument("--video_path", required=True, help="Path to input video")
    parser.add_argument("--expression_transformer_checkpoint", required=True, help="Path to expression transformer checkpoint")
    parser.add_argument("--output_dir", default="./output_clips", help="Output directory for clips")
    parser.add_argument("--subject_id", type=int, default=0, help="Subject ID for expression extraction")
    parser.add_argument("--K", type=int, default=10, help="Number of top sequences to return")
    parser.add_argument("--N", type=int, default=100, help="Number of sequences to sample and evaluate")
    parser.add_argument("--sequence_length", type=int, default=30, help="Length of each sequence in frames")
    parser.add_argument("--expression_weight", type=float, default=1.0, help="Weight for expression variation scoring")
    parser.add_argument("--position_weight", type=float, default=0.0, help="Weight for position stability scoring")
    parser.add_argument("--device", default="cpu", help="Device to run models on")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize analyzer
    logger.info("Initializing video expression analyzer...")
    analyzer = VideoExpressionAnalyzer(
        expression_transformer_checkpoint_path=args.expression_transformer_checkpoint,
        device=args.device
    )
    
    # Sample random clips
    logger.info(f"Sampling {args.N} clips from video...")
    clip_ranges = analyzer.sample_random_clips(args.video_path, args.N, args.sequence_length)
    
    if not clip_ranges:
        logger.error("No valid clips could be sampled from the video")
        return
    
    # Analyze clips
    logger.info("Analyzing clips...")
    frame_data = analyzer.analyze_video_clips(args.video_path, clip_ranges, args.subject_id)
    
    if not frame_data:
        logger.error("No valid frame data found")
        return
    
    # Find best sequences
    logger.info("Finding best sequences...")
    sequences = analyzer.find_best_sequences_from_clips(
        frame_data, 
        clip_ranges,
        K=args.K,
        expression_weight=args.expression_weight,
        position_weight=args.position_weight
    )
    
    if not sequences:
        logger.error("No valid sequences found")
        return
    
    # Extract video name prefix
    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    
    # Extract and save clips
    logger.info(f"Extracting {len(sequences)} clips...")
    for i, sequence in enumerate(sequences):
        try:
            # Extract cropped face frames for this sequence
            frames = extract_cropped_face_frames(args.video_path, sequence.start_frame, sequence.end_frame)
            
            if not frames:
                logger.warning(f"No frames extracted for sequence {i+1}")
                continue
            
            # Create output filename with new signature
            # Format: subject_<subject_id>_<video_name>_faces_<start_frame>_<end_frame>
            clip_id = f"subject_{args.subject_id}_{video_name}_faces_{sequence.start_frame}_{sequence.end_frame}"
            h5_path = os.path.join(args.output_dir, f"{clip_id}.h5")
            mp4_path = os.path.join(args.output_dir, f"{clip_id}.mp4")
            
            # Create metadata
            metadata = {
                'video_id': args.video_path,  # Use video_id key for H5 compatibility
                'subject_id': args.subject_id,
                'subject_label': {},  # Add empty subject_label for H5 compatibility
                'clip_id': clip_id,
                'start_frame': sequence.start_frame,
                'end_frame': sequence.end_frame,
                'num_frames': len(frames),
                'sequence_length': args.sequence_length,
                'expression_variation': sequence.expression_variation,
                'position_stability': sequence.position_stability,
                'combined_score': sequence.combined_score,
                'expression_weight': args.expression_weight,
                'position_weight': args.position_weight,
                'frame_size': '518x518'  # Indicate these are cropped face frames
            }
            
            # Save as H5
            h5_success = save_clip_as_h5(frames, h5_path, metadata)
            
            # Save as MP4
            mp4_success = save_clip_as_mp4(frames, mp4_path)
            
            if h5_success and mp4_success:
                logger.info(f"✅ Saved clip {i+1}/{len(sequences)}: {clip_id} (score: {sequence.combined_score:.8f})")
            else:
                logger.warning(f"Failed to save clip {i+1}")
                
        except Exception as e:
            logger.error(f"Error processing sequence {i+1}: {e}")
            continue
    
    logger.info(f"🎉 Clip generation completed! Output directory: {args.output_dir}")


if __name__ == "__main__":
    # Default parameters for easy script execution
    import sys
    
    # If no arguments provided, use defaults
    if len(sys.argv) == 1:
        # Set default values for quick testing
        sys.argv.extend([
            "--video_path", "/Users/ozgewhiting/Documents/EQLabs/datasets/CasualConversations/1140/1140_00.MP4",
            "--expression_transformer_checkpoint", "/Users/ozgewhiting/Documents/projects/dataset_utils/cloud_checkpoints/expression_transformer_epoch_5.pt",
            "--output_dir", "./output_clips",
            "--subject_id", "1",
            "--K", "10",
            "--N", "100",
            "--expression_weight", "1.0",
            "--position_weight", "0.0"
        ])
        print("🚀 Using default parameters for quick testing")
        print("   To use custom parameters, provide them as command line arguments")
    
    main()
    exit(0 if success else 1) 