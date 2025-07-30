#!/usr/bin/env python3
"""
Test script to compare old vs new video analysis approach
"""

import os
import sys
import logging
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clip_generation.video_expression_analyzer import VideoExpressionAnalyzer

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_video_analysis(video_path: str, expression_transformer_checkpoint: str):
    """Test both old and new video analysis approaches"""
    
    logger.info(f"Testing video analysis for: {video_path}")
    
    # Initialize analyzer
    analyzer = VideoExpressionAnalyzer(
        expression_transformer_checkpoint_path=expression_transformer_checkpoint,
        device="cpu"
    )
    
    # Test 1: Sample clips first
    logger.info("=== Testing new clip-based approach ===")
    try:
        # Sample 5 clips of 30 frames each
        clip_ranges = analyzer.sample_random_clips(video_path, N=5, sequence_length=30)
        logger.info(f"Sampled {len(clip_ranges)} clips: {clip_ranges}")
        
        # Analyze only the sampled clips
        frame_data = analyzer.analyze_video_clips(video_path, clip_ranges, subject_id=0)
        
        # Count faces detected
        faces_detected = sum(1 for fd in frame_data if fd.bbox is not None)
        total_frames = len(frame_data)
        
        logger.info(f"New approach results:")
        logger.info(f"  - Total frames analyzed: {total_frames}")
        logger.info(f"  - Frames with faces: {faces_detected}")
        logger.info(f"  - Face detection rate: {faces_detected/total_frames*100:.1f}%")
        
    except Exception as e:
        logger.error(f"New approach failed: {e}")
    
    # Test 2: Analyze first 150 frames (old approach simulation)
    logger.info("=== Testing old approach (first 150 frames) ===")
    try:
        # Create clip ranges for first 150 frames in chunks of 30
        old_clip_ranges = [(i, i+29) for i in range(0, 150, 30)]
        logger.info(f"Old approach clip ranges: {old_clip_ranges}")
        
        # Analyze these frames
        old_frame_data = analyzer.analyze_video_clips(video_path, old_clip_ranges, subject_id=0)
        
        # Count faces detected
        old_faces_detected = sum(1 for fd in old_frame_data if fd.bbox is not None)
        old_total_frames = len(old_frame_data)
        
        logger.info(f"Old approach results:")
        logger.info(f"  - Total frames analyzed: {old_total_frames}")
        logger.info(f"  - Frames with faces: {old_faces_detected}")
        logger.info(f"  - Face detection rate: {old_faces_detected/old_total_frames*100:.1f}%")
        
    except Exception as e:
        logger.error(f"Old approach failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Test video analysis approaches")
    parser.add_argument("video_path", help="Path to test video")
    parser.add_argument("expression_transformer_checkpoint", help="Path to expression transformer checkpoint")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        logger.error(f"Video file not found: {args.video_path}")
        return
    
    if not os.path.exists(args.expression_transformer_checkpoint):
        logger.error(f"Checkpoint file not found: {args.expression_transformer_checkpoint}")
        return
    
    test_video_analysis(args.video_path, args.expression_transformer_checkpoint)

if __name__ == "__main__":
    main() 