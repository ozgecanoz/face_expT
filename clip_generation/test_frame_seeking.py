#!/usr/bin/env python3
"""
Test frame seeking functionality
"""

import cv2
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_frame_seeking(video_path: str):
    """Test if frame seeking works correctly"""
    
    logger.info(f"Testing frame seeking for: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    logger.info(f"Video properties: {total_frames} frames, {fps} FPS")
    
    # Test seeking to different frames
    test_frames = [0, 10, 50, 100, min(200, total_frames-1)]
    
    for frame_idx in test_frames:
        if frame_idx >= total_frames:
            continue
            
        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # Check actual position
        actual_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        # Read frame
        ret, frame = cap.read()
        
        if ret:
            logger.info(f"Frame {frame_idx}: Seek successful, actual={actual_frame}, frame shape={frame.shape}")
        else:
            logger.warning(f"Frame {frame_idx}: Seek failed, actual={actual_frame}")
    
    cap.release()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python test_frame_seeking.py <video_path>")
        sys.exit(1)
    
    test_frame_seeking(sys.argv[1]) 