#!/usr/bin/env python3
"""
Test script for face extraction functionality
"""

import os
from mp4_utils import extract_face_sequence

def main():
    """Test face extraction on the Casual Conversations video"""
    
    # Video file path from previous test
    video_path = "../../EQLabs/datasets/CasualConversations/CasualConversationsA/1178/1178_08.MP4"
    output_folder = "test_output"
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        print("Please ensure the video file exists and the path is correct.")
        return
    
    print("=== Face Extraction Test ===")
    print(f"Video: {video_path}")
    
    # Test parameters
    timestamp = 20.0  # Start at 10 seconds
    output_folder = output_folder + "/t_" + str(timestamp)  # Output folder for both HDF5 and MP4 files
    # create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    face_size = (518, 518)
    num_frames = 15
    frame_skip = 0  # Skip 1 frame (every 2nd frame for 30fps)
    
    print(f"Extracting faces at timestamp: {timestamp}s")
    print(f"Output folder: {output_folder}")
    print(f"Face size: {face_size}")
    print(f"Number of frames: {num_frames}")
    print(f"Frame skip: {frame_skip}")
    print()
    
    try:
        # Extract face sequence
        result = extract_face_sequence(
            video_path=video_path,
            timestamp=timestamp,
            output_folder=output_folder,
            face_size=face_size,
            num_frames=num_frames,
            frame_skip=frame_skip,
            confidence_threshold=0.5
        )
        
        print("=== Extraction Results ===")
        print(f"HDF5 file: {result['hdf5_path']}")
        print(f"MP4 file: {result['mp4_path']}")
        print(f"Frames processed: {result['num_frames']}")
        print(f"Timestamp: {result['timestamp']}s")
        
        # Check if files were created
        if os.path.exists(result['hdf5_path']):
            hdf5_size = os.path.getsize(result['hdf5_path']) / (1024 * 1024)  # MB
            print(f"HDF5 file size: {hdf5_size:.2f} MB")
        
        if os.path.exists(result['mp4_path']):
            mp4_size = os.path.getsize(result['mp4_path']) / (1024 * 1024)  # MB
            print(f"MP4 file size: {mp4_size:.2f} MB")
        
        print("\nFace extraction completed successfully!")
        
    except Exception as e:
        print(f"Error during face extraction: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 