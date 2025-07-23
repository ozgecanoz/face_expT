#!/usr/bin/env python3
"""
Video Info Singleton

A simple script that takes a video file name from command line arguments
and calls get_video_info to display video information.
"""

import sys
import os
from mp4_utils import get_video_info


def main():
    """
    Main function that gets video information for a file passed via command line.
    """
    # Check if video file name is provided as command line argument
    if len(sys.argv) < 2:
        print("Usage: python video_info.py <video_file_name>")
        print("Example: python video_info.py my_video.mp4")
        sys.exit(1)
    
    # Get video file name from command line argument
    video_file = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(video_file):
        print(f"Error: Video file '{video_file}' not found.")
        print("Please provide a valid video file path.")
        sys.exit(1)
    
    try:
        # Get video information
        info = get_video_info(video_file)
        
        # Display video information in a formatted way
        print(f"\n=== Video Information for '{video_file}' ===")
        print(f"File Path: {info['file_path']}")
        print(f"File Size: {info['file_size_mb']} MB ({info['file_size_bytes']} bytes)")
        print(f"Duration: {info['duration_seconds']} seconds")
        print(f"Resolution: {info['resolution']}")
        print(f"Frame Rate: {info['fps']} fps")
        print(f"Total Frames: {info['frame_count']}")
        print(f"Codec: {info['codec']}")
        print(f"Aspect Ratio: {info['aspect_ratio']}")
        print("=" * 50)
        
    except Exception as e:
        print(f"Error getting video information: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 