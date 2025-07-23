#!/usr/bin/env python3
"""
Example usage of MP4 utilities

This script demonstrates how to use the various functions in mp4_utils.py
"""

import os
from mp4_utils import (
    extract_frames,
    extract_audio,
    get_video_info,
    extract_frames_at_timestamps,
    create_video_thumbnail,
    batch_extract_audio
)


def main():
    """Example usage of MP4 utilities"""
    
    # Example video file (replace with your actual video file)
    video_file = "example.mp4"
    
    # Check if example video exists
    if not os.path.exists(video_file):
        print(f"Example video file '{video_file}' not found.")
        print("Please place an MP4 file named 'example.mp4' in this directory to run the examples.")
        return
    
    print("=== MP4 Utilities Example ===\n")
    
    # 1. Get video information
    print("1. Getting video information...")
    try:
        info = get_video_info(video_file)
        print("Video Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        print()
    except Exception as e:
        print(f"Error getting video info: {e}\n")
    
    # 2. Extract frames (1 frame per second for first 10 seconds)
    print("2. Extracting frames...")
    try:
        frames = extract_frames(
            video_path=video_file,
            frame_rate=1,  # 1 frame per second
            start_time=0,
            end_time=10,  # First 10 seconds
            quality=90
        )
        print(f"Extracted {len(frames)} frames")
        print()
    except Exception as e:
        print(f"Error extracting frames: {e}\n")
    
    # 3. Extract audio
    print("3. Extracting audio...")
    try:
        audio_file = extract_audio(
            video_path=video_file,
            audio_format="mp3",
            start_time=0,
            end_time=30  # First 30 seconds
        )
        print(f"Audio extracted: {audio_file}")
        print()
    except Exception as e:
        print(f"Error extracting audio: {e}\n")
    
    # 4. Extract frames at specific timestamps
    print("4. Extracting frames at specific timestamps...")
    try:
        timestamps = [2.5, 5.0, 7.5, 10.0]  # Specific timestamps in seconds
        specific_frames = extract_frames_at_timestamps(
            video_path=video_file,
            timestamps=timestamps,
            quality=95
        )
        print(f"Extracted {len(specific_frames)} frames at specific timestamps")
        print()
    except Exception as e:
        print(f"Error extracting frames at timestamps: {e}\n")
    
    # 5. Create thumbnail
    print("5. Creating thumbnail...")
    try:
        thumbnail = create_video_thumbnail(
            video_path=video_file,
            timestamp=5.0,  # 5 seconds into the video
            size=(320, 240),
            quality=90
        )
        print(f"Thumbnail created: {thumbnail}")
        print()
    except Exception as e:
        print(f"Error creating thumbnail: {e}\n")
    
    # 6. Batch audio extraction (if multiple video files exist)
    print("6. Batch audio extraction...")
    video_files = [f for f in os.listdir('.') if f.endswith('.mp4')]
    if len(video_files) > 1:
        try:
            extracted_audio_files = batch_extract_audio(
                video_files=video_files,
                output_dir="batch_extracted_audio",
                audio_format="mp3"
            )
            print(f"Batch extracted {len(extracted_audio_files)} audio files")
        except Exception as e:
            print(f"Error in batch extraction: {e}")
    else:
        print("Only one MP4 file found, skipping batch extraction")
    
    print("\n=== Example completed ===")


if __name__ == "__main__":
    main() 