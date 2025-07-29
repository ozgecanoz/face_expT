#!/usr/bin/env python3
"""
Convert MOV files to MP4 format using OpenCV
Useful for converting QuickTime recordings to MP4 format
"""

import cv2
import os
import argparse
import sys
from pathlib import Path


def convert_mov_to_mp4(input_path, output_path=None, verbose=True):
    """
    Convert MOV file to MP4 format
    
    Args:
        input_path: Path to input MOV file
        output_path: Path for output MP4 file (optional, auto-generated if None)
        verbose: Whether to print progress information
    
    Returns:
        bool: True if conversion successful, False otherwise
    """
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return False
    
    # Generate output path if not provided
    if output_path is None:
        input_file = Path(input_path)
        output_path = input_file.with_suffix('.mp4')
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open input video: {input_path}")
        return False
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if verbose:
        print(f"Converting: {input_path}")
        print(f"Output: {output_path}")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps}")
        print(f"Total frames: {total_frames}")
        print(f"Duration: {total_frames / fps:.2f} seconds")
        print("-" * 50)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Could not create output video: {output_path}")
        cap.release()
        return False
    
    # Convert frames
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            out.write(frame)
            frame_count += 1
            
            if verbose and frame_count % 30 == 0:  # Print progress every 30 frames
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {frame_count}/{total_frames} frames ({progress:.1f}%)")
        
        # Cleanup
        cap.release()
        out.release()
        
        if verbose:
            print("-" * 50)
            print(f"Conversion complete!")
            print(f"Output: {output_path}")
            print(f"Processed {frame_count} frames")
            
            # Get output file size
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
                print(f"File size: {file_size:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        cap.release()
        out.release()
        return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Convert MOV files to MP4 format")
    parser.add_argument("input", help="Path to input MOV file")
    parser.add_argument("-o", "--output", help="Path for output MP4 file (optional)")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output")
    parser.add_argument("-b", "--batch", help="Convert all MOV files in directory")
    
    args = parser.parse_args()
    
    if args.batch:
        # Batch mode: convert all MOV files in directory
        input_dir = args.batch
        if not os.path.isdir(input_dir):
            print(f"Error: Directory not found: {input_dir}")
            sys.exit(1)
        
        mov_files = []
        for file in os.listdir(input_dir):
            if file.lower().endswith('.mov'):
                mov_files.append(os.path.join(input_dir, file))
        
        if not mov_files:
            print(f"No MOV files found in: {input_dir}")
            sys.exit(1)
        
        print(f"Found {len(mov_files)} MOV files to convert")
        successful = 0
        
        for mov_file in mov_files:
            print(f"\nConverting: {mov_file}")
            if convert_mov_to_mp4(mov_file, verbose=not args.quiet):
                successful += 1
        
        print(f"\nBatch conversion complete: {successful}/{len(mov_files)} successful")
        
    else:
        # Single file mode
        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}")
            sys.exit(1)
        
        success = convert_mov_to_mp4(args.input, args.output, verbose=not args.quiet)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 