#!/usr/bin/env python3
"""
Script to read and display dataset metadata
"""

import json
import os
import sys

def read_dataset_metadata(metadata_file):
    """Read and display dataset metadata"""
    
    if not os.path.exists(metadata_file):
        print(f"Metadata file not found: {metadata_file}")
        return
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    print("=== Dataset Metadata ===")
    print(f"Dataset folder: {metadata['dataset_folder']}")
    print(f"Output directory: {metadata['output_directory']}")
    print(f"Total videos processed: {metadata['total_videos_processed']}")
    print(f"Failed videos: {metadata['failed_videos']}")
    print(f"Total clips extracted: {metadata['total_clips_extracted']}")
    print(f"Clips per video: {metadata['clips_per_video']}")
    
    if 'dataset_stats' in metadata:
        stats = metadata['dataset_stats']
        print(f"\n=== Subject Statistics ===")
        print(f"Number of subjects: {stats['num_subjects']}")
        
        print(f"\n=== Subject Details ===")
        for subject_id, subject_data in stats['subjects'].items():
            print(f"\nSubject {subject_id}:")
            print(f"  Total clips: {subject_data['total_clips']}")
            
            if subject_data['label']:
                label = subject_data['label']
                print(f"  Age: {label.get('age', 'N/A')}")
                print(f"  Gender: {label.get('gender', 'N/A')}")
                print(f"  Skin type: {label.get('skin-type', 'N/A')}")
            
            print(f"  Video files: {len(subject_data['video_files'])}")
            print(f"  Clip files: {len(subject_data['clip_files'])}")
            
            # Show first few clip files
            if subject_data['clip_files']:
                print(f"  Sample clips:")
                for i, clip in enumerate(subject_data['clip_files'][:3]):
                    print(f"    {i+1}. {os.path.basename(clip['hdf5_file'])} (t={clip['timestamp']:.2f}s)")
                if len(subject_data['clip_files']) > 3:
                    print(f"    ... and {len(subject_data['clip_files']) - 3} more")

def main():
    """Main function"""
    
    if len(sys.argv) < 2:
        print("Usage: python read_dataset_metadata.py <metadata_file>")
        print("Example: python read_dataset_metadata.py test_output/dataset_metadata.json")
        return
    
    metadata_file = sys.argv[1]
    read_dataset_metadata(metadata_file)

if __name__ == "__main__":
    main() 