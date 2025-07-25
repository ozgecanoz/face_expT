#!/usr/bin/env python3
"""
Script to read and display dataset metadata with validation checks
"""

import json
import os
import sys
import h5py

def validate_clip_frames(hdf5_path):
    """Validate that a clip has exactly 30 frames"""
    try:
        with h5py.File(hdf5_path, 'r') as f:
            faces_group = f['faces']
            num_frames = len(faces_group.keys())
            return num_frames == 30
    except Exception as e:
        print(f"  ❌ Error reading {hdf5_path}: {e}")
        return False

def validate_video_files(video_files, base_dir):
    """Validate that all video files in metadata exist in directory"""
    missing_files = []
    for video_path in video_files:
        if not os.path.exists(video_path):
            missing_files.append(video_path)
    return missing_files

def read_dataset_metadata(metadata_file):
    """Read and display dataset metadata with validation"""
    
    if not os.path.exists(metadata_file):
        print(f"Metadata file not found: {metadata_file}")
        return
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    if 'dataset_stats' in metadata:
        stats = metadata['dataset_stats']
        
        print(f"\n=== Subject Details ===")
        total_clips_validated = 0
        total_clips_invalid = 0
        total_videos_missing = 0
        
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
            
            # Validate video files exist
            missing_videos = validate_video_files(subject_data['video_files'], os.path.dirname(metadata_file))
            if missing_videos:
                print(f"  ⚠️  Missing video files: {len(missing_videos)}")
                total_videos_missing += len(missing_videos)
                for missing in missing_videos[:3]:  # Show first 3
                    print(f"    - {os.path.basename(missing)}")
                if len(missing_videos) > 3:
                    print(f"    ... and {len(missing_videos) - 3} more")
            else:
                print(f"  ✅ All video files exist")
            
            # Validate clip files have 30 frames
            valid_clips = 0
            invalid_clips = 0
            for clip in subject_data['clip_files']:
                hdf5_path = clip['hdf5_file']
                if os.path.exists(hdf5_path):
                    if validate_clip_frames(hdf5_path):
                        valid_clips += 1
                    else:
                        invalid_clips += 1
                        print(f"  ❌ Invalid clip: {os.path.basename(hdf5_path)} (not 30 frames)")
                else:
                    invalid_clips += 1
                    print(f"  ❌ Missing clip file: {os.path.basename(hdf5_path)}")
            
            total_clips_validated += valid_clips
            total_clips_invalid += invalid_clips
            
            print(f"  ✅ Valid clips (30 frames): {valid_clips}")
            if invalid_clips > 0:
                print(f"  ❌ Invalid clips: {invalid_clips}")
            
            # Show first few clip files
            if subject_data['clip_files']:
                print(f"  Sample clips:")
                for i, clip in enumerate(subject_data['clip_files'][:3]):
                    status = "✅" if validate_clip_frames(clip['hdf5_file']) else "❌"
                    print(f"    {i+1}. {status} {os.path.basename(clip['hdf5_file'])} (t={clip['timestamp']:.2f}s)")
                if len(subject_data['clip_files']) > 3:
                    print(f"    ... and {len(subject_data['clip_files']) - 3} more")
        
        print("\n=== Dataset Metadata ===")
        print(f"Dataset folder: {metadata['dataset_folder']}")
        print(f"Output directory: {metadata['output_directory']}")
        print(f"Total videos processed: {metadata['total_videos_processed']}")
        print(f"Failed videos: {metadata['failed_videos']}")
        print(f"Total clips extracted: {metadata['total_clips_extracted']}")
        print(f"Clips per video: {metadata['clips_per_video']}")

        print(f"\n=== Subject Statistics ===")
        print(f"Number of subjects: {stats['num_subjects']}")
        
        print(f"\n=== Validation Summary ===")
        print(f"✅ Valid clips (30 frames): {total_clips_validated}")
        print(f"❌ Invalid clips: {total_clips_invalid}")
        print(f"❌ Missing video files: {total_videos_missing}")
        
        if total_clips_invalid == 0 and total_videos_missing == 0:
            print(f"🎉 Dataset validation passed! All clips have 30 frames and all files exist.")
        else:
            print(f"⚠️  Dataset validation failed! Found {total_clips_invalid} invalid clips and {total_videos_missing} missing video files.")

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