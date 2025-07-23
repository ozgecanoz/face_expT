#!/usr/bin/env python3
"""
Check dataset integrity and identify problematic HDF5 files
"""

import h5py
import os
import glob
import numpy as np
from tqdm import tqdm

def check_h5_file(file_path):
    """Check a single HDF5 file for integrity"""
    try:
        with h5py.File(file_path, 'r') as f:
            # Check if required groups exist
            if 'faces' not in f:
                return {'status': 'error', 'message': 'Missing faces group'}
            
            faces_group = f['faces']
            frame_count = len(faces_group.keys())
            
            # Check each frame
            frame_details = []
            for frame_name in sorted(faces_group.keys()):
                frame_group = faces_group[frame_name]
                face_count = len(frame_group.keys())
                
                if face_count == 0:
                    frame_details.append(f"Frame {frame_name}: No faces")
                elif face_count > 1:
                    frame_details.append(f"Frame {frame_name}: {face_count} faces (should be 1)")
                else:
                    frame_details.append(f"Frame {frame_name}: 1 face ✓")
            
            # Check metadata
            metadata_issues = []
            if 'metadata' in f:
                metadata = f['metadata']
                if 'num_frames' in metadata:
                    expected_frames = metadata['num_frames'][()]
                    if frame_count != expected_frames:
                        metadata_issues.append(f"Frame count mismatch: {frame_count} vs expected {expected_frames}")
            else:
                metadata_issues.append("Missing metadata group")
            
            return {
                'status': 'ok' if frame_count == 30 and all('1 face' in detail for detail in frame_details) else 'warning',
                'frame_count': frame_count,
                'frame_details': frame_details,
                'metadata_issues': metadata_issues,
                'file_path': file_path
            }
            
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e),
            'file_path': file_path
        }

def check_dataset_integrity(data_dir):
    """Check integrity of all HDF5 files in the dataset"""
    print(f"🔍 Checking dataset integrity: {data_dir}")
    
    # Find all HDF5 files
    h5_files = glob.glob(os.path.join(data_dir, "*.h5"))
    print(f"📁 Found {len(h5_files)} HDF5 files")
    
    if not h5_files:
        print("❌ No HDF5 files found")
        return
    
    # Check each file
    results = []
    problematic_files = []
    
    for file_path in tqdm(h5_files, desc="Checking files"):
        result = check_h5_file(file_path)
        results.append(result)
        
        if result['status'] != 'ok':
            problematic_files.append(result)
    
    # Print summary
    print("\n📊 Dataset Integrity Summary:")
    print("=" * 60)
    
    ok_files = [r for r in results if r['status'] == 'ok']
    warning_files = [r for r in results if r['status'] == 'warning']
    error_files = [r for r in results if r['status'] == 'error']
    
    print(f"✅ Valid files: {len(ok_files)}")
    print(f"⚠️  Warning files: {len(warning_files)}")
    print(f"❌ Error files: {len(error_files)}")
    
    # Show problematic files
    if problematic_files:
        print(f"\n🚨 Problematic Files:")
        print("=" * 60)
        
        for result in problematic_files:
            print(f"\n📄 {os.path.basename(result['file_path'])}")
            print(f"   Status: {result['status']}")
            
            if 'message' in result:
                print(f"   Error: {result['message']}")
            
            if 'frame_count' in result:
                print(f"   Frame count: {result['frame_count']}")
                
                if result['frame_count'] < 30:
                    print(f"   ⚠️  Missing frames: {30 - result['frame_count']}")
                
                if result['frame_details']:
                    print("   Frame details:")
                    for detail in result['frame_details'][:5]:  # Show first 5
                        print(f"     {detail}")
                    if len(result['frame_details']) > 5:
                        print(f"     ... and {len(result['frame_details']) - 5} more")
            
            if 'metadata_issues' in result and result['metadata_issues']:
                print("   Metadata issues:")
                for issue in result['metadata_issues']:
                    print(f"     {issue}")
    
    return problematic_files

def fix_dataset_issues(data_dir, problematic_files):
    """Suggest fixes for problematic files"""
    print(f"\n🔧 Suggested Fixes:")
    print("=" * 60)
    
    if not problematic_files:
        print("✅ No issues found!")
        return
    
    print("The following actions are recommended:")
    
    for result in problematic_files:
        if result['status'] == 'error':
            print(f"❌ Delete: {os.path.basename(result['file_path'])} - {result['message']}")
        elif 'frame_count' in result and result['frame_count'] < 30:
            print(f"⚠️  Re-extract: {os.path.basename(result['file_path'])} - Only {result['frame_count']} frames")
        elif 'frame_details' in result:
            multi_face_frames = [d for d in result['frame_details'] if 'faces (should be 1)' in d]
            if multi_face_frames:
                print(f"⚠️  Re-extract: {os.path.basename(result['file_path'])} - Multiple faces detected")
    
    print(f"\n💡 To fix these issues:")
    print("1. Delete problematic files")
    print("2. Re-run the serialization script with the updated single-face logic")
    print("3. The new logic will automatically skip clips with multiple faces")

def main():
    """Main function"""
    data_dir = "/Users/ozgewhiting/Documents/EQLabs/datasets_serial/CCA_val_db1"
    
    if not os.path.exists(data_dir):
        print(f"❌ Dataset directory not found: {data_dir}")
        return
    
    # Check dataset integrity
    problematic_files = check_dataset_integrity(data_dir)
    
    # Suggest fixes
    fix_dataset_issues(data_dir, problematic_files)

if __name__ == "__main__":
    main() 