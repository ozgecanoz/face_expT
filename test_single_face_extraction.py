#!/usr/bin/env python3
"""
Test script for single face extraction logic
"""

import os
import sys
from mp4_utils import extract_face_sequence

def test_single_face_extraction():
    """Test the updated face extraction with single face requirement"""
    
    # Test configuration
    video_path = "/Users/ozgewhiting/Documents/EQLabs/datasets/CasualConversations/CasualConversationsA/1140/1140_02.MP4"
    timestamp = 10.0  # Test timestamp
    output_folder = "./test_single_face_output"
    
    print("🧪 Testing Single Face Extraction")
    print(f"📹 Video: {video_path}")
    print(f"⏰ Timestamp: {timestamp}s")
    print(f"📁 Output: {output_folder}")
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        print("Please update the video_path to a valid file.")
        return
    
    print("✅ Video file found")
    
    # Test face extraction
    try:
        result = extract_face_sequence(
            video_path=video_path,
            timestamp=timestamp,
            output_folder=output_folder,
            face_size=(518, 518),
            num_frames=5,  # Test with fewer frames
            frame_skip=1,
            confidence_threshold=0.5,
            subject_id="test_subject",
            subject_label={"age": "30", "gender": "Female", "skin-type": "3"}
        )
        
        if result is None:
            print("❌ Face extraction failed - likely due to multiple faces or insufficient single faces")
        else:
            print("✅ Face extraction successful!")
            print(f"📊 Results:")
            print(f"  - Frames with single face: {result['frames_with_single_face']}")
            print(f"  - Total faces found: {result['total_faces']}")
            print(f"  - HDF5 file: {result['hdf5_path']}")
            print(f"  - MP4 file: {result['mp4_path']}")
            
            # Check if files were created
            if os.path.exists(result['hdf5_path']):
                print("✅ HDF5 file created")
            else:
                print("❌ HDF5 file not found")
                
            if os.path.exists(result['mp4_path']):
                print("✅ MP4 file created")
            else:
                print("❌ MP4 file not found")
    
    except Exception as e:
        print(f"❌ Error during face extraction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_face_extraction() 