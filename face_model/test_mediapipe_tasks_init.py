#!/usr/bin/env python3
"""
Test MediaPipe Tasks API initialization
"""

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import numpy as np

def test_mediapipe_tasks_import():
    """Test MediaPipe Tasks API imports"""
    print("🧪 Testing MediaPipe Tasks API imports...")
    
    try:
        print("  - Importing mediapipe.tasks.python...")
        from mediapipe.tasks import python
        print("    ✅ mediapipe.tasks.python imported")
        
        print("  - Importing mediapipe.tasks.python.vision...")
        from mediapipe.tasks.python import vision
        print("    ✅ mediapipe.tasks.python.vision imported")
        
        print("  - Importing mediapipe.framework.formats.landmark_pb2...")
        from mediapipe.framework.formats import landmark_pb2
        print("    ✅ landmark_pb2 imported")
        
        print("🎉 All MediaPipe Tasks imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ MediaPipe Tasks import failed: {e}")
        return False

def test_landmark_proto():
    """Test landmark protobuf creation"""
    print("\n🧪 Testing landmark protobuf creation...")
    
    try:
        # Create a dummy landmark
        landmark = landmark_pb2.NormalizedLandmark(x=0.5, y=0.5, z=0.0)
        print(f"    ✅ Created landmark: x={landmark.x}, y={landmark.y}, z={landmark.z}")
        
        # Create landmark list
        landmark_list = landmark_pb2.NormalizedLandmarkList()
        landmark_list.landmark.extend([landmark])
        print(f"    ✅ Created landmark list with {len(landmark_list.landmark)} landmarks")
        
        print("🎉 Landmark protobuf creation successful!")
        return True
        
    except Exception as e:
        print(f"❌ Landmark protobuf creation failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing MediaPipe Tasks API...")
    
    success1 = test_mediapipe_tasks_import()
    success2 = test_landmark_proto()
    
    if success1 and success2:
        print("\n🎉 All tests passed! MediaPipe Tasks API is ready to use.")
        exit(0)
    else:
        print("\n❌ Some tests failed. Please check MediaPipe installation.")
        exit(1)
