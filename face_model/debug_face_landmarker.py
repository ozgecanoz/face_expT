#!/usr/bin/env python3
"""
Debug MediaPipe Face Landmarker
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os

def test_face_landmarker():
    """Test the face landmarker with a simple image"""
    
    # Check if task file exists
    task_path = "/Users/ozgewhiting/Documents/projects/face_landmarker.task"
    if not os.path.exists(task_path):
        print(f"❌ Task file not found: {task_path}")
        return False
    
    print(f"✅ Task file found: {task_path}")
    print(f"   File size: {os.path.getsize(task_path)} bytes")
    
    try:
        # Initialize face landmarker
        print("🔄 Initializing face landmarker...")
        base_options = python.BaseOptions(model_asset_path=task_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1
        )
        face_landmarker = vision.FaceLandmarker.create_from_options(options)
        print("✅ Face landmarker initialized successfully")
        
        # Create a simple test image with a face-like pattern
        print("🖼️  Creating test image...")
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=test_image)
        print(f"✅ MediaPipe Image created: {mp_image.width}x{mp_image.height}")
        
        # Try to detect landmarks
        print("🔍 Attempting to detect landmarks...")
        detection_result = face_landmarker.detect(mp_image)
        
        print(f"   Detection result type: {type(detection_result)}")
        print(f"   Face landmarks: {len(detection_result.face_landmarks) if detection_result.face_landmarks else 0}")
        
        if detection_result.face_blendshapes:
            print(f"   Face blendshapes: {len(detection_result.face_blendshapes)}")
        
        if detection_result.facial_transformation_matrixes:
            print(f"   Facial transformation matrices: {len(detection_result.facial_transformation_matrixes)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_real_image():
    """Test with a real image if available"""
    
    # Try to find a test image
    test_image_paths = [
        "/Users/ozgewhiting/Documents/EQLabs/datasets/CasualConversations/CasualConversationsA/1221/1221_07.MP4",
        "test_image.jpg",
        "test_image.png"
    ]
    
    for image_path in test_image_paths:
        if os.path.exists(image_path):
            print(f"\n🖼️  Testing with: {image_path}")
            
            if image_path.endswith('.MP4') or image_path.endswith('.mp4'):
                # Extract first frame from video
                cap = cv2.VideoCapture(image_path)
                ret, frame = cap.read()
                cap.release()
                
                if ret:
                    print(f"   Extracted frame: {frame.shape}")
                    test_frame(frame)
                else:
                    print("   Failed to extract frame from video")
            else:
                # Load image directly
                frame = cv2.imread(image_path)
                if frame is not None:
                    print(f"   Loaded image: {frame.shape}")
                    test_frame(frame)
                else:
                    print("   Failed to load image")
            break
    else:
        print("⚠️  No test images found")

def test_frame(frame):
    """Test face landmarker with a specific frame"""
    
    try:
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Initialize face landmarker
        task_path = "/Users/ozgewhiting/Documents/projects/face_landmarker.task"
        base_options = python.BaseOptions(model_asset_path=task_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1
        )
        face_landmarker = vision.FaceLandmarker.create_from_options(options)
        
        # Detect landmarks
        detection_result = face_landmarker.detect(mp_image)
        
        print(f"   Detection result: {len(detection_result.face_landmarks) if detection_result.face_landmarks else 0} faces")
        
        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]
            print(f"   First face has {len(landmarks)} landmarks")
            
            # Check first few landmarks
            for i in range(min(5, len(landmarks))):
                landmark = landmarks[i]
                print(f"     Landmark {i}: x={landmark.x:.3f}, y={landmark.y:.3f}, z={landmark.z:.3f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error testing frame: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Debugging MediaPipe Face Landmarker...")
    
    # Test basic initialization
    success1 = test_face_landmarker()
    
    # Test with real image/video
    test_with_real_image()
    
    if success1:
        print("\n🎉 Basic tests completed!")
    else:
        print("\n❌ Basic tests failed!")
