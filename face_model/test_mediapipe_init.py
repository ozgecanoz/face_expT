#!/usr/bin/env python3
"""
Simple test to verify MediaPipe components can be initialized
"""

import mediapipe as mp
import numpy as np
import cv2

def test_mediapipe_initialization():
    """Test MediaPipe face detection and face mesh initialization"""
    print("🧪 Testing MediaPipe initialization...")
    
    try:
        # Test face detection
        print("  - Initializing face detection...")
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )
        print("    ✅ Face detection initialized")
        
        # Test face mesh
        print("  - Initializing face mesh...")
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("    ✅ Face mesh initialized")
        
        # Test drawing utilities
        print("  - Initializing drawing utilities...")
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        print("    ✅ Drawing utilities initialized")
        
        # Test with dummy image
        print("  - Testing with dummy image...")
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test face detection
        results_detection = face_detection.process(dummy_image)
        print(f"    ✅ Face detection processed dummy image: {len(results_detection.detections) if results_detection.detections else 0} faces detected")
        
        # Test face mesh
        results_mesh = face_mesh.process(dummy_image)
        print(f"    ✅ Face mesh processed dummy image: {len(results_mesh.multi_face_landmarks) if results_mesh.multi_face_landmarks else 0} faces detected")
        
        print("🎉 All MediaPipe components initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ MediaPipe initialization failed: {e}")
        return False

if __name__ == "__main__":
    success = test_mediapipe_initialization()
    exit(0 if success else 1)
