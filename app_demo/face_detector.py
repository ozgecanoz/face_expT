"""
Face Detector using MediaPipe
Extracts face regions from video frames using MediaPipe face detection
"""

import cv2
import mediapipe as mp
import numpy as np
import logging
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)


class MediaPipeFaceDetector:
    """Face detector using MediaPipe for real-time face detection"""
    
    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize MediaPipe face detector
        
        Args:
            confidence_threshold: Minimum confidence for face detection (0.0-1.0)
        """
        self.confidence_threshold = confidence_threshold
        
        # Initialize MediaPipe face detection
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=0,  # 0 for short-range model, 1 for full-range model
            min_detection_confidence=confidence_threshold
        )
        
        logger.info(f"MediaPipe Face Detector initialized with confidence threshold: {confidence_threshold}")
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect faces in a frame
        
        Args:
            frame: Input frame (BGR format from OpenCV)
            
        Returns:
            List of (x, y, width, height, confidence) tuples for detected faces
        """
        # Convert BGR to RGB (MediaPipe expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = self.face_detection.process(rgb_frame)
        
        detected_faces = []
        
        if results.detections:
            frame_height, frame_width = frame.shape[:2]
            
            for detection in results.detections:
                # Get bounding box coordinates
                bbox = detection.location_data.relative_bounding_box
                
                # Convert normalized coordinates to pixel coordinates
                x = int(bbox.xmin * frame_width)
                y = int(bbox.ymin * frame_height)
                width = int(bbox.width * frame_width)
                height = int(bbox.height * frame_height)
                
                # Get confidence score
                confidence = detection.score[0]
                
                detected_faces.append((x, y, width, height, confidence))
        
        return detected_faces
    
    def extract_face_region(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int], 
                           target_size: Tuple[int, int] = (518, 518)) -> Optional[np.ndarray]:
        """
        Extract and resize face region from frame
        
        Args:
            frame: Input frame
            face_bbox: (x, y, width, height) of detected face
            target_size: (width, height) for output face image
            
        Returns:
            Resized face image or None if extraction fails
        """
        x, y, width, height = face_bbox
        
        # Ensure coordinates are within frame bounds
        x = max(0, x)
        y = max(0, y)
        width = min(width, frame.shape[1] - x)
        height = min(height, frame.shape[0] - y)
        
        # Check if face region is valid
        if width <= 0 or height <= 0:
            return None
        
        # Extract face region
        face_region = frame[y:y+height, x:x+width]
        
        # Resize to target size
        try:
            resized_face = cv2.resize(face_region, target_size)
            
            # Ensure correct format: (H, W, C) with C=3
            if len(resized_face.shape) == 3 and resized_face.shape[2] == 3:
                # Already in correct format (H, W, C)
                return resized_face
            else:
                logger.warning(f"Unexpected face image shape: {resized_face.shape}")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to resize face region: {e}")
            return None
    
    def draw_detection_boxes(self, frame: np.ndarray, 
                           detected_faces: List[Tuple[int, int, int, int, float]]) -> np.ndarray:
        """
        Draw green detection boxes on frame
        
        Args:
            frame: Input frame
            detected_faces: List of (x, y, width, height, confidence) tuples
            
        Returns:
            Frame with detection boxes drawn
        """
        frame_with_boxes = frame.copy()
        
        for x, y, width, height, confidence in detected_faces:
            # Draw green rectangle
            cv2.rectangle(frame_with_boxes, (x, y), (x + width, y + height), (0, 255, 0), 2)
            
            # Draw confidence score
            confidence_text = f"{confidence:.2f}"
            cv2.putText(frame_with_boxes, confidence_text, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame_with_boxes
    
    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], List[Tuple[int, int, int, int, float]]]:
        """
        Process a frame to detect faces and extract the largest face
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (largest_face_image, all_detected_faces)
        """
        # Detect faces
        detected_faces = self.detect_faces(frame)
        
        if not detected_faces:
            return None, detected_faces
        
        # Find the largest face (highest area)
        largest_face = max(detected_faces, key=lambda face: face[2] * face[3])
        
        # Extract face region
        face_image = self.extract_face_region(frame, largest_face[:4])
        
        return face_image, detected_faces
    
    def close(self):
        """Close MediaPipe face detection"""
        if hasattr(self, 'face_detection'):
            self.face_detection.close()


def test_face_detector():
    """Test the face detector with webcam"""
    import cv2
    
    print("🧪 Testing MediaPipe Face Detector...")
    
    # Initialize face detector
    detector = MediaPipeFaceDetector(confidence_threshold=0.5)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return
    
    print("📹 Webcam opened successfully")
    print("Press 'q' to quit")
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            # Process frame
            face_image, detected_faces = detector.process_frame(frame)
            
            # Draw detection boxes
            frame_with_boxes = detector.draw_detection_boxes(frame, detected_faces)
            
            # Display frame
            cv2.imshow('Face Detection Test', frame_with_boxes)
            
            # Display face image if detected
            if face_image is not None:
                cv2.imshow('Extracted Face', face_image)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        print("✅ Face detector test completed!")


if __name__ == "__main__":
    test_face_detector() 