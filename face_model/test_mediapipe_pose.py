#!/usr/bin/env python3
"""
Test MediaPipe Pose Detection with Face Keypoints Visualization

This script inputs an MP4 video, uses MediaPipe to detect face bounding boxes and keypoints,
and creates an output video with:
- Left side: Input frame cropped to highest confidence face bounding box
- Right side: Cropped face with keypoints overlaid as colored dots

Keypoint colors:
- Nose tip: Yellow
- Eyes: Blue
- Mouth: Green
- Ears: Red
- Other facial landmarks: Cyan
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List
import logging

# Add models directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
from arcface_tokenizer import ArcFaceTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MediaPipePoseTester:
    """
    Test MediaPipe pose detection with face keypoints visualization
    """
    
    def __init__(self, 
                 face_detection_confidence: float = 0.5,
                 arcface_model_path: Optional[str] = None,
                 face_landmarker_task_path: str = "face_landmarker_v2_with_blendshapes.task"):
        """
        Initialize MediaPipe pose tester
        
        Args:
            face_detection_confidence: Face detection confidence threshold (0.0-1.0)
            arcface_model_path: Path to ArcFace ONNX model for similarity calculations
            face_landmarker_task_path: Path to MediaPipe face landmarker task file
        """
        self.face_detection_confidence = face_detection_confidence
        self.arcface_model_path = arcface_model_path
        self.face_landmarker_task_path = face_landmarker_task_path
        
        # Initialize MediaPipe
        self._setup_mediapipe()
        
        # Initialize ArcFace tokenizer if model path provided
        self.arcface_tokenizer = None
        if arcface_model_path:
            self._setup_arcface()
        
        logger.info(f"✅ MediaPipe pose tester initialized")
        logger.info(f"   Face detection confidence: {face_detection_confidence}")
        logger.info(f"   Face landmarker task: {face_landmarker_task_path}")
        if arcface_model_path:
            logger.info(f"   ArcFace model: {arcface_model_path}")
        else:
            logger.info(f"   ArcFace: Not enabled")
    
    def _setup_mediapipe(self):
        """Initialize MediaPipe face landmarker using Tasks API with fallback"""
        try:
            # Check if task file exists
            if not os.path.exists(self.face_landmarker_task_path):
                logger.error(f"❌ Face landmarker task file not found: {self.face_landmarker_task_path}")
                logger.error("Please download the face_landmarker_v2_with_blendshapes.task file")
                raise FileNotFoundError(f"Task file not found: {self.face_landmarker_task_path}")
            
            # Create FaceLandmarker using Tasks API
            base_options = python.BaseOptions(model_asset_path=self.face_landmarker_task_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                num_faces=1
            )
            self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
            
            # Also initialize fallback face detection
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=self.face_detection_confidence
            )
            
            # Drawing utilities
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            logger.info("✅ MediaPipe Tasks face landmarker initialized with fallback")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize MediaPipe: {e}")
            raise
    
    def _setup_arcface(self):
        """Initialize ArcFace tokenizer"""
        try:
            if not os.path.exists(self.arcface_model_path):
                logger.error(f"❌ ArcFace model not found: {self.arcface_model_path}")
                return
            
            self.arcface_tokenizer = ArcFaceTokenizer(self.arcface_model_path)
            logger.info("✅ ArcFace tokenizer initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize ArcFace: {e}")
            self.arcface_tokenizer = None
    
    def detect_face_and_keypoints(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[dict], Optional[np.ndarray]]:
        """
        Detect face and extract keypoints from image using MediaPipe Tasks face landmarker
        
        Args:
            image: Input image (RGB format)
            
        Returns:
            Tuple of (cropped_face, keypoints_data, annotated_face)
            - cropped_face: Face crop or None if no face detected
            - keypoints_data: Dictionary with keypoint information
            - annotated_face: Face crop with keypoints overlaid
        """
        try:
            # Convert to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = image
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            logger.debug(f"Processing image: shape={image_rgb.shape}, dtype={image_rgb.dtype}, range=[{image_rgb.min()}, {image_rgb.max()}]")
            
            # Create MediaPipe Image object with proper format
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            logger.debug(f"Created MediaPipe Image: {mp_image.width}x{mp_image.height}")
            
            # Try MediaPipe Tasks API first
            try:
                detection_result = self.face_landmarker.detect(mp_image)
                logger.debug(f"Tasks API - Detection result type: {type(detection_result)}")
                logger.debug(f"Tasks API - Face landmarks: {len(detection_result.face_landmarks) if detection_result.face_landmarks else 0}")
                
                if detection_result.face_landmarks:
                    # Use Tasks API result
                    logger.debug("Using Tasks API result")
                    return self._process_tasks_result(detection_result, image)
                    
            except Exception as e:
                logger.debug(f"Tasks API failed: {e}")
            
            # Fallback to old face detection method
            logger.debug("Falling back to face detection method")
            return self._process_fallback_detection(image)
            
        except Exception as e:
            logger.error(f"Error in face detection/keypoint extraction: {e}")
            return None, None, None
    
    def _process_tasks_result(self, detection_result, image):
        """Process MediaPipe Tasks API result"""
        try:
            # Get the first face landmarks
            face_landmarks = detection_result.face_landmarks[0]
            logger.debug(f"Processing face with {len(face_landmarks)} landmarks")
            
            # Calculate bounding box from landmarks
            x_coords = [landmark.x for landmark in face_landmarks]
            y_coords = [landmark.y for landmark in face_landmarks]
            
            # Convert relative coordinates to absolute
            h, w = image.shape[:2]
            x = int(min(x_coords) * w)
            y = int(min(y_coords) * h)
            width = int((max(x_coords) - min(x_coords)) * w)
            height = int((max(y_coords) - min(y_coords)) * h)
            
            logger.debug(f"Bounding box: x={x}, y={y}, w={width}, h={height}")
            logger.debug(f"Image dimensions: {w}x{h}")
            
            # Add some padding around the face
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            width = min(width + 2 * padding, w - x)
            height = min(height + 2 * padding, h - y)
            
            # Crop face
            face_crop = image[y:y+height, x:x+width]
            
            # Extract keypoints data
            keypoints_data = self._extract_keypoints_from_landmarks(face_landmarks, x, y, width, height, w, h)
            logger.debug(f"Extracted keypoints: {len(keypoints_data['all_landmarks']) if keypoints_data else 0}")
            
            # Create annotated face with landmarks
            annotated_face = self._draw_landmarks_on_face(face_crop, face_landmarks, x, y, width, height, w, h)
            
            return face_crop, keypoints_data, annotated_face
            
        except Exception as e:
            logger.error(f"Error processing Tasks API result: {e}")
            return None, None, None
    
    def _process_fallback_detection(self, image):
        """Process using fallback face detection method"""
        try:
            # Convert to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = image
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces using old method
            detection_results = self.face_detection.process(image_rgb)
            
            if not detection_results.detections:
                logger.debug("No face detected in fallback method")
                return None, None, None
            
            # Get the first face
            detection = detection_results.detections[0]
            
            # Get bounding box
            bbox = detection.location_data.relative_bounding_box
            
            # Convert relative coordinates to absolute
            h, w = image.shape[:2]
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            # Ensure coordinates are within bounds
            x = max(0, x)
            y = max(0, y)
            width = min(width, w - x)
            height = min(height, h - y)
            
            # Crop face
            face_crop = image[y:y+height, x:x+width]
            
            # Extract keypoints using old method
            keypoints_data = self._extract_keypoints_from_detection(detection, x, y, width, height, w, h)
            
            # Create simple annotated face (just the crop for now)
            annotated_face = face_crop
            
            logger.debug(f"Fallback method: extracted face crop {face_crop.shape}")
            return face_crop, keypoints_data, annotated_face
            
        except Exception as e:
            logger.error(f"Error in fallback detection: {e}")
            return None, None, None
    
    def _extract_keypoints_from_landmarks(self, face_landmarks, crop_x: int, crop_y: int, crop_w: int, crop_h: int, image_w: int, image_h: int) -> dict:
        """
        Extract keypoint data from MediaPipe face landmarks
        
        Args:
            face_landmarks: MediaPipe face landmarks
            crop_x, crop_y: Crop offset from original image
            crop_w, crop_h: Crop dimensions
            image_w, image_h: Original image dimensions
            
        Returns:
            Dictionary with keypoint information
        """
        keypoints_data = {
            'nose_tip': None,
            'left_eye': None,
            'right_eye': None,
            'mouth_center': None,
            'all_landmarks': []
        }
        
        # MediaPipe face mesh landmark indices for key facial features
        landmark_indices = {
            'nose_tip': 1,           # Nose tip
            'left_eye': 33,          # Left eye center
            'right_eye': 263,        # Right eye center
            'mouth_center': 13,      # Mouth center
        }
        
        # Extract key landmarks
        for name, idx in landmark_indices.items():
            if idx < len(face_landmarks):
                landmark = face_landmarks[idx]
                
                # Convert relative coordinates to crop-relative coordinates
                rel_x = landmark.x
                rel_y = landmark.y
                
                # Validate coordinate ranges
                if not (0.0 <= rel_x <= 1.0) or not (0.0 <= rel_y <= 1.0):
                    continue
                
                # Convert to crop-relative coordinates
                crop_rel_x = (rel_x - crop_x / image_w) * (image_w / crop_w)
                crop_rel_y = (rel_y - crop_y / image_h) * (image_h / crop_h)
                
                # Convert to pixel coordinates in crop
                pixel_x = int(crop_rel_x * crop_w)
                pixel_y = int(crop_rel_y * crop_h)
                
                # Ensure coordinates are within crop bounds
                pixel_x = max(0, min(pixel_x, crop_w - 1))
                pixel_y = max(0, min(pixel_y, crop_h - 1))
                
                keypoints_data[name] = (pixel_x, pixel_y)
                keypoints_data['all_landmarks'].append((name, pixel_x, pixel_y))
        
        # Only return keypoints data if we have at least some valid keypoints
        if len(keypoints_data['all_landmarks']) > 0:
            return keypoints_data
        else:
            return None
    
    def _extract_keypoints_from_detection(self, detection, crop_x: int, crop_y: int, crop_w: int, crop_h: int, image_w: int, image_h: int) -> dict:
        """
        Extract keypoint data from MediaPipe face detection (fallback method)
        
        Args:
            detection: MediaPipe face detection result
            crop_x, crop_y: Crop offset from original image
            crop_w, crop_h: Crop dimensions
            image_w, image_h: Original image dimensions
            
        Returns:
            Dictionary with keypoint information
        """
        keypoints_data = {
            'nose_tip': None,
            'left_eye': None,
            'right_eye': None,
            'mouth_center': None,
            'all_landmarks': []
        }
        
        # MediaPipe face detection keypoints
        keypoint_types = {
            'nose_tip': self.mp_face_detection.FaceKeyPoint.NOSE_TIP,
            'left_eye': self.mp_face_detection.FaceKeyPoint.LEFT_EYE,
            'right_eye': self.mp_face_detection.FaceKeyPoint.RIGHT_EYE,
            'mouth_center': self.mp_face_detection.FaceKeyPoint.MOUTH_CENTER
        }
        
        # Extract key landmarks
        for name, keypoint_type in keypoint_types.items():
            try:
                keypoint = self.mp_face_detection.get_key_point(detection, keypoint_type)
                
                # Check if keypoint is valid
                if keypoint is None:
                    continue
                
                # Validate keypoint coordinates
                if not hasattr(keypoint, 'x') or not hasattr(keypoint, 'y'):
                    continue
                
                # Convert relative coordinates to crop-relative coordinates
                rel_x = keypoint.x
                rel_y = keypoint.y
                
                # Convert to crop-relative coordinates
                crop_rel_x = (rel_x - crop_x / image_w) * (image_w / crop_w)
                crop_rel_y = (rel_y - crop_y / image_h) * (image_h / crop_h)
                
                # Convert to pixel coordinates in crop
                pixel_x = int(crop_rel_x * crop_w)
                pixel_y = int(crop_rel_y * crop_h)
                
                # Ensure coordinates are within crop bounds
                pixel_x = max(0, min(pixel_x, crop_w - 1))
                pixel_y = max(0, min(pixel_y, crop_h - 1))
                
                keypoints_data[name] = (pixel_x, pixel_y)
                keypoints_data['all_landmarks'].append((name, pixel_x, pixel_y))
                
            except Exception as e:
                logger.debug(f"Could not extract {name} keypoint: {e}")
                continue
        
        # Only return keypoints data if we have at least some valid keypoints
        if len(keypoints_data['all_landmarks']) > 0:
            return keypoints_data
        else:
            return None
    
    def compute_cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> Optional[float]:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            emb1: First normalized embedding
            emb2: Second normalized embedding
            
        Returns:
            Cosine similarity score or None if error
        """
        try:
            if emb1 is None or emb2 is None:
                return None
            
            # Check for invalid values
            if np.any(np.isnan(emb1)) or np.any(np.isinf(emb1)):
                return None
                
            if np.any(np.isnan(emb2)) or np.any(np.isinf(emb2)):
                return None
            
            # Compute cosine similarity
            cosine_sim = np.dot(emb1, emb2)
            
            # Check if result is valid
            if np.isnan(cosine_sim) or np.isinf(cosine_sim):
                return None
            
            return cosine_sim
            
        except Exception as e:
            logger.debug(f"Error computing cosine similarity: {e}")
            return None
    
    def _draw_landmarks_on_face(self, face_crop: np.ndarray, face_landmarks, crop_x: int, crop_y: int, crop_w: int, crop_h: int, image_w: int, image_h: int) -> np.ndarray:
        """
        Draw face landmarks on face crop using MediaPipe drawing utilities
        
        Args:
            face_crop: Face crop image
            face_landmarks: MediaPipe face landmarks
            crop_x, crop_y: Crop offset from original image
            crop_w, crop_h: Crop dimensions
            image_w, image_h: Original image dimensions
            
        Returns:
            Face crop with landmarks overlaid
        """
        try:
            # Create a copy of the face crop
            annotated_face = face_crop.copy()
            
            # Create a MediaPipe Image object for the face crop
            mp_face_crop = mp.Image(image_format=mp.ImageFormat.SRGB, data=annotated_face)
            
            # Create a new landmark list with coordinates adjusted for the crop
            adjusted_landmarks = []
            for landmark in face_landmarks:
                # Convert from original image coordinates to crop coordinates
                crop_rel_x = (landmark.x - crop_x / image_w) * (image_w / crop_w)
                crop_rel_y = (landmark.y - crop_y / image_h) * (image_h / crop_h)
                
                # Create new landmark with adjusted coordinates
                adjusted_landmark = landmark_pb2.NormalizedLandmark(
                    x=crop_rel_x,
                    y=crop_rel_y,
                    z=landmark.z
                )
                adjusted_landmarks.append(adjusted_landmark)
            
            # Create landmark list proto
            landmark_list = landmark_pb2.NormalizedLandmarkList()
            landmark_list.landmark.extend(adjusted_landmarks)
            
            # Draw the face mesh using MediaPipe drawing utilities
            self.mp_drawing.draw_landmarks(
                image=annotated_face,
                landmark_list=landmark_list,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            
            # Draw face contours
            self.mp_drawing.draw_landmarks(
                image=annotated_face,
                landmark_list=landmark_list,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
            )
            
            # Draw iris connections
            self.mp_drawing.draw_landmarks(
                image=annotated_face,
                landmark_list=landmark_list,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style()
            )
            
            return annotated_face
            
        except Exception as e:
            logger.error(f"Error drawing landmarks: {e}")
            return face_crop
    
    def process_video(self, input_video_path: str, output_video_path: str) -> bool:
        """
        Process input video and create output video with face keypoints
        
        Args:
            input_video_path: Path to input MP4 video
            output_video_path: Path to output MP4 video
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Open input video
            cap = cv2.VideoCapture(input_video_path)
            if not cap.isOpened():
                logger.error(f"❌ Could not open input video: {input_video_path}")
                return False
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"📹 Input video properties:")
            logger.info(f"   FPS: {fps}")
            logger.info(f"   Resolution: {width}x{height}")
            logger.info(f"   Total frames: {total_frames}")
            
            # Create output video writer
            # Output will be side-by-side: original crop | annotated crop
            output_width = width * 2  # Double width for side-by-side
            output_height = height
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))
            
            if not out.isOpened():
                logger.error(f"❌ Could not create output video: {output_video_path}")
                cap.release()
                return False
            
            frame_count = 0
            processed_frames = 0
            reference_embedding = None
            
            logger.info("🎬 Processing video frames...")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process frame
                face_crop, keypoints_data, annotated_face = self.detect_face_and_keypoints(frame)
                
                if face_crop is not None:
                    processed_frames += 1
                    
                    # Compute ArcFace similarity if tokenizer is available
                    similarity_text = ""
                    if self.arcface_tokenizer is not None:
                        try:
                            # Convert face crop to RGB [0,1] format for ArcFace
                            face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                            face_crop_normalized = face_crop_rgb.astype(np.float32) / 255.0
                            
                            # Get current frame embedding
                            current_embedding = self.arcface_tokenizer.forward(face_crop_normalized)
                            
                            if current_embedding is not None:
                                # Store first frame as reference
                                if reference_embedding is None:
                                    reference_embedding = current_embedding
                                    similarity_text = "Reference Frame"
                                else:
                                    # Compute similarity with reference
                                    similarity = self.compute_cosine_similarity(current_embedding, reference_embedding)
                                    if similarity is not None:
                                        similarity_text = f"Similarity: {similarity:.3f}"
                                    else:
                                        similarity_text = "Similarity: N/A"
                        except Exception as e:
                            logger.debug(f"ArcFace processing error: {e}")
                            similarity_text = "ArcFace: Error"
                    
                    # Create side-by-side output
                    output_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
                    
                    # Left side: original face crop
                    # Resize face crop to match output height
                    face_resized = cv2.resize(face_crop, (width, height))
                    output_frame[:, :width] = face_resized
                    
                    # Add similarity text to left frame (upper left corner)
                    if similarity_text:
                        cv2.putText(output_frame, similarity_text, 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        cv2.putText(output_frame, similarity_text, 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
                    
                    # Right side: annotated face crop
                    if annotated_face is not None:
                        annotated_resized = cv2.resize(annotated_face, (width, height))
                        output_frame[:, width:] = annotated_resized
                    
                    # Add frame counter and info
                    cv2.putText(output_frame, f"Frame: {frame_count}/{total_frames}", 
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    if keypoints_data:
                        cv2.putText(output_frame, f"Landmarks: {len(keypoints_data['all_landmarks'])}", 
                                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    # Write frame
                    out.write(output_frame)
                else:
                    # No face detected, create blank frame with message
                    output_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
                    cv2.putText(output_frame, "No face detected", 
                               (width//2 - 100, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(output_frame, f"Frame: {frame_count}/{total_frames}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    out.write(output_frame)
                
                # Progress update
                if frame_count % 30 == 0:
                    logger.info(f"   Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")
            
            # Cleanup
            cap.release()
            out.release()
            
            logger.info(f"✅ Video processing completed!")
            logger.info(f"   Total frames: {frame_count}")
            logger.info(f"   Frames with faces: {processed_frames}")
            logger.info(f"   Output saved to: {output_video_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing video: {e}")
            return False
    
    def process_single_image(self, input_image_path: str, output_image_path: str) -> bool:
        """
        Process single image and create output with face keypoints
        
        Args:
            input_image_path: Path to input image
            output_image_path: Path to output image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Read image
            image = cv2.imread(input_image_path)
            if image is None:
                logger.error(f"❌ Could not read input image: {input_image_path}")
                return False
            
            logger.info(f"🖼️  Processing image: {input_image_path}")
            
            # Process image
            face_crop, keypoints_data, annotated_face = self.detect_face_and_keypoints(image)
            
            if face_crop is None:
                logger.warning("No face detected in image")
                return False
            
            # Create side-by-side output
            height, width = face_crop.shape[:2]
            output_width = width * 2
            output_height = height
            
            output_image = np.zeros((output_height, output_width, 3), dtype=np.uint8)
            
            # Left side: original face crop
            output_image[:, :width] = face_crop
            
            # Right side: annotated face crop
            if annotated_face is not None:
                output_image[:, width:] = annotated_face
            
            # Add info text
            cv2.putText(output_image, "Original Crop", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(output_image, "Keypoints Overlay", (width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            if keypoints_data:
                cv2.putText(output_image, f"Landmarks: {len(keypoints_data['all_landmarks'])}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Save output
            cv2.imwrite(output_image_path, output_image)
            
            logger.info(f"✅ Image processing completed!")
            logger.info(f"   Output saved to: {output_image_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing image: {e}")
            return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test MediaPipe pose detection with face keypoints")
    parser.add_argument("--input", type=str, required=True,
                       help="Input video (MP4) or image path")
    parser.add_argument("--output", type=str, required=True,
                       help="Output video or image path")
    parser.add_argument("--face-detection-confidence", type=float, default=0.5,
                       help="Face detection confidence threshold (0.0-1.0, default: 0.5)")
    parser.add_argument("--arcface-model", type=str, default="/Users/ozgewhiting/Documents/projects/arc.onnx",
                       help="Path to ArcFace ONNX model for similarity calculations")
    parser.add_argument("--face-landmarker-task", type=str, #default="face_landmarker_v2_with_blendshapes.task",
                       default="/Users/ozgewhiting/Documents/projects/face_landmarker.task",
                       help="Path to MediaPipe face landmarker task file")
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file not found: {args.input}")
        return
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize tester
    tester = MediaPipePoseTester(
        face_detection_confidence=args.face_detection_confidence,
        arcface_model_path=args.arcface_model,
        face_landmarker_task_path=args.face_landmarker_task
    )
    
    # Determine if input is video or image
    input_ext = Path(args.input).suffix.lower()
    output_ext = Path(args.output).suffix.lower()
    
    if input_ext in ['.mp4', '.avi', '.mov', '.mkv']:
        # Process video
        if output_ext not in ['.mp4', '.avi', '.mov', '.mkv']:
            print("⚠️  Warning: Output extension doesn't match video input, using .mp4")
            args.output = str(Path(args.output).with_suffix('.mp4'))
        
        print(f"🎬 Processing video: {args.input}")
        success = tester.process_video(args.input, args.output)
        
    elif input_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        # Process image
        if output_ext not in ['.jpg', '.jpeg', '.png', '.bmp']:
            print("⚠️  Warning: Output extension doesn't match image input, using .png")
            args.output = str(Path(args.output).with_suffix('.png'))
        
        print(f"🖼️  Processing image: {args.input}")
        success = tester.process_single_image(args.input, args.output)
        
    else:
        print(f"❌ Error: Unsupported input format: {input_ext}")
        print("Supported formats: .mp4, .avi, .mov, .mkv, .jpg, .jpeg, .png, .bmp")
        return
    
    if success:
        print(f"🎉 Processing completed successfully!")
        print(f"📁 Output saved to: {args.output}")
    else:
        print(f"❌ Processing failed")


if __name__ == "__main__":
    main()
