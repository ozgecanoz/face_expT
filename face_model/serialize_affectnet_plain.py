#!/usr/bin/env python3
"""
Serialize AffectNet YOLO Format Dataset (Plain Version)

This script processes the AffectNet YOLO format dataset by:
1. Reading YOLO format images and labels
2. Using MediaPipe for face detection and cropping
3. Upsampling images to 518x518 resolution
4. Creating H5 files with 30 images each containing:
   - 'frames' key with cropped face images
   - Emotion labels per frame in metadata
5. Generating MP4 videos of the face clips

This is a simplified version without DINOv2 feature extraction or PCA projection.
It's designed to create the raw data needed for combined PCA computation later.

Usage:
    python serialize_affectnet_plain.py --input-dir /path/to/affectnet --output-dir /path/to/output --split train
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import cv2
import mediapipe as mp
import h5py
from tqdm import tqdm
from datetime import datetime

# Add the project root to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from data.affectnet_yolo_dataset import AffectNetYOLODataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AffectNetPlainSerializer:
    """
    Serializes AffectNet YOLO format dataset with face detection and cropping only.
    Creates H5 files with frames and emotion labels, plus MP4 videos.
    """
    
    def __init__(self, 
                 input_dir: str,
                 output_dir: str,
                 batch_size: int = 30,
                 target_resolution: int = 518,
                 create_videos: bool = True,
                 video_fps: float = 30.0,
                 face_detection_confidence: float = 0.5):
        """
        Initialize the AffectNet plain serializer.
        
        Args:
            input_dir: Path to AffectNet YOLO format dataset directory
            output_dir: Path to output directory for H5 files and videos
            batch_size: Number of images per H5 file (default: 30)
            target_resolution: Target resolution for face images (default: 518)
            create_videos: Whether to create MP4 videos
            video_fps: FPS for output videos
            face_detection_confidence: MediaPipe face detection confidence threshold (0.0-1.0, default: 0.5)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.target_resolution = target_resolution
        self.create_videos = create_videos
        self.video_fps = video_fps
        self.face_detection_confidence = face_detection_confidence
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize MediaPipe
        self._setup_mediapipe()
        
        logger.info(f"✅ AffectNet plain serializer initialized")
        logger.info(f"   Target resolution: {target_resolution}x{target_resolution}")
        logger.info(f"   Batch size: {batch_size}")
        logger.info(f"   Face detection confidence: {face_detection_confidence}")
    
    def _setup_mediapipe(self):
        """Initialize MediaPipe face detection."""
        try:
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_drawing = mp.solutions.drawing_utils
            
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1,  # 0 for short-range, 1 for full-range
                min_detection_confidence=self.face_detection_confidence
            )
            
            logger.info("✅ MediaPipe face detection initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize MediaPipe: {e}")
            raise
    
    def detect_and_crop_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect and crop face from image using MediaPipe.
        
        Args:
            image: Input image (RGB format)
            
        Returns:
            Cropped face image or None if no face detected
        """
        try:
            # Convert to RGB if needed
            if image.shape[2] == 3:
                image_rgb = image
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            results = self.face_detection.process(image_rgb)
            
            if not results.detections:
                logger.warning("No face detected in image")
                return None
            
            # Get the first (and presumably only) face
            detection = results.detections[0]
            
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
            
            # Resize to target resolution
            face_resized = cv2.resize(face_crop, (self.target_resolution, self.target_resolution))
            
            return face_resized
            
        except Exception as e:
            logger.error(f"Error in face detection/cropping: {e}")
            return None
    
    def create_face_clip_video(self, 
                              frames: np.ndarray,
                              batch_info: List[Dict[str, Any]],
                              output_path: str) -> bool:
        """
        Create MP4 video from face frames with emotion class names as text overlays.
        
        Args:
            frames: Face frames (N, C, H, W)
            batch_info: List of sample information containing emotion labels
            output_path: Output video path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.create_videos:
                return True
            
            num_frames = frames.shape[0]
            
            # Get video dimensions
            height = self.target_resolution
            width = self.target_resolution
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, self.video_fps, (width, height))
            
            for i in range(num_frames):
                # Get frame
                frame = frames[i]  # (C, H, W)
                
                # Convert frame to (H, W, C) - frames are already uint8 [0, 255]
                frame_hwc = frame.transpose(1, 2, 0)  # (H, W, C)
                frame_uint8 = frame_hwc  # Already uint8, no conversion needed
                
                # Add emotion class name as text overlay
                emotion_name = batch_info[i]['emotion_class_name']
                emotion_id = batch_info[i]['emotion_class_id']
                
                # Create text with emotion name and ID
                text = f"{emotion_name} ({emotion_id})"
                
                # Set text properties
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
                font_thickness = 2
                text_color = (255, 255, 255)  # White text
                outline_color = (0, 0, 0)     # Black outline
                
                # Get text size for positioning
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
                
                # Position text at top-left with some padding
                text_x = 20
                text_y = text_height + 20
                
                # Draw black outline first (for better visibility)
                cv2.putText(frame_uint8, text, (text_x, text_y), font, font_scale, 
                           outline_color, font_thickness + 1)
                
                # Draw white text on top
                cv2.putText(frame_uint8, text, (text_x, text_y), font, font_scale, 
                           text_color, font_thickness)
                
                # Write frame
                out.write(frame_uint8)
            
            # Release video writer
            out.release()
            
            logger.info(f"✅ Face clip video with emotion labels created: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating face clip video: {e}")
            return False
    
    def save_frames_to_h5(self, 
                          batch_frames: np.ndarray,
                          batch_info: List[Dict[str, Any]],
                          output_path: str) -> bool:
        """
        Save batch frames to H5 file matching CCA dataset format.
        
        Args:
            batch_frames: Batch of face frames (N, C, H, W)
            batch_info: List of sample information
            output_path: Output H5 file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with h5py.File(output_path, 'w') as f:
                # Create groups matching CCA format
                faces_group = f.create_group('faces')
                metadata_group = f.create_group('metadata')
                
                # Store face data in CCA format: faces/frame_XXX/face_000/data
                num_frames = batch_frames.shape[0]
                for i in range(num_frames):
                    # Create frame group
                    frame_group = faces_group.create_group(f'frame_{i:03d}')
                    
                    # Create face group (single face per frame)
                    face_group = frame_group.create_group('face_000')
                    
                    # Store face data (convert from CxHxW to HxWxC for consistency)
                    frame_data = batch_frames[i].transpose(1, 2, 0)  # (H, W, C)
                    face_group.create_dataset('data', data=frame_data, compression='gzip', compression_opts=9)
                    
                    # Store dummy bbox (full frame since we're already cropped)
                    face_group.create_dataset('bbox', data=[0, 0, self.target_resolution, self.target_resolution])
                    
                    # Store confidence (MediaPipe default)
                    face_group.create_dataset('confidence', data=0.5)
                    
                    # Store original size (same as target for cropped images)
                    face_group.create_dataset('original_size', data=[self.target_resolution, self.target_resolution])
                
                # Store metadata matching CCA format
                metadata_group.create_dataset('num_frames', data=num_frames)
                metadata_group.create_dataset('face_size', data=[self.target_resolution, self.target_resolution])
                metadata_group.create_dataset('fps', data=30.0)  # Default for image sequences
                metadata_group.create_dataset('extraction_date', data=datetime.now().isoformat())
                
                # Store subject ID (dummy for consistency with original dataset)
                metadata_group.create_dataset('subject_id', data='affectnet_0')
                
                # Store emotion information per frame
                emotion_class_ids = [info['emotion_class_id'] for info in batch_info]
                emotion_class_names = [info['emotion_class_name'] for info in batch_info]
                
                metadata_group.create_dataset('emotion_class_ids', data=emotion_class_ids)
                metadata_group.create_dataset('emotion_class_names', data=emotion_class_names)
                
                # Store original paths
                image_paths = [info['image_path'] for info in batch_info]
                metadata_group.create_dataset('original_image_paths', data=image_paths)
                
                # Store processing info
                metadata_group.create_dataset('processing_timestamp', data=datetime.now().isoformat())
                metadata_group.create_dataset('mediapipe_confidence', data=0.5)  # Default confidence threshold
                metadata_group.create_dataset('dataset_type', data='affectnet_plain')
            
            logger.debug(f"Frames saved to H5 file in CCA format: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving frames to {output_path}: {e}")
            return False
    
    def serialize_dataset(self, split: str, max_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Serialize the AffectNet dataset split.
        
        Args:
            split: Dataset split ('train', 'valid', 'test')
            max_samples: Maximum number of samples to process (for debugging)
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"🚀 Starting AffectNet plain serialization for {split} split...")
        logger.info(f"📋 Each H5 file will contain exactly {self.batch_size} frames (no missing frames)")
        logger.info(f"🔄 Loading all images into memory, then grouping successful face detections")
        start_time = datetime.now()
        
        # Load dataset
        dataset = AffectNetYOLODataset(self.input_dir, split, max_samples)
        
        # Get emotion class information
        class_info = dataset.get_emotion_class_info()
        logger.info(f"Dataset info: {class_info}")
        
        # Load ALL images into memory and detect faces
        total_samples = len(dataset)
        logger.info(f"Loading all {total_samples} images into memory...")
        all_samples = list(dataset)  # Load all samples at once
        
        logger.info("Detecting faces in all images...")
        successful_detections = []
        
        # Process all images with face detection
        for sample in tqdm(all_samples, desc="Face detection"):
            face_crop = self.detect_and_crop_face(sample['image'])
            if face_crop is not None:
                successful_detections.append({
                    'face_crop': face_crop,
                    'sample_info': sample
                })
        
        total_faces_detected = len(successful_detections)
        logger.info(f"✅ Face detection completed: {total_faces_detected} faces found out of {total_samples} images")
        
        # Group successful detections into batches of exactly batch_size
        num_complete_batches = total_faces_detected // self.batch_size
        total_frames_processed = num_complete_batches * self.batch_size
        
        logger.info(f"📦 Creating {num_complete_batches} complete batches of {self.batch_size} frames each")
        logger.info(f"📊 Total frames to process: {total_frames_processed}")
        logger.info(f"🗑️  Discarding {total_faces_detected % self.batch_size} remaining frames (incomplete batch)")
        
        # Statistics
        successful_batches = 0
        failed_batches = 0
        
        # Create progress bar for batch processing
        progress_bar = tqdm(range(num_complete_batches), desc="Creating H5 files")
        
        for batch_idx in progress_bar:
            try:
                # Get batch of exactly batch_size faces
                start_idx = batch_idx * self.batch_size
                end_idx = start_idx + self.batch_size
                batch_detections = successful_detections[start_idx:end_idx]
                
                # Extract face crops and sample info
                processed_images = [det['face_crop'] for det in batch_detections]
                batch_info = [det['sample_info'] for det in batch_detections]
                
                # Validate we have exactly the right number of frames
                assert len(processed_images) == self.batch_size, f"Expected {self.batch_size} frames, got {len(processed_images)}"
                
                # Convert to numpy array
                batch_frames = np.stack(processed_images, axis=0)  # (N, H, W, C)
                
                # Keep in uint8 [0, 255] range to match CCA format exactly
                batch_frames = batch_frames.astype(np.uint8)
                
                # Debug: log frame ranges
                logger.debug(f"Batch {batch_idx}: Frame range: [{batch_frames.min()}, {batch_frames.max()}]")
                logger.debug(f"Batch {batch_idx}: Exactly {len(processed_images)} frames, shape: {batch_frames.shape}")
                
                # Convert to (N, C, H, W) format for consistency
                batch_frames = batch_frames.transpose(0, 3, 1, 2)  # (N, C, H, W)
                
                # Create output filename
                output_filename = f"yolo_subject_{split}_{batch_idx:04d}_frames.h5"
                output_path = self.output_dir / output_filename
                
                # Save frames
                success = self.save_frames_to_h5(batch_frames, batch_info, str(output_path))
                
                if success:
                    # Create MP4 video
                    if self.create_videos:
                        video_path = str(output_path).replace('.h5', '_clip.mp4')
                        self.create_face_clip_video(batch_frames, batch_info, video_path)
                    
                    successful_batches += 1
                    
                    # Update progress
                    progress_bar.set_postfix({
                        'successful': successful_batches,
                        'failed': failed_batches,
                        'frames': total_frames_processed
                    })
                else:
                    failed_batches += 1
                
            except Exception as e:
                logger.error(f"❌ Error processing batch {batch_idx}: {e}")
                failed_batches += 1
                continue
        
        # Final statistics
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        results = {
            'split': split,
            'total_samples': total_samples,
            'successful_batches': successful_batches,
            'failed_batches': failed_batches,
            'total_frames_processed': total_frames_processed,
            'total_faces_detected': total_faces_detected,
            'processing_time_seconds': processing_time,
            'output_directory': str(self.output_dir)
        }
        
        logger.info(f"✅ Plain serialization completed for {split} split")
        logger.info(f"   Successful batches: {successful_batches}")
        logger.info(f"   Failed batches: {failed_batches}")
        logger.info(f"   Total frames processed: {total_frames_processed}")
        logger.info(f"   Total faces detected: {total_faces_detected}")
        logger.info(f"   Processing time: {processing_time:.2f} seconds")
        
        return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Serialize AffectNet YOLO format dataset with face detection and cropping only")
    
    # Required arguments
    parser.add_argument("--input-dir", type=str, required=True,
                       help="Path to AffectNet YOLO format dataset directory")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Path to output directory for H5 files and videos")
    parser.add_argument("--split", type=str, required=True, choices=["train", "valid", "test"],
                       help="Dataset split to process")
    
    # Optional arguments
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of samples to process (for debugging)")
    parser.add_argument("--batch-size", type=int, default=30,
                       help="Number of images per H5 file (default: 30)")
    parser.add_argument("--target-resolution", type=int, default=518,
                       help="Target resolution for face images (default: 518)")
    parser.add_argument("--no-videos", action="store_true",
                       help="Disable video creation")
    parser.add_argument("--video-fps", type=float, default=30.0,
                       help="FPS for output videos (default: 30.0)")
    parser.add_argument("--face-detection-confidence", type=float, default=0.5,
                       help="MediaPipe face detection confidence threshold (0.0-1.0, default: 0.5)")
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"❌ Error: Input directory not found: {args.input_dir}")
        return
    
    print("🚀 Starting AffectNet YOLO Format Plain Serialization")
    print("=" * 60)
    
    # Create serializer
    serializer = AffectNetPlainSerializer(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        target_resolution=args.target_resolution,
        create_videos=not args.no_videos,
        video_fps=args.video_fps,
        face_detection_confidence=args.face_detection_confidence
    )
    
    # Serialize dataset
    results = serializer.serialize_dataset(args.split, args.max_samples)
    
    # Print final results
    print(f"\n🎉 Plain serialization completed successfully!")
    print(f"   Split: {results['split']}")
    print(f"   Output directory: {results['output_directory']}")
    print(f"   Successful batches: {results['successful_batches']}")
    print(f"   Failed batches: {results['failed_batches']}")
    print(f"   Total frames processed: {results['total_frames_processed']}")
    print(f"   Total faces detected: {results['total_faces_detected']}")
    print(f"   Processing time: {results['processing_time_seconds']:.2f} seconds")


if __name__ == "__main__":
    main()
