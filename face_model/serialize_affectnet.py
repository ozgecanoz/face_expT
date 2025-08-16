#!/usr/bin/env python3
"""
Serialize AffectNet YOLO Format Dataset

This script processes the AffectNet YOLO format dataset by:
1. Reading YOLO format images and labels
2. Using MediaPipe for face detection and cropping
3. Upsampling images to 518x518 resolution
4. Extracting DINOv2 features using DINOv2BaseTokenizer
5. Projecting features to 384 dimensions using PCA
6. Creating H5 files with 30 images each
7. Generating side-by-side videos (cropped faces + PCA visualizations)

Usage:
    python serialize_affectnet.py --input-dir /path/to/affectnet --output-dir /path/to/output --split train
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import h5py
import cv2
import mediapipe as mp
from tqdm import tqdm
from datetime import datetime

# Add the project root to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from data.affectnet_yolo_dataset import AffectNetYOLODataset
from models.dinov2_tokenizer import DINOv2BaseTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AffectNetSerializer:
    """
    Serializes AffectNet YOLO format dataset with face detection, feature extraction, and PCA projection.
    """
    
    def __init__(self, 
                 input_dir: str,
                 output_dir: str,
                 pca_directions_path: str,
                 device: str = "auto",
                 batch_size: int = 30,
                 target_resolution: int = 518,
                 projected_dim: int = 384,
                 create_videos: bool = True,
                 video_fps: float = 30.0):
        """
        Initialize the AffectNet serializer.
        
        Args:
            input_dir: Path to AffectNet YOLO format dataset directory
            output_dir: Path to output directory for H5 files and videos
            pca_directions_path: Path to PCA directions JSON file
            device: Device to use for processing (auto/cuda/cpu)
            batch_size: Number of images per H5 file (default: 30)
            target_resolution: Target resolution for face images (default: 518)
            projected_dim: Target dimension for PCA projection (default: 384)
            create_videos: Whether to create side-by-side videos
            video_fps: FPS for output videos
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.pca_directions_path = Path(pca_directions_path)
        self.batch_size = batch_size
        self.target_resolution = target_resolution
        self.projected_dim = projected_dim
        self.create_videos = create_videos
        self.video_fps = video_fps
        
        # Auto-detect device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize MediaPipe
        self._setup_mediapipe()
        
        # Load DINOv2 tokenizer
        self._load_dinov2_tokenizer()
        
        # Load PCA directions
        self._load_pca_directions()
        
        logger.info(f"✅ AffectNet serializer initialized")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Target resolution: {target_resolution}x{target_resolution}")
        logger.info(f"   Projected dimension: {projected_dim}")
        logger.info(f"   Batch size: {batch_size}")
    
    def _setup_mediapipe(self):
        """Initialize MediaPipe face detection."""
        try:
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_drawing = mp.solutions.drawing_utils
            
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1,  # 0 for short-range, 1 for full-range
                min_detection_confidence=0.5
            )
            
            logger.info("✅ MediaPipe face detection initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize MediaPipe: {e}")
            raise
    
    def _load_dinov2_tokenizer(self):
        """Load DINOv2 base tokenizer."""
        try:
            self.tokenizer = DINOv2BaseTokenizer(device=self.device)
            logger.info("✅ DINOv2 base tokenizer loaded")
            
        except Exception as e:
            logger.error(f"❌ Failed to load DINOv2 tokenizer: {e}")
            raise
    
    def _load_pca_directions(self):
        """Load PCA directions from JSON file."""
        try:
            with open(self.pca_directions_path, 'r') as f:
                pca_data = json.load(f)
            
            self.pca_directions = np.array(pca_data['pca_components'])
            self.pca_explained_variance = np.array(pca_data['pca_explained_variance'])
            
            # Ensure we have enough components
            if self.pca_directions.shape[0] < self.projected_dim:
                raise ValueError(f"PCA directions have {self.pca_directions.shape[0]} components, but {self.projected_dim} requested")
            
            # Take top components
            self.pca_directions = self.pca_directions[:self.projected_dim]
            self.pca_explained_variance = self.pca_explained_variance[:self.projected_dim]
            
            logger.info(f"✅ PCA directions loaded: {self.pca_directions.shape}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load PCA directions: {e}")
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
    
    def extract_features(self, images: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Extract DINOv2 features from a batch of images.
        
        Args:
            images: List of face images
            
        Returns:
            Dictionary with features and metadata
        """
        try:
            # Convert to tensors
            image_tensors = []
            for img in images:
                # Convert to tensor and normalize
                img_tensor = torch.from_numpy(img).float() / 255.0  # [0, 1] range
                img_tensor = img_tensor.permute(2, 0, 1)  # (C, H, W)
                image_tensors.append(img_tensor)
            
            # Stack into batch
            batch_tensor = torch.stack(image_tensors, dim=0)  # (N, C, H, W)
            batch_tensor = batch_tensor.to(self.device)
            
            # Extract features
            with torch.no_grad():
                patch_tokens, pos_embeddings = self.tokenizer(batch_tensor)
            
            # Project to PCA space
            projected_features = self._project_to_pca(patch_tokens)
            
            return {
                'patch_tokens': patch_tokens.cpu().numpy(),
                'projected_features': projected_features,
                'frames': batch_tensor.cpu().numpy()
            }
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            raise
    
    def _project_to_pca(self, patch_tokens: torch.Tensor) -> np.ndarray:
        """
        Project patch tokens to PCA space.
        
        Args:
            patch_tokens: Patch tokens tensor (N, num_patches, token_dim)
            
        Returns:
            Projected features (N, projected_dim)
        """
        try:
            # Reshape to (N * num_patches, token_dim)
            N, num_patches, token_dim = patch_tokens.shape
            tokens_flat = patch_tokens.reshape(-1, token_dim)
            
            # Project using PCA directions
            projected = tokens_flat @ self.pca_directions.T  # (N * num_patches, projected_dim)
            
            # Reshape back to (N, num_patches, projected_dim)
            projected = projected.reshape(N, num_patches, self.projected_dim)
            
            return projected.cpu().numpy()
            
        except Exception as e:
            logger.error(f"Error in PCA projection: {e}")
            raise
    
    def create_pca_visualization(self, pca_features: np.ndarray, top_components: int = 3) -> np.ndarray:
        """
        Create RGB visualization from top PCA components.
        
        Args:
            pca_features: PCA features (num_patches, projected_dim)
            top_components: Number of top components to use
            
        Returns:
            RGB visualization image
        """
        try:
            # Take top components
            top_features = pca_features[:, :top_components]
            
            # Normalize to [0, 1] range
            features_norm = (top_features - top_features.min()) / (top_features.max() - top_features.min() + 1e-8)
            
            # Reshape to spatial dimensions (assuming square patches)
            num_patches = features_norm.shape[0]
            patch_size = int(np.sqrt(num_patches))
            
            if patch_size * patch_size != num_patches:
                # If not perfect square, pad or crop
                patch_size = int(np.sqrt(num_patches))
                if patch_size * patch_size < num_patches:
                    patch_size += 1
            
            # Reshape to (patch_size, patch_size, top_components)
            features_spatial = features_norm[:patch_size*patch_size].reshape(patch_size, patch_size, top_components)
            
            # Convert to uint8 for visualization
            features_uint8 = (features_spatial * 255).astype(np.uint8)
            
            # Resize to target resolution
            features_resized = cv2.resize(features_uint8, (self.target_resolution, self.target_resolution))
            
            return features_resized
            
        except Exception as e:
            logger.error(f"Error creating PCA visualization: {e}")
            raise
    
    def create_side_by_side_video(self, 
                                 frames: np.ndarray, 
                                 pca_features: np.ndarray,
                                 output_path: str) -> bool:
        """
        Create side-by-side video showing cropped faces and PCA visualizations.
        
        Args:
            frames: Face frames (N, C, H, W)
            pca_features: PCA features (N, num_patches, projected_dim)
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
            width = self.target_resolution * 2  # Side by side
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, self.video_fps, (width, height))
            
            for i in range(num_frames):
                # Get frame and PCA features
                frame = frames[i]  # (C, H, W)
                pca_feat = pca_features[i]  # (num_patches, projected_dim)
                
                # Convert frame to (H, W, C) and uint8
                frame_hwc = frame.transpose(1, 2, 0)  # (H, W, C)
                frame_uint8 = (frame_hwc * 255).astype(np.uint8)
                
                # Create PCA visualization
                pca_viz = self.create_pca_visualization(pca_feat)
                
                # Create side-by-side frame
                side_by_side = np.hstack([frame_uint8, pca_viz])
                
                # Write frame
                out.write(side_by_side)
            
            # Release video writer
            out.release()
            
            logger.info(f"✅ Side-by-side video created: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating side-by-side video: {e}")
            return False
    
    def save_features_to_h5(self, 
                           batch_features: Dict[str, np.ndarray],
                           batch_info: List[Dict[str, Any]],
                           output_path: str) -> bool:
        """
        Save batch features to H5 file.
        
        Args:
            batch_features: Dictionary with features and frames
            batch_info: List of sample information
            output_path: Output H5 file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with h5py.File(output_path, 'w') as f:
                # Create groups
                data_group = f.create_group('data')
                metadata_group = f.create_group('metadata')
                
                # Store feature data
                data_group.create_dataset('frames', 
                                        data=batch_features['frames'], 
                                        compression='gzip', 
                                        compression_opts=9)
                
                data_group.create_dataset('projected_features', 
                                        data=batch_features['projected_features'], 
                                        compression='gzip', 
                                        compression_opts=9)
                
                # Store metadata
                metadata_group.create_dataset('num_samples', data=len(batch_info))
                metadata_group.create_dataset('target_resolution', data=self.target_resolution)
                metadata_group.create_dataset('projected_dim', data=self.projected_dim)
                
                # Store emotion information
                emotion_class_ids = [info['emotion_class_id'] for info in batch_info]
                emotion_class_names = [info['emotion_class_name'].encode('utf-8') for info in batch_info]
                
                metadata_group.create_dataset('emotion_class_ids', data=emotion_class_ids)
                metadata_group.create_dataset('emotion_class_names', data=emotion_class_names)
                
                # Store original paths
                image_paths = [info['image_path'].encode('utf-8') for info in batch_info]
                metadata_group.create_dataset('original_image_paths', data=image_paths)
                
                # Store PCA information
                metadata_group.create_dataset('pca_components_used', data=self.projected_dim)
                metadata_group.create_dataset('pca_explained_variance_total', 
                                          data=np.sum(self.pca_explained_variance))
            
            logger.debug(f"Features saved to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving features to {output_path}: {e}")
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
        logger.info(f"🚀 Starting AffectNet serialization for {split} split...")
        start_time = datetime.now()
        
        # Load dataset
        dataset = AffectNetYOLODataset(self.input_dir, split, max_samples)
        
        # Get emotion class information
        class_info = dataset.get_emotion_class_info()
        logger.info(f"Dataset info: {class_info}")
        
        # Process in batches
        total_samples = len(dataset)
        num_batches = (total_samples + self.batch_size - 1) // self.batch_size
        
        logger.info(f"Processing {total_samples} samples in {num_batches} batches...")
        
        # Statistics
        successful_batches = 0
        failed_batches = 0
        total_frames_processed = 0
        
        # Create progress bar
        progress_bar = tqdm(range(num_batches), desc=f"Processing {split} batches")
        
        for batch_idx in progress_bar:
            try:
                # Get batch of samples
                batch_samples = dataset.get_sample_batch(self.batch_size)
                
                # Process images
                processed_images = []
                batch_info = []
                
                for sample in batch_samples:
                    # Detect and crop face
                    face_crop = self.detect_and_crop_face(sample['image'])
                    
                    if face_crop is not None:
                        processed_images.append(face_crop)
                        batch_info.append(sample)
                    else:
                        logger.warning(f"No face detected in {sample['image_path']}")
                
                if len(processed_images) == 0:
                    logger.warning(f"Batch {batch_idx}: No valid faces found")
                    failed_batches += 1
                    continue
                
                # Extract features
                batch_features = self.extract_features(processed_images)
                
                # Create output filename
                output_filename = f"yolo_subject_{split}_{batch_idx:04d}_features.h5"
                output_path = self.output_dir / output_filename
                
                # Save features
                success = self.save_features_to_h5(batch_features, batch_info, str(output_path))
                
                if success:
                    # Create side-by-side video
                    if self.create_videos:
                        video_path = str(output_path).replace('.h5', '_visualization.mp4')
                        self.create_side_by_side_video(
                            batch_features['frames'],
                            batch_features['projected_features'],
                            video_path
                        )
                    
                    successful_batches += 1
                    total_frames_processed += len(processed_images)
                    
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
            'processing_time_seconds': processing_time,
            'output_directory': str(self.output_dir)
        }
        
        logger.info(f"✅ Serialization completed for {split} split")
        logger.info(f"   Successful batches: {successful_batches}")
        logger.info(f"   Failed batches: {failed_batches}")
        logger.info(f"   Total frames processed: {total_frames_processed}")
        logger.info(f"   Processing time: {processing_time:.2f} seconds")
        
        return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Serialize AffectNet YOLO format dataset with face detection and feature extraction")
    
    # Required arguments
    parser.add_argument("--input-dir", type=str, required=True,
                       help="Path to AffectNet YOLO format dataset directory")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Path to output directory for H5 files and videos")
    parser.add_argument("--pca-directions", type=str, required=True,
                       help="Path to PCA directions JSON file")
    parser.add_argument("--split", type=str, required=True, choices=["train", "valid", "test"],
                       help="Dataset split to process")
    
    # Optional arguments
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of samples to process (for debugging)")
    parser.add_argument("--batch-size", type=int, default=30,
                       help="Number of images per H5 file (default: 30)")
    parser.add_argument("--target-resolution", type=int, default=518,
                       help="Target resolution for face images (default: 518)")
    parser.add_argument("--projected-dim", type=int, default=384,
                       help="Target dimension for PCA projection (default: 384)")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto/cuda/cpu)")
    parser.add_argument("--no-videos", action="store_true",
                       help="Disable video creation")
    parser.add_argument("--video-fps", type=float, default=30.0,
                       help="FPS for output videos (default: 30.0)")
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"❌ Error: Input directory not found: {args.input_dir}")
        return
    
    # Validate PCA directions file
    if not os.path.exists(args.pca_directions):
        print(f"❌ Error: PCA directions file not found: {args.pca_directions}")
        return
    
    print("🚀 Starting AffectNet YOLO Format Serialization")
    print("=" * 60)
    
    # Create serializer
    serializer = AffectNetSerializer(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        pca_directions_path=args.pca_directions,
        device=args.device,
        batch_size=args.batch_size,
        target_resolution=args.target_resolution,
        projected_dim=args.projected_dim,
        create_videos=not args.no_videos,
        video_fps=args.video_fps
    )
    
    # Serialize dataset
    results = serializer.serialize_dataset(args.split, args.max_samples)
    
    # Print final results
    print(f"\n🎉 Serialization completed successfully!")
    print(f"   Split: {results['split']}")
    print(f"   Output directory: {results['output_directory']}")
    print(f"   Successful batches: {results['successful_batches']}")
    print(f"   Failed batches: {results['failed_batches']}")
    print(f"   Total frames processed: {results['total_frames_processed']}")
    print(f"   Processing time: {results['processing_time_seconds']:.2f} seconds")


if __name__ == "__main__":
    main()
