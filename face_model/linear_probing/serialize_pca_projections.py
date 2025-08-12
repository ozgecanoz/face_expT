#!/usr/bin/env python3
"""
Serialize PCA Projections Script
Reads face dataset H5 files, extracts DINOv2 features using DINOv2BaseTokenizer,
projects features to 384 dimensions using PCA directions, and saves PCA-projected
features to new H5 files for downstream processing.

This script also creates side-by-side videos showing the original frames and
PCA visualizations (first 3 projected features as RGB values) for each clip.

The output H5 files contain only the PCA-projected features (384-dim) and metadata,
keeping file sizes minimal while preserving all necessary information for training.

This script is designed to work with the output from compute_pca_directions.py
and creates feature files that can be used for expression transformer training.
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
from torch.utils.data import DataLoader
from tqdm import tqdm
import h5py
from datetime import datetime

# Try to import OpenCV for video creation
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    logging.warning("OpenCV not available. Video creation will be disabled.")

# Add the project root to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.dataset import FaceDataset
from models.dinov2_tokenizer import DINOv2BaseTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PCAProjectionSerializer:
    """
    Serializes face datasets by extracting DINOv2 features, projecting them to PCA space,
    and saving both original and projected features to H5 files
    """
    
    def __init__(self, 
                 input_dataset_path: str,
                 output_dataset_path: str,
                 pca_directions_path: str,
                 device: str = "cuda",
                 batch_size: int = 8,
                 num_workers: int = 0,
                 pin_memory: bool = True,
                 persistent_workers: bool = False,
                 projected_dim: int = 384,
                 create_videos: bool = True,
                 video_fps: float = 30.0,
                 save_frames: bool = False):
        """
        Initialize the PCA projection serializer
        
        Args:
            input_dataset_path: Path to input dataset directory containing H5 files
            output_dataset_path: Path to output dataset directory for feature H5 files
            pca_directions_path: Path to PCA directions JSON file from compute_pca_directions.py
            device: Device to use for processing (cuda/cpu)
            batch_size: Batch size for processing
            num_workers: Number of data loader workers
            pin_memory: Whether to pin memory for GPU processing
            persistent_workers: Whether to use persistent workers
            projected_dim: Target dimension for PCA projection (default: 384)
            create_videos: Whether to create side-by-side videos (default: True)
            video_fps: FPS for output videos (default: 30.0)
            save_frames: Whether to save original frames to H5 files (default: False)
        """
        self.input_dataset_path = input_dataset_path
        self.output_dataset_path = output_dataset_path
        self.pca_directions_path = pca_directions_path
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.projected_dim = projected_dim
        self.create_videos = create_videos
        self.video_fps = video_fps
        self.save_frames = save_frames
        
        # Create output directory
        os.makedirs(output_dataset_path, exist_ok=True)
        
        # Check OpenCV availability for video creation
        if create_videos and not OPENCV_AVAILABLE:
            logger.warning("OpenCV not available. Disabling video creation.")
            create_videos = False
        
        # Load PCA directions
        self._load_pca_directions()
        
        # Initialize DINOv2 tokenizer
        logger.info(f"Initializing DINOv2BaseTokenizer on device: {device}")
        self.tokenizer = DINOv2BaseTokenizer(device=device)
        
        # Initialize counters
        self.processed_clips = 0
        self.successful_features = 0
        self.failed_clips = 0
        self.total_frames = 0
        self.successful_videos = 0
        self.failed_videos = 0
        
        # Validate tokenizer configuration
        self.embed_dim = self.tokenizer.get_embed_dim()
        self.num_patches = self.tokenizer.get_num_patches()
        self.input_size = self.tokenizer.get_input_size()
        
        # Validate PCA compatibility
        if self.embed_dim != self.original_embed_dim:
            raise ValueError(f"PCA embed dim ({self.original_embed_dim}) doesn't match tokenizer ({self.embed_dim})")
        
        logger.info(f"Tokenizer initialized:")
        logger.info(f"  Embedding dimension: {self.embed_dim}")
        logger.info(f"  Number of patches: {self.num_patches}")
        logger.info(f"  Input size: {self.input_size}x{self.input_size}")
        logger.info(f"  PCA projection: {self.embed_dim} -> {self.projected_dim}")
        
        # Statistics tracking
        # self.processed_clips = 0 # Moved to __init__
        # self.successful_features = 0 # Moved to __init__
        # self.failed_clips = 0 # Moved to __init__
        # self.total_frames = 0 # Moved to __init__
    
    def _load_pca_directions(self):
        """Load PCA directions from JSON file"""
        logger.info(f"Loading PCA directions from: {self.pca_directions_path}")
        
        try:
            with open(self.pca_directions_path, 'r') as f:
                pca_data = json.load(f)
            
            # Extract PCA components and metadata
            self.pca_components = np.array(pca_data['pca_components'])  # (n_components, embed_dim)
            self.pca_mean = np.array(pca_data['pca_mean'])  # (embed_dim,)
            self.pca_explained_variance = np.array(pca_data['pca_explained_variance'])  # (n_components,)
            
            # Validate PCA data
            self.n_components = self.pca_components.shape[0]
            self.original_embed_dim = self.pca_components.shape[1]
            
            # Validate projected dimension
            if self.projected_dim > self.n_components:
                logger.warning(f"Requested projected_dim ({self.projected_dim}) exceeds available PCA components ({self.n_components})")
                self.projected_dim = self.n_components
            
            logger.info(f"PCA directions loaded:")
            logger.info(f"  Components: {self.pca_components.shape}")
            logger.info(f"  Original embed dim: {self.original_embed_dim}")
            logger.info(f"  Target projected dim: {self.projected_dim}")
            logger.info(f"  Available components: {self.n_components}")
            logger.info(f"  Explained variance (top 10): {self.pca_explained_variance[:10].sum():.3f}")
            
        except Exception as e:
            logger.error(f"Failed to load PCA directions: {e}")
            raise
    
    def project_features_to_pca(self, features: np.ndarray) -> np.ndarray:
        """
        Project features to PCA space
        
        Args:
            features: (N, embed_dim) array of features
            
        Returns:
            projected_features: (N, projected_dim) array of projected features
        """
        # Center features using PCA mean
        centered_features = features - self.pca_mean
        
        # Project to PCA space using top components
        components_to_use = self.pca_components[:self.projected_dim]  # (projected_dim, embed_dim)
        
        # Project: (N, embed_dim) @ (embed_dim, projected_dim) = (N, projected_dim)
        projected_features = centered_features @ components_to_use.T
        
        logger.debug(f"Projected features from {features.shape} to {projected_features.shape}")
        return projected_features
    
    def load_dataset(self, max_samples: Optional[int] = None) -> DataLoader:
        """
        Load the face dataset using the same pattern as training scripts
        
        Args:
            max_samples: Maximum number of samples to load (for debugging)
            
        Returns:
            DataLoader for the dataset
        """
        logger.info(f"Loading dataset from: {self.input_dataset_path}")
        
        # Create dataset using the same FaceDataset class used in training
        dataset = FaceDataset(self.input_dataset_path, max_samples=max_samples)
        
        # Create dataloader with same settings as training
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Keep order for reproducibility
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False
        )
        
        logger.info(f"Dataset loaded: {len(dataset)} samples, {len(dataloader)} batches")
        return dataloader
    
    def process_batch(self, batch: Dict, batch_idx: int) -> List[Dict]:
        """
        Process a batch of clips through DINOv2 tokenizer
        
        Args:
            batch: Batch from dataloader containing 'frames' and 'subject_id'
            batch_idx: Batch index for logging
            
        Returns:
            List of feature dictionaries for each clip
        """
        batch_features = []
        
        try:
            # Extract frames and subject IDs from batch
            frames_list = batch['frames']  # List[Tensor(30, 3, 518, 518)]
            subject_ids = batch['subject_id']  # List[str]
            file_paths = batch['file_path']  # List[str]
            
            # Process each clip in the batch
            for clip_idx, (frames, subject_id, file_path) in enumerate(zip(frames_list, subject_ids, file_paths)):
                try:
                    # Extract clip ID from file path
                    clip_id = self._extract_clip_id(file_path)
                    
                    # Process single clip
                    clip_features = self._process_single_clip(
                        frames, subject_id, clip_id, file_path
                    )
                    
                    if clip_features is not None:
                        batch_features.append(clip_features)
                        self.successful_features += 1
                    else:
                        self.failed_clips += 1
                        
                except Exception as e:
                    logger.error(f"Error processing clip {clip_idx} in batch {batch_idx}: {e}")
                    self.failed_clips += 1
                    continue
            
            # Update statistics
            self.processed_clips += len(frames_list)
            self.total_frames += sum(frames.shape[0] for frames in frames_list)
            
            logger.debug(f"Batch {batch_idx}: Processed {len(frames_list)} clips, "
                        f"successful: {len(batch_features)}, failed: {len(frames_list) - len(batch_features)}")
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_idx}: {e}")
            self.failed_clips += len(batch.get('frames', []))
        
        return batch_features
    
    def _process_single_clip(self, frames: torch.Tensor, subject_id: str, 
                            clip_id: str, file_path: str) -> Optional[Dict]:
        """
        Process a single clip through DINOv2 tokenizer and project to PCA space
        
        Args:
            frames: (30, 3, 518, 518) tensor of face frames
            subject_id: Subject identifier
            clip_id: Unique clip identifier
            file_path: Original H5 file path
            
        Returns:
            Dictionary containing features and metadata, or None if failed
        """
        try:
            # Validate input
            if frames.shape[0] != 30:
                logger.warning(f"Clip {clip_id}: Expected 30 frames, got {frames.shape[0]}")
                return None
            
            if frames.shape[1:] != (3, 518, 518):
                logger.warning(f"Clip {clip_id}: Expected shape (3, 518, 518), got {frames.shape[1:]}")
                return None
            
            # Move frames to device
            frames_device = frames.to(self.device)
            
            # Extract DINOv2 features using tokenizer
            with torch.no_grad():
                patch_tokens, pos_embeddings = self.tokenizer(frames_device)
            
            # Validate output shapes
            expected_shape = (30, self.num_patches, self.embed_dim)
            if patch_tokens.shape != expected_shape:
                logger.error(f"Clip {clip_id}: Expected patch_tokens shape {expected_shape}, got {patch_tokens.shape}")
                return None
            
            # Move features back to CPU for numpy operations
            patch_tokens_cpu = patch_tokens.cpu().numpy()
            pos_embeddings_cpu = pos_embeddings.cpu().numpy()
            
            # Project features to PCA space
            projected_features = []
            for frame_idx in range(patch_tokens_cpu.shape[0]):
                frame_tokens = patch_tokens_cpu[frame_idx]  # (1369, 768)
                
                # Project to PCA space: (1369, 768) -> (1369, 384)
                frame_projected = self.project_features_to_pca(frame_tokens)
                projected_features.append(frame_projected)
            
            # Stack projected features
            projected_features = np.stack(projected_features, axis=0)  # (30, 1369, 384)
            
            # Create feature dictionary
            clip_features = {
                'clip_id': clip_id,
                'subject_id': subject_id,
                'original_file_path': file_path,
                'frames': frames,  # (30, 3, 518, 518) - Original frames for video creation
                'patch_tokens': patch_tokens_cpu,  # (30, 1369, 768) - Original DINOv2 features
                'pos_embeddings': pos_embeddings_cpu,  # (30, 1369, 768) - Original positional embeddings
                'projected_features': projected_features,  # (30, 1369, 384) - PCA-projected features
                'metadata': {
                    'num_frames': 30,
                    'original_embed_dim': self.embed_dim,
                    'projected_dim': self.projected_dim,
                    'num_patches': self.num_patches,
                    'input_size': self.input_size,
                    'extraction_date': datetime.now().isoformat(),
                    'tokenizer_type': 'DINOv2BaseTokenizer',
                    'device_used': str(self.device),
                    'pca_directions_file': os.path.basename(self.pca_directions_path),
                    'pca_explained_variance': self.pca_explained_variance[:self.projected_dim].tolist()
                }
            }
            
            return clip_features
            
        except Exception as e:
            logger.error(f"Error processing clip {clip_id}: {e}")
            return None
    
    def _extract_clip_id(self, file_path: str) -> str:
        """
        Extract clip ID from H5 file path
        
        Args:
            file_path: Path to H5 file
            
        Returns:
            Clip identifier
        """
        # Extract filename without extension
        filename = os.path.basename(file_path)
        clip_id = os.path.splitext(filename)[0]
        return clip_id
    
    def save_features_to_h5(self, clip_features: Dict, output_path: str):
        """
        Save extracted features to H5 file
        
        Args:
            clip_features: Dictionary containing features and metadata
            output_path: Path to save the H5 file
        """
        try:
            with h5py.File(output_path, 'w') as f:
                # Create groups
                data_group = f.create_group('data')
                metadata_group = f.create_group('metadata')
                
                # Store feature data
                if self.save_frames:
                    data_group.create_dataset('frames', 
                                            data=clip_features['frames'], 
                                            compression='gzip', 
                                            compression_opts=9)
                # data_group.create_dataset('patch_tokens', 
                #                         data=clip_features['patch_tokens'], 
                #                         compression='gzip', 
                #                         compression_opts=9)
                # data_group.create_dataset('pos_embeddings', 
                #                         data=clip_features['pos_embeddings'], 
                #                         compression_opts=9)
                data_group.create_dataset('projected_features', 
                                        data=clip_features['projected_features'], 
                                        compression='gzip', 
                                        compression_opts=9)
                
                # Store metadata
                metadata_group.create_dataset('clip_id', data=clip_features['clip_id'])
                metadata_group.create_dataset('subject_id', data=clip_features['subject_id'])
                metadata_group.create_dataset('original_file_path', data=clip_features['original_file_path'])
                
                # Store metadata dictionary
                for key, value in clip_features['metadata'].items():
                    if isinstance(value, (int, float)):
                        metadata_group.create_dataset(key, data=value)
                    elif isinstance(value, list):
                        metadata_group.create_dataset(key, data=value)
                    else:
                        metadata_group.create_dataset(key, data=str(value))
                
                # Store shape information for easy access
                # metadata_group.create_dataset('patch_tokens_shape', data=clip_features['patch_tokens'].shape)
                # metadata_group.create_dataset('pos_embeddings_shape', data=clip_features['pos_embeddings'].shape)
                metadata_group.create_dataset('projected_features_shape', data=clip_features['projected_features'].shape)
                
                # Store PCA information
                metadata_group.create_dataset('pca_components_used', data=self.projected_dim)
                metadata_group.create_dataset('pca_explained_variance_total', 
                                          data=np.sum(self.pca_explained_variance[:self.projected_dim]))
            
            logger.debug(f"Features saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving features to {output_path}: {e}")
            raise
    
    def serialize_dataset(self, max_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Main serialization function
        
        Args:
            max_samples: Maximum number of samples to process (for debugging)
            
        Returns:
            Dictionary with processing results
        """
        logger.info("🚀 Starting PCA projection serialization...")
        start_time = datetime.now()
        
        # Load dataset
        dataloader = self.load_dataset(max_samples)
        
        # Process all batches
        total_batches = len(dataloader)
        logger.info(f"Processing {total_batches} batches...")
        
        # Create progress bar
        progress_bar = tqdm(dataloader, desc="Processing batches", total=total_batches)
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Process batch
                batch_features = self.process_batch(batch, batch_idx)
                
                # Save features for each clip
                for clip_features in batch_features:
                    clip_id = clip_features['clip_id']
                    subject_id = clip_features['subject_id']
                    
                    # Create output filename
                    output_filename = f"{clip_id}_features.h5"
                    output_path = os.path.join(self.output_dataset_path, output_filename)
                    
                    # Save features
                    self.save_features_to_h5(clip_features, output_path)
                    
                    # Create video visualization
                    if self.create_videos and OPENCV_AVAILABLE:
                        logger.info(f"🎬 Creating video for {clip_id}...")
                        try:
                            # Create base path for video (same as H5 but without extension)
                            video_base_path = os.path.join(self.output_dataset_path, clip_id)
                            video_path = self.create_video_for_clip(clip_features, video_base_path)
                            
                            if video_path:
                                logger.info(f"🎬 Video created for {clip_id}: {os.path.basename(video_path)}")
                                self.successful_videos += 1
                            else:
                                logger.warning(f"⚠️ Failed to create video for {clip_id}")
                                self.failed_videos += 1
                        except Exception as e:
                            logger.error(f"❌ Error creating video for {clip_id}: {e}")
                            self.failed_videos += 1
                            # Continue processing other clips even if video creation fails
                    elif self.create_videos and not OPENCV_AVAILABLE:
                        logger.warning(f"📹 Video creation skipped for {clip_id} (OpenCV not available)")
                    else:
                        logger.debug(f"📹 Video creation skipped for {clip_id} (disabled)")
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Processed': f"{self.processed_clips}",
                    'Success': f"{self.successful_features}",
                    'Failed': f"{self.failed_clips}",
                    'Videos': f"{self.successful_videos}" if self.create_videos else "N/A"
                })
                
                # Memory cleanup for GPU
                if self.device != "cpu":
                    torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                continue
        
        progress_bar.close()
        
        # Calculate processing time
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Create summary
        summary = self._create_summary(processing_time)
        
        # Save summary to output directory
        summary_path = os.path.join(self.output_dataset_path, 'serialization_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"✅ Serialization completed in {processing_time:.1f} seconds")
        logger.info(f"📊 Summary: {self.successful_features} successful, {self.failed_clips} failed")
        if self.create_videos:
            logger.info(f"🎬 Videos: {self.successful_videos} successful, {self.failed_videos} failed")
        logger.info(f"📁 Output directory: {self.output_dataset_path}")
        logger.info(f"📄 Summary saved to: {summary_path}")
        
        return summary
    
    def _create_summary(self, processing_time: float) -> Dict[str, Any]:
        """
        Create summary of serialization process
        
        Args:
            processing_time: Total processing time in seconds
            
        Returns:
            Summary dictionary
        """
        summary = {
            'serialization_info': {
                'date': datetime.now().isoformat(),
                'processing_time_seconds': processing_time,
                'input_dataset_path': self.input_dataset_path,
                'output_dataset_path': self.output_dataset_path,
                'pca_directions_path': self.pca_directions_path,
                'device_used': str(self.device),
                'batch_size': self.batch_size,
                'num_workers': self.num_workers
            },
            'processing_stats': {
                'total_clips_processed': self.processed_clips,
                'successful_features_extracted': self.successful_features,
                'failed_clips': self.failed_clips,
                'total_frames_processed': self.total_frames,
                'success_rate': self.successful_features / max(self.processed_clips, 1),
                'video_creation_stats': {
                    'successful_videos': self.successful_videos,
                    'failed_videos': self.failed_videos,
                    'video_success_rate': self.successful_videos / max(self.successful_features, 1) if self.create_videos else 0
                }
            },
            'tokenizer_info': {
                'tokenizer_type': 'DINOv2BaseTokenizer',
                'embedding_dimension': self.embed_dim,
                'num_patches': self.num_patches,
                'input_size': self.input_size
            },
            'pca_projection_info': {
                'original_dimension': self.original_embed_dim,
                'projected_dimension': self.projected_dim,
                'pca_components_available': self.n_components,
                'explained_variance_top_10': float(self.pca_explained_variance[:10].sum()),
                'explained_variance_projected': float(np.sum(self.pca_explained_variance[:self.projected_dim])),
                'pca_directions_file': os.path.basename(self.pca_directions_path)
            },
            'output_info': {
                'file_format': 'h5',
                'compression': 'gzip',
                'compression_level': 9,
                'features_per_clip': {
                    'frames': f"(30, 3, {self.input_size}, {self.input_size})" if self.save_frames else "Not saved",
                    'patch_tokens': "Not saved (768-dim original features)",
                    'pos_embeddings': "Not saved (768-dim positional embeddings)",
                    'projected_features': f"(30, {self.num_patches}, {self.projected_dim})"
                },
                'video_creation': {
                    'enabled': self.create_videos,
                    'fps': self.video_fps,
                    'format': 'mp4',
                    'naming_convention': '{clip_id}_pca_visualization.mp4'
                }
            }
        }
        
        return summary
    
    def create_pca_visualization_frame(self, projected_features: np.ndarray) -> np.ndarray:
        """
        Create RGB visualization of PCA projected features for one frame
        
        Args:
            projected_features: (num_patches, projected_dim) PCA projected features
            
        Returns:
            rgb_image: RGB visualization as uint8
        """
        # Use first 3 components for RGB visualization
        n_components = min(3, projected_features.shape[1])
        
        # Extract first 3 components and normalize
        rgb_features = projected_features[:, :n_components]
        
        # Normalize to [0, 1] using min-max normalization
        rgb_normalized = (rgb_features - rgb_features.min(axis=0)) / (rgb_features.max(axis=0) - rgb_features.min(axis=0) + 1e-8)
        
        # If we have fewer than 3 components, pad with zeros
        if n_components < 3:
            padding = np.zeros((projected_features.shape[0], 3 - n_components))
            rgb_normalized = np.concatenate([rgb_normalized, padding], axis=1)
        
        # Create RGB image from PCA projections
        grid_size = int(np.sqrt(self.num_patches))  # Should be 37 for 518x518 input
        patch_size = self.input_size // grid_size   # Should be 14 for 518x518 input
        
        rgb_image = np.zeros((grid_size * patch_size, grid_size * patch_size, 3), dtype=np.float64)
        
        # Map patches to grid positions
        for i, projection in enumerate(rgb_normalized):
            if i >= grid_size * grid_size:  # Safety check
                break
            
            # Calculate grid position
            row = i // grid_size
            col = i % grid_size
            
            # Calculate pixel position
            pixel_row = row * patch_size
            pixel_col = col * patch_size
            
            # Fill patch with RGB values
            rgb_image[pixel_row:pixel_row+patch_size, pixel_col:pixel_col+patch_size] = projection
        
        # Convert to uint8 for OpenCV compatibility
        rgb_image_uint8 = (rgb_image * 255).astype(np.uint8)
        
        return rgb_image_uint8
    
    def create_side_by_side_video(self, original_frames: torch.Tensor, projected_features: np.ndarray, 
                                 output_path: str, fps: Optional[float] = None):
        """
        Create a side-by-side video showing original frames and PCA visualizations
        
        Args:
            original_frames: (T, 3, H, W) tensor of original frames
            projected_features: (T, num_patches, projected_dim) PCA projected features
            output_path: Path to save the video
            fps: Frames per second for output video (uses class default if None)
        """
        if not OPENCV_AVAILABLE:
            raise RuntimeError("OpenCV not available. Cannot create videos.")
            
        if fps is None:
            fps = self.video_fps
            
        logger.info(f"Creating side-by-side video with {len(original_frames)} frames at {fps} FPS")
        
        if len(original_frames) == 0:
            logger.warning("No frames to process for video creation")
            return
        
        # Convert frames to numpy and proper format
        frames_np = []
        pca_visualizations = []
        
        for frame_idx in range(len(original_frames)):
            # Convert original frame from tensor to numpy
            frame = original_frames[frame_idx].cpu().numpy()  # (3, H, W)
            frame = np.transpose(frame, (1, 2, 0))  # (H, W, 3)
            
            # Convert from [0, 1] to [0, 255] and to uint8
            frame_uint8 = (frame * 255).astype(np.uint8)
            frames_np.append(frame_uint8)
            
            # Create PCA visualization for this frame
            frame_features = projected_features[frame_idx]  # (num_patches, projected_dim)
            pca_viz = self.create_pca_visualization_frame(frame_features)
            pca_visualizations.append(pca_viz)
        
        # Get dimensions
        orig_height, orig_width = frames_np[0].shape[:2]
        pca_height, pca_width = pca_visualizations[0].shape[:2]
        
        logger.info(f"Original frame shape: {frames_np[0].shape}, dtype: {frames_np[0].dtype}")
        logger.info(f"PCA visualization shape: {pca_visualizations[0].shape}, dtype: {pca_visualizations[0].dtype}")
        
        # Resize PCA visualization to match original frame height
        target_height = orig_height
        target_width = int(pca_width * target_height / pca_height)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        total_width = orig_width + target_width
        out = cv2.VideoWriter(output_path, fourcc, fps, (total_width, target_height))
        
        logger.info(f"Video dimensions: {total_width}x{target_height}")
        logger.info(f"Original frame: {orig_width}x{orig_height}")
        logger.info(f"PCA visualization: {target_width}x{target_height}")
        
        # Process each frame
        for frame_idx, (orig_frame, pca_viz) in enumerate(zip(frames_np, pca_visualizations)):
            # Resize PCA visualization
            pca_resized = cv2.resize(pca_viz, (target_width, target_height))
            
            # Convert PCA visualization from RGB to BGR for OpenCV
            pca_bgr = cv2.cvtColor(pca_resized, cv2.COLOR_RGB2BGR)
            
            # Create side-by-side frame
            combined_frame = np.zeros((target_height, total_width, 3), dtype=np.uint8)
            combined_frame[:, :orig_width] = orig_frame
            combined_frame[:, orig_width:] = pca_bgr
            
            # Write frame
            out.write(combined_frame)
            
            if frame_idx % 10 == 0:
                logger.info(f"Processed frame {frame_idx}/{len(frames_np)}")
        
        # Release video writer
        out.release()
        logger.info(f"Video saved to: {output_path}")
    
    def create_video_for_clip(self, clip_features: Dict, output_base_path: str):
        """
        Create a side-by-side video for a single clip
        
        Args:
            clip_features: Dictionary containing clip features and metadata
            output_base_path: Base path for output files (without extension)
        """
        if not OPENCV_AVAILABLE:
            logger.error("OpenCV not available. Cannot create videos.")
            return None
            
        try:
            # Debug: log available keys
            logger.debug(f"Available keys in clip_features: {list(clip_features.keys())}")
            
            # Extract data
            frames = clip_features['frames']  # (T, 3, H, W) tensor
            projected_features = clip_features['projected_features']  # (T, num_patches, projected_dim)
            
            logger.debug(f"Frames shape: {frames.shape}, dtype: {frames.dtype}")
            logger.debug(f"Projected features shape: {projected_features.shape}, dtype: {projected_features.dtype}")
            
            # Validate data
            if frames is None or projected_features is None:
                logger.error("Frames or projected features are None")
                return None
                
            if len(frames.shape) != 4 or frames.shape[1] != 3:
                logger.error(f"Invalid frames shape: {frames.shape}, expected (T, 3, H, W)")
                return None
                
            if len(projected_features.shape) != 3:
                logger.error(f"Invalid projected features shape: {projected_features.shape}, expected (T, num_patches, projected_dim)")
                return None
                
            if frames.shape[0] != projected_features.shape[0]:
                logger.error(f"Mismatch in number of frames: {frames.shape[0]} vs {projected_features.shape[0]}")
                return None
            
            # Create video path
            video_path = output_base_path + '_pca_visualization.mp4'
            
            # Create side-by-side video
            self.create_side_by_side_video(frames, projected_features, video_path)
            
            logger.info(f"✅ Video created: {video_path}")
            return video_path
            
        except Exception as e:
            logger.error(f"❌ Failed to create video for clip: {e}")
            logger.error(f"Clip features keys: {list(clip_features.keys()) if clip_features else 'None'}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Serialize face dataset with DINOv2 features, PCA projection, and side-by-side video visualizations")
    
    # Required arguments
    '''
    parser.add_argument("--input-dataset", type=str, 
                       default="/Users/ozgewhiting/Documents/EQLabs/datasets_serial/CCA_train_db4_no_padding/CCA_train_db4_no_padding/",
                       help="Path to input dataset directory containing H5 files")
    parser.add_argument("--output-dataset", type=str, 
                       default="/Users/ozgewhiting/Documents/EQLabs/datasets_serial/CCA_train_db4_no_padding/CCA_train_db4_no_padding_features_pca_384/",
                       help="Path to output dataset directory for feature H5 files")
    parser.add_argument("--pca-directions", type=str, 
                       default="/Users/ozgewhiting/Documents/EQLabs/datasets_serial/CCA_train_db4_no_padding/pca_directions_dinov2_base_384.json",
                       help="Path to PCA directions JSON file from compute_pca_directions.py")
    '''
    parser.add_argument("--input-dataset", type=str, 
                       default="/mnt/dataset-storage/dbs/CCA_train_db4_no_padding_keywords_offset_1.0/",
                       help="Path to input dataset directory containing H5 files")
    parser.add_argument("--output-dataset", type=str, 
                       default="/mnt/dataset-storage/dbs/CCA_train_db4_no_padding_keywords_offset_1.0_features/",
                       help="Path to output dataset directory for feature H5 files")
    parser.add_argument("--pca-directions", type=str, 
                       default="/mnt/dataset-storage/dbs/pca_directions_dinov2_base_384.json",
                       help="Path to PCA directions JSON file from compute_pca_directions.py")

    
    # Optional arguments
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of samples to process (for debugging)")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size for processing")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto/cuda/cpu)")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Number of data loader workers")
    parser.add_argument("--pin-memory", action="store_true",
                       help="Pin memory for GPU processing")
    parser.add_argument("--persistent-workers", action="store_true",
                       help="Use persistent workers")
    parser.add_argument("--projected-dim", type=int, default=384,
                       help="Target dimension for PCA projection (default: 384)")
    parser.add_argument("--create-videos", action="store_true", default=True,
                       help="Create side-by-side videos showing original frames and PCA visualizations")
    parser.add_argument("--no-videos", action="store_true", default=False,
                       help="Disable video creation (overrides --create-videos)")
    parser.add_argument("--save-frames", action="store_true", default=False,
                       help="Save original frames to H5 files (increases file size significantly)")
    parser.add_argument("--video-fps", type=float, default=30.0,
                       help="FPS for output videos (default: 30.0)")
    
    args = parser.parse_args()
    
    # Auto-detect device if requested
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Auto-detected device: {args.device}")
    
    # Validate input dataset path
    if not os.path.exists(args.input_dataset):
        logger.error(f"Input dataset path does not exist: {args.input_dataset}")
        return False
    
    # Validate PCA directions path
    if not os.path.exists(args.pca_directions):
        logger.error(f"PCA directions path does not exist: {args.pca_directions}")
        return False
    
    # Handle video creation flags
    if args.no_videos:
        args.create_videos = False
    
    # Create serializer
    serializer = PCAProjectionSerializer(
        input_dataset_path=args.input_dataset,
        output_dataset_path=args.output_dataset,
        pca_directions_path=args.pca_directions,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        projected_dim=args.projected_dim,
        create_videos=args.create_videos,
        video_fps=args.video_fps,
        save_frames=args.save_frames
    )
    
    # Log video creation status
    if args.create_videos:
        logger.info(f"🎬 Video creation enabled: {args.video_fps} FPS")
    else:
        logger.info("📹 Video creation disabled")
    
    # Log frames saving status
    if args.save_frames:
        logger.info("💾 Original frames will be saved to H5 files (increases file size)")
    else:
        logger.info("💾 Original frames will not be saved to H5 files (saves disk space)")
    
    # Run serialization
    try:
        summary = serializer.serialize_dataset(max_samples=args.max_samples)
        logger.info("✅ Serialization completed successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Serialization failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 