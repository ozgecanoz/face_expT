#!/usr/bin/env python3
"""
Serialize PCA Projections for Face Dataset
Processes face clips, extracts DINOv2 features, projects to 384 dimensions using PCA,
and saves both projected features and visualization videos.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import h5py
import cv2
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import datetime

# Add the project root to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.dataset import FaceDataset
from models.dinov2_tokenizer import DINOv2BaseTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PCAProjectionSerializer:
    """
    Serializes PCA-projected DINOv2 features from face datasets
    """
    
    def __init__(self, pca_directions_path: str, device: str = "cuda"):
        """
        Initialize the serializer with PCA directions
        
        Args:
            pca_directions_path: Path to PCA directions JSON file
            device: Device to use for processing
        """
        self.device = torch.device(device)
        self.is_gpu = self.device.type == "cuda"
        
        # Load PCA directions
        logger.info(f"Loading PCA directions from: {pca_directions_path}")
        with open(pca_directions_path, 'r') as f:
            pca_data = json.load(f)
        
        # Extract PCA components and metadata
        self.pca_components = np.array(pca_data['pca_components'])  # (n_components, embed_dim)
        self.pca_mean = np.array(pca_data['pca_mean'])  # (embed_dim,)
        self.pca_explained_variance = np.array(pca_data['pca_explained_variance'])  # (n_components,)
        
        # Validate PCA data
        self.n_components = self.pca_components.shape[0]
        self.original_embed_dim = self.pca_components.shape[1]
        self.projected_dim = 384  # Target dimension
        
        logger.info(f"PCA components: {self.pca_components.shape}")
        logger.info(f"Original embed dim: {self.original_embed_dim}")
        logger.info(f"Target projected dim: {self.projected_dim}")
        logger.info(f"Explained variance: {self.pca_explained_variance[:10].sum():.3f} (top 10)")
        
        # Initialize DINOv2 base tokenizer
        logger.info("Initializing DINOv2 base tokenizer...")
        self.tokenizer = DINOv2BaseTokenizer(device=self.device)
        
        # Validate tokenizer configuration
        actual_embed_dim = self.tokenizer.get_embed_dim()
        if actual_embed_dim != self.original_embed_dim:
            raise ValueError(f"PCA embed dim ({self.original_embed_dim}) doesn't match tokenizer ({actual_embed_dim})")
        
        # GPU memory optimization
        if self.is_gpu:
            torch.cuda.empty_cache()
            logger.info(f"Initial GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
    
    def project_features_to_pca(self, features: np.ndarray, start_component: int = 0) -> np.ndarray:
        """
        Project features to PCA space
        
        Args:
            features: (N, embed_dim) array of features
            start_component: Starting component index for projection
        
        Returns:
            projected_features: (N, projected_dim) array of projected features
        """
        # Center features using PCA mean
        centered_features = features - self.pca_mean
        
        # Project to PCA space using top components
        end_component = min(start_component + self.projected_dim, self.n_components)
        components_to_use = self.pca_components[start_component:end_component]  # (projected_dim, embed_dim)
        
        # Project: (N, embed_dim) @ (embed_dim, projected_dim) = (N, projected_dim)
        projected_features = centered_features @ components_to_use.T
        
        logger.debug(f"Projected features from {features.shape} to {projected_features.shape}")
        return projected_features
    
    def create_rgb_visualization(self, features: np.ndarray, start_component: int = 0) -> np.ndarray:
        """
        Create RGB visualization using top 3 PCA components
        
        Args:
            features: (N, embed_dim) array of features
            start_component: Starting component index for visualization
        
        Returns:
            rgb_image: (H, W, 3) RGB image
        """
        # Project to top 3 components for RGB visualization
        rgb_projection = self.project_features_to_pca(features, start_component=start_component)[:, :3]
        
        # Normalize to 0-1 range
        rgb_projection = (rgb_projection - rgb_projection.min()) / (rgb_projection.max() - rgb_projection.min() + 1e-8)
        
        # Reshape to grid (assuming 37x37 patches for 518x518 images)
        grid_size = int(np.sqrt(features.shape[0]))
        if grid_size * grid_size != features.shape[0]:
            # If not perfect square, pad or truncate
            grid_size = int(np.sqrt(features.shape[0]))
            if grid_size * grid_size < features.shape[0]:
                grid_size += 1
        
        # Pad or truncate to fit grid
        target_size = grid_size * grid_size
        if rgb_projection.shape[0] < target_size:
            # Pad with zeros
            padding = np.zeros((target_size - rgb_projection.shape[0], 3))
            rgb_projection = np.vstack([rgb_projection, padding])
        elif rgb_projection.shape[0] > target_size:
            # Truncate
            rgb_projection = rgb_projection[:target_size]
        
        # Reshape to grid
        rgb_grid = rgb_projection.reshape(grid_size, grid_size, 3)
        
        # Scale to 0-255 and convert to uint8
        rgb_image = (rgb_grid * 255).astype(np.uint8)
        
        return rgb_image
    
    def process_batch(self, batch: Dict, batch_idx: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Process a batch of clips
        
        Args:
            batch: Batch from dataloader
            batch_idx: Batch index for logging
        
        Returns:
            projected_features_list: List of projected features per clip
            rgb_visualizations: List of RGB visualizations per clip
        """
        projected_features_list = []
        rgb_visualizations = []
        
        try:
            # Collect all frames from all clips in this batch
            all_frames = []
            all_clip_lengths = []
            
            for clip_idx, frames in enumerate(batch['frames']):
                # frames: (num_frames, 3, 518, 518) - already correct size
                num_frames = frames.shape[0]
                all_frames.append(frames)
                all_clip_lengths.append(num_frames)
            
            if all_frames:
                # Concatenate all frames into one large batch
                combined_frames = torch.cat(all_frames, dim=0)  # (total_frames, 3, 518, 518)
                
                # Move entire batch to GPU
                if self.is_gpu:
                    combined_frames = combined_frames.to(self.device)
                    logger.debug(f"Batch {batch_idx}: Moved {combined_frames.shape[0]} frames to GPU")
                
                # Extract DINOv2 features for all frames
                with torch.no_grad():
                    all_patch_tokens, _ = self.tokenizer(combined_frames)  # (total_frames, 1369, 768)
                
                # Split back by clip and process each
                start_idx = 0
                for clip_idx, clip_length in enumerate(all_clip_lengths):
                    clip_tokens = all_patch_tokens[start_idx:start_idx + clip_length]  # (clip_length, 1369, 768)
                    
                    # Process each frame in the clip
                    clip_projected_features = []
                    clip_rgb_visualizations = []
                    
                    for frame_idx in range(clip_length):
                        frame_tokens = clip_tokens[frame_idx]  # (1369, 768)
                        
                        # Move to CPU for numpy operations
                        frame_tokens_cpu = frame_tokens.cpu().numpy()
                        
                        # Project to PCA space
                        projected_features = self.project_features_to_pca(frame_tokens_cpu)  # (1369, 384)
                        clip_projected_features.append(projected_features)
                        
                        # Create RGB visualization
                        rgb_viz = self.create_rgb_visualization(frame_tokens_cpu)  # (37, 37, 3)
                        clip_rgb_visualizations.append(rgb_viz)
                    
                    # Stack features for this clip
                    clip_projected_features = np.stack(clip_projected_features, axis=0)  # (clip_length, 1369, 384)
                    projected_features_list.append(clip_projected_features)
                    
                    # Stack RGB visualizations for this clip
                    clip_rgb_viz = np.stack(clip_rgb_visualizations, axis=0)  # (clip_length, 37, 37, 3)
                    rgb_visualizations.append(clip_rgb_viz)
                    
                    start_idx += clip_length
                
                # GPU memory cleanup
                if self.is_gpu:
                    torch.cuda.empty_cache()
                    
        except torch.cuda.OutOfMemoryError as e:
            if self.is_gpu:
                logger.warning(f"GPU OOM at batch {batch_idx}. Clearing cache and retrying...")
                torch.cuda.empty_cache()
                
                # Try to process with smaller batches or skip
                try:
                    # Process clips one by one to reduce memory usage
                    for clip_idx, frames in enumerate(batch['frames']):
                        clip_projected_features = []
                        clip_rgb_visualizations = []
                        
                        for frame_idx in range(frames.shape[0]):
                            frame = frames[frame_idx:frame_idx+1].to(self.device)  # (1, 3, 518, 518)
                            
                            with torch.no_grad():
                                frame_tokens, _ = self.tokenizer(frame)  # (1, 1369, 768)
                            
                            frame_tokens_cpu = frame_tokens.squeeze(0).cpu().numpy()  # (1369, 768)
                            
                            # Project and visualize
                            projected_features = self.project_features_to_pca(frame_tokens_cpu)
                            rgb_viz = self.create_rgb_visualization(frame_tokens_cpu)
                            
                            clip_projected_features.append(projected_features)
                            clip_rgb_visualizations.append(rgb_viz)
                            
                            # Clean up frame
                            del frame, frame_tokens
                            torch.cuda.empty_cache()
                        
                        # Stack for this clip
                        clip_projected_features = np.stack(clip_projected_features, axis=0)
                        clip_rgb_viz = np.stack(clip_rgb_visualizations, axis=0)
                        
                        projected_features_list.append(clip_projected_features)
                        rgb_visualizations.append(clip_rgb_viz)
                    
                    logger.info("Successfully recovered from OOM with clip-by-clip processing")
                    
                except torch.cuda.OutOfMemoryError:
                    logger.error(f"Failed to recover from OOM, skipping batch {batch_idx}")
                    return [], []
            
            else:
                raise e
        
        except Exception as e:
            logger.error(f"Error processing batch {batch_idx}: {e}")
            return [], []
        
        return projected_features_list, rgb_visualizations
    
    def create_side_by_side_video(self, original_frames: np.ndarray, rgb_visualizations: np.ndarray, 
                                 output_path: str, fps: int = 30) -> None:
        """
        Create side-by-side video of original frames and RGB visualizations
        
        Args:
            original_frames: (T, H, W, 3) original video frames
            rgb_visualizations: (T, H, W, 3) RGB PCA visualizations
            output_path: Path to save the video
            fps: Frames per second
        """
        # Ensure frames are uint8
        if original_frames.dtype != np.uint8:
            original_frames = np.clip(original_frames, 0, 255).astype(np.uint8)
        
        if rgb_visualizations.dtype != np.uint8:
            rgb_visualizations = np.clip(rgb_visualizations, 0, 255).astype(np.uint8)
        
        # Resize RGB visualizations to match original frame height
        target_height = original_frames.shape[1]
        target_width = original_frames.shape[2]
        
        resized_viz = []
        for viz in rgb_visualizations:
            resized = cv2.resize(viz, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            resized_viz.append(resized)
        
        resized_viz = np.stack(resized_viz, axis=0)
        
        # Create side-by-side frames
        side_by_side_frames = []
        for orig_frame, viz_frame in zip(original_frames, resized_viz):
            # Concatenate horizontally
            combined_frame = np.hstack([orig_frame, viz_frame])
            side_by_side_frames.append(combined_frame)
        
        side_by_side_frames = np.stack(side_by_side_frames, axis=0)
        
        # Write video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, 
                                     (side_by_side_frames.shape[2], side_by_side_frames.shape[1]))
        
        for frame in side_by_side_frames:
            video_writer.write(frame)
        
        video_writer.release()
        logger.info(f"Side-by-side video saved to: {output_path}")
    
    def serialize_dataset(self, dataset_path: str, output_dir: str, max_samples: Optional[int] = None,
                         batch_size: int = 4, num_workers: int = 0, create_videos: bool = True) -> None:
        """
        Serialize PCA-projected features for entire dataset
        
        Args:
            dataset_path: Path to input dataset directory containing individual H5 files
            output_dir: Directory to save individual H5 files and videos
            max_samples: Maximum number of samples to process
            batch_size: Batch size for processing
            num_workers: Number of data loader workers
            create_videos: Whether to create visualization videos
        """
        logger.info(f"Starting dataset serialization...")
        logger.info(f"Input dataset directory: {dataset_path}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Max samples: {max_samples if max_samples else 'All'}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Create videos: {create_videos}")
        
        # Check if dataset_path is a directory containing H5 files
        if not os.path.isdir(dataset_path):
            raise ValueError(f"Dataset path must be a directory: {dataset_path}")
        
        # Find all H5 files in the dataset directory
        h5_files = []
        for file in os.listdir(dataset_path):
            if file.endswith('.h5'):
                h5_files.append(file)
        
        if not h5_files:
            raise ValueError(f"No H5 files found in dataset directory: {dataset_path}")
        
        # Sort files for consistent processing order
        h5_files.sort()
        
        # Limit samples if specified
        if max_samples:
            h5_files = h5_files[:max_samples]
        
        logger.info(f"Found {len(h5_files)} H5 files to process")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process batches
        total_clips = 0
        total_frames = 0
        
        # Create PCA metadata file
        pca_metadata_path = os.path.join(output_dir, "pca_metadata.json")
        pca_metadata = {
            'n_components': self.n_components,
            'original_embed_dim': self.original_embed_dim,
            'projected_dim': self.projected_dim,
            'pca_explained_variance': self.pca_explained_variance.tolist(),
            'pca_mean': self.pca_mean.tolist(),
            'model_name': 'dinov2-base',
            'input_image_size': 518,
            'num_patches': 1369,
            'input_dataset_directory': dataset_path,
            'input_files_count': len(h5_files),
            'created_at': str(datetime.datetime.now())
        }
        
        with open(pca_metadata_path, 'w') as f:
            json.dump(pca_metadata, f, indent=2)
        
        logger.info(f"PCA metadata saved to: {pca_metadata_path}")
        
        # Process H5 files in batches
        for batch_idx in range(0, len(h5_files), batch_size):
            batch_files = h5_files[batch_idx:batch_idx + batch_size]
            logger.info(f"Processing batch {batch_idx // batch_size + 1}: {len(batch_files)} files")
            
            for file_idx, h5_filename in enumerate(batch_files):
                try:
                    # Use the H5 filename as the base name (without .h5 extension)
                    base_name = h5_filename[:-3] if h5_filename.endswith('.h5') else h5_filename
                    clip_id = base_name
                    
                    logger.info(f"Processing {h5_filename} -> {clip_id}")
                    
                    # Load the H5 file to get frames
                    h5_file_path = os.path.join(dataset_path, h5_filename)
                    
                    try:
                        with h5py.File(h5_file_path, 'r') as h5_file:
                            # Check what datasets are available
                            available_datasets = list(h5_file.keys())
                            logger.debug(f"Available datasets in {h5_filename}: {available_datasets}")
                            
                            # Look for frames dataset (common names: 'frames', 'face_frames', 'data')
                            frames_dataset = None
                            for dataset_name in ['frames', 'face_frames', 'data', 'face_data']:
                                if dataset_name in h5_file:
                                    frames_dataset = h5_file[dataset_name]
                                    break
                            
                            if frames_dataset is None:
                                logger.warning(f"No frames dataset found in {h5_filename}, skipping")
                                continue
                            
                            # Get frames data
                            frames = frames_dataset[:]  # Load all frames
                            
                            # Validate frame dimensions
                            if len(frames.shape) != 4 or frames.shape[1] != 3:
                                logger.warning(f"Invalid frame shape in {h5_filename}: {frames.shape}, expected (T, 3, H, W)")
                                continue
                            
                            num_frames, channels, height, width = frames.shape
                            logger.info(f"Loaded {num_frames} frames with shape {frames.shape} from {h5_filename}")
                            
                            # Check if resizing is needed
                            if height != 518 or width != 518:
                                logger.info(f"Resizing frames from {height}x{width} to 518x518")
                                import torch.nn.functional as F
                                frames_tensor = torch.from_numpy(frames).float()
                                frames_resized = F.interpolate(
                                    frames_tensor, 
                                    size=(518, 518), 
                                    mode='bilinear', 
                                    align_corners=False,
                                    antialias=True
                                )
                                frames = frames_resized.numpy()
                            
                            # Process frames to get DINOv2 features
                            try:
                                # Convert to tensor and move to device
                                frames_tensor = torch.from_numpy(frames).float()
                                if self.is_gpu:
                                    frames_tensor = frames_tensor.to(self.device)
                                
                                # Extract DINOv2 features
                                with torch.no_grad():
                                    patch_tokens, _ = self.tokenizer(frames_tensor)  # (T, 1369, 768)
                                
                                # Process each frame
                                projected_features_list = []
                                rgb_visualizations = []
                                
                                for frame_idx in range(num_frames):
                                    frame_tokens = patch_tokens[frame_idx]  # (1369, 768)
                                    
                                    # Move to CPU for numpy operations
                                    frame_tokens_cpu = frame_tokens.cpu().numpy()
                                    
                                    # Project to PCA space
                                    projected_features = self.project_features_to_pca(frame_tokens_cpu)  # (1369, 384)
                                    projected_features_list.append(projected_features)
                                    
                                    # Create RGB visualization
                                    rgb_viz = self.create_rgb_visualization(frame_tokens_cpu)  # (37, 37, 3)
                                    rgb_visualizations.append(rgb_viz)
                                
                                # Stack features and visualizations
                                projected_features = np.stack(projected_features_list, axis=0)  # (T, 1369, 384)
                                rgb_viz = np.stack(rgb_visualizations, axis=0)  # (T, 37, 37, 3)
                                
                                # Create individual H5 file for this clip using original filename + _features
                                clip_h5_path = os.path.join(output_dir, f"{base_name}_features.h5")
                                
                                try:
                                    with h5py.File(clip_h5_path, 'w') as clip_h5:
                                        # Save projected features
                                        clip_h5.create_dataset(
                                            'projected_features', 
                                            data=projected_features,  # (T, 1369, 384)
                                            compression='gzip', 
                                            compression_opts=6
                                        )
                                        
                                        # Save clip metadata
                                        clip_metadata = {
                                            'clip_id': clip_id,
                                            'original_filename': h5_filename,
                                            'num_frames': projected_features.shape[0],
                                            'num_patches': projected_features.shape[1],
                                            'projected_dim': projected_features.shape[2],
                                            'original_shape': f"({projected_features.shape[0]}, {projected_features.shape[1]}, {projected_features.shape[2]})",
                                            'input_dataset_directory': dataset_path,
                                            'created_at': str(datetime.datetime.now())
                                        }
                                        
                                        # Store metadata as attributes
                                        for key, value in clip_metadata.items():
                                            clip_h5.attrs[key] = value
                                        
                                        # Store PCA metadata reference
                                        clip_h5.attrs['pca_metadata_file'] = os.path.basename(pca_metadata_path)
                                        
                                    logger.debug(f"Saved clip features to: {clip_h5_path}")
                                    
                                except Exception as e:
                                    logger.error(f"Failed to save H5 file for {clip_id}: {e}")
                                    continue
                                
                                # Create visualization video
                                if create_videos:
                                    try:
                                        # Convert frames to HWC format for OpenCV
                                        frames_hwc = frames.transpose(0, 2, 3, 1)  # (T, 518, 518, 3)
                                        
                                        # Create video path directly in output directory
                                        video_filename = f"{base_name}_pca_visualization.mp4"
                                        video_path = os.path.join(output_dir, video_filename)
                                        
                                        # Create side-by-side video
                                        self.create_side_by_side_video(
                                            frames_hwc, rgb_viz, video_path, fps=30
                                        )
                                        
                                    except Exception as e:
                                        logger.error(f"Failed to create video for {clip_id}: {e}")
                                
                                total_clips += 1
                                total_frames += projected_features.shape[0]
                                
                                # GPU memory cleanup
                                if self.is_gpu:
                                    torch.cuda.empty_cache()
                                
                            except Exception as e:
                                logger.error(f"Failed to process frames from {h5_filename}: {e}")
                                continue
                            
                    except Exception as e:
                        logger.error(f"Failed to load H5 file {h5_filename}: {e}")
                        continue
                    
                except Exception as e:
                    logger.error(f"Error processing file {h5_filename}: {e}")
                    continue
            
            # Log progress
            logger.info(f"Completed batch {batch_idx // batch_size + 1}")
            logger.info(f"Total clips processed: {total_clips}, Total frames: {total_frames}")
            if self.is_gpu:
                logger.info(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
        
        # Create summary file
        summary_path = os.path.join(output_dir, "serialization_summary.json")
        summary = {
            'total_clips': total_clips,
            'total_frames': total_frames,
            'input_dataset_directory': dataset_path,
            'input_files_processed': len(h5_files),
            'output_directory': output_dir,
            'pca_metadata_file': os.path.basename(pca_metadata_path),
            'batch_size': batch_size,
            'max_samples': max_samples,
            'device': str(self.device),
            'created_at': str(datetime.datetime.now()),
            'file_structure': {
                'individual_h5_files': f"{total_clips} files with pattern: {clip_id.split('_')[0]}_features.h5",
                'visualization_videos': f"{total_clips} files with pattern: {clip_id.split('_')[0]}_pca_visualization.mp4" if create_videos else "No videos created",
                'pca_metadata': os.path.basename(pca_metadata_path),
                'summary': os.path.basename(summary_path)
            }
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Serialization completed!")
        logger.info(f"Total clips processed: {total_clips}")
        logger.info(f"Total frames processed: {total_frames}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Individual H5 files: {total_clips} files")
        if create_videos:
            logger.info(f"Visualization videos: {total_clips} files")
        logger.info(f"Summary file: {summary_path}")
        logger.info(f"PCA metadata: {pca_metadata_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Serialize PCA-projected DINOv2 features from face dataset directory")
    
    # Required arguments
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to input dataset directory containing individual H5 files (e.g., /path/to/CCA_train_db4_no_padding_keywords_offset_1.0/)")
    parser.add_argument("--pca_directions_path", type=str, required=True,
                       help="Path to PCA directions JSON file")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save individual H5 files and videos")
    
    # Optional arguments
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of H5 files to process")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for processing (number of H5 files per batch)")
    parser.add_argument("--num_workers", type=int, default=0,
                       help="Number of data loader workers (not used for H5 file processing)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--no_videos", action="store_true",
                       help="Skip video creation")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.dataset_path):
        raise ValueError(f"Dataset directory does not exist: {args.dataset_path}")
    
    if not os.path.isdir(args.dataset_path):
        raise ValueError(f"Dataset path must be a directory: {args.dataset_path}")
    
    if not os.path.exists(args.pca_directions_path):
        raise ValueError(f"PCA directions path does not exist: {args.pca_directions_path}")
    
    # Check if directory contains H5 files
    h5_files = [f for f in os.listdir(args.dataset_path) if f.endswith('.h5')]
    if not h5_files:
        raise ValueError(f"No H5 files found in dataset directory: {args.dataset_path}")
    
    logger.info(f"Found {len(h5_files)} H5 files in dataset directory")
    logger.info(f"Sample files: {h5_files[:5]}...")
    
    # Create serializer
    serializer = PCAProjectionSerializer(args.pca_directions_path, args.device)
    
    # Serialize dataset
    serializer.serialize_dataset(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        create_videos=not args.no_videos
    )


if __name__ == "__main__":
    main() 