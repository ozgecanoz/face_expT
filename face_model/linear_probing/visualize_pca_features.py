#!/usr/bin/env python3
"""
Visualize PCA features from video clips
Loads a video clip and PCA directions, then generates RGB images using top 3 eigenvectors
"""

import torch
import numpy as np
import json
import cv2
import matplotlib.pyplot as plt
import argparse
import os
import logging
from typing import Dict, List, Tuple, Optional

# Add the project root to the path
import sys
sys.path.append('.')

# Add the face_model directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.dinov2_tokenizer import DINOv2Tokenizer
from data.dataset import FaceDataset
from torch.utils.data import DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_pca_results(pca_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Load PCA results from JSON file
    
    Args:
        pca_path: Path to PCA results JSON file
    
    Returns:
        pca_components: (n_components, 384) PCA components
        pca_mean: (384,) mean of features
        explained_variance_ratio: (n_components,) explained variance ratio
        configuration: Configuration dictionary with image size and model info
    """
    logger.info(f"Loading PCA results from: {pca_path}")
    
    with open(pca_path, 'r') as f:
        results = json.load(f)
    
    pca_components = np.array(results['pca_components'])
    pca_mean = np.array(results['pca_mean'])
    explained_variance_ratio = np.array(results['explained_variance_ratio'])
    
    # Extract configuration
    configuration = results.get('configuration', {})
    
    logger.info(f"Loaded PCA with {pca_components.shape[0]} components")
    logger.info(f"Feature dimension: {pca_components.shape[1]}")
    logger.info(f"Explained variance ratios: {explained_variance_ratio}")
    
    # Log configuration details
    if configuration:
        image_size = configuration.get('image_size', 'unknown')
        model_name = configuration.get('model_name', 'unknown')
        patches_per_frame = configuration.get('patches_per_frame', 'unknown')
        logger.info(f"PCA configuration: {image_size}x{image_size} images, {model_name}, {patches_per_frame} patches per frame")
    
    return pca_components, pca_mean, explained_variance_ratio, configuration


def extract_patch_features_from_video(
    video_path: str,
    pca_components: np.ndarray,
    pca_mean: np.ndarray,
    target_image_size: int = 518,
    model_name: str = 'vit_small_patch14_dinov2.lvd142m',
    device: str = "cpu"
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict]]:
    """
    Extract patch features from a video using DINOv2 tokenizer
    
    Args:
        video_path: Path to the video file
        pca_components: PCA components matrix
        pca_mean: PCA mean vector
        target_image_size: Target image size for processing
        model_name: Model name from PCA configuration
        device: Device to use
    
    Returns:
        patch_features_list: List of patch features for each frame
        original_frames: List of original frames
        metadata: List of metadata for each frame
    """
    logger.info(f"Loading video: {video_path}")
    logger.info(f"Target image size: {target_image_size}")
    logger.info(f"Model: {model_name}")
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")
    
    # Initialize tokenizer based on model type from PCA configuration
    # Check if this is a base model first (regardless of image size)
    if 'dinov2-base' in model_name.lower() or 'facebook/dinov2-base' in model_name.lower():
        from models.dinov2_tokenizer import DINOv2BaseTokenizer
        tokenizer = DINOv2BaseTokenizer(device=device)
        expected_patches = 1369  # Base model always has 1369 patches
        patch_size = 14
        grid_size = 37
        embed_dim = 768
        logger.info(f"Using DINOv2 base tokenizer with {expected_patches} patches (768-dim embeddings)")
    else:
        # For non-base models, determine configuration based on target size
        if target_image_size == 224:
            expected_patches = 196  # 14x14 grid
            patch_size = 16
            grid_size = 14
        else:
            expected_patches = 1369  # 37x37 grid
            patch_size = 14
            grid_size = 37
        
        from models.dinov2_tokenizer import DINOv2Tokenizer
        tokenizer = DINOv2Tokenizer(model_name=model_name, device=device)
        embed_dim = 384
        logger.info(f"Using DINOv2 tokenizer with {expected_patches} patches (384-dim embeddings)")
    
    # Validate tokenizer configuration
    actual_patches = tokenizer.get_num_patches()
    actual_input_size = tokenizer.get_input_size()
    if actual_patches != expected_patches or actual_input_size != target_image_size:
        raise ValueError(f"Tokenizer mismatch: expected {expected_patches} patches for {target_image_size}x{target_image_size}, "
                       f"got {actual_patches} patches for {actual_input_size}x{actual_input_size}")
    
    # Validate PCA components match embedding dimension
    if pca_components.shape[1] != embed_dim:
        raise ValueError(f"PCA components dimension mismatch: expected {embed_dim}, got {pca_components.shape[1]}")
    
    patch_features_list = []
    original_frames = []
    metadata = []
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        if width != target_image_size or height != target_image_size:
            logger.info(f"Resizing frame from {width}x{height} to {target_image_size}x{target_image_size}")
            frame_resized = cv2.resize(frame_rgb, (target_image_size, target_image_size))
        else:
            frame_resized = frame_rgb
        
        # Convert to tensor and normalize to [0, 1]
        frame_tensor = torch.from_numpy(frame_resized).float() / 255.0
        frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        frame_tensor = frame_tensor.to(device)
        
        # Get DINOv2 patch tokens
        with torch.no_grad():
            patch_tokens, _ = tokenizer(frame_tensor)  # (1, num_patches, embed_dim)
        
        # Extract patch features
        patch_features = patch_tokens.squeeze(0).cpu().numpy()  # (num_patches, embed_dim)
        
        # Store features and metadata
        patch_features_list.append(patch_features)
        original_frames.append(frame_rgb)
        
        metadata.append({
            'frame_idx': frame_idx,
            'target_image_size': target_image_size,
            'model_name': model_name,
            'patches_per_frame': patch_features.shape[0],
            'grid_size': grid_size,
            'patch_size': patch_size,
            'embed_dim': embed_dim,
            'original_size': f"{width}x{height}",
            'resized': width != target_image_size or height != target_image_size
        })
        
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            logger.info(f"Processed {frame_idx}/{total_frames} frames")
    
    cap.release()
    
    logger.info(f"Extracted features from {len(patch_features_list)} frames")
    logger.info(f"Feature shape: {patch_features_list[0].shape}")
    
    return patch_features_list, original_frames, metadata


def project_features_to_pca(
    patch_features: np.ndarray,
    pca_components: np.ndarray,
    pca_mean: np.ndarray,
    n_components: int = 3,
    start_component: int = 0
) -> np.ndarray:
    """
    Project patch features to PCA space
    
    Args:
        patch_features: (num_patches, embed_dim) patch features for one frame
        pca_components: (n_components, embed_dim) PCA components
        pca_mean: (embed_dim,) mean of features
        n_components: Number of components to use (default: 3 for RGB)
        start_component: Starting component index (default: 0 for first 3 components)
    
    Returns:
        pca_projections: (num_patches, n_components) PCA projections
    """
    # Center the features
    features_centered = patch_features - pca_mean
    
    # Project to PCA space starting from start_component
    end_component = start_component + n_components
    if end_component > pca_components.shape[0]:
        logger.warning(f"Requested components {start_component} to {end_component-1}, but only {pca_components.shape[0]} available")
        end_component = pca_components.shape[0]
        n_components = end_component - start_component
    
    selected_components = pca_components[start_component:end_component]
    pca_projections = features_centered @ selected_components.T  # (num_patches, n_components)
    
    logger.info(f"Projected to PCA components {start_component} to {end_component-1}")
    
    return pca_projections


def normalize_for_visualization(
    pca_projections: np.ndarray,
    method: str = "minmax"
) -> np.ndarray:
    """
    Normalize PCA projections for visualization
    
    Args:
        pca_projections: (1369, n_components) PCA projections
        method: Normalization method ("minmax", "zscore", "robust")
    
    Returns:
        normalized_projections: (1369, n_components) normalized projections in [0, 1]
    """
    if method == "minmax":
        # Min-max normalization to [0, 1]
        normalized = (pca_projections - pca_projections.min(axis=0)) / (pca_projections.max(axis=0) - pca_projections.min(axis=0))
    elif method == "zscore":
        # Z-score normalization then clip to [0, 1]
        normalized = (pca_projections - pca_projections.mean(axis=0)) / pca_projections.std(axis=0)
        normalized = np.clip((normalized + 3) / 6, 0, 1)  # Clip to [0, 1] assuming normal distribution
    elif method == "robust":
        # Robust normalization using percentiles
        q1 = np.percentile(pca_projections, 1, axis=0)
        q99 = np.percentile(pca_projections, 99, axis=0)
        normalized = (pca_projections - q1) / (q99 - q1)
        normalized = np.clip(normalized, 0, 1)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized


def create_patch_visualization_frame(
    pca_projections: np.ndarray,
    metadata: Dict
) -> np.ndarray:
    """
    Create RGB visualization of patch features for one frame
    
    Args:
        pca_projections: (num_patches, 3) normalized PCA projections
        metadata: Metadata dictionary containing grid_size and patch_size
    
    Returns:
        rgb_image: RGB visualization as uint8
    """
    # Extract grid configuration from metadata
    grid_size = metadata.get('grid_size', 37)  # Default to 37x37 for backward compatibility
    patch_size = metadata.get('patch_size', 14)  # Default to 14 for backward compatibility
    
    # Validate that we have the right number of patches
    expected_patches = grid_size * grid_size
    if pca_projections.shape[0] != expected_patches:
        raise ValueError(f"Expected {expected_patches} patches for {grid_size}x{grid_size} grid, got {pca_projections.shape[0]}")
    
    # Create RGB image from PCA projections
    rgb_image = np.zeros((grid_size * patch_size, grid_size * patch_size, 3), dtype=np.float64)
    
    # Map patches to grid positions
    for i, projection in enumerate(pca_projections):
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


def create_side_by_side_video(
    original_frames: List[np.ndarray],
    pca_visualizations: List[np.ndarray],
    output_path: str,
    fps: float = 30.0
):
    """
    Create a video with original frames and PCA visualizations side by side
    
    Args:
        original_frames: List of original video frames
        pca_visualizations: List of PCA visualization frames
        output_path: Path to save the video
        fps: Frames per second for output video
    """
    logger.info(f"Creating side-by-side video with {len(original_frames)} frames")
    
    if len(original_frames) == 0 or len(pca_visualizations) == 0:
        raise ValueError("No frames to process")
    
    # Get dimensions
    orig_height, orig_width = original_frames[0].shape[:2]
    pca_height, pca_width = pca_visualizations[0].shape[:2]
    
    logger.info(f"Original frame shape: {original_frames[0].shape}, dtype: {original_frames[0].dtype}")
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
    for frame_idx, (orig_frame, pca_viz) in enumerate(zip(original_frames, pca_visualizations)):
        # Ensure original frame is uint8
        if orig_frame.dtype != np.uint8:
            logger.warning(f"Converting original frame {frame_idx} from {orig_frame.dtype} to uint8")
            orig_frame = orig_frame.astype(np.uint8)
        
        # Ensure PCA visualization is uint8
        if pca_viz.dtype != np.uint8:
            logger.warning(f"Converting PCA visualization {frame_idx} from {pca_viz.dtype} to uint8")
            pca_viz = pca_viz.astype(np.uint8)
        
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
        
        if frame_idx % 100 == 0:
            logger.info(f"Processed frame {frame_idx}/{len(original_frames)}")
    
    # Release video writer
    out.release()
    logger.info(f"Video saved to: {output_path}")


def main():
    """Main function to visualize PCA features"""
    
    parser = argparse.ArgumentParser(description="Visualize PCA features from video")
    parser.add_argument("--video_path", type=str, 
    default="/Users/ozgewhiting/Documents/EQLabs/datasets_serial/CCA_train_db4_no_padding/CCA_train_db4_no_padding/subject_427_1566_00_faces_8_10.mp4", help="Path to video file")
    parser.add_argument("--pca_path", type=str, 
    default="/Users/ozgewhiting/Documents/EQLabs/datasets_serial/CCA_train_db4_no_padding/pca_directions_dinov2_base.json", help="Path to PCA results JSON")
    parser.add_argument("--output_path", type=str, 
    default="/Users/ozgewhiting/Documents/EQLabs/datasets_serial/CCA_train_db4_no_padding/subject_427_1566_00_faces_8_10_pca_visualization_components_6_8.mp4", help="Path to save video")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument("--normalization", type=str, default="minmax", 
                       choices=["minmax", "zscore", "robust"], help="Normalization method")
    parser.add_argument("--n_components", type=int, default=3, help="Number of PCA components to use")
    parser.add_argument("--start_component", type=int, 
    default=6, #default=0, 
    help="Starting PCA component index (0-based)")
    parser.add_argument("--fps", type=float, default=30.0, help="Output video FPS")
    
    args = parser.parse_args()
    
    logger.info(f"Starting PCA feature visualization")
    logger.info(f"Video: {args.video_path}")
    logger.info(f"PCA results: {args.pca_path}")
    logger.info(f"Output: {args.output_path}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Normalization: {args.normalization}")
    logger.info(f"Components: {args.n_components} starting from index {args.start_component}")
    logger.info(f"FPS: {args.fps}")
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Load PCA results
    pca_components, pca_mean, explained_variance_ratio, configuration = load_pca_results(args.pca_path)
    
    # Log component information
    total_components = pca_components.shape[0]
    end_component = args.start_component + args.n_components
    logger.info(f"Total PCA components available: {total_components}")
    logger.info(f"Using components {args.start_component} to {min(end_component-1, total_components-1)}")
    
    # Show explained variance for selected components
    if args.start_component < len(explained_variance_ratio):
        selected_variance = explained_variance_ratio[args.start_component:end_component]
        logger.info(f"Explained variance for selected components: {selected_variance}")
    
    # Determine target image size from configuration
    target_image_size = configuration.get('image_size', 518)
    if target_image_size not in [224, 518]:
        logger.warning(f"Unknown image size in configuration: {target_image_size}, defaulting to 518")
        target_image_size = 518
    
    logger.info(f"Using target image size: {target_image_size}x{target_image_size}")
    
    # Log configuration details
    if configuration:
        model_name = configuration.get('model_name', 'unknown')
        patches_per_frame = configuration.get('patches_per_frame', 'unknown')
        grid_size = configuration.get('grid_size', 'unknown')
        logger.info(f"PCA was computed with: {model_name}, {patches_per_frame} patches, {grid_size}x{grid_size} grid")
    
    # Extract patch features from video
    patch_features_list, original_frames, metadata = extract_patch_features_from_video(
        video_path=args.video_path,
        pca_components=pca_components,
        pca_mean=pca_mean,
        target_image_size=target_image_size,
        model_name=configuration.get('model_name', 'vit_small_patch14_dinov2.lvd142m'), # Pass model_name from configuration
        device=args.device
    )
    
    # Process each frame
    pca_visualizations = []
    
    logger.info("Processing frames for PCA visualization...")
    for frame_idx, patch_features in enumerate(patch_features_list):
        # Project features to PCA space
        pca_projections = project_features_to_pca(
            patch_features=patch_features,
            pca_components=pca_components,
            pca_mean=pca_mean,
            n_components=args.n_components,
            start_component=args.start_component
        )
        
        # Normalize for visualization
        normalized_projections = normalize_for_visualization(
            pca_projections=pca_projections,
            method=args.normalization
        )
        
        # Create visualization for this frame
        pca_viz = create_patch_visualization_frame(
            pca_projections=normalized_projections,
            metadata=metadata[frame_idx] # Pass metadata for this frame
        )
        
        pca_visualizations.append(pca_viz)
        
        if frame_idx % 50 == 0:
            logger.info(f"Processed frame {frame_idx}/{len(patch_features_list)}")
    
    # Create side-by-side video
    create_side_by_side_video(
        original_frames=original_frames,
        pca_visualizations=pca_visualizations,
        output_path=args.output_path,
        fps=args.fps
    )
    
    logger.info("PCA feature visualization completed successfully!")


if __name__ == "__main__":
    main() 