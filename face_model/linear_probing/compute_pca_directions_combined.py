#!/usr/bin/env python3
"""
Compute PCA directions from DINOv2 patch features from combined datasets
Loads from two dataset folders, extracts patch features, computes PCA, and saves results
Both datasets MUST have H5 files following the exact mp4_utils format:
- faces/frame_XXX/face_000/data (3D uint8 array: H, W, 3)
- faces/frame_XXX/face_000/bbox, confidence, original_size
- metadata group with required fields

Script will ERROR OUT if any H5 file has incorrect format.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import datetime
import h5py
import glob

# Add the project root to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.dinov2_tokenizer import DINOv2BaseTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CombinedH5Dataset(Dataset):
    """
    Dataset class that reads from multiple H5 files across two dataset folders
    Each H5 file must follow the exact mp4_utils format: 'faces/frame_XXX/face_000/data'
    """
    
    def __init__(self, dataset_paths: List[str], max_samples_per_dataset: Optional[int] = None, max_total_samples: Optional[int] = None):
        """
        Initialize combined dataset from multiple paths
        
        Args:
            dataset_paths: List of paths to dataset directories
            max_samples_per_dataset: Maximum H5 files to load from each dataset
            max_total_samples: Maximum total frames to load across all datasets
        """
        self.dataset_paths = dataset_paths
        self.max_samples_per_dataset = max_samples_per_dataset
        self.max_total_samples = max_total_samples
        self.samples = []
        
        # Load samples from all datasets
        total_invalid_files = 0
        for dataset_path in dataset_paths:
            invalid_count = self._load_dataset_samples(dataset_path)
            total_invalid_files += invalid_count
        
        # Apply total sample limit if specified
        if self.max_total_samples and len(self.samples) > self.max_total_samples:
            logger.info(f"Limiting total samples from {len(self.samples)} to {self.max_total_samples}")
            # Randomly sample to maintain diversity across datasets
            import random
            random.shuffle(self.samples)
            self.samples = self.samples[:self.max_total_samples]
        
        logger.info(f"Combined dataset: {len(self.samples)} total samples from {len(dataset_paths)} datasets")
        
        # Error out if any invalid files were found
        if total_invalid_files > 0:
            error_msg = f"❌ Found {total_invalid_files} invalid H5 files across all datasets. All files must follow the exact mp4_utils format."
            logger.error(error_msg)
            logger.error("Required H5 format:")
            logger.error("  - Top level: 'faces' and 'metadata' groups")
            logger.error("  - EXACTLY 30 frames: faces/frame_000 through faces/frame_029")
            logger.error("  - Each frame: faces/frame_XXX/face_000/data (3D uint8 array: H, W, 3)")
            logger.error("  - Each frame: faces/frame_XXX/face_000/bbox (4-element array)")
            logger.error("  - Each frame: faces/frame_XXX/face_000/confidence (float)")
            logger.error("  - Each frame: faces/frame_XXX/face_000/original_size (2-element array)")
            raise ValueError(error_msg)
    
    def _validate_h5_format(self, h5_file: Path) -> Tuple[bool, int, str]:
        """
        Strictly validate H5 file format matches mp4_utils structure
        
        Args:
            h5_file: Path to H5 file
            
        Returns:
            (is_valid, num_frames, error_message)
        """
        try:
            with h5py.File(h5_file, 'r') as f:
                # Check required top-level groups
                if 'faces' not in f:
                    return False, 0, f"Missing 'faces' group in {h5_file.name}"
                
                if 'metadata' not in f:
                    return False, 0, f"Missing 'metadata' group in {h5_file.name}"
                
                faces_group = f['faces']
                metadata_group = f['metadata']
                
                # Check frame structure
                frame_groups = [key for key in faces_group.keys() if key.startswith('frame_')]
                if not frame_groups:
                    return False, 0, f"No frame groups found in {h5_file.name}"
                
                # Validate first frame structure
                first_frame = frame_groups[0]
                if first_frame not in faces_group:
                    return False, 0, f"First frame group not accessible in {h5_file.name}"
                
                frame_group = faces_group[first_frame]
                
                # Check face structure
                if 'face_000' not in frame_group:
                    return False, 0, f"Missing 'face_000' in frame {first_frame} of {h5_file.name}"
                
                face_group = frame_group['face_000']
                
                # Check required face datasets
                required_datasets = ['data', 'bbox', 'confidence', 'original_size']
                for dataset_name in required_datasets:
                    if dataset_name not in face_group:
                        return False, 0, f"Missing '{dataset_name}' in face_000 of {h5_file.name}"
                
                # Check data shape and type
                data = face_group['data']
                if data.ndim != 3:
                    return False, 0, f"Face data must be 3D (H, W, C), got {data.ndim}D in {h5_file.name}"
                
                if data.shape[2] != 3:
                    return False, 0, f"Face data must have 3 channels, got {data.shape[2]} in {h5_file.name}"
                
                # Check if uint8
                if data.dtype != np.uint8:
                    return False, 0, f"Face data must be uint8, got {data.dtype} in {h5_file.name}"
                
                # Count total frames
                num_frames = len(frame_groups)
                
                # STRICT VALIDATION: Must have exactly 30 frames
                if num_frames != 30:
                    return False, 0, f"Must have exactly 30 frames, got {num_frames} in {h5_file.name}"
                
                # Validate all frame groups exist and are sequential
                expected_frames = [f'frame_{i:03d}' for i in range(30)]
                missing_frames = [frame for frame in expected_frames if frame not in faces_group]
                if missing_frames:
                    return False, 0, f"Missing frames in {h5_file.name}: {missing_frames[:5]}..."
                
                # Validate ALL 30 frames have the correct structure
                for frame_idx in range(30):
                    frame_key = f'frame_{frame_idx:03d}'
                    frame_group = faces_group[frame_key]
                    
                    # Check face structure
                    if 'face_000' not in frame_group:
                        return False, 0, f"Missing 'face_000' in {frame_key} of {h5_file.name}"
                    
                    face_group = frame_group['face_000']
                    
                    # Check required face datasets
                    required_datasets = ['data', 'bbox', 'confidence', 'original_size']
                    for dataset_name in required_datasets:
                        if dataset_name not in face_group:
                            return False, 0, f"Missing '{dataset_name}' in {frame_key}/face_000 of {h5_file.name}"
                    
                    # Check data shape and type for this frame
                    data = face_group['data']
                    if data.ndim != 3:
                        return False, 0, f"Face data must be 3D (H, W, C), got {data.ndim}D in {frame_key} of {h5_file.name}"
                    
                    if data.shape[2] != 3:
                        return False, 0, f"Face data must have 3 channels, got {data.shape[2]} in {frame_key} of {h5_file.name}"
                    
                    # Check if uint8
                    if data.dtype != np.uint8:
                        return False, 0, f"Face data must be uint8, got {data.dtype} in {frame_key} of {h5_file.name}"
                
                return True, num_frames, ""
                
        except Exception as e:
            return False, 0, f"Error validating {h5_file.name}: {str(e)}"
    
    def _load_dataset_samples(self, dataset_path: str):
        """Load samples from a single dataset directory with strict format validation"""
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            logger.warning(f"Dataset path does not exist: {dataset_path}")
            return
        
        # Find all H5 files
        h5_files = list(dataset_path.glob("*.h5"))
        logger.info(f"Found {len(h5_files)} H5 files in {dataset_path}")
        
        if self.max_samples_per_dataset:
            h5_files = h5_files[:self.max_samples_per_dataset]
            logger.info(f"Limited to {len(h5_files)} H5 files per dataset")
        
        valid_files = 0
        invalid_files = 0
        
        for h5_file in h5_files:
            try:
                # Strictly validate H5 format
                is_valid, num_frames, error_msg = self._validate_h5_format(h5_file)
                
                if not is_valid:
                    logger.error(f"❌ Invalid H5 format in {h5_file.name}: {error_msg}")
                    invalid_files += 1
                    continue
                
                # Format is valid, add frames as samples
                for frame_idx in range(num_frames):
                    self.samples.append({
                        'h5_path': str(h5_file),
                        'frame_idx': frame_idx,
                        'dataset_path': str(dataset_path)
                    })
                
                valid_files += 1
                logger.debug(f"✅ Validated {h5_file.name}: {num_frames} frames")
                        
            except Exception as e:
                logger.error(f"❌ Error reading {h5_file.name}: {e}")
                invalid_files += 1
                continue
        
        logger.info(f"Dataset {dataset_path}: {valid_files} valid files, {invalid_files} invalid files")
        
        if invalid_files > 0:
            logger.warning(f"⚠️  Found {invalid_files} invalid H5 files in {dataset_path}")
            logger.warning("All H5 files must follow the exact mp4_utils format")
        
        return invalid_files
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single frame from an H5 file"""
        sample = self.samples[idx]
        
        try:
            with h5py.File(sample['h5_path'], 'r') as f:
                # Read from CCA format: faces/frame_XXX/face_000/data
                frame_key = f'frame_{sample["frame_idx"]:03d}'
                face_key = 'face_000'
                
                if frame_key in f['faces'] and face_key in f['faces'][frame_key]:
                    frame_data = f['faces'][frame_key][face_key]['data']  # (H, W, 3)
                    
                    # Convert from (H, W, 3) to (3, H, W) for consistency
                    frame_data = np.transpose(frame_data, (2, 0, 1))  # (3, H, W)
                    
                    # Convert to torch tensor and normalize to [0, 1] for DINOv2
                    if frame_data.dtype == np.uint8:
                        frame_data = frame_data.astype(np.float32) / 255.0
                    
                    frame_tensor = torch.from_numpy(frame_data).float()
                    
                    return {
                        'frame': frame_tensor,
                        'h5_path': sample['h5_path'],
                        'frame_idx': sample['frame_idx'],
                        'dataset_path': sample['dataset_path']
                    }
                else:
                    logger.warning(f"Frame structure not found in {sample['h5_path']}")
                    # Return a dummy frame if structure is missing
                    dummy_frame = torch.zeros(3, 518, 518, dtype=torch.float32)
                    return {
                        'frame': dummy_frame,
                        'h5_path': sample['h5_path'],
                        'frame_idx': sample['frame_idx'],
                        'dataset_path': sample['dataset_path']
                    }
                
        except Exception as e:
            logger.error(f"Error loading frame from {sample['h5_path']}: {e}")
            # Return a dummy frame if loading fails
            dummy_frame = torch.zeros(3, 518, 518, dtype=torch.float32)
            return {
                'frame': dummy_frame,
                'h5_path': sample['h5_path'],
                'frame_idx': sample['frame_idx'],
                'dataset_path': sample['dataset_path']
            }


def collect_patch_features_combined(
    dataset_paths: List[str],
    max_samples_per_dataset: Optional[int] = None,
    batch_size: int = 8,
    device: str = "cpu",
    num_workers: int = 0,
    image_size: int = 518
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Collect patch features from combined datasets
    
    Args:
        dataset_paths: List of paths to dataset directories
        max_samples_per_dataset: Maximum samples to process from each dataset
        batch_size: Batch size for processing
        device: Device to use
        num_workers: Number of data loader workers
        image_size: Input image size (should be 518)
    
    Returns:
        patch_features: (N, embed_dim) array of patch features
        metadata: List of metadata for each patch
    """
    logger.info(f"Loading combined datasets from: {dataset_paths}")
    logger.info(f"Image size: {image_size}x{image_size}")
    logger.info(f"Device: {device}")
    
    # Initialize DINOv2 tokenizer
    if image_size == 518:
        tokenizer = DINOv2BaseTokenizer(device=device)
        expected_patches = 1369  # 37x37 grid
        embed_dim = 768
        logger.info(f"Using DINOv2 base model with {expected_patches} patches ({embed_dim}-dim embeddings)")
    else:
        raise ValueError(f"Unsupported image size: {image_size}. Must be 518 for this script.")
    
    # Load combined dataset
    dataset = CombinedH5Dataset(dataset_paths, max_samples_per_dataset)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device != "cpu"),
        persistent_workers=(num_workers > 0),
        drop_last=False
    )
    
    logger.info(f"Dataset loaded: {len(dataset)} samples, {len(dataloader)} batches")
    
    # Collect features incrementally
    all_features_list = []
    total_frames = 0
    
    logger.info("Processing batches and collecting features...")
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        try:
            # Extract frames from batch
            frames = batch['frame']  # (batch_size, 3, 518, 518)
            
            # Move to device
            frames = frames.to(device)
            
            # Extract patch tokens using DINOv2 tokenizer
            with torch.no_grad():
                patch_tokens, _ = tokenizer(frames)  # (batch_size, num_patches, embed_dim)
            
            # Move to CPU and convert to numpy
            patch_tokens = patch_tokens.cpu().numpy()
            
            # Keep original shape (batch_size, num_patches, embed_dim) for incremental PCA
            batch_size_actual = patch_tokens.shape[0]
            num_patches = patch_tokens.shape[1]
            
            # Add to our list with original shape
            all_features_list.append(patch_tokens)
            total_frames += batch_size_actual
            
            logger.debug(f"Batch {batch_idx}: Added {batch_size_actual * num_patches} patches, total frames: {total_frames}")
            
            # Memory cleanup
            if device != "cpu":
                torch.cuda.empty_cache()
            
            # Log progress every 10 batches
            if batch_idx % 10 == 0:
                logger.info(f"Processed {batch_idx + 1}/{len(dataloader)} batches")
                logger.info(f"Total frames processed: {total_frames}")
                if device != "cpu":
                    logger.info(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
                
                # Memory cleanup every 10 batches
                if device != "cpu":
                    torch.cuda.empty_cache()
                import gc
                gc.collect()
        
        except Exception as e:
            logger.error(f"Error processing batch {batch_idx}: {e}")
            continue
    
    logger.info(f"Feature collection completed!")
    logger.info(f"Total frames: {total_frames}")
    logger.info(f"Features list length: {len(all_features_list)}")
    
    # Final memory cleanup
    if device != "cpu":
        torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    if not all_features_list:
        raise RuntimeError("No features collected from datasets")
    
    return all_features_list, total_frames


def compute_pca_directions_incremental(features_list, n_components=384, device="cpu"):
    """
    Compute PCA directions incrementally by processing batches on GPU and transferring to CPU for PCA
    
    Args:
        features_list: List of feature arrays from different clips
        n_components: Number of PCA components to compute
        device: Device to use for feature processing
    
    Returns:
        pca_components: (n_components, embed_dim) PCA components
        pca_mean: (embed_dim,) PCA mean
        pca_explained_variance: (n_components,) Explained variance
    """
    logger.info(f"Computing PCA directions incrementally for {n_components} components...")
    logger.info(f"Processing {len(features_list)} feature clips")
    
    # Import incremental PCA
    from sklearn.decomposition import IncrementalPCA
    
    # Initialize incremental PCA
    ipca = IncrementalPCA(n_components=n_components, batch_size=1000)
    
    # Track memory usage
    is_gpu = device != "cpu"
    if is_gpu:
        logger.info(f"Initial GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
    
    total_samples_processed = 0
    total_features_processed = 0
    
    # Process features in batches
    for batch_idx, features in enumerate(tqdm(features_list, desc="Computing PCA incrementally")):
        try:
            # Validate features shape
            if len(features.shape) != 3:
                logger.warning(f"Skipping batch {batch_idx}: invalid shape {features.shape}")
                continue
            
            # Reshape features for PCA: (clip_length, num_patches, embed_dim) -> (total_patches, embed_dim)
            clip_length, num_patches, embed_dim = features.shape
            features_reshaped = features.reshape(-1, embed_dim)  # (clip_length * num_patches, embed_dim)
            
            logger.debug(f"Batch {batch_idx}: {features.shape} -> {features_reshaped.shape}")
            
            # Move features to GPU if needed for any processing
            if is_gpu:
                features_gpu = torch.from_numpy(features_reshaped).float().to(device)
                
                # Any GPU processing can go here if needed
                # For now, just transfer back to CPU
                features_cpu = features_gpu.cpu().numpy()
                
                # Clear GPU memory
                del features_gpu
                torch.cuda.empty_cache()
            else:
                features_cpu = features_reshaped
            
            # Perform incremental PCA on this batch
            ipca.partial_fit(features_cpu)
            
            # Update counters
            total_samples_processed += features_cpu.shape[0]
            total_features_processed += 1
            
            # Clear CPU memory
            del features_cpu, features_reshaped
            
            # Log progress and memory status
            if batch_idx % 10 == 0:
                logger.info(f"Processed {batch_idx + 1}/{len(features_list)} batches")
                logger.info(f"Total samples: {total_samples_processed}, Total features: {total_features_processed}")
                if is_gpu:
                    logger.info(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
                    logger.info(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1024**3:.1f} GB")
                
                # Memory cleanup every 10 batches
                if is_gpu:
                    torch.cuda.empty_cache()
                import gc
                gc.collect()
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_idx}: {e}")
            continue
    
    # Final memory cleanup
    if is_gpu:
        torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    # Get final PCA results
    logger.info("Finalizing PCA computation...")
    pca_components = ipca.components_  # (n_components, embed_dim)
    pca_mean = ipca.mean_  # (embed_dim,)
    pca_explained_variance = ipca.explained_variance_  # (n_components,)
    
    logger.info(f"PCA computation completed!")
    logger.info(f"Components shape: {pca_components.shape}")
    logger.info(f"Explained variance shape: {pca_explained_variance.shape}")
    logger.info(f"Total samples processed: {total_samples_processed}")
    logger.info(f"Total features processed: {total_features_processed}")
    
    return pca_components, pca_mean, pca_explained_variance


def extract_positional_embeddings(tokenizer, device: str) -> np.ndarray:
    """Extract positional embeddings from the tokenizer"""
    logger.info("Extracting positional embeddings...")
    
    try:
        # Create a dummy input to get positional embeddings
        dummy_input = torch.zeros(1, 3, 518, 518, device=device)
        with torch.no_grad():
            _, pos_embeddings = tokenizer(dummy_input)
        
        # Move to CPU and convert to numpy
        pos_embeddings = pos_embeddings.cpu().numpy()
        logger.info(f"Positional embeddings shape: {pos_embeddings.shape}")
        
        return pos_embeddings
        
    except Exception as e:
        logger.warning(f"Could not extract positional embeddings: {e}")
        return np.array([])


def save_pca_results(
    pca_components: np.ndarray,
    pca_mean: np.ndarray,
    pca_explained_variance: np.ndarray,
    pos_embeddings: np.ndarray,
    output_path: str,
    image_size: int,
    embed_dim: int,
    patches_per_frame: int,
    grid_size: int,
    patch_size: int,
    total_patches: int,
    unique_clips: int,
    pca_components_count: int,
    dataset_paths: List[str]
):
    """Save PCA results to JSON file"""
    
    results = {
        "metadata": {
            "creation_timestamp": datetime.datetime.now().isoformat(),
            "image_size": image_size,
            "embed_dim": embed_dim,
            "patches_per_frame": patches_per_frame,
            "grid_size": grid_size,
            "patch_size": patch_size,
            "total_patches": total_patches,
            "unique_clips": unique_clips,
            "pca_components_count": pca_components_count,
            "dataset_paths": dataset_paths,
            "description": "PCA directions computed from combined DINOv2 features"
        },
        "pca_components": pca_components.tolist(),
        "pca_mean": pca_mean.tolist(),
        "pca_explained_variance": pca_explained_variance.tolist(),
        "positional_embeddings": pos_embeddings.tolist() if pos_embeddings.size > 0 else []
    }
    
    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"PCA results saved to: {output_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Compute PCA directions from DINOv2 patch features from combined datasets")
    
    # Parse command line arguments
    parser.add_argument("--dataset_paths", type=str, nargs='+', required=True,
                       help="Paths to dataset directories (H5 files with 'faces/frame_XXX/face_000/data')")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Path to save PCA results JSON file")
    parser.add_argument("--max_samples_per_dataset", type=int, default=None,
                       help="Maximum samples to process from each dataset (default: all)")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for processing")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto/cpu/cuda)")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loader workers")
    parser.add_argument("--n_components", type=int, default=384,
                       help="Number of PCA components to compute")
    parser.add_argument("--image_size", type=int, default=518,
                       help="Input image size (default: 518)")
    
    args = parser.parse_args()
    
    # Auto-detect device if requested
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Auto-detected device: {args.device}")
    
    # Validate arguments
    logger.info(f"Validating arguments...")
    logger.info(f"Dataset paths: {args.dataset_paths}")
    logger.info(f"Output path: {args.output_path}")
    logger.info(f"Image size: {args.image_size}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Max samples per dataset: {args.max_samples_per_dataset}")
    
    # Validate dataset paths
    for dataset_path in args.dataset_paths:
        if not os.path.exists(dataset_path):
            error_msg = f"Dataset path does not exist: {dataset_path}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    if args.image_size != 518:
        error_msg = f"Image size must be 518 for this script, got {args.image_size}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Create output directory
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
    
    # GPU memory management setup
    is_gpu = args.device != "cpu"
    if is_gpu:
        torch.cuda.empty_cache()
        logger.info(f"Initial GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
        logger.info(f"Initial GPU Memory Reserved: {torch.cuda.memory_reserved() / 1024**3:.1f} GB")
    
    # Collect features from combined datasets
    logger.info("Collecting patch features from combined datasets...")
    features_list, total_frames = collect_patch_features_combined(
        dataset_paths=args.dataset_paths,
        max_samples_per_dataset=args.max_samples_per_dataset,
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers,
        image_size=args.image_size
    )
    
    # Compute PCA directions
    logger.info("Computing PCA directions...")
    pca_components, pca_mean, pca_explained_variance = compute_pca_directions_incremental(
        features_list, 
        n_components=args.n_components, 
        device=args.device
    )
    
    # Initialize tokenizer for positional embeddings
    tokenizer = DINOv2BaseTokenizer(device=args.device)
    
    # Extract positional embeddings
    logger.info("Extracting positional embeddings...")
    pos_embeddings = extract_positional_embeddings(tokenizer, args.device)
    
    # Save PCA results
    logger.info("Saving PCA results...")
    save_pca_results(
        pca_components=pca_components,
        pca_mean=pca_mean,
        pca_explained_variance=pca_explained_variance,
        pos_embeddings=pos_embeddings,
        output_path=args.output_path,
        image_size=args.image_size,
        embed_dim=768,  # DINOv2 base model
        patches_per_frame=1369,  # 37x37 grid
        grid_size=37,
        patch_size=14,
        total_patches=total_frames * 1369,
        unique_clips=total_frames,
        pca_components_count=args.n_components,
        dataset_paths=args.dataset_paths
    )
    
    logger.info(f"PCA directions saved to: {args.output_path}")
    logger.info("Processing completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Script failed with error: {e}")
        logger.error("Please check your arguments and dataset paths")
        raise
