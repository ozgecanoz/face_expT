#!/usr/bin/env python3
"""
Compute PCA directions from DINOv2 patch features
Loads a dataset, extracts patch features, computes PCA, and saves results
"""

import torch
import numpy as np
import json
import argparse
import os
import logging
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from sklearn.decomposition import PCA
from datetime import datetime

# Add the project root to the path
import sys
sys.path.append('.')

# Add the face_model directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.dataset import FaceDataset
from models.dinov2_tokenizer import DINOv2Tokenizer
from torch.utils.data import DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_dataset_image_size(dataset_path: str, expected_size: int = 518) -> bool:
    """
    Validate that the dataset contains images of the expected size
    
    Args:
        dataset_path: Path to the dataset
        expected_size: Expected image size (518 or 224)
    
    Returns:
        is_valid: True if dataset images match expected size
    """
    logger.info(f"Validating dataset image size...")
    
    try:
        # Load a small sample to check image dimensions
        temp_dataset = FaceDataset(dataset_path, max_samples=10)
        temp_dataloader = DataLoader(temp_dataset, batch_size=2, shuffle=False)
        
        for batch in temp_dataloader:
            frames = batch['frames'][0]  # Get first clip
            if frames.numel() > 0:
                height, width = frames.shape[2], frames.shape[3]
                logger.info(f"Sample image dimensions: {width}x{height}")
                
                if height == expected_size and width == expected_size:
                    logger.info(f"✅ Dataset images match expected size: {expected_size}x{expected_size}")
                    return True
                else:
                    logger.info(f"⚠️  Dataset images are {width}x{height}, expected {expected_size}x{expected_size}")
                    logger.info(f"   Images will be automatically resized during processing")
                    return False
        
        logger.warning("Could not determine image dimensions from dataset")
        return False
        
    except Exception as e:
        logger.warning(f"Could not validate dataset image size: {e}")
        return False


def collect_patch_features(
    dataset_path: str,
    max_samples: Optional[int] = None,
    batch_size: int = 8,
    device: str = "cpu",
    num_workers: int = 0,
    image_size: int = 518,
    model_name: str = 'vit_small_patch14_dinov2.lvd142m'
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Collect patch features from all images in the dataset
    
    Args:
        dataset_path: Path to the dataset
        max_samples: Maximum number of samples to process
        batch_size: Batch size for processing
        device: Device to use
        num_workers: Number of data loader workers
        image_size: Input image size (518 or 224)
        model_name: Model name to use ('dinov2-base' for Hugging Face model)
    
    Returns:
        patch_features: (N, embed_dim) array of patch features
        metadata: List of metadata for each patch
    """
    logger.info(f"Loading dataset from: {dataset_path}")
    logger.info(f"Image size: {image_size}x{image_size}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Device: {device}")
    
    # Validate dataset image size first
    dataset_has_correct_size = validate_dataset_image_size(dataset_path, image_size)
    
    # Load dataset
    dataset = FaceDataset(dataset_path, max_samples=max_samples)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,  # Keep order for reproducibility
        num_workers=num_workers,
        pin_memory=(device != "cpu"),  # Pin memory for GPU
        persistent_workers=(num_workers > 0),
        drop_last=False
    )
    
    logger.info(f"Dataset loaded: {len(dataset)} samples, {len(dataloader)} batches")
    
    # Initialize DINOv2 tokenizer with appropriate model
    if image_size == 224:
        model_name = 'vit_small_patch16_224.augreg_in21k'
        expected_patches = 196  # 14x14 grid
        logger.info(f"Using 224x224 model: {model_name} with {expected_patches} patches")
        tokenizer = DINOv2Tokenizer(model_name=model_name, device=device)
    elif image_size == 518:
        # Check if user wants the base model
        if 'dinov2-base' in model_name.lower():
            from models.dinov2_tokenizer import DINOv2BaseTokenizer
            model_name = 'facebook/dinov2-base'
            expected_patches = 1369  # 37x37 grid
            logger.info(f"Using 518x518 base model: {model_name} with {expected_patches} patches (768-dim embeddings)")
            tokenizer = DINOv2BaseTokenizer(device=device)
        else:
            model_name = 'vit_small_patch14_dinov2.lvd142m'
            expected_patches = 1369  # 37x37 grid
            logger.info(f"Using 518x518 model: {model_name} with {expected_patches} patches (384-dim embeddings)")
            tokenizer = DINOv2Tokenizer(model_name=model_name, device=device)
    else:
        raise ValueError(f"Unsupported image size: {image_size}. Must be 224 or 518.")
    
    # Validate tokenizer configuration
    actual_patches = tokenizer.get_num_patches()
    actual_input_size = tokenizer.get_input_size()
    if actual_patches != expected_patches or actual_input_size != image_size:
        raise ValueError(f"Tokenizer mismatch: expected {expected_patches} patches for {image_size}x{image_size}, "
                       f"got {actual_patches} patches for {actual_input_size}x{actual_input_size}")
    
    # Get embedding dimension from tokenizer
    embed_dim = tokenizer.get_embed_dim()
    logger.info(f"Embedding dimension: {embed_dim}")
    
    # GPU memory management setup
    is_gpu = device != "cpu"
    if is_gpu:
        torch.cuda.empty_cache()
        logger.info(f"Initial GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
        logger.info(f"Initial GPU Memory Reserved: {torch.cuda.memory_reserved() / 1024**3:.1f} GB")
    
    # Collect patch features with optimized memory management
    all_patch_features = []
    metadata = []
    gpu_features = []  # Accumulate features on GPU before CPU transfer
    
    # Determine optimal batch size for CPU transfers
    cpu_transfer_batch_size = 100 if is_gpu else 1  # Transfer to CPU every 100 features on GPU
    
    logger.info("Collecting patch features...")
    logger.info(f"Processing strategy: {'GPU batch processing' if is_gpu else 'CPU processing'}")
    if not dataset_has_correct_size:
        logger.info("Note: Images will be automatically resized during processing")
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        # Collect all frames from all clips in this batch
        all_frames = []
        all_subject_ids = []
        all_clip_lengths = []
        
        for clip_idx, (frames, subject_id) in enumerate(zip(batch['frames'], batch['subject_id'])):
            # frames: (num_frames, 3, H, W) - entire clip
            num_frames = frames.shape[0]
            original_height, original_width = frames.shape[2], frames.shape[3]
            
            # Validate input image size - assume dataset has 518x518 but model may need 224x224
            if original_height != image_size or original_width != image_size:
                if batch_idx == 0:  # Only log once per batch
                    logger.info(f"Resizing frames from {original_height}x{original_width} to {image_size}x{image_size}")
                # Resize frames using torch.nn.functional.interpolate
                import torch.nn.functional as F
                frames_resized = F.interpolate(
                    frames, 
                    size=(image_size, image_size), 
                    mode='bilinear', 
                    align_corners=False,
                    antialias=True
                )
                frames = frames_resized
            else:
                logger.debug(f"Frames already correct size: {image_size}x{image_size}")
            
            all_frames.append(frames)
            all_subject_ids.extend([subject_id] * num_frames)
            all_clip_lengths.append(num_frames)
        
        # Process entire batch at once if we have frames
        if all_frames:
            try:
                # Concatenate all frames into one large batch
                combined_frames = torch.cat(all_frames, dim=0)  # (total_frames, 3, image_size, image_size)
                
                # Move ENTIRE batch to GPU at once
                if is_gpu:
                    combined_frames = combined_frames.to(device)
                    logger.debug(f"Batch {batch_idx}: Moved {combined_frames.shape[0]} frames to GPU")
                
                # Process ALL frames in one forward pass
                with torch.no_grad():
                    all_patch_tokens, _ = tokenizer(combined_frames)  # (total_frames, num_patches, embed_dim)
                
                # Now split back by clip and accumulate features
                start_idx = 0
                for clip_idx, clip_length in enumerate(all_clip_lengths):
                    clip_tokens = all_patch_tokens[start_idx:start_idx + clip_length]  # (clip_length, num_patches, embed_dim)
                    clip_subject_id = batch['subject_id'][clip_idx]
                    
                    # Accumulate features on GPU for batch transfer
                    if is_gpu:
                        gpu_features.append(clip_tokens)  # Keep on GPU
                    else:
                        # For CPU, transfer immediately
                        clip_features = clip_tokens.cpu().numpy()
                        all_patch_features.append(clip_features)
                    
                    # Store metadata for each patch in this clip
                    for frame_idx in range(clip_length):
                        for patch_idx in range(clip_tokens.shape[1]):
                            metadata.append({
                                'batch_idx': batch_idx,
                                'clip_idx': clip_idx,
                                'frame_idx': frame_idx,
                                'patch_idx': patch_idx,
                                'subject_id': clip_subject_id,
                                'total_patches_per_frame': clip_tokens.shape[1],
                                'image_size': image_size,
                                'model_name': model_name,
                                'embed_dim': embed_dim,
                                'original_size': f"{original_height}x{original_width}",
                                'resized': original_height != image_size or original_width != image_size
                            })
                    
                    start_idx += clip_length
                
                # Batch transfer to CPU for GPU processing
                if is_gpu and len(gpu_features) >= cpu_transfer_batch_size:
                    try:
                        # Concatenate all accumulated features
                        batch_features = torch.cat(gpu_features, dim=0).cpu().numpy()
                        all_patch_features.extend(batch_features)
                        gpu_features = []  # Clear GPU memory
                        
                        # Log memory status
                        if batch_idx % 10 == 0:  # Log every 10 batches
                            logger.info(f"Batch {batch_idx}: Transferred {cpu_transfer_batch_size} features to CPU")
                            logger.info(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
                            logger.info(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1024**3:.1f} GB")
                    
                    except Exception as e:
                        logger.error(f"Error during batch transfer: {e}")
                        # Fallback: transfer features one by one
                        for feature in gpu_features:
                            all_patch_features.append(feature.cpu().numpy())
                        gpu_features = []
                
            except torch.cuda.OutOfMemoryError as e:
                if is_gpu:
                    logger.warning(f"GPU OOM at batch {batch_idx}. Clearing cache and retrying...")
                    torch.cuda.empty_cache()
                    
                    # Try to process with smaller batches or skip
                    try:
                        # Clear accumulated GPU features and transfer to CPU
                        if gpu_features:
                            batch_features = torch.cat(gpu_features, dim=0).cpu().numpy()
                            all_patch_features.extend(batch_features)
                            gpu_features = []
                            torch.cuda.empty_cache()
                        
                        # Retry with the same batch
                        combined_frames = torch.cat(all_frames, dim=0).to(device)
                        with torch.no_grad():
                            all_patch_tokens, _ = tokenizer(combined_frames)
                        
                        logger.info("Successfully recovered from OOM")
                        
                    except torch.cuda.OutOfMemoryError:
                        logger.error(f"Failed to recover from OOM, skipping batch {batch_idx}")
                        continue
                else:
                    raise e
            
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                continue
        
        # Final transfer for this batch if using GPU
        if is_gpu and gpu_features:
            try:
                batch_features = torch.cat(gpu_features, dim=0).cpu().numpy()
                all_patch_features.extend(batch_features)
                gpu_features = []
                torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"Error during final batch transfer: {e}")
                # Fallback: transfer features one by one
                for feature in gpu_features:
                    all_patch_features.append(feature.cpu().numpy())
                gpu_features = []
    
    # Concatenate all patch features
    patch_features = np.concatenate(all_patch_features, axis=0)  # (N, embed_dim)
    
    # Final GPU memory cleanup
    if is_gpu:
        torch.cuda.empty_cache()
        logger.info(f"Final GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
        logger.info(f"Final GPU Memory Reserved: {torch.cuda.memory_reserved() / 1024**3:.1f} GB")
    
    logger.info(f"Collected {patch_features.shape[0]} patch features from {len(metadata)} patches")
    logger.info(f"Feature shape: {patch_features.shape}")
    logger.info(f"Expected patches per frame: {expected_patches}")
    
    return patch_features, metadata


def compute_pca_directions(
    patch_features: np.ndarray,
    n_components: int = 10,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute PCA directions from patch features
    
    Args:
        patch_features: (N, embed_dim) array of patch features
        n_components: Number of PCA components to compute
        random_state: Random state for reproducibility
    
    Returns:
        pca_components: (n_components, embed_dim) PCA components
        pca_mean: (embed_dim,) mean of features
        explained_variance_ratio: (n_components,) explained variance ratio
    """
    logger.info(f"Computing PCA with {n_components} components...")
    logger.info(f"Input features shape: {patch_features.shape}")
    
    # Memory monitoring for large datasets
    feature_memory_gb = patch_features.nbytes / 1024**3
    logger.info(f"Feature array memory usage: {feature_memory_gb:.2f} GB")
    
    # Check if we have enough samples for the requested components
    if patch_features.shape[0] < n_components:
        logger.warning(f"Requested {n_components} components but only have {patch_features.shape[0]} samples")
        n_components = min(n_components, patch_features.shape[0])
        logger.info(f"Reduced to {n_components} components")
    
    # Initialize PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    
    # Fit PCA with progress monitoring
    logger.info("Fitting PCA...")
    pca.fit(patch_features)
    
    # Get results
    pca_components = pca.components_  # (n_components, embed_dim)
    pca_mean = pca.mean_  # (embed_dim,)
    explained_variance_ratio = pca.explained_variance_ratio_  # (n_components,)
    
    logger.info(f"PCA computed successfully")
    logger.info(f"Explained variance ratios: {explained_variance_ratio}")
    logger.info(f"Total explained variance: {np.sum(explained_variance_ratio):.4f}")
    
    # Log memory usage of results
    results_memory_gb = (pca_components.nbytes + pca_mean.nbytes + explained_variance_ratio.nbytes) / 1024**3
    logger.info(f"PCA results memory usage: {results_memory_gb:.3f} GB")
    
    return pca_components, pca_mean, explained_variance_ratio


def monitor_gpu_memory(device: str, stage: str = ""):
    """Monitor GPU memory usage"""
    if device == "cuda":
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - reserved
        
        logger.info(f"GPU Memory {stage}: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB, Free: {free:.2f} GB")


def save_pca_results(
    pca_components: np.ndarray,
    pca_mean: np.ndarray,
    explained_variance_ratio: np.ndarray,
    output_path: str,
    metadata: List[Dict],
    image_size: int,
    model_name: str
):
    """
    Save PCA results to JSON file
    
    Args:
        pca_components: PCA components matrix
        pca_mean: PCA mean vector
        explained_variance_ratio: Explained variance ratio for each component
        output_path: Path to save the results
        metadata: List of metadata for each patch
        image_size: Input image size used
        model_name: Model name used
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Extract some statistics from metadata
    total_patches = len(metadata)
    unique_subjects = len(set(m['subject_id'] for m in metadata))
    unique_clips = len(set((m['batch_idx'], m['clip_idx']) for m in metadata))
    
    # Get embedding dimension from first metadata entry
    embed_dim = metadata[0]['embed_dim'] if metadata else 384
    
    # Create results dictionary
    results = {
        'pca_components': pca_components.tolist(),
        'pca_mean': pca_mean.tolist(),
        'explained_variance_ratio': explained_variance_ratio.tolist(),
        'configuration': {
            'image_size': image_size,
            'model_name': model_name,
            'embed_dim': embed_dim,
            'patches_per_frame': pca_components.shape[1],
            'grid_size': int(np.sqrt(pca_components.shape[1])),
            'patch_size': image_size // int(np.sqrt(pca_components.shape[1])),
            'total_patches': total_patches,
            'unique_subjects': unique_subjects,
            'unique_clips': unique_clips,
            'pca_components': pca_components.shape[0],
            'timestamp': datetime.now().isoformat()
        },
        'metadata': metadata
    }
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"PCA results saved to: {output_path}")
    logger.info(f"Configuration: {image_size}x{image_size} images, {model_name} model, {embed_dim}-dim embeddings")
    logger.info(f"PCA components: {pca_components.shape[0]} components from {pca_components.shape[1]} features")
    logger.info(f"Total patches: {total_patches} from {unique_subjects} subjects, {unique_clips} clips")


def compute_optimal_batch_size(device: str, image_size: int, embed_dim: int, target_memory_gb: float = 0.7) -> int:
    """
    Compute optimal batch size based on available GPU memory
    
    Args:
        device: Device string ('cuda' or 'cpu')
        image_size: Input image size
        embed_dim: Embedding dimension
        target_memory_gb: Target GPU memory usage (fraction of total)
    
    Returns:
        optimal_batch_size: Recommended batch size
    """
    if device == "cpu":
        return 8  # Default for CPU
    
    try:
        # Get GPU memory info
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        available_memory = total_memory * target_memory_gb
        
        # Estimate memory per sample
        # Input: (batch_size, 3, image_size, image_size) - float32
        input_memory_per_sample = (3 * image_size * image_size * 4) / 1024**3  # GB
        
        # Output: (batch_size, num_patches, embed_dim) - float32
        num_patches = (image_size // 14) ** 2  # Approximate
        output_memory_per_sample = (num_patches * embed_dim * 4) / 1024**3  # GB
        
        # Model memory (rough estimate)
        model_memory = 0.5  # GB for DINOv2 base model
        
        # Available memory for batch processing
        batch_memory = available_memory - model_memory
        
        # Calculate optimal batch size
        memory_per_sample = input_memory_per_sample + output_memory_per_sample
        optimal_batch_size = max(1, int(batch_memory / memory_per_sample))
        
        # Cap at reasonable limits
        optimal_batch_size = min(optimal_batch_size, 32)
        
        logger.info(f"GPU Memory: {total_memory:.1f} GB total, {available_memory:.1f} GB available")
        logger.info(f"Memory per sample: {memory_per_sample:.3f} GB")
        logger.info(f"Optimal batch size: {optimal_batch_size}")
        
        return optimal_batch_size
        
    except Exception as e:
        logger.warning(f"Could not compute optimal batch size: {e}, using default")
        return 8


def main():
    """Main function to compute PCA directions"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Compute PCA directions from DINOv2 patch features")
    parser.add_argument("--image_size", type=int, default=518, 
                       help="Input image size (518 or 224)")
    parser.add_argument("--model_name", type=str, 
                       default='dinov2-base',
                       help="Model name to use ('dinov2-base' for Hugging Face model)")
    ''' # for local laptop
    parser.add_argument("--dataset_path", type=str, 
                       default="/Users/ozgewhiting/Documents/EQLabs/datasets_serial/CCA_train_db4_no_padding/CCA_train_db4_no_padding", 
                       help="Path to dataset directory")
    parser.add_argument("--output_path", type=str, 
                       default="/Users/ozgewhiting/Documents/EQLabs/datasets_serial/CCA_train_db4_no_padding/pca_directions_dinov2_base.json", 
                       help="Path to save PCA results")
    parser.add_argument("--max_samples", type=int, 
                       default=100, 
                       help="Maximum number of samples to process")
    parser.add_argument("--batch_size", type=int, default=None, 
                       help="Batch size for processing (auto-detect if None)")
    parser.add_argument("--device", type=str, default="auto", 
                       help="Device to use (auto/cpu/cuda)")
    parser.add_argument("--num_workers", type=int, default=None, 
                       help="Number of data loader workers (auto-detect if None)")
    parser.add_argument("--n_components", type=int, default=10,
                       help="Number of PCA components to compute")
    '''
    parser.add_argument("--dataset_path", type=str, 
                       default="/mnt/dataset-storage/dbs/CCA_train_db4_no_padding_keywords_offset_1.0", 
                       help="Path to dataset directory")
    parser.add_argument("--output_path", type=str, 
                       default="//mnt/dataset-storage/dbs/CCA_train_db4_keywords_offset_1.0_pca_dirs_base_1000.json", 
                       help="Path to save PCA results")
    parser.add_argument("--max_samples", type=int, 
                       default=1000, 
                       help="Maximum number of samples to process")
    parser.add_argument("--batch_size", type=int, default=16, 
                       help="Batch size for processing (auto-detect if None)")
    parser.add_argument("--device", type=str, default="cuda", 
                       help="Device to use (auto/cpu/cuda)")
    parser.add_argument("--num_workers", type=int, default=8, 
                       help="Number of data loader workers (auto-detect if None)")
    parser.add_argument("--n_components", type=int, default=384,
                       help="Number of PCA components to compute")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Auto-detect device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # Auto-detect optimal settings
    if args.batch_size is None:
        args.batch_size = compute_optimal_batch_size(device, args.image_size, 768 if 'dinov2-base' in args.model_name else 384)
    
    if args.num_workers is None:
        if device == "cuda":
            args.num_workers = min(8, os.cpu_count() or 4)  # More workers for GPU
        else:
            args.num_workers = min(4, os.cpu_count() or 2)  # Fewer workers for CPU
    
    logger.info("Starting PCA computation...")
    logger.info(f"Dataset path: {args.dataset_path}")
    logger.info(f"Output path: {args.output_path}")
    logger.info(f"Image size: {args.image_size}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Device: {device}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Number of workers: {args.num_workers}")
    logger.info(f"Max samples: {args.max_samples}")
    logger.info(f"PCA components: {args.n_components}")
    
    # GPU setup and memory info
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.cuda.empty_cache()
        monitor_gpu_memory(device, "Initial")
    
    # Collect patch features
    logger.info("="*50)
    logger.info("STAGE 1: Collecting patch features")
    logger.info("="*50)
    
    patch_features, metadata = collect_patch_features(
        dataset_path=args.dataset_path,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        device=device,
        num_workers=args.num_workers,
        image_size=args.image_size,
        model_name=args.model_name
    )
    
    monitor_gpu_memory(device, "After feature collection")
    
    # Compute PCA
    logger.info("="*50)
    logger.info("STAGE 2: Computing PCA")
    logger.info("="*50)
    
    pca_components, pca_mean, explained_variance_ratio = compute_pca_directions(
        patch_features, 
        n_components=args.n_components
    )
    
    monitor_gpu_memory(device, "After PCA computation")
    
    # Save results
    logger.info("="*50)
    logger.info("STAGE 3: Saving results")
    logger.info("="*50)
    
    save_pca_results(
        pca_components=pca_components,
        pca_mean=pca_mean,
        explained_variance_ratio=explained_variance_ratio,
        output_path=args.output_path,
        metadata=metadata,
        image_size=args.image_size,
        model_name=args.model_name
    )
    
    # Final memory cleanup
    if device == "cuda":
        torch.cuda.empty_cache()
        monitor_gpu_memory(device, "Final")
    
    logger.info(f"PCA results saved to: {args.output_path}")
    logger.info("="*50)
    logger.info("COMPLETED SUCCESSFULLY!")
    logger.info("="*50)


if __name__ == "__main__":
    main() 