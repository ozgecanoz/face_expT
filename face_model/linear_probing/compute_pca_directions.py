#!/usr/bin/env python3
"""
Compute PCA directions from DINOv2 patch features
Loads a dataset, extracts patch features, computes PCA, and saves results
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
from torch.utils.data import DataLoader
from tqdm import tqdm
import datetime

# Add the project root to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.dataset import FaceDataset
from models.dinov2_tokenizer import DINOv2Tokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
    logger.info(f"Mean shape: {pca_mean.shape}")
    logger.info(f"Explained variance shape: {pca_explained_variance.shape}")
    logger.info(f"Total samples processed: {total_samples_processed}")
    logger.info(f"Total features processed: {total_features_processed}")
    
    if is_gpu:
        logger.info(f"Final GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
    
    return pca_components, pca_mean, pca_explained_variance


def compute_pca_directions(features, n_components=384):
    """
    Legacy function - now calls incremental PCA for memory efficiency
    """
    logger.warning("Using incremental PCA for memory efficiency. Consider calling compute_pca_directions_incremental directly.")
    
    # Convert single features array to list format for incremental PCA
    if isinstance(features, np.ndarray):
        if len(features.shape) == 3:
            # (clips, frames_per_clip, embed_dim) -> list of (frames_per_clip, embed_dim)
            features_list = [features[i] for i in range(features.shape[0])]
        elif len(features.shape) == 2:
            # (samples, embed_dim) -> list of single samples
            features_list = [features[i:i+1] for i in range(features.shape[0])]
        else:
            raise ValueError(f"Unexpected features shape: {features.shape}")
    else:
        features_list = features
    
    return compute_pca_directions_incremental(features_list, n_components)


def monitor_gpu_memory(device: str, stage: str = ""):
    """Monitor GPU memory usage"""
    if device == "cuda":
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - reserved
        
        logger.info(f"GPU Memory {stage}: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB, Free: {free:.2f} GB")


def process_batch_for_pca(batch, batch_idx, tokenizer, device, image_size, model_name):
    """
    Process a batch for PCA computation, returning features on CPU to avoid memory issues
    
    Args:
        batch: Batch from dataloader
        batch_idx: Batch index for logging
        tokenizer: DINOv2 tokenizer
        device: Device to use
        image_size: Input image size
        model_name: Model name
    
    Returns:
        projected_features_list: List of feature arrays on CPU
        rgb_visualizations: List of RGB visualizations (not used for PCA)
    """
    projected_features_list = []
    rgb_visualizations = []
    
    try:
        # Collect all frames from all clips in this batch
        all_frames = []
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
            all_clip_lengths.append(num_frames)
        
        # Process entire batch at once if we have frames
        if all_frames:
            try:
                # Concatenate all frames into one large batch
                combined_frames = torch.cat(all_frames, dim=0)  # (total_frames, 3, image_size, image_size)
                
                # Move ENTIRE batch to GPU at once
                is_gpu = device != "cpu"
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
                    
                    # Move to CPU immediately to avoid GPU memory accumulation
                    clip_tokens_cpu = clip_tokens.cpu().numpy()
                    
                    # Store features on CPU (they'll be processed by incremental PCA)
                    projected_features_list.append(clip_tokens_cpu)
                    
                    # Create RGB visualization for debugging (optional)
                    # rgb_viz = create_rgb_visualization(clip_tokens_cpu)
                    # rgb_visualizations.append(rgb_viz)
                    
                    start_idx += clip_length
                
                # GPU memory cleanup
                if is_gpu:
                    torch.cuda.empty_cache()
                    
            except torch.cuda.OutOfMemoryError as e:
                if is_gpu:
                    logger.warning(f"GPU OOM at batch {batch_idx}. Clearing cache and retrying...")
                    torch.cuda.empty_cache()
                    
                    # Try to process with smaller batches or skip
                    try:
                        # Process clips one by one to reduce memory usage
                        for clip_idx, frames in enumerate(batch['frames']):
                            clip_projected_features = []
                            
                            for frame_idx in range(frames.shape[0]):
                                frame = frames[frame_idx:frame_idx+1].to(device)  # (1, 3, image_size, image_size)
                                
                                with torch.no_grad():
                                    frame_tokens, _ = tokenizer(frame)  # (1, num_patches, embed_dim)
                                
                                frame_tokens_cpu = frame_tokens.squeeze(0).cpu().numpy()  # (num_patches, embed_dim)
                                clip_projected_features.append(frame_tokens_cpu)
                                
                                # Clean up frame
                                del frame, frame_tokens
                                torch.cuda.empty_cache()
                            
                            # Stack for this clip
                            clip_projected_features = np.stack(clip_projected_features, axis=0)
                            projected_features_list.append(clip_projected_features)
                        
                        logger.info("Successfully recovered from OOM with clip-by-clip processing")
                        
                    except torch.cuda.OutOfMemoryError:
                        logger.error(f"Failed to recover from OOM, skipping batch {batch_idx}")
                        return [], []
                
                else:
                    raise e
            
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                return [], []
        
    except Exception as e:
        logger.error(f"Error in process_batch_for_pca: {e}")
        return [], []
    
    return projected_features_list, rgb_visualizations


def extract_positional_embeddings(tokenizer, device):
    """
    Extract learned positional embeddings from the DINOv2 tokenizer
    
    Args:
        tokenizer: DINOv2 tokenizer instance
        device: Device to use for extraction
    
    Returns:
        pos_embeddings: (num_patches, embed_dim) positional embeddings
    
    Raises:
        RuntimeError: If positional embeddings cannot be extracted
    """
    logger.info("Extracting positional embeddings from tokenizer...")
    
    try:
        # Check if tokenizer has positional embeddings
        if hasattr(tokenizer, 'model') and hasattr(tokenizer.model, 'embeddings'):
            # For Hugging Face models (DINOv2BaseTokenizer)
            if hasattr(tokenizer.model.embeddings, 'position_embeddings'):
                pos_embeddings = tokenizer.model.embeddings.position_embeddings.weight
                logger.info(f"Extracted positional embeddings from Hugging Face model: {pos_embeddings.shape}")
            else:
                raise RuntimeError("Hugging Face model doesn't have position_embeddings attribute")
        else:
            # For timm models (DINOv2Tokenizer)
            if hasattr(tokenizer.model, 'pos_embed'):
                pos_embeddings = tokenizer.model.pos_embed
                logger.info(f"Extracted positional embeddings from timm model: {pos_embeddings.shape}")
            else:
                raise RuntimeError("timm model doesn't have pos_embed attribute")
        
        # Move to CPU and convert to numpy
        if hasattr(pos_embeddings, 'cpu'):
            pos_embeddings = pos_embeddings.cpu().detach()
        
        # Ensure correct shape: (num_patches, embed_dim)
        if len(pos_embeddings.shape) == 3:
            # (1, num_patches, embed_dim) -> (num_patches, embed_dim)
            pos_embeddings = pos_embeddings.squeeze(0)
        elif len(pos_embeddings.shape) == 2:
            # Already (num_patches, embed_dim)
            pass
        else:
            raise RuntimeError(f"Unexpected positional embeddings shape: {pos_embeddings.shape}. Expected (num_patches, embed_dim) or (1, num_patches, embed_dim)")
        
        logger.info(f"Final positional embeddings shape: {pos_embeddings.shape}")
        return pos_embeddings
        
    except Exception as e:
        logger.error(f"Failed to extract positional embeddings: {e}")
        raise RuntimeError(f"Could not extract positional embeddings from tokenizer. This is required for the script to work. Error: {e}")


def save_pca_results(pca_components, pca_mean, pca_explained_variance, pos_embeddings, output_path, 
                    image_size, model_name, embed_dim, patches_per_frame, grid_size, 
                    patch_size, total_patches, unique_subjects, unique_clips, pca_components_count):
    """
    Save PCA results to JSON file
    
    Args:
        pca_components: PCA components array
        pca_mean: PCA mean array
        pca_explained_variance: PCA explained variance array
        pos_embeddings: Learned positional embeddings from tokenizer
        output_path: Path to save the JSON file
        image_size: Input image size
        model_name: Model name used
        embed_dim: Embedding dimension
        patches_per_frame: Number of patches per frame
        grid_size: Grid size (e.g., 37 for 37x37)
        patch_size: Patch size
        total_patches: Total number of patches
        unique_subjects: Number of unique subjects
        unique_clips: Number of unique clips
        pca_components_count: Number of PCA components
    """
    # Create results dictionary
    results = {
        'pca_components': pca_components.tolist(),
        'pca_mean': pca_mean.tolist(),
        'pca_explained_variance': pca_explained_variance.tolist(),
        'pos_embeddings': pos_embeddings.tolist(),
        'configuration': {
            'image_size': image_size,
            'model_name': model_name,
            'embed_dim': embed_dim,
            'patches_per_frame': patches_per_frame,
            'grid_size': grid_size,
            'patch_size': patch_size,
            'total_patches': total_patches,
            'unique_subjects': unique_subjects,
            'unique_clips': unique_clips,
            'pca_components': pca_components_count,
            'pos_embeddings_shape': list(pos_embeddings.shape),
            'pos_embeddings_type': 'learned',
            'timestamp': str(datetime.datetime.now())
        }
    }
    
    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"PCA results saved to: {output_path}")
    logger.info(f"Components: {pca_components.shape}")
    logger.info(f"Mean: {pca_mean.shape}")
    logger.info(f"Explained variance: {pca_explained_variance.shape}")
    logger.info(f"Positional embeddings: {pos_embeddings.shape}")
    logger.info(f"Positional embeddings type: learned")


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
    """Main function"""
    parser = argparse.ArgumentParser(description="Compute PCA directions from DINOv2 patch features")
    
    # Parse command line arguments
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
    
    # Cloud/remote defaults
    parser.add_argument("--dataset_path", type=str, 
                       default="/mnt/dataset-storage/dbs/CCA_train_db4_no_padding_keywords_offset_1.0", 
                       help="Path to dataset directory")
    parser.add_argument("--output_path", type=str, 
                       default="/mnt/dataset-storage/dbs/CCA_train_db4_keywords_offset_1.0_pca_dirs_base_1000.json", 
                       help="Path to save PCA results")
    parser.add_argument("--max_samples", type=int, 
                       default=1000, 
                       help="Maximum number of samples to process")
    parser.add_argument("--batch_size", type=int, default=16, 
                       help="Batch size for processing")
    parser.add_argument("--device", type=str, default="cuda", 
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--num_workers", type=int, default=8, 
                       help="Number of data loader workers")
    parser.add_argument("--n_components", type=int, default=384,
                       help="Number of PCA components to compute")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.dataset_path):
        raise ValueError(f"Dataset path does not exist: {args.dataset_path}")
    
    if args.image_size not in [224, 518]:
        raise ValueError(f"Image size must be 224 or 518, got {args.image_size}")
    
    # Create output directory
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Collect patch features incrementally to avoid memory issues
    logger.info("Collecting patch features incrementally...")
    
    # Load dataset
    dataset = FaceDataset(args.dataset_path, max_samples=args.max_samples)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,  # Keep order for reproducibility
        num_workers=args.num_workers,
        pin_memory=(args.device != "cpu"),
        persistent_workers=(args.num_workers > 0),
        drop_last=False
    )
    
    logger.info(f"Dataset loaded: {len(dataset)} samples, {len(dataloader)} batches")
    
    # Initialize DINOv2 tokenizer with appropriate model
    if args.image_size == 224:
        model_name = 'vit_small_patch16_224.augreg_in21k'
        expected_patches = 196  # 14x14 grid
        logger.info(f"Using 224x224 model: {model_name} with {expected_patches} patches")
        tokenizer = DINOv2Tokenizer(model_name=model_name, device=args.device)
    elif args.image_size == 518:
        # Check if user wants the base model
        if 'dinov2-base' in args.model_name.lower():
            from models.dinov2_tokenizer import DINOv2BaseTokenizer
            model_name = 'facebook/dinov2-base'
            expected_patches = 1369  # 37x37 grid
            logger.info(f"Using 518x518 base model: {model_name} with {expected_patches} patches (768-dim embeddings)")
            tokenizer = DINOv2BaseTokenizer(device=args.device)
        else:
            model_name = 'vit_small_patch14_dinov2.lvd142m'
            expected_patches = 1369  # 37x37 grid
            logger.info(f"Using 518x518 model: {model_name} with {expected_patches} patches (384-dim embeddings)")
            tokenizer = DINOv2Tokenizer(model_name=model_name, device=args.device)
    else:
        raise ValueError(f"Unsupported image size: {args.image_size}. Must be 224 or 518.")
    
    # Validate tokenizer configuration
    actual_patches = tokenizer.get_num_patches()
    actual_input_size = tokenizer.get_input_size()
    if actual_patches != expected_patches or actual_input_size != args.image_size:
        raise ValueError(f"Tokenizer mismatch: expected {expected_patches} patches for {args.image_size}x{args.image_size}, "
                       f"got {actual_patches} patches for {actual_input_size}x{actual_input_size}")
    
    # Get embedding dimension from tokenizer
    embed_dim = tokenizer.get_embed_dim()
    logger.info(f"Embedding dimension: {embed_dim}")
    
    # GPU memory management setup
    is_gpu = args.device != "cpu"
    if is_gpu:
        torch.cuda.empty_cache()
        logger.info(f"Initial GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
        logger.info(f"Initial GPU Memory Reserved: {torch.cuda.memory_reserved() / 1024**3:.1f} GB")
    
    # Collect features incrementally to avoid memory issues
    all_features_list = []
    total_clips = 0
    total_frames = 0
    
    logger.info("Processing batches and collecting features...")
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        try:
            # Process batch to get features
            projected_features_list, rgb_visualizations = process_batch_for_pca(
                batch, batch_idx, tokenizer, args.device, args.image_size, model_name
            )
            
            if projected_features_list:
                # Add features to our list (they're already on CPU from process_batch_for_pca)
                all_features_list.extend(projected_features_list)
                total_clips += len(projected_features_list)
                total_frames += sum(features.shape[0] for features in projected_features_list)
                
                logger.debug(f"Batch {batch_idx}: Added {len(projected_features_list)} clips, total: {total_clips}")
            
            # Memory cleanup after each batch
            if is_gpu:
                torch.cuda.empty_cache()
            
            # Log progress every 10 batches
            if batch_idx % 10 == 0:
                logger.info(f"Processed {batch_idx + 1}/{len(dataloader)} batches")
                logger.info(f"Total clips collected: {total_clips}, Total frames: {total_frames}")
                if is_gpu:
                    logger.info(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
                    logger.info(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1024**3:.1f} GB")
        
        except Exception as e:
            logger.error(f"Error processing batch {batch_idx}: {e}")
            continue
    
    logger.info(f"Feature collection completed!")
    logger.info(f"Total clips: {total_clips}, Total frames: {total_frames}")
    logger.info(f"Features list length: {len(all_features_list)}")
    
    if not all_features_list:
        raise RuntimeError("No features collected from dataset")
    
    # Compute PCA directions incrementally
    logger.info("Computing PCA directions incrementally...")
    pca_components, pca_mean, pca_explained_variance = compute_pca_directions_incremental(
        all_features_list, 
        n_components=args.n_components, 
        device=args.device
    )
    
    # Extract positional embeddings from tokenizer
    logger.info("Extracting positional embeddings from tokenizer...")
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
        model_name=model_name,
        embed_dim=embed_dim,
        patches_per_frame=expected_patches,
        grid_size=int(np.sqrt(expected_patches)),
        patch_size=args.image_size // int(np.sqrt(expected_patches)),
        total_patches=total_frames * expected_patches,
        unique_subjects=len(set(range(total_clips))),  # Approximate
        unique_clips=total_clips,
        pca_components_count=args.n_components
    )
    
    logger.info(f"PCA directions saved to: {args.output_path}")
    logger.info("Processing completed successfully!") 