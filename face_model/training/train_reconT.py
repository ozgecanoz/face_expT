#!/usr/bin/env python3
"""
Training script for ExpressionReconstructionModel using frozen ExpressionTransformer
Uses CCA database H5 clips and frozen ExpressionTransformer from checkpoint
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import logging
import uuid
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import json
import numpy as np

# Add the project root to the path
import sys
sys.path.append('.')

from data.dataset import FaceDataset, create_face_dataloader
from models.expression_transformer import ExpressionTransformer
from models.expression_reconstruction_model import ExpressionReconstructionModel
from models.dinov2_tokenizer import DINOv2BaseTokenizer
from models.arcface_tokenizer import ArcFaceTokenizer
from utils.checkpoint_utils import save_checkpoint, create_comprehensive_config, load_checkpoint_config, extract_model_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss for face image generation with identity preservation
    """
    
    def __init__(self, lambda_mse: float = 1.0, lambda_l1: float = 0.1, 
                 lambda_identity: float = 0.1, arcface_tokenizer=None):
        super().__init__()
        self.lambda_mse = lambda_mse
        self.lambda_l1 = lambda_l1
        self.lambda_identity = lambda_identity
        self.arcface_tokenizer = arcface_tokenizer
        
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        # Identity loss will be computed using ArcFace embeddings
        if self.arcface_tokenizer is None:
            logger.warning("⚠️  ArcFace tokenizer not provided, identity loss will be disabled")
    
    def forward(self, reconstructed: torch.Tensor, target: torch.Tensor, 
                subject_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            reconstructed: (B, 3, 518, 518) - Reconstructed face images
            target: (B, 3, 518, 518) - Target face images
            subject_ids: (B,) - Subject IDs for identity loss (optional)
            
        Returns:
            Combined reconstruction loss, mse_loss, l1_loss, identity_loss
        """
        mse_loss = self.mse_loss(reconstructed, target)
        l1_loss = self.l1_loss(reconstructed, target)
        
        # Initialize identity loss
        identity_loss = None
        
        # Compute identity loss if ArcFace tokenizer is available and subject IDs provided
        if self.arcface_tokenizer is not None and subject_ids is not None:
            try:
                identity_loss = self._compute_identity_loss(reconstructed, target, subject_ids)
            except Exception as e:
                logger.warning(f"⚠️  Failed to compute identity loss: {e}")
                identity_loss = None
        
        # Combine losses
        total_loss = self.lambda_mse * mse_loss + self.lambda_l1 * l1_loss
        
        if identity_loss is not None:
            total_loss += self.lambda_identity * identity_loss
        
        return total_loss, mse_loss, l1_loss, identity_loss
    
    def _compute_identity_loss(self, reconstructed: torch.Tensor, target: torch.Tensor, 
                              subject_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute identity loss using ArcFace embeddings
        
        Args:
            reconstructed: (B, 3, 518, 518) - Reconstructed face images
            target: (B, 3, 518, 518) - Target face images
            subject_ids: (B,) - Subject IDs
            
        Returns:
            Identity loss value
        """
        batch_size = reconstructed.shape[0]
        device = reconstructed.device
        
        # Convert tensors to numpy for ArcFace processing
        reconstructed_np = reconstructed.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        # Compute ArcFace embeddings for reconstructed and target images
        reconstructed_embeddings = []
        target_embeddings = []
        
        for i in range(batch_size):
            # Convert from [0,1] to [0,255] range for ArcFace
            recon_img = (reconstructed_np[i].transpose(1, 2, 0) * 255).astype(np.uint8)
            target_img = (target_np[i].transpose(1, 2, 0) * 255).astype(np.uint8)
            
            # Get embeddings
            recon_emb = self.arcface_tokenizer.forward(recon_img)
            target_emb = self.arcface_tokenizer.forward(target_img)
            
            if recon_emb is not None and target_emb is not None:
                reconstructed_embeddings.append(recon_emb)
                target_embeddings.append(target_emb)
            else:
                # If embedding extraction fails, use dummy embedding
                dummy_emb = np.zeros(512, dtype=np.float32)
                reconstructed_embeddings.append(dummy_emb)
                target_embeddings.append(dummy_emb)
        
        # Convert to tensors
        reconstructed_embeddings = torch.tensor(np.array(reconstructed_embeddings), device=device, dtype=torch.float32)
        target_embeddings = torch.tensor(np.array(target_embeddings), device=device, dtype=torch.float32)
        
        # Normalize embeddings
        reconstructed_embeddings = F.normalize(reconstructed_embeddings, p=2, dim=1)
        target_embeddings = F.normalize(target_embeddings, p=2, dim=1)
        
        # Compute cosine similarity loss (we want embeddings to be similar)
        # Convert to distance: |1 - cosine_similarity| to treat all deviations equally
        cosine_sim = F.cosine_similarity(reconstructed_embeddings, target_embeddings, dim=1)
        identity_loss = torch.mean(torch.abs(1 - cosine_sim))
        
        return identity_loss


def load_pca_projection(pca_json_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load PCA projection matrix from JSON file
    
    Args:
        pca_json_path: Path to PCA directions JSON file
        
    Returns:
        pca_components: (n_components, embed_dim) PCA components
        pca_mean: (embed_dim,) PCA mean vector
    """
    logger.info(f"Loading PCA projection from: {pca_json_path}")
    
    with open(pca_json_path, 'r') as f:
        pca_data = json.load(f)
    
    pca_components = np.array(pca_data['pca_components'])
    pca_mean = np.array(pca_data['pca_mean'])
    
    logger.info(f"PCA components shape: {pca_components.shape}")
    logger.info(f"PCA mean shape: {pca_mean.shape}")
    
    return pca_components, pca_mean


def prepare_reconstruction_data(batch, dinov2_tokenizer, pca_components, pca_mean, 
                              expression_transformer, device):
    """
    Prepare data for reconstruction training
    
    Args:
        batch: Batch from FaceDataset
        dinov2_tokenizer: DINOv2BaseTokenizer instance
        pca_components: PCA projection matrix
        pca_mean: PCA mean vector
        expression_transformer: Frozen ExpressionTransformer
        device: Device to move data to
        
    Returns:
        expression_tokens: (batch_size, T, 1, embed_dim) - Expression tokens per clip
        subject_ids: (batch_size,) - Subject IDs per clip
        original_frames: (batch_size, T, 3, 518, 518) - Original frames per clip
        clip_lengths: List of clip lengths
    """
    # Move PCA components to device once
    pca_components_tensor = torch.from_numpy(pca_components).float().to(device)
    pca_mean_tensor = torch.from_numpy(pca_mean).float().to(device)
    
    # Extract data from batch
    frames = batch['frames']  # List of (T, 3, 518, 518) tensors
    subject_ids = batch['subject_id']  # List of subject ID strings
    
    batch_size = len(frames)
    
    # Process each clip separately
    all_expression_tokens = []
    all_original_frames = []
    all_final_pos_embeddings = []
    clip_lengths = []
    
    for clip_idx, (clip_frames, subject_id) in enumerate(zip(frames, subject_ids)):
        # Move frames to device before processing
        clip_frames = clip_frames.to(device)
        
        # Get DINOv2 patch tokens
        patch_tokens, _ = dinov2_tokenizer(clip_frames)  # (T, 1369, 768)
        
        # Project to PCA space (keep on GPU)
        num_frames, num_patches, embed_dim = patch_tokens.shape
        
        # Reshape for PCA projection
        patch_tokens_reshaped = patch_tokens.reshape(-1, embed_dim)  # (T*1369, 768)
        
        # Center the data
        patch_tokens_centered = patch_tokens_reshaped - pca_mean_tensor
        
        # Project to PCA space
        projected_features = torch.mm(patch_tokens_centered, pca_components_tensor.T)  # (T*1369, 384)
        
        # Reshape back to (T, 1369, 384)
        projected_features = projected_features.reshape(num_frames, num_patches, -1)
        
        # Get expression tokens from frozen transformer using inference method
        with torch.no_grad():
            expression_tokens, _ = expression_transformer.inference(projected_features)  # (T, 1, 384), (T, 1369, 384) - ignore pos embeddings
        
        # Store data (keep clips separate)
        all_expression_tokens.append(expression_tokens)
        all_original_frames.append(clip_frames)
        clip_lengths.append(clip_frames.shape[0])
    
    # Stack clips (don't concatenate)
    expression_tokens = torch.stack(all_expression_tokens, dim=0)  # (batch_size, T, 1, 384)
    original_frames = torch.stack(all_original_frames, dim=0)  # (batch_size, T, 3, 518, 518)
    
    return expression_tokens, subject_ids, original_frames, clip_lengths


def train_expression_reconstruction(
    dataset_path: str,
    pca_json_path: str,
    expression_transformer_checkpoint_path: str,
    checkpoint_dir: str = "checkpoints",
    save_every_step: int = 100,
    batch_size: int = 8,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    max_samples: int = None,
    val_dataset_path: str = None,
    max_val_samples: int = None,
    device: str = "cpu",
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    drop_last: bool = True,
    # Learning rate scheduler parameters
    warmup_steps: int = 1000,
    min_lr: float = 1e-6,
    # Architecture configuration parameters
    embed_dim: int = 384,
    num_cross_attention_layers: int = 2,
    num_self_attention_layers: int = 2,
    num_heads: int = 8,
    dropout: float = 0.1,
    ff_dim: int = 1536,
    max_subjects: int = 3500,
    log_dir: str = None,
    # Memory management
    max_memory_fraction: float = 0.9,
    lambda_identity: float = 0.1,
    arcface_model_path: str = "/path/to/arcface.onnx"
):
    """
    Train the expression reconstruction model using frozen expression transformer
    
    Args:
        dataset_path: Path to the CCA dataset
        pca_json_path: Path to PCA projection JSON file
        expression_transformer_checkpoint_path: Path to expression transformer checkpoint
        checkpoint_dir: Directory to save checkpoints
        save_every_step: Save checkpoints every N steps
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        max_samples: Maximum number of samples to use
        val_dataset_path: Path to validation dataset
        max_val_samples: Maximum number of validation samples
        device: Device to use
        num_workers: Number of data loader workers
        pin_memory: Whether to pin memory
        persistent_workers: Whether to use persistent workers
        drop_last: Whether to drop last incomplete batch
        warmup_steps: Number of warmup steps for scheduler
        min_lr: Minimum learning rate
        embed_dim: Embedding dimension
        num_cross_attention_layers: Number of cross-attention layers
        num_self_attention_layers: Number of self-attention layers
        num_heads: Number of attention heads
        dropout: Dropout rate
        ff_dim: Feed-forward dimension
        max_subjects: Maximum number of subjects
        log_dir: Directory for TensorBoard logs
        max_memory_fraction: Maximum GPU memory fraction to use
        lambda_identity: Weight for identity loss
        arcface_model_path: Path to ArcFace ONNX model file
    """
    logger.info("Starting expression reconstruction training")
    logger.info(f"Device: {device}")
    
    # Set GPU memory fraction if using CUDA
    if device.type == "cuda" and max_memory_fraction < 1.0:
        torch.cuda.set_per_process_memory_fraction(max_memory_fraction)
        logger.info(f"Set GPU memory fraction to {max_memory_fraction}")
    
    # Load PCA projection
    pca_components, pca_mean = load_pca_projection(pca_json_path)
    
    # Initialize DINOv2 tokenizer
    logger.info("Initializing DINOv2 base tokenizer...")
    dinov2_tokenizer = DINOv2BaseTokenizer(device=device)
    logger.info("✅ DINOv2 base tokenizer initialized")
    
    # Initialize ArcFace tokenizer for identity loss
    logger.info("Initializing ArcFace tokenizer...")
    # ArcFaceTokenizer doesn't need device parameter - ONNX runtime handles it
    arcface_tokenizer = ArcFaceTokenizer(model_path=arcface_model_path)
    logger.info("✅ ArcFace tokenizer initialized")
    
    # Load frozen ExpressionTransformer from checkpoint
    logger.info(f"Loading frozen ExpressionTransformer from: {expression_transformer_checkpoint_path}")
    checkpoint = torch.load(expression_transformer_checkpoint_path, map_location=device)
    
    # Debug: Log checkpoint keys
    logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    # Extract config from checkpoint
    if 'config' in checkpoint:
        checkpoint_config = checkpoint['config']
        logger.info("✅ Found config in checkpoint, initializing ExpressionTransformer accordingly")
        
        # Initialize ExpressionTransformer with checkpoint config
        expression_transformer = ExpressionTransformer(
            embed_dim=checkpoint_config.get('expression_model', {}).get('expr_embed_dim', embed_dim),
            num_heads=checkpoint_config.get('expression_model', {}).get('expr_num_heads', num_heads),
            num_layers=checkpoint_config.get('expression_model', {}).get('expr_num_layers', 2),
            dropout=checkpoint_config.get('expression_model', {}).get('expr_dropout', dropout),
            ff_dim=checkpoint_config.get('expression_model', {}).get('expr_ff_dim', ff_dim),
            grid_size=checkpoint_config.get('expression_model', {}).get('expr_grid_size', 37)
        ).to(device)
        
        # Load model weights - handle both standalone and supervised model checkpoints
        if 'expression_transformer_state_dict' in checkpoint:
            # Check if the keys have the 'expression_transformer.' prefix
            expr_state_dict = checkpoint['expression_transformer_state_dict']
            if any(key.startswith('expression_transformer.') for key in expr_state_dict.keys()):
                # Keys have prefix - remove it
                clean_state_dict = {}
                for key, value in expr_state_dict.items():
                    if key.startswith('expression_transformer.'):
                        new_key = key[len('expression_transformer.'):]
                        clean_state_dict[new_key] = value
                    else:
                        clean_state_dict[key] = value
                
                expression_transformer.load_state_dict(clean_state_dict)
                logger.info("✅ ExpressionTransformer checkpoint loaded with matching config (removed prefix)")
            else:
                # Keys are clean - load directly
                expression_transformer.load_state_dict(expr_state_dict)
                logger.info("✅ ExpressionTransformer checkpoint loaded with matching config")
        elif 'model_state_dict' in checkpoint:
            # This is a supervised model checkpoint - extract ExpressionTransformer part
            model_state_dict = checkpoint['model_state_dict']
            # Filter keys to only include ExpressionTransformer parameters
            expr_state_dict = {}
            for key, value in model_state_dict.items():
                if key.startswith('expression_transformer.'):
                    # Remove the 'expression_transformer.' prefix
                    new_key = key[len('expression_transformer.'):]
                    expr_state_dict[new_key] = value
            
            if expr_state_dict:
                expression_transformer.load_state_dict(expr_state_dict)
                logger.info("✅ ExpressionTransformer extracted from supervised model checkpoint")
            else:
                raise ValueError("No ExpressionTransformer parameters found in supervised model checkpoint")
        else:
            raise ValueError("Checkpoint must contain either 'expression_transformer_state_dict' or 'model_state_dict'")
        
        # Update local variables to match checkpoint config
        embed_dim = checkpoint_config.get('expression_model', {}).get('expr_embed_dim', embed_dim)
        num_heads = checkpoint_config.get('expression_model', {}).get('expr_num_heads', num_heads)
        ff_dim = checkpoint_config.get('expression_model', {}).get('expr_ff_dim', ff_dim)
        
        logger.info(f"ExpressionTransformer config from checkpoint:")
        logger.info(f"  - embed_dim: {embed_dim}")
        logger.info(f"  - num_heads: {num_heads}")
        logger.info(f"  - ff_dim: {ff_dim}")
        
    else:
        logger.warning("⚠️  No config found in checkpoint, using provided arguments")
        # Initialize ExpressionTransformer with provided arguments
        expression_transformer = ExpressionTransformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=2,
            dropout=dropout,
            ff_dim=ff_dim,
            grid_size=37
        ).to(device)
        
        # Load model weights - handle both standalone and supervised model checkpoints
        if 'expression_transformer_state_dict' in checkpoint:
            # Check if the keys have the 'expression_transformer.' prefix
            expr_state_dict = checkpoint['expression_transformer_state_dict']
            if any(key.startswith('expression_transformer.') for key in expr_state_dict.keys()):
                # Keys have prefix - remove it
                clean_state_dict = {}
                for key, value in expr_state_dict.items():
                    if key.startswith('expression_transformer.'):
                        new_key = key[len('expression_transformer.'):]
                        clean_state_dict[new_key] = value
                    else:
                        clean_state_dict[key] = value
                
                expression_transformer.load_state_dict(clean_state_dict)
                logger.info("✅ ExpressionTransformer checkpoint loaded (removed prefix)")
            else:
                # Keys are clean - load directly
                expression_transformer.load_state_dict(expr_state_dict)
                logger.info("✅ ExpressionTransformer checkpoint loaded")
        elif 'model_state_dict' in checkpoint:
            # This is a supervised model checkpoint - extract ExpressionTransformer part
            model_state_dict = checkpoint['model_state_dict']
            # Filter keys to only include ExpressionTransformer parameters
            expr_state_dict = {}
            for key, value in model_state_dict.items():
                if key.startswith('expression_transformer.'):
                    # Remove the 'expression_transformer.' prefix
                    new_key = key[len('expression_transformer.'):]
                    expr_state_dict[new_key] = value
            
            if expr_state_dict:
                expression_transformer.load_state_dict(expr_state_dict)
                logger.info("✅ ExpressionTransformer extracted from supervised model checkpoint")
            else:
                raise ValueError("No ExpressionTransformer parameters found in supervised model checkpoint")
        else:
            raise ValueError("Checkpoint must contain either 'expression_transformer_state_dict' or 'model_state_dict'")
    
    # Freeze the ExpressionTransformer
    expression_transformer.requires_grad = False
    logger.info("🔒 ExpressionTransformer frozen (no gradients)")
    
    # Initialize ExpressionReconstructionModel
    logger.info("Initializing ExpressionReconstructionModel...")
    reconstruction_model = ExpressionReconstructionModel(
        embed_dim=embed_dim,
        num_cross_attention_layers=num_cross_attention_layers,
        num_self_attention_layers=num_self_attention_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout=dropout,
        max_subjects=max_subjects
    ).to(device)
    
    # Log model configuration
    model_config = reconstruction_model.get_config()
    logger.info(f"ExpressionReconstructionModel config:")
    for key, value in model_config.items():
        logger.info(f"  - {key}: {value}")
    
    # Load datasets
    train_dataset = FaceDataset(dataset_path, max_samples=max_samples)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=drop_last
    )
    
    val_dataloader = None
    if val_dataset_path:
        val_dataset = FaceDataset(val_dataset_path, max_samples=max_val_samples)
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            drop_last=False
        )
        logger.info(f"Validation dataset loaded: {len(val_dataset)} samples")
    
    logger.info(f"Training dataset loaded: {len(train_dataset)} samples")
    logger.info(f"Training batches: {len(train_dataloader)}")
    
    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(reconstruction_model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Cosine annealing scheduler with warmup
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=warmup_steps, 
        T_mult=2, 
        eta_min=min_lr
    )
    
    # Initialize loss function
    criterion = ReconstructionLoss(lambda_mse=1.0, lambda_l1=0.1, lambda_identity=lambda_identity, arcface_tokenizer=arcface_tokenizer)
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Setup TensorBoard logging
    if log_dir is None:
        log_dir = f"logs/reconT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # Training loop
    current_training_step = 0
    
    for epoch in range(num_epochs):
        reconstruction_model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Prepare data
                expression_tokens, subject_ids, original_frames, clip_lengths = prepare_reconstruction_data(
                    batch, dinov2_tokenizer, pca_components, pca_mean, expression_transformer, device
                )
                
                # Create subject ID to index mapping
                unique_subject_ids = list(set(subject_ids))
                subject_id_to_idx = {subject_id: idx for idx, subject_id in enumerate(unique_subject_ids)}
                
                # Convert string subject IDs to tensor indices
                subject_id_tensor = torch.tensor([subject_id_to_idx[sid] for sid in subject_ids], dtype=torch.long, device=device)
                
                # Process each clip separately
                batch_loss = 0.0
                batch_mse_loss = 0.0
                batch_l1_loss = 0.0
                batch_identity_loss = 0.0
                
                for clip_idx in range(len(subject_ids)):
                    # Get all frames for this clip
                    clip_expression_tokens = expression_tokens[clip_idx]  # (T, 1, 384) - all frames
                    clip_original_frames = original_frames[clip_idx]  # (T, 3, 518, 518) - all frames
                    num_frames = clip_expression_tokens.shape[0]
                    
                    # Create subject IDs for all frames (same subject for all frames in clip)
                    clip_subject_ids = torch.full((num_frames,), subject_id_tensor[clip_idx], dtype=torch.long, device=device)  # (T,)
                    
                    # Forward pass through reconstruction model for all frames of this clip
                    reconstructed_frames = reconstruction_model(
                        clip_subject_ids, clip_expression_tokens
                    )  # (T, 3, 518, 518)
                    
                    # Compute loss for all frames of this clip (including identity loss)
                    clip_loss, clip_mse_loss, clip_l1_loss, clip_identity_loss = criterion(
                        reconstructed_frames, clip_original_frames, clip_subject_ids
                    )
                    
                    # Accumulate losses
                    batch_loss += clip_loss
                    batch_mse_loss += clip_mse_loss
                    batch_l1_loss += clip_l1_loss
                    if clip_identity_loss is not None:
                        batch_identity_loss += clip_identity_loss
                
                # Average losses across clips
                total_loss = batch_loss / len(subject_ids)
                mse_loss = batch_mse_loss / len(subject_ids)
                l1_loss = batch_l1_loss / len(subject_ids)
                identity_loss = batch_identity_loss / len(subject_ids) if batch_identity_loss > 0 else None
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                # Update learning rate
                scheduler.step()
                
                # Update metrics
                epoch_loss += total_loss.item()
                num_batches += 1
                current_training_step += 1
                
                # Log to TensorBoard
                writer.add_scalar('Training/Step_Total_Loss', total_loss.item(), current_training_step)
                writer.add_scalar('Training/Step_MSE_Loss', mse_loss.item(), current_training_step)
                writer.add_scalar('Training/Step_L1_Loss', l1_loss.item(), current_training_step)
                if identity_loss is not None:
                    writer.add_scalar('Training/Step_Identity_Loss', identity_loss.item(), current_training_step)
                writer.add_scalar('Training/Learning_Rate', scheduler.get_last_lr()[0], current_training_step)
                
                # Save checkpoint every N steps
                if current_training_step % save_every_step == 0:
                    checkpoint_path = os.path.join(checkpoint_dir, f"reconT_step_{current_training_step}.pt")
                    
                    # Create config using reconstruction model's get_config method
                    model_config = reconstruction_model.get_config()
                    config = create_comprehensive_config(
                        # Reconstruction model parameters
                        recon_embed_dim=model_config['embed_dim'],
                        recon_num_cross_layers=model_config['num_cross_attention_layers'],
                        recon_num_self_layers=model_config['num_self_attention_layers'],
                        recon_num_heads=model_config['num_heads'],
                        recon_ff_dim=model_config['ff_dim'],
                        recon_dropout=model_config['dropout'],
                        recon_max_subjects=model_config['max_subjects'],
                        # Training parameters
                        learning_rate=learning_rate,
                        batch_size=batch_size,
                        num_epochs=num_epochs,
                        warmup_steps=warmup_steps,
                        min_lr=min_lr,
                        pca_json_path=pca_json_path
                    )
                    
                    try:
                        save_checkpoint(
                            model_state_dict=reconstruction_model.state_dict(),
                            optimizer_state_dict=optimizer.state_dict(),
                            scheduler_state_dict=scheduler.state_dict(),
                            epoch=epoch + 1,
                            avg_loss=total_loss.item(),
                            total_steps=current_training_step,
                            config=config,
                            checkpoint_path=checkpoint_path,
                            checkpoint_type="expression_reconstruction"
                        )
                        
                        logger.info(f"✅ Saved checkpoint: {checkpoint_path}")
                        print(f"💾 Checkpoint saved: {checkpoint_path}")
                        
                    except Exception as e:
                        logger.error(f"❌ Failed to save checkpoint: {e}")
                        print(f"❌ Checkpoint save failed: {e}")
                
                # Also save a checkpoint every epoch for safety
                if batch_idx == len(train_dataloader) - 1:  # Last batch of epoch
                    logger.info(f"🔄 Saving epoch checkpoint at step {current_training_step}")
                    epoch_checkpoint_path = os.path.join(checkpoint_dir, f"reconT_epoch_{epoch+1}_step_{current_training_step}.pt")
                    
                    # Create config for epoch checkpoint if not already created
                    if 'config' not in locals():
                        model_config = reconstruction_model.get_config()
                        config = create_comprehensive_config(
                            # Reconstruction model parameters
                            recon_embed_dim=model_config['embed_dim'],
                            recon_num_cross_layers=model_config['num_cross_attention_layers'],
                            recon_num_self_layers=model_config['num_self_attention_layers'],
                            recon_num_heads=model_config['num_heads'],
                            recon_ff_dim=model_config['ff_dim'],
                            recon_dropout=model_config['dropout'],
                            recon_max_subjects=model_config['max_subjects'],
                            # Training parameters
                            learning_rate=learning_rate,
                            batch_size=batch_size,
                            num_epochs=num_epochs,
                            warmup_steps=warmup_steps,
                            min_lr=min_lr,
                            pca_json_path=pca_json_path
                        )
                    
                    try:
                        save_checkpoint(
                            model_state_dict=reconstruction_model.state_dict(),
                            optimizer_state_dict=optimizer.state_dict(),
                            scheduler_state_dict=scheduler.state_dict(),
                            epoch=epoch + 1,
                            avg_loss=epoch_loss / num_batches,
                            total_steps=current_training_step,
                            config=config,
                            checkpoint_path=epoch_checkpoint_path,
                            checkpoint_type="expression_reconstruction"
                        )
                        
                        logger.info(f"✅ Saved epoch checkpoint: {epoch_checkpoint_path}")
                        print(f"💾 Epoch checkpoint saved: {epoch_checkpoint_path}")
                        
                    except Exception as e:
                        logger.error(f"❌ Failed to save epoch checkpoint: {e}")
                        print(f"❌ Epoch checkpoint save failed: {e}")
                
                # Update progress bar
                postfix_dict = {
                    'Total Loss': f'{total_loss.item():.5f}',
                    'MSE Loss': f'{mse_loss.item():.5f}',
                    'L1 Loss': f'{l1_loss.item():.5f}',
                    'Avg Loss': f'{epoch_loss / num_batches:.5f}',
                    'LR': f'{scheduler.get_last_lr()[0]:.2e}',
                    'Step': f'{current_training_step}',
                    'Next Checkpoint': f'Step {((current_training_step // save_every_step) + 1) * save_every_step}'
                }
                
                if identity_loss is not None:
                    postfix_dict['Identity Loss'] = f'{identity_loss.item():.5f}'
                
                progress_bar.set_postfix(postfix_dict)
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                continue
        
        avg_loss = epoch_loss / num_batches
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_loss:.4f}")
        
        # CUDA memory cleanup after each epoch
        if device.type == "cuda":
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            logger.info(f"🧹 Memory cleanup after epoch {epoch+1}")
        
        # Log epoch metrics to TensorBoard
        writer.add_scalar('Training/Epoch_Loss', avg_loss, epoch + 1)
        
        # Validate if validation dataloader is provided
        if val_dataloader is not None:
            val_loss = validate_expression_reconstruction(
                reconstruction_model, val_dataloader, criterion, dinov2_tokenizer, 
                pca_components, pca_mean, expression_transformer, device
            )
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Log validation metrics to TensorBoard
            writer.add_scalar('Validation/Epoch_Loss', val_loss, epoch + 1)
            
            # Memory cleanup after validation
            if device.type == "cuda":
                torch.cuda.empty_cache()
                import gc
                gc.collect()
    
    # Close TensorBoard writer
    writer.close()
    
    logger.info("Training completed!")
    logger.info(f"📊 TensorBoard logs saved to: {log_dir}")
    return reconstruction_model


def validate_expression_reconstruction(
    model,
    val_dataloader,
    criterion,
    dinov2_tokenizer,
    pca_components,
    pca_mean,
    expression_transformer,
    device="cpu"
):
    """
    Validate the expression reconstruction model
    """
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            try:
                # Prepare data
                expression_tokens, subject_ids, original_frames, clip_lengths = prepare_reconstruction_data(
                    batch, dinov2_tokenizer, pca_components, pca_mean, expression_transformer, device
                )
                
                # Create subject ID to index mapping for validation
                unique_subject_ids = list(set(subject_ids))
                subject_id_to_idx = {subject_id: idx for idx, subject_id in enumerate(unique_subject_ids)}
                
                # Convert string subject IDs to tensor indices
                subject_id_tensor = torch.tensor([subject_id_to_idx[sid] for sid in subject_ids], dtype=torch.long, device=device)
                
                # Process each clip separately
                batch_loss = 0.0
                
                for clip_idx in range(len(subject_ids)):
                    # Get all frames for this clip
                    clip_expression_tokens = expression_tokens[clip_idx]  # (1, 1, 384) - all frames
                    clip_original_frames = original_frames[clip_idx]  # (1, 3, 518, 518) - all frames
                    num_frames = clip_expression_tokens.shape[0]
                    
                    # Create subject IDs for all frames (same subject for all frames in clip)
                    clip_subject_ids = torch.full((num_frames,), subject_id_tensor[clip_idx], dtype=torch.long, device=device)  # (T,)
                    
                    # Forward pass through reconstruction model for all frames of this clip
                    reconstructed_frames = model(
                        clip_subject_ids, clip_expression_tokens
                    )  # (T, 3, 518, 518)
                    
                    # Compute loss for all frames of this clip (including identity loss)
                    clip_loss, _, _, _ = criterion(reconstructed_frames, clip_original_frames, clip_subject_ids)
                    
                    # Accumulate losses
                    batch_loss += clip_loss
                
                # Average losses across clips
                total_loss_val = batch_loss / len(subject_ids)
                
                # Update metrics
                total_loss += total_loss_val.item()
                num_batches += 1
                
            except Exception as e:
                logger.error(f"Error processing validation batch {batch_idx}: {e}")
                continue
    
    avg_loss = total_loss / num_batches
    logger.info(f"Validation - Avg Loss: {avg_loss:.4f}")
    
    return avg_loss


def test_expression_reconstruction():
    """Test the expression reconstruction model"""
    print("🧪 Testing Expression Reconstruction Model...")
    
    # Create model
    model = ExpressionReconstructionModel(
        embed_dim=384,
        num_cross_attention_layers=2,
        num_self_attention_layers=2,
        num_heads=8,
        ff_dim=1536,
        dropout=0.1,
        max_subjects=3500
    )
    print("✅ Model created successfully")
    
    # Test with dummy input
    batch_size = 2
    embed_dim = 384
    num_patches = 1369
    
    # Create dummy data
    subject_ids = torch.randint(0, 100, (batch_size,))
    expression_tokens = torch.randn(batch_size, 1, embed_dim)
    pos_embeddings = torch.randn(batch_size, num_patches, embed_dim)
    
    # Forward pass
    reconstructed = model(subject_ids, expression_tokens)
    
    print(f"Reconstructed shape: {reconstructed.shape}")
    print("✅ Expression reconstruction test passed!")


if __name__ == "__main__":
    test_expression_reconstruction()
