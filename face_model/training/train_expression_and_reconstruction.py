#!/usr/bin/env python3
"""
Joint Training of Expression Transformer and Expression Reconstruction Model
Combines expression extraction with face reconstruction using the same scheduler scheme
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

# Add the project root to the path
import sys
sys.path.append('.')

from data.dataset import FaceDataset
from models.expression_transformer import ExpressionTransformer
from models.expression_reconstruction_model import ExpressionReconstructionModel
from models.dinov2_tokenizer import DINOv2Tokenizer
from utils.scheduler_utils import CombinedLRLossWeightScheduler
from utils.checkpoint_utils import save_checkpoint, create_comprehensive_config, load_checkpoint_config, extract_model_config
from utils.visualization_utils import compute_cosine_similarity_distribution, plot_cosine_similarity_distribution
# Removed tensorboard histogram logging to avoid compatibility issues

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JointExpressionReconstructionModel(nn.Module):
    """
    Joint model combining Expression Transformer and Expression Reconstruction Model
    """
    
    def __init__(self, 
                 expr_embed_dim=384, expr_num_heads=4, expr_num_layers=2, expr_dropout=0.1, expr_max_subjects=3500, expr_ff_dim=1536,
                 recon_embed_dim=384, recon_num_cross_layers=2, recon_num_self_layers=2, recon_num_heads=8, recon_ff_dim=1536, recon_dropout=0.1):
        super().__init__()
        
        # Expression Transformer
        self.expression_transformer = ExpressionTransformer(
            embed_dim=expr_embed_dim,
            num_heads=expr_num_heads,
            num_layers=expr_num_layers,
            dropout=expr_dropout,
            max_subjects=expr_max_subjects,
            ff_dim=expr_ff_dim
        )
        
        # Expression Reconstruction Model
        self.expression_reconstruction = ExpressionReconstructionModel(
            embed_dim=recon_embed_dim,
            num_cross_attention_layers=recon_num_cross_layers,
            num_self_attention_layers=recon_num_self_layers,
            num_heads=recon_num_heads,
            ff_dim=recon_ff_dim,
            dropout=recon_dropout
        )
        
    def forward(self, face_images, subject_ids, tokenizer, clip_lengths=None):
        """
        Args:
            face_images: (total_frames, 3, 518, 518) - All face images
            subject_ids: (total_frames,) - Subject IDs for each frame
            tokenizer: DINOv2Tokenizer instance
            clip_lengths: List of clip lengths (optional)
        
        Returns:
            all_expression_tokens: List[Tensor (T, 1, D)] - Expression tokens per clip
            all_reconstructed_images: List[Tensor (T, 3, 518, 518)] - Reconstructed images per clip
        """
        if clip_lengths is None:
            # Single clip
            return self._forward_single_clip(face_images, subject_ids, tokenizer)
        
        # Multiple clips
        all_expression_tokens = []
        all_reconstructed_images = []
        
        start_idx = 0
        for clip_length in clip_lengths:
            end_idx = start_idx + clip_length
            
            # Extract clip data
            clip_images = face_images[start_idx:end_idx]  # (T, 3, 518, 518)
            clip_subject_ids = subject_ids[start_idx:end_idx]  # (T,)
            
            # Forward pass for this clip
            clip_expression_tokens, clip_reconstructed_images = self._forward_single_clip(
                clip_images, clip_subject_ids, tokenizer
            )
            
            all_expression_tokens.append(clip_expression_tokens)
            all_reconstructed_images.append(clip_reconstructed_images)
            
            start_idx = end_idx
        
        return all_expression_tokens, all_reconstructed_images
    
    def _forward_single_clip(self, face_images, subject_ids, tokenizer):
        """
        Forward pass for a single clip
        """
        # Get DINOv2 tokens and positional embeddings
        patch_tokens, pos_embeddings = tokenizer(face_images)  # (T, 1369, 384), (T, 1369, 384)
        
        # Extract expression tokens using Expression Transformer
        expression_tokens = []
        adjusted_pos_embeddings = []
        for t in range(patch_tokens.shape[0]):
            # Single frame processing
            frame_patch_tokens = patch_tokens[t:t+1]  # (1, 1369, 384)
            frame_pos_embeddings = pos_embeddings[t:t+1]  # (1, 1369, 384)
            frame_subject_id = subject_ids[t:t+1]  # (1,)
            
            # Get expression token
            expression_token = self.expression_transformer(
                frame_patch_tokens, frame_pos_embeddings, frame_subject_id
            )  # (1, 1, 384)
            
            expression_tokens.append(expression_token)
            
            # Create adjusted positional embeddings with delta embeddings
            # This ensures reconstruction loss affects the delta positional embeddings
            adjusted_pos = frame_pos_embeddings + self.expression_transformer.delta_pos_embed
            adjusted_pos_embeddings.append(adjusted_pos)
        
        # Stack expression tokens and adjusted positional embeddings
        expression_tokens = torch.cat(expression_tokens, dim=0)  # (T, 1, 384)
        adjusted_pos_embeddings = torch.cat(adjusted_pos_embeddings, dim=0)  # (T, 1369, 384)
        
        # Reconstruct images using Expression Reconstruction Model
        reconstructed_images = []
        for t in range(expression_tokens.shape[0]):
            # Get subject embedding for this frame
            subject_embedding = self.expression_transformer.subject_embeddings(subject_ids[t:t+1])  # (1, 384)
            subject_embedding = subject_embedding.unsqueeze(1)  # (1, 1, 384)
            
            # Get expression token for this frame
            expression_token = expression_tokens[t:t+1]  # (1, 1, 384)
            
            # Get adjusted positional embeddings for this frame (includes delta embeddings)
            frame_adjusted_pos_embeddings = adjusted_pos_embeddings[t:t+1]  # (1, 1369, 384)
            
            # Reconstruct image using adjusted positional embeddings
            reconstructed_image = self.expression_reconstruction(
                subject_embedding, expression_token, frame_adjusted_pos_embeddings
            )  # (1, 3, 518, 518)
            
            reconstructed_images.append(reconstructed_image)
        
        # Stack reconstructed images
        reconstructed_images = torch.cat(reconstructed_images, dim=0)  # (T, 3, 518, 518)
        
        return expression_tokens, reconstructed_images


class ExpressionReconstructionLoss(nn.Module):
    """
    Combined loss for expression reconstruction:
    1. Reconstruction loss between original and reconstructed images
    2. Temporal contrastive loss between consecutive expression tokens in a clip
    3. Diversity loss: penalize high similarity among expression tokens within a clip
    """
    
    def __init__(self, lambda_reconstruction=1.0, lambda_temporal=0.1, lambda_diversity=0.1):
        super().__init__()
        self.lambda_reconstruction = lambda_reconstruction
        self.lambda_temporal = lambda_temporal
        self.lambda_diversity = lambda_diversity
    
    def update_weights(self, new_weights: Dict[str, float]):
        """
        Update loss weights dynamically during training.
        
        Args:
            new_weights: Dictionary with keys 'lambda_reconstruction', 'lambda_temporal', 'lambda_diversity'
        """
        if 'lambda_reconstruction' in new_weights:
            self.lambda_reconstruction = new_weights['lambda_reconstruction']
        if 'lambda_temporal' in new_weights:
            self.lambda_temporal = new_weights['lambda_temporal']
        if 'lambda_diversity' in new_weights:
            self.lambda_diversity = new_weights['lambda_diversity']

    def forward(self, original_images, reconstructed_images, expression_tokens_by_clip):
        """
        Args:
            original_images: List[Tensor (T, 3, 518, 518)] - Original images per clip
            reconstructed_images: List[Tensor (T, 3, 518, 518)] - Reconstructed images per clip
            expression_tokens_by_clip: List[Tensor (T, 1, D)] - Expression tokens per clip

        Returns:
            total_loss: Scalar loss
        """
        reconstruction_losses = []
        temporal_losses = []
        diversity_losses = []

        for orig_clip, recon_clip, clip_tokens in zip(original_images, reconstructed_images, expression_tokens_by_clip):
            # ---- 1. Reconstruction Loss ----
            # MSE loss between original and reconstructed images
            reconstruction_loss = F.mse_loss(recon_clip, orig_clip)
            reconstruction_losses.append(reconstruction_loss)

            # ---- 2. Temporal Loss (dual objective: coherence + diversity) ----
            if clip_tokens.shape[0] > 1:
                lambda_coherence = 0.7
                lambda_contrast = 0.3

                tokens = clip_tokens.squeeze(1)  # (T, D) - already normalized

                # Encourage similarity: cosine_similarity ~ 1 → loss ~ 0
                adj_sim = F.cosine_similarity(tokens[:-1], tokens[1:], dim=-1)
                coherence_loss = (1 - adj_sim).mean()  # Want sim ≈ 1 → loss ≈ 0

                stride = 5
                if tokens.shape[0] > stride:
                    t1 = tokens[:-stride]
                    t2 = tokens[stride:]
                    contrast_sim = F.cosine_similarity(t1, t2, dim=-1)
                    contrast_loss = (contrast_sim.abs()).mean()  # Want sim ≈ 0 → loss ≈ 0
                else:
                    contrast_loss = 0.0
                
                temporal_loss = lambda_coherence * coherence_loss + lambda_contrast * contrast_loss
                temporal_losses.append(temporal_loss)

                # ---- 3. Diversity Loss (within-clip token dissimilarity) ----
                # Use every 5th frame for diversity computation (same as temporal contrast)
                diversity_tokens = clip_tokens[::stride].squeeze(1)  # (T//5, D)
                sim_matrix = diversity_tokens @ diversity_tokens.T  # (T//5, T//5)
                # Subtract diagonal (self-similarities = 1.0) to get only cross-token similarities
                T = diversity_tokens.shape[0]
                if T > 1:  # Need at least 2 tokens for diversity
                    #diversity_loss = (sim_matrix.sum() - T) / (T * (T - 1))
                    diversity_loss = torch.clamp((sim_matrix.sum() - T) / (T * (T - 1)), min=0.0)
                    diversity_losses.append(diversity_loss)

        # Aggregate losses
        total_reconstruction = torch.stack(reconstruction_losses).mean()
        total_temporal = torch.stack(temporal_losses).mean() if temporal_losses else 0.0
        total_diversity = torch.stack(diversity_losses).mean() if diversity_losses else 0.0

        total_loss = (
            self.lambda_reconstruction * total_reconstruction
            + self.lambda_temporal * total_temporal
            + self.lambda_diversity * total_diversity
        )

        # Return both total loss and individual components for logging
        loss_components = {
            'total': total_loss,
            'reconstruction': total_reconstruction,
            'temporal': total_temporal,
            'diversity': total_diversity
        }
        
        return total_loss, loss_components


def prepare_expression_reconstruction_data(batch, dinov2_tokenizer, device):
    """
    Prepare training data for expression reconstruction
    
    Args:
        batch: Batch from dataloader containing 'frames' and 'subject_id' for each sample
        dinov2_tokenizer: DINOv2Tokenizer instance
        device: Device to use
    
    Returns:
        face_images: (total_frames, 3, 518, 518) - All face images
        subject_ids: (total_frames,) - Subject IDs for each frame
        clip_lengths: List of clip lengths
    """
    # Extract frames from all clips in the batch
    all_frames = []
    subject_ids = []
    clip_lengths = []
    
    for i, (frames, subject_id) in enumerate(zip(batch['frames'], batch['subject_id'])):
        # frames: (num_frames, 3, 518, 518) for one clip
        num_frames = frames.shape[0]
        all_frames.append(frames)
        clip_lengths.append(num_frames)
        
        # Convert subject_id to integer if it's a string
        if isinstance(subject_id, str):
            # Try to convert string subject ID to integer
            try:
                subject_id_int = int(subject_id)
            except ValueError:
                # If conversion fails, use hash of string as subject ID
                subject_id_int = hash(subject_id) % 3500  # Ensure it's within max_subjects range
        else:
            subject_id_int = int(subject_id)
        
        # Repeat subject ID for all frames in this clip
        clip_subject_ids = torch.full((num_frames,), subject_id_int, dtype=torch.long)
        subject_ids.append(clip_subject_ids)
    
    # Concatenate all frames and subject IDs
    face_images = torch.cat(all_frames, dim=0).to(device)  # (total_frames, 3, 518, 518)
    subject_ids = torch.cat(subject_ids, dim=0).to(device)  # (total_frames,)
    
    return face_images, subject_ids, clip_lengths


def train_expression_and_reconstruction(
    dataset_path,
    expression_transformer_checkpoint_path=None,
    expression_reconstruction_checkpoint_path=None,
    joint_checkpoint_path=None,
    checkpoint_dir="checkpoints",
    save_every_step=500,  # Control both similarity plots and checkpoints
    batch_size=8,
    num_epochs=10,
    learning_rate=1e-4,
    max_samples=None,
    val_dataset_path=None,
    max_val_samples=None,
    device="cpu",
    num_workers=0,
    pin_memory=False,
    persistent_workers=False,
    drop_last=True,
    # Learning rate scheduler parameters
    warmup_steps=1000,
    min_lr=1e-6,
    # Loss weight scheduling parameters
    initial_lambda_reconstruction=0.1,
    initial_lambda_temporal=0.5,
    initial_lambda_diversity=0.5,
    warmup_lambda_reconstruction=0.3,
    warmup_lambda_temporal=0.4,
    warmup_lambda_diversity=0.4,
    final_lambda_reconstruction=0.5,
    final_lambda_temporal=0.3,
    final_lambda_diversity=0.3,
    # Architecture configuration parameters
    expr_embed_dim=384,
    expr_num_heads=4,
    expr_num_layers=2,
    expr_dropout=0.1,
    expr_max_subjects=3500,
    expr_ff_dim=1536,  # Feed-forward dimension for expression transformer
    recon_embed_dim=384,
    recon_num_cross_layers=2,
    recon_num_self_layers=2,
    recon_num_heads=8,
    recon_ff_dim=1536,
    recon_dropout=0.1,
    log_dir=None,
    freeze_expression_transformer=False
):
    """
    Train the joint expression and reconstruction model
    
    Args:
        dataset_path: Path to the dataset
        expression_transformer_checkpoint_path: Path to expression transformer checkpoint
        expression_reconstruction_checkpoint_path: Path to expression reconstruction checkpoint
        joint_checkpoint_path: Path to joint model checkpoint (preferred)
        checkpoint_dir: Directory to save checkpoints
        save_every_step: Save checkpoints and plots every N steps
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        max_samples: Maximum number of samples to use
        val_dataset_path: Path to validation dataset
        max_val_samples: Maximum number of validation samples
        device: Device to use for training
        num_workers: Number of data loader workers
        pin_memory: Whether to pin memory
        persistent_workers: Whether to use persistent workers
        drop_last: Whether to drop last incomplete batch
        warmup_steps: Number of warmup steps for scheduler
        min_lr: Minimum learning rate
        initial_lambda_reconstruction: Initial reconstruction loss weight
        initial_lambda_temporal: Initial temporal loss weight
        initial_lambda_diversity: Initial diversity loss weight
        warmup_lambda_reconstruction: Warmup reconstruction loss weight
        warmup_lambda_temporal: Warmup temporal loss weight
        warmup_lambda_diversity: Warmup diversity loss weight
        final_lambda_reconstruction: Final reconstruction loss weight
        final_lambda_temporal: Final temporal loss weight
        final_lambda_diversity: Final diversity loss weight
        expr_embed_dim: Expression transformer embedding dimension
        expr_num_heads: Expression transformer number of heads
        expr_num_layers: Expression transformer number of layers
        expr_dropout: Expression transformer dropout
        expr_max_subjects: Expression transformer max subjects
        expr_ff_dim: Expression transformer feed-forward dimension
        recon_embed_dim: Reconstruction model embedding dimension
        recon_num_cross_layers: Reconstruction model cross-attention layers
        recon_num_self_layers: Reconstruction model self-attention layers
        recon_num_heads: Reconstruction model number of heads
        recon_ff_dim: Reconstruction model feed-forward dimension
        recon_dropout: Reconstruction model dropout
        log_dir: Directory for TensorBoard logs
        freeze_expression_transformer: If True, freeze expression transformer and only train reconstruction model
    """
    logger.info(f"Starting joint expression and reconstruction training on device: {device}")
    
    # CUDA memory optimization settings
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        logger.info("Applied CUDA memory optimizations")
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load dataset
    dataset = FaceDataset(dataset_path, max_samples=max_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers, drop_last=drop_last)
    
    # Load validation dataset if provided
    val_dataloader = None
    if val_dataset_path is not None:
        val_dataset = FaceDataset(val_dataset_path, max_samples=max_val_samples)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers, drop_last=drop_last)
        logger.info(f"Validation dataset loaded with {len(val_dataloader)} batches")
        print(f"📊 Validation dataset: {len(val_dataset)} samples, {len(val_dataloader)} batches")
    
    logger.info(f"Training dataset loaded with {len(dataloader)} batches")
    
    # Calculate steps per epoch and total training steps
    steps_per_epoch = len(dataloader)
    total_training_steps = steps_per_epoch * num_epochs
    
    print(f"📊 Training dataset loaded: {len(dataloader)} batches per epoch")
    print(f"📈 Steps per epoch: {steps_per_epoch}")
    print(f"📈 Total training steps: {total_training_steps}")
    print(f"📦 Batch size: {batch_size}")
    print(f"📊 Dataset size: {len(dataset)} samples")
    
    # Initialize DINOv2 tokenizer
    dinov2_tokenizer = DINOv2Tokenizer(device=device)
    
    # Load from joint checkpoint if provided (preferred method)
    if joint_checkpoint_path is not None:
        logger.info(f"Loading joint model from checkpoint: {joint_checkpoint_path}")
        try:
            # Load checkpoint and extract configuration
            joint_checkpoint, config = load_checkpoint_config(joint_checkpoint_path, device)
            
            # Extract model parameters from config
            default_params = {
                'expr_embed_dim': expr_embed_dim,
                'expr_num_heads': expr_num_heads,
                'expr_num_layers': expr_num_layers,
                'expr_dropout': expr_dropout,
                'expr_max_subjects': expr_max_subjects,
                'expr_ff_dim': expr_ff_dim,
                'recon_embed_dim': recon_embed_dim,
                'recon_num_cross_layers': recon_num_cross_layers,
                'recon_num_self_layers': recon_num_self_layers,
                'recon_num_heads': recon_num_heads,
                'recon_ff_dim': recon_ff_dim,
                'recon_dropout': recon_dropout
            }
            
            extracted_params = extract_model_config(config, default_params)
            
            # Update parameters
            expr_embed_dim = extracted_params['expr_embed_dim']
            expr_num_heads = extracted_params['expr_num_heads']
            expr_num_layers = extracted_params['expr_num_layers']
            expr_dropout = extracted_params['expr_dropout']
            expr_max_subjects = extracted_params['expr_max_subjects']
            expr_ff_dim = extracted_params['expr_ff_dim']
            recon_embed_dim = extracted_params['recon_embed_dim']
            recon_num_cross_layers = extracted_params['recon_num_cross_layers']
            recon_num_self_layers = extracted_params['recon_num_self_layers']
            recon_num_heads = extracted_params['recon_num_heads']
            recon_ff_dim = extracted_params['recon_ff_dim']
            recon_dropout = extracted_params['recon_dropout']
            
            # Initialize joint model with architecture from checkpoint
            joint_model = JointExpressionReconstructionModel(
                expr_embed_dim=expr_embed_dim,
                expr_num_heads=expr_num_heads,
                expr_num_layers=expr_num_layers,
                expr_dropout=expr_dropout,
                expr_max_subjects=expr_max_subjects,
                expr_ff_dim=expr_ff_dim,
                recon_embed_dim=recon_embed_dim,
                recon_num_cross_layers=recon_num_cross_layers,
                recon_num_self_layers=recon_num_self_layers,
                recon_num_heads=recon_num_heads,
                recon_ff_dim=recon_ff_dim,
                recon_dropout=recon_dropout
            ).to(device)
            
            # Load the entire joint model state dict
            if 'joint_model_state_dict' in joint_checkpoint:
                joint_model.load_state_dict(joint_checkpoint['joint_model_state_dict'])
                logger.info(f"✅ Successfully loaded joint model from epoch {joint_checkpoint.get('epoch', 'unknown')}")
            else:
                # Try loading the entire checkpoint as state dict (for compatibility)
                joint_model.load_state_dict(joint_checkpoint)
                logger.info("✅ Successfully loaded joint model state dict directly")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load joint checkpoint: {str(e)}")
    
    # Initialize joint model with separate architectures for each component (fallback)
    else:
        joint_model = JointExpressionReconstructionModel(
            expr_embed_dim=expr_embed_dim,
            expr_num_heads=expr_num_heads,
            expr_num_layers=expr_num_layers,
            expr_dropout=expr_dropout,
            expr_max_subjects=expr_max_subjects,
            expr_ff_dim=expr_ff_dim,
            recon_embed_dim=recon_embed_dim,
            recon_num_cross_layers=recon_num_cross_layers,
            recon_num_self_layers=recon_num_self_layers,
            recon_num_heads=recon_num_heads,
            recon_ff_dim=recon_ff_dim,
            recon_dropout=recon_dropout
        ).to(device)
    
    # Handle freezing of expression transformer if requested
    if freeze_expression_transformer:
        logger.info("🔒 Freezing expression transformer parameters")
        for param in joint_model.expression_transformer.parameters():
            param.requires_grad = False
        
        # Only optimize reconstruction model parameters
        trainable_params = list(joint_model.expression_reconstruction.parameters())
        optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)
        logger.info(f"📊 Training only expression reconstruction model ({len(trainable_params)} parameter groups)")
    else:
        logger.info("🔄 Training both expression transformer and reconstruction model")
        optimizer = optim.AdamW(joint_model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Initialize loss function
    criterion = ExpressionReconstructionLoss(
        lambda_reconstruction=initial_lambda_reconstruction,
        lambda_temporal=initial_lambda_temporal,
        lambda_diversity=initial_lambda_diversity
    ).to(device)
    
    # Initialize scheduler
    initial_weights = {
        'lambda_reconstruction': initial_lambda_reconstruction,
        'lambda_temporal': initial_lambda_temporal,
        'lambda_diversity': initial_lambda_diversity
    }
    warmup_weights = {
        'lambda_reconstruction': warmup_lambda_reconstruction,
        'lambda_temporal': warmup_lambda_temporal,
        'lambda_diversity': warmup_lambda_diversity
    }
    final_weights = {
        'lambda_reconstruction': final_lambda_reconstruction,
        'lambda_temporal': final_lambda_temporal,
        'lambda_diversity': final_lambda_diversity
    }
    
    scheduler = CombinedLRLossWeightScheduler(
        optimizer=optimizer,
        initial_lr=learning_rate,
        warmup_lr=learning_rate,
        final_lr=min_lr,
        initial_weights=initial_weights,
        warmup_weights=warmup_weights,
        final_weights=final_weights,
        warmup_steps=warmup_steps,
        total_steps=total_training_steps,
        min_lr=min_lr
    )
    
    # Load optimizer state from checkpoint if provided
    if joint_checkpoint_path is not None and joint_checkpoint is not None:
        try:
            if 'optimizer_state_dict' in joint_checkpoint:
                optimizer.load_state_dict(joint_checkpoint['optimizer_state_dict'])
                logger.info("✅ Successfully loaded optimizer state from checkpoint")
            
            # Don't load scheduler state - use new parameters instead
            logger.info("📊 Using new weight scheduling parameters (not loading checkpoint scheduler state)")
            
        except Exception as e:
            logger.warning(f"Failed to load optimizer state: {str(e)}")
    
    # Initialize TensorBoard
    job_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if log_dir is None:
        log_dir = f"./logs/exp_recon_training_{job_id}_{timestamp}"
    else:
        log_dir = os.path.join(log_dir, f"exp_recon_training_{job_id}_{timestamp}")
    
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    logger.info(f"📊 TensorBoard logging to: {log_dir}")
    logger.info(f"🆔 Job ID: exp_recon_training_{job_id}")
    
    # Training loop
    joint_model.train()
    current_training_step = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Prepare data
            face_images, subject_ids, clip_lengths = prepare_expression_reconstruction_data(
                batch, dinov2_tokenizer, device
            )
            
            # Forward pass through joint model
            all_expression_tokens, all_reconstructed_images = joint_model(face_images, subject_ids, dinov2_tokenizer, clip_lengths)
            
            # Prepare data for loss computation
            original_images = []
            reconstructed_images = []
            expression_tokens_by_clip = []
            
            start_idx = 0
            for clip_idx, (clip_expression_tokens, clip_reconstructed_images) in enumerate(zip(all_expression_tokens, all_reconstructed_images)):
                clip_length = clip_lengths[clip_idx]
                end_idx = start_idx + clip_length
                
                # Extract original images for this clip
                clip_original_images = face_images[start_idx:end_idx]  # (T, 3, 518, 518)
                
                original_images.append(clip_original_images)
                reconstructed_images.append(clip_reconstructed_images)
                expression_tokens_by_clip.append(clip_expression_tokens)
                
                start_idx = end_idx
            
            # Compute loss
            loss, loss_components = criterion(original_images, reconstructed_images, expression_tokens_by_clip)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Step scheduler
            current_lr, current_weights = scheduler.step()
            criterion.update_weights(current_weights)
            
            # Update metrics
            epoch_loss += loss.item()
            num_batches += 1
            current_training_step += 1
            
            # Log to TensorBoard
            writer.add_scalar('Training/Total_Loss', loss.item(), current_training_step)
            writer.add_scalar('Training/Learning_Rate', current_lr, current_training_step)
            writer.add_scalar('Training/Loss_Weights/lambda_reconstruction', current_weights['lambda_reconstruction'], current_training_step)
            writer.add_scalar('Training/Loss_Weights/lambda_temporal', current_weights['lambda_temporal'], current_training_step)
            writer.add_scalar('Training/Loss_Weights/lambda_diversity', current_weights['lambda_diversity'], current_training_step)
            
            # Log individual loss components
            writer.add_scalar('Training/Loss_Components/Reconstruction', loss_components['reconstruction'].item(), current_training_step)
            writer.add_scalar('Training/Loss_Components/Temporal', loss_components['temporal'].item(), current_training_step)
            writer.add_scalar('Training/Loss_Components/Diversity', loss_components['diversity'].item(), current_training_step)
            
            # Save similarity plot and checkpoints every save_every_step steps (moved outside the clip count condition)
            if current_training_step % save_every_step == 0 and expression_tokens_by_clip:
                try:
                    # Compute similarity for single clip (within-clip similarities)
                    if len(expression_tokens_by_clip) >= 1:
                        # Compute similarity data for plotting
                        similarity_data = compute_cosine_similarity_distribution(expression_tokens_by_clip)
                                                # Log similarity statistics to TensorBoard
                        writer.add_scalar('Training/Similarity/Mean', similarity_data['mean_similarity'], current_training_step)
                        writer.add_scalar('Training/Similarity/Std', similarity_data['std_similarity'], current_training_step)
                        writer.add_scalar('Training/Similarity/Min', similarity_data['min_similarity'], current_training_step)
                        writer.add_scalar('Training/Similarity/Max', similarity_data['max_similarity'], current_training_step)
                        writer.add_scalar('Training/Similarity/Count', len(similarity_data['similarities']), current_training_step)
                        
                        # Save similarity plot
                        plot_path = os.path.join(log_dir, f'similarity_step_{current_training_step}.png')
                        plot_cosine_similarity_distribution(
                            similarity_data,
                            save_path=plot_path,
                            title=f"Cosine Similarity Distribution - Step {current_training_step} (Single Clip)"
                        )
                        logger.info(f"Saved similarity plot to: {plot_path}")
                    else:
                        logger.debug(f"Skipping similarity plot: no clips available")
                        
                except Exception as e:
                    logger.warning(f"Failed to save similarity plot: {e}")
                
                # Save checkpoints (always save, regardless of clip count)
                config = create_comprehensive_config(
                    expr_embed_dim=expr_embed_dim,
                    expr_num_heads=expr_num_heads,
                    expr_num_layers=expr_num_layers,
                    expr_dropout=expr_dropout,
                    expr_max_subjects=expr_max_subjects,
                    expr_ff_dim=expr_ff_dim,
                    recon_embed_dim=recon_embed_dim,
                    recon_num_cross_layers=recon_num_cross_layers,
                    recon_num_self_layers=recon_num_self_layers,
                    recon_num_heads=recon_num_heads,
                    recon_ff_dim=recon_ff_dim,
                    recon_dropout=recon_dropout,
                    lambda_reconstruction=initial_lambda_reconstruction,
                    lambda_temporal=initial_lambda_temporal,
                    lambda_diversity=initial_lambda_diversity,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    num_epochs=num_epochs,
                    warmup_steps=warmup_steps,
                    min_lr=min_lr,
                    initial_lambda_reconstruction=initial_lambda_reconstruction,
                    initial_lambda_temporal=initial_lambda_temporal,
                    initial_lambda_diversity=initial_lambda_diversity,
                    warmup_lambda_reconstruction=warmup_lambda_reconstruction,
                    warmup_lambda_temporal=warmup_lambda_temporal,
                    warmup_lambda_diversity=warmup_lambda_diversity,
                    final_lambda_reconstruction=final_lambda_reconstruction,
                    final_lambda_temporal=final_lambda_temporal,
                    final_lambda_diversity=final_lambda_diversity
                )
                
                # Joint model
                joint_step_path = os.path.join(checkpoint_dir, f"joint_expression_reconstruction_step_{current_training_step}.pt")
                save_checkpoint(
                    model_state_dict=joint_model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    scheduler_state_dict=scheduler.state_dict(),
                    epoch=epoch + 1,
                    avg_loss=loss.item(),
                    total_steps=current_training_step,
                    config=config,
                    checkpoint_path=joint_step_path,
                    checkpoint_type="joint"
                )
                
                # Individual models
                expr_step_path = os.path.join(checkpoint_dir, f"expression_transformer_step_{current_training_step}.pt")
                save_checkpoint(
                    model_state_dict=joint_model.expression_transformer.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    scheduler_state_dict=scheduler.state_dict(),
                    epoch=epoch + 1,
                    avg_loss=loss.item(),
                    total_steps=current_training_step,
                    config=config,
                    checkpoint_path=expr_step_path,
                    checkpoint_type="expression_transformer"
                )
                
                recon_step_path = os.path.join(checkpoint_dir, f"expression_reconstruction_step_{current_training_step}.pt")
                save_checkpoint(
                    model_state_dict=joint_model.expression_reconstruction.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    scheduler_state_dict=scheduler.state_dict(),
                    epoch=epoch + 1,
                    avg_loss=loss.item(),
                    total_steps=current_training_step,
                    config=config,
                    checkpoint_path=recon_step_path,
                    checkpoint_type="expression_reconstruction"
                )
                
                logger.info(f"Saved step checkpoints: {joint_step_path}, {expr_step_path}, {recon_step_path}")
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.5f}',
                'Avg Loss': f'{epoch_loss / num_batches:.5f}',
                'Recon': f'{loss_components["reconstruction"].item():.5f}',
                'Temporal': f'{loss_components["temporal"].item():.5f}',
                'Diversity': f'{loss_components["diversity"].item():.5f}',
                'LR': f'{current_lr:.2e}',
                'λ_recon': f'{current_weights["lambda_reconstruction"]:.2f}',
                'λ_temp': f'{current_weights["lambda_temporal"]:.2f}',
                'λ_div': f'{current_weights["lambda_diversity"]:.2f}',
                'Step': f'{current_training_step}'
            })
        
        avg_loss = epoch_loss / num_batches
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_loss:.4f}")
        
        # CUDA memory cleanup after each epoch
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        # Log epoch metrics to TensorBoard
        writer.add_scalar('Training/Epoch_Loss', avg_loss, epoch + 1)
        
        # Validate if validation dataloader is provided
        if val_dataloader is not None:
            val_loss = validate_expression_reconstruction(
                joint_model, val_dataloader, criterion, dinov2_tokenizer, device
            )
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Log validation metrics to TensorBoard
            writer.add_scalar('Validation/Epoch_Loss', val_loss, epoch + 1)
    
    # Removed tensorboard histogram logging to avoid compatibility issues
    
    # Close TensorBoard writer
    writer.close()
    
    logger.info("Training completed!")
    logger.info(f"📊 TensorBoard logs saved to: {log_dir}")
    return joint_model


def validate_expression_reconstruction(
    joint_model,
    val_dataloader,
    criterion,
    dinov2_tokenizer,
    device="cpu"
):
    """
    Validate the joint expression reconstruction model
    """
    joint_model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            # Prepare data
            face_images, subject_ids, clip_lengths = prepare_expression_reconstruction_data(
                batch, dinov2_tokenizer, device
            )
            
            # Forward pass through joint model
            all_expression_tokens, all_reconstructed_images = joint_model(face_images, subject_ids, dinov2_tokenizer, clip_lengths)
            
            # Prepare data for loss computation
            original_images = []
            reconstructed_images = []
            expression_tokens_by_clip = []
            
            start_idx = 0
            for clip_idx, (clip_expression_tokens, clip_reconstructed_images) in enumerate(zip(all_expression_tokens, all_reconstructed_images)):
                clip_length = clip_lengths[clip_idx]
                end_idx = start_idx + clip_length
                
                # Extract original images for this clip
                clip_original_images = face_images[start_idx:end_idx]
                
                original_images.append(clip_original_images)
                reconstructed_images.append(clip_reconstructed_images)
                expression_tokens_by_clip.append(clip_expression_tokens)
                
                start_idx = end_idx
            
            # Compute loss
            loss, loss_components = criterion(original_images, reconstructed_images, expression_tokens_by_clip)
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    logger.info(f"Validation - Avg Loss: {avg_loss:.4f}")
    
    return avg_loss


def test_joint_expression_reconstruction():
    """Test the joint expression reconstruction model"""
    print("🧪 Testing Joint Expression Reconstruction Model...")
    
    # Create model
    model = JointExpressionReconstructionModel()
    print("✅ Joint model created successfully")
    
    # Test with dummy input
    batch_size = 2
    num_frames = 5
    device = torch.device("cpu")
    
    # Create dummy data
    face_images = torch.randn(batch_size * num_frames, 3, 518, 518)
    subject_ids = torch.randint(0, 100, (batch_size * num_frames,))
    clip_lengths = [num_frames, num_frames]
    
    # Create dummy tokenizer
    class DummyTokenizer:
        def __call__(self, images):
            return torch.randn(images.shape[0], 1369, 384), torch.randn(images.shape[0], 1369, 384)
    
    tokenizer = DummyTokenizer()
    
    # Forward pass
    expression_tokens, reconstructed_images = model(face_images, subject_ids, tokenizer, clip_lengths)
    
    print(f"Expression tokens shape: {[t.shape for t in expression_tokens]}")
    print(f"Reconstructed images shape: {[r.shape for r in reconstructed_images]}")
    
    print("✅ Joint expression reconstruction test passed!")


if __name__ == "__main__":
    test_joint_expression_reconstruction() 