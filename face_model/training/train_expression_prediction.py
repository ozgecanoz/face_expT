"""
Joint Training: Expression Transformer + Prediction Head
Trains both Component C (Expression Transformer) and Component D (Transformer Decoder)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from tqdm import tqdm
import os
import logging
import sys
import uuid
import math
from datetime import datetime
from typing import Dict
sys.path.append('.')

from models.expression_transformer import ExpressionTransformer
from models.expression_transformer_decoder import ExpressionTransformerDecoder
from models.face_id_model import FaceIDModel
from models.dinov2_tokenizer import DINOv2Tokenizer
from data.dataset import FaceDataset
from utils.checkpoint_utils import (
    save_checkpoint,
    load_checkpoint_config,
    extract_model_config,
    create_comprehensive_config,
    validate_checkpoint_compatibility
)
from utils.scheduler_utils import CombinedLRLossWeightScheduler
from utils.visualization_utils import compute_cosine_similarity_distribution, plot_cosine_similarity_distribution
from utils.tensorboard_utils import log_model_parameters

logger = logging.getLogger(__name__)


class JointExpressionPredictionModel(nn.Module):
    """
    Joint model combining Components C and D
    Component C: Expression Transformer (extracts expression tokens using subject embeddings)
    Component D: Transformer Decoder (predicts next expression token)
    """
    
    def __init__(self, 
                 expr_embed_dim=384, expr_num_heads=4, expr_num_layers=2, expr_dropout=0.1, expr_max_subjects=3500,  # Added max_subjects
                 decoder_embed_dim=384, decoder_num_heads=4, decoder_num_layers=2, decoder_dropout=0.1,  # Changed from 8 to 4 heads
                 max_sequence_length=50):
        super().__init__()
        
        # Component C: Expression Transformer (trainable) - now uses subject embeddings
        self.expression_transformer = ExpressionTransformer(
            embed_dim=expr_embed_dim, 
            num_heads=expr_num_heads, 
            num_layers=expr_num_layers, 
            dropout=expr_dropout,
            max_subjects=expr_max_subjects  # Added max_subjects parameter
        )
        
        # Component D: Transformer Decoder (trainable)
        self.transformer_decoder = ExpressionTransformerDecoder(
            embed_dim=decoder_embed_dim,
            num_heads=decoder_num_heads,
            num_layers=decoder_num_layers,
            dropout=decoder_dropout,
            max_sequence_length=max_sequence_length
        )
        
        logger.info(f"Joint Expression Prediction Model initialized")
        logger.info(f"  Expression Transformer: {expr_num_layers} layers, {expr_num_heads} heads, {expr_embed_dim} dim, {expr_max_subjects} subjects")
        logger.info(f"  Transformer Decoder: {decoder_num_layers} layers, {decoder_num_heads} heads, {decoder_embed_dim} dim")
        
    def forward(self, face_images, subject_ids, tokenizer, clip_lengths=None):
        """
        Args:
            face_images: (total_frames, 3, 518, 518) - Input face images from multiple clips
            subject_ids: (total_frames,) - Subject IDs for each frame
            tokenizer: DINOv2Tokenizer instance
            clip_lengths: List of clip lengths for processing individual clips
        
        Returns:
            all_expression_tokens: List of expression tokens for each clip
            all_predicted_next_tokens: List of predicted next tokens for each clip
        """
        if clip_lengths is None:
            # Fallback: process as single clip
            return self._forward_single_clip(face_images, subject_ids, tokenizer)
        
        # Component A: Extract patch tokens using provided tokenizer
        patch_tokens, pos_embeddings = tokenizer(face_images)  # (total_frames, 1369, 384), (total_frames, 1369, 384)
        
        # Component C: Extract expression tokens for all frames using subject embeddings
        all_expression_tokens = self.expression_transformer(patch_tokens, pos_embeddings, subject_ids)  # (total_frames, 1, 384)
        
        # Component D: Predict next tokens for all clips using transformer decoder
        # Squeeze the middle dimension from (total_frames, 1, 384) to (total_frames, 384)
        all_expression_tokens_squeezed = all_expression_tokens.squeeze(1)  # (total_frames, 384)
        all_predicted_next_tokens = self.transformer_decoder(all_expression_tokens_squeezed, clip_lengths)
        
        # Split expression tokens by clip for return
        expression_tokens_by_clip = []
        current_idx = 0
        for clip_length in clip_lengths:
            clip_expression_tokens = all_expression_tokens[current_idx:current_idx + clip_length]
            expression_tokens_by_clip.append(clip_expression_tokens)
            current_idx += clip_length
        
        return expression_tokens_by_clip, all_predicted_next_tokens
    
    def _forward_single_clip(self, face_images, subject_ids, tokenizer):
        """
        Process a single clip (fallback for when clip_lengths is None)
        
        Args:
            face_images: (num_frames, 3, 518, 518) - Input face images
            subject_ids: (num_frames,) - Subject IDs for each frame
            tokenizer: DINOv2Tokenizer instance
        
        Returns:
            expression_tokens: (num_frames, 1, 384) - Expression tokens
            predicted_next_token: (1, 1, 384) - Predicted next token
        """
        # Component A: Extract patch tokens
        patch_tokens, pos_embeddings = tokenizer(face_images)  # (num_frames, 1369, 384), (num_frames, 1369, 384)
        
        # Component C: Extract expression tokens using subject embeddings
        expression_tokens = self.expression_transformer(patch_tokens, pos_embeddings, subject_ids)  # (num_frames, 1, 384)
        
        # Component D: Predict next token
        predicted_next_token = self.transformer_decoder._forward_single_clip(expression_tokens)  # (1, 1, 384)
        
        return [expression_tokens], [predicted_next_token]


class ExpressionPredictionLoss(nn.Module):
    """
    Loss function for expression prediction training
    """
    
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, predicted_next_token, actual_next_token):
        """
        Args:
            predicted_next_token: (1, 1, 384) - Predicted next token
            actual_next_token: (1, 1, 384) - Actual next token from expression transformer
        """
        # Simple MSE loss between predicted and actual next token
        #loss = self.mse_loss(predicted_next_token, actual_next_token)
        loss = 1 - F.cosine_similarity(predicted_next_token, actual_next_token, dim=-1).mean()
        
        return loss

class ExpressionPredictionLoss_v2(nn.Module):
    """
    Combined loss for expression prediction:
    1. Cosine similarity loss between predicted and actual next token (weighted by lambda_prediction)
    2. Temporal contrastive loss between consecutive expression tokens in a clip
    3. Diversity loss: penalize high similarity among expression tokens within a clip
    """
    def __init__(self, lambda_prediction=1.0, lambda_temporal=0.1, lambda_diversity=0.1):
        super().__init__()
        self.lambda_prediction = lambda_prediction
        self.lambda_temporal = lambda_temporal
        self.lambda_diversity = lambda_diversity
    
    def update_weights(self, new_weights: Dict[str, float]):
        """
        Update loss weights dynamically during training.
        
        Args:
            new_weights: Dictionary with keys 'lambda_prediction', 'lambda_temporal', 'lambda_diversity'
        """
        if 'lambda_prediction' in new_weights:
            self.lambda_prediction = new_weights['lambda_prediction']
        if 'lambda_temporal' in new_weights:
            self.lambda_temporal = new_weights['lambda_temporal']
        if 'lambda_diversity' in new_weights:
            self.lambda_diversity = new_weights['lambda_diversity']

    def forward(self, predicted_next_tokens, actual_next_tokens, expression_tokens_by_clip):
        """
        Args:
            predicted_next_tokens: List[Tensor (1, 1, D)] - Predicted next tokens per clip
            actual_next_tokens:    List[Tensor (1, 1, D)] - Actual next tokens per clip
            expression_tokens_by_clip: List[Tensor (T, 1, D)] - Expression tokens per clip

        Returns:
            total_loss: Scalar loss
        """
        cosine_losses = []
        temporal_losses = []
        diversity_losses = []

        for pred, target, clip_tokens in zip(predicted_next_tokens, actual_next_tokens, expression_tokens_by_clip):
            # ---- 1. Cosine Similarity Loss ----
            cosine_loss = 1 - F.cosine_similarity(pred.view(1, -1), target.view(1, -1)).mean()
            cosine_losses.append(cosine_loss)

            # ---- 2. Temporal Contrastive Loss (encourage change over time) ----
            if clip_tokens.shape[0] > 1:
                t1 = clip_tokens[:-1].squeeze(1)  # (T-1, D)
                t2 = clip_tokens[1:].squeeze(1)   # (T-1, D)
                temporal_sim = F.cosine_similarity(t1, t2, dim=-1)  # (T-1,)
                temporal_loss = 1 - temporal_sim.mean()
                temporal_losses.append(temporal_loss)

                # ---- 3. Diversity Loss (within-clip token dissimilarity) ----
                # clip_tokens are already normalized from ExpressionTransformer
                tokens = clip_tokens.squeeze(1)  # (T, D) - already normalized
                #sim_matrix = tokens @ tokens.T  # (T, T)
                # Subtract diagonal (self-similarities = 1.0) to get only cross-token similarities
                #mask = torch.eye(sim_matrix.shape[0], device=sim_matrix.device, dtype=torch.bool)
                #cross_similarities = sim_matrix[~mask]  # Exclude diagonal elements
                #diversity_loss = cross_similarities.mean()  # High if tokens are similar

                T = tokens.shape[0]
                sim_matrix = tokens @ tokens.T
                diversity_loss = (sim_matrix.sum() - T) / (T * (T - 1))
                diversity_losses.append(diversity_loss)

        # Aggregate losses
        total_cosine = torch.stack(cosine_losses).mean()
        total_temporal = torch.stack(temporal_losses).mean() if temporal_losses else 0.0
        total_diversity = torch.stack(diversity_losses).mean() if diversity_losses else 0.0

        total_loss = (
            self.lambda_prediction * total_cosine
            + self.lambda_temporal * total_temporal
            + self.lambda_diversity * total_diversity
        )

        # Return both total loss and individual components for logging
        loss_components = {
            'total': total_loss,
            'cosine': total_cosine,
            'temporal': total_temporal,
            'diversity': total_diversity
        }
        
        return total_loss, loss_components


def prepare_expression_training_data(batch, dinov2_tokenizer, device):
    """
    Prepare training data for expression prediction using subject embeddings
    
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


def train_expression_prediction(
    dataset_path,
    expression_transformer_checkpoint_path=None,
    transformer_decoder_checkpoint_path=None,
    joint_checkpoint_path=None,  # New parameter for joint checkpoint
    checkpoint_dir="checkpoints",
    save_every_epochs=2,
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
    initial_lambda_prediction=0.1,
    initial_lambda_temporal=0.5,
    initial_lambda_diversity=0.5,
    warmup_lambda_prediction=0.3,
    warmup_lambda_temporal=0.4,
    warmup_lambda_diversity=0.4,
    final_lambda_prediction=0.5,
    final_lambda_temporal=0.3,
    final_lambda_diversity=0.3,
    # Architecture configuration parameters
    expr_embed_dim=384,
    expr_num_heads=4,  # Changed from 8 to 4 to match your config
    expr_num_layers=2,
    expr_dropout=0.1,
    expr_max_subjects=3500,  # Added max_subjects parameter
    decoder_embed_dim=384,
    decoder_num_heads=4,  # Changed from 8 to 4 to match your config
    decoder_num_layers=2,
    decoder_dropout=0.1,
    max_sequence_length=50,
    log_dir=None
):
    """
    Train the joint expression prediction model using subject embeddings
    No longer requires a pre-trained face ID model
    """
    logger.info(f"Starting joint training on device: {device}")
    
    # CUDA memory optimization settings
    if device.type == "cuda":
        torch.cuda.empty_cache()  # Clear cache
        torch.backends.cudnn.benchmark = True  # Optimize for your GPU
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
    
    # Estimate training time (more realistic for CPU training)
    estimated_time_per_epoch = len(dataloader) * 4  # 125 seconds per batch for CPU VM
    total_estimated_time = estimated_time_per_epoch * num_epochs / 3600  # Convert to hours
    
    print(f"📊 Training dataset loaded: {len(dataloader)} batches per epoch")
    print(f"📈 Steps per epoch: {steps_per_epoch}")
    print(f"📈 Total training steps: {total_training_steps}")
    print(f"📦 Batch size: {batch_size}")
    print(f"📊 Dataset size: {len(dataset)} samples")
    print(f"⏱️  Estimated training time: {total_estimated_time:.1f} hours ({total_estimated_time/24:.1f} days)")
    print(f"💰 Estimated cost: ${total_estimated_time * 0.38:.1f} (at $0.38/hour)")
    
    # Initialize DINOv2 tokenizer
    dinov2_tokenizer = DINOv2Tokenizer(device=device)
    
    # Initialize checkpoint variables
    expr_checkpoint = None
    decoder_checkpoint = None
    
    # Load expression transformer checkpoint if provided to get architecture
    if expression_transformer_checkpoint_path is not None:
        if not os.path.exists(expression_transformer_checkpoint_path):
            raise FileNotFoundError(
                f"Expression transformer checkpoint not found: {expression_transformer_checkpoint_path}\n"
                f"Please train the expression transformer first or set this parameter to None"
            )
        
        logger.info(f"Loading expression transformer architecture from checkpoint: {expression_transformer_checkpoint_path}")
        try:
            # Load checkpoint and extract configuration
            _, config = load_checkpoint_config(expression_transformer_checkpoint_path, device)
            
            # Extract model parameters from config
            default_params = {
                'expr_embed_dim': expr_embed_dim,
                'expr_num_heads': expr_num_heads,
                'expr_num_layers': expr_num_layers,
                'expr_dropout': expr_dropout,
                'expr_max_subjects': expr_max_subjects,
                'lambda_temporal': lambda_temporal,
                'lambda_diversity': lambda_diversity
            }
            
            extracted_params = extract_model_config(config, default_params)
            
            # Update parameters
            expr_embed_dim = extracted_params['expr_embed_dim']
            expr_num_heads = extracted_params['expr_num_heads']
            expr_num_layers = extracted_params['expr_num_layers']
            expr_dropout = extracted_params['expr_dropout']
            expr_max_subjects = extracted_params['expr_max_subjects']
            lambda_temporal = extracted_params['lambda_temporal']
            lambda_diversity = extracted_params['lambda_diversity']
                
        except Exception as e:
            logger.warning(f"Failed to load expression transformer checkpoint for architecture: {str(e)}, using defaults")
    
    # Use provided architecture parameters for transformer decoder
    # (These are now passed as function parameters instead of hardcoded)
    
    # Load transformer decoder checkpoint if provided to get architecture
    if transformer_decoder_checkpoint_path is not None:
        if not os.path.exists(transformer_decoder_checkpoint_path):
            raise FileNotFoundError(
                f"Transformer decoder checkpoint not found: {transformer_decoder_checkpoint_path}\n"
                f"Please train the transformer decoder first or set this parameter to None"
            )
        
        logger.info(f"Loading transformer decoder architecture from checkpoint: {transformer_decoder_checkpoint_path}")
        try:
            # Load checkpoint and extract configuration
            _, config = load_checkpoint_config(transformer_decoder_checkpoint_path, device)
            
            # Extract model parameters from config
            default_params = {
                'decoder_embed_dim': decoder_embed_dim,
                'decoder_num_heads': decoder_num_heads,
                'decoder_num_layers': decoder_num_layers,
                'decoder_dropout': decoder_dropout,
                'max_sequence_length': max_sequence_length,
                'lambda_temporal': lambda_temporal,
                'lambda_diversity': lambda_diversity
            }
            
            extracted_params = extract_model_config(config, default_params)
            
            # Update parameters
            decoder_embed_dim = extracted_params['decoder_embed_dim']
            decoder_num_heads = extracted_params['decoder_num_heads']
            decoder_num_layers = extracted_params['decoder_num_layers']
            decoder_dropout = extracted_params['decoder_dropout']
            max_sequence_length = extracted_params['max_sequence_length']
            lambda_temporal = extracted_params['lambda_temporal']
            lambda_diversity = extracted_params['lambda_diversity']
                
        except Exception as e:
            logger.warning(f"Failed to load transformer decoder checkpoint for architecture: {str(e)}, using defaults")
    
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
                'decoder_embed_dim': decoder_embed_dim,
                'decoder_num_heads': decoder_num_heads,
                'decoder_num_layers': decoder_num_layers,
                'decoder_dropout': decoder_dropout,
                'max_sequence_length': max_sequence_length,
                'lambda_temporal': lambda_temporal,
                'lambda_diversity': lambda_diversity
            }
            
            extracted_params = extract_model_config(config, default_params)
            
            # Update parameters
            expr_embed_dim = extracted_params['expr_embed_dim']
            expr_num_heads = extracted_params['expr_num_heads']
            expr_num_layers = extracted_params['expr_num_layers']
            expr_dropout = extracted_params['expr_dropout']
            expr_max_subjects = extracted_params['expr_max_subjects']
            decoder_embed_dim = extracted_params['decoder_embed_dim']
            decoder_num_heads = extracted_params['decoder_num_heads']
            decoder_num_layers = extracted_params['decoder_num_layers']
            decoder_dropout = extracted_params['decoder_dropout']
            max_sequence_length = extracted_params['max_sequence_length']
            lambda_temporal = extracted_params['lambda_temporal']
            lambda_diversity = extracted_params['lambda_diversity']
            
            # Initialize joint model with architecture from checkpoint
            joint_model = JointExpressionPredictionModel(
                expr_embed_dim=expr_embed_dim,
                expr_num_heads=expr_num_heads,
                expr_num_layers=expr_num_layers,
                expr_dropout=expr_dropout,
                expr_max_subjects=expr_max_subjects,
                decoder_embed_dim=decoder_embed_dim,
                decoder_num_heads=decoder_num_heads,
                decoder_num_layers=decoder_num_layers,
                decoder_dropout=decoder_dropout,
                max_sequence_length=max_sequence_length
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
        joint_model = JointExpressionPredictionModel(
            expr_embed_dim=expr_embed_dim,
            expr_num_heads=expr_num_heads,
            expr_num_layers=expr_num_layers,
            expr_dropout=expr_dropout,
            expr_max_subjects=expr_max_subjects,  # Added max_subjects parameter
            decoder_embed_dim=decoder_embed_dim,
            decoder_num_heads=decoder_num_heads,
            decoder_num_layers=decoder_num_layers,
            decoder_dropout=decoder_dropout,
            max_sequence_length=max_sequence_length
        ).to(device)
    
    # Load expression transformer weights if checkpoint provided (fallback)
    if expression_transformer_checkpoint_path is not None:
        logger.info(f"Loading expression transformer weights from checkpoint: {expression_transformer_checkpoint_path}")
        try:
            # Load checkpoint if not already loaded
            if expr_checkpoint is None:
                expr_checkpoint = torch.load(expression_transformer_checkpoint_path, map_location=device)
            
            # Load the expression transformer state dict
            if 'expression_transformer_state_dict' in expr_checkpoint:
                joint_model.expression_transformer.load_state_dict(expr_checkpoint['expression_transformer_state_dict'])
                logger.info(f"Loaded expression transformer from epoch {expr_checkpoint.get('epoch', 'unknown')}")
            else:
                # Try loading the entire checkpoint as state dict (for compatibility)
                joint_model.expression_transformer.load_state_dict(expr_checkpoint)
                logger.info("Loaded expression transformer state dict directly")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load expression transformer checkpoint: {str(e)}")
    else:
        logger.info("No expression transformer checkpoint provided - training from scratch")
    
    # Load transformer decoder weights if checkpoint provided
    if transformer_decoder_checkpoint_path is not None:
        logger.info(f"Loading transformer decoder weights from checkpoint: {transformer_decoder_checkpoint_path}")
        try:
            # Load checkpoint if not already loaded
            if decoder_checkpoint is None:
                decoder_checkpoint = torch.load(transformer_decoder_checkpoint_path, map_location=device)
            
            # Load the transformer decoder state dict
            if 'transformer_decoder_state_dict' in decoder_checkpoint:
                joint_model.transformer_decoder.load_state_dict(decoder_checkpoint['transformer_decoder_state_dict'])
                logger.info(f"Loaded transformer decoder from epoch {decoder_checkpoint.get('epoch', 'unknown')}")
            else:
                # Try loading the entire checkpoint as state dict (for compatibility)
                joint_model.transformer_decoder.load_state_dict(decoder_checkpoint)
                logger.info("Loaded transformer decoder state dict directly")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load transformer decoder checkpoint: {str(e)}")
    else:
        logger.info("No transformer decoder checkpoint provided - training from scratch")
    
    # Initialize loss function with initial weights
    criterion = ExpressionPredictionLoss_v2(
        lambda_prediction=initial_lambda_prediction,
        lambda_temporal=initial_lambda_temporal,
        lambda_diversity=initial_lambda_diversity
    ).to(device)
    
    logger.info(f"Using ExpressionPredictionLoss_v2 with dynamic weight scheduling:")
    logger.info(f"  Initial weights: λ_pred={initial_lambda_prediction}, λ_temp={initial_lambda_temporal}, λ_div={initial_lambda_diversity}")
    logger.info(f"  Warmup weights: λ_pred={warmup_lambda_prediction}, λ_temp={warmup_lambda_temporal}, λ_div={warmup_lambda_diversity}")
    logger.info(f"  Final weights: λ_pred={final_lambda_prediction}, λ_temp={final_lambda_temporal}, λ_div={final_lambda_diversity}")
    
    # Log final model architecture for verification
    logger.info("Final model architecture:")
    logger.info(f"  Expression Transformer: {expr_num_layers} layers, {expr_num_heads} heads, {expr_embed_dim} dim, {expr_max_subjects} subjects")
    logger.info(f"  Transformer Decoder: {decoder_num_layers} layers, {decoder_num_heads} heads, {decoder_embed_dim} dim, {max_sequence_length} max_seq_len")
    logger.info(f"  Total parameters: {sum(p.numel() for p in joint_model.parameters()):,}")
    logger.info(f"  Trainable parameters: {sum(p.numel() for p in joint_model.parameters() if p.requires_grad):,}")
    
    # Track training step for LR scheduler resumption
    current_training_step = 0
    
    # Check if we're resuming from a checkpoint and need to restore training step
    checkpoint_to_check = None
    
    # Priority: joint checkpoint > individual checkpoints
    if joint_checkpoint_path is not None:
        checkpoint_to_check = joint_checkpoint_path
        logger.info(f"Will load from joint checkpoint: {joint_checkpoint_path}")
    elif expression_transformer_checkpoint_path is not None or transformer_decoder_checkpoint_path is not None:
        checkpoint_to_check = expression_transformer_checkpoint_path if expression_transformer_checkpoint_path else transformer_decoder_checkpoint_path
        logger.info(f"Will load from individual checkpoint: {checkpoint_to_check}")
    
    if checkpoint_to_check is not None:
        try:
            checkpoint = torch.load(checkpoint_to_check, map_location=device)
            if 'total_steps' in checkpoint:
                current_training_step = checkpoint['total_steps']
                logger.info(f"Resuming training from step {current_training_step}")
            else:
                logger.info("No training step found in checkpoint, starting from step 0")
        except Exception as e:
            logger.warning(f"Could not load training step from checkpoint: {e}, starting from step 0")
    
    # Initialize optimizer (train both expression transformer and transformer decoder)
    trainable_params = list(joint_model.expression_transformer.parameters()) + list(joint_model.transformer_decoder.parameters())
    optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)
    
    # Load optimizer state if available
    if joint_checkpoint_path is not None:
        try:
            joint_checkpoint = torch.load(joint_checkpoint_path, map_location=device)
            if 'optimizer_state_dict' in joint_checkpoint:
                optimizer.load_state_dict(joint_checkpoint['optimizer_state_dict'])
                logger.info("Loaded optimizer state from joint checkpoint")
        except Exception as e:
            logger.warning(f"Could not load optimizer state from joint checkpoint: {e}")
    elif expression_transformer_checkpoint_path is not None or transformer_decoder_checkpoint_path is not None:
        checkpoint_to_check = expression_transformer_checkpoint_path if expression_transformer_checkpoint_path else transformer_decoder_checkpoint_path
        try:
            checkpoint = torch.load(checkpoint_to_check, map_location=device)
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info("Loaded optimizer state from checkpoint")
        except Exception as e:
            logger.warning(f"Could not load optimizer state from checkpoint: {e}")
    
    # Calculate total training steps for scheduler
    total_training_steps = len(dataloader) * num_epochs
    
    # Define loss weight schedules using function parameters
    initial_weights = {
        'lambda_prediction': initial_lambda_prediction,
        'lambda_temporal': initial_lambda_temporal,
        'lambda_diversity': initial_lambda_diversity
    }
    warmup_weights = {
        'lambda_prediction': warmup_lambda_prediction,
        'lambda_temporal': warmup_lambda_temporal,
        'lambda_diversity': warmup_lambda_diversity
    }
    final_weights = {
        'lambda_prediction': final_lambda_prediction,
        'lambda_temporal': final_lambda_temporal,
        'lambda_diversity': final_lambda_diversity
    }
    
    # Initialize combined LR and loss weight scheduler
    scheduler = CombinedLRLossWeightScheduler(
        optimizer=optimizer,
        initial_lr=min_lr,
        warmup_lr=learning_rate,
        final_lr=min_lr,
        initial_weights=initial_weights,
        warmup_weights=warmup_weights,
        final_weights=final_weights,
        warmup_steps=warmup_steps,
        total_steps=total_training_steps,
        min_lr=min_lr
    )
    
    # Load scheduler state if available
    if joint_checkpoint_path is not None:
        try:
            joint_checkpoint = torch.load(joint_checkpoint_path, map_location=device)
            if 'scheduler_state_dict' in joint_checkpoint:
                scheduler.load_state_dict(joint_checkpoint['scheduler_state_dict'])
                logger.info("Loaded scheduler state from joint checkpoint")
            else:
                # Step the scheduler to the current training step if resuming
                if current_training_step > 0:
                    for _ in range(current_training_step):
                        scheduler.step()
                    logger.info(f"Advanced scheduler to step {current_training_step}")
        except Exception as e:
            logger.warning(f"Could not load scheduler state from joint checkpoint: {e}")
            # Fallback to stepping scheduler
            if current_training_step > 0:
                for _ in range(current_training_step):
                    scheduler.step()
                logger.info(f"Advanced scheduler to step {current_training_step}")
    elif expression_transformer_checkpoint_path is not None or transformer_decoder_checkpoint_path is not None:
        checkpoint_to_check = expression_transformer_checkpoint_path if expression_transformer_checkpoint_path else transformer_decoder_checkpoint_path
        try:
            checkpoint = torch.load(checkpoint_to_check, map_location=device)
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logger.info("Loaded scheduler state from checkpoint")
            else:
                # Step the scheduler to the current training step if resuming
                if current_training_step > 0:
                    for _ in range(current_training_step):
                        scheduler.step()
                    logger.info(f"Advanced scheduler to step {current_training_step}")
        except Exception as e:
            logger.warning(f"Could not load scheduler state from checkpoint: {e}")
            # Fallback to stepping scheduler
            if current_training_step > 0:
                for _ in range(current_training_step):
                    scheduler.step()
                logger.info(f"Advanced scheduler to step {current_training_step}")
    else:
        # Step the scheduler to the current training step if resuming
        if current_training_step > 0:
            for _ in range(current_training_step):
                scheduler.step()
            logger.info(f"Advanced scheduler to step {current_training_step}")
    
    logger.info(f"Combined LR and loss weight scheduler initialized:")
    logger.info(f"  LR schedule: {min_lr} -> {learning_rate} -> {min_lr}")
    logger.info(f"  Weight schedule:")
    logger.info(f"    Initial: {initial_weights}")
    logger.info(f"    Warmup: {warmup_weights}")
    logger.info(f"    Final: {final_weights}")
    logger.info(f"  Warmup steps: {warmup_steps}")
    logger.info(f"  Total training steps: {total_training_steps}")
    logger.info(f"  Current training step: {current_training_step}")
    logger.info("Expression transformer and transformer decoder parameters will be trained")
    
    # Initialize TensorBoard with log directory from config as base folder
    job_id = str(uuid.uuid4())[:8]  # Short unique ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if log_dir is None:
        # Fallback to default if no log_dir provided
        log_dir = f"./logs/exp_pred_training_{job_id}_{timestamp}"
    else:
        # Use provided log_dir as base folder and create unique subfolder
        log_dir = os.path.join(log_dir, f"exp_pred_training_{job_id}_{timestamp}")
    
    # Ensure the directory exists
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    logger.info(f"📊 TensorBoard logging to: {log_dir}")
    logger.info(f"🆔 Job ID: exp_pred_training_{job_id}")
    
    # Training loop
    joint_model.train()
    total_steps = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Prepare data
            face_images, subject_ids, clip_lengths = prepare_expression_training_data(
                batch, dinov2_tokenizer, device
            )
            
            # Forward pass through joint model
            all_expression_tokens, all_predicted_next_tokens = joint_model(face_images, subject_ids, dinov2_tokenizer, clip_lengths)
            
            # Prepare data for v2 loss computation
            predicted_next_tokens = []
            actual_next_tokens = []
            expression_tokens_by_clip = []
            
            for clip_idx, (clip_expression_tokens, clip_predicted_next_token) in enumerate(zip(all_expression_tokens, all_predicted_next_tokens)):
                if clip_expression_tokens.shape[0] > 1:
                    # Use the last token as the target
                    actual_next_token = clip_expression_tokens[-1:]  # (1, 1, 384)
                    predicted_next_tokens.append(clip_predicted_next_token)
                    actual_next_tokens.append(actual_next_token)
                    expression_tokens_by_clip.append(clip_expression_tokens)
            
            # Compute v2 loss if we have valid clips
            if predicted_next_tokens:
                loss, loss_components = criterion(predicted_next_tokens, actual_next_tokens, expression_tokens_by_clip)
            else:
                # No valid clips (all single frames)
                loss = torch.tensor(0.0, device=device)
                loss_components = {
                    'total': loss,
                    'cosine': torch.tensor(0.0, device=device),
                    'temporal': torch.tensor(0.0, device=device),
                    'diversity': torch.tensor(0.0, device=device)
                }
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Step the combined scheduler and get current LR and loss weights
            current_lr, current_weights = scheduler.step()
            
            # Update loss function weights dynamically
            criterion.update_weights(current_weights)
            
            # Update metrics
            epoch_loss += loss.item()
            num_batches += 1
            
            # Log to TensorBoard
            writer.add_scalar('Training/Batch_Loss', loss.item(), current_training_step)
            writer.add_scalar('Training/Avg_Loss', epoch_loss / num_batches, current_training_step)
            writer.add_scalar('Training/Learning_Rate', current_lr, current_training_step)
            
            # Log loss weights
            writer.add_scalar('Training/Loss_Weights/lambda_prediction', current_weights['lambda_prediction'], current_training_step)
            writer.add_scalar('Training/Loss_Weights/lambda_temporal', current_weights['lambda_temporal'], current_training_step)
            writer.add_scalar('Training/Loss_Weights/lambda_diversity', current_weights['lambda_diversity'], current_training_step)
            
            # Log individual loss components
            writer.add_scalar('Training/Loss_Components/Cosine', loss_components['cosine'].item(), current_training_step)
            writer.add_scalar('Training/Loss_Components/Temporal', loss_components['temporal'].item(), current_training_step)
            writer.add_scalar('Training/Loss_Components/Diversity', loss_components['diversity'].item(), current_training_step)
            
            # Compute and log cosine similarity distribution (every 100 steps to avoid overhead)
            if current_training_step % 100 == 0 and expression_tokens_by_clip:
                try:
                    # Ensure we have enough clips for meaningful analysis
                    if len(expression_tokens_by_clip) >= 2:
                        similarity_data = compute_cosine_similarity_distribution(expression_tokens_by_clip)
                        
                        # Log similarity statistics to TensorBoard
                        writer.add_scalar('Training/Similarity/Mean', similarity_data['mean_similarity'], current_training_step)
                        writer.add_scalar('Training/Similarity/Std', similarity_data['std_similarity'], current_training_step)
                        writer.add_scalar('Training/Similarity/Min', similarity_data['min_similarity'], current_training_step)
                        writer.add_scalar('Training/Similarity/Max', similarity_data['max_similarity'], current_training_step)
                        writer.add_scalar('Training/Similarity/Count', len(similarity_data['similarities']), current_training_step)
                        
                        # Save similarity plot every 300 steps
                        if current_training_step % 300 == 0:
                            plot_path = os.path.join(log_dir, f'similarity_step_{current_training_step}.png')
                            plot_cosine_similarity_distribution(
                                similarity_data,
                                save_path=plot_path,
                                title=f"Cosine Similarity Distribution - Step {current_training_step}"
                            )
                            logger.info(f"Saved similarity plot to: {plot_path}")
                    else:
                        logger.debug(f"Skipping similarity computation: only {len(expression_tokens_by_clip)} clips available")
                        
                except Exception as e:
                    logger.warning(f"Failed to compute similarity distribution: {e}")
                    # Continue training without similarity analysis
            
            # Update progress bar with detailed loss breakdown
            # Loss: Current batch total loss
            # Avg Loss: Running average loss for the epoch
            # Cosine: Cosine similarity loss component
            # Temporal: Temporal contrastive loss component  
            # Diversity: Diversity loss component
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.5f}',
                'Avg Loss': f'{epoch_loss / num_batches:.5f}',
                'Cosine': f'{loss_components["cosine"].item():.5f}',
                'Temporal': f'{loss_components["temporal"].item():.5f}',
                'Diversity': f'{loss_components["diversity"].item():.5f}',
                'LR': f'{current_lr:.2e}',
                'λ_pred': f'{current_weights["lambda_prediction"]:.2f}',
                'λ_temp': f'{current_weights["lambda_temporal"]:.2f}',
                'λ_div': f'{current_weights["lambda_diversity"]:.2f}',
                'Step': f'{current_training_step}'
            })
            
            current_training_step += 1
        
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
            val_loss = validate_expression_prediction(
                joint_model, val_dataloader, criterion, dinov2_tokenizer, device
            )
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Log validation metrics to TensorBoard
            writer.add_scalar('Validation/Epoch_Loss', val_loss, epoch + 1)
        
        # Save epoch checkpoints
        if (epoch + 1) % save_every_epochs == 0:
            # Create comprehensive config
            config = create_comprehensive_config(
                expr_embed_dim=expr_embed_dim,
                expr_num_heads=expr_num_heads,
                expr_num_layers=expr_num_layers,
                expr_dropout=expr_dropout,
                expr_max_subjects=expr_max_subjects,
                decoder_embed_dim=decoder_embed_dim,
                decoder_num_heads=decoder_num_heads,
                decoder_num_layers=decoder_num_layers,
                decoder_dropout=decoder_dropout,
                max_sequence_length=max_sequence_length,
                lambda_prediction=initial_lambda_prediction,  # Use initial weight for config
                lambda_temporal=initial_lambda_temporal,
                lambda_diversity=initial_lambda_diversity,
                learning_rate=learning_rate,
                batch_size=batch_size,
                num_epochs=num_epochs,
                warmup_steps=warmup_steps,
                min_lr=min_lr,
                # Scheduler parameters
                initial_lambda_prediction=initial_lambda_prediction,
                initial_lambda_temporal=initial_lambda_temporal,
                initial_lambda_diversity=initial_lambda_diversity,
                warmup_lambda_prediction=warmup_lambda_prediction,
                warmup_lambda_temporal=warmup_lambda_temporal,
                warmup_lambda_diversity=warmup_lambda_diversity,
                final_lambda_prediction=final_lambda_prediction,
                final_lambda_temporal=final_lambda_temporal,
                final_lambda_diversity=final_lambda_diversity
            )
            
            # Joint model
            joint_epoch_path = os.path.join(checkpoint_dir, f"joint_expression_prediction_epoch_{epoch+1}.pt")
            save_checkpoint(
                model_state_dict=joint_model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict(),
                epoch=epoch + 1,
                avg_loss=avg_loss,
                total_steps=current_training_step,
                config=config,
                checkpoint_path=joint_epoch_path,
                checkpoint_type="joint"
            )
            
            # Expression transformer
            expr_epoch_path = os.path.join(checkpoint_dir, f"expression_transformer_epoch_{epoch+1}.pt")
            save_checkpoint(
                model_state_dict=joint_model.expression_transformer.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict(),
                epoch=epoch + 1,
                avg_loss=avg_loss,
                total_steps=current_training_step,
                config=config,
                checkpoint_path=expr_epoch_path,
                checkpoint_type="expression_transformer"
            )
            
            # Transformer decoder
            decoder_epoch_path = os.path.join(checkpoint_dir, f"transformer_decoder_epoch_{epoch+1}.pt")
            save_checkpoint(
                model_state_dict=joint_model.transformer_decoder.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict(),
                epoch=epoch + 1,
                avg_loss=avg_loss,
                total_steps=current_training_step,
                config=config,
                checkpoint_path=decoder_epoch_path,
                checkpoint_type="transformer_decoder"
            )
            
            logger.info(f"Saved epoch checkpoints: {joint_epoch_path}, {expr_epoch_path}, {decoder_epoch_path}")
    
    # Log model parameters to TensorBoard
    # Commented out due to NumPy 2.x compatibility issues
    # for name, param in joint_model.named_parameters():
    #     if param.requires_grad:
    #         writer.add_histogram(f'Parameters/{name}', param.data, 0)
    
    # Use robust parameter logging with fallback options
    try:
        logging_results = log_model_parameters(writer, joint_model, global_step=0)
        logger.info(f"Parameter logging completed: {logging_results['histograms_successful']} histograms, {logging_results['statistics_successful']} statistics")
    except Exception as e:
        logger.warning(f"Could not log model parameters: {e}")
    
    # Close TensorBoard writer
    writer.close()
    
    # Create final similarity analysis if we have collected data
    try:
        # This would be populated during training if we collect similarity history
        # For now, we'll create a placeholder for future implementation
        logger.info("Training completed!")
        logger.info(f"📊 TensorBoard logs saved to: {log_dir}")
        logger.info(f"📈 Similarity plots saved to: {log_dir}/similarity_step_*.png")
    except Exception as e:
        logger.warning(f"Could not create final similarity analysis: {e}")
    
    logger.info("Training completed!")
    logger.info(f"📊 TensorBoard logs saved to: {log_dir}")
    return joint_model


def validate_expression_prediction(
    joint_model,
    val_dataloader,
    criterion,
    dinov2_tokenizer,
    device="cpu"
):
    """
    Validate the joint expression prediction model
    """
    joint_model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():  # No gradient computation for validation
        for batch_idx, batch in enumerate(val_dataloader):
            # Prepare data
            face_images, subject_ids, clip_lengths = prepare_expression_training_data(
                batch, dinov2_tokenizer, device
            )
            
            # Forward pass through joint model
            all_expression_tokens, all_predicted_next_tokens = joint_model(face_images, subject_ids, dinov2_tokenizer, clip_lengths)
            
            # Prepare data for v2 loss computation
            predicted_next_tokens = []
            actual_next_tokens = []
            expression_tokens_by_clip = []
            
            for clip_idx, (clip_expression_tokens, clip_predicted_next_token) in enumerate(zip(all_expression_tokens, all_predicted_next_tokens)):
                if clip_expression_tokens.shape[0] > 1:
                    # Use the last token as the target
                    actual_next_token = clip_expression_tokens[-1:]  # (1, 1, 384)
                    predicted_next_tokens.append(clip_predicted_next_token)
                    actual_next_tokens.append(actual_next_token)
                    expression_tokens_by_clip.append(clip_expression_tokens)
            
            # Compute v2 loss if we have valid clips
            if predicted_next_tokens:
                loss, loss_components = criterion(predicted_next_tokens, actual_next_tokens, expression_tokens_by_clip)
            else:
                # No valid clips (all single frames)
                loss = torch.tensor(0.0, device=device)
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    
    logger.info(f"Validation - Avg Loss: {avg_loss:.4f}")
    
    return avg_loss


def test_joint_expression_prediction():
    """Test the joint expression prediction model"""
    import torch
    
    # Create models
    joint_model = JointExpressionPredictionModel(
        expr_embed_dim=384,
        expr_num_heads=4,
        expr_num_layers=1,
        expr_dropout=0.1,
        expr_max_subjects=3500,  # Added max_subjects
        decoder_embed_dim=384,
        decoder_num_heads=4,
        decoder_num_layers=1,
        decoder_dropout=0.1,
        max_sequence_length=50
    )
    dinov2_tokenizer = DINOv2Tokenizer(device="cpu") # Changed to "cpu" for testing
    
    # Create dummy input simulating clips
    # Simulate 2 clips: first clip has 5 frames, second clip has 3 frames
    clip1_frames = torch.randn(5, 3, 518, 518)  # 5 frames
    clip2_frames = torch.randn(3, 3, 518, 518)  # 3 frames
    
    # Concatenate all frames
    face_images = torch.cat([clip1_frames, clip2_frames], dim=0)  # (8, 3, 518, 518)
    clip_lengths = [5, 3]  # Length of each clip
    
    # Create subject IDs for each clip (simulating real subject IDs from dataset)
    subject_ids = []
    current_idx = 0
    
    for i, clip_length in enumerate(clip_lengths):
        # Simulate realistic subject IDs (e.g., 123, 456, etc.)
        subject_id = 100 + i  # Subject 100 for first clip, 101 for second clip
        clip_subject_ids = torch.full((clip_length,), subject_id, dtype=torch.long)
        subject_ids.append(clip_subject_ids)
        current_idx += clip_length
    
    # Concatenate all subject IDs
    subject_ids = torch.cat(subject_ids, dim=0)  # (8,)
    
    # Forward pass through joint model
    all_expression_tokens, all_predicted_next_tokens = joint_model(face_images, subject_ids, dinov2_tokenizer, clip_lengths)
    
    print(f"Input face images shape: {face_images.shape}")
    print(f"Subject IDs shape: {subject_ids.shape}")
    print(f"Clip lengths: {clip_lengths}")
    print(f"Number of expression token clips: {len(all_expression_tokens)}")
    print(f"Number of predicted next token clips: {len(all_predicted_next_tokens)}")
    
    # Check that subject IDs are correct within each clip
    print(f"Subject IDs for clip 1 (frames 0-4): {subject_ids[0:5]}")
    print(f"Subject IDs for clip 2 (frames 5-7): {subject_ids[5:8]}")
    
    # Verify subject IDs are the same within clips
    assert torch.all(subject_ids[0:5] == 100), "Subject IDs should be 100 for clip 1"
    assert torch.all(subject_ids[5:8] == 101), "Subject IDs should be 101 for clip 2"
    
    # Test v2 loss function
    criterion = ExpressionPredictionLoss_v2(lambda_prediction=1.0, lambda_temporal=0.1, lambda_diversity=0.1)
    
    # Prepare data for v2 loss computation
    predicted_next_tokens = []
    actual_next_tokens = []
    expression_tokens_by_clip = []
    
    for clip_expression_tokens, clip_predicted_next_token in zip(all_expression_tokens, all_predicted_next_tokens):
        if clip_expression_tokens.shape[0] > 1:
            actual_next_token = clip_expression_tokens[-1:]
            predicted_next_tokens.append(clip_predicted_next_token)
            actual_next_tokens.append(actual_next_token)
            expression_tokens_by_clip.append(clip_expression_tokens)
    
    if predicted_next_tokens:
        loss, loss_components = criterion(predicted_next_tokens, actual_next_tokens, expression_tokens_by_clip)
        print(f"V2 Loss: {loss.item():.4f}")
        print(f"  - Cosine Loss: {loss_components['cosine'].item():.4f}")
        print(f"  - Temporal Loss: {loss_components['temporal'].item():.4f}")
        print(f"  - Diversity Loss: {loss_components['diversity'].item():.4f}")
    else:
        print("No valid clips for loss calculation")
    
    print("✅ Joint expression prediction test passed!")


if __name__ == "__main__":
    test_joint_expression_prediction() 