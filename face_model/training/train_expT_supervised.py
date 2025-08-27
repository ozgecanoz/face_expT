#!/usr/bin/env python3
"""
Supervised Training of Expression Transformer for Emotion Classification
Uses AffectNet dataset to train expression transformer with emotion classification head
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

from data.affectnet_dataset import AffectNetDataset
from models.expression_transformer import ExpressionTransformer
from models.dinov2_tokenizer import DINOv2BaseTokenizer
from utils.checkpoint_utils import save_checkpoint, create_comprehensive_config, load_checkpoint_config, extract_model_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExpTClassifierModel(nn.Module):
    """
    Expression Transformer with shallow classification head for emotion classification.
    The heavy lifting is done by the expression transformer, with just a simple
    linear projection to the output classes.
    """
    
    def __init__(self, 
                 embed_dim=384, 
                 num_heads=4, 
                 num_layers=2, 
                 dropout=0.1, 
                 ff_dim=1536, 
                 grid_size=37,
                 num_classes=8):
        super().__init__()
        
        # Expression Transformer (subject-invariant)
        self.expression_transformer = ExpressionTransformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            ff_dim=ff_dim,
            grid_size=grid_size
        )
        
        # Shallow classification head - let expression transformer do the heavy lifting
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, num_classes)
        )
        
    def forward(self, patch_tokens):
        """
        Args:
            patch_tokens: (batch_size, num_patches, embed_dim) - PCA projected DINOv2 features
            
        Returns:
            expression_tokens: (batch_size, 1, embed_dim) - Expression tokens
            logits: (batch_size, num_classes) - Emotion classification logits
        """
        # Extract expression tokens using Expression Transformer
        # This is where the heavy lifting happens - learning expressive representations
        expression_tokens = self.expression_transformer(patch_tokens)  # (batch_size, 1, embed_dim)
        
        # Extract the expression token (remove the sequence dimension)
        expression_token = expression_tokens.squeeze(1)  # (batch_size, embed_dim)
        
        # Simple linear projection to output classes
        # The expression transformer has already learned the discriminative features
        logits = self.classifier(expression_token)  # (batch_size, num_classes)
        
        return expression_tokens, logits


class EmotionClassificationLoss(nn.Module):
    """
    Cross-entropy loss for emotion classification
    """
    
    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (batch_size, num_classes) - Model predictions
            targets: (batch_size,) - Ground truth emotion class IDs
            
        Returns:
            loss: Scalar loss value
        """
        return self.cross_entropy(logits, targets)


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


def prepare_emotion_classification_data(batch, dinov2_tokenizer, pca_components, pca_mean, device):
    """
    Prepare data for emotion classification training
    
    Args:
        batch: Batch from FaceDataset
        dinov2_tokenizer: DINOv2BaseTokenizer instance
        pca_components: PCA projection matrix
        pca_mean: PCA mean vector
        device: Device to move data to
        
    Returns:
        face_images: (total_frames, 3, 518, 518) - Face images
        emotion_targets: (total_frames,) - Emotion class targets
        clip_lengths: List of clip lengths
    """
    # Extract data from batch
    frames = batch['frames']  # List of (T, 3, 518, 518) tensors
    emotion_class_ids = batch['emotion_class_ids']  # List of (T,) tensors
    
    # Process each clip
    all_face_images = []
    all_emotion_targets = []
    clip_lengths = []
    
    for clip_idx, (clip_frames, clip_emotions) in enumerate(zip(frames, emotion_class_ids)):
        # Move frames to device before processing
        clip_frames = clip_frames.to(device)
        
        # Get DINOv2 patch tokens
        patch_tokens, _ = dinov2_tokenizer(clip_frames)  # (T, 1369, 768)
        
        # Project to PCA space
        patch_tokens_np = patch_tokens.cpu().numpy()  # (T, 1369, 768)
        batch_size, num_patches, embed_dim = patch_tokens_np.shape
        
        # Reshape for PCA projection
        patch_tokens_reshaped = patch_tokens_np.reshape(-1, embed_dim)  # (T*1369, 768)
        
        # Center the data
        patch_tokens_centered = patch_tokens_reshaped - pca_mean
        
        # Project to PCA space
        projected_features = np.dot(patch_tokens_centered, pca_components.T)  # (T*1369, 384)
        
        # Reshape back to (T, 1369, 384)
        projected_features = projected_features.reshape(batch_size, num_patches, -1)
        
        # Convert back to tensor
        projected_features = torch.from_numpy(projected_features).float().to(device)
        
        # Store data
        all_face_images.append(projected_features)
        all_emotion_targets.append(clip_emotions)
        clip_lengths.append(clip_frames.shape[0])
    
    # Concatenate all clips
    face_images = torch.cat(all_face_images, dim=0)  # (total_frames, 1369, 384)
    emotion_targets = torch.cat(all_emotion_targets, dim=0)  # (total_frames,)
    
    # Move emotion targets to device
    emotion_targets = emotion_targets.to(device)
    
    return face_images, emotion_targets, clip_lengths


def train_expression_transformer_supervised(
    dataset_path,
    pca_json_path,
    expression_transformer_checkpoint_path=None,
    checkpoint_dir="checkpoints",
    save_every_step=500,
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
    # Architecture configuration parameters
    embed_dim=384,
    num_heads=4,
    num_layers=2,
    dropout=0.1,
    ff_dim=1536,
    grid_size=37,
    num_classes=8,
    log_dir=None,
    # Memory management
    max_memory_fraction=0.9
):
    """
    Train the expression transformer for supervised emotion classification
    
    Args:
        dataset_path: Path to the AffectNet dataset
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
        device: Device to use for training
        num_workers: Number of data loader workers
        pin_memory: Whether to pin memory
        persistent_workers: Whether to use persistent workers
        drop_last: Whether to drop last incomplete batch
        warmup_steps: Number of warmup steps for scheduler
        min_lr: Minimum learning rate
        embed_dim: Expression transformer embedding dimension
        num_heads: Expression transformer number of heads
        num_layers: Expression transformer number of layers
        dropout: Expression transformer dropout
        ff_dim: Expression transformer feed-forward dimension
        grid_size: Grid size for positional embeddings
        num_classes: Number of emotion classes
        log_dir: Directory for TensorBoard logs
        max_memory_fraction: Maximum GPU memory fraction to use (0.0-1.0, default: 0.9)
    """
    logger.info(f"Starting supervised expression transformer training on device: {device}")
    
    # CUDA memory optimization settings
    if device.type == "cuda":
        # Set memory fraction limit
        torch.cuda.set_per_process_memory_fraction(max_memory_fraction)
        logger.info(f"GPU Memory fraction limit set to: {max_memory_fraction}")
        
        # Clear cache
        torch.cuda.empty_cache()
        logger.info(f"Initial GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
        logger.info(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1024**3:.1f} GB")
    
    # Load PCA projection
    pca_components, pca_mean = load_pca_projection(pca_json_path)
    
    # Initialize DINOv2 tokenizer
    dinov2_tokenizer = DINOv2BaseTokenizer(device=device)
    logger.info("✅ DINOv2 base tokenizer initialized")
    
    # Load datasets
    train_dataset = AffectNetDataset(dataset_path, max_samples=max_samples)
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
        val_dataset = AffectNetDataset(val_dataset_path, max_samples=max_val_samples)
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
    
    # Log initial configuration
    logger.info(f"Initial model configuration:")
    logger.info(f"  - embed_dim: {embed_dim}")
    logger.info(f"  - num_heads: {num_heads}")
    logger.info(f"  - num_layers: {num_layers}")
    logger.info(f"  - dropout: {dropout}")
    logger.info(f"  - ff_dim: {ff_dim}")
    logger.info(f"  - grid_size: {grid_size}")
    logger.info(f"  - num_classes: {num_classes}")
    
    # Initialize model with config from checkpoint if provided
    if expression_transformer_checkpoint_path and os.path.exists(expression_transformer_checkpoint_path):
        logger.info(f"Loading expression transformer checkpoint: {expression_transformer_checkpoint_path}")
        checkpoint = torch.load(expression_transformer_checkpoint_path, map_location=device)
        
        # Extract config from checkpoint
        if 'config' in checkpoint:
            checkpoint_config = checkpoint['config']
            logger.info("✅ Found config in checkpoint, initializing model accordingly")
            
            # Initialize model with checkpoint config
            model = ExpTClassifierModel(
                embed_dim=checkpoint_config.get('expression_model', {}).get('expr_embed_dim', embed_dim),
                num_heads=checkpoint_config.get('expression_model', {}).get('expr_num_heads', num_heads),
                num_layers=checkpoint_config.get('expression_model', {}).get('expr_num_layers', num_layers),
                dropout=checkpoint_config.get('expression_model', {}).get('expr_dropout', dropout),
                ff_dim=checkpoint_config.get('expression_model', {}).get('expr_ff_dim', ff_dim),
                grid_size=checkpoint_config.get('expression_model', {}).get('expr_grid_size', grid_size),
                num_classes=checkpoint_config.get('supervised_model', {}).get('num_classes', num_classes)  # Load from checkpoint config
            ).to(device)
            
            # Load model weights - handle both supervised and standalone checkpoints
            if 'model_state_dict' in checkpoint:
                # This is a supervised model checkpoint
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("✅ Supervised model checkpoint loaded with matching config")
            elif 'expression_transformer_state_dict' in checkpoint:
                # This is a standalone ExpressionTransformer checkpoint
                model.load_state_dict(checkpoint['expression_transformer_state_dict'])
                logger.info("✅ Expression transformer checkpoint loaded with matching config")
            else:
                raise ValueError("Checkpoint must contain either 'model_state_dict' or 'expression_transformer_state_dict'")
            
            # Update local variables to match checkpoint config
            embed_dim = checkpoint_config.get('expression_model', {}).get('expr_embed_dim', embed_dim)
            num_heads = checkpoint_config.get('expression_model', {}).get('expr_num_heads', num_heads)
            num_layers = checkpoint_config.get('expression_model', {}).get('expr_num_layers', num_layers)
            dropout = checkpoint_config.get('expression_model', {}).get('expr_dropout', dropout)
            ff_dim = checkpoint_config.get('expression_model', {}).get('expr_ff_dim', ff_dim)
            grid_size = checkpoint_config.get('expression_model', {}).get('expr_grid_size', grid_size)
            num_classes = checkpoint_config.get('supervised_model', {}).get('num_classes', num_classes)
            
            logger.info(f"Model config from checkpoint:")
            logger.info(f"  - embed_dim: {embed_dim}")
            logger.info(f"  - num_heads: {num_heads}")
            logger.info(f"  - num_layers: {num_layers}")
            logger.info(f"  - dropout: {dropout}")
            logger.info(f"  - ff_dim: {ff_dim}")
            logger.info(f"  - grid_size: {grid_size}")
            logger.info(f"  - num_classes: {num_classes}")
            
        else:
            logger.warning("⚠️  No config found in checkpoint, using provided arguments")
            # Initialize model with provided arguments
            model = ExpTClassifierModel(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
                ff_dim=ff_dim,
                grid_size=grid_size,
                num_classes=num_classes
            ).to(device)
            
            # Load model weights
            model.load_state_dict(checkpoint['expression_transformer_state_dict'])
            logger.info("✅ Expression transformer checkpoint loaded")
            logger.info(f"⚠️  Note: num_classes not in checkpoint, using provided value: {num_classes}")
    else:
        # Initialize model with provided arguments (no checkpoint)
        logger.info("No checkpoint provided, initializing model with provided arguments")
        model = ExpTClassifierModel(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            ff_dim=ff_dim,
            grid_size=grid_size,
            num_classes=num_classes
        ).to(device)
    
    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Cosine annealing scheduler with warmup
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=warmup_steps, 
        T_mult=2, 
        eta_min=min_lr
    )
    
    # Initialize loss function
    criterion = EmotionClassificationLoss()
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Setup TensorBoard logging
    if log_dir is None:
        log_dir = f"logs/expT_supervised_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # Training loop
    current_training_step = 0
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Prepare data
                face_images, emotion_targets, clip_lengths = prepare_emotion_classification_data(
                    batch, dinov2_tokenizer, pca_components, pca_mean, device
                )
                
                # Forward pass
                expression_tokens, logits = model(face_images)
                
                # Compute loss
                loss = criterion(logits, emotion_targets)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update learning rate
                scheduler.step()
                
                # Update metrics
                epoch_loss += loss.item()
                num_batches += 1
                current_training_step += 1
                
                # Log to TensorBoard
                writer.add_scalar('Training/Step_Loss', loss.item(), current_training_step)
                writer.add_scalar('Training/Learning_Rate', scheduler.get_last_lr()[0], current_training_step)
                
                # Debug: Log step progress
                if current_training_step % 10 == 0:  # Log every 10 steps
                    logger.info(f"📊 Step {current_training_step}: Loss={loss.item():.5f}, LR={scheduler.get_last_lr()[0]:.2e}")
                    print(f"📊 Step {current_training_step}: Loss={loss.item():.5f}, LR={scheduler.get_last_lr()[0]:.2e}")
                
                # Save checkpoint every N steps
                if current_training_step % save_every_step == 0:
                    logger.info(f"🔄 Saving checkpoint at step {current_training_step}")
                    checkpoint_path = os.path.join(checkpoint_dir, f"expT_supervised_step_{current_training_step}.pt")
                    
                    # Create config (use updated values from checkpoint if available)
                    config = create_comprehensive_config(
                        expr_embed_dim=embed_dim,
                        expr_num_heads=num_heads,
                        expr_num_layers=num_layers,
                        expr_dropout=dropout,
                        expr_ff_dim=ff_dim,
                        expr_grid_size=grid_size,
                        num_classes=num_classes,
                        learning_rate=learning_rate,
                        batch_size=batch_size,
                        num_epochs=num_epochs,
                        warmup_steps=warmup_steps,
                        min_lr=min_lr,
                        pca_json_path=pca_json_path
                    )
                    
                    try:
                        save_checkpoint(
                            model_state_dict=model.state_dict(),
                            optimizer_state_dict=optimizer.state_dict(),
                            scheduler_state_dict=scheduler.state_dict(),
                            epoch=epoch + 1,
                            avg_loss=loss.item(),
                            total_steps=current_training_step,
                            config=config,
                            checkpoint_path=checkpoint_path,
                            checkpoint_type="expression_transformer"
                        )
                        
                        logger.info(f"✅ Saved checkpoint: {checkpoint_path}")
                        print(f"💾 Checkpoint saved: {checkpoint_path}")
                        
                    except Exception as e:
                        logger.error(f"❌ Failed to save checkpoint: {e}")
                        print(f"❌ Checkpoint save failed: {e}")
                
                # Also save a checkpoint every epoch for safety
                if batch_idx == len(train_dataloader) - 1:  # Last batch of epoch
                    logger.info(f"🔄 Saving epoch checkpoint at step {current_training_step}")
                    epoch_checkpoint_path = os.path.join(checkpoint_dir, f"expT_supervised_epoch_{epoch+1}_step_{current_training_step}.pt")
                    
                    # Create config for epoch checkpoint if not already created
                    if 'config' not in locals():
                        config = create_comprehensive_config(
                            expr_embed_dim=embed_dim,
                            expr_num_heads=num_heads,
                            expr_num_layers=num_layers,
                            expr_dropout=dropout,
                            expr_ff_dim=ff_dim,
                            expr_grid_size=grid_size,
                            num_classes=num_classes,
                            learning_rate=learning_rate,
                            batch_size=batch_size,
                            num_epochs=num_epochs,
                            warmup_steps=warmup_steps,
                            min_lr=min_lr,
                            pca_json_path=pca_json_path
                        )
                    
                    try:
                        save_checkpoint(
                            model_state_dict=model.state_dict(),
                            optimizer_state_dict=optimizer.state_dict(),
                            scheduler_state_dict=scheduler.state_dict(),
                            epoch=epoch + 1,
                            avg_loss=epoch_loss / num_batches,
                            total_steps=current_training_step,
                            config=config,
                            checkpoint_path=epoch_checkpoint_path,
                            checkpoint_type="expression_transformer"
                        )
                        
                        logger.info(f"✅ Saved epoch checkpoint: {epoch_checkpoint_path}")
                        print(f"💾 Epoch checkpoint saved: {epoch_checkpoint_path}")
                        
                    except Exception as e:
                        logger.error(f"❌ Failed to save epoch checkpoint: {e}")
                        print(f"❌ Epoch checkpoint save failed: {e}")
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.5f}',
                    'Avg Loss': f'{epoch_loss / num_batches:.5f}',
                    'LR': f'{scheduler.get_last_lr()[0]:.2e}',
                    'Step': f'{current_training_step}',
                    'Next Checkpoint': f'Step {((current_training_step // save_every_step) + 1) * save_every_step}'
                })
                
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
            val_loss = validate_expression_transformer_supervised(
                model, val_dataloader, criterion, dinov2_tokenizer, pca_components, pca_mean, device
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
    return model


def validate_expression_transformer_supervised(
    model,
    val_dataloader,
    criterion,
    dinov2_tokenizer,
    pca_components,
    pca_mean,
    device="cpu"
):
    """
    Validate the expression transformer supervised model
    """
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            try:
                # Prepare data
                face_images, emotion_targets, clip_lengths = prepare_emotion_classification_data(
                    batch, dinov2_tokenizer, pca_components, pca_mean, device
                )
                
                # Forward pass
                expression_tokens, logits = model(face_images)
                
                # Compute loss
                loss = criterion(logits, emotion_targets)
                
                # Update metrics
                total_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                logger.error(f"Error processing validation batch {batch_idx}: {e}")
                continue
    
    avg_loss = total_loss / num_batches
    logger.info(f"Validation - Avg Loss: {avg_loss:.4f}")
    
    return avg_loss


def test_expression_transformer_supervised():
    """Test the expression transformer supervised model"""
    print("🧪 Testing Expression Transformer Supervised Model...")
    
    # Create model
    model = ExpTClassifierModel()
    print("✅ Model created successfully")
    
    # Test with dummy input
    batch_size = 2
    num_patches = 1369
    embed_dim = 384
    
    # Create dummy data
    patch_tokens = torch.randn(batch_size, num_patches, embed_dim)
    
    # Forward pass
    expression_tokens, logits = model(patch_tokens)
    
    print(f"Expression tokens shape: {expression_tokens.shape}")
    print(f"Logits shape: {logits.shape}")
    
    print("✅ Expression transformer supervised test passed!")


if __name__ == "__main__":
    test_expression_transformer_supervised()
