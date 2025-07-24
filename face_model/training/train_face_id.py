#!/usr/bin/env python3
"""
Training script for Face ID Model (Component B)
Includes contrastive loss for better identity separation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import logging
import json
from tqdm import tqdm
import time
import uuid
from datetime import datetime

# Import our components
import sys
sys.path.append('.')

from models.dinov2_tokenizer import DINOv2Tokenizer
from models.face_id_model import FaceIDModel
from data.dataset import create_face_dataloader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def contrastive_loss_with_pairs(positive_pairs, negative_pairs, temperature=0.5):
    """
    Contrastive loss using explicit positive and negative pairs.
    """
    if len(positive_pairs) == 0 and len(negative_pairs) == 0:
        return torch.tensor(0.0, device=positive_pairs[0][0].device if positive_pairs else torch.device('cpu'), requires_grad=True)
    
    # Combine all pairs
    all_pairs = positive_pairs + negative_pairs
    labels = torch.cat([torch.ones(len(positive_pairs)), torch.zeros(len(negative_pairs))])
    
    # Create embeddings
    z_i = torch.stack([pair[0] for pair in all_pairs])
    z_j = torch.stack([pair[1] for pair in all_pairs])
    
    # Normalize
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    
    # Compute similarities
    similarities = torch.sum(z_i * z_j, dim=1) / temperature
    
    # Binary cross entropy loss
    loss = F.binary_cross_entropy_with_logits(similarities, labels)
    
    return loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for face ID tokens using NT-Xent loss
    Creates positive pairs from same subject and negative pairs from different subjects
    """
    
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, identity_tokens, subject_ids):
        """
        Args:
            identity_tokens: (B, 384) - Identity tokens for each clip
            subject_ids: (B,) - Subject IDs for each clip
        
        Returns:
            contrastive_loss: Scalar loss value
        """
        # Normalize identity tokens
        identity_tokens = F.normalize(identity_tokens, dim=1)
        
        # Convert subject_ids to tensor if it's a list
        if isinstance(subject_ids, list):
            # Convert string subject IDs to integers
            subject_ids = [int(sid) if isinstance(sid, str) else sid for sid in subject_ids]
            subject_ids = torch.tensor(subject_ids, device=identity_tokens.device)
        
        # Group tokens by subject ID
        unique_subjects = torch.unique(subject_ids)
        if len(unique_subjects) < 1:
            # Need at least 1 subject
            return torch.tensor(0.0, device=identity_tokens.device, requires_grad=True)
        
        # Create positive pairs (same subject) and negative pairs (different subjects)
        positive_pairs = []
        negative_pairs = []
        
        for subject_id in unique_subjects:
            # Get all tokens for this subject
            subject_mask = subject_ids == subject_id
            subject_tokens = identity_tokens[subject_mask]
            
            if len(subject_tokens) >= 2:
                # Create positive pairs from same subject (different frames)
                for i in range(len(subject_tokens)):
                    for j in range(i + 1, len(subject_tokens)):
                        positive_pairs.append((subject_tokens[i], subject_tokens[j]))
            
            # Create negative pairs with other subjects (if multiple subjects)
            other_subjects = unique_subjects[unique_subjects != subject_id]
            for other_subject_id in other_subjects:
                other_mask = subject_ids == other_subject_id
                other_tokens = identity_tokens[other_mask]
                
                for i in range(len(subject_tokens)):
                    for j in range(len(other_tokens)):
                        negative_pairs.append((subject_tokens[i], other_tokens[j]))
        
        # Sample pairs to avoid memory issues
        max_pairs = 50
        if len(positive_pairs) > max_pairs:
            positive_pairs = positive_pairs[:max_pairs]
        if len(negative_pairs) > max_pairs:
            negative_pairs = negative_pairs[:max_pairs]
        
        # Compute contrastive loss using both positive and negative pairs
        loss = contrastive_loss_with_pairs(positive_pairs, negative_pairs, self.temperature)
        return loss


def load_dataset_metadata(data_dir):
    """Load dataset metadata from JSON file"""
    metadata_file = os.path.join(data_dir, "dataset_metadata.json")
    
    if not os.path.exists(metadata_file):
        logger.warning(f"Dataset metadata not found: {metadata_file}")
        return None
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"Loaded dataset metadata:")
    logger.info(f"  Total subjects: {metadata['dataset_stats']['num_subjects']}")
    logger.info(f"  Total clips: {metadata['total_clips_extracted']}")
    logger.info(f"  Clips per video: {metadata['clips_per_video']}")
    
    return metadata


class FaceIDTrainer:
    """Trainer for Face ID Model with contrastive loss"""
    
    def __init__(self, config):
        self.config = config
        
        # Load dataset metadata
        self.dataset_metadata = load_dataset_metadata(config['training']['train_data_dir'])
        
        # Initialize models
        self.dinov2_tokenizer = DINOv2Tokenizer()
        
        # Load checkpoint if provided
        checkpoint_path = config['training'].get('checkpoint_path')
        if checkpoint_path is not None:
            logger.info(f"Loading Face ID Model from checkpoint: {checkpoint_path}")
            try:
                # Load checkpoint
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                
                # Get architecture from checkpoint config
                if 'config' in checkpoint and 'face_id_model' in checkpoint['config']:
                    face_id_config = checkpoint['config']['face_id_model']
                    logger.info(f"Using architecture from checkpoint: {face_id_config.get('num_layers', '?')} layers, {face_id_config.get('num_heads', '?')} heads")
                else:
                    # Fallback to config architecture
                    face_id_config = config['face_id_model']
                    logger.warning("No architecture config found in checkpoint, using config defaults")
                
                # Initialize model with checkpoint architecture
                self.face_id_model = FaceIDModel(**face_id_config)
                
                # Load state dict
                if 'face_id_model_state_dict' in checkpoint:
                    self.face_id_model.load_state_dict(checkpoint['face_id_model_state_dict'])
                    logger.info(f"Loaded Face ID Model from epoch {checkpoint.get('epoch', 'unknown')}")
                else:
                    # Try loading the entire checkpoint as state dict (for compatibility)
                    self.face_id_model.load_state_dict(checkpoint)
                    logger.info("Loaded Face ID Model state dict directly")
                
            except Exception as e:
                raise RuntimeError(f"Failed to load Face ID Model checkpoint: {str(e)}")
        else:
            # Train from scratch
            logger.info("Training Face ID Model from scratch")
            self.face_id_model = FaceIDModel(**config['face_id_model'])
        
        # Initialize loss functions
        self.consistency_loss = nn.MSELoss()  # For consistency within clips
        self.contrastive_loss = ContrastiveLoss(
            temperature=config['training'].get('contrastive_temperature', 0.5)
        )
        
        # Loss weights
        self.consistency_weight = config['training'].get('consistency_weight', 1.0)
        self.use_consistency_loss = config['training'].get('use_consistency_loss', True)
        self.contrastive_weight = config['training'].get('contrastive_weight', 1.0)
        
        # Log configuration
        logger.info(f"Training Configuration:")
        logger.info(f"  Consistency Loss: {'Enabled' if self.use_consistency_loss else 'Disabled'}")
        logger.info(f"  Contrastive Loss: First Frame Only")
        logger.info(f"  Loss Weights: Consistency={self.consistency_weight}, Contrastive={self.contrastive_weight}")
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.face_id_model.parameters(),
            lr=config['training']['learning_rate']
        )
        
        # Initialize TensorBoard with unique job ID
        job_id = str(uuid.uuid4())[:8]  # Short unique ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(config['training']['log_dir'], f'face_id_training_{job_id}_{timestamp}')
        self.writer = SummaryWriter(log_dir)
        
        logger.info(f"📊 TensorBoard logging to: {log_dir}")
        logger.info(f"🆔 Job ID: face_id_training_{job_id}")
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        
        # Track losses for plotting
        self.train_losses = []
        self.val_losses = []
        
        # Log dataset information
        if self.dataset_metadata:
            self._log_dataset_info()
        
        logger.info("Face ID Trainer initialized with contrastive loss")
    
    def _log_dataset_info(self):
        """Log dataset information to TensorBoard"""
        if not self.dataset_metadata:
            return
        
        stats = self.dataset_metadata['dataset_stats']
        
        # Log subject statistics
        for subject_id, subject_data in stats['subjects'].items():
            self.writer.add_scalar(f'Dataset/Subject_{subject_id}/clips', 
                                 subject_data['total_clips'], 0)
            
            if subject_data['label']:
                label = subject_data['label']
                self.writer.add_text(f'Dataset/Subject_{subject_id}/demographics',
                                   f"Age: {label.get('age', 'N/A')}, "
                                   f"Gender: {label.get('gender', 'N/A')}, "
                                   f"Skin-type: {label.get('skin-type', 'N/A')}", 0)
        
        # Log overall statistics
        self.writer.add_scalar('Dataset/total_subjects', stats['num_subjects'], 0)
        self.writer.add_scalar('Dataset/total_clips', self.dataset_metadata['total_clips_extracted'], 0)
    
    def train_epoch(self, dataloader):
        """Train for one epoch with contrastive loss"""
        self.face_id_model.train()
        
        epoch_loss = 0.0
        epoch_consistency_loss = 0.0
        epoch_contrastive_loss = 0.0
        num_batches = 0
        
        # Track loss per subject
        subject_losses = {}
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Get frames and subject IDs
            frames = batch['frames']  # (B, 30, 3, 518, 518)
            subject_ids = batch['subject_id']
            
            B, num_frames, C, H, W = frames.shape
            
            # Process each frame to get identity tokens
            identity_tokens = []
            
            for frame_idx in range(num_frames):
                # Get single frame
                frame = frames[:, frame_idx, :, :, :]  # (B, 3, 518, 518)
                
                # Get DINOv2 tokens
                with torch.no_grad():
                    patch_tokens, pos_emb = self.dinov2_tokenizer(frame)
                
                # Get identity token for this frame
                identity_token = self.face_id_model(patch_tokens, pos_emb)  # (B, 1, 384)
                identity_tokens.append(identity_token.squeeze(1))  # (B, 384)
            
            # Stack identity tokens across frames
            identity_tokens = torch.stack(identity_tokens, dim=1)  # (B, 30, 384)
            
            # Compute consistency loss (within clips)
            consistency_loss = self.face_id_model.get_identity_consistency_loss(identity_tokens)
            
            # Compute contrastive loss (between clips)
            # Use first frame identity token for contrastive loss
            clip_identity_tokens = identity_tokens[:, 0, :]  # (B, 384) - first frame only
            contrastive_loss = self.contrastive_loss(clip_identity_tokens, subject_ids)
            
            # Combined loss
            if hasattr(self, 'use_consistency_loss') and self.use_consistency_loss:
                total_loss = (self.consistency_weight * consistency_loss + 
                             self.contrastive_weight * contrastive_loss)
            else:
                # Use only contrastive loss
                total_loss = self.contrastive_weight * contrastive_loss
            
            # Track loss per subject
            for i, subject_id in enumerate(subject_ids):
                if subject_id not in subject_losses:
                    subject_losses[subject_id] = []
                subject_losses[subject_id].append(total_loss.item())
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += total_loss.item()
            epoch_consistency_loss += consistency_loss.item()
            epoch_contrastive_loss += contrastive_loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'total_loss': f'{total_loss.item():.4f}',
                'consistency': f'{consistency_loss.item():.4f}',
                'contrastive': f'{contrastive_loss.item():.6f}',  # More decimal places for small values
                'avg_loss': f'{epoch_loss / num_batches:.4f}'
            })
            
            # Log to TensorBoard
            self.writer.add_scalar('Loss/Total', total_loss.item(), self.global_step)
            self.writer.add_scalar('Loss/Consistency', consistency_loss.item(), self.global_step)
            self.writer.add_scalar('Loss/Contrastive', contrastive_loss.item(), self.global_step)
            self.writer.add_scalar('Loss/Average', epoch_loss / num_batches, self.global_step)
            
            # Log subject-specific losses
            for subject_id, losses in subject_losses.items():
                if len(losses) > 0:
                    avg_subject_loss = sum(losses) / len(losses)
                    self.writer.add_scalar(f'Loss/Subject_{subject_id}', avg_subject_loss, self.global_step)
            
            self.global_step += 1
        
        avg_loss = epoch_loss / num_batches
        avg_consistency_loss = epoch_consistency_loss / num_batches
        avg_contrastive_loss = epoch_contrastive_loss / num_batches
        
        logger.info(f"Epoch {self.epoch} - "
                   f"Total Loss: {avg_loss:.4f}, "
                   f"Consistency Loss: {avg_consistency_loss:.4f}, "
                   f"Contrastive Loss: {avg_contrastive_loss:.4f}")
        
        # Log subject-specific statistics
        for subject_id, losses in subject_losses.items():
            if len(losses) > 0:
                avg_subject_loss = sum(losses) / len(losses)
                logger.info(f"  Subject {subject_id}: {avg_subject_loss:.4f}")
        
        return avg_loss
    
    def _create_loss_vs_epoch_plot(self):
        """Create a custom plot showing train and validation loss vs epoch"""
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = list(range(len(self.train_losses)))
        
        if self.train_losses:
            ax.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        
        if self.val_losses:
            ax.plot(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss vs Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def validate_epoch(self, dataloader):
        """Validate for one epoch"""
        self.face_id_model.eval()
        
        epoch_loss = 0.0
        epoch_consistency_loss = 0.0
        epoch_contrastive_loss = 0.0
        num_batches = 0
        
        # Track loss per subject
        subject_losses = {}
        
        with torch.no_grad():  # No gradient computation for validation
            for batch_idx, batch in enumerate(dataloader):
                # Get frames and subject IDs
                frames = batch['frames']  # (B, 30, 3, 518, 518)
                subject_ids = batch['subject_id']
                
                B, num_frames, C, H, W = frames.shape
                
                # Process each frame to get identity tokens
                identity_tokens = []
                
                for frame_idx in range(num_frames):
                    # Get single frame
                    frame = frames[:, frame_idx, :, :, :]  # (B, 3, 518, 518)
                    
                    # Get DINOv2 tokens
                    patch_tokens, pos_emb = self.dinov2_tokenizer(frame)
                    
                    # Get identity token for this frame
                    identity_token = self.face_id_model(patch_tokens, pos_emb)  # (B, 1, 384)
                    identity_tokens.append(identity_token.squeeze(1))  # (B, 384)
                
                # Stack identity tokens across frames
                identity_tokens = torch.stack(identity_tokens, dim=1)  # (B, 30, 384)
                
                # Compute consistency loss (within clips)
                consistency_loss = self.face_id_model.get_identity_consistency_loss(identity_tokens)
                
                # Compute contrastive loss (between clips)
                # Use first frame identity token for contrastive loss
                clip_identity_tokens = identity_tokens[:, 0, :]  # (B, 384) - first frame only
                contrastive_loss = self.contrastive_loss(clip_identity_tokens, subject_ids)
                
                # Combined loss
                if hasattr(self, 'use_consistency_loss') and self.use_consistency_loss:
                    total_loss = (self.consistency_weight * consistency_loss + 
                                 self.contrastive_weight * contrastive_loss)
                else:
                    # Use only contrastive loss
                    total_loss = self.contrastive_weight * contrastive_loss
                
                # Track loss per subject
                for i, subject_id in enumerate(subject_ids):
                    if subject_id not in subject_losses:
                        subject_losses[subject_id] = []
                    subject_losses[subject_id].append(total_loss.item())
                
                # Update metrics
                epoch_loss += total_loss.item()
                epoch_consistency_loss += consistency_loss.item()
                epoch_contrastive_loss += contrastive_loss.item()
                num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        avg_consistency_loss = epoch_consistency_loss / num_batches
        avg_contrastive_loss = epoch_contrastive_loss / num_batches
        
        logger.info(f"Validation Epoch {self.epoch} - "
                   f"Total Loss: {avg_loss:.4f}, "
                   f"Consistency Loss: {avg_consistency_loss:.4f}, "
                   f"Contrastive Loss: {avg_contrastive_loss:.4f}")
        
        # Log validation metrics to TensorBoard
        self.writer.add_scalar('Validation/Loss', avg_loss, self.epoch)
        self.writer.add_scalar('Validation/Consistency', avg_consistency_loss, self.epoch)
        self.writer.add_scalar('Validation/Contrastive', avg_contrastive_loss, self.epoch)
        
        return avg_loss
    
    def train(self, train_dataloader, val_dataloader=None, num_epochs=10):
        """Train the model with optional validation"""
        logger.info(f"Starting training for {num_epochs} epochs")
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train one epoch
            train_loss = self.train_epoch(train_dataloader)
            
            # Track training loss
            self.train_losses.append(train_loss)
            
            # Log training epoch metrics
            self.writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
            
            # Validate if validation dataloader is provided
            if val_dataloader is not None:
                val_loss = self.validate_epoch(val_dataloader)
                
                # Track validation loss
                self.val_losses.append(val_loss)
                
                # Log validation epoch metrics
                self.writer.add_scalar('Epoch/Val_Loss', val_loss, epoch)
                
                # Track best validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    logger.info(f"New best validation loss: {best_val_loss:.4f}")
                
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}")
            
            # Create and log loss vs epoch plot
            if len(self.train_losses) > 0:
                fig = self._create_loss_vs_epoch_plot()
                self.writer.add_figure('Loss/Loss_vs_Epoch', 
                                     fig, 
                                     global_step=epoch,
                                     close=True)
            
            # Save checkpoint
            if (epoch + 1) % self.config['training']['save_every'] == 0:
                self.save_checkpoint(epoch)
        
        self.writer.close()
        logger.info("Training completed!")
    
    def save_checkpoint(self, epoch):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'face_id_model_state_dict': self.face_id_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'dataset_metadata': self.dataset_metadata
        }
        
        checkpoint_dir = self.config['training']['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, f'face_id_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")


def main():
    """Main training function"""
    
    # Configuration
    config = {
        'face_id_model': {
            'embed_dim': 384,
            'num_heads': 4,  # Reduced from 8 to 4 for optimized architecture
            'num_layers': 1,  # Reduced from 2 to 1 for optimized architecture
            'dropout': 0.1
        },
        'training': {
            'learning_rate': 1e-4,
            'batch_size': 4,  # Increased to ensure multiple subjects per batch
            'num_epochs': 10,
            'save_every': 2,
            'log_dir': './logs',
            'checkpoint_dir': './checkpoints',
            'train_data_dir': '/Users/ozgewhiting/Documents/EQLabs/datasets_serial/CCA_train_db1',  # Training data directory
            'val_data_dir': None,  # Validation data directory (optional)
            'max_train_samples': 20,  # For debugging, remove for full training
            'max_val_samples': 10,  # For debugging, remove for full training
            'contrastive_temperature': 0.1,
            'contrastive_margin': 1.0,
            'consistency_weight': 1.0,
            'contrastive_weight': 1.0
        }
    }
    
    # Create training dataloader
    train_dataloader = create_face_dataloader(
        data_dir=config['training']['train_data_dir'],
        batch_size=config['training']['batch_size'],
        max_samples=config['training']['max_train_samples']
    )
    
    # Create validation dataloader if validation data is provided
    val_dataloader = None
    if config['training']['val_data_dir'] is not None:
        val_dataloader = create_face_dataloader(
            data_dir=config['training']['val_data_dir'],
            batch_size=config['training']['batch_size'],
            max_samples=config['training']['max_val_samples']
        )
        logger.info(f"Validation dataloader created with {len(val_dataloader)} batches")
    
    # Create trainer
    trainer = FaceIDTrainer(config)
    
    # Start training
    trainer.train(train_dataloader, val_dataloader, config['training']['num_epochs'])


if __name__ == "__main__":
    main() 