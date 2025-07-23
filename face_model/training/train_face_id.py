#!/usr/bin/env python3
"""
Training script for Face ID Model (Component B)
Includes contrastive loss for better identity separation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import logging
import json
from tqdm import tqdm
import time

# Import our components
import sys
sys.path.append('.')

from models.dinov2_tokenizer import DINOv2Tokenizer
from models.face_id_model import FaceIDModel
from data.dataset import create_face_dataloader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for face ID tokens
    Pushes different subjects apart and pulls same subjects together
    """
    
    def __init__(self, temperature=0.1, margin=1.0):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        
    def forward(self, identity_tokens, subject_ids):
        """
        Args:
            identity_tokens: (B, 384) - Identity tokens for each clip
            subject_ids: (B,) - Subject IDs for each clip
        
        Returns:
            contrastive_loss: Scalar loss value
        """
        # Normalize identity tokens
        identity_tokens = nn.functional.normalize(identity_tokens, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(identity_tokens, identity_tokens.T) / self.temperature
        
        # Create labels for positive/negative pairs
        B = len(subject_ids)
        labels = torch.zeros(B, B, device=identity_tokens.device)
        
        for i in range(B):
            for j in range(B):
                if i != j and subject_ids[i] == subject_ids[j]:
                    labels[i, j] = 1  # Positive pair (same subject)
        
        # Compute contrastive loss using InfoNCE formulation
        positive_mask = labels == 1
        negative_mask = labels == 0
        
        # Positive loss (pull same subjects together)
        positive_loss = 0
        if positive_mask.sum() > 0:
            positive_similarities = similarity_matrix[positive_mask]
            # Use log-sum-exp trick for numerical stability
            positive_loss = -positive_similarities.mean()
        
        # Negative loss (push different subjects apart)
        negative_loss = 0
        if negative_mask.sum() > 0:
            negative_similarities = similarity_matrix[negative_mask]
            # Use margin-based loss to push different subjects apart
            negative_loss = torch.clamp(self.margin - negative_similarities, min=0).mean()
        
        # Scale the loss to be in a reasonable range (similar to consistency loss)
        total_loss = (positive_loss + negative_loss) * 0.01  # Scale factor
        
        return total_loss


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
        self.face_id_model = FaceIDModel(**config['face_id_model'])
        
        # Initialize loss functions
        self.consistency_loss = nn.MSELoss()  # For consistency within clips
        self.contrastive_loss = ContrastiveLoss(
            temperature=config['training'].get('contrastive_temperature', 0.1),
            margin=config['training'].get('contrastive_margin', 1.0)
        )
        
        # Loss weights
        self.consistency_weight = config['training'].get('consistency_weight', 1.0)
        self.contrastive_weight = config['training'].get('contrastive_weight', 1.0)
        self.use_consistency_loss = config['training'].get('use_consistency_loss', True)
        
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
        
        # Initialize TensorBoard
        log_dir = os.path.join(config['training']['log_dir'], 'face_id_training')
        self.writer = SummaryWriter(log_dir)
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        
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
                'contrastive': f'{contrastive_loss.item():.4f}',
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
            
            # Log training epoch metrics
            self.writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
            
            # Validate if validation dataloader is provided
            if val_dataloader is not None:
                val_loss = self.validate_epoch(val_dataloader)
                
                # Log validation epoch metrics
                self.writer.add_scalar('Epoch/Val_Loss', val_loss, epoch)
                
                # Track best validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    logger.info(f"New best validation loss: {best_val_loss:.4f}")
                
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}")
            
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
            'train_data_dir': '../test_output',  # Training data directory
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