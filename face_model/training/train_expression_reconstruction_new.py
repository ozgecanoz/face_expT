"""
Training Script for Expression Reconstruction Model (New)
Trains only the expression reconstruction model with:
- Frozen expression transformer from checkpoint
- Cached DINOv2 positional embeddings
- Clip-by-clip processing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import gc
import numpy as np

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dinov2_tokenizer import DINOv2Tokenizer
from models.expression_transformer import ExpressionTransformer
from models.expression_reconstruction_model import ExpressionReconstructionModel
from data.dataset import FaceDataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExpressionReconstructionTrainer:
    """
    Trainer for expression reconstruction model with frozen components
    """
    
    def __init__(self, 
                 expression_transformer_checkpoint_path: str,
                 device = "cpu",
                 embed_dim: int = 384,
                 num_cross_attention_layers: int = 2,
                 num_self_attention_layers: int = 2,
                 num_heads: int = 8,
                 ff_dim: int = 1536,
                 dropout: float = 0.1,
                 reconstruction_model_checkpoint_path: str = None):
        """
        Initialize the trainer
        
        Args:
            expression_transformer_checkpoint_path: Path to frozen expression transformer checkpoint
            device: Device to use for training
            embed_dim: Embedding dimension
            num_cross_attention_layers: Number of cross-attention layers
            num_self_attention_layers: Number of self-attention layers
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            dropout: Dropout rate
            reconstruction_model_checkpoint_path: Path to reconstruction model checkpoint (optional)
        """
        # Handle device parameter (can be string or torch.device)
        self.device = str(device) if hasattr(device, 'type') else device
        self.embed_dim = embed_dim
        
        # Load DINOv2 tokenizer first
        logger.info("Loading DINOv2 tokenizer...")
        # Use the original device object for DINOv2 tokenizer
        device_obj = device if hasattr(device, 'type') else torch.device(device)
        self.dinov2_tokenizer = DINOv2Tokenizer(device=device_obj)
        
        # Cache positional embeddings (they're the same for all images)
        dummy_image = torch.randn(1, 3, 518, 518).to(device_obj)
        _, pos_embeddings = self.dinov2_tokenizer(dummy_image)
        self.cached_pos_embeddings = pos_embeddings.to(device_obj)  # (1, 1369, 384)
        logger.info(f"✅ Cached positional embeddings shape: {self.cached_pos_embeddings.shape}")
        
        # Load expression transformer checkpoint to get architecture
        logger.info(f"Loading expression transformer checkpoint to get architecture: {expression_transformer_checkpoint_path}")
        checkpoint = torch.load(expression_transformer_checkpoint_path, map_location=device_obj)
        
        # Extract architecture from checkpoint
        if 'config' in checkpoint:
            config = checkpoint['config']
            embed_dim = config.get('embed_dim', embed_dim)
            num_heads = config.get('num_heads', num_heads)
            num_layers = config.get('num_layers', 2)
            dropout = config.get('dropout', dropout)
            max_subjects = config.get('max_subjects', 3500)
        else:
            # Default values if config not found
            num_layers = 2
            max_subjects = 3500
        
        logger.info(f"Expression transformer architecture: embed_dim={embed_dim}, num_heads={num_heads}, num_layers={num_layers}, max_subjects={max_subjects}")
        
        # Initialize expression transformer with correct architecture
        self.expression_transformer = ExpressionTransformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            max_subjects=max_subjects
        )
        
        # Load state dict
        self.expression_transformer.load_state_dict(checkpoint['model_state_dict'])
        
        # Freeze expression transformer
        for param in self.expression_transformer.parameters():
            param.requires_grad = False
        self.expression_transformer.eval()
        
        logger.info("✅ Expression transformer loaded and frozen")
        
        # Create or load expression reconstruction model
        if reconstruction_model_checkpoint_path is not None:
            logger.info(f"Loading expression reconstruction model from: {reconstruction_model_checkpoint_path}")
            recon_checkpoint = torch.load(reconstruction_model_checkpoint_path, map_location=device_obj)
            
            # Extract architecture from reconstruction checkpoint
            if 'config' in recon_checkpoint:
                recon_config = recon_checkpoint['config']
                embed_dim = recon_config.get('embed_dim', embed_dim)
                num_cross_attention_layers = recon_config.get('num_cross_attention_layers', num_cross_attention_layers)
                num_self_attention_layers = recon_config.get('num_self_attention_layers', num_self_attention_layers)
                num_heads = recon_config.get('num_heads', num_heads)
                ff_dim = recon_config.get('ff_dim', ff_dim)
                dropout = recon_config.get('dropout', dropout)
            
            self.reconstruction_model = ExpressionReconstructionModel(
                embed_dim=embed_dim,
                num_cross_attention_layers=num_cross_attention_layers,
                num_self_attention_layers=num_self_attention_layers,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout
            ).to(device_obj)
            
            self.reconstruction_model.load_state_dict(recon_checkpoint['model_state_dict'])
            logger.info("✅ Expression reconstruction model loaded from checkpoint")
        else:
            logger.info("Creating new expression reconstruction model from scratch")
            self.reconstruction_model = ExpressionReconstructionModel(
                embed_dim=embed_dim,
                num_cross_attention_layers=num_cross_attention_layers,
                num_self_attention_layers=num_self_attention_layers,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout
            ).to(device_obj)
            logger.info("✅ Expression reconstruction model created from scratch")
        
        # Move models to device
        self.expression_transformer = self.expression_transformer.to(device_obj)
        
        # Print model info
        total_params = sum(p.numel() for p in self.reconstruction_model.parameters())
        trainable_params = sum(p.numel() for p in self.reconstruction_model.parameters() if p.requires_grad)
        logger.info(f"📊 Model parameters:")
        logger.info(f"   Total parameters: {total_params:,}")
        logger.info(f"   Trainable parameters: {trainable_params:,}")
        logger.info(f"   Learnable patch embeddings: {self.reconstruction_model.patch_embeddings.numel():,}")
    
    def process_clip(self, face_images: torch.Tensor, subject_ids: torch.Tensor) -> tuple:
        """
        Process a single clip to get expression tokens and subject embeddings
        
        Args:
            face_images: (num_frames, 3, 518, 518) - Face images from a clip
            subject_ids: (num_frames,) - Subject IDs for each frame
            
        Returns:
            tuple: (expression_tokens, subject_embeddings)
                - expression_tokens: (num_frames, 1, 384)
                - subject_embeddings: (num_frames, 1, 384)
        """
        num_frames = face_images.shape[0]
        
        # Get patch tokens and positional embeddings using DINOv2 tokenizer
        patch_tokens, pos_embeddings = self.dinov2_tokenizer(face_images)  # (num_frames, 1369, 384), (num_frames, 1369, 384)
        
        # Use expression transformer's inference method
        # Always use no_grad for expression transformer since it's frozen
        with torch.no_grad():
            expression_tokens, subject_embeddings = self.expression_transformer.inference(
                patch_tokens, pos_embeddings, subject_ids
            )
        
        return expression_tokens, subject_embeddings
    
    def reconstruct_faces(self, expression_tokens: torch.Tensor, subject_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct faces using expression tokens and subject embeddings
        
        Args:
            expression_tokens: (num_frames, 1, 384) - Expression tokens
            subject_embeddings: (num_frames, 1, 384) - Subject embeddings
            
        Returns:
            torch.Tensor: (num_frames, 3, 518, 518) - Reconstructed faces
        """
        num_frames = expression_tokens.shape[0]
        
        # Expand cached positional embeddings to batch size
        pos_embeddings = self.cached_pos_embeddings.expand(num_frames, -1, -1)
        
        # Reconstruct faces
        reconstructed_faces = self.reconstruction_model(
            subject_embeddings, expression_tokens, pos_embeddings
        )
        
        return reconstructed_faces
    
    def compute_loss(self, reconstructed_faces: torch.Tensor, original_faces: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction loss
        
        Args:
            reconstructed_faces: (num_frames, 3, 518, 518) - Reconstructed faces
            original_faces: (num_frames, 3, 518, 518) - Original faces
            
        Returns:
            torch.Tensor: Reconstruction loss
        """
        # MSE loss between reconstructed and original faces
        loss = nn.MSELoss()(reconstructed_faces, original_faces)
        return loss


def train_expression_reconstruction_new(
    dataset_path: str,
    expression_transformer_checkpoint_path: str,
    reconstruction_model_checkpoint_path: str = None,
    checkpoint_dir: str = "checkpoints",
    save_every_epochs: int = 2,
    batch_size: int = 8,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    max_samples: int = None,
    val_dataset_path: str = None,
    max_val_samples: int = None,
    device = "cpu",
    embed_dim: int = 384,
    num_cross_attention_layers: int = 2,
    num_self_attention_layers: int = 2,
    num_heads: int = 8,
    ff_dim: int = 1536,
    dropout: float = 0.1
):
    """
    Train the expression reconstruction model
    
    Args:
        dataset_path: Path to training dataset
        expression_transformer_checkpoint_path: Path to frozen expression transformer checkpoint
        checkpoint_dir: Directory to save checkpoints
        save_every_epochs: Save checkpoint every N epochs
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        max_samples: Maximum number of samples to use (for debugging)
        val_dataset_path: Path to validation dataset
        max_val_samples: Maximum number of validation samples
        device: Device to use for training
        embed_dim: Embedding dimension
        num_cross_attention_layers: Number of cross-attention layers
        num_self_attention_layers: Number of self-attention layers
        num_heads: Number of attention heads
        ff_dim: Feed-forward dimension
        dropout: Dropout rate
    """
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Handle device parameter (can be string or torch.device)
    device_str = str(device) if hasattr(device, 'type') else device
    
    # Initialize trainer
    trainer = ExpressionReconstructionTrainer(
        expression_transformer_checkpoint_path=expression_transformer_checkpoint_path,
        reconstruction_model_checkpoint_path=reconstruction_model_checkpoint_path,
        device=device_str,
        embed_dim=embed_dim,
        num_cross_attention_layers=num_cross_attention_layers,
        num_self_attention_layers=num_self_attention_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout=dropout
    )
    
    # Load datasets
    logger.info(f"Loading training dataset from: {dataset_path}")
    train_dataset = FaceDataset(dataset_path, max_samples=max_samples)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    if val_dataset_path:
        logger.info(f"Loading validation dataset from: {val_dataset_path}")
        val_dataset = FaceDataset(val_dataset_path, max_samples=max_val_samples)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    else:
        val_dataloader = None
    
    # Setup optimizer and criterion
    optimizer = optim.AdamW(trainer.reconstruction_model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Training loop
    logger.info("🚀 Starting training...")
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training phase
        trainer.reconstruction_model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            face_images, subject_ids, clip_lengths = batch
            
            # Process each clip in the batch
            batch_loss = 0.0
            num_clips = len(clip_lengths)
            
            start_idx = 0
            for clip_idx in range(num_clips):
                clip_length = clip_lengths[clip_idx]
                
                # Extract clip frames
                clip_frames = face_images[start_idx:start_idx + clip_length]
                clip_subject_ids = subject_ids[start_idx:start_idx + clip_length]
                
                # Move to device
                clip_frames = clip_frames.to(device)
                clip_subject_ids = clip_subject_ids.to(device)
                
                # Process clip (get expression tokens and subject embeddings)
                expression_tokens, subject_embeddings = trainer.process_clip(clip_frames, clip_subject_ids)
                
                # Reconstruct faces using the reconstruction model
                reconstructed_faces = trainer.reconstruct_faces(expression_tokens, subject_embeddings)
                
                # Compute reconstruction loss
                clip_loss = criterion(reconstructed_faces, clip_frames)
                batch_loss += clip_loss
                
                start_idx += clip_length
            
            # Average loss over clips
            batch_loss = batch_loss / num_clips
            
            # Backward pass for reconstruction model
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            epoch_loss += batch_loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{batch_loss.item():.6f}",
                'avg_loss': f"{epoch_loss/num_batches:.6f}"
            })
        
        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)
        
        logger.info(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.6f}")
        
        # CUDA memory cleanup after each epoch
        if hasattr(device, 'type') and device.type == "cuda":
            torch.cuda.empty_cache()
        
        # Validation phase
        if val_dataloader is not None:
            val_loss = validate_expression_reconstruction_new(
                trainer, val_dataloader, criterion, device_str
            )
            val_losses.append(val_loss)
            logger.info(f"Epoch {epoch+1} - Val Loss: {val_loss:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % save_every_epochs == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"expression_reconstruction_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': trainer.reconstruction_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': val_losses[-1] if val_losses else None,
                'config': {
                    'embed_dim': embed_dim,
                    'num_cross_attention_layers': num_cross_attention_layers,
                    'num_self_attention_layers': num_self_attention_layers,
                    'num_heads': num_heads,
                    'ff_dim': ff_dim,
                    'dropout': dropout
                }
            }, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(checkpoint_dir, "expression_reconstruction_final.pt")
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': trainer.reconstruction_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'config': {
            'embed_dim': embed_dim,
            'num_cross_attention_layers': num_cross_attention_layers,
            'num_self_attention_layers': num_self_attention_layers,
            'num_heads': num_heads,
            'ff_dim': ff_dim,
            'dropout': dropout
        }
    }, final_checkpoint_path)
    logger.info(f"Final checkpoint saved: {final_checkpoint_path}")
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, checkpoint_dir)
    
    logger.info("🎉 Training completed!")
    
    return trainer.reconstruction_model


def validate_expression_reconstruction_new(
    trainer: ExpressionReconstructionTrainer,
    val_dataloader: DataLoader,
    criterion: nn.Module,
    device
) -> float:
    """
    Validate the expression reconstruction model
    
    Args:
        trainer: ExpressionReconstructionTrainer instance
        val_dataloader: Validation data loader
        criterion: Loss function
        device: Device to use
        
    Returns:
        float: Average validation loss
    """
    trainer.reconstruction_model.eval()
    val_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_dataloader:
            face_images, subject_ids, clip_lengths = batch
            
            # Process each clip in the batch
            batch_loss = 0.0
            num_clips = len(clip_lengths)
            
            start_idx = 0
            for clip_idx in range(num_clips):
                clip_length = clip_lengths[clip_idx]
                
                # Extract clip frames
                clip_frames = face_images[start_idx:start_idx + clip_length]
                clip_subject_ids = subject_ids[start_idx:start_idx + clip_length]
                
                # Move to device
                clip_frames = clip_frames.to(device)
                clip_subject_ids = clip_subject_ids.to(device)
                
                # Process clip (get expression tokens and subject embeddings)
                expression_tokens, subject_embeddings = trainer.process_clip(clip_frames, clip_subject_ids)
                
                # Reconstruct faces
                reconstructed_faces = trainer.reconstruct_faces(expression_tokens, subject_embeddings)
                
                # Compute loss
                clip_loss = criterion(reconstructed_faces, clip_frames)
                batch_loss += clip_loss
                
                start_idx += clip_length
            
            # Average loss over clips
            batch_loss = batch_loss / num_clips
            val_loss += batch_loss.item()
            num_batches += 1
    
    return val_loss / num_batches


def plot_training_curves(train_losses: list, val_losses: list, checkpoint_dir: str):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    if val_losses:
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    
    plt.title('Expression Reconstruction Training Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(checkpoint_dir, 'training_curves.png')
    plt.savefig(plot_path)
    plt.close()
    
    logger.info(f"Training curves saved: {plot_path}")


def test_expression_reconstruction_trainer():
    """Test the expression reconstruction trainer"""
    import torch
    
    print("🧪 Testing Expression Reconstruction Trainer...")
    
    # Create dummy checkpoint path (you'll need a real one for actual training)
    dummy_checkpoint_path = "dummy_checkpoint.pt"
    
    # Create a dummy checkpoint for testing
    dummy_model = ExpressionTransformer()
    torch.save({
        'model_state_dict': dummy_model.state_dict(),
        'epoch': 1
    }, dummy_checkpoint_path)
    
    try:
        # Test trainer initialization
        trainer = ExpressionReconstructionTrainer(
            expression_transformer_checkpoint_path=dummy_checkpoint_path,
            device="cpu"
        )
        print("✅ Trainer initialization successful")
        
        # Test clip processing
        num_frames = 30
        face_images = torch.randn(num_frames, 3, 518, 518)
        subject_ids = torch.randint(0, 100, (num_frames,))
        
        expression_tokens, subject_embeddings = trainer.process_clip(face_images, subject_ids)
        print(f"✅ Clip processing successful:")
        print(f"   Expression tokens shape: {expression_tokens.shape}")
        print(f"   Subject embeddings shape: {subject_embeddings.shape}")
        
        # Test face reconstruction
        reconstructed_faces = trainer.reconstruct_faces(expression_tokens, subject_embeddings)
        print(f"✅ Face reconstruction successful:")
        print(f"   Reconstructed faces shape: {reconstructed_faces.shape}")
        print(f"   Output range: [{reconstructed_faces.min().item():.4f}, {reconstructed_faces.max().item():.4f}]")
        
        # Test loss computation
        loss = trainer.compute_loss(reconstructed_faces, face_images)
        print(f"✅ Loss computation successful: {loss.item():.6f}")
        
        print("\n🎉 All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
    finally:
        # Clean up dummy checkpoint
        if os.path.exists(dummy_checkpoint_path):
            os.remove(dummy_checkpoint_path)


if __name__ == "__main__":
    # Run test
    test_expression_reconstruction_trainer() 