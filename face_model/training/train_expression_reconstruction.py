"""
Joint Training Script for Components C and E
Trains Expression Transformer and Face Reconstruction Model together
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
# import psutil  # Optional for memory monitoring

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dinov2_tokenizer import DINOv2Tokenizer
from models.face_id_model import FaceIDModel
from models.expression_transformer import ExpressionTransformer
from models.face_reconstruction_model import FaceReconstructionModel
from data.dataset import FaceDataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JointExpressionReconstructionModel(nn.Module):
    """
    Joint model combining Components A, C, and E
    Takes pre-computed face ID tokens as input
    """
    
    def __init__(self, embed_dim=384, num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        
        # Component C: Expression Transformer (trainable)
        self.expression_transformer = ExpressionTransformer(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            num_layers=num_layers, 
            dropout=dropout
        )
        
        # Component E: Face Reconstruction Model (trainable)
        self.reconstruction_model = FaceReconstructionModel(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        
        logger.info("Joint Expression-Reconstruction Model initialized")
        
    def forward(self, face_images, face_id_tokens, tokenizer):
        """
        Args:
            face_images: (total_frames, 3, 518, 518) - Input face images (all frames from all clips in batch)
            face_id_tokens: (total_frames, 1, 384) - Pre-computed face ID tokens (same for all frames in a clip)
            tokenizer: DINOv2Tokenizer instance to use for tokenization
        
        Returns:
            reconstructed_faces: (total_frames, 3, 518, 518) - Reconstructed face images
            expression_tokens: (total_frames, 1, 384) - Expression tokens
        """
        total_frames = face_images.shape[0]
        
        # Component A: Extract patch tokens using provided tokenizer
        patch_tokens, pos_embeddings = tokenizer(face_images)  # (total_frames, 1369, 384), (total_frames, 1369, 384)
        
        # Component C: Extract expression tokens
        expression_tokens = self.expression_transformer(patch_tokens, pos_embeddings, face_id_tokens)  # (total_frames, 1, 384)
        
        # Component E: Reconstruct faces
        reconstructed_faces = self.reconstruction_model(
            patch_tokens, pos_embeddings, face_id_tokens, expression_tokens
        )  # (total_frames, 3, 518, 518)
        
        return reconstructed_faces, expression_tokens


class JointLoss(nn.Module):
    """
    Joint loss function for expression reconstruction training
    """
    
    def __init__(self, reconstruction_weight=1.0, identity_weight=1.0):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.identity_weight = identity_weight
        
        # Reconstruction loss (MSE)
        self.reconstruction_loss = nn.MSELoss()
        
        # Identity preservation loss (L2 on face ID tokens)
        self.identity_loss = nn.MSELoss()
        
    def forward(self, reconstructed_faces, original_faces, face_id_tokens_orig, face_id_tokens_recon):
        """
        Args:
            reconstructed_faces: (total_frames, 3, 518, 518) - Reconstructed faces
            original_faces: (total_frames, 3, 518, 518) - Original faces
            face_id_tokens_orig: (total_frames, 1, 384) - Face ID tokens from original images
            face_id_tokens_recon: (total_frames, 1, 384) - Face ID tokens from reconstructed images
        """
        # Reconstruction loss
        recon_loss = self.reconstruction_loss(reconstructed_faces, original_faces)
        
        # Identity preservation loss
        identity_loss = self.identity_loss(face_id_tokens_recon, face_id_tokens_orig)
        
        # Total loss
        total_loss = (self.reconstruction_weight * recon_loss + 
                     self.identity_weight * identity_loss)
        
        return total_loss, recon_loss, identity_loss


def train_expression_reconstruction(
    dataset_path,
    face_id_checkpoint_path,
    expression_transformer_checkpoint_path=None,  # New parameter for expression transformer checkpoint
    checkpoint_dir="checkpoints",
    save_every_epochs=2,
    batch_size=8,
    num_epochs=10,
    learning_rate=1e-4,
    reconstruction_weight=1.0,
    identity_weight=1.0,
    max_samples=None,
    val_dataset_path=None,  # New parameter for validation dataset
    max_val_samples=None,   # New parameter for validation samples
    device="cpu"  # Force CPU for testing
):
    """
    Train the joint expression reconstruction model
    Requires a pre-trained face ID model checkpoint
    """
    logger.info(f"Starting joint training on device: {device}")
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load dataset
    dataset = FaceDataset(dataset_path, max_samples=max_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    
    # Load validation dataset if provided
    val_dataloader = None
    if val_dataset_path is not None:
        val_dataset = FaceDataset(val_dataset_path, max_samples=max_val_samples)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
        logger.info(f"Validation dataset loaded with {len(val_dataloader)} batches")
    
    logger.info(f"Training dataset loaded with {len(dataloader)} batches")
    
    # Initialize DINOv2 tokenizer for face ID token extraction
    dinov2_tokenizer = DINOv2Tokenizer()
    
    # Determine expression transformer architecture
    expr_embed_dim, expr_num_heads, expr_num_layers, expr_dropout = 384, 8, 2, 0.1  # Defaults
    
    # Load expression transformer checkpoint if provided to get architecture
    if expression_transformer_checkpoint_path is not None:
        if not os.path.exists(expression_transformer_checkpoint_path):
            raise FileNotFoundError(
                f"Expression transformer checkpoint not found: {expression_transformer_checkpoint_path}\n"
                f"Please train the expression transformer first or set this parameter to None"
            )
        
        logger.info(f"Loading expression transformer architecture from checkpoint: {expression_transformer_checkpoint_path}")
        try:
            # Load checkpoint to get architecture
            expr_checkpoint = torch.load(expression_transformer_checkpoint_path, map_location=device)
            
            # Get architecture from checkpoint config
            if 'config' in expr_checkpoint and 'expression_model' in expr_checkpoint['config']:
                expr_config = expr_checkpoint['config']['expression_model']
                expr_embed_dim = expr_config.get('embed_dim', 384)
                expr_num_heads = expr_config.get('num_heads', 8)
                expr_num_layers = expr_config.get('num_layers', 2)
                expr_dropout = expr_config.get('dropout', 0.1)
                logger.info(f"Using expression transformer architecture from checkpoint: {expr_num_layers} layers, {expr_num_heads} heads")
            else:
                logger.warning("No architecture config found in expression transformer checkpoint, using defaults")
                
        except Exception as e:
            logger.warning(f"Failed to load expression transformer checkpoint for architecture: {str(e)}, using defaults")
    
    # Initialize joint model with correct architecture
    joint_model = JointExpressionReconstructionModel(
        embed_dim=expr_embed_dim,
        num_heads=expr_num_heads,
        num_layers=expr_num_layers,
        dropout=expr_dropout
    ).to(device)
    
    # Initialize face ID model and load pre-trained checkpoint
    # First, load checkpoint to get the architecture
    if not os.path.exists(face_id_checkpoint_path):
        raise FileNotFoundError(
            f"Face ID model checkpoint not found: {face_id_checkpoint_path}\n"
            f"Please train the face ID model first using train_face_id.py"
        )
    
    # Load checkpoint to get architecture
    checkpoint = torch.load(face_id_checkpoint_path, map_location=device)
    
    # Get architecture from checkpoint config
    if 'config' in checkpoint and 'face_id_model' in checkpoint['config']:
        face_id_config = checkpoint['config']['face_id_model']
        embed_dim = face_id_config.get('embed_dim', 384)
        num_heads = face_id_config.get('num_heads', 8)
        num_layers = face_id_config.get('num_layers', 2)
        dropout = face_id_config.get('dropout', 0.1)
        logger.info(f"Loading face ID model with architecture from checkpoint: {num_layers} layers, {num_heads} heads")
    else:
        # Fallback to default architecture
        embed_dim, num_heads, num_layers, dropout = 384, 8, 2, 0.1
        logger.warning("No architecture config found in checkpoint, using defaults")
    
    # Initialize face ID model with correct architecture
    face_id_model = FaceIDModel(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    # Load face ID model checkpoint
    logger.info(f"Loading face ID model from checkpoint: {face_id_checkpoint_path}")
    try:
        if 'face_id_model_state_dict' in checkpoint:
            face_id_model.load_state_dict(checkpoint['face_id_model_state_dict'])
            logger.info(f"Loaded face ID model from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            # Try loading the entire checkpoint as state dict (for compatibility)
            face_id_model.load_state_dict(checkpoint)
            logger.info("Loaded face ID model state dict directly")
    except Exception as e:
        raise RuntimeError(f"Failed to load face ID model checkpoint: {str(e)}")
    
    # Freeze face ID model parameters
    for param in face_id_model.parameters():
        param.requires_grad = False
    
    logger.info("Face ID model loaded and frozen successfully")
    
    # Load expression transformer weights if checkpoint provided
    if expression_transformer_checkpoint_path is not None:
        logger.info(f"Loading expression transformer weights from checkpoint: {expression_transformer_checkpoint_path}")
        try:
            # Load the expression transformer state dict
            if 'expression_transformer_state_dict' in expr_checkpoint:
                joint_model.expression_transformer.load_state_dict(expr_checkpoint['expression_transformer_state_dict'])
                logger.info(f"Loaded expression transformer from epoch {expr_checkpoint.get('epoch', 'unknown')}")
            else:
                # Try loading the entire checkpoint as state dict (for compatibility)
                joint_model.expression_transformer.load_state_dict(expr_checkpoint)
                logger.info("Loaded expression transformer state dict directly")
            
            # Freeze expression transformer parameters
            for param in joint_model.expression_transformer.parameters():
                param.requires_grad = False
            
            logger.info("Expression transformer loaded and frozen successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load expression transformer checkpoint: {str(e)}")
    else:
        logger.info("No expression transformer checkpoint provided - training from scratch")
    
    # Initialize loss function
    criterion = JointLoss(
        reconstruction_weight=reconstruction_weight,
        identity_weight=identity_weight
    ).to(device)
    
    # Initialize optimizer (only train Components C and E, but C might be frozen)
    trainable_params = []
    
    # Add expression transformer parameters only if not frozen
    if expression_transformer_checkpoint_path is None:
        trainable_params.extend(list(joint_model.expression_transformer.parameters()))
        logger.info("Expression transformer parameters will be trained")
    else:
        logger.info("Expression transformer parameters are frozen")
    
    # Always add reconstruction model parameters
    trainable_params.extend(list(joint_model.reconstruction_model.parameters()))
    logger.info("Reconstruction model parameters will be trained")
    
    optimizer = optim.Adam(trainable_params, lr=learning_rate)
    
    # Training loop
    joint_model.train()
    total_steps = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_identity_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Monitor memory usage (optional)
            # if batch_idx % 10 == 0:
            #     process = psutil.Process()
            #     memory_info = process.memory_info()
            #     logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.1f} MB")
            
            # Extract frames and face ID tokens from all clips in the batch
            all_frames = []
            face_id_tokens = []
            
            for frames in batch['frames']:
                # frames: (num_frames, 3, 518, 518) for one clip
                num_frames = frames.shape[0]
                all_frames.append(frames)
                
                # Extract face ID token from first frame of this clip
                first_frame = frames[0:1]  # (1, 3, 518, 518) - keep on CPU initially
                
                with torch.no_grad():
                    first_frame_patches, first_frame_pos = dinov2_tokenizer(first_frame)
                    first_frame_face_id = face_id_model(first_frame_patches, first_frame_pos)  # (1, 1, 384)
                
                # Repeat face ID token for all frames in this clip
                clip_face_ids = first_frame_face_id.repeat(num_frames, 1, 1)  # (num_frames, 1, 384)
                face_id_tokens.append(clip_face_ids)
            
            # Concatenate all frames and face ID tokens
            face_images = torch.cat(all_frames, dim=0).to(device)  # (total_frames, 3, 518, 518)
            face_id_tokens = torch.cat(face_id_tokens, dim=0).to(device)  # (total_frames, 1, 384)
            
            # Forward pass with pre-computed face ID tokens
            reconstructed_faces, expression_tokens = joint_model(face_images, face_id_tokens, dinov2_tokenizer)
            
            # Get face ID tokens from reconstructed images for identity preservation
            with torch.no_grad():
                patch_tokens_recon, pos_embeddings_recon = dinov2_tokenizer(reconstructed_faces)
                face_id_tokens_recon = face_id_model(patch_tokens_recon, pos_embeddings_recon).to(device)
                
                # Explicitly delete intermediate tensors to free memory
                del patch_tokens_recon, pos_embeddings_recon
            
            # Compute loss
            try:
                total_loss, recon_loss, identity_loss = criterion(
                    reconstructed_faces, face_images, face_id_tokens, face_id_tokens_recon
                )
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                # Clear cache to prevent memory buildup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                else:
                    # Force garbage collection for CPU training
                    import gc
                    gc.collect()
                
                # Update metrics
                epoch_loss += total_loss.item()
                epoch_recon_loss += recon_loss.item()
                epoch_identity_loss += identity_loss.item()
                total_steps += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Total Loss': f'{total_loss.item():.4f}',
                    'Recon Loss': f'{recon_loss.item():.4f}',
                    'Identity Loss': f'{identity_loss.item():.4f}'
                })
                
                # Explicitly delete intermediate tensors after using them
                del reconstructed_faces, expression_tokens, face_id_tokens_recon
                del total_loss, recon_loss, identity_loss
                
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {str(e)}")
                logger.error(f"Batch shapes - reconstructed_faces: {reconstructed_faces.shape}, face_images: {face_images.shape}")
                logger.error(f"Batch shapes - face_id_tokens: {face_id_tokens.shape}, face_id_tokens_recon: {face_id_tokens_recon.shape}")
                continue
        
        # Log epoch statistics
        avg_loss = epoch_loss / len(dataloader)
        avg_recon_loss = epoch_recon_loss / len(dataloader)
        avg_identity_loss = epoch_identity_loss / len(dataloader)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                   f"Avg Total Loss: {avg_loss:.4f}, "
                   f"Avg Recon Loss: {avg_recon_loss:.4f}, "
                   f"Avg Identity Loss: {avg_identity_loss:.4f}")
        
        # Validate if validation dataloader is provided
        if val_dataloader is not None:
            val_loss, val_recon_loss, val_identity_loss = validate_expression_reconstruction(
                joint_model, face_id_model, val_dataloader, criterion, dinov2_tokenizer, device
            )
            logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                       f"Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save epoch checkpoints (only at end of epoch, not during training)
        if (epoch + 1) % save_every_epochs == 0:
            # Joint model
            joint_epoch_path = os.path.join(checkpoint_dir, f"joint_model_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'joint_model_state_dict': joint_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'avg_total_loss': avg_loss,
                'avg_recon_loss': avg_recon_loss,
                'avg_identity_loss': avg_identity_loss,
            }, joint_epoch_path)
            
            # Expression transformer
            expression_epoch_path = os.path.join(checkpoint_dir, f"expression_transformer_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'expression_transformer_state_dict': joint_model.expression_transformer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'avg_total_loss': avg_loss,
                'avg_recon_loss': avg_recon_loss,
                'avg_identity_loss': avg_identity_loss,
            }, expression_epoch_path)
            
            # Reconstruction model
            reconstruction_epoch_path = os.path.join(checkpoint_dir, f"reconstruction_model_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'reconstruction_model_state_dict': joint_model.reconstruction_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'avg_total_loss': avg_loss,
                'avg_recon_loss': avg_recon_loss,
                'avg_identity_loss': avg_identity_loss,
            }, reconstruction_epoch_path)
            
            logger.info(f"Saved epoch checkpoints: {joint_epoch_path}, {expression_epoch_path}, {reconstruction_epoch_path}")
    
    logger.info("Training completed!")
    return joint_model


def validate_expression_reconstruction(
    joint_model,
    face_id_model,
    val_dataloader,
    criterion,
    dinov2_tokenizer,
    device="cpu"
):
    """
    Validate the joint expression reconstruction model
    """
    joint_model.eval()
    face_id_model.eval()
    
    total_loss = 0.0
    total_recon_loss = 0.0
    total_identity_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():  # No gradient computation for validation
        for batch_idx, batch in enumerate(val_dataloader):
            # Extract frames and face ID tokens from all clips in the batch
            all_frames = []
            face_id_tokens = []
            
            for frames in batch['frames']:
                # frames: (num_frames, 3, 518, 518) for one clip
                num_frames = frames.shape[0]
                all_frames.append(frames)
                
                # Extract face ID token from first frame of this clip
                first_frame = frames[0:1]  # (1, 3, 518, 518)
                
                # Get DINOv2 tokens for first frame
                patch_tokens, pos_emb = dinov2_tokenizer(first_frame)
                
                # Get face ID token for first frame
                face_id_token = face_id_model(patch_tokens, pos_emb)  # (1, 1, 384)
                
                # Repeat face ID token for all frames in this clip
                clip_face_id_tokens = face_id_token.repeat(num_frames, 1, 1)  # (num_frames, 1, 384)
                face_id_tokens.append(clip_face_id_tokens)
            
            # Concatenate all frames and face ID tokens
            all_frames = torch.cat(all_frames, dim=0).to(device)  # (total_frames, 3, 518, 518)
            face_id_tokens = torch.cat(face_id_tokens, dim=0).to(device)  # (total_frames, 1, 384)
            
            # Forward pass through joint model
            reconstructed_faces, expression_tokens = joint_model(all_frames, face_id_tokens, dinov2_tokenizer)
            
            # Get face ID tokens from reconstructed images for identity preservation
            with torch.no_grad():
                patch_tokens_recon, pos_embeddings_recon = dinov2_tokenizer(reconstructed_faces)
                face_id_tokens_recon = face_id_model(patch_tokens_recon, pos_embeddings_recon)
            
            # Compute loss
            loss, recon_loss, identity_loss = criterion(
                reconstructed_faces, all_frames, face_id_tokens, face_id_tokens_recon
            )
            
            # Update metrics
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_identity_loss += identity_loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_recon_loss = total_recon_loss / num_batches
    avg_identity_loss = total_identity_loss / num_batches
    
    logger.info(f"Validation - Total Loss: {avg_loss:.4f}, "
               f"Reconstruction Loss: {avg_recon_loss:.4f}, "
               f"Identity Loss: {avg_identity_loss:.4f}")
    
    return avg_loss, avg_recon_loss, avg_identity_loss


def test_joint_model():
    """Test the joint model"""
    import torch
    
    # Create models
    joint_model = JointExpressionReconstructionModel()
    face_id_model = FaceIDModel()
    
    # Create dummy input simulating clips
    # Simulate 2 clips: first clip has 3 frames, second clip has 2 frames
    clip1_frames = torch.randn(3, 3, 518, 518)  # 3 frames
    clip2_frames = torch.randn(2, 3, 518, 518)  # 2 frames
    
    # Concatenate all frames
    face_images = torch.cat([clip1_frames, clip2_frames], dim=0)  # (5, 3, 518, 518)
    clip_lengths = [3, 2]  # Length of each clip
    
    # Efficiently extract face ID tokens from first frame of each clip
    face_id_tokens = []
    current_idx = 0
    
    for clip_length in clip_lengths:
        # Get first frame of this clip
        first_frame = face_images[current_idx:current_idx+1]  # (1, 3, 518, 518)
        
        # Extract face ID token from first frame
        with torch.no_grad():
            first_frame_patches, first_frame_pos = dinov2_tokenizer(first_frame)
            first_frame_face_id = face_id_model(first_frame_patches, first_frame_pos)  # (1, 1, 384)
        
        # Repeat face ID token for all frames in this clip
        clip_face_ids = first_frame_face_id.repeat(clip_length, 1, 1)  # (clip_length, 1, 384)
        face_id_tokens.append(clip_face_ids)
        
        current_idx += clip_length
    
    # Concatenate all face ID tokens
    face_id_tokens = torch.cat(face_id_tokens, dim=0)  # (5, 1, 384)
    
    # Forward pass with pre-computed face ID tokens
    reconstructed_faces, expression_tokens = joint_model(face_images, face_id_tokens, dinov2_tokenizer)
    
    print(f"Input face images shape: {face_images.shape}")
    print(f"Clip lengths: {clip_lengths}")
    print(f"Reconstructed faces shape: {reconstructed_faces.shape}")
    print(f"Face ID tokens shape: {face_id_tokens.shape}")
    print(f"Expression tokens shape: {expression_tokens.shape}")
    
    # Check that face ID tokens are the same within each clip
    print(f"Face ID token for clip 1 (frames 0-2): {face_id_tokens[0, 0, :5]}")  # First 5 values
    print(f"Face ID token for clip 1 (frame 1): {face_id_tokens[1, 0, :5]}")
    print(f"Face ID token for clip 1 (frame 2): {face_id_tokens[2, 0, :5]}")
    print(f"Face ID token for clip 2 (frames 3-4): {face_id_tokens[3, 0, :5]}")
    print(f"Face ID token for clip 2 (frame 4): {face_id_tokens[4, 0, :5]}")
    
    # Verify face ID tokens are the same within clips
    assert torch.allclose(face_id_tokens[0], face_id_tokens[1]), "Face ID tokens should be same within clip 1"
    assert torch.allclose(face_id_tokens[0], face_id_tokens[2]), "Face ID tokens should be same within clip 1"
    assert torch.allclose(face_id_tokens[3], face_id_tokens[4]), "Face ID tokens should be same within clip 2"
    
    # Test loss function
    criterion = JointLoss()
    total_loss, recon_loss, identity_loss = criterion(
        reconstructed_faces, face_images, face_id_tokens, face_id_tokens
    )
    
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Reconstruction loss: {recon_loss.item():.4f}")
    print(f"Identity loss: {identity_loss.item():.4f}")
    
    print("✅ Joint model test passed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train joint expression reconstruction model")
    parser.add_argument("--dataset_path", type=str, required=False, help="Path to dataset directory")
    parser.add_argument("--face_id_checkpoint_path", type=str, required=False, help="Path to face ID model checkpoint")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--save_every_epochs", type=int, default=2, help="Save checkpoint every N epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--reconstruction_weight", type=float, default=1.0, help="Reconstruction loss weight")
    parser.add_argument("--identity_weight", type=float, default=1.0, help="Identity loss weight")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum samples to load (for debugging)")
    parser.add_argument("--val_dataset_path", type=str, default=None, help="Path to validation dataset directory")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Maximum validation samples to load")
    parser.add_argument("--test", action="store_true", help="Run test instead of training")
    
    args = parser.parse_args()
    
    if args.test:
        test_joint_model()
    else:
        if not args.dataset_path or not args.face_id_checkpoint_path:
            parser.error("--dataset_path and --face_id_checkpoint_path are required when not running test")
        train_expression_reconstruction(
            dataset_path=args.dataset_path,
            face_id_checkpoint_path=args.face_id_checkpoint_path,
            checkpoint_dir=args.checkpoint_dir,
            save_every_epochs=args.save_every_epochs,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            reconstruction_weight=args.reconstruction_weight,
            identity_weight=args.identity_weight,
            max_samples=args.max_samples,
            val_dataset_path=args.val_dataset_path,
            max_val_samples=args.max_val_samples
        ) 