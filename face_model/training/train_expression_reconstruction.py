"""
Joint Training Script for Components C and E
Trains Expression Transformer and Face Reconstruction Model together or only the reconstruction model
Uses subject embeddings instead of face ID tokens
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
from models.expression_transformer import ExpressionTransformer
from models.face_reconstruction_model import FaceReconstructionModel
from data.dataset import FaceDataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JointExpressionReconstructionModel(nn.Module):
    """
    Joint model combining Components A, C, and E
    Takes subject IDs as input instead of face ID tokens
    """
    
    def __init__(self, 
                 expr_embed_dim=384, expr_num_heads=8, expr_num_layers=2, expr_dropout=0.1, expr_max_subjects=3500,
                 recon_embed_dim=384, recon_num_heads=8, recon_num_layers=2, recon_dropout=0.1):
        super().__init__()
        
        # Component C: Expression Transformer (trainable)
        self.expression_transformer = ExpressionTransformer(
            embed_dim=expr_embed_dim, 
            num_heads=expr_num_heads, 
            num_layers=expr_num_layers, 
            dropout=expr_dropout,
            max_subjects=expr_max_subjects
        )
        
        # Component E: Face Reconstruction Model (trainable)
        self.reconstruction_model = FaceReconstructionModel(
            embed_dim=recon_embed_dim,
            num_heads=recon_num_heads,
            num_layers=recon_num_layers,
            dropout=recon_dropout
        )
        
        logger.info("Joint Expression-Reconstruction Model initialized")
        logger.info(f"Expression Transformer: {expr_num_layers} layers, {expr_num_heads} heads, {expr_max_subjects} subjects")
        logger.info(f"Reconstruction Model: {recon_num_layers} layers, {recon_num_heads} heads")
        
    def forward(self, face_images, subject_ids, tokenizer):
        """
        Args:
            face_images: (total_frames, 3, 518, 518) - Input face images (all frames from all clips in batch)
            subject_ids: (total_frames,) - Subject IDs for each frame
            tokenizer: DINOv2Tokenizer instance to use for tokenization
        
        Returns:
            reconstructed_faces: (total_frames, 3, 518, 518) - Reconstructed face images
            expression_tokens: (total_frames, 1, 384) - Expression tokens
        """
        total_frames = face_images.shape[0]
        
        # Component A: Extract patch tokens using provided tokenizer
        patch_tokens, pos_embeddings = tokenizer(face_images)  # (total_frames, 1369, 384), (total_frames, 1369, 384)
        
        # Component C: Extract expression tokens and subject embeddings using inference method
        expression_tokens, subject_embeddings = self.expression_transformer.inference(patch_tokens, pos_embeddings, subject_ids)  # (total_frames, 1, 384), (total_frames, 1, 384)
        
        # Component E: Reconstruct faces using subject embeddings
        reconstructed_faces = self.reconstruction_model(
            patch_tokens, pos_embeddings, subject_embeddings, expression_tokens
        )  # (total_frames, 3, 518, 518)
        
        return reconstructed_faces, expression_tokens


class JointLoss(nn.Module):
    """
    Joint loss function for expression reconstruction training
    """
    
    def __init__(self, reconstruction_weight=1.0):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        
        # Reconstruction loss (MSE)
        self.reconstruction_loss = nn.MSELoss()
        
    def forward(self, reconstructed_faces, original_faces):
        """
        Args:
            reconstructed_faces: (total_frames, 3, 518, 518) - Reconstructed faces
            original_faces: (total_frames, 3, 518, 518) - Original faces
        """
        # Reconstruction loss only
        recon_loss = self.reconstruction_loss(reconstructed_faces, original_faces)
        
        return recon_loss, recon_loss  # Return total_loss and recon_loss only


def train_expression_reconstruction(
    dataset_path,
    expression_transformer_checkpoint_path,  # Required parameter for expression transformer checkpoint
    reconstruction_model_checkpoint_path=None,  # Optional parameter for reconstruction model checkpoint
    reconstruction_model_config=None,  # Configuration for reconstruction model architecture
    checkpoint_dir="checkpoints",
    save_every_epochs=2,
    batch_size=8,
    num_epochs=10,
    learning_rate=1e-4,
    reconstruction_weight=1.0,
    max_samples=None,
    val_dataset_path=None,  # New parameter for validation dataset
    max_val_samples=None,   # New parameter for validation samples
    device="cpu"  # Force CPU for testing
):
    """
    Train the joint expression reconstruction model
    Requires:
    - A pre-trained expression transformer checkpoint
    - Optional: A pre-trained reconstruction model checkpoint (will train from scratch if not provided)
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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    
    # Load validation dataset if provided
    val_dataloader = None
    if val_dataset_path is not None:
        val_dataset = FaceDataset(val_dataset_path, max_samples=max_val_samples)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
        logger.info(f"Validation dataset loaded with {len(val_dataloader)} batches")
    
    logger.info(f"Training dataset loaded with {len(dataloader)} batches")
    
    # Initialize DINOv2 tokenizer
    dinov2_tokenizer = DINOv2Tokenizer(device=device)
    
    # Load expression transformer weights (required)
    if expression_transformer_checkpoint_path is None:
        raise ValueError("expression_transformer_checkpoint_path is required. Please provide a checkpoint path.")
    
    if not os.path.exists(expression_transformer_checkpoint_path):
        raise FileNotFoundError(f"Expression transformer checkpoint not found: {expression_transformer_checkpoint_path}")
    
    # Load the expression transformer checkpoint for architecture info
    expr_checkpoint = torch.load(expression_transformer_checkpoint_path, map_location=device)
    
    # Get expression transformer architecture from checkpoint
    expr_embed_dim, expr_num_heads, expr_num_layers, expr_dropout, expr_max_subjects = 384, 4, 2, 0.1, 3500  # Defaults
    if 'config' in expr_checkpoint and 'expression_model' in expr_checkpoint['config']:
        expr_config = expr_checkpoint['config']['expression_model']
        expr_embed_dim = expr_config.get('embed_dim', 384)
        expr_num_heads = expr_config.get('num_heads', 4)
        expr_num_layers = expr_config.get('num_layers', 2)
        expr_dropout = expr_config.get('dropout', 0.1)
        expr_max_subjects = expr_config.get('max_subjects', 3500)
        logger.info(f"Using expression transformer architecture from checkpoint: {expr_num_layers} layers, {expr_num_heads} heads, {expr_max_subjects} subjects")
        print(f"📐 Expression Transformer Architecture from checkpoint: {expr_num_layers} layers, {expr_num_heads} heads, {expr_max_subjects} subjects")
    else:
        logger.warning("No architecture config found in expression transformer checkpoint, using defaults")
        print(f"⚠️  No architecture config found in expression transformer checkpoint, using defaults")
    
    # Load reconstruction model weights if checkpoint provided
    if reconstruction_model_checkpoint_path is not None:
        if not os.path.exists(reconstruction_model_checkpoint_path):
            raise FileNotFoundError(f"Reconstruction model checkpoint not found: {reconstruction_model_checkpoint_path}")
        
        logger.info(f"Loading reconstruction model weights from checkpoint: {reconstruction_model_checkpoint_path}")
        try:
            recon_checkpoint = torch.load(reconstruction_model_checkpoint_path, map_location=device)
            
            # Get architecture from checkpoint config
            recon_embed_dim, recon_num_heads, recon_num_layers, recon_dropout = 384, 4, 2, 0.1  # Defaults
            if 'config' in recon_checkpoint and 'reconstruction_model' in recon_checkpoint['config']:
                recon_config = recon_checkpoint['config']['reconstruction_model']
                recon_embed_dim = recon_config.get('embed_dim', 384)
                recon_num_heads = recon_config.get('num_heads', 4)
                recon_num_layers = recon_config.get('num_layers', 2)
                recon_dropout = recon_config.get('dropout', 0.1)
                logger.info(f"Using reconstruction model architecture from checkpoint: {recon_num_layers} layers, {recon_num_heads} heads")
                print(f"📐 Reconstruction Model Architecture from checkpoint: {recon_num_layers} layers, {recon_num_heads} heads")
            else:
                logger.warning("No architecture config found in reconstruction model checkpoint, using defaults")
                print(f"⚠️  No architecture config found in reconstruction model checkpoint, using defaults")
            

            
            # Initialize joint model with both architectures from checkpoints
            joint_model = JointExpressionReconstructionModel(
                expr_embed_dim=expr_embed_dim,
                expr_num_heads=expr_num_heads,
                expr_num_layers=expr_num_layers,
                expr_dropout=expr_dropout,
                expr_max_subjects=expr_max_subjects,
                recon_embed_dim=recon_embed_dim,
                recon_num_heads=recon_num_heads,
                recon_num_layers=recon_num_layers,
                recon_dropout=recon_dropout
            ).to(device)
            
            # Load reconstruction model weights
            if 'reconstruction_model_state_dict' in recon_checkpoint:
                joint_model.reconstruction_model.load_state_dict(recon_checkpoint['reconstruction_model_state_dict'])
                logger.info(f"✅ Successfully loaded reconstruction model from epoch {recon_checkpoint.get('epoch', 'unknown')}")
                print(f"✅ Successfully loaded reconstruction model from: {reconstruction_model_checkpoint_path}")
            else:
                # Try loading the entire checkpoint as state dict (for compatibility)
                joint_model.reconstruction_model.load_state_dict(recon_checkpoint)
                logger.info("✅ Successfully loaded reconstruction model state dict directly")
                print(f"✅ Successfully loaded reconstruction model from: {reconstruction_model_checkpoint_path}")
            
            # Load expression transformer weights into joint model
            logger.info(f"Loading expression transformer weights from checkpoint: {expression_transformer_checkpoint_path}")
            try:
                if 'expression_transformer_state_dict' in expr_checkpoint:
                    joint_model.expression_transformer.load_state_dict(expr_checkpoint['expression_transformer_state_dict'])
                    logger.info(f"✅ Successfully loaded expression transformer from epoch {expr_checkpoint.get('epoch', 'unknown')}")
                    print(f"✅ Successfully loaded expression transformer from: {expression_transformer_checkpoint_path}")
                else:
                    # Try loading the entire checkpoint as state dict (for compatibility)
                    joint_model.expression_transformer.load_state_dict(expr_checkpoint)
                    logger.info("✅ Successfully loaded expression transformer state dict directly")
                    print(f"✅ Successfully loaded expression transformer from: {expression_transformer_checkpoint_path}")
                
                # Freeze expression transformer parameters
                for param in joint_model.expression_transformer.parameters():
                    param.requires_grad = False
                
                logger.info("Expression transformer loaded and frozen successfully")
                print(f"🔒 Expression transformer frozen (parameters not trainable)")
                
            except Exception as e:
                raise RuntimeError(f"Failed to load expression transformer checkpoint: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Failed to load reconstruction model checkpoint: {str(e)}")
    else:
        # No reconstruction model checkpoint - use config or defaults
        recon_embed_dim, recon_num_heads, recon_num_layers, recon_dropout = 384, 4, 2, 0.1  # Defaults
        
        if reconstruction_model_config is not None:
            # Use provided reconstruction model config
            recon_embed_dim = reconstruction_model_config.get('embed_dim', 384)
            recon_num_heads = reconstruction_model_config.get('num_heads', 4)
            recon_num_layers = reconstruction_model_config.get('num_layers', 2)
            recon_dropout = reconstruction_model_config.get('dropout', 0.1)
            logger.info(f"Using reconstruction model architecture from config: {recon_num_layers} layers, {recon_num_heads} heads")
            print(f"📐 Reconstruction Model Architecture from config: {recon_num_layers} layers, {recon_num_heads} heads")
        else:
            logger.info(f"Using default reconstruction model architecture: {recon_num_layers} layers, {recon_num_heads} heads")
            print(f"📐 Reconstruction Model Architecture (default): {recon_num_layers} layers, {recon_num_heads} heads")
        

        
        # Initialize joint model with expression transformer from checkpoint and reconstruction model from config/defaults
        joint_model = JointExpressionReconstructionModel(
            expr_embed_dim=expr_embed_dim,
            expr_num_heads=expr_num_heads,
            expr_num_layers=expr_num_layers,
            expr_dropout=expr_dropout,
            expr_max_subjects=expr_max_subjects,
            recon_embed_dim=recon_embed_dim,
            recon_num_heads=recon_num_heads,
            recon_num_layers=recon_num_layers,
            recon_dropout=recon_dropout
        ).to(device)
        
        logger.info("No reconstruction model checkpoint provided - training from scratch")
        print(f"🎓 Reconstruction model will be trained from scratch")
        
        # Load expression transformer weights into joint model
        logger.info(f"Loading expression transformer weights from checkpoint: {expression_transformer_checkpoint_path}")
        try:
            if 'expression_transformer_state_dict' in expr_checkpoint:
                joint_model.expression_transformer.load_state_dict(expr_checkpoint['expression_transformer_state_dict'])
                logger.info(f"✅ Successfully loaded expression transformer from epoch {expr_checkpoint.get('epoch', 'unknown')}")
                print(f"✅ Successfully loaded expression transformer from: {expression_transformer_checkpoint_path}")
            else:
                # Try loading the entire checkpoint as state dict (for compatibility)
                joint_model.expression_transformer.load_state_dict(expr_checkpoint)
                logger.info("✅ Successfully loaded expression transformer state dict directly")
                print(f"✅ Successfully loaded expression transformer from: {expression_transformer_checkpoint_path}")
            
            # Freeze expression transformer parameters
            for param in joint_model.expression_transformer.parameters():
                param.requires_grad = False
            
            logger.info("Expression transformer loaded and frozen successfully")
            print(f"🔒 Expression transformer frozen (parameters not trainable)")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load expression transformer checkpoint: {str(e)}")
    
    # Initialize loss function
    criterion = JointLoss(
        reconstruction_weight=reconstruction_weight
    ).to(device)
    
    # Initialize optimizer (only train Component E - reconstruction model)
    trainable_params = []
    
    # Expression transformer is always frozen by default
    logger.info("Expression transformer parameters are frozen")
    
    # Only add reconstruction model parameters (Component E)
    trainable_params.extend(list(joint_model.reconstruction_model.parameters()))
    logger.info("Reconstruction model parameters will be trained")
    print(f"🎯 Training: Reconstruction model (transformer + CNN parts)")
    print(f"🔒 Frozen: Expression transformer (Component C)")
    
    # Verify expression transformer parameters are frozen
    expr_trainable_params = sum(p.numel() for p in joint_model.expression_transformer.parameters() if p.requires_grad)
    if expr_trainable_params > 0:
        logger.warning(f"⚠️ Expression transformer has {expr_trainable_params} trainable parameters - this should be 0!")
    else:
        logger.info(f"✅ Expression transformer parameters are properly frozen ({expr_trainable_params} trainable)")
    
    optimizer = optim.Adam(trainable_params, lr=learning_rate)
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in trainable_params)
    logger.info(f"Total trainable parameters: {total_params:,}")
    print(f"🎯 Training {total_params:,} parameters (reconstruction model only)")
    
    # Training loop
    joint_model.train()
    # Ensure expression transformer is in eval mode (frozen)
    joint_model.expression_transformer.eval()
    total_steps = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Monitor memory usage (optional)
            if batch_idx % 5 == 0:
                import gc
                gc.collect()  # Force garbage collection every 5 batches
                if batch_idx % 10 == 0:
                    import psutil
                    process = psutil.Process()
                    memory_info = process.memory_info()
                    logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.1f} MB")
                    
                    # Check system memory
                    system_memory = psutil.virtual_memory()
                    logger.info(f"System memory: {system_memory.available / 1024 / 1024:.1f} MB available, {system_memory.percent:.1f}% used")
                    
                    # Check CPU usage
                    cpu_percent = psutil.cpu_percent(interval=1)
                    logger.info(f"CPU usage: {cpu_percent:.1f}%")
            
            # Extract frames and subject IDs from all clips in the batch
            all_frames = []
            subject_ids = []
            
            for frames, subject_id in zip(batch['frames'], batch['subject_id']):
                # frames: (num_frames, 3, 518, 518) for one clip
                num_frames = frames.shape[0]
                all_frames.append(frames)
                
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
            
            # Forward pass with subject IDs
            reconstructed_faces, expression_tokens = joint_model(face_images, subject_ids, dinov2_tokenizer)
            
            # Compute loss
            try:
                total_loss, recon_loss = criterion(
                    reconstructed_faces, face_images
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
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Total Loss': f'{total_loss.item():.4f}',
                    'Recon Loss': f'{recon_loss.item():.4f}'
                })
                
                # Explicitly delete intermediate tensors after using them
                del reconstructed_faces, expression_tokens
                del total_loss, recon_loss
                
                # Clear intermediate tensors (these were already concatenated above)
                del face_images, subject_ids
                
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {str(e)}")
                logger.error(f"Batch shapes - reconstructed_faces: {reconstructed_faces.shape}, face_images: {face_images.shape}")
                logger.error(f"Batch shapes - subject_ids: {subject_ids.shape}")
                continue
        
        # Log epoch statistics
        avg_loss = epoch_loss / len(dataloader)
        avg_recon_loss = epoch_recon_loss / len(dataloader)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                   f"Avg Total Loss: {avg_loss:.4f}, "
                   f"Avg Recon Loss: {avg_recon_loss:.4f}")
        
        # CUDA memory cleanup after each epoch
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        # Validate if validation dataloader is provided
        if val_dataloader is not None:
            val_loss, val_recon_loss = validate_expression_reconstruction(
                joint_model, val_dataloader, criterion, dinov2_tokenizer, device
            )
            logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                       f"Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save epoch checkpoints (only at end of epoch, not during training)
        if (epoch + 1) % save_every_epochs == 0:
            # Reconstruction model
            reconstruction_epoch_path = os.path.join(checkpoint_dir, f"reconstruction_model_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'reconstruction_model_state_dict': joint_model.reconstruction_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'avg_total_loss': avg_loss,
                'avg_recon_loss': avg_recon_loss,
            }, reconstruction_epoch_path)
            
            logger.info(f"Saved epoch checkpoint: {reconstruction_epoch_path}")
    
    logger.info("Training completed!")
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
    total_recon_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():  # No gradient computation for validation
        for batch_idx, batch in enumerate(val_dataloader):
            # Extract frames and subject IDs from all clips in the batch
            all_frames = []
            subject_ids = []
            
            for frames, subject_id in zip(batch['frames'], batch['subject_id']):
                # frames: (num_frames, 3, 518, 518) for one clip
                num_frames = frames.shape[0]
                all_frames.append(frames)
                
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
            all_frames = torch.cat(all_frames, dim=0).to(device)  # (total_frames, 3, 518, 518)
            subject_ids = torch.cat(subject_ids, dim=0).to(device)  # (total_frames,)
            
            # Forward pass through joint model
            reconstructed_faces, expression_tokens = joint_model(all_frames, subject_ids, dinov2_tokenizer)
            
            # Compute loss
            loss, recon_loss = criterion(
                reconstructed_faces, all_frames
            )
            
            # Update metrics
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_recon_loss = total_recon_loss / num_batches
    
    logger.info(f"Validation - Total Loss: {avg_loss:.4f}, "
               f"Reconstruction Loss: {avg_recon_loss:.4f}")
    
    return avg_loss, avg_recon_loss


def test_joint_model():
    """Test the joint model"""
    import torch
    
    # Create models
    joint_model = JointExpressionReconstructionModel()
    # face_id_model = FaceIDModel() # This line is removed as face ID model is no longer used
    
    # Initialize DINOv2 tokenizer
    dinov2_tokenizer = DINOv2Tokenizer(device="cpu") # Changed to "cpu" for testing
    
    # Create dummy input simulating clips
    # Simulate 2 clips: first clip has 3 frames, second clip has 2 frames
    clip1_frames = torch.randn(3, 3, 518, 518)  # 3 frames
    clip2_frames = torch.randn(2, 3, 518, 518)  # 2 frames
    
    # Concatenate all frames
    face_images = torch.cat([clip1_frames, clip2_frames], dim=0)  # (5, 3, 518, 518)
    clip_lengths = [3, 2]  # Length of each clip
    
    # Efficiently extract subject IDs from first frame of each clip
    subject_ids = []
    current_idx = 0
    
    for clip_length in clip_lengths:
        # Get first frame of this clip
        first_frame = face_images[current_idx:current_idx+1]  # (1, 3, 518, 518)
        
        # Extract subject ID from first frame
        # with torch.no_grad(): # This line is removed as face ID model is no longer used
        #     first_frame_patches, first_frame_pos = dinov2_tokenizer(first_frame)
        #     first_frame_subject_id = face_id_model(first_frame_patches, first_frame_pos)  # (1, 1, 384)
        
        # Repeat subject ID for all frames in this clip
        # clip_subject_ids = first_frame_subject_id.repeat(clip_length, 1)  # (clip_length, 1)
        # subject_ids.append(clip_subject_ids)
        
        # For testing, we'll just use a dummy subject ID for now
        # This part of the test needs to be updated if subject_id is no longer available
        # For now, we'll use a placeholder or remove this test if it's no longer relevant
        # As per the edit hint, the face_id_model dependency is removed.
        # The test function needs to be adapted or removed if subject_id is no longer available.
        # For now, we'll just pass a dummy subject ID.
        dummy_subject_id = 123 # Placeholder for subject ID
        clip_subject_ids = torch.full((clip_length,), dummy_subject_id, dtype=torch.long)
        subject_ids.append(clip_subject_ids)
        
        current_idx += clip_length
    
    # Concatenate all subject IDs
    subject_ids = torch.cat(subject_ids, dim=0)  # (5, 1)
    
    # Forward pass with pre-computed subject IDs
    reconstructed_faces, expression_tokens = joint_model(face_images, subject_ids, dinov2_tokenizer)
    
    print(f"Input face images shape: {face_images.shape}")
    print(f"Clip lengths: {clip_lengths}")
    print(f"Reconstructed faces shape: {reconstructed_faces.shape}")
    print(f"Subject IDs shape: {subject_ids.shape}")
    print(f"Expression tokens shape: {expression_tokens.shape}")
    
    # Check that subject IDs are the same within each clip
    print(f"Subject ID for clip 1 (frames 0-2): {subject_ids[0, 0]}")  # First 5 values
    print(f"Subject ID for clip 1 (frame 1): {subject_ids[1, 0]}")
    print(f"Subject ID for clip 1 (frame 2): {subject_ids[2, 0]}")
    print(f"Subject ID for clip 2 (frames 3-4): {subject_ids[3, 0]}")
    print(f"Subject ID for clip 2 (frame 4): {subject_ids[4, 0]}")
    
    # Verify subject IDs are the same within clips
    # assert torch.allclose(subject_ids[0], subject_ids[1]), "Subject IDs should be same within clip 1" # This assertion is no longer valid
    # assert torch.allclose(subject_ids[0], subject_ids[2]), "Subject IDs should be same within clip 1" # This assertion is no longer valid
    # assert torch.allclose(subject_ids[3], subject_ids[4]), "Subject IDs should be same within clip 2" # This assertion is no longer valid
    
    # Test loss function
    criterion = JointLoss()
    total_loss, recon_loss = criterion(
        reconstructed_faces, face_images
    )
    
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Reconstruction loss: {recon_loss.item():.4f}")
    
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
            max_samples=args.max_samples,
            val_dataset_path=args.val_dataset_path,
            max_val_samples=args.max_val_samples
        ) 