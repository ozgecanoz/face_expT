"""
Joint Training: Expression Transformer + Prediction Head
Trains both Component C (Expression Transformer) and Component D (Transformer Decoder)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import logging
import sys
import uuid
from datetime import datetime
sys.path.append('.')

from models.expression_transformer import ExpressionTransformer
from models.expression_transformer_decoder import ExpressionTransformerDecoder
from models.face_id_model import FaceIDModel
from models.dinov2_tokenizer import DINOv2Tokenizer
from data.dataset import FaceDataset

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
        all_predicted_next_tokens = self.transformer_decoder(all_expression_tokens, clip_lengths)
        
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
        loss = self.mse_loss(predicted_next_token, actual_next_token)
        
        return loss


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
    max_sequence_length=50
):
    """
    Train the joint expression prediction model using subject embeddings
    No longer requires a pre-trained face ID model
    """
    logger.info(f"Starting joint expression prediction training on device: {device}")
    
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
    
    logger.info(f"Training dataset loaded with {len(dataloader)} batches")
    
    # Estimate training time (more realistic for CPU training)
    estimated_time_per_epoch = len(dataloader) * 126  # 125 seconds per batch
    total_estimated_time = estimated_time_per_epoch * num_epochs / 3600  # Convert to hours
    
    print(f"📊 Training dataset loaded: {len(dataloader)} batches per epoch")
    print(f"⏱️  Estimated training time: {total_estimated_time:.1f} hours ({total_estimated_time/24:.1f} days)")
    print(f"💰 Estimated cost: ${total_estimated_time * 0.38:.1f} (at $0.38/hour)")
    
    # Initialize DINOv2 tokenizer
    dinov2_tokenizer = DINOv2Tokenizer()
    
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
                expr_max_subjects = expr_config.get('max_subjects', 3500) # Added max_subjects
                logger.info(f"Using expression transformer architecture from checkpoint: {expr_num_layers} layers, {expr_num_heads} heads, {expr_max_subjects} subjects")
            else:
                logger.warning("No architecture config found in expression transformer checkpoint, using defaults")
                
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
            # Load checkpoint to get architecture
            decoder_checkpoint = torch.load(transformer_decoder_checkpoint_path, map_location=device)
            
            # Get architecture from checkpoint config
            if 'config' in decoder_checkpoint and 'transformer_decoder' in decoder_checkpoint['config']:
                decoder_config = decoder_checkpoint['config']['transformer_decoder']
                decoder_embed_dim = decoder_config.get('embed_dim', 384)
                decoder_num_heads = decoder_config.get('num_heads', 8)
                decoder_num_layers = decoder_config.get('num_layers', 2)
                decoder_dropout = decoder_config.get('dropout', 0.1)
                max_sequence_length = decoder_config.get('max_sequence_length', 50)
                logger.info(f"Using transformer decoder architecture from checkpoint: {decoder_num_layers} layers, {decoder_num_heads} heads")
            else:
                logger.warning("No architecture config found in transformer decoder checkpoint, using defaults")
                
        except Exception as e:
            logger.warning(f"Failed to load transformer decoder checkpoint for architecture: {str(e)}, using defaults")
    
    # Initialize joint model with separate architectures for each component
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
            
        except Exception as e:
            raise RuntimeError(f"Failed to load expression transformer checkpoint: {str(e)}")
    else:
        logger.info("No expression transformer checkpoint provided - training from scratch")
    
    # Load transformer decoder weights if checkpoint provided
    if transformer_decoder_checkpoint_path is not None:
        logger.info(f"Loading transformer decoder weights from checkpoint: {transformer_decoder_checkpoint_path}")
        try:
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
    
    # Initialize face ID model and load pre-trained checkpoint
    # if not os.path.exists(face_id_checkpoint_path):
    #     raise FileNotFoundError(
    #         f"Face ID model checkpoint not found: {face_id_checkpoint_path}\n"
    #         f"Please train the face ID model first using train_face_id.py"
    #     )
    
    # # Load checkpoint to get architecture
    # checkpoint = torch.load(face_id_checkpoint_path, map_location=device)
    
    # # Get architecture from checkpoint config
    # if 'config' in checkpoint and 'face_id_model' in checkpoint['config']:
    #     face_id_config = checkpoint['config']['face_id_model']
    #     embed_dim = face_id_config.get('embed_dim', 384)
    #     num_heads = face_id_config.get('num_heads', 8)
    #     num_layers = face_id_config.get('num_layers', 2)
    #     dropout = face_id_config.get('dropout', 0.1)
    #     logger.info(f"Loading face ID model with architecture from checkpoint: {num_layers} layers, {num_heads} heads")
    # else:
    #     # Fallback to default architecture
    #     embed_dim, num_heads, num_layers, dropout = 384, 8, 2, 0.1
    #     logger.warning("No architecture config found in checkpoint, using defaults")
    
    # # Initialize face ID model with correct architecture
    # face_id_model = FaceIDModel(
    #     embed_dim=embed_dim,
    #     num_heads=num_heads,
    #     num_layers=num_layers,
    #     dropout=dropout
    # ).to(device)
    
    # # Load face ID model checkpoint
    # logger.info(f"Loading face ID model from checkpoint: {face_id_checkpoint_path}")
    # try:
    #     if 'face_id_model_state_dict' in checkpoint:
    #         face_id_model.load_state_dict(checkpoint['face_id_model_state_dict'])
    #         logger.info(f"✅ Successfully loaded face ID model from epoch {checkpoint.get('epoch', 'unknown')}")
    #         print(f"✅ Successfully loaded face ID model from: {face_id_checkpoint_path}")
    #     else:
    #         # Try loading the entire checkpoint as state dict (for compatibility)
    #         face_id_model.load_state_dict(checkpoint)
    #         logger.info("✅ Successfully loaded face ID model state dict directly")
    #         print(f"✅ Successfully loaded face ID model from: {face_id_checkpoint_path}")
    # except Exception as e:
    #     raise RuntimeError(f"Failed to load face ID model checkpoint: {str(e)}")
    
    # # Freeze face ID model parameters
    # for param in face_id_model.parameters():
    #     param.requires_grad = False
    
    # logger.info("Face ID model loaded and frozen successfully")
    # print(f"🔒 Face ID model frozen (parameters not trainable)")
    
    # Initialize loss function
    criterion = ExpressionPredictionLoss().to(device)
    
    # Initialize optimizer (train both expression transformer and transformer decoder)
    trainable_params = list(joint_model.expression_transformer.parameters()) + list(joint_model.transformer_decoder.parameters())
    optimizer = optim.Adam(trainable_params, lr=learning_rate)
    
    logger.info("Expression transformer and transformer decoder parameters will be trained")
    
    # Initialize TensorBoard with unique job ID
    job_id = str(uuid.uuid4())[:8]  # Short unique ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/exp_pred_training_{job_id}_{timestamp}"
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
            
            # Compute loss for each clip
            batch_loss = 0.0
            num_valid_clips = 0
            
            for clip_idx, (clip_expression_tokens, clip_predicted_next_token) in enumerate(zip(all_expression_tokens, all_predicted_next_tokens)):
                if clip_expression_tokens.shape[0] > 1:
                    # Use the last token as the target
                    actual_next_token = clip_expression_tokens[-1:]  # (1, 1, 384)
                    clip_loss = criterion(clip_predicted_next_token, actual_next_token)
                    batch_loss += clip_loss
                    num_valid_clips += 1
                # Single frame clips are skipped (no prediction possible)
            
            # Average loss across clips
            if num_valid_clips > 0:
                loss = batch_loss / num_valid_clips
            else:
                # No valid clips (all single frames)
                loss = torch.tensor(0.0, device=device)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            num_batches += 1
            
            # Log to TensorBoard
            writer.add_scalar('Training/Batch_Loss', loss.item(), total_steps)
            writer.add_scalar('Training/Avg_Loss', epoch_loss / num_batches, total_steps)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{epoch_loss / num_batches:.4f}'
            })
            
            total_steps += 1
        
        avg_loss = epoch_loss / num_batches
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_loss:.4f}")
        
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
            # Joint model
            joint_epoch_path = os.path.join(checkpoint_dir, f"joint_expression_prediction_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'joint_model_state_dict': joint_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'avg_loss': avg_loss,
            }, joint_epoch_path)
            
            # Expression transformer
            expr_epoch_path = os.path.join(checkpoint_dir, f"expression_transformer_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'expression_transformer_state_dict': joint_model.expression_transformer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'avg_loss': avg_loss,
                'config': {
                    'expression_model': {
                        'embed_dim': expr_embed_dim,
                        'num_heads': expr_num_heads,
                        'num_layers': expr_num_layers,
                        'dropout': expr_dropout,
                        'max_subjects': expr_max_subjects # Added max_subjects
                    }
                }
            }, expr_epoch_path)
            
            # Transformer decoder
            decoder_epoch_path = os.path.join(checkpoint_dir, f"transformer_decoder_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'transformer_decoder_state_dict': joint_model.transformer_decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'avg_loss': avg_loss,
                'config': {
                    'transformer_decoder': {
                        'embed_dim': decoder_embed_dim,
                        'num_heads': decoder_num_heads,
                        'num_layers': decoder_num_layers,
                        'dropout': decoder_dropout,
                        'max_sequence_length': max_sequence_length
                    }
                }
            }, decoder_epoch_path)
            
            logger.info(f"Saved epoch checkpoints: {joint_epoch_path}, {expr_epoch_path}, {decoder_epoch_path}")
    
    # Log model parameters to TensorBoard
    for name, param in joint_model.named_parameters():
        if param.requires_grad:
            writer.add_histogram(f'Parameters/{name}', param.data, 0)
    
    # Close TensorBoard writer
    writer.close()
    
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
            
            # Compute loss for each clip
            batch_loss = 0.0
            num_valid_clips = 0
            
            for clip_idx, (clip_expression_tokens, clip_predicted_next_token) in enumerate(zip(all_expression_tokens, all_predicted_next_tokens)):
                if clip_expression_tokens.shape[0] > 1:
                    # Use the last token as the target
                    actual_next_token = clip_expression_tokens[-1:]  # (1, 1, 384)
                    clip_loss = criterion(clip_predicted_next_token, actual_next_token)
                    batch_loss += clip_loss
                    num_valid_clips += 1
                # Single frame clips are skipped (no prediction possible)
            
            # Average loss across clips
            if num_valid_clips > 0:
                loss = batch_loss / num_valid_clips
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
    dinov2_tokenizer = DINOv2Tokenizer()
    
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
    
    # Test loss function
    criterion = ExpressionPredictionLoss()
    total_loss = 0.0
    num_valid_clips = 0
    
    for clip_expression_tokens, clip_predicted_next_token in zip(all_expression_tokens, all_predicted_next_tokens):
        if clip_expression_tokens.shape[0] > 1:
            actual_next_token = clip_expression_tokens[-1:]
            clip_loss = criterion(clip_predicted_next_token, actual_next_token)
            total_loss += clip_loss.item()
            num_valid_clips += 1
    
    if num_valid_clips > 0:
        avg_loss = total_loss / num_valid_clips
        print(f"Average loss: {avg_loss:.4f}")
    else:
        print("No valid clips for loss calculation")
    
    print("✅ Joint expression prediction test passed!")


if __name__ == "__main__":
    test_joint_expression_prediction() 