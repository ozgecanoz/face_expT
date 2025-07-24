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
    Component C: Expression Transformer (extracts expression tokens)
    Component D: Transformer Decoder (predicts next expression token)
    """
    
    def __init__(self, 
                 expr_embed_dim=384, expr_num_heads=8, expr_num_layers=2, expr_dropout=0.1,
                 decoder_embed_dim=384, decoder_num_heads=8, decoder_num_layers=2, decoder_dropout=0.1, 
                 max_sequence_length=50):
        super().__init__()
        
        # Component C: Expression Transformer (trainable)
        self.expression_transformer = ExpressionTransformer(
            embed_dim=expr_embed_dim, 
            num_heads=expr_num_heads, 
            num_layers=expr_num_layers, 
            dropout=expr_dropout
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
        logger.info(f"  Expression Transformer: {expr_num_layers} layers, {expr_num_heads} heads, {expr_embed_dim} dim")
        logger.info(f"  Transformer Decoder: {decoder_num_layers} layers, {decoder_num_heads} heads, {decoder_embed_dim} dim")
        
    def forward(self, face_images, face_id_tokens, tokenizer, clip_lengths=None):
        """
        Args:
            face_images: (total_frames, 3, 518, 518) - Input face images from multiple clips
            face_id_tokens: (total_frames, 1, 384) - Face ID tokens (same for all frames in clip)
            tokenizer: DINOv2Tokenizer instance
            clip_lengths: List of clip lengths for processing individual clips
        
        Returns:
            all_expression_tokens: List of expression tokens for each clip
            all_predicted_next_tokens: List of predicted next tokens for each clip
        """
        if clip_lengths is None:
            # Fallback: process as single clip
            return self._forward_single_clip(face_images, face_id_tokens, tokenizer)
        
        # Component A: Extract patch tokens using provided tokenizer
        patch_tokens, pos_embeddings = tokenizer(face_images)  # (total_frames, 1369, 384), (total_frames, 1369, 384)
        
        # Component C: Extract expression tokens for all frames
        all_expression_tokens = self.expression_transformer(patch_tokens, pos_embeddings, face_id_tokens)  # (total_frames, 1, 384)
        
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
    
    def _forward_single_clip(self, face_images, face_id_tokens, tokenizer):
        """
        Process a single clip (fallback for when clip_lengths is None)
        
        Args:
            face_images: (clip_length, 3, 518, 518) - Input face images for one clip
            face_id_tokens: (clip_length, 1, 384) - Face ID tokens for one clip
            tokenizer: DINOv2Tokenizer instance
        
        Returns:
            expression_tokens: (clip_length, 1, 384) - Expression tokens for all frames
            predicted_next_token: (1, 1, 384) - Predicted next token for the last frame
        """
        total_frames = face_images.shape[0]
        
        # Component A: Extract patch tokens using provided tokenizer
        patch_tokens, pos_embeddings = tokenizer(face_images)  # (clip_length, 1369, 384), (clip_length, 1369, 384)
        
        # Component C: Extract expression tokens for all frames
        expression_tokens = self.expression_transformer(patch_tokens, pos_embeddings, face_id_tokens)  # (clip_length, 1, 384)
        
        # Component D: Predict next token using transformer decoder
        if total_frames > 1:
            predicted_next_token = self.transformer_decoder(expression_tokens)  # (1, 1, 384)
        else:
            # Single frame case
            predicted_next_token = torch.zeros(1, 1, 384, device=expression_tokens.device)
        
        return expression_tokens, predicted_next_token


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


def prepare_expression_training_data(batch, face_id_model, dinov2_tokenizer, device):
    """
    Prepare data for expression prediction training
    
    Args:
        batch: Batch from dataloader
        face_id_model: Frozen face ID model
        dinov2_tokenizer: DINOv2Tokenizer instance
        device: Device to use
    
    Returns:
        face_images: (total_frames, 3, 518, 518) - All face images
        face_id_tokens: (total_frames, 1, 384) - Face ID tokens (same for all frames in clip)
        clip_lengths: List of clip lengths
    """
    # Extract frames from all clips in the batch
    all_frames = []
    face_id_tokens = []
    clip_lengths = []
    
    for frames in batch['frames']:
        # frames: (num_frames, 3, 518, 518) for one clip
        num_frames = frames.shape[0]
        all_frames.append(frames)
        clip_lengths.append(num_frames)
        
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
    face_images = torch.cat(all_frames, dim=0).to(device)  # (total_frames, 3, 518, 518)
    face_id_tokens = torch.cat(face_id_tokens, dim=0).to(device)  # (total_frames, 1, 384)
    
    return face_images, face_id_tokens, clip_lengths


def train_expression_prediction(
    dataset_path,
    face_id_checkpoint_path,
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
    device="cpu"
):
    """
    Train the joint expression prediction model
    Requires a pre-trained face ID model checkpoint
    """
    logger.info(f"Starting joint expression prediction training on device: {device}")
    
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
    
    # Determine transformer decoder architecture
    decoder_embed_dim, decoder_num_heads, decoder_num_layers, decoder_dropout = 384, 8, 2, 0.1  # Defaults
    max_sequence_length = 50  # Default
    
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
            face_images, face_id_tokens, clip_lengths = prepare_expression_training_data(
                batch, face_id_model, dinov2_tokenizer, device
            )
            
            # Forward pass through joint model
            all_expression_tokens, all_predicted_next_tokens = joint_model(face_images, face_id_tokens, dinov2_tokenizer, clip_lengths)
            
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
        
        # Log epoch metrics to TensorBoard
        writer.add_scalar('Training/Epoch_Loss', avg_loss, epoch + 1)
        
        # Validate if validation dataloader is provided
        if val_dataloader is not None:
            val_loss = validate_expression_prediction(
                joint_model, face_id_model, val_dataloader, criterion, dinov2_tokenizer, device
            )
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
            
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
                        'dropout': expr_dropout
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
    face_id_model,
    val_dataloader,
    criterion,
    dinov2_tokenizer,
    device="cpu"
):
    """
    Validate the joint expression prediction model
    """
    joint_model.eval()
    face_id_model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():  # No gradient computation for validation
        for batch_idx, batch in enumerate(val_dataloader):
            # Prepare data
            face_images, face_id_tokens, clip_lengths = prepare_expression_training_data(
                batch, face_id_model, dinov2_tokenizer, device
            )
            
            # Forward pass through joint model
            all_expression_tokens, all_predicted_next_tokens = joint_model(face_images, face_id_tokens, dinov2_tokenizer, clip_lengths)
            
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
        decoder_embed_dim=384,
        decoder_num_heads=4,
        decoder_num_layers=1,
        decoder_dropout=0.1,
        max_sequence_length=50
    )
    face_id_model = FaceIDModel()
    dinov2_tokenizer = DINOv2Tokenizer()
    
    # Create dummy input simulating clips
    # Simulate 2 clips: first clip has 5 frames, second clip has 3 frames
    clip1_frames = torch.randn(5, 3, 518, 518)  # 5 frames
    clip2_frames = torch.randn(3, 3, 518, 518)  # 3 frames
    
    # Concatenate all frames
    face_images = torch.cat([clip1_frames, clip2_frames], dim=0)  # (8, 3, 518, 518)
    clip_lengths = [5, 3]  # Length of each clip
    
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
    face_id_tokens = torch.cat(face_id_tokens, dim=0)  # (8, 1, 384)
    
    # Forward pass with pre-computed face ID tokens
    all_expression_tokens, all_predicted_next_tokens = joint_model(face_images, face_id_tokens, dinov2_tokenizer, clip_lengths)
    
    print(f"Input face images shape: {face_images.shape}")
    print(f"Clip lengths: {clip_lengths}")
    print(f"Number of clips processed: {len(all_expression_tokens)}")
    
    # Assert correct number of clips
    assert len(all_expression_tokens) == len(clip_lengths), f"Expected {len(clip_lengths)} clips, got {len(all_expression_tokens)}"
    assert len(all_predicted_next_tokens) == len(clip_lengths), f"Expected {len(clip_lengths)} predictions, got {len(all_predicted_next_tokens)}"
    
    for clip_idx, (clip_expression_tokens, clip_predicted_next_token) in enumerate(zip(all_expression_tokens, all_predicted_next_tokens)):
        print(f"Clip {clip_idx}:")
        print(f"  Expression tokens shape: {clip_expression_tokens.shape}")
        print(f"  Predicted next token shape: {clip_predicted_next_token.shape}")
        
        # Assert shapes are correct
        expected_expr_shape = (clip_lengths[clip_idx], 1, 384)
        expected_pred_shape = (1, 1, 384)
        assert clip_expression_tokens.shape == expected_expr_shape, f"Clip {clip_idx}: Expected expression tokens shape {expected_expr_shape}, got {clip_expression_tokens.shape}"
        assert clip_predicted_next_token.shape == expected_pred_shape, f"Clip {clip_idx}: Expected prediction shape {expected_pred_shape}, got {clip_predicted_next_token.shape}"
    
    print(f"Face ID tokens shape: {face_id_tokens.shape}")
    
    # Check that face ID tokens are the same within each clip
    print(f"Face ID token for clip 1 (frames 0-4): {face_id_tokens[0, 0, :5]}")  # First 5 values
    print(f"Face ID token for clip 1 (frame 1): {face_id_tokens[1, 0, :5]}")
    print(f"Face ID token for clip 2 (frames 5-7): {face_id_tokens[5, 0, :5]}")
    print(f"Face ID token for clip 2 (frame 6): {face_id_tokens[6, 0, :5]}")
    
    # Assert face ID tokens are consistent within clips
    assert torch.allclose(face_id_tokens[0, 0, :], face_id_tokens[1, 0, :]), "Face ID tokens should be same within clip 1"
    assert torch.allclose(face_id_tokens[5, 0, :], face_id_tokens[6, 0, :]), "Face ID tokens should be same within clip 2"
    
    print("\n✅ Joint Expression Prediction Model test passed!")
    
    # Test with single clip batch
    print("\n🧪 Testing single clip batch:")
    
    # Create a batch with only one clip (3 frames)
    single_clip_frames = torch.randn(3, 3, 518, 518)  # 3 frames
    single_clip_lengths = [3]  # One clip with 3 frames
    
    # Extract face ID token from first frame
    with torch.no_grad():
        first_frame_patches, first_frame_pos = dinov2_tokenizer(single_clip_frames[0:1])
        first_frame_face_id = face_id_model(first_frame_patches, first_frame_pos)  # (1, 1, 384)
    
    # Repeat face ID token for all frames
    single_clip_face_id_tokens = first_frame_face_id.repeat(3, 1, 1)  # (3, 1, 384)
    
    print(f"Single clip frames shape: {single_clip_frames.shape}")
    print(f"Single clip face ID tokens shape: {single_clip_face_id_tokens.shape}")
    print(f"Single clip lengths: {single_clip_lengths}")
    
    # Forward pass with single clip
    single_clip_expression_tokens, single_clip_predicted_next_tokens = joint_model(
        single_clip_frames, single_clip_face_id_tokens, dinov2_tokenizer, single_clip_lengths
    )
    
    print(f"Single clip expression tokens: {len(single_clip_expression_tokens)} clips")
    print(f"Single clip predicted tokens: {len(single_clip_predicted_next_tokens)} predictions")
    
    # Assert single clip batch has correct number of elements
    assert len(single_clip_expression_tokens) == 1, f"Expected 1 clip, got {len(single_clip_expression_tokens)}"
    assert len(single_clip_predicted_next_tokens) == 1, f"Expected 1 prediction, got {len(single_clip_predicted_next_tokens)}"
    
    for clip_idx, (expr_tokens, pred_tokens) in enumerate(zip(single_clip_expression_tokens, single_clip_predicted_next_tokens)):
        print(f"  Clip {clip_idx}:")
        print(f"    Expression tokens shape: {expr_tokens.shape}")
        print(f"    Predicted tokens shape: {pred_tokens.shape}")
        
        # Assert shapes are correct for single clip
        expected_expr_shape = (3, 1, 384)  # 3 frames
        expected_pred_shape = (1, 1, 384)  # 1 prediction
        assert expr_tokens.shape == expected_expr_shape, f"Single clip: Expected expression tokens shape {expected_expr_shape}, got {expr_tokens.shape}"
        assert pred_tokens.shape == expected_pred_shape, f"Single clip: Expected prediction shape {expected_pred_shape}, got {pred_tokens.shape}"
    
    print("\n✅ Single clip batch test passed!")


if __name__ == "__main__":
    test_joint_expression_prediction() 