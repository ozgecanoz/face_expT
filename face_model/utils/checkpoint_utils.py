#!/usr/bin/env python3
"""
Checkpoint utilities for saving and loading model configurations
Provides reusable functions for consistent checkpoint handling across the codebase
"""

import torch
import os
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


def save_checkpoint(
    model_state_dict: Dict[str, torch.Tensor],
    optimizer_state_dict: Dict[str, Any],
    scheduler_state_dict: Dict[str, Any],
    epoch: int,
    avg_loss: float,
    total_steps: int,
    config: Dict[str, Any],
    checkpoint_path: str,
    checkpoint_type: str = "joint"
) -> None:
    """
    Save a checkpoint with comprehensive configuration
    
    Args:
        model_state_dict: Model state dictionary
        optimizer_state_dict: Optimizer state dictionary
        scheduler_state_dict: Scheduler state dictionary
        epoch: Current epoch number
        avg_loss: Average loss for this epoch
        total_steps: Total training steps completed
        config: Configuration dictionary containing model and training parameters
        checkpoint_path: Path where to save the checkpoint
        checkpoint_type: Type of checkpoint ("joint", "expression_transformer", "transformer_decoder", "expression_reconstruction")
    """
    # Ensure checkpoint directory exists
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    # Prepare checkpoint data
    checkpoint_data = {
        'epoch': epoch,
        'avg_loss': avg_loss,
        'total_steps': total_steps,
        'optimizer_state_dict': optimizer_state_dict,
        'scheduler_state_dict': scheduler_state_dict,
        'config': config
    }
    
    # Add model state dict with appropriate key
    if checkpoint_type == "joint":
        checkpoint_data['joint_model_state_dict'] = model_state_dict
    elif checkpoint_type == "expression_transformer":
        checkpoint_data['expression_transformer_state_dict'] = model_state_dict
    elif checkpoint_type == "transformer_decoder":
        checkpoint_data['transformer_decoder_state_dict'] = model_state_dict
    elif checkpoint_type == "expression_reconstruction":
        checkpoint_data['expression_reconstruction_state_dict'] = model_state_dict
    else:
        raise ValueError(f"Unknown checkpoint type: {checkpoint_type}")
    
    # Save checkpoint
    torch.save(checkpoint_data, checkpoint_path)
    logger.info(f"Saved {checkpoint_type} checkpoint to: {checkpoint_path}")


def load_checkpoint_config(
    checkpoint_path: str,
    device: str = "cpu"
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load configuration from a checkpoint file
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the checkpoint on
        
    Returns:
        Tuple of (checkpoint_data, extracted_config)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint_data = torch.load(checkpoint_path, map_location=device)
    
    # Extract configuration
    config = checkpoint_data.get('config', {})
    
    logger.info(f"Loaded checkpoint from: {checkpoint_path}")
    logger.info(f"Checkpoint epoch: {checkpoint_data.get('epoch', 'unknown')}")
    
    return checkpoint_data, config


def extract_model_config(
    config: Dict[str, Any],
    default_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Extract model configuration from checkpoint config
    
    Args:
        config: Configuration dictionary from checkpoint
        default_params: Default parameters to use as fallback
        
    Returns:
        Dictionary of extracted model parameters
    """
    extracted_params = default_params.copy()
    
    # Extract expression transformer parameters
    if 'expression_model' in config:
        expr_config = config['expression_model']
        for param_name in ['embed_dim', 'num_heads', 'num_layers', 'dropout', 'ff_dim', 'grid_size']:
            if param_name in expr_config:
                extracted_params[f'expr_{param_name}'] = expr_config[param_name]
                logger.info(f"Using expression transformer {param_name}: {expr_config[param_name]}")
    
    # Extract transformer decoder parameters
    if 'transformer_decoder' in config:
        decoder_config = config['transformer_decoder']
        for param_name in ['embed_dim', 'num_heads', 'num_layers', 'dropout', 'max_sequence_length']:
            if param_name in decoder_config:
                if param_name == 'max_sequence_length':
                    extracted_params[param_name] = decoder_config[param_name]
                else:
                    extracted_params[f'decoder_{param_name}'] = decoder_config[param_name]
                logger.info(f"Using transformer decoder {param_name}: {decoder_config[param_name]}")
    
    # Extract expression reconstruction parameters
    if 'expression_reconstruction' in config:
        recon_config = config['expression_reconstruction']
        for param_name in ['embed_dim', 'num_cross_attention_layers', 'num_self_attention_layers', 'num_heads', 'ff_dim', 'dropout', 'max_subjects']:
            if param_name in recon_config:
                extracted_params[f'recon_{param_name}'] = recon_config[param_name]
                logger.info(f"Using expression reconstruction {param_name}: {recon_config[param_name]}")
                # Note: max_subjects now controls both fixed base and learnable delta embeddings
                # All parameters are now properly stored in model config for checkpointing
    
    # Extract loss function parameters
    if 'loss_function' in config:
        loss_config = config['loss_function']
        for param_name in ['lambda_prediction', 'lambda_temporal', 'lambda_diversity']:
            if param_name in loss_config:
                extracted_params[param_name] = loss_config[param_name]
                logger.info(f"Using loss function {param_name}: {loss_config[param_name]}")
    
    # Extract training parameters (for logging)
    if 'training' in config:
        train_config = config['training']
        logger.info(f"Training parameters from checkpoint: "
                   f"LR={train_config.get('learning_rate', 'unknown')}, "
                   f"Batch={train_config.get('batch_size', 'unknown')}, "
                   f"Epochs={train_config.get('num_epochs', 'unknown')}")
    
    return extracted_params


def create_comprehensive_config(
    expr_embed_dim: int,
    expr_num_heads: int,
    expr_num_layers: int,
    expr_dropout: float,
    expr_ff_dim: int,
    expr_grid_size: int = 37,
    decoder_embed_dim: int = None,
    decoder_num_heads: int = None,
    decoder_num_layers: int = None,
    decoder_dropout: float = None,
    max_sequence_length: int = None,
    lambda_prediction: float = None,
    lambda_temporal: float = None,
    lambda_diversity: float = None,
    learning_rate: float = None,
    batch_size: int = None,
    num_epochs: int = None,
    warmup_steps: int = None,
    min_lr: float = None,
    # Reconstruction model parameters (optional)
    recon_embed_dim: int = None,
    recon_num_cross_layers: int = None,
    recon_num_self_layers: int = None,
    recon_num_heads: int = None,
    recon_ff_dim: int = None,
    recon_dropout: float = None,
    recon_max_subjects: int = None,
    # Loss parameters (optional)
    lambda_reconstruction: float = None,
    # Scheduler parameters (optional)
    initial_lambda_prediction: float = None,
    initial_lambda_temporal: float = None,
    initial_lambda_diversity: float = None,
    warmup_lambda_prediction: float = None,
    warmup_lambda_temporal: float = None,
    warmup_lambda_diversity: float = None,
    final_lambda_prediction: float = None,
    final_lambda_temporal: float = None,
    final_lambda_diversity: float = None,
    # Reconstruction scheduler parameters (optional)
    initial_lambda_reconstruction: float = None,
    warmup_lambda_reconstruction: float = None,
    final_lambda_reconstruction: float = None,
    # Supervised model parameters (optional)
    num_classes: int = None,
    pca_json_path: str = None
) -> Dict[str, Any]:
    """
    Create a comprehensive configuration dictionary for checkpoint saving
    
    Args:
        expr_embed_dim: Expression transformer embedding dimension
        expr_num_heads: Expression transformer number of heads
        expr_num_layers: Expression transformer number of layers
        expr_dropout: Expression transformer dropout
        expr_ff_dim: Expression transformer feed-forward dimension
        expr_grid_size: Expression transformer grid size
        decoder_embed_dim: Decoder embedding dimension (optional)
        decoder_num_heads: Decoder number of heads (optional)
        decoder_num_layers: Decoder number of layers (optional)
        decoder_dropout: Decoder dropout (optional)
        max_sequence_length: Maximum sequence length (optional)
        lambda_prediction: Prediction loss weight (optional)
        lambda_temporal: Temporal loss weight (optional)
        lambda_diversity: Diversity loss weight (optional)
        learning_rate: Learning rate (optional)
        batch_size: Batch size (optional)
        num_epochs: Number of epochs (optional)
        warmup_steps: Warmup steps (optional)
        min_lr: Minimum learning rate (optional)
        recon_embed_dim: Reconstruction model embedding dimension (optional)
        recon_num_cross_layers: Reconstruction cross-attention layers (optional)
        recon_num_self_layers: Reconstruction self-attention layers (optional)
        recon_num_heads: Reconstruction number of heads (optional)
        recon_ff_dim: Reconstruction feed-forward dimension (optional)
        recon_dropout: Reconstruction dropout (optional)
        recon_max_subjects: Reconstruction max subjects (optional)
        lambda_reconstruction: Reconstruction loss weight (optional)
        initial_lambda_prediction: Initial prediction loss weight (optional)
        warmup_lambda_prediction: Warmup prediction loss weight (optional)
        final_lambda_prediction: Final prediction loss weight (optional)
        initial_lambda_temporal: Initial temporal loss weight (optional)
        warmup_lambda_temporal: Warmup temporal loss weight (optional)
        final_lambda_temporal: Final temporal loss weight (optional)
        initial_lambda_diversity: Initial diversity loss weight (optional)
        warmup_lambda_diversity: Warmup diversity loss weight (optional)
        final_lambda_diversity: Final diversity loss weight (optional)
        initial_lambda_reconstruction: Initial reconstruction loss weight (optional)
        warmup_lambda_reconstruction: Warmup reconstruction loss weight (optional)
        final_lambda_reconstruction: Final reconstruction loss weight (optional)
        num_classes: Number of classes for supervised model (optional)
        pca_json_path: Path to PCA projection JSON (optional)
        
    Returns:
        Comprehensive configuration dictionary
    """
    config = {
        'expression_model': {
            'embed_dim': expr_embed_dim,
            'num_heads': expr_num_heads,
            'num_layers': expr_num_layers,
            'dropout': expr_dropout,
            'ff_dim': expr_ff_dim,
            'grid_size': expr_grid_size
        }
    }
    
    # Add transformer decoder config if provided
    if all(param is not None for param in [decoder_embed_dim, decoder_num_heads, decoder_num_layers, decoder_dropout, max_sequence_length]):
        config['transformer_decoder'] = {
            'embed_dim': decoder_embed_dim,
            'num_heads': decoder_num_heads,
            'num_layers': decoder_num_layers,
            'dropout': decoder_dropout,
            'max_sequence_length': max_sequence_length
        }
    
    # Add reconstruction model config if provided
    if all(param is not None for param in [recon_embed_dim, recon_num_cross_layers, recon_num_self_layers, recon_num_heads, recon_ff_dim, recon_dropout]):
        config['expression_reconstruction'] = {
            'embed_dim': recon_embed_dim,
            'num_cross_attention_layers': recon_num_cross_layers,
            'num_self_attention_layers': recon_num_self_layers,
            'num_heads': recon_num_heads,
            'ff_dim': recon_ff_dim,
            'dropout': recon_dropout,
            'max_subjects': recon_max_subjects if recon_max_subjects is not None else 3500
        }
    
    # Add supervised model config if provided
    if num_classes is not None:
        config['supervised_model'] = {
            'num_classes': num_classes,
            'pca_json_path': pca_json_path
        }
    
    # Add loss function config
    loss_config = {}
    if lambda_prediction is not None:
        loss_config['lambda_prediction'] = lambda_prediction
    if lambda_temporal is not None:
        loss_config['lambda_temporal'] = lambda_temporal
    if lambda_diversity is not None:
        loss_config['lambda_diversity'] = lambda_diversity
    if lambda_reconstruction is not None:
        loss_config['lambda_reconstruction'] = lambda_reconstruction
    
    if loss_config:
        config['loss_function'] = loss_config
    
    # Add training config if provided
    if all(param is not None for param in [learning_rate, batch_size, num_epochs, warmup_steps, min_lr]):
        config['training'] = {
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'warmup_steps': warmup_steps,
            'min_lr': min_lr
        }
    
    # Add scheduler parameters if provided
    scheduler_config = {}
    
    # Check if prediction scheduler parameters are provided
    if all(param is not None for param in [
        initial_lambda_prediction, initial_lambda_temporal, initial_lambda_diversity,
        warmup_lambda_prediction, warmup_lambda_temporal, warmup_lambda_diversity,
        final_lambda_prediction, final_lambda_temporal, final_lambda_diversity
    ]):
        scheduler_config['prediction'] = {
            'initial_weights': {
                'lambda_prediction': initial_lambda_prediction,
                'lambda_temporal': initial_lambda_temporal,
                'lambda_diversity': initial_lambda_diversity
            },
            'warmup_weights': {
                'lambda_prediction': warmup_lambda_prediction,
                'lambda_temporal': warmup_lambda_temporal,
                'lambda_diversity': warmup_lambda_diversity
            },
            'final_weights': {
                'lambda_prediction': final_lambda_prediction,
                'lambda_temporal': final_lambda_temporal,
                'lambda_diversity': final_lambda_diversity
            }
        }
    
    # Check if reconstruction scheduler parameters are provided
    if all(param is not None for param in [
        initial_lambda_reconstruction, initial_lambda_temporal, initial_lambda_diversity,
        warmup_lambda_reconstruction, warmup_lambda_temporal, warmup_lambda_diversity,
        final_lambda_reconstruction, final_lambda_temporal, final_lambda_diversity
    ]):
        scheduler_config['reconstruction'] = {
            'initial_weights': {
                'lambda_reconstruction': initial_lambda_reconstruction,
                'lambda_temporal': initial_lambda_temporal,
                'lambda_diversity': initial_lambda_diversity
            },
            'warmup_weights': {
                'lambda_reconstruction': warmup_lambda_reconstruction,
                'lambda_temporal': warmup_lambda_temporal,
                'lambda_diversity': warmup_lambda_diversity
            },
            'final_weights': {
                'lambda_reconstruction': final_lambda_reconstruction,
                'lambda_temporal': final_lambda_temporal,
                'lambda_diversity': final_lambda_diversity
            }
        }
    
    if scheduler_config:
        config['scheduler'] = scheduler_config
    
    return config


def validate_checkpoint_compatibility(
    checkpoint_path: str,
    expected_params: Dict[str, Any],
    device: str = "cpu"
) -> bool:
    """
    Validate that a checkpoint contains the expected parameters
    
    Args:
        checkpoint_path: Path to checkpoint file
        expected_params: Dictionary of expected parameter values
        device: Device to load checkpoint on
        
    Returns:
        True if checkpoint is compatible, False otherwise
    """
    try:
        checkpoint_data, config = load_checkpoint_config(checkpoint_path, device)
        extracted_params = extract_model_config(config, {})
        
        # Check that all expected parameters match
        for param_name, expected_value in expected_params.items():
            if param_name in extracted_params:
                actual_value = extracted_params[param_name]
                if actual_value != expected_value:
                    logger.warning(f"Parameter mismatch: {param_name} expected {expected_value}, got {actual_value}")
                    return False
            else:
                logger.warning(f"Missing parameter in checkpoint: {param_name}")
                return False
        
        logger.info("✅ Checkpoint compatibility validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Checkpoint validation failed: {e}")
        return False


def get_checkpoint_info(checkpoint_path: str, device: str = "cpu") -> Dict[str, Any]:
    """
    Get information about a checkpoint without loading the full model
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint on
        
    Returns:
        Dictionary containing checkpoint information
    """
    try:
        checkpoint_data, config = load_checkpoint_config(checkpoint_path, device)
        
        info = {
            'epoch': checkpoint_data.get('epoch', 'unknown'),
            'avg_loss': checkpoint_data.get('avg_loss', 'unknown'),
            'total_steps': checkpoint_data.get('total_steps', 'unknown'),
            'has_optimizer': 'optimizer_state_dict' in checkpoint_data,
            'has_scheduler': 'scheduler_state_dict' in checkpoint_data,
            'config_sections': list(config.keys()) if config else []
        }
        
        # Add model-specific info
        if 'joint_model_state_dict' in checkpoint_data:
            info['checkpoint_type'] = 'joint'
        elif 'expression_transformer_state_dict' in checkpoint_data:
            info['checkpoint_type'] = 'expression_transformer'
        elif 'transformer_decoder_state_dict' in checkpoint_data:
            info['checkpoint_type'] = 'transformer_decoder'
        elif 'expression_reconstruction_state_dict' in checkpoint_data:
            info['checkpoint_type'] = 'expression_reconstruction'
        else:
            info['checkpoint_type'] = 'unknown'
        
        return info
        
    except Exception as e:
        logger.error(f"Failed to get checkpoint info: {e}")
        return {'error': str(e)} 