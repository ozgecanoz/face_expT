#!/usr/bin/env python3
"""
TensorBoard utilities for robust parameter logging
"""

import torch
import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def safe_add_histogram(writer, tag: str, tensor: torch.Tensor, global_step: int) -> bool:
    """
    Safely add histogram to TensorBoard with NumPy compatibility handling.
    
    Args:
        writer: TensorBoard SummaryWriter
        tag: Tag for the histogram
        tensor: Tensor to log
        global_step: Global step number
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Convert to CPU and detach to avoid device issues
        tensor_cpu = tensor.cpu().detach()
        
        # Try different conversion methods for NumPy compatibility
        try:
            # Method 1: Direct conversion
            tensor_np = tensor_cpu.numpy()
        except Exception:
            try:
                # Method 2: Convert to float32 first
                tensor_np = tensor_cpu.float().numpy()
            except Exception:
                try:
                    # Method 3: Convert to list first
                    tensor_np = np.array(tensor_cpu.tolist())
                except Exception as e:
                    logger.warning(f"Could not convert tensor to numpy for histogram {tag}: {e}")
                    return False
        
        # Add histogram
        writer.add_histogram(tag, tensor_np, global_step)
        return True
        
    except Exception as e:
        logger.warning(f"Could not add histogram {tag}: {e}")
        return False


def safe_add_parameter_histograms(writer, model: torch.nn.Module, global_step: int = 0) -> Dict[str, bool]:
    """
    Safely add parameter histograms to TensorBoard.
    
    Args:
        writer: TensorBoard SummaryWriter
        model: PyTorch model
        global_step: Global step number
        
    Returns:
        Dictionary mapping parameter names to success status
    """
    results = {}
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            success = safe_add_histogram(writer, f'Parameters/{name}', param.data, global_step)
            results[name] = success
    
    return results


def add_parameter_statistics(writer, model: torch.nn.Module, global_step: int = 0) -> Dict[str, Dict[str, float]]:
    """
    Add parameter statistics (mean, std, min, max) to TensorBoard as scalars.
    
    Args:
        writer: TensorBoard SummaryWriter
        model: PyTorch model
        global_step: Global step number
        
    Returns:
        Dictionary mapping parameter names to their statistics
    """
    statistics = {}
    
    try:
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_data = param.data.cpu().detach()
                
                stats = {
                    'mean': param_data.mean().item(),
                    'std': param_data.std().item(),
                    'min': param_data.min().item(),
                    'max': param_data.max().item()
                }
                
                # Add to TensorBoard
                writer.add_scalar(f'Parameters/{name}_mean', stats['mean'], global_step)
                writer.add_scalar(f'Parameters/{name}_std', stats['std'], global_step)
                writer.add_scalar(f'Parameters/{name}_min', stats['min'], global_step)
                writer.add_scalar(f'Parameters/{name}_max', stats['max'], global_step)
                
                statistics[name] = stats
                
    except Exception as e:
        logger.warning(f"Could not add parameter statistics: {e}")
    
    return statistics


def log_model_parameters(writer, model: torch.nn.Module, global_step: int = 0) -> Dict[str, Any]:
    """
    Comprehensive parameter logging with fallback options.
    
    Args:
        writer: TensorBoard SummaryWriter
        model: PyTorch model
        global_step: Global step number
        
    Returns:
        Dictionary with logging results
    """
    results = {
        'histograms_successful': 0,
        'histograms_failed': 0,
        'statistics_successful': 0,
        'statistics_failed': 0,
        'parameter_count': 0
    }
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    results['parameter_count'] = total_params
    
    # Log parameter counts
    writer.add_scalar('Model/Total_Parameters', total_params, global_step)
    writer.add_scalar('Model/Trainable_Parameters', trainable_params, global_step)
    
    # Try histogram logging first
    histogram_results = safe_add_parameter_histograms(writer, model, global_step)
    
    for name, success in histogram_results.items():
        if success:
            results['histograms_successful'] += 1
        else:
            results['histograms_failed'] += 1
    
    # If histogram logging failed for most parameters, try statistics
    if results['histograms_failed'] > results['histograms_successful']:
        logger.info("Histogram logging had issues, falling back to parameter statistics")
        statistics = add_parameter_statistics(writer, model, global_step)
        results['statistics_successful'] = len(statistics)
        results['statistics_failed'] = len(histogram_results) - len(statistics)
    
    return results


def test_tensorboard_logging():
    """Test the TensorBoard logging utilities."""
    print("🧪 Testing TensorBoard logging utilities...")
    
    # Create a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 5)
    )
    
    # Create a dummy writer (we won't actually write files)
    from torch.utils.tensorboard import SummaryWriter
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as temp_dir:
        writer = SummaryWriter(temp_dir)
        
        # Test parameter logging
        results = log_model_parameters(writer, model, global_step=0)
        
        print(f"📊 Logging Results:")
        print(f"   Total parameters: {results['parameter_count']}")
        print(f"   Histograms successful: {results['histograms_successful']}")
        print(f"   Histograms failed: {results['histograms_failed']}")
        print(f"   Statistics successful: {results['statistics_successful']}")
        print(f"   Statistics failed: {results['statistics_failed']}")
        
        writer.close()
    
    print("✅ TensorBoard logging test completed!")
    return results


if __name__ == "__main__":
    test_tensorboard_logging() 