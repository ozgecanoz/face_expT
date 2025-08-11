#!/usr/bin/env python3
"""
Test script for GPU memory management in compute_pca_directions
"""

import torch
import logging
import sys
import os

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from compute_pca_directions import compute_optimal_batch_size, monitor_gpu_memory

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gpu_memory_management():
    """Test GPU memory management functions"""
    logger.info("Testing GPU memory management...")
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, skipping GPU tests")
        return False
    
    # Test optimal batch size computation
    logger.info("Testing optimal batch size computation...")
    
    # Test for DINOv2 base model (768-dim embeddings)
    batch_size_768 = compute_optimal_batch_size("cuda", 518, 768)
    logger.info(f"Optimal batch size for 518x518 with 768-dim: {batch_size_768}")
    
    # Test for regular DINOv2 model (384-dim embeddings)
    batch_size_384 = compute_optimal_batch_size("cuda", 518, 384)
    logger.info(f"Optimal batch size for 518x518 with 384-dim: {batch_size_384}")
    
    # Test for 224x224 model
    batch_size_224 = compute_optimal_batch_size("cuda", 224, 384)
    logger.info(f"Optimal batch size for 224x224 with 384-dim: {batch_size_224}")
    
    # Test memory monitoring
    logger.info("Testing GPU memory monitoring...")
    monitor_gpu_memory("cuda", "Test")
    
    # Test with different target memory fractions
    logger.info("Testing different memory targets...")
    for target in [0.5, 0.7, 0.9]:
        batch_size = compute_optimal_batch_size("cuda", 518, 768, target)
        logger.info(f"Target {target*100}% memory: batch size {batch_size}")
    
    logger.info("✅ GPU memory management tests completed!")
    return True

def test_cpu_fallback():
    """Test CPU fallback behavior"""
    logger.info("Testing CPU fallback...")
    
    batch_size_cpu = compute_optimal_batch_size("cpu", 518, 384)
    logger.info(f"CPU batch size: {batch_size_cpu}")
    
    # Test memory monitoring on CPU (should do nothing)
    monitor_gpu_memory("cpu", "CPU Test")
    
    logger.info("✅ CPU fallback tests completed!")
    return True

if __name__ == "__main__":
    logger.info("🧪 Testing GPU Memory Management...")
    
    success = True
    
    # Test CPU functionality
    success &= test_cpu_fallback()
    
    # Test GPU functionality if available
    if torch.cuda.is_available():
        success &= test_gpu_memory_management()
    else:
        logger.info("⚠️  CUDA not available, skipping GPU tests")
    
    if success:
        logger.info("🎉 All tests passed!")
        sys.exit(0)
    else:
        logger.error("💥 Some tests failed!")
        sys.exit(1) 