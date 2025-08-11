#!/usr/bin/env python3
"""
Test script for DINOv2BaseTokenizer
"""

import torch
import logging
import sys
import os

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.dinov2_tokenizer import DINOv2BaseTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_base_tokenizer():
    """Test the DINOv2BaseTokenizer class"""
    logger.info("Testing DINOv2BaseTokenizer...")
    
    try:
        # Initialize tokenizer
        tokenizer = DINOv2BaseTokenizer(device="cpu")
        
        # Test properties
        logger.info(f"Input size: {tokenizer.get_input_size()}")
        logger.info(f"Patch size: {tokenizer.get_patch_size()}")
        logger.info(f"Grid size: {tokenizer.get_grid_size()}")
        logger.info(f"Number of patches: {tokenizer.get_num_patches()}")
        logger.info(f"Embedding dimension: {tokenizer.get_embed_dim()}")
        
        # Test with dummy input
        batch_size = 2
        x = torch.randn(batch_size, 3, 518, 518)
        
        logger.info(f"Input shape: {x.shape}")
        
        # Forward pass
        patch_tokens, pos_emb = tokenizer(x)
        
        logger.info(f"Patch tokens shape: {patch_tokens.shape}")
        logger.info(f"Positional embeddings shape: {pos_emb.shape}")
        
        # Verify shapes
        expected_patches = 1369
        expected_embed_dim = 768
        
        assert patch_tokens.shape == (batch_size, expected_patches, expected_embed_dim), \
            f"Expected patch tokens shape ({batch_size}, {expected_patches}, {expected_embed_dim}), got {patch_tokens.shape}"
        
        assert pos_emb.shape == (batch_size, expected_patches, expected_embed_dim), \
            f"Expected pos embeddings shape ({batch_size}, {expected_patches}, {expected_embed_dim}), got {pos_emb.shape}"
        
        logger.info("✅ DINOv2BaseTokenizer test passed!")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.error("Install transformers with: pip install transformers")
        return False
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_base_tokenizer()
    if success:
        logger.info("🎉 All tests passed!")
        sys.exit(0)
    else:
        logger.error("💥 Tests failed!")
        sys.exit(1) 