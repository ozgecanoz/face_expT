#!/usr/bin/env python3
"""
Debug script to investigate DINOv2 patch count
"""

import torch
import timm
import numpy as np

def debug_patch_count():
    """Debug the actual patch count from DINOv2"""
    
    # Load DINOv2 model
    model_name = 'vit_small_patch14_dinov2.lvd142m'
    model = timm.create_model(model_name, pretrained=True)
    
    print(f"Model: {model_name}")
    print(f"Patch embed patch size: {model.patch_embed.patch_size}")
    print(f"Patch embed grid size: {model.patch_embed.grid_size}")
    print(f"Patch embed num patches: {model.patch_embed.num_patches}")
    print(f"Positional embedding shape: {model.pos_embed.shape}")
    
    # Test with different image sizes
    test_sizes = [518, 224, 448, 512]
    
    for size in test_sizes:
        print(f"\nTesting with image size: {size}x{size}")
        
        # Create dummy input
        x = torch.randn(1, 3, size, size)
        
        # Get patch embedding
        patch_embed = model.patch_embed(x)
        print(f"Patch embedding shape: {patch_embed.shape}")
        
        # Calculate expected patches
        patch_size = model.patch_embed.patch_size
        expected_patches = (size // patch_size) ** 2
        print(f"Expected patches: {expected_patches}")
        print(f"Actual patches: {patch_embed.shape[1]}")
        
        if expected_patches != patch_embed.shape[1]:
            print(f"❌ Mismatch! Expected {expected_patches}, got {patch_embed.shape[1]}")
        else:
            print(f"✅ Match! Both {patch_embed.shape[1]}")

if __name__ == "__main__":
    debug_patch_count() 