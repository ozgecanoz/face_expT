#!/usr/bin/env python3
"""
Debug script to trace DINOv2 forward pass and see where patch count changes
"""

import torch
import timm

def debug_forward_pass():
    """Debug the DINOv2 forward pass step by step"""
    
    # Load model
    model = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True)
    
    print("=== Model Configuration ===")
    print(f"Patch size: {model.patch_embed.patch_size}")
    print(f"Grid size: {model.patch_embed.grid_size}")
    print(f"Num patches: {model.patch_embed.num_patches}")
    print(f"Positional embedding shape: {model.pos_embed.shape}")
    
    # Create input
    x = torch.randn(1, 3, 518, 518)
    print(f"\n=== Input ===")
    print(f"Input shape: {x.shape}")
    
    # Step 1: Patch embedding
    print(f"\n=== Step 1: Patch Embedding ===")
    patch_embed = model.patch_embed(x)
    print(f"Patch embed output shape: {patch_embed.shape}")
    print(f"Expected shape: (1, 1369, 384)")
    print(f"Actual patches: {patch_embed.shape[1]}")
    
    # Step 2: Add positional embeddings
    print(f"\n=== Step 2: Positional Embeddings ===")
    print(f"Pos embed shape: {model.pos_embed.shape}")
    print(f"Patch embed shape: {patch_embed.shape}")
    
    # Check if shapes match
    if patch_embed.shape[1] != model.pos_embed.shape[1]:
        print(f"❌ Shape mismatch! Patch embed: {patch_embed.shape[1]}, Pos embed: {model.pos_embed.shape[1]}")
        
        # Try to understand why
        print(f"\n=== Debugging Shape Mismatch ===")
        print(f"Model was likely trained on a different image size")
        print(f"Let's check what image size the pos embed expects:")
        
        # Calculate expected image size from pos embed
        pos_embed_patches = model.pos_embed.shape[1] - 1  # Subtract class token
        expected_grid_size = int(pos_embed_patches ** 0.5)
        expected_img_size = expected_grid_size * 14
        print(f"Pos embed has {pos_embed_patches} patches")
        print(f"Expected grid size: {expected_grid_size}x{expected_grid_size}")
        print(f"Expected image size: {expected_img_size}x{expected_img_size}")
        
    else:
        print(f"✅ Shapes match!")
    
    # Step 3: Add pos embeddings (if shapes match)
    if patch_embed.shape[1] == model.pos_embed.shape[1]:
        x_with_pos = patch_embed + model.pos_embed
        print(f"\n=== Step 3: Added Positional Embeddings ===")
        print(f"Shape after adding pos embed: {x_with_pos.shape}")
    else:
        print(f"\n=== Step 3: Skipped (shape mismatch) ===")
    
    # Step 4: Try full forward pass
    print(f"\n=== Step 4: Full Forward Pass ===")
    try:
        with torch.no_grad():
            output = model.forward_features(x)
            print(f"Full forward pass output shape: {output.shape}")
            print(f"Expected: (1, 1370, 384) - 1369 patches + 1 class token")
            print(f"Actual patches: {output.shape[1] - 1}")
            print(f"Actual class tokens: {output.shape[1] - output.shape[1] + 1}")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
    
    # Step 5: Check if model supports dynamic image sizes
    print(f"\n=== Step 5: Model Flexibility ===")
    print(f"Model supports dynamic image sizes: {hasattr(model, 'flex_sizes')}")
    print(f"Model has interpolate_pos_encoding: {hasattr(model, 'interpolate_pos_encoding')}")

if __name__ == "__main__":
    debug_forward_pass() 