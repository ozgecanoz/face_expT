"""
Component A: DINOv2 Tokenizer (Frozen)
Extracts patch tokens from face images using pre-trained DINOv2
"""

import torch
import torch.nn as nn
import timm
import logging

logger = logging.getLogger(__name__)


class DINOv2Tokenizer(nn.Module):
    """
    Component A: Frozen DINOv2 tokenizer
    Input: Face image (518×518×3)
    Output: 1,369 patch tokens (384-dim each) + 1 class token (384-dim)
    """
    
    def __init__(self, model_name='vit_small_patch14_dinov2.lvd142m', device='cpu'):
        super().__init__()
        
        logger.info(f"Loading DINOv2 model: {model_name}")
        
        # Load pre-trained DINOv2
        self.model = timm.create_model(model_name, pretrained=True)
        
        # Move model to specified device
        self.model = self.model.to(device)
        logger.info(f"DINOv2 model moved to device: {device}")

        # Check patch embed properties
        print(f"Model patch size: {self.model.patch_embed.patch_size}")
        print(f"Model grid size: {self.model.patch_embed.grid_size}")
        print(f"Model num patches: {self.model.patch_embed.num_patches}")
        print(f"Positional embedding shape: {self.model.pos_embed.shape}")
    
        # Check actual output
        patch_embed = self.model.patch_embed(torch.randn(1, 3, 518, 518).to(device))
        print(f"Actual patch embed shape: {patch_embed.shape}")
        
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
        
        # Extract components we need
        self.patch_embed = self.model.patch_embed
        self.pos_embed = self.model.pos_embed
        self.blocks = self.model.blocks
        self.norm = self.model.norm
        
        # Patch size: 14×14, image size: 518×518
        # Num patches: (518/14)² = 37² = 1,369
        self.num_patches = 1369
        self.embed_dim = 384
        
        logger.info(f"DINOv2 tokenizer initialized with {self.num_patches} patches on device: {device}")
        
    def forward(self, x):
        """
        Args:
            x: (B, 3, 518, 518) - Batch of face images (already normalized to [0, 1])
        
        Returns:
            patch_tokens: (B, 1369, 384) - Patch tokens only
            pos_embeddings: (B, 1369, 384) - Positional embeddings for patches
        """
        B = x.shape[0]
        
        # Apply ImageNet normalization
        # ImageNet mean and std values
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        
        # Normalize: (x - mean) / std
        x = (x - mean) / std
        
        # Get positional embeddings for patches (skip class token position)
        pos_emb = self.model.pos_embed[:, 1:, :].expand(B, -1, -1)  # (B, 1369, 384)
        
        # Use model's forward_features method to get processed tokens through all transformer layers
        with torch.no_grad():
            # This handles class token, pos embedding, and transformer blocks
            x = self.model.forward_features(x)  # (B, 1370, 384)
            
        # Extract only patch tokens (skip class token)
        patch_tokens = x[:, 1:, :]  # (B, 1369, 384)
        
        return patch_tokens, pos_emb
    
    def get_patch_size(self):
        """Get the patch size used by the model"""
        return self.patch_embed.patch_size
    
    def get_embed_dim(self):
        """Get the embedding dimension"""
        return self.embed_dim


def test_dinov2_tokenizer():
    """Test the DINOv2 tokenizer"""
    import torch
    
    # Create tokenizer
    tokenizer = DINOv2Tokenizer(device="cpu")
    
    # Create dummy input
    batch_size = 2
    x = torch.randn(batch_size, 3, 518, 518)
    
    # Forward pass
    patch_tokens, pos_emb = tokenizer(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Patch tokens shape: {patch_tokens.shape}")
    print(f"Positional embeddings shape: {pos_emb.shape}")
    
    # Verify shapes
    assert patch_tokens.shape == (batch_size, 1369, 384)
    assert pos_emb.shape == (batch_size, 1369, 384)
    
    print("✅ DINOv2 tokenizer test passed!")


if __name__ == "__main__":
    test_dinov2_tokenizer() 