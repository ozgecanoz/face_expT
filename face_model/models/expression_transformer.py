"""
Component C: Expression Transformer (Learnable)
Learns to extract expression-specific features from face images using subject embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math

logger = logging.getLogger(__name__)


def create_2d_sinusoidal_positional_embeddings(grid_size, embed_dim):
    """
    Create fixed 2D sinusoidal positional embeddings for a grid.
    
    Args:
        grid_size: Size of the grid (e.g., 37 for 37x37)
        embed_dim: Embedding dimension
    
    Returns:
        pos_embeddings: (grid_size * grid_size, embed_dim) - Fixed positional embeddings
    """
    pos_embeddings = torch.zeros(grid_size * grid_size, embed_dim)
    
    # Create 2D grid coordinates
    y_pos = torch.arange(grid_size, dtype=torch.float32).unsqueeze(1).expand(grid_size, grid_size)
    x_pos = torch.arange(grid_size, dtype=torch.float32).unsqueeze(0).expand(grid_size, grid_size)
    
    # Flatten coordinates
    y_pos = y_pos.reshape(-1)  # (1369,)
    x_pos = x_pos.reshape(-1)  # (1369,)
    
    # Create sinusoidal embeddings
    for i in range(0, embed_dim, 2):
        if i + 1 < embed_dim:
            # Even dimensions: sin
            pos_embeddings[:, i] = torch.sin(x_pos / (10000 ** (i / embed_dim)))
            pos_embeddings[:, i + 1] = torch.sin(y_pos / (10000 ** ((i + 1) / embed_dim)))
        else:
            # Last dimension if odd
            pos_embeddings[:, i] = torch.sin(x_pos / (10000 ** (i / embed_dim)))
    
    return pos_embeddings


class ExpressionTransformer(nn.Module):
    """
    Component C: Expression Transformer
    Input: 1 frame × 1,369 patch tokens (384-dim each)
    Output: 1 expression token (384-dim)
    
    Implements strict attention: Q is learnable and updated by transformer layers,
    while K,V context remains fixed and doesn't get updated.
    
    Uses fixed 2D sinusoidal positional embeddings (37x37 grid) + learnable delta embeddings.
    Subject-invariant: learns to extract expression tokens without knowing subject identity.
    """
    
    def __init__(self, embed_dim=384, num_heads=8, num_layers=2, dropout=0.1, ff_dim=1536, grid_size=37):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.grid_size = grid_size
        self.num_patches = grid_size * grid_size  # 1369 for 37x37
        
        # Expression query initialization (learnable)
        self.expression_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Fixed 2D sinusoidal positional embeddings (frozen)
        pos_base = create_2d_sinusoidal_positional_embeddings(grid_size, embed_dim)
        self.register_buffer('pos_base', pos_base)  # (1369, 384) - frozen
        
        # Learnable delta positional embeddings (small magnitude)
        self.delta_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim) * 0.001)
        # Initialize very close to zero for stable training
        nn.init.trunc_normal_(self.delta_pos_embed, std=0.001)
        
        # Transformer decoder stack (cross-attends to memory only)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        # Final layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        logger.info(f"Expression Transformer initialized with {num_layers} layers, {num_heads} heads")
        logger.info(f"Feed-forward dimension: {ff_dim}")
        logger.info(f"Grid size: {grid_size}x{grid_size} = {self.num_patches} patches")
        logger.info(f"Positional embeddings: Fixed base + learnable delta (std=0.001)")
        logger.info(f"Subject-invariant: No subject embeddings")
        
    def forward(self, patch_tokens):
        """
        Args:
            patch_tokens: (B, 1369, 384) - 1 frame of patch tokens per batch
        
        Returns:
            expression_token: (B, 1, 384) - Expression token
        """
        B, num_patches, embed_dim = patch_tokens.shape
        assert num_patches == self.num_patches, f"Expected {self.num_patches} patches, got {num_patches}"
        
        # Initialize expression query for batch
        query = self.expression_query.expand(B, 1, embed_dim)
        
        # Prepare fixed K,V context (doesn't get updated)
        # Combine fixed base positional embeddings with learnable delta
        pos_embeddings = self.pos_base.unsqueeze(0).expand(B, -1, -1)  # (B, 1369, 384)
        patch_tokens_with_pos = patch_tokens + pos_embeddings + self.delta_pos_embed
        
        # Use patch tokens with positional embeddings as K, V context
        # No subject embeddings - purely expression-based
        kv_context = patch_tokens_with_pos  # (B, 1369, 384)
        
        # Cross-attend: decoder updates query based on memory
        decoded = self.decoder(tgt=query, memory=kv_context)  # (B, 1, D)
        
        # Final output projection and normalization
        expression_token = self.output_proj(decoded)
        expression_token = self.layer_norm(expression_token)
        expression_token = F.normalize(expression_token, dim=-1)  # (B, 1, D)
        
        return expression_token
    
    def inference(self, patch_tokens):
        """
        Inference method that returns expression tokens and final positional embeddings
        
        Args:
            patch_tokens: (B, 1369, 384) - 1 frame of patch tokens per batch
        
        Returns:
            expression_token: (B, 1, 384) - Expression token
            final_pos_embeddings: (B, 1369, 384) - Final positional embeddings (base + learned delta)
        """
        with torch.no_grad():
            B, num_patches, embed_dim = patch_tokens.shape
            assert num_patches == self.num_patches, f"Expected {self.num_patches} patches, got {num_patches}"
            
            # Initialize expression query for batch
            query = self.expression_query.expand(B, 1, embed_dim)
            
            # Prepare fixed K,V context (doesn't get updated)
            # Combine fixed base positional embeddings with learnable delta
            pos_embeddings = self.pos_base.unsqueeze(0).expand(B, -1, -1)  # (B, 1369, 384)
            final_pos_embeddings = pos_embeddings + self.delta_pos_embed  # (B, 1369, 384)
            patch_tokens_with_pos = patch_tokens + final_pos_embeddings
            
            # Use patch tokens with positional embeddings as K, V context
            # No subject embeddings - purely expression-based
            kv_context = patch_tokens_with_pos  # (B, 1369, 384)
            
            # Cross-attend: decoder updates query based on memory
            decoded = self.decoder(tgt=query, memory=kv_context)  # (B, 1, D)
            
            # Final output projection
            expression_token = self.output_proj(decoded)  # (B, 1, 384)
            
            # Final layer normalization
            expression_token = self.layer_norm(expression_token)
            expression_token = F.normalize(expression_token, dim=-1)  # (B, 1, D)
            
            return expression_token, final_pos_embeddings


def test_expression_transformer():
    """Test the expression transformer"""
    print("🧪 Testing Expression Transformer...")
    
    # Create model
    model = ExpressionTransformer(embed_dim=384, num_heads=4, num_layers=2, grid_size=37)
    print("✅ Expression Transformer created successfully")
    
    # Test with dummy input
    batch_size = 2
    num_patches = 1369
    embed_dim = 384
    
    patch_tokens = torch.randn(batch_size, num_patches, embed_dim)
    
    # Forward pass
    model.eval()  # Ensure model is in evaluation mode
    expression_token_forward = model(patch_tokens)
    
    print(f"Patch tokens shape: {patch_tokens.shape}")
    print(f"Expression token shape: {expression_token_forward.shape}")
    
    # Test inference method
    expression_token_inference, final_pos_embeddings = model.inference(patch_tokens)
    print(f"Inference method result shape: {expression_token_inference.shape}")
    print(f"Final positional embeddings shape: {final_pos_embeddings.shape}")
    
    # Test that forward and inference produce similar results (they should be identical)
    assert torch.allclose(expression_token_forward, expression_token_inference, atol=1e-6), "Forward and inference should produce identical results"
    
    # Test positional embeddings
    print(f"Fixed pos_base shape: {model.pos_base.shape}")
    print(f"Learnable delta_pos_embed shape: {model.delta_pos_embed.shape}")
    print(f"pos_base magnitude: {model.pos_base.norm():.4f}")
    print(f"delta_pos_embed magnitude: {model.delta_pos_embed.norm():.4f}")
    
    print("✅ Expression Transformer test passed!")


if __name__ == "__main__":
    test_expression_transformer() 