import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


# TransformerBlock removed - now using nn.TransformerDecoder for both cross and self attention


# CrossAttentionBlock removed - now using nn.TransformerDecoder with fixed memory


# Import the OptimizedCNNDecoder from the existing face reconstruction model
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from face_reconstruction_model import OptimizedCNNDecoder


class ExpressionReconstructionModel(nn.Module):
    """Expression reconstruction model using TransformerDecoder with fixed memory"""
    
    def __init__(self, 
                 embed_dim: int = 384,
                 num_patches: int = 1369,
                 num_cross_attention_layers: int = 2,
                 num_self_attention_layers: int = 2,
                 num_heads: int = 8,
                 ff_dim: int = 1536,
                 dropout: float = 0.1):
        """
        Args:
            embed_dim: Embedding dimension (default: 384)
            num_patches: Number of patches (default: 1369 for 37x37 grid)
            num_cross_attention_layers: Number of cross-attention transformer blocks
            num_self_attention_layers: Number of self-attention transformer blocks
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        
        # Learnable patch embeddings (Q vectors)
        self.patch_embeddings = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        
        # Cross-attention decoder layers (patches attend to subject + expression)
        # Using TransformerDecoderLayer for cross-attention with fixed memory
        cross_decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.cross_decoder = nn.TransformerDecoder(cross_decoder_layer, num_layers=num_cross_attention_layers)
        
        # Self-attention encoder layers (patches attend to each other)
        # Using TransformerEncoderLayer for pure intra-patch attention
        self_attention_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.self_attention_encoder = nn.TransformerEncoder(self_attention_encoder_layer, num_layers=num_self_attention_layers)
        
        # CNN decoder (reusing OptimizedCNNDecoder)
        self.decoder = OptimizedCNNDecoder(embed_dim)
        
    def forward(self, 
                subject_embedding: torch.Tensor,
                expression_token: torch.Tensor,
                pos_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            subject_embedding: (B, 1, embed_dim) - subject identity embedding
            expression_token: (B, 1, embed_dim) - expression token
            pos_embeddings: (B, num_patches, embed_dim) - positional embeddings from DINOv2
        Returns:
            torch.Tensor: (B, 3, 518, 518) - reconstructed face image
        """
        B = subject_embedding.shape[0]
        
        # Initialize patch embeddings with learnable vectors
        patch_vectors = self.patch_embeddings.expand(B, -1, -1)  # (B, num_patches, embed_dim)
        
        # Add positional embeddings
        patch_vectors = patch_vectors + pos_embeddings
        
        # Combine subject and expression tokens for fixed memory (K,V context)
        identity_expression = torch.cat([subject_embedding, expression_token], dim=1)  # (B, 2, embed_dim)
        
        # Cross-attention using TransformerDecoder: patches (query) attend to subject + expression (memory)
        # This follows the same pattern as ExpressionTransformer
        patch_vectors = self.cross_decoder(tgt=patch_vectors, memory=identity_expression)
        
        # Self-attention using TransformerEncoder: patches attend to each other
        # Pure intra-patch attention without external memory
        patch_vectors = self.self_attention_encoder(patch_vectors)
        
        # Reshape patch vectors to spatial grid for CNN decoder
        # (B, 1369, 384) -> (B, 37, 37, 384) -> (B, 384, 37, 37)
        B, num_patches, embed_dim = patch_vectors.shape
        spatial_features = patch_vectors.view(B, 37, 37, embed_dim)  # (B, 37, 37, 384)
        spatial_features = spatial_features.permute(0, 3, 1, 2)  # (B, 384, 37, 37)
        
        # Decode to image using OptimizedCNNDecoder
        reconstructed_image = self.decoder(spatial_features)
        
        return reconstructed_image


def create_expression_reconstruction_model(
    embed_dim: int = 384,
    num_patches: int = 1369,
    num_cross_attention_layers: int = 2,
    num_self_attention_layers: int = 2,
    num_heads: int = 8,
    ff_dim: int = 1536,
    dropout: float = 0.1
) -> ExpressionReconstructionModel:
    """Factory function to create expression reconstruction model"""
    
    return ExpressionReconstructionModel(
        embed_dim=embed_dim,
        num_patches=num_patches,
        num_cross_attention_layers=num_cross_attention_layers,
        num_self_attention_layers=num_self_attention_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout=dropout
    )


def test_expression_reconstruction_model():
    """Test the Expression Reconstruction Model"""
    import torch
    
    print("🧪 Testing Expression Reconstruction Model...")
    
    # Test 1: Basic model creation and forward pass
    print("\n📋 Test 1: Basic model creation and forward pass")
    
    # Create model with default parameters
    model = ExpressionReconstructionModel()
    print(f"✅ Model created successfully")
    print(f"   Embed dim: {model.embed_dim}")
    print(f"   Num patches: {model.num_patches}")
    print(f"   Cross-attention layers: {len(model.cross_attention_layers)}")
    print(f"   Self-attention layers: {len(model.self_attention_layers)}")
    
    # Create dummy input
    batch_size = 2
    embed_dim = 384
    num_patches = 1369
    
    subject_embedding = torch.randn(batch_size, 1, embed_dim)
    expression_token = torch.randn(batch_size, 1, embed_dim)
    pos_embeddings = torch.randn(batch_size, num_patches, embed_dim)
    
    print(f"✅ Input tensors created:")
    print(f"   Subject embedding: {subject_embedding.shape}")
    print(f"   Expression token: {expression_token.shape}")
    print(f"   Positional embeddings: {pos_embeddings.shape}")
    
    # Forward pass
    with torch.no_grad():
        reconstructed_image = model(subject_embedding, expression_token, pos_embeddings)
    
    print(f"✅ Forward pass successful:")
    print(f"   Output shape: {reconstructed_image.shape}")
    print(f"   Output min: {reconstructed_image.min().item():.4f}")
    print(f"   Output max: {reconstructed_image.max().item():.4f}")
    
    # Verify output is in valid range [0, 1]
    assert reconstructed_image.min() >= 0.0, f"Output min should be >= 0, got {reconstructed_image.min().item()}"
    assert reconstructed_image.max() <= 1.0, f"Output max should be <= 1, got {reconstructed_image.max().item()}"
    print("✅ Output is in valid range [0, 1]")
    
    # Test 2: Different batch sizes
    print("\n📋 Test 2: Different batch sizes")
    
    batch_sizes = [1, 4, 8]
    for bs in batch_sizes:
        subject_emb = torch.randn(bs, 1, embed_dim)
        expr_token = torch.randn(bs, 1, embed_dim)
        pos_emb = torch.randn(bs, num_patches, embed_dim)
        
        with torch.no_grad():
            output = model(subject_emb, expr_token, pos_emb)
        
        expected_shape = (bs, 3, 518, 518)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        print(f"✅ Batch size {bs}: {output.shape}")
    
    # Test 3: Different model configurations
    print("\n📋 Test 3: Different model configurations")
    
    configs = [
        {"num_cross_attention_layers": 1, "num_self_attention_layers": 1, "num_heads": 4},
        {"num_cross_attention_layers": 3, "num_self_attention_layers": 2, "num_heads": 8},
        {"num_cross_attention_layers": 1, "num_self_attention_layers": 3, "num_heads": 16},
    ]
    
    for i, config in enumerate(configs):
        print(f"   Testing config {i+1}: {config}")
        model_config = ExpressionReconstructionModel(**config)
        
        with torch.no_grad():
            output = model_config(subject_embedding, expression_token, pos_embeddings)
        
        assert output.shape == (batch_size, 3, 518, 518), f"Wrong output shape: {output.shape}"
        print(f"✅ Config {i+1} works correctly")
    
    # Test 4: Parameter count
    print("\n📋 Test 4: Parameter count")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Learnable patch embeddings: {model.patch_embeddings.numel():,}")
    
    # Test 5: Gradient flow
    print("\n📋 Test 5: Gradient flow")
    
    model.train()
    subject_emb = torch.randn(1, 1, embed_dim, requires_grad=True)
    expr_token = torch.randn(1, 1, embed_dim, requires_grad=True)
    pos_emb = torch.randn(1, num_patches, embed_dim)
    
    output = model(subject_emb, expr_token, pos_emb)
    loss = output.mean()
    loss.backward()
    
    assert subject_emb.grad is not None, "Gradients not flowing to subject embedding"
    assert expr_token.grad is not None, "Gradients not flowing to expression token"
    assert model.patch_embeddings.grad is not None, "Gradients not flowing to patch embeddings"
    
    print("✅ Gradient flow verified")
    
    # Test 6: Model components
    print("\n📋 Test 6: Model components")
    
    # Test cross-attention block
    cross_block = model.cross_attention_layers[0]
    query = torch.randn(1, num_patches, embed_dim)
    key_value = torch.randn(1, 2, embed_dim)
    
    with torch.no_grad():
        cross_output = cross_block(query, key_value)
    
    assert cross_output.shape == query.shape, f"Cross-attention output shape mismatch: {cross_output.shape}"
    print("✅ Cross-attention block works")
    
    # Test self-attention block
    self_block = model.self_attention_layers[0]
    input_tensor = torch.randn(1, num_patches, embed_dim)
    
    with torch.no_grad():
        self_output = self_block(input_tensor)
    
    assert self_output.shape == input_tensor.shape, f"Self-attention output shape mismatch: {self_output.shape}"
    print("✅ Self-attention block works")
    
    # Test decoder
    decoder = model.decoder
    spatial_features = torch.randn(1, embed_dim, 37, 37)
    
    with torch.no_grad():
        decoded = decoder(spatial_features)
    
    assert decoded.shape == (1, 3, 518, 518), f"Decoder output shape mismatch: {decoded.shape}"
    print("✅ CNN decoder works")
    
    print("\n🎉 All tests passed! Expression Reconstruction Model is working correctly.")
    
    return model


if __name__ == "__main__":
    # Run the test
    test_model = test_expression_reconstruction_model() 