import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Optional, Tuple

def orthogonal_rows_qr(num_rows: int, dim: int, *, device=None, dtype=None, seed: int = 0):
    """Return (num_rows, dim) with orthonormal rows (when num_rows <= dim)."""
    assert num_rows <= dim, "Cannot have more orthonormal rows than dim."
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    A = torch.randn(dim, num_rows, generator=g, device=device, dtype=dtype)   # (D, S)
    # Q has orthonormal columns; shape (D, S)
    Q, _ = torch.linalg.qr(A, mode="reduced")
    return Q.t().contiguous()   # (S, D) with orthonormal rows

def block_orthogonal_rows(num_rows: int, dim: int, *, device=None, dtype=None, seed: int = 0):
    g = torch.Generator(device=device); g.manual_seed(seed)
    blocks = []
    remaining = num_rows
    while remaining > 0:
        m = min(dim, remaining)
        A = torch.randn(dim, dim, generator=g, device=device, dtype=dtype)
        Q, _ = torch.linalg.qr(A, mode="reduced")  # (D, D)
        blocks.append(Q[:m, :])                    # take m rows
        remaining -= m
    M = torch.cat(blocks, dim=0)                   # (S, D)
    # Optional: small random rotation to reduce inter-block alignment
    R, _ = torch.linalg.qr(torch.randn(dim, dim, generator=g, device=device, dtype=dtype))
    M = (M @ R).contiguous()
    return M

def create_subject_bases(max_subjects: int, embed_dim: int, *, seed: int = 0, device=None, dtype=None):
    """
    Returns (max_subjects, embed_dim) subject base embeddings.
    If max_subjects <= embed_dim: orthonormal rows via QR.
    Else: block-orthonormal rows (orthonormal within each block of size embed_dim).
    """
    if max_subjects <= embed_dim:
        return orthogonal_rows_qr(max_subjects, embed_dim, device=device, dtype=dtype, seed=seed)
    else:
        return block_orthogonal_rows(max_subjects, embed_dim, device=device, dtype=dtype, seed=seed)

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
                 dropout: float = 0.1,
                 max_subjects: int = 3500):
        """
        Args:
            embed_dim: Embedding dimension (default: 384)
            num_patches: Number of patches (default: 1369 for 37x37 grid)
            num_cross_attention_layers: Number of cross-attention transformer blocks
            num_self_attention_layers: Number of self-attention transformer blocks
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            dropout: Dropout rate
            max_subjects: Maximum number of subjects for learnable subject embeddings
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.max_subjects = max_subjects
        self.num_cross_attention_layers = num_cross_attention_layers
        self.num_self_attention_layers = num_self_attention_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        
        # Fixed orthogonal subject embeddings (frozen)
        subject_base = create_subject_bases(max_subjects, embed_dim)
        self.register_buffer('subject_base', subject_base)  # (max_subjects, embed_dim) - frozen
        
        # Learnable delta subject embeddings (small magnitude)
        self.delta_subject_embed = nn.Parameter(torch.zeros(max_subjects, embed_dim) * 0.001)
        # Initialize very close to zero for stable training
        nn.init.trunc_normal_(self.delta_subject_embed, std=0.001)
        
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
        
        logger = logging.getLogger(__name__)
        logger.info(f"Expression Reconstruction Model initialized with {num_cross_attention_layers} cross-attention layers, {num_self_attention_layers} self-attention layers")
        logger.info(f"Feed-forward dimension: {ff_dim}")
        logger.info(f"Subject embeddings: {max_subjects} subjects, {embed_dim} dimensions (orthogonal base + learnable delta)")
        logger.info(f"Positional embeddings: {num_patches} patches, {embed_dim} dimensions")
        
    def forward(self, 
                subject_ids: torch.Tensor,
                expression_token: torch.Tensor,
                pos_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            subject_ids: (B,) - Subject IDs for each sample in batch
            expression_token: (B, 1, embed_dim) - Expression token
            pos_embeddings: (B, num_patches, embed_dim) - Positional embeddings
        Returns:
            torch.Tensor: (B, 3, 518, 518) - Reconstructed face image
        """
        B = subject_ids.shape[0]
        
        # Get subject embeddings: fixed base + learnable delta
        subject_base_embeddings = self.subject_base[subject_ids]  # (B, embed_dim)
        subject_delta_embeddings = self.delta_subject_embed[subject_ids]  # (B, embed_dim)
        subject_embedding = subject_base_embeddings + subject_delta_embeddings  # (B, embed_dim)
        subject_embedding = subject_embedding.unsqueeze(1)  # (B, 1, embed_dim)
        
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
    
    def get_config(self) -> dict:
        """
        Get model configuration for checkpointing
        
        Returns:
            dict: Model configuration parameters
        """
        return {
            'embed_dim': self.embed_dim,
            'num_patches': self.num_patches,
            'num_cross_attention_layers': self.num_cross_attention_layers,
            'num_self_attention_layers': self.num_self_attention_layers,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout': self.dropout,
            'max_subjects': self.max_subjects
        }


def create_expression_reconstruction_model(
    embed_dim: int = 384,
    num_patches: int = 1369,
    num_cross_attention_layers: int = 2,
    num_self_attention_layers: int = 2,
    num_heads: int = 8,
    ff_dim: int = 1536,
    dropout: float = 0.1,
    max_subjects: int = 3500
) -> ExpressionReconstructionModel:
    """Factory function to create expression reconstruction model"""
    
    return ExpressionReconstructionModel(
        embed_dim=embed_dim,
        num_patches=num_patches,
        num_cross_attention_layers=num_cross_attention_layers,
        num_self_attention_layers=num_self_attention_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout=dropout,
        max_subjects=max_subjects
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
    print(f"   Max subjects: {model.max_subjects}")
    print(f"   Subject embeddings: Fixed orthogonal base + learnable delta")
    print(f"   Cross-attention layers: {model.num_cross_attention_layers}")
    print(f"   Self-attention layers: {model.num_self_attention_layers}")
    print(f"   Num heads: {model.num_heads}")
    print(f"   FF dim: {model.ff_dim}")
    print(f"   Dropout: {model.dropout}")
    
    # Create dummy input
    batch_size = 2
    embed_dim = 384
    num_patches = 1369
    
    subject_ids = torch.randint(0, model.max_subjects, (batch_size,))
    expression_token = torch.randn(batch_size, 1, embed_dim)
    pos_embeddings = torch.randn(batch_size, num_patches, embed_dim)
    
    print(f"✅ Input tensors created:")
    print(f"   Subject IDs: {subject_ids.shape}")
    print(f"   Expression token: {expression_token.shape}")
    print(f"   Positional embeddings: {pos_embeddings.shape}")
    
    # Forward pass
    with torch.no_grad():
        reconstructed_image = model(subject_ids, expression_token, pos_embeddings)
    
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
        subject_ids = torch.randint(0, model.max_subjects, (bs,))
        expr_token = torch.randn(bs, 1, embed_dim)
        pos_emb = torch.randn(bs, num_patches, embed_dim)
        
        with torch.no_grad():
            output = model(subject_ids, expr_token, pos_emb)
        
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
        
        # Create new inputs for this specific model configuration
        test_batch_size = 2  # Use consistent batch size for testing
        test_subject_ids = torch.randint(0, model_config.max_subjects, (test_batch_size,))
        test_expression_token = torch.randn(test_batch_size, 1, config.get('embed_dim', embed_dim))
        test_pos_embeddings = torch.randn(test_batch_size, config.get('num_patches', num_patches), config.get('embed_dim', embed_dim))
        
        with torch.no_grad():
            output = model_config(test_subject_ids, test_expression_token, test_pos_embeddings)
        
        expected_shape = (test_batch_size, 3, 518, 518)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        print(f"✅ Config {i+1} works correctly")
    
    # Test 4: Parameter count
    print("\n📋 Test 4: Parameter count")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Learnable patch embeddings: {model.patch_embeddings.numel():,}")
    print(f"   Learnable subject embeddings: {model.delta_subject_embed.numel():,}") # Updated to reflect new learnable delta
    
    # Test 5: Gradient flow
    print("\n📋 Test 5: Gradient flow")
    
    model.train()
    subject_ids = torch.randint(0, model.max_subjects, (1,))
    expr_token = torch.randn(1, 1, embed_dim, requires_grad=True)
    pos_emb = torch.randn(1, num_patches, embed_dim)
    
    output = model(subject_ids, expr_token, pos_emb)
    loss = output.mean()
    loss.backward()
    
    # Check if gradients are flowing to learnable parameters
    assert model.delta_subject_embed.grad is not None, "Gradients not flowing to delta subject embeddings"
    assert expr_token.grad is not None, "Gradients not flowing to expression token"
    assert model.patch_embeddings.grad is not None, "Gradients not flowing to patch embeddings"
    
    print("✅ Gradient flow verified")
    
    # Test 6: Subject embedding properties
    print("\n📋 Test 6: Subject embedding properties")
    
    # Test orthogonal properties of base embeddings
    base_embeddings = model.subject_base  # (max_subjects, embed_dim)
    print(f"   Base subject embeddings shape: {base_embeddings.shape}")
    print(f"   Base embeddings magnitude: {base_embeddings.norm(dim=-1).mean():.4f}")
    
    # Test orthogonality (should be close to 0 for different subjects)
    if base_embeddings.shape[0] > 1:
        # Sample a subset of embeddings to avoid memory issues with large max_subjects
        sample_size = min(100, base_embeddings.shape[0])
        sample_indices = torch.randperm(base_embeddings.shape[0])[:sample_size]
        sample_embeddings = base_embeddings[sample_indices]
        
        print(f"   Testing orthogonality on sample of {sample_size} embeddings")
        
        # Compute pairwise cosine similarities on sample
        similarities = F.cosine_similarity(
            sample_embeddings.unsqueeze(1), 
            sample_embeddings.unsqueeze(0), 
            dim=-1
        )
        # Remove diagonal (self-similarities = 1.0)
        mask = ~torch.eye(sample_size, dtype=torch.bool)
        cross_similarities = similarities[mask]
        print(f"   Base embeddings cross-similarity mean: {cross_similarities.mean():.4f}")
        print(f"   Base embeddings cross-similarity std: {cross_similarities.std():.4f}")
        print(f"   Base embeddings max cross-similarity: {cross_similarities.max():.4f}")
        
        # Verify orthogonality is reasonable based on dimensions
        # For large numbers of subjects with small embedding dimensions, 
        # perfect orthogonality is not possible due to the Johnson-Lindenstrauss lemma
        max_cross_sim = cross_similarities.abs().max()
        
        # More lenient threshold for large subject counts with small embedding dimensions
        if model.max_subjects > embed_dim:
            # When max_subjects > embed_dim, we use block-orthogonal embeddings
            # Allow higher cross-similarity within reasonable bounds
            threshold = min(0.3, 0.1 + 0.2 * (model.max_subjects / embed_dim))
        else:
            # When max_subjects <= embed_dim, we can have true orthogonality
            threshold = 0.1
            
        print(f"   Orthogonality threshold: {threshold:.4f}")
        print(f"   Max cross-similarity: {max_cross_sim:.4f}")
        
        assert max_cross_sim < threshold, f"Base embeddings not sufficiently orthogonal: max cross-similarity {max_cross_sim:.4f} >= threshold {threshold:.4f}"
        print(f"✅ Orthogonality test passed (threshold: {threshold:.4f})")
    
    # Test delta embeddings
    delta_embeddings = model.delta_subject_embed  # (max_subjects, embed_dim)
    print(f"   Delta subject embeddings shape: {delta_embeddings.shape}")
    print(f"   Delta embeddings magnitude: {delta_embeddings.norm(dim=-1).mean():.4f}")
    
    print("✅ Subject embedding properties verified")
    
    # Test 7: Configuration method
    print("\n📋 Test 7: Configuration method")
    
    config = model.get_config()
    print(f"   Configuration retrieved: {list(config.keys())}")
    assert config['embed_dim'] == model.embed_dim, "Config embed_dim mismatch"
    assert config['num_patches'] == model.num_patches, "Config num_patches mismatch"
    assert config['max_subjects'] == model.max_subjects, "Config max_subjects mismatch"
    assert config['num_cross_attention_layers'] == model.num_cross_attention_layers, "Config num_cross_attention_layers mismatch"
    assert config['num_self_attention_layers'] == model.num_self_attention_layers, "Config num_self_attention_layers mismatch"
    print("✅ Configuration method works correctly")
    
    # Test 8: Model components
    print("\n📋 Test 8: Model components")
    
    # Test cross-attention block
    query = torch.randn(1, num_patches, embed_dim)
    key_value = torch.randn(1, 2, embed_dim)
    
    with torch.no_grad():
        cross_output = model.cross_decoder(query, key_value)
    
    assert cross_output.shape == query.shape, f"Cross-attention output shape mismatch: {cross_output.shape}"
    print("✅ Cross-attention block works")
    
    # Test self-attention block
    input_tensor = torch.randn(1, num_patches, embed_dim)
    
    with torch.no_grad():
        self_output = model.self_attention_encoder(input_tensor)
    
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