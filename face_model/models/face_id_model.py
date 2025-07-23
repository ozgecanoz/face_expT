"""
Component B: Face ID Model (Learnable)
Learns expression-invariant identity representations from all frames of a speaker
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class FaceIDModel(nn.Module):
    """
    Component B: Face ID Model
    Input: 1 frame × 1,369 patch tokens (384-dim each)
    Output: 1 identity token (384-dim) per subject
    
    Implements strict attention: Q (face ID token) is learnable and updated by transformer layers,
    while K,V (patch tokens) context remains fixed and doesn't get updated.
    """
    
    def __init__(self, embed_dim=384, num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Identity token initialization (learnable)
        self.identity_query = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Cross-attention layers that only update the query
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Feed-forward networks for query updates
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        
        # Layer normalization for query updates
        self.query_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        # Final layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        logger.info(f"Face ID Model initialized with {num_layers} layers, {num_heads} heads")
        
    def forward(self, patch_tokens, pos_embeddings):
        """
        Args:
            patch_tokens: (B, 1369, 384) - 1 frame of patch tokens per batch
            pos_embeddings: (B, 1369, 384) - Positional embeddings for patches
        
        Returns:
            identity_token: (B, 1, 384) - Identity token
        """
        B, num_patches, embed_dim = patch_tokens.shape
        
        # Initialize identity query for batch
        query = self.identity_query.expand(B, 1, embed_dim)
        
        # Prepare fixed K,V context (doesn't get updated)
        # Add positional embeddings to patch tokens
        kv_context = patch_tokens + pos_embeddings  # (B, 1369, 384)
        
        # Apply cross-attention layers that only update the query
        for i, (attention_layer, ffn_layer, query_norm) in enumerate(
            zip(self.cross_attention_layers, self.ffn_layers, self.query_norms)
        ):
            # Cross-attention: query attends to fixed kv_context
            # query: (B, 1, 384), kv_context: (B, 1369, 384)
            attn_output, _ = attention_layer(
                query=query,
                key=kv_context,
                value=kv_context
            )
            
            # Residual connection for query
            query = query + attn_output
            
            # Layer normalization
            query = query_norm(query)
            
            # Feed-forward network for query
            ffn_output = ffn_layer(query)
            
            # Residual connection for query
            query = query + ffn_output
        
        # Final projection and normalization
        identity_token = self.output_proj(query)
        identity_token = self.layer_norm(identity_token)
        
        return identity_token
    
    def get_identity_consistency_loss(self, identity_tokens):
        """
        Compute consistency loss to ensure identity tokens are similar across frames
        
        Args:
            identity_tokens: (B, 30, 384) - Identity tokens for all frames
        
        Returns:
            consistency_loss: Scalar loss value
        """
        # Compute variance across frames (lower variance = more consistent)
        variance = torch.var(identity_tokens, dim=1)  # (B, 384)
        consistency_loss = torch.mean(variance)
        
        return consistency_loss


def test_face_id_model():
    """Test the Face ID Model with contrastive loss"""
    import torch
    import torch.nn as nn
    import sys
    import os
    
    # Add the training directory to the path for imports
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'training'))
    
    # Create model
    model = FaceIDModel()
    
    # Create dummy input for multiple clips
    batch_size = 4  # Multiple clips to test contrastive loss
    num_patches = 1369
    embed_dim = 384
    
    patch_tokens = torch.randn(batch_size, num_patches, embed_dim)
    pos_embeddings = torch.randn(batch_size, num_patches, embed_dim)
    
    # Forward pass
    identity_token = model(patch_tokens, pos_embeddings)
    
    print(f"Patch tokens shape: {patch_tokens.shape}")
    print(f"Positional embeddings shape: {pos_embeddings.shape}")
    print(f"Identity token shape: {identity_token.shape}")
    
    # Verify shapes
    assert identity_token.shape == (batch_size, 1, embed_dim)
    
    # Test consistency loss with multiple frames
    identity_tokens_all_frames = torch.randn(batch_size, 30, embed_dim)
    consistency_loss = model.get_identity_consistency_loss(identity_tokens_all_frames)
    print(f"Consistency loss: {consistency_loss.item():.4f}")
    
    # Test contrastive loss
    try:
        from train_face_id import ContrastiveLoss
        
        # Create contrastive loss function
        contrastive_loss_fn = ContrastiveLoss(temperature=0.1, margin=1.0)
        
        # Create dummy subject IDs (2 subjects, 2 clips each)
        subject_ids = [0, 0, 1, 1]  # First 2 clips are subject 0, last 2 are subject 1
        
        # Get identity tokens for each clip (average across frames)
        clip_identity_tokens = identity_tokens_all_frames.mean(dim=1)  # (B, 384)
        
        # Compute contrastive loss
        contrastive_loss = contrastive_loss_fn(clip_identity_tokens, subject_ids)
        print(f"Contrastive loss: {contrastive_loss.item():.4f}")
        
        # Test with different subject configurations
        print("\nTesting different subject configurations:")
        
        # All same subject
        subject_ids_same = [0, 0, 0, 0]
        contrastive_loss_same = contrastive_loss_fn(clip_identity_tokens, subject_ids_same)
        print(f"  All same subject contrastive loss: {contrastive_loss_same.item():.4f}")
        
        # All different subjects
        subject_ids_diff = [0, 1, 2, 3]
        contrastive_loss_diff = contrastive_loss_fn(clip_identity_tokens, subject_ids_diff)
        print(f"  All different subjects contrastive loss: {contrastive_loss_diff.item():.4f}")
        
        # Mixed subjects
        subject_ids_mixed = [0, 0, 1, 2]
        contrastive_loss_mixed = contrastive_loss_fn(clip_identity_tokens, subject_ids_mixed)
        print(f"  Mixed subjects contrastive loss: {contrastive_loss_mixed.item():.4f}")
        
        # Test: All same subjects with identical tokens should give zero loss
        print("\nTesting zero loss case:")
        # Create identical identity tokens for all clips
        identical_tokens = torch.ones(batch_size, embed_dim)  # All tokens are identical
        identical_tokens = nn.functional.normalize(identical_tokens, dim=1)  # Normalize
        
        # All subjects are the same
        subject_ids_identical = [0, 0, 0, 0]
        contrastive_loss_identical = contrastive_loss_fn(identical_tokens, subject_ids_identical)
        print(f"  Identical tokens, same subjects contrastive loss: {contrastive_loss_identical.item():.4f}")
        
        # Verify that the loss is very close to zero
        assert contrastive_loss_identical.item() < 1e-6, f"Expected near-zero loss, got {contrastive_loss_identical.item()}"
        print("  ✅ Zero loss test passed!")
        
        print("✅ Contrastive loss test passed!")
        
    except ImportError as e:
        print(f"⚠️  Could not import ContrastiveLoss from training module: {e}")
        print("   This is expected if running the test from outside the training context.")
        print("   The contrastive loss functionality is available in the training script.")
    
    print("✅ Face ID Model test passed!")


if __name__ == "__main__":
    test_face_id_model() 