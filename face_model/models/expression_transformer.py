"""
Component C: Expression Transformer (Learnable)
Learns to extract expression-specific features from face images using subject embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class ExpressionTransformer(nn.Module):
    """
    Component C: Expression Transformer
    Input: 1 frame × 1,369 patch tokens (384-dim each) + Subject embedding (384-dim)
    Output: 1 expression token (384-dim)
    
    Implements strict attention: Q is learnable and updated by transformer layers,
    while K,V context remains fixed and doesn't get updated.
    """
    
    def __init__(self, embed_dim=384, num_heads=8, num_layers=2, dropout=0.1, max_subjects=3500):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_subjects = max_subjects
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_heads = num_heads
        
        # Learnable subject embeddings (replaces face ID model)
        # Typical usage: 100-1000 subjects for academic datasets, 1000-10000 for commercial
        self.subject_embeddings = nn.Embedding(max_subjects, embed_dim)
        
        # Expression query initialization (learnable)
        self.expression_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # delta position embeddings (learnable) will be added to Dino's positional embeddings
        self.delta_pos_embed = nn.Parameter(torch.zeros(1, 1369, embed_dim))
        nn.init.trunc_normal_(self.delta_pos_embed, std=0.02)
        
        # Transformer decoder stack (cross-attends to memory only)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        # Final layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        logger.info(f"Expression Transformer initialized with {num_layers} layers, {num_heads} heads")
        logger.info(f"Subject embeddings: {max_subjects} subjects, {embed_dim} dimensions")
        
    def forward(self, patch_tokens, pos_embeddings, subject_ids):
        """
        Args:
            patch_tokens: (B, 1369, 384) - 1 frame of patch tokens per batch
            pos_embeddings: (B, 1369, 384) - Positional embeddings for patches
            subject_ids: (B,) - Subject IDs for each sample in batch
        
        Returns:
            expression_token: (B, 1, 384) - Expression token
        """
        B, num_patches, embed_dim = patch_tokens.shape
        
        # Get subject embeddings for batch
        subject_embeddings = self.subject_embeddings(subject_ids)  # (B, 384)
        subject_embeddings = subject_embeddings.unsqueeze(1)  # (B, 1, 384)
        
        # Initialize expression query for batch
        query = self.expression_query.expand(B, 1, embed_dim)
        
        # Prepare fixed K,V context (doesn't get updated)
        # Add positional embeddings to patch tokens
        #patch_tokens_with_pos = patch_tokens + pos_embeddings
        patch_tokens_with_pos = patch_tokens + pos_embeddings + self.delta_pos_embed
        
        # Combine subject_embeddings and patch_tokens for K, V
        # subject_embeddings: (B, 1, 384), patch_tokens_with_pos: (B, 1369, 384)
        kv_context = torch.cat([subject_embeddings, patch_tokens_with_pos], dim=1)  # (B, 1370, 384)
        
        # Cross-attend: decoder updates query based on memory
        decoded = self.decoder(tgt=query, memory=kv_context)  # (B, 1, D)
        
        # Final output projection and normalization
        expression_token = self.output_proj(decoded)
        expression_token = self.layer_norm(expression_token)
        expression_token = F.normalize(expression_token, dim=-1)  # (B, 1, D)
        
        return expression_token
    
    def inference(self, patch_tokens, pos_embeddings, subject_ids):
        """
        Inference method that returns both expression tokens and subject embeddings
        
        Args:
            patch_tokens: (B, 1369, 384) - 1 frame of patch tokens per batch
            pos_embeddings: (B, 1369, 384) - Positional embeddings for patches
            subject_ids: (B,) - Subject IDs for each sample in batch
        
        Returns:
            expression_token: (B, 1, 384) - Expression token
            subject_embeddings: (B, 1, 384) - Subject embeddings used
        """
        with torch.no_grad():
            B, num_patches, embed_dim = patch_tokens.shape
            
            # Get subject embeddings for batch
            subject_embeddings = self.subject_embeddings(subject_ids)  # (B, 384)
            subject_embeddings = subject_embeddings.unsqueeze(1)  # (B, 1, 384)
            
            # Initialize expression query for batch
            query = self.expression_query.expand(B, 1, embed_dim)
            
            # Prepare fixed K,V context (doesn't get updated)
            # Add positional embeddings to patch tokens
            #patch_tokens_with_pos = patch_tokens + pos_embeddings
            patch_tokens_with_pos = patch_tokens + pos_embeddings + self.delta_pos_embed
            
            # Combine subject_embeddings and patch_tokens for K, V
            # subject_embeddings: (B, 1, 384), patch_tokens_with_pos: (B, 1369, 384)
            kv_context = torch.cat([subject_embeddings, patch_tokens_with_pos], dim=1)  # (B, 1370, 384)
            
            # Cross-attend: decoder updates query based on memory
            decoded = self.decoder(tgt=query, memory=kv_context)  # (B, 1, D)
            
            # Final output projection
            expression_token = self.output_proj(decoded)  # (B, 1, 384)
            
            # Final layer normalization
            expression_token = self.layer_norm(expression_token)
            expression_token = F.normalize(expression_token, dim=-1)  # (B, 1, D)
            
            return expression_token, subject_embeddings


def test_expression_transformer():
    """Test the expression transformer"""
    print("🧪 Testing Expression Transformer...")
    
    # Create model
    model = ExpressionTransformer(embed_dim=384, num_heads=4, num_layers=2, max_subjects=100)
    print("✅ Expression Transformer created successfully")
    
    # Test with dummy input
    batch_size = 2
    num_patches = 1369
    embed_dim = 384
    
    patch_tokens = torch.randn(batch_size, num_patches, embed_dim)
    pos_embeddings = torch.randn(batch_size, num_patches, embed_dim)
    subject_ids = torch.randint(0, model.max_subjects, (batch_size,))
    
    # Forward pass
    model.eval()  # Ensure model is in evaluation mode
    expression_token_forward = model(patch_tokens, pos_embeddings, subject_ids)
    
    print(f"Patch tokens shape: {patch_tokens.shape}")
    print(f"Positional embeddings shape: {pos_embeddings.shape}")
    print(f"Subject IDs shape: {subject_ids.shape}")
    print(f"Expression token shape: {expression_token_forward.shape}")
    
    # Test inference method
    expression_token_inference, subject_embeddings_inference = model.inference(patch_tokens, pos_embeddings, subject_ids)
    print(f"Inference method result shapes: expression={expression_token_inference.shape}, subject_embeddings={subject_embeddings_inference.shape}")
    
    # Test that forward and inference produce similar results (they should be identical)
    assert torch.allclose(expression_token_forward, expression_token_inference, atol=1e-6), "Forward and inference should produce identical results"
    
    print("✅ Expression Transformer test passed!")


if __name__ == "__main__":
    test_expression_transformer() 