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
    
    def __init__(self, embed_dim=384, num_heads=8, num_layers=2, dropout=0.1, max_subjects=100000):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_subjects = max_subjects
        
        # Learnable subject embeddings (replaces face ID model)
        self.subject_embeddings = nn.Embedding(max_subjects, embed_dim)
        
        # Expression query initialization (learnable)
        self.expression_query = nn.Parameter(torch.randn(1, 1, embed_dim))
        
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
        patch_tokens_with_pos = patch_tokens + pos_embeddings
        
        # Combine subject_embeddings and patch_tokens for K, V
        # subject_embeddings: (B, 1, 384), patch_tokens_with_pos: (B, 1369, 384)
        kv_context = torch.cat([subject_embeddings, patch_tokens_with_pos], dim=1)  # (B, 1370, 384)
        
        # Apply cross-attention layers that only update the query
        for i, (attention_layer, ffn_layer, query_norm) in enumerate(
            zip(self.cross_attention_layers, self.ffn_layers, self.query_norms)
        ):
            # Cross-attention: query attends to fixed kv_context
            # query: (B, 1, 384), kv_context: (B, 1370, 384)
            attn_output, _ = attention_layer(
                query=query,
                key=kv_context,
                value=kv_context
            )
            
            # Residual connection for query
            query = query + attn_output
            
            # Layer normalization
            query = query_norm(query)
            
            # Feed-forward network
            ffn_output = ffn_layer(query)
            
            # Residual connection for query
            query = query + ffn_output
        
        # Final output projection and normalization
        expression_token = self.output_proj(query)
        expression_token = self.layer_norm(expression_token)
        
        return expression_token
    
    def inference(self, patch_tokens, pos_embeddings, subject_ids):
        """
        Inference method for cached DINOv2 tokens
        
        Args:
            patch_tokens: (B, 1369, 384) - Cached DINOv2 patch tokens
            pos_embeddings: (B, 1369, 384) - Cached positional embeddings
            subject_ids: (B,) - Subject IDs for each sample in batch
        
        Returns:
            expression_token: (B, 1, 384) - Expression token
        """
        return self.forward(patch_tokens, pos_embeddings, subject_ids)


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
    expression_token = model(patch_tokens, pos_embeddings, subject_ids)
    
    print(f"Patch tokens shape: {patch_tokens.shape}")
    print(f"Positional embeddings shape: {pos_embeddings.shape}")
    print(f"Subject IDs shape: {subject_ids.shape}")
    print(f"Expression token shape: {expression_token.shape}")
    
    # Test inference method with same inputs
    with torch.no_grad():
        expression_token_inference = model.inference(patch_tokens, pos_embeddings, subject_ids)
    print(f"Inference method result shape: {expression_token_inference.shape}")
    
    # Verify inference matches forward pass
    diff = torch.abs(expression_token - expression_token_inference).max().item()
    print(f"Max difference between forward and inference: {diff}")
    print("✅ Inference method works correctly")
    
    # Test with different batch size
    batch_size_2 = 3
    patch_tokens_2 = torch.randn(batch_size_2, num_patches, embed_dim)
    pos_embeddings_2 = torch.randn(batch_size_2, num_patches, embed_dim)
    subject_ids_2 = torch.randint(0, model.max_subjects, (batch_size_2,))
    
    expression_token_2 = model(patch_tokens_2, pos_embeddings_2, subject_ids_2)
    print(f"Expression token shape (batch_size={batch_size_2}): {expression_token_2.shape}")
    
    print("✅ Expression Transformer test passed!")


if __name__ == "__main__":
    test_expression_transformer() 