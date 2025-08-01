"""
Component D: Expression Transformer Decoder
Predicts next frame's expression token from sequence of previous expression tokens
Similar to LLM training with causal masking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

logger = logging.getLogger(__name__)


class ExpressionTransformerDecoder(nn.Module):
    """
    Component D: Expression Transformer Decoder
    Input: N-1 expression tokens (384-dim each)
    Output: Predicted expression token for Frame N (384-dim)
    
    Uses causal masking for autoregressive prediction
    """
    
    def __init__(self, embed_dim=384, num_heads=8, num_layers=2, dropout=0.1, max_sequence_length=50):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_sequence_length = max_sequence_length
        self.dropout = dropout
        
        # Positional embeddings for sequence order
        self.pos_embeddings = nn.Parameter(torch.randn(1, max_sequence_length, embed_dim))
        
        # Transformer encoder with causal masking
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        # Causal mask for autoregressive prediction
        self.register_buffer('causal_mask', self._create_causal_mask())
        
        logger.info(f"Expression Transformer Decoder initialized with {num_layers} layers, {num_heads} heads")
        
    def _create_causal_mask(self):
        """Create causal mask for autoregressive prediction"""
        mask = torch.triu(torch.ones(self.max_sequence_length, self.max_sequence_length), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, expression_tokens, clip_lengths=None):
        """
        Args:
            expression_tokens: (total_tokens, 384) or (B, seq_len, 384) - Expression tokens
            clip_lengths: List[int] - Lengths of each clip (optional)

        Returns:
            predicted_tokens: List of predicted next tokens per clip
        """
        if clip_lengths is None:
            if expression_tokens.dim() == 2:
                expression_tokens = expression_tokens.unsqueeze(0)  # (1, seq_len, D)
            return self._forward_single_clip(expression_tokens)
        
        predicted_tokens = []
        current_idx = 0
        
        for clip_length in clip_lengths:
            # Extract tokens for this clip
            clip_tokens = expression_tokens[current_idx:current_idx + clip_length - 1]  # N-1 tokens for prediction
            
            if clip_tokens.shape[0] > 0:
                # Process this clip
                clip_prediction = self._forward_single_clip(clip_tokens.unsqueeze(0))
                predicted_tokens.append(clip_prediction)
            else:
                # Single frame clip - no prediction possible
                predicted_tokens.append(torch.zeros(1, 1, self.embed_dim, device=expression_tokens.device))
            
            current_idx += clip_length
        
        return predicted_tokens
    
    def _forward_single_clip(self, expression_tokens):
        """
        Process a single clip of expression tokens
        
        Args:
            expression_tokens: (1, seq_len, 384) - Expression tokens for one clip
        
        Returns:
            predicted_token: (1, 1, 384) - Predicted next expression token
        """
        batch_size, seq_len, dim = expression_tokens.shape

        # Add positional embeddings
        pos = self.pos_embeddings[:, :seq_len, :]  # (1, seq_len, D)
        tokens_with_pos = expression_tokens + pos

        # Causal mask: (seq_len, seq_len)
        causal_mask = self.causal_mask[:seq_len, :seq_len]

        # Transformer encoder
        encoded = self.encoder(tokens_with_pos, mask=causal_mask)

        # Normalize and project
        encoded = self.layer_norm(encoded)
        output = self.output_proj(encoded)

        return output[:, -1:, :]  # Predict next token (1, 1, 384)


def test_expression_transformer_decoder():
    """Test the expression transformer decoder"""
    import torch
    from expression_transformer_decoder import ExpressionTransformerDecoder

    # Create model
    decoder = ExpressionTransformerDecoder(
        embed_dim=384,
        num_heads=8,
        num_layers=2,
        dropout=0.1,
        max_sequence_length=50
    )

    # Test single batch (fallback mode)
    print("\U0001F9EA Testing single batch mode:")
    seq_length = 10
    embed_dim = 384

    # Simulate expression tokens from previous frames
    expression_tokens = torch.randn(1, seq_length, embed_dim)
    print(f"Input expression tokens shape: {expression_tokens.shape}")

    # Forward pass
    predicted_token = decoder(expression_tokens)
    print(f"Predicted next token shape: {predicted_token.shape}")
    print(f"Expected shape: (1, 1, {embed_dim})")
    assert predicted_token.shape == (1, 1, embed_dim)

    # Test with different sequence lengths
    for seq_len in [5, 15, 25]:
        tokens = torch.randn(1, seq_len, embed_dim)
        pred = decoder(tokens)
        print(f"Sequence length {seq_len}: Input {tokens.shape} -> Output {pred.shape}")
        assert pred.shape == (1, 1, embed_dim)

    # Test multiple clips mode
    print("\n\U0001F9EA Testing multiple clips mode:")
    clip1_tokens = torch.randn(4, embed_dim)  # N-1 tokens
    clip2_tokens = torch.randn(2, embed_dim)
    all_tokens = torch.cat([clip1_tokens, clip2_tokens], dim=0)  # (6, 384)
    clip_lengths = [5, 3]

    print(f"All tokens shape: {all_tokens.shape}")
    print(f"Clip lengths: {clip_lengths}")

    predicted_tokens = decoder(all_tokens, clip_lengths)
    print(f"Number of predictions: {len(predicted_tokens)}")
    assert len(predicted_tokens) == len(clip_lengths)

    for clip_idx, pred in enumerate(predicted_tokens):
        print(f"Clip {clip_idx} prediction shape: {pred.shape}")
        assert pred.shape == (1, 1, embed_dim)

    print("\n✅ Expression Transformer Decoder test passed!")


if __name__ == "__main__":
    test_expression_transformer_decoder()