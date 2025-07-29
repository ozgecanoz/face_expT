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
        
        # Positional embeddings for sequence order
        self.pos_embeddings = nn.Parameter(torch.randn(1, max_sequence_length, embed_dim))
        
        # Transformer encoder layers with causal masking
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Output projection
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
            expression_tokens: (total_tokens, 1, 384) - Concatenated expression tokens from multiple clips
            clip_lengths: List[int] - Length of each clip (for processing individual clips)
        
        Returns:
            predicted_tokens: List of predicted next tokens for each clip
        """
        if clip_lengths is None:
            # Fallback: process as single clip
            return self._forward_single_clip(expression_tokens)
        
        predicted_tokens = []
        current_idx = 0
        
        for clip_length in clip_lengths:
            # Extract tokens for this clip
            clip_tokens = expression_tokens[current_idx:current_idx + clip_length - 1]  # N-1 tokens for prediction
            
            if clip_tokens.shape[0] > 0:
                # Process this clip
                clip_prediction = self._forward_single_clip(clip_tokens)
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
            expression_tokens: (seq_len, 1, 384) - Expression tokens for one clip
        
        Returns:
            predicted_token: (1, 1, 384) - Predicted next expression token
        """
        seq_len = expression_tokens.shape[0]
        
        # Add positional embeddings
        pos_emb = self.pos_embeddings[:, :seq_len, :].transpose(0, 1) # (seq_len, 1, 384)
        tokens_with_pos = expression_tokens + pos_emb # (seq_len, 1, 384)
        
        # Reshape for transformer encoder (batch_first=True expects (batch, seq, dim))
        # one clip is one batch, each clip has a sequence of tokens, seq_len is the length of the sequence
        hidden_states = tokens_with_pos.transpose(0, 1) # (1, seq_len, 384) 
        
        for encoder_layer in self.encoder_layers:
            # Use causal mask for self-attention
            causal_mask = self.causal_mask[:seq_len, :seq_len]
            encoder_output = encoder_layer(
                src=hidden_states,
                src_mask=causal_mask
            )
            # Handle tuple return (output, attention_weights)
            if isinstance(encoder_output, tuple):
                hidden_states = encoder_output[0]
            else:
                hidden_states = encoder_output
        
        # Layer normalization
        hidden_states = self.layer_norm(hidden_states)
        
        # Output projection
        output = self.output_proj(hidden_states) # (1, seq_len, 384)
        
        # Return only the last token as prediction for next frame
        predicted_token = output[:, -1:, :]  # (1, 1, 384)
        
        return predicted_token


def test_expression_transformer_decoder():
    """Test the expression transformer decoder"""
    import torch
    
    # Create model
    decoder = ExpressionTransformerDecoder(
        embed_dim=384,
        num_heads=8,
        num_layers=2,
        dropout=0.1,
        max_sequence_length=50
    )
    
    # Test single batch (fallback mode)
    print("🧪 Testing single batch mode:")
    seq_length = 10
    embed_dim = 384
    
    # Simulate expression tokens from previous frames
    expression_tokens = torch.randn(seq_length, 1, embed_dim)
    
    print(f"Input expression tokens shape: {expression_tokens.shape}")
    
    # Forward pass
    predicted_token = decoder(expression_tokens)
    
    print(f"Predicted next token shape: {predicted_token.shape}")
    print(f"Expected shape: (1, 1, {embed_dim})")
    
    # Assert shape is correct
    assert predicted_token.shape == (1, 1, embed_dim), f"Expected (1, 1, {embed_dim}), got {predicted_token.shape}"
    
    # Test with different sequence lengths
    for seq_len in [5, 15, 25]:
        tokens = torch.randn(seq_len, 1, embed_dim)
        pred = decoder(tokens)
        print(f"Sequence length {seq_len}: Input {tokens.shape} -> Output {pred.shape}")
        
        # Assert shape is correct
        assert pred.shape == (1, 1, embed_dim), f"Expected (1, 1, {embed_dim}), got {pred.shape}"
    
    # Test multiple clips mode
    print("\n🧪 Testing multiple clips mode:")
    
    # Simulate 2 clips: first clip has 5 frames, second clip has 3 frames
    clip1_tokens = torch.randn(4, 1, embed_dim)  # 4 tokens (N-1 for 5-frame clip)
    clip2_tokens = torch.randn(2, 1, embed_dim)  # 2 tokens (N-1 for 3-frame clip)
    
    # Concatenate all tokens
    all_tokens = torch.cat([clip1_tokens, clip2_tokens], dim=0)  # (6, 1, 384)
    clip_lengths = [5, 3]  # Length of each clip
    
    print(f"All tokens shape: {all_tokens.shape}")
    print(f"Clip lengths: {clip_lengths}")
    
    # Forward pass
    predicted_tokens = decoder(all_tokens, clip_lengths)
    
    print(f"Number of predictions: {len(predicted_tokens)}")
    
    # Assert we have the correct number of predictions
    assert len(predicted_tokens) == len(clip_lengths), f"Expected {len(clip_lengths)} predictions, got {len(predicted_tokens)}"
    
    for clip_idx, pred in enumerate(predicted_tokens):
        print(f"Clip {clip_idx} prediction shape: {pred.shape}")
        
        # Assert each prediction has the correct shape
        assert pred.shape == (1, 1, embed_dim), f"Clip {clip_idx}: Expected (1, 1, {embed_dim}), got {pred.shape}"
    
    print("\n✅ Expression Transformer Decoder test passed!")


if __name__ == "__main__":
    test_expression_transformer_decoder() 