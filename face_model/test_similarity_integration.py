#!/usr/bin/env python3
"""
Test the similarity visualization integration with training
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.train_expression_prediction import JointExpressionPredictionModel
from utils.visualization_utils import compute_cosine_similarity_distribution, plot_cosine_similarity_distribution


def test_similarity_integration():
    """Test that similarity visualization works with the training model."""
    print("🧪 Testing similarity visualization integration...")
    
    # Create a dummy model
    model = JointExpressionPredictionModel(
        expr_embed_dim=384,
        decoder_embed_dim=384
    )
    
    # Create dummy expression tokens (simulating model output)
    torch.manual_seed(42)
    num_clips = 3
    expression_tokens_by_clip = []
    
    for i in range(num_clips):
        num_tokens = torch.randint(3, 8, (1,)).item()
        tokens = torch.randn(num_tokens, 1, 384)  # (num_tokens, 1, embed_dim)
        expression_tokens_by_clip.append(tokens)
    
    print(f"📊 Created {num_clips} clips with expression tokens")
    for i, tokens in enumerate(expression_tokens_by_clip):
        print(f"   Clip {i}: {tokens.shape[0]} tokens")
    
    # Compute similarity distribution
    similarity_data = compute_cosine_similarity_distribution(expression_tokens_by_clip)
    
    print(f"📈 Similarity Statistics:")
    print(f"   Mean: {similarity_data['mean_similarity']:.3f}")
    print(f"   Std: {similarity_data['std_similarity']:.3f}")
    print(f"   Min: {similarity_data['min_similarity']:.3f}")
    print(f"   Max: {similarity_data['max_similarity']:.3f}")
    print(f"   Total similarities: {len(similarity_data['similarities'])}")
    print(f"   Clips: {len(similarity_data['clip_similarities'])}")
    
    # Create and save plot
    fig = plot_cosine_similarity_distribution(
        similarity_data,
        save_path='test_similarity_integration.png',
        title="Test Similarity Integration"
    )
    
    print("✅ Similarity visualization integration test completed!")
    print("📊 Check 'test_similarity_integration.png' for the visualization")
    
    return fig


if __name__ == "__main__":
    test_similarity_integration() 