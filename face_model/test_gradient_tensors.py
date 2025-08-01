#!/usr/bin/env python3
"""
Test similarity computation with tensors that require gradients
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.visualization_utils import compute_cosine_similarity_distribution


def test_gradient_tensors():
    """Test that similarity computation works with tensors that require gradients."""
    print("🧪 Testing similarity computation with gradient tensors...")
    
    # Create tensors that require gradients (simulating training)
    torch.manual_seed(42)
    num_clips = 3
    expression_tokens_by_clip = []
    
    for i in range(num_clips):
        num_tokens = torch.randint(3, 8, (1,)).item()
        # Create tensors that require gradients
        tokens = torch.randn(num_tokens, 1, 384, requires_grad=True)
        expression_tokens_by_clip.append(tokens)
    
    print(f"📊 Created {num_clips} clips with gradient tensors")
    for i, tokens in enumerate(expression_tokens_by_clip):
        print(f"   Clip {i}: {tokens.shape[0]} tokens, requires_grad={tokens.requires_grad}")
    
    # Test similarity computation
    try:
        similarity_data = compute_cosine_similarity_distribution(expression_tokens_by_clip)
        
        print(f"📈 Similarity Statistics:")
        print(f"   Mean: {similarity_data['mean_similarity']:.3f}")
        print(f"   Std: {similarity_data['std_similarity']:.3f}")
        print(f"   Min: {similarity_data['min_similarity']:.3f}")
        print(f"   Max: {similarity_data['max_similarity']:.3f}")
        print(f"   Total similarities: {len(similarity_data['similarities'])}")
        print(f"   Clips: {len(similarity_data['clip_similarities'])}")
        
        print("✅ Successfully computed similarities with gradient tensors!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to compute similarities: {e}")
        return False


def test_training_simulation():
    """Simulate the actual training scenario with gradient computation."""
    print("\n🔄 Testing training simulation...")
    
    # Create a simple model
    model = torch.nn.Linear(384, 384)
    
    # Create input tensors that require gradients
    torch.manual_seed(42)
    num_clips = 2
    expression_tokens_by_clip = []
    
    for i in range(num_clips):
        num_tokens = torch.randint(3, 6, (1,)).item()
        # Create input tensors
        input_tokens = torch.randn(num_tokens, 1, 384, requires_grad=True)
        
        # Simulate model forward pass (this creates gradients)
        with torch.enable_grad():
            output_tokens = model(input_tokens)
            # Simulate some computation that creates gradients
            loss = output_tokens.sum()
            loss.backward()
        
        expression_tokens_by_clip.append(output_tokens)
    
    print(f"📊 Created {num_clips} clips with computed gradients")
    for i, tokens in enumerate(expression_tokens_by_clip):
        print(f"   Clip {i}: {tokens.shape[0]} tokens, requires_grad={tokens.requires_grad}")
    
    # Test similarity computation
    try:
        similarity_data = compute_cosine_similarity_distribution(expression_tokens_by_clip)
        
        print(f"📈 Training Simulation Results:")
        print(f"   Mean: {similarity_data['mean_similarity']:.3f}")
        print(f"   Std: {similarity_data['std_similarity']:.3f}")
        print(f"   Total similarities: {len(similarity_data['similarities'])}")
        
        print("✅ Successfully computed similarities in training simulation!")
        return True
        
    except Exception as e:
        print(f"❌ Failed in training simulation: {e}")
        return False


if __name__ == "__main__":
    success1 = test_gradient_tensors()
    success2 = test_training_simulation()
    
    if success1 and success2:
        print("\n🎉 All gradient tensor tests passed!")
    else:
        print("\n❌ Some gradient tensor tests failed!") 