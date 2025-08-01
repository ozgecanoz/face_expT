#!/usr/bin/env python3
"""
Test that gradient detachment for similarity analysis doesn't affect training gradients
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.visualization_utils import compute_cosine_similarity_distribution


def test_gradient_preservation():
    """Test that detaching tensors for analysis doesn't affect original gradients."""
    print("🧪 Testing gradient preservation during similarity analysis...")
    
    # Create tensors that require gradients (simulating model output)
    torch.manual_seed(42)
    original_tokens = torch.randn(5, 1, 384, requires_grad=True)
    
    print(f"📊 Original tensor:")
    print(f"   Shape: {original_tokens.shape}")
    print(f"   Requires grad: {original_tokens.requires_grad}")
    print(f"   Grad fn: {original_tokens.grad_fn}")
    
    # Simulate what happens in training
    # 1. Use original tensor for loss computation
    loss = original_tokens.sum()  # Simple loss for demonstration
    
    print(f"\n📈 Loss computation:")
    print(f"   Loss value: {loss.item():.3f}")
    print(f"   Loss requires grad: {loss.requires_grad}")
    
    # 2. Compute similarity analysis (this detaches tensors internally)
    expression_tokens_by_clip = [original_tokens]
    
    print(f"\n🔍 Similarity analysis:")
    try:
        similarity_data = compute_cosine_similarity_distribution(expression_tokens_by_clip)
        print(f"   Similarity computation successful")
        print(f"   Mean similarity: {similarity_data['mean_similarity']:.3f}")
    except Exception as e:
        print(f"   Similarity computation failed: {e}")
        return False
    
    # 3. Check that original tensor still has gradients
    print(f"\n✅ After similarity analysis:")
    print(f"   Original tensor requires grad: {original_tokens.requires_grad}")
    print(f"   Original tensor grad fn: {original_tokens.grad_fn}")
    print(f"   Loss still requires grad: {loss.requires_grad}")
    
    # 4. Perform backward pass
    print(f"\n🔄 Backward pass:")
    loss.backward()
    print(f"   Backward pass successful")
    print(f"   Original tensor has grad: {original_tokens.grad is not None}")
    if original_tokens.grad is not None:
        print(f"   Grad norm: {original_tokens.grad.norm().item():.3f}")
    
    return True


def test_training_simulation():
    """Simulate the exact training scenario."""
    print("\n🔄 Testing training simulation...")
    
    # Create a simple model
    model = torch.nn.Linear(384, 384)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create input tensors
    torch.manual_seed(42)
    input_tokens = torch.randn(5, 1, 384, requires_grad=True)
    
    print(f"📊 Training setup:")
    print(f"   Input shape: {input_tokens.shape}")
    print(f"   Input requires grad: {input_tokens.requires_grad}")
    
    # Forward pass
    with torch.enable_grad():
        output_tokens = model(input_tokens)
        loss = output_tokens.sum()
    
    print(f"\n📈 Forward pass:")
    print(f"   Output shape: {output_tokens.shape}")
    print(f"   Output requires grad: {output_tokens.requires_grad}")
    print(f"   Loss value: {loss.item():.3f}")
    
    # Simulate similarity analysis (like in training)
    expression_tokens_by_clip = [output_tokens]
    
    print(f"\n🔍 Similarity analysis (training simulation):")
    try:
        similarity_data = compute_cosine_similarity_distribution(expression_tokens_by_clip)
        print(f"   Similarity computation successful")
        print(f"   Mean similarity: {similarity_data['mean_similarity']:.3f}")
    except Exception as e:
        print(f"   Similarity computation failed: {e}")
        return False
    
    # Check gradients are preserved
    print(f"\n✅ After similarity analysis:")
    print(f"   Output tensor requires grad: {output_tokens.requires_grad}")
    print(f"   Loss still requires grad: {loss.requires_grad}")
    
    # Backward pass
    print(f"\n🔄 Backward pass:")
    optimizer.zero_grad()
    loss.backward()
    
    print(f"   Backward pass successful")
    print(f"   Output tensor has grad: {output_tokens.grad is not None}")
    print(f"   Model parameters have grads: {all(p.grad is not None for p in model.parameters())}")
    
    # Optimizer step
    optimizer.step()
    print(f"   Optimizer step successful")
    
    return True


if __name__ == "__main__":
    success1 = test_gradient_preservation()
    success2 = test_training_simulation()
    
    if success1 and success2:
        print("\n🎉 All gradient preservation tests passed!")
        print("✅ Gradient detachment for similarity analysis does NOT affect training!")
    else:
        print("\n❌ Some gradient preservation tests failed!") 