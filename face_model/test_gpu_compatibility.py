#!/usr/bin/env python3
"""
Test GPU compatibility of the scheduler and loss function
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.train_expression_prediction import ExpressionPredictionLoss_v2
from utils.scheduler_utils import CombinedLRLossWeightScheduler


def test_gpu_compatibility():
    """Test that the scheduler and loss function work on GPU."""
    print("🧪 Testing GPU compatibility...")
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, testing CPU only")
        device = torch.device("cpu")
    else:
        print("✅ CUDA available, testing GPU compatibility")
        device = torch.device("cuda")
    
    print(f"Using device: {device}")
    
    # Create dummy model and optimizer
    model = torch.nn.Linear(10, 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Test scheduler on device
    print("\n🔄 Testing scheduler on device...")
    scheduler = CombinedLRLossWeightScheduler(
        optimizer=optimizer,
        initial_lr=1e-6,
        warmup_lr=1e-4,
        final_lr=1e-6,
        initial_weights={'lambda_prediction': 0.1, 'lambda_temporal': 0.5, 'lambda_diversity': 0.5},
        warmup_weights={'lambda_prediction': 0.3, 'lambda_temporal': 0.4, 'lambda_diversity': 0.4},
        final_weights={'lambda_prediction': 0.5, 'lambda_temporal': 0.3, 'lambda_diversity': 0.3},
        warmup_steps=1000,
        total_steps=5000
    )
    
    # Test a few steps
    for step in range(3):
        lr, weights = scheduler.step()
        print(f"Step {step}: LR={lr:.2e}, λ_pred={weights['lambda_prediction']:.3f}")
    
    # Test loss function on device
    print("\n🔄 Testing loss function on device...")
    criterion = ExpressionPredictionLoss_v2(
        lambda_prediction=0.1,
        lambda_temporal=0.5,
        lambda_diversity=0.5
    ).to(device)
    
    # Create dummy tensors on device with proper shapes
    predicted_tokens = [torch.randn(1, 384, device=device)]  # (1, 384)
    actual_tokens = [torch.randn(1, 1, 384, device=device)]  # (1, 1, 384)
    expression_tokens = [torch.randn(3, 1, 384, device=device)]  # (3, 1, 384)
    
    # Test forward pass
    loss, components = criterion(predicted_tokens, actual_tokens, expression_tokens)
    print(f"Loss computed: {loss.item():.4f}")
    print(f"Loss device: {loss.device}")
    
    # Test weight updates
    new_weights = {'lambda_prediction': 0.5, 'lambda_temporal': 0.3, 'lambda_diversity': 0.2}
    criterion.update_weights(new_weights)
    print(f"Weights updated successfully")
    
    # Test checkpointing on device
    print("\n💾 Testing checkpointing on device...")
    state_dict = scheduler.state_dict()
    print(f"Scheduler state saved at step {scheduler.current_step}")
    
    # Create new scheduler and load state
    new_scheduler = CombinedLRLossWeightScheduler(
        optimizer=optimizer,
        initial_lr=1e-6,
        warmup_lr=1e-4,
        final_lr=1e-6,
        initial_weights={'lambda_prediction': 0.1, 'lambda_temporal': 0.5, 'lambda_diversity': 0.5},
        warmup_weights={'lambda_prediction': 0.3, 'lambda_temporal': 0.4, 'lambda_diversity': 0.4},
        final_weights={'lambda_prediction': 0.5, 'lambda_temporal': 0.3, 'lambda_diversity': 0.3},
        warmup_steps=1000,
        total_steps=5000
    )
    
    new_scheduler.load_state_dict(state_dict)
    print(f"Loaded scheduler state, current step: {new_scheduler.current_step}")
    
    # Verify values match
    lr1, weights1 = scheduler.get_last_lr(), scheduler.get_current_weights()
    lr2, weights2 = new_scheduler.get_last_lr(), new_scheduler.get_current_weights()
    
    assert abs(lr1 - lr2) < 1e-10, "LR mismatch after checkpointing"
    for key in weights1:
        assert abs(weights1[key] - weights2[key]) < 1e-10, f"Weight {key} mismatch after checkpointing"
    
    print("✅ Checkpointing test passed!")
    
    print("\n🎉 All GPU compatibility tests passed!")
    return True


if __name__ == "__main__":
    test_gpu_compatibility() 