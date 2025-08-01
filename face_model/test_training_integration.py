#!/usr/bin/env python3
"""
Test the training script integration with the new combined scheduler
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.train_expression_prediction import (
    JointExpressionPredictionModel,
    ExpressionPredictionLoss_v2,
    train_expression_prediction
)
from utils.scheduler_utils import CombinedLRLossWeightScheduler


def test_training_integration():
    """Test that the training script works with the new combined scheduler."""
    print("🧪 Testing training integration with combined scheduler...")
    
    # Create a dummy model
    model = JointExpressionPredictionModel(
        expr_embed_dim=384,
        decoder_embed_dim=384
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Create loss function
    criterion = ExpressionPredictionLoss_v2(
        lambda_prediction=1.0,
        lambda_temporal=0.1,
        lambda_diversity=0.1
    )
    
    # Test weight updating
    print("\n🔄 Testing loss weight updates:")
    initial_weights = criterion.lambda_prediction, criterion.lambda_temporal, criterion.lambda_diversity
    print(f"Initial weights: λ_pred={initial_weights[0]}, λ_temp={initial_weights[1]}, λ_div={initial_weights[2]}")
    
    new_weights = {'lambda_prediction': 0.5, 'lambda_temporal': 0.3, 'lambda_diversity': 0.2}
    criterion.update_weights(new_weights)
    
    updated_weights = criterion.lambda_prediction, criterion.lambda_temporal, criterion.lambda_diversity
    print(f"Updated weights: λ_pred={updated_weights[0]}, λ_temp={updated_weights[1]}, λ_div={updated_weights[2]}")
    
    # Test scheduler integration
    print("\n🔄 Testing scheduler integration:")
    
    # Create scheduler
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
    print("Testing scheduler stepping:")
    for step in range(5):
        lr, weights = scheduler.step()
        criterion.update_weights(weights)
        print(f"Step {step}: LR={lr:.2e}, λ_pred={weights['lambda_prediction']:.3f}, λ_temp={weights['lambda_temporal']:.3f}, λ_div={weights['lambda_diversity']:.3f}")
    
    # Test checkpointing
    print("\n💾 Testing scheduler checkpointing:")
    state_dict = scheduler.state_dict()
    print(f"Saved scheduler state at step {scheduler.current_step}")
    
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
    
    print("✅ Scheduler checkpointing test passed!")
    
    print("\n🎉 All training integration tests passed!")


if __name__ == "__main__":
    test_training_integration() 