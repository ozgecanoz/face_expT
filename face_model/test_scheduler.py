#!/usr/bin/env python3
"""
Test the combined LR and loss weight scheduler
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.scheduler_utils import CombinedLRLossWeightScheduler


def test_scheduler_behavior():
    """Test the scheduler behavior at different steps."""
    print("🧪 Testing CombinedLRLossWeightScheduler...")
    
    # Create a dummy optimizer
    dummy_params = [torch.nn.Parameter(torch.randn(10))]
    optimizer = torch.optim.Adam(dummy_params, lr=1e-4)
    
    # Test configuration
    initial_lr = 1e-6
    warmup_lr = 1e-4
    final_lr = 1e-6
    
    initial_weights = {'lambda_prediction': 0.1, 'lambda_temporal': 0.5, 'lambda_diversity': 0.5}
    warmup_weights = {'lambda_prediction': 0.3, 'lambda_temporal': 0.4, 'lambda_diversity': 0.4}
    final_weights = {'lambda_prediction': 0.5, 'lambda_temporal': 0.3, 'lambda_diversity': 0.3}
    
    warmup_steps = 1000
    total_steps = 5000
    
    # Create scheduler
    scheduler = CombinedLRLossWeightScheduler(
        optimizer=optimizer,
        initial_lr=initial_lr,
        warmup_lr=warmup_lr,
        final_lr=final_lr,
        initial_weights=initial_weights,
        warmup_weights=warmup_weights,
        final_weights=final_weights,
        warmup_steps=warmup_steps,
        total_steps=total_steps
    )
    
    # Test key points
    test_steps = [0, 500, 1000, 2000, 3000, 4000, 4999]
    
    print("\n📊 Scheduler behavior at key steps:")
    print("Step    | LR        | λ_pred | λ_temp | λ_div")
    print("-" * 50)
    
    for step in test_steps:
        # Reset scheduler to step
        scheduler.current_step = step
        
        # Get current values
        lr = scheduler.get_last_lr()
        weights = scheduler.get_current_weights()
        
        print(f"{step:6d} | {lr:8.2e} | {weights['lambda_prediction']:6.3f} | {weights['lambda_temporal']:6.3f} | {weights['lambda_diversity']:6.3f}")
    
    # Test stepping behavior
    print("\n🔄 Testing step-by-step behavior:")
    scheduler.current_step = 0
    
    # Test first few steps
    for step in range(5):
        lr, weights = scheduler.step()
        print(f"Step {step}: LR={lr:.2e}, λ_pred={weights['lambda_prediction']:.3f}, λ_temp={weights['lambda_temporal']:.3f}, λ_div={weights['lambda_diversity']:.3f}")
    
    # Test around warmup boundary
    print(f"\n🔄 Testing around warmup boundary (step {warmup_steps}):")
    scheduler.current_step = warmup_steps - 2
    
    for step in range(warmup_steps - 2, warmup_steps + 3):
        scheduler.current_step = step
        lr = scheduler.get_last_lr()
        weights = scheduler.get_current_weights()
        print(f"Step {step}: LR={lr:.2e}, λ_pred={weights['lambda_prediction']:.3f}, λ_temp={weights['lambda_temporal']:.3f}, λ_div={weights['lambda_diversity']:.3f}")
    
    # Test checkpointing
    print("\n💾 Testing checkpointing:")
    scheduler.current_step = 1000
    state_dict = scheduler.state_dict()
    print(f"Saved state at step {scheduler.current_step}")
    
    # Create new scheduler and load state
    new_scheduler = CombinedLRLossWeightScheduler(
        optimizer=optimizer,
        initial_lr=initial_lr,
        warmup_lr=warmup_lr,
        final_lr=final_lr,
        initial_weights=initial_weights,
        warmup_weights=warmup_weights,
        final_weights=final_weights,
        warmup_steps=warmup_steps,
        total_steps=total_steps
    )
    
    new_scheduler.load_state_dict(state_dict)
    print(f"Loaded state, current step: {new_scheduler.current_step}")
    
    # Verify values match
    lr1 = scheduler.get_last_lr()
    weights1 = scheduler.get_current_weights()
    lr2 = new_scheduler.get_last_lr()
    weights2 = new_scheduler.get_current_weights()
    
    assert abs(lr1 - lr2) < 1e-10, "LR mismatch after checkpointing"
    for key in weights1:
        assert abs(weights1[key] - weights2[key]) < 1e-10, f"Weight {key} mismatch after checkpointing"
    
    print("✅ Checkpointing test passed!")
    
    print("\n🎉 All scheduler tests passed!")


if __name__ == "__main__":
    test_scheduler_behavior() 