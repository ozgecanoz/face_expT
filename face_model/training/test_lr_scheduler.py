#!/usr/bin/env python3
"""
Test script for learning rate scheduler with warmup and cosine decay
"""

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import math

def test_lr_scheduler():
    """Test the learning rate scheduler implementation"""
    
    # Test parameters
    learning_rate = 5e-5
    warmup_steps = 1000
    min_lr = 1e-6
    total_steps = 5000
    
    # Create dummy optimizer
    dummy_params = [torch.nn.Parameter(torch.randn(10, 10))]
    optimizer = optim.AdamW(dummy_params, lr=learning_rate, weight_decay=0.01)
    
    # Create scheduler
    def get_lr_scheduler(optimizer, num_training_steps, warmup_steps, min_lr):
        """
        Create a learning rate scheduler with warmup and cosine decay
        """
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup from min_lr to learning_rate
                warmup_factor = float(current_step) / float(max(1, warmup_steps - 1))
                return min_lr / learning_rate + warmup_factor * (1 - min_lr / learning_rate)
            else:
                # Cosine decay
                progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                return max(min_lr / learning_rate, cosine_decay)
        
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    scheduler = get_lr_scheduler(optimizer, total_steps, warmup_steps, min_lr)
    
    # Track learning rates
    lrs = []
    steps = []
    
    print("Testing Learning Rate Scheduler:")
    print(f"Initial LR: {learning_rate}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Min LR: {min_lr}")
    print(f"Total steps: {total_steps}")
    print()
    
    for step in range(total_steps):
        current_lr = scheduler.get_last_lr()[0]
        lrs.append(current_lr)
        steps.append(step)
        
        # Print key points
        if step == 0:
            print(f"Step {step}: LR = {current_lr:.2e} (start)")
        elif step == warmup_steps - 1:
            print(f"Step {step}: LR = {current_lr:.2e} (end of warmup)")
        elif step == warmup_steps:
            print(f"Step {step}: LR = {current_lr:.2e} (start of decay)")
        elif step == total_steps - 1:
            print(f"Step {step}: LR = {current_lr:.2e} (end)")
        
        # Call optimizer.step() before scheduler.step() to avoid warning
        optimizer.step()
        scheduler.step()
    
    # Verify key properties
    assert abs(lrs[0] - min_lr) < 1e-10, f"Initial LR should be min_lr, got {lrs[0]}"
    assert abs(lrs[warmup_steps-1] - learning_rate) < 1e-8, f"LR at end of warmup should be learning_rate, got {lrs[warmup_steps-1]}"
    assert lrs[-1] >= min_lr, f"Final LR should be >= min_lr, got {lrs[-1]}"
    
    print(f"\n✅ Scheduler test passed!")
    print(f"   Initial LR: {lrs[0]:.2e}")
    print(f"   Peak LR: {max(lrs):.2e}")
    print(f"   Final LR: {lrs[-1]:.2e}")
    
    # Plot the learning rate curve
    plt.figure(figsize=(10, 6))
    plt.plot(steps, lrs)
    plt.axvline(x=warmup_steps, color='r', linestyle='--', label=f'Warmup end ({warmup_steps} steps)')
    plt.axhline(y=learning_rate, color='g', linestyle='--', label=f'Peak LR ({learning_rate:.2e})')
    plt.axhline(y=min_lr, color='orange', linestyle='--', label=f'Min LR ({min_lr:.2e})')
    plt.xlabel('Training Steps')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule: Warmup + Cosine Decay')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('lr_schedule_test.png', dpi=150, bbox_inches='tight')
    print(f"📊 Learning rate curve saved to: lr_schedule_test.png")
    
    return True

if __name__ == "__main__":
    test_lr_scheduler() 