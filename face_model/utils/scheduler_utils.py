#!/usr/bin/env python3
"""
Combined Learning Rate and Loss Weight Scheduler
Handles both LR scheduling and loss weight scheduling with warmup and linear interpolation
"""

import torch
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


class CombinedLRLossWeightScheduler:
    """
    Combined scheduler that handles both learning rate and loss weight scheduling.
    
    Loss weights follow this pattern:
    - Initial weights (step 0 to warmup_steps): Linear interpolation from initial to warmup weights
    - Warmup weights (warmup_steps to total_steps): Linear interpolation from warmup to final weights
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_lr: float,
        warmup_lr: float,
        final_lr: float,
        initial_weights: Dict[str, float],
        warmup_weights: Dict[str, float],
        final_weights: Dict[str, float],
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6
    ):
        """
        Initialize the combined scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            initial_lr: Initial learning rate
            warmup_lr: Learning rate at warmup_steps
            final_lr: Final learning rate at total_steps
            initial_weights: Initial loss weights {'lambda_prediction': 0.1, 'lambda_temporal': 0.5, 'lambda_diversity': 0.5}
            warmup_weights: Loss weights at warmup_steps
            final_weights: Final loss weights at total_steps
            warmup_steps: Number of warmup steps
            total_steps: Total number of training steps
            min_lr: Minimum learning rate
        """
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.warmup_lr = warmup_lr
        self.final_lr = final_lr
        self.initial_weights = initial_weights
        self.warmup_weights = warmup_weights
        self.final_weights = final_weights
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.current_step = 0
        
        # Validate weight dictionaries have same keys
        weight_keys = set(initial_weights.keys())
        assert weight_keys == set(warmup_weights.keys()), "Weight dictionaries must have same keys"
        assert weight_keys == set(final_weights.keys()), "Weight dictionaries must have same keys"
        
        logger.info(f"Initialized CombinedLRLossWeightScheduler:")
        logger.info(f"  LR: {initial_lr} -> {warmup_lr} -> {final_lr}")
        logger.info(f"  Weights: {initial_weights} -> {warmup_weights} -> {final_weights}")
        logger.info(f"  Steps: warmup={warmup_steps}, total={total_steps}")
    
    def step(self) -> Tuple[float, Dict[str, float]]:
        """
        Step the scheduler and return current LR and loss weights.
        
        Returns:
            Tuple of (current_lr, current_weights)
        """
        # Calculate current learning rate
        if self.current_step < self.warmup_steps:
            # Warmup phase: linear interpolation from initial_lr to warmup_lr
            progress = self.current_step / self.warmup_steps
            current_lr = self.initial_lr + (self.warmup_lr - self.initial_lr) * progress
        else:
            # Post-warmup phase: linear interpolation from warmup_lr to final_lr
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)  # Clamp to 1.0
            current_lr = self.warmup_lr + (self.final_lr - self.warmup_lr) * progress
        
        # Ensure LR doesn't go below minimum
        current_lr = max(current_lr, self.min_lr)
        
        # Calculate current loss weights
        current_weights = {}
        for key in self.initial_weights:
            if self.current_step < self.warmup_steps:
                # Warmup phase: linear interpolation from initial to warmup weights
                progress = self.current_step / self.warmup_steps
                initial_weight = self.initial_weights[key]
                warmup_weight = self.warmup_weights[key]
                current_weights[key] = initial_weight + (warmup_weight - initial_weight) * progress
            else:
                # Post-warmup phase: linear interpolation from warmup to final weights
                progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                progress = min(progress, 1.0)  # Clamp to 1.0
                warmup_weight = self.warmup_weights[key]
                final_weight = self.final_weights[key]
                current_weights[key] = warmup_weight + (final_weight - warmup_weight) * progress
        
        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr
        
        self.current_step += 1
        
        return current_lr, current_weights
    
    def get_last_lr(self) -> float:
        """Get the last learning rate."""
        if self.current_step == 0:
            return self.initial_lr
        
        # Calculate what the LR would be at current_step - 1
        step = self.current_step - 1
        if step < self.warmup_steps:
            progress = step / self.warmup_steps
            lr = self.initial_lr + (self.warmup_lr - self.initial_lr) * progress
        else:
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            lr = self.warmup_lr + (self.final_lr - self.warmup_lr) * progress
        
        return max(lr, self.min_lr)
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get the current loss weights."""
        if self.current_step == 0:
            return self.initial_weights.copy()
        
        # Calculate what the weights would be at current_step - 1
        step = self.current_step - 1
        current_weights = {}
        
        for key in self.initial_weights:
            if step < self.warmup_steps:
                progress = step / self.warmup_steps
                initial_weight = self.initial_weights[key]
                warmup_weight = self.warmup_weights[key]
                current_weights[key] = initial_weight + (warmup_weight - initial_weight) * progress
            else:
                progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                progress = min(progress, 1.0)
                warmup_weight = self.warmup_weights[key]
                final_weight = self.final_weights[key]
                current_weights[key] = warmup_weight + (final_weight - warmup_weight) * progress
        
        return current_weights
    
    def state_dict(self) -> Dict[str, Any]:
        """Get scheduler state for checkpointing."""
        return {
            'current_step': self.current_step,
            'initial_lr': self.initial_lr,
            'warmup_lr': self.warmup_lr,
            'final_lr': self.final_lr,
            'initial_weights': self.initial_weights,
            'warmup_weights': self.warmup_weights,
            'final_weights': self.final_weights,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'min_lr': self.min_lr
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load scheduler state from checkpoint."""
        self.current_step = state_dict['current_step']
        self.initial_lr = state_dict['initial_lr']
        self.warmup_lr = state_dict['warmup_lr']
        self.final_lr = state_dict['final_lr']
        self.initial_weights = state_dict['initial_weights']
        self.warmup_weights = state_dict['warmup_weights']
        self.final_weights = state_dict['final_weights']
        self.warmup_steps = state_dict['warmup_steps']
        self.total_steps = state_dict['total_steps']
        self.min_lr = state_dict['min_lr']


def test_combined_scheduler():
    """Test the combined scheduler to visualize LR and weight changes."""
    import matplotlib.pyplot as plt
    import numpy as np
    
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
    
    # Collect data
    steps = []
    lrs = []
    prediction_weights = []
    temporal_weights = []
    diversity_weights = []
    
    for step in range(total_steps):
        lr, weights = scheduler.step()
        steps.append(step)
        lrs.append(lr)
        prediction_weights.append(weights['lambda_prediction'])
        temporal_weights.append(weights['lambda_temporal'])
        diversity_weights.append(weights['lambda_diversity'])
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot learning rate
    ax1.plot(steps, lrs, 'b-', linewidth=2, label='Learning Rate')
    ax1.axvline(x=warmup_steps, color='r', linestyle='--', alpha=0.7, label='Warmup End')
    ax1.set_ylabel('Learning Rate')
    ax1.set_title('Learning Rate Schedule')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot loss weights
    ax2.plot(steps, prediction_weights, 'g-', linewidth=2, label='λ_prediction')
    ax2.plot(steps, temporal_weights, 'r-', linewidth=2, label='λ_temporal')
    ax2.plot(steps, diversity_weights, 'b-', linewidth=2, label='λ_diversity')
    ax2.axvline(x=warmup_steps, color='r', linestyle='--', alpha=0.7, label='Warmup End')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Loss Weight')
    ax2.set_title('Loss Weight Schedule')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('scheduler_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Scheduler test completed! Check 'scheduler_test.png' for visualization.")
    print(f"Final values at step {total_steps-1}:")
    print(f"  LR: {lrs[-1]:.2e}")
    print(f"  λ_prediction: {prediction_weights[-1]:.3f}")
    print(f"  λ_temporal: {temporal_weights[-1]:.3f}")
    print(f"  λ_diversity: {diversity_weights[-1]:.3f}")


if __name__ == "__main__":
    test_combined_scheduler() 