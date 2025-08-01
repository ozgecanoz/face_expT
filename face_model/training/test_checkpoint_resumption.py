#!/usr/bin/env python3
"""
Test script for checkpoint resumption with training step tracking
"""

import torch
import torch.optim as optim
import os
import tempfile
import shutil

def test_checkpoint_resumption():
    """Test checkpoint saving and loading with training step tracking"""
    
    print("Testing Checkpoint Resumption with Training Step Tracking...")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create dummy model and optimizer
        model = torch.nn.Linear(10, 1)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        
        # Create scheduler
        def get_lr_scheduler(optimizer, num_training_steps, warmup_steps, min_lr):
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    warmup_factor = float(current_step) / float(max(1, warmup_steps - 1))
                    return min_lr / 1e-4 + warmup_factor * (1 - min_lr / 1e-4)
                else:
                    progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
                    cosine_decay = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)))
                    return max(min_lr / 1e-4, cosine_decay)
            
            return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        scheduler = get_lr_scheduler(optimizer, 100, 10, 1e-6)
        
        # Simulate training for 5 steps
        current_training_step = 0
        for step in range(5):
            # Simulate training
            dummy_input = torch.randn(1, 10, requires_grad=True)
            output = model(dummy_input)
            loss = output.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            current_training_step += 1
            
            print(f"Step {step}: LR = {scheduler.get_last_lr()[0]:.2e}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(temp_dir, "test_checkpoint.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'total_steps': current_training_step,
            'epoch': 1
        }, checkpoint_path)
        
        print(f"✅ Saved checkpoint at step {current_training_step}")
        
        # Create new model and optimizer
        new_model = torch.nn.Linear(10, 1)
        new_optimizer = optim.AdamW(new_model.parameters(), lr=1e-4, weight_decay=0.01)
        new_scheduler = get_lr_scheduler(new_optimizer, 100, 10, 1e-6)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        
        # Load model state
        new_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        new_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        new_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Get training step
        loaded_training_step = checkpoint.get('total_steps', 0)
        
        print(f"✅ Loaded checkpoint:")
        print(f"   Training step: {loaded_training_step}")
        print(f"   Current LR: {new_scheduler.get_last_lr()[0]:.2e}")
        
        # Continue training for 3 more steps
        for step in range(3):
            dummy_input = torch.randn(1, 10, requires_grad=True)
            output = new_model(dummy_input)
            loss = output.sum()
            new_optimizer.zero_grad()
            loss.backward()
            new_optimizer.step()
            new_scheduler.step()
            loaded_training_step += 1
            
            print(f"Step {loaded_training_step}: LR = {new_scheduler.get_last_lr()[0]:.2e}")
        
        print("✅ Checkpoint resumption test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    finally:
        # Clean up
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    success = test_checkpoint_resumption()
    if success:
        print("✅ All checkpoint resumption tests passed!")
    else:
        print("❌ Checkpoint resumption tests failed!")
        exit(1) 