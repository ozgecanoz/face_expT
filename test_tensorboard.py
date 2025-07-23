#!/usr/bin/env python3
"""
Test script to verify TensorBoard works with PyTorch on Mac CPU
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import time

# Create log directory
log_dir = "./logs/test_run"
os.makedirs(log_dir, exist_ok=True)

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir)

# Simple test model
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)

# Test model
model = TestModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

print("Starting TensorBoard test...")
print(f"Log directory: {log_dir}")
print("Run 'tensorboard --logdir=./logs --port=6006' to view results")

# Simulate training
for step in range(100):
    # Generate dummy data
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)
    
    # Forward pass
    output = model(x)
    loss = criterion(output, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Log to TensorBoard
    writer.add_scalar('Loss/Train', loss.item(), step)
    writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], step)
    
    # Log some dummy images
    if step % 20 == 0:
        dummy_images = torch.randn(4, 3, 64, 64)  # 4 RGB images
        writer.add_images('Test_Images', dummy_images, step)
    
    if step % 10 == 0:
        print(f"Step {step}: Loss = {loss.item():.4f}")

writer.close()
print("Test completed! Check TensorBoard at http://localhost:6006") 