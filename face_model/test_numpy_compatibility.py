#!/usr/bin/env python3
"""
Test NumPy compatibility for TensorBoard logging
"""

import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.tensorboard_utils import safe_add_histogram, log_model_parameters


def test_numpy_compatibility():
    """Test that TensorBoard logging works with different NumPy versions."""
    print("🧪 Testing NumPy compatibility for TensorBoard logging...")
    
    # Create a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 5)
    )
    
    # Create a dummy writer
    from torch.utils.tensorboard import SummaryWriter
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        writer = SummaryWriter(temp_dir)
        
        # Test individual histogram logging
        print("📊 Testing individual histogram logging...")
        for name, param in model.named_parameters():
            success = safe_add_histogram(writer, f'Test/{name}', param.data, 0)
            print(f"   {name}: {'✅' if success else '❌'}")
        
        # Test comprehensive parameter logging
        print("\n📊 Testing comprehensive parameter logging...")
        results = log_model_parameters(writer, model, global_step=0)
        
        print(f"📈 Results:")
        print(f"   Total parameters: {results['parameter_count']}")
        print(f"   Histograms successful: {results['histograms_successful']}")
        print(f"   Histograms failed: {results['histograms_failed']}")
        print(f"   Statistics successful: {results['statistics_successful']}")
        print(f"   Statistics failed: {results['statistics_failed']}")
        
        writer.close()
    
    print("\n✅ NumPy compatibility test completed!")
    return results


if __name__ == "__main__":
    test_numpy_compatibility() 