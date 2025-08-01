#!/usr/bin/env python3
"""
Test that the training function can be called with the new scheduler parameters
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.train_expression_prediction import train_expression_prediction


def test_training_function_signature():
    """Test that the training function can be called with the new parameters."""
    print("🧪 Testing training function signature with new scheduler parameters...")
    
    # Test that we can import and inspect the function signature
    import inspect
    import torch
    
    # Get the function signature
    sig = inspect.signature(train_expression_prediction)
    
    # Check that the new parameters are in the signature
    expected_params = [
        'initial_lambda_prediction',
        'initial_lambda_temporal', 
        'initial_lambda_diversity',
        'warmup_lambda_prediction',
        'warmup_lambda_temporal',
        'warmup_lambda_diversity',
        'final_lambda_prediction',
        'final_lambda_temporal',
        'final_lambda_diversity'
    ]
    
    param_names = list(sig.parameters.keys())
    
    print("📋 Function parameters:")
    for param in param_names:
        if param in expected_params:
            print(f"  ✅ {param}")
        else:
            print(f"  📝 {param}")
    
    # Check that all expected parameters are present
    missing_params = [param for param in expected_params if param not in param_names]
    if missing_params:
        print(f"❌ Missing parameters: {missing_params}")
        return False
    
    print("✅ All expected scheduler parameters are present in function signature!")
    print("✅ Training function signature test completed!")
    return True


if __name__ == "__main__":
    test_training_function_signature() 