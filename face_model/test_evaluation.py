#!/usr/bin/env python3
"""
Test script for Face ID Model evaluation
"""

import torch
import os
import sys
sys.path.append('.')

from evaluate_face_id import FaceIDEvaluator

def test_evaluation_setup():
    """Test if evaluation setup works"""
    
    # Check for checkpoint
    checkpoint_path = "/Users/ozgewhiting/Documents/projects/dataset_utils/checkpoints/face_id_epoch_3.pth"
    data_dir = "/Users/ozgewhiting/Documents/EQLabs/datasets_serial/CCA_test_db1"
    
    print("🔍 Testing Face ID Model Evaluation Setup")
    print(f"📁 Checkpoint: {checkpoint_path}")
    print(f"📊 Dataset: {data_dir}")
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Please train the model first or update the checkpoint path.")
        return False
    
    print("✅ Checkpoint found")
    
    # Check if dataset exists
    if not os.path.exists(data_dir):
        print(f"❌ Dataset not found: {data_dir}")
        return False
    
    print("✅ Dataset found")
    
    # Test evaluator initialization
    try:
        evaluator = FaceIDEvaluator(checkpoint_path, data_dir)
        print("✅ Evaluator initialized successfully")
        
        # Test with a small sample
        print("\n🧪 Testing with 5 samples...")
        results = evaluator.run_evaluation(max_samples=5)
        
        print("✅ Evaluation test completed successfully!")
        print(f"📊 Processed {len(results['identity_tokens'])} samples")
        print(f"👥 Unique subjects: {len(set(results['subject_ids']))}")
        print(f"📈 Average consistency: {sum(results['frame_consistencies'])/len(results['frame_consistencies']):.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_evaluation_setup() 