#!/usr/bin/env python3
"""
Test script for evaluation functions
"""

import numpy as np
from training.eval_expT_supervised import compute_evaluation_metrics, print_evaluation_results


def test_evaluation_metrics():
    """Test the evaluation metrics computation"""
    print("🧪 Testing Evaluation Metrics...")
    
    # Create test data
    num_samples = 100
    num_classes = 8
    
    # Generate random predictions and targets
    np.random.seed(42)  # For reproducible results
    targets = np.random.randint(0, num_classes, num_samples)
    predictions = np.random.randint(0, num_classes, num_samples)
    probabilities = np.random.rand(num_samples, num_classes)
    
    # Normalize probabilities
    probabilities = probabilities / np.sum(probabilities, axis=1, keepdims=True)
    
    print(f"Test data: {num_samples} samples, {num_classes} classes")
    print(f"Targets range: {targets.min()} to {targets.max()}")
    print(f"Predictions range: {predictions.min()} to {predictions.max()}")
    
    # Test metrics computation
    try:
        results = compute_evaluation_metrics(predictions, targets, probabilities, num_classes)
        print("✅ Evaluation metrics computed successfully")
        
        # Test results printing
        print("\n📊 Testing results printing...")
        print_evaluation_results(results)
        print("✅ Results printing completed")
        
        # Verify results structure
        assert 'overall' in results, "Missing 'overall' key"
        assert 'per_class' in results, "Missing 'per_class' key"
        assert 'confusion_matrix' in results, "Missing 'confusion_matrix' key"
        assert 'predictions' in results, "Missing 'predictions' key"
        assert 'targets' in results, "Missing 'targets' key"
        assert 'probabilities' in results, "Missing 'probabilities' key"
        
        print("✅ All required keys present in results")
        
        # Verify overall metrics
        overall = results['overall']
        assert 'accuracy' in overall, "Missing 'accuracy' in overall metrics"
        assert 'macro_precision' in overall, "Missing 'macro_precision' in overall metrics"
        assert 'macro_recall' in overall, "Missing 'macro_recall' in overall metrics"
        assert 'macro_f1' in overall, "Missing 'macro_f1' in overall metrics"
        
        print("✅ Overall metrics structure correct")
        
        # Verify per-class metrics
        per_class = results['per_class']
        assert len(per_class) == num_classes, f"Expected {num_classes} classes, got {len(per_class)}"
        
        for class_id in range(num_classes):
            if class_id in per_class:
                class_metrics = per_class[class_id]
                assert 'samples' in class_metrics, f"Missing 'samples' for class {class_id}"
                assert 'correct' in class_metrics, f"Missing 'correct' for class {class_id}"
                assert 'accuracy' in class_metrics, f"Missing 'accuracy' for class {class_id}"
                assert 'precision' in class_metrics, f"Missing 'precision' for class {class_id}"
                assert 'recall' in class_metrics, f"Missing 'recall' for class {class_id}"
                assert 'f1_score' in class_metrics, f"Missing 'f1_score' for class {class_id}"
        
        print("✅ Per-class metrics structure correct")
        
        # Verify confusion matrix
        conf_matrix = np.array(results['confusion_matrix'])
        assert conf_matrix.shape == (num_classes, num_classes), f"Expected ({num_classes}, {num_classes}), got {conf_matrix.shape}"
        
        print("✅ Confusion matrix structure correct")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_class_names():
    """Test evaluation with custom class names"""
    print("\n🧪 Testing Evaluation with Custom Class Names...")
    
    # Create test data
    num_samples = 50
    num_classes = 8
    
    # Generate test data
    np.random.seed(123)
    targets = np.random.randint(0, num_classes, num_samples)
    predictions = np.random.randint(0, num_classes, num_samples)
    probabilities = np.random.rand(num_samples, num_classes)
    probabilities = probabilities / np.sum(probabilities, axis=1, keepdims=True)
    
    # Custom class names (AffectNet emotions)
    class_names = [
        "Neutral", "Happy", "Sad", "Surprise", 
        "Fear", "Disgust", "Anger", "Contempt"
    ]
    
    try:
        # Compute metrics
        results = compute_evaluation_metrics(predictions, targets, probabilities, num_classes)
        
        # Print with class names
        print_evaluation_results(results, class_names)
        print("✅ Custom class names test completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Custom class names test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 Starting Evaluation Function Tests")
    print("=" * 50)
    
    # Test basic evaluation
    test1_success = test_evaluation_metrics()
    
    # Test with custom class names
    test2_success = test_with_class_names()
    
    # Summary
    print("\n" + "=" * 50)
    if test1_success and test2_success:
        print("🎉 All evaluation tests passed!")
    else:
        print("💥 Some evaluation tests failed!")
    
    print("=" * 50) 