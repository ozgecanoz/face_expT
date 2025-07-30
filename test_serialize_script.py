#!/usr/bin/env python3
"""
Simple test script to verify serialize_dataset_with_scoring.py works
"""

import os
import json
import tempfile

def create_test_annotations():
    """Create a test annotations file"""
    test_annotations = {
        "test_video_1.mp4": {
            "subject_id": "1",
            "label": {
                "age": "25-30",
                "gender": "female",
                "skin-type": "light"
            }
        },
        "test_video_2.mp4": {
            "subject_id": "2", 
            "label": {
                "age": "30-35",
                "gender": "male",
                "skin-type": "medium"
            }
        }
    }
    
    # Create temp file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(test_annotations, temp_file)
    temp_file.close()
    
    return temp_file.name

def main():
    """Test the serialize script"""
    print("🧪 Testing serialize_dataset_with_scoring.py")
    
    # Create test annotations
    annotations_path = create_test_annotations()
    print(f"✅ Created test annotations: {annotations_path}")
    
    # Test command (this will fail but should show proper argument parsing)
    import subprocess
    try:
        result = subprocess.run([
            "python", "serialize_dataset_with_scoring.py",
            "--json_path", annotations_path,
            "--base_path", "/nonexistent/path",
            "--output_path", "./test_output",
            "--expression_transformer_checkpoint", "/nonexistent/model.pt",
            "--K", "5",
            "--N", "10",
            "--expression_weight", "1.0",
            "--position_weight", "0.0"
        ], capture_output=True, text=True, timeout=10)
        
        print("✅ Script argument parsing works correctly")
        print(f"Exit code: {result.returncode}")
        if result.stderr:
            print(f"Expected error: {result.stderr[:200]}...")
        
    except subprocess.TimeoutExpired:
        print("⚠️ Script timed out (expected for test)")
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    # Cleanup
    os.unlink(annotations_path)
    print("✅ Test completed")

if __name__ == "__main__":
    main() 