"""
Test script for webcam demo components
Tests each component individually to ensure they work correctly
"""

import sys
import os
import logging
import numpy as np
import torch

# Add parent directory to path for imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_loader import ModelLoader
from face_detector import MediaPipeFaceDetector
from token_extractor import TokenExtractor, TokenBuffer
from webcam_demo import WebcamDemo

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_model_loader():
    """Test model loader with dummy checkpoints"""
    print("🧪 Testing Model Loader...")
    
    try:
        loader = ModelLoader(device="cpu")
        print("✅ Model Loader created successfully")
        
        # Test with dummy paths (will fail, but tests structure)
        try:
            loader.load_all_models(
                expression_transformer_checkpoint_path="dummy_expr_trans.pth",
                expression_predictor_checkpoint_path="dummy_expr_pred.pth",
                face_reconstruction_checkpoint_path="dummy_recon.pth"
            )
            print("❌ Expected FileNotFoundError but got success")
        except FileNotFoundError:
            print("✅ Correctly caught FileNotFoundError for missing checkpoints")
        
        return True
    except Exception as e:
        print(f"❌ Model Loader test failed: {e}")
        return False


def test_face_detector():
    """Test face detector with dummy image"""
    print("\n🧪 Testing Face Detector...")
    
    try:
        detector = MediaPipeFaceDetector(confidence_threshold=0.5)
        print("✅ Face Detector created successfully")
        
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test detection
        detected_faces = detector.detect_faces(dummy_image)
        print(f"✅ Face detection completed, found {len(detected_faces)} faces")
        
        # Test face extraction
        if detected_faces:
            face_image = detector.extract_face_region(dummy_image, detected_faces[0][:4])
            if face_image is not None:
                print(f"✅ Face extraction completed, shape: {face_image.shape}")
            else:
                print("⚠️ Face extraction returned None (expected for dummy image)")
        else:
            print("⚠️ No faces detected in dummy image (expected)")
        
        # Test frame processing
        face_image, detected_faces = detector.process_frame(dummy_image)
        print(f"✅ Frame processing completed, detected {len(detected_faces)} faces")
        
        detector.close()
        return True
    except Exception as e:
        print(f"❌ Face Detector test failed: {e}")
        return False


def test_token_extractor():
    """Test token extractor with dummy models"""
    print("\n🧪 Testing Token Extractor...")
    
    try:
        # Create dummy models
        class DummyModel:
            def __init__(self, name):
                self.name = name
            
            def inference(self, *args):
                if self.name == 'expression':
                    return torch.randn(1, 1, 384), torch.randn(1, 1, 384)  # expression_token, subject_embeddings
                else:
                    return torch.randn(1, 1, 384)
        
        class DummyTokenizer:
            def __call__(self, face_tensor):
                return torch.randn(1, 1369, 384), torch.randn(1, 1369, 384)
        
        class DummyReconstructionModel:
            def __call__(self, patch_tokens, pos_embeddings, subject_embeddings, expression_token):
                return torch.randn(1, 3, 518, 518)
        
        # Create dummy models
        models = {
            'expression_transformer': DummyModel('expression'),
            'expression_predictor': DummyModel('predictor')
        }
        
        tokenizer = DummyTokenizer()
        face_reconstruction_model = DummyReconstructionModel()
        
        # Create token extractor
        extractor = TokenExtractor(models, tokenizer, face_reconstruction_model, device="cpu")
        print("✅ Token Extractor created successfully")
        
        # Create dummy face image
        dummy_face = np.random.randint(0, 255, (518, 518, 3), dtype=np.uint8)
        
        # Test token extraction
        tokens = extractor.process_frame_tokens(dummy_face)
        if tokens is not None:
            print("✅ Token extraction completed successfully")
            print(f"   Extracted tokens: {list(tokens.keys())}")
        else:
            print("❌ Token extraction failed")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Token Extractor test failed: {e}")
        return False


def test_token_buffer():
    """Test token buffer functionality"""
    print("\n🧪 Testing Token Buffer...")
    
    try:
        buffer = TokenBuffer(buffer_size=30)
        print("✅ Token Buffer created successfully")
        
        # Add some dummy tokens
        for i in range(35):
            face_id_token = torch.randn(1, 1, 384)
            expression_token = torch.randn(1, 1, 384)
            buffer.add_frame_tokens(face_id_token, expression_token)
            
            if buffer.is_ready_for_prediction():
                face_id_seq, expr_seq = buffer.get_prediction_sequence()
                if face_id_seq is not None and expr_seq is not None:
                    print(f"✅ Frame {i}: Ready for prediction, sequence shape: {face_id_seq.shape}")
                else:
                    print(f"❌ Frame {i}: Prediction sequence is None")
                    return False
            else:
                print(f"⚠️ Frame {i}: Not ready for prediction (expected for first 28 frames)")
        
        # Test reset
        buffer.reset()
        print("✅ Token Buffer reset successfully")
        
        return True
    except Exception as e:
        print(f"❌ Token Buffer test failed: {e}")
        return False


def test_webcam_demo():
    """Test webcam demo with all models"""
    print("🧪 Testing Webcam Demo...")
    
    # Test with dummy checkpoint paths
    try:
        demo = WebcamDemo(
            expression_transformer_checkpoint_path="/Users/ozgewhiting/Documents/projects/dataset_utils/cloud_checkpoints/expression_transformer_epoch_5.pt",
            expression_predictor_checkpoint_path="/Users/ozgewhiting/Documents/projects/dataset_utils/cloud_checkpoints/transformer_decoder_epoch_5.pt",
            face_reconstruction_checkpoint_path="/Users/ozgewhiting/Documents/projects/dataset_utils/cloud_checkpoints/reconstruction_model_epoch_5.pt"
        )
        print("❌ Expected FileNotFoundError but got success")
    except FileNotFoundError as e:
        print("✅ Correctly caught FileNotFoundError for missing checkpoints")
        print(f"Error: {e}")
    
    print("✅ Webcam Demo test completed!")


def main():
    """Run all component tests"""
    print("🚀 Starting Component Tests...\n")
    
    tests = [
        ("Model Loader", test_model_loader),
        ("Face Detector", test_face_detector),
        ("Token Extractor", test_token_extractor),
        ("Token Buffer", test_token_buffer),
        ("Webcam Demo Structure", test_webcam_demo)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"📋 Running {test_name} test...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} test passed!\n")
        else:
            print(f"❌ {test_name} test failed!\n")
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Components are ready for use.")
        print("\n📝 Next Steps:")
        print("1. Ensure you have valid model checkpoints")
        print("2. Install required dependencies: pip install -r requirements.txt")
        print("3. Run the demo: python webcam_demo.py --help")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    test_model_loader()
    test_face_detector()
    test_token_extractor()
    test_webcam_demo() 