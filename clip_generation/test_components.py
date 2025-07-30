#!/usr/bin/env python3
"""
Test script for clip generation components
"""

import os
import sys
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clip_generation.video_expression_analyzer import VideoExpressionAnalyzer, FrameData, SequenceScore
from clip_generation.clip_extractor import ClipExtractor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_video_expression_analyzer():
    """Test the VideoExpressionAnalyzer class"""
    logger.info("🧪 Testing VideoExpressionAnalyzer...")
    
    # Test with dummy checkpoint path (will fail but tests initialization)
    try:
        analyzer = VideoExpressionAnalyzer(
            expression_transformer_checkpoint_path="/dummy/path/model.pt",
            device="cpu"
        )
        logger.info("❌ Expected FileNotFoundError but got success")
        return False
    except FileNotFoundError:
        logger.info("✅ Correctly caught FileNotFoundError for missing checkpoint")
        return True
    except Exception as e:
        logger.info(f"✅ Caught expected error: {e}")
        return True


def test_clip_extractor():
    """Test the ClipExtractor class"""
    logger.info("🧪 Testing ClipExtractor...")
    
    # Create extractor
    extractor = ClipExtractor(output_dir="./test_output")
    
    # Test with dummy data
    dummy_sequences = [
        {
            'start_frame': 0,
            'end_frame': 29,
            'expression_variation': 0.5,
            'position_stability': 0.8,
            'combined_score': 0.65
        }
    ]
    
    # Test metadata creation
    try:
        # Create dummy analysis results directory
        os.makedirs("./test_analysis", exist_ok=True)
        
        # Create dummy sequences.json
        import json
        with open("./test_analysis/sequences.json", 'w') as f:
            json.dump(dummy_sequences, f)
        
        # Test metadata creation
        metadata_path = extractor.create_clip_metadata("./test_analysis", [])
        logger.info(f"✅ Created metadata: {metadata_path}")
        
        # Test summary report
        report_path = extractor.create_summary_report("./test_analysis", [])
        logger.info(f"✅ Created summary report: {report_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False
    finally:
        # Cleanup
        import shutil
        if os.path.exists("./test_output"):
            shutil.rmtree("./test_output")
        if os.path.exists("./test_analysis"):
            shutil.rmtree("./test_analysis")


def test_data_structures():
    """Test the data structures"""
    logger.info("🧪 Testing data structures...")
    
    # Test FrameData
    import torch
    frame_data = FrameData(
        frame_idx=0,
        bbox=(100, 100, 200, 200),
        confidence=0.95,
        expression_token=torch.randn(1, 1, 384),
        face_center=(200, 200)
    )
    logger.info(f"✅ FrameData created: {frame_data}")
    
    # Test SequenceScore
    sequence_score = SequenceScore(
        start_frame=0,
        end_frame=29,
        expression_variation=0.6,
        position_stability=0.8,
        combined_score=0.7,
        frame_data=[frame_data]
    )
    logger.info(f"✅ SequenceScore created: {sequence_score}")
    
    return True


def main():
    """Run all tests"""
    logger.info("🚀 Starting clip generation component tests...")
    
    tests = [
        test_data_structures,
        test_clip_extractor,
        test_video_expression_analyzer
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
                logger.info("✅ Test passed")
            else:
                logger.error("❌ Test failed")
        except Exception as e:
            logger.error(f"❌ Test failed with exception: {e}")
    
    logger.info(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed!")
        return True
    else:
        logger.error("❌ Some tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 