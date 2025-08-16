#!/usr/bin/env python3
"""
ArcFace Tokenizer for Face Recognition

This module provides an ArcFaceTokenizer class that loads a pre-trained ArcFace ONNX model
and extracts face embeddings from input images. The model is loaded in a frozen state
and only used for inference.
"""

import os
import numpy as np
import cv2
import onnxruntime as ort
from typing import Optional, Tuple, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArcFaceTokenizer:
    """
    ArcFace tokenizer for extracting face embeddings from images.
    
    This class loads a pre-trained ArcFace ONNX model and provides a forward method
    to extract normalized face embeddings from input images.
    """
    
    def __init__(self, model_path: str, target_size: Tuple[int, int] = (112, 112)):
        """
        Initialize ArcFace tokenizer.
        
        Args:
            model_path: Path to the ArcFace ONNX model file
            target_size: Target size for input images (height, width)
        """
        self.model_path = model_path
        self.target_size = target_size
        
        # Validate model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ArcFace model not found: {model_path}")
        
        # Load ONNX model
        logger.info(f"Loading ArcFace model from: {model_path}")
        try:
            # Create ONNX inference session
            self.sess = ort.InferenceSession(model_path)
            
            # Get input and output names
            self.input_name = self.sess.get_inputs()[0].name
            self.output_name = self.sess.get_outputs()[0].name
            
            # Get input shape info
            input_shape = self.sess.get_inputs()[0].shape
            logger.info(f"Model input shape: {input_shape}")
            
            # Verify output shape
            output_shape = self.sess.get_outputs()[0].shape
            logger.info(f"Model output shape: {output_shape}")
            
            logger.info("✅ ArcFace model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load ArcFace model: {str(e)}")
            raise
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for ArcFace model input.
        
        Args:
            image: Input image in RGB format with values in [0, 1] range
                  Can be (H, W, C) or (C, H, W) format
                  
        Returns:
            Preprocessed image ready for model input
        """
        try:
            # Ensure image is in float32 format
            if image.dtype != np.float32:
                image = image.astype(np.float32)
            
            # Check channel order and transpose if needed
            if image.shape[0] == 3:  # (C, H, W)
                image = np.transpose(image, (1, 2, 0))  # (H, W, C)
            
            # Resize to target size
            image_resized = cv2.resize(image, self.target_size)
            
            # Normalize from [0, 1] to [-1, 1] range
            image_normalized = (image_resized * 2.0) - 1.0
            
            # Add batch dimension
            image_batch = image_normalized[np.newaxis, ...]  # (1, H, W, C)
            
            return image_batch
            
        except Exception as e:
            logger.error(f"❌ Error preprocessing image: {str(e)}")
            raise
    
    def forward(self, image: np.ndarray) -> np.ndarray:
        """
        Extract face embedding from input image.
        
        Args:
            image: Input image in RGB format with values in [0, 1] range
                  Can be (H, W, C) or (C, H, W) format
                  
        Returns:
            Normalized face embedding vector
        """
        try:
            # Preprocess image
            preprocessed = self.preprocess_image(image)
            
            # Run inference
            embedding = self.sess.run([self.output_name], {self.input_name: preprocessed})[0][0]
            
            # Normalize embedding
            embedding_norm = embedding / np.linalg.norm(embedding)
            
            return embedding_norm
            
        except Exception as e:
            logger.error(f"❌ Error extracting embedding: {str(e)}")
            raise
    
    def extract_embeddings_batch(self, images: np.ndarray) -> np.ndarray:
        """
        Extract embeddings from a batch of images.
        
        Args:
            images: Batch of images with shape (N, H, W, C) or (N, C, H, W)
                   where N is batch size, values in [0, 1] range
                   
        Returns:
            Batch of normalized embeddings with shape (N, embedding_dim)
        """
        try:
            batch_size = images.shape[0]
            embeddings = []
            
            for i in range(batch_size):
                embedding = self.forward(images[i])
                embeddings.append(embedding)
            
            return np.array(embeddings)
            
        except Exception as e:
            logger.error(f"❌ Error extracting batch embeddings: {str(e)}")
            raise
    
    def get_embedding_dim(self) -> int:
        """
        Get the dimensionality of the face embeddings.
        
        Returns:
            Embedding dimension
        """
        try:
            # Create a dummy input to get output shape
            dummy_input = np.random.rand(1, *self.target_size, 3).astype(np.float32) * 2 - 1
            dummy_output = self.sess.run([self.output_name], {self.input_name: dummy_input})[0][0]
            return dummy_output.shape[0]
        except Exception as e:
            logger.error(f"❌ Error getting embedding dimension: {str(e)}")
            raise


def test_arcface_tokenizer():
    """
    Test function for ArcFaceTokenizer class.
    """
    print("🧪 Testing ArcFaceTokenizer class...")
    
    # Test with a dummy model path (this will fail, but tests the initialization logic)
    try:
        tokenizer = ArcFaceTokenizer("./dummy_model.onnx")
        print("❌ Expected failure - model file doesn't exist")
    except FileNotFoundError:
        print("✅ Correctly caught missing model file")
    
    # Test with random input data
    print("\n🧪 Testing preprocessing with random data...")
    
    # Create a mock tokenizer for testing preprocessing logic
    class MockArcFaceTokenizer:
        def __init__(self):
            self.target_size = (112, 112)
        
        def preprocess_image(self, image):
            # Copy the preprocessing logic from the real class
            if image.dtype != np.float32:
                image = image.astype(np.float32)
            
            if image.shape[0] == 3:  # (C, H, W)
                image = np.transpose(image, (1, 2, 0))  # (H, W, C)
            
            image_resized = cv2.resize(image, self.target_size)
            image_normalized = (image_resized * 2.0) - 1.0
            image_batch = image_normalized[np.newaxis, ...]
            
            return image_batch
    
    mock_tokenizer = MockArcFaceTokenizer()
    
    # Test different input formats
    test_cases = [
        ("(H, W, C) format", np.random.rand(224, 224, 3)),
        ("(C, H, W) format", np.random.rand(3, 224, 224)),
        ("Smaller size", np.random.rand(100, 100, 3)),
    ]
    
    for name, test_image in test_cases:
        try:
            preprocessed = mock_tokenizer.preprocess_image(test_image)
            print(f"   ✅ {name}: input {test_image.shape} -> output {preprocessed.shape}")
            print(f"      Output range: [{preprocessed.min():.3f}, {preprocessed.max():.3f}]")
        except Exception as e:
            print(f"   ❌ {name}: {str(e)}")
    
    print("\n🎉 ArcFaceTokenizer test completed!")


if __name__ == "__main__":
    test_arcface_tokenizer()
