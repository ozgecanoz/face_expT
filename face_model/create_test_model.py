#!/usr/bin/env python3
"""
Create a simple test frontalization model

This script creates a basic test model that can be used to test the video processing pipeline
while you work on getting the actual frontalization model.
"""

import torch
import torch.nn as nn
import os

class SimpleTestFrontalizationModel(nn.Module):
    """
    A simple test model that mimics the expected input/output interface
    of a frontalization model for testing purposes.
    
    This is NOT a real frontalization model - it just passes through the input
    with some basic transformations to test the video processing pipeline.
    """
    
    def __init__(self):
        super().__init__()
        
        # Simple convolutional layers for testing
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        
        # Activation and normalization
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm2d(64)
        
    def forward(self, x):
        """
        Forward pass - simple convolution operations
        
        Args:
            x: Input tensor of shape [batch, 3, 128, 128]
            
        Returns:
            Output tensor of shape [batch, 3, 128, 128] with values in [-1, 1]
        """
        # Apply convolutions
        x = self.relu(self.batch_norm(self.conv1(x)))
        x = self.relu(self.batch_norm(self.conv2(x)))
        x = self.conv3(x)
        
        # Apply tanh to get values in [-1, 1] range
        x = torch.tanh(x)
        
        return x

def create_test_model(save_path="./generator_v0.pt"):
    """
    Create and save a test frontalization model
    
    Args:
        save_path: Path to save the test model
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print("🔧 Creating test frontalization model...")
        
        # Create the model
        model = SimpleTestFrontalizationModel()
        
        # Set to evaluation mode
        model.eval()
        
        # Test the model with dummy input
        print("   Testing model with dummy input...")
        dummy_input = torch.randn(1, 3, 128, 128)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
        
        # Save the model
        print(f"   Saving model to: {save_path}")
        torch.save(model, save_path)
        
        print("✅ Test model created and saved successfully!")
        print(f"   Model saved to: {save_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating test model: {str(e)}")
        return False

def test_model_loading(save_path="./generator_v0.pt"):
    """
    Test loading the saved model
    
    Args:
        save_path: Path to the saved model
        
    Returns:
        bool: True if loading successful, False otherwise
    """
    try:
        print("\n🧪 Testing model loading...")
        
        # Load the model
        model = torch.load(save_path, map_location=torch.device('cpu'))
        
        # Set to evaluation mode
        model.eval()
        
        # Test inference
        dummy_input = torch.randn(1, 3, 128, 128)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"   Model loaded successfully")
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {str(e)}")
        return False

def main():
    """
    Main function
    """
    print("🚀 Test Frontalization Model Creator")
    print("=" * 50)
    
    # Check if model already exists
    if os.path.exists("./generator_v0.pt"):
        print("✅ Model already exists: ./generator_v0.pt")
        print("   Testing existing model...")
        if test_model_loading():
            print("\n🎉 Model is ready for use!")
            print("   You can now run: python test_frontalization_video.py")
        return
    
    # Create the test model
    success = create_test_model()
    
    if success:
        # Test the model
        test_model_loading()
        
        print("\n🎉 Test model is ready!")
        print("   You can now run: python test_frontalization_video.py")
        print("\n⚠️  Note: This is a TEST model that doesn't perform real frontalization.")
        print("   It's designed to test the video processing pipeline.")
        print("   Replace it with a real frontalization model when available.")
    else:
        print("\n❌ Failed to create test model")

if __name__ == "__main__":
    main()
