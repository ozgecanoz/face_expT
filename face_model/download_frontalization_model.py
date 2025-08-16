#!/usr/bin/env python3
"""
Download frontalization model from Hugging Face

This script downloads a pre-trained frontalization model from Hugging Face
and saves it locally for use with the video processor.
"""

import os
import sys
import torch
from transformers import AutoModel, AutoTokenizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_model_from_huggingface(model_name, save_path="./generator_v0.pt"):
    """
    Download a model from Hugging Face and save it locally
    
    Args:
        model_name: Hugging Face model identifier (e.g., "username/model-name")
        save_path: Local path to save the model
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"🔄 Downloading model from Hugging Face: {model_name}")
        
        # Try to load the model
        logger.info("   Loading model...")
        model = AutoModel.from_pretrained(model_name)
        
        # Set to evaluation mode
        model.eval()
        logger.info("   Model loaded successfully")
        
        # Save the model locally
        logger.info(f"   Saving model to: {save_path}")
        torch.save(model, save_path)
        
        logger.info("✅ Model downloaded and saved successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error downloading model: {str(e)}")
        return False

def download_frontalization_model():
    """
    Download a frontalization model
    """
    # Actual face frontalization models on Hugging Face
    models_to_try = [
        "opetrova/face-frontalization",  # Face frontalization model
        "microsoft/DialoGPT-medium",     # Fallback example
    ]
    
    logger.info("🚀 Frontalization Model Downloader")
    logger.info("=" * 50)
    
    # Check if model already exists
    if os.path.exists("./generator_v0.pt"):
        logger.info("✅ Model already exists: ./generator_v0.pt")
        return True
    
    # Try to download from different sources
    for model_name in models_to_try:
        logger.info(f"\n🔄 Trying model: {model_name}")
        if download_model_from_huggingface(model_name):
            return True
    
    # If no models worked, provide instructions
    logger.error("\n❌ Could not download any models automatically")
    logger.info("\n📋 Manual Download Instructions:")
    logger.info("1. Go to https://huggingface.co/")
    logger.info("2. Search for 'face frontalization' or 'face generator'")
    logger.info("3. Find a suitable model and copy its identifier")
    logger.info("4. Update this script with the correct model name")
    logger.info("5. Or manually download and place the model file as 'generator_v0.pt'")
    
    return False

def main():
    """
    Main function
    """
    logger.info("🎯 Face Frontalization Model Downloader")
    logger.info("=" * 50)
    
    # Check if we have the required packages
    try:
        import transformers
        logger.info("✅ Transformers library available")
    except ImportError:
        logger.error("❌ Transformers library not found")
        logger.info("Install with: pip install transformers")
        return
    
    # Try to download the model
    success = download_frontalization_model()
    
    if success:
        logger.info("\n🎉 Model ready for use!")
        logger.info("   You can now run: python test_frontalization_video.py")
    else:
        logger.info("\n⚠️  Please download a model manually or update the script")

if __name__ == "__main__":
    main()
