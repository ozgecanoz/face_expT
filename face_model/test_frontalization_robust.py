import gradio as gr
import numpy as np
import os
import sys

import torch
from torchvision import transforms
from torch.autograd import Variable

from PIL import Image
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

def load_frontalization_model(model_path="./generator_v0.pt"):
    """
    Load the frontalization generator model with error handling
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Loaded model or None if loading fails
    """
    try:
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            print("Please ensure the generator_v0.pt file is in the current directory")
            return None
            
        print(f"🔄 Loading frontalization model from: {model_path}")
        model = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Set model to evaluation mode
        model.eval()
        print("✅ Model loaded successfully and set to evaluation mode")
        
        return model
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return None

def frontalize(image, model):
    """
    Apply frontalization to an input image
    
    Args:
        image: Input numpy image array
        model: Loaded frontalization model
        
    Returns:
        Frontalized image as numpy array
    """
    try:
        if model is None:
            return image  # Return original image if model not loaded
            
        # Convert the test image to a [1, 3, 128, 128]-shaped torch tensor
        # (as required by the frontalization model)
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(128, 128)),
            transforms.ToTensor()
        ])
        
        input_tensor = torch.unsqueeze(preprocess(image), 0)
        print(f"📊 Input tensor shape: {input_tensor.shape}")
        
        # Use the saved model to generate an output (whose values go between -1 and 1,
        # and this will need to get fixed before the output is displayed)
        with torch.no_grad():  # Disable gradient computation for inference
            generated_image = model(Variable(input_tensor.type('torch.FloatTensor')))
        
        generated_image = generated_image.detach().squeeze().permute(1, 2, 0).numpy()
        generated_image = (generated_image + 1.0) / 2.0  # Convert from [-1,1] to [0,1]
        
        print(f"📊 Output image shape: {generated_image.shape}")
        print(f"📊 Output value range: [{generated_image.min():.3f}, {generated_image.max():.3f}]")
        
        return generated_image
        
    except Exception as e:
        print(f"❌ Error during frontalization: {str(e)}")
        return image  # Return original image on error

def create_gradio_interface():
    """
    Create and launch the Gradio interface
    """
    # Load the model
    model = load_frontalization_model()
    
    if model is None:
        print("⚠️  Running without model - interface will show original images")
    
    # Create the interface
    iface = gr.Interface(
        fn=lambda img: frontalize(img, model),
        inputs=gr.inputs.Image(type="numpy", label="Input Face Image"),
        outputs=gr.outputs.Image(label="Frontalized Face"),
        title="Face Frontalization Demo",
        description="Upload a face image to see the frontalized version",
        examples=[
            ["example_face1.jpg"] if os.path.exists("example_face1.jpg") else None,
            ["example_face2.jpg"] if os.path.exists("example_face2.jpg") else None,
        ],
        cache_examples=False
    )
    
    return iface

if __name__ == "__main__":
    print("🚀 Starting Face Frontalization Test")
    print("=" * 50)
    
    # Check if we're in the right directory
    current_dir = os.getcwd()
    print(f"📁 Current directory: {current_dir}")
    
    # List available model files
    model_files = [f for f in os.listdir('.') if f.endswith('.pt')]
    if model_files:
        print(f"🔍 Found model files: {model_files}")
    else:
        print("⚠️  No .pt model files found in current directory")
    
    # Create and launch interface
    interface = create_gradio_interface()
    print("🌐 Launching Gradio interface...")
    interface.launch(share=False, debug=True)
