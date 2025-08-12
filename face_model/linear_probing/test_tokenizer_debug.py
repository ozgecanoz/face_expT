#!/usr/bin/env python3
"""
Debug script to test DINOv2BaseTokenizer and positional embeddings extraction
"""

import os
import sys
import logging

# Add the project root to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.dinov2_tokenizer import DINOv2BaseTokenizer

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_tokenizer():
    """Test the DINOv2BaseTokenizer"""
    print("🧪 Testing DINOv2BaseTokenizer...")
    
    try:
        # Initialize tokenizer
        tokenizer = DINOv2BaseTokenizer(device="cpu")
        print("✅ Tokenizer initialized successfully")
        
        # Test forward pass
        import torch
        x = torch.randn(1, 3, 518, 518)
        print(f"Input shape: {x.shape}")
        
        patch_tokens, pos_embeddings = tokenizer(x)
        print(f"Patch tokens shape: {patch_tokens.shape}")
        print(f"Positional embeddings shape: {pos_embeddings.shape}")
        
        # Test positional embeddings extraction
        print("\n🔍 Testing positional embeddings extraction...")
        
        # Check model structure
        print(f"Model type: {type(tokenizer.model)}")
        print(f"Model attributes: {dir(tokenizer.model)}")
        
        if hasattr(tokenizer.model, 'embeddings'):
            print("✅ Found embeddings attribute")
            print(f"Embeddings attributes: {dir(tokenizer.model.embeddings)}")
            
            if hasattr(tokenizer.model.embeddings, 'position_embeddings'):
                print("✅ Found position_embeddings attribute")
                pos_emb = tokenizer.model.embeddings.position_embeddings
                print(f"Position embeddings type: {type(pos_emb)}")
                print(f"Position embeddings shape: {pos_emb.shape}")
                print(f"Position embeddings requires_grad: {pos_emb.requires_grad}")
                
                # Try to access the data
                try:
                    pos_emb_data = pos_emb.data
                    print(f"Position embeddings data shape: {pos_emb_data.shape}")
                except Exception as e:
                    print(f"Error accessing .data: {e}")
                
                # Try to access the weight
                try:
                    pos_emb_weight = pos_emb.weight
                    print(f"Position embeddings weight shape: {pos_emb_weight.shape}")
                except Exception as e:
                    print(f"Error accessing .weight: {e}")
                
            else:
                print("❌ No position_embeddings attribute found")
        else:
            print("❌ No embeddings attribute found")
        
        print("\n🎉 Tokenizer test completed!")
        
    except Exception as e:
        print(f"❌ Error testing tokenizer: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_tokenizer() 