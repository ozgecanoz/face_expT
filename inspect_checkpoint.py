#!/usr/bin/env python3
"""
Inspect expression transformer checkpoint to see actual architecture
"""

import torch
import sys
import os

def inspect_checkpoint(checkpoint_path):
    """Inspect checkpoint contents"""
    print(f"🔍 Inspecting checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"✅ Successfully loaded checkpoint")
        
        print(f"\n📋 Checkpoint keys:")
        for key in checkpoint.keys():
            print(f"  - {key}")
        
        if 'config' in checkpoint:
            print(f"\n⚙️  Config contents:")
            config = checkpoint['config']
            for key, value in config.items():
                print(f"  - {key}: {value}")
        
        if 'expression_transformer_state_dict' in checkpoint:
            state_dict = checkpoint['expression_transformer_state_dict']
            print(f"\n🔧 Expression Transformer State Dict keys:")
            for key in state_dict.keys():
                print(f"  - {key}")
        
        print(f"\n📊 Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"📊 Average Loss: {checkpoint.get('avg_loss', 'unknown')}")
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_checkpoint.py <checkpoint_path>")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    inspect_checkpoint(checkpoint_path) 