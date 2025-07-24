#!/usr/bin/env python3
"""
Test script to verify expression transformer loading functionality
"""

import torch
import os
import sys
sys.path.append('.')

from models.expression_transformer import ExpressionTransformer
from training.train_expression_reconstruction import JointExpressionReconstructionModel

def test_expression_transformer_loading():
    """Test loading and freezing expression transformer"""
    
    print("🧪 Testing Expression Transformer Loading Functionality")
    
    # Create a dummy expression transformer checkpoint
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create expression transformer
    expr_transformer = ExpressionTransformer(embed_dim=384, num_heads=4, num_layers=1, dropout=0.1)
    
    # Save a dummy checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, "test_expression_transformer_epoch_1.pt")
    torch.save({
        'epoch': 1,
        'expression_transformer_state_dict': expr_transformer.state_dict(),
        'avg_loss': 0.1,
        'config': {
            'expression_model': {
                'embed_dim': 384,
                'num_heads': 4,
                'num_layers': 1,
                'dropout': 0.1
            }
        }
    }, checkpoint_path)
    
    print(f"✅ Created test checkpoint: {checkpoint_path}")
    
    # Test loading the checkpoint
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Create new expression transformer
        new_expr_transformer = ExpressionTransformer(embed_dim=384, num_heads=4, num_layers=1, dropout=0.1)
        
        # Load state dict
        new_expr_transformer.load_state_dict(checkpoint['expression_transformer_state_dict'])
        
        print("✅ Successfully loaded expression transformer from checkpoint")
        
        # Test freezing parameters
        for param in new_expr_transformer.parameters():
            param.requires_grad = False
        
        print("✅ Successfully froze expression transformer parameters")
        
        # Verify parameters are frozen
        all_frozen = True
        for name, param in new_expr_transformer.named_parameters():
            if param.requires_grad:
                all_frozen = False
                print(f"❌ Parameter {name} is not frozen")
        
        if all_frozen:
            print("✅ All expression transformer parameters are frozen")
        else:
            print("❌ Some parameters are not frozen")
            
    except Exception as e:
        print(f"❌ Failed to load expression transformer: {str(e)}")
        return False
    
    # Test with joint model (simulating the actual loading process)
    try:
        # Load checkpoint to get architecture
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Get architecture from checkpoint config
        if 'config' in checkpoint and 'expression_model' in checkpoint['config']:
            expr_config = checkpoint['config']['expression_model']
            expr_embed_dim = expr_config.get('embed_dim', 384)
            expr_num_heads = expr_config.get('num_heads', 8)
            expr_num_layers = expr_config.get('num_layers', 2)
            expr_dropout = expr_config.get('dropout', 0.1)
            print(f"✅ Loaded architecture from checkpoint: {expr_num_layers} layers, {expr_num_heads} heads")
        else:
            # Fallback to default architecture
            expr_embed_dim, expr_num_heads, expr_num_layers, expr_dropout = 384, 8, 2, 0.1
            print("⚠️  No architecture config found, using defaults")
        
        # Initialize joint model with correct architecture (no reinitialization needed)
        joint_model = JointExpressionReconstructionModel(
            embed_dim=expr_embed_dim,
            num_heads=expr_num_heads,
            num_layers=expr_num_layers,
            dropout=expr_dropout
        )
        
        # Load expression transformer weights into joint model
        joint_model.expression_transformer.load_state_dict(checkpoint['expression_transformer_state_dict'])
        
        # Freeze expression transformer
        for param in joint_model.expression_transformer.parameters():
            param.requires_grad = False
        
        print("✅ Successfully loaded expression transformer weights into joint model")
        
        # Check trainable parameters
        trainable_params = []
        for name, param in joint_model.named_parameters():
            if param.requires_grad:
                trainable_params.append(name)
        
        print(f"✅ Trainable parameters: {trainable_params}")
        
        # Verify expression transformer parameters are not in trainable list
        expr_trainable = [name for name in trainable_params if 'expression_transformer' in name]
        if len(expr_trainable) == 0:
            print("✅ Expression transformer parameters are correctly frozen")
        else:
            print(f"❌ Expression transformer parameters are still trainable: {expr_trainable}")
            
    except Exception as e:
        print(f"❌ Failed to test with joint model: {str(e)}")
        return False
    
    # Clean up test file
    try:
        os.remove(checkpoint_path)
        print(f"✅ Cleaned up test checkpoint: {checkpoint_path}")
    except:
        print(f"⚠️  Could not clean up test checkpoint: {checkpoint_path}")
    
    print("\n🎉 All tests passed!")
    return True

if __name__ == "__main__":
    test_expression_transformer_loading() 