#!/usr/bin/env python3
"""
Test script for Expression Prediction Training
"""

import torch
import os
import sys
sys.path.append('.')

from models.expression_transformer_decoder import ExpressionTransformerDecoder
from training.train_expression_prediction import JointExpressionPredictionModel, ExpressionPredictionLoss

def test_transformer_decoder():
    """Test the transformer decoder model"""
    print("🧪 Testing Expression Transformer Decoder")
    
    # Create model
    decoder = ExpressionTransformerDecoder(
        embed_dim=384,
        num_heads=4,
        num_layers=1,
        dropout=0.1,
        max_sequence_length=50
    )
    
    # Test with different sequence lengths
    batch_size = 2
    
    for seq_len in [5, 10, 15]:
        # Create dummy input
        expression_tokens = torch.randn(batch_size, seq_len, 384)
        
        print(f"Input shape: {expression_tokens.shape}")
        
        # Forward pass
        predicted_token = decoder(expression_tokens)
        
        print(f"Output shape: {predicted_token.shape}")
        print(f"Expected: ({batch_size}, 1, 384)")
        
        # Verify output shape
        assert predicted_token.shape == (batch_size, 1, 384), f"Expected ({batch_size}, 1, 384), got {predicted_token.shape}"
    
    print("✅ Transformer Decoder test passed!")

def test_joint_model():
    """Test the joint expression prediction model"""
    print("\n🧪 Testing Joint Expression Prediction Model")
    
    # Create models
    joint_model = JointExpressionPredictionModel(
        expr_embed_dim=384,
        expr_num_heads=4,
        expr_num_layers=1,
        expr_dropout=0.1,
        decoder_embed_dim=384,
        decoder_num_heads=4,
        decoder_num_layers=1,
        decoder_dropout=0.1,
        max_sequence_length=50
    )
    
    # Create dummy input
    batch_size = 2
    num_frames = 8
    face_images = torch.randn(num_frames, 3, 518, 518)
    face_id_tokens = torch.randn(num_frames, 1, 384)
    
    # Mock tokenizer
    class MockTokenizer:
        def __call__(self, images):
            return torch.randn(images.shape[0], 1369, 384), torch.randn(images.shape[0], 1369, 384)
    
    tokenizer = MockTokenizer()
    
    print(f"Input face images shape: {face_images.shape}")
    print(f"Input face ID tokens shape: {face_id_tokens.shape}")
    
    # Forward pass (single clip)
    expression_tokens, predicted_next_tokens = joint_model(face_images, face_id_tokens, tokenizer)
    
    print(f"Expression tokens shape: {expression_tokens.shape}")
    print(f"Predicted next token shape: {predicted_next_tokens.shape}")
    
    # Verify shapes
    assert expression_tokens.shape == (num_frames, 1, 384), f"Expected ({num_frames}, 1, 384), got {expression_tokens.shape}"
    assert predicted_next_tokens.shape == (1, 1, 384), f"Expected (1, 1, 384), got {predicted_next_tokens.shape}"
    
    print("✅ Joint Model test passed!")

def test_loss_function():
    """Test the loss function"""
    print("\n🧪 Testing Expression Prediction Loss")
    
    # Create loss function
    criterion = ExpressionPredictionLoss()
    
    # Create dummy predictions and targets
    batch_size = 2
    seq_len = 1
    
    predicted_next_token = torch.randn(seq_len, 1, 384)
    actual_next_token = torch.randn(seq_len, 1, 384)
    
    print(f"Predicted token shape: {predicted_next_token.shape}")
    print(f"Actual token shape: {actual_next_token.shape}")
    
    # Compute loss
    loss = criterion(predicted_next_token, actual_next_token)
    
    print(f"Loss value: {loss.item():.4f}")
    
    # Verify loss is a scalar
    assert loss.dim() == 0, f"Expected scalar loss, got shape {loss.shape}"
    assert loss.item() >= 0, f"Expected non-negative loss, got {loss.item()}"
    
    print("✅ Loss Function test passed!")

def test_checkpoint_saving():
    """Test checkpoint saving and loading"""
    print("\n🧪 Testing Checkpoint Saving/Loading")
    
    # Create model
    decoder = ExpressionTransformerDecoder(
        embed_dim=384,
        num_heads=4,
        num_layers=1,
        dropout=0.1,
        max_sequence_length=50
    )
    
    # Create checkpoint directory
    checkpoint_dir = "test_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, "test_transformer_decoder.pt")
    torch.save({
        'epoch': 1,
        'transformer_decoder_state_dict': decoder.state_dict(),
        'avg_loss': 0.1,
        'config': {
            'transformer_decoder': {
                'embed_dim': 384,
                'num_heads': 4,
                'num_layers': 1,
                'dropout': 0.1,
                'max_sequence_length': 50
            }
        }
    }, checkpoint_path)
    
    print(f"✅ Saved checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create new model
    new_decoder = ExpressionTransformerDecoder(
        embed_dim=checkpoint['config']['transformer_decoder']['embed_dim'],
        num_heads=checkpoint['config']['transformer_decoder']['num_heads'],
        num_layers=checkpoint['config']['transformer_decoder']['num_layers'],
        dropout=checkpoint['config']['transformer_decoder']['dropout'],
        max_sequence_length=checkpoint['config']['transformer_decoder']['max_sequence_length']
    )
    
    # Load state dict
    new_decoder.load_state_dict(checkpoint['transformer_decoder_state_dict'])
    
    print("✅ Loaded checkpoint successfully")
    
    # Test that models are equivalent
    test_input = torch.randn(2, 10, 384)
    
    with torch.no_grad():
        output1 = decoder(test_input)
        output2 = new_decoder(test_input)
        
        # Check if outputs are close
        diff = torch.abs(output1 - output2).max().item()
        print(f"Max difference between original and loaded model: {diff:.6f}")
        
        assert diff < 1e-6, f"Models should be identical after loading, max diff: {diff}"
    
    print("✅ Checkpoint loading test passed!")
    
    # Clean up
    try:
        os.remove(checkpoint_path)
        os.rmdir(checkpoint_dir)
        print("✅ Cleaned up test files")
    except:
        print("⚠️  Could not clean up test files")

def main():
    """Run all tests"""
    print("🧪 Expression Prediction Training Tests")
    print("=" * 50)
    
    try:
        test_transformer_decoder()
        test_joint_model()
        test_loss_function()
        test_checkpoint_saving()
        
        print("\n🎉 All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 