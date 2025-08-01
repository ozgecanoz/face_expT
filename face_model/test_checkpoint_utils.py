#!/usr/bin/env python3
"""
Simple test for checkpoint utilities
"""

import torch
import tempfile
import os
from utils.checkpoint_utils import (
    save_checkpoint,
    load_checkpoint_config,
    extract_model_config,
    create_comprehensive_config,
    validate_checkpoint_compatibility
)


def test_checkpoint_utils():
    """Test the checkpoint utility functions"""
    print("🧪 Testing checkpoint utility functions...")
    
    # Test parameters
    test_params = {
        'expr_embed_dim': 256,
        'expr_num_heads': 8,
        'expr_num_layers': 3,
        'expr_dropout': 0.2,
        'expr_max_subjects': 5000,
        'decoder_embed_dim': 512,
        'decoder_num_heads': 16,
        'decoder_num_layers': 4,
        'decoder_dropout': 0.3,
        'max_sequence_length': 100,
        'lambda_prediction': 1.0,
        'lambda_temporal': 0.5,
        'lambda_diversity': 0.3,
        'learning_rate': 2e-4,
        'batch_size': 16,
        'num_epochs': 5,
        'warmup_steps': 500,
        'min_lr': 1e-5
    }
    
    # Create comprehensive config
    config = create_comprehensive_config(
        expr_embed_dim=test_params['expr_embed_dim'],
        expr_num_heads=test_params['expr_num_heads'],
        expr_num_layers=test_params['expr_num_layers'],
        expr_dropout=test_params['expr_dropout'],
        expr_max_subjects=test_params['expr_max_subjects'],
        decoder_embed_dim=test_params['decoder_embed_dim'],
        decoder_num_heads=test_params['decoder_num_heads'],
        decoder_num_layers=test_params['decoder_num_layers'],
        decoder_dropout=test_params['decoder_dropout'],
        max_sequence_length=test_params['max_sequence_length'],
        lambda_prediction=test_params['lambda_prediction'],
        lambda_temporal=test_params['lambda_temporal'],
        lambda_diversity=test_params['lambda_diversity'],
        learning_rate=test_params['learning_rate'],
        batch_size=test_params['batch_size'],
        num_epochs=test_params['num_epochs'],
        warmup_steps=test_params['warmup_steps'],
        min_lr=test_params['min_lr']
    )
    
    print("✅ Created comprehensive config")
    
    # Test saving and loading
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = os.path.join(temp_dir, "test_checkpoint.pt")
        
        # Save checkpoint
        save_checkpoint(
            model_state_dict={'test': torch.randn(10)},
            optimizer_state_dict={},
            scheduler_state_dict={},
            epoch=10,
            avg_loss=0.5,
            total_steps=1000,
            config=config,
            checkpoint_path=checkpoint_path,
            checkpoint_type="joint"
        )
        
        print("✅ Saved checkpoint")
        
        # Load checkpoint
        checkpoint_data, loaded_config = load_checkpoint_config(checkpoint_path, 'cpu')
        
        print("✅ Loaded checkpoint")
        
        # Extract parameters
        default_params = {
            'expr_embed_dim': 384,
            'expr_num_heads': 4,
            'expr_num_layers': 2,
            'expr_dropout': 0.1,
            'expr_max_subjects': 3500,
            'decoder_embed_dim': 384,
            'decoder_num_heads': 4,
            'decoder_num_layers': 2,
            'decoder_dropout': 0.1,
            'max_sequence_length': 50,
            'lambda_prediction': 0.5,
            'lambda_temporal': 0.1,
            'lambda_diversity': 0.1
        }
        
        extracted_params = extract_model_config(loaded_config, default_params)
        
        print("✅ Extracted model config")
        
        # Verify parameters match
        for param_name, expected_value in test_params.items():
            if param_name in extracted_params:
                actual_value = extracted_params[param_name]
                assert actual_value == expected_value, f"Parameter {param_name}: expected {expected_value}, got {actual_value}"
                print(f"✅ {param_name}: {actual_value}")
        
        # Test validation
        expected_params = {
            'expr_embed_dim': test_params['expr_embed_dim'],
            'expr_num_heads': test_params['expr_num_heads'],
            'expr_num_layers': test_params['expr_num_layers'],
            'expr_dropout': test_params['expr_dropout'],
            'expr_max_subjects': test_params['expr_max_subjects'],
            'decoder_embed_dim': test_params['decoder_embed_dim'],
            'decoder_num_heads': test_params['decoder_num_heads'],
            'decoder_num_layers': test_params['decoder_num_layers'],
            'decoder_dropout': test_params['decoder_dropout'],
            'max_sequence_length': test_params['max_sequence_length'],
            'lambda_prediction': test_params['lambda_prediction'],
            'lambda_temporal': test_params['lambda_temporal'],
            'lambda_diversity': test_params['lambda_diversity']
        }
        
        is_compatible = validate_checkpoint_compatibility(checkpoint_path, expected_params, 'cpu')
        assert is_compatible, "Checkpoint compatibility validation failed"
        
        print("✅ Checkpoint compatibility validation passed")
    
    print("🎉 All checkpoint utility tests passed!")


if __name__ == "__main__":
    test_checkpoint_utils() 