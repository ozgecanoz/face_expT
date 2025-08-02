#!/usr/bin/env python3
"""
Test script for expression reconstruction training
Runs a small training session with limited data
"""

import torch
import os
import sys
sys.path.append('.')

from training.train_expression_and_reconstruction import train_expression_and_reconstruction

def main():
    """Test the expression reconstruction training with limited data"""
    
    # Device detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    # Test configuration with limited data
    test_config = {
        'dataset_path': "/Users/ozgewhiting/Documents/EQLabs/datasets_serial/CCA_train_db3_expr/",
        'max_samples': 100,  # Limit to 100 samples for testing
        'val_dataset_path': "/Users/ozgewhiting/Documents/EQLabs/datasets_serial/CCA_train_db3_expr/",
        'max_val_samples': 50,  # Limit validation samples
        'checkpoint_dir': "./test_checkpoints_expression_reconstruction",
        'log_dir': "./test_logs_expression_reconstruction",
        'batch_size': 2,  # Small batch size for testing
        'num_epochs': 1,  # Just 1 epoch for testing
        'save_every_step': 50,  # Save more frequently for testing
        'learning_rate': 1e-4,
        'warmup_steps': 50,  # Short warmup for testing
        'min_lr': 1e-6,
        'num_workers': 0,  # No multiprocessing for testing
        'pin_memory': False,
        'persistent_workers': False,
        'drop_last': True,
        # Loss weight scheduling parameters
        'initial_lambda_reconstruction': 0.1,
        'initial_lambda_temporal': 0.5,
        'initial_lambda_diversity': 0.5,
        'warmup_lambda_reconstruction': 0.3,
        'warmup_lambda_temporal': 0.4,
        'warmup_lambda_diversity': 0.4,
        'final_lambda_reconstruction': 0.5,
        'final_lambda_temporal': 0.3,
        'final_lambda_diversity': 0.3,
        # Architecture configuration (smaller for testing)
        'expr_embed_dim': 384,
        'expr_num_heads': 4,
        'expr_num_layers': 2,
        'expr_dropout': 0.1,
        'expr_max_subjects': 100,  # Smaller for testing
        'recon_embed_dim': 384,
        'recon_num_cross_layers': 1,  # Smaller for testing
        'recon_num_self_layers': 1,  # Smaller for testing
        'recon_num_heads': 4,  # Smaller for testing
        'recon_ff_dim': 768,  # Smaller for testing
        'recon_dropout': 0.1
    }
    
    print("🧪 Testing Expression Reconstruction Training")
    print(f"📊 Dataset: {test_config['dataset_path']}")
    print(f"📦 Batch size: {test_config['batch_size']}")
    print(f"🔄 Epochs: {test_config['num_epochs']}")
    print(f"📈 Max samples: {test_config['max_samples']}")
    print(f"💾 Save every: {test_config['save_every_step']} steps")
    
    try:
        # Start training
        joint_model = train_expression_and_reconstruction(
            dataset_path=test_config['dataset_path'],
            max_samples=test_config['max_samples'],
            val_dataset_path=test_config['val_dataset_path'],
            max_val_samples=test_config['max_val_samples'],
            checkpoint_dir=test_config['checkpoint_dir'],
            log_dir=test_config['log_dir'],
            save_every_step=test_config['save_every_step'],
            batch_size=test_config['batch_size'],
            num_epochs=test_config['num_epochs'],
            learning_rate=test_config['learning_rate'],
            warmup_steps=test_config['warmup_steps'],
            min_lr=test_config['min_lr'],
            device=device,
            num_workers=test_config['num_workers'],
            pin_memory=test_config['pin_memory'],
            persistent_workers=test_config['persistent_workers'],
            drop_last=test_config['drop_last'],
            # Loss weight scheduling parameters
            initial_lambda_reconstruction=test_config['initial_lambda_reconstruction'],
            initial_lambda_temporal=test_config['initial_lambda_temporal'],
            initial_lambda_diversity=test_config['initial_lambda_diversity'],
            warmup_lambda_reconstruction=test_config['warmup_lambda_reconstruction'],
            warmup_lambda_temporal=test_config['warmup_lambda_temporal'],
            warmup_lambda_diversity=test_config['warmup_lambda_diversity'],
            final_lambda_reconstruction=test_config['final_lambda_reconstruction'],
            final_lambda_temporal=test_config['final_lambda_temporal'],
            final_lambda_diversity=test_config['final_lambda_diversity'],
            # Architecture configuration
            expr_embed_dim=test_config['expr_embed_dim'],
            expr_num_heads=test_config['expr_num_heads'],
            expr_num_layers=test_config['expr_num_layers'],
            expr_dropout=test_config['expr_dropout'],
            expr_max_subjects=test_config['expr_max_subjects'],
            recon_embed_dim=test_config['recon_embed_dim'],
            recon_num_cross_layers=test_config['recon_num_cross_layers'],
            recon_num_self_layers=test_config['recon_num_self_layers'],
            recon_num_heads=test_config['recon_num_heads'],
            recon_ff_dim=test_config['recon_ff_dim'],
            recon_dropout=test_config['recon_dropout']
        )
        
        print("✅ Training test completed successfully!")
        print(f"📊 Checkpoints saved to: {test_config['checkpoint_dir']}")
        print(f"📈 Logs saved to: {test_config['log_dir']}")
        
        # Test model inference
        print("\n🧪 Testing model inference...")
        joint_model.eval()
        
        # Create dummy input for testing
        dummy_images = torch.randn(2, 3, 518, 518).to(device)
        dummy_subject_ids = torch.randint(0, 100, (2,)).to(device)
        
        # Create dummy tokenizer
        class DummyTokenizer:
            def __call__(self, images):
                return torch.randn(images.shape[0], 1369, 384).to(images.device), torch.randn(images.shape[0], 1369, 384).to(images.device)
        
        tokenizer = DummyTokenizer()
        
        with torch.no_grad():
            expression_tokens, reconstructed_images = joint_model(dummy_images, dummy_subject_ids, tokenizer)
            
            print(f"✅ Inference test passed!")
            print(f"   Expression tokens shape: {expression_tokens.shape}")
            print(f"   Reconstructed images shape: {reconstructed_images.shape}")
        
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 All tests passed! The expression reconstruction training is working correctly.")
    else:
        print("\n💥 Tests failed. Please check the error messages above.") 