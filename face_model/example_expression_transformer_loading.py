#!/usr/bin/env python3
"""
Example: How to use expression transformer loading functionality
"""

import torch
import os
import sys
sys.path.append('.')

from training.train_expression_reconstruction import train_expression_reconstruction

def example_with_expression_transformer_checkpoint():
    """Example of training with a pre-trained expression transformer"""
    
    print("📚 Example: Training with Pre-trained Expression Transformer")
    print("=" * 60)
    
    # Configuration
    config = {
        'train_data_dir': "/Users/ozgewhiting/Documents/EQLabs/datasets_serial/CCA_train_db1",
        'val_data_dir': "/Users/ozgewhiting/Documents/EQLabs/datasets_serial/CCA_val_db1",
        'face_id_checkpoint_path': "/Users/ozgewhiting/Documents/projects/dataset_utils/face_model/checkpoints/face_id_epoch_1.pth",
        'expression_transformer_checkpoint_path': "/Users/ozgewhiting/Documents/projects/dataset_utils/face_model/checkpoints/expression_transformer_epoch_1.pt",  # Set this to load and freeze
        'checkpoint_dir': "/Users/ozgewhiting/Documents/projects/dataset_utils/face_model/checkpoints",
        'learning_rate': 1e-4,
        'batch_size': 2,
        'num_epochs': 2,
        'save_every_epochs': 1,
        'reconstruction_weight': 1.0,
        'identity_weight': 1.0,
        'max_train_samples': 10,  # Small sample for testing
        'max_val_samples': 5,
        'device': "cpu"
    }
    
    print("🔧 Configuration:")
    print(f"   📊 Training data: {config['train_data_dir']}")
    print(f"   📊 Validation data: {config['val_data_dir']}")
    print(f"   🧠 Face ID checkpoint: {config['face_id_checkpoint_path']}")
    print(f"   🎭 Expression transformer checkpoint: {config['expression_transformer_checkpoint_path']}")
    print(f"   💾 Checkpoint directory: {config['checkpoint_dir']}")
    print(f"   🎯 Learning rate: {config['learning_rate']}")
    print(f"   📦 Batch size: {config['batch_size']}")
    print(f"   🔄 Epochs: {config['num_epochs']}")
    
    # Check if expression transformer checkpoint exists
    if os.path.exists(config['expression_transformer_checkpoint_path']):
        print(f"\n✅ Expression transformer checkpoint found!")
        
        # Try to load and display architecture info
        try:
            checkpoint = torch.load(config['expression_transformer_checkpoint_path'], map_location='cpu')
            if 'config' in checkpoint and 'expression_model' in checkpoint['config']:
                expr_config = checkpoint['config']['expression_model']
                print(f"   📐 Architecture: {expr_config.get('num_layers', '?')} layers, {expr_config.get('num_heads', '?')} heads")
                print(f"   📏 Embed dim: {expr_config.get('embed_dim', '?')}")
                print(f"   💧 Dropout: {expr_config.get('dropout', '?')}")
            else:
                print(f"   ⚠️  No architecture info in checkpoint")
        except Exception as e:
            print(f"   ⚠️  Could not read checkpoint info: {str(e)}")
        
        print(f"\n🔒 The expression transformer will be loaded and frozen during training.")
        print(f"🎓 Only the reconstruction model will be trained.")
        
    else:
        print(f"\n⚠️  Expression transformer checkpoint not found: {config['expression_transformer_checkpoint_path']}")
        print(f"🎓 The expression transformer will be trained from scratch.")
    
    print("\n" + "=" * 60)
    print("🚀 Starting training...")
    
    # Start training
    train_expression_reconstruction(
        dataset_path=config['train_data_dir'],
        face_id_checkpoint_path=config['face_id_checkpoint_path'],
        expression_transformer_checkpoint_path=config['expression_transformer_checkpoint_path'],
        checkpoint_dir=config['checkpoint_dir'],
        save_every_epochs=config['save_every_epochs'],
        batch_size=config['batch_size'],
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        reconstruction_weight=config['reconstruction_weight'],
        identity_weight=config['identity_weight'],
        max_samples=config['max_train_samples'],
        val_dataset_path=config['val_data_dir'],
        max_val_samples=config['max_val_samples'],
        device=config['device']
    )
    
    print("\n✅ Training completed!")

def example_without_expression_transformer_checkpoint():
    """Example of training without a pre-trained expression transformer"""
    
    print("\n📚 Example: Training without Pre-trained Expression Transformer")
    print("=" * 60)
    
    # Configuration
    config = {
        'train_data_dir': "/Users/ozgewhiting/Documents/EQLabs/datasets_serial/CCA_train_db1",
        'val_data_dir': "/Users/ozgewhiting/Documents/EQLabs/datasets_serial/CCA_val_db1",
        'face_id_checkpoint_path': "/Users/ozgewhiting/Documents/projects/dataset_utils/face_model/checkpoints/face_id_epoch_1.pth",
        'expression_transformer_checkpoint_path': None,  # No pre-trained expression transformer
        'checkpoint_dir': "/Users/ozgewhiting/Documents/projects/dataset_utils/face_model/checkpoints",
        'learning_rate': 1e-4,
        'batch_size': 2,
        'num_epochs': 2,
        'save_every_epochs': 1,
        'reconstruction_weight': 1.0,
        'identity_weight': 1.0,
        'max_train_samples': 10,  # Small sample for testing
        'max_val_samples': 5,
        'device': "cpu"
    }
    
    print("🔧 Configuration:")
    print(f"   📊 Training data: {config['train_data_dir']}")
    print(f"   📊 Validation data: {config['val_data_dir']}")
    print(f"   🧠 Face ID checkpoint: {config['face_id_checkpoint_path']}")
    print(f"   🎭 Expression transformer checkpoint: None (train from scratch)")
    print(f"   💾 Checkpoint directory: {config['checkpoint_dir']}")
    print(f"   🎯 Learning rate: {config['learning_rate']}")
    print(f"   📦 Batch size: {config['batch_size']}")
    print(f"   🔄 Epochs: {config['num_epochs']}")
    
    print(f"\n🎓 Both expression transformer and reconstruction model will be trained from scratch.")
    
    print("\n" + "=" * 60)
    print("🚀 Starting training...")
    
    # Start training
    train_expression_reconstruction(
        dataset_path=config['train_data_dir'],
        face_id_checkpoint_path=config['face_id_checkpoint_path'],
        expression_transformer_checkpoint_path=config['expression_transformer_checkpoint_path'],
        checkpoint_dir=config['checkpoint_dir'],
        save_every_epochs=config['save_every_epochs'],
        batch_size=config['batch_size'],
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        reconstruction_weight=config['reconstruction_weight'],
        identity_weight=config['identity_weight'],
        max_samples=config['max_train_samples'],
        val_dataset_path=config['val_data_dir'],
        max_val_samples=config['max_val_samples'],
        device=config['device']
    )
    
    print("\n✅ Training completed!")

if __name__ == "__main__":
    print("🎭 Expression Transformer Loading Examples")
    print("=" * 60)
    
    # Example 1: With pre-trained expression transformer
    example_with_expression_transformer_checkpoint()
    
    # Example 2: Without pre-trained expression transformer
    example_without_expression_transformer_checkpoint() 