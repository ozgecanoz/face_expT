#!/usr/bin/env python3
"""
Run Expression Reconstruction Model training - Optimized for n2-standard-16 VM
"""

import torch
import os
import sys
sys.path.append('.')

from training.train_expression_reconstruction import train_expression_reconstruction

def main():
    """Run Expression Reconstruction Model training"""
    
    # Device detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Memory optimization based on device
    if device.type == "cpu":
        torch.set_num_threads(16)  # Use all 16 CPU cores
        torch.backends.cudnn.benchmark = False  # Disable for CPU
    else:
        # GPU optimizations
        torch.backends.cudnn.benchmark = True  # Enable for GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear GPU cache
    
    # Configuration - Optimized for n2-standard-16 VM with CCA_train_db2
    config = {
        'training': {
            'train_data_dir': "/mnt/dataset-storage/dbs/CCA_train_db2/",
            'val_data_dir': "/mnt/dataset-storage/dbs/CCA_val_db2/",
            'face_id_checkpoint_path': "/mnt/dataset-storage/face_model/checkpoints/face_id_epoch_0.pth",
            'expression_transformer_checkpoint_path': "/mnt/dataset-storage/face_model/checkpoints/expression_transformer_epoch_5.pt",  # Required
            'reconstruction_model_checkpoint_path': None,  # Optional - will train from scratch if not provided
            'log_dir': "/mnt/dataset-storage/face_model/logs",
            'checkpoint_dir': "/mnt/dataset-storage/face_model/checkpoints",
            'learning_rate': 1e-4,
            'batch_size': 8,  # Reduced from 16 to 8 for memory optimization
            'num_epochs': 5,
            'save_every_epochs': 1,   # Save checkpoint every epoch
            'reconstruction_weight': 1.0,
            'identity_weight': 1.0,
            'max_train_samples': None,  # Use all available samples
            'max_val_samples': 100,   # Limit validation samples for testing
            'num_workers': 2,  # Reduced from 4 to 2 for memory optimization
            'pin_memory': False,  # Not needed for CPU
            'persistent_workers': True,  # Keep workers alive for efficiency
            'drop_last': True  # Consistent batch sizes
        },
        'reconstruction_model': {
            'embed_dim': 384,
            'num_heads': 4,  # Optimized for CPU
            'num_layers': 2,  # Can handle more layers with 64GB RAM
            'dropout': 0.1
        }
    }
    
    print("🚀 Starting Expression Reconstruction Model Training on n2-standard-16 VM")
    print(f"🖥️  Device: {device}")
    print(f"📊 Training Dataset: {config['training']['train_data_dir']}")
    print(f"📊 Validation Dataset: {config['training']['val_data_dir']}")
    print(f"📈 Logs: {config['training']['log_dir']}")
    print(f"🎯 Learning rate: {config['training']['learning_rate']}")
    print(f"📦 Batch size: {config['training']['batch_size']}")
    print(f"🔄 Epochs: {config['training']['num_epochs']}")
    print(f"🧠 Reconstruction Model: {config['reconstruction_model']['num_layers']} layers, {config['reconstruction_model']['num_heads']} heads")
    print(f"🧵 Num workers: {config['training']['num_workers']}")
    print(f"💾 Memory optimization: drop_last={config['training']['drop_last']}")
    
    # Create directories if they don't exist
    os.makedirs(config['training']['log_dir'], exist_ok=True)
    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
    
    # Log checkpoint status
    print(f"🔒 Face ID Model: Will load from {config['training']['face_id_checkpoint_path']}")
    print(f"🔒 Expression Transformer: Will load from {config['training']['expression_transformer_checkpoint_path']}")
    
    if config['training']['reconstruction_model_checkpoint_path'] is not None:
        print(f"🔒 Reconstruction Model: Will load from {config['training']['reconstruction_model_checkpoint_path']}")
    else:
        print(f"🎓 Reconstruction Model: Will train from scratch")
    
    # Try to load and display architecture info for checkpoints
    try:
        # Face ID checkpoint info
        face_id_checkpoint = torch.load(config['training']['face_id_checkpoint_path'], map_location='cpu')
        if 'config' in face_id_checkpoint and 'face_id_model' in face_id_checkpoint['config']:
            face_id_config = face_id_checkpoint['config']['face_id_model']
            print(f"   📐 Face ID Architecture: {face_id_config.get('num_layers', '?')} layers, {face_id_config.get('num_heads', '?')} heads")
        else:
            print(f"   ⚠️  No Face ID architecture info in checkpoint")
    except Exception as e:
        print(f"   ⚠️  Could not read Face ID checkpoint info: {str(e)}")
    
    try:
        # Expression transformer checkpoint info
        expr_checkpoint = torch.load(config['training']['expression_transformer_checkpoint_path'], map_location='cpu')
        if 'config' in expr_checkpoint and 'expression_model' in expr_checkpoint['config']:
            expr_config = expr_checkpoint['config']['expression_model']
            print(f"   📐 Expression Transformer Architecture: {expr_config.get('num_layers', '?')} layers, {expr_config.get('num_heads', '?')} heads")
        else:
            print(f"   ⚠️  No Expression Transformer architecture info in checkpoint")
    except Exception as e:
        print(f"   ⚠️  Could not read Expression Transformer checkpoint info: {str(e)}")
    
    # Start training
    print("\n🎯 Starting training...")
    train_expression_reconstruction(
        dataset_path=config['training']['train_data_dir'],
        face_id_checkpoint_path=config['training']['face_id_checkpoint_path'],
        expression_transformer_checkpoint_path=config['training']['expression_transformer_checkpoint_path'],
        reconstruction_model_checkpoint_path=config['training']['reconstruction_model_checkpoint_path'],
        checkpoint_dir=config['training']['checkpoint_dir'],
        save_every_epochs=config['training']['save_every_epochs'],
        batch_size=config['training']['batch_size'],
        num_epochs=config['training']['num_epochs'],
        learning_rate=config['training']['learning_rate'],
        reconstruction_weight=config['training']['reconstruction_weight'],
        identity_weight=config['training']['identity_weight'],
        max_samples=config['training']['max_train_samples'],
        val_dataset_path=config['training']['val_data_dir'],
        max_val_samples=config['training']['max_val_samples'],
        device=device
    )
    
    print("\n✅ Training completed!")
    print(f"📊 TensorBoard logs will be saved with unique job ID (expression_reconstruction_training_<id>)")
    print(f"📊 Check TensorBoard logs at: {config['training']['log_dir']}/expression_reconstruction_training_*")
    print(f"💾 Checkpoints saved at: {config['training']['checkpoint_dir']}")

if __name__ == "__main__":
    main() 