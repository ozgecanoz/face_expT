#!/usr/bin/env python3
"""
Run Face ID Model training
"""

import torch
import os
import sys
sys.path.append('.')

from training.train_face_id import FaceIDTrainer
from data.dataset import create_face_dataloader

def main():
    """Run Face ID Model training"""
    
    # Device detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Memory optimization based on device
    if device.type == "cpu":
        torch.set_num_threads(4)  # Limit CPU threads
        torch.backends.cudnn.benchmark = False  # Disable for CPU
    else:
        # GPU optimizations
        torch.backends.cudnn.benchmark = True  # Enable for GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear GPU cache
    
    # Configuration - Optimized for CPU training
    config = {
        'training': {
            'log_dir': "/Users/ozgewhiting/Documents/projects/dataset_utils/face_model/logs",
            'checkpoint_dir': "/Users/ozgewhiting/Documents/projects/dataset_utils/face_model/checkpoints",
            'checkpoint_path': "/Users/ozgewhiting/Documents/projects/dataset_utils/face_model/checkpoints/face_id_epoch_1.pth",  # Set to path if you want to load from existing checkpoint
            'learning_rate': 1e-4,
            'batch_size': 4,  # Increased to ensure multiple subjects per batch for contrastive loss
            'num_epochs': 2,
            'save_every': 1,   # Save checkpoint every 2 epochs
            'contrastive_temperature': 0.5,  # Temperature for NT-Xent loss
            'consistency_weight': 1.0,
            'contrastive_weight': 1.0,  # Default weight for contrastive loss
            # Validation configuration
            'train_data_dir': "/Users/ozgewhiting/Documents/EQLabs/datasets_serial/CCA_train_db1",
            'val_data_dir': "/Users/ozgewhiting/Documents/EQLabs/datasets_serial/CCA_val_db1",  # Set to validation dataset path if available
            'max_train_samples': 20,  # if None, Use all samples for full training
            'max_val_samples': 20   # if None, Use all validation samples
        },
        'face_id_model': {
            'embed_dim': 384,
            'num_heads': 4,  # Updated to optimized architecture
            'num_layers': 1,  # Updated to optimized architecture
            'dropout': 0.1
        }
    }
    
    print("🚀 Starting Face ID Model Training")
    print(f"🖥️  Device: {device}")
    print(f"📊 Dataset: {config['training']['train_data_dir']}")
    print(f"📈 Logs: {config['training']['log_dir']}")
    print(f"🎯 Learning rate: {config['training']['learning_rate']}")
    print(f"📦 Batch size: {config['training']['batch_size']}")
    print(f"🔄 Epochs: {config['training']['num_epochs']}")
    
    # Log checkpoint status
    if config['training']['checkpoint_path'] is not None:
        print(f"🔒 Face ID Model: Will load from {config['training']['checkpoint_path']}")
        # Try to load and display architecture info
        try:
            checkpoint = torch.load(config['training']['checkpoint_path'], map_location='cpu')
            if 'config' in checkpoint and 'face_id_model' in checkpoint['config']:
                face_id_config = checkpoint['config']['face_id_model']
                print(f"   📐 Architecture: {face_id_config.get('num_layers', '?')} layers, {face_id_config.get('num_heads', '?')} heads")
            else:
                print(f"   ⚠️  No architecture info in checkpoint")
        except Exception as e:
            print(f"   ⚠️  Could not read checkpoint info: {str(e)}")
    else:
        print(f"🎓 Face ID Model: Will train from scratch")
    
    # Create trainer
    trainer = FaceIDTrainer(config)
    
    # Create training dataloader
    train_dataloader = create_face_dataloader(
        data_dir=config['training']['train_data_dir'],
        batch_size=config['training']['batch_size'],
        max_samples=config['training']['max_train_samples']
    )
    
    # Create validation dataloader if validation data is provided
    val_dataloader = None
    if config['training']['val_data_dir'] is not None:
        val_dataloader = create_face_dataloader(
            data_dir=config['training']['val_data_dir'],
            batch_size=config['training']['batch_size'],
            max_samples=config['training']['max_val_samples']
        )
        print(f"📊 Validation dataset loaded: {len(val_dataloader)} batches per epoch")
    
    print(f"\n📋 Training dataset loaded: {len(train_dataloader)} batches per epoch")
    
    # Start training
    print("\n🎯 Starting training...")
    trainer.train(train_dataloader, val_dataloader, config['training']['num_epochs'])
    
    print("\n✅ Training completed!")
    print(f"📊 TensorBoard logs will be saved with unique job ID (face_id_training_<id>)")
    print(f"📊 Check TensorBoard logs at: {config['training']['log_dir']}/face_id_training_*")

if __name__ == "__main__":
    main() 