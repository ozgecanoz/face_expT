#!/usr/bin/env python3
"""
Run Face ID Model training - Optimized for n2-standard-16 VM
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
        torch.set_num_threads(16)  # Use all 16 CPU cores
        torch.backends.cudnn.benchmark = False  # Disable for CPU
    else:
        # GPU optimizations
        torch.backends.cudnn.benchmark = True  # Enable for GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear GPU cache
    
    # Configuration - Optimized for n2-standard-16 VM with CCA_train_db1
    config = {
        'training': {
            'log_dir': "/mnt/dataset-storage/face_model/logs",
            'checkpoint_dir': "/mnt/dataset-storage/face_model/checkpoints",
            'checkpoint_path': None,  # Start from scratch
            'learning_rate': 1e-4,
            'batch_size': 16,  # Optimized for 64GB RAM
            'num_epochs': 5,  # 20 : More epochs for large dataset
            'save_every': 1,   # 3, Save checkpoint every 3 epochs
            'contrastive_temperature': 0.5,  # Temperature for NT-Xent loss
            'consistency_weight': 1.0,
            'contrastive_weight': 1.0,  # Default weight for contrastive loss
            'num_workers': 0,  # Single-threaded data loading for stability
            'pin_memory': False,  # Not needed for CPU
            'persistent_workers': False,  # Not applicable with num_workers=0
            'drop_last': True,  # Consistent batch sizes
            # Dataset configuration
            'train_data_dir': "/mnt/dataset-storage/dbs/CCA_train_db1",
            'val_data_dir': "/mnt/dataset-storage/dbs/CCA_val_db1/CCA_val_db1/",  # validation set for now
            'max_train_samples': None,  # Use all available samples
            'max_val_samples': 100    # set to None to use all available samples
        },
        'face_id_model': {
            'embed_dim': 384,
            'num_heads': 4,  # Optimized for CPU
            'num_layers': 2,  # Can handle more layers with 64GB RAM
            'dropout': 0.1
        }
    }
    
    print("🚀 Starting Face ID Model Training on n2-standard-16 VM")
    print(f"🖥️  Device: {device}")
    print(f"📊 Dataset: {config['training']['train_data_dir']}")
    print(f"📈 Logs: {config['training']['log_dir']}")
    print(f"🎯 Learning rate: {config['training']['learning_rate']}")
    print(f"📦 Batch size: {config['training']['batch_size']}")
    print(f"🔄 Epochs: {config['training']['num_epochs']}")
    print(f"🧵 Num workers: {config['training']['num_workers']}")
    print(f"💾 Memory optimization: CPU threads set to 16")
    
    # Create directories if they don't exist
    os.makedirs(config['training']['log_dir'], exist_ok=True)
    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
    
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
        max_samples=config['training']['max_train_samples'],
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory'],
        persistent_workers=config['training']['persistent_workers'],
        drop_last=config['training']['drop_last']
    )
    
    # Create validation dataloader if validation data is provided
    val_dataloader = None
    if config['training']['val_data_dir'] is not None:
        val_dataloader = create_face_dataloader(
            data_dir=config['training']['val_data_dir'],
            batch_size=config['training']['batch_size'],
            max_samples=config['training']['max_val_samples'],
            num_workers=config['training']['num_workers'],
            pin_memory=config['training']['pin_memory'],
            persistent_workers=config['training']['persistent_workers'],
            drop_last=config['training']['drop_last']
        )
        print(f"📊 Validation dataset loaded: {len(val_dataloader)} batches per epoch")
    
    print(f"\n📋 Training dataset loaded: {len(train_dataloader)} batches per epoch")
    
    # Estimate training time (more realistic for CPU training)
    estimated_time_per_epoch = len(train_dataloader) * 120  # More realistic: 30 seconds per batch
    total_estimated_time = estimated_time_per_epoch * config['training']['num_epochs'] / 3600  # Convert to hours
    
    print(f"⏱️  Estimated training time: {total_estimated_time:.1f} hours ({total_estimated_time/24:.1f} days)")
    print(f"💰 Estimated cost: ${total_estimated_time * 0.38:.1f} (at $0.38/hour)")
    
    # Start training
    print("\n🎯 Starting training...")
    trainer.train(train_dataloader, val_dataloader, config['training']['num_epochs'])
    
    print("\n✅ Training completed!")
    print(f"📊 TensorBoard logs will be saved with unique job ID (face_id_training_<id>)")
    print(f"📊 Check TensorBoard logs at: {config['training']['log_dir']}/face_id_training_*")
    print(f"💾 Checkpoints saved at: {config['training']['checkpoint_dir']}")

if __name__ == "__main__":
    main() 