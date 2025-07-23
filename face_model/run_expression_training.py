#!/usr/bin/env python3
"""
Run Expression Reconstruction Model training
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
            'train_data_dir': "/Users/ozgewhiting/Documents/EQLabs/datasets_serial/CCA_train_db1",
            'val_data_dir': "/Users/ozgewhiting/Documents/EQLabs/datasets_serial/CCA_val_db1",
            'face_id_checkpoint_path': "/Users/ozgewhiting/Documents/projects/dataset_utils/face_model/checkpoints/face_id_epoch_1.pth",
            'log_dir': "./logs",
            'checkpoint_dir': "/Users/ozgewhiting/Documents/projects/dataset_utils/face_model/checkpoints",
            'learning_rate': 1e-4,
            'batch_size': 2,  # Reduced for CPU memory
            'num_epochs': 2,
            'save_every_epochs': 1,   # Save checkpoint every 2 epochs
            'reconstruction_weight': 1.0,
            'identity_weight': 1.0,
            'max_train_samples': 50,  # pass None to use all samples for full training
            'max_val_samples': 20   # Limit validation samples for testing
        },
        'expression_model': {
            'embed_dim': 384,
            'num_heads': 4,  # Optimized architecture
            'num_layers': 1,  # Optimized architecture
            'dropout': 0.1
        }
    }
    
    print("🚀 Starting Expression Reconstruction Model Training")
    print(f"🖥️  Device: {device}")
    print(f"📊 Training Dataset: {config['training']['train_data_dir']}")
    print(f"📊 Validation Dataset: {config['training']['val_data_dir']}")
    print(f"📈 Logs: {config['training']['log_dir']}")
    print(f"🎯 Learning rate: {config['training']['learning_rate']}")
    print(f"📦 Batch size: {config['training']['batch_size']}")
    print(f"🔄 Epochs: {config['training']['num_epochs']}")
    print(f"🧠 Model: {config['expression_model']['num_layers']} layers, {config['expression_model']['num_heads']} heads")
    
    # Start training
    print("\n🎯 Starting training...")
    train_expression_reconstruction(
        dataset_path=config['training']['train_data_dir'],
        face_id_checkpoint_path=config['training']['face_id_checkpoint_path'],
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
    print(f"📊 Check TensorBoard logs at: {config['training']['log_dir']}/expression_training")

if __name__ == "__main__":
    main() 