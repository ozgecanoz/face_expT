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
    
    # Memory optimization for CPU training
    torch.set_num_threads(4)  # Limit CPU threads
    torch.backends.cudnn.benchmark = False  # Disable for CPU
    
    # Configuration - Optimized for CPU training
    config = {
        'training': {
            'data_dir': "/Users/ozgewhiting/Documents/EQLabs/datasets_serial/CCA_train_db1",
            'log_dir': "./logs",
            'checkpoint_dir': "./checkpoints",
            'learning_rate': 1e-4,
            'batch_size': 4,  # Increased to ensure multiple subjects per batch for contrastive loss
            'num_epochs': 10,
            'save_every': 2,   # Save checkpoint every 2 epochs
            'contrastive_temperature': 0.1,
            'contrastive_margin': 1.0,
            'consistency_weight': 1.0,
            'contrastive_weight': 0.1  # Reduced to balance with consistency loss
        },
        'face_id_model': {
            'embed_dim': 384,
            'num_heads': 4,  # Updated to optimized architecture
            'num_layers': 1,  # Updated to optimized architecture
            'dropout': 0.1
        }
    }
    
    print("🚀 Starting Face ID Model Training")
    print(f"📊 Dataset: {config['training']['data_dir']}")
    print(f"📈 Logs: {config['training']['log_dir']}")
    print(f"🎯 Learning rate: {config['training']['learning_rate']}")
    print(f"📦 Batch size: {config['training']['batch_size']}")
    print(f"🔄 Epochs: {config['training']['num_epochs']}")
    
    # Create trainer
    trainer = FaceIDTrainer(config)
    
    # Create dataloader
    dataloader = create_face_dataloader(
        data_dir=config['training']['data_dir'],
        batch_size=config['training']['batch_size']
    )
    
    print(f"\n📋 Dataset loaded: {len(dataloader)} batches per epoch")
    
    # Start training
    print("\n🎯 Starting training...")
    trainer.train(dataloader, config['training']['num_epochs'])
    
    print("\n✅ Training completed!")
    print(f"📊 Check TensorBoard logs at: {config['training']['log_dir']}/face_id_training")

if __name__ == "__main__":
    main() 