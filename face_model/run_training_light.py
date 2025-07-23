#!/usr/bin/env python3
"""
Lightweight Face ID Model training for CPU testing
"""

import torch
import os
import sys
sys.path.append('.')

from training.train_face_id import FaceIDTrainer
from data.dataset import create_face_dataloader

def main():
    """Run lightweight Face ID Model training"""
    
    # Memory optimization for CPU training
    torch.set_num_threads(2)  # Very limited CPU threads
    torch.backends.cudnn.benchmark = False
    
    # Lightweight configuration
    config = {
        'training': {
            'data_dir': "/Users/ozgewhiting/Documents/EQLabs/datasets_serial/CCA_train_db1",
            'log_dir': "./logs",
            'checkpoint_dir': "./checkpoints",
            'learning_rate': 1e-4,
            'batch_size': 4,  # Increased to ensure multiple subjects per batch for contrastive loss
            'num_epochs': 2,  # Fewer epochs for testing
            'save_every': 1,   # Save checkpoint every epoch
            'contrastive_temperature': 0.1,
            'contrastive_margin': 1.0,
            'consistency_weight': 1.0,
            'contrastive_weight': 0.1,  # Reduced to balance with consistency loss
            'use_consistency_loss': True  # Set to False to test only contrastive loss
        },
        'face_id_model': {
            'embed_dim': 384,
            'num_heads': 4,   # Updated to optimized architecture
            'num_layers': 1,  # Updated to optimized architecture
            'dropout': 0.1
        }
    }
    
    print("🚀 Starting Lightweight Face ID Model Training")
    print(f"📊 Dataset: {config['training']['data_dir']}")
    print(f"📈 Logs: {config['training']['log_dir']}")
    print(f"🎯 Learning rate: {config['training']['learning_rate']}")
    print(f"📦 Batch size: {config['training']['batch_size']}")
    print(f"🔄 Epochs: {config['training']['num_epochs']}")
    print(f"🧠 Model: {config['face_id_model']['num_layers']} layers, {config['face_id_model']['num_heads']} heads")
    
    # Create trainer
    trainer = FaceIDTrainer(config)
    
    # Create dataloader with limited samples for testing
    dataloader = create_face_dataloader(
        data_dir=config['training']['data_dir'],
        batch_size=config['training']['batch_size'],
        max_samples=50  # Limit to 50 samples for testing
    )
    
    print(f"\n📋 Dataset loaded: {len(dataloader)} batches per epoch (limited to 50 samples)")
    
    # Start training
    print("\n🎯 Starting lightweight training...")
    trainer.train(dataloader, config['training']['num_epochs'])
    
    print("\n✅ Lightweight training completed!")
    print(f"📊 Check TensorBoard logs at: {config['training']['log_dir']}/face_id_training")

if __name__ == "__main__":
    main() 