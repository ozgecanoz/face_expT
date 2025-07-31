#!/usr/bin/env python3
"""
Run Expression Reconstruction Training (New)
Trains only the expression reconstruction model with frozen expression transformer
"""

import torch
import os
import sys
sys.path.append('.')

from training.train_expression_reconstruction_new import train_expression_reconstruction_new

def main():
    """Run Expression Reconstruction Training (New)"""
    
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
            'train_data_dir': "/mnt/dataset-storage/dbs/CCA_train_db2/",
            'max_train_samples': None,  # pass None to use all samples for full training
            'val_data_dir': "/mnt/dataset-storage/dbs/CCA_val_db2/",
            'max_val_samples': 100,   # Limit validation samples for testing
            'expression_transformer_checkpoint_path': "/mnt/dataset-storage/face_model/checkpoints_with_subject_ids/expression_transformer_epoch_2.pt",  # Frozen expression transformer
            'reconstruction_model_checkpoint_path': None,  # Set to path if you want to continue training from checkpoint
            'log_dir': "/mnt/dataset-storage/face_model/logs",
            'checkpoint_dir': "/mnt/dataset-storage/face_model/checkpoints_with_subject_ids",
            'learning_rate': 1e-4,
            'batch_size': 4,  # for L4 GPU train-gpu-co 
            'num_epochs': 3,
            'save_every_epochs': 1,   # Save checkpoint every epoch
            'num_workers': 8,  # for L4 GPU (24 GB VRAM) train-gpu-co (it has 16 vCPUs), memory 64GB
            'pin_memory': True,  # for L4 GPU train-gpu-co 
            'persistent_workers': True,  # Keep workers alive for efficiency
            'drop_last': True  # Consistent batch sizes
        },
        'expression_reconstruction': {
            'embed_dim': 384,
            'num_cross_attention_layers': 2,
            'num_self_attention_layers': 2,
            'num_heads': 8,
            'ff_dim': 1536,
            'dropout': 0.1
        }
    }
    
    print("🚀 Starting Expression Reconstruction Training (New)")
    print(f"🖥️  Device: {device}")
    print(f"📊 Training Dataset: {config['training']['train_data_dir']}")
    print(f"📊 Validation Dataset: {config['training']['val_data_dir']}")
    print(f"📈 Logs: {config['training']['log_dir']}")
    print(f"🎯 Learning rate: {config['training']['learning_rate']}")
    print(f"📦 Batch size: {config['training']['batch_size']}")
    print(f"🔄 Epochs: {config['training']['num_epochs']}")
    print(f"🧠 Expression Reconstruction: {config['expression_reconstruction']['num_cross_attention_layers']} cross-attention, {config['expression_reconstruction']['num_self_attention_layers']} self-attention layers, {config['expression_reconstruction']['num_heads']} heads")
    print(f"🧵 Num workers: {config['training']['num_workers']}")
    print(f"💾 Memory optimization: drop_last={config['training']['drop_last']}")
    
    # Log checkpoint status
    if config['training']['expression_transformer_checkpoint_path'] is not None:
        print(f"🔒 Expression Transformer: Will load from {config['training']['expression_transformer_checkpoint_path']}")
        # Try to load and display architecture info
        try:
            checkpoint = torch.load(config['training']['expression_transformer_checkpoint_path'], map_location='cpu')
            if 'config' in checkpoint and 'expression_model' in checkpoint['config']:
                expr_config = checkpoint['config']['expression_model']
                print(f"   📐 Architecture: {expr_config.get('num_layers', '?')} layers, {expr_config.get('num_heads', '?')} heads")
            else:
                print(f"   ⚠️  No architecture info in checkpoint")
        except Exception as e:
            print(f"   ⚠️  Could not read checkpoint info: {str(e)}")
    else:
        print(f"❌ Expression Transformer checkpoint is required!")
        return
    
    if config['training']['reconstruction_model_checkpoint_path'] is not None:
        print(f"🔒 Reconstruction Model: Will load from {config['training']['reconstruction_model_checkpoint_path']}")
        # Try to load and display architecture info
        try:
            checkpoint = torch.load(config['training']['reconstruction_model_checkpoint_path'], map_location='cpu')
            if 'config' in checkpoint:
                recon_config = checkpoint['config']
                print(f"   📐 Architecture: {recon_config.get('num_cross_attention_layers', '?')} cross-attention, {recon_config.get('num_self_attention_layers', '?')} self-attention layers, {recon_config.get('num_heads', '?')} heads")
            else:
                print(f"   ⚠️  No architecture info in checkpoint")
        except Exception as e:
            print(f"   ⚠️  Could not read checkpoint info: {str(e)}")
    else:
        print(f"🎓 Reconstruction Model: Will train from scratch")
    
    print(f"\n🔗 Training Setup: Expression Transformer (frozen) + Expression Reconstruction Model (trainable)")
    
    # Start training
    print("\n🎯 Starting training...")
    train_expression_reconstruction_new(
        dataset_path=config['training']['train_data_dir'],
        expression_transformer_checkpoint_path=config['training']['expression_transformer_checkpoint_path'],
        reconstruction_model_checkpoint_path=config['training']['reconstruction_model_checkpoint_path'],
        checkpoint_dir=config['training']['checkpoint_dir'],
        save_every_epochs=config['training']['save_every_epochs'],
        batch_size=config['training']['batch_size'],
        num_epochs=config['training']['num_epochs'],
        learning_rate=config['training']['learning_rate'],
        max_samples=config['training']['max_train_samples'],
        val_dataset_path=config['training']['val_data_dir'],
        max_val_samples=config['training']['max_val_samples'],
        device=device,
        embed_dim=config['expression_reconstruction']['embed_dim'],
        num_cross_attention_layers=config['expression_reconstruction']['num_cross_attention_layers'],
        num_self_attention_layers=config['expression_reconstruction']['num_self_attention_layers'],
        num_heads=config['expression_reconstruction']['num_heads'],
        ff_dim=config['expression_reconstruction']['ff_dim'],
        dropout=config['expression_reconstruction']['dropout']
    )
    
    print("\n✅ Training completed!")
    print(f"📊 Checkpoints saved to: {config['training']['checkpoint_dir']}")

if __name__ == "__main__":
    main() 