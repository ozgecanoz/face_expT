#!/usr/bin/env python3
"""
Run Expression Prediction Training
Joint training of Expression Transformer + Transformer Decoder
"""

import torch
import os
import sys
sys.path.append('.')

from training.train_expression_prediction import train_expression_prediction

def main():
    """Run Expression Prediction Training"""
    
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
            'expression_transformer_checkpoint_path': None,  # Set to path if you want to load expression transformer
            'transformer_decoder_checkpoint_path': None,  # Set to path if you want to load transformer decoder
            'log_dir': "/Users/ozgewhiting/Documents/projects/dataset_utils/face_model/logs",
            'checkpoint_dir': "/Users/ozgewhiting/Documents/projects/dataset_utils/face_model/checkpoints",
            'learning_rate': 1e-4,
            'batch_size': 2,  # Reduced for CPU memory
            'num_epochs': 2,
            'save_every_epochs': 1,   # Save checkpoint every epoch
            'max_train_samples': 10,  # pass None to use all samples for full training
            'max_val_samples': 10   # Limit validation samples for testing
        },
        'expression_transformer': {
            'embed_dim': 384,
            'num_heads': 4,  # Optimized architecture
            'num_layers': 1,  # Optimized architecture
            'dropout': 0.1
        },
        'transformer_decoder': {
            'embed_dim': 384,
            'num_heads': 4,  # Optimized architecture
            'num_layers': 1,  # Optimized architecture
            'dropout': 0.1,
            'max_sequence_length': 50
        }
    }
    
    print("🚀 Starting Expression Prediction Training")
    print(f"🖥️  Device: {device}")
    print(f"📊 Training Dataset: {config['training']['train_data_dir']}")
    print(f"📊 Validation Dataset: {config['training']['val_data_dir']}")
    print(f"📈 Logs: {config['training']['log_dir']}")
    print(f"🎯 Learning rate: {config['training']['learning_rate']}")
    print(f"📦 Batch size: {config['training']['batch_size']}")
    print(f"🔄 Epochs: {config['training']['num_epochs']}")
    print(f"🧠 Expression Transformer: {config['expression_transformer']['num_layers']} layers, {config['expression_transformer']['num_heads']} heads")
    print(f"🧠 Transformer Decoder: {config['transformer_decoder']['num_layers']} layers, {config['transformer_decoder']['num_heads']} heads")
    
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
        print(f"🎓 Expression Transformer: Will train from scratch")
    
    if config['training']['transformer_decoder_checkpoint_path'] is not None:
        print(f"🔒 Transformer Decoder: Will load from {config['training']['transformer_decoder_checkpoint_path']}")
        # Try to load and display architecture info
        try:
            checkpoint = torch.load(config['training']['transformer_decoder_checkpoint_path'], map_location='cpu')
            if 'config' in checkpoint and 'transformer_decoder' in checkpoint['config']:
                decoder_config = checkpoint['config']['transformer_decoder']
                print(f"   📐 Architecture: {decoder_config.get('num_layers', '?')} layers, {decoder_config.get('num_heads', '?')} heads")
            else:
                print(f"   ⚠️  No architecture info in checkpoint")
        except Exception as e:
            print(f"   ⚠️  Could not read checkpoint info: {str(e)}")
    else:
        print(f"🎓 Transformer Decoder: Will train from scratch")
    
    print(f"\n🔗 Joint Training: Both Expression Transformer and Transformer Decoder will be trained together")
    print(f"🔒 Frozen: Face ID Model (Component A)")
    
    # Start training
    print("\n🎯 Starting training...")
    train_expression_prediction(
        dataset_path=config['training']['train_data_dir'],
        face_id_checkpoint_path=config['training']['face_id_checkpoint_path'],
        expression_transformer_checkpoint_path=config['training']['expression_transformer_checkpoint_path'],
        transformer_decoder_checkpoint_path=config['training']['transformer_decoder_checkpoint_path'],
        checkpoint_dir=config['training']['checkpoint_dir'],
        save_every_epochs=config['training']['save_every_epochs'],
        batch_size=config['training']['batch_size'],
        num_epochs=config['training']['num_epochs'],
        learning_rate=config['training']['learning_rate'],
        max_samples=config['training']['max_train_samples'],
        val_dataset_path=config['training']['val_data_dir'],
        max_val_samples=config['training']['max_val_samples'],
        device=device
    )
    
    print("\n✅ Training completed!")
    print(f"📊 TensorBoard logs will be saved with unique job ID (exp_pred_training_<id>)")
    print(f"📊 Check TensorBoard logs at: {config['training']['log_dir']}/exp_pred_training_*")

if __name__ == "__main__":
    main() 