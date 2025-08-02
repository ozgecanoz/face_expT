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
            #'train_data_dir': "/mnt/dataset-storage/dbs/CCA_train_db2/",
            #'max_train_samples': None,  # pass None to use all samples for full training
            #'val_data_dir': "/mnt/dataset-storage/dbs/CCA_val_db2/",
            #'max_val_samples': 100,   # Limit validation samples for testing
            #'checkpoint_dir': "/mnt/dataset-storage/face_model/checkpoints_with_subject_ids",
            'train_data_dir': "/mnt/dataset-storage/dbs/CCA_train_db4_no_padding_keywords_offset_1.0/", # this is with keyword based clip generation
            'max_train_samples': None,  # pass None to use all samples for full training
            'val_data_dir': "/mnt/dataset-storage/dbs/CCA_train_db4_no_padding/",   # this is with random clip generation
            'max_val_samples': 2000,   # Limit validation samples for testing
            'checkpoint_dir': "/mnt/dataset-storage/face_model/checkpoints_with_keywords3",
            'expression_transformer_checkpoint_path': None,  # Set to path if you want to load expression transformer
            'transformer_decoder_checkpoint_path': None,  # Set to path if you want to load transformer decoder
            'joint_checkpoint_path': None,  # Set to path if you want to load joint checkpoint (preferred)
            'log_dir': "/mnt/dataset-storage/face_model/logs",
            'learning_rate': 5e-5,
            'warmup_steps': 3000,  # Learning rate warmup steps
            'min_lr': 1e-6,  # Minimum learning rate after decay
            #'batch_size': 16,  # Optimized for 64GB RAM (cpu vm)
            'batch_size': 4,  # for L4 GPU train-gpu-co 
            'num_epochs': 2,
            'save_every_step': 300,   # Save similarity plots and checkpoints every 500 steps
            #'num_workers': 4,  # Parallel data loading with 16 vCPUs
            'num_workers': 8,  # for L4 GPU (24 GB VRAM) train-gpu-co (it has 16 vCPUs), memory 64GB, 
            #'pin_memory': False,  # Not needed for CPU
            'pin_memory': True,  # for L4 GPU train-gpu-co 
            'persistent_workers': True,  # Keep workers alive for efficiency
            'drop_last': True  # Consistent batch sizes
        },
        'scheduler': {
            # Loss weight scheduling parameters
            'initial_lambda_prediction': 0.01,  # Start with low prediction weight
            'initial_lambda_temporal': 0.1,    # Start with high temporal weight
            'initial_lambda_diversity': 0.5,   # Start with high diversity weight
            'warmup_lambda_prediction': 0.15,   # Prediction weight at warmup
            'warmup_lambda_temporal': 0.3,     # Temporal weight at warmup
            'warmup_lambda_diversity': 0.4,    # Diversity weight at warmup
            'final_lambda_prediction': 0.5,    # Final prediction weight (highest)
            'final_lambda_temporal': 0.2,      # Final temporal weight
            'final_lambda_diversity': 0.2      # Final diversity weight
        },
        'expression_transformer': {
            'embed_dim': 384,
            'num_heads': 8,  # Optimized architecture
            'num_layers': 4,  # Optimized architecture
            'dropout': 0.1,
            'max_subjects': 3011  # Added max_subjects parameter
        },
        'transformer_decoder': {
            'embed_dim': 384,
            'num_heads': 8,  # Optimized architecture
            'num_layers': 4,  # Optimized architecture
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
    print(f"🔥 Warmup steps: {config['training']['warmup_steps']}")
    print(f"📉 Min LR: {config['training']['min_lr']}")
    print(f"📦 Batch size: {config['training']['batch_size']}")
    print(f"🔄 Epochs: {config['training']['num_epochs']}")
    print(f"💾 Save every: {config['training']['save_every_step']} steps")
    print(f"🧠 Expression Transformer: {config['expression_transformer']['num_layers']} layers, {config['expression_transformer']['num_heads']} heads")
    print(f"🧠 Transformer Decoder: {config['transformer_decoder']['num_layers']} layers, {config['transformer_decoder']['num_heads']} heads")
    print(f"🧵 Num workers: {config['training']['num_workers']}")
    print(f"💾 Memory optimization: drop_last={config['training']['drop_last']}")
    print(f"⚖️  Loss Weight Schedule:")
    print(f"   Initial: λ_pred={config['scheduler']['initial_lambda_prediction']}, λ_temp={config['scheduler']['initial_lambda_temporal']}, λ_div={config['scheduler']['initial_lambda_diversity']}")
    print(f"   Warmup: λ_pred={config['scheduler']['warmup_lambda_prediction']}, λ_temp={config['scheduler']['warmup_lambda_temporal']}, λ_div={config['scheduler']['warmup_lambda_diversity']}")
    print(f"   Final: λ_pred={config['scheduler']['final_lambda_prediction']}, λ_temp={config['scheduler']['final_lambda_temporal']}, λ_div={config['scheduler']['final_lambda_diversity']}")
    

    
    
    # Log checkpoint status
    if config['training']['joint_checkpoint_path'] is not None:
        print(f"🔒 Joint Model: Will load from {config['training']['joint_checkpoint_path']}")
        # Try to load and display architecture info
        try:
            checkpoint = torch.load(config['training']['joint_checkpoint_path'], map_location='cpu')
            if 'total_steps' in checkpoint:
                print(f"   📊 Training step: {checkpoint['total_steps']}")
            if 'epoch' in checkpoint:
                print(f"   📅 Epoch: {checkpoint['epoch']}")
        except Exception as e:
            print(f"   ⚠️  Could not read checkpoint info: {str(e)}")
    elif config['training']['expression_transformer_checkpoint_path'] is not None:
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
    
    # Start training
    print("\n🎯 Starting training...")
    train_expression_prediction(
        dataset_path=config['training']['train_data_dir'],
        expression_transformer_checkpoint_path=config['training']['expression_transformer_checkpoint_path'],
        transformer_decoder_checkpoint_path=config['training']['transformer_decoder_checkpoint_path'],
        joint_checkpoint_path=config['training']['joint_checkpoint_path'],
        checkpoint_dir=config['training']['checkpoint_dir'],
        save_every_step=config['training']['save_every_step'],
        batch_size=config['training']['batch_size'],
        num_epochs=config['training']['num_epochs'],
        learning_rate=config['training']['learning_rate'],
        warmup_steps=config['training']['warmup_steps'],
        min_lr=config['training']['min_lr'],
        max_samples=config['training']['max_train_samples'],
        val_dataset_path=config['training']['val_data_dir'],
        max_val_samples=config['training']['max_val_samples'],
        device=device,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory'],
        persistent_workers=config['training']['persistent_workers'],
        drop_last=config['training']['drop_last'],
        # Loss weight scheduling parameters
        initial_lambda_prediction=config['scheduler']['initial_lambda_prediction'],
        initial_lambda_temporal=config['scheduler']['initial_lambda_temporal'],
        initial_lambda_diversity=config['scheduler']['initial_lambda_diversity'],
        warmup_lambda_prediction=config['scheduler']['warmup_lambda_prediction'],
        warmup_lambda_temporal=config['scheduler']['warmup_lambda_temporal'],
        warmup_lambda_diversity=config['scheduler']['warmup_lambda_diversity'],
        final_lambda_prediction=config['scheduler']['final_lambda_prediction'],
        final_lambda_temporal=config['scheduler']['final_lambda_temporal'],
        final_lambda_diversity=config['scheduler']['final_lambda_diversity'],
        # Architecture configuration
        expr_embed_dim=config['expression_transformer']['embed_dim'],
        expr_num_heads=config['expression_transformer']['num_heads'],
        expr_num_layers=config['expression_transformer']['num_layers'],
        expr_dropout=config['expression_transformer']['dropout'],
        expr_max_subjects=config['expression_transformer']['max_subjects'],  # Added max_subjects parameter
        decoder_embed_dim=config['transformer_decoder']['embed_dim'],
        decoder_num_heads=config['transformer_decoder']['num_heads'],
        decoder_num_layers=config['transformer_decoder']['num_layers'],
        decoder_dropout=config['transformer_decoder']['dropout'],
        max_sequence_length=config['transformer_decoder']['max_sequence_length'],
        log_dir=config['training']['log_dir']
    )
    
    print("\n✅ Training completed!")
    print(f"📊 TensorBoard logs saved to: {config['training']['log_dir']}/exp_pred_training_*")
    print(f"📊 Check TensorBoard logs at: {config['training']['log_dir']}/exp_pred_training_*")

if __name__ == "__main__":
    main() 