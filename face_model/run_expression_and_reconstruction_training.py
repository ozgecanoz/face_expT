#!/usr/bin/env python3
"""
Run Joint Expression and Reconstruction Training
Combines expression extraction with face reconstruction
"""

import torch
import os
import sys
sys.path.append('.')

# Set CUDA memory allocation configuration to prevent fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

from training.train_expression_and_reconstruction import train_expression_and_reconstruction

def main():
    """Run Joint Expression and Reconstruction Training"""
    
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
            'train_data_dir': "/mnt/dataset-storage/dbs/CCA_train_db4_no_padding_keywords_offset_1.0/",
            'max_train_samples': None,  # pass None to use all samples for full training
            #'val_data_dir': "/mnt/dataset-storage/dbs/CCA_train_db4_no_padding/",
            'val_data_dir': "/mnt/dataset-storage/dbs/CCA_val_db2/",
            'max_val_samples': None,   # Limit validation samples for testing
            'checkpoint_dir': "/mnt/dataset-storage/checkpoints",
            'expression_transformer_checkpoint_path': None,  # Set to path if you want to load expression transformer
            'expression_reconstruction_checkpoint_path': None,  # Set to path if you want to load expression reconstruction
            #'joint_checkpoint_path': "/mnt/dataset-storage/face_model/checkpoints_with_keywords6/joint_expression_reconstruction_step_600.pt",  # Set to path if you want to load joint checkpoint (preferred)
            'joint_checkpoint_path': None,
            'log_dir': "/mnt/dataset-storage/logs",
            'learning_rate': 5e-5,
            'warmup_steps': 3000,  # Learning rate warmup steps
            'min_lr': 1e-6,  # Minimum learning rate after decay
            #'batch_size': 1,  # for L4 GPU train-gpu-co 
            'batch_size': 8,  # for A100 GPU trainer-a100-co 
            'num_epochs': 3,
            'save_every_step': 600,   # Save similarity plots and checkpoints every 300 steps
            'num_workers': 8,  # for L4 GPU (24 GB VRAM) train-gpu-co (it has 16 vCPUs), memory 64GB, 
            'pin_memory': True,  # for L4 GPU train-gpu-co 
            'persistent_workers': True,  # Keep workers alive for efficiency
            'drop_last': True,  # Consistent batch sizes
            'freeze_expression_transformer': False  # Set to True to freeze expression transformer and only train reconstruction model
        },
        'scheduler': {
            # Loss weight scheduling parameters
            #'initial_lambda_reconstruction': 0.01,  # Start with low reconstruction weight
            #'initial_lambda_temporal': 0.4,    # Start with high temporal weight, lambda_coherence = 0.3, lambda_contrast = 0.7
            #'initial_lambda_diversity': 0.5,   # Start with high diversity weight
            #'warmup_lambda_reconstruction': 0.2,   # Reconstruction weight at warmup
            #'warmup_lambda_temporal': 0.3,     # Temporal weight at warmup
            #'warmup_lambda_diversity': 0.3,    # Diversity weight at warmup
            #'final_lambda_reconstruction': 0.5,    # Final reconstruction weight (highest)
            #'final_lambda_temporal': 0.2,      # Final temporal weight
            #'final_lambda_diversity': 0.2      # Final diversity weight
            
            # weights for from scratch training in /checkpoints_with_keywords8/joint_expression_reconstruction_step_5400.pt
            'initial_lambda_reconstruction': 0.1,  # Start with low reconstruction weight
            'initial_lambda_temporal': 0.4,    # Start with high temporal weight # also changed  lambda_coherence = 0.7, lambda_contrast = 0.3
            'initial_lambda_diversity': 0.3,   # Start with high diversity weight
            'warmup_lambda_reconstruction': 0.2,   # Reconstruction weight at warmup
            'warmup_lambda_temporal': 0.3,     # Temporal weight at warmup
            'warmup_lambda_diversity': 0.2,    # Diversity weight at warmup
            'final_lambda_reconstruction': 0.5,    # Final reconstruction weight (highest)
            'final_lambda_temporal': 0.2,      # Final temporal weight
            'final_lambda_diversity': 0.2      # Final diversity weight
            
            # weights for from frozen expression but reconstruction training from joint_expression_reconstruction_step_5400.pt above and save to /checkpoints_with_keywords9/
            #'initial_lambda_reconstruction': 1.0,  # Start with low reconstruction weight
            #'initial_lambda_temporal': 0.0,    # Start with high temporal weight # also changed  lambda_coherence = 0.7, lambda_contrast = 0.3
            #'initial_lambda_diversity': 0.0,   # Start with high diversity weight
            #'warmup_lambda_reconstruction': 1.0,   # Reconstruction weight at warmup
            #'warmup_lambda_temporal': 0.0,     # Temporal weight at warmup
            #'warmup_lambda_diversity': 0.0,    # Diversity weight at warmup
            #'final_lambda_reconstruction': 1.0,    # Final reconstruction weight (highest)
            #'final_lambda_temporal': 0.0,      # Final temporal weight
            #'final_lambda_diversity': 0.0      # Final diversity weight
        },
        'expression_transformer': {
            'embed_dim': 384,
            'num_heads': 4,  # Optimized architecture
            'num_layers': 4,  # Optimized architecture
            'dropout': 0.1,
            'max_subjects': 501,  # Added max_subjects parameter
            'ff_dim': 768  # Feed-forward dimension (4 * embed_dim)
        },
        'expression_reconstruction': {
            'embed_dim': 384,
            'num_cross_attention_layers': 2,
            'num_self_attention_layers': 2,
            'num_heads': 4,
            'ff_dim': 768,   # 1536 is 4x384, 768 is 2x384
            'dropout': 0.1
        }
    }
    
    print("🚀 Starting Joint Expression and Reconstruction Training")
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
    print(f"🧠 Expression Reconstruction: {config['expression_reconstruction']['num_cross_attention_layers']} cross layers, {config['expression_reconstruction']['num_self_attention_layers']} self layers, {config['expression_reconstruction']['num_heads']} heads")
    print(f"🧵 Num workers: {config['training']['num_workers']}")
    print(f"💾 Memory optimization: drop_last={config['training']['drop_last']}")
    print(f"⚖️  Loss Weight Schedule:")
    print(f"   Initial: λ_recon={config['scheduler']['initial_lambda_reconstruction']}, λ_temp={config['scheduler']['initial_lambda_temporal']}, λ_div={config['scheduler']['initial_lambda_diversity']}")
    print(f"   Warmup: λ_recon={config['scheduler']['warmup_lambda_reconstruction']}, λ_temp={config['scheduler']['warmup_lambda_temporal']}, λ_div={config['scheduler']['warmup_lambda_diversity']}")
    print(f"   Final: λ_recon={config['scheduler']['final_lambda_reconstruction']}, λ_temp={config['scheduler']['final_lambda_temporal']}, λ_div={config['scheduler']['final_lambda_diversity']}")
    
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
    
    if config['training']['expression_reconstruction_checkpoint_path'] is not None:
        print(f"🔒 Expression Reconstruction: Will load from {config['training']['expression_reconstruction_checkpoint_path']}")
        # Try to load and display architecture info
        try:
            checkpoint = torch.load(config['training']['expression_reconstruction_checkpoint_path'], map_location='cpu')
            if 'config' in checkpoint and 'expression_reconstruction' in checkpoint['config']:
                recon_config = checkpoint['config']['expression_reconstruction']
                print(f"   📐 Architecture: {recon_config.get('num_cross_attention_layers', '?')} cross layers, {recon_config.get('num_self_attention_layers', '?')} self layers, {recon_config.get('num_heads', '?')} heads")
            else:
                print(f"   ⚠️  No architecture info in checkpoint")
        except Exception as e:
            print(f"   ⚠️  Could not read checkpoint info: {str(e)}")
    else:
        print(f"🎓 Expression Reconstruction: Will train from scratch")
    
    # Log freeze status
    if config['training']['freeze_expression_transformer']:
        print(f"\n🔒 Frozen Training: Expression Transformer will be frozen, only Expression Reconstruction will be trained")
    else:
        print(f"\n🔗 Joint Training: Both Expression Transformer and Expression Reconstruction will be trained together")
    
    # Start training
    print("\n🎯 Starting training...")
    train_expression_and_reconstruction(
        dataset_path=config['training']['train_data_dir'],
        expression_transformer_checkpoint_path=config['training']['expression_transformer_checkpoint_path'],
        expression_reconstruction_checkpoint_path=config['training']['expression_reconstruction_checkpoint_path'],
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
        initial_lambda_reconstruction=config['scheduler']['initial_lambda_reconstruction'],
        initial_lambda_temporal=config['scheduler']['initial_lambda_temporal'],
        initial_lambda_diversity=config['scheduler']['initial_lambda_diversity'],
        warmup_lambda_reconstruction=config['scheduler']['warmup_lambda_reconstruction'],
        warmup_lambda_temporal=config['scheduler']['warmup_lambda_temporal'],
        warmup_lambda_diversity=config['scheduler']['warmup_lambda_diversity'],
        final_lambda_reconstruction=config['scheduler']['final_lambda_reconstruction'],
        final_lambda_temporal=config['scheduler']['final_lambda_temporal'],
        final_lambda_diversity=config['scheduler']['final_lambda_diversity'],
        # Architecture configuration
        expr_embed_dim=config['expression_transformer']['embed_dim'],
        expr_num_heads=config['expression_transformer']['num_heads'],
        expr_num_layers=config['expression_transformer']['num_layers'],
        expr_dropout=config['expression_transformer']['dropout'],
        expr_max_subjects=config['expression_transformer']['max_subjects'],
        expr_ff_dim=config['expression_transformer']['ff_dim'],
        recon_embed_dim=config['expression_reconstruction']['embed_dim'],
        recon_num_cross_layers=config['expression_reconstruction']['num_cross_attention_layers'],
        recon_num_self_layers=config['expression_reconstruction']['num_self_attention_layers'],
        recon_num_heads=config['expression_reconstruction']['num_heads'],
        recon_ff_dim=config['expression_reconstruction']['ff_dim'],
        recon_dropout=config['expression_reconstruction']['dropout'],
        log_dir=config['training']['log_dir'],
        freeze_expression_transformer=config['training']['freeze_expression_transformer']
    )
    
    print("\n✅ Training completed!")
    print(f"📊 TensorBoard logs saved to: {config['training']['log_dir']}/exp_recon_training_*")
    print(f"📊 Check TensorBoard logs at: {config['training']['log_dir']}/exp_recon_training_*")

if __name__ == "__main__":
    main() 