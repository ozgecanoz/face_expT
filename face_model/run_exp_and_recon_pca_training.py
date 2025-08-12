#!/usr/bin/env python3
"""
Run Joint Expression and Reconstruction Training with PCA Features
Combines expression extraction with face reconstruction using pre-computed PCA features
"""

import torch
import os
import sys
sys.path.append('.')

# Set CUDA memory allocation configuration to prevent fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'

from training.train_expression_and_reconstruction_with_pca_features import train_expression_and_reconstruction_with_pca_features

def cleanup_gpu_memory():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        import gc
        gc.collect()

def adjust_config_for_device(config, device):
    """Adjust configuration based on device type"""
    if device.type == "cpu":
        # CPU-optimized settings
        config['training']['batch_size'] = 2  # Even smaller for CPU
        config['training']['num_workers'] = 1  # Single worker for CPU
        config['training']['pin_memory'] = False
        config['training']['persistent_workers'] = False
        print("🔧 Configuration adjusted for CPU training")
    else:
        # GPU-optimized settings
        config['training']['batch_size'] = 8  # Larger batches for GPU
        config['training']['num_workers'] = 4  # More workers for GPU
        config['training']['pin_memory'] = True
        config['training']['persistent_workers'] = True
        print("🔧 Configuration optimized for GPU training")
    
    return config

def main():
    """Run Joint Expression and Reconstruction Training with PCA Features"""
    
    # Device detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Memory optimization based on device
    if device.type == "cpu":
        torch.set_num_threads(4)  # Limit CPU threads
        torch.backends.cudnn.benchmark = False  # Disable for CPU
        print(f"💻 CPU Training Mode - Using 4 threads")
    else:
        # GPU optimizations
        torch.backends.cudnn.benchmark = True  # Enable for GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear GPU cache
            # Limit memory usage to 70% of GPU memory (more conservative)
            torch.cuda.set_per_process_memory_fraction(0.7)
            # Force garbage collection
            import gc
            gc.collect()
            # Reset peak memory stats
            torch.cuda.reset_peak_memory_stats()
            print(f"🚀 GPU Training Mode - Memory limited to 70%")
    
    # Configuration - Optimized for PCA features training
    config = {
        'training': {
            'train_data_dir': "/Users/ozgewhiting/Documents/EQLabs/datasets_serial/CCA_train_db4_no_padding/CCA_train_db4_no_padding_features_pca_384/",
            'max_train_samples': None,  # pass None to use all samples for full training
            'val_data_dir': "/Users/ozgewhiting/Documents/EQLabs/datasets_serial/CCA_train_db4_no_padding/CCA_train_db4_no_padding_features_pca_384/",
            'max_val_samples': None,   # Limit validation samples for testing
            'checkpoint_dir': "./checkpoints_pca",
            'expression_transformer_checkpoint_path': None,  # Set to path if you want to load expression transformer
            'expression_reconstruction_checkpoint_path': None,  # Set to path if you want to load expression reconstruction
            'joint_checkpoint_path': None,  # Set to path if you want to load joint checkpoint (preferred)
            'log_dir': "./logs_pca",
            'learning_rate': 5e-5,
            'warmup_steps': 3000,  # Learning rate warmup steps
            'min_lr': 1e-6,  # Minimum learning rate after decay
            'batch_size': 4,  # Smaller batch size for CPU training
            'num_epochs': 3,
            'save_every_step': 100,   # Save similarity plots and checkpoints every 100 steps
            'num_workers': 2,  # Fewer workers for CPU training
            'pin_memory': False,  # Disable for CPU training
            'persistent_workers': False,  # Disable for CPU training
            'drop_last': True,  # Consistent batch sizes
            'freeze_expression_transformer': False,  # Set to True to freeze expression transformer and only train reconstruction model
            'feature_key': 'projected_features'  # Key for features in H5 file
        },
        'scheduler': {
            # Loss weight scheduling parameters - Same as original for consistency
            'initial_lambda_reconstruction': 0.1,  # Start with low reconstruction weight
            'initial_lambda_temporal': 0.4,    # Start with high temporal weight
            'initial_lambda_diversity': 0.4,   # Start with high diversity weight
            'warmup_lambda_reconstruction': 0.2,   # Reconstruction weight at warmup
            'warmup_lambda_temporal': 0.3,     # Temporal weight at warmup
            'warmup_lambda_diversity': 0.2,    # Diversity weight at warmup
            'final_lambda_reconstruction': 0.5,    # Final reconstruction weight (highest)
            'final_lambda_temporal': 0.2,      # Final temporal weight
            'final_lambda_diversity': 0.2      # Final diversity weight
        },
        'expression_transformer': {
            'embed_dim': 384,
            'num_heads': 4,  # Optimized architecture
            'num_layers': 4,  # Optimized architecture
            'dropout': 0.1,
            'max_subjects': 501,  # Added max_subjects parameter
            'ff_dim': 768  # Feed-forward dimension (2 * embed_dim)
        },
        'expression_reconstruction': {
            'embed_dim': 384,
            'num_cross_attention_layers': 2,
            'num_self_attention_layers': 2,
            'num_heads': 4,
            'ff_dim': 768,   # 2x384 for efficiency
            'dropout': 0.1
        }
    }
    
    print("🚀 Starting Joint Expression and Reconstruction Training with PCA Features")
    print(f"🖥️  Device: {device}")
    
    # Print device memory info
    if device.type == "cuda":
        print(f"💾 GPU Memory: {torch.cuda.get_device_name()}")
        print(f"💾 Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"💾 Allocated Memory: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
        print(f"💾 Reserved Memory: {torch.cuda.memory_reserved() / 1024**3:.1f} GB")
        print(f"💾 Free Memory: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3:.1f} GB")
    else:
        print(f"💻 CPU Training Mode")
        print(f"🧵 CPU Threads: {torch.get_num_threads()}")
    
    print(f"📊 Training Dataset: {config['training']['train_data_dir']}")
    print(f"📊 Validation Dataset: {config['training']['val_data_dir']}")
    print(f"📈 Logs: {config['training']['log_dir']}")
    print(f"🎯 Learning rate: {config['training']['learning_rate']}")
    print(f"🔥 Warmup steps: {config['training']['warmup_steps']}")
    print(f"📉 Min LR: {config['training']['min_lr']}")
    print(f"📦 Batch size: {config['training']['batch_size']}")
    print(f"🔄 Epochs: {config['training']['num_epochs']}")
    print(f"💾 Save every: {config['training']['save_every_step']} steps")
    print(f"🔑 Feature key: {config['training']['feature_key']}")
    
    # Final memory cleanup before training
    if device.type == "cuda":
        cleanup_gpu_memory()
        print(f"🧹 GPU memory cleanup completed")
        print(f"💾 Available after cleanup: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3:.1f} GB")
    else:
        import gc
        gc.collect()
        print(f"🧹 CPU memory cleanup completed")
    
    # Adjust configuration based on device
    config = adjust_config_for_device(config, device)
    
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
    
    # Log PCA features info
    print(f"\n🔑 PCA Features Training:")
    print(f"   📐 Input: PCA-projected features (384 dimensions)")
    print(f"   🧩 Patches: 1369 patches per frame (37×37 grid)")
    print(f"   🎬 Frames: 30 frames per clip")
    print(f"   🚀 Expected speedup: 10-50x faster than raw image training")
    
    # Start training
    print("\n🎯 Starting training...")
    train_expression_and_reconstruction_with_pca_features(
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
        freeze_expression_transformer=config['training']['freeze_expression_transformer'],
        feature_key=config['training']['feature_key']
    )
    
    print("\n✅ Training completed!")
    print(f"📊 TensorBoard logs saved to: {config['training']['log_dir']}/exp_recon_pca_training_*")
    print(f"📊 Check TensorBoard logs at: {config['training']['log_dir']}/exp_recon_pca_training_*")

if __name__ == "__main__":
    main() 