#!/usr/bin/env python3
"""
Run script for supervised expression transformer training
"""

import argparse
import torch
import os
from training.train_expT_supervised import train_expression_transformer_supervised


def main():
    parser = argparse.ArgumentParser(description="Train Expression Transformer for supervised emotion classification")
    
    # Required arguments
    parser.add_argument("--dataset-path", type=str, 
    default="/home/jupyter/dbs/AffectNet_518_train/",
                       help="Path to AffectNet dataset directory")
    parser.add_argument("--pca-json-path", type=str, 
    default="/home/jupyter/dbs/combined_pca_directions.json",
                       help="Path to PCA projection JSON file")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of samples to use")

    parser.add_argument("--val-dataset-path", type=str, 
    default="/home/jupyter/dbs/AffectNet_518_valid/",
                       help="Path to validation dataset")
    parser.add_argument("--max-val-samples", type=int, default=None,
                       help="Maximum number of validation samples")

    parser.add_argument("--log-dir", type=str, 
    default="/home/jupyter/logs/",
                       help="Directory for TensorBoard logs")
    parser.add_argument("--checkpoint-dir", type=str, 
    default="/home/jupyter/checkpoints/",
                       help="Directory to save checkpoints")
    
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Training batch size")
    parser.add_argument("--expression-transformer-checkpoint", type=str, default=None,
                       help="Path to expression transformer checkpoint to load")
    parser.add_argument("--num-epochs", type=int, default=5,
                       help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=1000,
                       help="Number of warmup steps for scheduler")
    parser.add_argument("--min-lr", type=float, default=1e-6,
                       help="Minimum learning rate")
    
    parser.add_argument("--save-every-step", type=int, default=500,
                       help="Save checkpoints every N steps")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda, cuda:0, etc.)")
    parser.add_argument("--num-workers", type=int, default=8,
                       help="Number of data loader workers")
    parser.add_argument("--pin-memory", action="store_true",
                       help="Pin memory for faster GPU transfer")
    parser.add_argument("--persistent-workers", action="store_true",
                       help="Use persistent workers")
    parser.add_argument("--drop-last", action="store_true", default=True,
                       help="Drop last incomplete batch")
    
    # Architecture parameters
    parser.add_argument("--embed-dim", type=int, default=384,
                       help="Expression transformer embedding dimension")
    parser.add_argument("--num-heads", type=int, default=4,
                       help="Expression transformer number of heads")
    parser.add_argument("--num-layers", type=int, default=4,
                       help="Expression transformer number of layers")
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="Expression transformer dropout")
    parser.add_argument("--ff-dim", type=int, default=1536,
                       help="Expression transformer feed-forward dimension")
    parser.add_argument("--grid-size", type=int, default=37,
                       help="Grid size for positional embeddings")
    parser.add_argument("--num-classes", type=int, default=8,
                       help="Number of emotion classes")
    
    # Memory management
    parser.add_argument("--max-memory-fraction", type=float, default=0.7,
                       help="Maximum GPU memory fraction to use (0.0-1.0, default: 0.9)")
    
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            print("💻 Using CPU")
    else:
        device = torch.device(args.device)
        print(f"🎯 Using device: {device}")
    
    # Validate paths
    if not os.path.exists(args.dataset_path):
        print(f"❌ Error: Dataset path not found: {args.dataset_path}")
        return
    
    if not os.path.exists(args.pca_json_path):
        print(f"❌ Error: PCA JSON path not found: {args.pca_json_path}")
        return
    
    if args.expression_transformer_checkpoint and not os.path.exists(args.expression_transformer_checkpoint):
        print(f"❌ Error: Checkpoint path not found: {args.expression_transformer_checkpoint}")
        return
    
    print("🚀 Starting Supervised Expression Transformer Training")
    print("=" * 60)
    print(f"Dataset: {args.dataset_path}")
    print(f"PCA Projection: {args.pca_json_path}")
    print(f"Device: {device}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Max Memory Fraction: {args.max_memory_fraction}")
    
    if args.expression_transformer_checkpoint:
        print(f"Checkpoint: {args.expression_transformer_checkpoint}")
        print("📋 Model config will be loaded from checkpoint")
    else:
        print("📋 Using provided model configuration")
        print(f"Embed Dim: {args.embed_dim}")
        print(f"Num Heads: {args.num_heads}")
        print(f"Num Layers: {args.num_layers}")
        print(f"Num Classes: {args.num_classes}")
    
    print("=" * 60)
    
    # Start training
    try:
        model = train_expression_transformer_supervised(
            dataset_path=args.dataset_path,
            pca_json_path=args.pca_json_path,
            expression_transformer_checkpoint_path=args.expression_transformer_checkpoint,
            checkpoint_dir=args.checkpoint_dir,
            save_every_step=args.save_every_step,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            max_samples=args.max_samples,
            val_dataset_path=args.val_dataset_path,
            max_val_samples=args.max_val_samples,
            device=device,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.persistent_workers,
            drop_last=args.drop_last,
            warmup_steps=args.warmup_steps,
            min_lr=args.min_lr,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
            ff_dim=args.ff_dim,
            grid_size=args.grid_size,
            num_classes=args.num_classes,
            log_dir=args.log_dir,
            max_memory_fraction=args.max_memory_fraction
        )
        
        print("🎉 Training completed successfully!")
        print(f"Model saved in: {args.checkpoint_dir}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
