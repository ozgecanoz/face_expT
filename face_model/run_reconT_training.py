#!/usr/bin/env python3
"""
Run script for expression reconstruction training
"""

import argparse
import torch
import os
from training.train_reconT import train_expression_reconstruction


def main():
    parser = argparse.ArgumentParser(description="Train ExpressionReconstructionModel using frozen ExpressionTransformer")
    
    # Required arguments
    parser.add_argument("--dataset-path", type=str, 
                        default="/mnt/dataset-storage/dbs/CCA_train_db4_no_padding_keywords_offset_1.0/",
                       help="Path to CCA dataset directory")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples to use")
    parser.add_argument("--pca-json-path", type=str, 
                        default="/mnt/dataset-storage/dbs/combined_pca_directions.json",
                       help="Path to PCA projection JSON file")
    parser.add_argument("--expression-transformer-checkpoint", type=str, 
                        default="/mnt/dataset-storage/checkpoints/expT_supervised_epoch_10_step_280.pt",
                       help="Path to ExpressionTransformer checkpoint to load")
    
    # Optional arguments
    parser.add_argument("--checkpoint-dir", type=str, 
                        default="/mnt/dataset-storage/checkpoints/",
                       help="Directory to save checkpoints")
    parser.add_argument("--save-every-step", type=int, default=2,
                       help="Save checkpoints every N steps")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--num-epochs", type=int, default=2,
                       help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=500,
                       help="Number of warmup steps for scheduler")
    parser.add_argument("--min-lr", type=float, default=1e-6,
                       help="Minimum learning rate")
    parser.add_argument("--lambda-identity", type=float, default=0.5,
                       help="Weight for identity loss (ArcFace-based)")

    # Validation arguments
    parser.add_argument("--val-dataset-path", type=str, 
                        default="/mnt/dataset-storage/dbs/CCA_val_db4_no_padding/",
                       help="Path to validation dataset")
    parser.add_argument("--max-val-samples", type=int, default=100,
                       help="Maximum number of validation samples")
    
    # Architecture parameters
    parser.add_argument("--embed-dim", type=int, default=384,
                       help="Embedding dimension")
    parser.add_argument("--num-cross-attention-layers", type=int, default=2,
                       help="Number of cross-attention layers")
    parser.add_argument("--num-self-attention-layers", type=int, default=2,
                       help="Number of self-attention layers")
    parser.add_argument("--num-heads", type=int, default=4,
                       help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="Dropout rate")
    parser.add_argument("--ff-dim", type=int, default=768,   # make it 2x embed_dim and not 4x (1536)
                       help="Feed-forward dimension")
    parser.add_argument("--max-subjects", type=int, default=501,
                       help="Maximum number of subjects")
    
    # System parameters
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda, cuda:0, etc.)")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Number of data loader workers")
    parser.add_argument("--pin-memory", action="store_true",
                       help="Pin memory for faster GPU transfer")
    parser.add_argument("--persistent-workers", action="store_true",
                       help="Use persistent workers")
    parser.add_argument("--drop-last", action="store_true", default=True,
                       help="Drop last incomplete batch")
    
    # Logging and memory
    parser.add_argument("--log-dir", type=str, default="/mnt/dataset-storage/logs",
                       help="Directory for TensorBoard logs")
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
    
    if not os.path.exists(args.expression_transformer_checkpoint):
        print(f"❌ Error: ExpressionTransformer checkpoint not found: {args.expression_transformer_checkpoint}")
        return
    
    print("🚀 Starting Expression Reconstruction Training")
    print("=" * 60)
    print(f"Dataset: {args.dataset_path}")
    print(f"PCA Projection: {args.pca_json_path}")
    print(f"ExpressionTransformer Checkpoint: {args.expression_transformer_checkpoint}")
    print(f"Device: {device}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Max Memory Fraction: {args.max_memory_fraction}")
    print(f"Embed Dim: {args.embed_dim}")
    print(f"Cross-Attention Layers: {args.num_cross_attention_layers}")
    print(f"Self-Attention Layers: {args.num_self_attention_layers}")
    print(f"Num Heads: {args.num_heads}")
    print(f"Max Subjects: {args.max_subjects}")
    print(f"Identity Loss Weight: {args.lambda_identity}")
    
    if args.val_dataset_path:
        print(f"Validation Dataset: {args.val_dataset_path}")
    
    print("=" * 60)
    
    # Start training
    try:
        model = train_expression_reconstruction(
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
            num_cross_attention_layers=args.num_cross_attention_layers,
            num_self_attention_layers=args.num_self_attention_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            ff_dim=args.ff_dim,
            max_subjects=args.max_subjects,
            log_dir=args.log_dir,
            max_memory_fraction=args.max_memory_fraction,
            lambda_identity=args.lambda_identity
        )
        
        print("🎉 Training completed successfully!")
        print(f"Model saved in: {args.checkpoint_dir}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
