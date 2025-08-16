#!/usr/bin/env python3
"""
Evaluation script for supervised expression transformer model
Loads model from checkpoint and evaluates performance on test dataset
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import json
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

# Add the project root to the path for imports
import sys
sys.path.append('.')

from data.dataset import FaceDataset
from models.dinov2_tokenizer import DINOv2BaseTokenizer
from utils.checkpoint_utils import load_checkpoint_config, extract_model_config
from training.train_expT_supervised import ExpTClassifierModel, load_pca_projection, prepare_emotion_classification_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model(
    model,
    dataloader,
    dinov2_tokenizer,
    pca_components,
    pca_mean,
    device,
    num_classes=8
):
    """
    Evaluate the model on the test dataset
    
    Args:
        model: Trained ExpTClassifierModel
        dataloader: DataLoader for test dataset
        dinov2_tokenizer: DINOv2BaseTokenizer instance
        pca_components: PCA projection matrix
        pca_mean: PCA mean vector
        device: Device to use
        num_classes: Number of emotion classes
        
    Returns:
        results: Dictionary containing evaluation metrics
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    logger.info("Starting evaluation...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            try:
                # Prepare data
                face_images, emotion_targets, clip_lengths = prepare_emotion_classification_data(
                    batch, dinov2_tokenizer, pca_components, pca_mean, device
                )
                
                # Forward pass
                expression_tokens, logits = model(face_images)
                
                # Get predictions
                probabilities = F.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(emotion_targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                continue
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_probabilities = np.array(all_probabilities)
    
    logger.info(f"Evaluation completed: {len(all_predictions)} samples processed")
    
    # Compute metrics
    results = compute_evaluation_metrics(all_predictions, all_targets, all_probabilities, num_classes)
    
    return results


def compute_evaluation_metrics(predictions, targets, probabilities, num_classes):
    """
    Compute comprehensive evaluation metrics
    
    Args:
        predictions: (N,) - Predicted class labels
        targets: (N,) - Ground truth class labels
        probabilities: (N, num_classes) - Predicted probabilities
        num_classes: Number of classes
        
    Returns:
        results: Dictionary containing all metrics
    """
    logger.info("Computing evaluation metrics...")
    
    # Basic accuracy
    accuracy = np.mean(predictions == targets)
    
    # Classification report
    class_report = classification_report(
        targets, predictions, 
        output_dict=True, 
        zero_division=0
    )
    
    # Confusion matrix
    conf_matrix = confusion_matrix(targets, predictions)
    
    # Per-class metrics
    per_class_metrics = {}
    for class_id in range(num_classes):
        class_mask = targets == class_id
        class_samples = np.sum(class_mask)
        
        if class_samples > 0:
            class_predictions = predictions[class_mask]
            class_correct = np.sum(class_predictions == class_id)
            
            per_class_metrics[class_id] = {
                'samples': int(class_samples),
                'correct': int(class_correct),
                'accuracy': float(class_correct / class_samples),
                'precision': float(class_report[str(class_id)]['precision']),
                'recall': float(class_report[str(class_id)]['recall']),
                'f1_score': float(class_report[str(class_id)]['f1-score'])
            }
        else:
            per_class_metrics[class_id] = {
                'samples': 0,
                'correct': 0,
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }
    
    # Overall metrics
    overall_metrics = {
        'accuracy': float(accuracy),
        'macro_precision': float(class_report['macro avg']['precision']),
        'macro_recall': float(class_report['macro avg']['recall']),
        'macro_f1': float(class_report['macro avg']['f1-score']),
        'weighted_precision': float(class_report['weighted avg']['precision']),
        'weighted_recall': float(class_report['weighted avg']['recall']),
        'weighted_f1': float(class_report['weighted avg']['f1-score'])
    }
    
    results = {
        'overall': overall_metrics,
        'per_class': per_class_metrics,
        'confusion_matrix': conf_matrix.tolist(),
        'predictions': predictions.tolist(),
        'targets': targets.tolist(),
        'probabilities': probabilities.tolist()
    }
    
    return results


def print_evaluation_results(results, class_names=None):
    """
    Print evaluation results in a formatted table
    
    Args:
        results: Results dictionary from compute_evaluation_metrics
        class_names: Optional list of class names
    """
    print("\n" + "="*80)
    print("🎯 EXPRESSION TRANSFORMER SUPERVISED EVALUATION RESULTS")
    print("="*80)
    
    # Overall metrics
    overall = results['overall']
    print(f"\n📊 OVERALL PERFORMANCE:")
    print(f"   Accuracy:           {overall['accuracy']:.4f}")
    print(f"   Macro Precision:    {overall['macro_precision']:.4f}")
    print(f"   Macro Recall:       {overall['macro_recall']:.4f}")
    print(f"   Macro F1-Score:     {overall['macro_f1']:.4f}")
    print(f"   Weighted Precision: {overall['weighted_precision']:.4f}")
    print(f"   Weighted Recall:    {overall['weighted_recall']:.4f}")
    print(f"   Weighted F1-Score:  {overall['weighted_f1']:.4f}")
    
    # Per-class metrics
    print(f"\n📋 PER-CLASS PERFORMANCE:")
    print("-" * 80)
    print(f"{'Class':<15} {'Samples':<10} {'Correct':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 80)
    
    per_class = results['per_class']
    for class_id in sorted(per_class.keys()):
        metrics = per_class[class_id]
        class_name = class_names[class_id] if class_names and class_id < len(class_names) else f"Class_{class_id}"
        
        print(f"{class_name:<15} {metrics['samples']:<10} {metrics['correct']:<10} "
              f"{metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
              f"{metrics['recall']:<10.4f} {metrics['f1_score']:<10.4f}")
    
    print("-" * 80)
    
    # Confusion matrix summary
    conf_matrix = np.array(results['confusion_matrix'])
    print(f"\n🔍 CONFUSION MATRIX SUMMARY:")
    print(f"   Total samples: {len(results['targets'])}")
    print(f"   Matrix shape: {conf_matrix.shape}")
    print(f"   Diagonal sum (correct predictions): {np.sum(np.diag(conf_matrix))}")
    print(f"   Off-diagonal sum (incorrect predictions): {np.sum(conf_matrix) - np.sum(np.diag(conf_matrix))}")


def save_evaluation_results(results, output_path):
    """
    Save evaluation results to JSON file
    
    Args:
        results: Results dictionary
        output_path: Path to save results
    """
    logger.info(f"Saving evaluation results to: {output_path}")
    
    # Create output directory
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("✅ Evaluation results saved")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Evaluate supervised expression transformer model")
    
    # Required arguments
    parser.add_argument("--checkpoint-path", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--dataset-path", type=str, required=True,
                       help="Path to test dataset directory")
    parser.add_argument("--pca-json-path", type=str, required=True,
                       help="Path to PCA projection JSON file")
    
    # Optional arguments
    parser.add_argument("--output-path", type=str, default="evaluation_results.json",
                       help="Path to save evaluation results (default: evaluation_results.json)")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Evaluation batch size (default: 8)")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto/cpu/cuda)")
    parser.add_argument("--num-workers", type=int, default=0,
                       help="Number of data loader workers")
    parser.add_argument("--pin-memory", action="store_true",
                       help="Pin memory for faster GPU transfer")
    parser.add_argument("--class-names", type=str, nargs='+', default=None,
                       help="Class names for better output formatting")
    
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
    if not os.path.exists(args.checkpoint_path):
        print(f"❌ Error: Checkpoint path not found: {args.checkpoint_path}")
        return
    
    if not os.path.exists(args.dataset_path):
        print(f"❌ Error: Dataset path not found: {args.dataset_path}")
        return
    
    if not os.path.exists(args.pca_json_path):
        print(f"❌ Error: PCA JSON path not found: {args.pca_json_path}")
        return
    
    print("🔍 Starting Expression Transformer Supervised Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Test Dataset: {args.dataset_path}")
    print(f"PCA Projection: {args.pca_json_path}")
    print(f"Device: {device}")
    print(f"Batch Size: {args.batch_size}")
    print("=" * 60)
    
    try:
        # Load checkpoint and extract config
        logger.info(f"Loading checkpoint: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        
        # Extract model config
        if 'config' in checkpoint:
            config = checkpoint['config']
            logger.info("✅ Found config in checkpoint")
            
            # Initialize model with checkpoint config
            model = ExpTClassifierModel(
                embed_dim=config.get('embed_dim', 384),
                num_heads=config.get('num_heads', 4),
                num_layers=config.get('num_layers', 2),
                dropout=config.get('dropout', 0.1),
                ff_dim=config.get('ff_dim', 1536),
                grid_size=config.get('grid_size', 37),
                num_classes=config.get('num_classes', 8)
            ).to(device)
            
            num_classes = config.get('num_classes', 8)
        else:
            logger.warning("⚠️  No config found in checkpoint, using default values")
            model = ExpTClassifierModel().to(device)
            num_classes = 8
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("✅ Model loaded successfully")
        
        # Load PCA projection
        pca_components, pca_mean = load_pca_projection(args.pca_json_path)
        
        # Initialize DINOv2 tokenizer
        dinov2_tokenizer = DINOv2BaseTokenizer(device=device)
        logger.info("✅ DINOv2 base tokenizer initialized")
        
        # Load test dataset
        test_dataset = FaceDataset(args.dataset_path)
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            drop_last=False
        )
        
        logger.info(f"Test dataset loaded: {len(test_dataset)} samples")
        
        # Evaluate model
        results = evaluate_model(
            model, test_dataloader, dinov2_tokenizer, 
            pca_components, pca_mean, device, num_classes
        )
        
        # Print results
        print_evaluation_results(results, args.class_names)
        
        # Save results
        save_evaluation_results(results, args.output_path)
        
        print(f"\n🎉 Evaluation completed successfully!")
        print(f"Results saved to: {args.output_path}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
