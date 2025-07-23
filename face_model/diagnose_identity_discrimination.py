#!/usr/bin/env python3
"""
Diagnose identity discrimination issues in Face ID Model
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
import logging
from tqdm import tqdm
import sys
sys.path.append('.')

from evaluate_face_id import FaceIDEvaluator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def diagnose_identity_discrimination(checkpoint_path, data_dir, max_samples=50):
    """Detailed diagnosis of identity discrimination"""
    
    print("🔍 Diagnosing Identity Discrimination Issues")
    print("=" * 60)
    
    # Create evaluator
    evaluator = FaceIDEvaluator(checkpoint_path, data_dir)
    
    # Extract identity tokens
    print("📊 Extracting identity tokens...")
    identity_tokens, subject_ids, frame_consistencies = evaluator.extract_identity_tokens(max_samples)
    
    print(f"📈 Extracted {len(identity_tokens)} identity tokens")
    print(f"👥 Unique subjects: {len(set(subject_ids))}")
    
    # Calculate detailed similarity analysis
    print("\n🔍 Calculating similarity matrices...")
    similarities = cosine_similarity(identity_tokens)
    
    # Separate within-subject and between-subject similarities
    within_subject_similarities = []
    between_subject_similarities = []
    within_pairs = []
    between_pairs = []
    
    for i in range(len(identity_tokens)):
        for j in range(i+1, len(identity_tokens)):
            similarity = similarities[i, j]
            
            if subject_ids[i] == subject_ids[j]:
                within_subject_similarities.append(similarity)
                within_pairs.append((i, j, subject_ids[i]))
            else:
                between_subject_similarities.append(similarity)
                between_pairs.append((i, j, subject_ids[i], subject_ids[j]))
    
    # Calculate statistics
    within_mean = np.mean(within_subject_similarities)
    within_std = np.std(within_subject_similarities)
    between_mean = np.mean(between_subject_similarities)
    between_std = np.std(between_subject_similarities)
    discrimination_score = between_mean - within_mean
    
    print("\n📊 Detailed Similarity Analysis:")
    print("=" * 60)
    print(f"Within-subject similarities:")
    print(f"  Mean: {within_mean:.6f}")
    print(f"  Std:  {within_std:.6f}")
    print(f"  Min:  {np.min(within_subject_similarities):.6f}")
    print(f"  Max:  {np.max(within_subject_similarities):.6f}")
    print(f"  Count: {len(within_subject_similarities)}")
    
    print(f"\nBetween-subject similarities:")
    print(f"  Mean: {between_mean:.6f}")
    print(f"  Std:  {between_std:.6f}")
    print(f"  Min:  {np.min(between_subject_similarities):.6f}")
    print(f"  Max:  {np.max(between_subject_similarities):.6f}")
    print(f"  Count: {len(between_subject_similarities)}")
    
    print(f"\nDiscrimination Score: {discrimination_score:.6f}")
    
    # Check if the model is producing identical outputs
    print(f"\n🔍 Checking for identical outputs...")
    unique_tokens = np.unique(identity_tokens, axis=0)
    print(f"Unique identity tokens: {len(unique_tokens)} out of {len(identity_tokens)}")
    
    if len(unique_tokens) == 1:
        print("🚨 CRITICAL ISSUE: All identity tokens are identical!")
        print("   This means the model is outputting the same token for all inputs.")
    elif len(unique_tokens) < len(identity_tokens) * 0.1:
        print("⚠️  WARNING: Very few unique identity tokens detected")
        print("   The model may be producing very similar outputs for different inputs.")
    
    # Check token variance
    token_variance = np.var(identity_tokens, axis=0)
    print(f"\n📊 Token Variance Analysis:")
    print(f"  Mean variance across dimensions: {np.mean(token_variance):.6f}")
    print(f"  Max variance: {np.max(token_variance):.6f}")
    print(f"  Min variance: {np.min(token_variance):.6f}")
    
    if np.mean(token_variance) < 1e-6:
        print("🚨 CRITICAL ISSUE: Very low token variance detected!")
        print("   The model is producing nearly identical tokens for all inputs.")
    
    # Create detailed visualizations
    create_diagnostic_plots(identity_tokens, subject_ids, similarities, 
                           within_subject_similarities, between_subject_similarities)
    
    return {
        'within_mean': within_mean,
        'between_mean': between_mean,
        'discrimination_score': discrimination_score,
        'unique_tokens': len(unique_tokens),
        'token_variance': np.mean(token_variance)
    }

def create_diagnostic_plots(identity_tokens, subject_ids, similarities, 
                           within_similarities, between_similarities):
    """Create diagnostic plots"""
    
    output_dir = "./diagnostic_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Similarity distribution comparison
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(within_similarities, bins=30, alpha=0.7, label='Within-subject', color='blue')
    plt.hist(between_similarities, bins=30, alpha=0.7, label='Between-subject', color='red')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Count')
    plt.title('Similarity Distribution Comparison')
    plt.legend()
    
    # 2. Similarity matrix heatmap
    plt.subplot(2, 2, 2)
    sns.heatmap(similarities[:20, :20], cmap='viridis', cbar=True)
    plt.title('Similarity Matrix (First 20 samples)')
    
    # 3. Token variance across dimensions
    plt.subplot(2, 2, 3)
    token_variance = np.var(identity_tokens, axis=0)
    plt.plot(token_variance)
    plt.xlabel('Token Dimension')
    plt.ylabel('Variance')
    plt.title('Token Variance Across Dimensions')
    
    # 4. Box plot of similarities
    plt.subplot(2, 2, 4)
    data_to_plot = [within_similarities, between_similarities]
    plt.boxplot(data_to_plot, labels=['Within-subject', 'Between-subject'])
    plt.ylabel('Cosine Similarity')
    plt.title('Similarity Distribution Box Plot')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'diagnostic_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Diagnostic plots saved to {output_dir}/")

def main():
    """Main diagnostic function"""
    
    checkpoint_path = "/Users/ozgewhiting/Documents/projects/dataset_utils/checkpoints/face_id_epoch_3.pth"
    data_dir = "/Users/ozgewhiting/Documents/EQLabs/datasets_serial/CCA_test_db1"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    # Run diagnosis
    results = diagnose_identity_discrimination(checkpoint_path, data_dir, max_samples=50)
    
    print("\n" + "="*60)
    print("DIAGNOSIS SUMMARY")
    print("="*60)
    print(f"Discrimination Score: {results['discrimination_score']:.6f}")
    print(f"Unique Tokens: {results['unique_tokens']}")
    print(f"Mean Token Variance: {results['token_variance']:.6f}")
    
    if results['discrimination_score'] < 0.01:
        print("\n🚨 ISSUE DETECTED: Poor identity discrimination")
        print("   Recommendations:")
        print("   1. Train the model for more epochs")
        print("   2. Increase model complexity (more layers/heads)")
        print("   3. Add identity classification loss")
        print("   4. Check training data quality")
    
    print("="*60)

if __name__ == "__main__":
    main() 