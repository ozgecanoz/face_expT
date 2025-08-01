#!/usr/bin/env python3
"""
Visualization utilities for training analysis
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def compute_cosine_similarity_distribution(expression_tokens_by_clip: List[torch.Tensor]) -> Dict[str, Any]:
    """
    Compute cosine similarity distribution across expression tokens in clips.
    
    Args:
        expression_tokens_by_clip: List of expression token tensors for each clip
                                 Each tensor has shape (num_tokens, 1, embed_dim)
    
    Returns:
        Dictionary containing similarity statistics and distribution data
    """
    all_similarities = []
    clip_similarities = []
    
    for clip_tokens in expression_tokens_by_clip:
        if clip_tokens.shape[0] < 2:
            continue  # Skip clips with only one token
            
        try:
            # Detach tokens to avoid gradient issues
            tokens_detached = clip_tokens.detach()
            
            # Reshape to (num_tokens, embed_dim)
            tokens = tokens_detached.squeeze(1)  # (num_tokens, embed_dim)
            
            # Compute pairwise cosine similarities
            # Normalize tokens for cosine similarity
            tokens_norm = F.normalize(tokens, p=2, dim=1)
            
            # Compute similarity matrix
            similarity_matrix = torch.mm(tokens_norm, tokens_norm.t())  # (num_tokens, num_tokens)
            
            # Get upper triangle (excluding diagonal) to avoid self-similarities
            upper_triangle = torch.triu(similarity_matrix, diagonal=1)
            
            # Extract non-zero similarities
            similarities = upper_triangle[upper_triangle > 0]
            
            if len(similarities) > 0:
                # Convert to numpy (already detached)
                similarities_np = similarities.cpu().numpy()
                all_similarities.extend(similarities_np)
                clip_similarities.append(similarities_np)
                
        except Exception as e:
            logger.warning(f"Failed to compute similarities for clip: {e}")
            continue
    
    if not all_similarities:
        return {
            'mean_similarity': 0.0,
            'std_similarity': 0.0,
            'min_similarity': 0.0,
            'max_similarity': 0.0,
            'similarities': [],
            'clip_similarities': []
        }
    
    similarities_array = np.array(all_similarities)
    
    return {
        'mean_similarity': float(np.mean(similarities_array)),
        'std_similarity': float(np.std(similarities_array)),
        'min_similarity': float(np.min(similarities_array)),
        'max_similarity': float(np.max(similarities_array)),
        'similarities': similarities_array,
        'clip_similarities': clip_similarities
    }


def plot_cosine_similarity_distribution(
    similarity_data: Dict[str, Any],
    save_path: str = None,
    title: str = "Cosine Similarity Distribution",
    show_plot: bool = False
) -> plt.Figure:
    """
    Plot cosine similarity distribution as a log histogram.
    
    Args:
        similarity_data: Dictionary from compute_cosine_similarity_distribution
        save_path: Path to save the plot (optional)
        title: Plot title
        show_plot: Whether to display the plot
    
    Returns:
        matplotlib Figure object
    """
    similarities = similarity_data['similarities']
    
    if len(similarities) == 0:
        logger.warning("No similarities to plot")
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Log histogram of all similarities
    ax1.hist(similarities, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_yscale('log')
    ax1.set_xlabel('Cosine Similarity')
    ax1.set_ylabel('Frequency (log scale)')
    ax1.set_title(f'{title} - All Tokens')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f'Mean: {similarity_data["mean_similarity"]:.3f}\n'
    stats_text += f'Std: {similarity_data["std_similarity"]:.3f}\n'
    stats_text += f'Min: {similarity_data["min_similarity"]:.3f}\n'
    stats_text += f'Max: {similarity_data["max_similarity"]:.3f}\n'
    stats_text += f'Count: {len(similarities)}'
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 2: Per-clip similarity distributions
    clip_similarities = similarity_data['clip_similarities']
    if len(clip_similarities) > 0:
        # Create violin plot for per-clip distributions
        ax2.violinplot(clip_similarities, showmeans=True, showmedians=True)
        ax2.set_xlabel('Clip Index')
        ax2.set_ylabel('Cosine Similarity')
        ax2.set_title('Per-Clip Similarity Distributions')
        ax2.grid(True, alpha=0.3)
        
        # Add clip statistics
        clip_means = [np.mean(clip_sims) for clip_sims in clip_similarities]
        clip_stds = [np.std(clip_sims) for clip_sims in clip_similarities]
        
        stats_text2 = f'Clips: {len(clip_similarities)}\n'
        stats_text2 += f'Avg Clip Mean: {np.mean(clip_means):.3f}\n'
        stats_text2 += f'Avg Clip Std: {np.mean(clip_stds):.3f}'
        
        ax2.text(0.02, 0.98, stats_text2, transform=ax2.transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved cosine similarity plot to: {save_path}")
    
    if show_plot:
        plt.show()
    
    return fig


def create_training_similarity_plot(
    similarity_history: List[Dict[str, Any]],
    save_path: str = None,
    title: str = "Cosine Similarity Evolution During Training"
) -> plt.Figure:
    """
    Create a plot showing how cosine similarity distribution evolves during training.
    
    Args:
        similarity_history: List of similarity data dictionaries from different training steps
        save_path: Path to save the plot
        title: Plot title
    
    Returns:
        matplotlib Figure object
    """
    if not similarity_history:
        logger.warning("No similarity history to plot")
        return None
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract statistics over time
    steps = list(range(len(similarity_history)))
    means = [data['mean_similarity'] for data in similarity_history]
    stds = [data['std_similarity'] for data in similarity_history]
    mins = [data['min_similarity'] for data in similarity_history]
    maxs = [data['max_similarity'] for data in similarity_history]
    
    # Plot 1: Mean similarity over time
    ax1.plot(steps, means, 'b-', linewidth=2, label='Mean Similarity')
    ax1.fill_between(steps, [m - s for m, s in zip(means, stds)], 
                     [m + s for m, s in zip(means, stds)], alpha=0.3, label='±1 Std')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Cosine Similarity')
    ax1.set_title('Mean Similarity Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Standard deviation over time
    ax2.plot(steps, stds, 'r-', linewidth=2, label='Std Similarity')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Standard Deviation')
    ax2.set_title('Similarity Spread Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Min/Max range over time
    ax3.plot(steps, maxs, 'g-', linewidth=2, label='Max Similarity')
    ax3.plot(steps, mins, 'orange', linewidth=2, label='Min Similarity')
    ax3.fill_between(steps, mins, maxs, alpha=0.3, label='Similarity Range')
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Cosine Similarity')
    ax3.set_title('Similarity Range Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Distribution comparison (first vs last)
    if len(similarity_history) >= 2:
        first_sims = similarity_history[0]['similarities']
        last_sims = similarity_history[-1]['similarities']
        
        ax4.hist(first_sims, bins=30, alpha=0.5, label=f'Step 0 (n={len(first_sims)})', color='blue')
        ax4.hist(last_sims, bins=30, alpha=0.5, label=f'Step {len(similarity_history)-1} (n={len(last_sims)})', color='red')
        ax4.set_xlabel('Cosine Similarity')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution Comparison: Start vs End')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training similarity plot to: {save_path}")
    
    return fig


def test_similarity_visualization():
    """Test the cosine similarity visualization functions."""
    print("🧪 Testing cosine similarity visualization...")
    
    # Create dummy expression tokens
    torch.manual_seed(42)
    num_clips = 5
    expression_tokens_by_clip = []
    
    for i in range(num_clips):
        num_tokens = np.random.randint(3, 8)
        tokens = torch.randn(num_tokens, 1, 384)  # (num_tokens, 1, embed_dim)
        expression_tokens_by_clip.append(tokens)
    
    # Compute similarity distribution
    similarity_data = compute_cosine_similarity_distribution(expression_tokens_by_clip)
    
    print(f"📊 Similarity Statistics:")
    print(f"   Mean: {similarity_data['mean_similarity']:.3f}")
    print(f"   Std: {similarity_data['std_similarity']:.3f}")
    print(f"   Min: {similarity_data['min_similarity']:.3f}")
    print(f"   Max: {similarity_data['max_similarity']:.3f}")
    print(f"   Total similarities: {len(similarity_data['similarities'])}")
    print(f"   Clips: {len(similarity_data['clip_similarities'])}")
    
    # Create and save plot
    fig = plot_cosine_similarity_distribution(
        similarity_data,
        save_path='test_cosine_similarity.png',
        title="Test Cosine Similarity Distribution"
    )
    
    print("✅ Cosine similarity visualization test completed!")
    print("📊 Check 'test_cosine_similarity.png' for the visualization")
    
    return fig


if __name__ == "__main__":
    test_similarity_visualization() 