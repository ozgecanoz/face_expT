#!/usr/bin/env python3
"""
Example: Cosine Similarity Analysis During Training
Shows how to create and save cosine similarity distribution plots
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.visualization_utils import (
    compute_cosine_similarity_distribution,
    plot_cosine_similarity_distribution,
    create_training_similarity_plot
)


def example_similarity_analysis():
    """Example of how to use cosine similarity analysis during training."""
    print("📊 Example: Cosine Similarity Analysis During Training")
    
    # Simulate training progression with different similarity distributions
    torch.manual_seed(42)
    
    # Create different stages of training (simulating model improvement)
    training_stages = [
        "Initial (Random)",      # High similarity (poor separation)
        "Early Training",        # Medium similarity
        "Mid Training",          # Lower similarity
        "Late Training",         # Even lower similarity
        "Final (Well-trained)"   # Lowest similarity (good separation)
    ]
    
    similarity_history = []
    
    for stage_idx, stage_name in enumerate(training_stages):
        print(f"\n🔄 Stage {stage_idx + 1}: {stage_name}")
        
        # Create expression tokens with different similarity characteristics
        num_clips = 5
        expression_tokens_by_clip = []
        
        for i in range(num_clips):
            num_tokens = torch.randint(3, 8, (1,)).item()
            
            # Simulate different training stages by controlling token similarity
            if stage_idx == 0:  # Initial - high similarity
                base_token = torch.randn(1, 384)
                tokens = base_token + 0.1 * torch.randn(num_tokens, 1, 384)
            elif stage_idx == 1:  # Early - medium similarity
                base_token = torch.randn(1, 384)
                tokens = base_token + 0.3 * torch.randn(num_tokens, 1, 384)
            elif stage_idx == 2:  # Mid - lower similarity
                tokens = torch.randn(num_tokens, 1, 384)
            elif stage_idx == 3:  # Late - even lower similarity
                tokens = torch.randn(num_tokens, 1, 384) * 1.2
            else:  # Final - lowest similarity (good separation)
                tokens = torch.randn(num_tokens, 1, 384) * 1.5
        
            expression_tokens_by_clip.append(tokens)
        
        # Compute similarity distribution
        similarity_data = compute_cosine_similarity_distribution(expression_tokens_by_clip)
        similarity_history.append(similarity_data)
        
        print(f"   📈 Mean similarity: {similarity_data['mean_similarity']:.3f}")
        print(f"   📊 Std similarity: {similarity_data['std_similarity']:.3f}")
        print(f"   📉 Similarity range: {similarity_data['min_similarity']:.3f} - {similarity_data['max_similarity']:.3f}")
        print(f"   🔢 Total similarities: {len(similarity_data['similarities'])}")
        
        # Save individual stage plot
        plot_path = f'similarity_stage_{stage_idx}_{stage_name.replace(" ", "_").lower()}.png'
        plot_cosine_similarity_distribution(
            similarity_data,
            save_path=plot_path,
            title=f"Cosine Similarity - {stage_name}"
        )
        print(f"   💾 Saved: {plot_path}")
    
    # Create training evolution plot
    print(f"\n📈 Creating training evolution plot...")
    evolution_plot_path = 'similarity_training_evolution.png'
    create_training_similarity_plot(
        similarity_history,
        save_path=evolution_plot_path,
        title="Cosine Similarity Evolution During Training"
    )
    print(f"💾 Saved training evolution: {evolution_plot_path}")
    
    # Summary
    print(f"\n🎯 Summary:")
    print(f"   📊 Created {len(training_stages)} training stage plots")
    print(f"   📈 Created 1 training evolution plot")
    print(f"   📉 Similarity decreased from {similarity_history[0]['mean_similarity']:.3f} to {similarity_history[-1]['mean_similarity']:.3f}")
    print(f"   📊 Standard deviation increased from {similarity_history[0]['std_similarity']:.3f} to {similarity_history[-1]['std_similarity']:.3f}")
    
    print(f"\n✅ Example completed! Check the generated PNG files for visualizations.")
    
    return similarity_history


if __name__ == "__main__":
    example_similarity_analysis() 