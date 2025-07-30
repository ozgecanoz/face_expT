#!/usr/bin/env python3
"""
Evaluate Face ID Model with t-SNE visualization and identity consistency analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import h5py
import os
import json
import logging
from tqdm import tqdm
import sys
sys.path.append('.')

from models.dinov2_tokenizer import DINOv2Tokenizer
from models.face_id_model import FaceIDModel
from data.dataset import create_face_dataloader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FaceIDEvaluator:
    """Evaluator for Face ID Model"""
    
    def __init__(self, checkpoint_path, data_dir):
        self.checkpoint_path = checkpoint_path
        self.data_dir = data_dir
        
        # Load models
        self.dinov2_tokenizer = DINOv2Tokenizer(device=device)
        self.face_id_model = self._load_model()
        
        # Load dataset metadata
        self.dataset_metadata = self._load_dataset_metadata()
        
        logger.info("Face ID Evaluator initialized")
    
    def _load_model(self):
        """Load trained Face ID Model from checkpoint"""
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        
        # Get model config from checkpoint
        config = checkpoint['config']
        model_config = config['face_id_model']
        
        # Create model with same config as training
        model = FaceIDModel(
            embed_dim=model_config['embed_dim'],
            num_heads=model_config['num_heads'],
            num_layers=model_config['num_layers'],
            dropout=model_config['dropout']
        )
        
        # Load trained weights
        model.load_state_dict(checkpoint['face_id_model_state_dict'])
        model.eval()
        
        logger.info(f"Loaded model from {self.checkpoint_path}")
        logger.info(f"Model config: {model_config}")
        return model
    
    def _load_dataset_metadata(self):
        """Load dataset metadata"""
        metadata_file = os.path.join(self.data_dir, "dataset_metadata.json")
        
        if not os.path.exists(metadata_file):
            logger.warning(f"Dataset metadata not found: {metadata_file}")
            return None
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Loaded evaluation dataset:")
        logger.info(f"  Total subjects: {metadata['dataset_stats']['num_subjects']}")
        logger.info(f"  Total clips: {metadata['total_clips_extracted']}")
        
        return metadata
    
    def extract_identity_tokens(self, max_samples=None):
        """Extract identity tokens from evaluation dataset"""
        logger.info("Extracting identity tokens...")
        
        # Create dataloader
        dataloader = create_face_dataloader(
            data_dir=self.data_dir,
            batch_size=1,  # Process one sample at a time
            max_samples=max_samples
        )
        
        identity_tokens = []
        subject_ids = []
        frame_consistencies = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting tokens")):
                frames = batch['frames']  # (1, 30, 3, 518, 518)
                subject_id = batch['subject_id'][0]
                
                # Extract identity token for each frame
                frame_tokens = []
                num_frames = frames.shape[1]  # Get actual number of frames
                
                for frame_idx in range(num_frames):
                    frame = frames[:, frame_idx, :, :, :]  # (1, 3, 518, 518)
                    
                    # Get DINOv2 tokens
                    patch_tokens, pos_emb = self.dinov2_tokenizer(frame)
                    
                    # Get identity token
                    identity_token = self.face_id_model(patch_tokens, pos_emb)  # (1, 1, 384)
                    frame_tokens.append(identity_token.squeeze(0).squeeze(0))  # (384,)
                
                # Stack all frame tokens
                frame_tokens = torch.stack(frame_tokens, dim=0)  # (num_frames, 384)
                
                # Calculate consistency across frames
                consistency = torch.var(frame_tokens, dim=0).mean().item()
                
                # Store results
                identity_tokens.append(frame_tokens.mean(dim=0).numpy())  # Average across frames
                subject_ids.append(subject_id)
                frame_consistencies.append(consistency)
        
        return np.array(identity_tokens), subject_ids, frame_consistencies
    
    def visualize_tsne(self, identity_tokens, subject_ids, output_dir="./evaluation_results"):
        """Visualize identity tokens with t-SNE"""
        logger.info("Creating t-SNE visualization...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(identity_tokens)-1))
        tokens_2d = tsne.fit_transform(identity_tokens)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Color by subject ID
        unique_subjects = list(set(subject_ids))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_subjects)))
        
        for i, subject_id in enumerate(unique_subjects):
            mask = [s == subject_id for s in subject_ids]
            plt.scatter(tokens_2d[mask, 0], tokens_2d[mask, 1], 
                       c=[colors[i]], label=f'Subject {subject_id}', alpha=0.7, s=50)
        
        plt.title('Face ID Model: Identity Token Clustering (t-SNE)', fontsize=16)
        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, "identity_tokens_tsne.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"t-SNE visualization saved to {plot_path}")
        
        return tokens_2d
    
    def analyze_identity_consistency(self, identity_tokens, subject_ids, frame_consistencies, output_dir="./evaluation_results"):
        """Analyze identity consistency across subjects"""
        logger.info("Analyzing identity consistency...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate consistency statistics
        consistency_stats = {}
        for subject_id in set(subject_ids):
            mask = [s == subject_id for s in subject_ids]
            subject_consistencies = [frame_consistencies[i] for i, m in enumerate(mask) if m]
            consistency_stats[subject_id] = {
                'mean_consistency': np.mean(subject_consistencies),
                'std_consistency': np.std(subject_consistencies),
                'num_clips': len(subject_consistencies)
            }
        
        # Create consistency histogram
        plt.figure(figsize=(10, 6))
        plt.hist(frame_consistencies, bins=30, alpha=0.7, edgecolor='black')
        plt.title('Identity Consistency Distribution', fontsize=16)
        plt.xlabel('Frame-to-Frame Variance (Lower = More Consistent)', fontsize=12)
        plt.ylabel('Number of Clips', fontsize=12)
        plt.axvline(np.mean(frame_consistencies), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(frame_consistencies):.4f}')
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, "identity_consistency_histogram.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create consistency per sample plot
        self._plot_consistency_per_sample(frame_consistencies, subject_ids, output_dir)
        
        # Save consistency statistics
        stats_path = os.path.join(output_dir, "consistency_statistics.json")
        with open(stats_path, 'w') as f:
            json.dump(consistency_stats, f, indent=2)
        
        logger.info(f"Consistency analysis saved to {output_dir}")
        
        return consistency_stats
    
    def _plot_consistency_per_sample(self, frame_consistencies, subject_ids, output_dir):
        """Plot consistency values per sample across all test samples"""
        logger.info("Creating consistency per sample plot...")
        
        # Create the plot
        plt.figure(figsize=(15, 8))
        
        # Create sample indices
        sample_indices = list(range(len(frame_consistencies)))
        
        # Color points by subject ID
        unique_subjects = list(set(subject_ids))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_subjects)))
        
        # Plot each sample
        for i, subject_id in enumerate(unique_subjects):
            mask = [s == subject_id for s in subject_ids]
            subject_indices = [j for j, m in enumerate(mask) if m]
            subject_consistencies = [frame_consistencies[j] for j in subject_indices]
            
            plt.scatter(subject_indices, subject_consistencies, 
                       c=[colors[i]], label=f'Subject {subject_id}', alpha=0.7, s=50)
        
        # Add mean line
        mean_consistency = np.mean(frame_consistencies)
        plt.axhline(y=mean_consistency, color='red', linestyle='--', 
                   label=f'Mean: {mean_consistency:.4f}', linewidth=2)
        
        # Add standard deviation bands
        std_consistency = np.std(frame_consistencies)
        plt.axhline(y=mean_consistency + std_consistency, color='orange', linestyle=':', 
                   label=f'+1σ: {mean_consistency + std_consistency:.4f}', alpha=0.7)
        plt.axhline(y=mean_consistency - std_consistency, color='orange', linestyle=':', 
                   label=f'-1σ: {mean_consistency - std_consistency:.4f}', alpha=0.7)
        
        # Customize plot
        plt.title('Identity Consistency per Sample', fontsize=16)
        plt.xlabel('Sample ID', fontsize=12)
        plt.ylabel('Frame-to-Frame Variance (Lower = More Consistent)', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, "consistency_per_sample.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Consistency per sample plot saved to {plot_path}")
        
        # Also create a box plot by subject
        self._plot_consistency_boxplot(frame_consistencies, subject_ids, output_dir)
    
    def _plot_consistency_boxplot(self, frame_consistencies, subject_ids, output_dir):
        """Create box plot of consistency by subject"""
        logger.info("Creating consistency box plot by subject...")
        
        # Group consistency values by subject
        subject_consistencies = {}
        for i, subject_id in enumerate(subject_ids):
            if subject_id not in subject_consistencies:
                subject_consistencies[subject_id] = []
            subject_consistencies[subject_id].append(frame_consistencies[i])
        
        # Create box plot
        plt.figure(figsize=(12, 6))
        
        # Prepare data for box plot
        labels = list(subject_consistencies.keys())
        data = list(subject_consistencies.values())
        
        # Create box plot
        bp = plt.boxplot(data, labels=labels, patch_artist=True)
        
        # Color boxes
        colors = plt.cm.tab20(np.linspace(0, 1, len(labels)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Customize plot
        plt.title('Identity Consistency Distribution by Subject', fontsize=16)
        plt.xlabel('Subject ID', fontsize=12)
        plt.ylabel('Frame-to-Frame Variance (Lower = More Consistent)', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, "consistency_boxplot_by_subject.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Consistency box plot saved to {plot_path}")
    
    def calculate_identity_discrimination(self, identity_tokens, subject_ids):
        """Calculate how well the model discriminates between different subjects"""
        logger.info("Calculating identity discrimination...")
        
        # Calculate cosine similarities
        similarities = cosine_similarity(identity_tokens)
        
        # Separate within-subject and between-subject similarities
        within_subject_similarities = []
        between_subject_similarities = []
        
        for i in range(len(identity_tokens)):
            for j in range(i+1, len(identity_tokens)):
                similarity = similarities[i, j]
                
                if subject_ids[i] == subject_ids[j]:
                    within_subject_similarities.append(similarity)
                else:
                    between_subject_similarities.append(similarity)
        
        # Calculate discrimination metrics
        within_mean = np.mean(within_subject_similarities)
        between_mean = np.mean(between_subject_similarities)
        discrimination_score = between_mean - within_mean
        
        logger.info(f"Identity Discrimination Analysis:")
        logger.info(f"  Within-subject similarity: {within_mean:.4f}")
        logger.info(f"  Between-subject similarity: {between_mean:.4f}")
        logger.info(f"  Discrimination score: {discrimination_score:.4f}")
        
        return {
            'within_subject_similarity': within_mean,
            'between_subject_similarity': between_mean,
            'discrimination_score': discrimination_score
        }
    
    def run_evaluation(self, max_samples=None):
        """Run complete evaluation"""
        logger.info("Starting Face ID Model evaluation...")
        
        # Extract identity tokens
        identity_tokens, subject_ids, frame_consistencies = self.extract_identity_tokens(max_samples)
        
        # Create visualizations
        tokens_2d = self.visualize_tsne(identity_tokens, subject_ids)
        
        # Analyze consistency
        consistency_stats = self.analyze_identity_consistency(identity_tokens, subject_ids, frame_consistencies)
        
        # Calculate discrimination
        discrimination_metrics = self.calculate_identity_discrimination(identity_tokens, subject_ids)
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*50)
        logger.info(f"Total samples evaluated: {len(identity_tokens)}")
        logger.info(f"Unique subjects: {len(set(subject_ids))}")
        logger.info(f"Average frame consistency: {np.mean(frame_consistencies):.4f}")
        logger.info(f"Identity discrimination score: {discrimination_metrics['discrimination_score']:.4f}")
        logger.info("="*50)
        
        return {
            'identity_tokens': identity_tokens,
            'subject_ids': subject_ids,
            'frame_consistencies': frame_consistencies,
            'tokens_2d': tokens_2d,
            'consistency_stats': consistency_stats,
            'discrimination_metrics': discrimination_metrics
        }


def main():
    """Run Face ID Model evaluation"""
    
    # Configuration
    checkpoint_path = "/Users/ozgewhiting/Documents/projects/dataset_utils/checkpoints/face_id_epoch_3.pth"  # Update with your checkpoint
    data_dir = "/Users/ozgewhiting/Documents/EQLabs/datasets_serial/CCA_test_db1"  # Update with your evaluation dataset
    
    print("🔍 Face ID Model Evaluation")
    print(f"📁 Checkpoint: {checkpoint_path}")
    print(f"📊 Dataset: {data_dir}")
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Please update the checkpoint_path in the script or train the model first.")
        return
    
    # Create evaluator
    evaluator = FaceIDEvaluator(checkpoint_path, data_dir)
    
    # Run evaluation
    results = evaluator.run_evaluation(max_samples=100)  # Limit to 100 samples for testing
    
    print("\n✅ Evaluation completed!")
    print("📊 Check evaluation_results/ folder for visualizations and statistics")


if __name__ == "__main__":
    main() 