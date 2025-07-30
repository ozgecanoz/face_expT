#!/usr/bin/env python3
"""
Prepare Identity Input for Webcam Demo
Extracts DINOv2 features from video clips and saves as JSON with subject ID
"""

import cv2
import numpy as np
import json
import logging
import argparse
from typing import Dict, Optional
import torch
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from face_model.models.dinov2_tokenizer import DINOv2Tokenizer
from face_model.models.expression_transformer import ExpressionTransformer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IdentityExtractor:
    """Extract identity features from video clips"""
    
    def __init__(self, device: str = "cpu", expression_transformer_checkpoint_path: str = None):
        """
        Initialize identity extractor
        
        Args:
            device: Device to use for computation
            expression_transformer_checkpoint_path: Path to Expression Transformer checkpoint for subject embeddings
        """
        self.device = device
        
        # Initialize DINOv2 tokenizer
        logger.info("📦 Loading DINOv2 tokenizer...")
        self.tokenizer = DINOv2Tokenizer(device=device)
        logger.info("✅ DINOv2 tokenizer loaded")
        
        # Initialize Expression Transformer for subject embeddings
        if expression_transformer_checkpoint_path:
            logger.info("😊 Loading Expression Transformer for subject embeddings...")
            self.expression_transformer = self._load_expression_transformer(expression_transformer_checkpoint_path)
            logger.info("✅ Expression Transformer loaded")
        else:
            logger.warning("⚠️ No Expression Transformer checkpoint provided - subject embeddings will not be extracted")
            self.expression_transformer = None
    
    def _load_expression_transformer(self, checkpoint_path: str) -> ExpressionTransformer:
        """Load Expression Transformer with architecture from checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Expression Transformer checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint to get architecture
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Get architecture from checkpoint config
        if 'config' in checkpoint and 'expression_model' in checkpoint['config']:
            expr_config = checkpoint['config']['expression_model']
            embed_dim = expr_config.get('embed_dim', 384)
            num_heads = expr_config.get('num_heads', 4)
            num_layers = expr_config.get('num_layers', 2)
            dropout = expr_config.get('dropout', 0.1)
            max_subjects = expr_config.get('max_subjects', 3500)
            logger.info(f"📐 Expression Transformer architecture: {num_layers} layers, {num_heads} heads, {max_subjects} subjects")
        else:
            # Fallback to default architecture
            embed_dim, num_heads, num_layers, dropout, max_subjects = 384, 4, 2, 0.1, 3500
            logger.warning("No architecture config found in Expression Transformer checkpoint, using defaults")
        
        # Initialize model with correct architecture
        expression_transformer = ExpressionTransformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            max_subjects=max_subjects
        ).to(self.device)
        
        # Load state dict
        if 'expression_transformer_state_dict' in checkpoint:
            expression_transformer.load_state_dict(checkpoint['expression_transformer_state_dict'])
            logger.info(f"Loaded Expression Transformer from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            # Try loading the entire checkpoint as state dict
            expression_transformer.load_state_dict(checkpoint)
            logger.info("Loaded Expression Transformer state dict directly")
        
        # Set to evaluation mode
        expression_transformer.eval()
        
        return expression_transformer
    
    def extract_first_frame(self, video_path: str) -> Optional[np.ndarray]:
        """
        Extract the first frame from a video file
        
        Args:
            video_path: Path to video file
            
        Returns:
            First frame as numpy array or None if failed
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return None
            
            # Read first frame
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                logger.error(f"Failed to read first frame from: {video_path}")
                return None
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            logger.info(f"✅ Extracted first frame from: {video_path}")
            logger.info(f"   Frame shape: {frame_rgb.shape}")
            
            return frame_rgb
            
        except Exception as e:
            logger.error(f"Error extracting first frame from {video_path}: {e}")
            return None
    
    def extract_identity_features(self, face_image: np.ndarray, subject_id: int) -> Dict[str, np.ndarray]:
        """
        Extract DINOv2 features and subject embeddings from face image
        
        Args:
            face_image: Face image (H, W, 3) in RGB format
            subject_id: Subject ID for this identity
            
        Returns:
            Dictionary containing extracted features
        """
        try:
            # Convert numpy array to torch tensor
            face_tensor = torch.from_numpy(face_image).float().to(self.device)
            
            # Normalize to [0, 1] if needed
            if face_tensor.max() > 1.0:
                face_tensor = face_tensor / 255.0
            
            # Ensure correct shape: (H, W, C) -> (C, H, W)
            if face_tensor.dim() == 3:
                if face_tensor.shape[0] == 3:  # (C, H, W) format
                    face_tensor = face_tensor.permute(1, 2, 0)  # (H, W, C)
                face_tensor = face_tensor.permute(2, 0, 1)  # (C, H, W)
            
            # Add batch dimension
            face_tensor = face_tensor.unsqueeze(0)  # (1, 3, 518, 518)
            
            # Extract DINOv2 tokens
            with torch.no_grad():
                patch_tokens, pos_embeddings = self.tokenizer(face_tensor)
            
            # Convert to numpy for JSON serialization
            patch_tokens_np = patch_tokens.squeeze(0).cpu().numpy()  # (1369, 384)
            pos_embeddings_np = pos_embeddings.squeeze(0).cpu().numpy()  # (1369, 384)
            
            # Extract subject embeddings if Expression Transformer is available
            subject_embeddings_np = None
            if self.expression_transformer is not None:
                # Create subject IDs tensor
                subject_ids = torch.tensor([subject_id], dtype=torch.long, device=self.device)
                
                # Extract subject embeddings using inference
                with torch.no_grad():
                    _, subject_embeddings = self.expression_transformer.inference(patch_tokens, pos_embeddings, subject_ids)
                
                # Convert to numpy
                subject_embeddings_np = subject_embeddings.squeeze(0).cpu().numpy()  # (384,)
                
                logger.info(f"✅ Extracted subject embeddings:")
                logger.info(f"   Subject embeddings shape: {subject_embeddings_np.shape}")
                logger.info(f"   Subject embeddings stats: mean={subject_embeddings_np.mean():.4f}, std={subject_embeddings_np.std():.4f}")
            
            logger.info(f"✅ Extracted DINOv2 features:")
            logger.info(f"   Patch tokens shape: {patch_tokens_np.shape}")
            logger.info(f"   Positional embeddings shape: {pos_embeddings_np.shape}")
            logger.info(f"   Patch tokens stats: mean={patch_tokens_np.mean():.4f}, std={patch_tokens_np.std():.4f}")
            logger.info(f"   Pos embeddings stats: mean={pos_embeddings_np.mean():.4f}, std={pos_embeddings_np.std():.4f}")
            
            features = {
                'patch_tokens': patch_tokens_np,
                'pos_embeddings': pos_embeddings_np
            }
            
            # Add subject embeddings if available
            if subject_embeddings_np is not None:
                features['subject_embeddings'] = subject_embeddings_np
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting identity features: {e}")
            return {}
    
    def process_video_clip(self, video_path: str, output_path: str, subject_id: int = 1) -> bool:
        """
        Process a video clip and extract identity features
        
        Args:
            video_path: Path to video file
            output_path: Path to save JSON file
            subject_id: Subject ID for this identity
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"🎬 Processing video: {video_path}")
            
            # Extract first frame
            first_frame = self.extract_first_frame(video_path)
            if first_frame is None:
                return False
            
            # Extract identity features
            features = self.extract_identity_features(first_frame, subject_id)
            if not features:
                return False
            
            # Create output data
            output_data = {
                'video_path': video_path,
                'subject_id': subject_id,
                'patch_tokens': features['patch_tokens'].tolist(),
                'pos_embeddings': features['pos_embeddings'].tolist(),
                'extraction_info': {
                    'frame_shape': first_frame.shape,
                    'patch_tokens_shape': features['patch_tokens'].shape,
                    'pos_embeddings_shape': features['pos_embeddings'].shape
                }
            }
            
            # Add subject embeddings to output data if available
            if 'subject_embeddings' in features:
                output_data['subject_embeddings'] = features['subject_embeddings'].tolist()
            
            # Save to JSON file
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            logger.info(f"✅ Identity features extracted successfully")
            logger.info(f"   Saved to: {output_path}")
            logger.info(f"   Subject ID: {subject_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Identity feature extraction failed: {e}")
            return False
#
# python3 ./app_demo/prepare_identity_input.py --face_id_checkpoint ./cloud_checkpoints/face_id_epoch_0.pth --video_path /Users/ozgewhiting/Documents/EQLabs/datasets_serial/CCA_small/1176_14_faces_1_70.mp4 --output_path ./app_demo/CCA_small_1176_14_faces_1_70.json
#
def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Extract identity features from video clips")
    parser.add_argument("--video_path", type=str, required=True, help="Path to video file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save JSON file")
    parser.add_argument("--subject_id", type=int, default=1, help="Subject ID for this identity")
    parser.add_argument("--expression_transformer_checkpoint", type=str, default=None, 
                       help="Path to Expression Transformer checkpoint for subject embeddings")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    
    args = parser.parse_args()
    
    # Create identity extractor
    extractor = IdentityExtractor(
        device=args.device,
        expression_transformer_checkpoint_path=args.expression_transformer_checkpoint
    )
    
    # Process video clip
    success = extractor.process_video_clip(args.video_path, args.output_path, subject_id=args.subject_id)
    
    if success:
        logger.info("✅ Identity feature extraction completed successfully!")
        return True
    else:
        logger.error("❌ Identity feature extraction failed!")
        return False


if __name__ == "__main__":
    exit(main()) 