#!/usr/bin/env python3
"""
Prepare Identity Input for Webcam Demo
Extracts face ID tokens and DINOv2 features from video clips and saves as JSON
"""

import json
import os
import sys
import argparse
import logging
import numpy as np
import torch
import cv2
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from face_model.models.dinov2_tokenizer import DINOv2Tokenizer
from face_model.models.face_id_model import FaceIDModel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IdentityExtractor:
    """Extract face ID tokens and DINOv2 features from video clips"""
    
    def __init__(self, face_id_checkpoint_path: str, device: str = "cpu"):
        """
        Initialize identity extractor
        
        Args:
            face_id_checkpoint_path: Path to Face ID model checkpoint
            device: Device to run models on
        """
        self.device = device
        self.face_id_checkpoint_path = face_id_checkpoint_path
        
        # Load DINOv2 tokenizer
        logger.info("📦 Loading DINOv2 tokenizer...")
        self.tokenizer = DINOv2Tokenizer()
        logger.info("✅ DINOv2 tokenizer loaded")
        
        # Load Face ID model
        logger.info("👤 Loading Face ID model...")
        self.face_id_model = self._load_face_id_model(face_id_checkpoint_path)
        logger.info("✅ Face ID model loaded")
    
    def _load_face_id_model(self, checkpoint_path: str) -> FaceIDModel:
        """Load Face ID model with architecture from checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Face ID checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint to get architecture
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Get architecture from checkpoint config
        if 'config' in checkpoint and 'face_id_model' in checkpoint['config']:
            face_id_config = checkpoint['config']['face_id_model']
            embed_dim = face_id_config.get('embed_dim', 384)
            num_heads = face_id_config.get('num_heads', 4)
            num_layers = face_id_config.get('num_layers', 2)
            dropout = face_id_config.get('dropout', 0.1)
            logger.info(f"📐 Face ID architecture: {num_layers} layers, {num_heads} heads")
        else:
            # Fallback to default architecture
            embed_dim, num_heads, num_layers, dropout = 384, 4, 2, 0.1
            logger.warning("No architecture config found in Face ID checkpoint, using defaults")
        
        # Initialize model with correct architecture
        face_id_model = FaceIDModel(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        ).to(self.device)
        
        # Store architecture info
        self.face_id_architecture = {
            'embed_dim': embed_dim,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'dropout': dropout
        }
        
        # Load state dict
        if 'face_id_model_state_dict' in checkpoint:
            face_id_model.load_state_dict(checkpoint['face_id_model_state_dict'])
            logger.info(f"Loaded Face ID model from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            # Try loading the entire checkpoint as state dict
            face_id_model.load_state_dict(checkpoint)
            logger.info("Loaded Face ID model state dict directly")
        
        # Set to evaluation mode
        face_id_model.eval()
        
        return face_id_model
    
    def extract_first_frame(self, video_path: str) -> Optional[np.ndarray]:
        """
        Extract first frame from video file
        
        Args:
            video_path: Path to video file
            
        Returns:
            First frame as numpy array (518x518x3) or None if failed
        """
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return None
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                logger.error(f"Could not open video file: {video_path}")
                return None
            
            # Read first frame
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                logger.error(f"Could not read first frame from: {video_path}")
                return None
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Check if frame is already 518x518
            if frame_rgb.shape[:2] != (518, 518):
                logger.warning(f"Frame size is {frame_rgb.shape[:2]}, expected (518, 518). Resizing...")
                frame_rgb = cv2.resize(frame_rgb, (518, 518))
            
            logger.info(f"✅ Extracted first frame: {frame_rgb.shape}")
            return frame_rgb
            
        except Exception as e:
            logger.error(f"Error extracting first frame: {e}")
            return None
    
    def extract_identity_features(self, face_image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract face ID token and DINOv2 features from face image
        
        Args:
            face_image: Face image (518x518x3) in RGB format
            
        Returns:
            Dictionary containing extracted features
        """
        try:
            # Convert to tensor and normalize
            face_tensor = torch.from_numpy(face_image).float() / 255.0  # (518, 518, 3)
            face_tensor = face_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, 518, 518)
            face_tensor = face_tensor.to(self.device)
            
            # Extract DINOv2 tokens
            with torch.no_grad():
                patch_tokens, pos_embeddings = self.tokenizer(face_tensor)
                
                # Extract face ID token
                face_id_token = self.face_id_model.inference(patch_tokens, pos_embeddings)
            
            # Convert to numpy for JSON serialization
            features = {
                'face_id_token': face_id_token.squeeze(0).cpu().numpy(),  # (1, 384)
                'patch_tokens': patch_tokens.squeeze(0).cpu().numpy(),    # (1369, 384)
                'pos_embeddings': pos_embeddings.squeeze(0).cpu().numpy() # (1369, 384)
            }
            
            logger.info("✅ Identity features extracted successfully")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting identity features: {e}")
            return None
    
    def process_video_clip(self, video_path: str, output_path: str) -> bool:
        """
        Process a video clip and save identity features to JSON
        
        Args:
            video_path: Path to video file
            output_path: Path to save JSON file
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"🎬 Processing video clip: {video_path}")
        
        # Extract first frame
        first_frame = self.extract_first_frame(video_path)
        if first_frame is None:
            return False
        
        # Extract identity features
        features = self.extract_identity_features(first_frame)
        if features is None:
            return False
        
        # Save to JSON
        try:
            # Convert numpy arrays to lists for JSON serialization
            json_data = {
                'video_path': video_path,
                'face_id_token': features['face_id_token'].tolist(),
                'patch_tokens': features['patch_tokens'].tolist(),
                'pos_embeddings': features['pos_embeddings'].tolist(),
                'frame_shape': first_frame.shape,
                'extraction_info': {
                    'device': self.device,
                    'face_id_checkpoint_path': self.face_id_checkpoint_path,
                    'face_id_model_architecture': self.face_id_architecture
                }
            }
            
            with open(output_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            logger.info(f"✅ Identity features saved to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving JSON file: {e}")
            return False
#
# python3 ./app_demo/prepare_identity_input.py --face_id_checkpoint ./cloud_checkpoints/face_id_epoch_0.pth --video_path /Users/ozgewhiting/Documents/EQLabs/datasets_serial/CCA_small/1176_14_faces_1_70.mp4 --output_path ./app_demo/CCA_small_1176_14_faces_1_70.json
#
def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Extract identity features from video clips")
    parser.add_argument("--face_id_checkpoint", type=str, required=True,
                       help="Path to Face ID model checkpoint")
    parser.add_argument("--video_path", type=str, default="identity_video.mp4",
                       help="Path to video file (should contain 518x518 face frames)")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Path to save JSON file with identity features")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to run models on (cpu/cuda)")
    
    args = parser.parse_args()
    
    # Create identity extractor
    extractor = IdentityExtractor(
        face_id_checkpoint_path=args.face_id_checkpoint,
        device=args.device
    )
    
    # Process video clip
    success = extractor.process_video_clip(args.video_path, args.output_path)
    
    if success:
        logger.info("🎉 Identity feature extraction completed successfully!")
        logger.info(f"📁 Output saved to: {args.output_path}")
        logger.info("💡 You can now use this JSON file with the webcam demo")
    else:
        logger.error("❌ Identity feature extraction failed")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 