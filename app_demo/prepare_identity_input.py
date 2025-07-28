#!/usr/bin/env python3
"""
Prepare Identity Input for Webcam Demo
Extracts DINOv2 features from video clips and saves as JSON with subject ID
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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IdentityExtractor:
    """Extract DINOv2 features from video clips"""
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize identity extractor
        
        Args:
            device: Device to run models on
        """
        self.device = device
        
        # Load DINOv2 tokenizer
        logger.info("📦 Loading DINOv2 tokenizer...")
        self.tokenizer = DINOv2Tokenizer()
        logger.info("✅ DINOv2 tokenizer loaded")
    
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
    
    def extract_identity_features(self, face_image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract DINOv2 features from face image
        
        Args:
            face_image: Face image (H, W, 3) in RGB format
            
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
            
            logger.info(f"✅ Extracted DINOv2 features:")
            logger.info(f"   Patch tokens shape: {patch_tokens_np.shape}")
            logger.info(f"   Positional embeddings shape: {pos_embeddings_np.shape}")
            logger.info(f"   Patch tokens stats: mean={patch_tokens_np.mean():.4f}, std={patch_tokens_np.std():.4f}")
            logger.info(f"   Pos embeddings stats: mean={pos_embeddings_np.mean():.4f}, std={pos_embeddings_np.std():.4f}")
            
            return {
                'patch_tokens': patch_tokens_np,
                'pos_embeddings': pos_embeddings_np
            }
            
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
            features = self.extract_identity_features(first_frame)
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
    parser.add_argument("--video_path", type=str, default="identity_video.mp4",
                       help="Path to video file (should contain 518x518 face frames)")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Path to save JSON file with identity features")
    parser.add_argument("--subject_id", type=int, default=1,
                       help="Subject ID for this identity")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to run models on (cpu/cuda)")
    
    args = parser.parse_args()
    
    # Create identity extractor
    extractor = IdentityExtractor(
        device=args.device
    )
    
    # Process video clip
    success = extractor.process_video_clip(args.video_path, args.output_path, subject_id=args.subject_id)
    
    if success:
        logger.info("🎉 Identity feature extraction completed successfully!")
        logger.info(f"📁 Output saved to: {args.output_path}")
        logger.info(f"👤 Subject ID: {args.subject_id}")
        logger.info("💡 You can now use this JSON file with the webcam demo")
    else:
        logger.error("❌ Identity feature extraction failed")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 