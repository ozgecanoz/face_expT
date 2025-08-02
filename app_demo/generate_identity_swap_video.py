#!/usr/bin/env python3
"""
Generate Identity Swap Video
Creates a video where expressions from input video are mapped to a different subject's identity
Uses pre-prepared identity features from JSON file
"""

import cv2
import numpy as np
import torch
import logging
import argparse
import os
import sys
import json
from typing import Optional, Dict, Any
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_loader import ModelLoader
from token_extractor import TokenExtractor

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IdentitySwapVideoGenerator:
    """Generate identity swap videos using pre-prepared identity features"""
    
    def __init__(self, 
                 expression_transformer_checkpoint_path: str,
                 face_reconstruction_checkpoint_path: str,
                 device: str = "cpu"):
        """
        Initialize identity swap video generator
        
        Args:
            expression_transformer_checkpoint_path: Path to Expression Transformer checkpoint
            face_reconstruction_checkpoint_path: Path to Face Reconstruction model checkpoint
            device: Device to run models on
        """
        self.device = device
        
        # Load models
        logger.info("🚀 Loading models...")
        model_loader = ModelLoader(device=device)
        self.models = model_loader.load_all_models(
            expression_transformer_checkpoint_path=expression_transformer_checkpoint_path,
            expression_predictor_checkpoint_path="dummy",  # Not needed for reconstruction
            face_reconstruction_checkpoint_path=face_reconstruction_checkpoint_path
        )
        
        # Initialize components
        self.tokenizer = self.models['tokenizer']
        self.token_extractor = TokenExtractor(
            expression_transformer=self.models['expression_transformer'],
            expression_predictor=self.models.get('expression_predictor'),  # May be None
            face_reconstruction_model=self.models.get('face_reconstruction_model'),  # May be None
            tokenizer=self.models['tokenizer'],
            device=self.device,
            subject_id=0  # Will be set per frame
        )
        
        # Store first frame expression token for cosine similarity calculation
        self.first_frame_expression_token = None
        
        logger.info("✅ Identity Swap Video Generator initialized successfully!")
    
    def _calculate_cosine_similarity(self, token1: torch.Tensor, token2: torch.Tensor) -> float:
        """
        Calculate cosine similarity between two expression tokens
        
        Args:
            token1: First expression token (1, 1, 384)
            token2: Second expression token (1, 1, 384)
            
        Returns:
            Cosine similarity value
        """
        # Flatten tokens to 1D vectors
        vec1 = token1.flatten()
        vec2 = token2.flatten()
        
        # Calculate cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0), dim=1)
        
        return cos_sim.item()
    
    def load_identity_features(self, identity_json_path: str) -> Dict[str, Any]:
        """
        Load identity features from JSON file
        
        Args:
            identity_json_path: Path to JSON file with identity features
            
        Returns:
            Dictionary containing identity features
        """
        if not os.path.exists(identity_json_path):
            raise FileNotFoundError(f"Identity features file not found: {identity_json_path}")
        
        with open(identity_json_path, 'r') as f:
            identity_features = json.load(f)
        
        # Convert numpy arrays back to tensors
        for key in ['patch_tokens', 'pos_embeddings', 'subject_embeddings']:
            if key in identity_features:
                identity_features[key] = torch.tensor(identity_features[key], dtype=torch.float32)
        
        logger.info(f"✅ Loaded identity features from: {identity_json_path}")
        logger.info(f"   Subject ID: {identity_features.get('subject_id', 'unknown')}")
        logger.info(f"   Available features: {list(identity_features.keys())}")
        
        return identity_features
    
    def process_video(self, 
                     input_video_path: str, 
                     output_video_path: str, 
                     identity_json_path: str,
                     input_subject_id: int = 0,
                     max_frames: Optional[int] = None) -> bool:
        """
        Process video and generate identity swap video
        
        Args:
            input_video_path: Path to input video with face frames
            output_video_path: Path to save output video
            identity_json_path: Path to JSON file with target identity features
            input_subject_id: Subject ID for the input video (for expression extraction)
            max_frames: Maximum number of frames to process (for testing)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load identity features
            identity_features = self.load_identity_features(identity_json_path)
            
            # Open input video
            cap = cv2.VideoCapture(input_video_path)
            if not cap.isOpened():
                logger.error(f"Could not open input video: {input_video_path}")
                return False
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"📹 Input video properties:")
            logger.info(f"   FPS: {fps}")
            logger.info(f"   Total frames: {total_frames}")
            logger.info(f"   Resolution: {width}x{height}")
            logger.info(f"   Input Subject ID: {input_subject_id}")
            logger.info(f"   Target Subject ID: {identity_features.get('subject_id', 'unknown')}")
            
            # Verify frame size is 518x518
            if width != 518 or height != 518:
                logger.warning(f"⚠️ Expected 518x518 frames, got {width}x{height}")
                logger.warning("   This script is designed for pre-cropped face frames")
            
            # Limit frames if specified
            if max_frames:
                total_frames = min(total_frames, max_frames)
                logger.info(f"   Processing first {total_frames} frames")
            
            # Set up output video writer
            # Output will be side-by-side, so double the width
            output_width = 518 * 2  # Side-by-side comparison
            output_height = 518
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))
            
            if not out.isOpened():
                logger.error(f"Could not create output video: {output_video_path}")
                cap.release()
                return False
            
            # Process frames
            processed_frames = 0
            successful_frames = 0
            
            logger.info("🎬 Processing video frames...")
            progress_bar = tqdm(total=total_frames, desc="Processing frames")
            
            for frame_idx in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame (frame is already 518x518 face crop)
                result = self._process_identity_swap_frame(frame, input_subject_id, identity_features, frame_idx)
                
                if result is not None:
                    # Create side-by-side comparison
                    comparison_frame = self._create_comparison_frame(
                        result['original_face'], 
                        result['swapped_face'],
                        result['cosine_similarity'],
                        result['expression_values']
                    )
                    
                    # Write to output video
                    out.write(comparison_frame)
                    successful_frames += 1
                
                processed_frames += 1
                progress_bar.update(1)
            
            progress_bar.close()
            
            # Cleanup
            cap.release()
            out.release()
            
            logger.info(f"✅ Identity swap video processing completed!")
            logger.info(f"   Processed frames: {processed_frames}")
            logger.info(f"   Successful swaps: {successful_frames}")
            logger.info(f"   Success rate: {successful_frames/processed_frames*100:.1f}%")
            logger.info(f"   Output saved to: {output_video_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing video: {e}")
            return False
    
    def _process_identity_swap_frame(self, face_frame: np.ndarray, input_subject_id: int, identity_features: Dict[str, Any], frame_idx: int = 0) -> Optional[Dict[str, Any]]:
        """
        Process a single frame for identity swapping
        
        Args:
            face_frame: Input face frame (518x518x3)
            input_subject_id: Subject ID for expression extraction
            identity_features: Dictionary with target identity features
            frame_idx: Frame index for tracking first frame
            
        Returns:
            Dictionary with original and identity-swapped faces, or None if failed
        """
        try:
            # Verify frame size
            if face_frame.shape != (518, 518, 3):
                logger.warning(f"⚠️ Expected 518x518x3 frame, got {face_frame.shape}")
                return None
            
            # Update token extractor with input subject ID for expression extraction
            self.token_extractor.subject_id = input_subject_id
            
            # Extract expression tokens from input frame
            logger.debug("Extracting tokens from input frame...")
            tokens = self.token_extractor.process_frame_tokens(face_frame)
            
            if tokens is None or tokens.get('expression_token') is None:
                logger.warning("Failed to extract tokens from input frame")
                return None
            
            logger.debug("Successfully extracted tokens from input frame")
            
            # Store first frame expression token for cosine similarity calculation
            if frame_idx == 0:
                self.first_frame_expression_token = tokens['expression_token'].clone()
                cosine_similarity = 1.0  # First frame has perfect similarity with itself
            else:
                # Calculate cosine similarity with first frame
                cosine_similarity = self._calculate_cosine_similarity(
                    self.first_frame_expression_token, 
                    tokens['expression_token']
                )
            
            # Get first three values of expression token for display
            expression_values = tokens['expression_token'].flatten()[:3].cpu().numpy()
            
            # Get target identity features
            target_patch_tokens = identity_features['patch_tokens'].to(self.device)
            target_pos_embeddings = identity_features['pos_embeddings'].to(self.device)
            target_subject_embeddings = identity_features['subject_embeddings'].to(self.device)
            
            # Ensure correct shapes - add batch dimensions if missing
            if len(target_patch_tokens.shape) == 2:
                target_patch_tokens = target_patch_tokens.unsqueeze(0)  # (1369, 384) -> (1, 1369, 384)
            
            if len(target_pos_embeddings.shape) == 2:
                target_pos_embeddings = target_pos_embeddings.unsqueeze(0)  # (1369, 384) -> (1, 1369, 384)
            
            # Fix subject_embeddings shape to be (1, 1, 384)
            logger.debug(f"Original target_subject_embeddings shape: {target_subject_embeddings.shape}")
            
            if len(target_subject_embeddings.shape) == 2:
                target_subject_embeddings = target_subject_embeddings.unsqueeze(0)  # (384,) -> (1, 384)
                target_subject_embeddings = target_subject_embeddings.unsqueeze(1)  # (1, 384) -> (1, 1, 384)
            elif len(target_subject_embeddings.shape) == 3 and target_subject_embeddings.shape[1] != 1:
                # If it's (1, 384, 384) or similar, squeeze to (1, 1, 384)
                target_subject_embeddings = target_subject_embeddings.squeeze(1)  # Remove extra dimension
                target_subject_embeddings = target_subject_embeddings.unsqueeze(1)  # Add back to (1, 1, 384)
            elif len(target_subject_embeddings.shape) == 4:
                # If it's (1, 1, 1, 384), squeeze to (1, 1, 384)
                target_subject_embeddings = target_subject_embeddings.squeeze(1)  # Remove extra dimension
            
            logger.debug(f"After shape fixing, target_subject_embeddings shape: {target_subject_embeddings.shape}")
            
            # Ensure it's exactly (1, 1, 384)
            if target_subject_embeddings.shape != (1, 1, 384):
                logger.warning(f"target_subject_embeddings shape is {target_subject_embeddings.shape}, forcing to (1, 1, 384)")
                # Force the shape by flattening and reshaping
                target_subject_embeddings = target_subject_embeddings.flatten()[:384].view(1, 1, 384)
            
            logger.debug("Starting face reconstruction...")
            # Reconstruct face using target identity but input expression
            swapped_face = self.token_extractor.reconstruct_face(
                target_subject_embeddings,
                tokens['expression_token'],
                target_patch_tokens,
                target_pos_embeddings
            )
            
            if swapped_face is None:
                logger.warning("Face reconstruction failed")
                return None
            
            logger.debug("Face reconstruction completed successfully")
            
            # Convert reconstructed face to numpy for display
            swapped_face = swapped_face.squeeze(0).cpu().numpy()  # (3, 518, 518)
            swapped_face = np.transpose(swapped_face, (1, 2, 0))  # (518, 518, 3)
            
            # Ensure both images are in the same format
            if face_frame.dtype != np.uint8:
                face_frame = (face_frame * 255).astype(np.uint8)
            
            if swapped_face.dtype != np.uint8:
                swapped_face = (swapped_face * 255).astype(np.uint8)
            
            return {
                'original_face': face_frame,
                'swapped_face': swapped_face,
                'cosine_similarity': cosine_similarity,
                'expression_values': expression_values
            }
            
        except Exception as e:
            logger.error(f"Error in _process_identity_swap_frame: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _create_comparison_frame(self, original_face: np.ndarray, swapped_face: np.ndarray, 
                                cosine_similarity: float = 1.0, expression_values: np.ndarray = None) -> np.ndarray:
        """
        Create side-by-side comparison frame with cosine similarity and expression token values
        
        Args:
            original_face: Original face image (518x518x3)
            swapped_face: Identity-swapped face image (518x518x3)
            cosine_similarity: Cosine similarity with first frame
            expression_values: First three values of expression token
            
        Returns:
            Side-by-side comparison frame (1036x518x3)
        """
        # Create side-by-side display (no resizing needed since both are 518x518)
        comparison_frame = np.hstack([original_face, swapped_face])
        
        # Add text labels
        cv2.putText(comparison_frame, "Original", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison_frame, "Identity Swapped", (568, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add cosine similarity
        cos_sim_text = f"Cosine Sim: {cosine_similarity:.8f}"
        cv2.putText(comparison_frame, cos_sim_text, (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add expression token values if provided
        if expression_values is not None:
            expr_text = f"Expr: [{expression_values[0]:.5f}, {expression_values[1]:.5f}, {expression_values[2]:.5f}]"
            cv2.putText(comparison_frame, expr_text, (50, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return comparison_frame


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate identity swap video using pre-prepared identity features")
    parser.add_argument("--input_video", type=str, 
    default="/Users/ozgewhiting/Documents/EQLabs/datasets_serial/CCA_small/1220_08_faces_53_07.mp4", help="Path to input video with 518x518 face frames")
    parser.add_argument("--input_subject_id", type=int, default=81, help="Subject ID for the input video")
    
    parser.add_argument("--output_video", type=str, 
    default="/Users/ozgewhiting/Documents/projects/cloud_checkpoints_with_keywords/1220_08_faces_53_07_reconstructed_w_subj_id_37_w_epoch_3.mp4", 
    help="Path to save output video")
    parser.add_argument("--identity_json", type=str, 
    default="/Users/ozgewhiting/Documents/projects/dataset_utils/app_demo/CCA_small_1176_14_faces_1_70_w_subj_id_37_w_embeddings.json",
    help="Path to JSON file with target identity features")
    parser.add_argument("--expression_transformer_checkpoint", type=str, 
                       default="/Users/ozgewhiting/Documents/projects/cloud_checkpoints_with_keywords/expression_transformer_epoch_2.pt",
                       help="Path to Expression Transformer checkpoint")
    parser.add_argument("--face_reconstruction_checkpoint", type=str, 
                       default="/Users/ozgewhiting/Documents/projects/cloud_checkpoints_with_subject_ids/reconstruction_model_epoch_3.pt",
                       help="Path to Face Reconstruction model checkpoint")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run models on")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum frames to process (for testing)")
    
    args = parser.parse_args()
    
    # Create identity swap generator
    generator = IdentitySwapVideoGenerator(
        expression_transformer_checkpoint_path=args.expression_transformer_checkpoint,
        face_reconstruction_checkpoint_path=args.face_reconstruction_checkpoint,
        device=args.device
    )
    
    # Process video
    success = generator.process_video(
        input_video_path=args.input_video,
        output_video_path=args.output_video,
        identity_json_path=args.identity_json,
        input_subject_id=args.input_subject_id,
        max_frames=args.max_frames
    )
    
    if success:
        logger.info("🎉 Identity swap video generation completed successfully!")
        return True
    else:
        logger.error("❌ Identity swap video generation failed!")
        return False


if __name__ == "__main__":
    exit(main()) 