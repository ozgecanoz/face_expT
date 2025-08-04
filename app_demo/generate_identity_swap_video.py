#!/usr/bin/env python3
"""
Generate Identity Swap Video
Creates a video where expressions from input video are mapped to a different subject's identity
Uses Expression Transformer and Expression Reconstruction models
"""

import cv2
import numpy as np
import torch
import logging
import argparse
import os
import sys
from typing import Optional, Dict, Any
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_loader import ModelLoader

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IdentitySwapVideoGenerator:
    """Generate identity swap videos using Expression Transformer and Expression Reconstruction models"""
    
    def __init__(self, 
                 expression_transformer_checkpoint_path: str,
                 expression_reconstruction_checkpoint_path: str,
                 device: str = "cpu"):
        """
        Initialize identity swap video generator
        
        Args:
            expression_transformer_checkpoint_path: Path to Expression Transformer checkpoint
            expression_reconstruction_checkpoint_path: Path to Expression Reconstruction model checkpoint
            device: Device to run models on
        """
        self.device = device
        
        # Load models
        logger.info("🚀 Loading models...")
        model_loader = ModelLoader(device=device)
        self.models = model_loader.load_all_models(
            expression_transformer_checkpoint_path=expression_transformer_checkpoint_path,
            expression_predictor_checkpoint_path="dummy",  # Not needed for reconstruction
            face_reconstruction_checkpoint_path=expression_reconstruction_checkpoint_path
        )
        
        # Initialize components
        self.tokenizer = self.models['tokenizer']
        self.expression_transformer = self.models['expression_transformer']
        self.expression_reconstruction_model = self.models.get('expression_reconstruction_model')
        
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
    
    def get_target_subject_embedding(self, target_subject_id: int) -> torch.Tensor:
        """
        Get the learned subject embedding for target subject ID
        
        Args:
            target_subject_id: Target subject ID
            
        Returns:
            Subject embedding tensor (1, 1, 384)
        """
        # Get subject embedding from Expression Transformer
        subject_embedding = self.expression_transformer.subject_embeddings(torch.tensor([target_subject_id], device=self.device))
        subject_embedding = subject_embedding.unsqueeze(1)  # (1, 1, 384)
        
        logger.info(f"✅ Retrieved subject embedding for target subject ID: {target_subject_id}")
        return subject_embedding
    
    def process_video(self, 
                     input_video_path: str, 
                     output_video_path: str, 
                     input_subject_id: int = 0,
                     target_subject_id: int = 0,
                     max_frames: Optional[int] = None) -> bool:
        """
        Process video and generate identity swap video
        
        Args:
            input_video_path: Path to input video with face frames
            output_video_path: Path to save output video
            input_subject_id: Subject ID for the input video (for expression extraction)
            target_subject_id: Target subject ID for identity swap
            max_frames: Maximum number of frames to process (for testing)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get target subject embedding
            target_subject_embedding = self.get_target_subject_embedding(target_subject_id)
            
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
            logger.info(f"   Target Subject ID: {target_subject_id}")
            
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
                result = self._process_identity_swap_frame(frame, input_subject_id, target_subject_embedding, frame_idx)
                
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
    
    def _process_identity_swap_frame(self, face_frame: np.ndarray, input_subject_id: int, target_subject_embedding: torch.Tensor, frame_idx: int = 0) -> Optional[Dict[str, Any]]:
        """
        Process a single frame for identity swapping
        
        Args:
            face_frame: Input face frame (518x518x3)
            input_subject_id: Subject ID for expression extraction
            target_subject_embedding: Target subject embedding tensor (1, 1, 384)
            frame_idx: Frame index for tracking first frame
            
        Returns:
            Dictionary with original and identity-swapped faces, or None if failed
        """
        try:
            # Verify frame size
            if face_frame.shape != (518, 518, 3):
                logger.warning(f"⚠️ Expected 518x518x3 frame, got {face_frame.shape}")
                return None
            
            # Convert numpy array to torch tensor
            face_tensor = torch.from_numpy(face_frame).float().to(self.device)
            
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
            
            # Create subject ID tensor for input subject
            subject_ids = torch.tensor([input_subject_id], dtype=torch.long, device=self.device)
            
            # Extract expression token using Expression Transformer
            with torch.no_grad():
                expression_token, _ = self.expression_transformer.inference(patch_tokens, pos_embeddings, subject_ids)
            
            logger.debug("Successfully extracted expression token from input frame")
            
            # Store first frame expression token for cosine similarity calculation
            if frame_idx == 0:
                self.first_frame_expression_token = expression_token.clone()
                cosine_similarity = 1.0  # First frame has perfect similarity with itself
            else:
                # Calculate cosine similarity with first frame
                cosine_similarity = self._calculate_cosine_similarity(
                    self.first_frame_expression_token, 
                    expression_token
                )
            
            # Get first three values of expression token for display
            expression_values = expression_token.flatten()[:3].cpu().numpy()
            
            # Add delta positional embeddings from Expression Transformer
            adjusted_pos_embeddings = pos_embeddings + self.expression_transformer.delta_pos_embed
            
            logger.debug("Starting face reconstruction...")
            # Reconstruct face using target subject embedding and input expression token
            with torch.no_grad():
                reconstructed_face = self.expression_reconstruction_model(
                    target_subject_embedding,
                    expression_token,
                    adjusted_pos_embeddings
                )
            
            if reconstructed_face is None:
                logger.warning("Face reconstruction failed")
                return None
            
            logger.debug("Face reconstruction completed successfully")
            
            # Convert reconstructed face to numpy for display
            swapped_face = reconstructed_face.squeeze(0).cpu().numpy()  # (3, 518, 518)
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
    parser = argparse.ArgumentParser(description="Generate identity swap video using pre-prepared identity features and Expression Reconstruction model")
    parser.add_argument("--input_video", type=str, 
    #default="/Users/ozgewhiting/Documents/EQLabs/datasets_serial/CCA_small/1220_08_faces_53_07.mp4", help="Path to input video with 518x518 face frames")
    #default="/Users/ozgewhiting/Documents/EQLabs/datasets_serial/CCA_train_db4_no_padding/CCA_train_db4_no_padding/subject_369_1508_09_faces_20_82.mp4",
   
    #default="/Users/ozgewhiting/Documents/EQLabs/datasets_serial/CCA_train_db4_no_padding/CCA_train_db4_no_padding/subject_100_1239_10_faces_21_09.mp4",
    #default="/Users/ozgewhiting/Documents/EQLabs/datasets_serial/CCA_train_db4_no_padding/CCA_train_db4_no_padding/subject_92_1231_00_faces_38_23.mp4",
    default="/Users/ozgewhiting/Documents/EQLabs/datasets_serial/CCA_train_db4_no_padding/CCA_train_db4_no_padding/subject_116_1255_00_faces_9_18.mp4",
    help="Path to input video with 518x518 face frames") 
    #parser.add_argument("--input_subject_id", type=int, default=100, help="Subject ID for the input video")
    #parser.add_argument("--input_subject_id", type=int, default=92, help="Subject ID for the input video")
    parser.add_argument("--input_subject_id", type=int, default=116, help="Subject ID for the input video")

    parser.add_argument("--target_subject_id", type=int, default=37, help="Target subject ID for identity swap")
    
    parser.add_argument("--output_video", type=str, 
    #default="/Users/ozgewhiting/Documents/projects/cloud_checkpoints_with_keywords/1220_08_faces_53_07_expression_identity_swapped.mp4", 
    #default="/Users/ozgewhiting/Documents/projects/cloud_checkpoints_with_keywords7/subject_369_1508_09_faces_20_82_expression_identity_swapped_w_subj_id_37.mp4", 
    #default="/Users/ozgewhiting/Documents/projects/checkpoints_with_keywords8/subject_100_1239_10_faces_21_09_expression_identity_swapped_w_subj_id_37_w_step_19800.mp4", 
    #default="/Users/ozgewhiting/Documents/projects/checkpoints_with_keywords8/subject_92_1231_00_faces_38_23_expression_identity_swapped_w_subj_id_37_w_step_5400_frozen_expr_trans.mp4", 
    default="/Users/ozgewhiting/Documents/projects/checkpoints_with_keywords8/subject_116_1255_00_faces_9_18_expression_identity_swapped_w_subj_id_37_w_step_5400_frozen_expr_trans.mp4", 
    help="Path to save output video")

    parser.add_argument("--expression_transformer_checkpoint", type=str, 
                       default="/Users/ozgewhiting/Documents/projects/checkpoints_with_keywords8/expression_transformer_step_5400.pt",
                       help="Path to Expression Transformer checkpoint")
    parser.add_argument("--expression_reconstruction_checkpoint", type=str, 
                       default="/Users/ozgewhiting/Documents/projects/checkpoints_with_keywords8/expression_reconstruction_step_5400.pt",
                       help="Path to Expression Reconstruction model checkpoint")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run models on")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum frames to process (for testing)")
    
    args = parser.parse_args()
    
    # Create identity swap generator
    generator = IdentitySwapVideoGenerator(
        expression_transformer_checkpoint_path=args.expression_transformer_checkpoint,
        expression_reconstruction_checkpoint_path=args.expression_reconstruction_checkpoint,
        device=args.device
    )
    
    # Process video
    success = generator.process_video(
        input_video_path=args.input_video,
        output_video_path=args.output_video,
        input_subject_id=args.input_subject_id,
        target_subject_id=args.target_subject_id,
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