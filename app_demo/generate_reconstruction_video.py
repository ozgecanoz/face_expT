#!/usr/bin/env python3
"""
Generate Reconstruction Comparison Video
Creates a side-by-side video showing original frames vs reconstructed frames using Expression Reconstruction model
Works with pre-cropped 518x518 face frames
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
from token_extractor import TokenExtractor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ReconstructionVideoGenerator:
    """Generate side-by-side reconstruction comparison videos from pre-cropped face frames using Expression Reconstruction model"""
    
    def __init__(self, 
                 expression_transformer_checkpoint_path: str,
                 expression_reconstruction_checkpoint_path: str,
                 device: str = "cpu"):
        """
        Initialize video generator
        
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
        self.token_extractor = TokenExtractor(
            expression_transformer=self.models['expression_transformer'],
            expression_predictor=self.models.get('expression_predictor'),  # May be None
            expression_reconstruction_model=self.models.get('expression_reconstruction_model'),  # May be None
            tokenizer=self.models['tokenizer'],
            device=self.device,
            subject_id=0  # Will be set per frame
        )
        
        logger.info("✅ Reconstruction Video Generator initialized successfully!")
    
    def process_video(self, 
                     input_video_path: str, 
                     output_video_path: str, 
                     subject_id: int,
                     target_subject_id: Optional[int] = None,
                     max_frames: Optional[int] = None) -> bool:
        """
        Process video and generate side-by-side comparison
        
        Args:
            input_video_path: Path to input video with pre-cropped 518x518 face frames
            output_video_path: Path to save output video
            subject_id: Subject ID for the input video
            target_subject_id: Subject ID to use for reconstruction (if different from input)
            max_frames: Maximum number of frames to process (for testing)
            
        Returns:
            True if successful, False otherwise
        """
        try:
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
            logger.info(f"   Subject ID: {subject_id}")
            logger.info(f"   Target Subject ID: {target_subject_id if target_subject_id else subject_id}")
            
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
                result = self._process_face_frame(frame, subject_id, target_subject_id)
                
                if result is not None:
                    # Create side-by-side comparison
                    comparison_frame = self._create_comparison_frame(
                        result['original_face'], 
                        result['reconstructed_face']
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
            
            logger.info(f"✅ Video processing completed!")
            logger.info(f"   Processed frames: {processed_frames}")
            logger.info(f"   Successful reconstructions: {successful_frames}")
            logger.info(f"   Success rate: {successful_frames/processed_frames*100:.1f}%")
            logger.info(f"   Output saved to: {output_video_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing video: {e}")
            return False
    
    def _process_face_frame(self, face_frame: np.ndarray, subject_id: int, target_subject_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Process a single 518x518 face frame
        
        Args:
            face_frame: Input face frame (518x518x3)
            subject_id: Subject ID for the input video
            target_subject_id: Target subject ID for reconstruction
            
        Returns:
            Dictionary with original and reconstructed faces, or None if failed
        """
        # Verify frame size
        if face_frame.shape != (518, 518, 3):
            logger.warning(f"⚠️ Expected 518x518x3 frame, got {face_frame.shape}")
            return None
        
        # Update token extractor with current subject ID
        current_subject_id = target_subject_id if target_subject_id else subject_id
        self.token_extractor.subject_id = current_subject_id
        
        # Extract tokens from face frame
        tokens = self.token_extractor.process_frame_tokens(face_frame)
        
        if tokens is None or tokens.get('expression_token') is None:
            return None
        
        # Reconstruct face
        reconstructed_face = self.token_extractor.reconstruct_face(
            tokens['subject_embeddings'],
            tokens['expression_token'],
            tokens['patch_tokens'],
            tokens['pos_embeddings']
        )
        
        if reconstructed_face is None:
            return None
        
        # Convert reconstructed face to numpy for display
        reconstructed_face = reconstructed_face.squeeze(0).cpu().numpy()  # (3, 518, 518)
        reconstructed_face = np.transpose(reconstructed_face, (1, 2, 0))  # (518, 518, 3)
        
        # Ensure both images are in the same format
        if face_frame.dtype != np.uint8:
            face_frame = (face_frame * 255).astype(np.uint8)
        
        if reconstructed_face.dtype != np.uint8:
            reconstructed_face = (reconstructed_face * 255).astype(np.uint8)
        
        return {
            'original_face': face_frame,
            'reconstructed_face': reconstructed_face
        }
    
    def _create_comparison_frame(self, original_face: np.ndarray, reconstructed_face: np.ndarray) -> np.ndarray:
        """
        Create side-by-side comparison frame
        
        Args:
            original_face: Original face image (518x518x3)
            reconstructed_face: Reconstructed face image (518x518x3)
            
        Returns:
            Side-by-side comparison frame (1036x518x3)
        """
        # Create side-by-side display (no resizing needed since both are 518x518)
        comparison_frame = np.hstack([original_face, reconstructed_face])
        
        # Add text labels
        cv2.putText(comparison_frame, "Original", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison_frame, "Reconstructed", (568, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return comparison_frame


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate reconstruction comparison video from pre-cropped face frames using Expression Reconstruction model")
    parser.add_argument("--input_video", type=str, 
    #default="/Users/ozgewhiting/Documents/EQLabs/datasets_serial/CCA_small/1176_14_faces_1_70.mp4", # subject_37
    #default="/Users/ozgewhiting/Documents/EQLabs/datasets_serial/CCA_train_db4_no_padding/CCA_train_db4_no_padding/subject_369_1508_09_faces_20_82.mp4", 
    default="/Users/ozgewhiting/Documents/EQLabs/datasets_serial/CCA_train_db4_no_padding/CCA_train_db4_no_padding/subject_100_1239_10_faces_21_09.mp4",
    help="Path to input video with 518x518 face frames")
    parser.add_argument("--subject_id", type=int, default=100, help="Subject ID for the input video")
    parser.add_argument("--output_video", type=str, 
    #default="/Users/ozgewhiting/Documents/projects/cloud_checkpoints_with_keywords7/1176_14_faces_1_70_reconstructed2_w_subj_id_37_w_keyword_epoch_2.mp4", 
    #default="/Users/ozgewhiting/Documents/projects/cloud_checkpoints_with_keywords7/subject_369_1508_09_faces_20_82_reconstructed.mp4", 
    default="/Users/ozgewhiting/Documents/projects/cloud_checkpoints_with_keywords7/subject_100_1239_10_faces_21_09_reconstructed_w_step_5400.mp4", 
    help="Path to save output video")
    parser.add_argument("--target_subject_id", type=int, default=None,
                       help="Target subject ID for reconstruction (if different from input)")
    parser.add_argument("--expression_transformer_checkpoint", type=str, 
                       default="/Users/ozgewhiting/Documents/projects/cloud_checkpoints_with_keywords7/expression_transformer_step_5400.pt",
                       help="Path to Expression Transformer checkpoint")
    parser.add_argument("--expression_reconstruction_checkpoint", type=str, 
                       default="/Users/ozgewhiting/Documents/projects/cloud_checkpoints_with_keywords7/expression_reconstruction_step_5400.pt",
                       help="Path to Expression Reconstruction model checkpoint")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run models on")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum frames to process (for testing)")
    
    args = parser.parse_args()
    
    # Create video generator
    generator = ReconstructionVideoGenerator(
        expression_transformer_checkpoint_path=args.expression_transformer_checkpoint,
        expression_reconstruction_checkpoint_path=args.expression_reconstruction_checkpoint,
        device=args.device
    )
    
    # Process video
    success = generator.process_video(
        input_video_path=args.input_video,
        output_video_path=args.output_video,
        subject_id=args.subject_id,
        target_subject_id=args.target_subject_id,
        max_frames=args.max_frames
    )
    
    if success:
        logger.info("🎉 Video generation completed successfully!")
        return True
    else:
        logger.error("❌ Video generation failed!")
        return False


if __name__ == "__main__":
    exit(main()) 