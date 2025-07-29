"""
Webcam Demo Application
Real-time face detection, token extraction, and expression prediction
"""

import cv2
import numpy as np
import torch
import logging
import argparse
import time
from typing import Optional, Dict, Any
import os
import json

# Import our components
from model_loader import ModelLoader
from face_detector import MediaPipeFaceDetector
from token_extractor import TokenExtractor, TokenBuffer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WebcamDemo:
    """Main webcam demo application"""
    
    def __init__(self, 
                 expression_transformer_checkpoint_path: str,
                 expression_predictor_checkpoint_path: str,
                 face_reconstruction_checkpoint_path: str,
                 identity_features_path: str = None,
                 device: str = "cpu",
                 confidence_threshold: float = 0.5,
                 subject_id: int = 0):
        """
        Initialize webcam demo
        
        Args:
            expression_transformer_checkpoint_path: Path to Expression Transformer checkpoint
            expression_predictor_checkpoint_path: Path to Expression Predictor checkpoint
            face_reconstruction_checkpoint_path: Path to Face Reconstruction model checkpoint
            identity_features_path: Path to JSON file with identity features (optional)
            device: Device to run models on
            confidence_threshold: Face detection confidence threshold
            subject_id: Subject ID for the current user
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.identity_features_path = identity_features_path
        self.subject_id = subject_id
        
        # Load identity features if provided
        self.identity_features = None
        self.first_frame_expression_token = None  # Store first frame's expression token for comparison
        if identity_features_path is not None:
            self.identity_features = self._load_identity_features(identity_features_path)
            if self.identity_features is not None:
                logger.info(f"✅ Loaded identity features from: {identity_features_path}")
                logger.info(f"   Identity source: {self.identity_features.get('video_path', 'unknown')}")
                logger.info(f"   Subject ID: {self.identity_features.get('subject_id', 'unknown')}")
            else:
                logger.warning(f"⚠️ Failed to load identity features from: {identity_features_path}")
                logger.warning("   Will use current frame's identity for reconstruction")
        else:
            logger.info("ℹ️ No identity features provided - will use current frame's identity")
        
        # Load all models
        logger.info("🚀 Loading models...")
        model_loader = ModelLoader(device=device)
        self.models = model_loader.load_all_models(
            expression_transformer_checkpoint_path=expression_transformer_checkpoint_path,
            expression_predictor_checkpoint_path=expression_predictor_checkpoint_path,
            face_reconstruction_checkpoint_path=face_reconstruction_checkpoint_path
        )
        
        # Initialize components
        self.tokenizer = self.models['tokenizer']
        self.face_detector = MediaPipeFaceDetector(confidence_threshold=confidence_threshold)
        self.token_extractor = TokenExtractor(
            expression_transformer=self.models['expression_transformer'],
            expression_predictor=self.models['expression_predictor'],
            face_reconstruction_model=self.models['face_reconstruction_model'],
            tokenizer=self.models['tokenizer'],
            device=self.device,
            subject_id=self.subject_id
        )
        self.token_buffer = TokenBuffer(buffer_size=30)
        
        # Initialize webcam
        self.cap = None
        
        # Performance tracking
        self.frame_count = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
        logger.info("✅ Webcam Demo initialized successfully!")
    
    def _load_identity_features(self, json_path: str) -> Optional[Dict[str, Any]]:
        """
        Load identity features from JSON file
        
        Args:
            json_path: Path to JSON file with identity features
            
        Returns:
            Dictionary containing identity features or None if failed
        """
        try:
            if not os.path.exists(json_path):
                logger.error(f"Identity features file not found: {json_path}")
                return None
            
            with open(json_path, 'r') as f:
                identity_data = json.load(f)
            
            # Validate required fields (no longer need face_id_token)
            required_fields = ['patch_tokens', 'pos_embeddings', 'subject_id', 'subject_embeddings']
            for field in required_fields:
                if field not in identity_data:
                    logger.error(f"Missing required field '{field}' in identity features file")
                    return None
            
            # Convert lists back to numpy arrays
            identity_features = {
                'patch_tokens': np.array(identity_data['patch_tokens'], dtype=np.float32),
                'pos_embeddings': np.array(identity_data['pos_embeddings'], dtype=np.float32),
                'subject_embeddings': np.array(identity_data['subject_embeddings'], dtype=np.float32),
                'subject_id': identity_data['subject_id'],
                'video_path': identity_data.get('video_path', 'unknown'),
                'frame_shape': identity_data.get('frame_shape', [518, 518, 3])
            }
            
            logger.info(f"✅ Loaded identity features:")
            logger.info(f"   Subject ID: {identity_features['subject_id']}")
            logger.info(f"   Patch tokens shape: {identity_features['patch_tokens'].shape}")
            logger.info(f"   Pos embeddings shape: {identity_features['pos_embeddings'].shape}")
            logger.info(f"   Subject embeddings shape: {identity_features['subject_embeddings'].shape}")
            
            return identity_features
            
        except Exception as e:
            logger.error(f"Error loading identity features: {e}")
            return None
    
    def _calculate_cosine_similarity(self, token1: torch.Tensor, token2: torch.Tensor) -> float:
        """
        Calculate cosine similarity between two tokens
        
        Args:
            token1: First token tensor
            token2: Second token tensor
            
        Returns:
            Cosine similarity value (0.0 to 1.0)
        """
        # Ensure tokens are 1D
        token1_flat = token1.squeeze().flatten()
        token2_flat = token2.squeeze().flatten()
        
        # Calculate cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            token1_flat.unsqueeze(0), 
            token2_flat.unsqueeze(0), 
            dim=1
        ).item()
        
        return cos_sim
    
    def start_webcam(self, camera_index: int = 0):
        """Start webcam capture"""
        self.cap = cv2.VideoCapture(camera_index)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam at index {camera_index}")
        
        logger.info(f"📹 Webcam started at index {camera_index}")
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single frame
        
        Args:
            frame: Input frame from webcam
            
        Returns:
            Dictionary containing processing results
        """
        # Detect faces and extract largest face
        face_image, detected_faces = self.face_detector.process_frame(frame)
        
        if face_image is None:
            return {
                'face_detected': False,
                'detected_faces': detected_faces,
                'tokens': None,
                'prediction': None,
                'reconstructed_face': None
            }
        
        # Extract tokens from face
        tokens = self.token_extractor.process_frame_tokens(face_image)
        
        if tokens is None:
            return {
                'face_detected': True,
                'detected_faces': detected_faces,
                'tokens': None,
                'prediction': None,
                'reconstructed_face': None
            }
        
        # Store first frame's expression token for comparison
        if self.first_frame_expression_token is None and tokens.get('expression_token') is not None:
            self.first_frame_expression_token = tokens['expression_token'].clone()
            logger.info("📸 Stored first frame's expression token for comparison")
            logger.info(f"   First frame expression token (first 5 values): {self.first_frame_expression_token.squeeze().cpu().numpy()[:5]}")
        
        # Compare current expression token with first frame's expression token
        if self.first_frame_expression_token is not None and tokens.get('expression_token') is not None:
            current_expr = tokens['expression_token'].squeeze()  # (384,)
            first_expr = self.first_frame_expression_token.squeeze()  # (384,)
            
            # Calculate cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(current_expr.unsqueeze(0), first_expr.unsqueeze(0), dim=1).item()
            
            # Store for display on frame
            self.current_expression_similarity = cos_sim
            
            # Print comparison info
            logger.info(f"🔍 Frame {self.token_buffer.get_frame_count()}:")
            logger.info(f"   Current expression token (first 5 values): {current_expr.cpu().numpy()[:5]}")
            logger.info(f"   First frame expression token (first 5 values): {first_expr.cpu().numpy()[:5]}")
            logger.info(f"   Cosine similarity: {cos_sim:.4f} (1.0 = identical, 0.0 = completely different)")
            
            # Log if expressions are very similar (potential issue)
            if cos_sim > 0.95:
                logger.warning(f"⚠️ Expression tokens are very similar (cos_sim={cos_sim:.4f}) - expressions may not be changing!")
            elif cos_sim < 0.5:
                logger.info(f"✅ Expression tokens are significantly different (cos_sim={cos_sim:.4f})")
        else:
            self.current_expression_similarity = None
        
        # Reconstruct face from tokens
        reconstructed_face = None
        if tokens.get('expression_token') is not None:
            # Get DINOv2 tokens from the token extraction
            patch_tokens = tokens.get('patch_tokens')
            pos_embeddings = tokens.get('pos_embeddings')
            subject_embeddings = tokens.get('subject_embeddings')
            
            if patch_tokens is not None and pos_embeddings is not None and subject_embeddings is not None:
                '''
                # When using identity features, use the loaded identity's subject embeddings
                if self.identity_features is not None:
                    # Get identity subject ID and precomputed subject embeddings
                    identity_subject_id = self.identity_features['subject_id']
                    
                    # Load precomputed subject embeddings from JSON
                    identity_subject_embeddings = torch.tensor(
                        self.identity_features['subject_embeddings'], 
                        dtype=torch.float32, 
                        device=self.device
                    ).squeeze().unsqueeze(0).unsqueeze(0)  # (1, 1, 384)
                    
                    # Load identity's DINOv2 tokens from JSON
                    identity_patch_tokens = torch.tensor(
                        self.identity_features['patch_tokens'], 
                        dtype=torch.float32, 
                        device=self.device
                    ).unsqueeze(0)  # (1, 1369, 384)
                    
                    identity_pos_embeddings = torch.tensor(
                        self.identity_features['pos_embeddings'], 
                        dtype=torch.float32, 
                        device=self.device
                    ).unsqueeze(0)  # (1, 1369, 384)
                    
                    logger.info(f"✅ Using precomputed subject embeddings for identity subject {identity_subject_id}")
                    logger.info(f"   Subject embeddings shape: {identity_subject_embeddings.shape}")
                    
                    # Reconstruct face using identity's subject embeddings and current frame's expression token
                    reconstructed_face = self.token_extractor.reconstruct_face(
                        identity_subject_embeddings,  # Use precomputed subject embeddings
                        tokens['expression_token'],   # Use current frame's expression token
                        identity_patch_tokens,        # Use identity's DINOv2 tokens
                        identity_pos_embeddings       # Use identity's positional embeddings
                    )
                    '''
                # Use current frame's identity and expression
                reconstructed_face = self.token_extractor.reconstruct_face(
                subject_embeddings,
                tokens['expression_token'],
                patch_tokens,
                pos_embeddings
                )
            else:
                    
                # Use current frame's identity and expression
                reconstructed_face = self.token_extractor.reconstruct_face(
                subject_embeddings,
                tokens['expression_token'],
                patch_tokens,
                pos_embeddings
                )
                
            # Convert to numpy for display
            reconstructed_face = reconstructed_face.squeeze(0).cpu().numpy()  # (3, 518, 518)
            reconstructed_face = np.transpose(reconstructed_face, (1, 2, 0))  # (518, 518, 3)
        
        # Add tokens to buffer
        self.token_buffer.add_frame_tokens(tokens['expression_token'])
        
        # Check if ready for prediction
        prediction = None
        if self.token_buffer.is_ready_for_prediction():
            logger.info("🔍 Ready for prediction, getting sequence...")
            expr_seq, _ = self.token_buffer.get_prediction_sequence()
            if expr_seq is not None:
                # Debug: Print sequence shape
                logger.info(f"Prediction sequence shape: {expr_seq.shape}")
                
                # Predict next expression token
                logger.info("🔍 Calling predict_next_expression...")
                try:
                    prediction = self.token_extractor.predict_next_expression(expr_seq)
                    logger.info("✅ Prediction successful")
                except Exception as e:
                    logger.error(f"❌ Error in predict_next_expression: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                
                # Debug: Print prediction shape
                if prediction is not None:
                    logger.info(f"Prediction shape: {prediction.shape}")
        
        return {
            'face_detected': True,
            'detected_faces': detected_faces,
            'tokens': tokens,
            'prediction': prediction,
            'reconstructed_face': reconstructed_face,
            'frame_count': self.token_buffer.get_frame_count()
        }
    
    def draw_frame_info(self, frame: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """
        Draw information on frame
        
        Args:
            frame: Input frame
            results: Processing results
            
        Returns:
            Frame with information drawn
        """
        # Draw detection boxes
        if results['detected_faces']:
            frame = self.face_detector.draw_detection_boxes(frame, results['detected_faces'])
        
        # Draw frame info
        frame_count = results.get('frame_count', 0)
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw FPS
        self.fps_counter += 1
        if self.fps_counter % 30 == 0:
            current_time = time.time()
            fps = 30 / (current_time - self.fps_start_time)
            self.fps_start_time = current_time
            self.current_fps = fps
        
        if hasattr(self, 'current_fps'):
            cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw prediction status
        if results.get('prediction') is not None:
            cv2.putText(frame, "PREDICTION READY", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif results.get('face_detected', False):
            cv2.putText(frame, "FACE DETECTED", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Draw identity source info
        if self.identity_features is not None:
            identity_source = os.path.basename(self.identity_features.get('video_path', 'unknown'))
            cv2.putText(frame, f"ID: {identity_source}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, "RECONSTRUCTING WITH LOADED IDENTITY", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(frame, f"Subject ID: {self.subject_id}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, "RECONSTRUCTING WITH CURRENT IDENTITY", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Draw expression similarity if available
        if hasattr(self, 'current_expression_similarity') and self.current_expression_similarity is not None:
            similarity_text = f"Expression similarity to first: {self.current_expression_similarity:.3f}"
            cv2.putText(frame, similarity_text, (10, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def print_token_info(self, results: Dict[str, Any]):
        """Print token information to console"""
        if results is None or 'tokens' not in results:
            return
        
        tokens = results['tokens']
        frame_num = results.get('frame_num', 0)
        
        # Print token statistics
        expression_token = tokens.get('expression_token')
        predicted_token = tokens.get('predicted_token')
        subject_ids = tokens.get('subject_ids')
        
        print(f"📊 Frame {frame_num} (Subject ID: {self.subject_id}):")
        
        if expression_token is not None:
            expr_stats = f"mean={expression_token.mean().item():.4f}, std={expression_token.std().item():.4f}"
            print(f"   Expression Token: {expr_stats}")
        
        if predicted_token is not None:
            pred_stats = f"mean={predicted_token.mean().item():.4f}, std={predicted_token.std().item():.4f}"
            print(f"   🔮 Prediction: {pred_stats}")
    
    def display_face_comparison(self, original_face: np.ndarray, reconstructed_face: np.ndarray) -> np.ndarray:
        """
        Display original and reconstructed faces side by side
        
        Args:
            original_face: Original cropped face image (518x518x3)
            reconstructed_face: Reconstructed face image (518x518x3)
            
        Returns:
            Combined display image
        """
        # Ensure both images are in the same format
        if original_face.dtype != np.uint8:
            original_face = (original_face * 255).astype(np.uint8)
        
        if reconstructed_face.dtype != np.uint8:
            reconstructed_face = (reconstructed_face * 255).astype(np.uint8)
        
        # Resize both images to a reasonable display size
        display_size = (400, 400)
        original_resized = cv2.resize(original_face, display_size)
        reconstructed_resized = cv2.resize(reconstructed_face, display_size)
        
        # Create side-by-side display
        combined_image = np.hstack([original_resized, reconstructed_resized])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        color = (255, 255, 255)  # White text
        
        # Add "Original" label
        cv2.putText(combined_image, "Original", (50, 50), font, font_scale, color, thickness)
        
        # Add "Reconstructed" label
        cv2.putText(combined_image, "Reconstructed", (450, 50), font, font_scale, color, thickness)
        
        return combined_image
    
    def run(self, camera_index: int = 0, print_tokens: bool = True):
        """
        Run the webcam demo
        
        Args:
            camera_index: Camera index to use
            print_tokens: Whether to print token information
        """
        try:
            # Start webcam
            self.start_webcam(camera_index)
            
            logger.info("🎥 Starting webcam demo...")
            logger.info("Press 'q' to quit, 'r' to reset buffer")
            
            while True:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to read frame")
                    break
                
                # Process frame
                results = self.process_frame(frame)
                
                # Draw information on frame
                frame = self.draw_frame_info(frame, results)
                
                # Display frame
                cv2.imshow('Webcam Demo', frame)
                
                # Display face comparison if reconstruction is available
                if results.get('reconstructed_face') is not None and results.get('face_detected', False):
                    # Get the original face image from the detector
                    original_face = self.face_detector.get_last_cropped_face()
                    if original_face is not None:
                        comparison_image = self.display_face_comparison(
                            original_face, 
                            results['reconstructed_face']
                        )
                        cv2.imshow('Face Comparison', comparison_image)
                
                # Print token information if requested
                if print_tokens and results.get('tokens') is not None:
                    self.print_token_info(results)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quit requested")
                    break
                elif key == ord('r'):
                    logger.info("Resetting token buffer")
                    self.token_buffer.reset()
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error in webcam demo: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap is not None:
            self.cap.release()
        
        if hasattr(self, 'face_detector'):
            self.face_detector.close()
        
        cv2.destroyAllWindows()
        logger.info("🧹 Cleanup completed")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Webcam Demo with Face Token Extraction")
    parser.add_argument("--expression_transformer_checkpoint", type=str, required=True,
                       help="Path to Expression Transformer checkpoint")
    parser.add_argument("--expression_predictor_checkpoint", type=str, required=True,
                       help="Path to Expression Predictor checkpoint")
    parser.add_argument("--face_reconstruction_checkpoint", type=str, required=True,
                       help="Path to Face Reconstruction model checkpoint")
    parser.add_argument("--identity_features", type=str, default=None,
                       help="Path to JSON file with identity features (optional)")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to run models on (cpu/cuda)")
    parser.add_argument("--camera", type=int, default=0,
                       help="Camera index to use")
    parser.add_argument("--confidence", type=float, default=0.5,
                       help="Face detection confidence threshold")
    parser.add_argument("--subject_id", type=int, default=0,
                       help="Subject ID for the current user")
    parser.add_argument("--no_print_tokens", action="store_true",
                       help="Disable token printing to console")
    
    args = parser.parse_args()
    
    # Create and run demo
    demo = WebcamDemo(
        expression_transformer_checkpoint_path=args.expression_transformer_checkpoint,
        expression_predictor_checkpoint_path=args.expression_predictor_checkpoint,
        face_reconstruction_checkpoint_path=args.face_reconstruction_checkpoint,
        identity_features_path=args.identity_features,
        device=args.device,
        confidence_threshold=args.confidence,
        subject_id=args.subject_id
    )
    
    demo.run(
        camera_index=args.camera,
        print_tokens=not args.no_print_tokens
    )


if __name__ == "__main__":
    main() 