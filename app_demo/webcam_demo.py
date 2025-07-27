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
                 face_id_checkpoint_path: str,
                 expression_transformer_checkpoint_path: str,
                 expression_predictor_checkpoint_path: str,
                 device: str = "cpu",
                 confidence_threshold: float = 0.5):
        """
        Initialize webcam demo
        
        Args:
            face_id_checkpoint_path: Path to Face ID model checkpoint
            expression_transformer_checkpoint_path: Path to Expression Transformer checkpoint
            expression_predictor_checkpoint_path: Path to Expression Predictor checkpoint
            device: Device to run models on
            confidence_threshold: Face detection confidence threshold
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        
        # Load all models
        logger.info("🚀 Loading models...")
        model_loader = ModelLoader(device=device)
        self.models = model_loader.load_all_models(
            face_id_checkpoint_path=face_id_checkpoint_path,
            expression_transformer_checkpoint_path=expression_transformer_checkpoint_path,
            expression_predictor_checkpoint_path=expression_predictor_checkpoint_path
        )
        
        # Initialize components
        self.tokenizer = self.models['tokenizer']
        self.face_detector = MediaPipeFaceDetector(confidence_threshold=confidence_threshold)
        self.token_extractor = TokenExtractor(self.models, self.tokenizer, device=device)
        self.token_buffer = TokenBuffer(buffer_size=30)
        
        # Initialize webcam
        self.cap = None
        
        # Performance tracking
        self.frame_count = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
        logger.info("✅ Webcam Demo initialized successfully!")
    
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
                'prediction': None
            }
        
        # Extract tokens from face
        tokens = self.token_extractor.process_frame_tokens(face_image)
        
        if tokens is None:
            return {
                'face_detected': True,
                'detected_faces': detected_faces,
                'tokens': None,
                'prediction': None
            }
        
        # Add tokens to buffer
        self.token_buffer.add_frame_tokens(
            tokens['face_id_token'],
            tokens['expression_token']
        )
        
        # Check if ready for prediction
        prediction = None
        if self.token_buffer.is_ready_for_prediction():
            face_id_seq, expr_seq = self.token_buffer.get_prediction_sequence()
            if face_id_seq is not None and expr_seq is not None:
                # Predict next expression token
                prediction = self.token_extractor.predict_next_expression(expr_seq)
        
        return {
            'face_detected': True,
            'detected_faces': detected_faces,
            'tokens': tokens,
            'prediction': prediction,
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
            cv2.putText(frame, "COLLECTING FRAMES", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "NO FACE DETECTED", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame
    
    def print_token_info(self, results: Dict[str, Any]):
        """Print token information to console"""
        if results.get('tokens') is not None:
            tokens = results['tokens']
            
            # Print token statistics
            face_id_token = tokens['face_id_token']
            expression_token = tokens['expression_token']
            
            print(f"\n📊 Frame {results.get('frame_count', 0)}:")
            print(f"   Face ID Token: mean={face_id_token.mean().item():.4f}, std={face_id_token.std().item():.4f}")
            print(f"   Expression Token: mean={expression_token.mean().item():.4f}, std={expression_token.std().item():.4f}")
            
            # Print prediction if available
            if results.get('prediction') is not None:
                prediction = results['prediction']
                print(f"   🔮 Prediction: mean={prediction.mean().item():.4f}, std={prediction.std().item():.4f}")
    
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
    parser.add_argument("--face_id_checkpoint", type=str, required=True,
                       help="Path to Face ID model checkpoint")
    parser.add_argument("--expression_transformer_checkpoint", type=str, required=True,
                       help="Path to Expression Transformer checkpoint")
    parser.add_argument("--expression_predictor_checkpoint", type=str, required=True,
                       help="Path to Expression Predictor checkpoint")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to run models on (cpu/cuda)")
    parser.add_argument("--camera", type=int, default=0,
                       help="Camera index to use")
    parser.add_argument("--confidence", type=float, default=0.5,
                       help="Face detection confidence threshold")
    parser.add_argument("--no_print_tokens", action="store_true",
                       help="Disable token printing to console")
    
    args = parser.parse_args()
    
    # Create and run demo
    demo = WebcamDemo(
        face_id_checkpoint_path=args.face_id_checkpoint,
        expression_transformer_checkpoint_path=args.expression_transformer_checkpoint,
        expression_predictor_checkpoint_path=args.expression_predictor_checkpoint,
        device=args.device,
        confidence_threshold=args.confidence
    )
    
    demo.run(
        camera_index=args.camera,
        print_tokens=not args.no_print_tokens
    )


if __name__ == "__main__":
    main() 