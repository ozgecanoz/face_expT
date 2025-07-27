"""
Token Extractor for Webcam Demo
Efficiently extracts face ID and expression tokens using cached DINOv2 tokens
"""

import torch
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional

# Add parent directory to path for imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


class TokenExtractor:
    """Efficiently extracts tokens from face images using cached DINOv2 tokens"""
    
    def __init__(self, models: Dict[str, Any], tokenizer: Any, device: str = "cpu"):
        """
        Initialize token extractor
        
        Args:
            models: Dictionary containing loaded models
            tokenizer: DINOv2 tokenizer
            device: Device to run models on
        """
        self.models = models
        self.tokenizer = tokenizer
        self.device = device
        
        # Extract models
        self.face_id_model = models['face_id_model']
        self.expression_transformer = models['expression_transformer']
        self.expression_predictor = models['expression_predictor']
        
        logger.info("Token Extractor initialized")
    
    def extract_tokens_from_face(self, face_image: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Extract all tokens from a single face image
        
        Args:
            face_image: Face image (518x518x3) in RGB format
            
        Returns:
            Dictionary containing extracted tokens
        """
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
        
        # Extract DINOv2 tokens (cached for reuse)
        with torch.no_grad():
            patch_tokens, pos_embeddings = self.tokenizer(face_tensor)
        
        # Extract face ID token
        face_id_token = self._extract_face_id_token(patch_tokens, pos_embeddings)
        
        # Extract expression token
        expression_token = self._extract_expression_token(patch_tokens, pos_embeddings, face_id_token)
        
        return {
            'patch_tokens': patch_tokens,
            'pos_embeddings': pos_embeddings,
            'face_id_token': face_id_token,
            'expression_token': expression_token
        }
    
    def _extract_face_id_token(self, patch_tokens: torch.Tensor, 
                              pos_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Extract face ID token using cached DINOv2 tokens
        
        Args:
            patch_tokens: DINOv2 patch tokens (1, 1369, 384)
            pos_embeddings: Positional embeddings (1, 1369, 384)
            
        Returns:
            Face ID token (1, 1, 384)
        """
        with torch.no_grad():
            # Use the new inference method for cached tokens
            face_id_token = self.face_id_model.inference(patch_tokens, pos_embeddings)
        
        return face_id_token
    
    def _extract_expression_token(self, patch_tokens: torch.Tensor, 
                                pos_embeddings: torch.Tensor,
                                face_id_token: torch.Tensor) -> torch.Tensor:
        """
        Extract expression token using cached DINOv2 tokens
        
        Args:
            patch_tokens: DINOv2 patch tokens (1, 1369, 384)
            pos_embeddings: Positional embeddings (1, 1369, 384)
            face_id_token: Face ID token (1, 1, 384)
            
        Returns:
            Expression token (1, 1, 384)
        """
        with torch.no_grad():
            # Use the new inference method for cached tokens
            expression_token = self.expression_transformer.inference(
                patch_tokens, pos_embeddings, face_id_token
            )
        
        return expression_token
    
    def predict_next_expression(self, expression_tokens: torch.Tensor) -> torch.Tensor:
        """
        Predict next expression token using the expression predictor
        
        Args:
            expression_tokens: Sequence of expression tokens (seq_len, 1, 384)
            
        Returns:
            Predicted expression token (1, 1, 384)
        """
        with torch.no_grad():
            # Use the existing _forward_single_clip method
            predicted_token = self.expression_predictor._forward_single_clip(expression_tokens)
        
        return predicted_token
    
    def process_frame_tokens(self, face_image: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Process a frame and extract all tokens
        
        Args:
            face_image: Face image (518x518x3) in RGB format
            
        Returns:
            Dictionary containing all extracted tokens
        """
        try:
            tokens = self.extract_tokens_from_face(face_image)
            return tokens
        except Exception as e:
            logger.error(f"Error extracting tokens: {e}")
            return None


class TokenBuffer:
    """Circular buffer for storing tokens efficiently"""
    
    def __init__(self, buffer_size: int = 30, token_dim: int = 384):
        """
        Initialize token buffer
        
        Args:
            buffer_size: Number of frames to store
            token_dim: Dimension of tokens
        """
        self.buffer_size = buffer_size
        self.token_dim = token_dim
        
        # Initialize circular buffers
        self.face_id_tokens = torch.zeros(buffer_size, 1, token_dim)
        self.expression_tokens = torch.zeros(buffer_size, 1, token_dim)
        
        # Track current position and frame count
        self.current_index = 0
        self.frame_count = 0
        
        logger.info(f"Token Buffer initialized with size {buffer_size}")
    
    def add_frame_tokens(self, face_id_token: torch.Tensor, 
                        expression_token: torch.Tensor) -> None:
        """
        Add tokens for a new frame
        
        Args:
            face_id_token: Face ID token (1, 1, 384)
            expression_token: Expression token (1, 1, 384)
        """
        # Store tokens in circular buffer
        self.face_id_tokens[self.current_index] = face_id_token.squeeze(0)
        self.expression_tokens[self.current_index] = expression_token.squeeze(0)
        
        # Update indices
        self.current_index = (self.current_index + 1) % self.buffer_size
        self.frame_count += 1
    
    def get_prediction_sequence(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the sequence of tokens for prediction (last 29 frames)
        
        Returns:
            Tuple of (face_id_sequence, expression_sequence) for prediction
        """
        if self.frame_count < 29:
            # Not enough frames for prediction
            return None, None
        
        # Get the last 29 frames (excluding current frame)
        start_idx = (self.current_index - 29) % self.buffer_size
        end_idx = (self.current_index - 1) % self.buffer_size
        
        if start_idx <= end_idx:
            # Normal sequence
            face_id_sequence = self.face_id_tokens[start_idx:end_idx+1]
            expression_sequence = self.expression_tokens[start_idx:end_idx+1]
        else:
            # Wrapped sequence
            face_id_sequence = torch.cat([
                self.face_id_tokens[start_idx:],
                self.face_id_tokens[:end_idx+1]
            ], dim=0)
            expression_sequence = torch.cat([
                self.expression_tokens[start_idx:],
                self.expression_tokens[:end_idx+1]
            ], dim=0)
        
        return face_id_sequence, expression_sequence
    
    def is_ready_for_prediction(self) -> bool:
        """Check if we have enough frames for prediction"""
        return self.frame_count >= 29
    
    def get_frame_count(self) -> int:
        """Get current frame count"""
        return self.frame_count
    
    def reset(self):
        """Reset the buffer"""
        self.face_id_tokens.zero_()
        self.expression_tokens.zero_()
        self.current_index = 0
        self.frame_count = 0
        logger.info("Token Buffer reset")


def test_token_extractor():
    """Test the token extractor with dummy data"""
    print("🧪 Testing Token Extractor...")
    
    # Create dummy models (these won't work, but we can test the structure)
    class DummyModel:
        def __init__(self, name):
            self.name = name
        
        def inference(self, *args):
            return torch.randn(1, 1, 384)
    
    # Create dummy tokenizer
    class DummyTokenizer:
        def __call__(self, face_tensor):
            return torch.randn(1, 1369, 384), torch.randn(1, 1369, 384)
    
    # Create dummy models
    models = {
        'face_id_model': DummyModel('face_id'),
        'expression_transformer': DummyModel('expression'),
        'expression_predictor': DummyModel('predictor')
    }
    
    tokenizer = DummyTokenizer()
    
    # Create token extractor
    extractor = TokenExtractor(models, tokenizer, device="cpu")
    
    # Create dummy face image
    dummy_face = np.random.randint(0, 255, (518, 518, 3), dtype=np.uint8)
    
    # Test token extraction
    try:
        tokens = extractor.process_frame_tokens(dummy_face)
        print("✅ Token extraction test passed!")
        print(f"Extracted tokens: {list(tokens.keys())}")
    except Exception as e:
        print(f"❌ Token extraction test failed: {e}")
    
    # Test token buffer
    print("\n🧪 Testing Token Buffer...")
    buffer = TokenBuffer(buffer_size=30)
    
    # Add some dummy tokens
    for i in range(35):
        face_id_token = torch.randn(1, 1, 384)
        expression_token = torch.randn(1, 1, 384)
        buffer.add_frame_tokens(face_id_token, expression_token)
        
        if buffer.is_ready_for_prediction():
            face_id_seq, expr_seq = buffer.get_prediction_sequence()
            print(f"Frame {i}: Ready for prediction, sequence shape: {face_id_seq.shape}")
        else:
            print(f"Frame {i}: Not ready for prediction")
    
    print("✅ Token Buffer test completed!")


if __name__ == "__main__":
    test_token_extractor() 