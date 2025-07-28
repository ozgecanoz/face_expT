"""
Token Extractor for Webcam Demo
Efficiently extracts expression tokens using cached DINOv2 tokens and subject embeddings
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
    """Extracts tokens from face images and predicts next expression token using subject embeddings"""
    
    def __init__(self, expression_transformer, expression_predictor, face_reconstruction_model, tokenizer, device="cpu", subject_id=0):
        self.expression_transformer = expression_transformer
        self.expression_predictor = expression_predictor
        self.face_reconstruction_model = face_reconstruction_model
        self.tokenizer = tokenizer
        self.device = device
        self.subject_id = subject_id  # Subject ID for the current user
        
        # Initialize circular buffer for 30 frames
        self.token_buffer = TokenBuffer(buffer_size=30)
    
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
        
        # Create subject ID tensor
        subject_ids = torch.tensor([self.subject_id], dtype=torch.long, device=self.device)
        
        # Extract expression token and subject embeddings using inference method
        expression_token, subject_embeddings = self.expression_transformer.inference(patch_tokens, pos_embeddings, subject_ids)
        
        return {
            'expression_token': expression_token,
            'subject_embeddings': subject_embeddings,
            'patch_tokens': patch_tokens,
            'pos_embeddings': pos_embeddings,
            'subject_ids': subject_ids
        }
    
    def _extract_expression_token(self, patch_tokens: torch.Tensor, 
                                pos_embeddings: torch.Tensor,
                                subject_ids: torch.Tensor) -> torch.Tensor:
        """
        Extract expression token using cached DINOv2 tokens and subject embeddings
        
        Args:
            patch_tokens: DINOv2 patch tokens (1, 1369, 384)
            pos_embeddings: Positional embeddings (1, 1369, 384)
            subject_ids: Subject IDs (1,)
            
        Returns:
            Expression token (1, 1, 384)
        """
        with torch.no_grad():
            # Use the new inference method for cached tokens with subject embeddings
            expression_token, _ = self.expression_transformer.inference(patch_tokens, pos_embeddings, subject_ids)
        
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
    
    def reconstruct_face(self, subject_embeddings: torch.Tensor, expression_token: torch.Tensor, 
                        patch_tokens: torch.Tensor, pos_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct face image from subject embeddings and expression tokens using actual DINOv2 tokens
        
        Args:
            subject_embeddings: (1, 1, 384) - Subject embeddings from Expression Transformer
            expression_token: (1, 1, 384) - Expression token
            patch_tokens: (1, 1369, 384) - DINOv2 patch tokens from input frame
            pos_embeddings: (1, 1369, 384) - Positional embeddings from input frame
            
        Returns:
            reconstructed_face: (1, 3, 518, 518) - Reconstructed face image
        """
        with torch.no_grad():
            # Use actual DINOv2 tokens from the input frame and subject embeddings from Expression Transformer
            reconstructed_face = self.face_reconstruction_model(
                patch_tokens, pos_embeddings, subject_embeddings, expression_token
            )
        
        return reconstructed_face
    
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
        
        # Initialize circular buffers (removed face_id_tokens since we use subject embeddings)
        self.expression_tokens = torch.zeros(buffer_size, 1, token_dim)
        
        # Track current position and frame count
        self.current_index = 0
        self.frame_count = 0
        
        logger.info(f"Token Buffer initialized with size {buffer_size}")
    
    def add_frame_tokens(self, expression_token: torch.Tensor) -> None:
        """
        Add expression token to the buffer
        
        Args:
            expression_token: Expression token (1, 1, 384)
        """
        # Store expression token
        self.expression_tokens[self.current_index] = expression_token
        
        # Update indices
        self.current_index = (self.current_index + 1) % self.buffer_size
        self.frame_count = min(self.frame_count + 1, self.buffer_size)
        
        logger.debug(f"Added expression token to buffer at index {self.current_index}")
    
    def get_prediction_sequence(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the sequence of tokens for prediction (last 29 frames)
        
        Returns:
            Tuple of (expression_tokens, None) for compatibility
        """
        if self.frame_count < 29:
            return None, None
        
        # Get the last 29 expression tokens
        start_idx = (self.current_index - 29) % self.buffer_size
        end_idx = self.current_index
        
        if start_idx < end_idx:
            # No wrap-around
            expression_sequence = self.expression_tokens[start_idx:end_idx]
        else:
            # Wrap-around case
            expression_sequence = torch.cat([
                self.expression_tokens[start_idx:],
                self.expression_tokens[:end_idx]
            ], dim=0)
        
        return expression_sequence, None  # Return None for face_id_tokens for compatibility
    
    def is_ready_for_prediction(self) -> bool:
        """Check if we have enough frames for prediction"""
        return self.frame_count >= 29
    
    def get_frame_count(self) -> int:
        """Get current frame count"""
        return self.frame_count
    
    def reset(self):
        """Reset the buffer"""
        self.expression_tokens.zero_()
        self.current_index = 0
        self.frame_count = 0
        logger.info("Token Buffer reset")


def test_token_extractor():
    """Test the token extractor"""
    print("🧪 Testing Token Extractor...")
    
    # Create dummy models
    class DummyModel:
        def __init__(self, name):
            self.name = name
        
        def inference(self, *args):
            return torch.randn(1, 1, 384), torch.randn(1, 1, 384) # Changed to return subject embeddings
    
    class DummyTokenizer:
        def __call__(self, face_tensor):
            return torch.randn(1, 1369, 384), torch.randn(1, 1369, 384)
    
    # Create dummy models
    expression_transformer = DummyModel('expression')
    expression_predictor = DummyModel('predictor')
    face_reconstruction_model = DummyModel('reconstruction')
    tokenizer = DummyTokenizer()
    
    # Create token extractor
    extractor = TokenExtractor(
        expression_transformer=expression_transformer,
        expression_predictor=expression_predictor,
        face_reconstruction_model=face_reconstruction_model,
        tokenizer=tokenizer,
        device="cpu",
        subject_id=0  # Use subject ID 0 for testing
    )
    print("✅ Token Extractor created successfully")
    
    # Create dummy face image
    dummy_face = np.random.randint(0, 255, (518, 518, 3), dtype=np.uint8)
    
    # Test token extraction
    tokens = extractor.process_frame_tokens(dummy_face)
    if tokens is not None:
        print("✅ Token extraction completed successfully")
        print(f"   Extracted tokens: {list(tokens.keys())}")
    else:
        print("❌ Token extraction failed")
        return False
    
    # Test token buffer
    buffer = TokenBuffer(buffer_size=30)
    print("✅ Token Buffer created successfully")
    
    # Add some dummy tokens
    for i in range(35):
        expression_token = torch.randn(1, 1, 384)
        buffer.add_frame_tokens(expression_token)
        
        if buffer.is_ready_for_prediction():
            expr_seq, _ = buffer.get_prediction_sequence()
            print(f"Frame {i}: Ready for prediction, sequence shape: {expr_seq.shape}")
        else:
            print(f"Frame {i}: Not ready for prediction")
    
    # Test reset
    buffer.reset()
    print("✅ Token Buffer reset successfully")
    
    print("✅ Token Extractor test passed!")
    return True


if __name__ == "__main__":
    test_token_extractor() 