"""
Model Loader for Webcam Demo
Loads Expression Transformer, Expression Predictor, and Face Reconstruction models from checkpoints
Uses subject embeddings instead of face ID model
"""

import torch
import os
import logging
from typing import Dict, Any, Tuple

# Add parent directory to path for imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from face_model.models.dinov2_tokenizer import DINOv2Tokenizer
from face_model.models.expression_transformer import ExpressionTransformer
from face_model.models.expression_transformer_decoder import ExpressionTransformerDecoder
from face_model.models.face_reconstruction_model import FaceReconstructionModel

logger = logging.getLogger(__name__)


class ModelLoader:
    """Loads and manages all models for the webcam demo"""
    
    def __init__(self, device="cpu"):
        self.device = device
        self.models = {}
        self.tokenizer = None
        
    def load_all_models(self, 
                        expression_transformer_checkpoint_path: str,
                        expression_predictor_checkpoint_path: str,
                        face_reconstruction_checkpoint_path: str) -> Dict[str, Any]:
        """
        Load all models from checkpoints
        
        Args:
            expression_transformer_checkpoint_path: Path to Expression Transformer checkpoint
            expression_predictor_checkpoint_path: Path to Expression Predictor checkpoint
            face_reconstruction_checkpoint_path: Path to Face Reconstruction model checkpoint
            
        Returns:
            Dictionary containing all loaded models and tokenizer
        """
        logger.info("🚀 Loading all models for webcam demo...")
        
        # Load DINOv2 tokenizer
        logger.info("📦 Loading DINOv2 tokenizer...")
        self.tokenizer = DINOv2Tokenizer()
        logger.info("✅ DINOv2 tokenizer loaded")
        
        # Load Expression Transformer
        logger.info("😊 Loading Expression Transformer...")
        expression_transformer = self._load_expression_transformer(expression_transformer_checkpoint_path)
        self.models['expression_transformer'] = expression_transformer
        logger.info("✅ Expression Transformer loaded")
        
        # Load Expression Predictor
        logger.info("🔮 Loading Expression Predictor...")
        expression_predictor = self._load_expression_predictor(expression_predictor_checkpoint_path)
        self.models['expression_predictor'] = expression_predictor
        logger.info("✅ Expression Predictor loaded")
        
        # Load Face Reconstruction model
        logger.info("🎨 Loading Face Reconstruction model...")
        face_reconstruction_model = self._load_face_reconstruction_model(face_reconstruction_checkpoint_path)
        self.models['face_reconstruction'] = face_reconstruction_model
        logger.info("✅ Face Reconstruction model loaded")
        
        logger.info("🎉 All models loaded successfully!")
        
        return {
            'tokenizer': self.tokenizer,
            'expression_transformer': expression_transformer,
            'expression_predictor': expression_predictor,
            'face_reconstruction_model': face_reconstruction_model
        }
    
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
            max_subjects = expr_config.get('max_subjects', 3500)  # Load from checkpoint
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
            max_subjects=max_subjects  # Use max_subjects from checkpoint
        ).to(self.device)
        
        # Load state dict
        if 'expression_transformer_state_dict' in checkpoint:
            state_dict = checkpoint['expression_transformer_state_dict']
            
            # Check for subject_embeddings.weight - required for demo
            if 'subject_embeddings.weight' not in state_dict:
                raise RuntimeError(
                    f"Checkpoint {checkpoint_path} is missing 'subject_embeddings.weight'. "
                    f"This checkpoint was trained with the old architecture (using face ID model). "
                    f"Please train a new Expression Transformer with subject embeddings for the demo to work."
                )
            
            expression_transformer.load_state_dict(state_dict)
            logger.info(f"Loaded Expression Transformer from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            # Try loading the entire checkpoint as state dict
            state_dict = checkpoint
            
            # Check for subject_embeddings.weight - required for demo
            if 'subject_embeddings.weight' not in state_dict:
                raise RuntimeError(
                    f"Checkpoint {checkpoint_path} is missing 'subject_embeddings.weight'. "
                    f"This checkpoint was trained with the old architecture (using face ID model). "
                    f"Please train a new Expression Transformer with subject embeddings for the demo to work."
                )
            
            expression_transformer.load_state_dict(state_dict)
            logger.info("Loaded Expression Transformer state dict directly")
        
        # Set to evaluation mode
        expression_transformer.eval()
        
        return expression_transformer
    
    def _load_expression_predictor(self, checkpoint_path: str) -> ExpressionTransformerDecoder:
        """Load Expression Predictor with architecture from checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Expression Predictor checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint to get architecture
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Get architecture from checkpoint config
        if 'config' in checkpoint and 'transformer_decoder' in checkpoint['config']:
            pred_config = checkpoint['config']['transformer_decoder']
            embed_dim = pred_config.get('embed_dim', 384)
            num_heads = pred_config.get('num_heads', 4)
            num_layers = pred_config.get('num_layers', 2)
            dropout = pred_config.get('dropout', 0.1)
            max_sequence_length = pred_config.get('max_sequence_length', 50)
            logger.info(f"📐 Expression Predictor architecture: {num_layers} layers, {num_heads} heads")
        else:
            # Fallback to default architecture
            embed_dim, num_heads, num_layers, dropout, max_sequence_length = 384, 4, 2, 0.1, 50
            logger.warning("No architecture config found in Expression Predictor checkpoint, using defaults")
        
        # Initialize model with correct architecture
        expression_predictor = ExpressionTransformerDecoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            max_sequence_length=max_sequence_length
        ).to(self.device)
        
        # Load state dict
        if 'transformer_decoder_state_dict' in checkpoint:
            expression_predictor.load_state_dict(checkpoint['transformer_decoder_state_dict'])
            logger.info(f"Loaded Expression Predictor from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            # Try loading the entire checkpoint as state dict
            expression_predictor.load_state_dict(checkpoint)
            logger.info("Loaded Expression Predictor state dict directly")
        
        # Set to evaluation mode
        expression_predictor.eval()
        
        return expression_predictor
    
    def _load_face_reconstruction_model(self, checkpoint_path: str) -> FaceReconstructionModel:
        """Load Face Reconstruction model with architecture from checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Face Reconstruction checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint to get architecture
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Get architecture from checkpoint config
        if 'config' in checkpoint and 'reconstruction_model' in checkpoint['config']:
            recon_config = checkpoint['config']['reconstruction_model']
            embed_dim = recon_config.get('embed_dim', 384)
            num_heads = recon_config.get('num_heads', 4)
            num_layers = recon_config.get('num_layers', 2)
            dropout = recon_config.get('dropout', 0.1)
            logger.info(f"📐 Face Reconstruction architecture: {num_layers} layers, {num_heads} heads")
        else:
            # Fallback to default architecture
            embed_dim, num_heads, num_layers, dropout = 384, 4, 2, 0.1
            logger.warning("No architecture config found in Face Reconstruction checkpoint, using defaults")
        
        # Initialize model with correct architecture
        face_reconstruction_model = FaceReconstructionModel(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        ).to(self.device)
        
        # Load state dict
        if 'reconstruction_model_state_dict' in checkpoint:
            face_reconstruction_model.load_state_dict(checkpoint['reconstruction_model_state_dict'])
            logger.info(f"Loaded Face Reconstruction model from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            # Try loading the entire checkpoint as state dict
            face_reconstruction_model.load_state_dict(checkpoint)
            logger.info("Loaded Face Reconstruction model state dict directly")
        
        # Set to evaluation mode
        face_reconstruction_model.eval()
        
        return face_reconstruction_model
    
    def get_models(self) -> Dict[str, Any]:
        """Get all loaded models"""
        return self.models
    
    def get_tokenizer(self) -> DINOv2Tokenizer:
        """Get the DINOv2 tokenizer"""
        return self.tokenizer


def test_model_loader():
    """Test the model loader"""
    print("🧪 Testing Model Loader...")
    
    # Create model loader
    loader = ModelLoader(device="cpu")
    
    # Test with dummy checkpoint paths (these won't exist, but we can test the structure)
    try:
        models = loader.load_all_models(
            expression_transformer_checkpoint_path="/Users/ozgewhiting/Documents/projects/dataset_utils/cloud_checkpoints/expression_transformer_epoch_5.pt",
            expression_predictor_checkpoint_path="/Users/ozgewhiting/Documents/projects/dataset_utils/cloud_checkpoints/transformer_decoder_epoch_5.pt",
            face_reconstruction_checkpoint_path="/Users/ozgewhiting/Documents/projects/dataset_utils/cloud_checkpoints/reconstruction_model_epoch_5.pt"
        )
        print("❌ Expected FileNotFoundError but got success")
    except FileNotFoundError as e:
        print("✅ Correctly caught FileNotFoundError for missing checkpoints")
        print(f"Error: {e}")
    
    print("✅ Model Loader test completed!")


if __name__ == "__main__":
    test_model_loader() 