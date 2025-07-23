"""
Component E: Face Reconstruction Model (Learnable)
Reconstructs face images from identity and expression tokens
Optimized for reduced parameter count (~70% reduction)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class FaceReconstructionModel(nn.Module):
    """
    Component E: Face Reconstruction Model (Optimized)
    Input: Patch tokens (1369×384) + Face ID token (1×384) + Expression token (1×384)
    Output: Reconstructed face image (518×518×3)
    """
    
    def __init__(self, embed_dim=384, num_heads=4, num_layers=1, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Transformer decoder for semantic fusion (reduced to 1 layer)
        self.transformer_decoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,  # Reduced from 8 to 4 heads
                dim_feedforward=embed_dim * 2,  # Reduced from 4x to 2x
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # CNN Decoder with optimized architecture
        self.cnn_decoder = OptimizedCNNDecoder(embed_dim)
        
        logger.info(f"Optimized Face Reconstruction Model initialized with {num_layers} transformer layer, {num_heads} heads")
        
    def forward(self, patch_tokens, pos_embeddings, face_id_token, expression_token):
        """
        Args:
            patch_tokens: (B, 1369, 384) - Patch tokens from Component A
            pos_embeddings: (B, 1369, 384) - Positional embeddings
            face_id_token: (B, 1, 384) - Face ID token from Component B
            expression_token: (B, 1, 384) - Expression token from Component C
        
        Returns:
            reconstructed_face: (B, 3, 518, 518) - Reconstructed face image
        """
        B, num_patches, embed_dim = patch_tokens.shape
        
        # Add positional embeddings to patch tokens
        patch_tokens_with_pos = patch_tokens + pos_embeddings
        
        # Combine face_id and expression tokens for K, V
        identity_expression = torch.cat([face_id_token, expression_token], dim=1)  # (B, 2, 384)
        
        # Cross-attention: patch tokens attend to [face_id, expression]
        # Concatenate patch tokens and identity_expression
        combined = torch.cat([patch_tokens_with_pos, identity_expression], dim=1)  # (B, 1371, 384)
        
        # Apply transformer decoder layers (only 1 layer now)
        transformed = combined
        for layer in self.transformer_decoder_layers:
            transformed = layer(transformed)
        
        # Extract transformed patch tokens (first 1369 tokens)
        transformed_patches = transformed[:, :num_patches, :]  # (B, 1369, 384)
        
        # Reshape to spatial grid (37×37×384)
        spatial_features = transformed_patches.view(B, 37, 37, embed_dim)  # (B, 37, 37, 384)
        spatial_features = spatial_features.permute(0, 3, 1, 2)  # (B, 384, 37, 37)
        
        # Pass through optimized CNN decoder
        reconstructed_face = self.cnn_decoder(spatial_features)  # (B, 3, 518, 518)
        
        return reconstructed_face


class OptimizedCNNDecoder(nn.Module):
    """
    Optimized CNN Decoder with reduced parameters
    Uses smaller channels, 3x3 kernels, and interpolation
    """
    
    def __init__(self, embed_dim=384):
        super().__init__()
        
        # Initial processing (reduced channels)
        self.initial_conv = nn.Conv2d(embed_dim, 256, kernel_size=3, padding=1)  # Reduced from 512
        self.initial_norm = nn.BatchNorm2d(256)
        self.initial_conv2 = nn.Conv2d(256, 512, kernel_size=3, padding=1)  # Reduced from 1024
        self.initial_norm2 = nn.BatchNorm2d(512)
        
        # Optimized upsampling layers with 3x3 kernels and reduced channels
        # First upsampling: 37 → 74
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)  # 3x3 instead of 4x4
        self.norm1 = nn.BatchNorm2d(256)
        
        # Second upsampling: 74 → 148  
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)  # 3x3 instead of 4x4
        self.norm2 = nn.BatchNorm2d(128)
        
        # Third upsampling: 148 → 296
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)  # 3x3 instead of 4x4
        self.norm3 = nn.BatchNorm2d(64)
        
        # Fourth upsampling: 296 → 518 (use interpolation instead of transpose conv)
        self.up4_conv = nn.Conv2d(64, 32, kernel_size=3, padding=1)  # Simple conv instead of transpose
        self.norm4 = nn.BatchNorm2d(32)
        
        # Final convolution to get 3 channels
        self.final_conv = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        
        # Activation function
        self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x):
        """
        Args:
            x: (B, 384, 37, 37) - Spatial features from transformer
        
        Returns:
            output: (B, 3, 518, 518) - Reconstructed face image
        """
        # Initial processing
        x = self.activation(self.initial_norm(self.initial_conv(x)))
        x = self.activation(self.initial_norm2(self.initial_conv2(x)))
        
        # Upsampling layers
        x = self.activation(self.norm1(self.up1(x)))      # (B, 256, 74, 74)
        x = self.activation(self.norm2(self.up2(x)))      # (B, 128, 148, 148)
        x = self.activation(self.norm3(self.up3(x)))      # (B, 64, 296, 296)
        
        # Use interpolation for final upsampling (more efficient than transpose conv)
        x = F.interpolate(x, size=(518, 518), mode='bilinear', align_corners=False)  # (B, 64, 518, 518)
        
        # Final processing
        x = self.activation(self.norm4(self.up4_conv(x)))  # (B, 32, 518, 518)
        x = self.final_conv(x)                            # (B, 3, 518, 518)
        
        # Apply sigmoid to get values in [0, 1] range
        x = torch.sigmoid(x)
        
        return x


def test_face_reconstruction_model():
    """Test the Optimized Face Reconstruction Model"""
    import torch
    
    # Create model
    model = FaceReconstructionModel()
    
    # Create dummy input
    batch_size = 2
    num_patches = 1369
    embed_dim = 384
    
    patch_tokens = torch.randn(batch_size, num_patches, embed_dim)
    pos_embeddings = torch.randn(batch_size, num_patches, embed_dim)
    face_id_token = torch.randn(batch_size, 1, embed_dim)
    expression_token = torch.randn(batch_size, 1, embed_dim)
    
    # Forward pass
    reconstructed_face = model(patch_tokens, pos_embeddings, face_id_token, expression_token)
    
    print(f"Patch tokens shape: {patch_tokens.shape}")
    print(f"Positional embeddings shape: {pos_embeddings.shape}")
    print(f"Face ID token shape: {face_id_token.shape}")
    print(f"Expression token shape: {expression_token.shape}")
    print(f"Reconstructed face shape: {reconstructed_face.shape}")
    
    # Check output range
    print(f"Output min: {reconstructed_face.min().item():.4f}")
    print(f"Output max: {reconstructed_face.max().item():.4f}")
    
    # Test with different batch sizes
    batch_size_2 = 1
    patch_tokens_2 = torch.randn(batch_size_2, num_patches, embed_dim)
    pos_embeddings_2 = torch.randn(batch_size_2, num_patches, embed_dim)
    face_id_token_2 = torch.randn(batch_size_2, 1, embed_dim)
    expression_token_2 = torch.randn(batch_size_2, 1, embed_dim)
    
    reconstructed_face_2 = model(patch_tokens_2, pos_embeddings_2, face_id_token_2, expression_token_2)
    print(f"Reconstructed face shape (batch_size={batch_size_2}): {reconstructed_face_2.shape}")
    
    print("✅ Optimized Face Reconstruction Model test passed!")


if __name__ == "__main__":
    test_face_reconstruction_model() 