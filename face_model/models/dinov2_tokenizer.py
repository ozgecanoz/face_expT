"""
Component A: DINOv2 Tokenizer (Frozen)
Extracts patch tokens from face images using pre-trained DINOv2
Supports both 518x518 and 224x224 input resolutions
Also supports facebook/dinov2-base model from Hugging Face
"""

import torch
import torch.nn as nn
import timm
import logging

logger = logging.getLogger(__name__)


class DINOv2Tokenizer(nn.Module):
    """
    Component A: Frozen DINOv2 tokenizer
    Supports two input resolutions:
    - 518×518×3 → 1,369 patch tokens (384-dim each) + 1 class token (384-dim)
    - 224×224×3 → 196 patch tokens (384-dim each) + 1 class token (384-dim)
    """
    
    def __init__(self, model_name='vit_small_patch14_dinov2.lvd142m', device='cpu'):
        super().__init__()
        
        logger.info(f"Loading DINOv2 model: {model_name}")
        
        # Load pre-trained DINOv2
        self.model = timm.create_model(model_name, pretrained=True)
        
        # Move model to specified device
        self.model = self.model.to(device)
        logger.info(f"DINOv2 model moved to device: {device}")

        # Check patch embed properties
        print(f"Model patch size: {self.model.patch_embed.patch_size}")
        print(f"Model grid size: {self.model.patch_embed.grid_size}")
        print(f"Model num patches: {self.model.patch_embed.num_patches}")
        print(f"Positional embedding shape: {self.model.pos_embed.shape}")
    
        # Determine input resolution and patch configuration based on model
        if 'patch16_224' in model_name:
            # 224x224 input with 16x16 patches
            self.input_size = 224
            self.patch_size = 16
            self.grid_size = 14  # 224/16 = 14
            self.num_patches = 196  # 14² = 196
            logger.info(f"Configured for 224x224 input: {self.grid_size}x{self.grid_size} grid, {self.num_patches} patches")
        else:
            # Default: 518x518 input with 14x14 patches
            self.input_size = 518
            self.patch_size = 14
            self.grid_size = 37  # 518/14 = 37
            self.num_patches = 1369  # 37² = 1369
            logger.info(f"Configured for 518x518 input: {self.grid_size}x{self.grid_size} grid, {self.num_patches} patches")
        
        # Test with appropriate input size
        test_input = torch.randn(1, 3, self.input_size, self.input_size).to(device)
        patch_embed = self.model.patch_embed(test_input)
        print(f"Actual patch embed shape: {patch_embed.shape}")
        
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
        
        # Extract components we need
        self.patch_embed = self.model.patch_embed
        self.pos_embed = self.model.pos_embed
        self.blocks = self.model.blocks
        self.norm = self.model.norm
        
        # Embedding dimension
        self.embed_dim = 384
        
        logger.info(f"DINOv2 tokenizer initialized with {self.num_patches} patches on device: {device}")
        logger.info(f"Input size: {self.input_size}x{self.input_size}, Patch size: {self.patch_size}x{self.patch_size}")
        
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) - Batch of face images (already normalized to [0, 1])
                H, W should match self.input_size (either 518 or 224)
        
        Returns:
            patch_tokens: (B, num_patches, 384) - Patch tokens only
            pos_embeddings: (B, num_patches, 384) - Positional embeddings for patches
        """
        B = x.shape[0]
        
        # Validate input dimensions
        expected_size = self.input_size
        if x.shape[2] != expected_size or x.shape[3] != expected_size:
            raise ValueError(f"Expected input size {expected_size}x{expected_size}, got {x.shape[2]}x{x.shape[3]}")
        
        # Apply ImageNet normalization
        # ImageNet mean and std values
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        
        # Normalize: (x - mean) / std
        x = (x - mean) / std
        
        # Get positional embeddings for patches (skip class token position)
        pos_emb = self.model.pos_embed[:, 1:, :].expand(B, -1, -1)  # (B, num_patches, 384)
        
        # Use model's forward_features method to get processed tokens through all transformer layers
        with torch.no_grad():
            # This handles class token, pos embedding, and transformer blocks
            x = self.model.forward_features(x)  # (B, num_patches+1, 384)
            
        # Extract only patch tokens (skip class token)
        patch_tokens = x[:, 1:, :]  # (B, num_patches, 384)
        
        return patch_tokens, pos_emb
    
    def get_patch_size(self):
        """Get the patch size used by the model"""
        return self.patch_size
    
    def get_grid_size(self):
        """Get the grid size (number of patches per side)"""
        return self.grid_size
    
    def get_num_patches(self):
        """Get the total number of patches"""
        return self.num_patches
    
    def get_input_size(self):
        """Get the expected input image size"""
        return self.input_size
    
    def get_embed_dim(self):
        """Get the embedding dimension"""
        return self.embed_dim


class DINOv2BaseTokenizer(nn.Module):
    """
    Component A: Frozen DINOv2 base tokenizer from Hugging Face
    Loads facebook/dinov2-base model
    - 518×518×3 → 1,369 patch tokens (768-dim each) + 1 class token (768-dim)
    """
    
    def __init__(self, device='cpu'):
        super().__init__()
        
        logger.info("Loading DINOv2 base model from Hugging Face: facebook/dinov2-base")
        
        try:
            from transformers import AutoModel
        except ImportError:
            raise ImportError("transformers library not found. Install with: pip install transformers")
        
        # Load pre-trained DINOv2 base model from Hugging Face
        self.model = AutoModel.from_pretrained('facebook/dinov2-base')
        
        # Move model to specified device
        self.model = self.model.to(device)
        logger.info(f"DINOv2 base model moved to device: {device}")

        # Configuration for facebook/dinov2-base
        self.input_size = 518
        self.patch_size = 14
        self.grid_size = 37  # 518/14 = 37
        self.num_patches = 1369  # 37² = 1369
        self.embed_dim = 768  # Base model uses 768 dimensions
        
        logger.info(f"Configured for 518x518 input: {self.grid_size}x{self.grid_size} grid, {self.num_patches} patches")
        logger.info(f"Embedding dimension: {self.embed_dim}")
        
        # Test with appropriate input size
        test_input = torch.randn(1, 3, self.input_size, self.input_size).to(device)
        with torch.no_grad():
            test_output = self.model(test_input)
        print(f"Test output shape: {test_output.last_hidden_state.shape}")
        
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
        
        logger.info(f"DINOv2 base tokenizer initialized with {self.num_patches} patches on device: {device}")
        logger.info(f"Input size: {self.input_size}x{self.input_size}, Patch size: {self.patch_size}x{self.patch_size}")
        
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) - Batch of face images (already normalized to [0, 1])
                H, W should match self.input_size (518)
        
        Returns:
            patch_tokens: (B, num_patches, 768) - Patch tokens only
            pos_embeddings: (B, num_patches, 768) - Positional embeddings for patches
        """
        B = x.shape[0]
        
        # Validate input dimensions
        expected_size = self.input_size
        if x.shape[2] != expected_size or x.shape[3] != expected_size:
            raise ValueError(f"Expected input size {expected_size}x{expected_size}, got {x.shape[2]}x{x.shape[3]}")
        
        # Apply ImageNet normalization
        # ImageNet mean and std values
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        
        # Normalize: (x - mean) / std
        x = (x - mean) / std
        
        # Forward pass through Hugging Face model
        with torch.no_grad():
            outputs = self.model(x)
            # Hugging Face model returns last_hidden_state with shape (B, num_patches+1, embed_dim)
            all_tokens = outputs.last_hidden_state  # (B, num_patches+1, 768)
        
        # Extract only patch tokens (skip class token)
        patch_tokens = all_tokens[:, 1:, :]  # (B, num_patches, 768)
        
        # For positional embeddings, we'll use the model's learned position embeddings
        # Note: Hugging Face models handle positional embeddings internally
        # We'll create a dummy positional embedding tensor for compatibility
        pos_emb = torch.zeros_like(patch_tokens)  # (B, num_patches, 768)
        
        return patch_tokens, pos_emb
    
    def get_patch_size(self):
        """Get the patch size used by the model"""
        return self.patch_size
    
    def get_grid_size(self):
        """Get the grid size (number of patches per side)"""
        return self.grid_size
    
    def get_num_patches(self):
        """Get the total number of patches"""
        return self.num_patches
    
    def get_input_size(self):
        """Get the expected input image size"""
        return self.input_size
    
    def get_embed_dim(self):
        """Get the embedding dimension"""
        return self.embed_dim


def test_dinov2_tokenizer():
    """Test the DINOv2 tokenizer with both model configurations"""
    import torch
    
    # Test 518x518 model (default)
    print("=== Testing 518x518 model ===")
    tokenizer_518 = DINOv2Tokenizer(device="cpu")
    
    # Create dummy input for 518x518
    batch_size = 2
    x_518 = torch.randn(batch_size, 3, 518, 518)
    
    # Forward pass
    patch_tokens_518, pos_emb_518 = tokenizer_518(x_518)
    
    print(f"Input shape: {x_518.shape}")
    print(f"Patch tokens shape: {patch_tokens_518.shape}")
    print(f"Positional embeddings shape: {pos_emb_518.shape}")
    
    # Verify shapes for 518x518
    assert patch_tokens_518.shape == (batch_size, 1369, 384)
    assert pos_emb_518.shape == (batch_size, 1369, 384)
    assert tokenizer_518.get_num_patches() == 1369
    assert tokenizer_518.get_input_size() == 518
    
    print("✅ 518x518 DINOv2 tokenizer test passed!")
    
    # Test 224x224 model
    print("\n=== Testing 224x224 model ===")
    tokenizer_224 = DINOv2Tokenizer(model_name='vit_small_patch16_224.augreg_in21k', device="cpu")
    
    # Create dummy input for 224x224
    x_224 = torch.randn(batch_size, 3, 224, 224)
    
    # Forward pass
    patch_tokens_224, pos_emb_224 = tokenizer_224(x_224)
    
    print(f"Input shape: {x_224.shape}")
    print(f"Patch tokens shape: {patch_tokens_224.shape}")
    print(f"Positional embeddings shape: {pos_emb_224.shape}")
    
    # Verify shapes for 224x224
    assert patch_tokens_224.shape == (batch_size, 196, 384)
    assert pos_emb_224.shape == (batch_size, 196, 384)
    assert tokenizer_224.get_num_patches() == 196
    assert tokenizer_224.get_input_size() == 224
    
    print("✅ 224x224 DINOv2 tokenizer test passed!")
    
    print("\n🎉 All DINOv2 tokenizer tests passed!")


def test_dinov2_base_tokenizer():
    """Test the DINOv2 base tokenizer from Hugging Face"""
    import torch
    
    print("=== Testing DINOv2 base tokenizer (facebook/dinov2-base) ===")
    
    try:
        tokenizer_base = DINOv2BaseTokenizer(device="cpu")
        
        # Create dummy input for 518x518
        batch_size = 2
        x_518 = torch.randn(batch_size, 3, 518, 518)
        
        # Forward pass
        patch_tokens_base, pos_emb_base = tokenizer_base(x_518)
        
        print(f"Input shape: {x_518.shape}")
        print(f"Patch tokens shape: {patch_tokens_base.shape}")
        print(f"Positional embeddings shape: {pos_emb_base.shape}")
        
        # Verify shapes for base model (768 dimensions)
        assert patch_tokens_base.shape == (batch_size, 1369, 768)
        assert pos_emb_base.shape == (batch_size, 1369, 768)
        assert tokenizer_base.get_num_patches() == 1369
        assert tokenizer_base.get_input_size() == 518
        assert tokenizer_base.get_embed_dim() == 768
        
        print("✅ DINOv2 base tokenizer test passed!")
        
    except ImportError as e:
        print(f"⚠️  Skipping DINOv2 base tokenizer test: {e}")
        print("   Install transformers with: pip install transformers")
    except Exception as e:
        print(f"❌ DINOv2 base tokenizer test failed: {e}")


def test_all_tokenizers():
    """Test all available DINOv2 tokenizer variants"""
    print("🧪 Testing all DINOv2 tokenizer variants...\n")
    
    # Test original tokenizers
    test_dinov2_tokenizer()
    
    print("\n" + "="*50 + "\n")
    
    # Test base tokenizer
    test_dinov2_base_tokenizer()
    
    print("\n🎉 All tokenizer tests completed!")


if __name__ == "__main__":
    test_all_tokenizers() 