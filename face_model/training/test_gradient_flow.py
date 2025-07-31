"""
Test gradient flow in expression reconstruction training
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train_expression_reconstruction_new import ExpressionReconstructionTrainer


def test_gradient_flow():
    """Test that gradients flow to the reconstruction model parameters"""
    
    print("🧪 Testing Gradient Flow...")
    
    # Create dummy checkpoint
    from models.expression_transformer import ExpressionTransformer
    dummy_model = ExpressionTransformer()
    dummy_checkpoint_path = "dummy_gradient_test.pt"
    torch.save({
        'model_state_dict': dummy_model.state_dict(),
        'epoch': 1
    }, dummy_checkpoint_path)
    
    try:
        # Initialize trainer
        trainer = ExpressionReconstructionTrainer(
            expression_transformer_checkpoint_path=dummy_checkpoint_path,
            device="cpu"
        )
        
        # Create dummy data
        num_frames = 5
        face_images = torch.randn(num_frames, 3, 518, 518)
        subject_ids = torch.randint(0, 100, (num_frames,))
        
        # Set model to training mode
        trainer.reconstruction_model.train()
        
        # Process clip
        expression_tokens, subject_embeddings = trainer.process_clip(face_images, subject_ids)
        
        # Reconstruct faces
        reconstructed_faces = trainer.reconstruct_faces(expression_tokens, subject_embeddings)
        
        # Compute loss
        criterion = nn.MSELoss()
        loss = criterion(reconstructed_faces, face_images)
        
        print(f"✅ Loss computed: {loss.item():.6f}")
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        has_gradients = False
        for name, param in trainer.reconstruction_model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if grad_norm > 0:
                    has_gradients = True
                    print(f"✅ Gradient flowing to {name}: norm = {grad_norm:.6f}")
        
        if has_gradients:
            print("✅ Gradient flow verified!")
        else:
            print("❌ No gradients detected!")
            return False
        
        # Check specific components
        patch_emb_grad = trainer.reconstruction_model.patch_embeddings.grad
        if patch_emb_grad is not None and patch_emb_grad.norm() > 0:
            print(f"✅ Patch embeddings gradients: norm = {patch_emb_grad.norm().item():.6f}")
        else:
            print("❌ No gradients to patch embeddings!")
            return False
        
        # Check that expression transformer is frozen
        expr_transformer_has_grad = False
        for param in trainer.expression_transformer.parameters():
            if param.grad is not None:
                expr_transformer_has_grad = True
                break
        
        if not expr_transformer_has_grad:
            print("✅ Expression transformer properly frozen (no gradients)")
        else:
            print("❌ Expression transformer not properly frozen!")
            return False
        
        print("\n🎉 Gradient flow test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists(dummy_checkpoint_path):
            os.remove(dummy_checkpoint_path)


if __name__ == "__main__":
    success = test_gradient_flow()
    if success:
        print("✅ All gradient flow tests passed!")
    else:
        print("❌ Gradient flow tests failed!")
        exit(1) 