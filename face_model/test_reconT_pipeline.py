#!/usr/bin/env python3
"""
Test script for the reconstruction training pipeline
Tests the entire pipeline with dummy data to ensure everything works correctly
"""

import torch
import torch.nn as nn
import os
import tempfile
import json
import numpy as np
from pathlib import Path

# Add the project root to the path
import sys
sys.path.append('.')

from training.train_reconT import (
    train_expression_reconstruction, 
    load_pca_projection, 
    prepare_reconstruction_data,
    ReconstructionLoss
)
from models.expression_transformer import ExpressionTransformer
from models.expression_reconstruction_model import ExpressionReconstructionModel
from models.dinov2_tokenizer import DINOv2BaseTokenizer
from data.dataset import FaceDataset


def create_dummy_cca_dataset(output_dir: str, num_clips: int = 4):
    """
    Create dummy CCA dataset for testing
    
    Args:
        output_dir: Directory to save dummy dataset
        num_clips: Number of dummy clips to create
    """
    print(f"🧪 Creating dummy CCA dataset with {num_clips} clips...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dummy H5 files
    for clip_idx in range(num_clips):
        h5_path = os.path.join(output_dir, f"dummy_clip_{clip_idx:03d}.h5")
        
        # Create dummy data structure
        import h5py
        
        with h5py.File(h5_path, 'w') as f:
            # Create frames data (30 frames, 3 channels, 518x518)
            frames_data = np.random.randint(0, 256, (30, 3, 518, 518), dtype=np.uint8)
            
            # Create metadata group with subject_id
            metadata_group = f.create_group('metadata')
            metadata_group.create_dataset('subject_id', data=f'dummy_subject_{clip_idx}'.encode('utf-8'))
            
            # Create faces group structure
            faces_group = f.create_group('faces')
            
            for frame_idx in range(30):
                frame_group = faces_group.create_group(f'frame_{frame_idx:03d}')
                face_group = frame_group.create_group('face_000')
                
                # Add frame data (transpose to match expected format: (518, 518, 3))
                frame_data = frames_data[frame_idx].transpose(1, 2, 0)  # (518, 518, 3)
                face_group.create_dataset('data', data=frame_data, dtype=np.uint8)
        
        print(f"  ✅ Created {h5_path}")
    
    print(f"✅ Dummy CCA dataset created in {output_dir}")
    return output_dir


def create_dummy_pca_json(output_path: str):
    """
    Create dummy PCA projection JSON file
    
    Args:
        output_path: Path to save dummy PCA JSON
    """
    print(f"🧪 Creating dummy PCA projection JSON...")
    
    # Create dummy PCA data
    pca_data = {
        'pca_components': np.random.randn(384, 768).astype(np.float32).tolist(),
        'pca_mean': np.random.randn(768).astype(np.float32).tolist()
    }
    
    with open(output_path, 'w') as f:
        json.dump(pca_data, f, indent=2)
    
    print(f"✅ Dummy PCA JSON created: {output_path}")
    return output_path


def create_dummy_expression_transformer_checkpoint(output_path: str):
    """
    Create dummy ExpressionTransformer checkpoint
    
    Args:
        output_path: Path to save dummy checkpoint
    """
    print(f"🧪 Creating dummy ExpressionTransformer checkpoint...")
    
    # Create dummy model
    model = ExpressionTransformer(
        embed_dim=384,
        num_heads=8,
        num_layers=2,
        dropout=0.1,
        ff_dim=1536,
        grid_size=37
    )
    
    # Create dummy checkpoint
    checkpoint = {
        'expression_transformer_state_dict': model.state_dict(),
        'config': {
            'expression_model': {
                'expr_embed_dim': 384,
                'expr_num_heads': 8,
                'expr_num_layers': 2,
                'expr_dropout': 0.1,
                'expr_ff_dim': 1536,
                'expr_grid_size': 37
            }
        }
    }
    
    # Save checkpoint
    torch.save(checkpoint, output_path)
    
    print(f"✅ Dummy ExpressionTransformer checkpoint created: {output_path}")
    return output_path


def test_reconstruction_loss():
    """Test the reconstruction loss function"""
    print("🧪 Testing ReconstructionLoss...")
    
    criterion = ReconstructionLoss(lambda_mse=1.0, lambda_l1=0.1)
    
    # Create dummy data
    batch_size = 2
    reconstructed = torch.randn(batch_size, 3, 518, 518)
    target = torch.randn(batch_size, 3, 518, 518)
    
    # Compute loss
    total_loss, mse_loss, l1_loss = criterion(reconstructed, target)
    
    print(f"  Total Loss: {total_loss.item():.6f}")
    print(f"  MSE Loss: {mse_loss.item():.6f}")
    print(f"  L1 Loss: {l1_loss.item():.6f}")
    
    assert total_loss.item() > 0, "Loss should be positive"
    print("✅ ReconstructionLoss test passed!")


def test_pca_loading():
    """Test PCA projection loading"""
    print("🧪 Testing PCA projection loading...")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_pca_path = f.name
    
    try:
        # Create dummy PCA data
        pca_data = {
            'pca_components': np.random.randn(384, 768).astype(np.float32).tolist(),
            'pca_mean': np.random.randn(768).astype(np.float32).tolist()
        }
        
        with open(temp_pca_path, 'w') as f:
            json.dump(pca_data, f)
        
        # Test loading
        pca_components, pca_mean = load_pca_projection(temp_pca_path)
        
        assert pca_components.shape == (384, 768), f"Expected shape (384, 768), got {pca_components.shape}"
        assert pca_mean.shape == (768,), f"Expected shape (768,), got {pca_mean.shape}"
        
        print("✅ PCA loading test passed!")
        
    finally:
        os.unlink(temp_pca_path)


def test_data_preparation():
    """Test data preparation function"""
    print("🧪 Testing data preparation...")
    
    # Create dummy data
    batch = {
        'frames': [
            torch.randn(30, 3, 518, 518),  # First clip
            torch.randn(30, 3, 518, 518)   # Second clip (same length)
        ],
        'subject_id': ['subject_1', 'subject_2']
    }
    
    # Create dummy components
    dinov2_tokenizer = DINOv2BaseTokenizer(device=torch.device("cpu"))
    pca_components = torch.randn(384, 768)
    pca_mean = torch.randn(768)
    
    # Create dummy frozen transformer
    expression_transformer = ExpressionTransformer(
        embed_dim=384,
        num_heads=8,
        num_layers=2,
        dropout=0.1,
        ff_dim=1536,
        grid_size=37
    )
    expression_transformer.requires_grad = False
    
    device = torch.device("cpu")
    
    # Test data preparation
    expression_tokens, subject_ids, original_frames, final_pos_embeddings, clip_lengths = prepare_reconstruction_data(
        batch, dinov2_tokenizer, pca_components.numpy(), pca_mean.numpy(), 
        expression_transformer, device
    )
    
    # Verify shapes
    batch_size = 2
    frames_per_clip = 30
    assert expression_tokens.shape == (batch_size, frames_per_clip, 1, 384), f"Expected shape ({batch_size}, {frames_per_clip}, 1, 384), got {expression_tokens.shape}"
    assert original_frames.shape == (batch_size, frames_per_clip, 3, 518, 518), f"Expected shape ({batch_size}, {frames_per_clip}, 3, 518, 518), got {original_frames.shape}"
    assert final_pos_embeddings.shape == (batch_size, frames_per_clip, 1369, 384), f"Expected shape ({batch_size}, {frames_per_clip}, 1369, 384), got {final_pos_embeddings.shape}"
    assert len(subject_ids) == 2, f"Expected 2 subject IDs, got {len(subject_ids)}"
    assert clip_lengths == [frames_per_clip, frames_per_clip], f"Expected clip lengths [{frames_per_clip}, {frames_per_clip}], got {clip_lengths}"
    
    print("✅ Data preparation test passed!")


def test_mini_training_pipeline():
    """Test a mini training pipeline with dummy data"""
    print("🧪 Testing mini training pipeline...")
    
    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        # Create dummy files
        dataset_path = temp_dir / "dummy_cca"
        pca_json_path = temp_dir / "dummy_pca.json"
        checkpoint_path = temp_dir / "dummy_expT.pt"
        
        # Create dummy data
        create_dummy_cca_dataset(str(dataset_path), num_clips=2)
        create_dummy_pca_json(str(pca_json_path))
        create_dummy_expression_transformer_checkpoint(str(checkpoint_path))
        
        # Test mini training (just a few steps)
        try:
            model = train_expression_reconstruction(
                dataset_path=str(dataset_path),
                pca_json_path=str(pca_json_path),
                expression_transformer_checkpoint_path=str(checkpoint_path),
                checkpoint_dir=str(temp_dir / "checkpoints"),
                save_every_step=5,
                batch_size=2,
                num_epochs=1,
                learning_rate=1e-4,
                max_samples=4,
                device=torch.device("cpu"),
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
                drop_last=False,
                warmup_steps=10,
                min_lr=1e-6,
                embed_dim=384,
                num_cross_attention_layers=1,
                num_self_attention_layers=1,
                num_heads=4,
                dropout=0.1,
                ff_dim=768,
                max_subjects=10,
                log_dir=str(temp_dir / "logs"),
                max_memory_fraction=1.0
            )
            
            print("✅ Mini training pipeline test passed!")
            
            # Check if checkpoints were created
            checkpoint_files = list((temp_dir / "checkpoints").glob("*.pt"))
            assert len(checkpoint_files) > 0, "No checkpoints were created"
            print(f"✅ Created {len(checkpoint_files)} checkpoint(s)")
            
        except Exception as e:
            print(f"❌ Mini training pipeline test failed: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """Run all tests"""
    print("🚀 Starting Reconstruction Training Pipeline Tests")
    print("=" * 60)
    
    try:
        # Test individual components
        test_reconstruction_loss()
        test_pca_loading()
        test_data_preparation()
        
        print("\n" + "=" * 60)
        print("🧪 Testing Mini Training Pipeline...")
        print("(This may take a few minutes)")
        print("=" * 60)
        
        # Test mini training pipeline
        test_mini_training_pipeline()
        
        print("\n" + "=" * 60)
        print("🎉 All tests passed! The reconstruction training pipeline is working correctly.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
