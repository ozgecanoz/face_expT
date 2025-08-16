#!/usr/bin/env python3
"""
Test script to verify checkpoint saving and loading functionality
for the supervised expression transformer training
"""

import os
import sys
import torch
import tempfile
import shutil
import json
import numpy as np
from pathlib import Path

# Add the project root to the path for imports
sys.path.append('.')

from training.train_expT_supervised import (
    ExpTClassifierModel, 
    prepare_emotion_classification_data,
    train_expression_transformer_supervised
)
from models.dinov2_tokenizer import DINOv2BaseTokenizer
from utils.checkpoint_utils import load_checkpoint_config, extract_model_config


def create_dummy_affectnet_dataset(temp_dir, num_samples=3):
    """Create a dummy AffectNet dataset for testing"""
    import h5py
    
    dataset_dir = os.path.join(temp_dir, "dummy_affectnet")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Create dummy H5 files
    for i in range(num_samples):
        h5_path = os.path.join(dataset_dir, f"dummy_affectnet_{i}.h5")
        
        with h5py.File(h5_path, 'w') as f:
            # Create faces group
            faces_group = f.create_group('faces')
            
            # Create 30 frames
            for j in range(30):
                frame_group = faces_group.create_group(f'frame_{j:03d}')
                face_group = frame_group.create_group('face_000')
                
                # Create dummy face data (518x518x3)
                face_data = np.random.randint(0, 256, (518, 518, 3), dtype=np.uint8)
                face_group.create_dataset('data', data=face_data)
                face_group.create_dataset('bbox', data=[0, 0, 518, 518])
                face_group.create_dataset('confidence', data=0.9)
                face_group.create_dataset('original_size', data=[518, 518])
            
            # Create metadata
            metadata_group = f.create_group('metadata')
            metadata_group.create_dataset('emotion_class_ids', data=np.random.randint(0, 8, 30))
            metadata_group.create_dataset('emotion_class_names', data=['happy'] * 30)
            metadata_group.create_dataset('subject_id', data=f'dummy_subject_{i}')
    
    return dataset_dir


def create_dummy_pca_json(temp_dir):
    """Create a dummy PCA projection JSON file"""
    pca_path = os.path.join(temp_dir, "dummy_pca.json")
    
    # Create dummy PCA data
    pca_data = {
        "pca_components": np.random.randn(384, 768).tolist(),  # 384 -> 768 projection (correct orientation)
        "pca_mean": np.random.randn(768).tolist(),
        "explained_variance_ratio": np.random.rand(384).tolist(),
        "n_components": 384,
        "n_features": 768
    }
    
    with open(pca_path, 'w') as f:
        json.dump(pca_data, f)
    
    return pca_path


def test_model_creation():
    """Test that the ExpTClassifierModel can be created"""
    print("🧪 Testing model creation...")
    
    try:
        model = ExpTClassifierModel(
            embed_dim=384,
            num_heads=4,
            num_layers=2,
            dropout=0.1,
            ff_dim=1536,
            grid_size=37,
            num_classes=8
        )
        
        print(f"✅ Model created successfully:")
        print(f"   Embed dim: {model.expression_transformer.embed_dim}")
        print(f"   Num heads: {model.expression_transformer.num_heads}")
        print(f"   Num layers: {model.expression_transformer.num_layers}")
        print(f"   Num classes: {model.classifier[0].out_features}")
        
        # Test forward pass
        batch_size = 2
        num_patches = 1369
        embed_dim = 384
        
        dummy_input = torch.randn(batch_size, num_patches, embed_dim)
        expression_tokens, logits = model(dummy_input)
        
        print(f"✅ Forward pass successful:")
        print(f"   Expression tokens shape: {expression_tokens.shape}")
        print(f"   Logits shape: {logits.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_checkpoint_saving_loading():
    """Test checkpoint saving and loading"""
    print("\n🧪 Testing checkpoint saving and loading...")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create model
        model = ExpTClassifierModel(
            embed_dim=384,
            num_heads=4,
            num_layers=2,
            dropout=0.1,
            ff_dim=1536,
            grid_size=37,
            num_classes=8
        )
        
        # Create optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        
        # Save checkpoint
        checkpoint_path = os.path.join(temp_dir, "test_checkpoint.pt")
        
        from utils.checkpoint_utils import save_checkpoint, create_comprehensive_config
        
        config = create_comprehensive_config(
            expr_embed_dim=384,
            expr_num_heads=4,
            expr_num_layers=2,
            expr_dropout=0.1,
            expr_ff_dim=1536,
            expr_grid_size=37,
            num_classes=8,
            learning_rate=1e-4,
            batch_size=16,
            num_epochs=10,
            warmup_steps=100,
            min_lr=1e-6,
            pca_json_path="dummy_pca.json"
        )
        
        save_checkpoint(
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            scheduler_state_dict=scheduler.state_dict(),
            epoch=1,
            avg_loss=1.5,
            total_steps=100,
            config=config,
            checkpoint_path=checkpoint_path,
            checkpoint_type="expression_transformer"
        )
        
        print(f"✅ Checkpoint saved to: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"✅ Checkpoint loaded successfully:")
        print(f"   Epoch: {checkpoint['epoch']}")
        print(f"   Total steps: {checkpoint['total_steps']}")
        print(f"   Avg loss: {checkpoint['avg_loss']}")
        print(f"   Checkpoint type: {checkpoint.get('checkpoint_type', 'N/A')}")
        
        # Test config loading
        checkpoint_data, loaded_config = load_checkpoint_config(checkpoint_path)
        print(f"✅ Config loaded successfully:")
        print(f"   Config structure: {list(loaded_config.keys())}")
        print(f"   Expression model keys: {list(loaded_config.get('expression_model', {}).keys())}")
        print(f"   Supervised model keys: {list(loaded_config.get('supervised_model', {}).keys())}")
        
        # Test model reconstruction from checkpoint
        new_model = ExpTClassifierModel(
            embed_dim=loaded_config['expression_model']['embed_dim'],
            num_heads=loaded_config['expression_model']['num_heads'],
            num_layers=loaded_config['expression_model']['num_layers'],
            dropout=loaded_config['expression_model']['dropout'],
            ff_dim=loaded_config['expression_model']['ff_dim'],
            grid_size=loaded_config['expression_model']['grid_size'],
            num_classes=loaded_config['supervised_model']['num_classes']
        )
        
        new_model.load_state_dict(checkpoint['expression_transformer_state_dict'])
        
        print(f"✅ Model reconstructed from checkpoint successfully")
        
        # Test forward pass with reconstructed model
        batch_size = 2
        num_patches = 1369
        embed_dim = loaded_config['expression_model']['embed_dim']
        
        dummy_input = torch.randn(batch_size, num_patches, embed_dim)
        expression_tokens, logits = new_model(dummy_input)
        
        print(f"✅ Reconstructed model forward pass successful:")
        print(f"   Expression tokens shape: {expression_tokens.shape}")
        print(f"   Logits shape: {logits.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Checkpoint test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def test_training_with_checkpoints():
    """Test the full training pipeline with checkpoint saving"""
    print("\n🧪 Testing training pipeline with checkpoints...")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create dummy dataset and PCA
        dataset_dir = create_dummy_affectnet_dataset(temp_dir, num_samples=3)
        pca_path = create_dummy_pca_json(temp_dir)
        
        print(f"✅ Created dummy dataset: {dataset_dir}")
        print(f"✅ Created dummy PCA: {pca_path}")
        
        # Run a very short training session
        print("🚀 Starting mini training session...")
        
        model = train_expression_transformer_supervised(
            dataset_path=dataset_dir,
            pca_json_path=pca_path,
            checkpoint_dir=os.path.join(temp_dir, "checkpoints"),
            save_every_step=10,  # Save every 10 steps
            batch_size=2,  # Small batch size
            num_epochs=2,  # Only 2 epochs
            max_samples=3,  # Only 3 samples
            device=torch.device("cpu"),  # Use CPU for testing
            num_workers=0,  # No workers for testing
            pin_memory=False,
            persistent_workers=False,
            drop_last=False,
            embed_dim=384,
            num_heads=4,
            num_layers=2,
            dropout=0.1,
            ff_dim=1536,
            grid_size=37,
            num_classes=8,
            log_dir=os.path.join(temp_dir, "logs"),
            max_memory_fraction=0.9
        )
        
        print("✅ Training completed successfully!")
        
        # Check if checkpoints were created
        checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
            print(f"✅ Found {len(checkpoints)} checkpoints:")
            for cp in checkpoints:
                print(f"   - {cp}")
        else:
            print("❌ No checkpoint directory found")
            return False
        
        # Test loading the last checkpoint
        if checkpoints:
            last_checkpoint = sorted(checkpoints)[-1]
            checkpoint_path = os.path.join(checkpoint_dir, last_checkpoint)
            
            print(f"🔄 Testing loading of: {last_checkpoint}")
            
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            checkpoint_data, loaded_config = load_checkpoint_config(checkpoint_path)
            
            print(f"✅ Checkpoint loaded successfully:")
            print(f"   Epoch: {checkpoint['epoch']}")
            print(f"   Total steps: {checkpoint['total_steps']}")
            print(f"   Config keys: {list(loaded_config.keys())}")
            
            # Reconstruct model from checkpoint
            new_model = ExpTClassifierModel(
                embed_dim=loaded_config['expression_model']['embed_dim'],
                num_heads=loaded_config['expression_model']['num_heads'],
                num_layers=loaded_config['expression_model']['num_layers'],
                dropout=loaded_config['expression_model']['dropout'],
                ff_dim=loaded_config['expression_model']['ff_dim'],
                grid_size=loaded_config['expression_model']['grid_size'],
                num_classes=loaded_config['supervised_model']['num_classes']
            )
            
            new_model.load_state_dict(checkpoint['expression_transformer_state_dict'])
            
            print(f"✅ Model reconstructed from training checkpoint successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def main():
    """Run all tests"""
    print("🚀 Starting Checkpoint Saving/Loading Tests")
    print("=" * 60)
    
    tests = [
        ("Model Creation", test_model_creation),
        ("Checkpoint Save/Load", test_checkpoint_saving_loading),
        ("Training Pipeline", test_training_with_checkpoints)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Checkpoint functionality is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
