#!/usr/bin/env python3
"""
Test script for Face ID Model training
"""

import torch
import json
import os
import sys
sys.path.append('.')

from training.train_face_id import load_dataset_metadata, FaceIDTrainer
from data.dataset import create_face_dataloader

def test_dataset_loading():
    """Test if we can load the dataset and metadata"""
    
    # Test metadata loading
    data_dir = "/Users/ozgewhiting/Documents/EQLabs/datasets_serial/CCA_train_db1"
    metadata = load_dataset_metadata(data_dir)
    
    if metadata is None:
        print("❌ Failed to load metadata")
        return False
    
    print("✅ Metadata loaded successfully")
    print(f"  Total subjects: {metadata['dataset_stats']['num_subjects']}")
    print(f"  Total clips: {metadata['total_clips_extracted']}")
    
    # Test dataloader creation
    try:
        dataloader = create_face_dataloader(
            data_dir=data_dir,
            batch_size=2
        )
        print("✅ Dataloader created successfully")
        
        # Test one batch
        for batch in dataloader:
            frames = batch['frames']
            subject_ids = batch['subject_id']
            print(f"✅ Batch loaded: frames shape {frames.shape}, subject_ids {subject_ids}")
            break
            
        return True
        
    except Exception as e:
        print(f"❌ Failed to create dataloader: {e}")
        return False

def test_training_setup():
    """Test if training setup works"""
    
    config = {
        'training': {
            'data_dir': "/Users/ozgewhiting/Documents/EQLabs/datasets_serial/CCA_train_db1",
            'log_dir': "./logs",
            'learning_rate': 1e-4
        },
        'face_id_model': {
            'embed_dim': 384,
            'num_heads': 8,
            'num_layers': 4,
            'dropout': 0.1
        }
    }
    
    try:
        trainer = FaceIDTrainer(config)
        print("✅ Trainer initialized successfully")
        return trainer
        
    except Exception as e:
        print(f"❌ Failed to initialize trainer: {e}")
        return None

if __name__ == "__main__":
    print("Testing dataset loading...")
    if test_dataset_loading():
        print("\nTesting training setup...")
        trainer = test_training_setup()
        if trainer:
            print("\n✅ All tests passed! Ready to train.")
        else:
            print("\n❌ Training setup failed.")
    else:
        print("\n❌ Dataset loading failed.") 