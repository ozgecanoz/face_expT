#!/usr/bin/env python3
"""
Dataset loader for AffectNet emotion classification data
"""

import h5py
import torch
from torch.utils.data import Dataset
import os
import glob
import logging
from typing import List, Dict, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class AffectNetDataset(Dataset):
    """
    Dataset for loading AffectNet emotion classification data from HDF5 files
    """
    
    def __init__(self, data_dir: str, max_samples: int = None):
        """
        Args:
            data_dir: Directory containing AffectNet HDF5 files
            max_samples: Maximum number of samples to load (for debugging)
        """
        self.data_dir = data_dir
        
        # Find all HDF5 files
        self.h5_files = glob.glob(os.path.join(data_dir, "*.h5"))
        self.h5_files.sort()
        
        if max_samples:
            self.h5_files = self.h5_files[:max_samples]
        
        logger.info(f"Found {len(self.h5_files)} AffectNet HDF5 files in {data_dir}")
        
        # Validate and create sample list
        self.samples = self._create_sample_list()
        logger.info(f"Created {len(self.samples)} samples")
    
    def _create_sample_list(self) -> List[Dict]:
        """Create list of samples for training"""
        samples = []
        
        for h5_file in self.h5_files:
            try:
                with h5py.File(h5_file, 'r') as f:
                    # Check if this is an AffectNet file
                    if 'metadata' not in f or 'emotion_class_ids' not in f['metadata']:
                        logger.warning(f"Skipping {h5_file}: not an AffectNet file")
                        continue
                    
                    # Get emotion class IDs
                    emotion_class_ids = f['metadata']['emotion_class_ids'][:]
                    
                    # Get number of frames
                    if 'faces' in f:
                        num_frames = len(f['faces'])
                    else:
                        logger.warning(f"Skipping {h5_file}: no faces group")
                        continue
                    
                    # Validate frame count
                    if num_frames != len(emotion_class_ids):
                        logger.warning(f"Skipping {h5_file}: frame count mismatch ({num_frames} vs {len(emotion_class_ids)})")
                        continue
                    
                    samples.append({
                        'file_path': h5_file,
                        'num_frames': num_frames,
                        'emotion_class_ids': emotion_class_ids
                    })
                    
            except Exception as e:
                logger.warning(f"Could not read {h5_file}: {e}")
                continue
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Load a single sample"""
        sample = self.samples[idx]
        
        with h5py.File(sample['file_path'], 'r') as f:
            # Load face data for all frames
            faces_group = f['faces']
            
            # Get all frame data
            frame_data = []
            for frame_name in sorted(faces_group.keys()):
                frame_group = faces_group[frame_name]
                
                # Get the first face from each frame
                face_name = sorted(frame_group.keys())[0]  # First face
                face_group = frame_group[face_name]
                
                # Load face data (RGB image)
                face_data = face_group['data'][:]  # (518, 518, 3)
                
                # Convert to tensor and normalize
                face_tensor = torch.from_numpy(face_data).float()
                face_tensor = face_tensor.permute(2, 0, 1)  # (3, 518, 518)
                face_tensor = face_tensor / 255.0  # Normalize to [0, 1]
                
                frame_data.append(face_tensor)
            
            # Stack all frames
            frames = torch.stack(frame_data, dim=0)  # (30, 3, 518, 518)
            
            # Convert emotion class IDs to tensor
            emotion_class_ids = torch.from_numpy(sample['emotion_class_ids']).long()
            
            return {
                'frames': frames,  # (30, 3, 518, 518)
                'emotion_class_ids': emotion_class_ids,  # (30,)
                'file_path': sample['file_path']
            }


def test_affectnet_dataset():
    """Test the AffectNet dataset"""
    print("🧪 Testing AffectNet Dataset...")
    
    # Create a test directory with sample data
    test_dir = "test_affectnet_data"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create a test H5 file
    test_file = os.path.join(test_dir, "test_affectnet_0.h5")
    
    try:
        with h5py.File(test_file, 'w') as f:
            # Create faces group
            faces_group = f.create_group('faces')
            
            # Create 30 frames
            for i in range(30):
                frame_group = faces_group.create_group(f'frame_{i:03d}')
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
            metadata_group.create_dataset('subject_id', data='test_subject')
        
        print(f"✅ Created test file: {test_file}")
        
        # Test dataset loading
        dataset = AffectNetDataset(test_dir)
        print(f"✅ Dataset loaded: {len(dataset)} samples")
        
        # Test sample loading
        sample = dataset[0]
        print(f"✅ Sample loaded:")
        print(f"   Frames shape: {sample['frames'].shape}")
        print(f"   Emotion class IDs shape: {sample['emotion_class_ids'].shape}")
        print(f"   File path: {sample['file_path']}")
        
        # Clean up
        os.remove(test_file)
        os.rmdir(test_dir)
        print("✅ Cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_affectnet_dataset()
