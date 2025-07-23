"""
Dataset loader for serialized face data
"""

import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import glob
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


class FaceDataset(Dataset):
    """
    Dataset for loading serialized face data from HDF5 files
    """
    
    def __init__(self, data_dir: str, max_samples: int = None):
        """
        Args:
            data_dir: Directory containing HDF5 files
            max_samples: Maximum number of samples to load (for debugging)
        """
        self.data_dir = data_dir
        
        # Find all HDF5 files
        self.h5_files = glob.glob(os.path.join(data_dir, "*.h5"))
        self.h5_files.sort()
        
        if max_samples:
            self.h5_files = self.h5_files[:max_samples]
        
        logger.info(f"Found {len(self.h5_files)} HDF5 files in {data_dir}")
        
        # Group files by subject ID
        self.subject_groups = self._group_by_subject()
        logger.info(f"Grouped into {len(self.subject_groups)} subjects")
        
        # Create sample list
        self.samples = self._create_sample_list()
        logger.info(f"Created {len(self.samples)} samples")
    
    def _group_by_subject(self) -> Dict[str, List[str]]:
        """Group HDF5 files by subject ID"""
        subject_groups = {}
        
        for h5_file in self.h5_files:
            try:
                with h5py.File(h5_file, 'r') as f:
                    subject_id = f['metadata']['subject_id'][()].decode('utf-8')
                    
                    if subject_id not in subject_groups:
                        subject_groups[subject_id] = []
                    subject_groups[subject_id].append(h5_file)
                    
            except Exception as e:
                logger.warning(f"Could not read subject ID from {h5_file}: {e}")
                continue
        
        return subject_groups
    
    def _create_sample_list(self) -> List[Dict]:
        """Create list of samples for training"""
        samples = []
        
        for subject_id, files in self.subject_groups.items():
            # For each subject, create samples from their clips
            for file_path in files:
                samples.append({
                    'file_path': file_path,
                    'subject_id': subject_id
                })
        
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
            
            return {
                'frames': frames,  # (30, 3, 518, 518)
                'subject_id': sample['subject_id'],
                'file_path': sample['file_path']
            }


def create_face_dataloader(data_dir: str, batch_size: int = 2, max_samples: int = None):
    """
    Create a DataLoader for face data
    
    Args:
        data_dir: Directory containing HDF5 files
        batch_size: Batch size for training
        max_samples: Maximum samples to load (for debugging)
    
    Returns:
        DataLoader
    """
    dataset = FaceDataset(data_dir, max_samples)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 for debugging
        pin_memory=False  # Use False for CPU
    )
    
    return dataloader


def test_dataset():
    """Test the dataset loader"""
    import os
    
    # Test with a small dataset
    data_dir = "../test_output"  # Adjust path as needed
    
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} not found. Please update the path.")
        return
    
    # Create dataset
    dataset = FaceDataset(data_dir, max_samples=5)
    
    if len(dataset) == 0:
        print("No samples found in dataset")
        return
    
    # Test loading a sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Frames shape: {sample['frames'].shape}")
    print(f"Subject ID: {sample['subject_id']}")
    print(f"File path: {sample['file_path']}")
    
    # Test dataloader
    dataloader = create_face_dataloader(data_dir, batch_size=2, max_samples=5)
    
    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"  Frames shape: {batch['frames'].shape}")
        print(f"  Subject IDs: {batch['subject_id']}")
        break
    
    print("✅ Dataset test passed!")


if __name__ == "__main__":
    test_dataset() 