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


class FaceFeaturesDataset(Dataset):
    """
    Dataset for loading PCA-projected face features from HDF5 files
    This dataset loads pre-computed DINOv2 features that have been projected to 384 dimensions
    """
    
    def __init__(self, data_dir: str, max_samples: int = None, feature_key: str = 'projected_features'):
        """
        Args:
            data_dir: Directory containing HDF5 files with PCA-projected features
            max_samples: Maximum number of samples to load (for debugging)
            feature_key: Key for features in H5 file (default: 'projected_features')
        """
        self.data_dir = data_dir
        self.feature_key = feature_key
        
        # Find all HDF5 files (should end with _features.h5)
        self.h5_files = glob.glob(os.path.join(data_dir, "*_features.h5"))
        self.h5_files.sort()
        
        if max_samples:
            self.h5_files = self.h5_files[:max_samples]
        
        logger.info(f"Found {len(self.h5_files)} feature HDF5 files in {data_dir}")
        
        # Group files by subject ID
        self.subject_groups = self._group_by_subject()
        logger.info(f"Grouped into {len(self.subject_groups)} subjects")
        
        # Create sample list
        self.samples = self._create_sample_list()
        logger.info(f"Created {len(self.samples)} samples")
        
        # Validate dataset structure
        self._validate_dataset()
    
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
    
    def _validate_dataset(self):
        """Validate that the dataset has the expected structure"""
        if len(self.samples) == 0:
            logger.warning("No samples found in dataset")
            return
        
        # Check first sample to validate structure
        try:
            sample = self[0]
            logger.info(f"Dataset validation successful:")
            logger.info(f"  Feature shape: {sample['features'].shape}")
            logger.info(f"  Frames shape: {sample['frames'].shape}")
            logger.info(f"  Subject ID: {sample['subject_id']}")
            logger.info(f"  Feature key: {self.feature_key}")
        except Exception as e:
            logger.error(f"Dataset validation failed: {e}")
            raise
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Load a single sample with PCA-projected features"""
        sample = self.samples[idx]
        
        with h5py.File(sample['file_path'], 'r') as f:
            # Load PCA-projected features
            features = f['data'][self.feature_key][:]  # (30, 1369, 384)
            
            # Convert to tensor
            features_tensor = torch.from_numpy(features).float()
            
            # Load original frames for reconstruction loss comparison
            frames = f['data']['frames'][:]  # (30, 3, 518, 518)
            frames_tensor = torch.from_numpy(frames).float()
            
            # Load metadata
            subject_id = f['metadata']['subject_id'][()].decode('utf-8')
            clip_id = f['metadata']['clip_id'][()].decode('utf-8')
            
            # Load additional metadata if available
            metadata = {}
            #for key in f['metadata'].keys():
            #    if key not in ['subject_id', 'clip_id']:
            #         try:
            #             value = f['metadata'][key][()]
            #             metadata[key] = value
            #         except:
            #             continue
            
            return {
                'features': features_tensor,  # (30, 1369, 384) - PCA-projected features
                'frames': frames_tensor,      # (30, 3, 518, 518) - Original frames for reconstruction loss
                'subject_id': subject_id,
                'clip_id': clip_id,
                'file_path': sample['file_path'],
                'metadata': metadata
            }
    
    def get_feature_dim(self) -> int:
        """Get the feature dimension"""
        if len(self.samples) == 0:
            return 0
        
        with h5py.File(self.samples[0]['file_path'], 'r') as f:
            return f['data'][self.feature_key].shape[-1]
    
    def get_num_patches(self) -> int:
        """Get the number of patches per frame"""
        if len(self.samples) == 0:
            return 0
        
        with h5py.File(self.samples[0]['file_path'], 'r') as f:
            return f['data'][self.feature_key].shape[1]
    
    def get_num_frames(self) -> int:
        """Get the number of frames per clip"""
        if len(self.samples) == 0:
            return 0
        
        with h5py.File(self.samples[0]['file_path'], 'r') as f:
            return f['data'][self.feature_key].shape[0]
    
    def get_frame_dimensions(self) -> Tuple[int, int]:
        """Get the frame dimensions (height, width)"""
        if len(self.samples) == 0:
            return (0, 0)
        
        with h5py.File(self.samples[0]['file_path'], 'r') as f:
            frames = f['data']['frames']
            return (frames.shape[2], frames.shape[3])  # (height, width)


def create_face_dataloader(data_dir: str, batch_size: int = 2, max_samples: int = None, 
                          num_workers: int = 0, pin_memory: bool = False, 
                          persistent_workers: bool = False, drop_last: bool = False):
    """
    Create a DataLoader for face data
    
    Args:
        data_dir: Directory containing HDF5 files
        batch_size: Batch size for training
        max_samples: Maximum samples to load (for debugging)
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory (use False for CPU)
        persistent_workers: Whether to keep workers alive between epochs
        drop_last: Whether to drop the last incomplete batch
    
    Returns:
        DataLoader
    """
    dataset = FaceDataset(data_dir, max_samples)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=drop_last
    )
    
    return dataloader


def create_face_features_dataloader(data_dir: str, batch_size: int = 2, max_samples: int = None, 
                                   num_workers: int = 0, pin_memory: bool = False, 
                                   persistent_workers: bool = False, drop_last: bool = False,
                                   feature_key: str = 'projected_features'):
    """
    Create a DataLoader for PCA-projected face features
    
    Args:
        data_dir: Directory containing HDF5 files with PCA-projected features
        batch_size: Batch size for training
        max_samples: Maximum samples to load (for debugging)
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory (use True for GPU training)
        persistent_workers: Whether to keep workers alive between epochs
        drop_last: Whether to drop the last incomplete batch
        feature_key: Key for features in H5 file (default: 'projected_features')
    
    Returns:
        DataLoader
    """
    dataset = FaceFeaturesDataset(data_dir, max_samples, feature_key)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=drop_last
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


def test_features_dataset():
    """Test the FaceFeaturesDataset loader"""
    import os
    
    # Test with a small dataset
    data_dir = "/Users/ozgewhiting/Documents/EQLabs/datasets_serial/CCA_train_db4_no_padding/CCA_train_db4_no_padding_features_pca_384/"  # Adjust path as needed
    
    if not os.path.exists(data_dir):
        print(f"Features data directory {data_dir} not found. Please update the path.")
        return
    
    # Create dataset
    dataset = FaceFeaturesDataset(data_dir, max_samples=5)
    
    if len(dataset) == 0:
        print("No samples found in features dataset")
        return
    
    # Test loading a sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Features shape: {sample['features'].shape}")
    print(f"Frames shape: {sample['frames'].shape}")
    print(f"Subject ID: {sample['subject_id']}")
    print(f"Clip ID: {sample['clip_id']}")
    print(f"File path: {sample['file_path']}")
    print(f"Metadata keys: {list(sample['metadata'].keys())}")
    
    # Test dataset properties
    print(f"Feature dimension: {dataset.get_feature_dim()}")
    print(f"Number of patches: {dataset.get_num_patches()}")
    print(f"Number of frames: {dataset.get_num_frames()}")
    
    # Test dataloader
    dataloader = create_face_features_dataloader(data_dir, batch_size=2, max_samples=5)
    
    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"  Features shape: {batch['features'].shape}")
        print(f"  Subject IDs: {batch['subject_id']}")
        print(f"  Clip IDs: {batch['clip_id']}")
        break
    
    print("✅ Features dataset test passed!")


if __name__ == "__main__":
    test_dataset()
    print("\n" + "="*50 + "\n")
    test_features_dataset() 