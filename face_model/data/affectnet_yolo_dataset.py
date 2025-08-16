#!/usr/bin/env python3
"""
AffectNet YOLO Format Dataset

This module provides an AffectNetYOLODataset class that reads YOLO format images and labels
from the AffectNet dataset, supporting both PNG and JPG extensions. The dataset provides
face images with emotion labels for supervised expression learning.
"""

import os
import glob
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import cv2
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AffectNetYOLODataset:
    """
    Dataset class for AffectNet YOLO format data.
    
    This class reads YOLO format images and labels, supporting both PNG and JPG extensions.
    It provides face images with emotion labels for supervised expression learning.
    """
    
    # Emotion class mapping (based on data.yaml)
    EMOTION_CLASSES = [
        "Anger",      # 0
        "Contempt",   # 1  
        "Disgust",    # 2
        "Fear",       # 3
        "Happy",      # 4
        "Neutral",    # 5
        "Sad",        # 6
        "Surprise"    # 7
    ]
    
    def __init__(self, data_dir: str, split: str = "train", max_samples: Optional[int] = None):
        """
        Initialize AffectNet YOLO dataset.
        
        Args:
            data_dir: Path to the AffectNet YOLO format dataset directory
            split: Dataset split ('train', 'valid', 'test')
            max_samples: Maximum number of samples to load (for debugging)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        
        # Validate split
        if split not in ["train", "valid", "test"]:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'valid', or 'test'")
        
        # Set paths
        self.images_dir = self.data_dir / split / "images"
        self.labels_dir = self.data_dir / split / "labels"
        
        # Validate directories exist
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")
        
        # Find all image files (support both PNG and JPG)
        self.image_files = self._find_image_files()
        
        if max_samples:
            self.image_files = self.image_files[:max_samples]
        
        logger.info(f"Found {len(self.image_files)} images in {split} split")
        
        # Validate that we have corresponding labels for all images
        self._validate_image_label_pairs()
        
        # Create sample list
        self.samples = self._create_sample_list()
        logger.info(f"Created {len(self.samples)} valid samples")
    
    def _find_image_files(self) -> List[Path]:
        """Find all image files (PNG and JPG) in the images directory."""
        image_files = []
        
        # Find PNG files
        png_files = glob.glob(str(self.images_dir / "*.png"))
        image_files.extend([Path(f) for f in png_files])
        
        # Find JPG files
        jpg_files = glob.glob(str(self.images_dir / "*.jpg"))
        image_files.extend([Path(f) for f in jpg_files])
        
        # Find JPEG files
        jpeg_files = glob.glob(str(self.images_dir / "*.jpeg"))
        image_files.extend([Path(f) for f in jpeg_files])
        
        # Sort for reproducibility
        image_files.sort()
        
        logger.info(f"Found {len(png_files)} PNG, {len(jpg_files)} JPG, {len(jpeg_files)} JPEG files")
        
        return image_files
    
    def _validate_image_label_pairs(self):
        """Validate that we have corresponding labels for all images."""
        missing_labels = []
        
        for image_file in self.image_files:
            # Get corresponding label file path
            label_file = self._get_label_path(image_file)
            
            if not label_file.exists():
                missing_labels.append(image_file.name)
        
        if missing_labels:
            logger.warning(f"Found {len(missing_labels)} images without corresponding labels")
            logger.warning(f"First few missing labels: {missing_labels[:5]}")
            
            # Remove images without labels
            self.image_files = [f for f in self.image_files 
                              if self._get_label_path(f).exists()]
            
            logger.info(f"After validation: {len(self.image_files)} valid image-label pairs")
    
    def _get_label_path(self, image_file: Path) -> Path:
        """Get the corresponding label file path for an image file."""
        # Replace image extension with .txt
        label_name = image_file.stem + ".txt"
        return self.labels_dir / label_name
    
    def _create_sample_list(self) -> List[Dict[str, Any]]:
        """Create list of samples with image and label information."""
        samples = []
        
        for image_file in self.image_files:
            label_file = self._get_label_path(image_file)
            
            try:
                # Read label information
                label_info = self._read_label_file(label_file)
                
                if label_info is not None:
                    samples.append({
                        'image_path': str(image_file),
                        'label_path': str(label_file),
                        'emotion_class_id': label_info['class_id'],
                        'emotion_class_name': label_info['class_name'],
                        'bounding_box': label_info['bbox']
                    })
                    
            except Exception as e:
                logger.warning(f"Failed to read label for {image_file.name}: {e}")
                continue
        
        return samples
    
    def _read_label_file(self, label_file: Path) -> Optional[Dict[str, Any]]:
        """Read and parse YOLO format label file."""
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            if not lines:
                return None
            
            # Parse first line (assuming single face per image)
            line = lines[0].strip()
            if not line:
                return None
            
            parts = line.split()
            if len(parts) != 5:
                logger.warning(f"Invalid label format in {label_file.name}: {line}")
                return None
            
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # Validate class ID
            if class_id < 0 or class_id >= len(self.EMOTION_CLASSES):
                logger.warning(f"Invalid class ID {class_id} in {label_file.name}")
                return None
            
            return {
                'class_id': class_id,
                'class_name': self.EMOTION_CLASSES[class_id],
                'bbox': {
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height
                }
            }
            
        except Exception as e:
            logger.error(f"Error reading label file {label_file}: {e}")
            return None
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample by index."""
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range")
        
        sample = self.samples[idx]
        
        # Load image
        image = self._load_image(sample['image_path'])
        
        return {
            'image': image,
            'emotion_class_id': sample['emotion_class_id'],
            'emotion_class_name': sample['emotion_class_name'],
            'bounding_box': sample['bounding_box'],
            'image_path': sample['image_path'],
            'label_path': sample['label_path']
        }
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load image from file path."""
        try:
            # Use OpenCV to load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise
    
    def get_emotion_class_info(self) -> Dict[str, Any]:
        """Get information about emotion classes."""
        class_counts = {}
        for sample in self.samples:
            class_name = sample['emotion_class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        return {
            'class_names': self.EMOTION_CLASSES,
            'num_classes': len(self.EMOTION_CLASSES),
            'class_counts': class_counts,
            'total_samples': len(self.samples)
        }
    
    def get_sample_batch(self, batch_size: int = 30) -> List[Dict[str, Any]]:
        """Get a batch of samples for processing."""
        if batch_size > len(self.samples):
            batch_size = len(self.samples)
        
        # Randomly sample batch_size samples
        indices = np.random.choice(len(self.samples), batch_size, replace=False)
        
        batch = []
        for idx in indices:
            batch.append(self[idx])
        
        return batch


def test_affectnet_yolo_dataset():
    """Test function for AffectNetYOLODataset class."""
    print("🧪 Testing AffectNetYOLODataset class...")
    
    # Test with a dummy path (this will fail, but tests the initialization logic)
    try:
        dataset = AffectNetYOLODataset("./dummy_path")
        print("❌ Expected failure - dataset path doesn't exist")
    except FileNotFoundError:
        print("✅ Correctly caught missing dataset path")
    
    # Test emotion class mapping
    print(f"\n📊 Emotion classes: {AffectNetYOLODataset.EMOTION_CLASSES}")
    print(f"   Total classes: {len(AffectNetYOLODataset.EMOTION_CLASSES)}")
    
    # Test label parsing logic
    print("\n🧪 Testing label parsing logic...")
    
    # Mock label data
    mock_label_line = "4 0.499 0.499 0.999 0.999"
    parts = mock_label_line.split()
    class_id = int(parts[0])
    class_name = AffectNetYOLODataset.EMOTION_CLASSES[class_id]
    
    print(f"   Mock label: {mock_label_line}")
    print(f"   Parsed class ID: {class_id}")
    print(f"   Parsed class name: {class_name}")
    
    print("\n🎉 AffectNetYOLODataset test completed!")


if __name__ == "__main__":
    test_affectnet_yolo_dataset()
