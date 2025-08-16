#!/usr/bin/env python3
"""
Test ArcFace Video Script
Reads face dataset H5 files, extracts ArcFace embeddings using ONNX runtime,
and computes cosine similarity between frames from two different H5 files.

This script is useful for:
- Face recognition and verification
- Computing similarity scores between different subjects
- Analyzing face embeddings across video sequences
"""

import cv2
import numpy as np
import os
import sys
import argparse
from pathlib import Path
import h5py
import gc
from typing import Dict, List, Tuple, Optional, Any

# Add models directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
from arcface_tokenizer import ArcFaceTokenizer

# Configure warnings
import warnings
warnings.filterwarnings('ignore')

# Set memory management
def setup_memory_management():
    """Setup memory management to prevent segmentation faults"""
    try:
        # Set OpenCV to use less memory
        cv2.setUseOptimized(True)
        cv2.setNumThreads(1)  # Reduce threading to prevent memory issues
        
        # Set numpy to use less memory
        np.set_printoptions(precision=3, suppress=True)
        
        print("🔧 Memory management configured")
    except Exception as e:
        print(f"⚠️  Memory management setup failed: {e}")

def check_memory_usage():
    """Check current memory usage"""
    try:
        # Simple memory check using sys
        import sys
        memory_mb = sys.getsizeof(globals()) / 1024 / 1024
        print(f"💾 Memory check completed")
        return memory_mb
    except Exception as e:
        print(f"⚠️  Could not check memory usage: {e}")
        return 0

def inspect_h5_structure(h5_path):
    """
    Inspect the structure of an H5 file to understand its layout
    
    Args:
        h5_path: Path to the H5 file
    """
    try:
        with h5py.File(h5_path, 'r') as f:
            print(f"\n🔍 H5 File Structure: {h5_path}")
            print("=" * 50)
            
            def print_structure(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"  Dataset: {name} - Shape: {obj.shape}, Dtype: {obj.dtype}")
                elif isinstance(obj, h5py.Group):
                    print(f"  Group: {name}")
            
            f.visititems(print_structure)
            
            print("\n📋 File Attributes:")
            for key, value in f.attrs.items():
                print(f"  {key}: {value}")
            
            print("=" * 50)
            
    except Exception as e:
        print(f"❌ Error inspecting H5 file: {str(e)}")

def load_h5_data(h5_path):
    """
    Load H5 file containing frames
    
    Args:
        h5_path: Path to the H5 file
        
    Returns:
        dict: Dictionary containing 'frames' and metadata
    """
    try:
        if not os.path.exists(h5_path):
            print(f"❌ H5 file not found: {h5_path}")
            return None
            
        print(f"🔄 Loading H5 data from: {h5_path}")
        
        # First inspect the file structure
        #inspect_h5_structure(h5_path)
        
        with h5py.File(h5_path, 'r') as f:
            # Load frames from data/frames
            if 'data/frames' in f:
                frames = f['data/frames'][:]
                print(f"   Frames shape: {frames.shape}")
            elif 'frames' in f:
                frames = f['frames'][:]
                print(f"   Frames shape: {frames.shape}")
            else:
                print("❌ No 'data/frames' or 'frames' found in H5 file")
                print(f"   Available keys: {list(f.keys())}")
                if 'data' in f:
                    print(f"   Data group keys: {list(f['data'].keys())}")
                return None
            
            # Get metadata
            metadata = {}
            if 'metadata' in f:
                for key in f['metadata'].keys():
                    try:
                        metadata[key] = f[f'metadata/{key}'][()]
                    except:
                        metadata[key] = str(f[f'metadata/{key}'][()])
            
            # Also get file attributes
            for key in f.attrs.keys():
                metadata[f'attr_{key}'] = f.attrs[key]
            
            print(f"✅ H5 data loaded successfully")
            return {
                'frames': frames,
                'metadata': metadata
            }
        
    except Exception as e:
        print(f"❌ Error loading H5 file: {str(e)}")
        return None

def load_arcface_tokenizer(model_path):
    """
    Load ArcFace tokenizer
    
    Args:
        model_path: Path to the ArcFace ONNX model file
        
    Returns:
        ArcFaceTokenizer instance
    """
    try:
        if not os.path.exists(model_path):
            print(f"❌ ArcFace model not found: {model_path}")
            print("Please download the ArcFace ONNX model and place it in the current directory")
            return None
            
        print(f"🔄 Loading ArcFace tokenizer from: {model_path}")
        
        # Create ArcFace tokenizer
        tokenizer = ArcFaceTokenizer(model_path)
        
        print(f"✅ ArcFace tokenizer loaded successfully")
        print(f"   Target size: {tokenizer.target_size}")
        print(f"   Embedding dim: {tokenizer.get_embedding_dim()}")
        
        return tokenizer
        
    except Exception as e:
        print(f"❌ Error loading ArcFace tokenizer: {str(e)}")
        return None



def extract_arcface_embedding(tokenizer, frame):
    """
    Extract ArcFace embedding from a frame using the tokenizer
    
    Args:
        tokenizer: ArcFaceTokenizer instance
        frame: Input frame in RGB format with values in [0, 1] range
        
    Returns:
        Normalized embedding vector
    """
    try:
        # Use the tokenizer's forward method
        embedding = tokenizer.forward(frame)
        
        if embedding is None:
            print(f"   ⚠️  Tokenizer returned None embedding")
            return None
        
        # Debug: check embedding values
        print(f"   🔍 Embedding stats: min={embedding.min():.6f}, max={embedding.max():.6f}, mean={embedding.mean():.6f}")
        print(f"   🔍 Embedding norm: {np.linalg.norm(embedding):.6f}")
        
        return embedding
        
    except Exception as e:
        print(f"❌ Error extracting embedding: {str(e)}")
        return None

def compute_cosine_similarity(emb1, emb2):
    """
    Compute cosine similarity between two embeddings
    
    Args:
        emb1: First normalized embedding
        emb2: Second normalized embedding
        
    Returns:
        Cosine similarity score
    """
    try:
        if emb1 is None or emb2 is None:
            print(f"   ⚠️  One or both embeddings are None")
            return None
        
        # Debug: check embedding shapes and values
        print(f"   🔍 Embedding shapes: emb1={emb1.shape}, emb2={emb2.shape}")
        print(f"   🔍 emb1 norm: {np.linalg.norm(emb1):.6f}")
        print(f"   🔍 emb2 norm: {np.linalg.norm(emb2):.6f}")
        
        # Check for invalid values
        if np.any(np.isnan(emb1)) or np.any(np.isinf(emb1)):
            print(f"   ⚠️  emb1 contains NaN or Inf values")
            return None
            
        if np.any(np.isnan(emb2)) or np.any(np.isinf(emb2)):
            print(f"   ⚠️  emb2 contains NaN or Inf values")
            return None
        
        # Verify embeddings are normalized
        if not np.isclose(np.linalg.norm(emb1), 1.0, atol=1e-6):
            print(f"   ⚠️  emb1 is not normalized: norm = {np.linalg.norm(emb1)}")
            return None
            
        if not np.isclose(np.linalg.norm(emb2), 1.0, atol=1e-6):
            print(f"   ⚠️  emb2 is not normalized: norm = {np.linalg.norm(emb2)}")
            return None
            
        # Compute cosine similarity
        cosine_sim = np.dot(emb1, emb2)
        
        # Check if result is valid
        if np.isnan(cosine_sim) or np.isinf(cosine_sim):
            print(f"   ⚠️  Cosine similarity result is invalid: {cosine_sim}")
            return None
        
        print(f"   🔍 Cosine similarity computed: {cosine_sim:.6f}")
        return cosine_sim
        
    except Exception as e:
        print(f"❌ Error computing cosine similarity: {str(e)}")
        return None

def save_debug_frame(frame, frame_idx, prefix="frame"):
    """
    Save a frame for debugging purposes
    
    Args:
        frame: Frame to save
        frame_idx: Frame index
        prefix: Prefix for filename
    """
    try:
        frame_debug_filename = f"{prefix}_{frame_idx}_debug.png"
        if frame.dtype != np.uint8:
            if frame.dtype == np.float32 or frame.dtype == np.float64:
                frame_save = (frame * 255).astype(np.uint8)
            else:
                frame_save = frame.astype(np.uint8)
        else:
            frame_save = frame
        cv2.imwrite(frame_debug_filename, frame_save)
        print(f"   💾 {prefix} {frame_idx} saved for debugging: {frame_debug_filename}")
    except Exception as e:
        print(f"   ⚠️  Failed to save debug frame: {str(e)}")

def process_arcface_comparison(input_h5_path, reference_h5_path, arcface_tokenizer, 
                               start_frame=0, end_frame=None, show_progress=True, debug=False):
    """
    Process H5 files and compute ArcFace similarities
    
    Args:
        input_h5_path: Path to input H5 file with frames
        reference_h5_path: Path to reference H5 file (first frame used as reference)
        arcface_model_path: Path to ArcFace ONNX model
        start_frame: Starting frame number (0-indexed)
        end_frame: Ending frame number (None for all frames)
        show_progress: Whether to show progress bar
        debug: Whether to save debug images
    """
    # Test tokenizer with a simple input to verify it works
    print("🧪 Testing ArcFace tokenizer with simple input...")
    test_input = np.random.rand(3, 224, 224).astype(np.float32)  # Random values in [0, 1], (C, H, W) format
    try:
        test_output = arcface_tokenizer.forward(test_input)
        print(f"   ✅ Tokenizer test successful: output shape={test_output.shape}")
        print(f"   🔍 Test output stats: min={test_output.min():.6f}, max={test_output.max():.6f}, mean={test_output.mean():.6f}")
    except Exception as e:
        print(f"   ❌ Tokenizer test failed: {str(e)}")
        return False
    
    # Load input H5 data
    input_data = load_h5_data(input_h5_path)
    if input_data is None:
        return False
    
    # Load reference H5 data
    reference_data = load_h5_data(reference_h5_path)
    if reference_data is None:
        return False
    
    # Get frame properties
    frames = input_data['frames']
    reference_frames = reference_data['frames']
    
    # Get reference frame (first frame)
    reference_frame = reference_frames[0]
    print(f"📸 Using first frame from reference H5 as comparison target")
    
    # Save reference frame for debugging
    if debug:
        save_debug_frame(reference_frame, 0, "reference")
    
    # Extract reference embedding directly using tokenizer
    reference_embedding = extract_arcface_embedding(arcface_tokenizer, reference_frame)
    if reference_embedding is None:
        return False
    
    print(f"✅ Reference embedding extracted successfully")
    
    # Get video properties
    total_frames = len(frames)
    height, width = frames[0].shape[:2]
    
    print(f"📹 Processing frames for ArcFace comparison:")
    print(f"   Resolution: {width}x{height}")
    print(f"   Total frames: {total_frames}")
    
    # Set frame range
    if end_frame is None:
        end_frame = total_frames
    
    frames_to_process = end_frame - start_frame
    print(f"   Processing frames: {start_frame} to {end_frame-1} ({frames_to_process} frames)")
    
    # Process frames
    similarities = []
    print(f"🔄 Processing frames for ArcFace comparison...")
    
    try:
        for i in range(frames_to_process):
            frame_idx = start_frame + i
            if frame_idx >= total_frames:
                print(f"⚠️  End of frames reached at index {frame_idx}")
                break
            
            try:
                # Get current frame
                frame = frames[frame_idx]
                
                # Save debug frame if requested
                if debug and frame_idx < 3:  # Save first 3 frames
                    save_debug_frame(frame, frame_idx, "input")
                
                # Extract embedding directly using tokenizer
                frame_embedding = extract_arcface_embedding(arcface_tokenizer, frame)
                if frame_embedding is None:
                    print(f"   ⚠️  Skipping frame {frame_idx} - embedding extraction failed")
                    continue
                
                # Compute cosine similarity
                similarity = compute_cosine_similarity(frame_embedding, reference_embedding)
                if similarity is None:
                    print(f"   ⚠️  Skipping frame {frame_idx} - similarity computation failed")
                    continue
                
                similarities.append((frame_idx, similarity))
                
                # Show progress
                if show_progress and i % 10 == 0:
                    progress = (i + 1) / frames_to_process * 100
                    print(f"   Progress: {progress:.1f}% ({i+1}/{frames_to_process})")
                    print(f"   Frame {frame_idx}: Similarity = {similarity:.4f}")
                
                # Memory cleanup every 10 frames
                if i % 10 == 0:
                    gc.collect()
                    check_memory_usage()
                    
            except Exception as e:
                print(f"   ❌ Error processing frame {frame_idx}: {str(e)}")
                continue
    
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
    
    finally:
        # Clean up
        gc.collect()
        print(f"✅ ArcFace comparison completed!")
    
    # Print results
    if similarities:
        print(f"\n📊 Similarity Results:")
        print("=" * 50)
        
        # Sort by similarity (highest first)
        #similarities.sort(key=lambda x: x[1], reverse=True)
        
        for frame_idx, similarity in similarities:
            print(f"   Frame {frame_idx:3d}: {similarity:.4f}")
        
        # Statistics
        similarities_values = [s[1] for s in similarities]
        avg_similarity = np.mean(similarities_values)
        max_similarity = np.max(similarities_values)
        min_similarity = np.min(similarities_values)
        
        print(f"\n📈 Statistics:")
        print(f"   Average similarity: {avg_similarity:.4f}")
        print(f"   Maximum similarity: {max_similarity:.4f}")
        print(f"   Minimum similarity: {min_similarity:.4f}")
        print(f"   Total frames processed: {len(similarities)}")
        
        # Save results to file
        results_filename = f"arcface_similarities_{Path(input_h5_path).stem}.txt"
        with open(results_filename, 'w') as f:
            f.write(f"ArcFace Similarity Results\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"Input H5: {input_h5_path}\n")
            f.write(f"Reference H5: {reference_h5_path}\n")
            f.write(f"Frames processed: {start_frame} to {end_frame-1}\n\n")
            
            f.write(f"Frame-by-frame similarities:\n")
            for frame_idx, similarity in similarities:
                f.write(f"  Frame {frame_idx:3d}: {similarity:.4f}\n")
            
            f.write(f"\nStatistics:\n")
            f.write(f"  Average similarity: {avg_similarity:.4f}\n")
            f.write(f"  Maximum similarity: {max_similarity:.4f}\n")
            f.write(f"  Minimum similarity: {min_similarity:.4f}\n")
            f.write(f"  Total frames processed: {len(similarities)}\n")
        
        print(f"💾 Results saved to: {results_filename}")
        
    else:
        print("❌ No similarities computed - processing failed for all frames")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Test ArcFace face recognition on H5 files')
    parser.add_argument('input_h5', nargs='?',
    default="/Users/ozgewhiting/Documents/EQLabs/datasets_serial/CCA_train_db4_no_padding/CCA_train_db4_no_padding_features_pca_384/subject_115_1254_00_faces_24_01_features.h5", 
    help='Path to input H5 file with frames')
    parser.add_argument('--reference-h5', '-r', 
    default="/Users/ozgewhiting/Documents/EQLabs/datasets_serial/CCA_train_db4_no_padding/CCA_train_db4_no_padding_features_pca_384/subject_115_1254_09_faces_1_52_features.h5", 
    help='Path to reference H5 file (first frame used as comparison target)')
    parser.add_argument('--arcface-model', '-m', 
    default="../arc.onnx", 
    help='Path to ArcFace ONNX model file')
    parser.add_argument('--start-frame', type=int, default=0, help='Starting frame number (0-indexed)')
    parser.add_argument('--end-frame', type=int, default=None, help='Ending frame number')
    parser.add_argument('--no-progress', action='store_true', help='Hide progress bar')
    parser.add_argument('--debug', action='store_true', help='Save debug images and show detailed logging')
    
    args = parser.parse_args()
    
    # Check input H5 file exists
    if args.input_h5 is None:
        print("❌ Error: Input H5 file is required")
        print("Usage: python test_arcface_video.py input.h5 --reference-h5 reference.h5 --arcface-model arcface.onnx")
        return
    
    if not os.path.exists(args.input_h5):
        print(f"❌ Error: Input H5 file not found: {args.input_h5}")
        return
    
    if not os.path.exists(args.reference_h5):
        print(f"❌ Error: Reference H5 file not found: {args.reference_h5}")
        return
    
    if not os.path.exists(args.arcface_model):
        print(f"❌ Error: ArcFace model not found: {args.arcface_model}")
        print("Please download the ArcFace ONNX model and place it in the current directory")
        return
    
    print("🚀 Starting ArcFace Face Recognition Test on H5 Files")
    print("=" * 60)
    
    # Setup memory management
    setup_memory_management()
    check_memory_usage()
    
    # Load ArcFace tokenizer
    arcface_tokenizer = load_arcface_tokenizer(args.arcface_model)
    if arcface_tokenizer is None:
        print("❌ Failed to load ArcFace tokenizer")
        return
    
    # Process ArcFace comparison
    success = process_arcface_comparison(
        args.input_h5, 
        args.reference_h5, 
        arcface_tokenizer,
        args.start_frame, 
        args.end_frame,
        not args.no_progress,
        args.debug
    )
    
    if success:
        print(f"\n🎉 ArcFace comparison completed successfully!")
        print(f"   Input H5: {args.input_h5}")
        print(f"   Reference H5: {args.reference_h5}")
        print(f"   ArcFace model: {args.arcface_model}")
    else:
        print(f"\n❌ ArcFace comparison failed!")

def test_debug_functionality():
    """Test if debug functionality works"""
    print("🧪 Testing debug functionality...")
    
    # Test frame saving
    test_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    save_debug_frame(test_frame, 0, "test")
    
    print("✅ Debug functionality test completed")
    return True

if __name__ == "__main__":
    # Test debug functionality if requested
    if len(sys.argv) > 1 and sys.argv[1] == "--test-debug":
        test_debug_functionality()
    else:
        main()
