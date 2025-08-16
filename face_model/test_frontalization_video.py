import cv2
import numpy as np
import os
import sys
import argparse
from pathlib import Path
import h5py
import gc

# PyTorch imports removed - no longer needed for homography-based approach
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

def warp_homography_from_features(orig1, orig2, feat1, feat2,
                                  detector='ORB',
                                  ratio_thresh=0.75,
                                  ransac_thresh=3.0):
    """
    Estimate H(feat1->feat2) using local features, convert it to H(orig1->orig2),
    and warp orig1 into orig2's frame.

    Args:
        orig1, orig2: original images (H1,W1,3)/(H2,W2,3), BGR or RGB (dtype uint8/float OK)
        feat1, feat2: feature images aligned to orig1/orig2 (Hf1,Wf1),(Hf2,Wf2). Can be 1- or 3-ch.
        detector: 'ORB' (fast) or 'SIFT' (better on low-texture; requires contrib build/license OK)
        ratio_thresh: Lowe ratio test
        ransac_thresh: RANSAC reprojection threshold in pixels (feature-image scale)

    Returns:
        warped1_to_2: orig1 warped into orig2 coordinates by homography
        H_orig: 3x3 homography mapping orig1 -> orig2 (float64)
        inlier_mask: inlier mask from RANSAC (Nx1 uint8) or None
    """

    def to_gray_u8(img):
        # Accept 1- or 3-channel, float or uint8; return uint8 grayscale
        if img.ndim == 3 and img.shape[2] >= 3:
            # Assume BGR/RGB; conversion is fine either way for grayscale
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            g = img.squeeze()
        g = g.astype(np.float32)
        if g.max() <= 1.0:
            g *= 255.0
        g = np.clip(g, 0, 255).astype(np.uint8)
        return g

    g1, g2 = to_gray_u8(feat1), to_gray_u8(feat2)

    # --- 1) Keypoints & descriptors ---
    if detector.upper() == 'SIFT':
        sift = cv2.SIFT_create()
        k1, d1 = sift.detectAndCompute(g1, None)
        k2, d2 = sift.detectAndCompute(g2, None)
        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=100))
    else:  # ORB (fast, free)
        orb = cv2.ORB_create(nfeatures=4000, scaleFactor=1.2, nlevels=8, edgeThreshold=15)
        k1, d1 = orb.detectAndCompute(g1, None)
        k2, d2 = orb.detectAndCompute(g2, None)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    if d1 is None or d2 is None or len(k1) < 4 or len(k2) < 4:
        raise RuntimeError("Not enough features to estimate homography.")

    # --- 2) Match + ratio test ---
    knn = matcher.knnMatch(d1, d2, k=2)
    good = []
    for m, n in knn:
        if m.distance < ratio_thresh * n.distance:
            good.append(m)
    if len(good) < 4:
        raise RuntimeError("Not enough good matches after ratio test.")

    pts1 = np.float32([k1[m.queryIdx].pt for m in good])
    pts2 = np.float32([k2[m.trainIdx].pt for m in good])

    # --- 3) Homography in FEATURE space (feat1 -> feat2) ---
    H_feat, inlier_mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
    if H_feat is None:
        raise RuntimeError("Homography estimation failed.")

    # --- 4) Lift to ORIGINAL image coordinates ---
    H1, W1 = orig1.shape[:2]
    H2, W2 = orig2.shape[:2]
    hf1, wf1 = g1.shape[:2]
    hf2, wf2 = g2.shape[:2]

    # S_down maps ORIGINAL -> FEATURE coordinates
    S1_down = np.array([[wf1 / W1, 0, 0],
                        [0, hf1 / H1, 0],
                        [0, 0, 1]], dtype=np.float64)
    S2_down = np.array([[wf2 / W2, 0, 0],
                        [0, hf2 / H2, 0],
                        [0, 0, 1]], dtype=np.float64)
    S2_down_inv = np.linalg.inv(S2_down)

    # H_orig maps orig1 -> orig2
    H_orig = S2_down_inv @ H_feat @ S1_down

    # --- 5) Warp orig1 into orig2's frame ---
    warped1_to_2 = cv2.warpPerspective(orig1, H_orig, (W2, H2),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REFLECT)

    return warped1_to_2, H_orig, inlier_mask

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
    Load H5 file containing PCA features and frames
    
    Args:
        h5_path: Path to the H5 file
        
    Returns:
        dict: Dictionary containing 'pca_features', 'frames', and metadata
    """
    try:
        if not os.path.exists(h5_path):
            print(f"❌ H5 file not found: {h5_path}")
            return None
            
        print(f"🔄 Loading H5 data from: {h5_path}")
        
        # First inspect the file structure
        inspect_h5_structure(h5_path)
        
        with h5py.File(h5_path, 'r') as f:
            # Load PCA features from data/projected_features
            if 'data/projected_features' in f:
                pca_features = f['data/projected_features'][:]
                print(f"   PCA features shape: {pca_features.shape}")
            elif 'projected_features' in f:
                pca_features = f['projected_features'][:]
                print(f"   PCA features shape: {pca_features.shape}")
            else:
                print("❌ No 'data/projected_features' or 'projected_features' found in H5 file")
                print(f"   Available keys: {list(f.keys())}")
                if 'data' in f:
                    print(f"   Data group keys: {list(f['data'].keys())}")
                return None
            
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
                'pca_features': pca_features,
                'frames': frames,
                'metadata': metadata
            }
        
    except Exception as e:
        print(f"❌ Error loading H5 file: {str(e)}")
        return None

def create_pca_rgb_visualization(pca_features, top_components=3, debug=False):
    """
    Create RGB visualization from top PCA components
    
    Args:
        pca_features: PCA features array (N, num_components)
        top_components: Number of top components to use (default: 3)
        debug: Whether to save debug images
        
    Returns:
        RGB image array (37, 37, 3) with values in [0, 255]
    """
    try:
        # Ensure we have enough components
        if pca_features.shape[1] < top_components:
            print(f"⚠️  Only {pca_features.shape[1]} components available, using all")
            top_components = pca_features.shape[1]
        
        # Take top components
        top_features = pca_features[:, :top_components]
        
        # Reshape to 37x37 grid (assuming 1369 patches = 37*37)
        if top_features.shape[0] == 1369:
            grid_size = 37
        elif top_features.shape[0] == 256:  # 16x16
            grid_size = 16
        else:
            # Try to find a square grid
            grid_size = int(np.sqrt(top_features.shape[0]))
            if grid_size * grid_size != top_features.shape[0]:
                print(f"⚠️  Non-square grid detected, using {grid_size}x{grid_size}")
        
        # Reshape to grid
        features_grid = top_features.reshape(grid_size, grid_size, top_components)
        
        # Normalize each component to [0, 1] range
        features_normalized = np.zeros_like(features_grid)
        for i in range(top_components):
            component = features_grid[:, :, i]
            min_val, max_val = component.min(), component.max()
            if max_val > min_val:
                features_normalized[:, :, i] = (component - min_val) / (max_val - min_val)
            else:
                features_normalized[:, :, i] = 0.5  # Default to middle value
        
        # Convert to RGB (scale to [0, 255])
        rgb_image = (features_normalized * 255).astype(np.uint8)
        
        # Ensure we have 3 channels
        if rgb_image.shape[2] == 1:
            rgb_image = np.repeat(rgb_image, 3, axis=2)
        elif rgb_image.shape[2] == 2:
            # Pad with zeros for the third channel
            rgb_image = np.concatenate([rgb_image, np.zeros((grid_size, grid_size, 1), dtype=np.uint8)], axis=2)
        
        # Enhance contrast for better feature detection
        # Apply histogram equalization to each channel
        enhanced_image = np.zeros_like(rgb_image)
        for i in range(rgb_image.shape[2]):
            channel = rgb_image[:, :, i]
            enhanced_image[:, :, i] = cv2.equalizeHist(channel)
        
        # Apply median smoothing to reduce salt and pepper noise
        # This creates smoother, cleaner images better for feature detection
        smoothed_image = np.zeros_like(enhanced_image)
        for i in range(enhanced_image.shape[2]):
            channel = enhanced_image[:, :, i]
            # Use 3x3 median filter for noise reduction
            smoothed_image[:, :, i] = cv2.medianBlur(channel, 3)
        
        # Apply additional Gaussian smoothing for even smoother results
        final_image = np.zeros_like(smoothed_image)
        for i in range(smoothed_image.shape[2]):
            channel = smoothed_image[:, :, i]
            # Use small Gaussian kernel (1.0 sigma) to maintain edge features while smoothing
            final_image[:, :, i] = cv2.GaussianBlur(channel, (3, 3), 1.0)
        
        print(f"✅ Enhanced and smoothed PCA RGB visualization created: {final_image.shape}")
        
        # Save visualization for debugging if requested
        if debug:
            debug_filename = f"pca_debug_{np.random.randint(1000, 9999)}.png"
            cv2.imwrite(debug_filename, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
            print(f"   💾 Debug image saved: {debug_filename}")
        
        return final_image
        
    except Exception as e:
        print(f"❌ Error creating PCA visualization: {str(e)}")
        return None

def frontalize_frame(frame, frame_pca_features, frontal_pca_features, detector='ORB'):
    """
    Apply homography-based frontalization to a single frame using PCA features
    
    Args:
        frame: Input frame as numpy array (BGR format from OpenCV)
        frame_pca_features: PCA features for the current frame (RGB format, 37x37)
        frontal_pca_features: Frontal PCA features reference (RGB format, 37x37)
        detector: Feature detector to use ('ORB' or 'SIFT')
        
    Returns:
        Frontalized frame as numpy array (BGR format)
    """
    try:
        if frame_pca_features is None or frontal_pca_features is None:
            return frame  # Return original frame if features not loaded
            
        # Ensure frame is in the correct format (BGR, uint8)
        if frame.dtype != np.uint8:
            if frame.dtype == np.float32 or frame.dtype == np.float64:
                # Convert from [0,1] to [0,255] range
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        
        # Convert BGR to RGB for the frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert PCA features to grayscale for better feature detection
        frame_pca_gray = cv2.cvtColor(frame_pca_features, cv2.COLOR_RGB2GRAY)
        frontal_pca_gray = cv2.cvtColor(frontal_pca_features, cv2.COLOR_RGB2GRAY)
        
        # Try different feature detection strategies
        homography_success = False
        H_pca = None
        
        # Strategy 1: Try with current detector
        try:
            warped_pca, H_pca, inlier_mask = warp_homography_from_features(
                frame_pca_gray, frontal_pca_gray, 
                frame_pca_gray, frontal_pca_gray,
                detector=detector,
                ratio_thresh=0.75,
                ransac_thresh=3.0
            )
            homography_success = True
            print(f"   ✅ Homography computed successfully with {detector}")
        except RuntimeError as e:
            print(f"   ⚠️  {detector} homography failed: {str(e)}")
        
        # Strategy 2: Try with different detector if first failed
        if not homography_success:
            alternate_detector = 'SIFT' if detector == 'ORB' else 'ORB'
            try:
                warped_pca, H_pca, inlier_mask = warp_homography_from_features(
                    frame_pca_gray, frontal_pca_gray, 
                    frame_pca_gray, frontal_pca_gray,
                    detector=alternate_detector,
                    ratio_thresh=0.6,  # More lenient ratio test
                    ransac_thresh=5.0   # More lenient RANSAC
                )
                homography_success = True
                print(f"   ✅ Homography computed successfully with {alternate_detector} (fallback)")
            except RuntimeError as e:
                print(f"   ⚠️  {alternate_detector} fallback also failed: {str(e)}")
        
        # Strategy 3: Try with even more lenient parameters
        if not homography_success:
            try:
                warped_pca, H_pca, inlier_mask = warp_homography_from_features(
                    frame_pca_gray, frontal_pca_gray, 
                    frame_pca_gray, frontal_pca_gray,
                    detector='ORB',
                    ratio_thresh=0.5,  # Very lenient ratio test
                    ransac_thresh=8.0   # Very lenient RANSAC
                )
                homography_success = True
                print(f"   ✅ Homography computed successfully with very lenient parameters")
            except RuntimeError as e:
                print(f"   ⚠️  All homography strategies failed: {str(e)}")
        
        if homography_success and H_pca is not None:
            # Scale homography from PCA space (37x37) to original frame space (518x518)
            pca_size = 37
            frame_height, frame_width = frame.shape[:2]
            
            # Scale factors
            scale_x = frame_width / pca_size
            scale_y = frame_height / pca_size
            
            # Create scaling matrices
            S_pca_to_frame = np.array([
                [scale_x, 0, 0],
                [0, scale_y, 0],
                [0, 0, 1]
            ], dtype=np.float64)
            
            S_frame_to_pca = np.array([
                [1/scale_x, 0, 0],
                [0, 1/scale_y, 0],
                [0, 0, 1]
            ], dtype=np.float64)
            
            # Transform homography from PCA space to frame space
            H_frame = S_pca_to_frame @ H_pca @ S_frame_to_pca
            
            # Warp the original frame using the scaled homography
            warped_frame = cv2.warpPerspective(
                frame_rgb, H_frame, (frame_width, frame_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT
            )
            
            # Convert RGB back to BGR for OpenCV
            warped_frame_bgr = cv2.cvtColor(warped_frame, cv2.COLOR_RGB2BGR)
            
            return warped_frame_bgr
        else:
            print(f"   ⚠️  All homography strategies failed, returning original frame")
            return frame
        
    except Exception as e:
        print(f"❌ Error during homography-based frontalization: {str(e)}")
        return frame  # Return original frame on error

def process_video(input_h5_path, output_video_path, frontal_h5_path, 
                  start_frame=0, end_frame=None, show_progress=True, detector='ORB', debug=False):
    """
    Process H5 file frame by frame, applying homography-based frontalization
    
    Args:
        input_h5_path: Path to input H5 file with PCA features and frames
        output_video_path: Path to output video file
        frontal_h5_path: Path to frontal reference H5 file
        start_frame: Starting frame number (0-indexed)
        end_frame: Ending frame number (None for all frames)
        show_progress: Whether to show progress bar
        detector: Feature detector to use ('ORB' or 'SIFT')
        debug: Whether to save debug images
    """
    # Load input H5 data
    input_data = load_h5_data(input_h5_path)
    if input_data is None:
        return False
    
    # Load frontal reference H5 data
    frontal_data = load_h5_data(frontal_h5_path)
    if frontal_data is None:
        return False
    
    # Get frame properties
    frames = input_data['frames']
    pca_features = input_data['pca_features']
    
    # Get frontal reference (first frame)
    frontal_frame = frontal_data['frames'][0]  # Use first frame as frontal reference
    frontal_pca_features = frontal_data['pca_features'][0]
    
    # Create frontal PCA visualization
    frontal_pca_rgb = create_pca_rgb_visualization(frontal_pca_features, debug=debug)
    if frontal_pca_rgb is None:
        return False
    
    # Get video properties
    total_frames = len(frames)
    height, width = frames[0].shape[:2]
    
    # Set frame range
    if end_frame is None:
        end_frame = total_frames
    
    frames_to_process = end_frame - start_frame
    print(f"📹 H5 data properties:")
    print(f"   Resolution: {width}x{height}")
    print(f"   Total frames: {total_frames}")
    print(f"   Processing frames: {start_frame} to {end_frame-1} ({frames_to_process} frames)")
    
    # Create video writer (using 30 FPS as default)
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"❌ Error: Could not create output video: {output_video_path}")
        return False
    
    print(f"🔄 Processing H5 frames...")
    
    try:
        for i in range(frames_to_process):
            frame_idx = start_frame + i
            if frame_idx >= total_frames:
                print(f"⚠️  End of frames reached at index {frame_idx}")
                break
            
            # Get current frame and PCA features
            frame = frames[frame_idx]
            frame_pca_features = pca_features[frame_idx]
            
            # Create PCA visualization for current frame
            frame_pca_rgb = create_pca_rgb_visualization(frame_pca_features)
            if frame_pca_rgb is None:
                print(f"   ⚠️  Skipping frame {frame_idx} - PCA visualization failed")
                continue
            
            # Apply frontalization
            frontalized_frame = frontalize_frame(frame, frame_pca_rgb, frontal_pca_rgb, detector)
            
            # Ensure frame is in correct format for video writing
            if frontalized_frame.dtype != np.uint8:
                if frontalized_frame.dtype == np.float32 or frontalized_frame.dtype == np.float64:
                    # Convert from [0,1] to [0,255] range
                    frontalized_frame = (frontalized_frame * 255).astype(np.uint8)
                else:
                    frontalized_frame = frontalized_frame.astype(np.uint8)
            
            # Write frame to output video
            out.write(frontalized_frame)
            
            # Show progress
            if show_progress and i % 10 == 0:
                progress = (i + 1) / frames_to_process * 100
                print(f"   Progress: {progress:.1f}% ({i+1}/{frames_to_process})")
                
                # Memory cleanup every 10 frames
                if i % 10 == 0:
                    gc.collect()
                    check_memory_usage()
    
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
    
    finally:
        # Clean up
        out.release()
        gc.collect()
        print(f"✅ Video processing completed!")
        print(f"   Output saved to: {output_video_path}")
    
    return True

def create_side_by_side_video(input_h5_path, output_video_path, frontal_h5_path,
                              start_frame=0, end_frame=None, show_progress=True, detector='ORB', debug=False):
    """
    Create a side-by-side comparison video (original vs frontalized)
    
    Args:
        input_h5_path: Path to input H5 file with PCA features and frames
        output_video_path: Path to output comparison video file
        frontal_h5_path: Path to frontal reference H5 file
        start_frame: Starting frame number (0-indexed)
        end_frame: Ending frame number (None for all frames)
        show_progress: Whether to show progress bar
        detector: Feature detector to use ('ORB' or 'SIFT')
        debug: Whether to save debug images
    """
    # Load input H5 data
    input_data = load_h5_data(input_h5_path)
    if input_data is None:
        return False
    
    # Load frontal reference H5 data
    frontal_data = load_h5_data(frontal_h5_path)
    if frontal_data is None:
        return False
    
    # Get frame properties
    frames = input_data['frames']
    pca_features = input_data['pca_features']
    
    # Get frontal reference (first frame)
    frontal_frame = frontal_data['frames'][0]  # Use first frame as frontal reference
    frontal_pca_features = frontal_data['pca_features'][0]
    
    # Create frontal PCA visualization
    frontal_pca_rgb = create_pca_rgb_visualization(frontal_pca_features)
    if frontal_pca_rgb is None:
        return False
    
    # Get video properties
    total_frames = len(frames)
    height, width = frames[0].shape[:2]
    
    print(f"📹 Creating side-by-side comparison video:")
    print(f"   Resolution: {width}x{height}")
    print(f"   Total frames: {total_frames}")
    
    # Set frame range
    if end_frame is None:
        end_frame = total_frames
    
    frames_to_process = end_frame - start_frame
    print(f"   Processing frames: {start_frame} to {end_frame-1} ({frames_to_process} frames)")
    
    # Create video writer for side-by-side (double width)
    fps = 30  # Default FPS
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width * 2, height))
    
    if not out.isOpened():
        print(f"❌ Error: Could not create output video: {output_video_path}")
        return False
    
    print(f"🔄 Processing H5 frames for side-by-side comparison...")
    
    try:
        for i in range(frames_to_process):
            frame_idx = start_frame + i
            if frame_idx >= total_frames:
                print(f"⚠️  End of frames reached at index {frame_idx}")
                break
            
            # Get current frame and PCA features
            frame = frames[frame_idx]
            frame_pca_features = pca_features[frame_idx]
            
            # Create PCA visualization for current frame
            frame_pca_rgb = create_pca_rgb_visualization(frame_pca_features)
            if frame_pca_rgb is None:
                print(f"   ⚠️  Skipping frame {frame_idx} - PCA visualization failed")
                continue
            
            # Apply frontalization
            frontalized_frame = frontalize_frame(frame, frame_pca_rgb, frontal_pca_rgb, detector)
            
            # Ensure both frames are in correct format for video writing
            if frame.dtype != np.uint8:
                if frame.dtype == np.float32 or frame.dtype == np.float64:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
            
            if frontalized_frame.dtype != np.uint8:
                if frontalized_frame.dtype == np.float32 or frontalized_frame.dtype == np.float64:
                    frontalized_frame = (frontalized_frame * 255).astype(np.uint8)
                else:
                    frontalized_frame = frontalized_frame.astype(np.uint8)
            
            # Create side-by-side frame
            side_by_side = np.hstack([frame, frontalized_frame])
            
            # Write frame to output video
            out.write(side_by_side)
            
            # Show progress
            if show_progress and i % 10 == 0:
                progress = (i + 1) / frames_to_process * 100
                print(f"   Progress: {progress:.1f}% ({i+1}/{frames_to_process})")
    
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
    
    finally:
        # Clean up
        out.release()
        print(f"✅ Side-by-side video completed!")
        print(f"   Output saved to: {output_video_path}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Test homography-based frontalization on video using PCA features')
    parser.add_argument('input_h5', nargs='?',
    default="/Users/ozgewhiting/Documents/EQLabs/datasets_serial/CCA_train_db4_no_padding/CCA_train_db4_no_padding_features_pca_384/subject_103_1242_00_faces_26_12_features.h5", 
    help='Path to input H5 file with PCA features and frames')
    parser.add_argument('--output', '-o', 
    default="/Users/ozgewhiting/Documents/EQLabs/datasets_serial/CCA_train_db4_no_padding/CCA_train_db4_no_padding_features_pca_384/subject_103_1242_00_faces_26_12_features_frontalized.mp4", 
    help='Path to output video file (auto-generated if not specified)')
    parser.add_argument('--frontal-h5', '-f', 
    default="/Users/ozgewhiting/Documents/EQLabs/datasets_serial/CCA_train_db4_no_padding/CCA_train_db4_no_padding_features_pca_384/subject_103_1242_00_faces_26_12_features.h5", help='Path to frontal reference H5 file')
    parser.add_argument('--detector', '-d', choices=['ORB', 'SIFT'], default='ORB', help='Feature detector to use (ORB is faster, SIFT is more robust)')
    parser.add_argument('--start-frame', type=int, default=0, help='Starting frame number (0-indexed)')
    parser.add_argument('--end-frame', type=int, default=None, help='Ending frame number')
    parser.add_argument('--side-by-side', '-s', action='store_true', help='Create side-by-side comparison video')
    parser.add_argument('--no-progress', action='store_true', help='Hide progress bar')
    parser.add_argument('--debug', action='store_true', help='Save debug images and show detailed logging')
    
    args = parser.parse_args()
    
    # Check input H5 file exists
    if args.input_h5 is None:
        print("❌ Error: Input H5 file is required")
        print("Usage: python test_frontalization_video.py input.h5 --frontal-h5 frontal.h5")
        return
    
    if not os.path.exists(args.input_h5):
        print(f"❌ Error: Input H5 file not found: {args.input_h5}")
        return
    
    if not os.path.exists(args.frontal_h5):
        print(f"❌ Error: Frontal reference H5 file not found: {args.frontal_h5}")
        return
    
    # Set default output path if not specified
    if args.output is None:
        input_path = Path(args.input_h5)
        if args.side_by_side:
            output_path = input_path.parent / f"{input_path.stem}_frontalized_comparison.mp4"
        else:
            output_path = input_path.parent / f"{input_path.stem}_frontalized.mp4"
    else:
        output_path = args.output
    
    print("🚀 Starting Homography-Based Face Frontalization from H5 Files")
    print("=" * 60)
    
    # Setup memory management
    setup_memory_management()
    check_memory_usage()
    
    # Load H5 data
    input_data = load_h5_data(args.input_h5)
    if input_data is None:
        print("❌ Cannot proceed without input H5 data. Exiting.")
        return
    
    frontal_data = load_h5_data(args.frontal_h5)
    if frontal_data is None:
        print("❌ Cannot proceed without frontal reference H5 data. Exiting.")
        return
    
    # Process video
    if args.side_by_side:
        success = create_side_by_side_video(
            args.input_h5, 
            output_path, 
            args.frontal_h5,
            args.start_frame, 
            args.end_frame,
            not args.no_progress,
            args.detector,
            args.debug
        )
    else:
        success = process_video(
            args.input_h5, 
            output_path, 
            args.frontal_h5,
            args.start_frame, 
            args.end_frame,
            not args.no_progress,
            args.detector,
            args.debug
        )
    
    if success:
        print(f"\n🎉 Video processing completed successfully!")
        print(f"   Input H5: {args.input_h5}")
        print(f"   Frontal reference H5: {args.frontal_h5}")
        print(f"   Output: {output_path}")
        print(f"   Detector: {args.detector}")
    else:
        print(f"\n❌ Video processing failed!")

def test_debug_functionality():
    """Test if debug functionality works"""
    print("🧪 Testing debug functionality...")
    
    # Test PCA visualization
    test_features = np.random.randn(1369, 384).astype(np.float32)
    test_viz = create_pca_rgb_visualization(test_features, debug=True)
    
    if test_viz is not None:
        print("✅ PCA visualization test passed")
        print(f"   Shape: {test_viz.shape}")
        print(f"   Dtype: {test_viz.dtype}")
        
        # Test saving debug frame
        test_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        save_debug_frame(test_frame, 0, "test")
        
        print("✅ Debug functionality test completed")
        return True
    else:
        print("❌ PCA visualization test failed")
        return False

if __name__ == "__main__":
    # Test debug functionality if requested
    if len(sys.argv) > 1 and sys.argv[1] == "--test-debug":
        test_debug_functionality()
    else:
        main()
