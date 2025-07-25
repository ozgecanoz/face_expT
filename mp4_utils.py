"""
MP4 Utilities Module

A collection of functions for extracting frames and audio from MP4 files.
Requires: opencv-python, moviepy, numpy
"""

import cv2
import numpy as np
import os
from typing import List, Tuple, Optional, Union, Dict
from moviepy import VideoFileClip, AudioFileClip
import tempfile
import logging
import h5py
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_frames(
    video_path: str, 
    output_dir: str = None, 
    frame_rate: int = 1,
    start_time: float = 0,
    end_time: Optional[float] = None,
    quality: int = 95
) -> List[str]:
    """
    Extract frames from an MP4 video file.
    
    Args:
        video_path (str): Path to the input MP4 file
        output_dir (str): Directory to save extracted frames (default: same as video)
        frame_rate (int): Number of frames to extract per second (default: 1)
        start_time (float): Start time in seconds (default: 0)
        end_time (float): End time in seconds (default: end of video)
        quality (int): JPEG quality for saved frames (1-100, default: 95)
    
    Returns:
        List[str]: List of paths to extracted frame files
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Create output directory if it doesn't exist
    if output_dir is None:
        output_dir = os.path.splitext(video_path)[0] + "_frames"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    # Calculate frame extraction parameters
    frame_interval = int(fps / frame_rate)
    start_frame = int(start_time * fps)
    
    if end_time is None:
        end_frame = total_frames
    else:
        end_frame = int(end_time * fps)
    
    extracted_frames = []
    frame_count = 0
    
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_frame = start_frame + frame_count
            
            if current_frame >= end_frame:
                break
            
            # Extract frame at specified interval
            if frame_count % frame_interval == 0:
                frame_filename = f"frame_{current_frame:06d}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                
                # Save frame with specified quality
                cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                extracted_frames.append(frame_path)
                
                logger.info(f"Extracted frame {current_frame} -> {frame_path}")
            
            frame_count += 1
            
    finally:
        cap.release()
    
    logger.info(f"Extracted {len(extracted_frames)} frames to {output_dir}")
    return extracted_frames


def extract_audio(
    video_path: str,
    output_path: str = None,
    audio_format: str = "mp3",
    start_time: float = 0,
    end_time: Optional[float] = None,
    audio_codec: str = "libmp3lame"
) -> str:
    """
    Extract audio from an MP4 video file.
    
    Args:
        video_path (str): Path to the input MP4 file
        output_path (str): Path for the output audio file (default: auto-generated)
        audio_format (str): Output audio format (mp3, wav, aac, etc.)
        start_time (float): Start time in seconds (default: 0)
        end_time (float): End time in seconds (default: end of video)
        audio_codec (str): Audio codec to use for encoding
    
    Returns:
        str: Path to the extracted audio file
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Generate output path if not provided
    if output_path is None:
        base_name = os.path.splitext(video_path)[0]
        output_path = f"{base_name}_audio.{audio_format}"
    
    try:
        # Load video clip
        video_clip = VideoFileClip(video_path)
        
        # Extract audio
        audio_clip = video_clip.audio
        
        if audio_clip is None:
            raise ValueError("No audio track found in the video file")
        
        # Trim audio if time limits are specified
        if start_time > 0 or end_time is not None:
            if end_time is None:
                end_time = audio_clip.duration
            audio_clip = audio_clip.subclip(start_time, end_time)
        
        # Write audio file
        audio_clip.write_audiofile(
            output_path,
            codec=audio_codec,
            verbose=False,
            logger=None
        )
        
        # Close clips to free memory
        audio_clip.close()
        video_clip.close()
        
        logger.info(f"Audio extracted successfully: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error extracting audio: {str(e)}")
        raise


def get_video_info(video_path: str) -> dict:
    """
    Get detailed information about an MP4 video file.
    
    Args:
        video_path (str): Path to the MP4 file
    
    Returns:
        dict: Dictionary containing video information
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    try:
        # Get basic video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        # Get video codec
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
        # Get file size
        file_size = os.path.getsize(video_path)
        
        info = {
            "file_path": video_path,
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "duration_seconds": round(duration, 2),
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "resolution": f"{width}x{height}",
            "codec": codec,
            "aspect_ratio": round(width / height, 2) if height > 0 else 0
        }
        
        return info
        
    finally:
        cap.release()


def extract_frames_at_timestamps(
    video_path: str,
    timestamps: List[float],
    output_dir: str = None,
    quality: int = 95
) -> List[str]:
    """
    Extract frames at specific timestamps from an MP4 video.
    
    Args:
        video_path (str): Path to the input MP4 file
        timestamps (List[float]): List of timestamps in seconds
        output_dir (str): Directory to save extracted frames
        quality (int): JPEG quality for saved frames (1-100)
    
    Returns:
        List[str]: List of paths to extracted frame files
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if output_dir is None:
        output_dir = os.path.splitext(video_path)[0] + "_frames"
    
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    extracted_frames = []
    
    try:
        for i, timestamp in enumerate(timestamps):
            frame_number = int(timestamp * fps)
            
            # Set position to specific frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if ret:
                frame_filename = f"frame_{timestamp:.2f}s_{i:03d}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                
                cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                extracted_frames.append(frame_path)
                
                logger.info(f"Extracted frame at {timestamp}s -> {frame_path}")
            else:
                logger.warning(f"Could not extract frame at timestamp {timestamp}s")
                
    finally:
        cap.release()
    
    return extracted_frames


def create_video_thumbnail(
    video_path: str,
    output_path: str = None,
    timestamp: float = 0,
    size: Tuple[int, int] = (320, 240),
    quality: int = 95
) -> str:
    """
    Create a thumbnail image from an MP4 video at a specific timestamp.
    
    Args:
        video_path (str): Path to the input MP4 file
        output_path (str): Path for the output thumbnail (default: auto-generated)
        timestamp (float): Timestamp in seconds to extract frame from
        size (Tuple[int, int]): Size of the thumbnail (width, height)
        quality (int): JPEG quality for the thumbnail (1-100)
    
    Returns:
        str: Path to the created thumbnail
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if output_path is None:
        base_name = os.path.splitext(video_path)[0]
        output_path = f"{base_name}_thumbnail.jpg"
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(timestamp * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        if ret:
            # Resize frame to thumbnail size
            resized_frame = cv2.resize(frame, size)
            
            # Save thumbnail
            cv2.imwrite(output_path, resized_frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            
            logger.info(f"Thumbnail created: {output_path}")
            return output_path
        else:
            raise ValueError(f"Could not extract frame at timestamp {timestamp}s")
            
    finally:
        cap.release()


def batch_extract_audio(
    video_files: List[str],
    output_dir: str = None,
    audio_format: str = "mp3"
) -> List[str]:
    """
    Extract audio from multiple MP4 files in batch.
    
    Args:
        video_files (List[str]): List of paths to MP4 files
        output_dir (str): Directory to save extracted audio files
        audio_format (str): Output audio format
    
    Returns:
        List[str]: List of paths to extracted audio files
    """
    if output_dir is None:
        output_dir = "extracted_audio"
    
    os.makedirs(output_dir, exist_ok=True)
    extracted_files = []
    
    for video_file in video_files:
        try:
            base_name = os.path.splitext(os.path.basename(video_file))[0]
            output_path = os.path.join(output_dir, f"{base_name}.{audio_format}")
            
            extracted_path = extract_audio(video_file, output_path, audio_format)
            extracted_files.append(extracted_path)
            
        except Exception as e:
            logger.error(f"Error processing {video_file}: {str(e)}")
    
    logger.info(f"Batch extraction complete. {len(extracted_files)} files processed.")
    return extracted_files


def extract_face_sequence(
    video_path: str,
    timestamp: float,
    output_folder: str,
    face_size: Tuple[int, int] = (518, 518),
    num_frames: int = 15,
    frame_skip: int = 1,
    confidence_threshold: float = 0.5,
    subject_id: str = None,
    subject_label: Dict = None
) -> dict:
    """
    Extract face sequences from an MP4 video at a specific timestamp.
    
    Args:
        video_path (str): Path to the input MP4 file
        timestamp (float): Start timestamp in seconds
        output_folder (str): Folder where both HDF5 and MP4 files will be written
        face_size (Tuple[int, int]): Size to resize face crops (default: 518x518)
        num_frames (int): Number of frames to extract (default: 15)
        frame_skip (int): Number of frames to skip between extractions (default: 1)
        confidence_threshold (float): Minimum confidence for face detection (default: 0.5)
        subject_id (str): Subject ID from JSON annotations
        subject_label (Dict): Subject label information (age, gender, skin-type)
    
    Returns:
        dict: Dictionary containing output paths and metadata
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize MediaPipe face detection
    import mediapipe as mp
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(
        model_selection=1,  # 0 for short-range, 1 for full-range
        min_detection_confidence=confidence_threshold
    )
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate frame positions
    start_frame = int(timestamp * fps)
    frame_interval = frame_skip + 1  # Skip 1 frame for 30fps = every 2nd frame
    
    # Validate timestamp and ensure enough frames are available
    if start_frame >= total_frames:
        raise ValueError(f"Timestamp {timestamp}s is beyond video duration")
    
    # Check if we have enough frames from this timestamp
    end_frame = start_frame + (num_frames * frame_interval)
    if end_frame > total_frames:
        logger.warning(f"Not enough frames available from timestamp {timestamp}s. Need {num_frames} frames but only {total_frames - start_frame} frames remaining")
        return None
    
    # Prepare data structures
    face_sequences = []
    frame_timestamps = []
    face_metadata = []
    
    # Generate output filenames with subject ID
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    timestamp_str = f"{timestamp:.2f}".replace('.', '_')
    
    # Include subject_id in filename if available
    if subject_id:
        hdf5_filename = f"subject_{subject_id}_{video_name}_faces_{timestamp_str}.h5"
        mp4_filename = f"subject_{subject_id}_{video_name}_faces_{timestamp_str}.mp4"
    else:
        hdf5_filename = f"{video_name}_faces_{timestamp_str}.h5"
        mp4_filename = f"{video_name}_faces_{timestamp_str}.mp4"
    
    hdf5_path = os.path.join(output_folder, hdf5_filename)
    mp4_path = os.path.join(output_folder, mp4_filename)
    
    # Prepare video writer for face sequence
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    face_video_writer = cv2.VideoWriter(mp4_path, fourcc, fps, face_size)
    
    try:
        # Extract frames and detect faces
        for i in range(num_frames):
            frame_number = start_frame + (i * frame_interval)
            
            if frame_number >= total_frames:
                logger.warning(f"Reached end of video at frame {frame_number}")
                break
            
            # Set position and read frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if not ret:
                logger.warning(f"Could not read frame {frame_number}")
                continue
            
            # Detect faces using MediaPipe
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)
            
            frame_timestamp = frame_number / fps
            frame_timestamps.append(frame_timestamp)
            
            # Process detected faces
            frame_faces = []
            if results.detections:
                # Check if more than one face is detected
                if len(results.detections) > 1:
                    logger.warning(f"Multiple faces detected in frame {frame_number} ({len(results.detections)} faces), discarding entire clip")
                    # Clean up any created files and return None
                    if os.path.exists(hdf5_path):
                        os.remove(hdf5_path)
                    if os.path.exists(mp4_path):
                        os.remove(mp4_path)
                    return None
                
                # Process the single detected face
                detection = results.detections[0]  # Take the first (and only) face
                
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                
                # Convert relative coordinates to absolute
                x = int(bbox.xmin * width)
                y = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)
                
                # Get confidence score
                confidence = detection.score[0]
                
                if confidence >= confidence_threshold:
                    # Extract face region with padding
                    padding = int(min(w, h) * 0.1)  # 10% padding
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(width, x + w + padding)
                    y2 = min(height, y + h + padding)
                    
                    # Crop and resize face
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size > 0:
                        face_resized = cv2.resize(face_crop, face_size)
                        
                        # Convert to RGB for storage
                        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                        
                        frame_faces.append({
                            'face_data': face_rgb,
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'original_size': [w, h]
                        })
                        
                        # Add to video writer
                        face_video_writer.write(face_resized)
            else:
                # No faces detected in this frame, discard entire clip
                logger.warning(f"No faces detected in frame {frame_number}, discarding entire clip")
                # Clean up any created files and return None
                if os.path.exists(hdf5_path):
                    os.remove(hdf5_path)
                if os.path.exists(mp4_path):
                    os.remove(mp4_path)
                return None
            
            face_sequences.append(frame_faces)
            
            # Log progress
            logger.info(f"Processed frame {frame_number} at {frame_timestamp:.2f}s - Found {len(frame_faces)} faces")
        
        # Close video writer
        face_video_writer.release()
        
        # Count total faces found and frames with single faces
        total_faces = sum(len(frame_faces) for frame_faces in face_sequences)
        frames_with_single_face = sum(1 for frame_faces in face_sequences if len(frame_faces) == 1)
        expected_frames = num_frames  # Should be 30 frames when frame_skip=0
        
        logger.info(f"Total faces found: {total_faces} across {len(face_sequences)} frames")
        logger.info(f"Frames with single face: {frames_with_single_face} out of {expected_frames}")
        
        # Check if we have enough frames with single faces
        if frames_with_single_face < expected_frames:
            logger.warning(f"Insufficient frames with single faces ({frames_with_single_face} < {expected_frames}), skipping clip")
            # Clean up created files
            if os.path.exists(hdf5_path):
                os.remove(hdf5_path)
            if os.path.exists(mp4_path):
                os.remove(mp4_path)
            return None
        
        # Save to HDF5 file
        with h5py.File(hdf5_path, 'w') as f:
            # Create groups
            faces_group = f.create_group('faces')
            metadata_group = f.create_group('metadata')
            
            # Store face data
            for i, frame_faces in enumerate(face_sequences):
                if frame_faces:
                    frame_group = faces_group.create_group(f'frame_{i:03d}')
                    
                    for j, face in enumerate(frame_faces):
                        face_group = frame_group.create_group(f'face_{j:03d}')
                        face_group.create_dataset('data', data=face['face_data'], compression='gzip', compression_opts=9)
                        face_group.create_dataset('bbox', data=face['bbox'])
                        face_group.create_dataset('confidence', data=face['confidence'])
                        face_group.create_dataset('original_size', data=face['original_size'])
            
            # Store metadata
            metadata_group.create_dataset('video_path', data=video_path)
            metadata_group.create_dataset('timestamp', data=timestamp)
            metadata_group.create_dataset('frame_timestamps', data=frame_timestamps)
            metadata_group.create_dataset('face_size', data=face_size)
            metadata_group.create_dataset('num_frames', data=num_frames)
            metadata_group.create_dataset('frame_skip', data=frame_skip)
            metadata_group.create_dataset('fps', data=fps)
            metadata_group.create_dataset('extraction_date', data=datetime.now().isoformat())
            
            # Store subject information if available
            if subject_id:
                metadata_group.create_dataset('subject_id', data=subject_id)
            
            if subject_label:
                # Store each label field separately
                for key, value in subject_label.items():
                    metadata_group.create_dataset(f'subject_{key}', data=str(value))
            
            # Store video properties
            video_info = get_video_info(video_path)
            for key, value in video_info.items():
                if isinstance(value, (int, float, str)):
                    metadata_group.create_dataset(f'video_{key}', data=str(value))
        
        logger.info(f"Face sequence extracted successfully:")
        logger.info(f"  HDF5 file: {hdf5_path}")
        logger.info(f"  MP4 file: {mp4_path}")
        logger.info(f"  Total frames processed: {len(face_sequences)}")
        logger.info(f"  Frames with single face: {frames_with_single_face}")
        logger.info(f"  Total faces found: {total_faces}")
        
        return {
            'hdf5_path': hdf5_path,
            'mp4_path': mp4_path,
            'num_frames': len(face_sequences),
            'total_faces': total_faces,
            'frames_with_single_face': frames_with_single_face,
            'timestamp': timestamp,
            'video_path': video_path,
            'subject_id': subject_id,
            'subject_label': subject_label
        }
        
    finally:
        cap.release()


if __name__ == "__main__":
    # Example usage
    video_path = "example.mp4"
    
    # Get video information
    try:
        info = get_video_info(video_path)
        print("Video Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    except FileNotFoundError:
        print(f"Example video file '{video_path}' not found. Please provide a valid MP4 file.") 