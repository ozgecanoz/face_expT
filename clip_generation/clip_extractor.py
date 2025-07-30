#!/usr/bin/env python3
"""
Clip Extractor
Extracts video clips based on analysis results from VideoExpressionAnalyzer
"""

import cv2
import numpy as np
import json
import os
import logging
from typing import List, Dict, Any
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClipExtractor:
    """
    Extracts video clips based on sequence analysis results
    """
    
    def __init__(self, output_dir: str = "./extracted_clips"):
        """
        Initialize clip extractor
        
        Args:
            output_dir: Directory to save extracted clips
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"✅ Clip Extractor initialized with output directory: {output_dir}")
    
    def extract_clips_from_analysis(self, 
                                   video_path: str, 
                                   analysis_results_path: str,
                                   clip_format: str = "mp4") -> List[str]:
        """
        Extract clips from video based on analysis results
        
        Args:
            video_path: Path to original video
            analysis_results_path: Path to analysis results directory
            clip_format: Format for output clips (mp4, avi, etc.)
            
        Returns:
            List of paths to extracted clips
        """
        logger.info(f"🎬 Extracting clips from: {video_path}")
        
        # Load analysis results
        sequences_file = os.path.join(analysis_results_path, "sequences.json")
        if not os.path.exists(sequences_file):
            raise FileNotFoundError(f"Analysis results not found: {sequences_file}")
        
        with open(sequences_file, 'r') as f:
            sequences = json.load(f)
        
        logger.info(f"📊 Loaded {len(sequences)} sequences from analysis")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"📹 Video properties: {width}x{height}, {fps} FPS")
        
        extracted_clips = []
        
        # Extract each sequence as a clip
        for i, sequence in enumerate(sequences):
            start_frame = sequence['start_frame']
            end_frame = sequence['end_frame']
            score = sequence['combined_score']
            
            logger.info(f"🎬 Extracting clip {i+1}/{len(sequences)}: "
                       f"frames {start_frame}-{end_frame}, score={score:.3f}")
            
            clip_path = self._extract_sequence_clip(
                cap, start_frame, end_frame, fps, width, height, i+1, clip_format
            )
            
            if clip_path:
                extracted_clips.append(clip_path)
        
        cap.release()
        
        logger.info(f"✅ Extracted {len(extracted_clips)} clips successfully")
        return extracted_clips
    
    def _extract_sequence_clip(self, 
                               cap: cv2.VideoCapture, 
                               start_frame: int, 
                               end_frame: int,
                               fps: int,
                               width: int,
                               height: int,
                               clip_number: int,
                               clip_format: str) -> str:
        """
        Extract a single sequence as a video clip
        
        Args:
            cap: OpenCV video capture object
            start_frame: Starting frame index
            end_frame: Ending frame index
            fps: Video frame rate
            width: Video width
            height: Video height
            clip_number: Clip number for naming
            clip_format: Output video format
            
        Returns:
            Path to extracted clip
        """
        # Create output filename
        clip_filename = f"clip_{clip_number:03d}_frames_{start_frame:06d}_{end_frame:06d}.{clip_format}"
        clip_path = os.path.join(self.output_dir, clip_filename)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') if clip_format == 'mp4' else cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            logger.error(f"Could not create video writer for: {clip_path}")
            return None
        
        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Extract frames
        num_frames = end_frame - start_frame + 1
        for frame_idx in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Failed to read frame at position {start_frame + frame_idx}")
                break
            
            out.write(frame)
        
        out.release()
        
        logger.info(f"✅ Extracted clip: {clip_filename}")
        return clip_path
    
    def create_clip_metadata(self, 
                            analysis_results_path: str,
                            extracted_clips: List[str]) -> str:
        """
        Create metadata file for extracted clips
        
        Args:
            analysis_results_path: Path to analysis results directory
            extracted_clips: List of paths to extracted clips
            
        Returns:
            Path to metadata file
        """
        # Load analysis results
        sequences_file = os.path.join(analysis_results_path, "sequences.json")
        with open(sequences_file, 'r') as f:
            sequences = json.load(f)
        
        # Create metadata
        metadata = {
            "extraction_info": {
                "total_clips": len(extracted_clips),
                "output_directory": self.output_dir,
                "analysis_results_path": analysis_results_path
            },
            "clips": []
        }
        
        for i, (clip_path, sequence) in enumerate(zip(extracted_clips, sequences)):
            clip_info = {
                "clip_number": i + 1,
                "clip_path": clip_path,
                "start_frame": sequence['start_frame'],
                "end_frame": sequence['end_frame'],
                "expression_variation": sequence['expression_variation'],
                "position_stability": sequence['position_stability'],
                "combined_score": sequence['combined_score'],
                "frame_count": sequence['end_frame'] - sequence['start_frame'] + 1
            }
            metadata["clips"].append(clip_info)
        
        # Save metadata
        metadata_path = os.path.join(self.output_dir, "clip_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Created clip metadata: {metadata_path}")
        return metadata_path
    
    def create_summary_report(self, 
                             analysis_results_path: str,
                             extracted_clips: List[str]) -> str:
        """
        Create a summary report of the extraction process
        
        Args:
            analysis_results_path: Path to analysis results directory
            extracted_clips: List of paths to extracted clips
            
        Returns:
            Path to summary report
        """
        # Load analysis results
        sequences_file = os.path.join(analysis_results_path, "sequences.json")
        with open(sequences_file, 'r') as f:
            sequences = json.load(f)
        
        # Calculate statistics
        scores = [seq['combined_score'] for seq in sequences]
        expr_vars = [seq['expression_variation'] for seq in sequences]
        pos_stabs = [seq['position_stability'] for seq in sequences]
        
        report_lines = [
            "CLIP EXTRACTION SUMMARY REPORT",
            "=" * 50,
            "",
            f"Total clips extracted: {len(extracted_clips)}",
            f"Output directory: {self.output_dir}",
            "",
            "SCORING STATISTICS:",
            f"  Average combined score: {np.mean(scores):.3f} ± {np.std(scores):.3f}",
            f"  Average expression variation: {np.mean(expr_vars):.3f} ± {np.std(expr_vars):.3f}",
            f"  Average position stability: {np.mean(pos_stabs):.3f} ± {np.std(pos_stabs):.3f}",
            "",
            "TOP 5 CLIPS BY SCORE:",
        ]
        
        # Sort sequences by score
        sorted_sequences = sorted(sequences, key=lambda x: x['combined_score'], reverse=True)
        
        for i, seq in enumerate(sorted_sequences[:5]):
            report_lines.append(
                f"  {i+1}. Frames {seq['start_frame']:06d}-{seq['end_frame']:06d}: "
                f"Score={seq['combined_score']:.3f}, "
                f"Expr={seq['expression_variation']:.3f}, "
                f"Pos={seq['position_stability']:.3f}"
            )
        
        report_lines.extend([
            "",
            "EXTRACTED CLIPS:",
        ])
        
        for i, clip_path in enumerate(extracted_clips):
            clip_name = os.path.basename(clip_path)
            report_lines.append(f"  {i+1}. {clip_name}")
        
        # Save report
        report_path = os.path.join(self.output_dir, "extraction_summary.txt")
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"✅ Created summary report: {report_path}")
        return report_path


def extract_clip_frames(video_path: str, start_frame: int, end_frame: int) -> List[np.ndarray]:
    """
    Extract frames from a video clip.
    
    Args:
        video_path (str): Path to video file
        start_frame (int): Starting frame index
        end_frame (int): Ending frame index
        
    Returns:
        List[np.ndarray]: List of frames as numpy arrays
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    for frame_idx in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames


def extract_cropped_face_frames(video_path: str, start_frame: int, end_frame: int, 
                               face_detector=None, confidence_threshold: float = 0.5) -> List[np.ndarray]:
    """
    Extract cropped face frames from a video clip.
    
    Args:
        video_path (str): Path to video file
        start_frame (int): Starting frame index
        end_frame (int): Ending frame index
        face_detector: MediaPipe face detector instance (optional)
        confidence_threshold (float): Minimum confidence for face detection
        
    Returns:
        List[np.ndarray]: List of cropped face frames (518x518) as numpy arrays
    """
    import cv2
    import mediapipe as mp
    
    # Initialize face detector if not provided
    if face_detector is None:
        mp_face_detection = mp.solutions.face_detection
        face_detector = mp_face_detection.FaceDetection(
            model_selection=1,  # 0 for short-range, 1 for full-range
            min_detection_confidence=confidence_threshold
        )
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cropped_frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    for frame_idx in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces in the frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb_frame)
        
        if results.detections:
            # Use the face with highest confidence
            best_detection = max(results.detections, key=lambda x: x.score[0])
            
            if best_detection.score[0] >= confidence_threshold:
                # Get bounding box
                bbox = best_detection.location_data.relative_bounding_box
                
                # Convert relative coordinates to absolute
                x = int(bbox.xmin * width)
                y = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)
                
                # Extract face region with padding (matching original approach)
                padding = int(min(w, h) * 0.1)  # 10% padding
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(width, x + w + padding)
                y2 = min(height, y + h + padding)
                
                # Crop face with padding
                face_crop = frame[y1:y2, x1:x2]
                
                if face_crop.size > 0:
                    # Resize to 518x518
                    face_crop = cv2.resize(face_crop, (518, 518))
                    cropped_frames.append(face_crop)
                else:
                    logger.warning(f"Empty face crop in frame {frame_idx}")
                    # Add a black frame as placeholder
                    cropped_frames.append(np.zeros((518, 518, 3), dtype=np.uint8))
            else:
                logger.warning(f"No face detected with sufficient confidence in frame {frame_idx}")
                # Add a black frame as placeholder
                cropped_frames.append(np.zeros((518, 518, 3), dtype=np.uint8))
        else:
            logger.warning(f"No faces detected in frame {frame_idx}")
            # Add a black frame as placeholder
            cropped_frames.append(np.zeros((518, 518, 3), dtype=np.uint8))
    
    cap.release()
    
    if face_detector is None:  # If we created the detector, close it
        face_detector.close()
    
    return cropped_frames


def save_clip_as_h5(frames: List[np.ndarray], output_path: str, metadata: Dict) -> bool:
    """
    Save clip frames as H5 file in the exact format from serialize_dataset.py.
    
    Args:
        frames (List[np.ndarray]): List of cropped face frames (518x518)
        output_path (str): Output H5 file path
        metadata (Dict): Metadata to save with the clip
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import h5py
        from datetime import datetime
        
        with h5py.File(output_path, 'w') as f:
            # Create groups (matching original format)
            faces_group = f.create_group('faces')
            metadata_group = f.create_group('metadata')
            
            # Store face data (matching original format)
            for i, frame in enumerate(frames):
                frame_group = faces_group.create_group(f'frame_{i:03d}')
                
                # Create face_000 group for each frame (matching original)
                face_group = frame_group.create_group('face_000')
                
                # Store face data with compression (matching original)
                face_group.create_dataset('data', data=frame, compression='gzip', compression_opts=9)
                
                # Store bbox (use full frame as bbox since these are already cropped)
                face_group.create_dataset('bbox', data=[0, 0, 518, 518])
                
                # Store confidence (default to 1.0 since these are pre-cropped)
                face_group.create_dataset('confidence', data=1.0)
                
                # Store original_size (518x518 since these are cropped)
                face_group.create_dataset('original_size', data=[518, 518])
            
            # Store metadata (matching original format)
            metadata_group.create_dataset('video_path', data=metadata.get('video_id', ''))
            metadata_group.create_dataset('timestamp', data=metadata.get('start_frame', 0) / 30.0)  # Approximate timestamp
            metadata_group.create_dataset('frame_timestamps', data=np.arange(len(frames)) / 30.0)  # Approximate timestamps
            metadata_group.create_dataset('face_size', data=[518, 518])
            metadata_group.create_dataset('num_frames', data=len(frames))
            metadata_group.create_dataset('frame_skip', data=0)
            metadata_group.create_dataset('fps', data=30)
            metadata_group.create_dataset('extraction_date', data=datetime.now().isoformat())
            
            # Store subject information (matching original format)
            if 'subject_id' in metadata:
                metadata_group.create_dataset('subject_id', data=metadata['subject_id'])
            
            if 'subject_label' in metadata and metadata['subject_label']:
                # Store each label field separately (matching original)
                for key, value in metadata['subject_label'].items():
                    metadata_group.create_dataset(f'subject_{key}', data=str(value))
            
            # Store additional metadata as attributes
            for key, value in metadata.items():
                if key not in ['video_id', 'subject_id', 'subject_label'] and isinstance(value, (str, int, float, bool)):
                    f.attrs[key] = value
        
        return True
    except Exception as e:
        logger.error(f"Failed to save H5 file {output_path}: {e}")
        return False


def save_clip_as_mp4(frames: List[np.ndarray], output_path: str, fps: int = 30) -> bool:
    """
    Save clip frames as MP4 file.
    
    Args:
        frames (List[np.ndarray]): List of frames
        output_path (str): Output MP4 file path
        fps (int): Frame rate for output video
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if not frames:
            return False
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        return True
    except Exception as e:
        logger.error(f"Failed to save MP4 file {output_path}: {e}")
        return False


def main():
    """Test the clip extractor"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract video clips from analysis results")
    parser.add_argument("--video_path", type=str, required=True, help="Path to original video")
    parser.add_argument("--analysis_results", type=str, required=True, help="Path to analysis results directory")
    parser.add_argument("--output_dir", type=str, default="./extracted_clips", help="Directory to save clips")
    parser.add_argument("--clip_format", type=str, default="mp4", help="Format for output clips")
    
    args = parser.parse_args()
    
    # Create clip extractor
    extractor = ClipExtractor(output_dir=args.output_dir)
    
    # Extract clips
    extracted_clips = extractor.extract_clips_from_analysis(
        args.video_path, 
        args.analysis_results,
        args.clip_format
    )
    
    if extracted_clips:
        # Create metadata and report
        extractor.create_clip_metadata(args.analysis_results, extracted_clips)
        extractor.create_summary_report(args.analysis_results, extracted_clips)
        
        logger.info("🎉 Clip extraction completed successfully!")
    else:
        logger.error("❌ No clips were extracted!")


if __name__ == "__main__":
    main() 