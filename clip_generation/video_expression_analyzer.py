#!/usr/bin/env python3
"""
Video Expression Analyzer
Analyzes videos to find sequences with high expression variation and stable face position
"""

import cv2
import numpy as np
import torch
import json
import os
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from tqdm import tqdm
import mediapipe as mp

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from face_model.models.dinov2_tokenizer import DINOv2Tokenizer
from face_model.models.expression_transformer import ExpressionTransformer
from app_demo.model_loader import ModelLoader

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FrameData:
    """Data structure for storing frame analysis results"""
    frame_idx: int
    bbox: Optional[Tuple[int, int, int, int]]  # (x, y, w, h) or None if no face
    confidence: float
    expression_token: Optional[torch.Tensor]  # None if no face or extraction failed
    face_center: Tuple[float, float]  # (x, y) center coordinates


@dataclass
class SequenceScore:
    """Data structure for sequence scoring results"""
    start_frame: int
    end_frame: int
    expression_variation: float
    position_stability: float
    combined_score: float
    frame_data: List[FrameData]


class VideoExpressionAnalyzer:
    """
    Analyzes videos to find sequences with high expression variation and stable face position
    """
    
    def __init__(self, 
                 expression_transformer_checkpoint_path: str,
                 device: str = "cpu",
                 face_confidence_threshold: float = 0.5):
        """
        Initialize the video expression analyzer
        
        Args:
            expression_transformer_checkpoint_path: Path to expression transformer checkpoint
            device: Device to run models on
            face_confidence_threshold: Minimum confidence for face detection
        """
        self.device = device
        self.face_confidence_threshold = face_confidence_threshold
        
        # Initialize MediaPipe face detection
        import mediapipe as mp
        mp_face_detection = mp.solutions.face_detection
        self.face_detection = mp_face_detection.FaceDetection(
            model_selection=1,  # 0 for short-range, 1 for full-range
            min_detection_confidence=face_confidence_threshold
        )
        
        # Initialize DINOv2 tokenizer
        from face_model.models.dinov2_tokenizer import DINOv2Tokenizer
        self.tokenizer = DINOv2Tokenizer(device=device)
        
        # Load expression transformer
        import torch
        checkpoint = torch.load(expression_transformer_checkpoint_path, map_location=device)
        
        # Get architecture from checkpoint
        if 'config' in checkpoint and 'expression_model' in checkpoint['config']:
            expr_config = checkpoint['config']['expression_model']
            embed_dim = expr_config.get('embed_dim', 384)
            num_heads = expr_config.get('num_heads', 4)
            num_layers = expr_config.get('num_layers', 2)
            dropout = expr_config.get('dropout', 0.1)
            max_subjects = expr_config.get('max_subjects', 3500)
        else:
            # Default architecture
            embed_dim, num_heads, num_layers, dropout, max_subjects = 384, 4, 2, 0.1, 3500
        
        # Initialize expression transformer
        from face_model.models.expression_transformer import ExpressionTransformer
        self.expression_transformer = ExpressionTransformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            max_subjects=max_subjects
        ).to(device)
        
        # Load weights
        if 'expression_transformer_state_dict' in checkpoint:
            self.expression_transformer.load_state_dict(checkpoint['expression_transformer_state_dict'])
        else:
            self.expression_transformer.load_state_dict(checkpoint)
        
        self.expression_transformer.eval()
        
        logger.info("✅ Video Expression Analyzer initialized successfully!")
    
    def sample_random_clips(self, video_path: str, N: int, sequence_length: int, min_duration: float = 5.0) -> List[Tuple[int, int]]:
        """
        Sample N random clips from a video
        
        Args:
            video_path: Path to input video
            N: Number of clips to sample
            sequence_length: Length of each clip in frames
            min_duration: Minimum duration in seconds to ensure video is long enough
            
        Returns:
            List of (start_frame, end_frame) tuples
        """
        # Open video to get properties
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        duration = total_frames / fps if fps > 0 else 0
        
        cap.release()
        
        logger.info(f"📹 Video properties: {total_frames} frames, {fps} FPS, {duration:.1f}s duration")
        
        if duration < min_duration:
            logger.warning(f"Video duration ({duration:.1f}s) is less than minimum ({min_duration}s)")
            return []
        
        if total_frames < sequence_length:
            logger.warning(f"Video has {total_frames} frames, less than sequence length {sequence_length}")
            return []
        
        # Calculate possible starting positions
        max_start_frame = total_frames - sequence_length
        if max_start_frame < 0:
            logger.warning(f"Cannot extract sequences of length {sequence_length} from video with {total_frames} frames")
            return []
        
        # Sample N random starting positions
        import random
        random.seed(42)  # For reproducible results
        
        if N > max_start_frame + 1:
            logger.warning(f"Requested {N} samples but only {max_start_frame + 1} possible starting positions")
            N = max_start_frame + 1
        
        start_positions = random.sample(range(max_start_frame + 1), N)
        
        # Create clip ranges
        clip_ranges = []
        for start_frame in start_positions:
            end_frame = start_frame + sequence_length - 1
            clip_ranges.append((start_frame, end_frame))
        
        logger.info(f"✅ Sampled {len(clip_ranges)} clips from video")
        return clip_ranges
    
    def _detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in a frame using MediaPipe
        
        Args:
            frame: Input frame (H, W, 3) in BGR format
            
        Returns:
            List of face detection results with bbox and confidence
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = self.face_detection.process(rgb_frame)
        
        faces = []
        if results.detections:
            for detection in results.detections:
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Get confidence
                confidence = detection.score[0]
                
                if confidence >= self.face_confidence_threshold:
                    faces.append({
                        'bbox': (x, y, width, height),
                        'confidence': confidence,
                        'center': (x + width // 2, y + height // 2)
                    })
        
        return faces
    
    def _extract_expression_token(self, face_image: np.ndarray, subject_id: int = 0) -> Optional[torch.Tensor]:
        """
        Extract expression token from a face image
        
        Args:
            face_image: Face image (H, W, 3) in RGB format
            subject_id: Subject ID for expression extraction
            
        Returns:
            Expression token tensor or None if failed
        """
        try:
            # Ensure face image is 518x518
            if face_image.shape[:2] != (518, 518):
                face_image = cv2.resize(face_image, (518, 518))
            
            # Convert to tensor and normalize
            face_tensor = torch.from_numpy(face_image).float().permute(2, 0, 1).unsqueeze(0)  # (1, 3, 518, 518)
            face_tensor = face_tensor / 255.0  # Normalize to [0, 1]
            face_tensor = face_tensor.to(self.device)
            
            # Extract DINOv2 tokens
            with torch.no_grad():
                patch_tokens, pos_embeddings = self.tokenizer(face_tensor)
                
                # Extract expression token
                subject_ids = torch.full((1,), subject_id, dtype=torch.long, device=self.device)
                expression_token, _ = self.expression_transformer.inference(patch_tokens, pos_embeddings, subject_ids)
            
            return expression_token
            
        except Exception as e:
            logger.warning(f"Failed to extract expression token: {e}")
            return None
    
    def _calculate_expression_variation(self, frame_data: List[FrameData]) -> float:
        """
        Calculate expression variation across a sequence of frames
        
        Args:
            frame_data: List of FrameData objects
            
        Returns:
            Expression variation score (higher = more variation)
        """
        if len(frame_data) < 3:
            return 0.0
        
        # Check that all frames have valid expression tokens
        for frame in frame_data:
            if frame.expression_token is None:
                return 0.0  # Invalid sequence
        
        # Get first, middle, and last frames
        first_frame = frame_data[0]
        middle_frame = frame_data[len(frame_data) // 2]
        last_frame = frame_data[-1]
        
        # Calculate cosine similarities between key frames
        similarities = []
        
        # First to middle
        token1 = first_frame.expression_token.flatten()
        token2 = middle_frame.expression_token.flatten()
        cos_sim_1 = torch.nn.functional.cosine_similarity(
            token1.unsqueeze(0), token2.unsqueeze(0), dim=1
        ).item()
        similarities.append(cos_sim_1)
        
        # Middle to last
        token2 = middle_frame.expression_token.flatten()
        token3 = last_frame.expression_token.flatten()
        cos_sim_2 = torch.nn.functional.cosine_similarity(
            token2.unsqueeze(0), token3.unsqueeze(0), dim=1
        ).item()
        similarities.append(cos_sim_2)
        
        # First to last
        token1 = first_frame.expression_token.flatten()
        token3 = last_frame.expression_token.flatten()
        cos_sim_3 = torch.nn.functional.cosine_similarity(
            token1.unsqueeze(0), token3.unsqueeze(0), dim=1
        ).item()
        similarities.append(cos_sim_3)
        
        # Print detailed similarity scores with maximum precision
        logger.info(f"📊 Expression Similarity Scores (frames {first_frame.frame_idx}, {middle_frame.frame_idx}, {last_frame.frame_idx}):")
        logger.info(f"   First→Middle: {cos_sim_1:.8f}")
        logger.info(f"   Middle→Last:  {cos_sim_2:.8f}")
        logger.info(f"   First→Last:   {cos_sim_3:.8f}")
        logger.info(f"   Average:      {np.mean(similarities):.8f}")
        
        # Expression variation = 1 - average similarity (higher variation = lower similarity)
        avg_similarity = np.mean(similarities)
        variation = 1.0 - avg_similarity
        
        logger.info(f"   Variation:    {variation:.8f}")
        
        return variation
    
    def _calculate_position_stability(self, frame_data: List[FrameData]) -> float:
        """
        Calculate position stability across a sequence of frames
        
        Args:
            frame_data: List of FrameData objects
            
        Returns:
            Position stability score (higher = more stable)
        """
        if len(frame_data) < 2:
            return 1.0
        
        # Check that all frames have valid face centers
        for frame in frame_data:
            if frame.bbox is None:
                return 0.0  # Invalid sequence
        
        # Calculate movement between consecutive frames
        movements = []
        for i in range(len(frame_data) - 1):
            center1 = frame_data[i].face_center
            center2 = frame_data[i + 1].face_center
            
            # Calculate Euclidean distance
            distance = np.sqrt((center2[0] - center1[0])**2 + (center2[1] - center1[1])**2)
            movements.append(distance)
        
        # Normalize by frame size (assume 518x518 for now)
        max_movement = np.sqrt(518**2 + 518**2)  # Diagonal of frame
        normalized_movements = [m / max_movement for m in movements]
        
        # Position stability = 1 - average normalized movement
        avg_movement = np.mean(normalized_movements)
        stability = 1.0 - avg_movement
        
        return max(0.0, stability)  # Ensure non-negative
    
    def _calculate_expression_variation_from_frames(self, face_data: np.ndarray, subject_id: int = 0) -> float:
        """
        Calculate expression variation from pre-extracted face frames
        
        Args:
            face_data: numpy array of shape (num_frames, height, width, channels) with face images
            subject_id: Subject ID for expression token extraction
            
        Returns:
            Expression variation score (higher = more variation)
        """
        if len(face_data) < 3:
            return 0.0
        
        # Extract expression tokens for key frames
        first_frame = face_data[0]
        middle_frame = face_data[len(face_data) // 2]
        last_frame = face_data[-1]
        
        # Convert to tensors and extract expression tokens
        try:
            # First frame
            first_tensor = torch.from_numpy(first_frame).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            first_tokens, _ = self.dinov2_tokenizer(first_tensor)
            first_expression_token, _ = self.expression_transformer.inference(first_tokens, first_tokens, torch.tensor([subject_id]))
            
            # Middle frame
            middle_tensor = torch.from_numpy(middle_frame).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            middle_tokens, _ = self.dinov2_tokenizer(middle_tensor)
            middle_expression_token, _ = self.expression_transformer.inference(middle_tokens, middle_tokens, torch.tensor([subject_id]))
            
            # Last frame
            last_tensor = torch.from_numpy(last_frame).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            last_tokens, _ = self.dinov2_tokenizer(last_tensor)
            last_expression_token, _ = self.expression_transformer.inference(last_tokens, last_tokens, torch.tensor([subject_id]))
            
        except Exception as e:
            logger.warning(f"Error extracting expression tokens: {e}")
            return 0.0
        
        # Calculate cosine similarities between key frames
        similarities = []
        
        # First to middle
        token1 = first_expression_token.flatten()
        token2 = middle_expression_token.flatten()
        cos_sim_1 = torch.nn.functional.cosine_similarity(
            token1.unsqueeze(0), token2.unsqueeze(0), dim=1
        ).item()
        similarities.append(cos_sim_1)
        
        # Middle to last
        token2 = middle_expression_token.flatten()
        token3 = last_expression_token.flatten()
        cos_sim_2 = torch.nn.functional.cosine_similarity(
            token2.unsqueeze(0), token3.unsqueeze(0), dim=1
        ).item()
        similarities.append(cos_sim_2)
        
        # First to last
        token1 = first_expression_token.flatten()
        token3 = last_expression_token.flatten()
        cos_sim_3 = torch.nn.functional.cosine_similarity(
            token1.unsqueeze(0), token3.unsqueeze(0), dim=1
        ).item()
        similarities.append(cos_sim_3)
        
        # Print detailed similarity scores with maximum precision
        logger.info(f"📊 Expression Similarity Scores (frames 0, {len(face_data)//2}, {len(face_data)-1}):")
        logger.info(f"   First→Middle: {cos_sim_1:.8f}")
        logger.info(f"   Middle→Last:  {cos_sim_2:.8f}")
        logger.info(f"   First→Last:   {cos_sim_3:.8f}")
        logger.info(f"   Average:      {np.mean(similarities):.8f}")
        
        # Expression variation = 1 - average similarity (higher variation = lower similarity)
        avg_similarity = np.mean(similarities)
        variation = 1.0 - avg_similarity
        
        logger.info(f"   Variation:    {variation:.8f}")
        
        return variation
    
    def analyze_video_clips(self, video_path: str, clip_ranges: List[Tuple[int, int]], subject_id: int = 0) -> List[FrameData]:
        """
        Analyze specific clips from a video
        
        Args:
            video_path: Path to input video
            clip_ranges: List of (start_frame, end_frame) tuples to analyze
            subject_id: Subject ID for expression extraction
            
        Returns:
            List of FrameData objects for each frame in the clips
        """
        logger.info(f"🎬 Analyzing {len(clip_ranges)} clips from video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        logger.info(f"📹 Video properties: {total_frames} frames, {fps} FPS")
        
        frame_data_list = []
        
        # Process each clip with progress indicator
        for clip_idx, (start_frame, end_frame) in enumerate(clip_ranges):
            logger.info(f"Processing clip {clip_idx+1}/{len(clip_ranges)}: frames {start_frame}-{end_frame}")
            
            # Process frames in this clip
            for frame_idx in range(start_frame, end_frame + 1):
                try:
                    # Seek to frame with timeout protection
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    
                    # Verify we're at the correct frame
                    actual_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    if actual_frame != frame_idx:
                        logger.warning(f"Seek failed: requested frame {frame_idx}, got frame {actual_frame}")
                        # Try to seek again
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        actual_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                        if actual_frame != frame_idx:
                            logger.error(f"Seek failed again for frame {frame_idx}")
                            continue
                    
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning(f"Could not read frame {frame_idx}")
                        continue
                    
                    # Detect faces
                    faces = self._detect_faces(frame)
                    
                    if faces:
                        logger.debug(f"Frame {frame_idx}: Found {len(faces)} faces")
                        # Use the face with highest confidence
                        best_face = max(faces, key=lambda x: x['confidence'])
                        bbox = best_face['bbox']
                        confidence = best_face['confidence']
                        center = best_face['center']
                        
                        logger.debug(f"Frame {frame_idx}: Best face confidence {confidence:.3f}, bbox {bbox}")
                        
                        # Crop face with padding (matching original approach)
                        x, y, w, h = bbox
                        padding = int(min(w, h) * 0.1)  # 10% padding
                        x1 = max(0, x - padding)
                        y1 = max(0, y - padding)
                        x2 = min(frame.shape[1], x + w + padding)
                        y2 = min(frame.shape[0], y + h + padding)
                        
                        face_image = frame[y1:y2, x1:x2]
                        
                        # Extract expression token
                        expression_token = self._extract_expression_token(face_image, subject_id)
                        
                        if expression_token is not None:
                            frame_data = FrameData(
                                frame_idx=frame_idx,
                                bbox=bbox,
                                confidence=confidence,
                                expression_token=expression_token,
                                face_center=center
                            )
                            frame_data_list.append(frame_data)
                            logger.debug(f"Frame {frame_idx}: Successfully extracted expression token")
                        else:
                            logger.warning(f"Failed to extract expression token for frame {frame_idx}")
                            # Add frame data with None bbox to indicate no valid face
                            frame_data = FrameData(
                                frame_idx=frame_idx,
                                bbox=None,
                                confidence=0.0,
                                expression_token=None,
                                face_center=(0, 0)
                            )
                            frame_data_list.append(frame_data)
                    else:
                        logger.debug(f"Frame {frame_idx}: No faces detected")
                        # Add frame data with None bbox to indicate no face detected
                        frame_data = FrameData(
                            frame_idx=frame_idx,
                            bbox=None,
                            confidence=0.0,
                            expression_token=None,
                            face_center=(0, 0)
                        )
                        frame_data_list.append(frame_data)
                        
                except KeyboardInterrupt:
                    logger.info("Analysis interrupted by user")
                    cap.release()
                    return frame_data_list
                except Exception as e:
                    logger.error(f"Error processing frame {frame_idx}: {e}")
                    continue
        
        cap.release()
        
        logger.info(f"✅ Video clip analysis completed: {len(frame_data_list)} frames with valid data")
        
        # Print expression token statistics for debugging
        if frame_data_list:
            self.print_expression_token_stats(frame_data_list)
        
        return frame_data_list
    
    def find_best_sequences(self, 
                           frame_data: List[FrameData], 
                           sequence_length: int = 30,
                           K: int = 10,
                           N: int = 100,
                           expression_weight: float = 0.7,
                           position_weight: float = 0.3) -> List[SequenceScore]:
        """
        Find the best sequences based on expression variation and position stability
        
        Args:
            frame_data: List of FrameData objects
            sequence_length: Length of sequences to extract
            K: Number of top sequences to return
            N: Number of sequences to sample and evaluate
            expression_weight: Weight for expression variation in scoring
            position_weight: Weight for position stability in scoring
            
        Returns:
            List of SequenceScore objects sorted by combined score (top K)
        """
        logger.info(f"🔍 Finding best {K} sequences from {N} samples of length {sequence_length}")
        
        if len(frame_data) < sequence_length:
            logger.warning(f"Not enough frames ({len(frame_data)}) for sequence length {sequence_length}")
            return []
        
        # Calculate total possible sequences
        total_possible_sequences = len(frame_data) - sequence_length + 1
        
        if N > total_possible_sequences:
            logger.warning(f"Requested {N} samples but only {total_possible_sequences} possible sequences available")
            N = total_possible_sequences
        
        # Randomly sample N starting positions
        import random
        random.seed(42)  # For reproducible results
        start_positions = random.sample(range(total_possible_sequences), N)
        
        # Evaluate sampled sequences
        valid_sequences = []
        for i, start_idx in enumerate(start_positions):
            end_idx = start_idx + sequence_length
            sequence_frames = frame_data[start_idx:end_idx]
            
            # Check that every frame in the sequence has at least one face detected
            all_frames_have_faces = True
            for frame in sequence_frames:
                if frame.bbox is None or frame.confidence < self.face_confidence_threshold:
                    all_frames_have_faces = False
                    break
            
            if not all_frames_have_faces:
                logger.debug(f"Skipping sequence {i+1}: frames {start_idx}-{end_idx} - missing faces in some frames")
                continue
            
            # Calculate scores
            expression_variation = self._calculate_expression_variation(sequence_frames)
            position_stability = self._calculate_position_stability(sequence_frames)
            
            # Combined score
            combined_score = (expression_weight * expression_variation + 
                            position_weight * position_stability)
            
            sequence_score = SequenceScore(
                start_frame=sequence_frames[0].frame_idx,
                end_frame=sequence_frames[-1].frame_idx,
                expression_variation=expression_variation,
                position_stability=position_stability,
                combined_score=combined_score,
                frame_data=sequence_frames
            )
            
            valid_sequences.append(sequence_score)
        
        if not valid_sequences:
            logger.warning("No valid sequences found with faces in all frames!")
            return []
        
        # Sort by combined score (highest first)
        valid_sequences.sort(key=lambda x: x.combined_score, reverse=True)
        
        # Return top K sequences
        top_sequences = valid_sequences[:K]
        
        logger.info(f"✅ Found {len(top_sequences)} best sequences from {len(valid_sequences)} valid samples")
        logger.info(f"📊 Score statistics:")
        logger.info(f"   - Best score: {top_sequences[0].combined_score:.8f}")
        logger.info(f"   - Worst score: {top_sequences[-1].combined_score:.8f}")
        logger.info(f"   - Average score: {np.mean([s.combined_score for s in top_sequences]):.8f}")
        logger.info(f"   - Valid sequences: {len(valid_sequences)}/{N} sampled")
        
        for i, seq in enumerate(top_sequences):
            logger.info(f"   Sequence {i+1}: frames {seq.start_frame}-{seq.end_frame}, "
                       f"score={seq.combined_score:.8f}, "
                       f"expr_var={seq.expression_variation:.8f}, "
                       f"pos_stab={seq.position_stability:.8f}")
        
        return top_sequences
    
    def find_best_sequences_from_clips(self, 
                                      frame_data: List[FrameData], 
                                      clip_ranges: List[Tuple[int, int]],
                                      K: int = 10,
                                      expression_weight: float = 1.0,
                                      position_weight: float = 0.0) -> List[SequenceScore]:
        """
        Find the best sequences from pre-analyzed clips
        
        Args:
            frame_data: List of FrameData objects from analyzed clips
            clip_ranges: List of (start_frame, end_frame) tuples that were analyzed
            K: Number of top sequences to return
            expression_weight: Weight for expression variation in scoring
            position_weight: Weight for position stability in scoring
            
        Returns:
            List of SequenceScore objects sorted by combined score (top K)
        """
        logger.info(f"🔍 Finding best {K} sequences from {len(clip_ranges)} analyzed clips")
        logger.info(f"📊 Weights: expression={expression_weight}, position={position_weight}")
        
        if not frame_data:
            logger.warning("No frame data available")
            return []
        
        # Group frame data by clip
        clip_sequences = []
        
        for clip_idx, (start_frame, end_frame) in enumerate(clip_ranges):
            # Find frames that belong to this clip
            clip_frames = []
            for frame in frame_data:
                if start_frame <= frame.frame_idx <= end_frame:
                    clip_frames.append(frame)
            
            if len(clip_frames) < 3:  # Need at least 3 frames for variation calculation
                logger.debug(f"Clip {clip_idx+1}: Insufficient frames ({len(clip_frames)})")
                continue
            
            # Check that every frame in the clip has a valid face
            all_frames_have_faces = True
            for frame in clip_frames:
                if frame.bbox is None or frame.confidence < self.face_confidence_threshold:
                    all_frames_have_faces = False
                    break
            
            if not all_frames_have_faces:
                logger.debug(f"Clip {clip_idx+1}: Missing faces in some frames")
                continue
            
            # Calculate scores
            expression_variation = self._calculate_expression_variation(clip_frames)
            position_stability = self._calculate_position_stability(clip_frames)
            
            # Combined score
            combined_score = (expression_weight * expression_variation + 
                            position_weight * position_stability)
            
            logger.info(f"📈 Clip {clip_idx+1} Scores:")
            logger.info(f"   Expression Variation: {expression_variation:.8f}")
            logger.info(f"   Position Stability:   {position_stability:.8f}")
            logger.info(f"   Combined Score:       {combined_score:.8f}")
            logger.info(f"   Weights: expr={expression_weight}, pos={position_weight}")
            
            sequence_score = SequenceScore(
                start_frame=clip_frames[0].frame_idx,
                end_frame=clip_frames[-1].frame_idx,
                expression_variation=expression_variation,
                position_stability=position_stability,
                combined_score=combined_score,
                frame_data=clip_frames
            )
            
            clip_sequences.append(sequence_score)
        
        if not clip_sequences:
            logger.warning("No valid sequences found from analyzed clips!")
            return []
        
        # Sort by combined score (highest first)
        clip_sequences.sort(key=lambda x: x.combined_score, reverse=True)
        
        # Return top K sequences
        top_sequences = clip_sequences[:K]
        
        logger.info(f"✅ Found {len(top_sequences)} best sequences from {len(clip_sequences)} valid clips")
        logger.info(f"📊 Score statistics:")
        logger.info(f"   - Best score: {top_sequences[0].combined_score:.8f}")
        logger.info(f"   - Worst score: {top_sequences[-1].combined_score:.8f}")
        logger.info(f"   - Average score: {np.mean([s.combined_score for s in top_sequences]):.8f}")
        logger.info(f"   - Valid clips: {len(clip_sequences)}/{len(clip_ranges)} analyzed")
        
        for i, seq in enumerate(top_sequences):
            logger.info(f"   Sequence {i+1}: frames {seq.start_frame}-{seq.end_frame}, "
                       f"score={seq.combined_score:.8f}, "
                       f"expr_var={seq.expression_variation:.8f}, "
                       f"pos_stab={seq.position_stability:.8f}")
        
        return top_sequences
    
    def save_analysis_results(self, 
                             frame_data: List[FrameData], 
                             sequences: List[SequenceScore],
                             output_dir: str):
        """
        Save analysis results to files
        
        Args:
            frame_data: List of FrameData objects
            sequences: List of SequenceScore objects
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save frame data
        frame_data_dict = []
        for frame in frame_data:
            frame_data_dict.append({
                'frame_idx': frame.frame_idx,
                'bbox': frame.bbox,
                'confidence': frame.confidence,
                'face_center': frame.face_center,
                'expression_token': frame.expression_token.cpu().numpy().tolist()
            })
        
        with open(os.path.join(output_dir, 'frame_data.json'), 'w') as f:
            json.dump(frame_data_dict, f, indent=2)
        
        # Save sequence scores
        sequence_data = []
        for seq in sequences:
            sequence_data.append({
                'start_frame': seq.start_frame,
                'end_frame': seq.end_frame,
                'expression_variation': seq.expression_variation,
                'position_stability': seq.position_stability,
                'combined_score': seq.combined_score,
                'frame_indices': [f.frame_idx for f in seq.frame_data]
            })
        
        with open(os.path.join(output_dir, 'sequences.json'), 'w') as f:
            json.dump(sequence_data, f, indent=2)
        
        logger.info(f"✅ Analysis results saved to: {output_dir}")

    def test_face_detection_on_frame(self, video_path: str, frame_idx: int = 0):
        """Test face detection on a specific frame"""
        logger.info(f"Testing face detection on frame {frame_idx} of {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return
        
        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        actual_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        logger.info(f"Seeked to frame {frame_idx}, actual frame: {actual_frame}")
        
        ret, frame = cap.read()
        if not ret:
            logger.error(f"Could not read frame {frame_idx}")
            cap.release()
            return
        
        logger.info(f"Frame shape: {frame.shape}")
        
        # Test face detection
        faces = self._detect_faces(frame)
        logger.info(f"Detected {len(faces)} faces")
        
        for i, face in enumerate(faces):
            logger.info(f"Face {i+1}: confidence={face['confidence']:.3f}, bbox={face['bbox']}")
        
        cap.release()
        return faces

    def print_expression_token_stats(self, frame_data: List[FrameData]):
        """Print statistics about expression tokens for debugging"""
        if not frame_data:
            logger.warning("No frame data to analyze")
            return
        
        valid_tokens = [fd.expression_token for fd in frame_data if fd.expression_token is not None]
        if not valid_tokens:
            logger.warning("No valid expression tokens found")
            return
        
        # Convert to tensor for analysis
        tokens_tensor = torch.stack(valid_tokens).squeeze(1)  # (num_frames, 384)
        
        logger.info(f"📊 Expression Token Statistics:")
        logger.info(f"   Number of valid tokens: {len(valid_tokens)}")
        logger.info(f"   Token shape: {tokens_tensor.shape}")
        logger.info(f"   Mean value: {tokens_tensor.mean().item():.8f}")
        logger.info(f"   Std value:  {tokens_tensor.std().item():.8f}")
        logger.info(f"   Min value:  {tokens_tensor.min().item():.8f}")
        logger.info(f"   Max value:  {tokens_tensor.max().item():.8f}")
        
        # Print first few values of first and last tokens
        if len(valid_tokens) >= 2:
            first_token = valid_tokens[0].flatten()
            last_token = valid_tokens[-1].flatten()
            
            logger.info(f"   First token (first 10 values): {first_token[:10].tolist()}")
            logger.info(f"   Last token (first 10 values):  {last_token[:10].tolist()}")
            
            # Calculate cosine similarity between first and last
            cos_sim = torch.nn.functional.cosine_similarity(
                first_token.unsqueeze(0), last_token.unsqueeze(0), dim=1
            ).item()
            logger.info(f"   First→Last similarity: {cos_sim:.8f}")
        
        return tokens_tensor


def main():
    """Test the video expression analyzer"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze video for expressive face sequences")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--expression_transformer_checkpoint", type=str, required=True, 
                       help="Path to Expression Transformer checkpoint")
    parser.add_argument("--output_dir", type=str, default="./analysis_results", 
                       help="Directory to save analysis results")
    parser.add_argument("--subject_id", type=int, default=0, help="Subject ID for expression extraction")
    parser.add_argument("--sequence_length", type=int, default=30, help="Length of sequences to extract")
    parser.add_argument("--num_sequences", type=int, default=10, help="Number of sequences to extract")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run models on")
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = VideoExpressionAnalyzer(
        expression_transformer_checkpoint_path=args.expression_transformer_checkpoint,
        device=args.device
    )
    
    # Analyze video
    frame_data = analyzer.analyze_video_clips(args.video_path, [(0, 100), (100, 200)])
    
    if frame_data:
        # Find best sequences
        sequences = analyzer.find_best_sequences(
            frame_data, 
            sequence_length=args.sequence_length,
            num_sequences=args.num_sequences
        )
        
        # Save results
        analyzer.save_analysis_results(frame_data, sequences, args.output_dir)
        
        logger.info("🎉 Video analysis completed successfully!")
    else:
        logger.error("❌ No valid frame data found!")


if __name__ == "__main__":
    main() 