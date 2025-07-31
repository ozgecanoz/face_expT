#!/usr/bin/env python3
"""
Cloud-based Dataset Serialization Script
Downloads videos from Google Cloud Storage and processes them using keyword timestamps
Uses multithreaded processing for efficiency on cloud VMs
"""

import os
import json
import logging
import argparse
import threading
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple
import time
import random
from pathlib import Path
import subprocess
import h5py
from datetime import datetime

# Import our utilities
import mp4_utils

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CloudDatasetSerializer:
    """Cloud-based dataset serialization with multithreaded processing"""
    
    def __init__(self, 
                 gcs_bucket: str,
                 output_base: str = "serialized_dataset_cloud",
                 num_threads: int = 8,
                 batch_size: int = 5,
                 temp_dir: str = "/tmp",
                 device: str = "cpu",
                 keyword_results_path: str = None,
                 annotations_path: str = None,
                 clips_per_video: int = 5,
                 subject_id_range: Tuple[int, int] = None):
        """
        Initialize the cloud serializer
        
        Args:
            gcs_bucket: GCS bucket name containing videos
            output_base: Base directory for output files
            num_threads: Number of worker threads
            batch_size: Number of videos per thread batch
            temp_dir: Directory for temporary downloads
            device: Device for processing (cpu/cuda)
            keyword_results_path: Path to out.json with keyword timestamps (keyword mode)
            annotations_path: Path to CasualConversations.json (random mode)
            clips_per_video: Number of random clips per video (random mode)
            subject_id_range: Tuple of (min_subject_id, max_subject_id) to filter subjects
        """
        self.gcs_bucket = gcs_bucket
        self.output_base = output_base
        self.num_threads = num_threads
        self.batch_size = batch_size
        self.temp_dir = temp_dir
        self.device = device
        self.keyword_results_path = keyword_results_path
        self.annotations_path = annotations_path
        self.clips_per_video = clips_per_video
        self.subject_id_range = subject_id_range
        
        # Determine operation mode
        if keyword_results_path and annotations_path:
            raise ValueError("Cannot specify both keyword_results_path and annotations_path")
        elif keyword_results_path:
            self.mode = "keyword"
            self.videos_data = self._load_keyword_results()
        elif annotations_path:
            self.mode = "random"
            self.videos_data = self._load_annotations()
        else:
            raise ValueError("Must specify either keyword_results_path or annotations_path")
        
        # Thread-safe counters
        self.processed_videos = 0
        self.successful_clips = 0
        self.failed_videos = 0
        self.skipped_clips = 0  # Track clips that don't have 30 frames
        self.lock = threading.Lock()
        
        # Create output directory
        os.makedirs(output_base, exist_ok=True)
        
        logger.info(f"Cloud Dataset Serializer initialized:")
        logger.info(f"  Mode: {self.mode}")
        logger.info(f"  GCS Bucket: {gcs_bucket}")
        logger.info(f"  Threads: {num_threads}")
        logger.info(f"  Batch Size: {batch_size}")
        logger.info(f"  Output: {output_base}")
        if self.subject_id_range:
            logger.info(f"  Subject ID Range: {self.subject_id_range[0]} to {self.subject_id_range[1]}")
        logger.info(f"  Videos to process: {len(self.videos_data)}")
        if self.mode == "random":
            logger.info(f"  Clips per video: {clips_per_video}")
    
    def _load_keyword_results(self) -> Dict[str, Any]:
        """Load keyword results from JSON file"""
        try:
            with open(self.keyword_results_path, 'r') as f:
                results = json.load(f)
            logger.info(f"Loaded keyword results: {len(results.get('videos', []))} videos")
            return results
        except Exception as e:
            logger.error(f"Failed to load keyword results: {e}")
            raise
    
    def _load_annotations(self) -> List[Dict[str, Any]]:
        """Load annotations from CasualConversations.json file"""
        try:
            with open(self.annotations_path, 'r') as f:
                annotations = json.load(f)
            
            # Extract video data from annotations
            videos_data = []
            for video_info in annotations:
                video_data = {
                    'video_path': video_info.get('video_path', ''),
                    'subject_id': video_info.get('subject_id', 'unknown'),
                    'age': video_info.get('age', 'unknown'),
                    'gender': video_info.get('gender', 'unknown'),
                    'skin_type': video_info.get('skin-type', 'unknown'),
                    'is_dark': video_info.get('is_dark', False)
                }
                videos_data.append(video_data)
            
            logger.info(f"Loaded annotations: {len(videos_data)} videos")
            return videos_data
        except Exception as e:
            logger.error(f"Failed to load annotations: {e}")
            raise
    
    def _download_video_from_gcs(self, gcs_path: str, local_path: str) -> bool:
        """Download a video from GCS to local storage"""
        try:
            gcs_uri = f"gs://{self.gcs_bucket}/{gcs_path}"
            logger.debug(f"Downloading {gcs_uri} -> {local_path}")
            
            # Use gsutil to download
            result = subprocess.run([
                'gsutil', 'cp', gcs_uri, local_path
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.debug(f"Successfully downloaded {gcs_path}")
                return True
            else:
                logger.error(f"Failed to download {gcs_path}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error downloading {gcs_path}: {e}")
            return False
    
    def _extract_timestamps_from_keywords(self, video_data: Dict[str, Any], keywords: List[str]) -> List[float]:
        """Extract timestamps from video's keyword data"""
        timestamps = []
        keywords_data = video_data.get('keywords', {})
        
        for keyword in keywords:
            if keyword in keywords_data:
                keyword_timestamps = keywords_data[keyword]
                if isinstance(keyword_timestamps, list):
                    timestamps.extend(keyword_timestamps)
                else:
                    # Handle single timestamp
                    timestamps.append(keyword_timestamps)
        
        # Remove duplicates and sort
        timestamps = sorted(list(set(timestamps)))
        return timestamps
    
    def _generate_random_timestamps(self, video_path: str, num_clips: int = 5, min_duration: float = 5.0) -> List[float]:
        """Generate random timestamps for video clips"""
        try:
            # Get video info to determine duration
            video_info = mp4_utils.get_video_info(video_path)
            duration = video_info['duration_seconds']
            
            if duration < min_duration:
                logger.warning(f"Video {video_path} duration ({duration}s) is shorter than minimum ({min_duration}s)")
                return []
            
            # Generate random timestamps
            timestamps = []
            max_start_time = duration - min_duration
            
            for _ in range(num_clips):
                # Generate random start time
                start_time = random.uniform(0, max_start_time)
                timestamps.append(start_time)
            
            # Sort timestamps
            timestamps.sort()
            return timestamps
            
        except Exception as e:
            logger.error(f"Error generating random timestamps for {video_path}: {e}")
            return []
    
    def _process_video_batch(self, batch_tasks: List[Dict[str, Any]], 
                           thread_id: int) -> Dict[str, Any]:
        """Process a batch of video tasks in a single thread"""
        batch_results = {
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'clips_created': 0,
            'errors': [],
            'skipped_clips': 0 # Track skipped clips in a batch
        }
        
        # Create thread-specific temp directory
        thread_temp_dir = os.path.join(self.temp_dir, f"thread_{thread_id}")
        os.makedirs(thread_temp_dir, exist_ok=True)
        
        for task in batch_tasks:
            try:
                video_path = task['video_path']
                subject_id = task['subject_id']
                subject_label = task['subject_label']
                mode = task['mode']
                
                # Download video from GCS first
                video_name = os.path.basename(video_path)
                local_video_path = os.path.join(thread_temp_dir, video_name)
                
                if not self._download_video_from_gcs(video_path, local_video_path):
                    logger.error(f"Thread {thread_id}: Failed to download {video_path}")
                    batch_results['failed'] += 1
                    continue
                
                # Get timestamps based on mode
                if mode == "keyword":
                    # Use pre-planned timestamps
                    timestamps = task['timestamps']
                else:  # random mode
                    # Generate random timestamps during processing
                    timestamps = self._generate_random_timestamps(
                        local_video_path, 
                        num_clips=task['clips_per_video'], 
                        min_duration=5.0
                    )
                    if not timestamps:
                        logger.warning(f"Thread {thread_id}: Could not generate random timestamps for {video_path}")
                        batch_results['failed'] += 1
                        continue
                
                # Process each timestamp
                for timestamp in timestamps:
                    try:
                        # Use mp4_utils.extract_face_sequence
                        result = mp4_utils.extract_face_sequence(
                            video_path=local_video_path,
                            timestamp=timestamp,
                            output_folder=self.output_base,
                            face_size=(518, 518),
                            num_frames=30,  # 30 frames for better sequences
                            frame_skip=0,   # No skipping for better quality
                            confidence_threshold=0.5,
                            subject_id=str(subject_id),
                            subject_label=subject_label
                        )
                        
                        if result is not None:
                            # Validate that the clip has exactly 30 frames
                            num_frames = result.get('num_frames', 0)
                            frames_with_single_face = result.get('frames_with_single_face', 0)
                            
                            if num_frames == 30 and frames_with_single_face == 30:
                                batch_results['clips_created'] += 1
                                logger.info(f"Thread {thread_id}: Created clip for {video_name} at {timestamp}s ({num_frames} frames)")
                            else:
                                logger.warning(f"Thread {thread_id}: Skipping clip for {video_name} at {timestamp}s - "
                                             f"expected 30 frames, got {num_frames} frames with {frames_with_single_face} single faces")
                                batch_results['skipped_clips'] += 1 # Increment skipped clips in batch
                        else:
                            logger.warning(f"Thread {thread_id}: No clip created for {video_name} at {timestamp}s")
                    
                    except Exception as e:
                        logger.error(f"Thread {thread_id}: Error processing {video_name} at {timestamp}s: {e}")
                        batch_results['errors'].append(f"{video_name}@{timestamp}s: {str(e)}")
                
                batch_results['successful'] += 1
                
                # Clean up downloaded video
                try:
                    os.remove(local_video_path)
                except:
                    pass
                
            except Exception as e:
                logger.error(f"Thread {thread_id}: Error processing video {task.get('video_path', 'unknown')}: {e}")
                batch_results['failed'] += 1
                batch_results['errors'].append(f"{task.get('video_path', 'unknown')}: {str(e)}")
        
        # Clean up thread temp directory
        try:
            shutil.rmtree(thread_temp_dir)
        except:
            pass
        
        return batch_results
    
    def _update_global_stats(self, batch_results: Dict[str, Any]):
        """Update global statistics thread-safely"""
        with self.lock:
            self.processed_videos += batch_results['processed']
            self.successful_clips += batch_results['clips_created']
            self.failed_videos += batch_results['failed']
            self.skipped_clips += batch_results.get('skipped_clips', 0)
    
    def _convert_subject_id_to_int(self, subject_id) -> int:
        """Convert subject ID to integer, handling string and numeric types"""
        if isinstance(subject_id, str):
            try:
                return int(subject_id)
            except ValueError:
                # If conversion fails, use hash of string as subject ID
                return hash(subject_id) % 3500  # Ensure it's within max_subjects range
        else:
            return int(subject_id)
    
    def _plan_processing(self, keywords: List[str] = None) -> List[Dict[str, Any]]:
        """
        Single-threaded planning phase to determine all clips to be generated
        
        Returns:
            List of video processing tasks with timestamps
        """
        logger.info("🔍 Starting planning phase...")
        
        processing_tasks = []
        filtered_videos = 0
        total_videos = 0
        
        # Get videos list based on mode
        if self.mode == "keyword":
            videos = self.videos_data.get('videos', [])
        else:  # random mode
            videos = self.videos_data
        
        for video_data in videos:
            total_videos += 1
            video_path = video_data['video_path']
            subject_id = video_data['subject_id']
            subject_label = {
                'age': video_data.get('age', 'unknown'),
                'gender': video_data.get('gender', 'unknown'),
                'skin-type': video_data.get('skin_type', 'unknown')
            }
            
            # Filter by subject_id_range if provided
            if self.subject_id_range:
                subject_id_int = self._convert_subject_id_to_int(subject_id)
                if not (self.subject_id_range[0] <= subject_id_int <= self.subject_id_range[1]):
                    logger.debug(f"Skipping video {video_path} with subject_id {subject_id} outside range {self.subject_id_range}")
                    continue
                filtered_videos += 1
            
            # Get timestamps based on mode
            if self.mode == "keyword":
                # Extract timestamps from keywords
                timestamps = self._extract_timestamps_from_keywords(video_data, keywords)
                if not timestamps:
                    logger.warning(f"No keyword timestamps found for {video_path}")
                    continue
            else:  # random mode
                # For planning, we'll estimate timestamps based on video duration
                # Actual timestamps will be generated during processing
                timestamps = [0.0] * self.clips_per_video  # Placeholder
            
            # Create processing task
            task = {
                'video_path': video_path,
                'subject_id': subject_id,
                'subject_label': subject_label,
                'timestamps': timestamps,
                'mode': self.mode,
                'clips_per_video': self.clips_per_video if self.mode == "random" else len(timestamps)
            }
            
            processing_tasks.append(task)
        
        logger.info(f"📋 Planning complete: {len(processing_tasks)} videos to process")
        if self.subject_id_range:
            logger.info(f"📊 Subject filtering: {filtered_videos} videos within range {self.subject_id_range} out of {total_videos} total")
        total_clips = sum(task['clips_per_video'] for task in processing_tasks)
        logger.info(f"📊 Total clips to generate: {total_clips}")
        
        return processing_tasks
    
    def _create_video_batches(self, processing_tasks: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Create batches of unique videos for multithreaded processing
        
        Args:
            processing_tasks: List of video processing tasks
            
        Returns:
            List of batches, each containing unique videos
        """
        logger.info(f"📦 Creating batches of {self.batch_size} videos for {self.num_threads} threads...")
        
        batches = []
        for i in range(0, len(processing_tasks), self.batch_size):
            batch = processing_tasks[i:i + self.batch_size]
            batches.append(batch)
        
        logger.info(f"✅ Created {len(batches)} batches")
        for i, batch in enumerate(batches):
            logger.info(f"  Batch {i}: {len(batch)} videos")
        
        return batches
    
    def serialize_dataset(self, keywords: List[str] = None) -> Dict[str, Any]:
        """
        Main serialization function with planning phase
        
        Args:
            keywords: List of keywords to process (keyword mode only)
            
        Returns:
            Dictionary with processing results
        """
        if self.mode == "keyword":
            if keywords is None:
                keywords = ['smile', 'natural', 'neutral', 'sad', 'surprised', 'expressions', 'happy', 'angry']
            logger.info(f"Starting keyword-based serialization")
            logger.info(f"Keywords: {keywords}")
        else:  # random mode
            keywords = None  # Not used in random mode
            logger.info(f"Starting random clip serialization")
            logger.info(f"Clips per video: {self.clips_per_video}")
        
        # Phase 1: Planning (single-threaded)
        logger.info("🚀 Phase 1: Planning processing tasks...")
        processing_tasks = self._plan_processing(keywords)
        
        if not processing_tasks:
            logger.error("No processing tasks generated")
            return {'error': 'No videos to process'}
        
        # Phase 2: Create batches (single-threaded)
        logger.info("📦 Phase 2: Creating video batches...")
        batches = self._create_video_batches(processing_tasks)
        
        # Phase 3: Multithreaded processing
        logger.info("⚡ Phase 3: Starting multithreaded processing...")
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(self._process_video_batch, batch, i): i
                for i, batch in enumerate(batches)
            }
            
            # Collect results as they complete
            completed_batches = 0
            for future in as_completed(future_to_batch):
                batch_id = future_to_batch[future]
                try:
                    batch_results = future.result()
                    self._update_global_stats(batch_results)
                    completed_batches += 1
                    
                    logger.info(f"Batch {batch_id} completed: "
                              f"{batch_results['successful']} successful, "
                              f"{batch_results['failed']} failed, "
                              f"{batch_results['clips_created']} clips created")
                    
                except Exception as e:
                    logger.error(f"Batch {batch_id} failed: {e}")
                    self.failed_videos += len(batches[batch_id])
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Create metadata file
        metadata = self._create_metadata(keywords, processing_time)
        
        logger.info(f"🎉 Cloud serialization completed in {processing_time:.2f} seconds")
        logger.info(f"📊 Results: {self.successful_clips} clips created, {self.failed_videos} videos failed, {self.skipped_clips} clips skipped (insufficient frames)")
        
        return metadata
    
    def _create_metadata(self, keywords: List[str], processing_time: float) -> Dict[str, Any]:
        """Create metadata file for the serialized dataset"""
        metadata = {
            'serialization_info': {
                'date': datetime.now().isoformat(),
                'processing_time_seconds': processing_time,
                'gcs_bucket': self.gcs_bucket,
                'mode': self.mode,
                'num_threads': self.num_threads,
                'batch_size': self.batch_size
            },
            'processing_stats': {
                'total_videos_processed': self.processed_videos,
                'successful_clips_created': self.successful_clips,
                'failed_videos': self.failed_videos,
                'skipped_clips': self.skipped_clips, # Add skipped clips to metadata
                'success_rate': self.successful_clips / max(self.processed_videos, 1)
            },
            'output_info': {
                'output_directory': self.output_base,
                'file_formats': ['h5', 'mp4'],
                'face_size': [518, 518],
                'frames_per_clip': 30,
                'frame_skip': 0,
                'confidence_threshold': 0.5
            }
        }
        
        # Add mode-specific information
        if self.mode == "keyword":
            metadata['serialization_info']['keyword_results_file'] = self.keyword_results_path
            metadata['serialization_info']['keywords_used'] = keywords
        else:  # random mode
            metadata['serialization_info']['annotations_file'] = self.annotations_path
            metadata['serialization_info']['clips_per_video'] = self.clips_per_video
        
        # Add subject ID range information if provided
        if self.subject_id_range:
            metadata['serialization_info']['subject_id_range'] = {
                'min': self.subject_id_range[0],
                'max': self.subject_id_range[1]
            }
        
        # Save metadata
        metadata_path = os.path.join(self.output_base, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to: {metadata_path}")
        return metadata


def main():
    """Main function for cloud dataset serialization"""
    parser = argparse.ArgumentParser(description='Cloud-based dataset serialization from GCS')
    parser.add_argument('--gcs-bucket', default='face-training-datasets', help='GCS bucket name containing videos')
    parser.add_argument('--output-base', 
    default='/mnt/dataset-storage/dbs/CCA_train_db3/', help='Base directory for output')
    parser.add_argument('--num-threads', type=int, default=8, help='Number of worker threads')
    parser.add_argument('--batch-size', type=int, default=5, help='Videos per thread batch')
    parser.add_argument('--temp-dir', default='/mnt/dataset-storage/tmp/', help='Directory for temporary downloads')
    parser.add_argument('--device', default='cpu', help='Processing device (cpu/cuda)')
    
    parser.add_argument('--subject-id-min', type=int, default=1, help='Minimum subject ID to process (random mode)')
    parser.add_argument('--subject-id-max', type=int, default=5, help='Maximum subject ID to process (random mode)')
    
    # Mode-specific arguments
    parser.add_argument('--keyword-results', help='Path to out.json with keyword timestamps (keyword mode)')
    parser.add_argument('--keywords', nargs='+', default=None, help='Keywords to process (keyword mode, default: all from JSON)')
    
    parser.add_argument('--annotations-path', 
    default='/mnt/dataset-storage/dbs/CC_annotations/CasualConversations.json', help='Path to CasualConversations.json (random mode)')
    parser.add_argument('--clips-per-video', type=int, default=2, help='Number of random clips per video (random mode)')
    
    args = parser.parse_args()
    
    # Validate mode-specific inputs
    if args.keyword_results and args.annotations_path:
        logger.error("Cannot specify both --keyword-results and --annotations-path")
        return False
    elif not args.keyword_results and not args.annotations_path:
        logger.error("Must specify either --keyword-results (keyword mode) or --annotations-path (random mode)")
        return False
    
    # Validate file existence
    if args.keyword_results and not os.path.exists(args.keyword_results):
        logger.error(f"Keyword results file not found: {args.keyword_results}")
        return False
    
    if args.annotations_path and not os.path.exists(args.annotations_path):
        logger.error(f"Annotations file not found: {args.annotations_path}")
        return False
    
    # Create serializer
    subject_id_range = None
    if args.subject_id_min is not None and args.subject_id_max is not None:
        subject_id_range = (args.subject_id_min, args.subject_id_max)
    
    serializer = CloudDatasetSerializer(
        gcs_bucket=args.gcs_bucket,
        output_base=args.output_base,
        num_threads=args.num_threads,
        batch_size=args.batch_size,
        temp_dir=args.temp_dir,
        device=args.device,
        keyword_results_path=args.keyword_results,
        annotations_path=args.annotations_path,
        clips_per_video=args.clips_per_video,
        subject_id_range=subject_id_range
    )
    
    # Run serialization
    try:
        metadata = serializer.serialize_dataset(args.keywords)
        logger.info("Cloud serialization completed successfully!")
        return True
    except Exception as e:
        logger.error(f"Cloud serialization failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 