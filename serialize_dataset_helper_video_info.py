#!/usr/bin/env python3
"""
Video Information Collection Script
Downloads videos from GCS temporarily to collect duration and other metadata,
then saves the information to a JSON file for use in serialization.
"""

import os
import json
import logging
import argparse
import tempfile
import shutil
from typing import List, Dict, Any, Optional, Tuple
import subprocess
from pathlib import Path
from datetime import datetime

# Import our utilities
import mp4_utils

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VideoInfoCollector:
    """Collect video information from GCS videos"""
    
    def __init__(self, 
                 gcs_bucket: str,
                 gcs_prefix: str = "",
                 temp_dir: str = "/tmp",
                 output_file: str = "CC_video_infos.json"):
        """
        Initialize the video info collector
        
        Args:
            gcs_bucket: GCS bucket name containing videos
            gcs_prefix: Prefix path in GCS bucket where videos are stored
            temp_dir: Directory for temporary downloads
            output_file: Output JSON file path
        """
        self.gcs_bucket = gcs_bucket
        self.gcs_prefix = gcs_prefix.rstrip('/')  # Remove trailing slash if present
        self.temp_dir = temp_dir
        self.output_file = output_file
        
        # Create temp directory
        os.makedirs(temp_dir, exist_ok=True)
        
        logger.info(f"Video Info Collector initialized:")
        logger.info(f"  GCS Bucket: {gcs_bucket}")
        if self.gcs_prefix:
            logger.info(f"  GCS Prefix: {self.gcs_prefix}")
        logger.info(f"  Temp Directory: {temp_dir}")
        logger.info(f"  Output File: {output_file}")
    
    def _download_video_from_gcs(self, gcs_path: str, local_path: str) -> bool:
        """Download a video from GCS to local storage"""
        try:
            # Construct full GCS path with prefix
            if self.gcs_prefix:
                full_gcs_path = f"{self.gcs_prefix}/{gcs_path}"
            else:
                full_gcs_path = gcs_path
            
            gcs_uri = f"gs://{self.gcs_bucket}/{full_gcs_path}"
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
    
    def _load_video_list_from_annotations(self, annotations_path: str) -> List[Dict[str, Any]]:
        """Load video list from CasualConversations.json file"""
        try:
            with open(annotations_path, 'r') as f:
                annotations = json.load(f)
            
            # Handle the original CasualConversations JSON format
            # Format: {subject_id: {files: [...], dark_files: [...], label: {...}}}
            videos_data = []
            
            if isinstance(annotations, dict):
                for subject_id, subject_data in annotations.items():
                    if isinstance(subject_data, dict):
                        # Get files list (excluding dark_files)
                        files = subject_data.get('files', [])
                        dark_files = subject_data.get('dark_files', [])
                        label = subject_data.get('label', {})
                        
                        # Process each file for this subject
                        for video_file in files:
                            # Skip dark files
                            if video_file in dark_files:
                                logger.debug(f"Skipping dark file: {video_file}")
                                continue
                            
                            # Create video data entry
                            video_data = {
                                'video_path': video_file,  # This is the relative path
                                'subject_id': subject_id,
                                'age': label.get('age', 'unknown'),
                                'gender': label.get('gender', 'unknown'),
                                'skin_type': label.get('skin-type', 'unknown'),
                                'is_dark': False
                            }
                            videos_data.append(video_data)
                    else:
                        logger.warning(f"Skipping non-dict subject data for {subject_id}: {type(subject_data)}")
            else:
                raise ValueError(f"Expected dict format, got {type(annotations)}")
            
            logger.info(f"Loaded video list: {len(videos_data)} videos from {len(annotations)} subjects")
            return videos_data
        except Exception as e:
            logger.error(f"Failed to load annotations: {e}")
            raise
    
    def _load_video_list_from_keywords(self, keyword_results_path: str) -> List[Dict[str, Any]]:
        """Load video list from keyword results JSON file"""
        try:
            with open(keyword_results_path, 'r') as f:
                results = json.load(f)
            
            videos = results.get('videos', [])
            logger.info(f"Loaded video list from keyword results: {len(videos)} videos")
            return videos
        except Exception as e:
            logger.error(f"Failed to load keyword results: {e}")
            raise
    
    def collect_video_info(self, video_list: List[Dict[str, Any]], 
                          subject_id_range: Tuple[int, int] = None) -> Dict[str, Any]:
        """
        Collect video information for all videos in the list
        
        Args:
            video_list: List of video data dictionaries
            subject_id_range: Optional tuple of (min_subject_id, max_subject_id) to filter
            
        Returns:
            Dictionary containing video information for all videos
        """
        video_infos = {}
        processed_count = 0
        failed_count = 0
        
        logger.info(f"Starting video info collection for {len(video_list)} videos...")
        
        for i, video_data in enumerate(video_list):
            video_path = video_data['video_path']
            subject_id = video_data.get('subject_id', 'unknown')
            
            # Filter by subject_id_range if provided
            if subject_id_range:
                try:
                    subject_id_int = int(subject_id)
                    if not (subject_id_range[0] <= subject_id_int <= subject_id_range[1]):
                        logger.debug(f"Skipping video {video_path} with subject_id {subject_id} outside range {subject_id_range}")
                        continue
                except (ValueError, TypeError):
                    logger.warning(f"Could not parse subject_id {subject_id} as integer")
                    continue
            
            logger.info(f"Processing video {i+1}/{len(video_list)}: {video_path}")
            
            try:
                # Download video temporarily
                video_name = os.path.basename(video_path)
                temp_video_path = os.path.join(self.temp_dir, f"temp_{video_name}")
                
                if self._download_video_from_gcs(video_path, temp_video_path):
                    # Get video info using mp4_utils
                    video_info = mp4_utils.get_video_info(temp_video_path)
                    
                    # Add metadata from video_data
                    video_info.update({
                        'gcs_path': video_path,
                        'subject_id': subject_id,
                        'age': video_data.get('age', 'unknown'),
                        'gender': video_data.get('gender', 'unknown'),
                        'skin_type': video_data.get('skin_type', 'unknown'),
                        'collection_timestamp': datetime.now().isoformat()
                    })
                    
                    video_infos[video_path] = video_info
                    processed_count += 1
                    
                    logger.info(f"✅ Collected info for {video_path}: {video_info['duration_seconds']}s, {video_info['resolution']}")
                    
                    # Clean up temp file
                    try:
                        os.remove(temp_video_path)
                    except:
                        pass
                else:
                    logger.error(f"❌ Failed to download {video_path}")
                    failed_count += 1
                    
            except Exception as e:
                logger.error(f"❌ Error processing {video_path}: {e}")
                failed_count += 1
        
        # Create final output structure
        output_data = {
            'collection_info': {
                'date': datetime.now().isoformat(),
                'gcs_bucket': self.gcs_bucket,
                'gcs_prefix': self.gcs_prefix,
                'total_videos': len(video_list),
                'processed_videos': processed_count,
                'failed_videos': failed_count,
                'success_rate': processed_count / len(video_list) if video_list else 0
            },
            'video_infos': video_infos
        }
        
        logger.info(f"🎉 Video info collection completed!")
        logger.info(f"📊 Results: {processed_count} processed, {failed_count} failed")
        logger.info(f"📈 Success rate: {output_data['collection_info']['success_rate']*100:.1f}%")
        
        return output_data
    
    def save_video_infos(self, video_infos: Dict[str, Any]):
        """Save video information to JSON file"""
        try:
            with open(self.output_file, 'w') as f:
                json.dump(video_infos, f, indent=2)
            logger.info(f"💾 Video information saved to: {self.output_file}")
        except Exception as e:
            logger.error(f"Failed to save video information: {e}")
            raise


def main():
    """Main function for video info collection"""
    parser = argparse.ArgumentParser(description='Collect video information from GCS')
    parser.add_argument('--gcs-bucket', default='face-training-datasets', help='GCS bucket name containing videos')
    parser.add_argument('--gcs-prefix', default='face_training_datasets/casual_conversations_full/', help='Prefix path in GCS bucket where videos are stored')
    parser.add_argument('--temp-dir', default='/tmp', help='Directory for temporary downloads')
    parser.add_argument('--output-file', default='CC_video_infos.json', help='Output JSON file path')
    
    # Input source arguments
    parser.add_argument('--annotations-path', 
                       default='/mnt/dataset-storage/dbs/CC_annotations/CasualConversations.json', 
                       help='Path to CasualConversations.json (random mode)')
    parser.add_argument('--keyword-results', help='Path to out.json with keyword timestamps (keyword mode)')
    
    # Filtering arguments
    parser.add_argument('--subject-id-min', type=int, help='Minimum subject ID to process')
    parser.add_argument('--subject-id-max', type=int, help='Maximum subject ID to process')
    
    args = parser.parse_args()
    
    # Validate input source
    if args.keyword_results and args.annotations_path:
        logger.error("Cannot specify both --keyword-results and --annotations-path")
        return False
    elif not args.keyword_results and not args.annotations_path:
        logger.error("Must specify either --keyword-results or --annotations-path")
        return False
    
    # Validate file existence
    if args.keyword_results and not os.path.exists(args.keyword_results):
        logger.error(f"Keyword results file not found: {args.keyword_results}")
        return False
    
    if args.annotations_path and not os.path.exists(args.annotations_path):
        logger.error(f"Annotations file not found: {args.annotations_path}")
        return False
    
    # Create collector
    collector = VideoInfoCollector(
        gcs_bucket=args.gcs_bucket,
        gcs_prefix=args.gcs_prefix,
        temp_dir=args.temp_dir,
        output_file=args.output_file
    )
    
    # Load video list based on input source
    if args.keyword_results:
        logger.info("Using keyword results mode")
        video_list = collector._load_video_list_from_keywords(args.keyword_results)
    else:
        logger.info("Using annotations mode")
        video_list = collector._load_video_list_from_annotations(args.annotations_path)
    
    # Set subject ID range
    subject_id_range = None
    if args.subject_id_min is not None and args.subject_id_max is not None:
        subject_id_range = (args.subject_id_min, args.subject_id_max)
        logger.info(f"Filtering by subject ID range: {subject_id_range}")
    
    # Collect video information
    try:
        video_infos = collector.collect_video_info(video_list, subject_id_range)
        collector.save_video_infos(video_infos)
        logger.info("Video info collection completed successfully!")
        return True
    except Exception as e:
        logger.error(f"Video info collection failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 