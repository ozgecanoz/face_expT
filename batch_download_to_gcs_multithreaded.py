#!/usr/bin/env python3
"""
Multithreaded batch download datasets from multiple URLs and upload to Google Cloud Storage
Designed to run on Google Cloud CPU VMs with parallel processing
"""

import os
import json
import logging
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import time
import requests
from tqdm import tqdm
from download_to_gcs import process_dataset_url, load_config, authenticate_gcloud, set_project, check_gcloud_installation

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Thread-safe counter for progress tracking
class ProgressTracker:
    def __init__(self, total):
        self.total = total
        self.successful = 0
        self.failed = 0
        self.lock = threading.Lock()
        self.thread_status = {}  # Track status of each thread
        self.thread_progress = {}  # Track progress of each thread
    
    def increment_success(self):
        with self.lock:
            self.successful += 1
    
    def increment_failed(self):
        with self.lock:
            self.failed += 1
    
    def get_progress(self):
        with self.lock:
            return self.successful, self.failed, self.successful + self.failed
    
    def update_thread_status(self, thread_id, filename, status, progress=None):
        """Update status for a specific thread"""
        with self.lock:
            self.thread_status[thread_id] = {
                'filename': filename,
                'status': status,
                'progress': progress,
                'timestamp': time.time()
            }
    
    def get_thread_statuses(self):
        """Get current status of all threads"""
        with self.lock:
            return self.thread_status.copy()

def download_with_progress(url, local_path, thread_id, tracker, filename):
    """Download file with progress tracking"""
    try:
        # Update status to downloading
        tracker.update_thread_status(thread_id, filename, "DOWNLOADING", 0)
        
        # Get file size
        response = requests.head(url, allow_redirects=True)
        total_size = int(response.headers.get('content-length', 0))
        
        # Download with progress
        response = requests.get(url, stream=True, allow_redirects=True)
        response.raise_for_status()
        
        downloaded_size = 0
        last_progress_update = 0
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    
                    # Calculate progress percentage
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        
                        # Update progress at 30%, 60%, 90% intervals
                        if progress >= 30 and last_progress_update < 30:
                            logger.info(f"🧵 Thread {thread_id} - {filename}: {progress:.1f}% downloaded")
                            last_progress_update = 30
                        elif progress >= 60 and last_progress_update < 60:
                            logger.info(f"🧵 Thread {thread_id} - {filename}: {progress:.1f}% downloaded")
                            last_progress_update = 60
                        elif progress >= 90 and last_progress_update < 90:
                            logger.info(f"🧵 Thread {thread_id} - {filename}: {progress:.1f}% downloaded")
                            last_progress_update = 90
                        
                        tracker.update_thread_status(thread_id, filename, "DOWNLOADING", progress)
        
        # Update status to completed download
        tracker.update_thread_status(thread_id, filename, "DOWNLOAD_COMPLETE", 100)
        logger.info(f"🧵 Thread {thread_id} - {filename}: Download completed (100%)")
        
        return True
        
    except Exception as e:
        tracker.update_thread_status(thread_id, filename, f"DOWNLOAD_ERROR: {str(e)}", 0)
        logger.error(f"🧵 Thread {thread_id} - {filename}: Download failed - {e}")
        return False

def upload_with_progress(local_path, bucket_name, remote_path, thread_id, tracker, filename):
    """Upload file with progress tracking"""
    try:
        # Update status to uploading
        tracker.update_thread_status(thread_id, filename, "UPLOADING", 0)
        logger.info(f"🧵 Thread {thread_id} - {filename}: Starting upload to gs://{bucket_name}/{remote_path}")
        
        # Get file size for progress calculation
        file_size = os.path.getsize(local_path)
        
        # Use gsutil with progress tracking
        import subprocess
        
        # Create gsutil command
        cmd = [
            'gsutil', '-q', 'cp', 
            local_path, 
            f'gs://{bucket_name}/{remote_path}'
        ]
        
        # Execute upload
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            tracker.update_thread_status(thread_id, filename, "UPLOAD_COMPLETE", 100)
            logger.info(f"🧵 Thread {thread_id} - {filename}: Upload completed (100%)")
            return True
        else:
            tracker.update_thread_status(thread_id, filename, f"UPLOAD_ERROR: {result.stderr}", 0)
            logger.error(f"🧵 Thread {thread_id} - {filename}: Upload failed - {result.stderr}")
            return False
            
    except Exception as e:
        tracker.update_thread_status(thread_id, filename, f"UPLOAD_ERROR: {str(e)}", 0)
        logger.error(f"🧵 Thread {thread_id} - {filename}: Upload failed - {e}")
        return False

def process_single_dataset(dataset, bucket_name, key_file, project_id, tracker, dataset_index, total_datasets):
    """Process a single dataset (thread worker function)"""
    thread_id = threading.current_thread().ident
    filename = dataset['name']
    
    try:
        logger.info(f"🧵 Thread {thread_id} - {filename}: Starting processing ({dataset_index}/{total_datasets})")
        
        start_time = time.time()
        
        # Extract filename from URL for local storage
        url = dataset['url']
        local_filename = url.split('/')[-1]
        local_path = f"/tmp/{local_filename}"
        
        # Download with progress tracking
        tracker.update_thread_status(thread_id, filename, "INITIALIZING", 0)
        
        download_success = download_with_progress(url, local_path, thread_id, tracker, filename)
        
        if not download_success:
            tracker.increment_failed()
            return False
        
        # Upload with progress tracking
        upload_success = upload_with_progress(local_path, bucket_name, dataset['remote_path'], thread_id, tracker, filename)
        
        # Clean up local file
        try:
            os.remove(local_path)
        except:
            pass
        
        elapsed_time = time.time() - start_time
        
        if upload_success:
            tracker.increment_success()
            tracker.update_thread_status(thread_id, filename, "COMPLETED", 100)
            logger.info(f"🧵 Thread {thread_id} - {filename}: Successfully completed in {elapsed_time:.1f}s")
            return True
        else:
            tracker.increment_failed()
            tracker.update_thread_status(thread_id, filename, "FAILED", 0)
            logger.error(f"🧵 Thread {thread_id} - {filename}: Failed after {elapsed_time:.1f}s")
            return False
            
    except Exception as e:
        tracker.increment_failed()
        tracker.update_thread_status(thread_id, filename, f"ERROR: {str(e)}", 0)
        logger.error(f"🧵 Thread {thread_id} - {filename}: Unexpected error - {e}")
        return False

def progress_monitor(tracker, total_datasets, stop_event):
    """Monitor and report progress with detailed thread status"""
    while not stop_event.is_set():
        successful, failed, completed = tracker.get_progress()
        thread_statuses = tracker.get_thread_statuses()
        
        if completed > 0:
            logger.info(f"📊 Overall Progress: {completed}/{total_datasets} completed ({successful} successful, {failed} failed)")
            
            # Show active thread statuses
            active_threads = {tid: status for tid, status in thread_statuses.items() 
                            if status['status'] not in ['COMPLETED', 'FAILED', 'ERROR']}
            
            if active_threads:
                logger.info("🔄 Active Threads:")
                for thread_id, status in active_threads.items():
                    progress_str = f"{status['progress']:.1f}%" if status['progress'] is not None else "N/A"
                    logger.info(f"   Thread {thread_id}: {status['filename']} - {status['status']} ({progress_str})")
        
        if completed >= total_datasets:
            break
        
        time.sleep(30)  # Report every 30 seconds

def load_dataset_config(dataset_config_file):
    """Load dataset configuration from file"""
    try:
        with open(dataset_config_file, 'r') as f:
            datasets = json.load(f)
        logger.info(f"✅ Loaded dataset config from: {dataset_config_file}")
        return datasets
    except FileNotFoundError:
        logger.error(f"❌ Dataset config file not found: {dataset_config_file}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid dataset config file: {e}")
        return None

def create_sample_dataset_config():
    """Create a sample dataset configuration file"""
    sample_config = {
        "datasets": [
            {
                "name": "example_dataset_1",
                "url": "https://example.com/dataset1.zip",
                "remote_path": "datasets/dataset1"
            },
            {
                "name": "example_dataset_2", 
                "url": "https://example.com/dataset2.zip",
                "remote_path": "datasets/dataset2"
            }
        ]
    }
    
    config_file = "dataset_urls.json"
    try:
        with open(config_file, 'w') as f:
            json.dump(sample_config, f, indent=2)
        logger.info(f"✅ Created sample config: {config_file}")
        return config_file
    except Exception as e:
        logger.error(f"❌ Failed to create sample config: {e}")
        return None

def main():
    """Main multithreaded batch download function"""
    parser = argparse.ArgumentParser(description='Multithreaded batch download datasets from URLs and upload to Google Cloud Storage')
    parser.add_argument('--dataset-config', default='CC_dataset_urls.json', help='Dataset configuration file')
    parser.add_argument('--config', default='gcp_config.json', help='GCP configuration file')
    parser.add_argument('--bucket-name', help='GCS bucket name (overrides config)')
    parser.add_argument('--create-sample', action='store_true', help='Create a sample dataset configuration file')
    parser.add_argument('--max-workers', type=int, default=4, help='Maximum number of worker threads (default: 4)')
    
    args = parser.parse_args()
    
    # Create sample config if requested
    if args.create_sample:
        create_sample_dataset_config()
        return True
    
    logger.info("🚀 Starting multithreaded batch download and upload to Google Cloud Storage")
    logger.info(f"🔧 Using {args.max_workers} worker threads")
    
    # Check prerequisites
    if not check_gcloud_installation():
        return False
    
    # Load GCP configuration
    gcp_config = load_config(args.config)
    if not gcp_config:
        logger.error("Please create gcp_config.json first")
        return False
    
    # Load dataset configuration
    datasets = load_dataset_config(args.dataset_config)
    if not datasets:
        logger.error(f"Please create {args.dataset_config} with your dataset URLs")
        logger.info("Use --create-sample to generate a sample configuration file")
        return False
    
    # Use command line args or config
    bucket_name = args.bucket_name or gcp_config.get('bucket_name', 'face-training-datasets')
    project_id = gcp_config.get('project_id')
    
    if not project_id:
        logger.error("No project ID found in config")
        return False
    
    try:
        # Load config to get key file
        key_file = gcp_config.get('key_file')
        
        # Authenticate if key file is provided
        if key_file:
            if not authenticate_gcloud(key_file):
                return False
        else:
            logger.info("✅ Using VM's built-in service account")
        
        # Set project
        if not set_project(project_id):
            return False
        
        # Get dataset list
        dataset_list = datasets.get('datasets', [])
        total_datasets = len(dataset_list)
        
        if total_datasets == 0:
            logger.warning("⚠️ No datasets found in configuration")
            return False
        
        logger.info(f"📋 Found {total_datasets} datasets to process")
        
        # Create progress tracker
        tracker = ProgressTracker(total_datasets)
        
        # Start progress monitor
        stop_event = threading.Event()
        progress_thread = threading.Thread(
            target=progress_monitor, 
            args=(tracker, total_datasets, stop_event)
        )
        progress_thread.start()
        
        # Process datasets with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            # Submit all tasks
            future_to_dataset = {}
            for i, dataset in enumerate(dataset_list, 1):
                future = executor.submit(
                    process_single_dataset,
                    dataset,
                    bucket_name,
                    key_file,
                    project_id,
                    tracker,
                    i,
                    total_datasets
                )
                future_to_dataset[future] = dataset
            
            # Wait for all tasks to complete
            for future in as_completed(future_to_dataset):
                dataset = future_to_dataset[future]
                try:
                    result = future.result()
                except Exception as e:
                    logger.error(f"❌ Unexpected error processing {dataset['name']}: {e}")
        
        # Stop progress monitor
        stop_event.set()
        progress_thread.join()
        
        # Final summary
        successful, failed, completed = tracker.get_progress()
        logger.info(f"🎉 Batch processing completed!")
        logger.info(f"✅ Successful: {successful}/{total_datasets}")
        logger.info(f"❌ Failed: {failed}/{total_datasets}")
        logger.info(f"📁 Datasets available in: gs://{bucket_name}/")
        
        return successful == total_datasets
        
    except Exception as e:
        logger.error(f"❌ Batch process failed: {e}")
        return False

if __name__ == "__main__":
    main() 