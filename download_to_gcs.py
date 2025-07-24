#!/usr/bin/env python3
"""
Download dataset from remote URLs and upload directly to Google Cloud Storage
Designed to run on Google Cloud CPU VMs
"""

import os
import json
import requests
import subprocess
import logging
import argparse
from pathlib import Path
from urllib.parse import urlparse
import tempfile
import zipfile
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, check=True):
    """Run a shell command and return the result"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if check and result.returncode != 0:
            logger.error(f"Command failed: {command}")
            logger.error(f"Error: {result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, command)
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        raise

def check_gcloud_installation():
    """Check if gcloud CLI is installed"""
    try:
        result = run_command("gcloud --version", check=False)
        if result.returncode == 0:
            logger.info("✅ gcloud CLI is installed")
            return True
        else:
            logger.error("❌ gcloud CLI is not installed")
            return False
    except FileNotFoundError:
        logger.error("❌ gcloud CLI is not found in PATH")
        return False

def authenticate_gcloud(key_file_path):
    """Authenticate gcloud with service account key"""
    if key_file_path is None:
        # VM uses built-in service account - no authentication needed
        logger.info("✅ Using VM's built-in service account")
        return True
    
    try:
        run_command(f"gcloud auth activate-service-account --key-file={key_file_path}")
        logger.info("✅ Authenticated with service account")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Authentication failed: {e}")
        return False

def set_project(project_id):
    """Set the GCP project"""
    try:
        run_command(f"gcloud config set project {project_id}")
        logger.info(f"✅ Set project to: {project_id}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to set project: {e}")
        return False

def download_file(url, local_path):
    """Download a file from URL with progress tracking"""
    try:
        logger.info(f"📥 Downloading: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        logger.info(f"   Progress: {percent:.1f}% ({downloaded}/{total_size} bytes)")
        
        logger.info(f"✅ Download completed: {local_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        return False

def extract_zip(zip_path, extract_dir):
    """Extract a zip file"""
    try:
        logger.info(f"📦 Extracting: {zip_path}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        logger.info(f"✅ Extraction completed: {extract_dir}")
        return True
    except Exception as e:
        logger.error(f"❌ Extraction failed: {e}")
        return False

def upload_to_gcs(local_path, bucket_name, remote_path):
    """Upload directory to GCS"""
    try:
        gs_path = f"gs://{bucket_name}/{remote_path}"
        logger.info(f"📤 Uploading {local_path} to {gs_path}")
        
        # Use rsync for efficient upload
        run_command(f"gsutil -m rsync -r {local_path} {gs_path}")
        
        logger.info(f"✅ Upload completed: {gs_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Upload failed: {e}")
        return False

def verify_upload(bucket_name, remote_path):
    """Verify the upload by listing files"""
    try:
        gs_path = f"gs://{bucket_name}/{remote_path}"
        result = run_command(f"gsutil ls -r {gs_path}")
        
        files = result.stdout.strip().split('\n')
        file_count = len([f for f in files if f and not f.endswith('/')])
        
        logger.info(f"✅ Upload verified: {file_count} files in {gs_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Verification failed: {e}")
        return False

def load_config(config_file="gcp_config.json"):
    """Load configuration from file"""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        logger.info(f"✅ Loaded config from: {config_file}")
        return config
    except FileNotFoundError:
        logger.error(f"❌ Config file not found: {config_file}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid config file: {e}")
        return None

def process_dataset_url(url, bucket_name, remote_path, key_file, project_id):
    """Process a single dataset URL"""
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"📁 Using temporary directory: {temp_dir}")
        
        # Download the file
        filename = os.path.basename(urlparse(url).path)
        if not filename:
            filename = "dataset.zip"
        
        local_file = os.path.join(temp_dir, filename)
        if not download_file(url, local_file):
            return False
        
        # Extract if it's a zip file
        extract_dir = os.path.join(temp_dir, "extracted")
        if filename.lower().endswith('.zip'):
            if not extract_zip(local_file, extract_dir):
                return False
            upload_path = extract_dir
        else:
            # If not a zip file, upload the file directly
            upload_path = local_file
        
        # Upload to GCS
        if not upload_to_gcs(upload_path, bucket_name, remote_path):
            return False
        
        # Verify upload
        if not verify_upload(bucket_name, remote_path):
            return False
        
        return True

def main():
    """Main download and upload function"""
    parser = argparse.ArgumentParser(description='Download dataset from URL and upload to Google Cloud Storage')
    parser.add_argument('--url', required=True, help='URL of the dataset file to download')
    parser.add_argument('--remote-path', required=True, help='Remote path in GCS bucket')
    parser.add_argument('--key-file', default='dataset-uploader-key.json', help='Service account key file')
    parser.add_argument('--config', default='gcp_config.json', help='Configuration file')
    parser.add_argument('--bucket-name', help='GCS bucket name (overrides config)')
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting download and upload to Google Cloud Storage")
    
    # Check prerequisites
    if not check_gcloud_installation():
        return False
    
    # Load configuration
    config = load_config(args.config)
    if not config:
        logger.error("Please create gcp_config.json first")
        return False
    
    # Use command line args or config
    key_file = args.key_file if os.path.exists(args.key_file) else config.get('key_file', args.key_file)
    bucket_name = args.bucket_name or config.get('bucket_name', 'face-training-datasets')
    project_id = config.get('project_id')
    
    if not project_id:
        logger.error("No project ID found in config")
        return False
    
    try:
        # Authenticate
        if not authenticate_gcloud(key_file):
            return False
        
        # Set project
        if not set_project(project_id):
            return False
        
        # Process the dataset
        if not process_dataset_url(args.url, bucket_name, args.remote_path, key_file, project_id):
            return False
        
        logger.info("🎉 Download and upload completed successfully!")
        logger.info(f"📁 Dataset location: gs://{bucket_name}/{args.remote_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Process failed: {e}")
        return False

if __name__ == "__main__":
    main() 