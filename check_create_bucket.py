#!/usr/bin/env python3
"""
Check and create GCS bucket if needed
"""

import subprocess
import json
import logging

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

def check_and_create_bucket():
    """Check if bucket exists and create if needed"""
    
    # Load config
    try:
        with open('gcp_config.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.error("❌ gcp_config.json not found")
        return False
    
    bucket_name = config.get('bucket_name')
    project_id = config.get('project_id')
    region = config.get('region', 'us-central1')
    
    if not bucket_name or not project_id:
        logger.error("❌ Missing bucket_name or project_id in config")
        return False
    
    logger.info(f"🔍 Checking bucket: gs://{bucket_name}")
    
    # Check if bucket exists
    try:
        result = run_command(f"gsutil ls -b gs://{bucket_name}", check=False)
        if result.returncode == 0:
            logger.info(f"✅ Bucket already exists: gs://{bucket_name}")
            return True
    except Exception as e:
        logger.warning(f"⚠️ Error checking bucket: {e}")
    
    # Create bucket
    logger.info(f"📦 Creating bucket: gs://{bucket_name}")
    try:
        run_command(f"gsutil mb -p {project_id} -c STANDARD -l {region} gs://{bucket_name}")
        logger.info(f"✅ Successfully created bucket: gs://{bucket_name}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to create bucket: {e}")
        return False

if __name__ == "__main__":
    check_and_create_bucket() 