#!/usr/bin/env python3
"""
Test script to debug GCS upload issues
"""

import subprocess
import logging
import tempfile
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_gcs_access():
    """Test basic GCS access"""
    logger.info("🔍 Testing GCS access...")
    
    # Load config to get key file
    try:
        import json
        with open('gcp_config.json', 'r') as f:
            config = json.load(f)
        key_file = config.get('key_file')
        project_id = config.get('project_id')
        
        if key_file:
            logger.info(f"🔑 Using service account key: {key_file}")
            # Authenticate with service account
            auth_result = subprocess.run(
                f"gcloud auth activate-service-account --key-file={key_file}",
                shell=True, capture_output=True, text=True
            )
            if auth_result.returncode != 0:
                logger.error(f"❌ Authentication failed: {auth_result.stderr}")
                return False
            logger.info("✅ Authenticated with service account")
        
        if project_id:
            # Set project
            project_result = subprocess.run(
                f"gcloud config set project {project_id}",
                shell=True, capture_output=True, text=True
            )
            if project_result.returncode == 0:
                logger.info(f"✅ Set project to: {project_id}")
        
    except Exception as e:
        logger.error(f"❌ Failed to load config: {e}")
        return False
    
    # Test 1: List buckets
    try:
        result = subprocess.run("gsutil ls", shell=True, capture_output=True, text=True)
        logger.info(f"✅ Bucket listing successful")
        logger.info(f"Buckets: {result.stdout}")
    except Exception as e:
        logger.error(f"❌ Bucket listing failed: {e}")
        return False
    
    # Test 2: Check specific bucket
    bucket_name = "face-training-datasets"
    try:
        result = subprocess.run(f"gsutil ls -b gs://{bucket_name}", shell=True, capture_output=True, text=True)
        logger.info(f"✅ Bucket {bucket_name} exists")
    except Exception as e:
        logger.error(f"❌ Bucket {bucket_name} check failed: {e}")
        return False
    
    # Test 3: Try to create a test file
    try:
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            temp_file = f.name
        
        logger.info(f"📝 Created test file: {temp_file}")
        
        # Upload test file
        result = subprocess.run(
            f"gsutil cp {temp_file} gs://{bucket_name}/test_upload.txt",
            shell=True, capture_output=True, text=True
        )
        
        if result.returncode == 0:
            logger.info("✅ Test upload successful")
            
            # Clean up
            subprocess.run(f"gsutil rm gs://{bucket_name}/test_upload.txt", shell=True, capture_output=True)
            os.unlink(temp_file)
            logger.info("🧹 Cleaned up test file")
        else:
            logger.error(f"❌ Test upload failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test upload failed: {e}")
        return False
    
    return True

def test_rsync_upload():
    """Test rsync upload specifically"""
    logger.info("🔍 Testing rsync upload...")
    
    bucket_name = "face-training-datasets"
    
    # Create a test directory with files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create some test files
        test_files = ["file1.txt", "file2.txt", "subdir/file3.txt"]
        
        for file_path in test_files:
            full_path = os.path.join(temp_dir, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w') as f:
                f.write(f"test content for {file_path}")
        
        logger.info(f"📁 Created test directory: {temp_dir}")
        logger.info(f"📄 Test files: {test_files}")
        
        # Try rsync upload
        try:
            result = subprocess.run(
                f"gsutil -m rsync -r {temp_dir}/ gs://{bucket_name}/test_rsync/",
                shell=True, capture_output=True, text=True
            )
            
            if result.returncode == 0:
                logger.info("✅ Rsync upload successful")
                
                # List uploaded files
                list_result = subprocess.run(
                    f"gsutil ls -r gs://{bucket_name}/test_rsync/",
                    shell=True, capture_output=True, text=True
                )
                logger.info(f"📋 Uploaded files: {list_result.stdout}")
                
                # Clean up
                subprocess.run(f"gsutil -m rm -r gs://{bucket_name}/test_rsync/", shell=True, capture_output=True)
                logger.info("🧹 Cleaned up test files")
            else:
                logger.error(f"❌ Rsync upload failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Rsync upload failed: {e}")
            return False
    
    return True

if __name__ == "__main__":
    logger.info("🚀 Starting GCS upload tests...")
    
    if test_gcs_access():
        logger.info("✅ Basic GCS access test passed")
        
        if test_rsync_upload():
            logger.info("✅ Rsync upload test passed")
            logger.info("🎉 All tests passed! Upload should work now.")
        else:
            logger.error("❌ Rsync upload test failed")
    else:
        logger.error("❌ Basic GCS access test failed") 