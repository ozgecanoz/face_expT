#!/usr/bin/env python3
"""
Setup Google Cloud Platform authentication for secure dataset upload
Creates a service account, generates a key file, and grants necessary permissions
"""

import os
import json
import subprocess
import logging
from pathlib import Path

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

def get_project_id():
    """Get the current GCP project ID"""
    try:
        result = run_command("gcloud config get-value project")
        project_id = result.stdout.strip()
        if project_id:
            logger.info(f"📋 Current project ID: {project_id}")
            return project_id
        else:
            logger.error("❌ No project ID configured")
            return None
    except subprocess.CalledProcessError:
        logger.error("❌ Failed to get project ID")
        return None

def create_service_account(project_id, service_account_name="dataset-uploader"):
    """Create a service account for dataset upload"""
    try:
        # Create service account
        email = f"{service_account_name}@{project_id}.iam.gserviceaccount.com"
        run_command(f"gcloud iam service-accounts create {service_account_name} --display-name='Dataset Uploader' --description='Service account for uploading datasets to GCS'")
        logger.info(f"✅ Created service account: {email}")
        return email
    except subprocess.CalledProcessError as e:
        if "already exists" in str(e):
            logger.info(f"✅ Service account already exists: {service_account_name}")
            return f"{service_account_name}@{project_id}.iam.gserviceaccount.com"
        else:
            logger.error(f"❌ Failed to create service account: {e}")
            raise

def create_key_file(service_account_email, key_file_path="dataset-uploader-key.json"):
    """Create a key file for the service account"""
    try:
        run_command(f"gcloud iam service-accounts keys create {key_file_path} --iam-account={service_account_email}")
        logger.info(f"✅ Created key file: {key_file_path}")
        return key_file_path
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to create key file: {e}")
        raise

def grant_permissions(service_account_email):
    """Grant necessary permissions to the service account"""
    try:
        # Grant Storage Admin role for full GCS access
        run_command(f"gcloud projects add-iam-policy-binding {service_account_email.split('@')[1].split('.')[0]} --member=serviceAccount:{service_account_email} --role=roles/storage.admin")
        logger.info("✅ Granted Storage Admin permissions")
        
        # Grant additional roles if needed
        run_command(f"gcloud projects add-iam-policy-binding {service_account_email.split('@')[1].split('.')[0]} --member=serviceAccount:{service_account_email} --role=roles/storage.objectViewer")
        logger.info("✅ Granted Storage Object Viewer permissions")
        
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to grant permissions: {e}")
        raise

def create_gitignore_entry(key_file_path):
    """Add the key file to .gitignore"""
    try:
        with open('.gitignore', 'a') as f:
            f.write(f"\n# Google Cloud Service Account Key\n{key_file_path}\n")
        logger.info(f"✅ Added {key_file_path} to .gitignore")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to update .gitignore: {e}")
        raise

def create_config_file(project_id, key_file_path):
    """Create a configuration file for the upload script"""
    config = {
        "project_id": project_id,
        "key_file": key_file_path,
        "bucket_name": "face-training-datasets",
        "region": "us-central1"
    }
    
    config_file = "gcp_config.json"
    try:
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"✅ Created config file: {config_file}")
        return config_file
    except Exception as e:
        logger.error(f"❌ Failed to create config file: {e}")
        raise

def main():
    """Main setup function"""
    logger.info("🚀 Setting up GCP authentication for dataset upload")
    
    # Check gcloud installation
    if not check_gcloud_installation():
        logger.error("Please install gcloud CLI first: https://cloud.google.com/sdk/docs/install")
        return False
    
    # Get project ID
    project_id = get_project_id()
    if not project_id:
        logger.error("Please set up a GCP project first: gcloud config set project YOUR_PROJECT_ID")
        return False
    
    try:
        # Create service account
        service_account_email = create_service_account(project_id)
        
        # Create key file
        key_file_path = create_key_file(service_account_email)
        
        # Grant permissions
        grant_permissions(service_account_email)
        
        # Add to .gitignore
        create_gitignore_entry(key_file_path)
        
        # Create config file
        create_config_file(project_id, key_file_path)
        
        logger.info("🎉 GCP authentication setup completed!")
        logger.info(f"📁 Key file: {key_file_path}")
        logger.info(f"📁 Config file: gcp_config.json")
        logger.info("⚠️  Keep your key file secure and never commit it to git!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        return False

if __name__ == "__main__":
    main() 