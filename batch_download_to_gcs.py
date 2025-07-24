#!/usr/bin/env python3
"""
Batch download datasets from multiple URLs and upload to Google Cloud Storage
Designed to run on Google Cloud CPU VMs
"""

import os
import json
import logging
import argparse
from download_to_gcs import process_dataset_url, load_config, authenticate_gcloud, set_project, check_gcloud_installation

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    """Main batch download function"""
    parser = argparse.ArgumentParser(description='Batch download datasets from URLs and upload to Google Cloud Storage')
    parser.add_argument('--dataset-config', default='CC_dataset_urls.json', help='Dataset configuration file')
    parser.add_argument('--config', default='gcp_config.json', help='GCP configuration file')
    parser.add_argument('--bucket-name', help='GCS bucket name (overrides config)')
    parser.add_argument('--create-sample', action='store_true', help='Create a sample dataset configuration file')
    
    args = parser.parse_args()
    
    # Create sample config if requested
    if args.create_sample:
        create_sample_dataset_config()
        return True
    
    logger.info("🚀 Starting batch download and upload to Google Cloud Storage")
    
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
        
        # Process each dataset
        successful = 0
        total = len(datasets.get('datasets', []))
        
        for i, dataset in enumerate(datasets.get('datasets', []), 1):
            logger.info(f"📦 Processing dataset {i}/{total}: {dataset['name']}")
            
            try:
                if process_dataset_url(
                    dataset['url'], 
                    bucket_name, 
                    dataset['remote_path'], 
                    key_file,  # Pass the key file for authentication
                    project_id
                ):
                    successful += 1
                    logger.info(f"✅ Successfully processed: {dataset['name']} ({i}/{total})")
                else:
                    logger.error(f"❌ Failed to process: {dataset['name']} ({i}/{total})")
            except Exception as e:
                logger.error(f"❌ Error processing {dataset['name']} ({i}/{total}): {e}")
            
            # Add a summary every 10 datasets
            if i % 10 == 0:
                logger.info(f"📊 Progress summary: {i}/{total} datasets processed, {successful} successful")
        
        logger.info(f"🎉 Batch processing completed!")
        logger.info(f"✅ Successful: {successful}/{total}")
        logger.info(f"📁 Datasets available in: gs://{bucket_name}/")
        
        return successful == total
        
    except Exception as e:
        logger.error(f"❌ Batch process failed: {e}")
        return False

if __name__ == "__main__":
    main() 