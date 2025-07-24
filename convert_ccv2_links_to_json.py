#!/usr/bin/env python3
"""
Convert CCV2_download_links.txt to JSON format for batch download
"""

import json
import csv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_ccv2_links_to_json():
    """Convert CCV2_download_links.txt to JSON format"""
    
    input_file = "CCV2_download_links.txt"
    output_file = "CCV2_dataset_urls.json"
    
    try:
        # Read the tab-separated file
        datasets = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            # Skip header line
            next(f)
            
            for line in f:
                line = line.strip()
                if line:
                    # Split by tab
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        file_name = parts[0].strip()
                        cdn_link = parts[1].strip()
                        
                        # Keep the full URL with query parameters (they're essential for Facebook CDN)
                        clean_url = cdn_link
                        
                        dataset_entry = {
                            "name": file_name,
                            "url": clean_url,
                            "remote_path": "face_training_datasets/casual_conversations_v2"
                        }
                        
                        datasets.append(dataset_entry)
                        logger.info(f"📄 Added: {file_name}")
        
        # Create the JSON structure
        json_data = {
            "datasets": datasets
        }
        
        # Write to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)
        
        logger.info(f"✅ Successfully converted {len(datasets)} datasets to {output_file}")
        logger.info(f"📊 Summary:")
        logger.info(f"   - Total files: {len(datasets)}")
        logger.info(f"   - Remote path: face_training_datasets/casual_conversations_v2")
        logger.info(f"   - URLs: Full Facebook CDN URLs with authentication parameters")
        
        # Show some examples
        logger.info(f"📋 Sample entries:")
        for i, dataset in enumerate(datasets[:5]):
            logger.info(f"   {i+1}. {dataset['name']}")
        
        if len(datasets) > 5:
            logger.info(f"   ... and {len(datasets) - 5} more files")
        
        return True
        
    except FileNotFoundError:
        logger.error(f"❌ File not found: {input_file}")
        return False
    except Exception as e:
        logger.error(f"❌ Error converting file: {e}")
        return False

def main():
    """Main function"""
    logger.info("🔄 Converting CCV2_download_links.txt to JSON format...")
    
    success = convert_ccv2_links_to_json()
    
    if success:
        logger.info("✅ Conversion completed successfully!")
        logger.info("📁 You can now use CCV2_dataset_urls.json with batch_download_to_gcs.py")
    else:
        logger.error("❌ Conversion failed!")

if __name__ == "__main__":
    main() 