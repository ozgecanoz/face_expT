#!/usr/bin/env python3
"""
Extract download links and zip file names from a webpage
"""

import requests
from bs4 import BeautifulSoup
import json
import argparse
import logging
import re
from urllib.parse import urljoin, urlparse
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_page_content(url):
    """Get the HTML content of a webpage"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.text
    except Exception as e:
        logger.error(f"❌ Failed to fetch page: {e}")
        return None

def extract_zip_links(html_content, base_url):
    """Extract zip file download links from HTML content"""
    soup = BeautifulSoup(html_content, 'html.parser')
    links = []
    
    # Find all links
    for link in soup.find_all('a', href=True):
        href = link.get('href')
        text = link.get_text(strip=True)
        
        # Make URL absolute
        absolute_url = urljoin(base_url, href)
        
        # Check if it's a zip file
        if is_zip_file(absolute_url):
            filename = extract_filename(absolute_url, text)
            links.append({
                'url': absolute_url,
                'filename': filename,
                'link_text': text
            })
    
    return links

def is_zip_file(url):
    """Check if URL points to a zip file"""
    zip_extensions = ['.zip', '.tar.gz', '.tar.bz2', '.rar', '.7z']
    url_lower = url.lower()
    
    # Check file extension
    for ext in zip_extensions:
        if ext in url_lower:
            return True
    
    # Check if URL contains zip-related keywords
    zip_keywords = ['zip', 'download', 'dataset', 'data']
    url_lower = url_lower.lower()
    for keyword in zip_keywords:
        if keyword in url_lower:
            return True
    
    return False

def extract_filename(url, link_text):
    """Extract a meaningful filename from URL or link text"""
    # Try to get filename from URL path
    parsed_url = urlparse(url)
    path = parsed_url.path
    filename = os.path.basename(path)
    
    # If filename is empty or generic, use link text
    if not filename or filename in ['', '/', 'download', 'Download']:
        # Clean up link text
        filename = re.sub(r'[^\w\s\-_.]', '', link_text)
        filename = re.sub(r'\s+', '_', filename)
        if not filename:
            filename = 'dataset'
    
    # Add .zip extension if missing
    if not any(filename.lower().endswith(ext) for ext in ['.zip', '.tar.gz', '.tar.bz2', '.rar', '.7z']):
        filename += '.zip'
    
    return filename

def generate_dataset_config(links, output_file='dataset_urls.json'):
    """Generate dataset configuration from extracted links"""
    datasets = []
    
    for i, link in enumerate(links, 1):
        # Create a clean name for the dataset
        name = link['filename'].replace('.zip', '').replace('.tar.gz', '').replace('.tar.bz2', '')
        name = re.sub(r'[^\w\s\-_]', '', name)
        name = re.sub(r'\s+', '_', name)
        
        # Create remote path
        remote_path = f"datasets/{name}"
        
        datasets.append({
            'name': name,
            'url': link['url'],
            'remote_path': remote_path,
            'original_filename': link['filename'],
            'link_text': link['link_text']
        })
    
    config = {'datasets': datasets}
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"✅ Generated config file: {output_file}")
    return config

def print_links_summary(links):
    """Print a summary of found links"""
    logger.info(f"📦 Found {len(links)} potential download links:")
    
    for i, link in enumerate(links, 1):
        logger.info(f"  {i}. {link['filename']}")
        logger.info(f"     URL: {link['url']}")
        logger.info(f"     Text: {link['link_text']}")
        logger.info("")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Extract download links from a webpage')
    parser.add_argument('url', help='URL of the webpage to scrape')
    parser.add_argument('--output', default='dataset_urls.json', help='Output configuration file')
    parser.add_argument('--print-only', action='store_true', help='Only print links, don\'t generate config')
    parser.add_argument('--verbose', action='store_true', help='Show detailed information')
    
    args = parser.parse_args()
    
    logger.info(f"🔍 Scraping webpage: {args.url}")
    
    # Get page content
    html_content = get_page_content(args.url)
    if not html_content:
        return False
    
    # Extract links
    links = extract_zip_links(html_content, args.url)
    
    if not links:
        logger.warning("⚠️  No zip file links found on the page")
        logger.info("💡 Try checking the page manually or use --verbose for more info")
        return False
    
    # Print summary
    print_links_summary(links)
    
    if not args.print_only:
        # Generate config file
        config = generate_dataset_config(links, args.output)
        
        logger.info(f"🎉 Successfully extracted {len(links)} download links")
        logger.info(f"📁 Configuration saved to: {args.output}")
        logger.info("💡 Edit the file to customize remote paths and names")
        
        if args.verbose:
            logger.info("\n📋 Generated configuration:")
            print(json.dumps(config, indent=2))
    
    return True

if __name__ == "__main__":
    main() 