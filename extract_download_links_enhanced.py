#!/usr/bin/env python3
"""
Enhanced extract download links with better handling of modern websites
"""

import requests
from bs4 import BeautifulSoup
import json
import argparse
import logging
import re
from urllib.parse import urljoin, urlparse
import os
import time
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_page_content(url, max_retries=3):
    """Get the HTML content of a webpage with enhanced headers and retry logic"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Cache-Control': 'max-age=0'
    }
    
    session = requests.Session()
    session.headers.update(headers)
    
    for attempt in range(max_retries):
        try:
            logger.info(f"🔍 Attempt {attempt + 1}/{max_retries} to fetch: {url}")
            
            # Add a small delay between retries
            if attempt > 0:
                delay = random.uniform(1, 3)
                logger.info(f"⏳ Waiting {delay:.1f}s before retry...")
                time.sleep(delay)
            
            response = session.get(url, timeout=30)
            response.raise_for_status()
            
            logger.info(f"✅ Successfully fetched page (status: {response.status_code})")
            return response.text
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"❌ HTTP Error {e.response.status_code}: {e}")
            if e.response.status_code == 403:
                logger.info("💡 This might be a protected page. Try manual inspection.")
            elif e.response.status_code == 404:
                logger.error("❌ Page not found. Check the URL.")
            elif e.response.status_code == 429:
                logger.warning("⚠️  Rate limited. Waiting before retry...")
                time.sleep(5)
            else:
                logger.error(f"❌ HTTP error: {e}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Request failed: {e}")
            
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
    
    logger.error(f"❌ Failed to fetch page after {max_retries} attempts")
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
        
        # Clean URL by removing query parameters
        clean_url = absolute_url.split('?')[0]
        
        # Check if it's a zip file
        if is_zip_file(clean_url):
            filename = extract_filename(clean_url, text)
            links.append({
                'url': clean_url,
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
    zip_keywords = ['zip', 'download', 'dataset', 'data', 'file']
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

def save_html_for_inspection(html_content, filename='page_content.html'):
    """Save HTML content for manual inspection"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.info(f"💾 Saved page content to: {filename}")
        logger.info("🔍 You can open this file in a browser to inspect the page manually")
    except Exception as e:
        logger.error(f"❌ Failed to save HTML: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Extract download links from a webpage')
    parser.add_argument('url', help='URL of the webpage to scrape')
    parser.add_argument('--output', default='dataset_urls.json', help='Output configuration file')
    parser.add_argument('--print-only', action='store_true', help='Only print links, don\'t generate config')
    parser.add_argument('--verbose', action='store_true', help='Show detailed information')
    parser.add_argument('--save-html', action='store_true', help='Save HTML content for manual inspection')
    parser.add_argument('--retries', type=int, default=3, help='Number of retry attempts')
    
    args = parser.parse_args()
    
    logger.info(f"🔍 Scraping webpage: {args.url}")
    
    # Get page content
    html_content = get_page_content(args.url, args.retries)
    if not html_content:
        logger.error("❌ Could not fetch page content")
        logger.info("💡 Try these alternatives:")
        logger.info("   1. Check if the URL is correct")
        logger.info("   2. Try accessing the page manually in a browser")
        logger.info("   3. The page might require JavaScript or authentication")
        logger.info("   4. Use --save-html to inspect the page content")
        return False
    
    # Save HTML for inspection if requested
    if args.save_html:
        save_html_for_inspection(html_content)
    
    # Extract links
    links = extract_zip_links(html_content, args.url)
    
    if not links:
        logger.warning("⚠️  No zip file links found on the page")
        logger.info("💡 This could mean:")
        logger.info("   1. The page uses JavaScript to load links")
        logger.info("   2. Links are in a different format")
        logger.info("   3. The page requires authentication")
        logger.info("   4. Try --save-html to inspect the page manually")
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