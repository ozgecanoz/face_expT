#!/usr/bin/env python3
"""
Selenium-based script to extract download links from Meta AI Casual Conversations dataset page
"""

import json
import time
import logging
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from urllib.parse import urljoin, urlparse
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_firefox_driver():
    """Set up Firefox driver with appropriate options"""
    options = Options()
    options.add_argument('--headless')  # Run in background
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    
    # Add user agent to mimic real browser
    options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0')
    
    try:
        driver = webdriver.Firefox(options=options)
        logger.info("✅ Firefox driver created successfully")
        return driver
    except Exception as e:
        logger.error(f"❌ Failed to create Firefox driver: {e}")
        return None


def extract_download_links(driver, url):
    """Extract download links from the Meta AI dataset page"""
    try:
        logger.info(f"🌐 Navigating to: {url}")
        driver.get(url)
        
        # Wait for page to load
        time.sleep(3)
        
        # Wait for content to be present
        wait = WebDriverWait(driver, 10)
        
        # Try to find download links - Meta AI might have different structures
        download_links = []
        
        # Method 1: Look for links with zip files
        logger.info("🔍 Searching for download links...")
        
        # Find all links on the page
        all_links = driver.find_elements(By.TAG_NAME, "a")
        logger.info(f"Found {len(all_links)} total links on the page")
        
        for link in all_links:
            try:
                href = link.get_attribute('href')
                text = link.text.strip()
                
                if href and text:
                    # Check if it's a download link
                    if any(keyword in text.lower() for keyword in ['download', 'zip', 'part', 'cc_']):
                        logger.info(f"📦 Found potential download link: {text} -> {href}")
                        download_links.append({
                            'text': text,
                            'url': href,
                            'filename': text if text.endswith('.zip') else f"{text}.zip"
                        })
                    
                    # Also check for zip files in href
                    elif href and ('.zip' in href.lower() or 'download' in href.lower()):
                        filename = text if text else href.split('/')[-1]
                        logger.info(f"📦 Found zip link: {filename} -> {href}")
                        download_links.append({
                            'text': text,
                            'url': href,
                            'filename': filename
                        })
                        
            except Exception as e:
                logger.warning(f"Error processing link: {e}")
                continue
        
        # Method 2: Look for buttons that might trigger downloads
        buttons = driver.find_elements(By.TAG_NAME, "button")
        logger.info(f"Found {len(buttons)} buttons on the page")
        
        for button in buttons:
            try:
                text = button.text.strip()
                if any(keyword in text.lower() for keyword in ['download', 'zip', 'part']):
                    logger.info(f"🔘 Found download button: {text}")
                    # Note: We can't get direct URLs from buttons, but we can log them
            except Exception as e:
                continue
        
        # Method 3: Look for any text that might indicate download sections
        page_text = driver.page_source.lower()
        if 'casual conversations' in page_text:
            logger.info("✅ Found 'Casual Conversations' content on page")
        
        if 'download' in page_text:
            logger.info("✅ Found download-related content on page")
        
        # Save page source for inspection
        with open('meta_ai_page_source.html', 'w', encoding='utf-8') as f:
            f.write(driver.page_source)
        logger.info("💾 Saved page source to meta_ai_page_source.html for inspection")
        
        return download_links
        
    except TimeoutException:
        logger.error("❌ Page load timeout")
        return []
    except Exception as e:
        logger.error(f"❌ Error extracting links: {e}")
        return []


def generate_dataset_config(links):
    """Generate dataset configuration from extracted links"""
    config = {
        "datasets": []
    }
    
    for i, link in enumerate(links):
        # Clean the URL by removing query parameters
        clean_url = link['url'].split('?')[0]
        
        dataset_entry = {
            "name": link['filename'],
            "url": clean_url,
            "remote_path": "face_training_datasets/casual_conversations"
        }
        
        config["datasets"].append(dataset_entry)
        logger.info(f"📝 Added dataset: {link['filename']}")
    
    return config


def main():
    """Main function to extract download links"""
    url = "https://ai.meta.com/datasets/casual-conversations-downloads/"
    
    logger.info("🚀 Starting Selenium-based link extraction")
    
    # Set up driver
    driver = setup_firefox_driver()
    if not driver:
        return False
    
    try:
        # Extract links
        links = extract_download_links(driver, url)
        
        if links:
            logger.info(f"✅ Found {len(links)} download links")
            
            # Generate config
            config = generate_dataset_config(links)
            
            # Save to file
            with open('CC_dataset_urls.json', 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info("💾 Saved dataset configuration to CC_dataset_urls.json")
            
            # Print summary
            print("\n📋 Extracted Download Links:")
            for link in links:
                print(f"  • {link['filename']}")
            
            return True
        else:
            logger.warning("⚠️ No download links found")
            logger.info("💡 The page might have dynamic content or different structure")
            logger.info("📄 Check meta_ai_page_source.html for page content")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error in main: {e}")
        return False
    finally:
        driver.quit()
        logger.info("🔚 Firefox driver closed")


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Link extraction completed successfully!")
        print("📁 Check CC_dataset_urls.json for the extracted links")
    else:
        print("\n❌ Link extraction failed")
        print("🔍 Check the logs and meta_ai_page_source.html for debugging") 