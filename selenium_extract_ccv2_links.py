#!/usr/bin/env python3
"""
Selenium script to extract download links from CCv2 (Casual Conversations v2) dataset page
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
    """Set up Firefox driver with network monitoring capabilities"""
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

def save_page_source(driver, filename="ccv2_page_source.html"):
    """Save page source for inspection"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(driver.page_source)
        logger.info(f"💾 Saved page source to {filename}")
    except Exception as e:
        logger.error(f"❌ Failed to save page source: {e}")

def extract_ccv2_download_links(driver, url):
    """Extract download links from the CCv2 dataset page"""
    try:
        logger.info(f"🌐 Navigating to: {url}")
        driver.get(url)
        
        # Wait for page to load
        time.sleep(5)
        
        # Save initial page source
        save_page_source(driver, "ccv2_page_source.html")
        
        # Wait for content to be present
        wait = WebDriverWait(driver, 15)
        
        # Look for zip file patterns in page content
        logger.info("🔍 Searching for zip file patterns in page content...")
        zip_patterns = [
            r'CCv2_[^"]*\.zip',
            r'ccv2_[^"]*\.zip',
            r'casual_conversations_v2[^"]*\.zip'
        ]
        
        page_text = driver.page_source
        found_files = []
        
        for pattern in zip_patterns:
            matches = re.findall(pattern, page_text, re.IGNORECASE)
            for match in matches:
                if match not in found_files:
                    found_files.append(match)
                    logger.info(f"📄 Found zip filename: {match}")
        
        # Look for download buttons or forms
        logger.info("🔍 Looking for download buttons or forms...")
        
        # Look for forms
        forms = driver.find_elements(By.TAG_NAME, "form")
        logger.info(f"Found {len(forms)} forms on the page")
        
        for i, form in enumerate(forms):
            try:
                action = form.get_attribute('action')
                method = form.get_attribute('method')
                logger.info(f"📋 Form {i+1}: action='{action}', method='{method}'")
                
                # Look for download buttons in this form
                buttons = form.find_elements(By.TAG_NAME, "button")
                for j, button in enumerate(buttons):
                    button_text = button.text.strip()
                    if button_text and ('download' in button_text.lower() or 'ccv2' in button_text.lower()):
                        logger.info(f"🔘 Found download button: '{button_text}'")
            except Exception as e:
                logger.warning(f"⚠️ Error examining form {i+1}: {e}")
        
        # Execute JavaScript to find download elements
        logger.info("🔍 Executing JavaScript to find download elements...")
        try:
            # Look for elements containing CCv2 or download text
            js_code = """
            var elements = document.querySelectorAll('*');
            var downloadElements = [];
            for (var i = 0; i < elements.length; i++) {
                var element = elements[i];
                var text = element.textContent || element.innerText || '';
                if (text.toLowerCase().includes('ccv2') || 
                    text.toLowerCase().includes('download') ||
                    text.toLowerCase().includes('casual conversations')) {
                    downloadElements.push({
                        tag: element.tagName,
                        text: text.substring(0, 100),
                        id: element.id,
                        className: element.className
                    });
                }
            }
            return downloadElements;
            """
            
            download_elements = driver.execute_script(js_code)
            logger.info(f"🔍 JavaScript found {len(download_elements)} potential download elements")
            
            for i, element in enumerate(download_elements[:10]):  # Show first 10
                logger.info(f"📄 Element: {element['tag']} - '{element['text']}'")
                
        except Exception as e:
            logger.warning(f"⚠️ JavaScript execution failed: {e}")
        
        # Search for all links on the page
        logger.info("🔍 Searching for all links on the page...")
        all_links = driver.find_elements(By.TAG_NAME, "a")
        logger.info(f"Found {len(all_links)} total links")
        
        potential_download_links = []
        for link in all_links:
            try:
                href = link.get_attribute('href')
                text = link.text.strip()
                
                if href and ('download' in href.lower() or 'ccv2' in href.lower() or 'zip' in href.lower()):
                    logger.info(f"🔗 Potential download link: {text} -> {href}")
                    potential_download_links.append({
                        'text': text,
                        'url': href
                    })
            except Exception as e:
                continue
        
        if not found_files and not potential_download_links:
            logger.warning("⚠️ No download links found")
            logger.info("💡 The page might have dynamic content or different structure")
            logger.info("📄 Check ccv2_page_source.html for page content")
            return []
        
        # Return found files for now (we'll need to get actual URLs)
        return found_files
        
    except Exception as e:
        logger.error(f"❌ Error extracting links: {e}")
        return []

def main():
    """Main function"""
    # CCv2 dataset URL (you'll need to find the correct URL)
    ccv2_url = "https://ai.meta.com/datasets/casual-conversations-v2-downloads/"
    
    logger.info("🚀 Starting CCv2 download link extraction...")
    logger.info(f"🌐 Target URL: {ccv2_url}")
    
    # Set up driver
    driver = setup_firefox_driver()
    if not driver:
        logger.error("❌ Failed to set up Firefox driver")
        return
    
    try:
        # Extract download links
        found_files = extract_ccv2_download_links(driver, ccv2_url)
        
        if found_files:
            logger.info(f"✅ Found {len(found_files)} potential files:")
            for file in found_files:
                logger.info(f"   📄 {file}")
            
            # Create a sample JSON structure
            sample_data = {
                "datasets": [
                    {
                        "name": file,
                        "url": "TO_BE_EXTRACTED",  # Will need actual URL
                        "remote_path": "face_training_datasets/casual_conversations_v2"
                    }
                    for file in found_files
                ]
            }
            
            # Save sample structure
            with open("ccv2_sample_structure.json", "w") as f:
                json.dump(sample_data, f, indent=2)
            
            logger.info("📁 Saved sample structure to ccv2_sample_structure.json")
            logger.info("💡 You'll need to manually extract the actual download URLs")
            
        else:
            logger.warning("⚠️ No files found. Check the page source for different patterns.")
            
    except Exception as e:
        logger.error(f"❌ Error during extraction: {e}")
    
    finally:
        driver.quit()
        logger.info("🔚 Firefox driver closed")

if __name__ == "__main__":
    main() 