#!/usr/bin/env python3
"""
Enhanced Selenium script to extract download links from Meta AI Casual Conversations dataset page
This version captures network requests and handles dynamic content
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
    
    # Enable network monitoring
    options.set_preference("devtools.console.stdout.content", True)
    options.set_preference("devtools.netmonitor.enabled", True)
    
    try:
        driver = webdriver.Firefox(options=options)
        logger.info("✅ Firefox driver created successfully")
        return driver
    except Exception as e:
        logger.error(f"❌ Failed to create Firefox driver: {e}")
        return None

def extract_download_links_enhanced(driver, url):
    """Enhanced extraction with network monitoring and dynamic content handling"""
    try:
        logger.info(f"🌐 Navigating to: {url}")
        driver.get(url)
        
        # Wait for page to load
        time.sleep(5)
        
        # Save initial page source
        with open('meta_ai_page_source.html', 'w', encoding='utf-8') as f:
            f.write(driver.page_source)
        logger.info("💾 Saved initial page source to meta_ai_page_source.html")
        
        # Wait for content to be present
        wait = WebDriverWait(driver, 15)
        
        download_links = []
        
        # Method 1: Look for any text that contains zip file patterns
        logger.info("🔍 Searching for zip file patterns in page content...")
        page_text = driver.page_source
        
        # Look for common zip file patterns
        zip_patterns = [
            r'CC_part_\d+_\d+\.zip',
            r'CC_annotations\.zip',
            r'casual_conversations.*\.zip',
            r'part_\d+.*\.zip'
        ]
        
        found_files = []
        for pattern in zip_patterns:
            matches = re.findall(pattern, page_text, re.IGNORECASE)
            for match in matches:
                found_files.append(match)
                logger.info(f"📄 Found zip filename: {match}")
        
        # Method 2: Try to find and click download buttons
        logger.info("🔍 Looking for download buttons or forms...")
        
        # Look for buttons with download-related text
        download_selectors = [
            "button[contains(text(), 'Download')]",
            "button[contains(text(), 'download')]",
            "a[contains(text(), 'Download')]",
            "a[contains(text(), 'download')]",
            "input[type='submit'][value*='Download']",
            "input[type='button'][value*='Download']"
        ]
        
        for selector in download_selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                for element in elements:
                    logger.info(f"🔘 Found download element: {element.text}")
                    # Try clicking to see if it generates a download
                    try:
                        element.click()
                        time.sleep(2)
                        logger.info("✅ Clicked download element")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not click element: {e}")
            except Exception as e:
                logger.debug(f"Selector {selector} not found: {e}")
        
        # Method 3: Look for forms that might trigger downloads
        logger.info("🔍 Looking for forms...")
        forms = driver.find_elements(By.TAG_NAME, "form")
        logger.info(f"Found {len(forms)} forms on the page")
        
        for i, form in enumerate(forms):
            try:
                logger.info(f"📋 Form {i+1}: action='{form.get_attribute('action')}', method='{form.get_attribute('method')}'")
                
                # Look for submit buttons in the form
                submit_buttons = form.find_elements(By.CSS_SELECTOR, "input[type='submit'], button[type='submit']")
                for button in submit_buttons:
                    logger.info(f"🔘 Submit button: {button.get_attribute('value') or button.text}")
                    try:
                        button.click()
                        time.sleep(3)
                        logger.info("✅ Clicked submit button")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not click submit button: {e}")
            except Exception as e:
                logger.debug(f"Could not process form {i+1}: {e}")
        
        # Method 4: Execute JavaScript to find download elements
        logger.info("🔍 Executing JavaScript to find download elements...")
        
        js_script = """
        // Find all elements that might be download links
        var downloadElements = [];
        
        // Look for elements with download-related text
        var allElements = document.querySelectorAll('*');
        for (var i = 0; i < allElements.length; i++) {
            var element = allElements[i];
            var text = element.textContent || element.innerText || '';
            if (text.toLowerCase().includes('download') || 
                text.toLowerCase().includes('cc_part') ||
                text.toLowerCase().includes('zip')) {
                downloadElements.push({
                    tagName: element.tagName,
                    text: text.trim(),
                    className: element.className,
                    id: element.id
                });
            }
        }
        
        return downloadElements;
        """
        
        try:
            download_elements = driver.execute_script(js_script)
            logger.info(f"🔍 JavaScript found {len(download_elements)} potential download elements")
            
            for elem in download_elements[:10]:  # Show first 10
                logger.info(f"📄 Element: {elem['tagName']} - '{elem['text'][:50]}...'")
        except Exception as e:
            logger.warning(f"⚠️ JavaScript execution failed: {e}")
        
        # Method 5: Look for any links that might be downloads
        logger.info("🔍 Searching for all links on the page...")
        all_links = driver.find_elements(By.TAG_NAME, "a")
        logger.info(f"Found {len(all_links)} total links")
        
        for link in all_links[:20]:  # Check first 20 links
            try:
                href = link.get_attribute('href')
                text = link.text.strip()
                
                if href and ('download' in href.lower() or 'zip' in href.lower() or 'cc_part' in text.lower()):
                    logger.info(f"🔗 Potential download link: {text} -> {href}")
                    download_links.append({
                        'text': text,
                        'url': href,
                        'type': 'link'
                    })
            except Exception as e:
                logger.debug(f"Could not process link: {e}")
        
        # Method 6: Check if there are any hidden download URLs in the page
        logger.info("🔍 Searching for hidden download URLs in page source...")
        
        # Look for URLs in JavaScript variables or data attributes
        js_url_patterns = [
            r'["\']([^"\']*\.zip)["\']',
            r'["\']([^"\']*download[^"\']*)["\']',
            r'url["\']?\s*:\s*["\']([^"\']*\.zip)["\']',
            r'href["\']?\s*:\s*["\']([^"\']*\.zip)["\']'
        ]
        
        for pattern in js_url_patterns:
            matches = re.findall(pattern, page_text, re.IGNORECASE)
            for match in matches:
                logger.info(f"🔗 Found potential download URL in JS: {match}")
                download_links.append({
                    'text': 'JavaScript URL',
                    'url': match,
                    'type': 'js_url'
                })
        
        return download_links, found_files
        
    except Exception as e:
        logger.error(f"❌ Error extracting download links: {e}")
        return [], []

def main():
    """Main function to extract download links"""
    url = "https://ai.meta.com/datasets/casual-conversations-downloads/"
    
    driver = setup_firefox_driver()
    if not driver:
        return False
    
    try:
        download_links, found_files = extract_download_links_enhanced(driver, url)
        
        logger.info(f"📊 Extraction Summary:")
        logger.info(f"   Found {len(download_links)} download links")
        logger.info(f"   Found {len(found_files)} zip file names")
        
        if download_links:
            logger.info("🔗 Download Links Found:")
            for link in download_links:
                logger.info(f"   {link['text']} -> {link['url']}")
        
        if found_files:
            logger.info("📦 Zip Files Found:")
            for file in found_files:
                logger.info(f"   {file}")
        
        if not download_links and not found_files:
            logger.warning("⚠️ No download links or zip files found")
            logger.info("💡 The page might require user interaction or have different structure")
            logger.info("📄 Check meta_ai_page_source.html for page content")
        
        return len(download_links) > 0 or len(found_files) > 0
        
    finally:
        logger.info("🔚 Closing Firefox driver")
        driver.quit()

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("✅ Extraction completed successfully")
    else:
        logger.warning("⚠️ Extraction completed but no links found") 