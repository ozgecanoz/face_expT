#!/usr/bin/env python3
"""
Script to submit the download form and capture the actual download URLs
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

def submit_download_form(driver, url):
    """Submit the download form and capture the response"""
    try:
        logger.info(f"🌐 Navigating to: {url}")
        driver.get(url)
        
        # Wait for page to load
        time.sleep(5)
        
        # Save initial page source
        with open('meta_ai_page_source.html', 'w', encoding='utf-8') as f:
            f.write(driver.page_source)
        logger.info("💾 Saved initial page source to meta_ai_page_source.html")
        
        # Look for the download form
        logger.info("🔍 Looking for download form...")
        
        # Method 1: Find form by action URL
        forms = driver.find_elements(By.TAG_NAME, "form")
        download_form = None
        
        for form in forms:
            action = form.get_attribute('action')
            if action and 'casual-conversations-downloads' in action:
                download_form = form
                logger.info(f"📋 Found download form: action='{action}', method='{form.get_attribute('method')}'")
                break
        
        if not download_form:
            logger.error("❌ Download form not found")
            return False
        
        # Look for the "Download Dataset" button
        logger.info("🔍 Looking for 'Download Dataset' button...")
        
        # Try different selectors for the button
        button_selectors = [
            "button[type='submit']",
            "input[type='submit']",
            "button:contains('Download')",
            "input[value*='Download']",
            "button",
            "input[type='button']"
        ]
        
        download_button = None
        for selector in button_selectors:
            try:
                buttons = download_form.find_elements(By.CSS_SELECTOR, selector)
                for button in buttons:
                    text = button.text.strip() or button.get_attribute('value') or ''
                    logger.info(f"🔘 Found button: '{text}'")
                    if 'download' in text.lower() or 'dataset' in text.lower():
                        download_button = button
                        logger.info(f"✅ Found download button: '{text}'")
                        break
                if download_button:
                    break
            except Exception as e:
                logger.debug(f"Selector {selector} failed: {e}")
        
        if not download_button:
            # Try to find any button in the form
            all_buttons = download_form.find_elements(By.TAG_NAME, "button")
            if all_buttons:
                download_button = all_buttons[0]
                logger.info(f"🔘 Using first button found: '{download_button.text}'")
            else:
                logger.error("❌ No download button found")
                return False
        
        # Before clicking, let's check if there are any hidden fields we need to fill
        logger.info("🔍 Checking for form fields...")
        hidden_inputs = download_form.find_elements(By.CSS_SELECTOR, "input[type='hidden']")
        for hidden_input in hidden_inputs:
            name = hidden_input.get_attribute('name')
            value = hidden_input.get_attribute('value')
            logger.info(f"📝 Hidden field: {name} = {value}")
        
        # Also check for any checkboxes or radio buttons
        checkboxes = download_form.find_elements(By.CSS_SELECTOR, "input[type='checkbox']")
        for checkbox in checkboxes:
            name = checkbox.get_attribute('name')
            value = checkbox.get_attribute('value')
            logger.info(f"☑️ Checkbox: {name} = {value}")
            # Check the checkbox if it's related to terms/agreement
            if 'agree' in name.lower() or 'terms' in name.lower() or 'accept' in name.lower():
                if not checkbox.is_selected():
                    checkbox.click()
                    logger.info(f"✅ Checked checkbox: {name}")
        
        # Now submit the form
        logger.info("🚀 Submitting download form...")
        
        # Method 1: Click the button
        try:
            download_button.click()
            logger.info("✅ Clicked download button")
            time.sleep(5)  # Wait for response
        except Exception as e:
            logger.warning(f"⚠️ Could not click button: {e}")
            
            # Method 2: Submit the form directly
            try:
                download_form.submit()
                logger.info("✅ Submitted form directly")
                time.sleep(5)  # Wait for response
            except Exception as e:
                logger.error(f"❌ Could not submit form: {e}")
                return False
        
        # Check if we got redirected or if there's a new page
        current_url = driver.current_url
        logger.info(f"📍 Current URL after submission: {current_url}")
        
        # Save the page source after submission
        with open('meta_ai_after_submit.html', 'w', encoding='utf-8') as f:
            f.write(driver.page_source)
        logger.info("💾 Saved page source after submission to meta_ai_after_submit.html")
        
        # Look for download links in the response
        page_text = driver.page_source
        
        # Look for zip file patterns
        zip_patterns = [
            r'CC_part_\d+_\d+\.zip',
            r'CC_annotations\.zip',
            r'casual_conversations.*\.zip',
            r'part_\d+.*\.zip',
            r'https://[^"\']*\.zip',
            r'["\']([^"\']*\.zip)["\']'
        ]
        
        found_files = []
        for pattern in zip_patterns:
            matches = re.findall(pattern, page_text, re.IGNORECASE)
            for match in matches:
                found_files.append(match)
                logger.info(f"📄 Found zip filename: {match}")
        
        # Look for any links that might be downloads
        all_links = driver.find_elements(By.TAG_NAME, "a")
        download_links = []
        
        for link in all_links:
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
        
        # Check if we got any response
        if len(found_files) > 0 or len(download_links) > 0:
            logger.info("✅ Found download content after form submission!")
            return True
        else:
            logger.warning("⚠️ No download content found after form submission")
            logger.info("💡 The form might require additional interaction or have different behavior")
            return False
        
    except Exception as e:
        logger.error(f"❌ Error submitting download form: {e}")
        return False

def main():
    """Main function to submit download form"""
    url = "https://ai.meta.com/datasets/casual-conversations-downloads/"
    
    driver = setup_firefox_driver()
    if not driver:
        return False
    
    try:
        success = submit_download_form(driver, url)
        
        if success:
            logger.info("✅ Form submission completed successfully")
            logger.info("📄 Check meta_ai_after_submit.html for the response")
        else:
            logger.warning("⚠️ Form submission completed but no download links found")
            logger.info("💡 The page might require additional interaction")
        
        return success
        
    finally:
        logger.info("🔚 Closing Firefox driver")
        driver.quit()

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("✅ Form submission completed successfully")
    else:
        logger.warning("⚠️ Form submission completed but no download links found") 