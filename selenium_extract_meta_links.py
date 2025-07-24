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
    
    # Enable network logging
    options.set_preference('devtools.console.stdout.content', True)
    options.set_preference('devtools.netmonitor.enabled', True)
    
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
        time.sleep(5)
        
        # Wait for content to be present
        wait = WebDriverWait(driver, 15)
        
        # Save initial page source for inspection
        with open('meta_ai_page_source.html', 'w', encoding='utf-8') as f:
            f.write(driver.page_source)
        logger.info("💾 Saved initial page source to meta_ai_page_source.html")
        
        download_links = []
        
        # Method 1: Look for clickable elements with zip file names
        logger.info("🔍 Searching for clickable download elements...")
        
        # Find all clickable elements (links, buttons, divs, spans)
        clickable_elements = []
        
        # Look for links
        links = driver.find_elements(By.TAG_NAME, "a")
        clickable_elements.extend(links)
        
        # Look for buttons
        buttons = driver.find_elements(By.TAG_NAME, "button")
        clickable_elements.extend(buttons)
        
        # Look for divs and spans that might be clickable
        divs = driver.find_elements(By.TAG_NAME, "div")
        spans = driver.find_elements(By.TAG_NAME, "span")
        
        # Filter divs and spans that might be clickable (have onclick, cursor pointer, etc.)
        for div in divs:
            try:
                onclick = div.get_attribute('onclick')
                style = div.get_attribute('style')
                class_name = div.get_attribute('class')
                
                if (onclick or 
                    (style and 'cursor: pointer' in style) or
                    (class_name and any(keyword in class_name.lower() for keyword in ['click', 'download', 'link']))):
                    clickable_elements.append(div)
            except:
                continue
        
        for span in spans:
            try:
                onclick = span.get_attribute('onclick')
                style = span.get_attribute('style')
                class_name = span.get_attribute('class')
                
                if (onclick or 
                    (style and 'cursor: pointer' in style) or
                    (class_name and any(keyword in class_name.lower() for keyword in ['click', 'download', 'link']))):
                    clickable_elements.append(span)
            except:
                continue
        
        logger.info(f"Found {len(clickable_elements)} potentially clickable elements")
        
        # Look for elements with zip file names
        zip_patterns = [
            r'CC_part_\d+_\d+\.zip',
            r'CC_annotations\.zip',
            r'CC_\w+\.zip',
            r'part_\d+_\d+\.zip',
            r'annotations\.zip'
        ]
        
        for element in clickable_elements:
            try:
                text = element.text.strip()
                if not text:
                    continue
                
                # Check if text matches zip file patterns
                for pattern in zip_patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        logger.info(f"📦 Found clickable element with zip name: {text}")
                        
                        # Try to click and capture the generated URL
                        try:
                            # Store current URL
                            current_url = driver.current_url
                            
                            # Click the element
                            logger.info(f"🖱️ Clicking on: {text}")
                            element.click()
                            
                            # Wait a moment for any redirect or new window
                            time.sleep(2)
                            
                            # Check if URL changed
                            new_url = driver.current_url
                            if new_url != current_url:
                                logger.info(f"✅ URL changed after click: {new_url}")
                                download_links.append({
                                    'text': text,
                                    'url': new_url,
                                    'filename': text
                                })
                            else:
                                # Check if a new window/tab opened
                                if len(driver.window_handles) > 1:
                                    # Switch to new window
                                    driver.switch_to.window(driver.window_handles[-1])
                                    new_url = driver.current_url
                                    logger.info(f"✅ New window opened with URL: {new_url}")
                                    download_links.append({
                                        'text': text,
                                        'url': new_url,
                                        'filename': text
                                    })
                                    # Close new window and switch back
                                    driver.close()
                                    driver.switch_to.window(driver.window_handles[0])
                                else:
                                    logger.warning(f"⚠️ Click on {text} didn't generate a new URL")
                            
                        except Exception as click_error:
                            logger.warning(f"⚠️ Error clicking on {text}: {click_error}")
                            continue
                        
                        break  # Found a match, move to next element
                        
            except Exception as e:
                logger.warning(f"Error processing element: {e}")
                continue
        
        # Method 2: Look for any text that contains zip file names
        page_text = driver.page_source
        for pattern in zip_patterns:
            matches = re.findall(pattern, page_text, re.IGNORECASE)
            for match in matches:
                logger.info(f"📄 Found zip filename in page text: {match}")
        
        # Method 3: Try to execute JavaScript to find clickable elements
        logger.info("🔍 Trying JavaScript approach to find download elements...")
        try:
            # Execute JavaScript to find elements with zip file names
            js_script = """
            function findDownloadElements() {
                const elements = [];
                const zipPatterns = [
                    /CC_part_\\d+_\\d+\\.zip/i,
                    /CC_annotations\\.zip/i,
                    /CC_\\w+\\.zip/i,
                    /part_\\d+_\\d+\\.zip/i,
                    /annotations\\.zip/i
                ];
                
                // Find all elements with text content
                const allElements = document.querySelectorAll('*');
                
                for (let element of allElements) {
                    const text = element.textContent.trim();
                    if (!text) continue;
                    
                    for (let pattern of zipPatterns) {
                        if (pattern.test(text)) {
                            elements.push({
                                text: text,
                                tagName: element.tagName,
                                className: element.className,
                                id: element.id,
                                onclick: element.onclick,
                                href: element.href
                            });
                        }
                    }
                }
                
                return elements;
            }
            return findDownloadElements();
            """
            
            js_results = driver.execute_script(js_script)
            logger.info(f"🔍 JavaScript found {len(js_results)} elements with zip names")
            
            for result in js_results:
                logger.info(f"📦 JS found: {result['text']} (tag: {result['tagName']}, class: {result['className']})")
                
                # Try to click this element
                try:
                    # Find the element by text content
                    element = driver.find_element(By.XPATH, f"//*[contains(text(), '{result['text']}')]")
                    
                    # Store current URL
                    current_url = driver.current_url
                    
                    # Click the element
                    logger.info(f"🖱️ Clicking on JS-found element: {result['text']}")
                    element.click()
                    
                    # Wait for any changes
                    time.sleep(3)
                    
                    # Check if URL changed
                    new_url = driver.current_url
                    if new_url != current_url:
                        logger.info(f"✅ URL changed after JS click: {new_url}")
                        download_links.append({
                            'text': result['text'],
                            'url': new_url,
                            'filename': result['text']
                        })
                    else:
                        # Check for new windows
                        if len(driver.window_handles) > 1:
                            driver.switch_to.window(driver.window_handles[-1])
                            new_url = driver.current_url
                            logger.info(f"✅ New window opened with URL: {new_url}")
                            download_links.append({
                                'text': result['text'],
                                'url': new_url,
                                'filename': result['text']
                            })
                            driver.close()
                            driver.switch_to.window(driver.window_handles[0])
                        else:
                            logger.warning(f"⚠️ JS click on {result['text']} didn't generate a new URL")
                            
                except Exception as click_error:
                    logger.warning(f"⚠️ Error clicking JS-found element {result['text']}: {click_error}")
                    continue
                    
        except Exception as js_error:
            logger.warning(f"⚠️ JavaScript approach failed: {js_error}")
        
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