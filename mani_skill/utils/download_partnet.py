import time
import json
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests
from datetime import datetime

BASE_URL = "https://sapien.ucsd.edu/browse"

# Define all categories to download and their counts
CATEGORIES = [
    ("Bottle", 57),
    ("Box", 28),
    ("Bucket", 36),
    ("Camera", 37),
    ("Cart", 61),
    ("Chair", 81),
    ("Clock", 31),
    ("CoffeeMachine", 54),
    ("Dishwasher", 48),
    ("Dispenser", 57),
    ("Display", 37),
    ("Door", 36),
    ("Eyeglasses", 65),
    ("Fan", 81),
    ("Faucet", 84),
    ("FoldingChair", 26),
    ("Globe", 61),
    ("Kettle", 29),
    ("Keyboard", 37),
    ("KitchenPot", 25),
    ("Knife", 44),
    ("Lamp", 45),
    ("Laptop", 55),
    ("Lighter", 28),
    ("Microwave", 16),
    ("Mouse", 14),
    ("Oven", 30),
    ("Pen", 48),
    ("Phone", 18),
    ("Pliers", 25),
    ("Printer", 29),
    ("Refrigerator", 44),
    ("Remote", 49),
    ("Safe", 30),
    ("Scissors", 47),
    ("Stapler", 23),
    ("StorageFurniture", 346),
    ("Suitcase", 24),
    ("Switch", 70),
    ("Table", 101),
    ("Toaster", 25),
    ("Toilet", 69),
    ("TrashCan", 70),
    ("USB", 51),
    ("WashingMachine", 17),
    ("Window", 58)
]

def get_download_directory():
    """Get the correct download directory path"""
    # Get the user's home directory
    home_dir = os.path.expanduser("~")
    
    # Build the download directory path
    download_dir = os.path.join(home_dir, "partnet_downloads")
    
    # Ensure the directory exists
    os.makedirs(download_dir, exist_ok=True)
    
    print(f"Download directory set to: {download_dir}")
    return download_dir

class SAPIENSpecificCategoriesDownloader:
    def __init__(self):
        self.chrome_options = webdriver.ChromeOptions()
        self.chrome_options.add_argument('--no-sandbox')
        self.chrome_options.add_argument('--disable-dev-shm-usage')
        
        # Get the correct download directory
        self.download_dir = get_download_directory()
        
        # Set Chrome's download directory to the correct path
        prefs = {"download.default_directory": self.download_dir}
        self.chrome_options.add_experimental_option("prefs", prefs)
        
        self.driver = None
        self.all_downloads = {}  # Store download info for all categories
        self.page_mode = '1'  # Default to automatic page turning
        self.use_async = False  # Default to not using async mode
        
    def start(self):
        """Start the browser"""
        print("Starting browser...")
        self.driver = webdriver.Chrome(options=self.chrome_options)
        self.driver.maximize_window()
        
    def navigate_and_login(self):
        """Navigate to the page and log in"""
        print("Visiting the SAPIEN browse page...")
        self.driver.get(BASE_URL)
        
        input("\nPlease log in to the website first (if needed), then press Enter to continue...")
        
    def select_category(self, category_name, expected_count):
        """Select a specific category"""
        print(f"\nPreparing to select category: {category_name} (expected {expected_count} items)")
        
        # Try to click the category via JavaScript
        js_code = f"""
        function selectCategory() {{
            // Find elements containing the category name
            const elements = document.querySelectorAll('*');
            for (let elem of elements) {{
                if (elem.textContent.trim() === '{category_name}' || 
                    elem.textContent.includes('{category_name} (')) {{
                    // Check if the element is clickable
                    if (elem.tagName === 'A' || elem.tagName === 'BUTTON' || 
                        elem.onclick || elem.style.cursor === 'pointer') {{
                        elem.click();
                        return true;
                    }}
                }}
            }}
            return false;
        }}
        
        return selectCategory();
        """
        
        try:
            success = self.driver.execute_script(js_code)
            if success:
                print(f"Successfully auto-selected '{category_name}'")
                time.sleep(5)  # Increase wait time to ensure the page fully loads
                return True
        except Exception as e:
            print(f"Auto-selection failed: {e}")
        
        # If auto-selection fails, prompt for manual action
        input(f"\nPlease manually click the '{category_name}' category, wait for the page to finish loading, then press Enter...")
        return True
    
    def extract_all_pages_async(self, category_name, expected_count):
        """Use async JavaScript to extract all pages at once (similar to the original code)"""
        print(f"\nExtracting all links for '{category_name}' category using async mode...")
        
        js_code = """
        async function extractAllLinks() {
            let allLinks = {};
            let currentPage = 1;
            
            // Function: get all download links on the current page
            function getCurrentPageLinks() {
                let links = [];
                
                // First try a.download
                const downloadElements = document.querySelectorAll('a.download');
                
                downloadElements.forEach(elem => {
                    const href = elem.getAttribute('href');
                    if (href && href.includes('.zip')) {
                        const match = href.match(/\/(\d+)\.zip/);
                        if (match) {
                            links.push({
                                id: match[1],
                                url: href,
                                filename: match[1] + '.zip'
                            });
                        }
                    }
                });
                
                // If none found, try other selectors
                if (links.length === 0) {
                    const altLinks = document.querySelectorAll('a[href*="/api/download/compressed/"]');
                    altLinks.forEach(elem => {
                        const href = elem.getAttribute('href');
                        if (href && href.includes('.zip')) {
                            const match = href.match(/\/(\d+)\.zip/);
                            if (match) {
                                links.push({
                                    id: match[1],
                                    url: href,
                                    filename: match[1] + '.zip'
                                });
                            }
                        }
                    });
                }
                
                return links;
            }
            
            // Function: get the current page number
            function getCurrentPageNumber() {
                const activeCell = document.querySelector('div.cell.active');
                return activeCell ? parseInt(activeCell.textContent) : 1;
            }
            
            // Function: click a specific page number
            function clickPage(pageNum) {
                const cells = document.querySelectorAll('div.cell');
                for (let cell of cells) {
                    if (cell.textContent.trim() === pageNum.toString()) {
                        cell.click();
                        return true;
                    }
                }
                return false;
            }
            
            // Function: wait for the page to load
            function waitForLoad(timeout = 5000) {
                return new Promise(resolve => setTimeout(resolve, timeout));
            }
            
            // Main loop
            let maxPages = 50;
            let consecutiveEmptyPages = 0;
            
            for (let page = 1; page <= maxPages; page++) {
                console.log(`Processing page ${page}...`);
                
                // Get links on the current page
                const pageLinks = getCurrentPageLinks();
                let newCount = 0;
                
                pageLinks.forEach(link => {
                    if (!allLinks[link.id]) {
                        allLinks[link.id] = link;
                        newCount++;
                    }
                });
                
                console.log(`Page ${page}: found ${pageLinks.length} links, ${newCount} of them new`);
                
                if (newCount === 0) {
                    consecutiveEmptyPages++;
                    if (consecutiveEmptyPages >= 2) {
                        console.log('No new links for 2 consecutive pages, stopping');
                        break;
                    }
                } else {
                    consecutiveEmptyPages = 0;
                }
                
                // Check if there is a next page
                const hasNextPage = clickPage(page + 1);
                if (!hasNextPage) {
                    console.log('Reached the last page');
                    break;
                }
                
                // Wait for the page to load
                await waitForLoad();
            }
            
            return Object.values(allLinks);
        }
        
        // Run the extraction
        return await extractAllLinks();
        """
        
        try:
            # Execute the JavaScript code
            all_links = self.driver.execute_script(js_code)
            print(f"Async extraction complete, found {len(all_links)} unique links (expected: {expected_count})")
            return all_links
        except Exception as e:
            print(f"Async extraction failed: {e}")
            print("Falling back to the regular method...")
            return None
    
    def extract_all_pages_for_category(self, category_name, expected_count):
        """Extract all page links for a category"""
        # If async mode is enabled, try async extraction first
        if hasattr(self, 'use_async') and self.use_async:
            links = self.extract_all_pages_async(category_name, expected_count)
            if links is not None:
                return links
        
        # Use the regular method
        print(f"\nStarting to extract all links for '{category_name}' category...")
        
        all_links = {}  # Use a dict to deduplicate automatically
        page_num = 1
        max_pages = 50  # Set a maximum page count to prevent infinite loops
        consecutive_empty_pages = 0
        
        while page_num <= max_pages:
            print(f"Processing page {page_num}...")
            
            # Extract the current page
            page_links = self.extract_current_page_links()
            
            # Deduplicate and add new links
            new_links_count = 0
            if page_links:
                for link in page_links:
                    if link['id'] not in all_links:
                        all_links[link['id']] = link
                        new_links_count += 1
                
                print(f"Page {page_num}: found {len(page_links)} links, {new_links_count} of them new (total unique: {len(all_links)})")
                
                # If this page has no new links, increment the empty page count
                if new_links_count == 0:
                    consecutive_empty_pages += 1
                    if consecutive_empty_pages >= 2:
                        print("No new links for 2 consecutive pages, likely the end")
                        break
                else:
                    consecutive_empty_pages = 0
            else:
                print(f"No links found on page {page_num}")
                # If no links were found on the first page, run debug
                if page_num == 1 and hasattr(self, '_debug_mode') and self._debug_mode:
                    print("\nNo links found on the first page, starting debug...")
                    self.debug_page_structure()
                    
                consecutive_empty_pages += 1
                if consecutive_empty_pages >= 2:
                    print("No links for 2 consecutive pages, stopping")
                    break
            
            # Try to turn to the next page
            has_next = self.go_to_next_page()
            if not has_next:
                print("No more pages")
                # If fewer links than expected were collected, show debug info
                if len(all_links) < expected_count * 0.8:  # Less than 80% of expected
                    print(f"\nWarning: only collected {len(all_links)} links, fewer than the expected {expected_count}")
                    if hasattr(self, 'page_mode') and self.page_mode == '1':
                        print("Attempting pagination debug...")
                        self.debug_pagination()
                        manual = input("\nContinue by manually turning pages? (y/n): ").strip().lower()
                        if manual == 'y':
                            original_mode = self.page_mode
                            self.page_mode = '3'  # Temporarily switch to manual mode
                            has_next = self.go_to_next_page()
                            if has_next:
                                page_num += 1
                                print("Waiting for the page to load...")
                                time.sleep(3)
                                # Restore the original mode
                                self.page_mode = original_mode
                                continue
                break
                
            page_num += 1
            print("Waiting for the page to load...")
            time.sleep(3)  # Increase wait time to ensure the page fully loads
        
        # Convert to a list
        unique_links = list(all_links.values())
        
        print(f"\nFound {len(unique_links)} unique links for '{category_name}' category (expected: {expected_count})")
        
        return unique_links
    
    def extract_current_page_links(self):
        """Extract links from the current page"""
        js_code = """
        function extractCurrentPageLinks() {
            let links = [];
            
            // Use selectors proven effective in the original code
            // First try a.download
            let downloadElements = document.querySelectorAll('a.download');
            
            if (downloadElements.length > 0) {
                console.log('Found a.download elements:', downloadElements.length);
                downloadElements.forEach(elem => {
                    const href = elem.getAttribute('href');
                    if (href && href.includes('.zip')) {
                        const match = href.match(/\/(\d+)\.zip/);
                        if (match) {
                            links.push({
                                id: match[1],
                                url: href,
                                filename: match[1] + '.zip'
                            });
                        }
                    }
                });
            }
            
            // If none found, try other selectors
            if (links.length === 0) {
                const selectors = [
                    'a[href*="/api/download/compressed/"]',
                    'a[href*=".zip"]',
                    'button.download',
                    '[data-download-url]',
                    'a[download]'
                ];
                
                // Use a Set for deduplication
                const foundUrls = new Set();
                
                selectors.forEach(selector => {
                    try {
                        const elements = document.querySelectorAll(selector);
                        elements.forEach(elem => {
                            let href = elem.getAttribute('href') || 
                                      elem.getAttribute('data-download-url') || 
                                      elem.getAttribute('data-url');
                            
                            if (href && href.includes('.zip')) {
                                const match = href.match(/\/(\d+)\.zip/);
                                if (match && !foundUrls.has(match[1])) {
                                    foundUrls.add(match[1]);
                                    links.push({
                                        id: match[1],
                                        url: href,
                                        filename: match[1] + '.zip'
                                    });
                                }
                            }
                        });
                    } catch (e) {
                        console.error('Error with selector:', selector, e);
                    }
                });
            }
            
            // Final fallback method
            if (links.length === 0) {
                console.log('Using fallback method to find links...');
                document.querySelectorAll('a').forEach(elem => {
                    const href = elem.getAttribute('href');
                    if (href && href.includes('download') && href.includes('.zip')) {
                        const match = href.match(/\/(\d+)\.zip/);
                        if (match) {
                            links.push({
                                id: match[1],
                                url: href,
                                filename: match[1] + '.zip'
                            });
                        }
                    }
                });
            }
            
            console.log('Found', links.length, 'links on current page');
            return links;
        }
        
        return extractCurrentPageLinks();
        """
        
        try:
            links = self.driver.execute_script(js_code)
            # Add debug output
            if not links and hasattr(self, '_debug_mode') and self._debug_mode:
                print("No links found, attempting debug...")
                self.debug_page_structure()
            return links
        except Exception as e:
            print(f"Error extracting links from current page: {e}")
            return []
    
    def go_to_next_page(self):
        """Go to the next page"""
        # If in manual mode, ask the user to turn the page manually
        if hasattr(self, 'page_mode') and self.page_mode == '3':
            response = input("\nIs there a next page? If so, manually click it and press Enter (or enter n if not): ").strip().lower()
            return response != 'n'
        
        js_code = """
        function goToNextPage() {
            // SAPIEN-specific method: find div.cell page-number elements
            function getCurrentPageNumber() {
                const activeCell = document.querySelector('div.cell.active');
                return activeCell ? parseInt(activeCell.textContent) : null;
            }
            
            const currentPage = getCurrentPageNumber();
            console.log('Current page number:', currentPage);
            
            if (currentPage !== null) {
                // Method 1: click the next page number
                const cells = document.querySelectorAll('div.cell');
                for (let cell of cells) {
                    const pageNum = parseInt(cell.textContent.trim());
                    if (!isNaN(pageNum) && pageNum === currentPage + 1) {
                        console.log('Clicking page number:', pageNum);
                        cell.click();
                        return true;
                    }
                }
            }
            
            // Fallback: find all possible page-number elements
            const pageSelectors = ['div.cell', '.page-item', '.pagination-item'];
            for (let selector of pageSelectors) {
                const elements = document.querySelectorAll(selector);
                
                // Find the currently active page number
                let currentIndex = -1;
                for (let i = 0; i < elements.length; i++) {
                    if (elements[i].classList.contains('active') || 
                        elements[i].classList.contains('current') ||
                        elements[i].getAttribute('aria-current') === 'page') {
                        currentIndex = i;
                        break;
                    }
                }
                
                // If the current page is found, click the next one
                if (currentIndex >= 0 && currentIndex < elements.length - 1) {
                    const nextElement = elements[currentIndex + 1];
                    if (nextElement && !nextElement.classList.contains('disabled')) {
                        console.log('Clicking next element:', nextElement);
                        nextElement.click();
                        return true;
                    }
                }
            }
            
            // Look for the "next page" button
            const nextPatterns = ['Next', '>', '→', 'next'];
            for (let pattern of nextPatterns) {
                const elements = document.querySelectorAll('button, a, span[onclick], div[onclick]');
                for (let elem of elements) {
                    const text = elem.textContent.trim();
                    if ((text === pattern || text.toLowerCase() === pattern.toLowerCase()) && 
                        !elem.disabled && 
                        !elem.classList.contains('disabled')) {
                        console.log('Found next button:', elem);
                        elem.click();
                        return true;
                    }
                }
            }
            
            console.log('No next page found');
            return false;
        }
        
        return goToNextPage();
        """
        
        try:
            result = self.driver.execute_script(js_code)
            if result:
                print("Successfully turned to the next page")
                time.sleep(1)  # Brief wait to ensure the page starts loading
                return True
            else:
                # If in semi-automatic mode, ask for manual action when auto fails
                if hasattr(self, 'page_mode') and self.page_mode == '2':
                    response = input("\nAuto page turning failed. Is there a next page? If so, click it manually and press Enter (or enter n if not): ").strip().lower()
                    return response != 'n'
                return False
        except Exception as e:
            print(f"Error while turning the page: {e}")
            # If in semi-automatic mode, ask for manual action on error
            if hasattr(self, 'page_mode') and self.page_mode == '2':
                response = input("\nError turning the page. Is there a next page? If so, click it manually and press Enter (or enter n if not): ").strip().lower()
                return response != 'n'
            return False
    
    def debug_page_structure(self):
        """Debug the page structure to find possible download links"""
        js_code = """
        function debugPageStructure() {
            let info = {
                all_links: [],
                potential_downloads: [],
                page_info: {}
            };
            
            // Get all links
            const links = document.querySelectorAll('a');
            links.forEach(link => {
                const href = link.getAttribute('href');
                const text = link.textContent.trim();
                
                if (href) {
                    info.all_links.push({
                        href: href,
                        text: text.substring(0, 50),
                        className: link.className
                    });
                    
                    // Check if it might be a download link
                    if (href.includes('download') || 
                        href.includes('.zip') || 
                        href.includes('/api/') ||
                        text.toLowerCase().includes('download')) {
                        info.potential_downloads.push({
                            href: href,
                            text: text,
                            parent: link.parentElement.tagName,
                            parentClass: link.parentElement.className
                        });
                    }
                }
            });
            
            // Get basic page info
            info.page_info = {
                title: document.title,
                url: window.location.href,
                total_links: links.length,
                has_pagination: document.querySelector('.pagination, .pager, [class*="page"]') ? true : false
            };
            
            return info;
        }
        
        return debugPageStructure();
        """
        
        try:
            debug_info = self.driver.execute_script(js_code)
            print("\n=== Page structure debug ===")
            print(f"Page title: {debug_info['page_info']['title']}")
            print(f"Total links: {debug_info['page_info']['total_links']}")
            print(f"Potential download links: {len(debug_info['potential_downloads'])}")
            
            if debug_info['potential_downloads']:
                print("\nPotential download links:")
                for link in debug_info['potential_downloads'][:5]:
                    print(f"  URL: {link['href']}")
                    print(f"  Text: {link['text']}")
                    print(f"  Parent element: {link['parent']} (class: {link['parentClass']})")
                    print()
                    
            return debug_info
        except Exception as e:
            print(f"Error debugging page structure: {e}")
            return None
    
    def debug_pagination(self):
        """Debug pagination elements"""
        js_code = """
        function debugPagination() {
            let info = {
                pagination_containers: [],
                clickable_elements: [],
                active_elements: []
            };
            
            // Find pagination containers
            const paginationSelectors = [
                '.pagination', '.pager', '[class*="pagination"]', 
                '[class*="page"]', '.nav-links', '.page-numbers'
            ];
            
            paginationSelectors.forEach(selector => {
                const elements = document.querySelectorAll(selector);
                elements.forEach(el => {
                    if (el.children.length > 0) {
                        info.pagination_containers.push({
                            selector: selector,
                            className: el.className,
                            childCount: el.children.length,
                            innerHTML: el.innerHTML.substring(0, 200)
                        });
                    }
                });
            });
            
            // Find possible page-turning buttons
            const buttons = document.querySelectorAll('a, button, span[onclick], div[onclick]');
            buttons.forEach(btn => {
                const text = btn.textContent.trim();
                if (text && text.length < 20 && 
                    (text.match(/\d+/) || text.match(/next|prev|>/i))) {
                    info.clickable_elements.push({
                        tag: btn.tagName,
                        text: text,
                        className: btn.className,
                        href: btn.href || 'none',
                        onclick: btn.onclick ? 'has onclick' : 'no onclick'
                    });
                }
            });
            
            // Find currently active page elements
            const activeSelectors = ['.active', '.current', '[aria-current]', '.selected', 'div.cell.active'];
            activeSelectors.forEach(selector => {
                const elements = document.querySelectorAll(selector);
                elements.forEach(el => {
                    info.active_elements.push({
                        selector: selector,
                        text: el.textContent.trim(),
                        className: el.className
                    });
                });
            });
            
            // Specifically check div.cell elements (SAPIEN-specific)
            const cells = document.querySelectorAll('div.cell');
            if (cells.length > 0) {
                info.div_cells = [];
                cells.forEach(cell => {
                    const hasOnClick = cell.onclick !== null;
                    const parentHasOnClick = cell.parentElement && cell.parentElement.onclick !== null;
                    info.div_cells.push({
                        text: cell.textContent.trim(),
                        className: cell.className,
                        isActive: cell.classList.contains('active'),
                        hasOnClick: hasOnClick,
                        parentHasOnClick: parentHasOnClick,
                        cursor: window.getComputedStyle(cell).cursor
                    });
                });
            }
            
            return info;
        }
        
        return debugPagination();
        """
        
        try:
            debug_info = self.driver.execute_script(js_code)
            print("\n=== Pagination debug info ===")
            print(f"Found {len(debug_info['pagination_containers'])} possible pagination containers")
            print(f"Found {len(debug_info['clickable_elements'])} possible page-turning elements")
            print(f"Found {len(debug_info['active_elements'])} active-state elements")
            
            if debug_info['clickable_elements']:
                print("\nClickable element examples:")
                for elem in debug_info['clickable_elements'][:10]:
                    print(f"  {elem['tag']}: '{elem['text']}' (class: {elem['className']})")
            
            if 'div_cells' in debug_info and debug_info['div_cells']:
                print(f"\nFound {len(debug_info['div_cells'])} div.cell elements:")
                for cell in debug_info['div_cells']:
                    active = " [ACTIVE]" if cell['isActive'] else ""
                    clickable = " [CLICKABLE]" if (cell['hasOnClick'] or cell['cursor'] == 'pointer') else ""
                    print(f"  Page {cell['text']}: {cell['className']}{active}{clickable}")
                    
            return debug_info
        except Exception as e:
            print(f"Error debugging pagination: {e}")
            return None
    
    def download_links(self, links, category_name):
        """Download all links for a given category"""
        if not links:
            print(f"No links found for '{category_name}' category")
            return
        
        # Create the category directory using the correct download directory
        category_dir = os.path.join(self.download_dir, category_name)
        if not os.path.exists(category_dir):
            os.makedirs(category_dir)
        
        print(f"Files for category '{category_name}' will be downloaded to: {category_dir}")
        
        # Save the link list for this category
        links_file = os.path.join(category_dir, "links.json")
        with open(links_file, 'w', encoding='utf-8') as f:
            json.dump(links, f, indent=2)
        
        # Get cookies and headers
        cookies = self.driver.get_cookies()
        session = requests.Session()
        for cookie in cookies:
            session.cookies.set(cookie['name'], cookie['value'])
        
        headers = {
            'User-Agent': self.driver.execute_script("return navigator.userAgent;"),
            'Referer': BASE_URL
        }
        
        # Download files
        successful = 0
        failed = []
        
        print(f"\nStarting to download {len(links)} files for '{category_name}' category...")
        
        for i, link in enumerate(links, 1):
            filepath = os.path.join(category_dir, link['filename'])
            
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                if file_size > 1000:  # Files larger than 1KB are considered valid
                    print(f"[{i}/{len(links)}] {link['filename']} already exists, skipping")
                    successful += 1
                    continue
                else:
                    print(f"[{i}/{len(links)}] {link['filename']} file too small, re-downloading")
                    os.remove(filepath)
            
            try:
                # Make sure the URL is complete
                url = link['url']
                if not url.startswith('http'):
                    if url.startswith('/'):
                        url = 'https://sapien.ucsd.edu' + url
                    else:
                        url = BASE_URL.rsplit('/', 1)[0] + '/' + url
                
                print(f"[{i}/{len(links)}] Downloading: {link['filename']}")
                response = session.get(url, headers=headers, timeout=60, stream=True)
                response.raise_for_status()
                
                # Stream the response to the file
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                successful += 1
                time.sleep(1)  # Avoid making requests too quickly
                
            except Exception as e:
                print(f"[{i}/{len(links)}] Failed: {str(e)}")
                failed.append({**link, 'error': str(e)})
        
        # Save the download record for this category
        self.all_downloads[category_name] = {
            'total': len(links),
            'successful': successful,
            'failed': len(failed),
            'failed_links': failed
        }
        
        print(f"\n'{category_name}' download complete! Successful: {successful}/{len(links)}")
        
    def save_summary(self):
        """Save a download summary"""
        summary = {
            'download_time': datetime.now().isoformat(),
            'download_directory': self.download_dir,
            'total_categories': len(CATEGORIES),
            'categories': self.all_downloads
        }
        
        summary_file = os.path.join(self.download_dir, 'download_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Generate a detailed report
        report_lines = [
            "=== SAPIEN Dataset Download Report ===",
            f"Download time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Download directory: {self.download_dir}",
            f"Total categories: {len(CATEGORIES)}",
            ""
        ]
        
        total_expected = sum(count for _, count in CATEGORIES)
        total_downloaded = 0
        total_successful = 0
        
        for cat_name, expected_count in CATEGORIES:
            if cat_name in self.all_downloads:
                info = self.all_downloads[cat_name]
                total_downloaded += info['total']
                total_successful += info['successful']
                status = "✓" if info['successful'] == expected_count else "✗"
                report_lines.append(
                    f"{status} {cat_name}: {info['successful']}/{info['total']} "
                    f"(expected: {expected_count})"
                )
            else:
                report_lines.append(f"✗ {cat_name}: not downloaded (expected: {expected_count})")
        
        report_lines.extend([
            "",
            f"Total: {total_successful}/{total_downloaded} successfully downloaded",
            f"Expected total: {total_expected}",
            ""
        ])
        
        # Save the report
        report_file = os.path.join(self.download_dir, 'download_report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        # Print the report
        print("\n" + '\n'.join(report_lines))
        print("\nDetailed information saved to:")
        print(f"- {summary_file} (JSON format)")
        print(f"- {report_file} (text report)")
        
    def run(self):
        """Run the downloader"""
        try:
            self.start()
            self.navigate_and_login()
            
            print(f"\nPreparing to download {len(CATEGORIES)} categories, {sum(c for _, c in CATEGORIES)} items in total")
            print(f"Files will be downloaded to: {self.download_dir}")
            
            # Let the user choose the download mode
            print("\nChoose download mode:")
            print("1. Download all categories")
            print("2. Download specific categories")
            print("3. Start downloading from a specific category (for resuming interrupted downloads)")
            
            choice = input("\nPlease choose (1/2/3): ").strip()
            
            # New: ask about the page-turning mode
            print("\nChoose page-turning mode:")
            print("1. Automatic (recommended)")
            print("2. Semi-automatic (manual fallback when auto fails)")
            print("3. Manual")
            print("4. Async one-shot extraction (similar to the original code)")
            
            page_mode = input("\nPlease choose (1/2/3/4): ").strip()
            if page_mode == '4':
                self.use_async = True
                self.page_mode = '1'  # Use automatic page turning if async mode fails
            else:
                self.use_async = False
                self.page_mode = page_mode if page_mode in ['1', '2', '3'] else '1'
            
            # New: ask whether to enable debug mode
            debug = input("\nEnable debug mode? (y/n): ").strip().lower()
            self._debug_mode = (debug == 'y')
            
            categories_to_download = []
            
            if choice == '1':
                categories_to_download = CATEGORIES
            elif choice == '2':
                print("\nAvailable categories:")
                for i, (cat, count) in enumerate(CATEGORIES, 1):
                    print(f"{i:2d}. {cat} ({count})")
                
                selected = input("\nEnter category numbers to download (comma-separated, e.g. 1,3,5): ")
                indices = [int(x.strip()) - 1 for x in selected.split(',')]
                categories_to_download = [CATEGORIES[i] for i in indices if 0 <= i < len(CATEGORIES)]
            elif choice == '3':
                print("\nAvailable categories:")
                for i, (cat, count) in enumerate(CATEGORIES, 1):
                    print(f"{i:2d}. {cat} ({count})")
                
                start_idx = int(input("\nStart from which number? : ").strip()) - 1
                if 0 <= start_idx < len(CATEGORIES):
                    categories_to_download = CATEGORIES[start_idx:]
            
            if not categories_to_download:
                print("No categories selected")
                return
            
            print(f"\nWill download {len(categories_to_download)} categories")
            confirm = input("Confirm to start downloading? (y/n): ").lower().strip()
            
            if confirm != 'y':
                return
            
            # Iterate over each category
            for idx, (category_name, expected_count) in enumerate(categories_to_download, 1):
                print(f"\n\n{'='*60}")
                print(f"Processing category {idx}/{len(categories_to_download)}: {category_name}")
                print(f"{'='*60}")
                
                # Return to the main page
                self.driver.get(BASE_URL)
                time.sleep(2)
                
                # Select the category
                if self.select_category(category_name, expected_count):
                    # Extract all links for this category
                    links = self.extract_all_pages_for_category(category_name, expected_count)
                    
                    # Download
                    self.download_links(links, category_name)
                    
                    # Brief pause
                    if idx < len(categories_to_download):
                        print(f"\nFinished '{category_name}' category, resting for 3 seconds...")
                        time.sleep(3)
            
            # Save the summary
            self.save_summary()
            
        except KeyboardInterrupt:
            print("\n\nDownload interrupted by user")
            self.save_summary()
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            import traceback
            traceback.print_exc()
            self.save_summary()
        finally:
            input("\nPress Enter to close the browser...")
            if self.driver:
                self.driver.quit()

if __name__ == "__main__":
    downloader = SAPIENSpecificCategoriesDownloader()
    downloader.run()