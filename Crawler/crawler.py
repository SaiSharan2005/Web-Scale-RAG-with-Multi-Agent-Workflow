#!/usr/bin/env python3
"""
Domain-Restricted Web Crawler

A Python web crawler that:
- Crawls all pages within a single domain
- Respects robots.txt rules
- Extracts visible text content
- Handles errors gracefully
- Saves results to JSON/CSV

Usage:
    python crawler.py https://example.com
"""

import requests
import time
import json
import csv
import logging
import re
import os
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser
from collections import deque
from bs4 import BeautifulSoup, Comment
import argparse
import sys
from typing import Set, Dict, List, Optional, Tuple
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    from dotenv import load_dotenv
    # Load crawler-specific environment file
    load_dotenv('crawler.env')
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


class DomainCrawler:
    def __init__(self, start_url: str, delay: float = 1.0, user_agent: str = "*", debug_text_extraction: bool = False, limit: Optional[int] = None, respect_robots: bool = True, custom_disallow: List[str] = None):
        """
        Initialize the crawler with configuration.
        
        Args:
            start_url: The starting URL to begin crawling
            delay: Delay between requests in seconds
            user_agent: User agent string for robots.txt checking
            debug_text_extraction: Enable detailed text extraction debugging
            limit: Maximum number of pages to crawl (None for unlimited)
            respect_robots: Whether to respect robots.txt rules
            custom_disallow: List of custom paths to disallow (e.g., ['/about', '/admin'])
        """
        self.start_url = start_url.rstrip('/')
        self.domain = urlparse(start_url).netloc
        self.delay = delay
        self.user_agent = user_agent
        self.debug_text_extraction = debug_text_extraction
        self.limit = 30
        self.respect_robots = respect_robots
        self.custom_disallow = custom_disallow or []
        
        # Data structures for crawling
        self.visited_urls: Set[str] = set()
        self.url_queue: deque = deque([self.start_url])
        self.results: Dict[str, str] = {}
        self.failed_urls: List[Tuple[str, str]] = []
        
        # HTTP session for connection reuse
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; DomainCrawler/1.0; +https://example.com/bot)'
        })
        
        # Robots.txt parser
        self.rp: Optional[RobotFileParser] = None
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def normalize_url(self, url: str) -> str:
        """
        Normalize URL to avoid duplicates.
        
        Args:
            url: Raw URL to normalize
            
        Returns:
            Normalized URL string
        """
        # Parse URL components
        parsed = urlparse(url)
        
        # Remove fragment (anchor)
        normalized = urlunparse((
            parsed.scheme,
            parsed.netloc.lower(),  # Lowercase domain
            parsed.path.rstrip('/') if parsed.path != '/' else '/',  # Remove trailing slash except root
            parsed.params,
            parsed.query,
            ''  # Remove fragment
        ))
        
        return normalized

    def is_same_domain(self, url: str) -> bool:
        """
        Check if URL belongs to the same domain as start URL.
        
        Args:
            url: URL to check
            
        Returns:
            True if same domain, False otherwise
        """
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower() == self.domain.lower()
        except Exception:
            return False

    def load_robots_txt(self) -> bool:
        """
        Load and parse robots.txt for the domain.
        
        Returns:
            True if robots.txt was successfully loaded, False otherwise
        """
        if not self.respect_robots:
            self.logger.info("Robots.txt checking disabled")
            return False
            
        try:
            robots_url = f"{urlparse(self.start_url).scheme}://{self.domain}/robots.txt"
            self.logger.info(f"Loading robots.txt from: {robots_url}")
            
            self.rp = RobotFileParser()
            self.rp.set_url(robots_url)
            self.rp.read()
            
            self.logger.info("Successfully loaded robots.txt")
            return True
            
        except Exception as e:
            self.logger.warning(f"Could not load robots.txt: {e}")
            self.logger.info("Proceeding without robots.txt restrictions")
            return False

    def can_fetch(self, url: str) -> bool:
        """
        Check if URL can be fetched according to robots.txt and custom rules.
        
        Args:
            url: URL to check
            
        Returns:
            True if URL can be fetched, False otherwise
        """
        # Check custom disallow rules first
        parsed_url = urlparse(url)
        path = parsed_url.path
        
        for disallow_path in self.custom_disallow:
            if path == disallow_path or path.startswith(disallow_path + '/'):
                self.logger.info(f"Custom rule disallows: {url}")
                return False
        
        # Check robots.txt rules
        if not self.respect_robots or self.rp is None:
            return True
        
        try:
            can_fetch = self.rp.can_fetch(self.user_agent, url)
            if not can_fetch:
                self.logger.info(f"Robots.txt disallows: {url}")
            return can_fetch
        except Exception as e:
            self.logger.warning(f"Error checking robots.txt for {url}: {e}")
            return True  # Be permissive on error

    def extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """
        Extract all links from a BeautifulSoup object.
        
        Args:
            soup: BeautifulSoup object of the page
            base_url: Base URL for resolving relative links
            
        Returns:
            List of normalized URLs
        """
        links = []
        
        # Find all anchor tags with href attributes
        for link in soup.find_all('a', href=True):
            href = link['href'].strip()
            
            # Skip empty hrefs, anchors, and common non-page links
            if not href or href.startswith('#') or href.startswith('mailto:') or href.startswith('tel:'):
                continue
            
            # Skip common file extensions that aren't web pages
            skip_extensions = {'.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', 
                             '.jpg', '.jpeg', '.png', '.gif', '.svg', '.ico', '.zip', 
                             '.tar', '.gz', '.mp3', '.mp4', '.avi', '.mov'}
            
            if any(href.lower().endswith(ext) for ext in skip_extensions):
                continue
            
            # Resolve relative URLs
            try:
                absolute_url = urljoin(base_url, href)
                normalized_url = self.normalize_url(absolute_url)
                
                # Only add if it's the same domain AND allowed by robots.txt
                if self.is_same_domain(normalized_url) and self.can_fetch(normalized_url):
                    links.append(normalized_url)
                elif self.is_same_domain(normalized_url) and not self.can_fetch(normalized_url):
                    self.logger.debug(f"Robots.txt disallows: {normalized_url}")
                    
            except Exception as e:
                self.logger.debug(f"Error processing link {href}: {e}")
                continue
        
        return links

    def extract_text(self, soup: BeautifulSoup) -> str:
        """
        Extract visible text from BeautifulSoup object.
        
        Args:
            soup: BeautifulSoup object of the page
            
        Returns:
            Cleaned visible text
        """
        # Create a copy to avoid modifying the original
        soup_copy = BeautifulSoup(str(soup), 'html.parser')
        
        # Remove script and style elements
        for script in soup_copy(["script", "style", "meta", "link", "noscript"]):
            script.decompose()
        
        # Remove comments
        comments = soup_copy.find_all(string=lambda text: isinstance(text, Comment))
        for comment in comments:
            comment.extract()
        
        # First, try to extract from main content areas
        main_content = ""
        
        # Look for main content containers first
        main_selectors = [
            'main', '[role="main"]', '#main', '.main',
            'article', '.content', '#content', '.page-content',
            '.entry-content', '.post-content', '.article-content'
        ]
        
        for selector in main_selectors:
            main_elements = soup_copy.select(selector)
            if main_elements:
                for element in main_elements:
                    main_text = element.get_text(separator=' ', strip=True)
                    if len(main_text) > 100:  # Only use if substantial content
                        main_content += main_text + " "
                        self.logger.debug(f"Found main content in {selector}: {len(main_text)} chars")
        
        # If we found substantial main content, use it
        if len(main_content.strip()) > 100:
            text = main_content.strip()
        else:
            # Fall back to extracting from body, but be more conservative with removal
            
            # Only remove obvious navigation elements
            for nav in soup_copy(["nav"]):
                nav.decompose()
            
            # Remove only very specific navigation patterns to avoid over-filtering
            specific_nav_patterns = [
                {'class': re.compile(r'^nav$|^navigation$|^navbar$|^menu$', re.I)},
                {'id': re.compile(r'^nav$|^navigation$|^navbar$|^menu$', re.I)},
                {'class': re.compile(r'^header$|^site-header$', re.I)},
                {'id': re.compile(r'^header$|^site-header$', re.I)},
                {'class': re.compile(r'^footer$|^site-footer$', re.I)},
                {'id': re.compile(r'^footer$|^site-footer$', re.I)}
            ]
            
            for pattern in specific_nav_patterns:
                for element in soup_copy.find_all(attrs=pattern):
                    element.decompose()
            
            # Get all text
            text = soup_copy.get_text(separator=' ', strip=True)
        
        # Clean up whitespace but preserve structure
        # Split by multiple whitespace and rejoin with single spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Log the extraction result for debugging
        self.logger.debug(f"Extracted {len(text)} characters of text")
        if len(text) == 0:
            # If we got nothing, try a very basic extraction
            self.logger.warning("No text extracted, trying basic extraction")
            basic_soup = BeautifulSoup(str(soup), 'html.parser')
            for script in basic_soup(["script", "style"]):
                script.decompose()
            text = basic_soup.get_text(separator=' ', strip=True)
            text = re.sub(r'\s+', ' ', text).strip()
            self.logger.debug(f"Basic extraction yielded {len(text)} characters")
        
        return text

    def crawl_page(self, url: str) -> Optional[str]:
        """
        Crawl a single page and extract its content.
        
        Args:
            url: URL to crawl
            
        Returns:
            Extracted text content or None if failed
        """
        try:
            self.logger.info(f"Crawling: {url}")
            
            # Check robots.txt
            if not self.can_fetch(url):
                self.logger.info(f"Robots.txt disallows crawling: {url}")
                return None
            
            # Make request
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type:
                self.logger.debug(f"Skipping non-HTML content: {url} ({content_type})")
                return None
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract new links and add to queue (only if we haven't reached the limit)
            if self.limit is None or len(self.results) < self.limit:
                new_links = self.extract_links(soup, url)
                for link in new_links:
                    if link not in self.visited_urls and link not in self.url_queue:
                        self.url_queue.append(link)
                        self.logger.debug(f"Added to queue: {link}")
            else:
                self.logger.debug(f"Skipping link extraction - limit reached ({len(self.results)}/{self.limit})")
            
            # Extract text content
            text_content = self.extract_text(soup)
            
            # Debug: Save HTML if text extraction fails
            if self.debug_text_extraction and len(text_content) < 50:
                debug_file = f"debug_{urlparse(url).path.replace('/', '_') or 'index'}.html"
                with open(debug_file, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                self.logger.warning(f"Low text extraction for {url}. HTML saved to {debug_file}")
            
            self.logger.info(f"Successfully crawled: {url} ({len(text_content)} characters)")
            return text_content
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Request error for {url}: {e}"
            self.logger.error(error_msg)
            self.failed_urls.append((url, str(e)))
            return None
            
        except Exception as e:
            error_msg = f"Unexpected error crawling {url}: {e}"
            self.logger.error(error_msg)
            self.failed_urls.append((url, str(e)))
            return None

    def crawl(self) -> Dict[str, str]:
        """
        Main crawling method that processes the entire domain.
        
        Returns:
            Dictionary mapping URLs to their text content
        """
        self.logger.info(f"Starting crawl of domain: {self.domain}")
        
        # Load robots.txt
        self.load_robots_txt()
        
        # Process URLs from queue
        total_urls = len(self.url_queue)
        processed = 0
        
        if TQDM_AVAILABLE:
            pbar = tqdm(total=total_urls, desc="Crawling pages", unit="page")
        
        while self.url_queue:
            # Check if we've reached the limit
            if self.limit is not None and len(self.results) >= self.limit:
                self.logger.info(f"Reached crawl limit of {self.limit} pages")
                break
                
            url = self.url_queue.popleft()
            
            # Skip if already visited
            if url in self.visited_urls:
                continue
                
            # Mark as visited
            self.visited_urls.add(url)
            
            # Crawl the page
            text_content = self.crawl_page(url)
            if text_content:
                self.results[url] = text_content
                
                # Check if we've reached the limit after adding content
                if self.limit is not None and len(self.results) >= self.limit:
                    self.logger.info(f"Reached crawl limit of {self.limit} pages")
                    break
            
            processed += 1
            if TQDM_AVAILABLE:
                pbar.update(1)
                pbar.set_postfix({"Found": len(self.results), "Queue": len(self.url_queue), "Limit": self.limit or "∞"})
            
            # Rate limiting
            if self.delay > 0:
                time.sleep(self.delay)
        
        if TQDM_AVAILABLE:
            pbar.close()
        
        self.logger.info(f"Crawling complete. Visited {len(self.visited_urls)} pages, "
                        f"extracted content from {len(self.results)} pages")
        
        if self.failed_urls:
            self.logger.warning(f"Failed to crawl {len(self.failed_urls)} URLs")
        
        return self.results

    def save_results(self, output_file: str = "crawl_results"):
        """
        Save crawling results to JSON and CSV files.
        
        Args:
            output_file: Base filename for output files (without extension)
        """
        # Save as JSON
        json_file = f"{output_file}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                'domain': self.domain,
                'start_url': self.start_url,
                'total_pages': len(self.results),
                'failed_urls': len(self.failed_urls),
                'results': self.results,
                'failed': [{'url': url, 'error': error} for url, error in self.failed_urls]
            }, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Results saved to {json_file}")
        
        # Save as CSV
        csv_file = f"{output_file}.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['URL', 'Text_Length', 'Text_Content'])
            
            for url, text in self.results.items():
                writer.writerow([url, len(text), text])
        
        self.logger.info(f"Results saved to {csv_file}")
        
        # Save failed URLs if any
        if self.failed_urls:
            failed_file = f"{output_file}_failed.csv"
            with open(failed_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['URL', 'Error'])
                writer.writerows(self.failed_urls)
            
            self.logger.info(f"Failed URLs saved to {failed_file}")


def main():
    """Main function to run the crawler from command line."""
    parser = argparse.ArgumentParser(description='Crawl all pages within a domain')
    parser.add_argument('url', help='Starting URL to crawl')
    parser.add_argument('--delay', type=float, 
                       help='Delay between requests in seconds (default from .env or 1.0)')
    parser.add_argument('--output', default='crawl_results',
                       help='Output filename base (default: crawl_results)')
    parser.add_argument('--user-agent', 
                       help='User agent for robots.txt checking (default from .env or *)')
    parser.add_argument('--limit', type=int,
                       help='Maximum number of pages to crawl (default from .env or unlimited)')
    parser.add_argument('--no-robots', action='store_true', default=None,
                       help='Disable robots.txt checking')
    parser.add_argument('--disallow', nargs='+',
                       help='Custom paths to disallow (e.g., --disallow /about /admin)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate URL
    try:
        parsed = urlparse(args.url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("Invalid URL format")
    except Exception as e:
        print(f"Error: Invalid URL '{args.url}': {e}")
        sys.exit(1)
    
    # Load configuration from crawler.env file
    delay = args.delay
    if delay is None:
        delay = float(os.getenv('CRAWLER_DELAY', '1.0'))
    
    user_agent = args.user_agent
    if user_agent is None:
        user_agent = os.getenv('CRAWLER_USER_AGENT', '*')
    
    limit = args.limit
    if limit is None:
        limit_str = os.getenv('CRAWLER_LIMIT')
        limit = int(limit_str) if limit_str else None
    
    # Check if robots.txt should be disabled
    respect_robots = not args.no_robots
    if args.no_robots is None:  # If not specified via command line
        respect_robots_str = os.getenv('CRAWLER_RESPECT_ROBOTS', 'true')
        respect_robots = respect_robots_str.lower() == 'true'
    
    # Get custom disallow paths
    custom_disallow = args.disallow if args.disallow else []
    if not custom_disallow:  # If not specified via command line
        disallow_str = os.getenv('CRAWLER_DISALLOW', '')
        if disallow_str:
            custom_disallow = [path.strip() for path in disallow_str.split(',') if path.strip()]
    
    # Get output filename
    output_file = args.output
    if output_file == 'crawl_results':  # Default value
        output_file = os.getenv('CRAWLER_OUTPUT', 'crawl_results')
    
    # Check verbose mode
    verbose = args.verbose
    if not verbose:  # If not specified via command line
        verbose_str = os.getenv('CRAWLER_VERBOSE', 'false')
        verbose = verbose_str.lower() == 'true'
    
    # Create and run crawler
    try:
        crawler = DomainCrawler(
            start_url=args.url,
            delay=delay,
            user_agent=user_agent,
            debug_text_extraction=verbose,
            limit=limit,
            respect_robots=respect_robots,
            custom_disallow=custom_disallow
        )
        
        results = crawler.crawl()
        crawler.save_results(output_file)
        
        print(f"\nCrawling Summary:")
        print(f"Domain: {crawler.domain}")
        print(f"Total pages crawled: {len(results)}")
        print(f"Failed URLs: {len(crawler.failed_urls)}")
        print(f"Results saved to: {output_file}.json and {output_file}.csv")
        
    except KeyboardInterrupt:
        print("\nCrawling interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during crawling: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 