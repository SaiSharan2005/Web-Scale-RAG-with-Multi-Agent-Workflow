# Domain-Restricted Web Crawler

A powerful, efficient, and respectful Python web crawler designed to crawl all pages within a single domain while respecting robots.txt rules and extracting clean, structured text content.

## 🌟 Features

- **Domain-Restricted Crawling**: Only crawls pages within the specified domain
- **Robots.txt Compliance**: Automatically respects robots.txt rules
- **Intelligent Text Extraction**: Extracts clean, readable text content from web pages
- **Rate Limiting**: Configurable delays between requests to be respectful to servers
- **Error Handling**: Graceful handling of network errors and malformed pages
- **Multiple Output Formats**: Saves results in both JSON and CSV formats
- **Comprehensive Logging**: Detailed logging for debugging and monitoring
- **Link Discovery**: Automatically discovers and follows internal links
- **Content Filtering**: Removes navigation, headers, footers, and other non-content elements

## 📋 Requirements

- Python 3.7+
- Required packages (see installation section)

## 🚀 Installation

### Option 1: Using pip

```bash
# Clone the repository
git clone <repository-url>
cd web-crawler

# Install required packages
pip install -r requirements.txt
```

### Option 2: Manual Installation

```bash
# Install required packages
pip install requests beautifulsoup4 lxml
```

### Option 3: Using conda

```bash
conda install requests beautifulsoup4 lxml
```

## 📦 Dependencies

The crawler requires the following Python packages:

- `requests` - HTTP library for making requests
- `beautifulsoup4` - HTML parsing and text extraction
- `lxml` - XML/HTML parser (recommended for better performance)
- `urllib3` - HTTP client (included with requests)

## 🎯 Quick Start

### Basic Usage

```bash
# Crawl a website
python crawler.py https://example.com

# Crawl with custom delay
python crawler.py https://example.com --delay 2.0

# Crawl with verbose logging
python crawler.py https://example.com --verbose

# Specify output filename
python crawler.py https://example.com --output my_crawl_results
```

### Command Line Options

```bash
python crawler.py [URL] [OPTIONS]

Arguments:
  URL                     Starting URL to crawl

Options:
  --delay DELAY           Delay between requests in seconds (default: 1.0)
  --output OUTPUT         Output filename base (default: crawl_results)
  --user-agent USER_AGENT User agent for robots.txt checking (default: *)
  --verbose, -v           Enable verbose logging
  --help, -h              Show help message
```

## 🔧 Advanced Usage

### Programmatic Usage

```python
from crawler import DomainCrawler

# Initialize crawler
crawler = DomainCrawler(
    start_url="https://example.com",
    delay=1.0,
    user_agent="MyBot/1.0",
    debug_text_extraction=True
)

# Start crawling
results = crawler.crawl()

# Save results
crawler.save_results("my_results")

# Access results
for url, text in results.items():
    print(f"URL: {url}")
    print(f"Text length: {len(text)}")
    print(f"Preview: {text[:100]}...")
```

### Custom Configuration

```python
# Create crawler with custom settings
crawler = DomainCrawler(
    start_url="https://example.com",
    delay=2.0,  # 2 second delay between requests
    user_agent="MyCustomBot/1.0 (+https://example.com/bot)",
    debug_text_extraction=True  # Enable debug mode
)

# Crawl and get results
results = crawler.crawl()

# Check for failed URLs
if crawler.failed_urls:
    print(f"Failed to crawl {len(crawler.failed_urls)} URLs:")
    for url, error in crawler.failed_urls:
        print(f"  {url}: {error}")
```

## 📊 Output Formats

### JSON Output

The crawler saves results in a structured JSON format:

```json
{
  "domain": "example.com",
  "start_url": "https://example.com",
  "total_pages": 25,
  "failed_urls": 2,
  "results": {
    "https://example.com": "Extracted text content...",
    "https://example.com/about": "About page content...",
    "https://example.com/contact": "Contact information..."
  },
  "failed": [
    {
      "url": "https://example.com/private",
      "error": "403 Forbidden"
    }
  ]
}
```

### CSV Output

Results are also saved in CSV format with columns:
- `URL`: The crawled URL
- `Text_Length`: Number of characters in extracted text
- `Text_Content`: The full extracted text content

### Failed URLs

Failed URLs are saved in a separate CSV file with columns:
- `URL`: The URL that failed
- `Error`: The error message

## 🛠️ Configuration

### Environment Variables

You can set environment variables for default configuration:

```bash
export CRAWLER_DELAY=2.0
export CRAWLER_USER_AGENT="MyBot/1.0"
export CRAWLER_OUTPUT="my_crawl"
```

### Configuration File

Create a `crawler_config.json` file:

```json
{
  "default_delay": 1.5,
  "default_user_agent": "MyBot/1.0",
  "default_output": "crawl_results",
  "max_pages": 1000,
  "timeout": 30,
  "retry_attempts": 3
}
```

## 🔍 Text Extraction Features

### Intelligent Content Detection

The crawler uses multiple strategies to extract meaningful content:

1. **Main Content Areas**: Looks for semantic HTML elements like `<main>`, `<article>`, `.content`
2. **Navigation Filtering**: Removes navigation menus, headers, and footers
3. **Script/Style Removal**: Excludes JavaScript and CSS content
4. **Comment Removal**: Strips HTML comments
5. **Whitespace Normalization**: Cleans up excessive whitespace

### Content Filtering

The crawler automatically filters out:
- Navigation menus (`nav`, `.navigation`, `.navbar`)
- Headers and footers (`.header`, `.footer`)
- Script and style content
- HTML comments
- Common non-content elements

## 🚦 Rate Limiting and Respect

### Built-in Respect

- **Robots.txt Compliance**: Automatically reads and respects robots.txt
- **Configurable Delays**: Set delays between requests (default: 1 second)
- **User-Agent Identification**: Proper user agent string for identification
- **Error Handling**: Graceful handling of server errors

### Best Practices

```bash
# Be respectful with longer delays for large sites
python crawler.py https://example.com --delay 3.0

# Use a descriptive user agent
python crawler.py https://example.com --user-agent "MyResearchBot/1.0 (+https://example.com/bot)"
```

## 📈 Performance and Monitoring

### Logging Levels

- **INFO**: Default level with crawl progress
- **DEBUG**: Detailed information about text extraction
- **WARNING**: Non-critical issues
- **ERROR**: Failed requests and errors

### Monitoring Progress

```python
# Enable verbose logging
crawler = DomainCrawler(
    start_url="https://example.com",
    debug_text_extraction=True
)

# Monitor progress through logs
results = crawler.crawl()
```

## 🔧 Troubleshooting

### Common Issues

1. **Connection Errors**
   ```bash
   # Increase timeout
   python crawler.py https://example.com --delay 5.0
   ```

2. **Low Text Extraction**
   ```bash
   # Enable debug mode to see what's happening
   python crawler.py https://example.com --verbose
   ```

3. **Robots.txt Issues**
   ```bash
   # Check if robots.txt is blocking
   python crawler.py https://example.com --user-agent "*"
   ```

### Debug Mode

Enable debug mode to get detailed information:

```bash
python crawler.py https://example.com --verbose
```

This will:
- Save HTML files for pages with low text extraction
- Show detailed text extraction process
- Log all discovered links
- Provide more detailed error information

## 📝 Examples

### Example 1: Basic Website Crawl

```bash
# Crawl a simple website
python crawler.py https://example.com
```

### Example 2: Large Site with Respectful Crawling

```bash
# Crawl a large site with longer delays
python crawler.py https://large-site.com --delay 3.0 --output large_site_crawl
```

### Example 3: Research Project

```bash
# Crawl for research with custom user agent
python crawler.py https://research-site.com \
  --user-agent "ResearchBot/1.0 (+https://university.edu/bot)" \
  --delay 2.0 \
  --output research_data \
  --verbose
```

### Example 4: Programmatic Usage

```python
from crawler import DomainCrawler
import json

# Initialize crawler
crawler = DomainCrawler(
    start_url="https://example.com",
    delay=1.5,
    user_agent="MyBot/1.0"
)

# Crawl the site
results = crawler.crawl()

# Process results
for url, text in results.items():
    # Do something with the text
    word_count = len(text.split())
    print(f"{url}: {word_count} words")

# Save results
crawler.save_results("processed_results")

# Check for errors
if crawler.failed_urls:
    print(f"Failed URLs: {len(crawler.failed_urls)}")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This crawler is designed for educational and research purposes. Always:

- Respect robots.txt files
- Use appropriate delays between requests
- Follow the website's terms of service
- Don't overload servers with too many requests
- Use a descriptive user agent string

## 🆘 Support

If you encounter any issues:

1. Check the troubleshooting section
2. Enable verbose logging with `--verbose`
3. Check the logs for detailed error information
4. Open an issue on GitHub with:
   - The command you ran
   - The error message
   - The URL you were trying to crawl
   - Your Python version

## 📚 Additional Resources

- [Pinecone Vector Database Integration](Pinecone.py) - Vector storage for crawled content
- [BeautifulSoup Documentation](https://www.crummy.com/software/BeautifulSoup/)
- [Requests Documentation](https://requests.readthedocs.io/)
- [Robots.txt Specification](https://www.robotstxt.org/)

---

**Happy Crawling! 🕷️** 
