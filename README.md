# SHL Product Catalog Scraper

This Python script extracts assessment data from SHL's product catalog, focusing on Pre-packaged Job Solutions.

## Features

- Extracts comprehensive data from SHL's product catalog
- Handles pagination to navigate through all pages
- Accesses detailed description pages for each assessment
- Extracts assessment name, URL, remote testing status, adaptive/IRT support, duration, test type, and detailed descriptions
- Stores results in both JSON and CSV formats
- Includes error handling, retry logic, and logging

## Requirements

- Python 3.8+
- Chrome browser installed
- ChromeDriver installed and accessible in PATH

## Installation

1. Clone this repository:

```
git clone <repository-url>
cd shl-scraper
```

2. Install the required dependencies:

```
pip install -r requirements.txt
```

3. Make sure you have Chrome and ChromeDriver installed.

## Usage

Simply run the main script:

```
python shl_scraper.py
```

The script will:

1. Navigate to SHL's product catalog
2. Extract data from Pre-packaged Job Solutions section
3. Handle pagination through all pages
4. Visit each assessment's detail page
5. Save the extracted data in both JSON and CSV formats

## Output

- `output/shl_assessments.json`: JSON file containing all extracted data
- `output/shl_assessments.csv`: CSV file with the same data for easier analysis
- `shl_scraper.log`: Log file with detailed information about the scraping process

## Customization

You can modify the following constants in the script to customize the behavior:

- `MAX_RETRIES`: Number of retry attempts for failed operations
- `WAIT_TIME`: Maximum wait time for page elements to load (in seconds)
- `REQUEST_DELAY`: Delay between requests to respect the website (in seconds)

## Disclaimer

This script is for educational purposes only. Make sure to respect SHL's terms of service and website policies when using it. Use responsibly and ethically with appropriate delays between requests.
