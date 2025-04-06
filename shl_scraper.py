import requests
from bs4 import BeautifulSoup
import json
import csv
import time
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("shl_scraper.log"),
        logging.StreamHandler()
    ]
)

BASE_URL = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/"
HEADERS = {
    'User-Agent': 'Mozilla/5.0',
    'Accept-Language': 'en-US, en;q=0.5'
}

def is_yes(td_tag):
    return td_tag.find("span", class_="catalogue__circle -yes") is not None

def extract_detail_page_data(detail_url):
    try:
        logging.info(f"Fetching detail page: {detail_url}")
        detail_res = requests.get(detail_url, headers=HEADERS)
        detail_soup = BeautifulSoup(detail_res.content, "html.parser")

        def get_section_text(heading):
            h4 = detail_soup.find("h4", string=lambda s: s and heading.lower() in s.lower())
            if h4:
                p = h4.find_next_sibling("p")
                return p.get_text(strip=True) if p else ""
            return ""

        desc_div = detail_soup.find("div", class_="product-catalogue-training-calendar__row typ")
        description = desc_div.find("p").get_text(strip=True) if desc_div else "Description not found"

        return {
            "Description": description,
            "Job Levels": get_section_text("Job Levels"),
            "Languages": get_section_text("Languages"),
            "Assessment Length": get_section_text("Assessment length")
        }

    except Exception as e:
        logging.error(f"Error fetching detail page {detail_url}: {e}")
        return {
            "Description": f"Error fetching: {e}",
            "Job Levels": "",
            "Languages": "",
            "Assessment Length": ""
        }

def scrape_products(product_type, max_items=400):
    """
    Scrape products of a specific type (1 for Individual Test Solutions, 2 for Pre-packaged Job Solutions)
    """
    logging.info(f"Starting to scrape {product_type} products")
    products = []
    
    # We assume max items, incrementing 12 per page
    for start in range(0, max_items, 12):
        page_url = f"{CATALOG_URL}?start={start}&type={product_type}" if start > 0 else f"{CATALOG_URL}?type={product_type}"
        logging.info(f"Fetching page: {page_url}")
        
        try:
            res = requests.get(page_url, headers=HEADERS)
            soup = BeautifulSoup(res.content, "html.parser")

            # Find the table containing the product type
            table_header = soup.find("th", string="Individual Test Solutions" if product_type == 1 else "Pre-packaged Job Solutions")
            if not table_header:
                logging.info(f"No more pages found for product type {product_type}")
                break  # No more pages

            # Get the <table> containing this <th>
            table_tag = table_header.find_parent("table")
            if not table_tag:
                logging.warning(f"Table not found for product type {product_type}")
                continue

            rows = table_tag.find_all("tr")[1:]  # Skip header row
            
            if not rows:
                logging.info(f"No rows found on page for product type {product_type}")
                break

            for row in rows:
                cols = row.find_all("td")
                if len(cols) < 4:
                    continue

                title_tag = cols[0].find("a")
                if not title_tag:
                    continue

                title = title_tag.get_text(strip=True)
                relative_link = title_tag["href"]
                full_link = BASE_URL + relative_link

                # Scrape data from the detail page
                detail_data = extract_detail_page_data(full_link)
                
                # Add a small delay to be respectful to the server
                time.sleep(1)

                product_data = {
                    "Product Type": "Individual Test Solutions" if product_type == 1 else "Pre-packaged Job Solutions",
                    "Title": title,
                    "Remote Testing": "Yes" if is_yes(cols[1]) else "No",
                    "Adaptive/IRT": "Yes" if is_yes(cols[2]) else "No",
                    "Test Type": cols[3].get_text(strip=True),
                    "Description": detail_data["Description"],
                    "Job Levels": detail_data["Job Levels"],
                    "Languages": detail_data["Languages"],
                    "Assessment Length": detail_data["Assessment Length"],
                    "Detail Page": full_link
                }
                
                products.append(product_data)
                logging.info(f"Added product: {title}")
            
            # Add a delay between pages
            time.sleep(2)
            
        except Exception as e:
            logging.error(f"Error processing page {page_url}: {e}")
    
    logging.info(f"Finished scraping {len(products)} {product_type} products")
    return products

def main():
    all_products = []
    
    # Scrape Individual Test Solutions (type=1)
    individual_products = scrape_products(1)
    all_products.extend(individual_products)
    
    # Scrape Pre-packaged Job Solutions (type=2)
    package_products = scrape_products(2)
    all_products.extend(package_products)
    
    # Save to files
    logging.info(f"Saving {len(all_products)} products to JSON and CSV files")
    
    with open("shl_products.json", "w", encoding="utf-8") as jf:
        json.dump(all_products, jf, indent=4)

    with open("shl_products.csv", "w", newline='', encoding='utf-8') as cf:
        writer = csv.DictWriter(cf, fieldnames=all_products[0].keys())
        writer.writeheader()
        writer.writerows(all_products)
    
    logging.info("Scraping completed successfully")

if __name__ == "__main__":
    main()
