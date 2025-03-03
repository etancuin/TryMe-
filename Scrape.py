import requests
import time
import random
from typing import Dict, List, Optional
from dataclasses import dataclass
import re
import backoff

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import googlemaps

import hashlib
import psycopg2
from contextlib import contextmanager
import logging

@dataclass
class Restaurant:
    name: str
    address: str
    phone: Optional[str] = None
    website: Optional[str] = None
    cuisine: Optional[List[str]] = None
    price_range: Optional[str] = None
    rating: Optional[float] = None
    hours: Optional[Dict[str, str]] = None
    source: str = None
    
    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'address': self.address,
            'phone': self.phone,
            'website': self.website,
            'cuisine': self.cuisine,
            'price_range': self.price_range,
            'rating': self.rating,
            'hours': self.hours,
            'source': self.source
        }
    
    def get_hash(self) -> str:
        """Create a unique hash based on name and address"""
        string_to_hash = f"{self.name.lower()}{self.address.lower()}".encode('utf-8')
        return hashlib.md5(string_to_hash).hexdigest()


class DatabaseManager:
    def __init__(self, db_config: Dict[str, str]):
        """
        Initialize database connection with configuration
        
        db_config should contain: {
            'dbname': 'your_db_name',
            'user': 'your_user',
            'password': 'your_password',
            'host': 'your_host',
            'port': 'your_port'
        }
        """
        self.db_config = db_config
        self.setup_database()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = psycopg2.connect(**self.db_config)
            yield conn
        except Exception as e:
            logging.error(f"Database connection error: {e}")
            raise
        finally:
            if conn is not None:
                conn.close()

    def setup_database(self):
        """Set up database and required tables"""
        # First, connect to default database to create our database if it doesn't exist
        temp_config = self.db_config.copy()
        temp_config['dbname'] = 'postgres'
        
        try:
            with psycopg2.connect(**temp_config) as conn:
                conn.autocommit = True
                with conn.cursor() as cur:
                    # Check if database exists
                    cur.execute(
                        "SELECT 1 FROM pg_database WHERE datname = %s",
                        (self.db_config['dbname'],)
                    )
                    if not cur.fetchone():
                        # Create database if it doesn't exist
                        cur.execute(f"CREATE DATABASE {self.db_config['dbname']}")
        
        except Exception as e:
            logging.error(f"Error setting up database: {e}")
            raise

        # Now create tables in our database
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Create restaurants table
                    cur.execute('''
                        CREATE TABLE IF NOT EXISTS restaurants (
                            id SERIAL PRIMARY KEY,
                            hash TEXT UNIQUE,
                            name TEXT NOT NULL,
                            address TEXT NOT NULL,
                            phone TEXT,
                            website TEXT,
                            cuisine JSONB,
                            price_range TEXT,
                            rating REAL,
                            hours JSONB,
                            source TEXT,
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                        )
                    ''')
                    
                    # Create index on hash
                    cur.execute('''
                        CREATE INDEX IF NOT EXISTS idx_restaurants_hash 
                        ON restaurants(hash)
                    ''')
                    
                    # Create trigger for updated_at
                    cur.execute('''
                        CREATE OR REPLACE FUNCTION update_updated_at_column()
                        RETURNS TRIGGER AS $$
                        BEGIN
                            NEW.updated_at = CURRENT_TIMESTAMP;
                            RETURN NEW;
                        END;
                        $$ language 'plpgsql'
                    ''')
                    
                    # Create trigger if it doesn't exist
                    cur.execute('''
                        DO $$
                        BEGIN
                            IF NOT EXISTS (
                                SELECT 1 
                                FROM pg_trigger 
                                WHERE tgname = 'update_restaurants_updated_at'
                            ) THEN
                                CREATE TRIGGER update_restaurants_updated_at
                                    BEFORE UPDATE ON restaurants
                                    FOR EACH ROW
                                    EXECUTE FUNCTION update_updated_at_column();
                            END IF;
                        END;
                        $$
                    ''')
                    
                    conn.commit()
        
        except Exception as e:
            logging.error(f"Error creating tables: {e}")
            raise

class RestaurantScraper:
    def __init__(self, city: str, google_api_key: str, db_manager: DatabaseManager):
        self.city = city
        self.db_manager = db_manager
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.gmaps = googlemaps.Client(key=google_api_key)
        self.driver = None
        self.max_retries = 3
        self.base_delay = 1
        self.restaurants: List[Restaurant] = []
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename='scraper.log'
        )
        
    def setup_selenium(self):
        """Initialize Selenium WebDriver with error handling"""
        try:
            options = webdriver.ChromeOptions()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument(f'user-agent={self.headers["User-Agent"]}')
            self.driver = webdriver.Chrome(options=options)
            self.driver.set_page_load_timeout(30)
        except WebDriverException as e:
            logging.error(f"Selenium setup error: {e}")
            raise

    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, TimeoutException),
        max_tries=3
    )
    def make_request(self, url: str) -> requests.Response:
        """Make HTTP request with exponential backoff retry logic"""
        time.sleep(self.base_delay + random.uniform(0, 2))
        response = requests.get(url, headers=self.headers, timeout=10)
        response.raise_for_status()
        return response

    def clean_phone_number(self, phone: str) -> str:
        """Standardize phone number format"""
        if not phone:
            return None
        digits = re.sub(r'\D', '', phone)
        if len(digits) == 10:
            return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
        return phone

    def clean_address(self, address: str) -> str:
        """Standardize address format"""
        if not address:
            return None
        # Remove extra whitespace and normalize commas
        address = re.sub(r'\s+', ' ', address).strip()
        address = re.sub(r'\s*,\s*', ', ', address)
        return address

    def scrape_google_maps(self):
        """Scrape restaurant data using Google Maps Places API"""
        try:
            location = self.gmaps.geocode(self.city)[0]['geometry']['location']
            restaurants_result = self.gmaps.places_nearby(
                location=location,
                radius=5000,
                type='restaurant'
            )
            
            while True:
                for place in restaurants_result.get('results', []):
                    try:
                        place_details = self.gmaps.place(
                            place['place_id'],
                            fields=['name', 'formatted_address', 'formatted_phone_number',
                                'website', 'opening_hours', 'price_level', 'rating', 'types']
                        )
                        
                        details = place_details['result']
                        restaurant = Restaurant(
                            name=details.get('name'),
                            address=self.clean_address(details.get('formatted_address')),
                            phone=self.clean_phone_number(details.get('formatted_phone_number')),
                            website=details.get('website'),
                            cuisine=[t for t in details.get('types', []) 
                                if t not in ['restaurant', 'food', 'establishment']],
                            price_range='$' * details.get('price_level', 0),
                            rating=details.get('rating'),
                            hours=details.get('opening_hours', {}).get('weekday_text', []),
                            source='Google Maps'
                        )
                        
                        self.db_manager.insert_restaurant(restaurant)
                        time.sleep(2)
                        
                    except Exception as e:
                        logging.error(f"Error processing Google Maps place: {e}")
                        continue
                
                if 'next_page_token' in restaurants_result:
                    time.sleep(2)
                    restaurants_result = self.gmaps.places_nearby(
                        location=location,
                        radius=5000,
                        type='restaurant',
                        page_token=restaurants_result['next_page_token']
                    )
                else:
                    break
                    
        except Exception as e:
            logging.error(f"Error scraping Google Maps: {e}")

    def scrape_yelp(self):
        """Scrape restaurant data from Yelp"""
        if not self.driver:
            self.setup_selenium()
            
        base_url = f"https://www.yelp.com/search?find_desc=restaurants&find_loc={self.city}"
        
        try:
            self.driver.get(base_url)
            
            while True:
                restaurant_elements = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_all_elements_located(
                        (By.CSS_SELECTOR, "div.business-card")
                    )
                )
                
                for element in restaurant_elements:
                    try:
                        restaurant = Restaurant(
                            name=self._safe_extract_selenium("h3.business-name"),
                            address=self.clean_address(
                                self._safe_extract_selenium("address.business-address")
                            ),
                            phone=self.clean_phone_number(
                                self._safe_extract_selenium("span.business-phone")
                            ),
                            website=self._safe_extract_selenium("a.business-website"),
                            cuisine=self._get_yelp_categories(),
                            price_range=self._safe_extract_selenium("span.price-range"),
                            rating=self._get_yelp_rating(),
                            hours=self._get_hours_selenium(),
                            source='Yelp'
                        )
                        
                        self.db_manager.insert_restaurant(restaurant)
                        time.sleep(random.uniform(1, 3))
                        
                    except Exception as e:
                        logging.error(f"Error processing Yelp restaurant: {e}")
                        continue
                
                if not self._next_page():
                    break
                    
        except Exception as e:
            logging.error(f"Error scraping Yelp: {e}")

    def scrape_tripadvisor(self):
        """Scrape restaurant data from TripAdvisor"""
        if not self.driver:
            self.setup_selenium()
            
        base_url = f"https://www.tripadvisor.com/Restaurants-{self.city}"
        
        try:
            self.driver.get(base_url)
            
            while True:
                restaurant_elements = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_all_elements_located(
                        (By.CSS_SELECTOR, "div[data-test='restaurant-item']")
                    )
                )
                
                for element in restaurant_elements:
                    try:
                        restaurant = Restaurant(
                            name=self._safe_extract_selenium("h1.restaurant-name"),
                            address=self.clean_address(
                                self._safe_extract_selenium("span.address")
                            ),
                            phone=self.clean_phone_number(
                                self._safe_extract_selenium("span.phone")
                            ),
                            website=self._safe_extract_selenium("a.website"),
                            cuisine=self._get_tripadvisor_categories(),
                            price_range=self._safe_extract_selenium("span.price-range"),
                            rating=self._get_tripadvisor_rating(),
                            hours=self._get_hours_selenium(),
                            source='TripAdvisor'
                        )
                        
                        self.db_manager.insert_restaurant(restaurant)
                        time.sleep(random.uniform(1, 3))
                        
                    except Exception as e:
                        logging.error(f"Error processing TripAdvisor restaurant: {e}")
                        continue
                
                if not self._next_page():
                    break
                    
        except Exception as e:
            logging.error(f"Error scraping TripAdvisor: {e}")

    def _safe_extract_selenium(self, selector: str) -> Optional[str]:
        """Safely extract data from Selenium elements"""
        try:
            element = WebDriverWait(self.driver, 5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, selector))
            )
            return element.text.strip()
        except:
            return None

    def _next_page(self) -> bool:
        """Handle pagination with error checking"""
        try:
            next_button = self.driver.find_element(By.CSS_SELECTOR, "a.next")
            if not next_button.is_enabled():
                return False
            next_button.click()
            time.sleep(2)
            return True
        except:
            return False

    def _get_hours_selenium(self) -> Dict[str, str]:
        """Extract business hours using Selenium"""
        hours = {}
        try:
            hours_elements = self.driver.find_elements(By.CSS_SELECTOR, "div.hours-row")
            for element in hours_elements:
                day = element.find_element(By.CSS_SELECTOR, "span.day").text
                time = element.find_element(By.CSS_SELECTOR, "span.time").text
                hours[day] = time
        except Exception as e:
            logging.debug(f"Error extracting hours: {e}")
        return hours

    def cleanup(self):
        """Clean up resources"""
        if self.driver:
            self.driver.quit()

def main():
    # Configuration
    GOOGLE_API_KEY = "your_api_key_here"
    city = "San Francisco, CA"
    db_path = "restaurants.db"
    
    # Initialize database and scraper
    db_manager = DatabaseManager(db_path)
    scraper = RestaurantScraper(city, GOOGLE_API_KEY, db_manager)
    
    try:
        # Scrape from all sources
        scraper.scrape_google_maps()
        scraper.scrape_yelp()
        scraper.scrape_tripadvisor()
        
    except Exception as e:
        logging.error(f"Main scraping error: {e}")
        
    finally:
        scraper.cleanup()

if __name__ == "__main__":
    main()