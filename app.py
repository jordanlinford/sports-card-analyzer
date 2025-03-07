import subprocess
subprocess.run(["pip", "show", "opencv-python"])
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from ebaysdk.finding import Connection as Finding
import logging
from PIL import Image
import io
from scipy import stats
import time
import plotly.graph_objects as go
import os
import json
import shutil
from typing import Dict, Any, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('brown')
from textblob import TextBlob
import random
import plotly.express as px
from functools import wraps
import threading
import re
import unicodedata
import cv2

# Your remaining Streamlit code...

# Configure logging
logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# API Configuration
EBAY_CONFIG = {
    'app_id': 'JordanLi-SportsCa-PRD-0dec1ac47-a1385cd5',
    'dev_id': '889db8e2-9d05-4405-848f-1bea7f0c4f52',
    'cert_id': 'PRD-dec1ac475699-2f77-4bf0-b699-f386'
}

GOOGLE_CONFIG = {
    'api_key': 'AIzaSyCpfrr3sXeMzHZ08kr62H3UX3_SGHikJLU',
    'search_engine_id': '264fee7c07fbc428e'
}

# Add cache configuration at the top of your file
st.cache_data.clear()

# Rate limiter for eBay API
_EBAY_LAST_CALL = 0
_EBAY_LOCK = threading.Lock()
_MAX_RETRIES = 3
_BACKOFF_FACTOR = 2
_MIN_INTERVAL = 5.0
_MAX_DAILY_CALLS = 5000
_DAILY_CALL_COUNT = 0
_LAST_RESET = time.time()
_LOCK_TIMEOUT = 30

def _load_daily_stats():
    try:
        stats_file = 'ebay_api_stats.json'
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                stats = json.load(f)
                if stats.get('date') == datetime.now().strftime('%Y-%m-%d'):
                    return stats.get('calls', 0), stats.get('last_reset', time.time())
    except Exception as e:
        logger.warning(f"Error loading API stats: {e}")
    return 0, time.time()

def _save_daily_stats(calls, last_reset):
    try:
        stats_file = 'ebay_api_stats.json'
        with open(stats_file, 'w') as f:
            json.dump({
                'date': datetime.now().strftime('%Y-%m-%d'),
                'calls': calls,
                'last_reset': last_reset,
                'last_call': time.time()
            }, f)
    except Exception as e:
        logger.warning(f"Error saving API stats: {e}")

def rate_limit_ebay(min_interval=_MIN_INTERVAL):
    """Decorator to rate limit eBay API calls with improved exponential backoff and request tracking"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            global _EBAY_LAST_CALL, _DAILY_CALL_COUNT, _LAST_RESET
            
            # Load persisted stats
            _DAILY_CALL_COUNT, _LAST_RESET = _load_daily_stats()
            
            lock_acquired = False
            try:
                # Try to acquire lock with timeout
                lock_acquired = _EBAY_LOCK.acquire(timeout=_LOCK_TIMEOUT)
                if not lock_acquired:
                    logger.error("Failed to acquire rate limit lock")
                    return None
                
                current_time = time.time()
                
                # Reset daily counter if 24 hours have passed
                if current_time - _LAST_RESET >= 86400:
                    _DAILY_CALL_COUNT = 0
                    _LAST_RESET = current_time
                    _save_daily_stats(_DAILY_CALL_COUNT, _LAST_RESET)
                
                # Check daily limit with buffer
                if _DAILY_CALL_COUNT >= (_MAX_DAILY_CALLS * 0.95):  # 95% of limit
                    logger.error("Approaching daily API call limit")
                    return None
                
                for attempt in range(_MAX_RETRIES):
                    time_since_last_call = current_time - _EBAY_LAST_CALL
                    
                    # Calculate dynamic backoff time
                    backoff_time = min_interval * (_BACKOFF_FACTOR ** attempt)
                    jitter = random.uniform(0, 1)  # Add randomness to prevent thundering herd
                    sleep_time = max(0, backoff_time - time_since_last_call + jitter)
                    
                    if sleep_time > 0:
                        logger.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds (attempt {attempt + 1})")
                        time.sleep(sleep_time)
                    
                    try:
                        _EBAY_LAST_CALL = time.time()
                        _DAILY_CALL_COUNT += 1
                        _save_daily_stats(_DAILY_CALL_COUNT, _LAST_RESET)
                        
                        # Execute the API call
                        result = func(*args, **kwargs)
                        
                        # If successful, return immediately
                        return result
                        
                    except Exception as e:
                        if "Service call has exceeded" in str(e):
                            if attempt < _MAX_RETRIES - 1:
                                # Increase backoff for next attempt
                                sleep_time = min_interval * (_BACKOFF_FACTOR ** (attempt + 1))
                                logger.warning(f"Rate limit hit, backing off for {sleep_time:.2f} seconds")
                                time.sleep(sleep_time)
                                continue
                            else:
                                logger.error("Max retries exceeded for rate limit")
                        else:
                            logger.error(f"Unexpected error in API call: {str(e)}")
                        raise
                
                return None
                
            finally:
                if lock_acquired:
                    _EBAY_LOCK.release()
                    
        return wrapper
    return decorator

@st.cache_data(ttl=3600)  # Cache for 1 hour
@rate_limit_ebay()  # Use default interval
def get_ebay_sales(query, condition="raw"):
    """Fetch eBay sales data with enhanced error handling and consistent data format"""
    try:
        # Construct search query based on condition
        search_query = query
        if condition.lower() == "raw":
            search_query += " -PSA -BGS -SGC"  # Exclude graded cards
        else:
            search_query += f" {condition.upper()}"  # Add grade to search
            
        logger.info(f"Attempting eBay search for {condition} with query: {search_query}")
        
        try:
            api = Finding(appid=EBAY_CONFIG['app_id'], config_file=None)
            response = api.execute('findItemsAdvanced', {
                'keywords': search_query,
                'itemFilter': [
                    {'name': 'SoldItemsOnly', 'value': 'true'},
                    {'name': 'EndTimeFrom', 'value': (datetime.now() - timedelta(days=90)).isoformat()}
                ],
                'paginationInput': {'entriesPerPage': '50', 'pageNumber': '1'},  # Reduced from 100 to 50
                'sortOrder': 'EndTimeSoonest',
                'outputSelector': ['SellerInfo', 'StoreInfo', 'PictureURLLarge']
            })
            
            if hasattr(response.reply, 'searchResult') and hasattr(response.reply.searchResult, 'item'):
                items = response.reply.searchResult.item
                
                # Process items into DataFrame
                data = []
                for item in items:
                    try:
                        if hasattr(item, 'sellingStatus') and hasattr(item.sellingStatus, 'currentPrice'):
                            price = float(item.sellingStatus.currentPrice.value)
                            date = pd.to_datetime(item.listingInfo.endTime)
                            title = item.title
                            link = item.viewItemURL
                            
                            data.append({
                                'Date': date,
                                'Price': price,
                                'Title': title,
                                'Link': link,
                                'Condition': condition
                            })
                    except (AttributeError, ValueError) as e:
                        logger.warning(f"Error processing item: {str(e)}")
                        continue
                
                if data:
                    df = pd.DataFrame(data)
                    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')  # Convert to numeric, replacing errors with NaN
                    df = df.dropna(subset=['Price'])  # Remove rows with NaN prices
                    df['Price'] = df['Price'].astype(float)  # Convert to float
                    df = df.sort_values('Date', ascending=False)  # Sort by date
                    return df
                
        except Exception as e:
            if "Service call has exceeded" in str(e):
                logger.error("eBay API rate limit exceeded, falling back to dummy data")
            else:
                logger.error(f"eBay API error: {str(e)}, falling back to dummy data")
        
        # If we get here, either the API call failed or no results were found
        logger.info(f"Generating dummy data for {condition}")
        return generate_dummy_sales(query, condition)
        
    except Exception as e:
        logger.error(f"Error in data fetching: {str(e)}")
        return generate_dummy_sales(query, condition)

def generate_dummy_sales(query, condition="raw", base_price=100):
    """Generate realistic dummy sales data with consistent format"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        
        # Generate random number of sales (between 15-30) over 90 days
        num_sales = np.random.randint(15, 30)
        
        # Generate random dates
        dates = sorted([
            start_date + timedelta(days=np.random.randint(0, 90))
            for _ in range(num_sales)
        ])
        
        # Adjust price based on condition and add trend/volatility
        if condition.lower() == "raw":
            price_mean = base_price
            price_std = base_price * 0.1
            trend_factor = 0.001  # Slight upward trend
        elif condition.lower() == "psa 9":
            price_mean = base_price * 3
            price_std = base_price * 0.15
            trend_factor = 0.002  # Moderate upward trend
        else:  # PSA 10
            price_mean = base_price * 5
            price_std = base_price * 0.2
            trend_factor = 0.003  # Stronger upward trend
        
        # Generate prices with trend and randomness
        prices = []
        for i, date in enumerate(dates):
            # Add trend
            trend = price_mean * (1 + trend_factor * i)
            # Add randomness
            price = max(10, np.random.normal(trend, price_std))
            prices.append(price)
        
        # Create DataFrame with consistent column names
        df = pd.DataFrame({
            'Date': dates,
            'Price': prices,
            'Title': [f"{query} {condition} Sale {i+1}" for i in range(len(dates))],
            'Link': ['#'] * len(dates)
        })
        
        # Ensure price is float and sort by date
        df['Price'] = df['Price'].astype(float)
        return df.sort_values('Date', ascending=False)
        
    except Exception as e:
        logger.error(f"Error generating dummy data: {str(e)}")
        # Return minimal valid DataFrame
        return pd.DataFrame({
            'Date': [datetime.now()],
            'Price': [float(base_price)],
            'Title': [f"{query} {condition}"],
            'Link': ['#']
        })

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_image(query):
    """Fetch card image using Google Custom Search API with enhanced error handling and caching"""
    try:
        # Validate configuration
        if not GOOGLE_CONFIG.get('api_key') or not GOOGLE_CONFIG.get('search_engine_id'):
            logger.error("Google API configuration missing")
            return None

        # Extract card details
        card_number = None
        year = None
        set_name = None
        player_name = None
        parallel = None
        manufacturer = None
        
        # Enhanced card number patterns
        card_number_patterns = [
            r'#(\d+)',                    # Standard #123 format
            r'(?<!\d)(\d{1,3})(?=\s|$)', # Standalone numbers
            r'number\s*(\d+)',           # "number 123" format
            r'(?:card|no)[.\s]*(\d+)',   # "card 123" or "no. 123" format
            r'(?<=\s)(\d{1,3})(?=\s|$)'  # Space-separated number
        ]
        
        for pattern in card_number_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                card_number = match.group(1)
                break

        # Enhanced year extraction
        year_patterns = [
            r'(?:19|20)(\d{2})',     # Full year (1900s or 2000s)
            r"'(\d{2})",             # Apostrophe year
            r'(?<!\d)(\d{2})(?!\d)'  # Standalone 2-digit year
        ]
        
        for pattern in year_patterns:
            match = re.search(pattern, query)
            if match:
                year_digits = match.group(1)
                year = f"20{year_digits}" if int(year_digits) < 50 else f"19{year_digits}"
                break

        # Enhanced set name and manufacturer detection
        set_patterns = {
            'prizm': ['prizm', 'prism'],
            'topps': ['topps', 'bowman', 'heritage', 'chrome', 'update'],
            'panini': ['donruss', 'select', 'optic', 'mosaic', 'chronicles'],
            'upper deck': ['upper deck', 'spx', 'ud'],
            'fleer': ['fleer', 'ultra', 'flair'],
            'score': ['score']
        }
        
        query_lower = query.lower()
        for set_name_key, variations in set_patterns.items():
            if any(var in query_lower for var in variations):
                set_name = set_name_key
                # Set manufacturer based on set name
                if set_name_key in ['prizm', 'donruss', 'select', 'optic', 'mosaic']:
                    manufacturer = 'panini'
                elif set_name_key in ['topps', 'bowman']:
                    manufacturer = 'topps'
                break

        # Parallel/variation detection
        parallel_patterns = [
            'refractor', 'parallel', 'prizm', 'wave', 'mosaic', 'holo',
            'gold', 'silver', 'blue', 'red', 'green', 'purple', 'orange',
            'atomic', 'ice', 'cracked ice'
        ]
        
        for pattern in parallel_patterns:
            if pattern in query_lower:
                parallel = pattern
                break

        # Enhanced player name extraction
        # Remove card number, year, and set name portions first
        name_query = query
        if card_number:
            name_query = re.sub(r'#?\d+', '', name_query)
        if year:
            name_query = re.sub(r'(?:19|20)\d{2}', '', name_query)
        if set_name:
            name_query = re.sub(re.escape(set_name), '', name_query, flags=re.IGNORECASE)
        
        # Extract first two words that look like a name
        name_parts = [
            word for word in name_query.split()
            if word.isalpha() and len(word) > 1 and not word.lower() in ['card', 'rookie', 'rc']
        ][:2]
        
        if name_parts:
            player_name = ' '.join(name_parts)

        # Clean and normalize the query
        query = unicodedata.normalize('NFKD', query).encode('ASCII', 'ignore').decode('ASCII')
        
        # Enhanced search strategies with better scoring
        search_strategies = [
            # Strategy 1: Most specific with all details
            {
                'terms': [
                    f'"{player_name}"' if player_name else "",
                    f'#{card_number}' if card_number else "",
                    year if year else "",
                    set_name if set_name else "",
                    parallel if parallel else "",
                    manufacturer if manufacturer else "",
                    "sports card",
                    "-reprint",
                    "-lot",
                    "-case",
                    "-box"
                ],
                'weight': 2.0
            },
            # Strategy 2: Focus on card number and set
            {
                'terms': [
                    f'"{player_name}"' if player_name else "",
                    set_name if set_name else "",
                    f'#{card_number}' if card_number else "",
                    year if year else "",
                    "card",
                    "-reprint",
                    "-lot"
                ],
                'weight': 1.5
            },
            # Strategy 3: Manufacturer and player focus
            {
                'terms': [
                    f'"{player_name}"' if player_name else "",
                    manufacturer if manufacturer else "",
                    "sports card",
                    year if year else "",
                    "-reprint"
                ],
                'weight': 1.2
            },
            # Strategy 4: Original query as fallback
            {
                'terms': [
                    f'"{query}"',
                    "sports card",
                    "trading card",
                    "-reprint"
                ],
                'weight': 1.0
            }
        ]

        all_scored_images = []
        
        for strategy in search_strategies:
            try:
                # Filter out empty terms and join
                search_terms = [term for term in strategy['terms'] if term]
                enhanced_query = " ".join(search_terms)
                
                url = "https://www.googleapis.com/customsearch/v1"
                params = {
                    'key': GOOGLE_CONFIG['api_key'],
                    'cx': GOOGLE_CONFIG['search_engine_id'],
                    'q': enhanced_query,
                    'searchType': 'image',
                    'num': 10,
                    'imgSize': 'LARGE',
                    'imgType': 'photo',
                    'safe': 'active'
                }

                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                if 'items' in data:
                    for item in data['items']:
                        try:
                            score = 0
                            title = item.get('title', '').lower()
                            snippet = item.get('snippet', '').lower()
                            full_text = f"{title} {snippet}"
                            
                            # Enhanced scoring based on matching criteria
                            if card_number and f"#{card_number}" in full_text:
                                score += 15
                            elif card_number and card_number in full_text:
                                score += 10
                                
                            if year and year in full_text:
                                score += 8
                                
                            if set_name and set_name.lower() in full_text:
                                score += 12
                                
                            if manufacturer and manufacturer.lower() in full_text:
                                score += 5
                                
                            if parallel and parallel.lower() in full_text:
                                score += 10
                                
                            if player_name and player_name.lower() in full_text:
                                score += 15
                                
                            # Negative scoring for unwanted terms
                            unwanted_terms = ['reprint', 'lot', 'case', 'box', 'wrapper', 'pack', 'sealed']
                            for term in unwanted_terms:
                                if term in full_text:
                                    score -= 20
                            
                            # Bonus for card-specific terms
                            card_terms = ['card', 'trading card', 'sports card']
                            for term in card_terms:
                                if term in full_text:
                                    score += 5
                            
                            # Penalize auction site watermarks if not from primary sources
                            if any(site in full_text.lower() for site in ['ebay', 'pwcc', 'comc']) and 'stock' not in full_text:
                                score -= 3
                            
                            # Adjust score based on strategy weight
                            score *= strategy['weight']
                            
                            # Image quality scoring
                            image_data = item.get('image', {})
                            width = image_data.get('width', 0)
                            height = image_data.get('height', 0)
                            
                            # Prefer images with good dimensions
                            if width >= 300 and height >= 400:
                                score += 10
                            elif width >= 200 and height >= 300:
                                score += 5
                            
                            # Penalize very small or very large images
                            if width < 200 or height < 300:
                                score -= 10
                            if width > 2000 or height > 2000:
                                score -= 5
                            
                            if score > 0:  # Only keep images with positive scores
                                all_scored_images.append({
                                    'url': item['link'],
                                    'score': score,
                                    'strategy': search_terms[0],
                                    'metadata': {
                                        'width': width,
                                        'height': height
                                    }
                                })
                                
                        except Exception as e:
                            logger.warning(f"Error processing image result: {e}")
                            continue

            except Exception as e:
                logger.warning(f"Search strategy failed: {e}")
                continue

        # Sort by score and validate images
        if all_scored_images:
            all_scored_images.sort(key=lambda x: x['score'], reverse=True)
            
            # Try each image until we find a valid one
            for img in all_scored_images[:5]:  # Try top 5 images
                try:
                    img_response = requests.get(img['url'], timeout=5)
                    if img_response.status_code == 200:
                        img_data = Image.open(io.BytesIO(img_response.content))
                        width, height = img_data.size
                        
                        if width >= 200 and height >= 300:
                            logger.info(
                                f"Selected image with score {img['score']} from strategy: {img['strategy']}\n"
                                f"Dimensions: {width}x{height}"
                            )
                            return img['url']
                            
                except Exception as e:
                    logger.warning(f"Failed to validate image {img['url']}: {e}")
                    continue

        # If no good images found, try eBay as fallback
        try:
            api = Finding(appid=EBAY_CONFIG['app_id'], config_file=None)
            
            # Construct optimized eBay search
            ebay_query_parts = []
            if player_name:
                ebay_query_parts.append(f'"{player_name}"')
            if year:
                ebay_query_parts.append(year)
            if set_name:
                ebay_query_parts.append(set_name)
            if card_number:
                ebay_query_parts.append(f"#{card_number}")
            if parallel:
                ebay_query_parts.append(parallel)
            
            # Add card type if we can determine it
            if 'baseball' in query_lower or 'topps' in query_lower:
                ebay_query_parts.append('baseball card')
            elif 'basketball' in query_lower or 'prizm' in query_lower:
                ebay_query_parts.append('basketball card')
            else:
                ebay_query_parts.append('sports card')
            
            ebay_query = ' '.join(ebay_query_parts)
            
            response = api.execute('findItemsAdvanced', {
                'keywords': ebay_query,
                'itemFilter': [
                    {'name': 'SoldItemsOnly', 'value': 'true'}
                ],
                'outputSelector': ['PictureURLSuperSize', 'PictureURLLarge'],
                'paginationInput': {'entriesPerPage': '5'}  # Try top 5 results
            })
            
            if hasattr(response.reply, 'searchResult') and hasattr(response.reply.searchResult, 'item'):
                for item in response.reply.searchResult.item[:5]:
                    if hasattr(item, 'pictureURLSuperSize'):
                        return item.pictureURLSuperSize
                    elif hasattr(item, 'pictureURLLarge'):
                        return item.pictureURLLarge
                    
        except Exception as e:
            logger.warning(f"eBay API fallback failed: {e}")

        logger.warning(f"No valid image found for query: {query}")
        return "https://via.placeholder.com/300x400.png?text=Card+Image+Not+Found"

    except Exception as e:
        logger.error(f"Critical error in fetch_image: {e}")
        return "https://via.placeholder.com/300x400.png?text=Error+Loading+Image"

def get_cached_ebay_sold_prices(query):
    """Mock function for eBay sales data"""
    return pd.DataFrame({
        'Date': pd.date_range(end=pd.Timestamp.now(), periods=3),
        'Price': np.random.uniform(50, 500, 3)
    })

def clean_prices(df):
    """Clean and format price data"""
    if df.empty:
        return pd.DataFrame(columns=['Date', 'Price'])
    return df.head(3)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def build_card_database():
    """
    Build a database of card features using Google Custom Search results.
    Caches the results for 1 hour to avoid excessive API calls.
    """
    try:
        # List of popular sports cards to seed the database
        seed_cards = [
            "Tom Brady Rookie Card Bowman Chrome",
            "Patrick Mahomes Rookie Card Prizm",
            "Justin Herbert Rookie Card Prizm",
            "Joe Burrow Rookie Card Prizm",
            "Trevor Lawrence Rookie Card Prizm",
            "Josh Allen Rookie Card Prizm",
            "Justin Jefferson Rookie Card Prizm",
            "Ja'Marr Chase Rookie Card Prizm",
            "Mac Jones Rookie Card Prizm",
            "Jayson Tatum Rookie Card Prizm"
        ]
        
        card_database = {}
        
        for query in seed_cards:
            try:
                # Fetch image URL using existing fetch_image function
                image_url = fetch_image(query)
                if not image_url:
                    continue
                
                # Download and process the image
                response = requests.get(image_url)
                if response.status_code != 200:
                    continue
                
                # Convert to numpy array
                nparr = np.frombuffer(response.content, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                
                # Process image features (similar to search function)
                # Resize image
                img = cv2.resize(img, (224, 224))
                
                # Convert to different color spaces
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                
                # Extract features
                hist_bins = 16
                color_features = []
                
                # BGR histogram
                for i in range(3):
                    hist = cv2.calcHist([img], [i], None, [hist_bins], [0, 256])
                    hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
                    color_features.extend(hist)
                
                # HSV histogram
                for i in range(3):
                    hist = cv2.calcHist([hsv], [i], None, [hist_bins], [0, 256])
                    hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
                    color_features.extend(hist)
                
                # Edge features
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
                
                # Shape features
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                shape_features = []
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(largest_contour)
                    perimeter = cv2.arcLength(largest_contour, True)
                    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                    shape_features = [area / (img.shape[0] * img.shape[1]), circularity]
                else:
                    shape_features = [0, 0]
                
                # Combine features
                features = np.concatenate([
                    np.array(color_features, dtype=np.float32),
                    np.array([edge_density], dtype=np.float32),
                    np.array(shape_features, dtype=np.float32)
                ])
                
                # Store in database
                card_id = f"card_{len(card_database)}"
                card_database[card_id] = {
                    'features': features,
                    'name': query.split(" Rookie")[0],
                    'set': "Prizm" if "Prizm" in query else "Bowman Chrome",
                    'year': "2020",  # You could parse this from the image or query
                    'image_url': image_url
                }
                
            except Exception as e:
                logger.warning(f"Error processing card {query}: {str(e)}")
                continue
        
        return card_database
        
    except Exception as e:
        logger.error(f"Error building card database: {str(e)}")
        return {}

def load_card_database():
    """
    Load the card database with real card data from Google Custom Search.
    """
    return build_card_database()

def get_card_details(card_id):
    """
    Get details for a specific card from the database.
    """
    card_database = load_card_database()
    card_data = card_database.get(card_id, {})
    if card_data:
        # Get real market data from eBay
        query = f"{card_data['name']} rookie {card_data['set']} {card_data['year']}"
        try:
            sales_data = get_ebay_sales(query, "raw")
            if not sales_data.empty:
                last_sale = f"${sales_data['Price'].iloc[0]:,.2f}"
                market_price = f"${sales_data['Price'].mean():,.2f}"
            else:
                last_sale = "N/A"
                market_price = "N/A"
        except Exception as e:
            logger.warning(f"Error fetching market data: {str(e)}")
            last_sale = "N/A"
            market_price = "N/A"
            
        return {
            'name': card_data['name'],
            'set': card_data['set'],
            'year': card_data['year'],
            'last_sale': last_sale,
            'market_price': market_price,
            'image_url': card_data['image_url']
        }
    return None

def get_cached_search_by_image_results(image_bytes):
    """
    Process an uploaded image and find similar trading cards using advanced image processing.
    Uses a real card database built from Google Custom Search results.
    """
    try:
        # Convert image bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            logger.error("Failed to decode image")
            return pd.DataFrame()
            
        # Process image (same as before)
        img = cv2.resize(img, (224, 224))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Extract features
        hist_bins = 16
        color_features = []
        
        # BGR histogram
        for i in range(3):
            hist = cv2.calcHist([img], [i], None, [hist_bins], [0, 256])
            hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
            color_features.extend(hist)
        
        # HSV histogram
        for i in range(3):
            hist = cv2.calcHist([hsv], [i], None, [hist_bins], [0, 256])
            hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
            color_features.extend(hist)
        
        # Edge features
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Shape features
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            shape_features = [area / (img.shape[0] * img.shape[1]), circularity]
        else:
            shape_features = [0, 0]
        
        # Combine features
        query_features = np.concatenate([
            np.array(color_features, dtype=np.float32),
            np.array([edge_density], dtype=np.float32),
            np.array(shape_features, dtype=np.float32)
        ])
        
        # Load card database
        card_database = load_card_database()
        
        # Find similar cards
        similar_cards = []
        for card_id, card_data in card_database.items():
            try:
                card_features = card_data['features']
                
                # Calculate similarity scores
                color_sim = 1 - cv2.compareHist(
                    query_features[:hist_bins*6],
                    card_features[:hist_bins*6],
                    cv2.HISTCMP_INTERSECT
                ) / hist_bins
                
                edge_sim = 1 - abs(query_features[-3] - card_features[-3])
                shape_sim = 1 - np.linalg.norm(query_features[-2:] - card_features[-2:])
                
                # Weighted similarity score
                similarity = (
                    color_sim * 0.5 +
                    edge_sim * 0.3 +
                    shape_sim * 0.2
                )
                
                # Calculate confidence score
                confidence_score = similarity
                if edge_density < 0.01:
                    confidence_score *= 0.8
                if np.std(query_features[:hist_bins*6]) < 0.1:
                    confidence_score *= 0.8
                
                similar_cards.append({
                    'card_id': card_id,
                    'similarity': similarity,
                    'confidence_score': confidence_score
                })
                
            except Exception as e:
                logger.warning(f"Error comparing with card {card_id}: {str(e)}")
                continue
        
        # Sort by similarity
        similar_cards.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Get top matches
        top_matches = similar_cards[:10]
        
        # Convert to DataFrame
        results = []
        for match in top_matches:
            card_details = get_card_details(match['card_id'])
            if card_details:
                confidence_level = 'High' if match['confidence_score'] > 0.8 else \
                                 'Medium' if match['confidence_score'] > 0.6 else 'Low'
                
                results.append({
                    'Card Name': card_details['name'],
                    'Set': card_details['set'],
                    'Year': card_details['year'],
                    'Similarity': f"{match['similarity']:.1%}",
                    'Confidence': confidence_level,
                    'Last Sale': card_details['last_sale'],
                    'Market Price': card_details['market_price'],
                    'Image URL': card_details['image_url']
                })
        
        return pd.DataFrame(results)
        
    except Exception as e:
        logger.error(f"Error in image search: {str(e)}")
        return pd.DataFrame()

def calculate_metrics(sales_data):
    """Calculate market metrics from sales data"""
    try:
        if sales_data is None or sales_data.empty:
            return {
                'avg': 0.0,
                'median': 0.0,
                'min': 0.0,
                'max': 0.0,
                'std': 0.0,
                'volatility': 0.0,
                'trend': 0.0,
                'volume': 0,
                'market_score': 0.0,
                'liquidity': 0.0,
                'volatility_metrics': {
                    'short_term': 0.0,
                    'long_term': 0.0,
                    'risk_adjusted_return': 0.0
                },
                'trend_metrics': {
                    'short_term': 0.0,
                    'medium_term': 0.0,
                    'long_term': 0.0,
                    'strength': 0.0,
                    'reliability': 0.0,
                    'cycle_position': 'unknown'
                }
            }

        # Ensure we have a Price column and Date column
        if 'Price' not in sales_data.columns:
            if 'Sale Price' in sales_data.columns:
                sales_data['Price'] = sales_data['Sale Price'].astype(float)
            else:
                raise KeyError("No Price or Sale Price column found in sales data")
        
        if 'Date' not in sales_data.columns:
            raise KeyError("No Date column found in sales data")

        # Convert dates and sort by date
        sales_data['Date'] = pd.to_datetime(sales_data['Date'])
        sales_data = sales_data.sort_values('Date')
        prices = sales_data['Price'].astype(float)

        # Calculate basic statistics
        metrics = {
            'avg': float(prices.mean()),
            'median': float(prices.median()),
            'min': float(prices.min()),
            'max': float(prices.max()),
            'std': float(prices.std()) if len(prices) > 1 else 0.0,
            'volume': len(prices)
        }

        # Calculate liquidity score (0-100)
        try:
            # Calculate days between first and last sale
            date_range = (sales_data['Date'].max() - sales_data['Date'].min()).days
            if date_range == 0:  # If all sales are on the same day
                date_range = 1
            
            # Calculate average sales per day
            sales_per_day = len(sales_data) / date_range
            
            # Calculate consistency of sales
            daily_sales = sales_data.groupby(sales_data['Date'].dt.date).size()
            sales_consistency = len(daily_sales) / date_range  # Ratio of days with sales
            
            # Calculate price stability (inverse of coefficient of variation)
            price_stability = 1 - (metrics['std'] / metrics['avg']) if metrics['avg'] > 0 else 0
            
            # Calculate recent activity (last 7 days)
            recent_date = sales_data['Date'].max()
            week_ago = recent_date - pd.Timedelta(days=7)
            recent_sales = len(sales_data[sales_data['Date'] >= week_ago])
            recent_activity = min(1.0, recent_sales / 7)  # Cap at 1.0
            
            # Combine factors into liquidity score
            liquidity_score = (
                (min(1.0, sales_per_day) * 30) +     # Up to 30 points for frequency (normalized)
                (sales_consistency * 30) +            # Up to 30 points for consistency
                (price_stability * 20) +              # Up to 20 points for price stability
                (recent_activity * 20)                # Up to 20 points for recent activity
            )
            
            metrics['liquidity'] = float(min(100, max(0, liquidity_score)))
            
        except Exception as e:
            logger.warning(f"Error calculating liquidity: {str(e)}")
            metrics['liquidity'] = 0.0
        
        # Enhanced Volatility Calculations
        if len(prices) > 1:
            try:
                # Calculate returns
                returns = np.diff(prices) / prices[:-1]
                
                # Short-term volatility (7-day)
                if len(returns) >= 7:
                    short_term_vol = np.std(returns[-7:]) * np.sqrt(252) * 100
                else:
                    short_term_vol = np.std(returns) * np.sqrt(252) * 100
                
                # Long-term volatility (30-day or all available data)
                long_term_vol = np.std(returns) * np.sqrt(252) * 100
                
                # Calculate annualized return
                total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
                days_held = (sales_data['Date'].max() - sales_data['Date'].min()).days
                annualized_return = ((1 + total_return) ** (365 / max(days_held, 1)) - 1) * 100
                
                # Calculate risk-adjusted return (Sharpe Ratio like metric)
                risk_free_rate = 0.04  # Assume 4% risk-free rate
                risk_adjusted_return = (annualized_return - risk_free_rate) / long_term_vol if long_term_vol > 0 else 0
                
                # Store volatility metrics
                metrics['volatility_metrics'] = {
                    'short_term': float(short_term_vol),
                    'long_term': float(long_term_vol),
                    'risk_adjusted_return': float(risk_adjusted_return)
                }
                
                # Overall volatility score (weighted average of short and long term, normalized to 0-100)
                volatility_score = (short_term_vol * 0.7 + long_term_vol * 0.3)
                metrics['volatility'] = float(min(100, volatility_score))  # Cap at 100
                
            except Exception as e:
                logger.warning(f"Error calculating enhanced volatility metrics: {str(e)}")
                metrics['volatility'] = (metrics['std'] / metrics['avg'] * 100) if metrics['avg'] > 0 else 0.0
                metrics['volatility_metrics'] = {
                    'short_term': 0.0,
                    'long_term': 0.0,
                    'risk_adjusted_return': 0.0
                }
        else:
            metrics['volatility'] = 0.0
            metrics['volatility_metrics'] = {
                'short_term': 0.0,
                'long_term': 0.0,
                'risk_adjusted_return': 0.0
            }
        
        # Calculate comprehensive market trends
        if len(prices) > 1:
            try:
                recent_date = sales_data['Date'].max()
                
                # Define timeframes for trend analysis
                timeframes = {
                    'short': 7,    # 7 days
                    'medium': 30,  # 30 days
                    'long': 90     # 90 days
                }
                
                trend_metrics = {}
                moving_averages = {}
                
                # Calculate trends for each timeframe
                for period, days in timeframes.items():
                    period_start = recent_date - pd.Timedelta(days=days)
                    period_data = sales_data[sales_data['Date'] >= period_start]
                    
                    if not period_data.empty and len(period_data) > 1:
                        # Calculate price trend
                        x = np.arange(len(period_data))
                        y = period_data['Price'].values
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                        
                        # Normalize slope to percentage
                        avg_price = np.mean(y)
                        normalized_slope = (slope / avg_price) * 100 if avg_price > 0 else 0
                        
                        # Calculate moving average
                        ma_window = min(5, len(period_data))
                        moving_avg = period_data['Price'].rolling(window=ma_window).mean()
                        
                        # Store metrics
                        trend_metrics[f'{period}_term'] = float(normalized_slope)
                        moving_averages[period] = float(moving_avg.iloc[-1]) if not moving_avg.empty else float(y[-1])
                
                # Calculate trend strength (0-100)
                if 'short_term' in trend_metrics and 'medium_term' in trend_metrics:
                    # Normalize trend values to 0-100 scale
                    short_term_score = min(100, max(0, trend_metrics['short_term'] + 50))  # Center around 50
                    medium_term_score = min(100, max(0, trend_metrics['medium_term'] + 50))
                    long_term_score = min(100, max(0, trend_metrics.get('long_term', 0) + 50))
                    
                    trend_strength = (
                        short_term_score * 0.5 +    # 50% weight to short-term
                        medium_term_score * 0.3 +   # 30% weight to medium-term
                        long_term_score * 0.2       # 20% weight to long-term
                    )
                else:
                    trend_strength = 50  # Neutral if not enough data
                
                # Calculate trend reliability (0-100)
                if len(prices) >= 3:
                    # Use R-squared value from regression
                    x = np.arange(len(prices))
                    _, _, r_value, _, _ = stats.linregress(x, prices)
                    trend_reliability = float(min(100, max(0, r_value * 100)))
                else:
                    trend_reliability = 0
                
                # Detect market cycle position
                if len(prices) >= 5:
                    recent_trend = trend_metrics.get('short_term', 0)
                    recent_ma = moving_averages.get('short', prices.iloc[-1])
                    long_ma = moving_averages.get('long', prices.mean())
                    
                    if recent_trend > 0 and prices.iloc[-1] > recent_ma > long_ma:
                        cycle_position = 'uptrend'
                    elif recent_trend < 0 and prices.iloc[-1] < recent_ma < long_ma:
                        cycle_position = 'downtrend'
                    elif recent_trend > 0 and prices.iloc[-1] < long_ma:
                        cycle_position = 'accumulation'
                    elif recent_trend < 0 and prices.iloc[-1] > long_ma:
                        cycle_position = 'distribution'
                    else:
                        cycle_position = 'sideways'
                else:
                    cycle_position = 'unknown'
                
                # Store trend metrics
                metrics['trend_metrics'] = {
                    'short_term': trend_metrics.get('short_term', 0.0),
                    'medium_term': trend_metrics.get('medium_term', 0.0),
                    'long_term': trend_metrics.get('long_term', 0.0),
                    'strength': trend_strength,
                    'reliability': trend_reliability,
                    'cycle_position': cycle_position
                }
                
                # Calculate trend impact (0-100)
                trend_impact = (
                    trend_strength * 0.6 +          # 60% weight to trend strength
                    trend_reliability * 0.4         # 40% weight to trend reliability
                )
                
                # Calculate final market score (0-100)
                market_score = (
                    (trend_impact * 0.4) +                # Trend components: 40% weight
                    (metrics['liquidity'] * 0.3) +        # Liquidity: 30% weight
                    ((100 - metrics['volatility']) * 0.3) # Inverse volatility: 30% weight
                )
                
                metrics['market_score'] = float(min(100, max(0, market_score)))
                metrics['trend'] = trend_metrics.get('medium_term', 0.0)
                
            except Exception as e:
                logger.error(f"Error calculating market trends: {str(e)}")
                metrics.update({
                    'market_score': 0.0,
                    'trend': 0.0,
                    'trend_metrics': {
                        'short_term': 0.0,
                        'medium_term': 0.0,
                        'long_term': 0.0,
                        'strength': 0.0,
                        'reliability': 0.0,
                        'cycle_position': 'unknown'
                    }
                })
        else:
            metrics.update({
                'market_score': 0.0,
                'trend': 0.0,
                'trend_metrics': {
                    'short_term': 0.0,
                    'medium_term': 0.0,
                    'long_term': 0.0,
                    'strength': 0.0,
                    'reliability': 0.0,
                    'cycle_position': 'unknown'
                }
            })
            
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        return {
            'avg': 0.0,
            'median': 0.0,
            'min': 0.0,
            'max': 0.0,
            'std': 0.0,
            'volatility': 0.0,
            'trend': 0.0,
            'volume': 0,
            'market_score': 0.0,
            'liquidity': 0.0,
            'volatility_metrics': {
                'short_term': 0.0,
                'long_term': 0.0,
                'risk_adjusted_return': 0.0
            },
            'trend_metrics': {
                'short_term': 0.0,
                'medium_term': 0.0,
                'long_term': 0.0,
                'strength': 0.0,
                'reliability': 0.0,
                'cycle_position': 'unknown'
            }
        }

def get_sales_data(query):
    """Fetch and process sales data for a given query"""
    try:
        # Initialize empty DataFrames with correct columns
        empty_df = pd.DataFrame(columns=['Date', 'Price', 'Title', 'Condition', 'Link'])
        raw_sales = empty_df.copy()
        psa9_sales = empty_df.copy()
        psa10_sales = empty_df.copy()
        metrics = {}
        img_url = None

        # Fetch card image
        with st.spinner('Fetching card image...'):
            img_url = fetch_image(query)
            if img_url:
                st.image(img_url, width=200)

        # Fetch raw sales data
        with st.spinner('Fetching raw sales data...'):
            raw_sales = get_ebay_sales(query, condition="raw")
            if raw_sales is not None and not raw_sales.empty:
                raw_sales = clean_prices(raw_sales)
                metrics['raw'] = calculate_metrics(raw_sales)

        # Fetch PSA 9 sales data
        with st.spinner('Fetching PSA 9 sales data...'):
            psa9_sales = get_ebay_sales(query, condition="psa 9")
            if psa9_sales is not None and not psa9_sales.empty:
                psa9_sales = clean_prices(psa9_sales)
                metrics['psa9'] = calculate_metrics(psa9_sales)

        # Fetch PSA 10 sales data
        with st.spinner('Fetching PSA 10 sales data...'):
            psa10_sales = get_ebay_sales(query, condition="psa 10")
            if psa10_sales is not None and not psa10_sales.empty:
                psa10_sales = clean_prices(psa10_sales)
                metrics['psa10'] = calculate_metrics(psa10_sales)

        return {
            'raw_sales': raw_sales,
            'psa9_sales': psa9_sales,
            'psa10_sales': psa10_sales,
            'metrics': metrics,
            'img_url': img_url
        }

    except Exception as e:
        logging.error(f"Error in get_sales_data: {str(e)}")
        st.error("Error fetching sales data. Please try again.")
        return None

def run_text_search(query):
    """Run a text search for a card and display results"""
    try:
        # Clear previous search results from session state
        if 'search_metrics' in st.session_state:
            st.session_state.search_metrics = {}
        
        # Part 1: Data Fetching
        with st.spinner('Fetching card image...'):
            img_url = fetch_image(query)
            if img_url:
                try:
                    st.image(img_url, caption=query, use_container_width=True)
                except Exception as e:
                    logger.error(f"Error displaying image from URL {img_url}: {e}")
                    st.warning("Could not display card image, but continuing with market analysis...")
            else:
                st.warning("No card image found")
        
        with st.spinner('Analyzing market data...'):
            # Get sales data with fallback to dummy data
            raw_sales = get_ebay_sales(query, "raw")
            time.sleep(1)  # Rate limiting
            psa9_sales = get_ebay_sales(query, "psa 9")
            time.sleep(1)  # Rate limiting
            psa10_sales = get_ebay_sales(query, "psa 10")
            
            # Calculate metrics for each condition
            metrics = {}
            for grade, sales in [("Raw", raw_sales), ("PSA 9", psa9_sales), ("PSA 10", psa10_sales)]:
                if not sales.empty:
                    metrics[grade] = calculate_metrics(sales)
            
            # Store results in session state
            st.session_state.search_metrics = metrics
            st.session_state.raw_sales = raw_sales
            st.session_state.psa9_sales = psa9_sales
            st.session_state.psa10_sales = psa10_sales
        
        # Part 2: Display Analysis
        raw_tab, psa9_tab, psa10_tab = st.tabs(["Raw", "PSA 9", "PSA 10"])
        
        with raw_tab:
            st.subheader("Raw Analysis")
            if not raw_sales.empty:
                display_market_analysis(raw_sales, metrics.get("Raw", {}), f"{query} (Raw)")
            else:
                st.warning("No raw sales data available")
        
        with psa9_tab:
            st.subheader("PSA 9 Analysis")
            if not psa9_sales.empty:
                display_market_analysis(psa9_sales, metrics.get("PSA 9", {}), f"{query} (PSA 9)")
            else:
                st.warning("No PSA 9 sales data available")
        
        with psa10_tab:
            st.subheader("PSA 10 Analysis")
            if not psa10_sales.empty:
                display_market_analysis(psa10_sales, metrics.get("PSA 10", {}), f"{query} (PSA 10)")
            else:
                st.warning("No PSA 10 sales data available")
        
        # Part 3: Add to Collection
        with st.expander("➕ Add to Collection"):
            col1, col2 = st.columns(2)
            with col1:
                player_name = st.text_input("Player Name", key="add_player_name")
            with col2:
                card_desc = st.text_input("Card Description", key="add_card_desc")
            
            if st.button("Add to Collection", key="add_to_collection_button"):
                if player_name and card_desc:
                    # Calculate trend scores
                    trend_scores = {}
                    for grade in ['Raw', 'PSA 9', 'PSA 10']:
                        if grade in metrics:
                            trend_scores[grade] = '📈' if metrics[grade]['avg'] > metrics[grade].get('min', 0) else '📉'
                    
                    result = add_to_collection(
                        player_name, 
                        card_desc, 
                        metrics, 
                        0,  # No purchase price in collection add
                        trend_scores
                    )
                    if result is not None:
                        st.success("Card added to collection!")
                else:
                    st.error("Please enter both player name and card description")
        
        # Part 4: Standalone Profit Analysis
        with st.expander("💰 Profit Calculator", expanded=True):
            purchase_price = st.number_input(
                "Enter Purchase Price ($):",
                min_value=0.0,
                step=0.01,
                key=f"profit_calc_price_{query}"
            )
            
            if purchase_price > 0:
                display_profit_analysis(metrics, purchase_price)
            else:
                st.info("Enter a purchase price above to calculate potential profits")
            
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        st.error(f"Error in analysis: {str(e)}")

class CardMarketAnalyzer:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
    def prepare_features(self, df):
        """Prepare features for price prediction"""
        df['Date'] = pd.to_datetime(df['Date'])
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['DayOfMonth'] = df['Date'].dt.day
        df['Month'] = df['Date'].dt.month
        df['DaysSinceStart'] = (df['Date'] - df['Date'].min()).dt.days
        return df
    
    def predict_prices(self, historical_data, days_to_predict=30):
        """Predict future prices using machine learning"""
        try:
            if historical_data.empty:
                return None, None, 0
            
            # Prepare data
            df = self.prepare_features(historical_data.copy())
            
            # Prepare features for training
            X = df[['DayOfWeek', 'DayOfMonth', 'Month', 'DaysSinceStart']].values
            y = df['Price'].values
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            
            # Generate future dates
            last_date = df['Date'].max()
            future_dates = pd.date_range(start=last_date, periods=days_to_predict + 1)[1:]
            
            # Create future features
            future_df = pd.DataFrame({'Date': future_dates})
            future_df = self.prepare_features(future_df)
            future_X = self.scaler.transform(
                future_df[['DayOfWeek', 'DayOfMonth', 'Month', 'DaysSinceStart']].values
            )
            
            # Make predictions
            predictions = self.model.predict(future_X)
            
            # Calculate confidence intervals
            predictions_std = np.std([
                tree.predict(future_X) for tree in self.model.estimators_
            ], axis=0)
            
            confidence_score = max(0.5, min(0.9, self.model.score(X_scaled, y)))
            
            return predictions, predictions_std, confidence_score
            
        except Exception as e:
            logger.error(f"Error in price prediction: {e}")
            return None, None, 0
    
    def analyze_sentiment(self, card_name, recent_prices):
        """Analyze market sentiment for a card"""
        try:
            # Calculate price trends
            price_change = 0
            if len(recent_prices) > 1:
                price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] * 100
            
            # Simulate social media sentiment (replace with actual API calls in production)
            sentiment_score = np.random.normal(0.2, 0.3)  # Slightly positive bias
            
            # Adjust sentiment based on price trends
            if price_change > 10:
                sentiment_score += 0.2
            elif price_change < -10:
                sentiment_score -= 0.2
            
            # Generate market signals
            signals = {
                'price_trend': 'Bullish' if price_change > 5 else 'Bearish' if price_change < -5 else 'Neutral',
                'sentiment': 'Positive' if sentiment_score > 0.2 else 'Negative' if sentiment_score < -0.2 else 'Neutral',
                'price_change': price_change,
                'confidence': min(1.0, max(0.0, 0.5 + abs(sentiment_score))),
                'recommendation': self._generate_recommendation(price_change, sentiment_score)
            }
            
            return signals
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return None
    
    def _generate_recommendation(self, price_change, sentiment):
        """Generate trading recommendation based on price change and sentiment"""
        if price_change > 15 and sentiment > 0.3:
            return "Strong Sell - Price spike with positive sentiment suggests peak"
        elif price_change > 10 and sentiment > 0:
            return "Consider Selling - Positive momentum but watch for reversal"
        elif price_change < -15 and sentiment < -0.3:
            return "Strong Buy - Price drop with negative sentiment suggests bottom"
        elif price_change < -10 and sentiment < 0:
            return "Consider Buying - Market may be oversold"
        else:
            return "Hold - No strong signals"

def display_predictions_and_alerts(sales_data, card_name=""):
    """Display price predictions and market alerts with enhanced analysis"""
    try:
        # Generate unique timestamp for keys
        timestamp = int(time.time() * 1000)
        
        # Validate input data
        if not isinstance(sales_data, pd.DataFrame):
            st.warning("Invalid sales data format")
            return
            
        if sales_data.empty:
            st.warning("Insufficient data for predictions")
            return
            
        # Prepare and validate data
        filtered_data = None
        try:
            filtered_data = sales_data.copy()
            filtered_data['Price'] = pd.to_numeric(filtered_data['Price'], errors='coerce')
            filtered_data = filtered_data.dropna(subset=['Price'])
            
            if filtered_data.empty:
                st.warning("No valid price data available for analysis")
                return
                
            if len(filtered_data) < 2:
                st.warning("Insufficient data points for trend analysis")
                return
        except Exception as e:
            logging.error(f"Error preparing data: {str(e)}")
            st.error("Error processing sales data")
            return
            
        # Initialize market analyzer and calculate metrics
        try:
            analyzer = CardMarketAnalyzer()
            
            # Calculate current price metrics
            current_price = filtered_data['Price'].iloc[-1]
            week_ago_idx = -min(7, len(filtered_data))
            week_ago_price = filtered_data['Price'].iloc[week_ago_idx]
            price_change = ((current_price - week_ago_price) / week_ago_price * 100)
            
            # Check for significant price movements
            if abs(price_change) >= 15:
                alert_color = "green" if price_change > 0 else "red"
                st.markdown(
                    f"""<div style='padding: 10px; border-radius: 5px; background-color: rgba({255 if price_change < 0 else 0}, {255 if price_change > 0 else 0}, 0, 0.1)'>
                    🚨 <strong>Market Alert:</strong> {card_name} has {'increased' if price_change > 0 else 'decreased'} by {abs(price_change):.1f}% in the past {min(7, len(filtered_data))} days.<br>
                    {'Consider selling based on current market conditions.' if price_change > 0 else 'Potential buying opportunity.'}
                    </div>""",
                    unsafe_allow_html=True
                )
            
            # Display current price and weekly change
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Current Price",
                    f"${current_price:.2f}",
                    f"{price_change:+.1f}% ({min(7, len(filtered_data))}d)",
                    key=f"price_metric_{timestamp}"
                )
            
            # Calculate market sentiment
            try:
                sentiment = analyzer.analyze_sentiment(
                    card_name,
                    filtered_data['Price'].values
                )
                
                if sentiment and isinstance(sentiment, dict):
                    st.subheader("📊 Market Analysis")
                    
                    # Display sentiment metrics in columns
                    cols = st.columns(3)
                    
                    # Price Trend
                    with cols[0]:
                        trend = sentiment.get('price_trend', 'Neutral')
                        change = sentiment.get('price_change', 0)
                        color = (
                            'green' if trend == 'Bullish'
                            else 'red' if trend == 'Bearish'
                            else 'orange'
                        )
                        st.markdown(
                            f"""
                            **Price Trend:**
                            <p style='color: {color}; font-size: 20px;'>
                            {trend} ({change:+.1f}%)
                            </p>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    # Market Sentiment
                    with cols[1]:
                        sent = sentiment.get('sentiment', 'Neutral')
                        color = (
                            'green' if sent == 'Positive'
                            else 'red' if sent == 'Negative'
                            else 'orange'
                        )
                        st.markdown(
                            f"""
                            **Market Sentiment:**
                            <p style='color: {color}; font-size: 20px;'>
                            {sent}
                            </p>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    # Recommendation
                    with cols[2]:
                        rec = sentiment.get('recommendation', 'Monitor market conditions')
                        st.markdown(
                            f"""
                            **Recommendation:**
                            <p style='color: blue; font-size: 20px;'>
                            {rec}
                            </p>
                            """,
                            unsafe_allow_html=True
                        )
            except Exception as e:
                logging.error(f"Error calculating sentiment: {str(e)}")
                st.warning("Unable to generate market sentiment analysis")
        except Exception as e:
            logging.error(f"Error in market analysis calculations: {str(e)}")
            st.error("Error processing market analysis")
                
    except Exception as e:
        logging.error(f"Critical error in display_predictions_and_alerts: {str(e)}")
        st.error("An error occurred while analyzing market data")

def calculate_break_even(purchase_price: float, grading_cost: float = 50, shipping_cost: float = 15, selling_fees: float = 0.1287) -> dict:
    """Calculate break-even price considering all costs"""
    try:
        total_costs = purchase_price + grading_cost + shipping_cost
        break_even_price = total_costs / (1 - selling_fees)
        
        return {
            'break_even_price': break_even_price,
            'total_costs': total_costs,
            'cost_breakdown': {
                'purchase_price': purchase_price,
                'grading_cost': grading_cost,
                'shipping_cost': shipping_cost,
                'selling_fees': break_even_price * selling_fees
            }
        }
    except Exception as e:
        logger.error(f"Error in break-even calculation: {e}")
        return None

def project_collection_value(collection_df: pd.DataFrame, analyzer: CardMarketAnalyzer, months: int = 12) -> dict:
    """Project future value of entire collection"""
    try:
        if collection_df.empty:
            return None
            
        total_current_value = 0
        total_projected_value = 0
        biggest_gainers = []
        
        for _, row in collection_df.iterrows():
            # Current values (use PSA 10 price as default)
            current_value = max(row['Last PSA 10 Price'], row['Last PSA 9 Price'], row['Last Raw Price'])
            total_current_value += current_value
            
            # Create historical data for prediction
            historical_data = pd.DataFrame({
                'Date': pd.date_range(end=pd.Timestamp.now(), periods=90),
                'Price': np.linspace(
                    current_value * 0.9,  # Simulate some price history
                    current_value,
                    90
                )
            })
            
            # Get price prediction
            predictions, _, _ = analyzer.predict_prices(historical_data, days_to_predict=months*30)
            if predictions is not None:
                projected_value = predictions[-1]
                projected_change = ((projected_value - current_value) / current_value * 100)
                
                total_projected_value += projected_value
                
                if projected_change > 10:  # Track significant gainers
                    biggest_gainers.append({
                        'card': f"{row['Player']} - {row['Card Description']}",
                        'current_value': current_value,
                        'projected_value': projected_value,
                        'change_percent': projected_change
                    })
        
        return {
            'current_value': total_current_value,
            'projected_value': total_projected_value,
            'projected_change': ((total_projected_value - total_current_value) / total_current_value * 100),
            'biggest_gainers': sorted(biggest_gainers, key=lambda x: x['change_percent'], reverse=True)
        }
        
    except Exception as e:
        logger.error(f"Error in collection value projection: {e}")
        return None

def display_trade_analyzer():
    """Display trade analyzer interface"""
    try:
        st.title("💱 Trade Analyzer")
        
        # Generate a unique session ID for this analysis
        if "trade_analyzer_session" not in st.session_state:
            st.session_state.trade_analyzer_session = f"{int(time.time() * 1000000)}_{random.randint(10000, 99999)}"
        
        session_id = st.session_state.trade_analyzer_session
        
        # Card 1 (Your Card) Input
        st.subheader("Your Card")
        card1_player = st.text_input("Player Name (Your Card)", key=f"card1_player_{session_id}")
        card1_desc = st.text_input("Card Description (Your Card)", key=f"card1_desc_{session_id}")
        card1_condition = st.selectbox("Condition (Your Card)", ["raw", "psa 9", "psa 10"], key=f"card1_condition_{session_id}")
        card1_price = st.number_input("Current Market Value (Your Card)", min_value=0.0, value=0.0, key=f"card1_price_{session_id}")
        
        # Card 2 (Their Card) Input
        st.subheader("Their Card")
        card2_player = st.text_input("Player Name (Their Card)", key=f"card2_player_{session_id}")
        card2_desc = st.text_input("Card Description (Their Card)", key=f"card2_desc_{session_id}")
        card2_condition = st.selectbox("Condition (Their Card)", ["raw", "psa 9", "psa 10"], key=f"card2_condition_{session_id}")
        card2_price = st.number_input("Current Market Value (Their Card)", min_value=0.0, value=0.0, key=f"card2_price_{session_id}")
        
        if st.button("Analyze Trade", key=f"analyze_trade_{session_id}"):
            if not all([card1_player, card1_desc, card2_player, card2_desc]):
                st.warning("Please fill in all card details")
                return
                
            with st.spinner("Analyzing cards..."):
                # Generate unique chart IDs
                chart_id1 = f"chart1_{int(time.time() * 1000000)}_{random.randint(100000, 999999)}"
                chart_id2 = f"chart2_{int(time.time() * 1000000)}_{random.randint(100000, 999999)}"
                
                # Create progress container
                progress_container = st.empty()
                progress_container.progress(0)
                
                try:
                    # Card 1 analysis
                    search_query1 = f"{card1_player} {card1_desc}"
                    card1_sales = get_ebay_sales(search_query1, card1_condition)
                    progress_container.progress(25)
                    
                    # Calculate metrics for card 1
                    card1_metrics = calculate_metrics(card1_sales)
                    card1_stats = {
                        'avg_price': float(card1_sales['Price'].mean() if not card1_sales.empty else card1_price),
                        'max_price': float(card1_sales['Price'].max() if not card1_sales.empty else card1_price),
                        'min_price': float(card1_sales['Price'].min() if not card1_sales.empty else card1_price),
                        'sales_count': str(len(card1_sales) if not card1_sales.empty else 0),
                        'volatility': f"{float(card1_metrics.get('volatility', 0)):.1f}%" if card1_metrics else "0.0%",
                        'trend': f"{float(card1_metrics.get('trend', 0)):.1f}%" if card1_metrics else "0.0%"
                    }
                    progress_container.progress(50)
                    
                    # Card 2 analysis
                    search_query2 = f"{card2_player} {card2_desc}"
                    card2_sales = get_ebay_sales(search_query2, card2_condition)
                    progress_container.progress(75)
                    
                    # Calculate metrics for card 2
                    card2_metrics = calculate_metrics(card2_sales)
                    card2_stats = {
                        'avg_price': float(card2_sales['Price'].mean() if not card2_sales.empty else card2_price),
                        'max_price': float(card2_sales['Price'].max() if not card2_sales.empty else card2_price),
                        'min_price': float(card2_sales['Price'].min() if not card2_sales.empty else card2_price),
                        'sales_count': str(len(card2_sales) if not card2_sales.empty else 0),
                        'volatility': f"{float(card2_metrics.get('volatility', 0)):.1f}%" if card2_metrics else "0.0%",
                        'trend': f"{float(card2_metrics.get('trend', 0)):.1f}%" if card2_metrics else "0.0%"
                    }
                    progress_container.progress(100)
                    
                    # Clear progress bar
                    progress_container.empty()
                    
                    # Display comparison
                    st.success("Analysis complete!")
                    
                    # Create comparison charts
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Your Card")
                        fig1 = create_comparison_chart(card1_sales, card1_player, card1_desc)
                        st.plotly_chart(fig1, use_container_width=True, key=chart_id1)
                    
                    with col2:
                        st.subheader("Their Card")
                        fig2 = create_comparison_chart(card2_sales, card2_player, card2_desc)
                        st.plotly_chart(fig2, use_container_width=True, key=chart_id2)
                    
                    # Display metrics comparison with proper type handling
                    metrics_data = {
                        'Metric': ['Average Price', 'Maximum Price', 'Minimum Price', 'Sales Count', 'Volatility', 'Price Trend'],
                        'Your Card': [
                            f"${card1_stats['avg_price']:.2f}",
                            f"${card1_stats['max_price']:.2f}",
                            f"${card1_stats['min_price']:.2f}",
                            card1_stats['sales_count'],
                            card1_stats['volatility'],
                            card1_stats['trend']
                        ],
                        'Their Card': [
                            f"${card2_stats['avg_price']:.2f}",
                            f"${card2_stats['max_price']:.2f}",
                            f"${card2_stats['min_price']:.2f}",
                            card2_stats['sales_count'],
                            card2_stats['volatility'],
                            card2_stats['trend']
                        ]
                    }
                    
                    # Create DataFrame with all string values
                    metrics_comparison = pd.DataFrame(metrics_data)
                    metrics_comparison = metrics_comparison.astype(str)
                    
                    st.subheader("📊 Metrics Comparison")
                    st.dataframe(metrics_comparison.set_index('Metric'))
                    
                    # Display trade analysis
                    value_diff = card2_stats['avg_price'] - card1_stats['avg_price']
                    value_diff_pct = (value_diff / card1_stats['avg_price'] * 100) if card1_stats['avg_price'] > 0 else 0
                    
                    st.subheader("📈 Trade Analysis")
                    
                    # Value comparison
                    value_color = 'green' if value_diff >= 0 else 'red'
                    st.markdown(
                        f"""
                        **Value Difference:** <span style='color:{value_color}'>${abs(value_diff):.2f} ({'gain' if value_diff >= 0 else 'loss'})</span><br>
                        **Percentage Difference:** <span style='color:{value_color}'>{abs(value_diff_pct):.1f}%</span>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Risk assessment
                    volatility_diff = float(card2_metrics.get('volatility', 0)) - float(card1_metrics.get('volatility', 0))
                    risk_color = 'red' if volatility_diff > 5 else 'green' if volatility_diff < -5 else 'orange'
                    st.markdown(
                        f"""
                        **Risk Assessment:**
                        - Volatility Difference: <span style='color:{risk_color}'>{volatility_diff:+.1f}%</span>
                        - {'Higher' if volatility_diff > 5 else 'Lower' if volatility_diff < -5 else 'Similar'} risk profile
                        """,
                        unsafe_allow_html=True
                    )
                    
                except Exception as e:
                    logger.error(f"Error during trade analysis: {str(e)}")
                    st.error("An error occurred during the trade analysis. Please try again.")
                    progress_container.empty()
    
    except Exception as e:
        logger.error(f"Error in display_trade_analyzer: {str(e)}")
        st.error("Error initializing trade analyzer")

def create_comparison_chart(sales_data, player_name, card_desc):
    """Create a comparison chart with consistent styling and error handling"""
    try:
        if sales_data is None or sales_data.empty:
            # Create empty chart with message
            fig = go.Figure()
            fig.add_annotation(
                text="No sales data available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=14)
            )
            fig.update_layout(
                title=f"{player_name} {card_desc}",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                showlegend=False
            )
            return fig
        
        # Ensure data types are correct
        sales_data = sales_data.copy()
        sales_data['Date'] = pd.to_datetime(sales_data['Date'])
        sales_data['Price'] = pd.to_numeric(sales_data['Price'], errors='coerce')
        sales_data = sales_data.dropna(subset=['Price'])
        
        # Create base scatter plot
        fig = go.Figure()
        
        # Add sales data points
        fig.add_trace(go.Scatter(
            x=sales_data['Date'],
            y=sales_data['Price'],
            mode='markers',
            name='Sales',
            marker=dict(
                size=8,
                color='blue',
                opacity=0.6
            ),
            hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br>" +
                         "<b>Price:</b> $%{y:.2f}<br>" +
                         "<extra></extra>"
        ))
        
        # Add trend line if enough data points
        if len(sales_data) > 2:
            z = np.polyfit(range(len(sales_data)), sales_data['Price'], 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(
                x=sales_data['Date'],
                y=p(range(len(sales_data))),
                name='Trend',
                line=dict(color='red', dash='dash'),
                hovertemplate="<b>Trend Price:</b> $%{y:.2f}<br>" +
                             "<extra></extra>"
            ))
        
        # Add 7-day moving average if enough data
        if len(sales_data) >= 7:
            sales_data['MA7'] = sales_data['Price'].rolling(window=7, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=sales_data['Date'],
                y=sales_data['MA7'],
                name='7-day MA',
                line=dict(color='green'),
                hovertemplate="<b>7-day Avg:</b> $%{y:.2f}<br>" +
                             "<extra></extra>"
            ))
        
        # Update layout
        fig.update_layout(
            title=f"{player_name} {card_desc}",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        # Format axes
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating comparison chart: {str(e)}")
        # Return a basic error chart
        fig = go.Figure()
        fig.add_annotation(
            text="Error creating chart",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color='red')
        )
        return fig

def display_market_analysis(sales_data, metrics, card_name):
    """Display comprehensive market analysis with enhanced trend metrics"""
    try:
        if sales_data is None or sales_data.empty:
            st.warning("No sales data available for market analysis")
            return
            
        st.subheader("📊 Market Analysis")
        
        # Market Score and Key Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            market_score = metrics.get('market_score', 0)
            score_color = (
                'green' if market_score >= 70
                else 'orange' if market_score >= 40
                else 'red'
            )
            st.markdown(
                f"""
                **Market Score**
                <p style='color: {score_color}; font-size: 24px; font-weight: bold;'>
                {market_score:.1f}/100
                </p>
                """,
                unsafe_allow_html=True
            )
            st.info(
                """
                Market Score combines:
                • Trend Impact (40%)
                • Market Liquidity (30%)
                • Price Stability (30%)
                
                Higher scores indicate stronger, more stable markets with good trading opportunities.
                """
            )
            
        with col2:
            avg_price = metrics.get('avg', 0)
            std_price = metrics.get('std', 0)
            st.markdown(
                f"""
                **Average Price**
                <p style='font-size: 24px;'>
                ${avg_price:.2f} ± ${std_price:.2f}
                </p>
                """,
                unsafe_allow_html=True
            )
            
        with col3:
            liquidity = metrics.get('liquidity', 0)
            volume = metrics.get('volume', 0)
            liquidity_color = (
                'green' if liquidity >= 70
                else 'orange' if liquidity >= 40
                else 'red'
            )
            st.markdown(
                f"""
                **Market Liquidity**
                <p style='color: {liquidity_color}; font-size: 24px;'>
                {liquidity:.1f}% ({volume} sales)
                </p>
                """,
                unsafe_allow_html=True
            )
            st.info(
                """
                Liquidity Score factors:
                • Sales Frequency (30%)
                • Trading Consistency (30%)
                • Price Stability (20%)
                • Recent Activity (20%)
                
                Higher liquidity means easier buying/selling with less price impact.
                """
            )
            
        # Trend Analysis Section
        st.markdown("### 📈 Trend Analysis")
        
        # Get trend metrics
        trend_metrics = metrics.get('trend_metrics', {})
        cycle_position = trend_metrics.get('cycle_position', 'unknown')
        
        # Display trend metrics in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Price Trends")
            trends_df = pd.DataFrame({
                'Timeframe': ['Short-term (7d)', 'Medium-term (30d)', 'Long-term (90d)'],
                'Trend': [
                    trend_metrics.get('short_term', 0),
                    trend_metrics.get('medium_term', 0),
                    trend_metrics.get('long_term', 0)
                ]
            })
            
            # Format trend values with arrows and colors
            def format_trend(value):
                if abs(value) < 0.01:
                    return "→ Neutral"
                elif value > 0:
                    return f"↗ Rising (${value:.2f}/day)"
                else:
                    return f"↘ Falling (${value:.2f}/day)"
                    
            for idx, row in trends_df.iterrows():
                trend_value = row['Trend']
                trend_text = format_trend(trend_value)
                color = 'green' if trend_value > 0 else 'red' if trend_value < 0 else 'gray'
                st.markdown(
                    f"""
                    **{row['Timeframe']}**  
                    <span style='color: {color};'>{trend_text}</span>
                    """,
                    unsafe_allow_html=True
                )
                
        with col2:
            st.markdown("#### Trend Strength & Reliability")
            
            # Trend Strength
            strength = trend_metrics.get('strength', 0)
            st.markdown(
                f"""
                **Trend Strength:** {strength:.1f}/100
                <div style='width: 100%; background-color: #eee; height: 20px; border-radius: 10px;'>
                    <div style='width: {strength}%; background-color: {'green' if strength >= 70 else 'orange' if strength >= 40 else 'red'}; 
                         height: 100%; border-radius: 10px;'></div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Trend Reliability
            reliability = trend_metrics.get('reliability', 0)
            st.markdown(
                f"""
                **Trend Reliability:** {reliability:.1f}/100
                <div style='width: 100%; background-color: #eee; height: 20px; border-radius: 10px;'>
                    <div style='width: {reliability}%; background-color: {'green' if reliability >= 70 else 'orange' if reliability >= 40 else 'red'}; 
                         height: 100%; border-radius: 10px;'></div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Market Cycle Position
            cycle_colors = {
                'uptrend': 'green',
                'downtrend': 'red',
                'accumulation': 'blue',
                'distribution': 'orange',
                'sideways': 'gray',
                'unknown': 'gray'
            }
            
            st.markdown(
                f"""
                **Market Cycle:** <span style='color: {cycle_colors[cycle_position]};'>{cycle_position.title()}</span>
                """,
                unsafe_allow_html=True
            )
        
        # Price Chart
        st.markdown("### 💹 Price History")
        
        # Create price chart with Plotly
        fig = go.Figure()
        
        # Add price scatter plot
        fig.add_trace(go.Scatter(
            x=sales_data['Date'],
            y=sales_data['Price'],
            mode='markers',
            name='Sales',
            marker=dict(
                size=8,
                color='blue',
                opacity=0.6
            )
        ))
        
        # Add trend line
        if len(sales_data) > 1:
            x = np.arange(len(sales_data))
            y = sales_data['Price'].values
            slope, intercept, _, _, _ = stats.linregress(x, y)
            trend_line = slope * x + intercept
            fig.add_trace(go.Scatter(
                x=sales_data['Date'],
                y=trend_line,
                mode='lines',
                name='Trend',
                line=dict(
                    color='red',
                    dash='dash'
                )
            ))
            
            # Add moving averages if enough data points
            if len(sales_data) >= 5:
                # 7-day moving average
                ma7 = sales_data['Price'].rolling(window=min(7, len(sales_data))).mean()
                fig.add_trace(go.Scatter(
                    x=sales_data['Date'],
                    y=ma7,
                    mode='lines',
                    name='7-day MA',
                    line=dict(color='green')
                ))
                
                # 30-day moving average
                if len(sales_data) >= 30:
                    ma30 = sales_data['Price'].rolling(window=30).mean()
                    fig.add_trace(go.Scatter(
                        x=sales_data['Date'],
                        y=ma30,
                        mode='lines',
                        name='30-day MA',
                        line=dict(color='orange')
                    ))
        
        # Update layout
        fig.update_layout(
            title=f"Price History for {card_name}",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode='x unified',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Market Summary
        st.markdown("### 📝 Market Summary")
        
        # Generate market summary based on metrics
        summary_points = []
        
        # Trend summary
        trend_direction = "upward" if trend_metrics.get('medium_term', 0) > 0 else "downward"
        trend_strength_text = (
            "strong" if trend_metrics.get('strength', 0) >= 70
            else "moderate" if trend_metrics.get('strength', 0) >= 40
            else "weak"
        )
        
        summary_points.append(
            f"• The market shows a {trend_strength_text} {trend_direction} trend "
            f"with {trend_metrics.get('reliability', 0):.1f}% reliability."
        )
        
        # Market cycle insight
        cycle_insights = {
            'uptrend': "The market is in a clear upward trend with strong buying pressure.",
            'downtrend': "The market is experiencing a downward trend with selling pressure.",
            'accumulation': "The market appears to be in an accumulation phase, suggesting potential future growth.",
            'distribution': "The market shows signs of distribution, indicating possible trend reversal.",
            'sideways': "The market is moving sideways with no clear directional trend.",
            'unknown': "Insufficient data to determine market cycle position."
        }
        summary_points.append(f"• {cycle_insights.get(cycle_position, '')}")
        
        # Volatility insight
        volatility = metrics.get('volatility', 0)
        volatility_text = (
            "high" if volatility > 30
            else "moderate" if volatility > 15
            else "low"
        )
        summary_points.append(
            f"• Market volatility is {volatility_text} ({volatility:.1f}%), "
            f"with a risk-adjusted return of {metrics.get('volatility_metrics', {}).get('risk_adjusted_return', 0):.2f}."
        )
        
        # Liquidity insight
        liquidity_text = (
            "highly liquid" if liquidity >= 70
            else "moderately liquid" if liquidity >= 40
            else "illiquid"
        )
        summary_points.append(
            f"• The market is {liquidity_text} with {volume} sales recorded "
            f"and a liquidity score of {liquidity:.1f}%."
        )
        
        # Display summary points
        for point in summary_points:
            st.markdown(point)
        
        # Recent Sales Table
        st.markdown("### 📊 Recent Sales")
        
        # Display recent sales in a formatted table
        recent_sales = sales_data.sort_values('Date', ascending=False).head(5)
        if not recent_sales.empty:
            # Format the data for display
            display_df = recent_sales.copy()
            display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
            display_df['Price'] = display_df['Price'].apply(lambda x: f"${x:.2f}")
            
            # Reorder columns and rename them
            display_df = display_df[['Date', 'Price', 'Title', 'Link']]
            display_df.columns = ['Date', 'Price', 'Item Title', 'eBay Link']
            
            # Display the table
            st.dataframe(
                display_df,
                column_config={
                    "eBay Link": st.column_config.LinkColumn()
                },
                hide_index=True
            )
        else:
            st.info("No recent sales data available")
            
    except Exception as e:
        logger.error(f"Error in market analysis display: {str(e)}")
        st.error("Error displaying market analysis")

def display_profit_analysis(metrics, purchase_price):
    """Display profit analysis for different card grades"""
    try:
        if not metrics:
            st.warning("No metrics available for profit analysis")
            return

        st.subheader("💰 Profit Analysis")
        
        # Input fields for costs
        col1, col2, col3 = st.columns(3)
        with col1:
            grading_cost = st.number_input("Grading Cost ($)", min_value=0.0, value=24.99, step=5.0)
        with col2:
            shipping_cost = st.number_input("Shipping Cost ($)", min_value=0.0, value=15.0, step=1.0)
        with col3:
            selling_fees = st.number_input("Selling Fees (%)", min_value=0.0, max_value=100.0, value=12.87, step=0.01) / 100

        # Store best ROI for recommendation
        best_roi = {'grade': None, 'roi': -float('inf')}
        
        # Calculate and display profit analysis for each grade
        for grade in ['Raw', 'PSA 9', 'PSA 10']:
            if grade in metrics:
                grade_metrics = metrics[grade]
                avg_price = grade_metrics.get('avg', 0)
                
                # Calculate costs and fees
                total_cost = purchase_price
                if grade != 'Raw':
                    total_cost += grading_cost + shipping_cost
                selling_fee_amount = avg_price * selling_fees
                shipping_to_buyer = shipping_cost if grade == 'Raw' else 0
                
                # Calculate profit
                net_profit = avg_price - total_cost - selling_fee_amount - shipping_to_buyer
                roi = (net_profit / total_cost * 100) if total_cost > 0 else 0
                
                # Track best ROI
                if roi > best_roi['roi']:
                    best_roi = {'grade': grade, 'roi': roi, 'profit': net_profit}
                
                # Display analysis
                st.write(f"**{grade} Analysis:**")
                
                # Metrics row
                metric_cols = st.columns(4)
                with metric_cols[0]:
                    st.metric(
                        f"{grade} Average Sale Price",
                        f"${avg_price:,.2f}"
                    )
                with metric_cols[1]:
                    st.metric(
                        f"{grade} Total Cost",
                        f"${total_cost:,.2f}"
                    )
                with metric_cols[2]:
                    st.metric(
                        f"{grade} Net Profit",
                        f"${net_profit:,.2f}"
                    )
                with metric_cols[3]:
                    st.metric(
                        f"{grade} ROI",
                        f"{roi:.1f}%"
                    )
                
                # Details row
                detail_cols = st.columns(2)
                with detail_cols[0]:
                    st.write("**Cost Breakdown:**")
                    st.write(f"- Purchase Price: ${purchase_price:,.2f}")
                    if grade != 'Raw':
                        st.write(f"- Grading Cost: ${grading_cost:,.2f}")
                        st.write(f"- Shipping to Grader: ${shipping_cost:,.2f}")
                    if grade == 'Raw':
                        st.write(f"- Shipping to Buyer: ${shipping_to_buyer:,.2f}")
                    st.write(f"- Selling Fees: ${selling_fee_amount:,.2f}")
                
                with detail_cols[1]:
                    st.write("**Break-even Analysis:**")
                    break_even = total_cost / (1 - selling_fees)
                    st.write(f"- Break-even Price: ${break_even:,.2f}")
                    margin = avg_price - break_even
                    st.write(f"- Margin over Break-even: ${margin:,.2f}")
                
                st.markdown("---")  # Add separator between grades

        # Final Recommendation
        st.subheader("📋 Final Recommendation")
        if best_roi['grade']:
            recommendation_color = (
                "green" if best_roi['roi'] > 20
                else "orange" if best_roi['roi'] > 0
                else "red"
            )
            
            if best_roi['grade'] == 'Raw':
                action = "Sell raw" if best_roi['roi'] > 10 else "Hold raw"
            else:
                action = f"Grade to {best_roi['grade']}" if best_roi['roi'] > 20 else "Hold raw"
            
            st.markdown(f"""
            <div style='padding: 20px; border-radius: 5px; background-color: rgba(0, 0, 0, 0.05);'>
                <p style='color: {recommendation_color}; font-size: 18px; font-weight: bold;'>
                    {action} (Best ROI: {best_roi['roi']:.1f}% as {best_roi['grade']})
                </p>
                <p>
                    • Expected Profit: ${best_roi['profit']:,.2f}<br>
                    • Best Grade Option: {best_roi['grade']}<br>
                    • {'Consider grading if condition is excellent.' if best_roi['grade'] != 'Raw' else 'Selling raw provides the best return.'}
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Unable to generate recommendation - insufficient data")

    except Exception as e:
        logger.error(f"Error in display_profit_analysis: {str(e)}")
        st.error("Error displaying profit analysis")

def generate_recommendation(price_data, volume_data, volatility):
    """Generate market recommendation with safe calculations"""
    try:
        # Input validation
        if not isinstance(price_data, pd.Series) or not isinstance(volume_data, pd.Series):
            return "Invalid data format", {
                'trend_score': 'N/A',
                'volume_score': 'N/A',
                'volatility_score': 'N/A'
            }

        if len(price_data) < 2 or len(volume_data) < 2:
            return "Insufficient data for recommendation", {
                'trend_score': 'N/A',
                'volume_score': 'N/A',
                'volatility_score': 'N/A'
            }

        # Calculate price trend safely
        recent_price = float(price_data.iloc[-1])
        previous_price = float(price_data.iloc[0])
        try:
            price_trend = ((recent_price - previous_price) / max(0.01, abs(previous_price))) * 100
        except (ValueError, ZeroDivisionError):
            price_trend = 0

        # Calculate volume trend safely
        recent_volume = float(volume_data.iloc[-1])
        previous_volume = float(volume_data.iloc[0])
        try:
            volume_trend = ((recent_volume - previous_volume) / max(0.01, abs(previous_volume))) * 100
        except (ValueError, ZeroDivisionError):
            volume_trend = 0

        # Prepare scores with safe formatting
        scores = {
            'trend_score': f"{price_trend:+.1f}%" if not pd.isna(price_trend) else 'N/A',
            'volume_score': f"{volume_trend:+.1f}%" if not pd.isna(volume_trend) else 'N/A',
            'volatility_score': f"{volatility:.1f}%" if volatility is not None and not pd.isna(volatility) else 'N/A'
        }

        # Generate recommendation with thresholds
        if pd.isna(price_trend) or pd.isna(volume_trend):
            recommendation = "Unable to generate recommendation due to invalid data"
        elif price_trend > 10 and volume_trend > 0:
            recommendation = "Strong Buy - Price trending up with increasing volume"
        elif price_trend < -10 and volume_trend < 0:
            recommendation = "Consider Selling - Price trending down with decreasing volume"
        elif price_trend > 5:
            recommendation = "Moderate Buy - Price showing positive trend"
        elif price_trend < -5:
            recommendation = "Hold/Watch - Price showing negative trend"
        else:
            recommendation = "Market Stable - Monitor for changes"

        return recommendation, scores

    except Exception as e:
        logger.error(f"Error generating recommendation: {str(e)}")
        return "Unable to generate recommendation", {
            'trend_score': 'N/A',
            'volume_score': 'N/A',
            'volatility_score': 'N/A'
        }

def display_collection_manager():
    """Display the collection manager interface"""
    try:
        st.subheader("📊 Collection Overview")
        
        # Initialize collection in session state if not exists
        if 'collection' not in st.session_state:
            st.session_state.collection = pd.DataFrame(columns=[
                'Player', 'Card Description', 'Purchase Price',
                'Last Raw Price', 'Last PSA 9 Price', 'Last PSA 10 Price',
                'Date Added', 'Last Updated'
            ])
        
        # Display current collection
        if not st.session_state.collection.empty:
            st.dataframe(
                st.session_state.collection.style.format({
                    'Purchase Price': '${:,.2f}',
                    'Last Raw Price': '${:,.2f}',
                    'Last PSA 9 Price': '${:,.2f}',
                    'Last PSA 10 Price': '${:,.2f}'
                }),
                hide_index=True
            )
            
            # Collection Statistics
            total_value = st.session_state.collection['Last Raw Price'].sum()
            total_cost = st.session_state.collection['Purchase Price'].sum()
            total_profit = total_value - total_cost
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Collection Value", f"${total_value:,.2f}")
            with col2:
                st.metric("Total Investment", f"${total_cost:,.2f}")
            with col3:
                st.metric("Potential Profit", f"${total_profit:,.2f}", 
                         f"{(total_profit/total_cost*100 if total_cost > 0 else 0):,.1f}%")
            
            # Export collection button
            if st.button("Export Collection"):
                try:
                    st.session_state.collection.to_csv("card_collection.csv", index=False)
                    st.success("Collection exported to card_collection.csv")
                except Exception as e:
                    logger.error(f"Error exporting collection: {e}")
                    st.error("Error exporting collection")
        else:
            st.info("Your collection is empty. Add cards using the Search & Analysis tab.")
            
    except Exception as e:
        logger.error(f"Error in collection manager: {str(e)}")
        st.error("Error displaying collection manager")

def add_to_collection(player_name: str, card_desc: str, metrics: dict, purchase_price: float, trend_scores: dict) -> bool:
    """Add a card to the collection with proper error handling"""
    try:
        # Initialize collection if not exists
        if 'collection' not in st.session_state:
            st.session_state.collection = pd.DataFrame(columns=[
                'Player', 'Card Description', 'Purchase Price',
                'Last Raw Price', 'Last PSA 9 Price', 'Last PSA 10 Price',
                'Date Added', 'Last Updated'
            ])
        
        # Get current prices from metrics
        raw_price = metrics.get('Raw', {}).get('avg', 0)
        psa9_price = metrics.get('PSA 9', {}).get('avg', 0)
        psa10_price = metrics.get('PSA 10', {}).get('avg', 0)
        
        # Create new card entry
        new_card = pd.DataFrame([{
            'Player': player_name,
            'Card Description': card_desc,
            'Purchase Price': purchase_price,
            'Last Raw Price': raw_price,
            'Last PSA 9 Price': psa9_price,
            'Last PSA 10 Price': psa10_price,
            'Date Added': pd.Timestamp.now().strftime('%Y-%m-%d'),
            'Last Updated': pd.Timestamp.now().strftime('%Y-%m-%d')
        }])
        
        # Add to collection
        st.session_state.collection = pd.concat([st.session_state.collection, new_card], ignore_index=True)
        return True
        
    except Exception as e:
        logger.error(f"Error adding to collection: {str(e)}")
        st.error("Error adding card to collection")
        return False

def run_image_search(uploaded_file):
    """Handle image search functionality with improved result display"""
    try:
        # Validate image format
        if uploaded_file.type not in ["image/jpeg", "image/jpg", "image/png"]:
            st.error("Please upload a valid image file (JPG or PNG)")
            return
            
        # Create two columns for layout
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Card", use_container_width=True)
        
        with col2:
            # Show processing status
            with st.spinner("Processing image..."):
                # Convert image to bytes for processing
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format=image.format)
                img_byte_arr = img_byte_arr.getvalue()
                
                # Get search results
                results = get_cached_search_by_image_results(img_byte_arr)
        
        if not results.empty:
            # Group results by confidence level
            high_confidence = results[results['Confidence'] == 'High']
            medium_confidence = results[results['Confidence'] == 'Medium']
            low_confidence = results[results['Confidence'] == 'Low']
            
            # Display results based on confidence
            if not high_confidence.empty:
                st.subheader("🎯 Best Matches")
                display_results_group(high_confidence)
                
            if not medium_confidence.empty:
                st.subheader("📊 Possible Matches")
                display_results_group(medium_confidence)
                
            if not low_confidence.empty:
                with st.expander("📋 Other Potential Matches"):
                    st.info("These matches have lower confidence but might still be relevant.")
                    display_results_group(low_confidence)
            
            # Add option to switch to text search
            st.divider()
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write("Can't find what you're looking for?")
            with col2:
                if st.button("Try Text Search Instead"):
                    st.session_state.show_text_search = True
            
            # Show text search if requested
            if st.session_state.get('show_text_search', False):
                st.subheader("🔍 Search by Text")
                query = st.text_input("Enter card details:", key="manual_search")
                if query:
                    run_text_search(query)
                    
        else:
            st.warning("No matches found. Try adjusting the image or search by text instead.")
            query = st.text_input("Enter card details:", key="fallback_search")
            if query:
                run_text_search(query)
            
    except Exception as e:
        logger.error(f"Error in image search: {str(e)}")
        st.error("Error processing image search. Please try again with a different image.")

def display_results_group(results_df):
    """Helper function to display a group of search results"""
    for _, row in results_df.iterrows():
        with st.container():
            cols = st.columns([1, 2, 1])
            with cols[0]:
                st.image(row['Image URL'], width=100)
            with cols[1]:
                st.write(f"**{row['Card Name']}**")
                st.write(f"Set: {row['Set']} ({row['Year']})")
                st.write(f"Match: {row['Similarity']} ({row['Confidence']} confidence)")
            with cols[2]:
                st.metric("Last Sale", row['Last Sale'])
                st.metric("Market Price", row['Market Price'])
                if st.button("Select", key=f"select_{row['Card Name']}"):
                    # When a card is selected, run text search with the card details
                    query = f"{row['Card Name']} {row['Set']} {row['Year']}"
                    st.session_state.selected_card = query
                    run_text_search(query)
                    return
        st.divider()

def main():
    """Main function to run the Streamlit application"""
    st.title("Sports Card Market Analyzer")
    
    # Create tabs for different features
    tab1, tab2, tab3 = st.tabs([
        "🔍 Search by Text",
        "💰 Trade Analyzer",
        "📊 Collection Manager"
    ])
    
    with tab1:
        st.header("Search by Text")
        query = st.text_input("Enter card name (e.g., 'Tom Brady Rookie Card')")
        if query:
            run_text_search(query)
    
    with tab2:
        display_trade_analyzer()
        
    with tab3:
        display_collection_manager()

if __name__ == "__main__":
    main()

