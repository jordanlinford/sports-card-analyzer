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
from typing import Dict, Any, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_ebay_sales(query, condition="raw"):
    """Fetch eBay sales data with fallback to dummy data"""
    try:
        # Add delay between API calls
        time.sleep(2)
        
        api = Finding(appid=EBAY_CONFIG['app_id'], config_file=None)
        
        if condition.lower() == "raw":
            filter_query = f"{query} -PSA -BGS -SGC"
        elif condition.lower() == "psa 9":
            filter_query = f"{query} PSA 9"
        else:  # PSA 10
            filter_query = f"{query} PSA 10"

        print(f"\nAttempting eBay search for {condition} with query: {filter_query}")
        
        # Try eBay API first
        try:
            response = api.execute('findItemsAdvanced', {
                'keywords': filter_query,
                'itemFilter': [
                    {'name': 'SoldItemsOnly', 'value': 'true'},
                    {'name': 'EndTimeFrom', 'value': (datetime.now() - timedelta(days=90)).isoformat()}
                ],
                'paginationInput': {'entriesPerPage': '100', 'pageNumber': '1'},
                'sortOrder': 'EndTimeSoonest'
            })
            
            if hasattr(response.reply, 'searchResult') and hasattr(response.reply.searchResult, 'item'):
                items = response.reply.searchResult.item
                sales_data = []
                for item in items:
                    try:
                        sales_data.append({
                            'Date': pd.to_datetime(item.listingInfo.endTime),
                            'Price': float(item.sellingStatus.currentPrice.value),
                            'Title': item.title
                        })
                    except AttributeError:
                        continue
                
                if sales_data:
                    df = pd.DataFrame(sales_data)
                    return df.sort_values('Date', ascending=False)
        
        except Exception as e:
            print(f"eBay API error: {str(e)}, falling back to dummy data")
        
        # Fallback to dummy data
        print(f"Generating dummy data for {condition}")
        base_price = 100  # Adjust this based on the card type
        return generate_dummy_sales(query, condition, base_price)
    
    except Exception as e:
        print(f"Error in data fetching: {str(e)}")
        return generate_dummy_sales(query, condition)

def fetch_image(query):
    """Fetch card image using Google Custom Search API"""
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'key': GOOGLE_CONFIG['api_key'],
        'cx': GOOGLE_CONFIG['search_engine_id'],
        'q': f"{query} sports card",
        'searchType': 'image',
        'num': 1
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if 'items' in data:
            return data['items'][0]['link']
        return None
    except Exception as e:
        st.error(f"Error fetching image: {e}")
        return None

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

def generate_dummy_sales(query, condition="raw", base_price=100):
    """Generate realistic dummy sales data for 90 days"""
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
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Price': prices,
        'Title': [f"{query} {condition} Sale {i+1}" for i in range(len(dates))]
    })
    
    # Sort by date descending
    return df.sort_values('Date', ascending=False)

def get_cached_search_by_image_results(image_bytes):
    """Mock function for image search results"""
    return pd.DataFrame()

def calculate_profit_analysis(raw_sales, psa9_sales, psa10_sales, purchase_price):
    """Calculate profit analysis for different grades"""
    try:
        # Calculate average prices for each condition
        raw_avg = raw_sales['Price'].mean() if not raw_sales.empty else 0
        psa9_avg = psa9_sales['Price'].mean() if not psa9_sales.empty else 0
        psa10_avg = psa10_sales['Price'].mean() if not psa10_sales.empty else 0
        
        # Create profit analysis dataframe
        return pd.DataFrame({
            'Grade': ['Raw', 'PSA 9', 'PSA 10'],
            'Avg Price': [raw_avg, psa9_avg, psa10_avg],
            'Profit': [
                raw_avg - purchase_price,
                psa9_avg - purchase_price,
                psa10_avg - purchase_price
            ],
            'ROI': [
                ((raw_avg - purchase_price) / purchase_price * 100) if purchase_price > 0 else 0,
                ((psa9_avg - purchase_price) / purchase_price * 100) if purchase_price > 0 else 0,
                ((psa10_avg - purchase_price) / purchase_price * 100) if purchase_price > 0 else 0
            ]
        })
    except Exception as e:
        logger.error(f"Error in profit analysis: {e}")
        return pd.DataFrame()

def calculate_market_metrics(sales_data):
    """Calculate market volatility and trend scores"""
    try:
        if sales_data.empty:
            return 0, 3  # Default values
            
        prices = sales_data['Price'].values
        dates = np.arange(len(prices))
        
        # Calculate volatility
        volatility = (prices.std() / prices.mean() * 100) if len(prices) > 1 else 0
        
        # Calculate trend
        if len(prices) > 1:
            slope, _, _, _, _ = stats.linregress(dates, prices)
            avg_price = prices.mean()
            # Normalize slope to a 1-5 scale
            trend_score = 3  # Default neutral score
            if slope > 0:
                trend_score = 4 if slope > avg_price * 0.1 else 3.5
            else:
                trend_score = 2 if slope < -avg_price * 0.1 else 2.5
        else:
            trend_score = 3
            
        return volatility, trend_score
        
    except Exception as e:
        logger.error(f"Error calculating market metrics: {e}")
        return 0, 3

def generate_investment_recommendation(volatility, trend_score, profit_potential):
    """Generate investment recommendation based on market metrics"""
    try:
        # Base recommendation on combined factors
        if trend_score >= 4 and volatility < 30:
            base_rec = "Strong upward trend with low volatility. "
            if profit_potential > 50:
                return base_rec + "High profit potential for grading. Consider buying raw and grading."
            else:
                return base_rec + "Moderate profit potential. Consider holding current position."
                
        elif trend_score >= 3.5 and volatility < 50:
            base_rec = "Positive trend with moderate volatility. "
            if profit_potential > 30:
                return base_rec + "Consider grading if condition is strong."
            else:
                return base_rec + "Monitor market conditions before making moves."
                
        elif trend_score < 2.5 or volatility > 70:
            return "High risk market conditions. Consider waiting for market stabilization."
            
        else:
            return "Neutral market conditions. Monitor trends and reassess regularly."
            
    except Exception as e:
        logger.error(f"Error generating recommendation: {e}")
        return "Unable to generate recommendation due to insufficient data."

def run_text_search(query):
    try:
        # Clear previous search results from session state
        if 'search_metrics' in st.session_state:
            st.session_state.search_metrics = {}
        if 'search_purchase_price' in st.session_state:
            st.session_state.search_purchase_price = 0
        
        # Part 1: Data Fetching
        with st.spinner('Fetching card image...'):
            img_url = fetch_image(query)
            if img_url:
                st.image(img_url, use_container_width=True)
            else:
                st.warning("No card image found")
        
        with st.spinner('Analyzing market data...'):
            # Get sales data with fallback to dummy data
            raw_sales = get_ebay_sales(query, "raw")
            time.sleep(1)
            psa9_sales = get_ebay_sales(query, "psa 9")
            time.sleep(1)
            psa10_sales = get_ebay_sales(query, "psa 10")
            
            # Store the data in session state
            st.session_state.raw_sales = raw_sales
            st.session_state.psa9_sales = psa9_sales
            st.session_state.psa10_sales = psa10_sales
            
            # Format the DataFrames
            for df in [raw_sales, psa9_sales, psa10_sales]:
                if not df.empty:
                    df.rename(columns={
                        'Date': 'Sale Date',
                        'Price': 'Sale Price'
                    }, inplace=True)
                    df['Sale Date'] = df['Sale Date'].dt.strftime('%m/%d/%Y')
            
            # Calculate metrics once
            metrics = {}
            for grade, data in [("Raw", raw_sales), ("PSA 9", psa9_sales), ("PSA 10", psa10_sales)]:
                if not data.empty:
                    prices = data['Sale Price'].values
                    metrics[grade] = {
                        'avg': np.mean(prices),
                        'min': np.min(prices),
                        'max': np.max(prices),
                        'std': np.std(prices),
                        'volatility': (np.std(prices) / np.mean(prices)) * 100
                    }
            
            # Store metrics in session state
            st.session_state.search_metrics = metrics
            
            # Display Analysis Sections
            display_market_analysis(raw_sales, psa9_sales, psa10_sales, metrics, query)
            
            # Purchase price input
            purchase_price = st.number_input(
                "💰 Enter Purchase Price ($):",
                min_value=0.0,
                step=0.01,
                key=f"purchase_price_input_{query}",
                help="Enter the actual price paid or your target purchase price"
            )
            
            # Store purchase price in session state
            st.session_state.search_purchase_price = purchase_price
            
            if purchase_price > 0:
                display_profit_analysis(metrics, purchase_price)
        
    except Exception as e:
        st.error(f"Error in analysis: {e}")
        logger.error(f"Detailed error: {str(e)}")

def run_image_search(uploaded_file):
    try:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        image_bytes = uploaded_file.getvalue()
        results_df = get_cached_search_by_image_results(image_bytes)
        
        if results_df.empty:
            st.info("No results found for image search.")
            dummy_data = generate_dummy_sales("Raw")
            st.dataframe(dummy_data.style.format({'Price': '${:.2f}'}), hide_index=True)
        else:
            st.dataframe(results_df.style.format({'Price': '${:.2f}'}))
            
    except Exception as e:
        st.error(f"Error in image search: {e}")

def validate_metrics(metrics: Dict[str, Any]) -> bool:
    """Validate metrics data structure"""
    required_grades = ['Raw', 'PSA 9', 'PSA 10']
    required_fields = ['avg', 'min', 'max', 'volatility']
    
    try:
        for grade in required_grades:
            if grade not in metrics:
                return False
            for field in required_fields:
                if field not in metrics[grade]:
                    return False
            # Validate numeric values
            if not all(isinstance(metrics[grade][field], (int, float)) for field in required_fields):
                return False
    except Exception:
        return False
    return True

def create_backup(filename: str = 'card_collection.csv') -> bool:
    """Create a backup of the collection file"""
    try:
        if os.path.exists(filename):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir = 'collection_backups'
            os.makedirs(backup_dir, exist_ok=True)
            
            # Create backup
            backup_path = os.path.join(backup_dir, f'collection_backup_{timestamp}.csv')
            shutil.copy2(filename, backup_path)
            
            # Clean old backups (keep last 5)
            backups = sorted([f for f in os.listdir(backup_dir) if f.startswith('collection_backup_')])
            for old_backup in backups[:-5]:
                os.remove(os.path.join(backup_dir, old_backup))
            
            return True
    except Exception as e:
        st.error(f"Backup creation failed: {str(e)}")
        return False

def validate_card_data(player_name: str, card_desc: str) -> tuple[bool, str]:
    """Validate card input data"""
    if not player_name or not card_desc:
        return False, "Player name and card description are required"
    if len(player_name) > 100 or len(card_desc) > 200:
        return False, "Input length exceeds maximum allowed"
    if any(char in player_name + card_desc for char in ['<', '>', '/', '\\']):
        return False, "Invalid characters detected"
    return True, ""

def add_to_collection(player_name: str, card_desc: str, metrics: Dict[str, Any], purchase_price: float, trend_scores: Dict[str, str]) -> Optional[pd.DataFrame]:
    """Add or update a card in the collection with enhanced validation and backup"""
    try:
        # Validate inputs
        is_valid, error_msg = validate_card_data(player_name, card_desc)
        if not is_valid:
            st.error(error_msg)
            return None
            
        # Validate metrics
        if not validate_metrics(metrics):
            st.error("Invalid metrics data structure")
            return None
            
        # Create backup before modification
        if not create_backup():
            if not st.checkbox("Continue without backup?"):
                return None
        
        # Load existing collection
        try:
            collection_df = pd.read_csv('card_collection.csv')
            # Validate loaded data
            if not all(col in collection_df.columns for col in [
                'Player', 'Card Description', 'Last Raw Price', 'Last PSA 9 Price', 
                'Last PSA 10 Price', 'Price Paid'
            ]):
                st.error("Corrupted collection file detected")
                return None
        except FileNotFoundError:
            collection_df = pd.DataFrame(columns=[
                'Player', 'Card Description', 'Last Raw Price', 'Last PSA 9 Price', 
                'Last PSA 10 Price', 'Price Paid', 'Raw ROI', 'PSA 9 ROI', 'PSA 10 ROI',
                'Raw Volatility', 'PSA 9 Volatility', 'PSA 10 Volatility',
                'Raw Trend', 'PSA 9 Trend', 'PSA 10 Trend', 'Last Updated'
            ])
        
        # Create new entry with safe value extraction
        new_entry = {
            'Player': player_name[:100],  # Limit length
            'Card Description': card_desc[:200],  # Limit length
            'Last Raw Price': float(metrics.get('Raw', {}).get('avg', 0)),
            'Last PSA 9 Price': float(metrics.get('PSA 9', {}).get('avg', 0)),
            'Last PSA 10 Price': float(metrics.get('PSA 10', {}).get('avg', 0)),
            'Price Paid': float(purchase_price),
            'Raw ROI': ((metrics.get('Raw', {}).get('avg', 0) - purchase_price) / purchase_price * 100) if purchase_price > 0 else 0,
            'PSA 9 ROI': ((metrics.get('PSA 9', {}).get('avg', 0) - purchase_price) / purchase_price * 100) if purchase_price > 0 else 0,
            'PSA 10 ROI': ((metrics.get('PSA 10', {}).get('avg', 0) - purchase_price) / purchase_price * 100) if purchase_price > 0 else 0,
            'Raw Volatility': float(metrics.get('Raw', {}).get('volatility', 0)),
            'PSA 9 Volatility': float(metrics.get('PSA 9', {}).get('volatility', 0)),
            'PSA 10 Volatility': float(metrics.get('PSA 10', {}).get('volatility', 0)),
            'Raw Trend': str(trend_scores.get('Raw', 'N/A')),
            'PSA 9 Trend': str(trend_scores.get('PSA 9', 'N/A')),
            'PSA 10 Trend': str(trend_scores.get('PSA 10', 'N/A')),
            'Last Updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Validate numeric values
        for key, value in new_entry.items():
            if key.endswith(('Price', 'ROI', 'Volatility')):
                try:
                    new_entry[key] = float(value)
                    if new_entry[key] < 0:
                        new_entry[key] = 0
                except (ValueError, TypeError):
                    new_entry[key] = 0
        
        # Update or append
        mask = (collection_df['Player'] == player_name) & \
               (collection_df['Card Description'] == card_desc)
        
        if mask.any():
            collection_df.loc[mask] = new_entry
        else:
            collection_df = pd.concat([
                collection_df,
                pd.DataFrame([new_entry])
            ], ignore_index=True)
        
        # Save with error handling
        try:
            collection_df.to_csv('card_collection.csv', index=False)
        except Exception as e:
            st.error(f"Error saving collection: {str(e)}")
            return None
        
        return collection_df
        
    except Exception as e:
        st.error(f"Error in add_to_collection: {str(e)}")
        return None

def restore_backup():
    """Restore collection from backup"""
    try:
        backup_dir = 'collection_backups'
        if not os.path.exists(backup_dir):
            st.error("No backups found")
            return False
            
        backups = sorted([f for f in os.listdir(backup_dir) if f.startswith('collection_backup_')])
        if not backups:
            st.error("No backups available")
            return False
            
        selected_backup = st.selectbox(
            "Select backup to restore",
            backups,
            format_func=lambda x: x.replace('collection_backup_', '').replace('.csv', '')
        )
        
        if st.button("Restore Selected Backup"):
            backup_path = os.path.join(backup_dir, selected_backup)
            shutil.copy2(backup_path, 'card_collection.csv')
            st.success("Backup restored successfully!")
            return True
            
    except Exception as e:
        st.error(f"Error restoring backup: {str(e)}")
        return False

def display_collection_insights(collection_df: pd.DataFrame):
    """Display collection tracking insights and projections"""
    try:
        if collection_df.empty:
            st.info("Add cards to your collection to see insights and projections.")
            return
            
        st.header("📈 Collection Insights")
        
        # Initialize analyzer for predictions
        analyzer = CardMarketAnalyzer()
        
        # Calculate current collection value
        current_value = collection_df[['Last Raw Price', 'Last PSA 9 Price', 'Last PSA 10 Price']].max(axis=1).sum()
        
        # Project future values
        projections = project_collection_value(collection_df, analyzer)
        
        if projections:
            # Display current and projected values
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Current Collection Value",
                    f"${current_value:,.2f}"
                )
            with col2:
                st.metric(
                    "Projected Value (12 Months)",
                    f"${projections['projected_value']:,.2f}",
                    f"{projections['projected_change']:+.1f}%"
                )
            
            # Show biggest gainers
            if projections.get('biggest_gainers') and len(projections['biggest_gainers']) > 0:
                st.subheader("🚀 Biggest Potential Gainers")
                for gainer in projections['biggest_gainers'][:min(3, len(projections['biggest_gainers']))]:
                    st.markdown(f"""
                    **{gainer['card']}**
                    - Current: ${gainer['current_value']:,.2f}
                    - Projected: ${gainer['projected_value']:,.2f} ({gainer['change_percent']:+.1f}%)
                    """)
            
            # ROI Projections
            st.subheader("📊 ROI Projections")
            
            # Create projection periods
            periods = [3, 6, 12]
            proj_data = []
            
            for months in periods:
                proj = project_collection_value(collection_df, analyzer, months)
                if proj:
                    proj_data.append({
                        'Period': f"{months} Months",
                        'Projected Value': proj['projected_value'],
                        'Projected Change': proj['projected_change']
                    })
            
            if proj_data:
                proj_df = pd.DataFrame(proj_data)
                
                # Create ROI chart
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=proj_df['Period'],
                    y=proj_df['Projected Change'],
                    text=proj_df['Projected Change'].apply(lambda x: f"{x:+.1f}%"),
                    textposition='auto',
                ))
                
                fig.update_layout(
                    title='Projected ROI by Time Period',
                    yaxis_title='Projected Change (%)',
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True, key="collection_insights_roi_chart")
            
            # Break-Even Analysis
            st.subheader("💰 Break-Even Analysis")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                grading_cost = st.number_input("Grading Cost ($)", value=50.0, step=5.0, key="grading_cost")
            with col2:
                shipping_cost = st.number_input("Shipping Cost ($)", value=15.0, step=1.0, key="shipping_cost")
            with col3:
                selling_fees = st.number_input("Selling Fees (%)", value=12.87, step=0.1, key="selling_fees") / 100
            
            if st.button("Calculate Break-Even Prices"):
                st.markdown("### Break-Even Analysis by Card")
                
                for _, row in collection_df.iterrows():
                    break_even = calculate_break_even(
                        row['Price Paid'],
                        grading_cost,
                        shipping_cost,
                        selling_fees
                    )
                    
                    if break_even:
                        with st.expander(f"{row['Player']} - {row['Card Description']}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"""
                                **Purchase Price:** ${row['Price Paid']:,.2f}
                                **Break-Even Price:** ${break_even['break_even_price']:,.2f}
                                **Total Costs:** ${break_even['total_costs']:,.2f}
                                """)
                            with col2:
                                st.markdown(f"""
                                **Cost Breakdown:**
                                - Grading: ${break_even['cost_breakdown']['grading_cost']:,.2f}
                                - Shipping: ${break_even['cost_breakdown']['shipping_cost']:,.2f}
                                - Selling Fees: ${break_even['cost_breakdown']['selling_fees']:,.2f}
                                """)
            
            # Monthly Change Alert
            last_month = pd.Timestamp.now() - pd.DateOffset(months=1)
            recent_cards = collection_df[pd.to_datetime(collection_df['Last Updated']) > last_month]
            
            if not recent_cards.empty:
                monthly_change = ((current_value - recent_cards['Price Paid'].sum()) / recent_cards['Price Paid'].sum() * 100)
                
                if abs(monthly_change) >= 5:  # Alert for significant changes
                    alert_message = f"""
                    📢 Your collection's value {'increased' if monthly_change > 0 else 'decreased'} by {abs(monthly_change):.1f}% this month.
                    """
                    
                    if projections.get('biggest_gainers') and len(projections['biggest_gainers']) > 0:
                        top_gainer = projections['biggest_gainers'][0]
                        alert_message += f"""
                        Biggest gainer: {top_gainer['card']} ({top_gainer['change_percent']:+.1f}%)
                        {'Consider selling now!' if top_gainer['change_percent'] > 15 else 'Monitor for selling opportunities.'}
                        """
                    
                    st.info(alert_message)
        
    except Exception as e:
        st.error(f"Error displaying collection insights: {e}")
        logger.error(f"Detailed error in collection insights: {e}")

def display_collection_manager(metrics=None, purchase_price=0):
    """Display and manage the collection with enhanced tracking features"""
    if metrics is None:
        metrics = {}
        
    st.markdown("---")
    st.header("📚 Collection Manager")
    
    # Load existing collection
    try:
        df = pd.read_csv('card_collection.csv')
    except FileNotFoundError:
        df = pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading collection: {str(e)}")
        df = pd.DataFrame()
    
    # Add collection insights if there are cards in the collection
    if not df.empty:
        display_collection_insights(df)
    
    # Add to collection section
    with st.expander("➕ Add Current Card to Collection", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            player_name = st.text_input("Player Name", key="collection_manager_player_input")
        with col2:
            card_desc = st.text_input("Card Description", key="collection_manager_desc_input")
        
        if st.button("Add to Collection", key="collection_manager_add_button"):
            if player_name and card_desc:
                if not metrics:
                    st.error("Please perform a card analysis first")
                    return
                
                # Calculate trend scores
                trend_scores = {}
                for grade in ['Raw', 'PSA 9', 'PSA 10']:
                    if grade in metrics:
                        trend_scores[grade] = '📈' if metrics[grade]['avg'] > metrics[grade].get('min', 0) else '📉'
                
                result = add_to_collection(player_name, card_desc, metrics, purchase_price, trend_scores)
                if result is not None:
                    st.success("Card added to collection!")
                    df = result
            else:
                st.error("Please enter both player name and card description")
    
    # Display collection
    if not df.empty:
        st.subheader("Your Collection")
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            search = st.text_input("🔍 Search Collection", "", key="collection_manager_search_input")
        with col2:
            sort_by = st.selectbox(
                "Sort By",
                ["Last Updated", "Player", "PSA 10 ROI", "PSA 10 Price"],
                key="collection_manager_sort_select"
            )
        
        try:
            # Filter and sort collection
            if search:
                mask = df['Player'].str.contains(search, case=False) | \
                       df['Card Description'].str.contains(search, case=False)
                df = df[mask]
            
            df = df.sort_values(sort_by, ascending=False)
            
            # Display collection with formatting
            st.dataframe(
                df.style
                .format({
                    'Last Raw Price': '${:,.2f}',
                    'Last PSA 9 Price': '${:,.2f}',
                    'Last PSA 10 Price': '${:,.2f}',
                    'Price Paid': '${:,.2f}',
                    'Raw ROI': '{:,.1f}%',
                    'PSA 9 ROI': '{:,.1f}%',
                    'PSA 10 ROI': '{:,.1f}%',
                    'Raw Volatility': '{:,.1f}%',
                    'PSA 9 Volatility': '{:,.1f}%',
                    'PSA 10 Volatility': '{:,.1f}%'
                })
                .background_gradient(subset=['PSA 10 ROI'], cmap='RdYlGn')
                .set_properties(**{
                    'background-color': 'rgba(255, 255, 255, 0.1)',
                    'color': 'white'
                })
            )
        except Exception as e:
            st.error(f"Error displaying collection: {str(e)}")
        
        # Export and management options
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Download Collection (CSV)", key="download_collection_button"):
                try:
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "Click to Download",
                        csv,
                        "card_collection.csv",
                        "text/csv",
                        key='download_csv_button'
                    )
                except Exception as e:
                    st.error(f"Error downloading collection: {str(e)}")
                    
        with col2:
            if st.button("🗑️ Clear Collection", key="clear_collection_button"):
                try:
                    if os.path.exists('card_collection.csv'):
                        os.remove('card_collection.csv')
                    st.success("Collection cleared!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clearing collection: {str(e)}")
        
        # Remove individual cards
        with st.expander("🗑️ Remove Cards"):
            to_remove = st.multiselect(
                "Select cards to remove",
                df.apply(lambda x: f"{x['Player']} - {x['Card Description']}", axis=1),
                key="remove_cards_multiselect"
            )
            if to_remove and st.button("Remove Selected Cards", key="remove_selected_cards_button"):
                try:
                    for card in to_remove:
                        player, desc = card.split(" - ", 1)
                        mask = (df['Player'] == player) & \
                               (df['Card Description'] == desc)
                        df = df[~mask]
                    df.to_csv('card_collection.csv', index=False)
                    st.success("Selected cards removed!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error removing cards: {str(e)}")

    # Add backup management
    with st.expander("💾 Backup Management"):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Create Backup", key="create_backup_button"):
                if create_backup():
                    st.success("Backup created successfully!")
        with col2:
            restore_backup()

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
        if not isinstance(sales_data, pd.DataFrame) or sales_data.empty:
            st.warning("No sales data available for analysis")
            return

        # Initialize market analyzer
        analyzer = CardMarketAnalyzer()
        
        # Current price metrics
        current_price = sales_data['Sale Price'].iloc[-1]
        week_ago_price = sales_data['Sale Price'].iloc[-7] if len(sales_data) > 7 else sales_data['Sale Price'].iloc[0]
        price_change = ((current_price - week_ago_price) / week_ago_price * 100)
        
        # Check for significant price movements
        if abs(price_change) >= 15:
            st.warning(f"""🚨 Alert: {card_name} has {'increased' if price_change > 0 else 'decreased'} by {abs(price_change):.1f}% in the past 7 days.
            {'Consider selling based on current market conditions.' if price_change > 0 else 'Potential buying opportunity.'}""")
        
        # Display current price and weekly change
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Price", f"${current_price:.2f}", 
                     f"{price_change:+.1f}% (7d)")
        
        # Prepare data for prediction
        prediction_data = sales_data.copy()
        prediction_data['Date'] = pd.to_datetime(prediction_data['Sale Date'])
        prediction_data['Price'] = prediction_data['Sale Price']
        
        # Generate price predictions
        predictions, pred_std, confidence = analyzer.predict_prices(prediction_data)
        
        if predictions is not None:
            # Create prediction chart
            fig = go.Figure()
            
            # Historical prices
            fig.add_trace(go.Scatter(
                x=prediction_data['Date'],
                y=prediction_data['Price'],
                name='Historical Prices',
                mode='lines+markers'
            ))
            
            # Predictions
            future_dates = pd.date_range(
                start=prediction_data['Date'].iloc[-1], 
                periods=len(predictions) + 1
            )[1:]
            
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=predictions,
                name='Predicted Price',
                line=dict(dash='dash')
            ))
            
            # Confidence intervals
            fig.add_trace(go.Scatter(
                x=future_dates.tolist() + future_dates.tolist()[::-1],
                y=(predictions + 2*pred_std).tolist() + 
                  (predictions - 2*pred_std).tolist()[::-1],
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% Confidence Interval'
            ))
            
            fig.update_layout(
                title='Price History and 30-Day Forecast',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display prediction metrics
            pred_price = predictions[-1]
            pred_change = ((pred_price - current_price) / current_price * 100)
            
            with col2:
                st.metric("30-Day Forecast", f"${pred_price:.2f}", 
                         f"{pred_change:+.1f}% (Confidence: {confidence*100:.1f}%)")
        
        # Market sentiment analysis
        sentiment = analyzer.analyze_sentiment(card_name, sales_data['Sale Price'].values)
        
        if sentiment:
            st.subheader("📊 Market Analysis")
            
            # Display sentiment metrics
            cols = st.columns(3)
            
            with cols[0]:
                st.markdown(f"""
                **Price Trend:**
                <p style='color: {'green' if sentiment['price_trend'] == 'Bullish' else 'red' if sentiment['price_trend'] == 'Bearish' else 'orange'}; font-size: 20px;'>
                {sentiment['price_trend']} ({sentiment['price_change']:+.1f}%)
                </p>
                """, unsafe_allow_html=True)
            
            with cols[1]:
                st.markdown(f"""
                **Market Sentiment:**
                <p style='color: {'green' if sentiment['sentiment'] == 'Positive' else 'red' if sentiment['sentiment'] == 'Negative' else 'orange'}; font-size: 20px;'>
                {sentiment['sentiment']}
                </p>
                """, unsafe_allow_html=True)
            
            with cols[2]:
                st.markdown(f"""
                **Recommendation:**
                <p style='color: blue; font-size: 20px;'>
                {sentiment['recommendation']}
                </p>
                """, unsafe_allow_html=True)
                
    except Exception as e:
        st.error(f"Error in analysis: {str(e)}")
        logger.error(f"Detailed error in display_predictions_and_alerts: {str(e)}")

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

def analyze_trade(card1_data: Dict[str, Any], card2_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze a potential trade between two cards"""
    try:
        analysis = {
            'fair_trade': False,
            'recommendation': '',
            'value_difference': 0,
            'risk_assessment': '',
            'future_outlook': '',
            'detailed_metrics': {}
        }

        # Extract current values (use PSA 10 as default)
        card1_value = max(
            card1_data['Last PSA 10 Price'],
            card1_data['Last PSA 9 Price'],
            card1_data['Last Raw Price']
        )
        card2_value = max(
            card2_data['Last PSA 10 Price'],
            card2_data['Last PSA 9 Price'],
            card2_data['Last Raw Price']
        )

        # Calculate value difference
        value_diff = card1_value - card2_value
        value_diff_pct = (value_diff / card1_value) * 100

        # Compare volatility
        card1_volatility = max(
            card1_data['Raw Volatility'],
            card1_data['PSA 9 Volatility'],
            card1_data['PSA 10 Volatility']
        )
        card2_volatility = max(
            card2_data['Raw Volatility'],
            card2_data['PSA 9 Volatility'],
            card2_data['PSA 10 Volatility']
        )

        # Compare ROI potential
        card1_roi = max(
            card1_data['Raw ROI'],
            card1_data['PSA 9 ROI'],
            card1_data['PSA 10 ROI']
        )
        card2_roi = max(
            card2_data['Raw ROI'],
            card2_data['PSA 9 ROI'],
            card2_data['PSA 10 ROI']
        )

        # Store detailed metrics
        analysis['detailed_metrics'] = {
            'card1': {
                'value': card1_value,
                'volatility': card1_volatility,
                'roi_potential': card1_roi
            },
            'card2': {
                'value': card2_value,
                'volatility': card2_volatility,
                'roi_potential': card2_roi
            }
        }

        # Determine if trade is fair (within 10% value difference)
        analysis['fair_trade'] = abs(value_diff_pct) <= 10

        # Generate risk assessment
        if card1_volatility < card2_volatility:
            analysis['risk_assessment'] = "Trading for a more volatile card"
        elif card1_volatility > card2_volatility:
            analysis['risk_assessment'] = "Trading for a more stable card"
        else:
            analysis['risk_assessment'] = "Similar volatility levels"

        # Generate future outlook
        if card2_roi > card1_roi:
            analysis['future_outlook'] = "Positive - Higher growth potential"
        elif card2_roi < card1_roi:
            analysis['future_outlook'] = "Negative - Lower growth potential"
        else:
            analysis['future_outlook'] = "Neutral - Similar growth potential"

        # Generate recommendation
        recommendation = []
        if analysis['fair_trade']:
            recommendation.append("✅ Fair value trade")
        else:
            recommendation.append("⚠️ Value mismatch")
            if value_diff > 0:
                recommendation.append(f"You're giving up ${abs(value_diff):.2f} in value")
            else:
                recommendation.append(f"You're gaining ${abs(value_diff):.2f} in value")

        if card2_roi > card1_roi:
            recommendation.append("📈 Better long-term potential")
        if card2_volatility < card1_volatility:
            recommendation.append("🛡️ Lower risk profile")

        analysis['recommendation'] = " | ".join(recommendation)

        return analysis

    except Exception as e:
        logger.error(f"Error in trade analysis: {e}")
        return None

def display_trade_analyzer():
    """Display the trade analysis interface"""
    st.header("🔄 Trade Analyzer")
    
    try:
        # Load collection
        try:
            collection_df = pd.read_csv('card_collection.csv')
            if collection_df.empty:
                st.warning("Your collection is empty. Add some cards first!")
                return
        except FileNotFoundError:
            st.warning("Please add cards to your collection first to use the Trade Analyzer")
            return
        except Exception as e:
            st.error(f"Error loading collection: {str(e)}")
            return

        # Create card selection interface
        col1, col2 = st.columns(2)
        
        # Prepare card options
        card_options = collection_df.apply(lambda x: f"{x['Player']} - {x['Card Description']}", axis=1).tolist()
        
        with col1:
            st.subheader("Your Card")
            your_card = st.selectbox(
                "Select your card to trade",
                options=card_options,
                key="your_card_select"
            )

        with col2:
            st.subheader("Their Card")
            their_card = st.selectbox(
                "Select card you're trading for",
                options=card_options,
                key="their_card_select"
            )

        if your_card == their_card:
            st.warning("Please select different cards to compare")
            return

        if st.button("Analyze Trade", key="analyze_trade_button"):
            if not your_card or not their_card:
                st.warning("Please select both cards to analyze the trade")
                return
                
            try:
                # Get card data
                your_card_player, your_card_desc = your_card.split(" - ", 1)
                their_card_player, their_card_desc = their_card.split(" - ", 1)

                your_card_data = collection_df[
                    (collection_df['Player'] == your_card_player) & 
                    (collection_df['Card Description'] == your_card_desc)
                ].iloc[0]

                their_card_data = collection_df[
                    (collection_df['Player'] == their_card_player) & 
                    (collection_df['Card Description'] == their_card_desc)
                ].iloc[0]

                # Analyze trade
                analysis = analyze_trade(your_card_data, their_card_data)

                if analysis:
                    # Display trade analysis
                    st.markdown("### Trade Analysis")

                    # Value Comparison
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "Your Card Value",
                            f"${analysis['detailed_metrics']['card1']['value']:,.2f}",
                            f"{analysis['detailed_metrics']['card1']['roi_potential']:+.1f}% ROI"
                        )
                    with col2:
                        st.metric(
                            "Their Card Value",
                            f"${analysis['detailed_metrics']['card2']['value']:,.2f}",
                            f"{analysis['detailed_metrics']['card2']['roi_potential']:+.1f}% ROI"
                        )

                    # Trade Recommendation
                    st.markdown(f"""
                    ### Recommendation
                    {analysis['recommendation']}

                    **Risk Assessment:** {analysis['risk_assessment']}  
                    **Future Outlook:** {analysis['future_outlook']}
                    """)

                    # Detailed Metrics Comparison
                    st.markdown("### Detailed Comparison")
                    comparison_df = pd.DataFrame({
                        'Metric': ['Current Value', 'Volatility', 'ROI Potential'],
                        'Your Card': [
                            f"${analysis['detailed_metrics']['card1']['value']:,.2f}",
                            f"{analysis['detailed_metrics']['card1']['volatility']:.1f}%",
                            f"{analysis['detailed_metrics']['card1']['roi_potential']:+.1f}%"
                        ],
                        'Their Card': [
                            f"${analysis['detailed_metrics']['card2']['value']:,.2f}",
                            f"{analysis['detailed_metrics']['card2']['volatility']:.1f}%",
                            f"{analysis['detailed_metrics']['card2']['roi_potential']:+.1f}%"
                        ]
                    })
                    
                    st.dataframe(comparison_df)

                    # Add trade insights
                    with st.expander("📈 Trade Insights"):
                        st.markdown(f"""
                        **Value Analysis:**
                        - {'Fair trade in terms of value' if analysis['fair_trade'] else 'Significant value difference'}
                        - {'Consider requesting additional value to balance the trade' if not analysis['fair_trade'] else ''}

                        **Risk Profile:**
                        - Volatility: {analysis['risk_assessment']}
                        - {"Consider if you're comfortable with the increased risk" if 'more volatile' in analysis['risk_assessment'] else ''}

                        **Growth Potential:**
                        - {analysis['future_outlook']}
                        - ROI Difference: {analysis['detailed_metrics']['card2']['roi_potential'] - analysis['detailed_metrics']['card1']['roi_potential']:+.1f}%
                        """)
                else:
                    st.error("Unable to analyze trade. Please try again.")
                    
            except Exception as e:
                st.error(f"Error analyzing trade: {str(e)}")
                logger.error(f"Detailed error in trade analysis: {str(e)}")

    except Exception as e:
        st.error(f"Error in trade analyzer: {str(e)}")
        logger.error(f"Detailed error in trade analyzer: {str(e)}")

def display_market_analysis(raw_sales, psa9_sales, psa10_sales, metrics, card_name):
    """Display comprehensive market analysis for a card"""
    try:
        st.header("📊 Market Analysis")
        
        # Display price metrics for each grade
        cols = st.columns(3)
        grades = [("Raw", raw_sales), ("PSA 9", psa9_sales), ("PSA 10", psa10_sales)]
        
        for i, (grade, sales) in enumerate(grades):
            with cols[i]:
                if grade in metrics:
                    st.metric(
                        f"{grade} Market Price",
                        f"${metrics[grade]['avg']:.2f}",
                        f"±${metrics[grade]['std']:.2f}"
                    )
                    
                    # Price range
                    st.markdown(f"""
                    **Price Range:**
                    - High: ${metrics[grade]['max']:.2f}
                    - Low: ${metrics[grade]['min']:.2f}
                    - Volatility: {metrics[grade]['volatility']:.1f}%
                    """)
                else:
                    st.warning(f"No {grade} data available")
        
        # Display sales data tables
        st.subheader("Recent Sales Data")
        tabs = st.tabs(["Raw", "PSA 9", "PSA 10"])
        
        for tab, (grade, sales) in zip(tabs, grades):
            with tab:
                if not sales.empty:
                    st.dataframe(
                        sales.style.format({
                            'Sale Price': '${:.2f}'
                        }),
                        hide_index=True
                    )
                else:
                    st.info(f"No recent {grade} sales data available")
        
        # Market Trends and Predictions
        st.subheader("📈 Market Trends")
        
        # Create combined price trend chart
        fig = go.Figure()
        
        for grade, sales in grades:
            if not sales.empty:
                fig.add_trace(go.Scatter(
                    x=pd.to_datetime(sales['Sale Date']),
                    y=sales['Sale Price'],
                    name=f"{grade} Sales",
                    mode='lines+markers'
                ))
        
        fig.update_layout(
            title=f'Price Trends - {card_name}',
            xaxis_title='Date',
            yaxis_title='Sale Price ($)',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True, key="market_analysis_trend_chart")
        
        # Display detailed predictions and alerts
        for grade, sales in grades:
            if not sales.empty:
                with st.expander(f"{grade} Detailed Analysis"):
                    display_predictions_and_alerts(sales, f"{card_name} ({grade})")
        
        # Market Health Score
        st.subheader("🏥 Market Health")
        
        # Calculate market health metrics
        health_metrics = {}
        for grade, sales in grades:
            if grade in metrics:
                # Calculate various health indicators
                volatility = metrics[grade]['volatility']
                price_stability = max(0, 100 - volatility)
                sales_volume = len(sales) if not sales.empty else 0
                volume_score = min(100, sales_volume * 5)  # 20 sales = 100%
                
                # Combine into overall health score
                health_score = (price_stability * 0.7 + volume_score * 0.3)
                
                health_metrics[grade] = {
                    'health_score': health_score,
                    'stability': price_stability,
                    'volume_score': volume_score
                }
        
        # Display health scores
        cols = st.columns(len(grades))
        for i, (grade, _) in enumerate(grades):
            with cols[i]:
                if grade in health_metrics:
                    metrics = health_metrics[grade]
                    st.metric(
                        f"{grade} Market Health",
                        f"{metrics['health_score']:.1f}%",
                        f"Volume: {metrics['volume_score']:.1f}%"
                    )
                    
                    # Health status
                    status = "Healthy" if metrics['health_score'] >= 70 else \
                            "Stable" if metrics['health_score'] >= 50 else \
                            "Volatile"
                    
                    st.markdown(f"""
                    **Status:** {status}
                    - Price Stability: {metrics['stability']:.1f}%
                    """)
                else:
                    st.warning(f"No health data for {grade}")
        
        # Investment Recommendations
        st.subheader("💡 Investment Insights")
        
        for grade, sales in grades:
            if grade in metrics and not sales.empty:
                with st.expander(f"{grade} Investment Analysis"):
                    # Calculate metrics for recommendations
                    volatility = metrics[grade]['volatility']
                    avg_price = metrics[grade]['avg']
                    price_trend = (sales['Sale Price'].iloc[-1] - sales['Sale Price'].iloc[0]) / \
                                sales['Sale Price'].iloc[0] * 100 if len(sales) > 1 else 0
                    
                    # Generate recommendations
                    st.markdown(f"""
                    **Price Trend:** {price_trend:+.1f}% over available data
                    **Market Volatility:** {volatility:.1f}%
                    
                    **Recommendation:**
                    {generate_investment_recommendation(volatility, price_trend, avg_price)}
                    """)
        
    except Exception as e:
        st.error(f"Error in market analysis display: {str(e)}")
        logger.error(f"Detailed error in display_market_analysis: {str(e)}")

def display_profit_analysis(metrics, purchase_price):
    """Display profit analysis for different grades"""
    try:
        st.subheader("💰 Profit Analysis")
        
        # Calculate profit metrics for each grade
        profit_data = []
        for grade in ['Raw', 'PSA 9', 'PSA 10']:
            if grade in metrics:
                avg_price = metrics[grade]['avg']
                profit = avg_price - purchase_price
                roi = (profit / purchase_price * 100) if purchase_price > 0 else 0
                
                profit_data.append({
                    'Grade': grade,
                    'Market Price': avg_price,
                    'Profit': profit,
                    'ROI': roi
                })
        
        if profit_data:
            # Create profit comparison chart
            fig = go.Figure()
            
            # Add ROI bars
            fig.add_trace(go.Bar(
                x=[data['Grade'] for data in profit_data],
                y=[data['ROI'] for data in profit_data],
                name='ROI (%)',
                text=[f"{roi:+.1f}%" for roi in [data['ROI'] for data in profit_data]],
                textposition='auto',
            ))
            
            fig.update_layout(
                title='ROI Comparison by Grade',
                yaxis_title='Return on Investment (%)',
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True, key="profit_analysis_roi_chart")
            
            # Display detailed metrics
            cols = st.columns(len(profit_data))
            for i, data in enumerate(profit_data):
                with cols[i]:
                    st.metric(
                        f"{data['Grade']} Profit",
                        f"${data['Profit']:,.2f}",
                        f"{data['ROI']:+.1f}% ROI"
                    )
            
            # Break-even analysis
            st.subheader("📊 Break-Even Analysis")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                grading_cost = st.number_input("Grading Cost ($)", value=50.0, step=5.0)
            with col2:
                shipping_cost = st.number_input("Shipping Cost ($)", value=15.0, step=1.0)
            with col3:
                selling_fees = st.number_input("Selling Fees (%)", value=12.87, step=0.1) / 100
            
            break_even = calculate_break_even(purchase_price, grading_cost, shipping_cost, selling_fees)
            
            if break_even:
                st.markdown(f"""
                ### Break-Even Analysis
                - Purchase Price: ${purchase_price:,.2f}
                - Break-Even Price: ${break_even['break_even_price']:,.2f}
                - Total Costs: ${break_even['total_costs']:,.2f}
                
                **Cost Breakdown:**
                - Grading: ${break_even['cost_breakdown']['grading_cost']:,.2f}
                - Shipping: ${break_even['cost_breakdown']['shipping_cost']:,.2f}
                - Selling Fees: ${break_even['cost_breakdown']['selling_fees']:,.2f}
                """)
        
    except Exception as e:
        st.error(f"Error in profit analysis: {str(e)}")
        logger.error(f"Detailed error in display_profit_analysis: {str(e)}")

def main():
    st.title("Sports Card Market Analysis")
    
    # Initialize session state for metrics if not exists
    if 'search_metrics' not in st.session_state:
        st.session_state.search_metrics = {}
    if 'search_purchase_price' not in st.session_state:
        st.session_state.search_purchase_price = 0
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "Search & Analysis"
    
    # Create tabs for different features
    tab1, tab2, tab3 = st.tabs(["Search & Analysis", "Collection Manager", "Trade Analyzer"])
    
    with tab1:
        st.session_state.current_tab = "Search & Analysis"
        # Search interface
        search_type = st.radio("Search Type", ["Text Search", "Image Search"], key="search_type_radio")
        
        if search_type == "Text Search":
            query = st.text_input("Enter card details (e.g., '2020 Prizm Justin Herbert RC'):", key="text_search_query")
            if query:
                run_text_search(query)
        else:
            uploaded_file = st.file_uploader("Upload card image", type=["jpg", "jpeg", "png"], key="image_upload")
            if uploaded_file:
                run_image_search(uploaded_file)
    
    with tab2:
        st.session_state.current_tab = "Collection Manager"
        # Only pass metrics if we're coming from search
        if st.session_state.get('search_metrics'):
            display_collection_manager(
                metrics=st.session_state.search_metrics,
                purchase_price=st.session_state.search_purchase_price
            )
        else:
            display_collection_manager()
    
    with tab3:
        st.session_state.current_tab = "Trade Analyzer"
        display_trade_analyzer()

if __name__ == "__main__":
    main()

