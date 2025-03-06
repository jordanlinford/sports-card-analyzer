import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import requests
import os
from dotenv import load_dotenv
import random
import math
import plotly.graph_objects as go

# Load environment variables
load_dotenv()

def get_card_image(query):
    """Get card image using Google Custom Search API"""
    try:
        # Google Custom Search API configuration
        GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
        GOOGLE_CSE_ID = os.getenv('GOOGLE_CSE_ID')
        
        # Validate credentials
        if not GOOGLE_API_KEY or GOOGLE_API_KEY == 'your_api_key_here':
            st.error("⚠️ Invalid Google API Key. Please check your .env file.")
            st.info("To set up the Google Custom Search API:\n" +
                   "1. Enable the API at: https://console.cloud.google.com/apis/library/customsearch.googleapis.com\n" +
                   "2. Create credentials at: https://console.cloud.google.com/apis/credentials\n" +
                   "3. Set up a Custom Search Engine at: https://programmablesearchengine.google.com/\n" +
                   "4. Update your .env file with the credentials")
            return None
            
        if not GOOGLE_CSE_ID or GOOGLE_CSE_ID == 'your_search_engine_id_here':
            st.error("⚠️ Invalid Search Engine ID. Please check your .env file.")
            return None
        
        # Clean and enhance the search query
        search_terms = [
            f'"{query}"',  # Exact match for the main query
            "sports card",
            "trading card",
            "front view",
            "-reprint",
            "-lot",
            "-case",
            "-box",
            "-pack",
            "-break",
            "high resolution"
        ]
        enhanced_query = " ".join(search_terms)
        
        # Build the search URL
        search_url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'q': enhanced_query,
            'key': GOOGLE_API_KEY,
            'cx': GOOGLE_CSE_ID,
            'searchType': 'image',
            'num': 1,
            'imgSize': 'LARGE',
            'imgType': 'photo',
            'safe': 'active'
        }
        
        st.info("🔍 Searching for card image...")
        
        response = requests.get(search_url, params=params, timeout=10)
        
        if response.status_code == 403:
            st.error("⚠️ API Access Error (403)")
            st.info("Please ensure you have:\n" +
                   "1. Enabled the Custom Search API\n" +
                   "2. Created valid API credentials\n" +
                   "3. Set up a Custom Search Engine\n" +
                   "4. Waited a few minutes after enabling the API")
            return None
        
        if response.status_code == 200:
            data = response.json()
            
            if 'items' in data and len(data['items']) > 0:
                image_url = data['items'][0]['link']
                
                # Verify image URL is accessible
                try:
                    img_response = requests.head(image_url, timeout=5)
                    if img_response.status_code == 200:
                        return image_url
                    else:
                        st.warning(f"Found image URL but couldn't access it (Status: {img_response.status_code})")
                except:
                    st.warning("Found image URL but couldn't verify access")
                    return image_url  # Return URL anyway as it might still work
            else:
                st.warning("No images found for this card")
                if 'error' in data:
                    st.error(f"API Error: {data['error'].get('message', 'Unknown error')}")
        else:
            st.error(f"API request failed (Status: {response.status_code})")
            try:
                error_data = response.json()
                if 'error' in error_data:
                    st.error(f"Error details: {error_data['error'].get('message', 'Unknown error')}")
            except:
                st.error(f"Raw error response: {response.text[:500]}")
        
        return None
    
    except requests.Timeout:
        st.error("Request timed out while fetching image")
        return None
    except requests.RequestException as e:
        st.error(f"Network error while fetching image: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Unexpected error while fetching image: {str(e)}")
        return None

def format_currency(value):
    """Format value as currency"""
    return f"${value:,.2f}"

def calculate_metrics(data):
    """Calculate key metrics from price data"""
    metrics = {
        'mean': data['Price'].mean(),
        'median': data['Price'].median(),
        'min': data['Price'].min(),
        'max': data['Price'].max(),
        'std': data['Price'].std(),
        'count': len(data),
        'last_price': data['Price'].iloc[-1],
        'week_trend': data['Price'].iloc[-7:].mean() - data['Price'].iloc[-14:-7].mean(),
        'volatility': data['Price'].std() / data['Price'].mean() * 100  # Coefficient of variation
    }
    return metrics

def calculate_profit_metrics(current_price, purchase_price):
    """Calculate profit metrics"""
    profit = current_price - purchase_price
    roi = (profit / purchase_price * 100) if purchase_price > 0 else 0
    return profit, roi

def calculate_grading_multiplier(current_price):
    """Calculate dynamic grading multipliers based on card value with enhanced precision"""
    # Value ranges and corresponding multipliers
    if current_price < 20:
        return {
            'psa9': 2.5,
            'psa10': 5.0,
            'grading_recommendation': '⚠️ Grading may not be cost-effective at this price point'
        }
    elif current_price < 50:
        return {
            'psa9': 2.8,
            'psa10': 5.5,
            'grading_recommendation': '📊 Consider grading only if confident of PSA 10 potential'
        }
    elif current_price < 100:
        return {
            'psa9': 2.3,
            'psa10': 4.5,
            'grading_recommendation': '📈 Grading can be profitable with PSA 9+ potential'
        }
    elif current_price < 250:
        return {
            'psa9': 2.0,
            'psa10': 4.0,
            'grading_recommendation': '✨ Good candidate for grading if condition is excellent'
        }
    elif current_price < 500:
        return {
            'psa9': 1.8,
            'psa10': 3.5,
            'grading_recommendation': '🎯 Professional grading recommended for value protection'
        }
    elif current_price < 1000:
        return {
            'psa9': 1.6,
            'psa10': 3.0,
            'grading_recommendation': '🔒 Grading essential for authentication and preservation'
        }
    else:
        return {
            'psa9': 1.5,
            'psa10': 2.5,
            'grading_recommendation': '💎 Premium card - professional grading and authentication crucial'
        }

def analyze_market_trends(price_data, volume_data):
    """Analyze market trends with enhanced metrics"""
    trends = {
        'momentum': 0,
        'support_levels': [],
        'resistance_levels': [],
        'market_sentiment': 'NEUTRAL'
    }
    
    if len(price_data) < 7:
        return trends
    
    # Calculate momentum using RSI
    price_changes = price_data.diff()
    gains = price_changes.clip(lower=0)
    losses = -price_changes.clip(upper=0)
    avg_gain = gains.rolling(window=14, min_periods=1).mean()
    avg_loss = losses.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, 0.00001)  # Prevent division by zero
    rsi = 100 - (100 / (1 + rs))
    trends['momentum'] = rsi.iloc[-1]
    
    # Identify support and resistance levels
    window = 20
    rolling_min = price_data.rolling(window=window, center=True).min()
    rolling_max = price_data.rolling(window=window, center=True).max()
    
    trends['support_levels'] = [
        price for price in rolling_min.unique() 
        if price > price_data.iloc[-1] * 0.8 
        and price < price_data.iloc[-1] * 1.2
    ][:3]  # Top 3 support levels
    
    trends['resistance_levels'] = [
        price for price in rolling_max.unique() 
        if price > price_data.iloc[-1] * 0.8 
        and price < price_data.iloc[-1] * 1.2
    ][:3]  # Top 3 resistance levels
    
    # Determine market sentiment
    recent_trend = price_data.iloc[-7:].mean() - price_data.iloc[-14:-7].mean()
    volume_trend = volume_data.iloc[-7:].mean() - volume_data.iloc[-14:-7].mean()
    
    if recent_trend > 0 and volume_trend > 0:
        trends['market_sentiment'] = 'BULLISH'
    elif recent_trend < 0 and volume_trend < 0:
        trends['market_sentiment'] = 'BEARISH'
    elif recent_trend > 0 and volume_trend < 0:
        trends['market_sentiment'] = 'CAUTIOUSLY BULLISH'
    elif recent_trend < 0 and volume_trend > 0:
        trends['market_sentiment'] = 'CAUTIOUSLY BEARISH'
    
    return trends

def calculate_market_health(price_data, volume_data, grade=None):
    """Enhanced market health calculation"""
    if len(price_data) < 30:
        return {
            'score': 50,
            'indicators': ['Insufficient data for market health analysis']
        }
    
    # Initialize score and indicators
    score = 50
    indicators = []
    
    # Adjust volatility thresholds based on grade
    volatility_thresholds = {
        'PSA 10': {'high': 25, 'low': 8},  # More stable prices
        'PSA 9': {'high': 30, 'low': 10},
        'BGS 9.5': {'high': 28, 'low': 9},
        'Raw': {'high': 35, 'low': 12},     # More volatile
        None: {'high': 30, 'low': 10}       # Default thresholds
    }
    
    thresholds = volatility_thresholds.get(grade, volatility_thresholds[None])
    
    # Calculate price metrics with increased weight on recent data
    recent_prices = price_data[-30:]
    very_recent_prices = price_data[-7:]
    price_trend = (very_recent_prices.mean() - price_data[:-7].mean()) / price_data[:-7].mean() * 100
    price_volatility = recent_prices.std() / recent_prices.mean() * 100
    
    # Calculate moving averages with exponential weighting
    ma7 = price_data[-7:].mean()
    ma30 = price_data[-30:].mean()
    ma90 = price_data[-90:].mean() if len(price_data) >= 90 else price_data.mean()
    
    # Volume analysis with increased weight (30% of total score)
    recent_volume = len(price_data[-7:])
    past_volume = max(1, len(price_data[-14:-7]))  # Prevent division by zero
    volume_change = ((recent_volume - past_volume) / past_volume * 100)
    
    # Enhanced trend analysis (40% of total score)
    if price_trend > 15:
        score += 25
        indicators.append('🚀 Strong bullish trend')
    elif price_trend > 7:
        score += 15
        indicators.append('📈 Positive price momentum')
    elif price_trend < -15:
        score -= 25
        indicators.append('📉 Significant price decline')
    elif price_trend < -7:
        score -= 15
        indicators.append('⚠️ Negative price pressure')
    
    # Moving average analysis (15% of total score)
    if ma7 > ma30 and ma30 > ma90:
        score += 10
        indicators.append('📊 Strong upward trend')
    elif ma7 < ma30 and ma30 < ma90:
        score -= 10
        indicators.append('📊 Downward trend')
    
    # Grade-specific volatility analysis (15% of total score)
    if price_volatility > thresholds['high']:
        score -= 15
        indicators.append('⚡ High volatility risk')
    elif price_volatility > (thresholds['high'] + thresholds['low']) / 2:
        score -= 8
        indicators.append('📊 Above average volatility')
    elif price_volatility < thresholds['low']:
        score += 10
        indicators.append('🎯 Stable price action')
    
    # Volume trend analysis with increased impact
    if volume_change > 20:
        score += 15
        indicators.append('📈 Increasing market activity')
    elif volume_change < -20:
        score -= 15
        indicators.append('📉 Decreasing market activity')
    
    # Price stability check with grade-specific thresholds
    stability_thresholds = {
        'PSA 10': {'stable': 12, 'volatile': 35},
        'PSA 9': {'stable': 15, 'volatile': 40},
        'BGS 9.5': {'stable': 13, 'volatile': 38},
        'Raw': {'stable': 18, 'volatile': 45},
        None: {'stable': 15, 'volatile': 40}
    }
    
    stability = stability_thresholds.get(grade, stability_thresholds[None])
    price_range = (price_data.max() - price_data.min()) / price_data.mean() * 100
    
    if price_range < stability['stable']:
        score += 10
        indicators.append('🎯 Price consolidation')
    elif price_range > stability['volatile']:
        score -= 10
        indicators.append('⚠️ Wide price swings')
    
    # Add trend analysis
    market_trends = analyze_market_trends(price_data, volume_data)
    
    # Adjust score based on momentum
    if market_trends['momentum'] > 70:
        score += 10
        indicators.append('💪 Strong momentum')
    elif market_trends['momentum'] < 30:
        score -= 10
        indicators.append('⚠️ Weak momentum')
    
    # Add support/resistance analysis
    if price_data.iloc[-1] < min(market_trends['support_levels'] or [float('inf')]):
        indicators.append('📉 Below support level')
        score -= 5
    elif price_data.iloc[-1] > max(market_trends['resistance_levels'] or [float('-inf')]):
        indicators.append('📈 Above resistance level')
        score += 5
    
    # Add sentiment impact
    sentiment_scores = {
        'BULLISH': 10,
        'CAUTIOUSLY BULLISH': 5,
        'CAUTIOUSLY BEARISH': -5,
        'BEARISH': -10,
        'NEUTRAL': 0
    }
    score += sentiment_scores[market_trends['market_sentiment']]
    
    # Ensure score stays within 0-100 range
    score = max(0, min(100, score))
    
    # Sort indicators by importance
    indicators = sorted(indicators, key=lambda x: x.startswith('🚀') or x.startswith('📈'), reverse=True)
    
    return {
        'score': round(score),
        'indicators': indicators[:4]  # Limit to top 4 most important indicators
    }

def generate_recommendation(price_data, volume_data, volatility, market_health_score, current_price, purchase_price=None, grade=None):
    """Generate recommendation with grade-specific considerations"""
    try:
        if len(price_data) < 30:
            return {
                'action': 'HOLD',
                'confidence': 'LOW',
                'reasons': ['Insufficient historical data for analysis (minimum 30 days required)']
            }, {
                'trend_score': '0.0',
                'trend_details': 'Insufficient data',
                'volatility': f"{volatility:.1f}%",
                'market_health': market_health_score
            }
        
        # Grade-specific volatility thresholds
        volatility_thresholds = {
            'PSA 10': {'high': 20, 'moderate': 12},
            'PSA 9': {'high': 25, 'moderate': 15},
            'BGS 9.5': {'high': 22, 'moderate': 13},
            'Raw': {'high': 30, 'moderate': 18},
            None: {'high': 25, 'moderate': 15}
        }
        
        thresholds = volatility_thresholds.get(grade, volatility_thresholds[None])
        
        # Calculate trends with grade-specific considerations
        short_term_price = price_data[-7:].mean()
        mid_term_price = price_data[-30:].mean()
        long_term_price = price_data[:-30].mean() if len(price_data) > 30 else price_data.mean()
        
        # Weighted trends based on grade stability
        grade_weights = {
            'PSA 10': {'short': 0.35, 'mid': 0.35, 'volume': 0.30},
            'PSA 9': {'short': 0.40, 'mid': 0.30, 'volume': 0.30},
            'BGS 9.5': {'short': 0.35, 'mid': 0.35, 'volume': 0.30},
            'Raw': {'short': 0.45, 'mid': 0.25, 'volume': 0.30},
            None: {'short': 0.40, 'mid': 0.30, 'volume': 0.30}
        }
        
        weights = grade_weights.get(grade, grade_weights[None])
        
        # Calculate weighted trends with safe division
        short_term_trend = ((short_term_price - mid_term_price) / mid_term_price * 100) if mid_term_price > 0 else 0
        mid_term_trend = ((mid_term_price - long_term_price) / long_term_price * 100) if long_term_price > 0 else 0
        
        # Volume trend with safety check
        recent_volume = len(price_data[-7:])
        previous_volume = len(price_data[-14:-7])
        volume_trend = ((recent_volume - previous_volume) / max(1, previous_volume) * 100)
        
        # Calculate weighted trend score
        trend_score = (
            (short_term_trend * weights['short']) +
            (mid_term_trend * weights['mid']) +
            (volume_trend * weights['volume'])
        )
        
        # Generate trend details
        trend_details = []
        if abs(short_term_trend) > 5:
            trend_details.append(f"{'↑' if short_term_trend > 0 else '↓'} {abs(short_term_trend):.1f}% 7-day price trend")
        if abs(mid_term_trend) > 5:
            trend_details.append(f"{'↑' if mid_term_trend > 0 else '↓'} {abs(mid_term_trend):.1f}% 30-day price trend")
        if abs(volume_trend) > 10:
            trend_details.append(f"{'↑' if volume_trend > 0 else '↓'} {abs(volume_trend):.1f}% volume change")
        
        # Get market signals with adjusted thresholds
        signals = {
            'trend': 'STRONG POSITIVE' if trend_score > 15 else 
                    'POSITIVE' if trend_score > 5 else 
                    'STRONG NEGATIVE' if trend_score < -15 else 
                    'NEGATIVE' if trend_score < -5 else 'NEUTRAL',
            'volatility': 'HIGH' if volatility > thresholds['high'] else 'LOW' if volatility < thresholds['moderate'] else 'MODERATE',
            'market_health': market_health_score
        }
        
        recommendation = {
            'action': 'HOLD',
            'confidence': 'MEDIUM',
            'reasons': []
        }
        
        # Enhanced recommendation logic based on combined signals
        if signals['market_health'] == 'STRONG':
            if 'STRONG POSITIVE' in signals['trend']:
                if signals['volatility'] == 'LOW':
                    recommendation['action'] = 'BUY'
                    recommendation['confidence'] = 'HIGH'
                    recommendation['reasons'].append('Strong upward momentum with low volatility')
                    recommendation['reasons'].append('Excellent conditions for long-term investment')
                else:
                    recommendation['action'] = 'SELL'
                    recommendation['confidence'] = 'HIGH'
                    recommendation['reasons'].append('Peak market conditions with high volatility')
                    if purchase_price and current_price and current_price > purchase_price * 1.5:
                        recommendation['reasons'].append('Optimal time to take profits')
            elif 'POSITIVE' in signals['trend']:
                recommendation['action'] = 'HOLD'
                recommendation['confidence'] = 'HIGH'
                recommendation['reasons'].append('Strong market with steady growth')
                recommendation['reasons'].append('Good conditions for holding position')
        
        elif signals['market_health'] == 'WEAK':
            if 'STRONG NEGATIVE' in signals['trend']:
                recommendation['action'] = 'HOLD'
                recommendation['confidence'] = 'HIGH'
                recommendation['reasons'].append('Significant market weakness')
                recommendation['reasons'].append('Wait for trend reversal signals')
            else:
                recommendation['action'] = 'HOLD'
                recommendation['confidence'] = 'MEDIUM'
                recommendation['reasons'].append('Weak market conditions')
                recommendation['reasons'].append('Monitor for recovery signals')
        
        else:  # MODERATE market health
            if 'POSITIVE' in signals['trend']:
                recommendation['action'] = 'HOLD'
                recommendation['confidence'] = 'HIGH'
                recommendation['reasons'].append('Improving market conditions')
                if signals['volatility'] == 'LOW':
                    recommendation['reasons'].append('Stable growth pattern emerging')
            else:
                recommendation['action'] = 'HOLD'
                recommendation['confidence'] = 'MEDIUM'
                recommendation['reasons'].append('Neutral market conditions')
                if signals['volatility'] == 'LOW':
                    recommendation['reasons'].append('Low risk environment for current positions')
        
        return recommendation, {
            'trend_score': f"{trend_score:+.1f}%",
            'trend_details': ' | '.join(trend_details) if trend_details else 'Stable market conditions',
            'volatility': f"{volatility:.1f}%",
            'market_health': market_health_score
        }
        
    except Exception as e:
        logger.error(f"Error generating recommendation: {e}")
        return {
            'action': 'HOLD',
            'confidence': 'LOW',
            'reasons': ['Error generating recommendation']
        }, {
            'trend_score': 'N/A',
            'trend_details': 'Error in analysis',
            'volatility': 'N/A',
            'market_health': 'N/A'
        }

@st.cache_data
def load_data(query, date_range, grade_filter):
    """Load and process card sales data"""
    # Generate sample data with consistent lengths
    dates = pd.date_range(end=datetime.now(), periods=date_range)
    
    # Generate dynamic price data with enhanced realism
    base_price = 150
    price_data = []
    trend_direction = random.choice([-1, 1])  # Random trend direction
    volatility_factor = random.uniform(0.05, 0.15)  # 5-15% volatility
    
    # Adjust base price based on grade
    grade_multipliers = {
        "PSA 10": 2.5,
        "PSA 9": 1.5,
        "BGS 9.5": 2.0,
        "Raw": 1.0
    }
    
    if grade_filter != "All Grades":
        base_price *= grade_multipliers[grade_filter]
    
    for i in range(date_range):
        # Long-term trend (0.1-0.3% daily change)
        trend_factor = i * trend_direction * random.uniform(0.001, 0.003) * base_price
        
        # Daily volatility (-volatility_factor to +volatility_factor of base price)
        daily_volatility = random.uniform(-volatility_factor, volatility_factor) * base_price
        
        # Market sentiment cycles (sine wave with 30-day period)
        sentiment = math.sin(2 * math.pi * i / 30) * volatility_factor * base_price * 0.5
        
        # Combine all factors
        price = base_price + trend_factor + daily_volatility + sentiment
        
        # Ensure price stays positive and add some mean reversion
        price = max(price, base_price * 0.5)  # Price won't go below 50% of base
        price = min(price, base_price * 2.0)  # Price won't go above 200% of base
        
        price_data.append(round(price, 2))
    
    # Create sample data with consistent lengths
    sample_data = pd.DataFrame({
        'Date': dates,
        'Price': price_data,
        'Condition': [grade_filter if grade_filter != "All Grades" else random.choice(["Raw", "PSA 9", "PSA 10", "BGS 9.5"]) for _ in range(date_range)],
        'Seller': [f'eBay Seller {i % 10 + 1}' for i in range(date_range)],
        'Sale Type': ['Auction' if i % 3 == 0 else 'Buy It Now' for i in range(date_range)]
    })
    
    return sample_data

def metric_with_tooltip(label, value, help_text):
    """Display a metric with a tooltip in Streamlit"""
    st.markdown(
        f"""
        <div class="metric-container" title="{help_text}">
            <p style="color: #666; margin-bottom: 0;">{label}</p>
            <h3 style="margin: 0;">{value}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

def main():
    # Add custom CSS
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1E88E5;
            text-align: center;
            margin-bottom: 2rem;
            padding: 1rem;
            background: linear-gradient(90deg, #E3F2FD 0%, #BBDEFB 100%);
            border-radius: 10px;
        }
        .metric-container {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .section-header {
            color: #1976D2;
            font-weight: 600;
            margin-top: 2rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #BBDEFB;
        }
        .recommendation-card {
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
            background: #fff;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .indicator-card {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            margin: 0.5rem 0;
            text-align: center;
        }
        .chart-container {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }
        .data-table {
            margin-top: 1rem;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .grade-selector {
            background-color: #f8f9fa;
            padding: 0.5rem;
            border-radius: 5px;
            margin-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # Main title with custom styling
    st.markdown('<h1 class="main-header">📊 Sports Card Value Analyzer</h1>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### 🎛️ Analysis Controls")
        date_range = st.slider(
            "📅 Date Range (days)",
            min_value=7,
            max_value=365,
            value=30,
            step=1
        )
        
        st.markdown("### 💲 Price Range Analysis")
        if 'filtered_data' in locals():
            # Calculate price statistics
            price_min = float(filtered_data['Price'].min())
            price_max = float(filtered_data['Price'].max())
            price_mean = float(filtered_data['Price'].mean())
            
            # Create price range slider with actual data bounds
            price_range = st.slider(
                "Filter Price Range ($)",
                min_value=price_min,
                max_value=price_max,
                value=(price_min, price_max),
                step=1.0,
                format="$%d"
            )
            
            # Show price distribution
            st.markdown("#### Price Distribution")
            hist_data = filtered_data[
                (filtered_data['Price'] >= price_range[0]) &
                (filtered_data['Price'] <= price_range[1])
            ]
            if not hist_data.empty:
                fig = px.histogram(
                    hist_data,
                    x='Price',
                    nbins=20,
                    title='',
                    height=100
                )
                fig.update_layout(
                    margin=dict(l=0, r=0, t=0, b=0),
                    yaxis_title='',
                    xaxis_title='',
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Show summary stats
            st.markdown(f"**Average Price:** ${price_mean:.2f}")
            st.markdown(f"**Price Range:** ${price_min:.2f} - ${price_max:.2f}")
        else:
            st.info("Enter a card name to see price analysis")
            price_range = (0, 1000000)  # Default range when no data
    
    # Main search interface
    st.markdown('<h3 class="section-header">🔍 Card Search</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Enter card details",
            placeholder="e.g., 2020 Justin Herbert Prizm PSA 10"
        )
    
    with col2:
        st.markdown('<div class="grade-selector">', unsafe_allow_html=True)
        grade_filter = st.selectbox(
            "📊 Grade Filter",
            ["All Grades", "PSA 10", "PSA 9", "BGS 9.5", "Raw"],
            help="Filter sales by card grade"
        )
        
        if grade_filter != "All Grades":
            st.markdown(f"""
            <div style='font-size: 0.8rem; color: #666;'>
                Showing only {grade_filter} sales
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    if query:
        try:
            with st.spinner("🔍 Searching for card..."):
                image_url = get_card_image(query)
                
            # Data loading with progress
            data_load_state = st.progress(0)
            for i in range(100):
                data_load_state.progress(i + 1)
            filtered_data = load_data(query, date_range, grade_filter)
            data_load_state.empty()

            # Add export functionality
            if not filtered_data.empty:
                csv = filtered_data.to_csv(index=False)
                st.download_button(
                    label="📥 Export Data",
                    data=csv,
                    file_name=f"{query}_market_data.csv",
                    mime="text/csv"
                )

            # Enhanced error handling
            if len(filtered_data) < 5:
                st.warning("Limited data available. Analysis may be less accurate.")
            
            # Calculate metrics first
            metrics = calculate_metrics(filtered_data)
            
            # Add tooltips to key metrics
            metric_with_tooltip(
                "Average Price",
                format_currency(metrics['mean']),
                "Average price over selected time period"
            )

            # Display metrics with enhanced styling
            st.markdown('<h3 class="section-header">📈 Market Overview</h3>', unsafe_allow_html=True)
            
            with st.container():
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                row1_col1, row1_col2, row1_col3, row1_col4 = st.columns(4)
                with row1_col1:
                    st.metric(
                        "💰 Average Price",
                        format_currency(metrics['mean']),
                        delta=format_currency(metrics['week_trend'])
                    )
                with row1_col2:
                    st.metric("📊 Median Price", format_currency(metrics['median']))
                with row1_col3:
                    st.metric("⬇️ Lowest Sale", format_currency(metrics['min']))
                with row1_col4:
                    st.metric("⬆️ Highest Sale", format_currency(metrics['max']))
                st.markdown('</div>', unsafe_allow_html=True)
            
            with st.container():
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                row2_col1, row2_col2, row2_col3, row2_col4 = st.columns(4)
                with row2_col1:
                    st.metric("🏷️ Last Sale", format_currency(metrics['last_price']))
                with row2_col2:
                    st.metric("📊 Total Sales", str(metrics['count']))
                with row2_col3:
                    st.metric("📏 Price StdDev", format_currency(metrics['std']))
                with row2_col4:
                    st.metric("📈 Volatility", f"{metrics['volatility']:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Profit Analysis Section with enhanced styling
            st.markdown('<h3 class="section-header">💰 Profit Analysis</h3>', unsafe_allow_html=True)
            
            with st.container():
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                purchase_price = st.number_input(
                    "Enter Your Purchase Price ($)",
                    min_value=0.0,
                    value=0.0,
                    step=0.01
                )
                
                if purchase_price > 0:
                    current_price = metrics['mean']
                    profit, roi = calculate_profit_metrics(current_price, purchase_price)
                    
                    if grade_filter in ["Raw", "All Grades"]:
                        # Show grading potential for raw cards
                        col1, col2, col3 = st.columns(3)
                        grading_multipliers = calculate_grading_multiplier(current_price)
                        
                        with col1:
                            st.metric(
                                "📦 Raw Condition",
                                format_currency(current_price),
                                delta=f"{roi:.1f}% ROI"
                            )
                        
                        psa9_price = current_price * grading_multipliers['psa9']
                        psa9_profit, psa9_roi = calculate_profit_metrics(psa9_price, purchase_price)
                        
                        with col2:
                            st.metric(
                                "🏅 PSA 9 Estimate",
                                format_currency(psa9_price),
                                delta=f"{psa9_roi:.1f}% ROI"
                            )
                        
                        psa10_price = current_price * grading_multipliers['psa10']
                        psa10_profit, psa10_roi = calculate_profit_metrics(psa10_price, purchase_price)
                        
                        with col3:
                            st.metric(
                                "🏆 PSA 10 Estimate",
                                format_currency(psa10_price),
                                delta=f"{psa10_roi:.1f}% ROI"
                            )
                        
                        st.info(f"🔍 Grading Analysis: {grading_multipliers['grading_recommendation']}")
                    else:
                        # Show simple profit analysis for graded cards
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                f"💰 Current {grade_filter} Value",
                                format_currency(current_price),
                                delta=f"{roi:.1f}% ROI"
                            )
                        with col2:
                            st.metric(
                                "💵 Profit/Loss",
                                format_currency(profit),
                                delta=format_currency(metrics['week_trend']) + " (7-day trend)"
                            )
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Market Health Score with enhanced visualization
            st.markdown('<h3 class="section-header">🎯 Market Health</h3>', unsafe_allow_html=True)
            
            with st.container():
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                market_health = calculate_market_health(filtered_data['Price'], filtered_data['Price'], grade_filter)
                
                # Progress bar and score
                progress_color = 'green' if market_health['score'] >= 70 else 'red' if market_health['score'] <= 30 else 'orange'
                st.progress(market_health['score'] / 100)
                st.markdown(f"""
                    <h2 style='text-align: center; color: {progress_color}; margin: 1rem 0;'>
                        {market_health['score']}/100
                    </h2>
                """, unsafe_allow_html=True)
                
                # Display indicators in a grid
                if market_health['indicators']:
                    indicator_cols = st.columns(min(4, len(market_health['indicators'])))
                    for idx, indicator in enumerate(market_health['indicators'][:4]):  # Limit to 4 indicators
                        with indicator_cols[idx % len(indicator_cols)]:
                            st.markdown(f"""
                            <div class="indicator-card">
                                {indicator}
                            </div>
                            """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Market Analysis & Recommendation with enhanced styling
            st.markdown('<h3 class="section-header">📊 Market Analysis & Recommendation</h3>', unsafe_allow_html=True)
            
            with st.container():
                st.markdown('<div class="recommendation-card">', unsafe_allow_html=True)
                recommendation, signals = generate_recommendation(
                    filtered_data['Price'],
                    filtered_data['Price'],
                    metrics['volatility'],
                    market_health['score'],
                    metrics['mean'],
                    purchase_price if 'purchase_price' in locals() else None,
                    grade_filter
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    trend_color = 'green' if float(signals['trend_score'].rstrip('%')) > 0 else 'red' if float(signals['trend_score'].rstrip('%')) < 0 else 'gray'
                    st.markdown(f"""
                    <h4 style='color: {trend_color};'>Market Trend: {signals['trend_score']}</h4>
                    <p>{signals['trend_details']}</p>
                    """, unsafe_allow_html=True)
                
                with col2:
                    health_color = 'green' if market_health['score'] >= 70 else 'red' if market_health['score'] <= 30 else 'orange'
                    volatility_color = 'red' if float(signals['volatility'].rstrip('%')) > 20 else 'green' if float(signals['volatility'].rstrip('%')) < 10 else 'orange'
                    st.markdown(f"""
                    <h4>Market Metrics</h4>
                    <p style='color: {health_color};'>Health Score: {market_health['score']}/100</p>
                    <p style='color: {volatility_color};'>Volatility: {signals['volatility']}</p>
                    """, unsafe_allow_html=True)
                
                recommendation_color = {
                    'SELL': '#ef5350',
                    'HOLD': '#42a5f5',
                    'BUY': '#66bb6a'
                }[recommendation['action']]
                
                st.markdown(f"""
                <h3 style='color: {recommendation_color}; text-align: center; margin: 1rem 0;'>
                    Recommendation: {recommendation['action']}
                </h3>
                <p style='text-align: center; font-weight: 500;'>
                    Confidence Level: {recommendation['confidence']}
                </p>
                """, unsafe_allow_html=True)
                
                st.markdown("**Reasons:**")
                for reason in recommendation['reasons']:
                    st.markdown(f"- {reason}")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Charts with enhanced styling
            st.markdown('<h3 class="section-header">📈 Price Analysis</h3>', unsafe_allow_html=True)
            
            with st.container():
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                fig = px.line(
                    filtered_data,
                    x='Date',
                    y='Price',
                    title='Price History',
                    template='plotly_white'
                )
                fig.update_traces(mode='lines+markers')
                fig.update_layout(
                    hovermode='x unified',
                    yaxis_title='Price ($)',
                    showlegend=False,
                    title_x=0.5
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Sales breakdown with enhanced styling
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.subheader("Sale Type Distribution")
                sale_type_counts = filtered_data['Sale Type'].value_counts()
                st.bar_chart(sale_type_counts)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.subheader("Weekly Price Trends")
                weekly_avg = filtered_data.set_index('Date')['Price'].resample('W').mean()
                st.line_chart(weekly_avg)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Detailed sales table with enhanced styling
            st.markdown('<h3 class="section-header">📋 Recent Sales History</h3>', unsafe_allow_html=True)
            
            with st.container():
                st.markdown('<div class="data-table">', unsafe_allow_html=True)
                display_data = filtered_data.copy()
                display_data['Date'] = display_data['Date'].dt.strftime('%Y-%m-%d')
                st.dataframe(
                    display_data.sort_values('Date', ascending=False),
                    column_config={
                        'Date': st.column_config.DateColumn('Sale Date'),
                        'Price': st.column_config.NumberColumn(
                            'Price',
                            format='$%.2f'
                        ),
                        'Condition': 'Condition',
                        'Seller': 'Seller',
                        'Sale Type': 'Sale Type'
                    },
                    hide_index=True,
                    use_container_width=True
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Add grade distribution chart
            if grade_filter == "All Grades":
                st.markdown('<h3 class="section-header">📊 Grade Distribution</h3>', unsafe_allow_html=True)
                with st.container():
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    grade_dist = filtered_data['Condition'].value_counts()
                    
                    # Calculate average price by grade
                    grade_prices = filtered_data.groupby('Condition')['Price'].agg(['mean', 'count']).round(2)
                    grade_prices.columns = ['Avg Price', 'Count']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.bar_chart(grade_dist)
                    
                    with col2:
                        st.dataframe(
                            grade_prices,
                            column_config={
                                'Avg Price': st.column_config.NumberColumn(
                                    'Average Price',
                                    format='$%.2f'
                                ),
                                'Count': st.column_config.NumberColumn(
                                    'Number of Sales'
                                )
                            },
                            use_container_width=True
                        )
                    st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please try again or contact support if the issue persists.")
    else:
        st.info("Enter a card name above to see market analysis.")

if __name__ == "__main__":
    main() 