"""
STREAMLIT UI FOR STOXAI
Beautiful interactive dashboard for LSTM + News sentiment analysis
Run: streamlit run streamlit_app.py
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime
import json

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="📈 StoxAI Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_URL = "http://127.0.0.1:8001"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data(ttl=3600)
def fetch_prediction(symbol, analysis_type="full", openrouter_key=None, newsorg_key=None):
    """Fetch prediction from API with caching"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={
                "symbol": symbol, 
                "analysis_type": analysis_type,
                "openrouter_key": openrouter_key,
                "newsorg_key": newsorg_key
            },
            timeout=600
        )
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"API Error: {response.status_code}"
    except requests.Timeout:
        return None, "⏱️ Request timeout - LSTM training in progress. Try again in 1-2 minutes."
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

def format_currency(value):
    """Format as Indian Rupee"""
    return f"₹{value:,.2f}"

def get_sentiment_emoji(score):
    """Get emoji for sentiment score"""
    if score > 0.2:
        return "🟢"
    elif score < -0.2:
        return "🔴"
    else:
        return "🟡"

def get_recommendation_color(rec):
    """Get color for recommendation"""
    if rec == "BUY":
        return "green"
    elif rec == "SELL":
        return "red"
    else:
        return "orange"

# ============================================================================
# HEADER
# ============================================================================

st.title("📈 StoxAI Dashboard")
st.markdown("*AI-Powered Analysis: LSTM Deep Learning + Real-time News Sentiment*")

# API Keys Section in Main Area
with st.expander("🔑 API Keys Setup (Required)", expanded=True):
    col_k1, col_k2 = st.columns(2)
    with col_k1:
        st.markdown('<p style="font-size:14px; font-weight:600; margin-bottom:0px;">OpenRouter API Key <a href="https://openrouter.ai/settings/keys" target="_blank" style="font-size: 12px; color: #2563EB;">(CLICK TO GET IT)</a></p>', unsafe_allow_html=True)
        openrouter_key = st.text_input("OpenRouter API Key", type="password", help="Required for AI analysis", key="openrouter_key", label_visibility="collapsed")
    with col_k2:
        st.markdown('<p style="font-size:14px; font-weight:600; margin-bottom:0px;">NewsOrg API Key <a href="https://newsapi.org/" target="_blank" style="font-size: 12px; color: #2563EB;">(CLICK TO GET IT)</a></p>', unsafe_allow_html=True)
        newsorg_key = st.text_input("NewsOrg API Key", type="password", help="Required for fetching news", key="newsorg_key", label_visibility="collapsed")
    
    if not openrouter_key or not newsorg_key:
        st.warning("⚠️ Please enter both API keys above to enable analysis.")

st.divider()

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.title("⚙️ Configuration")

# Stock symbol input
col1, col2 = st.sidebar.columns([2, 1])
with col1:
    symbol = st.text_input(
        "Stock Symbol",
        value="RELIANCE.NS",
        help="e.g., RELIANCE.NS, TCS.NS, INFY.NS, SBIN.NS"
    ).strip().upper()

with col2:
    st.write("")  # Spacer
    st.write("")
    if st.button("🔄", help="Refresh"):
        st.rerun()

# Analysis type
analysis_type = st.sidebar.radio(
    "Analysis Type",
    ["full", "lstm", "news"],
    format_func=lambda x: {
        "full": "📊 Full (LSTM + News)",
        "lstm": "🤖 LSTM Only",
        "news": "📰 News Only"
    }[x]
)

# Analyze button
analyze_btn = st.sidebar.button("🔍 Analyze Stock", use_container_width=True)

st.sidebar.divider()

# Cache info

# Cache info
if st.sidebar.button("🗑️ Clear Cache", use_container_width=True):
    st.cache_data.clear()
    try:
        requests.get(f"{API_URL}/cache-clear")
        st.sidebar.success("✅ Cache cleared!")
    except:
        st.sidebar.warning("⚠️ Could not clear API cache")

# ============================================================================
# API STATUS
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.subheader("🔗 API Status")

try:
    health = requests.get(f"{API_URL}/", timeout=5)
    if health.status_code == 200:
        st.sidebar.success("✅ API Running")
        st.sidebar.caption("http://127.0.0.1:8001")
    else:
        st.sidebar.error("❌ API Error")
except:
    st.sidebar.error("❌ API Offline")
    st.sidebar.caption("Start with: python integrated_api.py")

# ============================================================================
# MAIN CONTENT
# ============================================================================

if analyze_btn or symbol:
    if not openrouter_key or not newsorg_key:
        st.error("❌ Please ensure API keys are entered above.")
    else:
        with st.spinner(f"🔄 Analyzing {symbol}... (1-2 min)"):
            data, error = fetch_prediction(symbol, analysis_type, openrouter_key, newsorg_key)
    
    if error:
        st.error(error)
    elif data:
        lstm_data = data.get('lstm', {})
        news_data = data.get('news', {})
        combined = data.get('combined_prediction', {})
        timestamp = data.get('timestamp', '')
        
        st.caption(f"📅 {timestamp}")
        
        # ====================================================================
        # TOP METRICS
        # ====================================================================
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if combined and combined.get('combined_predicted_price'):
                value = combined['combined_predicted_price']
                current = news_data.get('current_price', 0) if news_data else lstm_data.get('predictions_6mo', {}).get('last_price', 0)
                delta = value - current if current else 0
                st.metric(
                    "Predicted Price",
                    format_currency(value),
                    delta=f"{delta:+.2f}",
                    help="6-month forecast"
                )
            elif lstm_data.get('predictions_6mo'):
                st.metric(
                    "Predicted Price",
                    format_currency(lstm_data['predictions_6mo'].get('avg', 0)),
                    help="6-month LSTM forecast"
                )
        
        with col2:
            if combined and combined.get('recommendation'):
                rec = combined['recommendation']
                st.metric(
                    "Recommendation",
                    f":{get_recommendation_color(rec)}[{rec}]",
                    delta=combined.get('confidence', 'N/A')
                )
        
        with col3:
            if combined:
                sentiment = combined.get('sentiment_score', 0)
                st.metric(
                    f"Sentiment {get_sentiment_emoji(sentiment)}",
                    f"{sentiment:+.2f}",
                    help="Range: -1 (bearish) to +1 (bullish)"
                )
        
        with col4:
            if lstm_data.get('predictions_6mo'):
                current = lstm_data['predictions_6mo'].get('last_price', 0)
                predicted = lstm_data['predictions_6mo'].get('avg', 0)
                change = ((predicted - current) / current * 100) if current else 0
                st.metric(
                    "Expected Change",
                    f"{change:+.2f}%",
                    help="6-month change estimate"
                )
        
        st.divider()
        
        # ====================================================================
        # TABS
        # ====================================================================
        
        tabs = st.tabs(["📊 Price Forecast", "📰 News", "🎯 Analysis", "📋 Data"])
        
        # TAB 1: PRICE FORECAST
        with tabs[0]:
            if lstm_data.get('predictions_6mo'):
                pred = lstm_data['predictions_6mo']
                predictions = pred.get('predictions', [])
                
                col_chart, col_stats = st.columns([3, 1])
                
                with col_chart:
                    df = pd.DataFrame({
                        'Day': range(1, len(predictions) + 1),
                        'Price': predictions
                    })
                    
                    fig = px.line(
                        df, x='Day', y='Price',
                        title='6-Month Price Prediction (LSTM)',
                        markers=True,
                        template='plotly_white'
                    )
                    
                    fig.add_hline(
                        y=pred.get('last_price'),
                        line_dash="dash",
                        line_color="red",
                        annotation_text="Current",
                        annotation_position="right"
                    )
                    
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col_stats:
                    st.write("**6M Stats**")
                    st.metric("Current", format_currency(pred.get('last_price', 0)))
                    st.metric("Average", format_currency(pred.get('avg', 0)))
                    st.metric("High", format_currency(pred.get('high', 0)))
                    st.metric("Low", format_currency(pred.get('low', 0)))
            
            if lstm_data.get('predictions_1mo'):
                st.markdown("---")
                pred_1m = lstm_data['predictions_1mo']
                preds_1m = pred_1m.get('predictions', [])
                
                col_chart, col_stats = st.columns([3, 1])
                
                with col_chart:
                    df_1m = pd.DataFrame({
                        'Day': range(1, len(preds_1m) + 1),
                        'Price': preds_1m
                    })
                    
                    fig_1m = px.line(
                        df_1m, x='Day', y='Price',
                        title='1-Month Price Prediction (LSTM)',
                        markers=True,
                        template='plotly_white'
                    )
                    
                    fig_1m.update_layout(height=400)
                    st.plotly_chart(fig_1m, use_container_width=True)
                
                with col_stats:
                    st.write("**1M Stats**")
                    st.metric("Average", format_currency(pred_1m.get('avg', 0)))
                    st.metric("High", format_currency(pred_1m.get('high', 0)))
                    st.metric("Low", format_currency(pred_1m.get('low', 0)))
        
        # TAB 2: NEWS
        with tabs[1]:
            if news_data:
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric("Articles", news_data.get('articles_count', 0))
                
                with col_b:
                    sentiment = news_data.get('sentiment_score', 0)
                    st.metric(f"Sentiment {get_sentiment_emoji(sentiment)}", f"{sentiment:+.2f}")
                
                with col_c:
                    pos = news_data.get('positive_count', 0)
                    neg = news_data.get('negative_count', 0)
                    st.metric("Breakdown", f"🟢{pos} 🔴{neg}")
                
                st.divider()
                
                # Sentiment pie
                sentiment_data = {
                    'Positive': news_data.get('positive_count', 0),
                    'Negative': news_data.get('negative_count', 0),
                    'Neutral': news_data.get('articles_count', 0) - news_data.get('positive_count', 0) - news_data.get('negative_count', 0)
                }
                
                if sum(sentiment_data.values()) > 0:
                    fig_pie = px.pie(
                        values=sentiment_data.values(),
                        names=sentiment_data.keys(),
                        title='Sentiment Distribution',
                        color_discrete_map={'Positive': '#28a745', 'Negative': '#dc3545', 'Neutral': '#ffc107'}
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                st.divider()
                
                # Articles
                st.subheader("📰 Latest Articles")
                articles = news_data.get('articles', [])
                
                if articles:
                    for i, article in enumerate(articles[:5], 1):
                        with st.expander(f"{i}. {article.get('title', 'N/A')[:60]}..."):
                            st.write(f"**Source:** {article.get('source', {}).get('name', 'N/A')}")
                            st.write(f"**Date:** {article.get('publishedAt', 'N/A')[:10]}")
                            st.write(article.get('description', 'N/A'))
                            if article.get('url'):
                                st.markdown(f"[🔗 Read Full]({article['url']})")
                else:
                    st.info("No news articles found")
            else:
                st.info("No news data available")
        
        # TAB 3: ANALYSIS
        with tabs[2]:
            if combined:
                st.subheader("🎯 Combined Analysis")
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.write("**Weight Distribution**")
                    weight_data = {
                        'Component': ['LSTM (60%)', 'News (40%)'],
                        'Weight': [60, 40]
                    }
                    df_weight = pd.DataFrame(weight_data)
                    fig_weight = px.bar(
                        df_weight, x='Component', y='Weight',
                        color_discrete_sequence=['#1f77b4', '#ff7f0e']
                    )
                    st.plotly_chart(fig_weight, use_container_width=True)
                
                with col_b:
                    st.write("**Price Calculation**")
                    st.metric("LSTM Price", format_currency(combined.get('lstm_predicted_price', 0)))
                    st.metric("Sentiment Adj", f"{combined.get('sentiment_adjustment', 0):+.2f}")
                    st.metric("Final Price", format_currency(combined.get('combined_predicted_price', 0)))
                
                st.divider()
                
                col_info1, col_info2 = st.columns(2)
                
                with col_info1:
                    st.info("""
                    **🤖 LSTM Model**
                    - 2-layer neural network (64→32 units)
                    - Trained on 4 timeframes
                    - Dynamic lookback (10-60 days)
                    - 50 epochs, 20% dropout
                    """)
                
                with col_info2:
                    st.info("""
                    **📰 News Analysis**
                    - NewsOrg API (15 articles)
                    - Last 30 days
                    - AI sentiment analysis
                    - Recency-weighted
                    """)
            else:
                st.info("Combined analysis not available")
        
        # TAB 4: DATA
        with tabs[3]:
            st.subheader("📋 Full Response Data")
            
            col_json, col_csv = st.columns(2)
            
            with col_json:
                json_str = json.dumps(data, indent=2)
                st.download_button(
                    "📥 Download JSON",
                    json_str,
                    f"{symbol}_pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json"
                )
            
            with col_csv:
                if lstm_data.get('predictions_6mo'):
                    preds = lstm_data['predictions_6mo'].get('predictions', [])
                    df_csv = pd.DataFrame({
                        'Day': range(1, len(preds) + 1),
                        'Predicted_Price': preds
                    })
                    st.download_button(
                        "📊 Download CSV",
                        df_csv.to_csv(index=False),
                        f"{symbol}_preds_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv"
                    )
            
            st.divider()
            st.write("**Raw JSON Response:**")
            st.json(data)

# ============================================================================
# INFO SIDEBAR
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📚 How It Works

1. **LSTM Pipeline** (60% weight)
   - Historical price data
   - Deep learning model
   - 30 & 180 day forecast

2. **News Pipeline** (40% weight)
   - Recent articles
   - AI sentiment analysis
   - Score: -1 to +1

3. **Combined**
   - Weighted average
   - BUY/SELL/HOLD
   - Confidence level

### 🎯 Signals
- **BUY**: >5% upside
- **SELL**: >5% downside
- **HOLD**: Neutral
""")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 🚀 Quick Start

**Terminal 1:**
```bash
python api.py
```

**Terminal 2:**
```bash
python integrated_api.py
```

**Terminal 3:**
```bash
streamlit run streamlit_app.py
```

### 📖 Docs
- See INTEGRATION_GUIDE.md
- API: http://127.0.0.1:8001
""")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
<div style='text-align: center; font-size: 0.9em; color: gray;'>
    <p>📈 <b>StoxAI</b> | LSTM Deep Learning + News Sentiment</p>
    <p>⚠️ Educational purposes only. Always do your own research before investing.</p>
</div>
""", unsafe_allow_html=True)
