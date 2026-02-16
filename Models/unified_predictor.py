"""
================================================================================
UNIFIED STOCK PREDICTION & NEWS ANALYSIS INTEGRATION
================================================================================
Integrates:
1. StockPredictionPipeline (LSTM Deep Learning model)
2. AdvancedStockNewsPrediction (News analysis with sentiment)
3. FastAPI endpoints for easy access

Usage:
    # Run as API server
    python unified_predictor.py
    
    # Then make requests to:
    # POST http://localhost:8001/predict
    # GET http://localhost:8001/lstm-only/RELIANCE.NS
    # GET http://localhost:8001/news-only/RELIANCE.NS
================================================================================
"""

import requests
import pandas as pd
import numpy as np
import json
import os
import time
from datetime import datetime, timedelta
# from dotenv import load_dotenv  # Removed for deployment flexibility
import warnings
warnings.filterwarnings('ignore')

# ML Models
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# FastAPI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List

# Load environment
# Load environment (Optional - rely on request parameters)
# load_dotenv() 
API_BASE_URL = os.getenv("STOCK_API_BASE_URL", "http://127.0.0.1:8000")
OPENROUTER_API_KEY = None # Was os.getenv("OPENROUTER_API_KEY")
NEWSORG_API_KEY = None # Was os.getenv("NEWSORG_API_KEY")
OPENROUTER_URL = "https://openrouter.io/api/v1/chat/completions"
NEWSORG_URL = "https://newsapi.org/v2/everything"

# ============================================================================
# LSTM STOCK PREDICTION PIPELINE (FROM stock_prediction.py)
# ============================================================================

class StockPredictionPipeline:
    """Train LSTM models and generate price predictions"""
    
    def __init__(self, symbol="RELIANCE.NS"):
        self.symbol = symbol
        self.results = {}
        self.intervals = {
            '1d': (['1m'], '1d'),
            '1wk': (['5m', '15m', '1h'], '1wk'),
            '1mo': (['1h', '1d'], '1mo'),
            '6mo': (['1d'], '6mo')
        }
    
    def fetch_data(self, intervals, period):
        """Fetch stock data from API"""
        for interval in intervals:
            params = {"symbol": self.symbol, "interval": interval}
            try:
                resp = requests.get(f"{API_BASE_URL}/candles", params=params)
                resp.raise_for_status()
                data = resp.json()
                
                if "error" in data or not data.get('data'):
                    continue
                
                df = pd.DataFrame(data["data"])
                if len(df) == 0:
                    continue
                
                return df, interval
            except Exception as e:
                continue
        
        return None, None
    
    def get_lookback_for_interval(self, num_data_points):
        """Dynamically determine lookback window"""
        if num_data_points >= 200:
            return 60
        elif num_data_points >= 100:
            return 30
        elif num_data_points >= 50:
            return 20
        else:
            return 10
    
    def prepare_features(self, df, period, interval):
        """Prepare LSTM features"""
        if df is None or len(df) < 20:
            return None, None, None, 0, None
        
        df = df.sort_values(by='Datetime' if 'Datetime' in df.columns else 'Date').reset_index(drop=True)
        prices = df['Close'].values.astype(float)
        lookback = self.get_lookback_for_interval(len(df))
        
        scaler = MinMaxScaler()
        prices_scaled = scaler.fit_transform(prices.reshape(-1, 1)).flatten()
        
        X, y = [], []
        for i in range(len(prices_scaled) - lookback):
            X.append(prices_scaled[i:i+lookback])
            y.append(prices_scaled[i+lookback])
        
        X = np.array(X).reshape(-1, lookback, 1)
        y = np.array(y)
        
        return X, y, scaler, lookback, df
    
    def train_lstm_model(self, X, y):
        """Train LSTM model"""
        try:
            model = Sequential([
                LSTM(64, activation='relu', return_sequences=True, input_shape=(X.shape[1], 1)),
                Dropout(0.2),
                LSTM(32, activation='relu'),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(1)
            ])
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            model.fit(X, y, epochs=50, batch_size=32, verbose=0)
            return model, True
        except Exception as e:
            return None, False
    
    def predict_future(self, model, X, scaler, lookback, last_price, days_or_periods):
        """Generate future price predictions"""
        if last_price is None:
            return []
        
        predictions = []
        current_sequence = X[-1].copy()
        price_history = scaler.inverse_transform(X.reshape(-1, lookback)).flatten()
        price_mean = price_history.mean()
        price_std = price_history.std()
        
        for i in range(days_or_periods):
            try:
                next_pred_norm = model.predict(current_sequence.reshape(1, lookback, 1), verbose=0)[0][0]
                next_pred = float(scaler.inverse_transform([[next_pred_norm]])[0][0])
                
                if abs(next_pred - price_mean) > 3 * price_std:
                    next_pred = price_mean + np.sign(next_pred - price_mean) * 2 * price_std
                
                predictions.append(float(next_pred))
                current_sequence = np.append(current_sequence[1:], [[next_pred_norm]])
            except:
                if predictions:
                    predictions.append(predictions[-1])
                else:
                    predictions.append(last_price)
        
        return predictions
    
    def run(self):
        """Run complete pipeline for all periods"""
        for period, (interval_list, agg_period) in self.intervals.items():
            df, used_interval = self.fetch_data(interval_list, agg_period)
            
            if df is None:
                continue
            
            X, y, scaler, lookback, full_df = self.prepare_features(df, agg_period, used_interval)
            
            if X is None:
                continue
            
            model, success = self.train_lstm_model(X, y)
            
            if success:
                self.results[period] = {
                    'model': model,
                    'scaler': scaler,
                    'lookback': lookback,
                    'period': agg_period,
                    'interval': used_interval,
                    'df': full_df,
                    'X': X,
                }
    
    def get_predictions(self, period='6mo', days=180):
        """Get predictions for specific period"""
        if period not in self.results:
            return None
        
        data = self.results[period]
        model = data['model']
        scaler = data['scaler']
        lookback = data['lookback']
        df = data['df']
        X = data['X']
        
        last_price = float(df['Close'].values[-1])
        predictions = self.predict_future(model, X, scaler, lookback, last_price, days)
        
        return {
            'symbol': self.symbol,
            'period': period,
            'last_price': last_price,
            'predictions': predictions,
            'high': max(predictions) if predictions else None,
            'low': min(predictions) if predictions else None,
            'avg': np.mean(predictions) if predictions else None,
        }


# ============================================================================
# NEWS ANALYSIS & SENTIMENT PIPELINE (FROM news.py)
# ============================================================================

class AdvancedStockNewsPrediction:
    """Analyze news and generate sentiment-based predictions"""
    
    def __init__(self, symbol="RELIANCE.NS", openrouter_key=None, newsorg_key=None):
        self.symbol = symbol
        self.company_name = symbol.split('.')[0]
        self.results = {}
        self.openrouter_key = openrouter_key or os.getenv("OPENROUTER_API_KEY")
        self.newsorg_key = newsorg_key or os.getenv("NEWSORG_API_KEY")
    
    def fetch_current_price(self):
        """Fetch current price"""
        try:
            resp = requests.get(f"{API_BASE_URL}/candles", 
                              params={"symbol": self.symbol, "interval": "1d"})
            data = resp.json()
            if data.get('data') and len(data['data']) > 0:
                return float(data['data'][-1]['Close'])
        except:
            pass
        return None
    
    def fetch_news_from_newsorg(self):
        """Fetch real news from NewsOrg"""
        try:
            from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            to_date = datetime.now().strftime('%Y-%m-%d')
            
            params = {
                "q": self.company_name,
                "sortBy": "publishedAt",
                "language": "en",
                "from": from_date,
                "to": to_date,
                "apiKey": self.newsorg_key,
                "pageSize": 15
            }
            
            resp = requests.get(NEWSORG_URL, params=params)
            data = resp.json()
            
            if data.get('status') == 'ok' and data.get('articles'):
                return data['articles']
        except:
            pass
        
        return []
    
    def analyze_news_sentiment(self, articles):
        """Analyze sentiment using OpenRouter AI"""
        if not articles:
            return []
        
        articles_text = ""
        for idx, article in enumerate(articles[:10], 1):
            articles_text += f"{idx}. [{article.get('publishedAt', 'N/A')[:10]}] {article.get('title', 'N/A')}\n"
            articles_text += f"   {article.get('description', 'N/A')[:100]}...\n"
        
        prompt = f"""Analyze sentiment of these {self.company_name} articles:

{articles_text}

Return JSON with analyzed_articles array containing: headline, sentiment (positive/negative/neutral), 
impact_area, market_reaction_potential (1-10), key_points (list).
Only return valid JSON."""
        
        try:
            response = requests.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {self.openrouter_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.6,
                },
                timeout=30
            )
            
            result = response.json()
            if 'choices' in result:
                text = result['choices'][0]['message']['content']
                json_start = text.find('{')
                json_end = text.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    analysis = json.loads(text[json_start:json_end])
                    return analysis.get('analyzed_articles', [])
        except:
            pass
        
        return []
    
    def run(self):
        """Run complete news analysis"""
        current_price = self.fetch_current_price()
        if not current_price:
            return None
        
        articles = self.fetch_news_from_newsorg()
        sentiment_analysis = self.analyze_news_sentiment(articles) if articles else []
        
        # Calculate overall sentiment
        positive = sum(1 for a in sentiment_analysis if a.get('sentiment') == 'positive')
        negative = sum(1 for a in sentiment_analysis if a.get('sentiment') == 'negative')
        total = len(sentiment_analysis) if sentiment_analysis else 1
        sentiment_score = (positive - negative) / total
        
        return {
            'symbol': self.symbol,
            'current_price': current_price,
            'articles_count': len(articles),
            'articles': articles[:5],
            'sentiment_analysis': sentiment_analysis,
            'sentiment_score': sentiment_score,
            'positive_count': positive,
            'negative_count': negative,
        }


# ============================================================================
# UNIFIED INTEGRATION CLASS
# ============================================================================

class UnifiedStockPredictor:
    """Combines LSTM predictions with news sentiment analysis"""
    
    def __init__(self, symbol="RELIANCE.NS", openrouter_key=None, newsorg_key=None):
        self.symbol = symbol
        self.lstm_pipeline = StockPredictionPipeline(symbol)
        self.news_pipeline = AdvancedStockNewsPrediction(symbol, openrouter_key, newsorg_key)
        self.lstm_results = None
        self.news_results = None
    
    def run_lstm_analysis(self):
        """Run LSTM pipeline"""
        print(f"🚀 Running LSTM analysis for {self.symbol}...")
        self.lstm_pipeline.run()
        self.lstm_results = {
            'trained_periods': list(self.lstm_pipeline.results.keys()),
            'predictions_6mo': self.lstm_pipeline.get_predictions('6mo', 180),
            'predictions_1mo': self.lstm_pipeline.get_predictions('1mo', 30),
        }
        return self.lstm_results
    
    def run_news_analysis(self):
        """Run news analysis pipeline"""
        print(f"📰 Running news analysis for {self.symbol}...")
        self.news_results = self.news_pipeline.run()
        return self.news_results
    
    def run_full_analysis(self):
        """Run both pipelines"""
        lstm_data = self.run_lstm_analysis()
        news_data = self.run_news_analysis()
        
        return {
            'symbol': self.symbol,
            'timestamp': datetime.now().isoformat(),
            'lstm': lstm_data,
            'news': news_data,
            'combined_prediction': self._combine_predictions(lstm_data, news_data)
        }
    
    def _combine_predictions(self, lstm_data, news_data):
        """Combine LSTM and news predictions"""
        if not lstm_data or not news_data:
            return None
        
        lstm_pred = lstm_data.get('predictions_6mo')
        
        if not lstm_pred:
            return None
        
        # Weight: 60% LSTM, 40% News Sentiment
        sentiment_factor = news_data.get('sentiment_score', 0)
        lstm_price = lstm_pred['avg']
        
        # Adjust LSTM prediction based on news sentiment
        sentiment_adjustment = lstm_price * (sentiment_factor * 0.03)  # ±3% based on sentiment
        
        combined_price = lstm_price + sentiment_adjustment
        
        return {
            'lstm_predicted_price': round(lstm_price, 4),
            'sentiment_adjustment': round(sentiment_adjustment, 4),
            'combined_predicted_price': round(combined_price, 4),
            'sentiment_score': round(sentiment_factor, 2),
            'confidence': 'high' if abs(sentiment_factor) < 0.5 else 'medium',
            'recommendation': self._get_recommendation(combined_price, news_data['current_price']),
        }
    
    def _get_recommendation(self, predicted_price, current_price):
        """Generate buy/sell recommendation"""
        change_pct = ((predicted_price - current_price) / current_price) * 100
        
        if change_pct > 5:
            return "BUY"
        elif change_pct < -5:
            return "SELL"
        else:
            return "HOLD"


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(title="Unified Stock Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class PredictionRequest(BaseModel):
    symbol: str = "RELIANCE.NS"
    analysis_type: str = "full"  # "lstm", "news", "full"
    openrouter_key: Optional[str] = None
    newsorg_key: Optional[str] = None

class PredictionResponse(BaseModel):
    symbol: str
    timestamp: str
    lstm: Optional[Dict]
    news: Optional[Dict]
    combined_prediction: Optional[Dict]

# Cache for predictions
prediction_cache = {}

@app.get("/")
def home():
    return {
        "message": "Unified Stock Prediction API is running!",
        "endpoints": {
            "full_prediction": "POST /predict",
            "lstm_only": "GET /lstm-only/{symbol}",
            "news_only": "GET /news-only/{symbol}",
            "clear_cache": "GET /cache-clear"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_stock(request: PredictionRequest):
    """
    Unified prediction endpoint
    
    Request JSON:
    {
        "symbol": "RELIANCE.NS",
        "analysis_type": "full"  // "lstm", "news", or "full"
    }
    """
    try:
        # Check cache
        cache_key = f"{request.symbol}_{request.analysis_type}"
        if cache_key in prediction_cache:
            cached_time = prediction_cache[cache_key]['timestamp']
            if (datetime.now() - datetime.fromisoformat(cached_time)).total_seconds() < 3600:
                print(f"✓ Returning cached result for {cache_key}")
                return prediction_cache[cache_key]
        
        predictor = UnifiedStockPredictor(request.symbol, request.openrouter_key, request.newsorg_key)
        
        if request.analysis_type == "lstm":
            lstm_data = predictor.run_lstm_analysis()
            result = {
                'symbol': request.symbol,
                'timestamp': datetime.now().isoformat(),
                'lstm': lstm_data,
                'news': None,
                'combined_prediction': None
            }
        elif request.analysis_type == "news":
            news_data = predictor.run_news_analysis()
            result = {
                'symbol': request.symbol,
                'timestamp': datetime.now().isoformat(),
                'lstm': None,
                'news': news_data,
                'combined_prediction': None
            }
        else:  # full
            result = predictor.run_full_analysis()
        
        # Cache result
        prediction_cache[cache_key] = result
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/lstm-only/{symbol}")
def lstm_prediction(symbol: str):
    """Quick LSTM prediction only"""
    try:
        predictor = UnifiedStockPredictor(symbol) # Quick endpoints might miss keys if not passed in query params, usually these are for test. 
        # But let's leave them as relying on Env if not passed.
        # Actually, let's just leave it default for GET requests
        predictor = UnifiedStockPredictor(symbol)
        lstm_data = predictor.run_lstm_analysis()
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'lstm': lstm_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/news-only/{symbol}")
def news_prediction(symbol: str):
    """Quick news analysis only"""
    try:
        predictor = UnifiedStockPredictor(symbol)
        news_data = predictor.run_news_analysis()
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'news': news_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cache-clear")
def clear_cache():
    """Clear prediction cache"""
    prediction_cache.clear()
    return {"message": "Cache cleared", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    print("=" * 80)
    print("UNIFIED STOCK PREDICTION API - Starting...")
    print("=" * 80)
    print("🚀 Server running at: http://127.0.0.1:8001")
    print("\n📚 Available Endpoints:")
    print("  • POST   /predict           - Full analysis (LSTM + News)")
    print("  • GET    /lstm-only/{symbol} - LSTM only")
    print("  • GET    /news-only/{symbol} - News only")
    print("  • GET    /cache-clear       - Clear cache")
    print("=" * 80)
    uvicorn.run(app, host="127.0.0.1", port=8001)
