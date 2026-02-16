"""
SIMPLIFIED UNIFIED API - Import-Based Integration
Directly uses stock_prediction.py and news.py as modules
"""

import os
from datetime import datetime
from dotenv import load_dotenv

# Import from your existing modules
from stock_prediction import StockPredictionPipeline
from news import AdvancedStockNewsPrediction

# FastAPI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict

load_dotenv()

app = FastAPI(title="Unified Stock Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class PredictionRequest(BaseModel):
    symbol: str = "RELIANCE.NS"
    analysis_type: str = "full"  # "lstm", "news", "full"

# Cache
prediction_cache = {}

@app.get("/")
def home():
    return {
        "message": "Unified Stock Prediction API",
        "endpoints": {
            "full": "POST /predict",
            "lstm": "GET /lstm-only/{symbol}",
            "news": "GET /news-only/{symbol}"
        }
    }

@app.post("/predict")
def predict_stock(request: PredictionRequest):
    """Run LSTM + News analysis and combine results"""
    try:
        # Check cache
        cache_key = f"{request.symbol}_{request.analysis_type}"
        if cache_key in prediction_cache:
            cached_time = prediction_cache[cache_key]['timestamp']
            if (datetime.now() - datetime.fromisoformat(cached_time)).total_seconds() < 3600:
                return prediction_cache[cache_key]
        
        if request.analysis_type == "lstm":
            # Run LSTM only
            lstm_pipeline = StockPredictionPipeline(request.symbol)
            lstm_pipeline.run()
            
            result = {
                'symbol': request.symbol,
                'timestamp': datetime.now().isoformat(),
                'lstm': {
                    'trained_periods': list(lstm_pipeline.results.keys()),
                    'predictions_6mo': lstm_pipeline.get_predictions('6mo', 180),
                    'predictions_1mo': lstm_pipeline.get_predictions('1mo', 30),
                },
                'news': None,
                'combined_prediction': None
            }
        
        elif request.analysis_type == "news":
            # Run News only
            news_pipeline = AdvancedStockNewsPrediction(request.symbol)
            news_data = news_pipeline.run()
            
            result = {
                'symbol': request.symbol,
                'timestamp': datetime.now().isoformat(),
                'lstm': None,
                'news': news_data,
                'combined_prediction': None
            }
        
        else:  # full
            # Run both and combine
            lstm_pipeline = StockPredictionPipeline(request.symbol)
            news_pipeline = AdvancedStockNewsPrediction(request.symbol)
            
            # Run LSTM
            lstm_pipeline.run()
            lstm_pred = lstm_pipeline.get_predictions('6mo', 180)
            
            # Run News
            news_data = news_pipeline.run()
            
            # Combine predictions
            if lstm_pred and news_data:
                lstm_price = lstm_pred['avg']
                sentiment_score = news_data.get('sentiment_score', 0)
                sentiment_adjustment = lstm_price * (sentiment_score * 0.03)
                combined_price = lstm_price + sentiment_adjustment
                
                change_pct = ((combined_price - news_data['current_price']) / news_data['current_price']) * 100
                
                if change_pct > 5:
                    recommendation = "BUY"
                elif change_pct < -5:
                    recommendation = "SELL"
                else:
                    recommendation = "HOLD"
                
                combined = {
                    'lstm_predicted_price': round(lstm_price, 4),
                    'sentiment_adjustment': round(sentiment_adjustment, 4),
                    'combined_predicted_price': round(combined_price, 4),
                    'sentiment_score': round(sentiment_score, 2),
                    'confidence': 'high' if abs(sentiment_score) < 0.5 else 'medium',
                    'recommendation': recommendation,
                }
            else:
                combined = None
            
            result = {
                'symbol': request.symbol,
                'timestamp': datetime.now().isoformat(),
                'lstm': {
                    'trained_periods': list(lstm_pipeline.results.keys()),
                    'predictions_6mo': lstm_pred,
                    'predictions_1mo': lstm_pipeline.get_predictions('1mo', 30),
                },
                'news': news_data,
                'combined_prediction': combined
            }
        
        # Cache result
        prediction_cache[cache_key] = result
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/lstm-only/{symbol}")
def lstm_only(symbol: str):
    """LSTM prediction only"""
    try:
        lstm_pipeline = StockPredictionPipeline(symbol)
        lstm_pipeline.run()
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'lstm': {
                'trained_periods': list(lstm_pipeline.results.keys()),
                'predictions_6mo': lstm_pipeline.get_predictions('6mo', 180),
                'predictions_1mo': lstm_pipeline.get_predictions('1mo', 30),
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/news-only/{symbol}")
def news_only(symbol: str):
    """News analysis only"""
    try:
        news_pipeline = AdvancedStockNewsPrediction(symbol)
        news_data = news_pipeline.run()
        
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
    print("=" * 70)
    print("UNIFIED STOCK PREDICTION API - Starting...")
    print("=" * 70)
    print("📚 Endpoints:")
    print("  • POST   /predict           - Full (LSTM + News)")
    print("  • GET    /lstm-only/{symbol} - LSTM only")
    print("  • GET    /news-only/{symbol} - News only")
    print("  • GET    /cache-clear       - Clear cache")
    print("=" * 70)
    uvicorn.run(app, host="127.0.0.1", port=8001)
