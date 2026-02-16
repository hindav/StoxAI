import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

API_BASE_URL = "http://127.0.0.1:8000" 

class StockPredictionPipeline:
    def __init__(self, symbol="RELIANCE.NS"):
        self.symbol = symbol
        self.results = {}
        # Mapping: period -> (preferred interval, fallback intervals)
        self.intervals = {
            '1d': (['1m'], '1d'),          
            '1wk': (['1h'], '1wk'),
            '1mo': (['1h'], '1mo'),   
            '6mo': (['1d'], '6mo')        
        }
        
    def fetch_data(self, intervals, period):
        """Fetch stock data from API, trying intervals in order"""
        for interval in intervals:
            print(f"\n📥 Fetching data for period: {period} with interval: {interval}")
            params = {"symbol": self.symbol, "interval": interval}
            
            try:
                resp = requests.get(f"{API_BASE_URL}/candles", params=params)
                resp.raise_for_status()
                data = resp.json()
                
                if "error" in data or not data.get('data'):
                    print(f"   ✗ No data for {interval} interval, trying next...")
                    continue
                
                df = pd.DataFrame(data["data"])
                if len(df) == 0:
                    print(f"   ✗ Empty data for {interval} interval, trying next...")
                    continue
                    
                print(f"   ✓ Retrieved {len(df)} data points ({interval} interval)")
                return df, interval
            except Exception as e:
                print(f"   ✗ Error with {interval}: {str(e)[:50]}..., trying next...")
                continue
        
        print(f"   ✗ Failed to fetch data for {period} with any interval")
        return None, None
    
    def get_lookback_for_interval(self, num_data_points):
        """Dynamically determine lookback window based on data size"""
        if num_data_points >= 100:
            return 5  # Standard lookback for large datasets
        elif num_data_points >= 50:
            return 4
        elif num_data_points >= 20:
            return 3
        else:
            return 2  # Minimum lookback for very small datasets
    
    def prepare_features(self, df, period, interval):
        """Prepare features for ML models with dynamic lookback"""
        if df is None or len(df) < 6:  # Need at least lookback + 1
            return None, None, None, 0
        
        df = df.sort_values(by='Datetime' if 'Datetime' in df.columns else 'Date').reset_index(drop=True)
        
        # Use Close price
        prices = df['Close'].values.astype(float)
        
        # Determine lookback dynamically
        lookback = self.get_lookback_for_interval(len(df))
        
        # Create features using lookback window
        X, y = [], []
        for i in range(len(prices) - lookback):
            X.append(prices[i:i+lookback])
            y.append(prices[i+lookback])
        
        X = np.array(X)
        y = np.array(y)
        
        # Normalize
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, y, scaler, lookback
    
    def train_models(self, X, y, period, interval):
        """Train multiple models and return results"""
        # Require minimum 10 samples for training (80-20 split = 8 train, 2 test)
        min_samples = 10
        
        if X is None or len(X) < min_samples:
            print(f"   ⚠️  Insufficient data for modeling (need {min_samples}+, got {len(X) if X is not None else 0})")
            return {}
        
        # For very small datasets, use 70-30 or 60-40 split
        if len(X) < 15:
            test_size = 0.3
        elif len(X) < 20:
            test_size = 0.25
        else:
            test_size = 0.2
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, random_state=42),
            'Support Vector Machine': SVR(kernel='rbf', C=100, gamma='scale'),
            'XGBoost': xgb.XGBRegressor(n_estimators=50, random_state=42, verbosity=0)
        }
        
        model_results = {}
        
        for name, model in models.items():
            print(f"   🤖 Training {name}...", end=" ")
            start_time = time.time()
            
            try:
                model.fit(X_train, y_train)
                
                train_time = time.time() - start_time
                
                # Predictions
                y_pred = model.predict(X_test)
                
                # Metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Predict next value
                next_pred = model.predict(X_test[-1:].reshape(1, -1))[0]
                
                model_results[name] = {
                    'model': model,
                    'train_time': train_time,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'next_prediction': next_pred,
                    'y_test': y_test,
                    'y_pred': y_pred
                }
                
                print(f"✓ ({train_time:.3f}s)")
            except Exception as e:
                print(f"✗ Error: {str(e)[:30]}")
        
        return model_results
    
    def run(self):
        """Run complete pipeline"""
        print(f"\n{'='*70}")
        print(f"Stock Prediction Pipeline - {self.symbol}")
        print(f"Smart Interval Selection with Fallback")
        print(f"{'='*70}")
        
        for period, (interval_list, agg_period) in self.intervals.items():
            print(f"\n{'─'*70}")
            print(f"PERIOD: {agg_period.upper()}")
            print(f"{'─'*70}")
            
            # Fetch data with fallback intervals
            df, used_interval = self.fetch_data(interval_list, agg_period)
            if df is None:
                print(f"   ⚠️  Skipping {agg_period.upper()} - no data available")
                continue
            
            # Prepare features with dynamic lookback
            X, y, scaler, lookback = self.prepare_features(df, agg_period, used_interval)
            if X is None:
                print(f"   ✗ Insufficient raw data for feature preparation (need 6+ data points)")
                continue
            
            print(f"   ℹ️  Using lookback window: {lookback}")
            
            # Check if we have enough samples
            if len(X) < 10:
                print(f"   ⚠️  Only {len(X)} samples available (minimum recommended: 10)")
                print(f"   💡 Data is limited but will attempt training with adjusted parameters...")
            
            # Train models
            print(f"\n🚀 Training models on {len(X)} samples...")
            model_results = self.train_models(X, y, agg_period, used_interval)
            
            # Only save and print if models were successfully trained
            if model_results:
                self.results[period] = {
                    'data': df,
                    'models': model_results,
                    'sample_count': len(X),
                    'lookback': lookback,
                    'period': agg_period,
                    'interval': used_interval
                }
                
                # Print results
                self.print_results(period)
            else:
                print(f"   ⚠️  No models were successfully trained for {agg_period.upper()}")
    
    def print_results(self, period):
        """Print detailed results for a period"""
        if period not in self.results:
            return
        
        data = self.results[period]
        models = data['models']
        
        if not models:
            print(f"   ⚠️  No models to display for {data['period'].upper()}")
            return
        
        print(f"\n📊 RESULTS for {data['period'].upper()} ({data['interval'].upper()} interval)")
        print(f"   Total samples: {data['sample_count']}")
        print(f"   Lookback window: {data['lookback']}")
        print(f"\n{'Model':<25} {'Train Time':<15} {'RMSE':<12} {'MAE':<12} {'R²':<10}")
        print(f"   {'-'*74}")
        
        for name, result in models.items():
            print(f"   {name:<25} {result['train_time']:.4f}s        {result['rmse']:.4f}      {result['mae']:.4f}      {result['r2']:.4f}")
        
        # Best model
        best_model = max(models.items(), key=lambda x: x[1]['r2'])
        print(f"\n   🏆 Best Model: {best_model[0]} (R² = {best_model[1]['r2']:.4f})")
        print(f"   Next Prediction: {best_model[1]['next_prediction']:.4f}")
    
    def print_summary(self):
        """Print overall summary"""
        print(f"\n{'='*70}")
        print(f"SUMMARY - STOCK PREDICTION RESULTS")
        print(f"{'='*70}")
        
        if not self.results:
            print("   ⚠️  No intervals had sufficient data for modeling.")
            print("   💡 Tip: Check your API connection or data availability.")
            return
        
        total_all_time = 0
        
        period_order = ['1d', '1wk', '1mo', '6mo']
        
        for period in period_order:
            if period not in self.results:
                continue
            
            data = self.results[period]
            models = data['models']
            if not models:
                continue
            
            total_time = sum(m['train_time'] for m in models.values())
            total_all_time += total_time
            best_model = max(models.items(), key=lambda x: x[1]['r2'])
            
            print(f"\n{data['period'].upper()} ({data['interval'].upper()} interval):")
            print(f"  Data Points: {data['sample_count']}")
            print(f"  Lookback: {data['lookback']}")
            print(f"  Total Training Time: {total_time:.3f}s")
            print(f"  Models Trained: {len(models)}")
            print(f"  Best Model: {best_model[0]}")
            print(f"  Best R² Score: {best_model[1]['r2']:.4f}")
        
        print(f"\n{'─'*70}")
        print(f"⏱️  TOTAL TRAINING TIME ACROSS ALL PERIODS: {total_all_time:.3f}s")
        print(f"{'─'*70}")

# Run the pipeline
if __name__ == "__main__":
    symbol = input("Enter stock symbol (e.g., RELIANCE.NS, TCS.NS, INFY.NS): ").strip() or "RELIANCE.NS"
    
    pipeline = StockPredictionPipeline(symbol=symbol)
    pipeline.run()
    pipeline.print_summary()
    
    print(f"\n✅ Pipeline completed successfully!")
