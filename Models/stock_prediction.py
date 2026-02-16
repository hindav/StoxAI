import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Models
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

API_BASE_URL = "http://127.0.0.1:8000"  # Change if needed

class StockPredictionPipeline:
    def __init__(self, symbol="RELIANCE.NS"):
        self.symbol = symbol
        self.results = {}
        # Mapping: period -> (preferred interval, fallback intervals)
        self.intervals = {
            '1d': (['1m'], '1d'),           # 1-day: prefer 1-min
            '1wk': (['5m', '15m', '1h'], '1wk'),  # 1-week: fallback from 5m to 1h
            '1mo': (['1h', '1d'], '1mo'),   # 1-month: use hourly or daily
            '6mo': (['1d'], '6mo')          # 6-month: use daily
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
        if num_data_points >= 200:
            return 60
        elif num_data_points >= 100:
            return 30
        elif num_data_points >= 50:
            return 20
        else:
            return 10
    
    def prepare_features(self, df, period, interval):
        """Prepare features for LSTM with dynamic lookback"""
        if df is None or len(df) < 20:
            return None, None, None, 0, None
        
        df = df.sort_values(by='Datetime' if 'Datetime' in df.columns else 'Date').reset_index(drop=True)
        
        # Use Close price
        prices = df['Close'].values.astype(float)
        
        # Determine lookback dynamically
        lookback = self.get_lookback_for_interval(len(df))
        
        # Normalize prices
        scaler = MinMaxScaler()
        prices_scaled = scaler.fit_transform(prices.reshape(-1, 1)).flatten()
        
        # Create sequences for LSTM
        X, y = [], []
        for i in range(len(prices_scaled) - lookback):
            X.append(prices_scaled[i:i+lookback])
            y.append(prices_scaled[i+lookback])
        
        X = np.array(X)
        y = np.array(y)
        
        # Reshape for LSTM [samples, timesteps, features]
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        return X, y, scaler, lookback, df
    
    def train_lstm_model(self, X, y):
        """Train LSTM model for time-series prediction"""
        print(f"   🤖 Training LSTM (Deep Learning)...", end=" ")
        start_time = time.time()
        
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
            
            # Train silently
            model.fit(X, y, epochs=50, batch_size=32, verbose=0)
            
            train_time = time.time() - start_time
            print(f"✓ ({train_time:.2f}s)")
            return model, train_time
        except Exception as e:
            print(f"✗ Error: {str(e)[:30]}")
            return None, 0
    
    def predict_future(self, model, X, scaler, lookback, last_price, days_or_periods, interval_type):
        """Predict future prices with smooth progression"""
        if last_price is None:
            print("   ⚠️  Error: last_price is None")
            return []
        
        predictions = []
        current_sequence = X[-1].copy()  # Last known sequence (normalized)
        
        # Get price range for variance
        price_history = scaler.inverse_transform(X.reshape(-1, lookback)).flatten()
        price_mean = price_history.mean()
        price_std = price_history.std()
        
        print(f"   📊 Historical Mean: ₹{price_mean:.4f}, Std Dev: ₹{price_std:.4f}")
        
        for i in range(days_or_periods):
            try:
                # Predict next normalized value
                next_pred_norm = model.predict(current_sequence.reshape(1, lookback, 1), verbose=0)[0][0]
                
                # Denormalize to actual price
                next_pred = float(scaler.inverse_transform([[next_pred_norm]])[0][0])
                
                # Apply realistic variance (±1-2% per step for short-term, smaller for long-term)
                if days_or_periods <= 7:
                    variance_factor = 0.015  # 1.5% max variance
                else:
                    variance_factor = 0.005  # 0.5% max variance for long-term
                
                # Smooth prediction towards mean if too far
                if abs(next_pred - price_mean) > 3 * price_std:
                    next_pred = price_mean + np.sign(next_pred - price_mean) * 2 * price_std
                
                predictions.append(float(next_pred))
                
                # Update sequence (shift left, add new normalized prediction)
                current_sequence = np.append(current_sequence[1:], [[next_pred_norm]])
                
            except Exception as e:
                print(f"   ⚠️  Error during prediction {i+1}: {str(e)[:40]}")
                if predictions:
                    predictions.append(predictions[-1])  # Repeat last value
                else:
                    predictions.append(last_price)
        
        return predictions
    
    def run(self, progress_callback=None):
        """Run complete pipeline"""
        print(f"\n{'='*70}")
        print(f"Stock Prediction Pipeline - {self.symbol}")
        print(f"Advanced Prediction with LSTM (Deep Learning)")
        print(f"{'='*70}")
        
        periods = list(self.intervals.items())
        total_steps = len(periods)
        
        for i, (period, (interval_list, agg_period)) in enumerate(periods):
            # Calculate base progress (0-100)
            base_progress = int((i / total_steps) * 100)
            if progress_callback:
                progress_callback(base_progress, f"Analyzing {agg_period.upper()} data...")
            
            print(f"\n{'─'*70}")
            print(f"PERIOD: {agg_period.upper()}")
            print(f"{'─'*70}")
            
            # Fetch data with fallback intervals
            df, used_interval = self.fetch_data(interval_list, agg_period)
            if df is None:
                print(f"   ⚠️  Skipping {agg_period.upper()} - no data available")
                continue
            
            # Prepare features with dynamic lookback
            X, y, scaler, lookback, full_df = self.prepare_features(df, agg_period, used_interval)
            if X is None:
                print(f"   ✗ Insufficient raw data for feature preparation (need 20+ data points)")
                continue
            
            print(f"   ℹ️  Using lookback window: {lookback}")
            print(f"   📊 Total data points: {len(full_df)}")
            
            # Train LSTM
            print(f"\n🚀 Training LSTM on {len(X)} sequences...")
            if progress_callback:
                progress_callback(base_progress + 10, f"Training LSTM model for {agg_period.upper()}...")
                
            model, train_time = self.train_lstm_model(X, y)
            
            if model:
                self.results[period] = {
                    'model': model,
                    'scaler': scaler,
                    'lookback': lookback,
                    'period': agg_period,
                    'interval': used_interval,
                    'df': full_df,
                    'X': X,
                    'train_time': train_time
                }
                
                # Show previous data
                self.show_previous_data(period)
            else:
                print(f"   ⚠️  Failed to train model for {agg_period.upper()}")
        
        if progress_callback:
            progress_callback(100, "Technical analysis complete")
    
    def show_previous_data(self, period):
        """Show last 5 days/periods of previous data"""
        data = self.results[period]
        df = data['df']
        
        print(f"\n📈 PREVIOUS DATA (Last 5 {data['period'].upper()} entries):")
        print(f"{'Date/Time':<25} {'High':<12} {'Low':<12} {'Close':<12}")
        print(f"{'-'*65}")
        
        # Get last 5 rows
        last_5 = df.tail(5)
        for idx, row in last_5.iterrows():
            datetime_str = str(row.get('Datetime', row.get('Date', 'N/A')))[:20]
            high = f"{row['High']:.4f}"
            low = f"{row['Low']:.4f}"
            close = f"{row['Close']:.4f}"
            print(f"{datetime_str:<25} {high:<12} {low:<12} {close:<12}")
    
    def predict_next_7_days(self):
        """Predict next 7 days using 1D model"""
        if '1d' not in self.results:
            print("\n❌ No 1D model available for 7-day prediction")
            return
        
        print(f"\n{'='*70}")
        print(f"🔮 PREDICTION: NEXT 7 DAYS (LSTM)")
        print(f"{'='*70}")
        
        data = self.results['1d']
        model = data['model']
        scaler = data['scaler']
        lookback = data['lookback']
        df = data['df']
        X = data['X']
        
        # Get last prices for reference
        last_price = float(df['Close'].values[-1])
        
        # Predict next 7 prices
        print(f"\n📥 Last Price (1M interval): ₹{last_price:.4f}")
        predictions = self.predict_future(model, X, scaler, lookback, last_price, 7, '1d')
        
        if not predictions:
            print("   ⚠️  No valid predictions generated")
            return
        
        # Get last date and create future dates
        last_date = pd.to_datetime(df.iloc[-1].get('Datetime', df.iloc[-1].get('Date')))
        
        print(f"\n{'Day':<6} {'Date':<20} {'Predicted Price':<20} {'Change %':<15}")
        print(f"{'-'*65}")
        
        for i, pred in enumerate(predictions):
            future_date = last_date + timedelta(minutes=i+1)
            change_pct = ((pred - last_price) / last_price) * 100
            print(f"{i+1:<6} {str(future_date):<20} ₹{pred:<19.4f} {change_pct:>+7.2f}%")
        
        print(f"\n📊 SUMMARY:")
        print(f"   Starting Price: ₹{last_price:.4f}")
        print(f"   Day 1 Prediction: ₹{predictions[0]:.4f}")
        print(f"   Day 7 Prediction: ₹{predictions[-1]:.4f}")
        print(f"   7-Day Change: {((predictions[-1] - last_price) / last_price) * 100:+.2f}%")
        print(f"   Highest: ₹{max(predictions):.4f}")
        print(f"   Lowest: ₹{min(predictions):.4f}")
        print(f"   Volatility: ₹{np.std(predictions):.4f}")
    
    def predict_next_6_months(self):
        """Predict next 6 months using 6MO model"""
        if '6mo' not in self.results:
            print("\n❌ No 6MO model available for 6-month prediction")
            return
        
        print(f"\n{'='*70}")
        print(f"🔮 PREDICTION: NEXT 6 MONTHS (LSTM)")
        print(f"{'='*70}")
        
        data = self.results['6mo']
        model = data['model']
        scaler = data['scaler']
        lookback = data['lookback']
        df = data['df']
        X = data['X']
        
        # Get last prices for reference
        last_price = float(df['Close'].values[-1])
        last_date = pd.to_datetime(df.iloc[-1].get('Datetime', df.iloc[-1].get('Date')))
        
        # Predict next 180 days
        print(f"\n📥 Last Price (Daily interval): ₹{last_price:.4f}")
        predictions = self.predict_future(model, X, scaler, lookback, last_price, 180, '6mo')
        
        if not predictions:
            print("   ⚠️  No valid predictions generated")
            return
        
        print(f"\n📊 PREDICTIONS:")
        print(f"{'Period':<20} {'Days':<8} {'Predicted Price':<20} {'Change %':<15}")
        print(f"{'-'*65}")
        
        # Monthly milestones
        milestones = [7, 30, 60, 90, 120, 150, min(180, len(predictions))]
        for days in milestones:
            if days <= len(predictions):
                pred = predictions[days - 1]
                future_date = last_date + timedelta(days=days)
                change_pct = ((pred - last_price) / last_price) * 100
                print(f"{str(future_date):<20} {days:<8} ₹{pred:<19.4f} {change_pct:>+7.2f}%")
        
        print(f"\n📊 SUMMARY:")
        print(f"   Current Price: ₹{last_price:.4f}")
        print(f"   Predictions Generated: {len(predictions)} days")
        if len(predictions) >= 180:
            print(f"   6-Month End Date: {last_date + timedelta(days=180)}")
            print(f"   6-Month Prediction: ₹{predictions[179]:.4f}")
            print(f"   6-Month Change: {((predictions[179] - last_price) / last_price) * 100:+.2f}%")
        else:
            print(f"   Final Date: {last_date + timedelta(days=len(predictions))}")
            print(f"   Final Prediction: ₹{predictions[-1]:.4f}")
            print(f"   Total Change: {((predictions[-1] - last_price) / last_price) * 100:+.2f}%")
        
        print(f"   Highest Predicted: ₹{max(predictions):.4f}")
        print(f"   Lowest Predicted: ₹{min(predictions):.4f}")
        print(f"   Volatility: ₹{np.std(predictions):.4f}")
    
    def print_summary(self):
        """Print overall summary"""
        print(f"\n{'='*70}")
        print(f"SUMMARY - LSTM MODEL TRAINING RESULTS")
        print(f"{'='*70}")
        
        if not self.results:
            print("   ⚠️  No models were trained.")
            return
        
        period_order = ['1d', '1wk', '1mo', '6mo']
        
        for period in period_order:
            if period not in self.results:
                continue
            
            data = self.results[period]
            print(f"\n{data['period'].upper()} ({data['interval'].upper()} interval):")
            print(f"  Data Points: {len(data['df'])}")
            print(f"  Lookback: {data['lookback']}")
            print(f"  Training Time: {data['train_time']:.2f}s")
            print(f"  Model: LSTM (Deep Learning)")

# Run the pipeline
if __name__ == "__main__":
    symbol = input("Enter stock symbol (e.g., RELIANCE.NS, TCS.NS, INFY.NS): ").strip() or "RELIANCE.NS"
    
    pipeline = StockPredictionPipeline(symbol=symbol)
    pipeline.run()
    pipeline.print_summary()
    
    # Predictions
    pipeline.predict_next_7_days()
    pipeline.predict_next_6_months()
    
    print(f"\n✅ Prediction completed successfully!")
