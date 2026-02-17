from flask import Flask, render_template, jsonify, request, Response, stream_with_context
from flask_cors import CORS
import subprocess
import sys
import time
import os
# Suppress TensorFlow OneDNN messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import requests
import json
import numpy as np
from datetime import datetime
# from dotenv import load_dotenv
import io
import contextlib

import threading
import queue

# Redirect stdout to suppress prints
import warnings
warnings.filterwarnings('ignore')

# load_dotenv()

app = Flask(__name__)
CORS(app)

# Global state
api_process = None
api_running = False

def suppress_prints():
    """Context manager to suppress all print statements"""
    return contextlib.redirect_stdout(io.StringIO())

class PredictionEngine:
    def __init__(self, symbol, openrouter_key=None, newsorg_key=None):
        self.symbol = symbol
        self.openrouter_key = openrouter_key
        self.newsorg_key = newsorg_key
        self.api_url = "http://127.0.0.1:8000"
        
    def ensure_api_running(self):
        """Check if API is running, start if needed"""
        global api_process, api_running
        
        try:
            resp = requests.get(f"{self.api_url}/", timeout=2)
            if resp.status_code == 200:
                api_running = True
                return True
        except:
            pass
        
        if not api_running:
            try:
                # Start API using uvicorn as a module
                # In deployment, we need to be careful with paths and environments
                print("Attempting to start internal API server...")
                api_process = subprocess.Popen(
                    [sys.executable, "-m", "uvicorn", "Api.api:app", "--host", "127.0.0.1", "--port", "8000"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=os.getcwd() # Ensure we run from the project root
                )
                
                # Check for immediate failure
                try:
                    outs, errs = api_process.communicate(timeout=0.5)
                    if api_process.returncode is not None and api_process.returncode != 0:
                        print(f"API failed to start immediately. Error: {errs.decode('utf-8')}")
                        return False
                except subprocess.TimeoutExpired:
                    # This is good, it means it's running
                    pass

                for i in range(15):
                    time.sleep(1)
                    try:
                        resp = requests.get(f"{self.api_url}/", timeout=1)
                        if resp.status_code == 200:
                            api_running = True
                            print("Internal API server started successfully.")
                            return True
                    except:
                        if api_process.poll() is not None:
                            # Process died
                            _, errs = api_process.communicate()
                            print(f"API process died during startup. Error: {errs.decode('utf-8')}")
                            return False
                        continue
            except Exception as e:
                print(f"Exception starting API: {e}")
                pass
        
        return api_running
    
    def run_technical_analysis(self, progress_callback=None):
        """Run LSTM technical analysis"""
        with suppress_prints():
            from Models.stock_prediction import StockPredictionPipeline
            
            pipeline = StockPredictionPipeline(symbol=self.symbol)
            pipeline.run(progress_callback=progress_callback)
            
            results = {'1day': None, '7day': None, '30day': None, '6month': None}
            
            # Use '6mo' model (Daily Data) for all forecasts if available
            # This ensures 1 step = 1 day
            if '6mo' in pipeline.results:
                data = pipeline.results['6mo']
                model = data['model']
                scaler = data['scaler']
                lookback = data['lookback']
                df = data['df']
                X = data['X']
                last_price = float(df['Close'].values[-1])
                
                # Predict next 180 days
                predictions = pipeline.predict_future(
                    model, X, scaler, lookback, last_price, 180, '6mo'
                )
                
                if predictions:
                    # 1-Day Result (Index 0)
                    if len(predictions) >= 1:
                        results['1day'] = {
                            'current_price': last_price,
                            'day_1_price': predictions[0],
                            'change_pct': ((predictions[0] - last_price) / last_price) * 100
                        }
                    
                    # 7-Day Result (Index 6)
                    if len(predictions) >= 7:
                        results['7day'] = {
                            'current_price': last_price,
                            'predictions': predictions[:7],
                            'day_7_price': predictions[6],
                            'change_pct': ((predictions[6] - last_price) / last_price) * 100
                        }
                    
                    # 30-Day Result (Index 29)
                    if len(predictions) >= 30:
                        results['30day'] = {
                            'current_price': last_price,
                            'day_30_price': predictions[29],
                            'change_pct': ((predictions[29] - last_price) / last_price) * 100
                        }
                    
                    # 6-Month Result (Index 179 or last)
                    if len(predictions) >= 180:
                        final_idx = 179
                    else:
                        final_idx = len(predictions) - 1
                        
                    results['6month'] = {
                        'current_price': last_price,
                        'day_180_price': predictions[final_idx],
                        'change_pct': ((predictions[final_idx] - last_price) / last_price) * 100
                    }
            return results
    
    def run_news_analysis(self):
        """Run news sentiment analysis"""
        with suppress_prints():
            from Models.news import AdvancedStockNewsPrediction
            
            predictor = AdvancedStockNewsPrediction(
                symbol=self.symbol,
                openrouter_api_key=self.openrouter_key,
                newsorg_api_key=self.newsorg_key
            )
            result = predictor.run()
            
            if result:
                return {
                    'current_price': result['current_price'],
                    'sentiment_analysis': result['sentiment_analysis'],
                    'market_trend': result['market_trend'],
                    'events_volatility': result['events_volatility'],
                    'predictions': result['predictions']
                }
            return None
    
    def calculate_confidence(self, tech_change, news_change):
        """Calculate confidence score"""
        same_direction = (tech_change * news_change) > 0
        diff = abs(tech_change - news_change)
        
        if same_direction:
            if diff < 2:
                return 90
            elif diff < 5:
                return 75
            elif diff < 10:
                return 60
            else:
                return 45
        else:
            return 30
    
    def generate_final_prediction(self, technical_results, news_results):
        """Combine results"""
        TECHNICAL_WEIGHT = 0.50
        NEWS_WEIGHT = 0.50
        
        current_price = news_results['current_price']
        
        # Helper for combination
        def combine_predictions(tech_res_key, news_price_target, news_change_pct, tech_w=0.5, news_w=0.5):
            if not technical_results[tech_res_key]:
                return None
            
            tech_data = technical_results[tech_res_key]
            tech_price = tech_data[f'day_{"1" if tech_res_key == "1day" else "7" if tech_res_key == "7day" else "30" if tech_res_key == "30day" else "180"}_price']
            tech_change = tech_data['change_pct']
            
            final_change = (tech_change * tech_w + news_change_pct * news_w)
            final_price = current_price * (1 + final_change / 100)
            
            return {
                'current_price': current_price,
                'technical_prediction': tech_price,
                'technical_change_pct': tech_change,
                'news_prediction': news_price_target,
                'news_change_pct': news_change_pct,
                'final_prediction': final_price,
                'final_change_pct': final_change,
                'confidence_score': self.calculate_confidence(tech_change, news_change_pct)
            }

        # News data prep
        news_preds_30d = news_results['predictions']['predictions_30day']
        news_30day_change = news_results['predictions']['total_impact']
        
        # 1-Day Prediction
        prediction_1day = None
        if technical_results['1day'] and news_preds_30d:
            news_1day_price = news_preds_30d[0]
            news_1day_change = ((news_1day_price - current_price) / current_price) * 100
            prediction_1day = combine_predictions('1day', news_1day_price, news_1day_change)
            
        # 7-Day Prediction
        prediction_7day = None
        if technical_results['7day'] and news_preds_30d:
            news_7day_price = np.mean(news_preds_30d[:7])
            news_7day_change = ((news_7day_price - current_price) / current_price) * 100
            prediction_7day = combine_predictions('7day', news_7day_price, news_7day_change)
            
        # 30-Day Prediction
        prediction_30day = None
        if technical_results['30day']:
            news_30day_price = news_results['predictions']['expected_price_30day']
            prediction_30day = combine_predictions('30day', news_30day_price, news_30day_change)

        # 6-Month Prediction
        # For long term, weight technicals higher (80%) and use 30-day news trend as general sentiment (20%)
        prediction_6month = None
        if technical_results['6month']:
            # Project news sentiment linearly but with decay, or just use 30-day change as a constant offset
            # Here we'll use the 30-day change percentage as the "news opinion" for the long term direction
            prediction_6month = combine_predictions('6month', 
                                                  current_price * (1 + news_30day_change/100), 
                                                  news_30day_change, 
                                                  tech_w=0.8, news_w=0.2)
        
        # Generate recommendation
        recommendation = None
        if prediction_30day:
            change = prediction_30day['final_change_pct']
            confidence = prediction_30day['confidence_score']
            
            if change > 5 and confidence >= 60:
                recommendation = 'STRONG BUY'
            elif change > 2 and confidence >= 50:
                recommendation = 'BUY'
            elif change < -5 and confidence >= 60:
                recommendation = 'STRONG SELL'
            elif change < -2 and confidence >= 50:
                recommendation = 'SELL'
            else:
                recommendation = 'HOLD'
        
        return {
            'symbol': self.symbol,
            'current_price': current_price,
            'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            '1_day': prediction_1day,
            '7_day': prediction_7day,
            '30_day': prediction_30day,
            '6_month': prediction_6month,
            'recommendation': recommendation,
            'news_impact': {
                'sentiment': news_results['predictions']['sentiment_impact'],
                'market': news_results['predictions']['market_impact'],
                'events': news_results['predictions']['event_direction_impact']
            }
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/search', methods=['GET'])
def search_stocks():
    query = request.args.get('q', '')
    if not query:
        return jsonify([])
    
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount=10&newsCount=0"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        data = response.json()
        
        if 'quotes' in data:
            results = [{
                'symbol': item['symbol'],
                'name': item.get('shortname', item.get('longname', item['symbol'])),
                'exch': item.get('exchDisp', 'Unknown')
            } for item in data['quotes'] if item.get('isYahooFinance', True)] # Filter out non-finance items if possible steps
            return jsonify(results)
        return jsonify([])
    except Exception as e:
        print(f"Search error: {e}")
        return jsonify([])
            
@app.route('/api/predict', methods=['POST'])
def predict():
    @stream_with_context
    def generate():
        try:
            data = request.json
            symbol = data.get('symbol', 'RELIANCE.NS')
            openrouter_key = data.get('openrouter_key')
            newsorg_key = data.get('newsorg_key')
            
            engine = PredictionEngine(symbol, openrouter_key, newsorg_key)
            
            # Step 1: Ensure API is running
            yield f"data: {json.dumps({'step': 1, 'status': 'loading', 'message': 'Starting API server...'})}\n\n"
            
            if not engine.ensure_api_running():
                yield f"data: {json.dumps({'step': 1, 'status': 'error', 'message': 'Failed to start API'})}\n\n"
                return
                
            yield f"data: {json.dumps({'step': 1, 'status': 'completed', 'message': 'API server ready'})}\n\n"
            
            # Step 2: Technical Analysis
            # Step 2: Technical Analysis
            yield f"data: {json.dumps({'step': 2, 'status': 'loading', 'message': 'Starting technical analysis...', 'progress': 0})}\n\n"
            
            q = queue.Queue()
            
            def technical_worker():
                try:
                    def on_progress(p, m):
                        q.put({'type': 'progress', 'progress': p, 'message': m})
                    
                    res = engine.run_technical_analysis(progress_callback=on_progress)
                    q.put({'type': 'result', 'data': res})
                except Exception as e:
                    import traceback
                    q.put({'type': 'error', 'error': str(e), 'traceback': traceback.format_exc()})

            t = threading.Thread(target=technical_worker)
            t.start()
            
            technical_results = None
            
            while True:
                try:
                    # Wait for items with timeout
                    item = q.get(timeout=0.1)
                    
                    if item['type'] == 'progress':
                        yield f"data: {json.dumps({'step': 2, 'status': 'loading', 'message': item['message'], 'progress': item['progress']})}\n\n"
                    elif item['type'] == 'result':
                        technical_results = item['data']
                        break
                    elif item['type'] == 'error':
                        raise Exception(f"Technical Analysis Failed: {item['error']}")
                except queue.Empty:
                    if not t.is_alive():
                        # If thread died without result/error
                        if technical_results is None:
                             raise Exception("Technical analysis thread terminated unexpectedly")
                        break
            
            yield f"data: {json.dumps({'step': 2, 'status': 'completed', 'message': 'Technical analysis complete', 'progress': 100})}\n\n"
            
            # Step 3: News Analysis
            yield f"data: {json.dumps({'step': 3, 'status': 'loading', 'message': 'Analyzing news sentiment...'})}\n\n"
            
            news_results = engine.run_news_analysis()
            
            yield f"data: {json.dumps({'step': 3, 'status': 'completed', 'message': 'News analysis complete'})}\n\n"
            
            # Step 4: Final Prediction
            yield f"data: {json.dumps({'step': 4, 'status': 'loading', 'message': 'Generating final prediction...'})}\n\n"
            
            final_results = engine.generate_final_prediction(technical_results, news_results)
            
            yield f"data: {json.dumps({'step': 4, 'status': 'completed', 'message': 'Prediction complete', 'results': final_results})}\n\n"
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            traceback_msg = traceback.format_exc()
            yield f"data: {json.dumps({'status': 'error', 'message': error_msg, 'traceback': traceback_msg})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    print("\n" + "="*70)
    print("STOCK PREDICTION DASHBOARD")
    print("Flask Server with Modern UI")
    print("="*70)
    print("\nStarting server at http://localhost:5000")
    print("Open your browser and navigate to the URL above\n")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)