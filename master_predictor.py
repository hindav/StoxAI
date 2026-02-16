import subprocess
import sys
import time
import os
import requests
import json
import numpy as np
from datetime import datetime
# from dotenv import load_dotenv

# Import your modules
from Models.stock_prediction import StockPredictionPipeline
from Models.news import AdvancedStockNewsPrediction

# load_dotenv()

class MasterStockPredictor:
    def __init__(self, symbol="RELIANCE.NS"):
        self.symbol = symbol
        self.api_process = None
        self.api_url = "http://127.0.0.1:8000"
        self.technical_results = None
        self.news_results = None
        self.final_prediction = None
        
    def start_api_server(self):
        """Launch FastAPI server in background"""
        print("\n" + "="*85)
        print("STEP 1: LAUNCHING API SERVER")
        print("="*85)
        
        # Check if API is already running
        try:
            resp = requests.get(f"{self.api_url}/", timeout=2)
            if resp.status_code == 200:
                print("✓ API server already running")
                return True
        except:
            pass
        
        # Start API server
        print("🚀 Starting FastAPI server...")
        try:
            self.api_process = subprocess.Popen(
                [sys.executable, "api.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for server to start
            for i in range(10):
                time.sleep(1)
                try:
                    resp = requests.get(f"{self.api_url}/", timeout=1)
                    if resp.status_code == 200:
                        print("✓ API server started successfully")
                        return True
                except:
                    print(f"   Waiting for API... ({i+1}/10)")
            
            print("✗ API server failed to start")
            return False
            
        except Exception as e:
            print(f"✗ Error starting API: {str(e)}")
            return False
    
    def run_technical_analysis(self):
        """Run LSTM-based technical analysis"""
        print("\n" + "="*85)
        print("STEP 2: TECHNICAL ANALYSIS (LSTM)")
        print("="*85)
        
        try:
            pipeline = StockPredictionPipeline(symbol=self.symbol)
            pipeline.run()
            
            # Get 7-day and 6-month predictions
            technical_7day = None
            technical_6month = None
            
            if '1d' in pipeline.results:
                data = pipeline.results['1d']
                model = data['model']
                scaler = data['scaler']
                lookback = data['lookback']
                df = data['df']
                X = data['X']
                last_price = float(df['Close'].values[-1])
                
                predictions_7day = pipeline.predict_future(
                    model, X, scaler, lookback, last_price, 7, '1d'
                )
                
                if predictions_7day:
                    technical_7day = {
                        'current_price': last_price,
                        'predictions': predictions_7day,
                        'day_7_price': predictions_7day[-1],
                        'change_pct': ((predictions_7day[-1] - last_price) / last_price) * 100
                    }
            
            if '6mo' in pipeline.results:
                data = pipeline.results['6mo']
                model = data['model']
                scaler = data['scaler']
                lookback = data['lookback']
                df = data['df']
                X = data['X']
                last_price = float(df['Close'].values[-1])
                
                predictions_6month = pipeline.predict_future(
                    model, X, scaler, lookback, last_price, 180, '6mo'
                )
                
                if predictions_6month and len(predictions_6month) >= 30:
                    technical_6month = {
                        'current_price': last_price,
                        'day_30_price': predictions_6month[29],
                        'change_pct': ((predictions_6month[29] - last_price) / last_price) * 100
                    }
            
            self.technical_results = {
                '7day': technical_7day,
                '30day': technical_6month
            }
            
            print("\n✓ Technical analysis completed")
            return True
            
        except Exception as e:
            print(f"\n✗ Technical analysis failed: {str(e)}")
            return False
    
    def run_news_analysis(self):
        """Run news sentiment analysis"""
        print("\n" + "="*85)
        print("STEP 3: NEWS SENTIMENT ANALYSIS")
        print("="*85)
        
        try:
            predictor = AdvancedStockNewsPrediction(symbol=self.symbol)
            result = predictor.run()
            
            if result:
                self.news_results = {
                    'current_price': result['current_price'],
                    'sentiment_analysis': result['sentiment_analysis'],
                    'market_trend': result['market_trend'],
                    'events_volatility': result['events_volatility'],
                    'predictions': result['predictions']
                }
                
                print("\n✓ News analysis completed")
                return True
            else:
                print("\n✗ News analysis failed")
                return False
                
        except Exception as e:
            print(f"\n✗ News analysis failed: {str(e)}")
            return False
    
    def generate_final_prediction(self):
        """Combine technical and news analysis for final prediction"""
        print("\n" + "="*85)
        print("STEP 4: GENERATING FINAL COMPREHENSIVE PREDICTION")
        print("="*85)
        
        if not self.technical_results or not self.news_results:
            print("✗ Missing required analysis results")
            return False
        
        # Weight allocation
        TECHNICAL_WEIGHT = 0.50  # 50% weight to LSTM technical analysis
        NEWS_WEIGHT = 0.50       # 50% weight to news sentiment
        
        # Get current price (use news price as it's most recent)
        current_price = self.news_results['current_price']
        
        # === 7-DAY PREDICTION ===
        technical_7day = self.technical_results.get('7day')
        news_30day = self.news_results['predictions']
        
        if technical_7day and news_30day:
            # Technical prediction for 7 days
            tech_7day_change = technical_7day['change_pct']
            
            # News prediction (use avg of first 7 days from 30-day forecast)
            news_predictions = news_30day['predictions_30day'][:7]
            news_7day_price = np.mean(news_predictions)
            news_7day_change = ((news_7day_price - current_price) / current_price) * 100
            
            # Weighted average
            final_7day_change = (tech_7day_change * TECHNICAL_WEIGHT + 
                               news_7day_change * NEWS_WEIGHT)
            
            final_7day_price = current_price * (1 + final_7day_change / 100)
            
            prediction_7day = {
                'current_price': current_price,
                'technical_prediction': technical_7day['day_7_price'],
                'technical_change_pct': tech_7day_change,
                'news_prediction': news_7day_price,
                'news_change_pct': news_7day_change,
                'final_prediction': final_7day_price,
                'final_change_pct': final_7day_change,
                'confidence_score': self.calculate_confidence(
                    tech_7day_change, news_7day_change
                )
            }
        else:
            prediction_7day = None
        
        # === 30-DAY PREDICTION ===
        technical_30day = self.technical_results.get('30day')
        
        if technical_30day and news_30day:
            # Technical prediction for 30 days
            tech_30day_change = technical_30day['change_pct']
            
            # News prediction for 30 days
            news_30day_price = news_30day['expected_price_30day']
            news_30day_change = news_30day['total_impact']
            
            # Weighted average
            final_30day_change = (tech_30day_change * TECHNICAL_WEIGHT + 
                                news_30day_change * NEWS_WEIGHT)
            
            final_30day_price = current_price * (1 + final_30day_change / 100)
            
            prediction_30day = {
                'current_price': current_price,
                'technical_prediction': technical_30day['day_30_price'],
                'technical_change_pct': tech_30day_change,
                'news_prediction': news_30day_price,
                'news_change_pct': news_30day_change,
                'final_prediction': final_30day_price,
                'final_change_pct': final_30day_change,
                'confidence_score': self.calculate_confidence(
                    tech_30day_change, news_30day_change
                ),
                'news_impact_breakdown': news_30day.get('impact_breakdown', {})
            }
        else:
            prediction_30day = None
        
        self.final_prediction = {
            '7_day': prediction_7day,
            '30_day': prediction_30day,
            'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'weights': {
                'technical_analysis': TECHNICAL_WEIGHT * 100,
                'news_sentiment': NEWS_WEIGHT * 100
            }
        }
        
        return True
    
    def calculate_confidence(self, tech_change, news_change):
        """Calculate confidence score based on agreement between models"""
        # If both agree on direction, confidence is higher
        same_direction = (tech_change * news_change) > 0
        
        # Calculate magnitude difference
        diff = abs(tech_change - news_change)
        
        if same_direction:
            # High agreement = high confidence
            if diff < 2:
                return 90  # Very high confidence
            elif diff < 5:
                return 75  # High confidence
            elif diff < 10:
                return 60  # Medium confidence
            else:
                return 45  # Lower confidence
        else:
            # Opposing signals = low confidence
            return 30
    
    def print_final_report(self):
        """Print comprehensive final report"""
        if not self.final_prediction:
            print("✗ No final prediction available")
            return
        
        print("\n" + "="*85)
        print("FINAL COMPREHENSIVE STOCK PREDICTION REPORT")
        print(f"Symbol: {self.symbol}")
        print(f"Analysis Time: {self.final_prediction['analysis_time']}")
        print("="*85)
        
        # Model weights
        print(f"\n📊 MODEL WEIGHTS:")
        print(f"   • Technical Analysis (LSTM): {self.final_prediction['weights']['technical_analysis']:.0f}%")
        print(f"   • News Sentiment Analysis: {self.final_prediction['weights']['news_sentiment']:.0f}%")
        
        # 7-Day Prediction
        if self.final_prediction['7_day']:
            pred = self.final_prediction['7_day']
            print(f"\n" + "-"*85)
            print(f"🎯 7-DAY PREDICTION")
            print("-"*85)
            print(f"   Current Price: ₹{pred['current_price']:.4f}")
            print(f"\n   Technical Model:")
            print(f"      • Predicted Price: ₹{pred['technical_prediction']:.4f}")
            print(f"      • Change: {pred['technical_change_pct']:+.2f}%")
            print(f"\n   News Sentiment Model:")
            print(f"      • Predicted Price: ₹{pred['news_prediction']:.4f}")
            print(f"      • Change: {pred['news_change_pct']:+.2f}%")
            
            direction = "📈 UP" if pred['final_change_pct'] > 0 else "📉 DOWN"
            confidence_icon = "🟢" if pred['confidence_score'] >= 70 else "🟡" if pred['confidence_score'] >= 50 else "🔴"
            
            print(f"\n   {confidence_icon} FINAL PREDICTION:")
            print(f"      • Expected Price (Day 7): ₹{pred['final_prediction']:.4f}")
            print(f"      • Expected Change: {pred['final_change_pct']:+.2f}% {direction}")
            print(f"      • Confidence Score: {pred['confidence_score']}/100")
        
        # 30-Day Prediction
        if self.final_prediction['30_day']:
            pred = self.final_prediction['30_day']
            print(f"\n" + "-"*85)
            print(f"🎯 30-DAY PREDICTION")
            print("-"*85)
            print(f"   Current Price: ₹{pred['current_price']:.4f}")
            print(f"\n   Technical Model:")
            print(f"      • Predicted Price: ₹{pred['technical_prediction']:.4f}")
            print(f"      • Change: {pred['technical_change_pct']:+.2f}%")
            print(f"\n   News Sentiment Model:")
            print(f"      • Predicted Price: ₹{pred['news_prediction']:.4f}")
            print(f"      • Change: {pred['news_change_pct']:+.2f}%")
            
            direction = "📈 UP" if pred['final_change_pct'] > 0 else "📉 DOWN"
            confidence_icon = "🟢" if pred['confidence_score'] >= 70 else "🟡" if pred['confidence_score'] >= 50 else "🔴"
            
            print(f"\n   {confidence_icon} FINAL PREDICTION:")
            print(f"      • Expected Price (Day 30): ₹{pred['final_prediction']:.4f}")
            print(f"      • Expected Change: {pred['final_change_pct']:+.2f}% {direction}")
            print(f"      • Confidence Score: {pred['confidence_score']}/100")
            
            if 'news_impact_breakdown' in pred:
                print(f"\n   📰 News Impact Breakdown:")
                breakdown = pred['news_impact_breakdown']
                print(f"      • News Sentiment: {breakdown.get('news_sentiment', 0)}% weight")
                print(f"      • Market/Sector: {breakdown.get('market_sector', 0)}% weight")
                print(f"      • Events/Volatility: {breakdown.get('events_volatility', 0)}% weight")
        
        # Investment Recommendation
        print(f"\n" + "="*85)
        print("💡 RECOMMENDATION")
        print("="*85)
        
        if self.final_prediction['30_day']:
            pred = self.final_prediction['30_day']
            change = pred['final_change_pct']
            confidence = pred['confidence_score']
            
            if change > 5 and confidence >= 60:
                recommendation = "STRONG BUY"
                icon = "🟢🟢"
            elif change > 2 and confidence >= 50:
                recommendation = "BUY"
                icon = "🟢"
            elif change < -5 and confidence >= 60:
                recommendation = "STRONG SELL"
                icon = "🔴🔴"
            elif change < -2 and confidence >= 50:
                recommendation = "SELL"
                icon = "🔴"
            else:
                recommendation = "HOLD"
                icon = "🟡"
            
            print(f"   {icon} {recommendation}")
            print(f"\n   Rationale:")
            if confidence >= 70:
                print(f"      • High model agreement (Confidence: {confidence}/100)")
            elif confidence >= 50:
                print(f"      • Moderate model agreement (Confidence: {confidence}/100)")
            else:
                print(f"      • Low model agreement - proceed with caution (Confidence: {confidence}/100)")
            
            if abs(change) > 5:
                print(f"      • Significant price movement expected ({change:+.2f}%)")
            elif abs(change) > 2:
                print(f"      • Moderate price movement expected ({change:+.2f}%)")
            else:
                print(f"      • Minimal price movement expected ({change:+.2f}%)")
        
        print(f"\n" + "="*85)
        print("⚠️  DISCLAIMER")
        print("="*85)
        print("   This prediction is for educational purposes only.")
        print("   Not financial advice. Invest at your own risk.")
        print("="*85 + "\n")
    
    def save_results(self):
        """Save results to JSON file"""
        if not self.final_prediction:
            return
        
        filename = f"{self.symbol.replace('.', '_')}_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump({
                    'symbol': self.symbol,
                    'final_prediction': self.final_prediction,
                    'technical_results': {
                        '7day': self.technical_results['7day'] if self.technical_results and self.technical_results.get('7day') else None,
                        '30day': self.technical_results['30day'] if self.technical_results and self.technical_results.get('30day') else None
                    } if self.technical_results else None,
                    'news_summary': {
                        'current_price': self.news_results['current_price'],
                        'total_impact': self.news_results['predictions']['total_impact'],
                        'sentiment_score': self.news_results['predictions']['sentiment_score']
                    } if self.news_results else None
                }, f, indent=2, default=str)
            
            print(f"✓ Results saved to: {filename}")
        except Exception as e:
            print(f"✗ Failed to save results: {str(e)}")
    
    def cleanup(self):
        """Clean up resources"""
        if self.api_process:
            print("\n🛑 Shutting down API server...")
            self.api_process.terminate()
            self.api_process.wait(timeout=5)
            print("✓ API server stopped")
    
    def run(self):
        """Main execution pipeline"""
        try:
            # Step 1: Start API
            if not self.start_api_server():
                return False
            
            time.sleep(2)  # Give API time to stabilize
            
            # Step 2: Technical Analysis
            if not self.run_technical_analysis():
                return False
            
            # Step 3: News Analysis
            if not self.run_news_analysis():
                return False
            
            # Step 4: Generate Final Prediction
            if not self.generate_final_prediction():
                return False
            
            # Step 5: Print Report
            self.print_final_report()
            
            # Step 6: Save Results
            self.save_results()
            
            return True
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Process interrupted by user")
            return False
        except Exception as e:
            print(f"\n\n✗ Fatal error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self.cleanup()

if __name__ == "__main__":
    print("\n" + "="*85)
    print("MASTER STOCK PREDICTION SYSTEM")
    print("Combines LSTM Technical Analysis + News Sentiment Analysis")
    print("="*85)
    
    # Validate environment
    required_keys = ['OPENROUTER_API_KEY', 'NEWSORG_API_KEY']
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        print("\n✗ Missing required API keys in .env:")
        for key in missing_keys:
            print(f"   • {key}")
        print("\nPlease add them to your .env file")
        sys.exit(1)
    
    symbol = input("\n📊 Enter stock symbol (e.g., RELIANCE.NS, TCS.NS): ").strip() or "RELIANCE.NS"
    
    predictor = MasterStockPredictor(symbol=symbol)
    success = predictor.run()
    
    if success:
        print("\n✅ Analysis completed successfully!")
    else:
        print("\n❌ Analysis failed. Check errors above.")
    
    sys.exit(0 if success else 1)
