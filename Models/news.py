import requests
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
# from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Load environment variables from .env
# load_dotenv()

API_BASE_URL = os.getenv("STOCK_API_BASE_URL", "http://127.0.0.1:8000")
OPENROUTER_API_KEY = None # os.getenv("OPENROUTER_API_KEY")
NEWSORG_API_KEY = None # os.getenv("NEWSORG_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
NEWSORG_URL = "https://newsapi.org/v2/everything"

class AdvancedStockNewsPrediction:
    def __init__(self, symbol="RELIANCE.NS", openrouter_api_key=None, newsorg_api_key=None):
        self.symbol = symbol
        self.company_name = symbol.split('.')[0]  # Extract company name
        self.results = {}
        
        # Initialize API keys (Argument > Environment Variable)
        self.openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        self.newsorg_api_key = newsorg_api_key or os.getenv("NEWSORG_API_KEY")
        
        # Validate API keys
        if not self.openrouter_api_key:
            print("❌ OPENROUTER_API_KEY not provided")
        if not self.newsorg_api_key:
            print("❌ NEWSORG_API_KEY not provided")
    
    def fetch_current_price(self):
        """Fetch current stock price from API"""
        print(f"\n📥 Fetching current price for {self.symbol}...")
        params = {"symbol": self.symbol, "interval": "1d"}
        
        try:
            resp = requests.get(f"{API_BASE_URL}/candles", params=params)
            resp.raise_for_status()
            data = resp.json()
            
            if data.get('data') and len(data['data']) > 0:
                current_price = float(data['data'][-1]['Close'])
                print(f"   ✓ Current Price: ₹{current_price:.4f}")
                return current_price
            else:
                print(f"   ✗ Failed to fetch price")
                return None
        except Exception as e:
            print(f"   ✗ Error: {str(e)[:50]}")
            return None
    
    def fetch_news_from_newsorg(self):
        """Fetch real news from NewsOrg API with date filtering"""
        print(f"\n📰 Fetching real news from NewsOrg for {self.company_name}...")
        
        try:
            # Calculate date range (last 30 days)
            from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            to_date = datetime.now().strftime('%Y-%m-%d')
            
            params = {
                "q": self.company_name,
                "sortBy": "publishedAt",
                "language": "en",
                "from": from_date,
                "to": to_date,
                "apiKey": self.newsorg_api_key,
                "pageSize": 15
            }
            
            resp = requests.get(NEWSORG_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
            
            if data.get('status') == 'ok' and data.get('articles'):
                articles = data['articles']
                print(f"   ✓ Retrieved {len(articles)} news articles (Last 30 days)")
                return articles
            else:
                print(f"   ✗ No articles found or API error: {data.get('message', 'Unknown')}")
                return []
        except Exception as e:
            print(f"   ✗ Error fetching news: {str(e)[:50]}")
            return []
    
    def calculate_news_age_impact(self, published_date):
        """Calculate how fresh/old a news is and its impact weight"""
        try:
            pub_date = datetime.fromisoformat(published_date.replace('Z', '+00:00'))
            days_old = (datetime.now(pub_date.tzinfo) - pub_date).days
            
            # Exponential decay: newer news has more weight
            if days_old <= 1:
                age_weight = 1.0  # Very fresh news
            elif days_old <= 7:
                age_weight = 0.8  # Fresh (last week)
            elif days_old <= 14:
                age_weight = 0.5  # Medium age
            else:
                age_weight = 0.2  # Older news
            
            return age_weight, days_old
        except:
            return 0.3, 30
    
    def analyze_news_sentiment(self, articles):
        """Analyze sentiment of news articles with age consideration"""
        print(f"\n🔍 Analyzing sentiment of {len(articles)} articles with age weighting...")
        
        if not articles:
            return []
        
        # Prepare article summary with dates
        articles_text = ""
        for idx, article in enumerate(articles[:10], 1):
            published = article.get('publishedAt', 'N/A')[:10]
            age_weight, days_old = self.calculate_news_age_impact(article.get('publishedAt', ''))
            articles_text += f"{idx}. [{published} - {days_old}d ago] {article.get('title', 'N/A')}\n"
            articles_text += f"   Description: {article.get('description', 'N/A')[:100]}...\n"
            articles_text += f"   Age Weight: {age_weight:.1f}/1.0\n\n"
        
        prompt = f"""
Analyze the following {self.company_name} stock news articles considering how RECENT or OLD each news is.
Older news has less impact on price movements. Recent news (last 3 days) has HIGH impact.

{articles_text}

For each article, determine:
1. Sentiment (positive/negative/neutral)
2. Impact Area (earnings, expansion, acquisition, regulatory, competition, technology, market)
3. Recency Score (1-10, where 10 = very recent and hot topic, 1 = old and forgotten)
4. Market Reaction Potential (1-10, considering both sentiment AND recency)
5. Expected Duration of Impact (days)

Format as JSON:
{{
    "analyzed_articles": [
        {{
            "headline": "headline",
            "sentiment": "positive/negative/neutral",
            "impact_area": "category",
            "recency_score": 9,
            "market_reaction_potential": 8,
            "expected_duration_days": 3,
            "key_points": ["point1", "point2"]
        }}
    ]
}}

Only return valid JSON.
"""
        
        try:
            response = requests.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {self.openrouter_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.6,
                }
            )
            response.raise_for_status()
            
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                text = result['choices'][0]['message']['content']
                
                try:
                    json_start = text.find('{')
                    json_end = text.rfind('}') + 1
                    if json_start != -1 and json_end > json_start:
                        json_str = text[json_start:json_end]
                        analysis = json.loads(json_str)
                        print(f"   ✓ Sentiment analysis with recency completed")
                        return analysis.get('analyzed_articles', [])
                except:
                    print(f"   ✗ Failed to parse sentiment JSON")
                    return []
        except Exception as e:
            print(f"   ✗ Error analyzing sentiment: {str(e)[:50]}")
        
        return []
    
    def analyze_market_sector_trend(self, symbol, current_price):
        """Analyze overall market and sector trends affecting stock"""
        print(f"\n📊 Analyzing market & sector trends impacting {symbol}...")
        
        prompt = f"""
For {symbol} stock (current price: ₹{current_price:.2f}), comprehensively analyze:

1. Overall market trend (bullish/bearish/mixed) and its impact on stock
2. Sector performance and growth trajectory
3. Industry headwinds vs tailwinds
4. Market volatility context (high/medium/low) and risk level
5. Macroeconomic factors (interest rates, inflation, GDP growth, etc.)
6. Competitive landscape changes and market disruption risks
7. Regulatory environment impact

Format as JSON:
{{
    "market_trend": "bullish/bearish/mixed",
    "market_trend_strength": 7,
    "market_trend_influence_on_stock": "high/medium/low",
    "sector_performance": "strong/average/weak",
    "sector_growth_outlook": 7,
    "industry_tailwinds": ["tailwind1", "tailwind2"],
    "industry_headwinds": ["headwind1", "headwind2"],
    "market_volatility": "high/medium/low",
    "volatility_risk_level": 6,
    "macroeconomic_factors": ["factor1", "factor2"],
    "competitive_changes": "description",
    "regulatory_impact": "positive/negative/neutral"
}}

Only return valid JSON.
"""
        
        try:
            response = requests.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {self.openrouter_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.5,
                }
            )
            response.raise_for_status()
            
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                text = result['choices'][0]['message']['content']
                
                try:
                    json_start = text.find('{')
                    json_end = text.rfind('}') + 1
                    if json_start != -1 and json_end > json_start:
                        json_str = text[json_start:json_end]
                        trend = json.loads(json_str)
                        print(f"   ✓ Market & sector trend analysis completed")
                        return trend
                except:
                    print(f"   ✗ Failed to parse trend JSON")
                    return None
        except Exception as e:
            print(f"   ✗ Error analyzing trends: {str(e)[:50]}")
        
        return None
    
    def analyze_events_and_volatility(self, articles):
        """Identify major events, volatility triggers, and their duration"""
        print(f"\n⚡ Analyzing events, volatility & catalysts...")
        
        articles_text = "\n".join([f"- [{a.get('publishedAt', 'N/A')[:10]}] {a.get('title', 'N/A')}" 
                                   for a in articles[:7]])
        
        prompt = f"""
From these {self.company_name} news articles, identify SPECIFIC EVENTS and their impact:

{articles_text}

Identify and analyze:
1. EARNINGS-RELATED: earnings announcement, earnings miss/beat, guidance change
2. ACQUISITIONS: M&A activity, partnership announcements
3. REGULATORY: policy changes, regulatory approvals/rejections, compliance issues
4. VOLATILITY EVENTS: stock split, bonus announcement, IPO, listing changes
5. OPERATIONAL: product launches, plant shutdowns, expansion, capacity changes
6. MARKET SHOCKS: sudden price movements, competitor actions, market crashes
7. CATALYSTS: upcoming events that could move the stock

For each event:
- Type and name
- Volatility impact score (1-10, where 10 = extreme volatility expected)
- Expected duration (days the impact will be felt)
- Direction impact (up/down/mixed)
- Confidence level (high/medium/low)

Format as JSON:
{{
    "identified_events": [
        {{
            "event_type": "earnings/acquisition/regulatory/volatility/operational/market_shock",
            "event_name": "specific name",
            "event_date": "2025-12-15",
            "volatility_impact": 8,
            "expected_duration_days": 7,
            "expected_direction": "up/down/mixed",
            "confidence_level": "high",
            "details": "why this matters"
        }}
    ],
    "overall_volatility_score": 7,
    "next_major_catalyst": "event name and timing",
    "risk_events": ["risk1", "risk2"],
    "opportunity_events": ["opp1", "opp2"]
}}

Only return valid JSON.
"""
        
        try:
            response = requests.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {self.openrouter_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.6,
                }
            )
            response.raise_for_status()
            
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                text = result['choices'][0]['message']['content']
                
                try:
                    json_start = text.find('{')
                    json_end = text.rfind('}') + 1
                    if json_start != -1 and json_end > json_start:
                        json_str = text[json_start:json_end]
                        events = json.loads(json_str)
                        print(f"   ✓ Events & volatility analysis completed")
                        return events
                except:
                    print(f"   ✗ Failed to parse events JSON")
                    return None
        except Exception as e:
            print(f"   ✗ Error analyzing events: {str(e)[:50]}")
        
        return None
    
    def predict_stock_price(self, current_price, sentiment_analysis, market_trend, events_volatility):
        """Generate comprehensive price prediction considering all factors"""
        print(f"\n🔮 Generating price predictions (30-day forecast)...")
        
        # 1. NEWS & SENTIMENT IMPACT (40% weight)
        positive_count = sum(1 for a in sentiment_analysis if a.get('sentiment') == 'positive')
        negative_count = sum(1 for a in sentiment_analysis if a.get('sentiment') == 'negative')
        total_articles = max(len(sentiment_analysis), 1)
        
        sentiment_score = (positive_count - negative_count) / total_articles
        recency_weighted_sentiment = 0
        
        # Weight recent news more heavily
        for article in sentiment_analysis:
            multiplier = article.get('recency_score', 5) / 10
            direction = 1 if article.get('sentiment') == 'positive' else -1 if article.get('sentiment') == 'negative' else 0
            reaction_potential = article.get('market_reaction_potential', 5) / 10
            recency_weighted_sentiment += direction * multiplier * reaction_potential
        
        recency_weighted_sentiment /= total_articles
        sentiment_impact = recency_weighted_sentiment * 0.04  # 0-4% impact
        
        # 2. MARKET & SECTOR TREND IMPACT (35% weight)
        market_impact = 0
        if market_trend:
            trend = market_trend.get('market_trend', 'mixed')
            strength = market_trend.get('market_trend_strength', 5) / 10
            influence = market_trend.get('market_trend_influence_on_stock', 'medium')
            
            influence_weight = {'high': 1.0, 'medium': 0.6, 'low': 0.3}.get(influence, 0.6)
            
            if trend == 'bullish':
                market_impact = 0.035 * strength * influence_weight
            elif trend == 'bearish':
                market_impact = -0.035 * strength * influence_weight
        
        # 3. EVENTS & VOLATILITY IMPACT (25% weight)
        volatility_impact = 0
        event_direction_impact = 0
        
        if events_volatility and isinstance(events_volatility, dict):
            vol_score = events_volatility.get('overall_volatility_score', 5) / 10
            
            # Volatility creates uncertainty
            volatility_impact = (np.random.random() - 0.5) * 0.04 * vol_score
            
            # Check identified events direction
            for event in events_volatility.get('identified_events', []):
                direction = event.get('expected_direction', 'mixed')
                confidence = {'high': 1.0, 'medium': 0.7, 'low': 0.4}.get(event.get('confidence_level', 'medium'), 0.7)
                vol_rating = event.get('volatility_impact', 5) / 10
                
                if direction == 'up':
                    event_direction_impact += 0.02 * confidence * vol_rating
                elif direction == 'down':
                    event_direction_impact -= 0.02 * confidence * vol_rating
            
            event_count = len(events_volatility.get('identified_events', []))
            if event_count > 0:
                event_direction_impact /= event_count
        
        # Total expected change
        total_impact = sentiment_impact + market_impact + event_direction_impact + (volatility_impact * 0.5)
        
        # Cap at ±8% for realism
        total_impact = np.clip(total_impact, -0.08, 0.08)
        
        # Generate 30-day predictions (gradual convergence)
        predictions = []
        for day in range(1, 31):
            # Gradual convergence to target price
            progress = (day / 30) ** 1.5  # Non-linear convergence (faster initially)
            day_price = current_price * (1 + total_impact * progress)
            
            # Add intra-day volatility
            daily_vol = (np.random.random() - 0.5) * current_price * 0.02
            day_price += daily_vol
            
            predictions.append(round(day_price, 4))
        
        return {
            'predictions_30day': predictions,
            'sentiment_score': sentiment_score,
            'sentiment_impact': sentiment_impact * 100,
            'market_impact': market_impact * 100,
            'event_direction_impact': event_direction_impact * 100,
            'volatility_impact': volatility_impact * 100,
            'total_impact': total_impact * 100,
            'expected_price_30day': round(current_price * (1 + total_impact), 4),
            'impact_breakdown': {
                'news_sentiment': 40,
                'market_sector': 35,
                'events_volatility': 25
            }
        }
    
    def print_comprehensive_report(self, symbol, current_price, articles, sentiment_analysis, 
                                   market_trend, events_volatility, predictions):
        """Print professional comprehensive analysis report"""
        print(f"\n{'='*85}")
        print(f"ADVANCED STOCK NEWS PREDICTION REPORT - {symbol.upper()}")
        print(f"{'='*85}")
        
        print(f"\n📈 CURRENT STATUS:")
        print(f"   Stock Symbol: {symbol}")
        print(f"   Current Price: ₹{current_price:.4f}")
        print(f"   Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Latest News Summary
        if articles:
            print(f"\n📰 LATEST NEWS ({len(articles)} articles from last 30 days):")
            print(f"{'-'*85}")
            for idx, article in enumerate(articles[:5], 1):
                title = article.get('title', 'N/A')[:65]
                published = article.get('publishedAt', 'N/A')[:10]
                source = article.get('source', {}).get('name', 'N/A')[:20]
                
                # Calculate age
                age_weight, days_old = self.calculate_news_age_impact(article.get('publishedAt', ''))
                recency_indicator = "🔥" if days_old <= 1 else "⚡" if days_old <= 7 else "📰"
                
                print(f"{idx}. {recency_indicator} [{published}] {title}...")
                print(f"   Source: {source} | Age: {days_old}d old (Weight: {age_weight:.1f})")
        
        # Sentiment Analysis with Recency
        if sentiment_analysis:
            print(f"\n💭 SENTIMENT ANALYSIS (with Recency Weighting):")
            print(f"{'-'*85}")
            positive = sum(1 for a in sentiment_analysis if a.get('sentiment') == 'positive')
            negative = sum(1 for a in sentiment_analysis if a.get('sentiment') == 'negative')
            neutral = sum(1 for a in sentiment_analysis if a.get('sentiment') == 'neutral')
            
            print(f"   Overall: {positive} Positive | {negative} Negative | {neutral} Neutral")
            print(f"\n   Top News by Market Reaction Potential:")
            print(f"   {'#':<3} {'Sentiment':<12} {'Impact Area':<20} {'Reaction Potential':<20} {'Recency':<10}")
            print(f"   {'-'*85}")
            
            sorted_articles = sorted(sentiment_analysis, 
                                   key=lambda x: x.get('market_reaction_potential', 0), 
                                   reverse=True)
            
            for idx, article in enumerate(sorted_articles[:5], 1):
                sentiment = article.get('sentiment', 'neutral').upper()
                impact_area = article.get('impact_area', 'general')[:18]
                reaction = article.get('market_reaction_potential', 5)
                recency = article.get('recency_score', 5)
                
                sentiment_icon = "🟢" if sentiment == "POSITIVE" else "🔴" if sentiment == "NEGATIVE" else "🟡"
                
                print(f"   {idx:<3} {sentiment_icon} {sentiment:<10} {impact_area:<20} {reaction:.1f}/10 ({recency:.1f} fresh)")
        
        # Market & Sector Trend
        if market_trend:
            print(f"\n📊 MARKET & SECTOR TREND ANALYSIS:")
            print(f"{'-'*85}")
            
            trend = market_trend.get('market_trend', 'N/A').upper()
            strength = market_trend.get('market_trend_strength', 5)
            trend_influence = market_trend.get('market_trend_influence_on_stock', 'medium').upper()
            
            trend_icon = "📈" if trend == "BULLISH" else "📉" if trend == "BEARISH" else "↔️"
            print(f"   {trend_icon} Market Trend: {trend} (Strength: {strength}/10)")
            print(f"      └─ Influence on {symbol}: {trend_influence}")
            
            print(f"\n   Sector: {market_trend.get('sector_performance', 'N/A').upper()}")
            print(f"   Sector Growth Outlook: {market_trend.get('sector_growth_outlook', 5)}/10")
            print(f"   Market Volatility: {market_trend.get('market_volatility', 'N/A').upper()}")
            
            if market_trend.get('industry_tailwinds'):
                print(f"\n   ✅ Industry Tailwinds (Positive Factors):")
                for tailwind in market_trend.get('industry_tailwinds', []):
                    print(f"      • {tailwind}")
            
            if market_trend.get('industry_headwinds'):
                print(f"\n   ⚠️  Industry Headwinds (Negative Factors):")
                for headwind in market_trend.get('industry_headwinds', []):
                    print(f"      • {headwind}")
            
            if market_trend.get('macroeconomic_factors'):
                print(f"\n   💼 Macroeconomic Factors:")
                for factor in market_trend.get('macroeconomic_factors', []):
                    print(f"      • {factor}")
        
        # Events & Volatility
        if events_volatility:
            print(f"\n⚡ EVENTS & VOLATILITY ANALYSIS:")
            print(f"{'-'*85}")
            print(f"   Overall Volatility Score: {events_volatility.get('overall_volatility_score', 5)}/10")
            
            if events_volatility.get('identified_events'):
                print(f"\n   📅 Identified Events (Catalysts):")
                for event in events_volatility.get('identified_events', [])[:4]:
                    event_type = event.get('event_type', 'unknown').upper()
                    event_name = event.get('event_name', 'N/A')
                    volatility = event.get('volatility_impact', 5)
                    duration = event.get('expected_duration_days', 0)
                    direction = event.get('expected_direction', 'mixed').upper()
                    
                    direction_icon = "📈" if direction == "UP" else "📉" if direction == "DOWN" else "↔️"
                    print(f"      • {event_type}: {event_name}")
                    print(f"        {direction_icon} Direction: {direction} | Volatility: {volatility}/10 | Duration: {duration}d")
            
            print(f"\n   ⚠️  Risk Events to Watch:")
            for risk in events_volatility.get('risk_events', [])[:3]:
                print(f"      ⚠️  {risk}")
            
            if events_volatility.get('opportunity_events'):
                print(f"\n   ✨ Opportunity Events:")
                for opp in events_volatility.get('opportunity_events', [])[:3]:
                    print(f"      ✨ {opp}")
            
            if events_volatility.get('next_major_catalyst'):
                print(f"\n   🎯 Next Major Catalyst: {events_volatility.get('next_major_catalyst', 'N/A')}")
        
        # Price Predictions with Impact Breakdown
        if predictions:
            print(f"\n🔮 PRICE PREDICTION & IMPACT ANALYSIS:")
            print(f"{'-'*85}")
            
            print(f"   Impact Breakdown:")
            print(f"      • News & Sentiment: {predictions['sentiment_impact']:+.2f}% (40% weight)")
            print(f"      • Market & Sector Trend: {predictions['market_impact']:+.2f}% (35% weight)")
            print(f"      • Events & Volatility: {predictions['event_direction_impact']:+.2f}% + {predictions['volatility_impact']:+.2f}% (25% weight)")
            print(f"      {'─' * 75}")
            print(f"      • TOTAL EXPECTED IMPACT: {predictions['total_impact']:+.2f}%")
            
            print(f"\n   Price Forecast:")
            print(f"      Current Price: ₹{current_price:.4f}")
            print(f"      Expected Price (30 days): ₹{predictions['expected_price_30day']:.4f}")
            change_amount = predictions['expected_price_30day'] - current_price
            change_pct = (change_amount / current_price) * 100
            change_icon = "📈" if change_amount > 0 else "📉"
            print(f"      {change_icon} Expected Change: {change_amount:+.4f} ({change_pct:+.2f}%)")
            
            # 30-day forecast table
            print(f"\n   📈 30-DAY PRICE FORECAST:")
            print(f"   {'Week':<8} {'Day 7':<15} {'Day 14':<15} {'Day 21':<15} {'Day 30':<15}")
            print(f"   {'-'*85}")
            
            predictions_list = predictions['predictions_30day']
            for week in range(4):
                day7 = predictions_list[min(6, len(predictions_list)-1)]
                day14 = predictions_list[min(13, len(predictions_list)-1)]
                day21 = predictions_list[min(20, len(predictions_list)-1)]
                day30 = predictions_list[-1] if week == 3 else ""
                
                if week == 3:
                    print(f"   W{week+1:<6} ₹{day7:<14.4f} ₹{day14:<14.4f} ₹{day21:<14.4f} ₹{day30:<14.4f}")
                else:
                    print(f"   W{week+1:<6} ₹{day7:<14.4f} ₹{day14:<14.4f} ₹{day21:<14.4f}")
            
            print(f"\n   Summary Statistics:")
            print(f"      Highest Predicted (30d): ₹{max(predictions_list):.4f}")
            print(f"      Lowest Predicted (30d): ₹{min(predictions_list):.4f}")
            print(f"      Average Predicted: ₹{np.mean(predictions_list):.4f}")
            print(f"      Volatility Range: ₹{max(predictions_list) - min(predictions_list):.4f}")
        
        print(f"\n{'='*85}")
        print(f"✅ Analysis completed at {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*85}\n")
    
    def run(self):
        """Run complete analysis pipeline"""
        print(f"\n{'='*85}")
        print(f"ADVANCED STOCK NEWS PREDICTION ENGINE v2.0")
        print(f"With NewsOrg API + OpenRouter AI + Recency Weighting")
        print(f"{'='*85}")
        
        # Step 1: Fetch current price
        current_price = self.fetch_current_price()
        if not current_price:
            print("❌ Failed to fetch current price")
            return
        
        # Step 2: Fetch real news from NewsOrg
        articles = self.fetch_news_from_newsorg()
        if not articles:
            print("⚠️  No news articles retrieved")
            articles = []
        
        # Step 3: Analyze sentiment with recency weighting
        sentiment_analysis = []
        if articles:
            sentiment_analysis = self.analyze_news_sentiment(articles)
        
        # Step 4: Analyze market & sector trends
        market_trend = self.analyze_market_sector_trend(self.symbol, current_price)
        
        # Step 5: Analyze events & volatility
        events_volatility = None
        if articles:
            events_volatility = self.analyze_events_and_volatility(articles)
        
        # Step 6: Generate price predictions
        predictions = self.predict_stock_price(current_price, sentiment_analysis, market_trend, events_volatility)
        
        # Step 7: Print comprehensive report
        self.print_comprehensive_report(self.symbol, current_price, articles, sentiment_analysis, 
                                       market_trend, events_volatility, predictions)
        
        return {
            'symbol': self.symbol,
            'current_price': current_price,
            'articles': articles,
            'sentiment_analysis': sentiment_analysis,
            'market_trend': market_trend,
            'events_volatility': events_volatility,
            'predictions': predictions
        }

# Main execution
if __name__ == "__main__":
    print("\n" + "="*85)
    print("ADVANCED STOCK NEWS PREDICTION WITH NEWSORG API v2.0")
    print("="*85)
    
    # Check for required API keys
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    if not OPENROUTER_API_KEY:
        print("\n❌ Error: OPENROUTER_API_KEY not found in .env")
        print("   Add to .env: OPENROUTER_API_KEY=your_openrouter_key")
        exit(1)
    
    NEWSORG_API_KEY = os.getenv("NEWSORG_API_KEY")
    if not NEWSORG_API_KEY:
        print("\n❌ Error: NEWSORG_API_KEY not found in .env")
        print("   Add to .env: NEWSORG_API_KEY=your_newsorg_api_key")
        print("   Get free key from: https://newsapi.org")
        exit(1)
    
    print("\n✓ All API Keys loaded from .env file")
    
    symbol = input("\n📊 Enter stock symbol (e.g., RELIANCE.NS, TCS.NS, INFY.NS): ").strip() or "RELIANCE.NS"
    
    predictor = AdvancedStockNewsPrediction(symbol=symbol)
    result = predictor.run()
    
    if result:
        print(f"\n✅ ADVANCED analysis completed successfully!")
        print(f"\n💾 Analysis results available for {result['symbol']}")