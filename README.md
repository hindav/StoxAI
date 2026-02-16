# AI-Powered Stock Prediction System

This project is a comprehensive stock market prediction application that combines **Deep Learning (LSTM)** for technical analysis with **Large Language Models (GPT via OpenRouter)** for fundamental news sentiment analysis. It provides multi-timeframe forecasts (7-day, 30-day, 6-month) and actionable investment recommendations.

## 🚀 Features

-   **Dual-Engine Analysis**:
    -   **Technical**: LSTM (Long Short-Term Memory) neural networks trained on historical price data.
    -   **Fundamental**: AI-powered sentiment analysis of real-time news articles using GPT-3.5/4.
-   **Multi-Timeframe Predictions**: Short-term (7 days), medium-term (30 days), and long-term (6 months).
-   **Consolidated Insights**: Combines technical and fundamental signals into a weighted final prediction with confidence scores.
-   **Interactive Dashboards**:
    -   **Streamlit UI**: A data-science focused dashboard with detailed charts and metrics.
    -   **Flask UI**: A modern web interface for easy access.
-   **Comprehensive Reporting**: Detailed breakdown of market trends, sector performance, volatility events, and buy/sell signals.

## 🛠️ Installation

1.  **Clone the repository** (if applicable) or navigate to the project directory.

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up Environment Variables**:
    Create a `.env` file in the root directory with your API keys:
    ```env
    OPENROUTER_API_KEY=your_openrouter_api_key
    NEWSORG_API_KEY=your_newsapi_key
    STOCK_API_BASE_URL=http://127.0.0.1:8000
    ```
    *Note: You can get a free NewsAPI key from [newsapi.org](https://newsapi.org).*

## 🖥️ Usage

You can run the application in three ways depending on your preference:

### Option 1: Streamlit Dashboard (Recommended for Analytics)
This provides the most detailed visualization and control.

1.  Start the integrated API server:
    ```bash
    python Models/integrated_api.py
    ```
2.  In a separate terminal, launch the Streamlit app:
    ```bash
    streamlit run streamlit_ui.py
    ```
3.  Open your browser at `http://localhost:8501`.

### Option 2: Flask Web Application
A simplified web interface.

1.  Start the Flask server:
    ```bash
    python flask_app.py
    ```
2.  Open your browser at `http://localhost:5000`.

### Option 3: Command Line Interface (CLI)
Run a quick analysis directly in your terminal.

1.  Run the master predictor:
    ```bash
    python master_predictor.py
    ```
2.  Enter the stock symbol (e.g., `RELIANCE.NS`, `TCS.NS`) when prompted.

## 🧠 Workflow

1.  **Data Ingestion**:
    -   Fetches historical stock candle data via the internal API.
    -   Retrieves the latest news articles for the target company via NewsAPI.

2.  **Technical Analysis (LSTM Pipeline)**:
    -   Preprocesses data with dynamic lookback windows.
    -   Trains a custom LSTM neural network on multiple timeframes (Daily, Weekly, Monthly).
    -   Generates future price sequences for 7, 30, and 180 days.

3.  **Fundamental Analysis (News Pipeline)**:
    -   **Sentiment Analysis**: Evaluates news articles for positive/negative sentiment, weighted by recency.
    -   **Market Context**: Analyzes broader sector trends and market conditions.
    -   **Event Detection**: Identifies catalysts like earnings, mergers, or regulatory changes.
    -   **LLM Processing**: Uses OpenRouter (GPT) to synthesize this information into quantitative impact scores.

4.  **Ensemble Prediction**:
    -   Combines the LSTM price targets with the News Sentiment impact.
    -   Applies weighting (default: 50% Technical / 50% Fundamental) to calculate the final target price.
    -   Calculates a **Confidence Score** based on the agreement between the two models.

5.  **Output**:
    -   Generates a final recommendation: **STRONG BUY**, **BUY**, **HOLD**, **SELL**, or **STRONG SELL**.
    -   Visualizes the predicted price path and key metrics.

## 📂 Project Structure

-   `flask_app.py`: Main entry point for the Flask web application.
-   `streamlit_ui.py`: Main entry point for the Streamlit dashboard.
-   `master_predictor.py`: CLI orchestration script.
-   `Models/`: Contains the core logic.
    -   `stock_prediction.py`: LSTM model implementation.
    -   `news.py`: News fetching and LLM analysis logic.
    -   `integrated_api.py`: FastAPI backend for the Streamlit app.
-   `Api/`: Internal data API (likely for raw stock data).

## ⚠️ Disclaimer
This tool is for educational purposes only. Always conduct your own research before making investment decisions.
