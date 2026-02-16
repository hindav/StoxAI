from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd

app = FastAPI()

# Allow all origins for frontend usage
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/candles")
def candles(symbol: str = "RELIANCE.NS", interval: str = "1m"):

    # Yahoo limit for 1-minute data
    if interval == "1m":
        period = "5d"
    else:
        period = "1y"

    # Download data
    data = yf.download(symbol, period=period, interval=interval)

    # Handle empty response
    if data.empty:
        return {"error": "No data returned from Yahoo Finance"}

    # **FIX: Flatten multi-index columns if present**
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Reset index so Datetime appears as a column
    data = data.reset_index()

    # Convert Datetime or Date column to string
    if "Datetime" in data.columns:
        data["Datetime"] = data["Datetime"].astype(str)
    if "Date" in data.columns:
        data["Date"] = data["Date"].astype(str)

    # Convert all numpy types → pure Python
    data = data.astype(object)
    for col in data.columns:
        data[col] = data[col].apply(
            lambda x: x.item() if hasattr(x, "item") else x
        )

    # Convert to JSON-safe list of objects
    candles_json = data.to_dict(orient="records")

    return {
        "symbol": symbol,
        "interval": interval,
        "length": len(candles_json),
        "data": candles_json
    }


@app.get("/")
def home():
    return {"message": "Stock API is running!"}