import streamlit as st
import requests
import pandas as pd
import json

st.set_page_config(page_title="Stock Viewer", layout="wide")
st.title("Stock Viewer")

# Sidebar inputs
with st.sidebar:
    symbol = st.text_input("Symbol", "RELIANCE.NS")
    interval = st.selectbox("Interval", ["1m"])
    fetch = st.button("Fetch Data")

if not fetch:
    st.info("Enter symbol & click Fetch")
    st.stop()

# Fetch API
url = f"http://localhost:8000/candles?symbol={symbol}&interval={interval}"
res = requests.get(url).text

# Convert broken JSON
res = res.replace('""', '"')  # Fix malformed JSON
data = json.loads(res)

# Create DataFrame
df = pd.DataFrame(data["data"])

# Expand JSON-string columns
for col in df.columns:
    # Detect malformed JSON strings
    if df[col].astype(str).str.startswith("{").any():
        df[col] = df[col].astype(str).apply(lambda x: json.loads(x.replace('""', '"')))
        expanded = pd.json_normalize(df[col])
        df = pd.concat([df.drop(columns=[col]), expanded], axis=1)

# Convert date column
date_col = None
for c in ["Datetime", "Date", "timestamp"]:
    if c in df.columns:
        date_col = c
        df[c] = pd.to_datetime(df[c])
        break

# Sort by date if exists
if date_col:
    df = df.sort_values(date_col)

# Display table
st.subheader(f"Data for {symbol} ({interval})")
st.dataframe(df, height=600, use_container_width=True)

# Download button
csv = df.to_csv(index=False)
st.download_button("Download CSV", csv, f"{symbol}_{interval}.csv", "text/csv") 
