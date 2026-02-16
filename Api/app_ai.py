import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Stock Analyzer", page_icon="📈", layout="wide")

st.title('📈 Indian Stock Data Viewer')

# Sidebar for inputs
with st.sidebar:
    st.header("Configuration")
    symbol = st.text_input('Enter Symbol', 'RELIANCE.NS')
    interval = st.selectbox('Interval', ['1m','5m','15m','1h','1d','1wk','1mo'])
    fetch_button = st.button('🔄 Fetch Data', use_container_width=True)

if fetch_button:
    with st.spinner('Fetching data...'):
        url = f'http://localhost:8000/candles?symbol={symbol}&interval={interval}'
        try:
            response = requests.get(url)
            data = response.json()
            
            if "error" in data:
                st.error(data["error"])
            else:
                # Extract the actual data array
                df = pd.DataFrame(data['data'])
                
                import pytz
                ist = pytz.timezone("Asia/Kolkata")

                # Convert datetime safely
                if "Datetime" in df.columns:
                    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")
                    df["Datetime"] = df["Datetime"].dt.tz_convert(ist)
                    df = df.sort_values("Datetime")

                elif "Date" in df.columns:
                    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
                    df["Date"] = df["Date"].dt.tz_convert(ist)
                    df = df.sort_values("Date")


                
                # Display key metrics in columns
                st.success(f"✅ Fetched {len(df)} candles for {data['symbol']}")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Current Price", f"₹{df['Close'].iloc[-1]:.2f}")
                with col2:
                    change = df['Close'].iloc[-1] - df['Close'].iloc[0]
                    pct_change = (change / df['Close'].iloc[0]) * 100
                    st.metric("Change", f"₹{change:.2f}", f"{pct_change:.2f}%")
                with col3:
                    st.metric("High", f"₹{df['High'].max():.2f}")
                with col4:
                    st.metric("Low", f"₹{df['Low'].min():.2f}")
                with col5:
                    st.metric("Avg Volume", f"{df['Volume'].mean():,.0f}")
                
                # Tabs for different views
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Chart", "📋 Data Table", "📈 Statistics", "📉 Technical"])
                
                with tab1:
                    # Candlestick chart with volume
                    fig = make_subplots(
                        rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_heights=[0.7, 0.3],
                        subplot_titles=(f'{symbol} Price', 'Volume')
                    )
                    
                    # Candlestick
                    date_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
                    fig.add_trace(
                        go.Candlestick(
                            x=df[date_col],
                            open=df['Open'],
                            high=df['High'],
                            low=df['Low'],
                            close=df['Close'],
                            name='Price'
                        ),
                        row=1, col=1
                    )
                    
                    # Volume bars
                    colors = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i] else 'green' 
                             for i in range(len(df))]
                    fig.add_trace(
                        go.Bar(x=df[date_col], y=df['Volume'], name='Volume', marker_color=colors),
                        row=2, col=1
                    )
                    
                    fig.update_layout(
                        height=600,
                        xaxis_rangeslider_visible=False,
                        hovermode='x unified',
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    st.dataframe(
                        df.style.format({
                            'Open': '₹{:.2f}',
                            'High': '₹{:.2f}',
                            'Low': '₹{:.2f}',
                            'Close': '₹{:.2f}',
                            'Volume': '{:,.0f}'
                        }),
                        use_container_width=True,
                        height=400
                    )
                    
                    # Download button
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "📥 Download CSV",
                        csv,
                        f"{symbol}_{interval}_data.csv",
                        "text/csv"
                    )
                
                with tab3:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Price Statistics")
                        stats_df = pd.DataFrame({
                            'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Range'],
                            'Value': [
                                f"₹{df['Close'].mean():.2f}",
                                f"₹{df['Close'].median():.2f}",
                                f"₹{df['Close'].std():.2f}",
                                f"₹{df['Close'].min():.2f}",
                                f"₹{df['Close'].max():.2f}",
                                f"₹{df['Close'].max() - df['Close'].min():.2f}"
                            ]
                        })
                        st.dataframe(stats_df, hide_index=True, use_container_width=True)
                    
                    with col2:
                        st.subheader("Volume Statistics")
                        vol_stats_df = pd.DataFrame({
                            'Metric': ['Total Volume', 'Avg Volume', 'Max Volume', 'Min Volume'],
                            'Value': [
                                f"{df['Volume'].sum():,.0f}",
                                f"{df['Volume'].mean():,.0f}",
                                f"{df['Volume'].max():,.0f}",
                                f"{df['Volume'].min():,.0f}"
                            ]
                        })
                        st.dataframe(vol_stats_df, hide_index=True, use_container_width=True)
                    
                    # Distribution plot
                    st.subheader("Price Distribution")
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Histogram(x=df['Close'], nbinsx=30, name='Close Price'))
                    fig_hist.update_layout(height=300, template='plotly_white')
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with tab4:
                    # Calculate simple moving averages
                    df['SMA_20'] = df['Close'].rolling(window=min(20, len(df))).mean()
                    df['SMA_50'] = df['Close'].rolling(window=min(50, len(df))).mean()
                    
                    # Price with SMAs
                    fig_tech = go.Figure()
                    date_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
                    
                    fig_tech.add_trace(go.Scatter(x=df[date_col], y=df['Close'], 
                                                  mode='lines', name='Close', line=dict(color='blue')))
                    fig_tech.add_trace(go.Scatter(x=df[date_col], y=df['SMA_20'], 
                                                  mode='lines', name='SMA 20', line=dict(color='orange', dash='dash')))
                    fig_tech.add_trace(go.Scatter(x=df[date_col], y=df['SMA_50'], 
                                                  mode='lines', name='SMA 50', line=dict(color='red', dash='dash')))
                    
                    fig_tech.update_layout(height=400, template='plotly_white', 
                                          title='Price with Moving Averages',
                                          hovermode='x unified')
                    st.plotly_chart(fig_tech, use_container_width=True)
                    
                    # Returns
                    st.subheader("Returns Analysis")
                    df['Returns'] = df['Close'].pct_change() * 100
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Avg Return", f"{df['Returns'].mean():.4f}%")
                    with col2:
                        st.metric("Volatility (Std)", f"{df['Returns'].std():.4f}%")
                    with col3:
                        st.metric("Max Return", f"{df['Returns'].max():.4f}%")
                    
        except Exception as e:
            st.error(f"Error: {str(e)}")
else:
    st.info("👆 Configure settings in the sidebar and click 'Fetch Data' to begin") 