import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Pro Investment Dashboard", layout="wide")
st.title("📊 Multi-Equity Analysis Dashboard")

# --- 1. Sidebar Settings ---
st.sidebar.header("Configuration")

# Period Mapping
period_map = {
    "1 Week": "5d",
    "1 Month": "1mo",
    "YTD": "ytd",
    "1 Year": "1y",
    "3 Years": "3y",
    "5 Years": "5y"
}
selected_label = st.sidebar.selectbox("Select Timeframe", list(period_map.keys()), index=3)
period = period_map[selected_label]

# Ticker Selection (Stocks & ETFs work the same in yfinance)
default_tickers = ["AAPL", "TSLA", "VOO", "QQQ", "MSFT"]
tickers = st.sidebar.multiselect("Select up to 5 Equities", 
                                 options=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "VOO", "SPY", "QQQ", "VTI", "SCHD"],
                                 default=default_tickers[:3])

# Custom Ticker Input
custom_ticker = st.sidebar.text_input("Add a custom ticker (e.g. NVDA)").upper()
if custom_ticker and custom_ticker not in tickers:
    tickers.append(custom_ticker)

if len(tickers) > 5:
    st.warning("Please select only up to 5 tickers for the best view.")
    tickers = tickers[:5]

# --- 2. Main Dashboard Logic ---
if tickers:
    # Fetch Dow Jones for the Index Graph
    dow_data = yf.download("^DJI", period=period)['Close']
    
    tabs = st.tabs(["Performance Comparison", "Individual Deep Dive", "Market Context"])

    with tabs[0]:
        st.subheader(f"Relative Growth ({selected_label})")
        # Download all selected tickers
        all_data = yf.download(tickers, period=period)['Close']
        
        # Normalize to 100 for percentage growth comparison
        normalized_data = (all_data / all_data.iloc[0]) * 100
        st.line_chart(normalized_data)
        st.caption("All assets normalized to 100 at start of period.")

    with tabs[1]:
        st.subheader("Financial Highlights")
        for t in tickers:
            with st.expander(f"Detailed Stats: {t}"):
                stock = yf.Ticker(t)
                info = stock.info
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Price", f"${info.get('currentPrice', 'N/A')}")
                c2.metric("Forward P/E", f"{info.get('forwardPE', 'N/A')}")
                c3.metric("Profit Margin", f"{info.get('profitMargins', 0)*100:.2f}%")
                c4.metric("ROE", f"{info.get('returnOnEquity', 0)*100:.2f}%")
                
                # Extended Table
                extra_stats = {
                    "Market Cap": f"{info.get('marketCap', 0):,}",
                    "52 Week High": info.get('fiftyTwoWeekHigh'),
                    "52 Week Low": info.get('fiftyTwoWeekLow'),
                    "Dividend Yield": f"{info.get('dividendYield', 0)*100:.2f}%",
                    "Total Revenue": info.get('totalRevenue')
                }
                st.write(pd.DataFrame([extra_stats], index=["Value"]).T)

    with tabs[2]:
        st.subheader("Dow Jones Industrial Average (^DJI)")
        st.line_chart(dow_data)
        
        # Calculate Dow Change
        dow_perf = ((dow_data.iloc[-1] / dow_data.iloc[0]) - 1) * 100
        st.metric("Dow Jones Period Perf", f"{dow_perf:.2f}%")

else:
    st.info("Please select at least one ticker in the sidebar.")