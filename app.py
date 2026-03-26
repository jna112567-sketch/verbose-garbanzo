import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import xml.etree.ElementTree as ET
import requests
import requests_cache

st.set_page_config(page_title="Pro Terminal", layout="wide")
st.title("🏛️ Professional Equity Terminal")

# --- 1. Sidebar Configuration & Ticker Selection ---
st.sidebar.header("Market Settings")

POPULAR_ASSETS = {
    "AAPL": "Apple Inc. (AAPL)", "MSFT": "Microsoft Corp. (MSFT)", "NVDA": "NVIDIA (NVDA)",
    "GOOGL": "Alphabet / Google (GOOGL)", "AMZN": "Amazon (AMZN)", "TSLA": "Tesla (TSLA)",
    "META": "Meta Platforms (META)", "VOO": "Vanguard S&P 500 ETF (VOO)", 
    "QQQ": "Invesco QQQ Trust (QQQ)", "SPY": "SPDR S&P 500 ETF (SPY)"
}

selected_from_dropdown = st.sidebar.multiselect(
    "Search Companies / ETFs", 
    options=list(POPULAR_ASSETS.keys()), 
    format_func=lambda x: POPULAR_ASSETS[x],
    default=["AAPL", "VOO", "MSFT"]
)

custom_ticker_input = st.sidebar.text_input("Other Tickers (comma separated)", placeholder="e.g. PLTR, UBER")
custom_tickers = [t.strip().upper() for t in custom_ticker_input.split(",") if t.strip()]

# MASTER WATCHLIST
tickers = list(set(selected_from_dropdown + custom_tickers))

st.sidebar.subheader("💼 My Portfolio Holdings")
portfolio = {}
with st.sidebar.expander("Enter Your Shares & Cost"):
    for t in tickers:
        col1, col2 = st.columns(2)
        shares = col1.number_input(f"{t} Shares", min_value=0.0, value=0.0, step=1.0, key=f"sh_{t}")
        cost = col2.number_input(f"Avg Cost ($)", min_value=0.0, value=0.0, step=1.0, key=f"cst_{t}")
        if shares > 0:
            portfolio[t] = {"shares": shares, "cost": cost}

st.sidebar.divider()

period_map = {"1W": "5d", "1M": "1mo", "YTD": "ytd", "1Y": "1y", "3Y": "3y", "5Y": "5y"}
selected_label = st.sidebar.selectbox("Timeframe", list(period_map.keys()), index=3)
period = period_map[selected_label]

graph_type = st.sidebar.radio("Graph Display Type", ["Performance (%)", "Price ($)"])

benchmark_map = {"S&P 500": "^GSPC", "Dow Jones": "^DJI", "NASDAQ": "^IXIC"}
selected_bench = st.sidebar.selectbox("Market Benchmark", list(benchmark_map.keys()))
bench_symbol = benchmark_map[selected_bench]

# --- 2. Advanced Math & Helper Functions ---
def get_max_drawdown(series):
    roll_max = series.cummax()
    drawdown = (series - roll_max) / roll_max
    return drawdown.min()

def get_sharpe(returns, risk_free_rate=0.04):
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    if ann_vol == 0: return 0
    return (ann_ret - risk_free_rate) / ann_vol

def get_sortino(returns, risk_free_rate=0.04):
    ann_ret = returns.mean() * 252
    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(252)
    if pd.isna(downside_vol) or downside_vol == 0: return 0
    return (ann_ret - risk_free_rate) / downside_vol

def get_beta(asset_ret, bench_ret):
    aligned = pd.concat([asset_ret, bench_ret], axis=1).dropna()
    if aligned.empty or len(aligned.columns) < 2: return 1.0
    cov_matrix = np.cov(aligned.iloc[:,0], aligned.iloc[:,1])
    return cov_matrix[0,1] / cov_matrix[1,1]

def get_sentiment_rss(ticker):
    try:
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
        root = ET.fromstring(resp.text)
        titles = [item.find('title').text.lower() for item in root.findall('.//item')[:12]]
        if not titles: return "Neutral ⚪"
        text_blob = " ".join(titles)
        pos_words = ['buy', 'growth', 'bull', 'upgrade', 'beat', 'positive', 'surge', 'jump', 'gain']
        neg_words = ['sell', 'risk', 'bear', 'downgrade', 'miss', 'negative', 'drop', 'fall', 'lawsuit']
        pos_score = sum(1 for w in pos_words if w in text_blob)
        neg_score = sum(1 for w in neg_words if w in text_blob)
        if pos_score > neg_score: return f"Positive 🟢 ({pos_score}:{neg_score})"
        elif neg_score > pos_score: return f"Negative 🔴 ({neg_score}:{pos_score})"
        else: return "Neutral ⚪"
    except: return "Neutral ⚪"

def get_full_news_rss(ticker):
    try:
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
        root = ET.fromstring(resp.text)
        news_items = []
        for item in root.findall('.//item')[:10]:
            news_items.append({
                'title': item.find('title').text if item.find('title') is not None else "No Title",
                'link': item.find('link').text if item.find('link') is not None else "#",
                'pubDate': item.find('pubDate').text if item.find('pubDate') is not None else "Unknown Date"
            })
        return news_items
    except:
        return []

def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_comps(ticker):
    comps_dict = {
        "AAPL": ["MSFT", "GOOGL", "DELL", "HPQ"],
        "MSFT": ["AAPL", "GOOGL", "ORCL", "IBM"],
        "NVDA": ["AMD", "INTC", "QCOM", "AVGO"],
        "GOOGL": ["META", "MSFT", "AMZN"],
        "AMZN": ["WMT", "TGT", "BABA", "EBAY"],
        "TSLA": ["F", "GM", "RIVN", "LCID"],
        "META": ["GOOGL", "SNAP", "PINS"],
        "VOO": ["SPY", "IVV", "SPLG"],
        "QQQ": ["VGT", "XLK", "IYW"],
        "SPY": ["VOO", "IVV"]
    }
    return comps_dict.get(ticker, ["(Search within same Sector/Industry)"])

FINANCIAL_TERMS = {
    "Total Revenue": "The total money brought in by operations (Top Line). \n\n*Healthy Range: Year-over-Year growth is expected.*",
    "Gross Profit": "Revenue minus direct costs to produce goods. \n\n*Healthy Range: Higher is better; implies pricing power.*",
    "Operating Expense": "Costs to run the day-to-day business (e.g., Marketing, R&D).",
    "EBITDA": "Earnings Before Interest, Taxes, Depreciation, and Amortization. \n\n*Healthy Range: Positive and growing. Wall Street's favorite cash proxy.*",
    "Net Income": "Total profit after all expenses (Bottom Line). \n\n*Healthy Range: Consistently positive.*",
    "Total Assets": "Everything the company owns of value.",
    "Total Liabilities": "Everything the company owes (Debt, payables).",
    "Stockholders Equity": "Assets minus Liabilities (Net Worth). \n\n*Healthy Range: Should ideally increase year-over-year.*",
    "Free Cash Flow": "Cash generated after supporting operations and maintaining capital assets. \n\n*Healthy Range: High positive numbers indicate safety and room for dividends.*"
}

# --- NEW: Data Caching Function ---
@st.cache_data(ttl=3600)
def fetch_ticker_data(symbols_tuple, time_period):
    """
    Fetch data allowing yf to manage its own curl_cffi session.
    Using list(symbols_tuple) as yf.download expects a list or string.
    """
    # Simply call download without passing a manual session object
    df = yf.download(
        tickers=list(symbols_tuple), 
        period=time_period, 
        progress=False,
        group_by='column'
    )
    
    # Logic to handle MultiIndex columns vs Single Ticker Series
    if len(symbols_tuple) > 1:
        return df['Close']
    return df

# --- 3. Main Application Logic ---
if tickers:
    try:
        all_symbols = list(set(tickers + [bench_symbol]))
        data = fetch_ticker_data(tuple(all_symbols), period)
        daily_returns = data.ffill().pct_change(fill_method=None).dropna()

        tab_market, tab_details, tab_portfolio, tab_financials, tab_mc, tab_news = st.tabs([
            "📈 Market Overview", "🔍 Asset Details", "💼 Portfolio", "🧾 Financials", "🎲 Monte Carlo", "📰 News Feed"
        ])

        # ==========================================
        # TAB 1: MARKET OVERVIEW
        # ==========================================
        with tab_market:
            st.subheader("Market Analysis & Risk Matrix")
            tab1_tickers = st.multiselect("Select assets to compare on this page:", tickers, default=tickers, key="tab1_ticks")
            
            if not tab1_tickers:
                st.info("Please select at least one asset to view the market comparison.")
            else:
                fig = go.Figure()
                symbols_to_plot = tab1_tickers.copy()
                if graph_type == "Performance (%)": symbols_to_plot.append(bench_symbol)

                for t in symbols_to_plot:
                    is_bench = (t == bench_symbol)
                    label = f"{selected_bench} (Ref)" if is_bench else t
                    y_val = (data[t] / data[t].iloc[0] - 1) * 100 if graph_type == "Performance (%)" else data[t]
                    line_style = dict(width=4, dash='dot') if is_bench else dict(width=2)
                    fig.add_trace(go.Scatter(x=data.index, y=y_val, name=label, line=line_style))

                fig.update_layout(hovermode="x unified", template="plotly_dark", height=450)
                st.plotly_chart(fig, width="stretch")

                st.divider()
                
                st.subheader("⚖️ Advanced Risk & Return Matrix")
                comp_data = {}
                for t in tab1_tickers:
                    ret_series = daily_returns[t]
                    bench_ret_series = daily_returns[bench_symbol]
                    
                    ret_total = ((data[t].iloc[-1] / data[t].iloc[0]) - 1) * 100
                    vol = ret_series.std() * np.sqrt(252)
                    mdd = get_max_drawdown(data[t])
                    sharpe = get_sharpe(ret_series)
                    sortino = get_sortino(ret_series)
                    beta = get_beta(ret_series, bench_ret_series)
                    var_95 = ret_series.quantile(0.05) * 100 

                    comp_data[t] = {
                        "Total Return": f"{ret_total:.2f}%",
                        "Volatility (Ann.)": f"{vol:.2%}",
                        "Max Drawdown": f"{mdd:.2%}",
                        "Beta vs Bench": f"{beta:.2f}",
                        "Sharpe Ratio": f"{sharpe:.2f}",
                        "Sortino Ratio": f"{sortino:.2f}",
                        "Daily VaR (95%)": f"{var_95:.2f}%",
                        "AI Sentiment": get_sentiment_rss(t)
                    }
                
                comp_df = pd.DataFrame(comp_data).T
                # FIX: Added astype(str) and width="stretch" to prevent PyArrow errors
                st.dataframe(comp_df.astype(str), width="stretch")
                
                csv_data = comp_df.to_csv().encode('utf-8')
                st.download_button(label="📥 Export Matrix to CSV", data=csv_data, file_name='advanced_risk_matrix.csv', mime='text/csv', key="t1_csv")

                st.subheader("🧬 Asset Correlation Matrix")
                if len(tab1_tickers) >= 2:
                    corr = daily_returns[tab1_tickers].corr()
                    fig_heat = px.imshow(corr, text_auto=".2f", color_continuous_scale='RdBu_r')
                    fig_heat.update_layout(height=400)
                    st.plotly_chart(fig_heat, width="stretch")
                else:
                    st.info("Select at least 2 assets in the filter above to generate a correlation heatmap.")

                with st.expander("📚 Market Overview Glossary & Formulas"):
                    st.markdown("""
                    * **Volatility (Ann.):** `Volatility = StdDev(Daily Returns) × √(252)`
                      * *S&P 500 average is ~15-20%. Over 30% is considered highly risky.*
                    * **Max Drawdown:** `MDD = (Trough Value - Peak Value) / Peak Value`
                      * *0% to -20% is a normal correction. Worse than -30% is a severe crash.*
                    * **Beta vs Bench:** `Beta = Covariance(Asset, Market) / Variance(Market)`
                      * *1.0 = moves exactly with market. >1.0 = more aggressive. <1.0 = defensive/stable.*
                    * **Sharpe Ratio:** `Sharpe = (Asset Return - Risk-Free Rate) / Asset Volatility`
                      * *>1.0 is Good, >2.0 is Excellent, <1.0 means the risk might not be worth the reward.*
                    * **Sortino Ratio:** `Sortino = (Asset Return - Risk-Free Rate) / Downside Volatility`
                      * *Similar to Sharpe, but only penalizes downward volatility (drops).*
                    * **Daily VaR (95%):** The 5th percentile of daily returns. On 95% of trading days, your daily loss should not exceed this percentage.
                    """)

        # ==========================================
        # TAB 2: INDIVIDUAL ASSET DETAILS
        # ==========================================
        with tab_details:
            st.subheader("Deep Dive Analytics (Valuation & Research)")
            
            tab2_tickers = st.multiselect("Select assets to view details for:", tickers, default=tickers, key="tab2_ticks")
            
            with st.expander("🧠 How are the AI & Technical Signals Calculated?"):
                st.markdown("""
                * **Short-Term Signal (RSI):** `RSI = 100 - [100 / (1 + (Average Gain / Average Loss))]`
                    * 🟢 **BUY:** RSI is below 45 (cooling off) AND the current price is within 2% of the 50-day average.
                    * 🔴 **SELL:** RSI is over 65 (asset is 'overbought' and due for a pullback).
                    * 🟡 **HOLD:** Asset is trading in a normal middle-ground range.
                * **Long-Term Trend (SMA):** `SMA = (Sum of Prices over N periods) / N`
                    * 🟢 **BULLISH:** The 50-Day Moving Average is higher than the 200-Day Moving Average (Golden Cross).
                    * 🔴 **BEARISH:** The 50-Day Moving Average is below the 200-Day Moving Average (Death Cross).
                * **AI Sentiment:** Scrapes the Yahoo Finance RSS feed for the asset's last 12 news headlines. It counts "Bullish" words versus "Bearish" words. The ratio is displayed as `(Positive:Negative)`.
                """)

            if not tab2_tickers:
                st.info("Please select at least one asset from the dropdown above to view its details.")

            for t in tab2_tickers:
                with st.container(border=True):
                    t_obj = yf.Ticker(t)
                    info = t_obj.info
                    quote_type = info.get('quoteType', 'EQUITY')
                    
                    st.markdown(f"### {info.get('shortName', t)} ({t})")
                    st.caption(f"Asset Class: {quote_type} | Industry: {info.get('industry', info.get('category', 'N/A'))}")
                    
                    rsi_series = calculate_rsi(data[t])
                    current_rsi = rsi_series.iloc[-1]
                    price_vs_sma50 = data[t].iloc[-1] / data[t].rolling(50).mean().iloc[-1]
                    
                    st_signal = "🟢 BUY" if current_rsi < 45 and price_vs_sma50 > 0.98 else "🔴 SELL" if current_rsi > 65 else "🟡 HOLD"
                    lt_signal = "🟢 BULLISH" if data[t].rolling(50).mean().iloc[-1] > data[t].rolling(200).mean().iloc[-1] else "🔴 BEARISH"

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Current Price", f"${data[t].iloc[-1]:.2f}")
                    c2.metric("Short-Term Signal (RSI)", f"{st_signal} ({current_rsi:.1f})")
                    c3.metric("Long-Term Trend (SMA)", lt_signal)
                    c4.metric("AI Sentiment", get_sentiment_rss(t))

                    st.divider()

                    if quote_type == 'ETF':
                        st.markdown("**Fund Characteristics & Performance**")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            # FIX: Added astype(str) and width="stretch"
                            st.dataframe(pd.Series({
                                "YTD Return": f"{info.get('ytdReturn', 0)*100:.2f}%" if info.get('ytdReturn') else "N/A",
                                "3-Year Avg Return": f"{info.get('threeYearAverageReturn', 0)*100:.2f}%" if info.get('threeYearAverageReturn') else "N/A",
                                "5-Year Avg Return": f"{info.get('fiveYearAverageReturn', 0)*100:.2f}%" if info.get('fiveYearAverageReturn') else "N/A"
                            }, name="Performance").astype(str), width="stretch")
                        with col2:
                            st.dataframe(pd.Series({
                                "Fund Family": info.get('fundFamily', 'N/A'),
                                "Dividend Yield": f"{info.get('yield', 0)*100:.2f}%" if info.get('yield') else "N/A",
                                "Beta (3Y)": info.get('beta3Year', 'N/A')
                            }, name="Metrics").astype(str), width="stretch")
                        with col3:
                            st.dataframe(pd.Series({
                                "Total Assets": f"${info.get('totalAssets', 0):,}" if info.get('totalAssets') else "N/A",
                                "NAV Price": f"${info.get('navPrice', 0):.2f}" if info.get('navPrice') else "N/A",
                                "52-Week Range": f"${info.get('fiftyTwoWeekLow', 0)} - ${info.get('fiftyTwoWeekHigh', 0)}" if info.get('fiftyTwoWeekLow') else "N/A"
                            }, name="Size & Profile").astype(str), width="stretch")
                    else:
                        st.markdown("**Analyst Valuation & Financial Health**")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.dataframe(pd.Series({
                                "Market Cap": f"${info.get('marketCap', 0):,}" if info.get('marketCap') else "N/A",
                                "Trailing P/E": info.get('trailingPE', 'N/A'),
                                "Forward P/E": info.get('forwardPE', 'N/A'),
                                "EV / EBITDA": info.get('enterpriseToEbitda', 'N/A'),
                                "Price to Book (P/B)": info.get('priceToBook', 'N/A'),
                                "Price to Sales (P/S)": info.get('priceToSalesTrailing12Months', 'N/A')
                            }, name="Valuation Multiples").astype(str), width="stretch")
                        with col2:
                            st.dataframe(pd.Series({
                                "Return on Equity (ROE)": f"{info.get('returnOnEquity', 0)*100:.2f}%" if info.get('returnOnEquity') else "N/A",
                                "Return on Assets (ROA)": f"{info.get('returnOnAssets', 0)*100:.2f}%" if info.get('returnOnAssets') else "N/A",
                                "Gross Margin": f"{info.get('grossMargins', 0)*100:.2f}%" if info.get('grossMargins') else "N/A",
                                "Operating Margin": f"{info.get('operatingMargins', 0)*100:.2f}%" if info.get('operatingMargins') else "N/A"
                            }, name="Profitability").astype(str), width="stretch")
                        with col3:
                            st.dataframe(pd.Series({
                                "Current Ratio": info.get('currentRatio', 'N/A'),
                                "Debt to Equity": info.get('debtToEquity', 'N/A'),
                                "Free Cash Flow": f"${info.get('freeCashflow', 0):,}" if info.get('freeCashflow') else "N/A",
                                "Earnings Growth": f"{info.get('earningsGrowth', 0)*100:.2f}%" if info.get('earningsGrowth') else "N/A",
                                "Revenue Growth": f"{info.get('revenueGrowth', 0)*100:.2f}%" if info.get('revenueGrowth') else "N/A"
                            }, name="Health & Growth").astype(str), width="stretch")

                        st.divider()
                        
                        st.markdown("**Ownership & Market Position**")
                        col_own, col_peer = st.columns(2)
                        with col_own:
                            st.write("**Top Institutional Holders**")
                            try:
                                holders = t_obj.institutional_holders
                                if holders is not None and not holders.empty:
                                    holders_clean = holders[['Holder', 'Shares', 'Value']].head(5)
                                    holders_clean['Shares'] = holders_clean['Shares'].apply(lambda x: f"{x:,.0f}")
                                    holders_clean['Value'] = holders_clean['Value'].apply(lambda x: f"${x:,.0f}")
                                    st.dataframe(holders_clean.astype(str), hide_index=True, width="stretch")
                                else:
                                    st.write("Holder data not available.")
                            except:
                                st.write("Holder data not available.")
                        
                        with col_peer:
                            st.write("**Comparable Companies (Peers)**")
                            st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                            st.write(f"**Industry:** {info.get('industry', 'N/A')}")
                            
                            comps = get_comps(t)
                            st.write(f"**Example Peers:** {', '.join(comps)}")
                            
                            st.caption("🔍 *Analyst Tip:* Compare P/E and EV/EBITDA against these peers to see if the stock is relatively over or undervalued.")

            with st.expander("📚 Asset Details Glossary & Formulas"):
                st.markdown("""
                ### Equity Valuation Multiples
                * **P/E (Price-to-Earnings):** `P/E = Share Price / Earnings Per Share (EPS)`
                  * *S&P 500 average is ~15-25x. Lower implies cheap/value, higher implies high-growth expectations.*
                * **EV / EBITDA:** `EV/EBITDA = (Market Cap + Total Debt - Cash) / EBITDA`
                  * *Under 10x is generally considered healthy/undervalued.*
                * **P/B (Price-to-Book):** `P/B = Share Price / Book Value Per Share`
                  * *< 1.0 means it trades for less than its assets are worth; 1.0 - 3.0 is typical.*
                
                ### Profitability & Health
                * **ROE (Return on Equity):** `ROE = Net Income / Shareholders' Equity`
                  * *> 15% is generally considered strong.*
                * **ROA (Return on Assets):** `ROA = Net Income / Total Assets`
                  * *> 5% is generally good, highly industry-dependent.*
                * **Current Ratio:** `Current Ratio = Current Assets / Current Liabilities`
                  * *1.5 to 3.0 is very healthy. < 1.0 means potential liquidity trouble.*
                * **Debt to Equity:** `D/E = Total Liabilities / Shareholders' Equity`
                  * *< 1.0 is conservative. > 2.0 is highly leveraged.*
                
                ### ETF Specifics
                * **Fund Family:** The financial institution that manages the ETF (e.g., BlackRock, Vanguard).
                * **NAV (Net Asset Value):** `NAV = (Total Assets - Total Liabilities) / Total Shares Outstanding`
                  * *Should closely match the ETF's current trading price.*
                * **52-Week Range:** Shows volatility and current momentum by displaying the absolute lowest and highest price over the past year.
                """)

        # ==========================================
        # TAB 3: PORTFOLIO TRACKER
        # ==========================================
        with tab_portfolio:
            st.subheader("Live Portfolio Performance")
            
            if not portfolio:
                st.info("👈 Enter your shares and average cost in the sidebar to unlock Portfolio Tracking.")
            else:
                owned_tickers = list(portfolio.keys())
                tab3_tickers = st.multiselect("Filter your portfolio view:", owned_tickers, default=owned_tickers, key="tab3_ticks")
                
                if not tab3_tickers:
                    st.warning("All portfolio assets have been filtered out. Select an asset to view.")
                else:
                    port_data = []
                    total_cost = total_value = 0
                    
                    for t in tab3_tickers:
                        p_info = portfolio[t]
                        current_price = data[t].iloc[-1]
                        shares = p_info['shares']
                        cost_basis = p_info['cost']
                        pos_cost = shares * cost_basis
                        pos_value = shares * current_price
                        pnl_dollars = pos_value - pos_cost
                        pnl_pct = (pnl_dollars / pos_cost) * 100 if pos_cost > 0 else 0
                        total_cost += pos_cost
                        total_value += pos_value
                        
                        port_data.append({
                            "Asset": t, "Shares": shares, "Avg Cost": f"${cost_basis:.2f}",
                            "Current Price": f"${current_price:.2f}", "Total Value": f"${pos_value:.2f}",
                            "P&L ($)": f"${pnl_dollars:.2f}", "P&L (%)": f"{pnl_pct:.2f}%"
                        })
                    
                    total_pnl = total_value - total_cost
                    total_pnl_pct = (total_pnl / total_cost) * 100 if total_cost > 0 else 0
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Selected Portfolio Value", f"${total_value:,.2f}")
                    c2.metric("Selected Profit/Loss ($)", f"${total_pnl:,.2f}")
                    c3.metric("Selected Profit/Loss (%)", f"{total_pnl_pct:.2f}%")
                    
                    st.divider()
                    col_tab, col_pie = st.columns([2, 1])
                    with col_tab:
                        st.dataframe(pd.DataFrame(port_data).astype(str), width="stretch")
                    with col_pie:
                        pie_df = pd.DataFrame([{"Asset": d["Asset"], "Value": float(d["Total Value"].replace('$','').replace(',',''))} for d in port_data])
                        fig_pie = px.pie(pie_df, values='Value', names='Asset', hole=0.4, template="plotly_dark")
                        fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0))
                        st.plotly_chart(fig_pie, width="stretch")

        # ==========================================
        # TAB 4: RAW FINANCIAL STATEMENTS
        # ==========================================
        with tab_financials:
            st.subheader("Raw Accounting Data & Financials")
            
            fin_col1, fin_col2 = st.columns([2, 1])
            with fin_col1:
                fin_ticker = st.selectbox("Select Asset to View Financials:", tickers, key="tab4_ticks")
                fin_obj = yf.Ticker(fin_ticker)
                
                if fin_obj.info.get('quoteType') == 'ETF':
                    st.warning(f"{fin_ticker} is an ETF. ETFs do not have traditional corporate financial statements.")
                else:
                    statement_type = st.radio("Select Statement", ["Income Statement", "Balance Sheet", "Cash Flow"], horizontal=True)
            
            with fin_col2:
                st.write("**📚 Term Lookup**")
                lookup_term = st.selectbox("Select a term to understand it:", ["--- Select a Term ---"] + list(FINANCIAL_TERMS.keys()))
                if lookup_term != "--- Select a Term ---":
                    st.info(FINANCIAL_TERMS[lookup_term])

            if fin_obj.info.get('quoteType') != 'ETF':
                df_fin = fin_obj.financials if statement_type == "Income Statement" else fin_obj.balance_sheet if statement_type == "Balance Sheet" else fin_obj.cashflow
                
                if not df_fin.empty:
                    df_fin = df_fin.dropna(how='all')
                    for col in df_fin.columns:
                        df_fin[col] = df_fin[col].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) and isinstance(x, (int, float)) else "-")
                    st.dataframe(df_fin.astype(str), width="stretch")
                else:
                    st.error("Data not available.")

        # ==========================================
        # TAB 5: MONTE CARLO SIMULATION
        # ==========================================
        with tab_mc:
            st.subheader("🎲 Monte Carlo Price Forecasting")
            mc_ticker = st.selectbox("Select Asset to Simulate:", tickers, key="mc_tick")
            days_to_sim = st.slider("Days to Simulate (Future)", 30, 252, 252, 30)
            
            if st.button(f"Run Simulation for {mc_ticker}"):
                with st.spinner("Calculating 100 alternate futures..."):
                    hist_data = data[mc_ticker].dropna()
                    log_returns = np.log(1 + hist_data.pct_change()).dropna()
                    
                    u = log_returns.mean()
                    var = log_returns.var()
                    drift = u - (0.5 * var)
                    stdev = log_returns.std()
                    
                    simulations = 100
                    Z = np.random.standard_normal((days_to_sim, simulations))
                    daily_sim_returns = np.exp(drift + stdev * Z)
                    
                    price_paths = np.zeros_like(daily_sim_returns)
                    price_paths[0] = hist_data.iloc[-1]
                    for t in range(1, days_to_sim):
                        price_paths[t] = price_paths[t-1] * daily_sim_returns[t]
                    
                    fig_mc = go.Figure()
                    for i in range(simulations):
                        fig_mc.add_trace(go.Scatter(y=price_paths[:, i], mode='lines', line=dict(width=1, color='rgba(0, 150, 255, 0.1)'), showlegend=False))
                    
                    fig_mc.update_layout(title=f"{mc_ticker} - 100 Paths", yaxis_title="Price ($)", template="plotly_dark", height=500)
                    st.plotly_chart(fig_mc, width="stretch")
                    
                    final_prices = price_paths[-1]
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Avg Expected Price", f"${np.mean(final_prices):.2f}")
                    c2.metric("Worst Case (5th %)", f"${np.percentile(final_prices, 5):.2f}")
                    c3.metric("Best Case (95th %)", f"${np.percentile(final_prices, 95):.2f}")

            with st.expander("📚 Monte Carlo Glossary & Formulas"):
                st.markdown("""
                * **Monte Carlo Simulation:** A statistical technique that generates hundreds of random variations of the future to calculate probability and risk.
                * **Drift:** `Drift = Average Daily Return - (Variance / 2)`
                  * *The stock's historical average direction, adjusted for volatility drag.*
                * **Randomness (Simulated Return):** `Next Price = Current Price × e^(Drift + Volatility × Random Value)`
                  * *The system injects random volatility matching the stock's historical wildness. High volatility stocks will show a wider spread of lines.*
                * **Worst Case (5th Percentile):** Out of 100 possible futures simulated, 95 of them ended up *better* than this price. It serves as a strong "worst-case scenario" warning.
                * **Best Case (95th Percentile):** Only 5% of the simulated futures resulted in a price higher than this.
                """)

        # ==========================================
        # TAB 6: LIVE NEWS FEED
        # ==========================================
        with tab_news:
            st.subheader("📰 Live Market News")
            news_ticker = st.selectbox("Select Asset for News Feed:", tickers, key="news_tick")
            
            st.markdown(f"**Latest Verified Headlines for {news_ticker}**")
            news_items = get_full_news_rss(news_ticker)
            
            if news_items:
                for item in news_items:
                    with st.container(border=True):
                        st.markdown(f"#### [{item['title']}]({item['link']})")
                        st.caption(f"📅 Published: {item['pubDate']}")
            else:
                st.info(f"No recent news found for {news_ticker}.")

    except Exception as e:
        st.error(f"Waiting for valid data... Error: {e}")
else:
    st.info("Select or enter tickers in the sidebar.")

    # --- SIDEBAR HEALTH CHECK ---
with st.sidebar:
    st.divider()
    try:
        # A lightweight test call to check connectivity
        test_data = yf.Ticker("AAPL").fast_info['lastPrice']
        st.sidebar.markdown("● **API Status:** <span style='color:green'>Online</span>", unsafe_allow_html=True)
    except Exception:
        st.sidebar.markdown("● **API Status:** <span style='color:red'>Rate Limited</span>", unsafe_allow_html=True)
        st.sidebar.warning("Yahoo is throttling this IP. Please wait 15-30 mins.")