import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from risk_investExpect import MomentumStrategy, load_portfolio
import os

st.set_page_config(page_title="Dual Momentum Dashboard", layout="wide", page_icon="📈")

st.title("📈 Dual Momentum Investment Strategy")
st.markdown("""
This dashboard implements the **Dual Momentum Strategy**. 
It combines **Relative Momentum** (buying top performers) and **Absolute Momentum** (moving to cash during market downturns).
""")

st.sidebar.header("Strategy Parameters")
top_n = st.sidebar.slider("Top N Stocks", 1, 10, 5)
momentum_window = st.sidebar.slider("Momentum Window (Months)", 1, 12, 6)
lookback_years = st.sidebar.slider("Backtest Years", 1, 20, 10)

strategy = MomentumStrategy(
    top_n=top_n, 
    momentum_window=momentum_window, 
    lookback_years=lookback_years
)

tab1, tab2 = st.tabs(["🚀 Today's Recommendation", "📊 Backtest Analysis"])

with tab1:
    st.header("Today's Investment Recommendation")
    
    if st.button("Generate Recommendation"):
        # Capital Input moved outside button to persist state
        pass
    
    # Capital Input
    capital = st.number_input("Total Capital to Invest ($)", min_value=1000, value=10000, step=1000)
    
    if st.button("Analyze & Recommend"):
        with st.spinner("Analyzing market and calculating momentum..."):
            portfolio_file = "my_portfolio.csv"
            my_stocks = load_portfolio(portfolio_file)
            
            rec = strategy.recommend_portfolio(my_stocks, investment_capital=capital)
            
            st.subheader("Market Analysis (SPY Trend)")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("SPY Price", f"${rec['SPY Price']:.2f}")
            col2.metric("SPY SMA200", f"${rec['SPY SMA200']:.2f}")
            col3.metric("Date", str(rec["Date"]))
            
            spy_data = rec["SPY Data"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=spy_data.index, y=spy_data['Close'], name='SPY Price'))
            fig.add_trace(go.Scatter(x=spy_data.index, y=spy_data['SMA200'], name='SMA200', line=dict(dash='dash')))
            fig.update_layout(title="SPY Trend Analysis", xaxis_title="Date", yaxis_title="Price")
            # use_container_width=True is deprecated in recent versions
            st.plotly_chart(fig, use_container_width=True) 
            
            st.success("✅ **Always-Invest Strategy:** Top momentum stocks recommended below.")
            
            top_df = pd.DataFrame(rec["Top Stocks"])
            # Update st.dataframe to avoid deprecation warning if possible, but use_container_width is standard for now.
            # The warning suggests width='stretch' for st.dataframe
            st.dataframe(
                top_df.style.format({
                    "Momentum": "{:.2f}%", 
                    "Price": "${:.2f}",
                    "Cost": "${:,.2f}"
                }),
                use_container_width=True 
            )
            
            st.info(f"💰 **Allocation Summary:** Total Cost: **${rec['Total Cost']:,.2f}** | Leftover Cash: **${rec['Leftover Cash']:,.2f}**")
            
            if "Analysis" in rec:
                st.subheader("Portfolio Check")
                analysis = rec["Analysis"]
                if analysis["Keep"]:
                    st.info(f"🟢 **Keep (Still in top ranking):** {', '.join(analysis['Keep'])}")
                if analysis["Sell"]:
                    st.error(f"🔴 **Sell (Dropped from ranking):** {', '.join(analysis['Sell'])}")
                if analysis["Buy"]:
                    st.success(f"🔵 **New Buy Recommendation:** {', '.join(analysis['Buy'])}")
                
                if not analysis["Sell"] and not analysis["Buy"]:
                    st.balloons()
                    st.success("✨ Your portfolio is already optimal!")

with tab2:
    st.header("Backtest Performance Analysis")
    interval = st.selectbox("Rebalancing Interval", ["1mo", "2wk", "1wk"], index=0)
    
    if st.button("Run Simulation"):
        with st.spinner("Running historical simulation... This may take a minute."):
            results, summary = strategy.run_simulation(interval)
            
            if results is not None:
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("CAGR", f"{summary['CAGR']:.2f}%")
                m2.metric("Max Drawdown", f"{summary['MDD']:.2f}%")
                m3.metric("Avg Drawdown", f"{summary['Avg Drawdown']:.2f}%")
                m4.metric("Total Return", f"{summary['Total Return']:.2f}%")
                m5.metric("Final Value", f"${summary['Final Value']:,.0f}")
                
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=results['Date'], y=results['Value'], name='Portfolio Value'))
                fig2.update_layout(title="Equity Curve ($10,000 Initial)", xaxis_title="Date", yaxis_title="Portfolio Value ($)")
                st.plotly_chart(fig2, use_container_width=True)
                
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=results['Date'], y=results['Drawdown'], name='Drawdown', fill='tozeroy', line=dict(color='red')))
                fig3.update_layout(title="Historical Drawdown (%)", xaxis_title="Date", yaxis_title="Drawdown (%)")
                st.plotly_chart(fig3, use_container_width=True)
                
                st.subheader("Simulation History")
                st.dataframe(
                    results[['Date', 'Value', 'Return', 'Holdings']].style.format({"Value": "${:,.0f}", "Return": "{:+.2f}%"}),
                    use_container_width=True
                )
            else:
                st.error("Failed to run simulation. Please check your data connection.")

st.markdown("---")
st.caption("Data provided by yfinance. Past performance does not guarantee future results.")
