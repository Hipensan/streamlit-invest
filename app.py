import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from risk_investExpect import MomentumStrategy

st.set_page_config(page_title="Dual Momentum Dashboard", layout="wide", page_icon="📈")

st.title("📈 Dual Momentum Investment Strategy")
st.markdown("""
This dashboard implements the **Dual Momentum Strategy**. 
It focuses on **Relative Momentum** (buying top performers) and stays invested.
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
    
    # Capital Input
    capital = st.number_input("Total Capital to Invest ($)", min_value=1000, value=10000, step=1000)
    chart_window = st.selectbox("Chart Window", ["3M", "6M", "1Y"], index=2)
    
    if st.button("Analyze & Recommend"):
        with st.spinner("Analyzing market and calculating momentum..."):
            rec = strategy.recommend_portfolio(None, investment_capital=capital)

            st.subheader("Market Analysis (SPY Trend)")
            
            col1, col2 = st.columns(2)
            col1.metric("SPY Price", f"${rec['SPY Price']:.2f}")
            col2.metric("Date", str(rec["Date"]))
            
            spy_data = rec["SPY Data"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=spy_data.index, y=spy_data['Close'], name='SPY Price'))
            fig.update_layout(title="SPY Price", xaxis_title="Date", yaxis_title="Price")
            # use_container_width=True is deprecated in recent versions
            st.plotly_chart(fig, use_container_width=True) 
            
            st.success("✅ **Always-Invest Strategy:** Top momentum stocks recommended below.")
            
            top_df = pd.DataFrame(rec["Top Stocks"])
            display_cols = [
                "Ticker",
                "Momentum",
                "Return 3M",
                "Return 6M",
                "Return 12M",
                "Volatility (63d)",
                "Price",
                "Shares to Buy",
                "Cost"
            ]
            display_cols = [c for c in display_cols if c in top_df.columns]
            # Update st.dataframe to avoid deprecation warning if possible, but use_container_width is standard for now.
            # The warning suggests width='stretch' for st.dataframe
            st.dataframe(
                top_df[display_cols].style.format({
                    "Momentum": "{:.2f}%", 
                    "Return 3M": "{:.2f}%",
                    "Return 6M": "{:.2f}%",
                    "Return 12M": "{:.2f}%",
                    "Volatility (63d)": "{:.2f}%",
                    "Price": "${:.2f}",
                    "Cost": "${:,.2f}"
                }),
                use_container_width=True 
            )

            if not top_df.empty:
                st.subheader("Stock Details")
                cols = st.columns(3)
                for idx, row in top_df.iterrows():
                    col = cols[idx % 3]
                    with col:
                        st.markdown(f"**{row['Ticker']}**")
                        st.metric("Momentum", f"{row['Momentum']:.2f}%")
                        r6 = row.get("Return 6M")
                        st.metric("Return 6M", "N/A" if pd.isna(r6) else f"{r6:.2f}%")
                        r3 = row.get("Return 3M")
                        r12 = row.get("Return 12M")
                        vol = row.get("Volatility (63d)")
                        caption = (
                            f"Return 3M: {'N/A' if pd.isna(r3) else f'{r3:.2f}%'} | "
                            f"Return 12M: {'N/A' if pd.isna(r12) else f'{r12:.2f}%'} | "
                            f"Volatility (63d): {'N/A' if pd.isna(vol) else f'{vol:.2f}%'}"
                        )
                        st.caption(caption)
            
            st.info(f"💰 **Allocation Summary:** Total Cost: **${rec['Total Cost']:,.2f}** | Leftover Cash: **${rec['Leftover Cash']:,.2f}**")

            if "Alternates" in rec and rec["Alternates"]:
                st.subheader("Alternate Candidates")
                alt_df = pd.DataFrame(rec["Alternates"])
                alt_display = [
                    "Ticker",
                    "Momentum",
                    "Return 3M",
                    "Return 6M",
                    "Return 12M",
                    "Volatility (63d)",
                    "Price"
                ]
                alt_display = [c for c in alt_display if c in alt_df.columns]
                st.dataframe(
                    alt_df[alt_display].style.format({
                        "Momentum": "{:.2f}%",
                        "Return 3M": "{:.2f}%",
                        "Return 6M": "{:.2f}%",
                        "Return 12M": "{:.2f}%",
                        "Volatility (63d)": "{:.2f}%",
                        "Price": "${:.2f}"
                    }),
                    use_container_width=True
                )
            
            # Portfolio comparison removed (no CSV portfolio tracking)

            if "Top Prices" in rec and not rec["Top Prices"].empty:
                st.subheader("Momentum Score & Price Trends")
                momentum_chart = go.Figure()
                momentum_chart.add_trace(
                    go.Bar(
                        x=top_df["Ticker"],
                        y=top_df["Momentum"],
                        name="Momentum (%)"
                    )
                )
                momentum_chart.update_layout(
                    title="Momentum Scores",
                    xaxis_title="Ticker",
                    yaxis_title="Momentum (%)"
                )
                st.plotly_chart(momentum_chart, use_container_width=True)

                lookback_map = {"3M": 63, "6M": 126, "1Y": 252}
                lookback_days = lookback_map.get(chart_window, 252)
                price_df = rec["Top Prices"].copy().tail(lookback_days)
                normalized = price_df / price_df.iloc[0]
                price_chart = go.Figure()
                for col in normalized.columns:
                    price_chart.add_trace(
                        go.Scatter(x=normalized.index, y=normalized[col], name=col)
                    )
                price_chart.update_layout(
                    title="Normalized Price Trends (1.0 = start)",
                    xaxis_title="Date",
                    yaxis_title="Normalized Price"
                )
                st.plotly_chart(price_chart, use_container_width=True)

with tab2:

    st.header("Backtest Performance Analysis")
    strategy_name = st.selectbox(
        "Strategy",
        ["Momentum", "Blended Momentum (3/6/12)", "Volatility-Adjusted Momentum", "Buy & Hold SPY"],
        index=0
    )
    interval = st.selectbox("Rebalancing Interval", ["1mo", "2wk", "1wk"], index=0)
    
    if st.button("Run Simulation"):
        with st.spinner("Running historical simulation... This may take a minute."):
            results, summary = strategy.run_simulation(interval, strategy_name=strategy_name)
            
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

    if st.button("Compare Intervals"):
        with st.spinner("Running interval comparison... This may take a minute."):
            rows = []
            for iv in ["1mo", "2wk", "1wk"]:
                results, summary = strategy.run_simulation(iv, strategy_name=strategy_name)
                if results is None:
                    continue
                rows.append({
                    "Interval": iv,
                    "CAGR": summary["CAGR"],
                    "MDD": summary["MDD"],
                    "Avg Drawdown": summary["Avg Drawdown"],
                    "Total Return": summary["Total Return"],
                    "Final Value": summary["Final Value"]
                })
            if rows:
                compare_df = pd.DataFrame(rows)
                st.subheader("Interval Comparison")
                st.dataframe(
                    compare_df.style.format({
                        "CAGR": "{:.2f}%",
                        "MDD": "{:.2f}%",
                        "Avg Drawdown": "{:.2f}%",
                        "Total Return": "{:.2f}%",
                        "Final Value": "${:,.0f}"
                    }),
                    use_container_width=True
                )
            else:
                st.error("Failed to run interval comparison. Please check your data connection.")

st.markdown("---")
st.caption("Data provided by yfinance. Past performance does not guarantee future results.")
