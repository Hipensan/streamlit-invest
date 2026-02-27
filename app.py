import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from risk_investExpect import MomentumStrategy

st.set_page_config(page_title="듀얼 모멘텀 대시보드", layout="wide", page_icon="📈")

st.title("듀얼 모멘텀 투자 전략")
st.markdown(
    """
이 대시보드는 **듀얼 모멘텀 전략**을 구현합니다.
**상대 모멘텀**(성과가 좋은 종목 매수)을 중심으로 항상 투자 상태를 유지합니다.
"""
)

st.sidebar.header("전략 파라미터")
top_n = st.sidebar.slider("상위 종목 수 (N)", 1, 10, 5)
momentum_window = st.sidebar.slider("모멘텀 기간 (개월)", 1, 12, 6)
lookback_years = st.sidebar.slider("백테스트 기간 (년)", 1, 20, 10)

strategy = MomentumStrategy(
    top_n=top_n,
    momentum_window=momentum_window,
    lookback_years=lookback_years,
)

tab1, tab2, tab3 = st.tabs(["오늘의 추천", "백테스트 분석", "자산 계산기"])

with tab1:
    st.header("오늘의 투자 추천")

    capital = st.number_input("총 투자 금액 ($)", min_value=1000, value=10000, step=1000)
    chart_window_label = st.selectbox("차트 기간", ["3개월", "6개월", "1년"], index=2)

    if st.button("분석 및 추천"):
        with st.spinner("시장 데이터를 분석하고 모멘텀을 계산하는 중..."):
            rec = strategy.recommend_portfolio(None, investment_capital=capital)

            st.subheader("시장 분석 (SPY 추세)")

            col1, col2 = st.columns(2)
            col1.metric("SPY 가격", f"${rec['SPY Price']:.2f}")
            col2.metric("날짜", str(rec["Date"]))

            spy_data = rec["SPY Data"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=spy_data.index, y=spy_data["Close"], name="SPY 가격"))
            fig.update_layout(title="SPY 가격", xaxis_title="날짜", yaxis_title="가격")
            st.plotly_chart(fig, use_container_width=True)

            st.success("상시 투자 전략 기준으로 모멘텀 상위 종목을 추천합니다.")

            top_df = pd.DataFrame(rec["Top Stocks"])
            raw_display_cols = [
                "Ticker",
                "Momentum",
                "Return 3M",
                "Return 6M",
                "Return 12M",
                "Volatility (63d)",
                "Price",
                "Shares to Buy",
                "Cost",
            ]
            raw_display_cols = [c for c in raw_display_cols if c in top_df.columns]
            col_map = {
                "Ticker": "티커",
                "Momentum": "모멘텀",
                "Return 3M": "3개월 수익률",
                "Return 6M": "6개월 수익률",
                "Return 12M": "12개월 수익률",
                "Volatility (63d)": "변동성(63일)",
                "Price": "현재가",
                "Shares to Buy": "매수 수량",
                "Cost": "투자금액",
            }
            display_df = top_df[raw_display_cols].rename(columns=col_map)

            st.dataframe(
                display_df.style.format(
                    {
                        "모멘텀": "{:.2f}%",
                        "3개월 수익률": "{:.2f}%",
                        "6개월 수익률": "{:.2f}%",
                        "12개월 수익률": "{:.2f}%",
                        "변동성(63일)": "{:.2f}%",
                        "현재가": "${:.2f}",
                        "투자금액": "${:,.2f}",
                    }
                ),
                use_container_width=True,
            )

            if not top_df.empty:
                st.subheader("종목 상세")
                cols = st.columns(3)
                for idx, row in top_df.iterrows():
                    col = cols[idx % 3]
                    with col:
                        st.markdown(f"**{row['Ticker']}**")
                        st.metric("모멘텀", f"{row['Momentum']:.2f}%")
                        r6 = row.get("Return 6M")
                        st.metric("6개월 수익률", "없음" if pd.isna(r6) else f"{r6:.2f}%")
                        r3 = row.get("Return 3M")
                        r12 = row.get("Return 12M")
                        vol = row.get("Volatility (63d)")
                        caption = (
                            f"3개월 수익률: {'없음' if pd.isna(r3) else f'{r3:.2f}%'} | "
                            f"12개월 수익률: {'없음' if pd.isna(r12) else f'{r12:.2f}%'} | "
                            f"변동성(63일): {'없음' if pd.isna(vol) else f'{vol:.2f}%'}"
                        )
                        st.caption(caption)

            st.info(
                f"배분 요약: 총 투자금액 **${rec['Total Cost']:,.2f}** | 남은 현금 **${rec['Leftover Cash']:,.2f}**"
            )

            if "Alternates" in rec and rec["Alternates"]:
                st.subheader("대체 후보 종목")
                alt_df = pd.DataFrame(rec["Alternates"])
                alt_raw_cols = [
                    "Ticker",
                    "Momentum",
                    "Return 3M",
                    "Return 6M",
                    "Return 12M",
                    "Volatility (63d)",
                    "Price",
                ]
                alt_raw_cols = [c for c in alt_raw_cols if c in alt_df.columns]
                alt_display_df = alt_df[alt_raw_cols].rename(columns=col_map)
                st.dataframe(
                    alt_display_df.style.format(
                        {
                            "모멘텀": "{:.2f}%",
                            "3개월 수익률": "{:.2f}%",
                            "6개월 수익률": "{:.2f}%",
                            "12개월 수익률": "{:.2f}%",
                            "변동성(63일)": "{:.2f}%",
                            "현재가": "${:.2f}",
                        }
                    ),
                    use_container_width=True,
                )

            if "Top Prices" in rec and not rec["Top Prices"].empty:
                st.subheader("모멘텀 점수 및 가격 추이")
                momentum_chart = go.Figure()
                momentum_chart.add_trace(
                    go.Bar(x=top_df["Ticker"], y=top_df["Momentum"], name="모멘텀 (%)")
                )
                momentum_chart.update_layout(
                    title="모멘텀 점수", xaxis_title="티커", yaxis_title="모멘텀 (%)"
                )
                st.plotly_chart(momentum_chart, use_container_width=True)

                lookback_map = {"3개월": 63, "6개월": 126, "1년": 252}
                lookback_days = lookback_map.get(chart_window_label, 252)
                price_df = rec["Top Prices"].copy().tail(lookback_days)
                normalized = price_df / price_df.iloc[0]
                price_chart = go.Figure()
                for col in normalized.columns:
                    price_chart.add_trace(go.Scatter(x=normalized.index, y=normalized[col], name=col))
                price_chart.update_layout(
                    title="정규화 가격 추이 (1.0 = 시작점)",
                    xaxis_title="날짜",
                    yaxis_title="정규화 가격",
                )
                st.plotly_chart(price_chart, use_container_width=True)

with tab2:
    st.header("백테스트 성과 분석")

    strategy_label_to_value = {
        "모멘텀": "Momentum",
        "혼합 모멘텀 (3/6/12)": "Blended Momentum (3/6/12)",
        "변동성 조정 모멘텀": "Volatility-Adjusted Momentum",
        "SPY 매수 후 보유": "Buy & Hold SPY",
    }
    strategy_label = st.selectbox("전략", list(strategy_label_to_value.keys()), index=0)
    strategy_name = strategy_label_to_value[strategy_label]

    interval_label_to_value = {"1개월": "1mo", "2주": "2wk", "1주": "1wk"}
    interval_label = st.selectbox("리밸런싱 간격", list(interval_label_to_value.keys()), index=0)
    interval = interval_label_to_value[interval_label]

    if st.button("시뮬레이션 실행"):
        with st.spinner("과거 시뮬레이션 실행 중... 잠시만 기다려주세요."):
            results, summary = strategy.run_simulation(interval, strategy_name=strategy_name)

            if results is not None:
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("연복리수익률(CAGR)", f"{summary['CAGR']:.2f}%")
                m2.metric("최대 낙폭", f"{summary['MDD']:.2f}%")
                m3.metric("평균 낙폭", f"{summary['Avg Drawdown']:.2f}%")
                m4.metric("총 수익률", f"{summary['Total Return']:.2f}%")
                m5.metric("최종 자산", f"${summary['Final Value']:,.0f}")

                fig2 = go.Figure()
                fig2.add_trace(
                    go.Scatter(x=results["Date"], y=results["Value"], name="포트폴리오 가치")
                )
                fig2.update_layout(
                    title="자산 곡선 ($10,000 시작)",
                    xaxis_title="날짜",
                    yaxis_title="포트폴리오 가치 ($)",
                )
                st.plotly_chart(fig2, use_container_width=True)

                fig3 = go.Figure()
                fig3.add_trace(
                    go.Scatter(
                        x=results["Date"],
                        y=results["Drawdown"],
                        name="낙폭",
                        fill="tozeroy",
                        line=dict(color="red"),
                    )
                )
                fig3.update_layout(title="과거 낙폭 (%)", xaxis_title="날짜", yaxis_title="낙폭 (%)")
                st.plotly_chart(fig3, use_container_width=True)

                st.subheader("시뮬레이션 이력")
                history_df = results[["Date", "Value", "Return", "Holdings"]].rename(
                    columns={
                        "Date": "날짜",
                        "Value": "가치",
                        "Return": "수익률",
                        "Holdings": "보유 종목",
                    }
                )
                st.dataframe(
                    history_df.style.format({"가치": "${:,.0f}", "수익률": "{:+.2f}%"}),
                    use_container_width=True,
                )
            else:
                st.error("시뮬레이션 실행에 실패했습니다. 데이터 연결을 확인해주세요.")

    if st.button("간격별 비교"):
        with st.spinner("간격별 비교 실행 중... 잠시만 기다려주세요."):
            rows = []
            for iv_label, iv_value in interval_label_to_value.items():
                results, summary = strategy.run_simulation(iv_value, strategy_name=strategy_name)
                if results is None:
                    continue
                rows.append(
                    {
                        "간격": iv_label,
                        "연복리수익률(CAGR)": summary["CAGR"],
                        "최대 낙폭": summary["MDD"],
                        "평균 낙폭": summary["Avg Drawdown"],
                        "총 수익률": summary["Total Return"],
                        "최종 자산": summary["Final Value"],
                    }
                )
            if rows:
                compare_df = pd.DataFrame(rows)
                st.subheader("간격 비교")
                st.dataframe(
                    compare_df.style.format(
                        {
                            "연복리수익률(CAGR)": "{:.2f}%",
                            "최대 낙폭": "{:.2f}%",
                            "평균 낙폭": "{:.2f}%",
                            "총 수익률": "{:.2f}%",
                            "최종 자산": "${:,.0f}",
                        }
                    ),
                    use_container_width=True,
                )
            else:
                st.error("간격 비교 실행에 실패했습니다. 데이터 연결을 확인해주세요.")

with tab3:
    st.header("자산 계산기")
    st.caption("행을 추가/삭제한 뒤 종목 이름, 주식 수, 현재가를 입력하세요.")

    if "asset_rows" not in st.session_state:
        st.session_state.asset_rows = [{"name": "", "shares": 0.0, "price": 0.0}]

    add_col, remove_col = st.columns([1, 1])
    with add_col:
        if st.button("+ 행 추가", key="add_asset_row"):
            st.session_state.asset_rows.append({"name": "", "shares": 0.0, "price": 0.0})
    with remove_col:
        if st.button("- 행 삭제", key="remove_asset_row"):
            if len(st.session_state.asset_rows) > 1:
                st.session_state.asset_rows.pop()

    total_asset_value = 0.0
    st.markdown("### 보유 종목")
    header_cols = st.columns([3, 2, 2, 2])
    header_cols[0].markdown("**이름**")
    header_cols[1].markdown("**주식 수**")
    header_cols[2].markdown("**현재가**")
    header_cols[3].markdown("**소계**")

    for i, row in enumerate(st.session_state.asset_rows):
        cols = st.columns([3, 2, 2, 2])
        name = cols[0].text_input(
            "이름",
            value=row["name"],
            key=f"asset_name_{i}",
            label_visibility="collapsed",
        )
        shares = cols[1].number_input(
            "주식 수",
            min_value=0.0,
            value=float(row["shares"]),
            step=1.0,
            key=f"asset_shares_{i}",
            label_visibility="collapsed",
        )
        price = cols[2].number_input(
            "현재가",
            min_value=0.0,
            value=float(row["price"]),
            step=0.01,
            key=f"asset_price_{i}",
            label_visibility="collapsed",
        )
        subtotal = shares * price
        cols[3].markdown(f"${subtotal:,.2f}")

        st.session_state.asset_rows[i] = {"name": name, "shares": shares, "price": price}
        total_asset_value += subtotal

    st.markdown("---")
    st.metric("총 평가금액", f"${total_asset_value:,.2f}")

st.markdown("---")
st.caption("데이터 제공: yfinance. 과거 수익률은 미래 수익을 보장하지 않습니다.")
