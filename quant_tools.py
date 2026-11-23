from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import streamlit as st


@st.cache_data(show_spinner=False, max_entries=2)
def load_close_prices(path: Path, modified_ns: int) -> pd.DataFrame:
    # The modification time invalidates this entry when the lab refreshes its cache.
    columns = [name for name in pq.read_schema(path).names if name.startswith("CLOSE__")]
    prices = pd.read_parquet(path, columns=columns)
    prices.columns = prices.columns.str.removeprefix("CLOSE__")
    return prices.where(prices > 0).dropna(how="all", axis=1).dropna(how="all").sort_index()


def render_asset_comparison(cache_path: Path) -> None:
    st.title("종목 비교")
    st.caption("저장된 수정종가로 같은 기간의 누적 수익률과 최대 낙폭을 비교합니다.")
    try:
        prices = load_close_prices(cache_path, cache_path.stat().st_mtime_ns)
    except FileNotFoundError:
        st.info("저장된 가격 데이터가 없습니다. 리밸런싱 연구실에서 백테스트를 실행하면 종목을 비교할 수 있습니다.")
        return
    except (OSError, ValueError):
        st.error("가격 데이터를 읽을 수 없습니다. 연구실에서 가격 캐시를 새로고침해 주세요.")
        return
    if prices.empty:
        st.info("비교에 사용할 수정종가가 없습니다.")
        return

    latest = prices.index[-1]
    st.caption(f"가격 데이터 기간 · {prices.index[0]:%Y-%m-%d} ~ {latest:%Y-%m-%d}")
    default = [ticker for ticker in ("SPY", "AAPL", "MSFT") if ticker in prices]
    left, right = st.columns([3, 1])
    tickers = left.multiselect("비교 종목", sorted(prices.columns), default=default or list(prices.columns[:2]))
    periods = {"3개월": 3, "6개월": 6, "1년": 12, "3년": 36, "전체": None}
    period = right.selectbox("조회 기간", list(periods), index=2)
    if not tickers:
        st.info("비교할 종목을 선택하세요.")
        return

    months = periods[period]
    window = prices.loc[latest - pd.DateOffset(months=months):] if months else prices
    common = window[tickers].dropna()
    if len(common) < 2:
        st.info("선택한 종목의 공통 거래일이 부족합니다. 조회 기간이나 종목을 변경하세요.")
        return

    performance = common.div(common.iloc[0]).sub(1).mul(100)
    drawdown = common.div(common.cummax()).sub(1).mul(100)
    st.subheader("누적 수익률")
    st.caption(f"공통 거래일 {len(common):,}일 · {common.index[0]:%Y-%m-%d} ~ {common.index[-1]:%Y-%m-%d}")
    st.line_chart(performance, y_label="수익률 (%)", x_label="거래일")
    summary = pd.DataFrame({"누적 수익률": performance.iloc[-1], "최대 낙폭": drawdown.min()})
    st.dataframe(
        summary.rename_axis("종목").reset_index().sort_values("누적 수익률", ascending=False),
        hide_index=True,
        column_config={col: st.column_config.NumberColumn(format="%.2f%%") for col in summary},
    )


def contribution_projection(initial: int, monthly: int, annual_rate: float, years: int) -> pd.DataFrame:
    monthly_factor = (1 + annual_rate) ** (1 / 12)
    balance = float(initial)
    rows = [(0, initial, balance)]
    for month in range(1, years * 12 + 1):
        balance = balance * monthly_factor + monthly
        rows.append((month, initial + monthly * month, balance))
    return pd.DataFrame(rows, columns=["경과 개월", "납입 원금", "예상 자산"]).set_index("경과 개월")


def render_contribution_simulator() -> None:
    st.title("적립식 시뮬레이터")
    st.caption("연 수익률이 일정하다고 가정하고 매월 말 추가 납입합니다. 세금과 수수료는 제외합니다.")
    left, right = st.columns(2)
    initial = left.number_input("초기 투자금 (원)", min_value=0, value=10_000_000, step=1_000_000)
    monthly = right.number_input("월 납입액 (원)", min_value=0, value=300_000, step=50_000)
    rate = left.number_input("가정 연 수익률 (%)", min_value=-100.0, max_value=100.0, value=5.0, step=0.5)
    years = right.slider("투자 기간 (년)", 1, 50, 10)
    projection = contribution_projection(initial, monthly, rate / 100, years)
    principal, balance = projection.iloc[-1]
    first, second, third = st.columns(3)
    first.metric("총 납입 원금", f"{principal:,.0f}원")
    second.metric("예상 투자 손익", f"{balance - principal:,.0f}원")
    third.metric("예상 최종 자산", f"{balance:,.0f}원")
    st.subheader("월별 자산 추이")
    st.line_chart(projection, x_label="경과 개월", y_label="금액 (원)")
