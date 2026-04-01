from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from rebalance_lab.backtest import CONTROL_TICKERS, MonthlyBacktester, evaluate_strategies, save_result_artifacts
from rebalance_lab.data import fetch_sp500_snapshot, load_or_refresh_price_cache, persist_universe_snapshot
from rebalance_lab.planner import build_purchase_plan
from run_pipeline import build_eligibility_map


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CACHE_DIR = OUTPUT_DIR / "cache"


def ensure_rebalance_frequency_column(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if "rebalance_frequency" not in normalized.columns:
        normalized["rebalance_frequency"] = "monthly"
    normalized["rebalance_frequency"] = normalized["rebalance_frequency"].fillna("monthly").astype(str)
    return normalized


def parse_top_n_text(raw: str) -> list[int]:
    values = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    if not values:
        raise ValueError("Top N 값을 하나 이상 입력해야 합니다.")
    return values


def parse_frequency_text(raw: str) -> list[str]:
    supported = ["weekly", "monthly", "bimonthly", "quarterly", "semiannual", "annual"]
    values = []
    for part in raw.split(","):
        value = part.strip().lower()
        if not value:
            continue
        if value not in supported:
            raise ValueError(f"지원하지 않는 리밸런싱 주기입니다: {value}")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("리밸런싱 주기를 하나 이상 입력해야 합니다.")
    return values


def run_backtest(
    *,
    start: str,
    end: str | None,
    top_n_values: list[int],
    rebalance_frequencies: list[str],
    initial_capital: float,
    transaction_cost_bps: float,
    force_refresh: bool,
) -> dict[str, object]:
    universe_snapshot = fetch_sp500_snapshot()
    persist_universe_snapshot(universe_snapshot, OUTPUT_DIR / "sp500_snapshot.csv")

    tickers = sorted(set(universe_snapshot.tickers + CONTROL_TICKERS))
    price_bundle = load_or_refresh_price_cache(
        tickers=tickers,
        start=start,
        end=end,
        cache_path=CACHE_DIR / "price_history.parquet",
        force_refresh=force_refresh,
    )
    open_prices = price_bundle.open_prices
    close_prices = price_bundle.close_prices
    available_universe = [ticker for ticker in universe_snapshot.tickers if ticker in close_prices.columns]
    open_prices = open_prices.loc[:, available_universe + CONTROL_TICKERS]
    close_prices = close_prices.loc[:, available_universe + CONTROL_TICKERS]
    eligible_from = build_eligibility_map(universe_snapshot.metadata, available_universe)

    backtester = MonthlyBacktester(
        open_prices=open_prices,
        close_prices=close_prices,
        universe=available_universe,
        transaction_cost_bps=transaction_cost_bps,
        eligible_from=eligible_from,
        initial_capital=initial_capital,
        rebalance_frequency=rebalance_frequencies[0],
    )
    runs = evaluate_strategies(
        backtester,
        top_n_values=top_n_values,
        rebalance_frequencies=rebalance_frequencies,
    )
    summary, top_n_summary, best_run = save_result_artifacts(backtester, runs, OUTPUT_DIR)

    summary = ensure_rebalance_frequency_column(summary)
    top_n_summary = ensure_rebalance_frequency_column(top_n_summary)
    equity_curves = pd.read_csv(OUTPUT_DIR / "equity_curves.csv", index_col=0, parse_dates=True)
    current_holdings = pd.read_csv(OUTPUT_DIR / "current_model_portfolio.csv")
    latest_recommendations = pd.read_csv(OUTPUT_DIR / "latest_recommendations.csv")
    monthly_portfolio_history = ensure_rebalance_frequency_column(
        pd.read_csv(OUTPUT_DIR / "monthly_portfolio_history.csv")
    )
    trade_log = ensure_rebalance_frequency_column(pd.read_csv(OUTPUT_DIR / "trade_log.csv"))
    rebalance_summary = ensure_rebalance_frequency_column(pd.read_csv(OUTPUT_DIR / "rebalance_summary.csv"))

    return {
        "requested_start": start,
        "requested_end": end,
        "rebalance_frequencies": rebalance_frequencies,
        "latest_market_date": close_prices.index[-1].strftime("%Y-%m-%d"),
        "universe_size": len(available_universe),
        "backtester": backtester,
        "runs": runs,
        "summary": summary,
        "top_n_summary": top_n_summary,
        "best_run": best_run,
        "best_row": summary.iloc[0],
        "equity_curves": equity_curves,
        "current_holdings": current_holdings,
        "latest_recommendations": latest_recommendations,
        "monthly_portfolio_history": monthly_portfolio_history,
        "trade_log": trade_log,
        "rebalance_summary": rebalance_summary,
    }


def load_existing_outputs() -> dict[str, object] | None:
    required = [
        OUTPUT_DIR / "backtest_summary.csv",
        OUTPUT_DIR / "top_n_summary.csv",
        OUTPUT_DIR / "equity_curves.csv",
        OUTPUT_DIR / "current_model_portfolio.csv",
        OUTPUT_DIR / "latest_recommendations.csv",
        OUTPUT_DIR / "monthly_portfolio_history.csv",
        OUTPUT_DIR / "trade_log.csv",
        OUTPUT_DIR / "rebalance_summary.csv",
    ]
    if not all(path.exists() for path in required):
        return None

    summary = ensure_rebalance_frequency_column(pd.read_csv(OUTPUT_DIR / "backtest_summary.csv"))
    top_n_summary = ensure_rebalance_frequency_column(pd.read_csv(OUTPUT_DIR / "top_n_summary.csv"))
    equity_curves = pd.read_csv(OUTPUT_DIR / "equity_curves.csv", index_col=0, parse_dates=True)
    current_holdings = pd.read_csv(OUTPUT_DIR / "current_model_portfolio.csv")
    latest_recommendations = pd.read_csv(OUTPUT_DIR / "latest_recommendations.csv")
    monthly_portfolio_history = ensure_rebalance_frequency_column(
        pd.read_csv(OUTPUT_DIR / "monthly_portfolio_history.csv")
    )
    trade_log = ensure_rebalance_frequency_column(pd.read_csv(OUTPUT_DIR / "trade_log.csv"))
    rebalance_summary = ensure_rebalance_frequency_column(pd.read_csv(OUTPUT_DIR / "rebalance_summary.csv"))

    latest_market_date = ""
    if "as_of" in latest_recommendations.columns and not latest_recommendations.empty:
        latest_market_date = str(latest_recommendations["as_of"].iloc[0])

    return {
        "requested_start": "",
        "requested_end": None,
        "rebalance_frequencies": sorted(summary["rebalance_frequency"].dropna().astype(str).unique().tolist())
        if "rebalance_frequency" in summary.columns
        else ["monthly"],
        "latest_market_date": latest_market_date,
        "universe_size": None,
        "backtester": None,
        "runs": None,
        "summary": summary,
        "top_n_summary": top_n_summary,
        "best_row": summary.iloc[0],
        "equity_curves": equity_curves,
        "current_holdings": current_holdings,
        "latest_recommendations": latest_recommendations,
        "monthly_portfolio_history": monthly_portfolio_history,
        "trade_log": trade_log,
        "rebalance_summary": rebalance_summary,
    }


def pct(value: float) -> str:
    return f"{value:.2%}"


def num(value: float) -> str:
    return f"{value:,.2f}"


def frequency_label(value: str) -> str:
    mapping = {
        "weekly": "주간",
        "monthly": "월간",
        "bimonthly": "격월",
        "quarterly": "분기",
        "semiannual": "반기",
        "annual": "연간",
    }
    return mapping.get(value, value)


def build_run_label(strategy: str, rebalance_frequency: str, top_n: int) -> str:
    return f"{strategy} | {frequency_label(rebalance_frequency)} | Top N {int(top_n)}"


def resolve_selected_run(results: dict[str, object], selection_label: str | None) -> tuple[pd.Series, object | None]:
    summary = results["summary"]
    runs = results.get("runs")
    if runs is None or results.get("backtester") is None:
        return summary.iloc[0], None

    for row in summary.itertuples(index=False):
        label = build_run_label(row.strategy, row.rebalance_frequency, row.top_n)
        if label == selection_label:
            selected_row = summary[
                (summary["strategy"] == row.strategy)
                & (summary["rebalance_frequency"] == row.rebalance_frequency)
                & (summary["top_n"] == row.top_n)
            ].iloc[0]
            selected_run = next(
                run
                for run in runs
                if run.strategy.name == row.strategy
                and run.rebalance_frequency == row.rebalance_frequency
                and run.top_n == row.top_n
            )
            return selected_row, selected_run
    return summary.iloc[0], results.get("best_run")


def materialize_selected_results(results: dict[str, object], selected_row: pd.Series, selected_run: object | None) -> dict[str, object]:
    if selected_run is None or results.get("backtester") is None:
        selected = results.copy()
        selected["selected_row"] = selected_row
        return selected

    backtester = results["backtester"]
    backtester.set_rebalance_frequency(selected_run.rebalance_frequency)
    current_holdings = backtester.latest_holdings(selected_run)
    ranking_date, ranking_frame = backtester.latest_ranking_snapshot(selected_run.strategy, selected_run.top_n)
    ranking_frame.insert(0, "as_of", ranking_date.strftime("%Y-%m-%d"))

    selected = results.copy()
    selected["selected_row"] = selected_row
    selected["current_holdings"] = current_holdings
    selected["latest_recommendations"] = ranking_frame
    selected["monthly_portfolio_history"] = selected_run.portfolio_history
    selected["trade_log"] = selected_run.trade_log
    selected["rebalance_summary"] = selected_run.rebalance_summary
    return selected


def render_metric_strip(best_row: pd.Series, latest_market_date: str, universe_size: int | None) -> None:
    st.markdown(
        f"""
        <div style="
            background: rgba(255,255,255,0.86);
            border: 1px solid rgba(19,38,47,0.08);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: 0 14px 30px rgba(19,38,47,0.06);
            margin-bottom: 0.8rem;
        ">
            <div style="font-size:0.9rem; color:#47606a; margin-bottom:0.35rem;">현재 최고 조합</div>
            <div style="font-size:1.25rem; font-weight:700; color:#13262f; word-break:break-word;">
                전략: {best_row["strategy"]}
            </div>
            <div style="font-size:1rem; color:#2f4a54; margin-top:0.35rem;">
                주기: {frequency_label(str(best_row["rebalance_frequency"]))} | Top N: {int(best_row["top_n"])}
            </div>
            <div style="font-size:0.92rem; color:#5d727a; margin-top:0.35rem; word-break:break-word;">
                설명: {best_row["description"]}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("총수익률", pct(float(best_row["total_return"])))
    col2.metric("CAGR", pct(float(best_row["cagr"])))
    col3.metric("최대낙폭", pct(float(best_row["max_drawdown"])))
    col4.metric("최종자산", num(float(best_row["final_equity"])))

    meta_parts = []
    if latest_market_date:
        meta_parts.append(f"최신 장 마감 데이터 기준일: {latest_market_date}")
    if universe_size is not None:
        meta_parts.append(f"유니버스 수: {universe_size}개 종목")
    meta_parts.append(f"실제 성과 집계 구간: {best_row['start_date']} ~ {best_row['end_date']}")
    meta_parts.append("신호 계산: 주기 말 종가")
    meta_parts.append("실행 시점: 다음 거래일 시가")
    st.caption(" | ".join(meta_parts))


def render_equity_chart(equity_curves: pd.DataFrame, summary: pd.DataFrame) -> None:
    top_labels = [
        f"{row.strategy}_{row.rebalance_frequency}_n{int(row.top_n)}"
        for row in summary.head(8).itertuples(index=False)
    ]
    selected_columns = [column for column in top_labels if column in equity_curves.columns]
    if "spy_buy_and_hold" in equity_curves.columns:
        selected_columns.append("spy_buy_and_hold")
    chart_df = equity_curves[selected_columns].reset_index()
    date_column = chart_df.columns[0]
    chart_df = chart_df.rename(columns={date_column: "date"})
    melted = chart_df.melt(id_vars="date", var_name="series", value_name="equity")
    fig = px.line(
        melted,
        x="date",
        y="equity",
        color="series",
        log_y=True,
        template="plotly_white",
    )
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)


def render_holdings(current_holdings: pd.DataFrame) -> None:
    stocks = current_holdings[current_holdings["ticker"] != "CASH"].copy()
    if stocks.empty:
        st.info("현재 보유 종목이 없습니다.")
        return
    fig = px.bar(
        stocks.sort_values("market_value", ascending=False),
        x="ticker",
        y="market_value",
        color="weight",
        color_continuous_scale="Tealgrn",
        template="plotly_white",
    )
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    display = stocks[["ticker", "shares", "latest_price", "market_value", "weight"]].copy()
    display.columns = ["종목", "보유주수", "최신종가", "평가금액", "비중"]
    display["최신종가"] = display["최신종가"].map(lambda x: f"{x:,.2f}")
    display["평가금액"] = display["평가금액"].map(num)
    display["비중"] = display["비중"].map(pct)
    st.dataframe(display, use_container_width=True, hide_index=True)

    cash_row = current_holdings[current_holdings["ticker"] == "CASH"]
    if not cash_row.empty:
        row = cash_row.iloc[0]
        st.caption(f"남은 현금: {num(float(row['market_value']))} ({pct(float(row['weight']))})")


def render_rebalance_explorer(
    monthly_portfolio_history: pd.DataFrame,
    trade_log: pd.DataFrame,
    rebalance_summary: pd.DataFrame,
) -> None:
    if rebalance_summary.empty:
        st.info("리밸런싱 내역이 없습니다.")
        return

    choices = (
        rebalance_summary[["rebalance_no", "effective_date", "signal_date"]]
        .assign(label=lambda df: df["rebalance_no"].astype(str) + " | " + df["effective_date"])
    )
    selected_label = st.selectbox("리밸런싱 회차", choices["label"].tolist(), index=len(choices) - 1)
    selected_no = int(choices.loc[choices["label"] == selected_label, "rebalance_no"].iloc[0])

    summary_row = rebalance_summary[rebalance_summary["rebalance_no"] == selected_no].iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("신호 기준일", str(summary_row["signal_date"]))
    col2.metric("실행일", str(summary_row["effective_date"]))
    col3.metric("회전율", pct(float(summary_row["turnover"])))
    col4.metric("수수료", num(float(summary_row["fees"])))
    st.caption("신호는 기준일 종가로 계산하고, 실제 매매는 실행일 시가에 이뤄집니다.")

    portfolio = monthly_portfolio_history[monthly_portfolio_history["rebalance_no"] == selected_no].copy()
    trades = trade_log[trade_log["rebalance_no"] == selected_no].copy()
    if "execution_price" not in portfolio.columns:
        portfolio["execution_price"] = pd.NA

    left, right = st.columns([1.2, 1.0])
    with left:
        display_portfolio = portfolio.copy()
        display_portfolio["price"] = display_portfolio["price"].map(lambda x: f"{x:,.2f}")
        display_portfolio["execution_price"] = display_portfolio["execution_price"].map(
            lambda x: "" if pd.isna(x) else f"{float(x):,.2f}"
        )
        display_portfolio["market_value"] = display_portfolio["market_value"].map(num)
        display_portfolio["weight"] = display_portfolio["weight"].map(pct)
        display_portfolio["target_weight"] = display_portfolio["target_weight"].map(
            lambda x: "" if pd.isna(x) else pct(float(x))
        )
        display_portfolio = display_portfolio.rename(
            columns={
                "ticker": "종목",
                "shares": "보유주수",
                "execution_price": "실행시가",
                "price": "당일종가",
                "market_value": "평가금액",
                "weight": "실제비중",
                "target_weight": "목표비중",
                "rank_on_signal": "신호순위",
            }
        )
        st.dataframe(
            display_portfolio[
                ["종목", "보유주수", "실행시가", "당일종가", "평가금액", "실제비중", "목표비중", "신호순위"]
            ],
            use_container_width=True,
            hide_index=True,
        )
    with right:
        if trades.empty:
            st.info("이 회차에는 거래가 없습니다.")
        else:
            display_trades = trades.copy()
            display_trades["price"] = display_trades["price"].map(lambda x: f"{x:,.2f}")
            display_trades["notional"] = display_trades["notional"].map(num)
            display_trades["fee"] = display_trades["fee"].map(num)
            display_trades = display_trades.rename(
                columns={
                    "action": "구분",
                    "ticker": "종목",
                    "shares": "주수",
                    "price": "체결시가",
                    "notional": "거래금액",
                    "fee": "수수료",
                    "pre_shares": "이전보유",
                    "post_shares": "변경후보유",
                    "rank_on_signal": "신호순위",
                }
            )
            st.dataframe(
                display_trades[
                    ["구분", "종목", "주수", "체결시가", "거래금액", "수수료", "이전보유", "변경후보유", "신호순위"]
                ],
                use_container_width=True,
                hide_index=True,
            )


def render_recommendations(latest_recommendations: pd.DataFrame) -> None:
    recommendations = latest_recommendations.copy()
    if recommendations.empty:
        st.info("추천 데이터가 없습니다.")
        return
    if "as_of" in recommendations.columns:
        st.caption(
            f"신호 기준일: {recommendations['as_of'].iloc[0]} 종가 기준입니다. "
            "실제 주문 체결은 다음 거래일 시가 기준입니다."
        )
    display = recommendations[["ticker", "score", "latest_price", "ret_1m", "ret_6m", "ret_12m", "selected"]].copy()
    display["score"] = display["score"].map(lambda x: f"{x:,.4f}")
    display["latest_price"] = display["latest_price"].map(lambda x: f"{x:,.2f}")
    for column in ["ret_1m", "ret_6m", "ret_12m"]:
        display[column] = display[column].map(pct)
    display = display.rename(
        columns={
            "ticker": "종목",
            "score": "점수",
            "latest_price": "기준종가",
            "ret_1m": "1개월",
            "ret_6m": "6개월",
            "ret_12m": "12개월",
            "selected": "현재선정",
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True)


def build_signal_target_holdings(latest_recommendations: pd.DataFrame, top_n: int) -> pd.DataFrame:
    recommendations = latest_recommendations.copy()
    if recommendations.empty:
        return pd.DataFrame(columns=["ticker", "weight", "latest_price", "market_value"])
    if "selected" in recommendations.columns:
        selected = recommendations[recommendations["selected"] == True].copy()
    else:
        selected = recommendations.head(top_n).copy()
    if selected.empty:
        selected = recommendations.head(top_n).copy()
    selected = selected.dropna(subset=["ticker", "latest_price"]).copy()
    if selected.empty:
        return pd.DataFrame(columns=["ticker", "weight", "latest_price", "market_value"])
    selected["weight"] = 1.0 / len(selected)
    selected["market_value"] = selected["latest_price"]
    base = selected[["ticker", "weight", "latest_price", "market_value"]].reset_index(drop=True)
    cash_row = pd.DataFrame([{"ticker": "CASH", "weight": 0.0, "latest_price": 1.0, "market_value": 0.0}])
    return pd.concat([base, cash_row], ignore_index=True)


def render_buy_plan_section(
    holdings_frame: pd.DataFrame,
    budget: float,
    title: str,
    caption: str,
) -> None:
    st.markdown(f"**{title}**")
    st.caption(caption)
    plan, cash_left = build_purchase_plan(current_holdings=holdings_frame, budget=budget)

    if plan.empty:
        st.info("매수 계획을 만들 수 없습니다.")
        return

    invested = float(plan["actual_value"].sum())
    col1, col2, col3 = st.columns(3)
    col1.metric("예산", num(budget))
    col2.metric("투입금액", num(invested))
    col3.metric("남는현금", num(cash_left))

    fig = px.bar(
        plan,
        x="ticker",
        y="actual_value",
        color="actual_weight",
        color_continuous_scale="YlGnBu",
        template="plotly_white",
    )
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    display = plan.copy()
    for column in ["target_weight", "actual_weight", "weight_gap"]:
        display[column] = display[column].map(pct)
    for column in ["latest_price", "target_value", "actual_value"]:
        display[column] = display[column].map(num)
    display = display.rename(
        columns={
            "ticker": "종목",
            "target_weight": "목표비중",
            "latest_price": "기준가격",
            "target_value": "목표금액",
            "shares_to_buy": "매수주수",
            "actual_value": "실제금액",
            "actual_weight": "실제비중",
            "weight_gap": "비중오차",
        }
    )
    st.dataframe(
        display[
            ["종목", "목표비중", "기준가격", "목표금액", "매수주수", "실제금액", "실제비중", "비중오차"]
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.caption("소수점 주식은 허용하지 않으므로 정수 주수로만 계산합니다. 남는 현금은 깔끔하게 배분되지 않은 금액입니다.")


def render_downloads() -> None:
    st.subheader("파일 내려받기")
    download_specs = [
        ("backtest_summary.csv", "전체 결과 CSV"),
        ("top_n_summary.csv", "Top N 요약 CSV"),
        ("current_model_portfolio.csv", "현재 포트폴리오 CSV"),
        ("latest_recommendations.csv", "최신 추천 CSV"),
        ("monthly_portfolio_history.csv", "월별 포트폴리오 CSV"),
        ("trade_log.csv", "매매 로그 CSV"),
        ("rebalance_summary.csv", "리밸런싱 요약 CSV"),
    ]
    cols = st.columns(3)
    for index, (filename, label) in enumerate(download_specs):
        path = OUTPUT_DIR / filename
        if not path.exists():
            continue
        with cols[index % 3]:
            st.download_button(
                label=label,
                data=path.read_bytes(),
                file_name=filename,
                mime="text/csv",
                use_container_width=True,
            )


def main() -> None:
    st.set_page_config(page_title="월별 리밸런싱 연구실", page_icon="chart_with_upwards_trend", layout="wide")
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(247, 201, 72, 0.20), transparent 28%),
                radial-gradient(circle at top right, rgba(18, 106, 117, 0.18), transparent 24%),
                linear-gradient(180deg, #f6f1e7 0%, #edf3ef 100%);
            color: #13262f;
        }
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.82);
            border: 1px solid rgba(19,38,47,0.08);
            border-radius: 18px;
            padding: 0.8rem 1rem;
            box-shadow: 0 14px 30px rgba(19,38,47,0.06);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("월별 리밸런싱 연구실")
    st.caption(
        "여러 리밸런싱 전략과 주기를 백테스트하고, 현재 추천 종목과 실제 매매 로그, 예산별 매수 계획까지 한 화면에서 확인합니다."
    )

    with st.sidebar:
        st.subheader("실행 설정")
        start_value = st.date_input("시작일", value=date(2014, 1, 1))
        end_enabled = st.checkbox("종료일 지정", value=False)
        end_value = st.date_input("종료일", value=date.today(), disabled=not end_enabled)
        top_n_text = st.text_input("Top N 목록", value="5,10,15,20,25,30")
        frequency_text = st.text_input(
            "리밸런싱 주기 목록",
            value="monthly,quarterly",
            help="입력 가능: weekly, monthly, bimonthly, quarterly, semiannual, annual",
        )
        initial_capital = st.number_input("초기 자본", min_value=10000.0, value=1_000_000.0, step=10000.0)
        transaction_cost_bps = st.number_input("거래 비용 bps", min_value=0.0, value=10.0, step=1.0)
        force_refresh = st.checkbox("가격 캐시 강제 새로고침", value=False)
        run_clicked = st.button("백테스트 실행", use_container_width=True, type="primary")
        load_clicked = st.button("기존 결과 불러오기", use_container_width=True)

    if "results" not in st.session_state:
        existing = load_existing_outputs()
        if existing is not None:
            st.session_state["results"] = existing

    if run_clicked:
        try:
            top_n_values = parse_top_n_text(top_n_text)
            rebalance_frequencies = parse_frequency_text(frequency_text)
            with st.spinner("백테스트를 실행 중입니다. 조합이 많으면 몇 분 정도 걸릴 수 있습니다."):
                st.session_state["results"] = run_backtest(
                    start=start_value.isoformat(),
                    end=end_value.isoformat() if end_enabled else None,
                    top_n_values=top_n_values,
                    rebalance_frequencies=rebalance_frequencies,
                    initial_capital=initial_capital,
                    transaction_cost_bps=transaction_cost_bps,
                    force_refresh=force_refresh,
                )
            st.success("백테스트가 완료되었습니다.")
        except Exception as exc:
            st.error(f"백테스트 실행 중 오류가 발생했습니다: {exc}")

    if load_clicked:
        existing = load_existing_outputs()
        if existing is None:
            st.warning("저장된 결과가 없습니다. 먼저 백테스트를 실행하세요.")
        else:
            st.session_state["results"] = existing
            st.success("저장된 결과를 불러왔습니다.")

    results = st.session_state.get("results")
    if results is None:
        st.info("왼쪽에서 조건을 설정한 뒤 `백테스트 실행` 또는 `기존 결과 불러오기`를 선택하세요.")
        return

    results["summary"] = ensure_rebalance_frequency_column(results["summary"])
    results["top_n_summary"] = ensure_rebalance_frequency_column(results["top_n_summary"])
    results["trade_log"] = ensure_rebalance_frequency_column(results["trade_log"])
    results["monthly_portfolio_history"] = ensure_rebalance_frequency_column(results["monthly_portfolio_history"])
    results["rebalance_summary"] = ensure_rebalance_frequency_column(results["rebalance_summary"])
    if "rebalance_frequencies" not in results or not results["rebalance_frequencies"]:
        results["rebalance_frequencies"] = sorted(results["summary"]["rebalance_frequency"].unique().tolist())
    results["best_row"] = results["summary"].iloc[0]
    best_row = results["best_row"]

    runs = results.get("runs")
    if runs is not None and results.get("backtester") is not None:
        labels = [
            build_run_label(row.strategy, row.rebalance_frequency, row.top_n)
            for row in results["summary"].itertuples(index=False)
        ]
        default_label = build_run_label(best_row["strategy"], best_row["rebalance_frequency"], best_row["top_n"])
        selected_label = st.selectbox(
            "보고 싶은 조합 선택",
            labels,
            index=labels.index(default_label) if default_label in labels else 0,
        )
        selected_row, selected_run = resolve_selected_run(results, selected_label)
        view_results = materialize_selected_results(results, selected_row, selected_run)
        st.caption("선택한 조합 기준으로 아래 포트폴리오, 추천, 리밸런싱 로그가 갱신됩니다.")
    else:
        selected_row, selected_run = best_row, None
        view_results = materialize_selected_results(results, selected_row, selected_run)
        st.info("현재는 저장된 결과만 불러온 상태라 최고 조합 기준만 볼 수 있습니다. 조합 선택은 앱에서 백테스트를 새로 실행하면 사용할 수 있습니다.")

    render_metric_strip(selected_row, str(view_results.get("latest_market_date", "")), view_results.get("universe_size"))
    st.caption(
        "비교한 주기: " + ", ".join(frequency_label(value) for value in results.get("rebalance_frequencies", ["monthly"]))
    )

    tab_overview, tab_recommend, tab_rebalance, tab_results = st.tabs(
        ["개요", "추천 및 매수 계획", "리밸런싱 로그", "전체 결과"]
    )

    with tab_overview:
        left, right = st.columns([1.35, 1.0])
        with left:
            st.subheader("누적 수익 곡선")
            render_equity_chart(results["equity_curves"], results["summary"])
        with right:
            st.subheader("주기별 / Top N별 최고 결과")
            top_n_summary = results["top_n_summary"].copy()
            display = top_n_summary[["rebalance_frequency", "top_n", "strategy", "total_return", "cagr", "max_drawdown"]].copy()
            display["rebalance_frequency"] = display["rebalance_frequency"].map(frequency_label)
            display["total_return"] = display["total_return"].map(pct)
            display["cagr"] = display["cagr"].map(pct)
            display["max_drawdown"] = display["max_drawdown"].map(pct)
            display.columns = ["주기", "Top N", "전략", "총수익률", "CAGR", "최대낙폭"]
            st.dataframe(display, use_container_width=True, hide_index=True)

        st.subheader("현재 모델 포트폴리오")
        render_holdings(view_results["current_holdings"])

    with tab_recommend:
        left, right = st.columns([1.1, 1.0])
        with left:
            st.subheader("최신 추천 종목")
            render_recommendations(view_results["latest_recommendations"])
        with right:
            st.subheader("매수 계획")
            budget = st.number_input("현재 보유 금액", min_value=0.0, value=10_000.0, step=1000.0)
            rebalance_summary = view_results["rebalance_summary"]
            last_trade_date = (
                str(rebalance_summary["effective_date"].iloc[-1])
                if not rebalance_summary.empty and "effective_date" in rebalance_summary.columns
                else "N/A"
            )
            signal_as_of = (
                str(view_results["latest_recommendations"]["as_of"].iloc[0])
                if "as_of" in view_results["latest_recommendations"].columns and not view_results["latest_recommendations"].empty
                else "N/A"
            )
            render_buy_plan_section(
                holdings_frame=view_results["current_holdings"],
                budget=budget,
                title="마지막 실제 체결 포트폴리오 기준",
                caption=f"마지막 실행일은 {last_trade_date} 입니다. 실제로 마지막 리밸런싱이 체결된 포트폴리오를 기준으로 계산합니다.",
            )
            signal_holdings = build_signal_target_holdings(
                view_results["latest_recommendations"],
                top_n=int(selected_row["top_n"]),
            )
            render_buy_plan_section(
                holdings_frame=signal_holdings,
                budget=budget,
                title="다음 리밸런싱 추천 신호 기준",
                caption=(
                    f"신호 기준일은 {signal_as_of} 종가입니다. "
                    "다음 거래일 시가에 실행된다고 가정하기 전의 계획용 계산이며, 현재는 최신 종가를 기준 가격으로 사용합니다."
                ),
            )

    with tab_rebalance:
        st.subheader("리밸런싱 실행 내역")
        render_rebalance_explorer(
            view_results["monthly_portfolio_history"],
            view_results["trade_log"],
            view_results["rebalance_summary"],
        )

    with tab_results:
        st.subheader("전체 전략 결과")
        summary_table = results["summary"].copy()
        summary_table["rebalance_frequency"] = summary_table["rebalance_frequency"].map(frequency_label)
        for column in ["total_return", "cagr", "annual_volatility", "max_drawdown", "avg_monthly_turnover"]:
            if column in summary_table.columns:
                summary_table[column] = summary_table[column].map(pct)
        if "sharpe" in summary_table.columns:
            summary_table["sharpe"] = summary_table["sharpe"].map(lambda x: f"{x:,.2f}")
        summary_table["initial_capital"] = summary_table["initial_capital"].map(num)
        summary_table["final_equity"] = summary_table["final_equity"].map(num)
        summary_table = summary_table.rename(
            columns={
                "strategy": "전략",
                "description": "설명",
                "rebalance_frequency": "주기",
                "top_n": "Top N",
                "start_date": "시작일",
                "end_date": "종료일",
                "initial_capital": "초기자본",
                "final_equity": "최종자산",
                "total_return": "총수익률",
                "cagr": "CAGR",
                "annual_volatility": "연변동성",
                "sharpe": "샤프",
                "max_drawdown": "최대낙폭",
                "avg_monthly_turnover": "평균회전율",
                "rebalance_count": "리밸런싱횟수",
            }
        )
        st.dataframe(summary_table, use_container_width=True, hide_index=True)
        render_downloads()


if __name__ == "__main__":
    main()
