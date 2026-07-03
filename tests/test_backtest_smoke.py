from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rebalance_lab.backtest import MonthlyBacktester, evaluate_strategies
from rebalance_lab.data import PriceBundle
import rebalance_lab.data as data_module
from rebalance_lab.planner import build_purchase_plan
from rebalance_lab.strategies import Strategy


def test_monthly_backtester_runs_on_synthetic_prices() -> None:
    index = pd.bdate_range("2023-01-02", periods=320)
    close_prices = pd.DataFrame(
        {
            "AAA": np.linspace(100, 180, len(index)),
            "BBB": np.linspace(100, 140, len(index)),
            "CCC": np.linspace(100, 90, len(index)),
            "SPY": np.linspace(100, 150, len(index)),
        },
        index=index,
    )
    open_prices = close_prices * 0.99
    volume_prices = pd.DataFrame(
        {
            "AAA": np.linspace(1_000_000, 1_800_000, len(index)),
            "BBB": np.linspace(900_000, 1_100_000, len(index)),
            "CCC": np.linspace(700_000, 500_000, len(index)),
            "SPY": np.linspace(2_000_000, 2_400_000, len(index)),
        },
        index=index,
    )
    backtester = MonthlyBacktester(
        open_prices=open_prices,
        close_prices=close_prices,
        volume_prices=volume_prices,
        universe=["AAA", "BBB", "CCC"],
        transaction_cost_bps=0.0,
        initial_capital=10_000.0,
        rebalance_shift_days=0,
    )
    run = backtester.run(
        Strategy(name="momentum_12m", description="synthetic test"),
        top_n=2,
    )
    assert run.equity_curve.iloc[-1] > 1.0
    assert run.metrics.rebalance_count > 0
    assert not backtester.latest_holdings(run).empty
    assert not run.trade_log.empty
    assert not run.portfolio_history.empty
    assert (run.shares_history.fillna(0.0) % 1 == 0).all().all()
    assert "execution_price" in run.portfolio_history.columns
    assert run.trade_log["price"].notna().all()


def test_high_liquidity_strategy_prefers_highly_liquid_names() -> None:
    index = pd.bdate_range("2023-01-02", periods=320)
    close_prices = pd.DataFrame(
        {
            "AAA": np.linspace(100, 210, len(index)),
            "BBB": np.linspace(100, 210, len(index)),
            "CCC": np.linspace(100, 120, len(index)),
            "SPY": np.linspace(100, 170, len(index)),
        },
        index=index,
    )
    open_prices = close_prices * 0.99
    volume_prices = pd.DataFrame(
        {
            "AAA": np.concatenate([np.full(len(index) - 5, 1_000_000.0), np.full(5, 5_000_000.0)]),
            "BBB": np.full(len(index), 1_000_000.0),
            "CCC": np.full(len(index), 600_000.0),
            "SPY": np.full(len(index), 3_000_000.0),
        },
        index=index,
    )
    backtester = MonthlyBacktester(
        open_prices=open_prices,
        close_prices=close_prices,
        volume_prices=volume_prices,
        universe=["AAA", "BBB", "CCC"],
        transaction_cost_bps=0.0,
        initial_capital=10_000.0,
        rebalance_shift_days=0,
    )

    selected, ranking = backtester.select_target_tickers(
        Strategy(name="high_liquidity", description="synthetic volume strategy"),
        date=index[-2],
        top_n=1,
    )

    assert selected == ["AAA"]
    assert ranking.index[0] == "AAA"


def test_inverse_volatility_allocation_overweights_lower_vol_names() -> None:
    index = pd.bdate_range("2023-01-02", periods=320)
    aaa = np.linspace(100, 180, len(index))
    bbb = np.linspace(100, 180, len(index)) + 8.0 * np.sin(np.linspace(0, 24, len(index)))
    close_prices = pd.DataFrame(
        {
            "AAA": aaa,
            "BBB": bbb,
            "CCC": np.linspace(100, 120, len(index)),
            "SPY": np.linspace(100, 150, len(index)),
        },
        index=index,
    )
    open_prices = close_prices * 0.99
    volume_prices = pd.DataFrame(
        {
            "AAA": np.full(len(index), 1_000_000.0),
            "BBB": np.full(len(index), 1_000_000.0),
            "CCC": np.full(len(index), 800_000.0),
            "SPY": np.full(len(index), 2_500_000.0),
        },
        index=index,
    )
    backtester = MonthlyBacktester(
        open_prices=open_prices,
        close_prices=close_prices,
        volume_prices=volume_prices,
        universe=["AAA", "BBB", "CCC"],
        transaction_cost_bps=0.0,
        initial_capital=10_000.0,
        rebalance_shift_days=0,
    )

    weights = backtester._build_target_weights(
        Strategy(name="test_inverse_vol", description="risk-aware", allocation_mode="inverse_volatility"),
        selected=["AAA", "BBB"],
        signal_date=index[-2],
    )

    assert abs(weights.sum() - 1.0) < 1e-9
    assert weights["AAA"] > weights["BBB"]


def test_turnover_guard_keeps_existing_holdings_when_trade_is_too_small() -> None:
    index = pd.bdate_range("2023-01-02", periods=320)
    close_prices = pd.DataFrame(
        {
            "AAA": np.linspace(100, 180, len(index)),
            "BBB": np.linspace(100, 180, len(index)) + 4.0 * np.sin(np.linspace(0, 12, len(index))),
            "CCC": np.linspace(100, 110, len(index)),
            "SPY": np.linspace(100, 150, len(index)),
        },
        index=index,
    )
    open_prices = close_prices * 0.99
    volume_prices = pd.DataFrame(
        {
            "AAA": np.full(len(index), 1_100_000.0),
            "BBB": np.full(len(index), 1_100_000.0),
            "CCC": np.full(len(index), 900_000.0),
            "SPY": np.full(len(index), 2_500_000.0),
        },
        index=index,
    )
    backtester = MonthlyBacktester(
        open_prices=open_prices,
        close_prices=close_prices,
        volume_prices=volume_prices,
        universe=["AAA", "BBB", "CCC"],
        transaction_cost_bps=0.0,
        initial_capital=10_000.0,
        rebalance_shift_days=0,
    )
    current_shares = pd.Series({"AAA": 45, "BBB": 45, "CCC": 0}, dtype=int)
    current_cash = 200.0
    signal_date = index[-2]
    effective_date = index[-1]

    next_shares, next_cash, trade_rows, portfolio_rows, rebalance_row = backtester._rebalance_portfolio(
        strategy=Strategy(
            name="test_turnover_guard",
            description="skip small rebalance",
            allocation_mode="inverse_volatility",
            min_turnover=0.95,
        ),
        top_n=2,
        signal_date=signal_date,
        effective_date=effective_date,
        rebalance_no=1,
        selected=["AAA", "BBB"],
        ranking=pd.Series([2.0, 1.0], index=["AAA", "BBB"]),
        current_shares=current_shares,
        current_cash=current_cash,
    )

    assert next_shares.equals(current_shares)
    assert next_cash == current_cash
    assert trade_rows == []
    assert rebalance_row["skipped_by_turnover"] is True
    assert rebalance_row["turnover"] == 0.0
    stock_rows = [row for row in portfolio_rows if row["ticker"] != "CASH"]
    assert stock_rows
    assert any(row["target_weight"] > 0 for row in stock_rows)


def test_rebalance_frequency_changes_trade_count() -> None:
    index = pd.bdate_range("2023-01-02", periods=320)
    close_prices = pd.DataFrame(
        {
            "AAA": np.linspace(100, 180, len(index)),
            "BBB": np.linspace(90, 160, len(index)),
            "CCC": np.linspace(110, 130, len(index)),
            "SPY": np.linspace(100, 150, len(index)),
        },
        index=index,
    )
    open_prices = close_prices * 0.99
    volume_prices = pd.DataFrame(
        {
            "AAA": np.linspace(1_000_000, 1_300_000, len(index)),
            "BBB": np.linspace(950_000, 1_250_000, len(index)),
            "CCC": np.linspace(800_000, 900_000, len(index)),
            "SPY": np.linspace(2_000_000, 2_200_000, len(index)),
        },
        index=index,
    )
    backtester = MonthlyBacktester(
        open_prices=open_prices,
        close_prices=close_prices,
        volume_prices=volume_prices,
        universe=["AAA", "BBB", "CCC"],
        transaction_cost_bps=0.0,
        initial_capital=10_000.0,
        rebalance_frequency="monthly",
        rebalance_shift_days=0,
    )
    monthly_run = backtester.run(Strategy(name="momentum_6m", description="synthetic test"), top_n=2)
    backtester.set_rebalance_frequency("quarterly")
    quarterly_run = backtester.run(Strategy(name="momentum_6m", description="synthetic test"), top_n=2)
    assert monthly_run.metrics.rebalance_count >= quarterly_run.metrics.rebalance_count


def test_evaluate_strategies_can_filter_selected_strategies() -> None:
    index = pd.bdate_range("2023-01-02", periods=320)
    close_prices = pd.DataFrame(
        {
            "AAA": np.linspace(100, 180, len(index)),
            "BBB": np.linspace(90, 160, len(index)),
            "CCC": np.linspace(110, 130, len(index)),
            "SPY": np.linspace(100, 150, len(index)),
        },
        index=index,
    )
    open_prices = close_prices * 0.99
    volume_prices = pd.DataFrame(
        {
            "AAA": np.full(len(index), 1_000_000.0),
            "BBB": np.full(len(index), 1_000_000.0),
            "CCC": np.full(len(index), 900_000.0),
            "SPY": np.full(len(index), 2_000_000.0),
        },
        index=index,
    )
    backtester = MonthlyBacktester(
        open_prices=open_prices,
        close_prices=close_prices,
        volume_prices=volume_prices,
        universe=["AAA", "BBB", "CCC"],
        transaction_cost_bps=0.0,
        initial_capital=10_000.0,
        rebalance_frequency="monthly",
        rebalance_shift_days=0,
    )

    runs = evaluate_strategies(
        backtester,
        top_n_values=[2],
        rebalance_frequencies=["monthly"],
        strategy_names=["momentum_6m", "low_volatility"],
    )

    assert [run.strategy.name for run in runs] == ["momentum_6m", "low_volatility"]


def test_rebalance_contribution_is_added_to_cash_flow() -> None:
    index = pd.bdate_range("2023-01-02", periods=320)
    close_prices = pd.DataFrame(
        {
            "AAA": np.linspace(100, 180, len(index)),
            "BBB": np.linspace(90, 160, len(index)),
            "CCC": np.linspace(110, 130, len(index)),
            "SPY": np.linspace(100, 150, len(index)),
        },
        index=index,
    )
    open_prices = close_prices * 0.99
    volume_prices = pd.DataFrame(
        {
            "AAA": np.full(len(index), 1_000_000.0),
            "BBB": np.full(len(index), 1_000_000.0),
            "CCC": np.full(len(index), 900_000.0),
            "SPY": np.full(len(index), 2_000_000.0),
        },
        index=index,
    )
    backtester = MonthlyBacktester(
        open_prices=open_prices,
        close_prices=close_prices,
        volume_prices=volume_prices,
        universe=["AAA", "BBB", "CCC"],
        transaction_cost_bps=0.0,
        initial_capital=10_000.0,
        rebalance_frequency="monthly",
        rebalance_contribution=1_000.0,
        rebalance_shift_days=0,
    )

    run = backtester.run(Strategy(name="momentum_6m", description="synthetic test"), top_n=2)

    assert not run.rebalance_summary.empty
    expected_contributions = 1_000.0 * len(run.rebalance_summary)
    assert run.rebalance_summary["cash_contribution"].sum() == expected_contributions
    assert run.metrics.total_contributions == expected_contributions
    assert run.metrics.total_invested_capital == 10_000.0 + expected_contributions
    assert run.metrics.final_equity > run.metrics.total_invested_capital


def test_purchase_plan_uses_integer_shares_and_leaves_cash() -> None:
    holdings = pd.DataFrame(
        [
            {"ticker": "AAA", "weight": 0.5, "latest_price": 110.0, "market_value": 550.0},
            {"ticker": "BBB", "weight": 0.3, "latest_price": 55.0, "market_value": 330.0},
            {"ticker": "CCC", "weight": 0.2, "latest_price": 25.0, "market_value": 120.0},
            {"ticker": "CASH", "weight": 0.0, "latest_price": 1.0, "market_value": 0.0},
        ]
    )
    plan, cash_left = build_purchase_plan(holdings, budget=1000.0)
    assert not plan.empty
    assert (plan["shares_to_buy"] % 1 == 0).all()
    assert cash_left >= 0
    assert plan["actual_value"].sum() <= 1000.0 + 1e-9


def test_rebalance_shift_and_adjustment() -> None:
    index = pd.bdate_range("2023-01-02", "2023-02-28")
    # Remove 2023-01-24 to simulate a non-trading day (since 1월 31일 - 7일 = 24일)
    index = index[index != "2023-01-24"]

    close_prices = pd.DataFrame(
        {
            "AAA": np.linspace(100, 120, len(index)),
            "SPY": np.linspace(100, 110, len(index)),
        },
        index=index,
    )
    open_prices = close_prices * 0.99

    backtester = MonthlyBacktester(
        open_prices=open_prices,
        close_prices=close_prices,
        volume_prices=None,
        universe=["AAA"],
        rebalance_frequency="monthly",
        rebalance_shift_days=7,
        non_trading_day_adjustment="prior",
    )

    signal_dates = backtester._build_signal_dates()
    # 2023-01-31 is the last trading day of Jan 2023.
    # 7 days earlier is 2023-01-24, which we removed.
    # The prior trading day should be 2023-01-23 (Monday).
    assert pd.Timestamp("2023-01-23") in signal_dates
    assert pd.Timestamp("2023-01-24") not in signal_dates



def test_price_cache_refreshes_when_requested_range_is_not_covered(monkeypatch) -> None:
    cached_index = pd.bdate_range("2020-01-02", periods=5)
    cached_bundle = PriceBundle(
        open_prices=pd.DataFrame({"AAA": [1, 2, 3, 4, 5]}, index=cached_index),
        close_prices=pd.DataFrame({"AAA": [1, 2, 3, 4, 5]}, index=cached_index),
        volume_prices=pd.DataFrame({"AAA": [10, 11, 12, 13, 14]}, index=cached_index),
    )
    cache_dir = Path("tests_cache")
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / "prices.parquet"
    data_module._bundle_to_cache_frame(cached_bundle).to_parquet(cache_path)

    refreshed_index = pd.bdate_range("2019-01-02", periods=5)
    refreshed_bundle = PriceBundle(
        open_prices=pd.DataFrame({"AAA": [10, 11, 12, 13, 14]}, index=refreshed_index),
        close_prices=pd.DataFrame({"AAA": [10, 11, 12, 13, 14]}, index=refreshed_index),
        volume_prices=pd.DataFrame({"AAA": [20, 21, 22, 23, 24]}, index=refreshed_index),
    )

    calls = {"count": 0}

    def fake_download_price_history(tickers, start, end=None, chunk_size=100):
        calls["count"] += 1
        return refreshed_bundle

    monkeypatch.setattr(data_module, "download_price_history", fake_download_price_history)
    loaded = data_module.load_or_refresh_price_cache(
        tickers=["AAA"],
        start="2019-01-01",
        end=None,
        cache_path=cache_path,
        force_refresh=False,
    )
    assert calls["count"] == 1
    assert loaded.close_prices.index.min() == refreshed_index.min()
    assert loaded.volume_prices.index.min() == refreshed_index.min()
    if cache_path.exists():
        cache_path.unlink()
    if cache_dir.exists():
        cache_dir.rmdir()


def test_price_cache_refreshes_when_cached_bundle_has_no_volume(monkeypatch) -> None:
    cached_index = pd.bdate_range("2020-01-02", periods=5)
    legacy_cache = pd.concat(
        [
            pd.DataFrame({"OPEN__AAA": [1, 2, 3, 4, 5]}, index=cached_index),
            pd.DataFrame({"CLOSE__AAA": [1, 2, 3, 4, 5]}, index=cached_index),
        ],
        axis=1,
    )
    cache_dir = Path("tests_cache_legacy")
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / "prices.parquet"
    legacy_cache.to_parquet(cache_path)

    refreshed_bundle = PriceBundle(
        open_prices=pd.DataFrame({"AAA": [10, 11, 12, 13, 14]}, index=cached_index),
        close_prices=pd.DataFrame({"AAA": [10, 11, 12, 13, 14]}, index=cached_index),
        volume_prices=pd.DataFrame({"AAA": [20, 21, 22, 23, 24]}, index=cached_index),
    )

    calls = {"count": 0}

    def fake_download_price_history(tickers, start, end=None, chunk_size=100):
        calls["count"] += 1
        return refreshed_bundle

    monkeypatch.setattr(data_module, "download_price_history", fake_download_price_history)
    loaded = data_module.load_or_refresh_price_cache(
        tickers=["AAA"],
        start="2020-01-01",
        end=None,
        cache_path=cache_path,
        force_refresh=False,
    )

    assert calls["count"] == 1
    assert not loaded.volume_prices.empty
    if cache_path.exists():
        cache_path.unlink()
    if cache_dir.exists():
        cache_dir.rmdir()


def test_risk_parity_allocation() -> None:
    index = pd.bdate_range("2023-01-02", periods=65)
    close_prices = pd.DataFrame(
        {
            "AAA": np.linspace(100, 150, len(index)),
            "BBB": np.linspace(100, 110, len(index)) + np.sin(np.arange(len(index))),
            "CCC": np.linspace(100, 120, len(index)),
            "SPY": np.linspace(100, 130, len(index)),
        },
        index=index,
    )
    open_prices = close_prices * 0.99

    backtester = MonthlyBacktester(
        open_prices=open_prices,
        close_prices=close_prices,
        volume_prices=None,
        universe=["AAA", "BBB", "CCC"],
        transaction_cost_bps=0.0,
        initial_capital=10_000.0,
        rebalance_frequency="monthly",
        rebalance_shift_days=0,
        allocation_mode_override="risk_parity",
    )

    weights = backtester._build_target_weights(
        Strategy(name="momentum_6m", description="test"),
        selected=["AAA", "BBB"],
        signal_date=index[-1],
    )

    assert abs(weights.sum() - 1.0) < 1e-9
    # 변동성이 더 작은 AAA의 비중이 더 크게 배분되어야 함
    assert weights["AAA"] > weights["BBB"]


def test_mean_variance_allocation() -> None:
    index = pd.bdate_range("2023-01-02", periods=130)
    close_prices = pd.DataFrame(
        {
            "AAA": np.linspace(100, 200, len(index)),
            "BBB": np.linspace(100, 105, len(index)),
            "CCC": np.linspace(100, 110, len(index)),
            "SPY": np.linspace(100, 120, len(index)),
        },
        index=index,
    )
    open_prices = close_prices * 0.99

    backtester = MonthlyBacktester(
        open_prices=open_prices,
        close_prices=close_prices,
        volume_prices=None,
        universe=["AAA", "BBB", "CCC"],
        transaction_cost_bps=0.0,
        initial_capital=10_000.0,
        rebalance_frequency="monthly",
        rebalance_shift_days=0,
        allocation_mode_override="mean_variance",
    )

    weights = backtester._build_target_weights(
        Strategy(name="momentum_6m", description="test"),
        selected=["AAA", "BBB", "CCC"],
        signal_date=index[-1],
    )

    assert abs(weights.sum() - 1.0) < 1e-9
    assert weights["AAA"] <= 0.40001


def test_hybrid_strategy_blending() -> None:
    index = pd.bdate_range("2023-01-02", periods=130)
    close_prices = pd.DataFrame(
        {
            "AAA": np.linspace(100, 150, len(index)),
            "BBB": np.linspace(100, 120, len(index)),
            "CCC": np.linspace(100, 105, len(index)),
            "SPY": np.linspace(100, 125, len(index)),
        },
        index=index,
    )
    open_prices = close_prices * 0.99

    hybrid_w = {"momentum_3m": 0.5, "momentum_12m": 0.5}

    backtester = MonthlyBacktester(
        open_prices=open_prices,
        close_prices=close_prices,
        volume_prices=None,
        universe=["AAA", "BBB", "CCC"],
        transaction_cost_bps=0.0,
        initial_capital=10_000.0,
        rebalance_frequency="monthly",
        rebalance_shift_days=0,
        hybrid_weights=hybrid_w,
    )

    score = backtester.score_strategy("hybrid_strategy", index[-1])
    assert not score.empty
    assert "AAA" in score.index
    assert "BBB" in score.index


def test_non_momentum_strategies_smoke() -> None:
    index = pd.bdate_range("2023-01-02", periods=130)
    close_prices = pd.DataFrame(
        {
            "AAA": np.linspace(100, 150, len(index)),
            "BBB": np.linspace(100, 110, len(index)) + np.sin(np.arange(len(index))),
            "CCC": np.linspace(100, 120, len(index)),
            "SPY": np.linspace(100, 130, len(index)),
        },
        index=index,
    )
    open_prices = close_prices * 0.99
    volume_prices = pd.DataFrame(
        {
            "AAA": np.full(len(index), 1_000_000.0),
            "BBB": np.full(len(index), 1_200_000.0),
            "CCC": np.full(len(index), 800_000.0),
            "SPY": np.full(len(index), 3_000_000.0),
        },
        index=index,
    )

    backtester = MonthlyBacktester(
        open_prices=open_prices,
        close_prices=close_prices,
        volume_prices=volume_prices,
        universe=["AAA", "BBB", "CCC"],
        transaction_cost_bps=0.0,
        initial_capital=10_000.0,
        rebalance_frequency="monthly",
        rebalance_shift_days=0,
    )

    for strat_name in ["low_volatility", "mean_reversion_rsi", "high_liquidity", "min_drawdown"]:
        selected, ranking = backtester.select_target_tickers(
            Strategy(name=strat_name, description="non-momentum factor test"),
            date=index[-1],
            top_n=2,
        )
        assert len(selected) == 2
        assert not ranking.empty


def test_rebalance_pnl_tracking() -> None:
    index = pd.bdate_range("2023-01-02", periods=45)
    close_prices = pd.DataFrame(
        {
            "AAA": [100.0] * 30 + [120.0] * 15,
            "BBB": [100.0] * 30 + [80.0] * 15,
            "SPY": [100.0] * 45,
        },
        index=index,
    )
    open_prices = close_prices.copy()
    volume_prices = pd.DataFrame(
        {
            "AAA": [1_000_000.0] * 45,
            "BBB": [1_000_000.0] * 45,
            "SPY": [1_000_000.0] * 45,
        },
        index=index,
    )

    backtester = MonthlyBacktester(
        open_prices=open_prices,
        close_prices=close_prices,
        volume_prices=volume_prices,
        universe=["AAA", "BBB"],
        transaction_cost_bps=0.0,
        initial_capital=10_000.0,
        rebalance_frequency="monthly",
        rebalance_shift_days=0,
    )

    selected = ["AAA"]
    ranking = pd.Series([1.0, 0.0], index=["AAA", "BBB"])
    current_shares = pd.Series([0, 0], index=["AAA", "BBB"])
    current_cash = 10_000.0

    current_shares, current_cash, trade_rows, _, rebalance_row = backtester._rebalance_portfolio(
        strategy=Strategy(name="test_strat", description=""),
        top_n=1,
        signal_date=index[19],
        effective_date=index[20],
        rebalance_no=1,
        selected=selected,
        ranking=ranking,
        current_shares=current_shares,
        current_cash=current_cash,
    )

    assert backtester.average_costs["AAA"] == 100.0
    assert pd.isna(trade_rows[0]["realized_pnl"])

    selected2 = ["BBB"]
    ranking2 = pd.Series([0.0, 1.0], index=["AAA", "BBB"])
    current_shares, current_cash, trade_rows2, _, rebalance_row2 = backtester._rebalance_portfolio(
        strategy=Strategy(name="test_strat", description=""),
        top_n=1,
        signal_date=index[39],
        effective_date=index[40],
        rebalance_no=2,
        selected=selected2,
        ranking=ranking2,
        current_shares=current_shares,
        current_cash=current_cash,
    )

    sell_trade = [t for t in trade_rows2 if t["ticker"] == "AAA" and t["action"] == "SELL"][0]
    assert sell_trade["price"] == 120.0
    assert sell_trade["purchase_price"] == 100.0
    expected_shares = sell_trade["shares"]
    assert sell_trade["realized_pnl"] == expected_shares * 20.0
    assert abs(sell_trade["realized_pnl_pct"] - 0.20) < 1e-5


def test_annual_tax_deduction_in_december() -> None:
    index = pd.bdate_range("2023-11-01", periods=45)
    close_prices = pd.DataFrame(
        {
            "AAA": [100.0] * 20 + [200.0] * 25,
            "BBB": [100.0] * 20 + [100.0] * 25,
            "SPY": [100.0] * 45,
        },
        index=index,
    )
    open_prices = close_prices.copy()
    volume_prices = pd.DataFrame(
        {
            "AAA": [1_000_000.0] * 45,
            "BBB": [1_000_000.0] * 45,
            "SPY": [1_000_000.0] * 45,
        },
        index=index,
    )

    backtester = MonthlyBacktester(
        open_prices=open_prices,
        close_prices=close_prices,
        volume_prices=volume_prices,
        universe=["AAA", "BBB"],
        transaction_cost_bps=0.0,
        initial_capital=20_000.0,
        rebalance_frequency="monthly",
        rebalance_shift_days=0,
    )
    backtester.rebalance_schedule = [
        (index[18], index[19]),
        (index[39], index[40]),
    ]

    selected = ["AAA"]
    ranking = pd.Series([1.0, 0.0], index=["AAA", "BBB"])
    current_shares = pd.Series([0, 0], index=["AAA", "BBB"])
    current_cash = 20_000.0

    current_shares, current_cash, trade_rows, _, rebalance_row = backtester._rebalance_portfolio(
        strategy=Strategy(name="test_strat", description=""),
        top_n=1,
        signal_date=index[18],
        effective_date=index[19],
        rebalance_no=1,
        selected=selected,
        ranking=ranking,
        current_shares=current_shares,
        current_cash=current_cash,
    )

    selected2 = ["BBB"]
    ranking2 = pd.Series([0.0, 1.0], index=["AAA", "BBB"])

    current_shares, current_cash, trade_rows2, _, rebalance_row2 = backtester._rebalance_portfolio(
        strategy=Strategy(name="test_strat", description=""),
        top_n=1,
        signal_date=index[39],
        effective_date=index[40],
        rebalance_no=2,
        selected=selected2,
        ranking=ranking2,
        current_shares=current_shares,
        current_cash=current_cash,
    )

    assert rebalance_row2["tax"] > 0
    expected_tax = (20_000.0 - 2000.0) * 0.22
    assert abs(rebalance_row2["tax"] - expected_tax) < 1e-4
    assert rebalance_row2["cash_after"] == current_cash
    assert abs(current_cash - 40.0) < 1e-4
