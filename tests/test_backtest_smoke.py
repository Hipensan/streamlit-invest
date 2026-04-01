from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rebalance_lab.backtest import MonthlyBacktester
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
    backtester = MonthlyBacktester(
        open_prices=open_prices,
        close_prices=close_prices,
        universe=["AAA", "BBB", "CCC"],
        transaction_cost_bps=0.0,
        initial_capital=10_000.0,
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
    backtester = MonthlyBacktester(
        open_prices=open_prices,
        close_prices=close_prices,
        universe=["AAA", "BBB", "CCC"],
        transaction_cost_bps=0.0,
        initial_capital=10_000.0,
        rebalance_frequency="monthly",
    )
    monthly_run = backtester.run(Strategy(name="momentum_6m", description="synthetic test"), top_n=2)
    backtester.set_rebalance_frequency("quarterly")
    quarterly_run = backtester.run(Strategy(name="momentum_6m", description="synthetic test"), top_n=2)
    assert monthly_run.metrics.rebalance_count >= quarterly_run.metrics.rebalance_count


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


def test_price_cache_refreshes_when_requested_range_is_not_covered(monkeypatch) -> None:
    cached_index = pd.bdate_range("2020-01-02", periods=5)
    cached_bundle = PriceBundle(
        open_prices=pd.DataFrame({"AAA": [1, 2, 3, 4, 5]}, index=cached_index),
        close_prices=pd.DataFrame({"AAA": [1, 2, 3, 4, 5]}, index=cached_index),
    )
    cache_dir = Path("tests_cache")
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / "prices.parquet"
    data_module._bundle_to_cache_frame(cached_bundle).to_parquet(cache_path)

    refreshed_index = pd.bdate_range("2019-01-02", periods=5)
    refreshed_bundle = PriceBundle(
        open_prices=pd.DataFrame({"AAA": [10, 11, 12, 13, 14]}, index=refreshed_index),
        close_prices=pd.DataFrame({"AAA": [10, 11, 12, 13, 14]}, index=refreshed_index),
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
    if cache_path.exists():
        cache_path.unlink()
    if cache_dir.exists():
        cache_dir.rmdir()
