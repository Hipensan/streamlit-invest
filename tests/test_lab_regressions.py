from contextlib import nullcontext
from io import BytesIO

import numpy as np
import pandas as pd
import pytest

import app
from rebalance_lab.backtest import MonthlyBacktester
from rebalance_lab.data import PriceBundle, UniverseSnapshot
from rebalance_lab.planner import build_purchase_plan
from rebalance_lab.strategies import Strategy


def test_year_end_tax_and_history_are_independent_of_cutoff():
    index = pd.bdate_range("2026-01-01", "2027-01-08")
    prices = pd.DataFrame({"AAA": 100., "BBB": 100., "SPY": 100.}, index=index)
    prices.loc["2026-05-01":, "AAA"] = 200

    def replay(end):
        bars = prices.loc[:end]
        engine = MonthlyBacktester(bars, bars, None, ["AAA", "BBB"], transaction_cost_bps=0,
                                   rebalance_frequency="quarterly", rebalance_shift_days=30)

        def select(strategy, date, top_n):
            ticker = "AAA" if date.month < 5 else "BBB"
            return [ticker], pd.Series({ticker: 1.})

        engine.select_target_tickers = select
        return engine.run(Strategy("test", ""), top_n=1)

    full = replay("2027-01-08")
    for end in ("2026-06-01", "2026-09-01", "2026-12-02"):
        short = replay(end)
        pd.testing.assert_series_equal(short.equity_curve, full.equity_curve.loc[:end])
        pd.testing.assert_frame_equal(short.trade_log, full.trade_log.query("effective_date <= @end"))
        pd.testing.assert_frame_equal(short.rebalance_summary, full.rebalance_summary.query("effective_date <= @end"))
    tax = full.rebalance_summary.set_index("effective_date").tax
    assert tax["2026-06-01"] == 0
    assert tax["2026-12-02"] == pytest.approx((10_000 - 2000) * 0.22)


def test_missing_open_freezes_only_that_holding_until_it_can_trade():
    index = pd.bdate_range("2026-01-01", "2026-09-02")
    closes = pd.DataFrame(100., index=index, columns=["AAA", "BBB", "CCC", "SPY"])
    opens = closes.copy()
    opens.loc["2026-06-01", "AAA"] = np.nan
    closes.loc["2026-06-01", "AAA"] = np.nan
    engine = MonthlyBacktester(opens, closes, None, ["AAA", "BBB", "CCC"], transaction_cost_bps=0,
                               rebalance_frequency="quarterly", rebalance_shift_days=30)

    def select(strategy, date, top_n):
        tickers = ["AAA", "BBB"] if date.month < 5 else ["CCC"]
        return tickers, pd.Series(1., index=tickers)

    engine.select_target_tickers = select
    run = engine.run(Strategy("test", ""), top_n=2)
    assert run.shares_history.loc["2026-06-01"].to_dict() == {"AAA": 50, "BBB": 0, "CCC": 50}
    assert (run.trade_log.price > 0).all()
    assert set(run.trade_log.query("effective_date == '2026-06-01'").ticker) == {"BBB", "CCC"}
    assert run.trade_log.query("effective_date == '2026-09-01' and action == 'SELL'").ticker.tolist() == ["AAA"]
    mark = run.portfolio_history.query("effective_date == '2026-06-01' and ticker == 'AAA'").iloc[0]
    assert pd.isna(mark.execution_price) and mark.market_value == 5000
    assert np.allclose(run.equity_curve, 1)
    assert np.allclose(run.cash_series.loc["2026-03-02":], 0)


def test_factor_values_and_hybrid_warmup():
    index = pd.bdate_range("2023-01-03", periods=253)
    prices = pd.DataFrame({"AAA": 100., "UP": np.arange(253) + 100.,
                           "DOWN": 400. - np.arange(253), "SPY": 100.}, index=index)
    prices.loc[index[153], "AAA"] = 40
    prices.loc[index[-22], "AAA"] = 150
    prices.loc[index[-1], "AAA"] = 180
    engine = MonthlyBacktester(prices, prices, None, ["AAA", "UP", "DOWN", "SPY"],
                               eligible_from={"UP": index[-1] + pd.Timedelta(days=1)},
                               hybrid_weights={"momentum_12m": 0.8, "momentum_3m": 0.2})
    last = index[-1]
    assert engine.score_strategy("momentum_12m_skip_1m", last)["AAA"] == pytest.approx(0.5)
    assert engine.score_strategy("min_drawdown", last)["AAA"] == pytest.approx(-0.6)
    assert engine.rsi_14.loc[last, ["UP", "DOWN", "SPY"]].tolist() == [100., 0., 50.]
    assert engine.rsi_14.iloc[:14].isna().all().all()
    hybrid = Strategy("hybrid_strategy", "")
    assert engine.select_target_tickers(hybrid, index[80], 5)[0] == []
    assert set(engine.select_target_tickers(hybrid, last, 5)[0]) == {"AAA", "DOWN", "SPY"}


def test_first_run_restores_weights_and_downloads_selected_run(tmp_path, monkeypatch):
    index = pd.bdate_range("2023-01-03", periods=320)
    time = np.arange(len(index))
    prices = pd.DataFrame({"AAA": 100 + time + np.sin(time),
                           "BBB": 100 + time + 8 * np.sin(time), "SPY": 100.}, index=index)
    snapshot = UniverseSnapshot(str(index[-1].date()), ["AAA", "BBB"], pd.DataFrame({"Symbol": ["AAA", "BBB"]}))
    monkeypatch.setattr(app, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(app, "cached_fetch_sp500_snapshot", lambda: snapshot)
    monkeypatch.setattr(app, "cached_load_or_refresh_price_cache",
                        lambda **kwargs: PriceBundle(prices, prices, prices * 1000))
    result = app.run_backtest(start=str(index[0].date()), end=str(index[-1].date()), top_n_values=[2],
                              rebalance_frequencies=["monthly", "quarterly"], strategy_names=["momentum_3m"],
                              initial_capital=10_000., rebalance_contribution=0, transaction_cost_bps=7,
                              force_refresh=False, rebalance_shift_days=0, allocation_override="inverse_volatility")
    restored = app.load_existing_outputs()
    assert not list(tmp_path.glob("daily_signal*"))
    assert restored is not None
    weights = (1 / prices[["AAA", "BBB"]].pct_change().tail(63).std())
    weights /= weights.sum()
    for view in (result, restored):
        holdings = app.build_signal_target_holdings(view["latest_recommendations"], 2)
        plan, cash = build_purchase_plan(holdings, 25_000)
        assert plan.set_index("ticker").target_weight.to_dict() == pytest.approx(weights.to_dict())
        assert cash + plan.actual_value.sum() == pytest.approx(25_000)
    unselected = restored["latest_recommendations"].assign(selected=False)
    plan, cash = build_purchase_plan(app.build_signal_target_holdings(unselected, 2), 10_000)
    assert plan.empty and cash == 10_000

    row = result["summary"].iloc[1]
    run = next(r for r in result["runs"] if r.rebalance_frequency == row.rebalance_frequency)
    view = app.materialize_selected_results(result, row, run)
    downloads = {}
    monkeypatch.setattr(app.st, "columns", lambda n: [nullcontext() for _ in range(n)])
    monkeypatch.setattr(app.st, "download_button", lambda **kw: downloads.update({kw["file_name"]: kw["data"]}))
    app.render_downloads(view)
    assert len(downloads) == 7
    for filename, key in (("trade_log.csv", "trade_log"), ("rebalance_summary.csv", "rebalance_summary"),
                          ("monthly_portfolio_history.csv", "monthly_portfolio_history"),
                          ("current_model_portfolio.csv", "current_holdings"),
                          ("latest_recommendations.csv", "latest_recommendations")):
        export = pd.read_csv(BytesIO(downloads[filename]))
        pd.testing.assert_frame_equal(export, view[key].reset_index(drop=True), check_dtype=False)
    assert set(pd.read_csv(BytesIO(downloads["trade_log.csv"])).rebalance_frequency) == {row.rebalance_frequency}
    assert set(pd.read_csv(BytesIO(downloads["backtest_summary.csv"])).rebalance_frequency) == {"monthly", "quarterly"}
