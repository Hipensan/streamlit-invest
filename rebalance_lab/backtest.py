from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rebalance_lab.strategies import Strategy, build_strategy_library


CONTROL_TICKERS = ["SPY"]


@dataclass
class BacktestMetrics:
    strategy: str
    description: str
    rebalance_frequency: str
    top_n: int
    start_date: str
    end_date: str
    initial_capital: float
    final_equity: float
    total_return: float
    cagr: float
    annual_volatility: float
    sharpe: float
    max_drawdown: float
    avg_monthly_turnover: float
    rebalance_count: int


@dataclass
class StrategyRun:
    strategy: Strategy
    rebalance_frequency: str
    top_n: int
    metrics: BacktestMetrics
    daily_returns: pd.Series
    equity_curve: pd.Series
    shares_history: pd.DataFrame
    cash_series: pd.Series
    weight_history: pd.DataFrame
    rankings: dict[pd.Timestamp, pd.Series]
    trade_log: pd.DataFrame
    portfolio_history: pd.DataFrame
    rebalance_summary: pd.DataFrame


class MonthlyBacktester:
    def __init__(
        self,
        open_prices: pd.DataFrame,
        close_prices: pd.DataFrame,
        universe: list[str],
        transaction_cost_bps: float = 10.0,
        benchmark_ticker: str = "SPY",
        eligible_from: dict[str, pd.Timestamp] | None = None,
        initial_capital: float = 1_000_000.0,
        rebalance_frequency: str = "monthly",
    ) -> None:
        self.open_prices = open_prices.sort_index().copy()
        self.close_prices = close_prices.sort_index().copy()
        common_universe = set(self.open_prices.columns).intersection(self.close_prices.columns)
        self.universe = [ticker for ticker in universe if ticker in common_universe]
        self.transaction_cost = transaction_cost_bps / 10_000.0
        self.benchmark_ticker = benchmark_ticker
        self.initial_capital = float(initial_capital)
        self.rebalance_frequency = rebalance_frequency
        self.eligible_from = {ticker: None for ticker in self.universe}
        if eligible_from:
            for ticker, value in eligible_from.items():
                if ticker in self.eligible_from:
                    self.eligible_from[ticker] = value

        self.trade_prices = self.open_prices[self.universe].astype(float)
        self.daily_returns = self.close_prices.pct_change(fill_method=None).fillna(0.0)
        self.ret_21 = self.close_prices / self.close_prices.shift(21) - 1.0
        self.ret_63 = self.close_prices / self.close_prices.shift(63) - 1.0
        self.ret_126 = self.close_prices / self.close_prices.shift(126) - 1.0
        self.ret_252 = self.close_prices / self.close_prices.shift(252) - 1.0
        self.vol_63 = self.daily_returns.rolling(63).std() * np.sqrt(252)
        self.sma_200 = self.close_prices.rolling(200).mean()
        self.high_252 = self.close_prices.rolling(252).max()
        self.rebalance_schedule: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        self._refresh_rebalance_schedule()

    def _refresh_rebalance_schedule(self) -> None:
        signal_dates = self._build_signal_dates()
        self.rebalance_schedule: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        for signal_date in signal_dates:
            signal_position = self.close_prices.index.get_loc(signal_date)
            if signal_position + 1 >= len(self.close_prices.index):
                break
            effective_date = self.close_prices.index[signal_position + 1]
            self.rebalance_schedule.append((signal_date, effective_date))

    def set_rebalance_frequency(self, rebalance_frequency: str) -> None:
        self.rebalance_frequency = rebalance_frequency
        self._refresh_rebalance_schedule()

    def _build_signal_dates(self) -> pd.Index:
        frequency = self.rebalance_frequency.lower()
        if frequency == "weekly":
            grouped = self.close_prices.groupby(pd.Grouper(freq="W-FRI")).tail(1).index
            return pd.Index(grouped)
        if frequency == "monthly":
            grouped = self.close_prices.groupby(self.close_prices.index.to_period("M")).tail(1).index
            return pd.Index(grouped)
        if frequency == "bimonthly":
            grouped = self.close_prices.groupby(self.close_prices.index.to_period("M")).tail(1)
            month_ends = grouped.index
            month_numbers = month_ends.to_period("M").month
            return pd.Index(month_ends[month_numbers % 2 == 0])
        if frequency == "quarterly":
            grouped = self.close_prices.groupby(self.close_prices.index.to_period("Q")).tail(1).index
            return pd.Index(grouped)
        if frequency == "semiannual":
            grouped = self.close_prices.groupby(self.close_prices.index.to_period("Q")).tail(1)
            quarter_ends = grouped.index
            quarter_numbers = quarter_ends.to_period("Q").quarter
            return pd.Index(quarter_ends[quarter_numbers.isin([2, 4])])
        if frequency == "annual":
            grouped = self.close_prices.groupby(self.close_prices.index.to_period("Y")).tail(1).index
            return pd.Index(grouped)
        raise ValueError(f"Unsupported rebalance frequency: {self.rebalance_frequency}")

    def _benchmark_uptrend(self, date: pd.Timestamp) -> bool:
        benchmark_price = self.close_prices.at[date, self.benchmark_ticker]
        benchmark_sma = self.sma_200.at[date, self.benchmark_ticker]
        return pd.notna(benchmark_price) and pd.notna(benchmark_sma) and benchmark_price > benchmark_sma

    def score_strategy(self, strategy_name: str, date: pd.Timestamp) -> pd.Series:
        universe = self.universe
        ret_21 = self.ret_21.loc[date, universe]
        ret_63 = self.ret_63.loc[date, universe]
        ret_126 = self.ret_126.loc[date, universe]
        ret_252 = self.ret_252.loc[date, universe]
        vol_63 = self.vol_63.loc[date, universe]
        above_sma = self.close_prices.loc[date, universe] > self.sma_200.loc[date, universe]
        high_252 = self.high_252.loc[date, universe]
        breakout = (self.close_prices.loc[date, universe] / high_252) - 1.0
        eligibility_mask = pd.Series(True, index=universe, dtype=bool)
        for ticker in universe:
            eligible_from = self.eligible_from.get(ticker)
            if eligible_from is not None and date < eligible_from:
                eligibility_mask.at[ticker] = False

        if strategy_name == "momentum_3m":
            return ret_63.where(eligibility_mask)
        if strategy_name == "momentum_6m":
            return ret_126.where(eligibility_mask)
        if strategy_name == "momentum_12m":
            return ret_252.where(eligibility_mask)
        if strategy_name == "momentum_12m_skip_1m":
            return (ret_252 - ret_21).where(eligibility_mask)
        if strategy_name == "risk_adjusted_momentum":
            return (ret_252 / vol_63).where(eligibility_mask)
        if strategy_name == "trend_filtered_momentum":
            if not self._benchmark_uptrend(date):
                return pd.Series(dtype=float)
            return ret_252.where(above_sma & eligibility_mask)
        if strategy_name == "low_vol_momentum":
            score = ret_126 - (0.75 * vol_63)
            return score.where(eligibility_mask)
        if strategy_name == "breakout_52w":
            score = (0.60 * ret_126) + (0.40 * breakout.rank(pct=True))
            return score.where(above_sma & eligibility_mask)
        if strategy_name == "blend_momentum":
            score = (0.45 * ret_252) + (0.35 * ret_126) - (0.20 * vol_63) - (0.10 * ret_21.rank(pct=True))
            score = score.where(above_sma & eligibility_mask)
            if not self._benchmark_uptrend(date):
                return pd.Series(dtype=float)
            return score
        raise ValueError(f"Unknown strategy: {strategy_name}")

    def select_target_tickers(
        self,
        strategy: Strategy,
        date: pd.Timestamp,
        top_n: int,
    ) -> tuple[list[str], pd.Series]:
        score = self.score_strategy(strategy.name, date)
        score = score.replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False)
        if strategy.name in {"trend_filtered_momentum"}:
            score = score[score > 0]
        return score.head(top_n).index.tolist(), score

    def _evaluate_cash_after_trade(
        self,
        current_shares: pd.Series,
        current_cash: float,
        target_shares: pd.Series,
        price_row: pd.Series,
    ) -> tuple[float, pd.Series, pd.Series, float, float, float]:
        price_row = price_row.fillna(0.0)
        share_delta = (target_shares - current_shares).astype(int)
        buy_shares = share_delta.clip(lower=0)
        sell_shares = (-share_delta).clip(lower=0)
        buy_notional = float((buy_shares * price_row).sum())
        sell_notional = float((sell_shares * price_row).sum())
        fees = (buy_notional + sell_notional) * self.transaction_cost
        cash_after = float(current_cash + sell_notional - buy_notional - fees)
        return cash_after, buy_shares, sell_shares, buy_notional, sell_notional, fees

    def _build_target_shares(
        self,
        selected: list[str],
        price_row: pd.Series,
        total_equity: float,
        current_shares: pd.Series,
        current_cash: float,
    ) -> tuple[pd.Series, float, pd.Series, pd.Series, float, float, float]:
        target_shares = pd.Series(0, index=self.universe, dtype=int)
        selected_prices = price_row.reindex(selected).dropna()
        selected_prices = selected_prices[selected_prices > 0]
        if selected_prices.empty or total_equity <= 0:
            cash_after, buy_shares, sell_shares, buy_notional, sell_notional, fees = (
                self._evaluate_cash_after_trade(current_shares, current_cash, target_shares, price_row)
            )
            return (
                target_shares,
                cash_after,
                buy_shares,
                sell_shares,
                buy_notional,
                sell_notional,
                fees,
            )

        per_name_budget = total_equity / len(selected_prices)
        seed_shares = np.floor(per_name_budget / selected_prices).astype(int)
        target_shares.loc[selected_prices.index] = seed_shares

        while True:
            current_values = target_shares.loc[selected_prices.index] * selected_prices
            affordable_candidates = []
            for ticker in current_values.sort_values().index:
                test_target = target_shares.copy()
                test_target.at[ticker] += 1
                cash_after, *_ = self._evaluate_cash_after_trade(
                    current_shares=current_shares,
                    current_cash=current_cash,
                    target_shares=test_target,
                    price_row=price_row,
                )
                if cash_after >= -1e-9:
                    affordable_candidates.append((ticker, cash_after))
            if not affordable_candidates:
                break
            candidate = affordable_candidates[0][0]
            target_shares.at[candidate] += 1

        cash_after, buy_shares, sell_shares, buy_notional, sell_notional, fees = (
            self._evaluate_cash_after_trade(current_shares, current_cash, target_shares, price_row)
        )
        while cash_after < -1e-9:
            invested = target_shares[target_shares > 0].index
            if invested.empty:
                break
            invested_values = (target_shares.loc[invested] * price_row.loc[invested]).sort_values(ascending=False)
            ticker_to_reduce = invested_values.index[0]
            target_shares.at[ticker_to_reduce] -= 1
            cash_after, buy_shares, sell_shares, buy_notional, sell_notional, fees = (
                self._evaluate_cash_after_trade(current_shares, current_cash, target_shares, price_row)
            )

        return (
            target_shares,
            cash_after,
            buy_shares,
            sell_shares,
            buy_notional,
            sell_notional,
            fees,
        )

    def _rebalance_portfolio(
        self,
        strategy: Strategy,
        top_n: int,
        signal_date: pd.Timestamp,
        effective_date: pd.Timestamp,
        rebalance_no: int,
        selected: list[str],
        ranking: pd.Series,
        current_shares: pd.Series,
        current_cash: float,
    ) -> tuple[pd.Series, float, list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
        execution_price_row = self.trade_prices.loc[effective_date].fillna(0.0)
        close_price_row = self.close_prices.loc[effective_date, self.universe].fillna(0.0)
        portfolio_value_before = float((current_shares * execution_price_row).sum() + current_cash)
        (
            target_shares,
            cash_after,
            buy_shares,
            sell_shares,
            buy_notional,
            sell_notional,
            fees,
        ) = self._build_target_shares(
            selected=selected,
            price_row=execution_price_row,
            total_equity=portfolio_value_before,
            current_shares=current_shares,
            current_cash=current_cash,
        )
        traded_notional = buy_notional + sell_notional
        turnover = traded_notional / portfolio_value_before if portfolio_value_before > 0 else 0.0
        portfolio_value_after = float((target_shares * close_price_row).sum() + cash_after)

        trade_rows: list[dict[str, object]] = []
        share_delta = target_shares - current_shares
        ranked_lookup = {ticker: rank + 1 for rank, ticker in enumerate(ranking.index.tolist())}
        for ticker in share_delta.index:
            delta = int(share_delta.at[ticker])
            if delta == 0:
                continue
            action = "BUY" if delta > 0 else "SELL"
            shares = abs(delta)
            price = float(execution_price_row.at[ticker])
            notional = shares * price
            fee = notional * self.transaction_cost
            trade_rows.append(
                {
                    "strategy": strategy.name,
                    "rebalance_frequency": self.rebalance_frequency,
                    "top_n": top_n,
                    "rebalance_no": rebalance_no,
                    "signal_date": signal_date.strftime("%Y-%m-%d"),
                    "effective_date": effective_date.strftime("%Y-%m-%d"),
                    "action": action,
                    "ticker": ticker,
                    "shares": shares,
                    "price": price,
                    "notional": notional,
                    "fee": fee,
                    "pre_shares": int(current_shares.at[ticker]),
                    "post_shares": int(target_shares.at[ticker]),
                    "rank_on_signal": ranked_lookup.get(ticker),
                }
            )

        holdings_value = (target_shares * close_price_row).astype(float)
        total_equity = float(holdings_value.sum() + cash_after)
        target_weight = 1.0 / len(selected) if selected else 0.0
        portfolio_rows: list[dict[str, object]] = []
        for ticker in target_shares[target_shares > 0].index:
            market_value = float(holdings_value.at[ticker])
            weight = market_value / total_equity if total_equity > 0 else 0.0
            portfolio_rows.append(
                {
                    "strategy": strategy.name,
                    "rebalance_frequency": self.rebalance_frequency,
                    "top_n": top_n,
                    "rebalance_no": rebalance_no,
                    "signal_date": signal_date.strftime("%Y-%m-%d"),
                    "effective_date": effective_date.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "shares": int(target_shares.at[ticker]),
                    "price": float(close_price_row.at[ticker]),
                    "execution_price": float(execution_price_row.at[ticker]),
                    "market_value": market_value,
                    "weight": weight,
                    "target_weight": target_weight,
                    "cash_after": cash_after,
                    "total_equity": total_equity,
                    "rank_on_signal": ranked_lookup.get(ticker),
                }
            )
        portfolio_rows.append(
            {
                "strategy": strategy.name,
                "rebalance_frequency": self.rebalance_frequency,
                "top_n": top_n,
                "rebalance_no": rebalance_no,
                "signal_date": signal_date.strftime("%Y-%m-%d"),
                "effective_date": effective_date.strftime("%Y-%m-%d"),
                "ticker": "CASH",
                "shares": 0,
                "price": 1.0,
                "execution_price": np.nan,
                "market_value": cash_after,
                "weight": cash_after / total_equity if total_equity > 0 else 0.0,
                "target_weight": np.nan,
                "cash_after": cash_after,
                "total_equity": total_equity,
                "rank_on_signal": np.nan,
            }
        )

        rebalance_row = {
            "strategy": strategy.name,
            "rebalance_frequency": self.rebalance_frequency,
            "top_n": top_n,
            "rebalance_no": rebalance_no,
            "signal_date": signal_date.strftime("%Y-%m-%d"),
            "effective_date": effective_date.strftime("%Y-%m-%d"),
            "selected_count": len(selected),
            "portfolio_value_before": portfolio_value_before,
            "portfolio_value_after": portfolio_value_after,
            "buy_notional": buy_notional,
            "sell_notional": sell_notional,
            "fees": fees,
            "traded_notional": traded_notional,
            "turnover": turnover,
            "cash_after": cash_after,
        }
        return target_shares, cash_after, trade_rows, portfolio_rows, rebalance_row

    def run(self, strategy: Strategy, top_n: int) -> StrategyRun:
        index = self.close_prices.index
        shares_history = pd.DataFrame(0, index=index, columns=self.universe, dtype=int)
        cash_series = pd.Series(index=index, dtype=float)
        rankings: dict[pd.Timestamp, pd.Series] = {}
        trade_rows: list[dict[str, object]] = []
        portfolio_rows: list[dict[str, object]] = []
        rebalance_rows: list[dict[str, object]] = []

        current_shares = pd.Series(0, index=self.universe, dtype=int)
        current_cash = self.initial_capital
        schedule_lookup = {effective_date: signal_date for signal_date, effective_date in self.rebalance_schedule}
        rebalance_no = 0

        for date in index:
            signal_date = schedule_lookup.get(date)
            if signal_date is not None:
                rebalance_no += 1
                selected, ranking = self.select_target_tickers(strategy=strategy, date=signal_date, top_n=top_n)
                rankings[signal_date] = ranking
                (
                    current_shares,
                    current_cash,
                    new_trade_rows,
                    new_portfolio_rows,
                    rebalance_row,
                ) = self._rebalance_portfolio(
                    strategy=strategy,
                    top_n=top_n,
                    signal_date=signal_date,
                    effective_date=date,
                    rebalance_no=rebalance_no,
                    selected=selected,
                    ranking=ranking,
                    current_shares=current_shares,
                    current_cash=current_cash,
                )
                trade_rows.extend(new_trade_rows)
                portfolio_rows.extend(new_portfolio_rows)
                rebalance_rows.append(rebalance_row)

            shares_history.loc[date] = current_shares.values
            cash_series.loc[date] = current_cash

        holdings_value_close = shares_history.astype(float) * self.close_prices[self.universe]
        total_equity = holdings_value_close.sum(axis=1) + cash_series
        equity_curve = total_equity / self.initial_capital
        daily_returns = total_equity.pct_change(fill_method=None).fillna(0.0)
        weight_history = holdings_value_close.div(total_equity.replace(0.0, np.nan), axis=0).fillna(0.0)
        trade_log = pd.DataFrame(trade_rows)
        portfolio_history = pd.DataFrame(portfolio_rows)
        rebalance_summary = pd.DataFrame(rebalance_rows)
        metrics = self._build_metrics(
            strategy=strategy,
            top_n=top_n,
            daily_returns=daily_returns,
            equity_curve=equity_curve,
            rebalance_summary=rebalance_summary,
        )
        return StrategyRun(
            strategy=strategy,
            rebalance_frequency=self.rebalance_frequency,
            top_n=top_n,
            metrics=metrics,
            daily_returns=daily_returns,
            equity_curve=equity_curve,
            shares_history=shares_history,
            cash_series=cash_series,
            weight_history=weight_history,
            rankings=rankings,
            trade_log=trade_log,
            portfolio_history=portfolio_history,
            rebalance_summary=rebalance_summary,
        )

    def _build_metrics(
        self,
        strategy: Strategy,
        top_n: int,
        daily_returns: pd.Series,
        equity_curve: pd.Series,
        rebalance_summary: pd.DataFrame,
    ) -> BacktestMetrics:
        if rebalance_summary.empty:
            observed = equity_curve.copy()
        else:
            first_trade_date = pd.Timestamp(rebalance_summary.iloc[0]["effective_date"])
            observed = equity_curve[equity_curve.index >= first_trade_date]
        observed_returns = daily_returns.loc[observed.index]
        total_return = observed.iloc[-1] - 1.0
        annualized_return = observed.iloc[-1] ** (252 / max(len(observed), 1)) - 1.0
        annualized_vol = observed_returns.std() * np.sqrt(252)
        sharpe = annualized_return / annualized_vol if annualized_vol > 0 else np.nan
        rolling_max = observed.cummax()
        drawdown = observed / rolling_max - 1.0
        max_drawdown = drawdown.min()
        avg_monthly_turnover = float(rebalance_summary["turnover"].mean()) if not rebalance_summary.empty else 0.0
        rebalance_count = int((rebalance_summary["traded_notional"] > 0).sum()) if not rebalance_summary.empty else 0
        return BacktestMetrics(
            strategy=strategy.name,
            description=strategy.description,
            rebalance_frequency=self.rebalance_frequency,
            top_n=top_n,
            start_date=observed.index[0].strftime("%Y-%m-%d"),
            end_date=observed.index[-1].strftime("%Y-%m-%d"),
            initial_capital=self.initial_capital,
            final_equity=float(observed.iloc[-1] * self.initial_capital),
            total_return=float(total_return),
            cagr=float(annualized_return),
            annual_volatility=float(annualized_vol),
            sharpe=float(sharpe),
            max_drawdown=float(max_drawdown),
            avg_monthly_turnover=avg_monthly_turnover,
            rebalance_count=rebalance_count,
        )

    def benchmark_equity(self) -> pd.Series:
        benchmark_returns = (
            self.close_prices[self.benchmark_ticker].pct_change(fill_method=None).fillna(0.0)
        )
        return (1.0 + benchmark_returns).cumprod()

    def latest_holdings(self, run: StrategyRun) -> pd.DataFrame:
        latest_date = run.shares_history.index[-1]
        shares = run.shares_history.loc[latest_date]
        prices = self.close_prices.loc[latest_date, self.universe]
        values = (shares * prices).astype(float)
        cash_value = float(run.cash_series.loc[latest_date])
        total_equity = float(values.sum() + cash_value)

        rows: list[dict[str, object]] = []
        for ticker in values[values > 0].sort_values(ascending=False).index:
            market_value = float(values.at[ticker])
            rows.append(
                {
                    "as_of": latest_date.strftime("%Y-%m-%d"),
                    "asset_type": "stock",
                    "ticker": ticker,
                    "shares": int(shares.at[ticker]),
                    "latest_price": float(prices.at[ticker]),
                    "market_value": market_value,
                    "weight": market_value / total_equity if total_equity > 0 else 0.0,
                }
            )
        rows.append(
            {
                "as_of": latest_date.strftime("%Y-%m-%d"),
                "asset_type": "cash",
                "ticker": "CASH",
                "shares": 0,
                "latest_price": 1.0,
                "market_value": cash_value,
                "weight": cash_value / total_equity if total_equity > 0 else 0.0,
            }
        )
        return pd.DataFrame(rows)

    def latest_ranking_snapshot(self, strategy: Strategy, top_n: int) -> tuple[pd.Timestamp, pd.DataFrame]:
        latest_date = self.close_prices.index[-1]
        selected, score = self.select_target_tickers(strategy=strategy, date=latest_date, top_n=top_n)
        frame = pd.DataFrame({"ticker": score.index, "score": score.values})
        if frame.empty:
            return latest_date, frame
        frame["latest_price"] = self.close_prices.loc[latest_date, frame["ticker"]].values
        frame["ret_1m"] = self.ret_21.loc[latest_date, frame["ticker"]].values
        frame["ret_6m"] = self.ret_126.loc[latest_date, frame["ticker"]].values
        frame["ret_12m"] = self.ret_252.loc[latest_date, frame["ticker"]].values
        frame["vol_3m"] = self.vol_63.loc[latest_date, frame["ticker"]].values
        frame["selected"] = frame["ticker"].isin(selected)
        return latest_date, frame.head(max(top_n, 20)).reset_index(drop=True)


def evaluate_strategies(
    backtester: MonthlyBacktester,
    top_n_values: list[int],
    rebalance_frequencies: list[str],
) -> list[StrategyRun]:
    runs: list[StrategyRun] = []
    for rebalance_frequency in rebalance_frequencies:
        backtester.set_rebalance_frequency(rebalance_frequency)
        for top_n in top_n_values:
            for strategy in build_strategy_library():
                runs.append(backtester.run(strategy=strategy, top_n=top_n))
    return runs


def save_result_artifacts(
    backtester: MonthlyBacktester,
    runs: list[StrategyRun],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, StrategyRun]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([run.metrics.__dict__ for run in runs]).sort_values(
        by=["total_return", "cagr"],
        ascending=False,
    )
    summary.to_csv(output_dir / "backtest_summary.csv", index=False)

    top_n_summary = (
        summary.sort_values(
            by=["rebalance_frequency", "top_n", "total_return", "cagr"],
            ascending=[True, True, False, False],
        )
        .drop_duplicates(subset=["rebalance_frequency", "top_n"], keep="first")
        .sort_values(by="total_return", ascending=False)
    )
    top_n_summary.to_csv(output_dir / "top_n_summary.csv", index=False)

    equity_curves = pd.DataFrame(
        {
            f"{run.strategy.name}_{run.rebalance_frequency}_n{run.top_n}": run.equity_curve
            for run in runs
        }
        | {"spy_buy_and_hold": backtester.benchmark_equity()}
    )
    equity_curves.to_csv(output_dir / "equity_curves.csv")

    plt.figure(figsize=(13, 8))
    for column in equity_curves.columns:
        line_width = 2.0 if column == "spy_buy_and_hold" else 1.1
        alpha = 0.95 if column == "spy_buy_and_hold" else 0.65
        plt.plot(equity_curves.index, equity_curves[column], label=column, linewidth=line_width, alpha=alpha)
    plt.yscale("log")
    plt.title("Monthly Rebalancing Equity Curves by Strategy and Top N")
    plt.xlabel("Date")
    plt.ylabel("Equity (log scale)")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "equity_curves.png", dpi=150)
    plt.close()

    best_strategy_name = summary.iloc[0]["strategy"]
    best_frequency = summary.iloc[0]["rebalance_frequency"]
    best_top_n = int(summary.iloc[0]["top_n"])
    best_run = next(
        run
        for run in runs
        if run.strategy.name == best_strategy_name
        and run.rebalance_frequency == best_frequency
        and run.top_n == best_top_n
    )

    current_holdings = backtester.latest_holdings(best_run)
    current_holdings.to_csv(output_dir / "current_model_portfolio.csv", index=False)

    ranking_date, ranking_frame = backtester.latest_ranking_snapshot(best_run.strategy, best_run.top_n)
    ranking_frame.insert(0, "as_of", ranking_date.strftime("%Y-%m-%d"))
    ranking_frame.to_csv(output_dir / "latest_recommendations.csv", index=False)

    best_run.trade_log.to_csv(output_dir / "trade_log.csv", index=False)
    best_run.portfolio_history.to_csv(output_dir / "monthly_portfolio_history.csv", index=False)
    best_run.rebalance_summary.to_csv(output_dir / "rebalance_summary.csv", index=False)

    return summary, top_n_summary, best_run
