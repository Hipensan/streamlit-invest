from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scipy.optimize as sco

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
    total_contributions: float
    total_invested_capital: float
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
        volume_prices: pd.DataFrame | None,
        universe: list[str],
        transaction_cost_bps: float = 7.0,
        benchmark_ticker: str = "SPY",
        eligible_from: dict[str, pd.Timestamp] | None = None,
        initial_capital: float = 10_000.0,
        rebalance_frequency: str = "monthly",
        rebalance_contribution: float = 0.0,
        rebalance_shift_days: int = 30,
        non_trading_day_adjustment: str = "prior",
        allocation_mode_override: str | None = None,
        hybrid_weights: dict[str, float] | None = None,
    ) -> None:
        self.open_prices = open_prices.sort_index().copy()
        self.close_prices = close_prices.sort_index().copy()
        if volume_prices is None:
            self.volume_prices = pd.DataFrame(index=self.close_prices.index, columns=self.close_prices.columns, dtype=float)
        else:
            self.volume_prices = volume_prices.sort_index().reindex(
                index=self.close_prices.index,
                columns=self.close_prices.columns,
            )
        self.volume_prices = self.volume_prices.astype(float)
        common_universe = set(self.open_prices.columns).intersection(self.close_prices.columns)
        self.universe = [ticker for ticker in universe if ticker in common_universe]
        self.transaction_cost = transaction_cost_bps / 10_000.0
        self.benchmark_ticker = benchmark_ticker
        self.initial_capital = float(initial_capital)
        self.rebalance_contribution = float(rebalance_contribution)
        self.rebalance_frequency = rebalance_frequency
        self.rebalance_shift_days = rebalance_shift_days
        self.non_trading_day_adjustment = non_trading_day_adjustment
        self.allocation_mode_override = allocation_mode_override
        self.hybrid_weights = hybrid_weights or {}
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
        self.volume_ma_21 = self.volume_prices.rolling(21).mean()
        self.rel_volume_21 = self.volume_prices / self.volume_ma_21.replace(0.0, np.nan)
        self.dollar_volume_21 = (self.close_prices * self.volume_prices).rolling(21).mean()
        self.high_126 = self.close_prices.rolling(126).max()
        self.drawdown_126 = self.close_prices / self.high_126 - 1.0
        self.high_252 = self.close_prices.rolling(252).max()
        # RSI 14일 연산 매트릭스 사전 계산 (Wilder's 방식)
        delta = self.close_prices.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1.0/14.0, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0/14.0, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        self.rsi_14 = 100.0 - (100.0 / (1.0 + rs))
        self.rsi_14 = self.rsi_14.fillna(50.0)

        self.rebalance_schedule: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        self.average_costs: dict[str, float] = {}
        self.usd_krw_rate = 1300.0
        self.annual_realized_pnl = 0.0
        self.current_year = None
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
            original_dates = pd.Index(grouped)
        elif frequency == "monthly":
            grouped = self.close_prices.groupby(self.close_prices.index.to_period("M")).tail(1).index
            original_dates = pd.Index(grouped)
        elif frequency == "bimonthly":
            grouped = self.close_prices.groupby(self.close_prices.index.to_period("M")).tail(1)
            month_ends = grouped.index
            month_numbers = month_ends.to_period("M").month
            original_dates = pd.Index(month_ends[month_numbers % 2 == 0])
        elif frequency == "quarterly":
            grouped = self.close_prices.groupby(self.close_prices.index.to_period("Q")).tail(1).index
            original_dates = pd.Index(grouped)
        elif frequency == "semiannual":
            grouped = self.close_prices.groupby(self.close_prices.index.to_period("Q")).tail(1)
            quarter_ends = grouped.index
            quarter_numbers = quarter_ends.to_period("Q").quarter
            original_dates = pd.Index(quarter_ends[quarter_numbers.isin([2, 4])])
        elif frequency == "annual":
            grouped = self.close_prices.groupby(self.close_prices.index.to_period("Y")).tail(1).index
            original_dates = pd.Index(grouped)
        else:
            raise ValueError(f"Unsupported rebalance frequency: {self.rebalance_frequency}")

        # 미완성 기간(주, 월, 분기, 연)의 임시 말일 제거 로직
        if not original_dates.empty:
            last_orig = original_dates[-1]
            max_data_date = self.close_prices.index[-1]
            if frequency == "weekly":
                period_end = last_orig + pd.offsets.Week(0, weekday=4)
            elif frequency in ["monthly", "bimonthly"]:
                period_end = last_orig + pd.offsets.MonthEnd(0)
            elif frequency in ["quarterly", "semiannual"]:
                period_end = last_orig + pd.offsets.QuarterEnd(0)
            elif frequency == "annual":
                period_end = last_orig + pd.offsets.YearEnd(0)
            else:
                period_end = last_orig

            if period_end > max_data_date:
                original_dates = original_dates[:-1]

        if self.rebalance_shift_days == 0:
            return original_dates

        adjusted_dates = []
        trading_days = self.close_prices.index
        for d in original_dates:
            target = d - pd.Timedelta(days=self.rebalance_shift_days)
            if target in trading_days:
                adjusted_dates.append(target)
            else:
                if self.non_trading_day_adjustment == "prior":
                    prior_days = trading_days[trading_days < target]
                    if not prior_days.empty:
                        adjusted_dates.append(prior_days[-1])
                    else:
                        following_days = trading_days[trading_days > target]
                        if not following_days.empty:
                            adjusted_dates.append(following_days[0])
                elif self.non_trading_day_adjustment == "following":
                    following_days = trading_days[trading_days > target]
                    if not following_days.empty:
                        adjusted_dates.append(following_days[0])
                    else:
                        prior_days = trading_days[trading_days < target]
                        if not prior_days.empty:
                            adjusted_dates.append(prior_days[-1])
                elif self.non_trading_day_adjustment == "nearest":
                    idx = trading_days.get_indexer([target], method="nearest")[0]
                    adjusted_dates.append(trading_days[idx])
                else:
                    adjusted_dates.append(target)

        adjusted_dates = sorted(list(set(adjusted_dates)))
        return pd.Index(adjusted_dates)

    def _benchmark_uptrend(self, date: pd.Timestamp) -> bool:
        benchmark_price = self.close_prices.at[date, self.benchmark_ticker]
        benchmark_sma = self.sma_200.at[date, self.benchmark_ticker]
        return pd.notna(benchmark_price) and pd.notna(benchmark_sma) and benchmark_price > benchmark_sma

    def score_strategy(self, strategy_name: str, date: pd.Timestamp) -> pd.Series:
        universe = self.universe
        if strategy_name == "hybrid_strategy":
            combined_rank = pd.Series(0.0, index=universe)
            total_weight = 0.0
            for sub_name, weight in self.hybrid_weights.items():
                if weight <= 0:
                    continue
                sub_score = self.score_strategy(sub_name, date)
                if sub_score.empty:
                    continue
                sub_rank = sub_score.rank(pct=True)
                combined_rank = combined_rank.add(sub_rank * weight, fill_value=0.0)
                total_weight += weight
            if total_weight > 0:
                return combined_rank / total_weight
            return pd.Series(0.0, index=universe)
        ret_21 = self.ret_21.loc[date, universe]
        ret_63 = self.ret_63.loc[date, universe]
        ret_126 = self.ret_126.loc[date, universe]
        ret_252 = self.ret_252.loc[date, universe]
        vol_63 = self.vol_63.loc[date, universe]
        rel_volume_21 = self.rel_volume_21.loc[date, universe]
        dollar_volume_21 = self.dollar_volume_21.loc[date, universe]
        drawdown_126 = self.drawdown_126.loc[date, universe]
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
        if strategy_name == "low_vol_momentum":
            score = ret_126 - (0.75 * vol_63)
            return score.where(eligibility_mask)
        if strategy_name == "breakout_52w":
            score = (0.60 * ret_126) + (0.40 * breakout.rank(pct=True))
            return score.where(above_sma & eligibility_mask)
        if strategy_name == "drawdown_filtered_momentum":
            score = (0.55 * ret_252) + (0.45 * ret_126) - (0.20 * vol_63)
            score = score.where((drawdown_126 > -0.15) & above_sma & eligibility_mask)
            if not self._benchmark_uptrend(date):
                return pd.Series(dtype=float)
            return score
        if strategy_name == "low_volatility":
            return (-vol_63).where(eligibility_mask)
        if strategy_name == "mean_reversion_rsi":
            rsi_val = self.rsi_14.loc[date, universe]
            return (-rsi_val).where(eligibility_mask)
        if strategy_name == "high_liquidity":
            return dollar_volume_21.where(eligibility_mask)
        if strategy_name == "min_drawdown":
            return drawdown_126.where(eligibility_mask)
        raise ValueError(f"Unknown strategy: {strategy_name}")

    def select_target_tickers(
        self,
        strategy: Strategy,
        date: pd.Timestamp,
        top_n: int,
    ) -> tuple[list[str], pd.Series]:
        score = self.score_strategy(strategy.name, date)
        score = score.replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False)
        if strategy.name in {"drawdown_filtered_momentum"}:
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

    def _build_target_weights(
        self,
        strategy: Strategy,
        selected: list[str],
        signal_date: pd.Timestamp,
    ) -> pd.Series:
        weights = pd.Series(dtype=float)
        if not selected:
            return weights

        mode = self.allocation_mode_override or strategy.allocation_mode
        if mode == "inverse_volatility":
            volatility = self.vol_63.loc[signal_date, selected].replace(0.0, np.nan)
            inverse_vol = 1.0 / volatility
            inverse_vol = inverse_vol.replace([np.inf, -np.inf], np.nan).dropna()
            if not inverse_vol.empty and float(inverse_vol.sum()) > 0:
                weights = inverse_vol / inverse_vol.sum()
        elif mode == "risk_parity":
            weights = self._calculate_risk_parity_weights(selected, signal_date)
        elif mode == "mean_variance":
            weights = self._calculate_mvo_weights(selected, signal_date)

        if weights.empty:
            weights = pd.Series(1.0 / len(selected), index=selected, dtype=float)
        return weights.sort_values(ascending=False)

    def _calculate_risk_parity_weights(self, selected: list[str], signal_date: pd.Timestamp) -> pd.Series:
        n = len(selected)
        if n == 0:
            return pd.Series(dtype=float)

        recent_returns = self.daily_returns.loc[:signal_date, selected].tail(63)
        if len(recent_returns) < 5 or recent_returns.isnull().all().all():
            return pd.Series(1.0 / n, index=selected, dtype=float)

        cov = recent_returns.cov().fillna(0.0).values * 252
        if np.all(np.diag(cov) == 0):
            return pd.Series(1.0 / n, index=selected, dtype=float)

        def objective(w):
            port_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
            if port_vol == 0:
                return 0
            rc = w * np.dot(cov, w) / port_vol
            diff = rc[:, np.newaxis] - rc[np.newaxis, :]
            return np.sum(diff ** 2)

        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        bounds = [(0.0, 1.0) for _ in range(n)]
        init_w = np.ones(n) / n

        res = sco.minimize(objective, init_w, method='SLSQP', bounds=bounds, constraints=cons)
        if res.success:
            return pd.Series(res.x, index=selected, dtype=float)
        return pd.Series(init_w, index=selected, dtype=float)

    def _calculate_mvo_weights(self, selected: list[str], signal_date: pd.Timestamp) -> pd.Series:
        n = len(selected)
        if n == 0:
            return pd.Series(dtype=float)

        recent_returns = self.daily_returns.loc[:signal_date, selected].tail(126)
        if len(recent_returns) < 5 or recent_returns.isnull().all().all():
            return pd.Series(1.0 / n, index=selected, dtype=float)

        expected_returns = recent_returns.mean().fillna(0.0).values * 252
        cov = recent_returns.cov().fillna(0.0).values * 252

        if np.all(np.diag(cov) == 0):
            return pd.Series(1.0 / n, index=selected, dtype=float)

        rf = 0.02
        def objective(w):
            port_return = np.dot(w, expected_returns)
            port_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
            if port_vol == 0:
                return 0
            return - (port_return - rf) / port_vol

        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        max_bound = max(1.0 / n, 0.4)
        bounds = [(0.0, max_bound) for _ in range(n)]
        init_w = np.ones(n) / n

        res = sco.minimize(objective, init_w, method='SLSQP', bounds=bounds, constraints=cons)
        if res.success:
            return pd.Series(res.x, index=selected, dtype=float)
        return pd.Series(init_w, index=selected, dtype=float)

    def _build_target_shares(
        self,
        selected: list[str],
        target_weights: pd.Series,
        price_row: pd.Series,
        total_equity: float,
        current_shares: pd.Series,
        current_cash: float,
    ) -> tuple[pd.Series, pd.Series, float, pd.Series, pd.Series, float, float, float]:
        target_shares = pd.Series(0, index=self.universe, dtype=int)
        selected_prices = price_row.reindex(selected).dropna()
        selected_prices = selected_prices[selected_prices > 0]
        if selected_prices.empty or total_equity <= 0:
            cash_after, buy_shares, sell_shares, buy_notional, sell_notional, fees = (
                self._evaluate_cash_after_trade(current_shares, current_cash, target_shares, price_row)
            )
            return (
                target_shares,
                pd.Series(dtype=float),
                cash_after,
                buy_shares,
                sell_shares,
                buy_notional,
                sell_notional,
                fees,
            )

        normalized_weights = target_weights.reindex(selected_prices.index).fillna(0.0)
        if float(normalized_weights.sum()) <= 0:
            normalized_weights = pd.Series(1.0 / len(selected_prices), index=selected_prices.index, dtype=float)
        else:
            normalized_weights = normalized_weights / normalized_weights.sum()
        desired_values = total_equity * normalized_weights
        seed_shares = np.floor(desired_values / selected_prices).astype(int)
        target_shares.loc[selected_prices.index] = seed_shares

        while True:
            current_values = target_shares.loc[selected_prices.index] * selected_prices
            affordable_candidates = []
            shortfall = (desired_values - current_values).sort_values(ascending=False)
            for ticker in shortfall.index:
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
            invested_values = target_shares.loc[invested] * price_row.loc[invested]
            overweight = (invested_values - desired_values.reindex(invested).fillna(0.0)).sort_values(ascending=False)
            ticker_to_reduce = overweight.index[0]
            target_shares.at[ticker_to_reduce] -= 1
            cash_after, buy_shares, sell_shares, buy_notional, sell_notional, fees = (
                self._evaluate_cash_after_trade(current_shares, current_cash, target_shares, price_row)
            )

        return (
            target_shares,
            normalized_weights,
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
        cash_contribution: float = 0.0,
    ) -> tuple[pd.Series, float, list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
        if self.current_year is None or signal_date.year != self.current_year:
            self.annual_realized_pnl = 0.0
            self.current_year = signal_date.year

        is_last_of_year = True
        for next_sig, next_eff in self.rebalance_schedule[rebalance_no:]:
            if next_sig.year == signal_date.year:
                is_last_of_year = False
                break

        execution_price_row = self.trade_prices.loc[effective_date].fillna(0.0)
        close_price_row = self.close_prices.loc[effective_date, self.universe].fillna(0.0)
        portfolio_value_before_contribution = float((current_shares * execution_price_row).sum() + current_cash)
        current_cash = float(current_cash + cash_contribution)
        portfolio_value_before = float((current_shares * execution_price_row).sum() + current_cash)
        target_weights = self._build_target_weights(strategy=strategy, selected=selected, signal_date=signal_date)

        # 연도 최종 거래일 세금 가예측 (Pass 1)
        estimated_tax = 0.0
        if is_last_of_year:
            temp_target_shares, _, _, _, _, _, _, _ = self._build_target_shares(
                selected=selected,
                target_weights=target_weights,
                price_row=execution_price_row,
                total_equity=portfolio_value_before,
                current_shares=current_shares,
                current_cash=current_cash,
            )
            # 임시 매도로 인한 실현손익 계산
            temp_realized_pnl = 0.0
            temp_share_delta = temp_target_shares - current_shares
            for ticker in temp_share_delta.index:
                delta = int(temp_share_delta.at[ticker])
                if delta < 0:
                    shares = abs(delta)
                    price = float(execution_price_row.at[ticker])
                    notional = shares * price
                    prev_cost = float(self.average_costs.get(ticker, 0.0))
                    if prev_cost > 0:
                        temp_realized_pnl += (notional - (shares * prev_cost))
            
            estimated_annual_pnl = self.annual_realized_pnl + temp_realized_pnl
            tax_free_limit = 2000.0
            if estimated_annual_pnl > tax_free_limit:
                estimated_tax = (estimated_annual_pnl - tax_free_limit) * 0.22

        # 세금 선공제 조정 (Pass 2)
        tax = estimated_tax
        equity_for_allocation = portfolio_value_before - tax
        current_cash_for_allocation = current_cash - tax

        (
            target_shares,
            target_weights,
            cash_after,
            buy_shares,
            sell_shares,
            buy_notional,
            sell_notional,
            fees,
        ) = self._build_target_shares(
            selected=selected,
            target_weights=target_weights,
            price_row=execution_price_row,
            total_equity=equity_for_allocation,
            current_shares=current_shares,
            current_cash=current_cash_for_allocation,
        )
        traded_notional = buy_notional + sell_notional
        turnover = traded_notional / portfolio_value_before if portfolio_value_before > 0 else 0.0
        skipped_by_turnover = False
        if traded_notional > 0 and turnover < strategy.min_turnover:
            skipped_by_turnover = True
            target_shares = current_shares.copy()
            cash_after = float(current_cash)
            buy_shares = pd.Series(0, index=self.universe, dtype=int)
            sell_shares = pd.Series(0, index=self.universe, dtype=int)
            buy_notional = 0.0
            sell_notional = 0.0
            fees = 0.0
            traded_notional = 0.0
            turnover = 0.0
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

            # 평단가 및 PnL 계산
            purchase_price = np.nan
            realized_pnl = np.nan
            realized_pnl_pct = np.nan

            pre_shares = int(current_shares.at[ticker])
            post_shares = int(target_shares.at[ticker])

            if delta < 0:  # 매도 (SELL / TRIM)
                prev_cost = float(self.average_costs.get(ticker, 0.0))
                if prev_cost > 0:
                    purchase_price = prev_cost
                    purchase_notional = shares * prev_cost
                    realized_pnl = notional - purchase_notional
                    realized_pnl_pct = realized_pnl / purchase_notional if purchase_notional > 0 else 0.0
                    self.annual_realized_pnl += realized_pnl

                if post_shares == 0:
                    self.average_costs[ticker] = 0.0
            else:  # 매수 (BUY / ADD)
                prev_cost = float(self.average_costs.get(ticker, 0.0))
                if pre_shares + shares > 0:
                    new_cost = (pre_shares * prev_cost + shares * price) / (pre_shares + shares)
                    self.average_costs[ticker] = new_cost
                else:
                    self.average_costs[ticker] = price

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
                    "allocation_mode": strategy.allocation_mode,
                    "pre_shares": pre_shares,
                    "post_shares": post_shares,
                    "purchase_price": purchase_price,
                    "realized_pnl": realized_pnl,
                    "realized_pnl_pct": realized_pnl_pct,
                    "rank_on_signal": ranked_lookup.get(ticker),
                }
            )

        holdings_value = (target_shares * close_price_row).astype(float)
        total_equity = float(holdings_value.sum() + cash_after)
        
        # 연간 세금 결산 후 실현손익 리셋
        if is_last_of_year:
            self.annual_realized_pnl = 0.0

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
                    "target_weight": float(target_weights.get(ticker, 0.0)),
                    "allocation_mode": strategy.allocation_mode,
                    "cash_contribution": cash_contribution,
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
                "allocation_mode": strategy.allocation_mode,
                "cash_contribution": cash_contribution,
                "cash_after": cash_after,
                "total_equity": total_equity,
                "rank_on_signal": np.nan,
            }
        )

        rebalance_row = {
            "strategy": strategy.name,
            "allocation_mode": strategy.allocation_mode,
            "rebalance_frequency": self.rebalance_frequency,
            "top_n": top_n,
            "rebalance_no": rebalance_no,
            "signal_date": signal_date.strftime("%Y-%m-%d"),
            "effective_date": effective_date.strftime("%Y-%m-%d"),
            "selected_count": len(selected),
            "cash_contribution": cash_contribution,
            "portfolio_value_before_contribution": portfolio_value_before_contribution,
            "portfolio_value_before": portfolio_value_before,
            "portfolio_value_after": portfolio_value_after,
            "buy_notional": buy_notional,
            "sell_notional": sell_notional,
            "fees": fees,
            "traded_notional": traded_notional,
            "turnover": turnover,
            "target_turnover_threshold": strategy.min_turnover,
            "skipped_by_turnover": skipped_by_turnover,
            "cash_after": cash_after,
            "tax": tax,
        }
        return target_shares, cash_after, trade_rows, portfolio_rows, rebalance_row

    def _execute_schedule(
        self,
        strategy: Strategy,
        top_n: int,
        schedule: list[tuple[pd.Timestamp, pd.Timestamp]],
        apply_contributions: bool = True,
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict[pd.Timestamp, pd.Series], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        index = self.close_prices.index
        shares_history = pd.DataFrame(0, index=index, columns=self.universe, dtype=int)
        cash_series = pd.Series(index=index, dtype=float)
        contribution_series = pd.Series(0.0, index=index, dtype=float)
        rankings: dict[pd.Timestamp, pd.Series] = {}
        trade_rows: list[dict[str, object]] = []
        portfolio_rows: list[dict[str, object]] = []
        rebalance_rows: list[dict[str, object]] = []

        current_shares = pd.Series(0, index=self.universe, dtype=int)
        current_cash = self.initial_capital
        schedule_lookup = {effective_date: signal_date for signal_date, effective_date in schedule}
        rebalance_no = 0

        for date in index:
            signal_date = schedule_lookup.get(date)
            if signal_date is not None:
                rebalance_no += 1
                cash_contribution = self.rebalance_contribution if apply_contributions else 0.0
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
                    cash_contribution=cash_contribution,
                )
                contribution_series.loc[date] = cash_contribution
                trade_rows.extend(new_trade_rows)
                portfolio_rows.extend(new_portfolio_rows)
                rebalance_rows.append(rebalance_row)

            shares_history.loc[date] = current_shares.values
            cash_series.loc[date] = current_cash

        return (
            shares_history,
            cash_series,
            contribution_series,
            rankings,
            pd.DataFrame(trade_rows),
            pd.DataFrame(portfolio_rows),
            pd.DataFrame(rebalance_rows),
        )

    def run(self, strategy: Strategy, top_n: int) -> StrategyRun:
        self.average_costs = {ticker: 0.0 for ticker in self.universe}
        self.annual_realized_pnl = 0.0
        self.current_year = None
        (
            shares_history,
            cash_series,
            contribution_series,
            rankings,
            trade_log,
            portfolio_history,
            rebalance_summary,
        ) = self._execute_schedule(strategy=strategy, top_n=top_n, schedule=self.rebalance_schedule)
        holdings_value_close = shares_history.astype(float) * self.close_prices[self.universe]
        total_equity = holdings_value_close.sum(axis=1) + cash_series
        daily_returns = self._cash_flow_adjusted_returns(total_equity, contribution_series)
        equity_curve = (1.0 + daily_returns).cumprod()
        weight_history = holdings_value_close.div(total_equity.replace(0.0, np.nan), axis=0).fillna(0.0)
        metrics = self._build_metrics(
            strategy=strategy,
            top_n=top_n,
            daily_returns=daily_returns,
            equity_curve=equity_curve,
            rebalance_summary=rebalance_summary,
            final_equity=float(total_equity.iloc[-1]),
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



    def _cash_flow_adjusted_returns(self, total_equity: pd.Series, contribution_series: pd.Series) -> pd.Series:
        previous_equity = total_equity.shift(1)
        adjusted_returns = (total_equity - contribution_series).div(previous_equity.replace(0.0, np.nan)) - 1.0
        adjusted_returns.iloc[0] = 0.0
        return adjusted_returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _build_metrics(
        self,
        strategy: Strategy,
        top_n: int,
        daily_returns: pd.Series,
        equity_curve: pd.Series,
        rebalance_summary: pd.DataFrame,
        final_equity: float,
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
        total_contributions = (
            float(rebalance_summary["cash_contribution"].sum())
            if not rebalance_summary.empty and "cash_contribution" in rebalance_summary.columns
            else 0.0
        )
        return BacktestMetrics(
            strategy=strategy.name,
            description=strategy.description,
            rebalance_frequency=self.rebalance_frequency,
            top_n=top_n,
            start_date=observed.index[0].strftime("%Y-%m-%d"),
            end_date=observed.index[-1].strftime("%Y-%m-%d"),
            initial_capital=self.initial_capital,
            total_contributions=total_contributions,
            total_invested_capital=self.initial_capital + total_contributions,
            final_equity=final_equity,
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
        frame["ret_3m"] = self.ret_63.loc[latest_date, frame["ticker"]].values
        frame["ret_6m"] = self.ret_126.loc[latest_date, frame["ticker"]].values
        frame["ret_12m"] = self.ret_252.loc[latest_date, frame["ticker"]].values
        frame["vol_3m"] = self.vol_63.loc[latest_date, frame["ticker"]].values
        frame["rel_volume_1m"] = self.rel_volume_21.loc[latest_date, frame["ticker"]].values
        frame["drawdown_6m"] = self.drawdown_126.loc[latest_date, frame["ticker"]].values
        frame["selected"] = frame["ticker"].isin(selected)
        return latest_date, frame.head(max(top_n, 20)).reset_index(drop=True)

    def get_next_scheduled_rebalance(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        latest_market_date = self.close_prices.index[-1]
        return calculate_next_rebalance_date(
            latest_market_date=latest_market_date,
            rebalance_frequency=self.rebalance_frequency,
            rebalance_shift_days=self.rebalance_shift_days,
            non_trading_day_adjustment=self.non_trading_day_adjustment,
        )




def evaluate_strategies(
    backtester: MonthlyBacktester,
    top_n_values: list[int],
    rebalance_frequencies: list[str],
    strategy_names: list[str] | None = None,
) -> list[StrategyRun]:
    runs: list[StrategyRun] = []
    strategies = build_strategy_library()
    if backtester.hybrid_weights:
        hybrid_desc = "Blended: " + ", ".join(f"{k}({v:.1f})" for k, v in backtester.hybrid_weights.items())
        strategies.append(Strategy(name="hybrid_strategy", description=hybrid_desc))

    if strategy_names is not None:
        selected_names = list(dict.fromkeys(strategy_names))
        strategy_lookup = {strategy.name: strategy for strategy in strategies}
        unsupported = [name for name in selected_names if name not in strategy_lookup]
        if unsupported:
            raise ValueError(f"Unsupported strategies: {', '.join(unsupported)}")
        strategies = [strategy_lookup[name] for name in selected_names]
    if not strategies:
        raise ValueError("At least one strategy must be selected.")
    for rebalance_frequency in rebalance_frequencies:
        backtester.set_rebalance_frequency(rebalance_frequency)
        for top_n in top_n_values:
            for strategy in strategies:
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


def calculate_next_rebalance_date(
    latest_market_date: pd.Timestamp,
    rebalance_frequency: str,
    rebalance_shift_days: int,
    non_trading_day_adjustment: str = "prior",
) -> tuple[pd.Timestamp, pd.Timestamp]:
    future_index = pd.bdate_range(
        start=latest_market_date - pd.Timedelta(days=45),
        end=latest_market_date + pd.Timedelta(days=450)
    )
    temp_close = pd.DataFrame(index=future_index)
    
    frequency = rebalance_frequency.lower()
    if frequency == "weekly":
        grouped = temp_close.groupby(pd.Grouper(freq="W-FRI")).tail(1).index
        original_dates = pd.Index(grouped)
    elif frequency == "monthly":
        grouped = temp_close.groupby(temp_close.index.to_period("M")).tail(1).index
        original_dates = pd.Index(grouped)
    elif frequency == "bimonthly":
        grouped = temp_close.groupby(temp_close.index.to_period("M")).tail(1)
        month_ends = grouped.index
        month_numbers = month_ends.to_period("M").month
        original_dates = pd.Index(month_ends[month_numbers % 2 == 0])
    elif frequency == "quarterly":
        grouped = temp_close.groupby(temp_close.index.to_period("Q")).tail(1).index
        original_dates = pd.Index(grouped)
    elif frequency == "semiannual":
        grouped = temp_close.groupby(temp_close.index.to_period("Q")).tail(1)
        quarter_ends = grouped.index
        quarter_numbers = quarter_ends.to_period("Q").quarter
        original_dates = pd.Index(quarter_ends[quarter_numbers.isin([2, 4])])
    elif frequency == "annual":
        grouped = temp_close.groupby(temp_close.index.to_period("Y")).tail(1).index
        original_dates = pd.Index(grouped)
    else:
        original_dates = pd.Index([latest_market_date])

    if rebalance_shift_days == 0:
        adjusted_dates = original_dates
    else:
        adjusted_dates = []
        trading_days = temp_close.index
        for d in original_dates:
            target = d - pd.Timedelta(days=rebalance_shift_days)
            if target in trading_days:
                adjusted_dates.append(target)
            else:
                if non_trading_day_adjustment == "prior":
                    prior_days = trading_days[trading_days < target]
                    if not prior_days.empty:
                        adjusted_dates.append(prior_days[-1])
                    else:
                        following_days = trading_days[trading_days > target]
                        if not following_days.empty:
                            adjusted_dates.append(following_days[0])
                elif non_trading_day_adjustment == "following":
                    following_days = trading_days[trading_days > target]
                    if not following_days.empty:
                        adjusted_dates.append(following_days[0])
                    else:
                        prior_days = trading_days[trading_days < target]
                        if not prior_days.empty:
                            adjusted_dates.append(prior_days[-1])
                elif non_trading_day_adjustment == "nearest":
                    idx = trading_days.get_indexer([target], method="nearest")[0]
                    adjusted_dates.append(trading_days[idx])
                else:
                    adjusted_dates.append(target)

        adjusted_dates = sorted(list(set(adjusted_dates)))

    next_signal = None
    for sd in adjusted_dates:
        if sd > latest_market_date:
            next_signal = sd
            break

    if next_signal is None:
        next_signal = latest_market_date + pd.Timedelta(days=30)

    trading_days = temp_close.index
    idx = trading_days.get_loc(next_signal)
    if idx + 1 < len(trading_days):
        next_effective = trading_days[idx + 1]
    else:
        next_effective = next_signal + pd.Timedelta(days=1)

    return next_signal, next_effective
