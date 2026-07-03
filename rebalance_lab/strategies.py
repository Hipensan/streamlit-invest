from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Strategy:
    name: str
    description: str
    allocation_mode: str = "equal_weight"
    min_turnover: float = 0.0


def equal_weight(selection: pd.Index | list[str], all_tickers: list[str]) -> pd.Series:
    weights = pd.Series(0.0, index=all_tickers, dtype=float)
    if len(selection) == 0:
        return weights
    weights.loc[list(selection)] = 1.0 / len(selection)
    return weights


def sanitize_score(score: pd.Series) -> pd.Series:
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    return score.sort_values(ascending=False)


def momentum_top_n(
    score: pd.Series,
    tickers: list[str],
    top_n: int,
    positive_only: bool = False,
) -> pd.Series:
    score = sanitize_score(score)
    if positive_only:
        score = score[score > 0]
    selection = score.head(top_n).index
    return equal_weight(selection, tickers)


def build_strategy_library() -> list[Strategy]:
    return [
        Strategy(
            name="momentum_3m",
            description="Top N stocks by trailing 3-month return",
        ),
        Strategy(
            name="momentum_6m",
            description="Top N stocks by trailing 6-month return",
        ),
        Strategy(
            name="momentum_12m",
            description="Top N stocks by trailing 12-month return",
        ),
        Strategy(
            name="momentum_12m_skip_1m",
            description="Top N stocks by 12-month return excluding the latest 1-month move",
        ),
        Strategy(
            name="risk_adjusted_momentum",
            description="Top N by 12-month return divided by 3-month volatility",
        ),
        Strategy(
            name="low_vol_momentum",
            description="Top N by 6-month return penalized by 3-month volatility",
        ),
        Strategy(
            name="breakout_52w",
            description="Top N by proximity to 52-week highs blended with 6-month momentum",
        ),
        Strategy(
            name="drawdown_filtered_momentum",
            description="Top N by blended momentum while avoiding names in deeper 6-month drawdowns",
            allocation_mode="inverse_volatility",
            min_turnover=0.04,
        ),
        Strategy(
            name="low_volatility",
            description="Top N stocks by trailing 3-month volatility (lowest vol first)",
        ),
        Strategy(
            name="mean_reversion_rsi",
            description="Top N stocks by trailing 14-day RSI (most oversold first)",
        ),
        Strategy(
            name="high_liquidity",
            description="Top N stocks by trailing 1-month average daily trading volume (highest value first)",
        ),
        Strategy(
            name="min_drawdown",
            description="Top N stocks by trailing 6-month maximum drawdown (smallest drawdown first)",
        ),
    ]
