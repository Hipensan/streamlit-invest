from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Strategy:
    name: str
    description: str


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
            name="trend_filtered_momentum",
            description="Hold top N 12-month momentum names only when SPY and stock are above 200DMA",
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
            name="blend_momentum",
            description="Top N by blended 6m/12m momentum and lower volatility",
        ),
    ]
