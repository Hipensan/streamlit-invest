from __future__ import annotations

from functools import lru_cache

import exchange_calendars as xcals
import pandas as pd


@lru_cache(maxsize=16)
def _nyse_sessions(first_year: int, last_year: int) -> pd.DatetimeIndex:
    calendar = xcals.get_calendar("XNYS", start=f"{first_year}-01-01", end=f"{last_year}-12-31")
    return calendar.sessions


def rebalance_dates(
    start: pd.Timestamp,
    end: pd.Timestamp,
    frequency: str,
    shift_days: int,
    adjustment: str,
    observed_sessions: pd.DatetimeIndex | None = None,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return signal dates in [start, end] and the corresponding next sessions."""
    if shift_days < 0:
        raise ValueError("리밸런싱을 앞당기는 일수는 0 이상이어야 합니다.")
    methods = {"prior": "pad", "following": "backfill", "nearest": "nearest"}
    if adjustment not in methods:
        raise ValueError(f"Unsupported non-trading day adjustment: {adjustment}")

    # Allow for holidays around both the period end and the shifted target.
    margin = pd.Timedelta(weeks=2)
    shift = pd.Timedelta(days=shift_days)
    first = start - margin
    last = end + shift + margin
    sessions = _nyse_sessions(first.year, (last + margin).year)
    if frequency == "weekly":
        ends = pd.date_range(first, last, freq=pd.offsets.Week(weekday=4))
    else:
        months = {"monthly": 1, "bimonthly": 2, "quarterly": 3, "semiannual": 6, "annual": 12}
        if frequency not in months:
            raise ValueError(f"Unsupported rebalance frequency: {frequency}")
        ends = pd.date_range(first, last, freq=pd.offsets.MonthEnd())
        ends = ends[ends.month % months[frequency] == 0]

    # Anchor to the exchange's final session even when prices end mid-period.
    close_positions = sessions.get_indexer(ends, method="pad")
    period_closes = sessions[close_positions[close_positions >= 0]]
    targets = period_closes - shift
    if observed_sessions is not None:
        observed_range = sessions[(sessions >= observed_sessions[0]) & (sessions <= observed_sessions[-1])]
        sessions = sessions.difference(observed_range.difference(observed_sessions))
    locations = sessions.get_indexer(targets, method=methods[adjustment])
    return [
        (sessions[i], sessions[i + 1])
        for i in sorted(set(locations))
        if 0 <= i < len(sessions) - 1 and start <= sessions[i] <= end
    ]


def calculate_next_rebalance_date(
    latest_market_date: pd.Timestamp,
    rebalance_frequency: str,
    rebalance_shift_days: int,
    non_trading_day_adjustment: str = "prior",
) -> tuple[pd.Timestamp, pd.Timestamp]:
    dates = rebalance_dates(
        latest_market_date - pd.Timedelta(weeks=2),
        latest_market_date + pd.Timedelta(days=400),
        rebalance_frequency.lower(),
        rebalance_shift_days,
        non_trading_day_adjustment,
    )
    return next(pair for pair in dates if pair[1] > latest_market_date)
