from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
import yfinance as yf


WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/135.0.0.0 Safari/537.36"
    )
}


@dataclass(frozen=True)
class UniverseSnapshot:
    as_of: str
    tickers: list[str]
    metadata: pd.DataFrame


@dataclass(frozen=True)
class PriceBundle:
    open_prices: pd.DataFrame
    close_prices: pd.DataFrame


def normalize_ticker(symbol: str) -> str:
    return symbol.strip().upper().replace(".", "-")


def fetch_sp500_snapshot() -> UniverseSnapshot:
    response = requests.get(WIKI_SP500_URL, headers=REQUEST_HEADERS, timeout=30)
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    constituents = tables[0].copy()
    constituents["Symbol"] = constituents["Symbol"].map(normalize_ticker)
    constituents = constituents.drop_duplicates(subset=["Symbol"]).reset_index(drop=True)
    return UniverseSnapshot(
        as_of=pd.Timestamp.utcnow().strftime("%Y-%m-%d"),
        tickers=constituents["Symbol"].tolist(),
        metadata=constituents,
    )


def _extract_field_frame(raw: pd.DataFrame | pd.Series, field: str) -> pd.DataFrame:
    if isinstance(raw, pd.Series):
        return raw.to_frame(name=field)
    if isinstance(raw.columns, pd.MultiIndex):
        if field in raw.columns.get_level_values(0):
            frame = raw[field].copy()
        else:
            raise ValueError(f"{field} field was not returned by yfinance.")
    else:
        frame = raw.copy()
        if field in frame.columns:
            frame = frame[[field]].rename(columns={field: frame.columns[0]})
    frame.columns = [normalize_ticker(str(column)) for column in frame.columns]
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    frame = frame.sort_index()
    return frame


def download_price_history(
    tickers: Iterable[str],
    start: str,
    end: str | None = None,
    chunk_size: int = 100,
) -> PriceBundle:
    ticker_list = [normalize_ticker(ticker) for ticker in tickers]
    chunks = [
        ticker_list[position : position + chunk_size]
        for position in range(0, len(ticker_list), chunk_size)
    ]
    open_frames: list[pd.DataFrame] = []
    close_frames: list[pd.DataFrame] = []
    for chunk in chunks:
        raw = yf.download(
            tickers=chunk,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        open_frame = _extract_field_frame(raw, "Open")
        close_frame = _extract_field_frame(raw, "Close")
        open_frames.append(open_frame)
        close_frames.append(close_frame)
    combined_open = pd.concat(open_frames, axis=1)
    combined_open = combined_open.loc[:, ~combined_open.columns.duplicated()]
    combined_open = combined_open.dropna(axis=1, how="all")
    combined_close = pd.concat(close_frames, axis=1)
    combined_close = combined_close.loc[:, ~combined_close.columns.duplicated()]
    combined_close = combined_close.dropna(axis=1, how="all")
    common_columns = sorted(set(combined_open.columns).intersection(combined_close.columns))
    return PriceBundle(
        open_prices=combined_open[common_columns],
        close_prices=combined_close[common_columns],
    )


def _bundle_to_cache_frame(bundle: PriceBundle) -> pd.DataFrame:
    open_frame = bundle.open_prices.copy()
    close_frame = bundle.close_prices.copy()
    open_frame.columns = [f"OPEN__{column}" for column in open_frame.columns]
    close_frame.columns = [f"CLOSE__{column}" for column in close_frame.columns]
    combined = pd.concat([open_frame, close_frame], axis=1).sort_index()
    return combined


def _cache_frame_to_bundle(frame: pd.DataFrame) -> PriceBundle | None:
    open_columns = [column for column in frame.columns if str(column).startswith("OPEN__")]
    close_columns = [column for column in frame.columns if str(column).startswith("CLOSE__")]
    if not open_columns or not close_columns:
        return None
    open_prices = frame[open_columns].copy()
    open_prices.columns = [str(column).replace("OPEN__", "", 1) for column in open_prices.columns]
    close_prices = frame[close_columns].copy()
    close_prices.columns = [str(column).replace("CLOSE__", "", 1) for column in close_prices.columns]
    common_columns = sorted(set(open_prices.columns).intersection(close_prices.columns))
    return PriceBundle(
        open_prices=open_prices[common_columns].sort_index(),
        close_prices=close_prices[common_columns].sort_index(),
    )


def _bundle_covers_request(bundle: PriceBundle, start: str, end: str | None) -> bool:
    if bundle.close_prices.empty or bundle.open_prices.empty:
        return False
    requested_start = pd.Timestamp(start).tz_localize(None)
    available_start = bundle.close_prices.index.min()
    if pd.isna(available_start) or available_start > requested_start:
        return False
    if end is not None:
        requested_end = pd.Timestamp(end).tz_localize(None)
        available_end = bundle.close_prices.index.max()
        if pd.isna(available_end) or available_end < requested_end:
            return False
    return True


def load_or_refresh_price_cache(
    tickers: Iterable[str],
    start: str,
    end: str | None,
    cache_path: Path,
    force_refresh: bool = False,
) -> PriceBundle:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not force_refresh:
        cached = pd.read_parquet(cache_path)
        bundle = _cache_frame_to_bundle(cached)
        if bundle is not None and _bundle_covers_request(bundle, start=start, end=end):
            return bundle
    bundle = download_price_history(tickers=tickers, start=start, end=end)
    _bundle_to_cache_frame(bundle).to_parquet(cache_path)
    return bundle


def persist_universe_snapshot(snapshot: UniverseSnapshot, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.metadata.to_csv(path, index=False)
