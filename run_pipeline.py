from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rebalance_lab.backtest import CONTROL_TICKERS, MonthlyBacktester, evaluate_strategies, save_result_artifacts
from rebalance_lab.data import fetch_sp500_snapshot, load_or_refresh_price_cache, persist_universe_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare monthly stock rebalancing strategies and generate latest recommendations."
    )
    parser.add_argument("--start", default="2014-01-01", help="Backtest start date, YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="Optional end date, YYYY-MM-DD")
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Run only a single top-N value instead of a sweep",
    )
    parser.add_argument(
        "--top-n-values",
        default="5,10,15,20,25,30",
        help="Comma-separated top-N values to compare",
    )
    parser.add_argument(
        "--rebalance-frequency",
        default=None,
        help="Run a single rebalance frequency: weekly, monthly, bimonthly, quarterly, semiannual, annual",
    )
    parser.add_argument(
        "--rebalance-frequencies",
        default="monthly",
        help="Comma-separated rebalance frequencies to compare",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=1_000_000.0,
        help="Initial capital for the integer-share simulation",
    )
    parser.add_argument(
        "--transaction-cost-bps",
        type=float,
        default=10.0,
        help="One-way cost applied on traded notional in basis points",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore cached price data and download again",
    )
    return parser.parse_args()


def parse_top_n_values(args: argparse.Namespace) -> list[int]:
    if args.top_n is not None:
        return [args.top_n]
    values = []
    for raw in args.top_n_values.split(","):
        raw = raw.strip()
        if raw:
            values.append(int(raw))
    unique_values = sorted(set(values))
    if not unique_values:
        raise ValueError("At least one top-N value must be provided.")
    return unique_values


def parse_rebalance_frequencies(args: argparse.Namespace) -> list[str]:
    supported = ["weekly", "monthly", "bimonthly", "quarterly", "semiannual", "annual"]
    if args.rebalance_frequency is not None:
        values = [args.rebalance_frequency.strip().lower()]
    else:
        values = [value.strip().lower() for value in args.rebalance_frequencies.split(",") if value.strip()]
    unique_values = []
    for value in values:
        if value not in supported:
            raise ValueError(f"Unsupported rebalance frequency: {value}")
        if value not in unique_values:
            unique_values.append(value)
    if not unique_values:
        raise ValueError("At least one rebalance frequency must be provided.")
    return unique_values


def main() -> None:
    args = parse_args()
    top_n_values = parse_top_n_values(args)
    rebalance_frequencies = parse_rebalance_frequencies(args)
    project_root = Path(__file__).resolve().parent
    output_dir = project_root / "outputs"
    cache_dir = output_dir / "cache"

    universe_snapshot = fetch_sp500_snapshot()
    persist_universe_snapshot(universe_snapshot, output_dir / "sp500_snapshot.csv")

    tickers = sorted(set(universe_snapshot.tickers + CONTROL_TICKERS))
    price_bundle = load_or_refresh_price_cache(
        tickers=tickers,
        start=args.start,
        end=args.end,
        cache_path=cache_dir / "price_history.parquet",
        force_refresh=args.force_refresh,
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
        transaction_cost_bps=args.transaction_cost_bps,
        eligible_from=eligible_from,
        initial_capital=args.initial_capital,
        rebalance_frequency=rebalance_frequencies[0],
    )
    runs = evaluate_strategies(
        backtester,
        top_n_values=top_n_values,
        rebalance_frequencies=rebalance_frequencies,
    )
    summary, top_n_summary, best_run = save_result_artifacts(backtester, runs, output_dir)

    latest_market_date = close_prices.index[-1].strftime("%Y-%m-%d")
    best_row = summary.iloc[0]
    current_holdings = pd.read_csv(output_dir / "current_model_portfolio.csv")
    latest_recommendations = pd.read_csv(output_dir / "latest_recommendations.csv")
    write_report(
        output_dir=output_dir,
        latest_market_date=latest_market_date,
        universe_size=len(available_universe),
        best_row=best_row,
        top_n_summary=top_n_summary,
        rebalance_frequencies=rebalance_frequencies,
        current_holdings=current_holdings,
        latest_recommendations=latest_recommendations,
    )

    print(f"Latest market data: {latest_market_date}")
    print(f"Universe size used: {len(available_universe)} stocks")
    print(f"Top-N values tested: {', '.join(str(value) for value in top_n_values)}")
    print(f"Rebalance frequencies tested: {', '.join(rebalance_frequencies)}")
    print(f"Best strategy: {best_row['strategy']} ({best_row['description']})")
    print(f"Best Frequency: {best_row['rebalance_frequency']}")
    print(f"Best Top N: {int(best_row['top_n'])}")
    print("Execution assumption: signal on period-end close, trade on next trading day open")
    print(f"Initial capital: {best_row['initial_capital']:.2f}")
    print(f"Total return: {best_row['total_return']:.2%}")
    print(f"CAGR: {best_row['cagr']:.2%}")
    print(f"Max drawdown: {best_row['max_drawdown']:.2%}")
    print()
    print("Best result by Top N:")
    for row in top_n_summary.itertuples(index=False):
        print(
            f"  freq={row.rebalance_frequency}, N={row.top_n}: strategy={row.strategy}, "
            f"total_return={row.total_return:.2%}, CAGR={row.cagr:.2%}"
        )
    print()
    print("Current model portfolio:")
    if current_holdings.empty:
        print("  No active holdings.")
    else:
        for row in current_holdings.itertuples(index=False):
            if row.ticker == "CASH":
                print(f"  CASH: value={row.market_value:.2f}, weight={row.weight:.2%}")
            else:
                print(
                    f"  {row.ticker}: shares={int(row.shares)}, weight={row.weight:.2%}, "
                    f"latest_price={row.latest_price:.2f}, value={row.market_value:.2f}"
                )
    print()
    print("Latest ranking snapshot for the best strategy:")
    for row in latest_recommendations.itertuples(index=False):
        print(
            "  "
            f"{row.ticker}: score={row.score:.4f}, price={row.latest_price:.2f}, "
            f"1m={row.ret_1m:.2%}, 6m={row.ret_6m:.2%}, 12m={row.ret_12m:.2%}"
        )
    print()
    print("Saved files:")
    print("  outputs/backtest_summary.csv")
    print("  outputs/top_n_summary.csv")
    print("  outputs/current_model_portfolio.csv")
    print("  outputs/latest_recommendations.csv")
    print("  outputs/monthly_portfolio_history.csv")
    print("  outputs/trade_log.csv")
    print("  outputs/rebalance_summary.csv")
    print("  outputs/latest_report.md")


def write_report(
    output_dir: Path,
    latest_market_date: str,
    universe_size: int,
    best_row: pd.Series,
    top_n_summary: pd.DataFrame,
    rebalance_frequencies: list[str],
    current_holdings: pd.DataFrame,
    latest_recommendations: pd.DataFrame,
) -> None:
    lines = [
        "# Monthly Rebalancing Report",
        "",
        f"- Latest completed market close used: {latest_market_date}",
        f"- Universe size used: {universe_size} stocks",
        f"- Rebalance frequencies tested: {', '.join(rebalance_frequencies)}",
        "- Execution assumption: signal on period-end close, trade on next trading day open",
        f"- Best strategy: {best_row['strategy']}",
        f"- Description: {best_row['description']}",
        f"- Best rebalance frequency: {best_row['rebalance_frequency']}",
        f"- Best top N: {int(best_row['top_n'])}",
        f"- Initial capital: {best_row['initial_capital']:.2f}",
        f"- Total return: {best_row['total_return']:.2%}",
        f"- CAGR: {best_row['cagr']:.2%}",
        f"- Annual volatility: {best_row['annual_volatility']:.2%}",
        f"- Sharpe: {best_row['sharpe']:.2f}",
        f"- Max drawdown: {best_row['max_drawdown']:.2%}",
        f"- Final equity: {best_row['final_equity']:.2f}",
        "",
        "## Best Result By Top N",
        "",
    ]
    for row in top_n_summary.itertuples(index=False):
        lines.append(
            f"- freq={row.rebalance_frequency}, N={row.top_n}: {row.strategy}, "
            f"total return {row.total_return:.2%}, CAGR {row.cagr:.2%}"
        )
    lines.extend(["", "## Current Model Portfolio", ""])
    if current_holdings.empty:
        lines.append("- No active holdings")
    else:
        for row in current_holdings.itertuples(index=False):
            if row.ticker == "CASH":
                lines.append(f"- CASH: value {row.market_value:.2f}, weight {row.weight:.2%}")
            else:
                lines.append(
                    f"- {row.ticker}: shares {int(row.shares)}, latest price {row.latest_price:.2f}, "
                    f"value {row.market_value:.2f}, weight {row.weight:.2%}"
                )
    lines.extend(["", "## Latest Ranking Snapshot", ""])
    for row in latest_recommendations.itertuples(index=False):
        lines.append(
            f"- {row.ticker}: score {row.score:.4f}, price {row.latest_price:.2f}, "
            f"1m {row.ret_1m:.2%}, 6m {row.ret_6m:.2%}, 12m {row.ret_12m:.2%}"
        )
    lines.extend(
        [
            "",
            "## Exported Files",
            "",
            "- outputs/backtest_summary.csv: all strategy x top-N combinations",
            "- outputs/top_n_summary.csv: best result for each top-N",
            "- outputs/current_model_portfolio.csv: current integer-share portfolio plus cash",
            "- outputs/latest_recommendations.csv: ranked candidates for the best strategy",
            "- outputs/monthly_portfolio_history.csv: portfolio after each rebalance",
            "- outputs/trade_log.csv: every buy and sell with prices",
            "- outputs/rebalance_summary.csv: one row per rebalance event",
        ]
    )
    report_path = output_dir / "latest_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_eligibility_map(metadata: pd.DataFrame, universe: list[str]) -> dict[str, pd.Timestamp]:
    eligible_from: dict[str, pd.Timestamp] = {}
    columns = {column.lower(): column for column in metadata.columns}
    symbol_column = columns.get("symbol")
    date_added_column = columns.get("date added")
    if symbol_column is None or date_added_column is None:
        return eligible_from

    reduced = metadata[[symbol_column, date_added_column]].copy()
    reduced[date_added_column] = pd.to_datetime(reduced[date_added_column], errors="coerce")
    reduced = reduced.dropna(subset=[symbol_column]).drop_duplicates(subset=[symbol_column], keep="last")
    for _, row in reduced.iterrows():
        symbol = row[symbol_column]
        date_added = row[date_added_column]
        if symbol in universe and pd.notna(date_added):
            eligible_from[symbol] = pd.Timestamp(date_added).tz_localize(None)
    return eligible_from


if __name__ == "__main__":
    main()
