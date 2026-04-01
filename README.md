# Monthly Rebalancing Lab

This CLI project compares monthly stock rebalancing strategies under the same rules, then generates the latest stock recommendations from the best-performing strategy.

## Run

```bash
python run_pipeline.py
```

Web dashboard:

```bash
streamlit run app.py
```

Example options:

```bash
python run_pipeline.py --start 2016-01-01 --top-n-values 5,10,15,20 --transaction-cost-bps 10
python run_pipeline.py --top-n 10
python run_pipeline.py --rebalance-frequencies monthly,quarterly
python run_pipeline.py --rebalance-frequency weekly
python run_pipeline.py --initial-capital 1000000
python run_pipeline.py --force-refresh
```

## Output Files

- `outputs/backtest_summary.csv`: strategy x top-N performance summary
- `outputs/top_n_summary.csv`: best-performing strategy for each top-N
- `outputs/equity_curves.csv`: equity curves for each strategy and `SPY`
- `outputs/equity_curves.png`: equity curve chart
- `outputs/current_model_portfolio.csv`: current integer-share model portfolio plus cash
- `outputs/latest_recommendations.csv`: latest ranked recommendations
- `outputs/monthly_portfolio_history.csv`: portfolio snapshot after each rebalance
- `outputs/trade_log.csv`: every buy/sell with execution price
- `outputs/rebalance_summary.csv`: one row per rebalance event
- `outputs/sp500_snapshot.csv`: current S&P 500 constituent snapshot used for the run

## Included Strategies

- `momentum_3m`: top N by trailing 3-month return
- `momentum_6m`: top N by trailing 6-month return
- `momentum_12m`: top N by trailing 12-month return
- `momentum_12m_skip_1m`: top N by 12-month return excluding the latest 1-month move
- `risk_adjusted_momentum`: top N by 12-month return divided by 3-month volatility
- `trend_filtered_momentum`: top N by 12-month momentum only when `SPY` and the stock are above 200DMA
- `low_vol_momentum`: top N by 6-month return penalized by 3-month volatility
- `breakout_52w`: top N by proximity to 52-week highs blended with 6-month momentum
- `blend_momentum`: top N by a blend of 6m/12m momentum and volatility

## Simulation Rules

- Signals are computed on the last trading day of each chosen rebalance period and traded on the next trading day's open.
- Rebalance frequency is configurable: `weekly`, `monthly`, `bimonthly`, `quarterly`, `semiannual`, `annual`.
- Shares are integer-only. Fractional shares are not allowed.
- The allocator aims for near-equal dollar exposure across selected names and keeps leftover cash when exact equality is impossible.
- Trade logs include buy/sell price, share count, notional, and fee.
- The Streamlit app includes a buy-plan tab where you can enter your current capital and see how many shares to buy for the current model portfolio.

## Caveats

- Free data only; using the current S&P 500 membership for the full history introduces survivorship bias.
- This project is for research and learning purposes, not investment advice.
