# Monthly Rebalancing Report

- Latest completed market close used: 2026-07-02
- Universe size used: 503 stocks
- Rebalance frequencies tested: monthly
- Strategies tested: momentum_6m, hybrid_strategy
- Execution assumption: signal on period-end close, trade on next trading day open
- Best strategy: momentum_6m
- Description: Top N stocks by trailing 6-month return
- Best rebalance frequency: monthly
- Best top N: 10
- Initial capital: 10000.00
- Rebalance contribution: 0.00
- Total contributions: 0.00
- Total invested capital: 10000.00
- Total return: 1121.82%
- CAGR: 22.23%
- Annual volatility: 27.87%
- Sharpe: 0.80
- Max drawdown: -36.46%
- Final equity: 122181.96

## Best Result By Top N

- freq=monthly, N=10: momentum_6m, total return 1121.82%, CAGR 22.23%

## Current Model Portfolio

- INTC: shares 123, latest price 120.35, value 14803.05, weight 12.12%
- SNDK: shares 8, latest price 1745.00, value 13960.00, weight 11.43%
- WDC: shares 25, latest price 539.00, value 13475.00, weight 11.03%
- DELL: shares 33, latest price 394.32, value 13012.56, weight 10.65%
- MU: shares 13, latest price 975.56, value 12682.28, weight 10.38%
- STX: shares 15, latest price 820.16, value 12302.40, weight 10.07%
- HPE: shares 267, latest price 41.23, value 11008.41, weight 9.01%
- COHR: shares 33, latest price 333.36, value 11000.88, weight 9.00%
- LITE: shares 14, latest price 728.32, value 10196.48, weight 8.35%
- CIEN: shares 23, latest price 422.46, value 9716.58, weight 7.95%
- CASH: value 24.32, weight 0.02%

## Latest Ranking Snapshot

- SNDK: score 6.2642, price 1745.00, 1m 1.67%, 3m 151.90%, 6m 626.42%, 12m 3781.23%
- MU: score 2.3352, price 975.56, 1m -8.32%, 3m 165.21%, 6m 233.52%, 12m 708.91%
- INTC: score 2.2265, price 120.35, 1m 11.51%, 3m 150.57%, 6m 222.65%, 12m 426.70%
- DELL: score 2.1056, price 394.32, 1m -9.42%, 3m 133.52%, 6m 210.56%, 12m 228.86%
- WDC: score 2.0637, price 539.00, 1m -4.25%, 3m 81.08%, 6m 206.37%, 12m 746.59%
- STX: score 1.9355, price 820.16, 1m -11.43%, 3m 93.97%, 6m 193.55%, 12m 470.14%
- MRVL: score 1.8307, price 245.29, 1m -15.65%, 3m 129.98%, 6m 183.07%, 12m 222.60%
- MRNA: score 1.6228, price 79.76, 1m 74.76%, 3m 59.42%, 6m 162.28%, 12m 178.01%
- AMD: score 1.4047, price 517.82, 1m -0.71%, 3m 146.33%, 6m 140.47%, 12m 280.44%
- AMAT: score 1.3254, price 603.04, 1m 23.06%, 3m 70.66%, 6m 132.54%, 12m 230.57%
- GLW: score 1.2221, price 196.79, 1m -1.80%, 3m 38.43%, 6m 122.21%, 12m 278.29%
- FLEX: score 1.2192, price 136.86, 1m -14.16%, 3m 100.56%, 6m 121.92%, 12m 182.24%
- DVA: score 1.0639, price 234.91, 1m 25.03%, 3m 56.25%, 6m 106.39%, 12m 60.12%
- LRCX: score 1.0260, price 351.41, 1m 5.16%, 3m 58.40%, 6m 102.60%, 12m 265.00%
- LITE: score 0.9622, price 728.32, 1m -29.23%, 3m -4.75%, 6m 96.22%, 12m 696.07%
- FTNT: score 0.9456, price 156.25, 1m 4.96%, 3m 92.54%, 6m 94.56%, 12m 52.54%
- KLAC: score 0.8989, price 235.55, 1m 15.17%, 3m 55.18%, 6m 89.89%, 12m 163.75%
- DDOG: score 0.8938, price 260.36, 1m -3.26%, 3m 119.40%, 6m 89.38%, 12m 96.75%
- TER: score 0.8782, price 369.09, 1m -5.99%, 3m 18.27%, 6m 87.82%, 12m 302.56%
- PANW: score 0.8628, price 348.06, 1m 17.12%, 3m 116.63%, 6m 86.28%, 12m 76.16%

## Daily Action Signals

- outputs/daily_signals.csv: daily buy/add/trim/sell guidance based on the selected strategy

## Exported Files

- outputs/backtest_summary.csv: all strategy x top-N combinations
- outputs/top_n_summary.csv: best result for each top-N
- outputs/current_model_portfolio.csv: current integer-share portfolio plus cash
- outputs/latest_recommendations.csv: ranked candidates for the best strategy
- outputs/monthly_portfolio_history.csv: portfolio after each rebalance
- outputs/trade_log.csv: every buy and sell with prices
- outputs/rebalance_summary.csv: one row per rebalance event
- outputs/daily_signals.csv: daily action guidance and target-weight gaps
