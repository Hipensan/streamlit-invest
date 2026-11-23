# Monthly Rebalancing Report

- Latest completed market close used: 2026-11-23
- Universe size used: 503 stocks
- Rebalance frequencies tested: quarterly
- Strategies tested: momentum_3m, momentum_12m, momentum_6m
- Execution assumption: signal on scheduled close, trade on next trading day open
- Best strategy: momentum_3m
- Description: Top N stocks by trailing 3-month return
- Best rebalance frequency: quarterly
- Best top N: 5
- Initial capital: 10000.00
- Rebalance contribution: 2000.00
- Total contributions: 8000.00
- Total invested capital: 18000.00
- Total return: 64.73%
- CAGR: 66.75%
- Annual volatility: 54.19%
- Sharpe: 1.23
- Max drawdown: -35.98%
- Final equity: 27118.62

## Best Result By Top N

- freq=quarterly, N=5: momentum_3m, total return 64.73%, CAGR 66.75%

## Current Model Portfolio

- MRNA: shares 35, latest price 212.08, value 7422.80, weight 27.37%
- CRL: shares 17, latest price 338.26, value 5750.42, weight 21.20%
- MPC: shares 13, latest price 357.69, value 4649.97, weight 17.15%
- VLO: shares 14, latest price 329.40, value 4611.60, weight 17.01%
- VEEV: shares 17, latest price 268.65, value 4567.05, weight 16.84%
- CASH: value 116.78, weight 0.43%

## Latest Ranking Snapshot

- OKTA: score 0.5794, price 206.28, 1m -3.18%, 3m 57.94%, 6m 119.89%, 12m nan%
- MRNA: score 0.3353, price 212.08, 1m 5.99%, 3m 33.53%, 6m 350.95%, 12m nan%
- PTC: score 0.2968, price 196.00, 1m 1.14%, 3m 29.68%, 6m 35.28%, 12m nan%
- IDXX: score 0.2491, price 687.02, 1m 22.60%, 3m 24.91%, 6m 23.30%, 12m nan%
- MTD: score 0.2305, price 1725.52, 1m 3.31%, 3m 23.05%, 6m 56.68%, 12m nan%
- WMT: score 0.2296, price 129.58, 1m 8.08%, 3m 22.96%, 6m 9.55%, 12m nan%
- RVTY: score 0.2124, price 151.29, 1m -4.71%, 3m 21.24%, 6m 56.41%, 12m nan%
- APP: score 0.2041, price 373.92, 1m 43.68%, 3m 20.41%, 6m -27.29%, 12m nan%
- ACN: score 0.1987, price 222.29, 1m -10.86%, 3m 19.87%, 6m 28.12%, 12m nan%
- CDW: score 0.1968, price 170.06, 1m 3.00%, 3m 19.68%, 6m 57.36%, 12m nan%
- VLTO: score 0.1941, price 117.78, 1m 12.89%, 3m 19.41%, 6m 37.26%, 12m nan%
- ALLE: score 0.1843, price 189.26, 1m 5.74%, 3m 18.43%, 6m 44.91%, 12m nan%
- HAS: score 0.1805, price 113.53, 1m -2.75%, 3m 18.05%, 6m 31.84%, 12m nan%
- VTRS: score 0.1752, price 19.65, 1m 9.17%, 3m 17.52%, 6m 21.76%, 12m nan%
- PG: score 0.1742, price 169.60, 1m 5.33%, 3m 17.42%, 6m 20.31%, 12m nan%
- SWKS: score 0.1738, price 77.12, 1m -7.56%, 3m 17.38%, 6m -7.55%, 12m nan%
- MAS: score 0.1722, price 85.15, 1m 12.45%, 3m 17.22%, 6m 24.60%, 12m nan%
- MAR: score 0.1637, price 415.63, 1m 6.56%, 3m 16.37%, 6m 11.60%, 12m nan%
- ULTA: score 0.1634, price 625.02, 1m 0.09%, 3m 16.34%, 6m 21.12%, 12m nan%
- BIIB: score 0.1628, price 256.71, 1m 2.40%, 3m 16.28%, 6m 32.96%, 12m nan%

## Exported Files

- outputs/backtest_summary.csv: all strategy x top-N combinations
- outputs/top_n_summary.csv: best result for each top-N
- outputs/current_model_portfolio.csv: current integer-share portfolio plus cash
- outputs/latest_recommendations.csv: ranked candidates for the best strategy
- outputs/monthly_portfolio_history.csv: portfolio after each rebalance
- outputs/trade_log.csv: every buy and sell with prices
- outputs/rebalance_summary.csv: one row per rebalance event
