# Monthly Rebalancing Report

- Latest completed market close used: 2026-03-30
- Universe size used: 503 stocks
- Rebalance frequencies tested: monthly
- Execution assumption: signal on period-end close, trade on next trading day open
- Best strategy: momentum_6m
- Description: Top N stocks by trailing 6-month return
- Best rebalance frequency: monthly
- Best top N: 5
- Initial capital: 1000000.00
- Total return: 1103.83%
- CAGR: 22.77%
- Annual volatility: 30.78%
- Sharpe: 0.74
- Max drawdown: -38.30%
- Final equity: 12038266.81

## Best Result By Top N

- freq=monthly, N=5: momentum_6m, total return 1103.83%, CAGR 22.77%

## Current Model Portfolio

- CIEN: shares 7430, latest price 365.00, value 2711950.00, weight 22.53%
- WDC: shares 9722, latest price 251.67, value 2446735.72, weight 20.32%
- SNDK: shares 4248, latest price 572.50, value 2431980.00, weight 20.20%
- TER: shares 8465, latest price 276.35, value 2339302.80, weight 19.43%
- MU: shares 6550, latest price 321.80, value 2107789.92, weight 17.51%
- CASH: value 508.36, weight 0.00%

## Latest Ranking Snapshot

- SNDK: score 4.8948, price 572.50, 1m -9.89%, 6m 489.48%, 12m 982.64%
- LITE: score 3.0733, price 654.79, 1m -6.58%, 6m 307.33%, 12m 923.43%
- CIEN: score 1.5717, price 365.00, 1m 4.67%, 6m 157.17%, 12m 480.66%
- WDC: score 1.3577, price 251.67, 1m -9.98%, 6m 135.77%, 12m 503.09%
- COHR: score 1.0530, price 219.65, 1m -15.17%, 6m 105.30%, 12m 223.49%
- MU: score 1.0491, price 321.80, 1m -21.93%, 6m 104.91%, 12m 254.31%
- TER: score 1.0447, price 276.35, 1m -13.65%, 6m 104.47%, 12m 223.73%
- ALB: score 1.0226, price 177.22, 1m -0.57%, 6m 102.26%, 12m 141.25%
- MRNA: score 0.9694, price 48.23, 1m -9.97%, 6m 96.94%, 12m 50.16%
- DOW: score 0.8726, price 41.87, 1m 36.25%, 6m 87.26%, 12m 26.85%
- APA: score 0.8084, price 43.74, 1m 44.02%, 6m 80.84%, 12m 116.98%
- LYB: score 0.7347, price 82.38, 1m 44.96%, 6m 73.47%, 12m 26.86%
- VRT: score 0.6907, price 234.22, 1m -8.09%, 6m 69.07%, 12m 207.27%
- STX: score 0.6788, price 362.43, 1m -10.98%, 6m 67.88%, 12m 319.54%
- GLW: score 0.6286, price 128.55, 1m -14.52%, 6m 62.86%, 12m 180.26%
- AMAT: score 0.5896, price 323.12, 1m -13.21%, 6m 58.96%, 12m 120.75%
- FIX: score 0.5846, price 1273.18, 1m -10.88%, 6m 58.46%, 12m 284.25%
- HAL: score 0.5811, price 39.26, 1m 9.58%, 6m 58.11%, 12m 59.18%
- LRCX: score 0.5624, price 199.93, 1m -14.42%, 6m 56.24%, 12m 169.21%
- BG: score 0.5608, price 126.28, 1m 4.67%, 6m 56.08%, 12m 70.17%

## Exported Files

- outputs/backtest_summary.csv: all strategy x top-N combinations
- outputs/top_n_summary.csv: best result for each top-N
- outputs/current_model_portfolio.csv: current integer-share portfolio plus cash
- outputs/latest_recommendations.csv: ranked candidates for the best strategy
- outputs/monthly_portfolio_history.csv: portfolio after each rebalance
- outputs/trade_log.csv: every buy and sell with prices
- outputs/rebalance_summary.csv: one row per rebalance event
