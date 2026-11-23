# 퀀트 투자 연구소

Streamlit 상단 메뉴에서 세 가지 투자 도구를 사용할 수 있습니다.

- **리밸런싱 연구실**에서 전략과 주기별 백테스트, 추천 종목, 매수 계획과 거래 내역을 확인합니다. 백테스트 기간은 기본적으로 실행일 기준 최근 10년이며, 최근 1·3·5년 또는 직접 지정으로 변경할 수 있습니다. 종료일을 지정하면 해당 날짜를 기준으로 계산합니다.
- **종목 비교**에서 저장된 가격 데이터의 종목을 선택하고 공통 거래일 기준으로 누적 수익률과 최대 낙폭을 비교합니다. 가격 데이터는 연구실의 가격 캐시를 갱신할 때 업데이트됩니다.
- **적립식 시뮬레이터**에서 초기 투자금, 월 납입액, 가정 연 수익률과 투자 기간에 따른 원금과 자산 추이를 살펴봅니다.

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
python run_pipeline.py --start 2016-01-01 --top-n-values 5,10,15,20 --transaction-cost-bps 7
python run_pipeline.py --top-n 10
python run_pipeline.py --rebalance-frequencies monthly,quarterly
python run_pipeline.py --rebalance-frequency weekly
python run_pipeline.py --strategies momentum_6m,blend_momentum
python run_pipeline.py --initial-capital 10000
python run_pipeline.py --rebalance-contribution 100000
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

- Each period is anchored to its final NYSE trading session. The configured shift subtracts calendar days, with the default holiday adjustment moving the signal to the preceding session. Signals use the close and trades use the next session's open. The summary stores the next pending signal and execution dates for each run.
- Rebalance frequency is configurable: `weekly`, `monthly`, `bimonthly`, `quarterly`, `semiannual`, `annual`.
- Strategies can be filtered in the Streamlit sidebar or with `--strategies` in the CLI.
- Optional cash contributions can be added immediately before every scheduled rebalance.
- Shares are integer-only. Fractional shares are not allowed.
- The allocator aims for near-equal dollar exposure across selected names and keeps leftover cash when exact equality is impossible.
- Trade logs include buy/sell price, share count, notional, and fee.
- A missing or invalid opening price freezes that stock's shares for the rebalance. Holdings use the latest valid close for valuation, or that session's open if the close is missing.
- The simulation settles its annual tax assumption on the last scheduled execution of the calendar year, using realized sales. Ending a backtest earlier leaves that year's settlement pending.
- Hybrid strategies require a valid score for every positively weighted factor. Saved recommendations include the strategy's target weights, and the dashboard exports the portfolio and trade logs of the selected run.
- The Streamlit app includes a buy-plan tab where you can enter your current capital and see how many shares to buy for the current model portfolio.

## Caveats

- Free data only; using the current S&P 500 membership for the full history introduces survivorship bias.
- This project is for research and learning purposes, not investment advice.
