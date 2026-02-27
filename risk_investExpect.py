import pandas as pd
import numpy as np
import yfinance as yf
import requests
import streamlit as st
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
from datetime import datetime
import argparse

warnings.filterwarnings("ignore")

@st.cache_data(ttl=60 * 10, show_spinner=False)
def _cached_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    df = pd.read_html(StringIO(response.text))[0]
    tickers = df['Symbol'].tolist()
    return [t.replace('.', '-') for t in tickers]

@st.cache_data(ttl=60 * 10, show_spinner=False)
def _cached_spy_prices(start_date):
    spy = yf.download("SPY", start=start_date, progress=False, auto_adjust=True)
    if spy.index.tz is not None:
        spy.index = spy.index.tz_localize(None)
    
    if isinstance(spy['Close'], pd.DataFrame):
        close_series = spy['Close'].iloc[:, 0]
    else:
        close_series = spy['Close']
        
    spy_df = pd.DataFrame(index=spy.index)
    spy_df['Close'] = close_series
    return spy_df

@st.cache_data(ttl=60 * 10, show_spinner=False)
def _cached_price_data(tickers, start_date):
    prices = {}

    def fetch(ticker):
        try:
            t = yf.Ticker(ticker)
            df = t.history(start=start_date, auto_adjust=True)
            if df.empty:
                df = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)
            if not df.empty and 'Close' in df.columns:
                c = df['Close']
                if isinstance(c, pd.DataFrame):
                    return ticker, c.iloc[:, 0]
                return ticker, c
        except:
            return ticker, None
        return ticker, None

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(fetch, t): t for t in tickers}
        for future in as_completed(futures):
            ticker, series = future.result()
            if series is not None:
                if series.index.tz is not None:
                    series.index = series.index.tz_localize(None)
                prices[ticker] = series
    
    if not prices:
        return pd.DataFrame()

    price_df = pd.concat(prices, axis=1)
    price_df.sort_index(inplace=True)
    return price_df

class MomentumStrategy:
    def __init__(self, top_n=5, momentum_window=6, lookback_years=10):
        self.top_n = top_n
        self.momentum_window = momentum_window
        self.lookback_years = lookback_years

    @staticmethod
    def get_sp500_tickers():
        try:
            return _cached_sp500_tickers()
        except Exception as e:
            print(f"??? ??????????? ???: {e}")
            return []

    def get_price_data(self, tickers, start_date=None):
        if start_date is None:
            start_date = (datetime.now() - pd.DateOffset(years=self.lookback_years + 2)).strftime('%Y-%m-%d')
        return _cached_price_data(tuple(tickers), start_date)

    @staticmethod
    def get_spy_prices(start_date):
        return _cached_spy_prices(start_date)

    def run_simulation(self, interval='1mo', strategy_name="Momentum", buy_lag_days=1):
        tickers = self.get_sp500_tickers()
        if not tickers: return None, None
        price_df = self.get_price_data(tickers)
        if price_df.empty: return None, None
        spy_df = self.get_spy_prices(price_df.index[0])

        if interval == '2wk':
            resample_rule, periods_per_year = '2W-FRI', 26
        elif interval == '1wk':
            resample_rule, periods_per_year = 'W-FRI', 52
        else:
            resample_rule, periods_per_year = 'M', 12

        rebalance_prices = price_df.resample(resample_rule).last()
        # Drop the last resampled row if it is labeled after the last actual trading date
        # (e.g., month-end label when the month is not finished yet).
        last_actual_date = price_df.index[-1]
        if not rebalance_prices.empty and rebalance_prices.index[-1] > last_actual_date:
            rebalance_prices = rebalance_prices.iloc[:-1]
        # Use daily momentum (same logic as recommendation) for consistency
        lookback_days = int(self.momentum_window * 21)

        portfolio_value = 10000.0
        history = []
        start_idx = 1
        # Find the first rebalance with enough data
        start_idx = 1
        for i in range(1, len(rebalance_prices) - 1):
            current_date = rebalance_prices.index[i]
            current_prices = price_df.loc[:current_date]
            if len(current_prices) > lookback_days:
                start_idx = i
                break
        buy_hold_cash = None
        spy_close = spy_df['Close'] if isinstance(spy_df, pd.DataFrame) and 'Close' in spy_df.columns else None
        held_stocks = []  # Track holdings across rebalance periods
        leftover_cash = 0.0  # Track cash across rebalance periods
        history = []
        start_idx = 1
        buy_hold_shares = None
        buy_hold_cash = None
        spy_close = spy_df['Close'] if isinstance(spy_df, pd.DataFrame) and 'Close' in spy_df.columns else None
        held_stocks = []  # Track holdings across rebalance periods
        history = []
        start_idx = 1
        buy_hold_shares = None
        buy_hold_cash = None
        spy_close = spy_df['Close'] if isinstance(spy_df, pd.DataFrame) and 'Close' in spy_df.columns else None

        def price_on_or_before(df, date, ticker):
            series = df[ticker].loc[:date].dropna()
            if series.empty:
                return np.nan
            return series.iloc[-1]

        def series_on_or_before(series, date):
            s = series.loc[:date].dropna()
            if s.empty:
                return np.nan
            return s.iloc[-1]

        def trading_date_on_or_before(index, date, lag_days=0):
            candidates = index[index <= date]
            if len(candidates) == 0:
                return None
            pos = max(0, len(candidates) - 1 - lag_days)
            return candidates[pos]

        def compute_scores(current_prices):
            if strategy_name == "Momentum":
                scores = current_prices.pct_change(lookback_days).iloc[-1].dropna()
            elif strategy_name == "Blended Momentum (3/6/12)":
                lookbacks = [63, 126, 252]
                parts = []
                for lb in lookbacks:
                    if len(current_prices) > lb:
                        parts.append(current_prices.pct_change(lb).iloc[-1])
                if not parts:
                    return pd.Series(dtype=float)
                scores = pd.concat(parts, axis=1).mean(axis=1).dropna()
            elif strategy_name == "Volatility-Adjusted Momentum":
                if len(current_prices) <= lookback_days:
                    return pd.Series(dtype=float)
                mom = current_prices.pct_change(lookback_days).iloc[-1]
                daily_returns = current_prices.pct_change().tail(lookback_days)
                vol = daily_returns.std()
                scores = (mom / vol.replace(0, np.nan)).dropna()
            else:
                scores = current_prices.pct_change(lookback_days).iloc[-1].dropna()
            return scores

        for i in range(start_idx, len(rebalance_prices) - 1):
            current_date = rebalance_prices.index[i]
            next_date = rebalance_prices.index[i+1]
            buy_date = trading_date_on_or_before(price_df.index, current_date, 0)
            sell_date = trading_date_on_or_before(price_df.index, next_date, 1)
            if buy_date is None or sell_date is None or buy_date >= sell_date:
                continue


            current_cash = portfolio_value
            target_allocation = current_cash / self.top_n

            current_prices = price_df.loc[:buy_date]
            if len(current_prices) <= lookback_days:
                continue

            if strategy_name == "Buy & Hold SPY":
                if spy_close is None or spy_close.empty:
                    continue
                if buy_hold_shares is None:
                    current_price = series_on_or_before(spy_close, buy_date)
                    if pd.isna(current_price) or current_price <= 0:
                        continue
                    buy_hold_shares = int(current_cash / current_price)
                    buy_hold_cash = current_cash - (buy_hold_shares * current_price)
                next_price = series_on_or_before(spy_close, sell_date)
                if pd.isna(next_price) or next_price <= 0:
                    next_price = series_on_or_before(spy_close, buy_date)
                portfolio_value = buy_hold_cash + (buy_hold_shares * next_price)
                holdings_display = f"Cash(${buy_hold_cash:.0f}) + SPY({buy_hold_shares})"
                history.append({
                    "Date": sell_date,
                    "Value": portfolio_value,
                    "Return": (portfolio_value - current_cash) / current_cash * 100,
                    "Holdings": holdings_display
                })
                continue

            score_series = compute_scores(current_prices)
            if score_series.empty:
                continue
            top_series = score_series.sort_values(ascending=False).head(self.top_n)
            top_stocks = top_series.index.tolist()

            # First rebalance: buy all top stocks
            if i == start_idx:
                held_stocks = []
                invested_amount = 0

                for ticker in top_stocks:
                    current_price = price_on_or_before(price_df, buy_date, ticker)
                    if pd.isna(current_price) or current_price <= 0:
                        continue
                    shares = int(target_allocation / current_price)
                    cost = shares * current_price
                    held_stocks.append({
                        'ticker': ticker,
                        'shares': shares,
                        'price': current_price,
                        'cost': cost
                    })
                    invested_amount += cost

                leftover_cash = current_cash - invested_amount

                # Reinvest leftover cash
                while True:
                    bought_something = False
                    for item in held_stocks:
                        if leftover_cash >= item['price']:
                            leftover_cash -= item['price']
                            item['shares'] += 1
                            item['cost'] += item['price']
                            bought_something = True
                    if not bought_something:
                        break
            # Subsequent rebalances: keep existing holdings, adjust differences
            else:
                held_tickers = {item['ticker'] for item in held_stocks}
                top_set = set(top_stocks)
                
                # Keepers: existing holdings still in top stocks
                keepers = [item for item in held_stocks if item['ticker'] in top_set]
                
                # New stocks: top stocks not currently held
                new_stocks = [t for t in top_stocks if t not in held_tickers]
                
                # Removed stocks: currently held but not in top stocks
                removed_stocks = [item for item in held_stocks if item['ticker'] not in top_set]
                
                # Calculate target portfolio value (sell removed stocks first)
                sell_value = 0
                for item in removed_stocks:
                    sell_price = price_on_or_before(price_df, sell_date, item['ticker'])
                    if pd.isna(sell_price) or sell_price <= 0:
                        sell_price = price_on_or_before(price_df, buy_date, item['ticker'])
                    sell_value += item['shares'] * sell_price
                
                # Include previous leftover cash
                portfolio_value = current_cash + sell_value + leftover_cash
                leftover_cash = portfolio_value
                leftover_cash = portfolio_value
                
                # Allocate to keepers (existing holdings)
                new_held_stocks = []
                total_invested = 0
                
                for item in keepers:
                    current_price = price_on_or_before(price_df, buy_date, item['ticker'])
                    if pd.isna(current_price) or current_price <= 0:
                        continue
                    target_shares = int(target_allocation / current_price)
                    
                    # Buy or sell to reach target
                    if target_shares > item['shares']:
                        # Buy more
                        buy_shares = target_shares - item['shares']
                        buy_cost = buy_shares * current_price
                        if leftover_cash >= buy_cost:
                            new_shares = target_shares
                            new_cost = new_shares * current_price
                            leftover_cash -= buy_cost
                        else:
                            # Not enough cash, buy as much as possible
                            new_shares = item['shares'] + int(leftover_cash / current_price)
                            new_cost = new_shares * current_price
                            leftover_cash -= (new_shares - item['shares']) * current_price
                    elif target_shares < item['shares']:
                        # Sell excess
                        sell_shares = item['shares'] - target_shares
                        sell_price_for_excess = price_on_or_before(price_df, sell_date, item['ticker'])
                        if pd.isna(sell_price_for_excess) or sell_price_for_excess <= 0:
                            sell_price_for_excess = current_price
                        leftover_cash += sell_shares * sell_price_for_excess
                        new_shares = target_shares
                        new_cost = new_shares * current_price
                    else:
                        # No change
                        new_shares = item['shares']
                        new_cost = new_shares * current_price
                    
                    new_held_stocks.append({
                        'ticker': item['ticker'],
                        'shares': new_shares,
                        'price': current_price,
                        'cost': new_cost
                    })
                    total_invested += new_cost
                
                # Buy new stocks with remaining cash
                for ticker in new_stocks:
                    current_price = price_on_or_before(price_df, buy_date, ticker)
                    if pd.isna(current_price) or current_price <= 0:
                        continue
                    shares = int(target_allocation / current_price)
                    cost = shares * current_price
                    
                    if leftover_cash >= cost:
                        new_held_stocks.append({
                            'ticker': ticker,
                            'shares': shares,
                            'price': current_price,
                            'cost': cost
                        })
                        leftover_cash -= cost
                        total_invested += cost
                
                # Reinvest leftover cash
                while True:
                    bought_something = False
                    for item in new_held_stocks:
                        if leftover_cash >= item['price']:
                            leftover_cash -= item['price']
                            item['shares'] += 1
                            item['cost'] += item['price']
                            bought_something = True
                    if not bought_something:
                        break
                
            next_portfolio_value = leftover_cash
            stock_details = []

            for item in held_stocks:
                if item['shares'] > 0:
                    stock_details.append(f"{item['ticker']}({item['shares']} 개)")

                    next_price = price_on_or_before(price_df, sell_date, item['ticker'])
                    if pd.isna(next_price) or next_price <= 0:
                        next_price = price_on_or_before(price_df, buy_date, item['ticker'])

                    next_portfolio_value += item['shares'] * next_price

            portfolio_value = next_portfolio_value
            holdings_display = f"Cash(${leftover_cash:.0f}) + " + ", ".join(stock_details)

            history.append({
                "Date": sell_date,
                "Value": portfolio_value,
                "Return": (portfolio_value - current_cash) / current_cash * 100,
                "Holdings": holdings_display
            })

        results = pd.DataFrame(history)
        if results.empty: return None, None

        years = (results['Date'].iloc[-1] - results['Date'].iloc[0]).days / 365.25
        cagr = (portfolio_value / 10000) ** (1 / years) - 1
        results['Peak'] = results['Value'].cummax()
        results['Drawdown'] = (results['Value'] - results['Peak']) / results['Peak'] * 100
        mdd = results['Drawdown'].min()
        avg_dd = results['Drawdown'].mean()

        summary = {
            "Final Value": portfolio_value,
            "Total Return": (portfolio_value - 10000) / 10000 * 100,
            "CAGR": cagr * 100,
            "MDD": mdd,
            "Avg Drawdown": avg_dd,
            "Years": years
        }
        return results, summary

    def recommend_portfolio(self, my_stocks=None, investment_capital=10000):
        end_date = datetime.now()
        start_date_spy = end_date - pd.DateOffset(days=400)
        spy_df = self.get_spy_prices(start_date_spy)
        
        current_price = float(spy_df['Close'].iloc[-1])

        recommendation = {
            "SPY Price": current_price,
            "Date": spy_df.index[-1].date(),
            "SPY Data": spy_df.tail(252)
        }

        tickers = self.get_sp500_tickers()
        start_date_stocks = (end_date - pd.DateOffset(days=550)).strftime('%Y-%m-%d')
        price_df = self.get_price_data(tickers, start_date=start_date_stocks)
        
        lookback_days = int(self.momentum_window * 21)
        momentum = price_df.pct_change(lookback_days).iloc[-1].dropna()
        momentum_sorted = momentum.sort_values(ascending=False)
        top_stocks = momentum_sorted.head(self.top_n)
        alternate_stocks = momentum_sorted.head(self.top_n + 3).iloc[self.top_n:]

        def _stock_metrics(series):
            metrics = {
                "Return 3M": np.nan,
                "Return 6M": np.nan,
                "Return 12M": np.nan,
                "Volatility (63d)": np.nan
            }
            if series is None or series.empty:
                return metrics
            if len(series) > 63:
                metrics["Return 3M"] = series.pct_change(63).iloc[-1] * 100
            if len(series) > 126:
                metrics["Return 6M"] = series.pct_change(126).iloc[-1] * 100
            if len(series) > 252:
                metrics["Return 12M"] = series.pct_change(252).iloc[-1] * 100
            daily_returns = series.pct_change().tail(63)
            if daily_returns.dropna().size > 2:
                metrics["Volatility (63d)"] = daily_returns.std() * np.sqrt(252) * 100
            return metrics
        
        target_allocation = investment_capital / self.top_n
        
        # 1차 분배: 동일 비중 할당
        rec_stocks = []
        invested_cost = 0

        for ticker, mom in top_stocks.items():
            series = price_df[ticker].dropna()
            if series.empty:
                continue
            price = series.iloc[-1]

            if pd.isna(price) or price <= 0:
                continue

            shares = int(target_allocation / price)
            cost = shares * price
            invested_cost += cost

            metrics = _stock_metrics(series)
            rec_stocks.append({
                "Ticker": ticker,
                "Momentum": mom * 100,
                "Price": price,
                "Shares to Buy": shares,
                "Cost": cost,
                "Return 3M": metrics["Return 3M"],
                "Return 6M": metrics["Return 6M"],
                "Return 12M": metrics["Return 12M"],
                "Volatility (63d)": metrics["Volatility (63d)"]
            })
            
        # 2차 분배: 남은 현금 재투자 (모멘텀 순위 우선)
        leftover_cash = investment_capital - invested_cost

        alternates = []
        for ticker, mom in alternate_stocks.items():
            series = price_df[ticker].dropna()
            if series.empty:
                continue
            price = series.iloc[-1]
            if pd.isna(price) or price <= 0:
                continue
            metrics = _stock_metrics(series)
            alternates.append({
                "Ticker": ticker,
                "Momentum": mom * 100,
                "Price": price,
                "Return 3M": metrics["Return 3M"],
                "Return 6M": metrics["Return 6M"],
                "Return 12M": metrics["Return 12M"],
                "Volatility (63d)": metrics["Volatility (63d)"]
            })

        while True:
            bought_something = False
            for item in rec_stocks:
                if leftover_cash >= item['Price']:
                    leftover_cash -= item['Price']
                    item['Shares to Buy'] += 1
                    item['Cost'] += item['Price']
                    bought_something = True
            
            if not bought_something:
                break
        
        recommendation["Top Stocks"] = rec_stocks
        recommendation["Alternates"] = alternates
        top_tickers_list = top_stocks.index.tolist()
        top_prices = price_df[top_tickers_list].dropna(how='all').tail(252)
        recommendation["Top Prices"] = top_prices
        recommendation["Total Cost"] = sum(x['Cost'] for x in rec_stocks)
        recommendation["Leftover Cash"] = leftover_cash
        
        if my_stocks:
            recommendation["Analysis"] = {
                "Keep": [t for t in my_stocks if t in top_tickers_list],
                "Sell": [t for t in my_stocks if t not in top_tickers_list],
                "Buy": [t for t in top_tickers_list if t not in my_stocks]
            }
        
        return recommendation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dual Momentum Strategy")
    parser.add_argument("--backtest", action="store_true", help="백테스트 시뮬레이션 실행")
    parser.add_argument("--interval", type=str, default="1mo", choices=["1mo", "2wk", "1wk"], help="리밸런싱 주기")
    args = parser.parse_args()

    strategy = MomentumStrategy()
    if args.backtest:
        results, summary = strategy.run_simulation(args.interval)
        if results is not None:
            print(f"CAGR: {summary['CAGR']:.2f}%, MDD: {summary['MDD']:.2f}%")
            results.to_csv("Momentum_Strategy_Result.csv", index=False)
    else:
        rec = strategy.recommend_portfolio(None)
        print(rec)

