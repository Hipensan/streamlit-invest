import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
from datetime import datetime
import argparse
import os

warnings.filterwarnings("ignore")

class MomentumStrategy:
    def __init__(self, top_n=5, momentum_window=6, lookback_years=10):
        self.top_n = top_n
        self.momentum_window = momentum_window
        self.lookback_years = lookback_years

    @staticmethod
    def get_sp500_tickers():
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers)
            df = pd.read_html(StringIO(response.text))[0]
            tickers = df['Symbol'].tolist()
            return [t.replace('.', '-') for t in tickers]
        except Exception as e:
            print(f"티커 리스트 다운로드 실패: {e}")
            return []

    def get_price_data(self, tickers, start_date=None):
        if start_date is None:
            start_date = (datetime.now() - pd.DateOffset(years=self.lookback_years + 2)).strftime('%Y-%m-%d')
        
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

    @staticmethod
    def get_spy_filter(start_date):
        spy = yf.download("SPY", start=start_date, progress=False, auto_adjust=True)
        if spy.index.tz is not None:
            spy.index = spy.index.tz_localize(None)
        
        if isinstance(spy['Close'], pd.DataFrame):
            close_series = spy['Close'].iloc[:, 0]
        else:
            close_series = spy['Close']
            
        spy_df = pd.DataFrame(index=spy.index)
        spy_df['Close'] = close_series
        spy_df['SMA200'] = close_series.rolling(window=200).mean()
        return spy_df

    def run_simulation(self, interval='1mo'):
        tickers = self.get_sp500_tickers()
        if not tickers: return None, None
        price_df = self.get_price_data(tickers)
        if price_df.empty: return None, None
        spy_df = self.get_spy_filter(price_df.index[0])

        if interval == '2wk':
            resample_rule, periods_per_year = '2W-FRI', 26
            window = int(self.momentum_window * 26 / 12)
        elif interval == '1wk':
            resample_rule, periods_per_year = 'W-FRI', 52
            window = int(self.momentum_window * 52 / 12)
        else:
            resample_rule, periods_per_year = 'M', 12
            window = self.momentum_window

        monthly_prices = price_df.resample(resample_rule).last()
        momentum_df = monthly_prices.pct_change(window)
        
        portfolio_value = 10000.0
        history = []
        start_idx = window
        
        for i in range(start_idx, len(monthly_prices) - 1):
            current_date = monthly_prices.index[i]
            next_date = monthly_prices.index[i+1]
            
            # 리밸런싱: 현재 자산을 현금화했다고 가정하고 다시 분배
            # (실제로는 매도/매수 수수료가 발생하지만 여기서는 무시하거나 필요시 추가 가능)
            current_cash = portfolio_value
            target_allocation = current_cash / self.top_n

            current_momentum = momentum_df.iloc[i]
            top_series = current_momentum.sort_values(ascending=False).head(self.top_n)
            top_stocks = top_series.index.tolist()
            
            held_stocks = [] # (ticker, shares, buy_price)
            invested_amount = 0
            stock_details = []

            for ticker in top_stocks:
                current_price = monthly_prices.loc[current_date, ticker]
                
                # 가격 데이터 유효성 검사
                if pd.isna(current_price) or current_price <= 0:
                    continue

                # 정수 단위 주식 수 계산 (Floor)
                shares = int(target_allocation / current_price)
                cost = shares * current_price
                
                # 기본 정보 저장 (0주여도 리스트에는 일단 포함, 재투자에서 추가될 수 있음)
                # 재투자 편의를 위해 일단 0주인 종목도 후보군에 둡니다.
                held_stocks.append({
                    'ticker': ticker, 
                    'shares': shares, 
                    'price': current_price,
                    'cost': cost
                })
                invested_amount += cost
            
            # 남은 현금 (자투리 돈)
            leftover_cash = current_cash - invested_amount

            # 2차: 자투리 현금 재투자 (모멘텀 순위대로 1주씩 추가 매수 시도)
            # held_stocks는 이미 모멘텀 순으로 정렬되어 있음 (top_stocks 순서)
            while True:
                bought_something = False
                for item in held_stocks:
                    if leftover_cash >= item['price']:
                        leftover_cash -= item['price']
                        item['shares'] += 1
                        item['cost'] += item['price']
                        bought_something = True
                        # 한 바퀴 돌 때마다 가장 높은 순위부터 다시 기회를 주려면 break 할 수도 있으나,
                        # 여기서는 순서대로 골고루(Round Robin) 분배하는 것이 일반적임.
                        # 다만 "모멘텀 강한 놈에게 몰아주기"를 원한다면 break 후 다시 처음부터 시작해야 함.
                        # "강한 놈 더 사기" 전략 채택:
                        # break 
                
                # Round Robin 방식 (순서대로 1주씩)으로 계속 돌면서 살 수 있는 거 다 사기
                if not bought_something:
                    break
            
            # 최종 포트폴리오 구성
            next_portfolio_value = leftover_cash
            stock_details = []
            
            for item in held_stocks:
                if item['shares'] > 0:
                    m_score = top_series[item['ticker']] * 100
                    stock_details.append(f"{item['ticker']}({item['shares']}주)")
                    
                    next_price = monthly_prices.loc[next_date, item['ticker']]
                    if pd.isna(next_price) or next_price <= 0:
                        next_price = monthly_prices.loc[current_date, item['ticker']]
                    
                    next_portfolio_value += item['shares'] * next_price

            portfolio_value = next_portfolio_value
            holdings_display = f"현금(${leftover_cash:.0f}) + " + ", ".join(stock_details)

            history.append({
                "Date": next_date,
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
        spy_df = self.get_spy_filter(start_date_spy)
        
        current_price = float(spy_df['Close'].iloc[-1])
        current_sma = float(spy_df['SMA200'].iloc[-1])

        recommendation = {
            "SPY Price": current_price,
            "SPY SMA200": current_sma,
            "Date": spy_df.index[-1].date(),
            "SPY Data": spy_df.tail(252)
        }

        tickers = self.get_sp500_tickers()
        start_date_stocks = (end_date - pd.DateOffset(days=550)).strftime('%Y-%m-%d')
        price_df = self.get_price_data(tickers, start_date=start_date_stocks)
        
        lookback_days = int(self.momentum_window * 21)
        momentum = price_df.pct_change(lookback_days).iloc[-1].dropna()
        top_stocks = momentum.sort_values(ascending=False).head(self.top_n)
        
        target_allocation = investment_capital / self.top_n
        
        # 1차 분배: 동일 비중 할당
        rec_stocks = []
        invested_cost = 0

        for ticker, mom in top_stocks.items():
            # 마지막 가격이 NaN일 수 있으므로 유효한 마지막 값을 가져옴
            series = price_df[ticker].dropna()
            if series.empty:
                continue
            price = series.iloc[-1]
            
            # 가격이 0 이하이거나 NaN이면 건너뜀 (안전장치)
            if pd.isna(price) or price <= 0:
                continue

            shares = int(target_allocation / price)
            cost = shares * price
            invested_cost += cost
            
            rec_stocks.append({
                "Ticker": ticker,
                "Momentum": mom * 100,
                "Price": price,
                "Shares to Buy": shares,
                "Cost": cost
            })
            
        # 2차 분배: 남은 현금 재투자 (모멘텀 순위 우선)
        leftover_cash = investment_capital - invested_cost
        
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
        recommendation["Total Cost"] = sum(x['Cost'] for x in rec_stocks)
        recommendation["Leftover Cash"] = leftover_cash
        
        if my_stocks:
            top_tickers_list = top_stocks.index.tolist()
            recommendation["Analysis"] = {
                "Keep": [t for t in my_stocks if t in top_tickers_list],
                "Sell": [t for t in my_stocks if t not in top_tickers_list],
                "Buy": [t for t in top_tickers_list if t not in my_stocks]
            }
        
        return recommendation


def load_portfolio(filepath):
    if not os.path.exists(filepath):
        pd.DataFrame(columns=["Ticker", "Date", "Price"]).to_csv(filepath, index=False)
        return []
    try:
        df = pd.read_csv(filepath)
        return df['Ticker'].astype(str).str.upper().str.strip().tolist()
    except:
        return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dual Momentum Strategy")
    parser.add_argument("--backtest", action="store_true", help="백테스트 시뮬레이션 실행")
    parser.add_argument("--portfolio", type=str, default="my_portfolio.csv", help="포트폴리오 파일 경로")
    parser.add_argument("--interval", type=str, default="1mo", choices=["1mo", "2wk", "1wk"], help="리밸런싱 주기")
    args = parser.parse_args()

    strategy = MomentumStrategy()
    if args.backtest:
        results, summary = strategy.run_simulation(args.interval)
        if results is not None:
            print(f"CAGR: {summary['CAGR']:.2f}%, MDD: {summary['MDD']:.2f}%")
            results.to_csv("Momentum_Strategy_Result.csv", index=False)
    else:
        my_stocks = load_portfolio(args.portfolio)
        rec = strategy.recommend_portfolio(my_stocks)
        print(rec)

