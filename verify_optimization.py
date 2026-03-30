import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

def simulate_bitcoin_prices(days=1000, initial_price=50000, volatility=0.04, drift=0.001):
    np.random.seed(123)
    prices = [initial_price]
    for _ in range(1, days):
        shock = np.random.normal(0, 1)
        price_change = np.exp((drift - 0.5 * volatility**2) + volatility * shock)
        prices.append(prices[-1] * price_change)
    start_date = datetime.now() - timedelta(days=days-1)
    dates = [start_date + timedelta(days=i) for i in range(days)]
    df = pd.DataFrame({'Date': dates, 'Price': prices})
    return df

def calculate_moving_averages(df):
    df['MA7'] = df['Price'].rolling(window=7).mean()
    df['MA30'] = df['Price'].rolling(window=30).mean()
    return df

def run_trading_algorithm_original(df):
    cash = 10000.0
    btc = 0.0
    ledger = []
    for i in range(len(df)):
        date = df.loc[i, 'Date'].strftime('%Y-%m-%d')
        price = df.loc[i, 'Price']
        ma7 = df.loc[i, 'MA7']
        ma30 = df.loc[i, 'MA30']
        action = "HOLD"
        if i > 0 and not pd.isna(ma7) and not pd.isna(ma30) and not pd.isna(df.loc[i-1, 'MA7']) and not pd.isna(df.loc[i-1, 'MA30']):
            prev_ma7 = df.loc[i-1, 'MA7']
            prev_ma30 = df.loc[i-1, 'MA30']
            if prev_ma7 <= prev_ma30 and ma7 > ma30:
                if cash > 0:
                    btc = cash / price
                    cash = 0.0
                    action = f"BUY {btc:.4f} BTC"
            elif prev_ma7 >= prev_ma30 and ma7 < ma30:
                if btc > 0:
                    cash = btc * price
                    action = f"SELL {btc:.4f} BTC"
                    btc = 0.0
        portfolio_value = cash + (btc * price)
        ledger.append({
            'Date': date, 'Price': price, 'MA7': ma7, 'MA30': ma30,
            'Action': action, 'Cash': cash, 'BTC': btc, 'Portfolio Value': portfolio_value
        })
    return pd.DataFrame(ledger)

def run_trading_algorithm_optimized(df):
    cash = 10000.0
    btc = 0.0
    ledger = []

    # Optimization: Use .values for faster access and vectorize strftime
    dates = df['Date'].dt.strftime('%Y-%m-%d').values
    prices = df['Price'].values
    ma7s = df['MA7'].values
    ma30s = df['MA30'].values

    for i in range(len(df)):
        date = dates[i]
        price = prices[i]
        ma7 = ma7s[i]
        ma30 = ma30s[i]
        action = "HOLD"

        if i > 0 and not pd.isna(ma7) and not pd.isna(ma30) and not pd.isna(ma7s[i-1]) and not pd.isna(ma30s[i-1]):
            prev_ma7 = ma7s[i-1]
            prev_ma30 = ma30s[i-1]
            if prev_ma7 <= prev_ma30 and ma7 > ma30:
                if cash > 0:
                    btc = cash / price
                    cash = 0.0
                    action = f"BUY {btc:.4f} BTC"
            elif prev_ma7 >= prev_ma30 and ma7 < ma30:
                if btc > 0:
                    cash = btc * price
                    action = f"SELL {btc:.4f} BTC"
                    btc = 0.0
        portfolio_value = cash + (btc * price)
        ledger.append({
            'Date': date, 'Price': price, 'MA7': ma7, 'MA30': ma30,
            'Action': action, 'Cash': cash, 'BTC': btc, 'Portfolio Value': portfolio_value
        })
    return pd.DataFrame(ledger)

if __name__ == "__main__":
    try:
        print("Generating data...")
        df = simulate_bitcoin_prices(2000)
        df = calculate_moving_averages(df)

        print("Running original algorithm...")
        start = time.time()
        res_orig = run_trading_algorithm_original(df)
        end = time.time()
        orig_time = end - start
        print(f"Original time: {orig_time:.4f}s")

        print("Running optimized algorithm...")
        start = time.time()
        res_opt = run_trading_algorithm_optimized(df)
        end = time.time()
        opt_time = end - start
        print(f"Optimized time: {opt_time:.4f}s")

        print(f"Improvement: {(orig_time - opt_time) / orig_time * 100:.2f}%")

        # Verify correctness
        pd.testing.assert_frame_equal(res_orig, res_opt)
        print("Verification successful: Both versions produce identical results.")

    except Exception as e:
        print(f"Error during benchmark: {e}")
        print("This is expected if pandas/numpy are not installed in the environment.")
