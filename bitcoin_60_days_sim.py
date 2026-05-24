import pandas as pd
import numpy as np
import secrets
from datetime import datetime, timezone

def simulate_bitcoin_prices(days=60, initial_price=50000.0, volatility=0.04, drift=0.001):
    """Simulate Bitcoin prices using Geometric Brownian Motion."""
    rng = np.random.default_rng(secrets.randbits(32))

    shocks = rng.normal(0, 1, days - 1)
    price_changes = np.exp((drift - 0.5 * volatility**2) + volatility * shocks)
    prices = initial_price * np.cumprod(np.insert(price_changes, 0, 1.0))

    start_date = datetime.now(timezone.utc) - pd.Timedelta(days=days-1)
    dates = pd.date_range(start=start_date, periods=days)

    df = pd.DataFrame({'Date': dates, 'Price': prices})
    return df

def calculate_moving_averages(df):
    """Calculate 7-day and 30-day Moving Averages."""
    df['MA7'] = df['Price'].rolling(window=7).mean()
    df['MA30'] = df['Price'].rolling(window=30).mean()
    return df

def run_trading_algorithm(df):
    """Implement Golden Cross trading algorithm."""
    dates = df['Date'].dt.strftime('%Y-%m-%d').values
    prices = df['Price'].values
    ma7s = df['MA7'].values
    ma30s = df['MA30'].values

    n = len(df)

    cash = 10000.0
    btc = 0.0

    actions = np.full(n, "HOLD", dtype=object)

    valid = pd.notna(ma7s) & pd.notna(ma30s)

    buys = np.zeros(n, dtype=bool)
    sells = np.zeros(n, dtype=bool)

    valid_cross = valid[1:] & valid[:-1]
    prev_ma7s = ma7s[:-1]
    prev_ma30s = ma30s[:-1]
    curr_ma7s = ma7s[1:]
    curr_ma30s = ma30s[1:]

    buys[1:] = valid_cross & (prev_ma7s <= prev_ma30s) & (curr_ma7s > curr_ma30s)
    sells[1:] = valid_cross & (prev_ma7s >= prev_ma30s) & (curr_ma7s < curr_ma30s)

    event_indices = np.where(buys | sells)[0]

    cash_arr = np.zeros(n, dtype=np.float64)
    btc_arr = np.zeros(n, dtype=np.float64)

    last_idx = 0

    for i in event_indices:
        cash_arr[last_idx:i] = cash
        btc_arr[last_idx:i] = btc

        price = prices[i]

        if buys[i] and cash > 0:
            btc = cash / price
            cash = 0.0
            actions[i] = f"BUY {btc:.4f} BTC"
        elif sells[i] and btc > 0:
            cash = btc * price
            actions[i] = f"SELL {btc:.4f} BTC"
            btc = 0.0

        cash_arr[i] = cash
        btc_arr[i] = btc
        last_idx = i + 1

    if last_idx < n:
        cash_arr[last_idx:] = cash
        btc_arr[last_idx:] = btc

    portfolio_value = cash_arr + (btc_arr * prices)

    return pd.DataFrame({
        'Date': dates,
        'Price': prices,
        'MA7': ma7s,
        'MA30': ma30s,
        'Action': actions,
        'Cash': cash_arr,
        'BTC': btc_arr,
        'Portfolio Value': portfolio_value
    })

if __name__ == "__main__":
    print("Simulating 60 days of Bitcoin prices...")
    df = simulate_bitcoin_prices(60)
    df = calculate_moving_averages(df)

    print("\nRunning Golden Cross trading algorithm...")
    ledger_df = run_trading_algorithm(df)

    print("\n--- Daily Ledger ---")
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)

    out_df = ledger_df.copy()
    out_df['Price'] = out_df['Price'].round(2)
    out_df['MA7'] = out_df['MA7'].round(2)
    out_df['MA30'] = out_df['MA30'].round(2)
    out_df['Portfolio Value'] = out_df['Portfolio Value'].round(2)

    print(out_df[['Date', 'Price', 'MA7', 'MA30', 'Action', 'Portfolio Value']])

    final_value = ledger_df.iloc[-1]['Portfolio Value']
    initial_value = 10000.0
    roi = ((final_value - initial_value) / initial_value) * 100

    print("\n--- Final Portfolio Performance ---")
    print(f"Initial Value: ${initial_value:.2f}")
    print(f"Final Value:   ${final_value:.2f}")
    print(f"Return on Investment (ROI): {roi:.2f}%")
