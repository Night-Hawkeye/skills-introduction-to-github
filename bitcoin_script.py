import pandas as pd
import numpy as np
import secrets
from datetime import datetime, timedelta, timezone

def simulate_bitcoin_prices(days=60, initial_price=50000.0, volatility=0.04, drift=0.001):
    """Simulate Bitcoin prices using Geometric Brownian Motion."""
    # Use secrets for secure randomness
    np.random.seed(secrets.randbits(32))

    prices = [initial_price]
    for _ in range(1, days):
        shock = np.random.normal(0, 1)
        price_change = np.exp((drift - 0.5 * volatility**2) + volatility * shock)
        prices.append(prices[-1] * price_change)

    # Generate dates using timezone-aware datetime
    start_date = datetime.now(timezone.utc) - timedelta(days=days - 1)
    dates = [start_date + timedelta(days=i) for i in range(days)]

    df = pd.DataFrame({'Date': dates, 'Price': prices})
    return df

def calculate_moving_averages(df):
    """Calculate 7-day and 30-day Moving Averages."""
    df['MA7'] = df['Price'].rolling(window=7).mean()
    df['MA30'] = df['Price'].rolling(window=30).mean()
    return df

def run_trading_algorithm(df):
    """Implement Golden Cross trading algorithm."""
    cash = 10000.0  # Initial capital
    btc = 0.0

    ledger = []

    df = df.copy()

    prev_ma7 = df['MA7'].shift(1)
    prev_ma30 = df['MA30'].shift(1)

    valid = df['MA7'].notna() & df['MA30'].notna() & prev_ma7.notna() & prev_ma30.notna()

    buy_mask = valid & (prev_ma7 <= prev_ma30) & (df['MA7'] > df['MA30'])
    sell_mask = valid & (prev_ma7 >= prev_ma30) & (df['MA7'] < df['MA30'])

    actions = np.full(len(df), "HOLD", dtype=object)
    cashes = np.full(len(df), cash)
    btcs = np.full(len(df), btc)

    signal_indices = np.where(buy_mask | sell_mask)[0]
    prices = df['Price'].values

    last_idx = 0
    for i in signal_indices:
        cashes[last_idx:i] = cash
        btcs[last_idx:i] = btc

        price = prices[i]
        if buy_mask[i] and cash > 0:
            btc = cash / price
            cash = 0.0
            actions[i] = f"BUY {btc:.4f} BTC"
        elif sell_mask[i] and btc > 0:
            sold_btc = btc
            cash = btc * price
            btc = 0.0
            actions[i] = f"SELL {sold_btc:.4f} BTC"

        cashes[i] = cash
        btcs[i] = btc
        last_idx = i + 1

    cashes[last_idx:] = cash
    btcs[last_idx:] = btc

    df['Action'] = actions
    df['Cash'] = cashes
    df['BTC'] = btcs
    df['Portfolio Value'] = cashes + btcs * prices

    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

    return df[['Date', 'Price', 'MA7', 'MA30', 'Action', 'Cash', 'BTC', 'Portfolio Value']]

if __name__ == "__main__":
    print("Simulating 60 days of Bitcoin prices...")
    df = simulate_bitcoin_prices(60)
    df = calculate_moving_averages(df)

    print("\nRunning Golden Cross trading algorithm...")
    ledger_df = run_trading_algorithm(df)

    print("\n--- Daily Ledger ---")
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)

    # Format floats for cleaner output
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
