import pandas as pd
import numpy as np
import secrets
from datetime import datetime, timedelta, timezone

def simulate_bitcoin_prices(days=60, initial_price=50000, volatility=0.04, drift=0.001):
    """Simulate Bitcoin prices using Geometric Brownian Motion."""
    # Use seed 123 so there is actually enough movement to trigger a Golden Cross for demonstration
    rng = np.random.default_rng(123)

    shocks = rng.normal(0, 1, days - 1)
    price_changes = np.exp((drift - 0.5 * volatility**2) + volatility * shocks)
    prices = initial_price * np.cumprod(np.insert(price_changes, 0, 1.0))

    # Fast datetime generation
    start_date = datetime.now(timezone.utc) - timedelta(days=days-1)
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
    cash = 10000.0  # Initial capital
    btc = 0.0

    ledger = []

    n = len(df)
    if n == 0:
        return pd.DataFrame(columns=['Date', 'Price', 'MA7', 'MA30', 'Action', 'Cash', 'BTC', 'Portfolio Value'])

    dates = df['Date'].dt.strftime('%Y-%m-%d').values
    prices = df['Price'].values
    ma7s = df['MA7'].values
    ma30s = df['MA30'].values

    # Pre compute boolean mask to safely handle NaNs
    valid = pd.notna(ma7s) & pd.notna(ma30s)

    prev_ma7 = np.roll(ma7s, 1)
    prev_ma30 = np.roll(ma30s, 1)
    prev_ma7[0] = ma7s[0]
    prev_ma30[0] = ma30s[0]
    prev_valid = np.roll(valid, 1)
    prev_valid[0] = False

    # Vectorized signals
    with np.errstate(invalid='ignore'):
        buy_signals = valid & prev_valid & (prev_ma7 <= prev_ma30) & (ma7s > ma30s)
        sell_signals = valid & prev_valid & (prev_ma7 >= prev_ma30) & (ma7s < ma30s)

    signal_indices = np.where(buy_signals | sell_signals)[0]

    # Initialize states
    cash_arr = np.zeros(n)
    btc_arr = np.zeros(n)
    action_arr = np.full(n, "HOLD", dtype=object)

    current_cash = cash
    current_btc = btc

    last_idx = 0
    for idx in signal_indices:
        # Fill intermediate non-signal states
        cash_arr[last_idx:idx] = current_cash
        btc_arr[last_idx:idx] = current_btc

        price = prices[idx]
        if buy_signals[idx]:
            if current_cash > 0:
                current_btc = current_cash / price
                current_cash = 0.0
                action_arr[idx] = f"BUY {current_btc:.4f} BTC"
        elif sell_signals[idx]:
            if current_btc > 0:
                current_cash = current_btc * price
                action_arr[idx] = f"SELL {current_btc:.4f} BTC"
                current_btc = 0.0

        cash_arr[idx] = current_cash
        btc_arr[idx] = current_btc
        last_idx = idx + 1

    # Fill remaining
    cash_arr[last_idx:] = current_cash
    btc_arr[last_idx:] = current_btc

    portfolio_value = cash_arr + (btc_arr * prices)

    return pd.DataFrame({
        'Date': dates,
        'Price': prices,
        'MA7': ma7s,
        'MA30': ma30s,
        'Action': action_arr,
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
