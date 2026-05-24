import pandas as pd
import numpy as np
import secrets
from datetime import datetime, timedelta, timezone

def simulate_bitcoin_prices(days=60, initial_price=50000, volatility=0.04, drift=0.001):
    """Simulate Bitcoin prices using Geometric Brownian Motion."""
    # Use secrets for cryptographic secure random generation
    seed = secrets.randbits(32)
    rng = np.random.default_rng(seed)

    shocks = rng.normal(0, 1, days - 1)
    price_changes = np.exp((drift - 0.5 * volatility**2) + volatility * shocks)
    prices = initial_price * np.cumprod(np.insert(price_changes, 0, 1.0))

    # Generate dates - using timezone.utc instead of utcnow()
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

    # Pre-extract numpy arrays
    dates = df['Date'].dt.strftime('%Y-%m-%d').values
    prices = df['Price'].to_numpy()
    ma7 = df['MA7'].to_numpy()
    ma30 = df['MA30'].to_numpy()

    n = len(df)

    # Calculate shifted values for previous day
    prev_ma7 = np.roll(ma7, 1)
    prev_ma30 = np.roll(ma30, 1)

    # Find valid indices
    valid_mask = ~np.isnan(ma7) & ~np.isnan(ma30) & ~np.isnan(prev_ma7) & ~np.isnan(prev_ma30)
    if n > 0:
        valid_mask[0] = False

    # Find buy and sell signals
    buy_signals = valid_mask & (prev_ma7 <= prev_ma30) & (ma7 > ma30)
    sell_signals = valid_mask & (prev_ma7 >= prev_ma30) & (ma7 < ma30)

    signal_indices = np.nonzero(buy_signals | sell_signals)[0]

    # Pre-allocate output arrays
    cash_arr = np.empty(n, dtype=np.float64)
    btc_arr = np.empty(n, dtype=np.float64)
    action_arr = np.full(n, "HOLD", dtype=object)

    last_idx = 0
    for idx in signal_indices:
        # Fill arrays up to idx with current cash/btc
        cash_arr[last_idx:idx] = cash
        btc_arr[last_idx:idx] = btc

        if buy_signals[idx] and cash > 0: # BUY
            price = prices[idx]
            btc = cash / price
            cash = 0.0
            action_arr[idx] = f"BUY {btc:.4f} BTC"
        elif sell_signals[idx] and btc > 0: # SELL
            price = prices[idx]
            cash = btc * price
            action_arr[idx] = f"SELL {btc:.4f} BTC"
            btc = 0.0

        cash_arr[idx] = cash
        btc_arr[idx] = btc
        last_idx = idx + 1

    # Fill remaining
    cash_arr[last_idx:] = cash
    btc_arr[last_idx:] = btc

    # Portfolio value
    portfolio_value = cash_arr + (btc_arr * prices)

    # Create final dataframe using a dictionary
    return pd.DataFrame({
        'Date': dates,
        'Price': prices,
        'MA7': ma7,
        'MA30': ma30,
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
