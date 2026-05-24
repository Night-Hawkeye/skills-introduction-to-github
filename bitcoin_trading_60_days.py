import pandas as pd
import numpy as np
import secrets
from datetime import datetime, timedelta, timezone

def simulate_bitcoin_prices(days=60, initial_price=50000, volatility=0.04, drift=0.001):
    """Simulate Bitcoin prices using Geometric Brownian Motion."""
    seed = secrets.randbits(32)
    rng = np.random.default_rng(seed)

    shocks = rng.normal(0, 1, days - 1)
    price_changes = np.exp((drift - 0.5 * volatility**2) + volatility * shocks)
    prices = initial_price * np.cumprod(np.insert(price_changes, 0, 1.0))

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

    dates = df['Date'].dt.strftime('%Y-%m-%d').values
    prices = df['Price'].values
    ma7s = df['MA7'].values
    ma30s = df['MA30'].values

    # Pre-calculate condition for optimization
    valid = pd.notna(ma7s) & pd.notna(ma30s)

    n = len(df)

    # Pre-allocate arrays for fastest appending
    actions = np.full(n, "HOLD", dtype=object)
    cashes = np.zeros(n)
    btcs = np.zeros(n)
    portfolio_values = np.zeros(n)

    # We can pre-calculate the complete combined valid mask for i and i-1
    # which avoids the `i > 0 and valid[i] and valid[i-1]` in the loop
    valid_both = valid & np.roll(valid, 1)
    valid_both[0] = False

    for i in range(n):
        if valid_both[i]:
            prev_ma7 = ma7s[i-1]
            prev_ma30 = ma30s[i-1]
            ma7 = ma7s[i]
            ma30 = ma30s[i]

            # Golden Cross: MA7 crosses above MA30 -> BUY
            if prev_ma7 <= prev_ma30 and ma7 > ma30:
                if cash > 0:
                    btc = cash / prices[i]
                    cash = 0.0
                    actions[i] = f"BUY {btc:.4f} BTC"

            # Death Cross: MA7 crosses below MA30 -> SELL
            elif prev_ma7 >= prev_ma30 and ma7 < ma30:
                if btc > 0:
                    cash = btc * prices[i]
                    actions[i] = f"SELL {btc:.4f} BTC"
                    btc = 0.0

        cashes[i] = cash
        btcs[i] = btc
        portfolio_values[i] = cash + btc * prices[i]

    return pd.DataFrame({
        'Date': dates,
        'Price': prices,
        'MA7': ma7s,
        'MA30': ma30s,
        'Action': actions,
        'Cash': cashes,
        'BTC': btcs,
        'Portfolio Value': portfolio_values
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
