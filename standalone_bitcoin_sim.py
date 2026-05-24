import pandas as pd
import numpy as np
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
    dates = df['Date'].dt.strftime('%Y-%m-%d').tolist()
    prices = df['Price'].tolist()
    ma7s = df['MA7'].tolist()
    ma30s = df['MA30'].tolist()
    valid = (pd.notna(df['MA7']) & pd.notna(df['MA30'])).tolist()

    n = len(df)

    if n == 0:
        return pd.DataFrame({
            'Date': dates,
            'Price': prices,
            'MA7': ma7s,
            'MA30': ma30s,
            'Action': [],
            'Cash': [],
            'BTC': [],
            'Portfolio Value': []
        })

    actions = ["HOLD"] * n
    cashes = [0.0] * n
    btcs = [0.0] * n
    portfolios = [0.0] * n

    cash = 10000.0  # Initial capital
    btc = 0.0

    prev_ma7_val = ma7s[0]
    prev_ma30_val = ma30s[0]
    prev_valid = valid[0]

    for i in range(n):
        price = prices[i]
        ma7 = ma7s[i]
        ma30 = ma30s[i]
        v = valid[i]

        action = "HOLD"

        if i > 0 and v and prev_valid:
            # Golden Cross: MA7 crosses above MA30 -> BUY
            if prev_ma7_val <= prev_ma30_val and ma7 > ma30:
                if cash > 0:
                    btc = cash / price
                    cash = 0.0
                    action = f"BUY {btc:.4f} BTC"

            # Death Cross: MA7 crosses below MA30 -> SELL
            elif prev_ma7_val >= prev_ma30_val and ma7 < ma30:
                if btc > 0:
                    cash = btc * price
                    action = f"SELL {btc:.4f} BTC"
                    btc = 0.0

        actions[i] = action
        cashes[i] = cash
        btcs[i] = btc
        portfolios[i] = cash + (btc * price)

        prev_ma7_val = ma7
        prev_ma30_val = ma30
        prev_valid = v

    return pd.DataFrame({
        'Date': dates,
        'Price': prices,
        'MA7': ma7s,
        'MA30': ma30s,
        'Action': actions,
        'Cash': cashes,
        'BTC': btcs,
        'Portfolio Value': portfolios
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
