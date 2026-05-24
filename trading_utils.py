import pandas as pd
import numpy as np
import secrets
from datetime import datetime, timezone

def simulate_bitcoin_prices(days=60, initial_price=50000.0, volatility=0.04, drift=0.001, seed=None):
    """Simulate Bitcoin prices using Geometric Brownian Motion."""
    if seed is None:
        seed = secrets.randbits(32)
    # Use provided seed or secrets for secure randomness
    rng = np.random.default_rng(seed)

    days_to_simulate = max(0, days - 1)
    if days_to_simulate > 0:
        shocks = rng.normal(0, 1, days_to_simulate)
        price_changes = np.exp((drift - 0.5 * volatility**2) + volatility * shocks)
        prices = np.concatenate(([initial_price], initial_price * np.cumprod(price_changes))).tolist()
    else:
        prices = [initial_price]

    # Fast datetime generation
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
    cash = 10000.0  # Initial capital
    btc = 0.0

    ledger = []

    # Use .values for performant loop over DataFrame
    dates = df['Date'].dt.strftime('%Y-%m-%d').values
    prices = df['Price'].values
    ma7s = df['MA7'].values
    ma30s = df['MA30'].values

    valid = pd.notna(ma7s) & pd.notna(ma30s)

    for i in range(len(df)):
        date = dates[i]
        price = prices[i]
        ma7 = ma7s[i]
        ma30 = ma30s[i]

        action = "HOLD"

        if i > 0 and valid[i] and valid[i-1]:
            prev_ma7 = ma7s[i-1]
            prev_ma30 = ma30s[i-1]

            # Golden Cross: MA7 crosses above MA30 -> BUY
            if prev_ma7 <= prev_ma30 and ma7 > ma30:
                if cash > 0:
                    btc = cash / price
                    cash = 0.0
                    action = f"BUY {btc:.4f} BTC"

            # Death Cross: MA7 crosses below MA30 -> SELL
            elif prev_ma7 >= prev_ma30 and ma7 < ma30:
                if btc > 0:
                    cash = btc * price
                    action = f"SELL {btc:.4f} BTC"
                    btc = 0.0

        portfolio_value = cash + (btc * price)
        ledger.append({
            'Date': date,
            'Price': price,
            'MA7': ma7,
            'MA30': ma30,
            'Action': action,
            'Cash': cash,
            'BTC': btc,
            'Portfolio Value': portfolio_value
        })

    return pd.DataFrame(ledger)
