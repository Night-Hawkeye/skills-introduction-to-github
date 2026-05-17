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

    # Use .values for performant loop over DataFrame
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
