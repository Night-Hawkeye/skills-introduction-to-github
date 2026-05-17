import pandas as pd
import numpy as np
import secrets
from datetime import datetime, timedelta, timezone

def simulate_bitcoin_prices(days=60, initial_price=50000.0, volatility=0.04, drift=0.001):
    """Simulate Bitcoin prices using Geometric Brownian Motion."""
    # Secure randomness
    seed = secrets.randbits(32)
    rng = np.random.default_rng(seed)

    # Generate prices using GBM
    shocks = rng.normal(0, 1, days - 1)
    price_changes = np.exp((drift - 0.5 * volatility**2) + volatility * shocks)
    prices = initial_price * np.cumprod(np.insert(price_changes, 0, 1.0))

    # Generate dates using timezone-aware datetime and pandas date_range for efficiency
    start_date = datetime.now(timezone.utc) - timedelta(days=days-1)
    dates = pd.date_range(start=start_date, periods=days)

    df = pd.DataFrame({'Date': dates, 'Price': prices})
    return df

def calculate_moving_averages(df):
    """Calculate 7-day and 30-day Moving Averages."""
    df['MA7'] = df['Price'].rolling(window=7).mean()
    df['MA30'] = df['Price'].rolling(window=30).mean()
    return df

def run_trading_algorithm(df, initial_balance=10000.0):
    balance = initial_balance
    btc_held = 0.0
    portfolio_value = balance

    print("\n--- Daily Trading Ledger ---")

    # Track daily portfolio value
    df['Portfolio_Value'] = 0.0
    df['Position'] = 'Hold'

    for i in range(len(df)):
        current_price = df['Price'].iloc[i]
        date = df['Date'].iloc[i]

        # Golden Cross Logic requires at least 30 days to have both MAs
        if i >= 30:
            ma7_current = df['MA7'].iloc[i]
            ma30_current = df['MA30'].iloc[i]
            ma7_prev = df['MA7'].iloc[i-1]
            ma30_prev = df['MA30'].iloc[i-1]

            # Buy Signal: MA7 crosses ABOVE MA30
            if ma7_prev <= ma30_prev and ma7_current > ma30_current and balance > 0:
                btc_to_buy = balance / current_price
                btc_held += btc_to_buy
                balance = 0.0
                df.at[i, 'Position'] = 'Buy'
                print(f"{date.strftime('%Y-%m-%d')} - BUY: {btc_to_buy:.6f} BTC at ${current_price:.2f}")

            # Sell Signal: MA7 crosses BELOW MA30
            elif ma7_prev >= ma30_prev and ma7_current < ma30_current and btc_held > 0:
                proceeds = btc_held * current_price
                balance += proceeds
                btc_held = 0.0
                df.at[i, 'Position'] = 'Sell'
                print(f"{date.strftime('%Y-%m-%d')} - SELL: Proceeds ${proceeds:.2f} at ${current_price:.2f}")

        # Calculate daily portfolio value
        portfolio_value = balance + (btc_held * current_price)
        df.at[i, 'Portfolio_Value'] = portfolio_value

    print("\n--- Final Portfolio Performance ---")
    print(f"Initial Balance: ${initial_balance:.2f}")
    print(f"Final Balance:   ${portfolio_value:.2f}")
    roi = ((portfolio_value - initial_balance) / initial_balance) * 100
    print(f"Return on Invest: {roi:.2f}%")

    return df

if __name__ == '__main__':
    print("Simulating Bitcoin Prices...")
    df = simulate_bitcoin_prices(days=60)

    print("Calculating Moving Averages...")
    df = calculate_moving_averages(df)

    run_trading_algorithm(df)
