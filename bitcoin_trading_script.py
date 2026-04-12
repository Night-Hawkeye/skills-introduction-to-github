import pandas as pd
import numpy as np
import secrets
from datetime import datetime, timedelta

def simulate_bitcoin_prices(days=60, initial_price=50000.0, volatility=0.04, drift=0.001):
    """Simulate Bitcoin prices using Geometric Brownian Motion."""
    # Use secrets for secure randomness
    rng = np.random.default_rng(secrets.randbits(32))

    prices = [initial_price]
    for _ in range(1, days):
        shock = rng.normal(0, 1)
        price_change = np.exp((drift - 0.5 * volatility**2) + volatility * shock)
        prices.append(prices[-1] * price_change)

    # Generate dates
    dates = [datetime.today() - timedelta(days=days - i - 1) for i in range(days)]
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
