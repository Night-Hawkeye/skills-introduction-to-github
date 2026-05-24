import pandas as pd
import numpy as np
import secrets
from datetime import datetime, timedelta

def simulate_bitcoin_prices(days=60, initial_price=50000.0, volatility=0.04, drift=0.001):
    """Simulate Bitcoin prices using Geometric Brownian Motion."""
    # Use secrets for secure randomness as per guidelines
    rng = np.random.default_rng(secrets.randbits(32))

    shocks = rng.normal(0, 1, days - 1)
    price_changes = np.exp((drift - 0.5 * volatility**2) + volatility * shocks)
    prices = np.concatenate(([initial_price], initial_price * np.cumprod(price_changes))).tolist()

    # Generate dates efficiently
    start_date = datetime.today() - timedelta(days=days - 1)
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
    ma7 = df['MA7'].values
    ma30 = df['MA30'].values

    valid = pd.notna(ma7) & pd.notna(ma30)
    valid_prev = np.roll(valid, 1)
    valid_prev[0] = False
    mask = valid & valid_prev

    prev_ma7 = np.roll(ma7, 1)
    prev_ma30 = np.roll(ma30, 1)

    buy_signal = mask & (prev_ma7 <= prev_ma30) & (ma7 > ma30)
    sell_signal = mask & (prev_ma7 >= prev_ma30) & (ma7 < ma30)

    # Calculate state (1 for holding BTC, 0 for holding Cash)
    s = pd.Series(np.nan, index=df.index)
    s[buy_signal] = 1
    s[sell_signal] = 0
    s = s.ffill().fillna(0)
    state_ffill = s.values

    # Calculate returns
    price_ratio = np.ones(len(df))
    price_ratio[1:] = prices[1:] / prices[:-1]

    port_returns = np.ones(len(df))
    port_returns[1:] = np.where(state_ffill[:-1] == 1, price_ratio[1:], 1.0)

    pv = 10000.0 * np.cumprod(port_returns)

    # Calculate Cash and BTC
    calc_cash = np.where(state_ffill == 1, 0.0, pv)
    calc_btc = np.where(state_ffill == 1, pv / prices, 0.0)

    # Generate actions
    state_change = np.zeros(len(df))
    state_change[1:] = state_ffill[1:] - state_ffill[:-1]
    is_buy = state_change == 1
    is_sell = state_change == -1

    actions = np.full(len(df), "HOLD", dtype=object)

    for i in np.where(is_buy)[0]:
        actions[i] = f"BUY {calc_btc[i]:.4f} BTC"
    for i in np.where(is_sell)[0]:
        actions[i] = f"SELL {calc_btc[i-1]:.4f} BTC"

    return pd.DataFrame({
        'Date': dates,
        'Price': prices,
        'MA7': ma7,
        'MA30': ma30,
        'Action': actions,
        'Cash': calc_cash,
        'BTC': calc_btc,
        'Portfolio Value': pv
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
