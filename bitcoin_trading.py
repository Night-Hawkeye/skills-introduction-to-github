import pandas as pd
import numpy as np
import secrets
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Optional

@dataclass
class SimulationConfig:
    days: int = 60
    initial_price: float = 50000.0
    volatility: float = 0.04
    drift: float = 0.001
    seed: Optional[int] = None


def simulate_bitcoin_prices(config: SimulationConfig = None):
    """Simulate Bitcoin prices using Geometric Brownian Motion."""
    if config is None:
        config = SimulationConfig()

    # Vectorized random number generation for performance
    seed = config.seed
    if seed is None:
        seed = secrets.randbits(128)
    rng = np.random.default_rng(seed)
    shocks = rng.normal(0, 1, config.days - 1)
    log_returns = (config.drift - 0.5 * config.volatility**2) + config.volatility * shocks
    prices = np.empty(config.days)
    prices[0] = config.initial_price
    if config.days > 1:
        prices[1:] = config.initial_price * np.exp(np.cumsum(log_returns))

    # Fast datetime generation
    start_date = datetime.now(timezone.utc) - timedelta(days=config.days - 1)
    dates = pd.date_range(start=start_date, periods=config.days)

    df = pd.DataFrame({'Date': dates, 'Price': prices})
    return df

def calculate_moving_averages(df):
    """Calculate 7-day and 30-day Moving Averages."""
    if len(df) == 0:
        df['MA7'] = pd.Series([], dtype=float)
        df['MA30'] = pd.Series([], dtype=float)
        return df

    df['MA7'] = df['Price'].rolling(window=7).mean()
    df['MA30'] = df['Price'].rolling(window=30).mean()
    return df

def _generate_signals(ma7, ma30, index):
    if len(ma7) == 0:
        return np.array([])
    valid = pd.notna(ma7) & pd.notna(ma30)

    prev_ma7 = np.roll(ma7, 1)
    prev_ma30 = np.roll(ma30, 1)
    prev_valid = np.roll(valid, 1)
    prev_valid[0] = False

    with np.errstate(invalid='ignore'):
        buy_signal = (prev_ma7 <= prev_ma30) & (ma7 > ma30) & valid & prev_valid
        sell_signal = (prev_ma7 >= prev_ma30) & (ma7 < ma30) & valid & prev_valid

    signals = pd.Series(np.nan, index=index)
    signals.loc[buy_signal] = 1
    signals.loc[sell_signal] = 0
    return signals.ffill().fillna(0).values

def _calculate_btc_returns(prices):
    btc_returns = np.zeros(len(prices))
    prev_prices = prices[:-1]
    curr_prices = prices[1:]
    safe_div = np.where(prev_prices != 0, prev_prices, 1.0)
    btc_returns[1:] = np.where(prev_prices != 0, (curr_prices - prev_prices) / safe_div, 0.0)
    return btc_returns

def _calculate_strategy_returns(btc_returns, position):
    prev_position = np.roll(position, 1)
    prev_position[0] = 0
    return btc_returns * prev_position

def _calculate_asset_holdings(portfolio_value, prices, position):
    safe_prices = np.where(prices != 0, prices, 1.0)
    btc_held = np.where((position == 1) & (prices != 0), portfolio_value / safe_prices, 0.0)
    cash_held = np.where(position == 0, portfolio_value, 0.0)
    return btc_held, cash_held

def _calculate_portfolio(prices, position, initial_cash):
    if len(prices) == 0:
        return np.array([]), np.array([]), np.array([])

    btc_returns = _calculate_btc_returns(prices)
    strat_returns = _calculate_strategy_returns(btc_returns, position)

    portfolio_value = initial_cash * np.cumprod(1 + strat_returns)

    # Zero out portfolio value if it goes negative due to weird floating point (not typical but safe)
    portfolio_value = np.maximum(portfolio_value, 0.0)

    btc_held, cash_held = _calculate_asset_holdings(portfolio_value, prices, position)

    return portfolio_value, cash_held, btc_held

def _generate_actions(position, portfolio_value, btc_held):
    if len(position) == 0:
        return np.array([])
    prev_position = np.roll(position, 1)
    prev_position[0] = 0

    prev_btc_held = np.roll(btc_held, 1)
    prev_btc_held[0] = 0.0

    is_buy = (position == 1) & (prev_position == 0) & (portfolio_value > 0)
    # We sell if we were holding BTC and now we are not. And we only sell if we actually held BTC.
    is_sell = (position == 0) & (prev_position == 1) & (prev_btc_held > 0)

    action = np.full(len(position), "HOLD", dtype=object)

    buy_indices = np.where(is_buy)[0]
    sell_indices = np.where(is_sell)[0]

    if len(buy_indices) > 0:
        action[buy_indices] = [f"BUY {val:.4f} BTC" for val in btc_held[buy_indices].tolist()]

    if len(sell_indices) > 0:
        action[sell_indices] = [f"SELL {val:.4f} BTC" for val in prev_btc_held[sell_indices].tolist()]

    return action

def run_trading_algorithm(df):
    """Implement Golden Cross trading algorithm."""
    if len(df) == 0:
        return pd.DataFrame()

    initial_cash = 10000.0

    dates = df['Date'].dt.strftime('%Y-%m-%d').values
    prices = df['Price'].values
    ma7 = df['MA7'].values
    ma30 = df['MA30'].values

    position = _generate_signals(ma7, ma30, df.index)
    portfolio_value, cash_held, btc_held = _calculate_portfolio(prices, position, initial_cash)
    action = _generate_actions(position, portfolio_value, btc_held)

    return pd.DataFrame({
        'Date': dates,
        'Price': prices,
        'MA7': ma7,
        'MA30': ma30,
        'Action': action,
        'Cash': cash_held,
        'BTC': btc_held,
        'Portfolio Value': portfolio_value
    })

if __name__ == "__main__":
    print("Simulating 60 days of Bitcoin prices...")
    df = simulate_bitcoin_prices(SimulationConfig(days=60))
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
