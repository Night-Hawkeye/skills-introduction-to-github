import pytest
import numpy as np
import secrets
from datetime import datetime, timedelta
import pandas as pd
from bitcoin_trader import simulate_bitcoin_prices

def simulate_bitcoin_prices_vectorized(days=60, initial_price=50000, volatility=0.04, drift=0.001):
    """Simulate Bitcoin prices using Geometric Brownian Motion."""
    # Use seed 123 so there is actually enough movement to trigger a Golden Cross for demonstration
    np.random.seed(secrets.randbits(32))

    shocks = np.random.normal(0, 1, days - 1)
    price_changes = np.exp((drift - 0.5 * volatility**2) + volatility * shocks)
    prices = initial_price * np.cumprod(np.insert(price_changes, 0, 1.0))

    # Generate dates
    start_date = datetime.now() - timedelta(days=days-1)
    dates = pd.date_range(start=start_date, periods=days)

    df = pd.DataFrame({'Date': dates, 'Price': prices})
    return df

def test_benchmark_simulate_prices_original(benchmark):
    benchmark(simulate_bitcoin_prices, days=100000)

def test_benchmark_simulate_prices_vectorized(benchmark):
    benchmark(simulate_bitcoin_prices_vectorized, days=100000)
