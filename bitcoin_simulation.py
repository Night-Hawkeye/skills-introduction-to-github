import numpy as np
import pandas as pd
import secrets

def simulate_bitcoin_prices(days=60, initial_price=50000, volatility=0.04, drift=0.001):
    """Simulate Bitcoin prices using Geometric Brownian Motion."""
    # Use dynamically generated secure seed
    rng = np.random.default_rng(secrets.randbits(128))

    shocks = rng.normal(0, 1, days - 1)
    price_changes = np.exp((drift - 0.5 * volatility**2) + volatility * shocks)
    prices = initial_price * np.cumprod(np.insert(price_changes, 0, 1.0))
    return pd.DataFrame({'Price': prices})
