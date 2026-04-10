import pytest
import pandas as pd
from bitcoin_trader import simulate_bitcoin_prices

def test_simulate_bitcoin_prices():
    df = simulate_bitcoin_prices(days=10)
    assert len(df) == 10
    assert 'Date' in df.columns
    assert 'Price' in df.columns
