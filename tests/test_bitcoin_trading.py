import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import secrets
from unittest.mock import patch

# Import the function to be tested
from bitcoin_trading import simulate_bitcoin_prices

def test_simulate_bitcoin_prices_basic():
    """Test the basic properties of the returned DataFrame."""
    days = 30
    initial_price = 40000
    df = simulate_bitcoin_prices(days=days, initial_price=initial_price, volatility=0.05, drift=0.002)

    # Check type
    assert isinstance(df, pd.DataFrame)

    # Check columns
    assert 'Date' in df.columns
    assert 'Price' in df.columns

    # Check shape
    assert df.shape == (days, 2)

    # Check initial price
    assert df.iloc[0]['Price'] == initial_price

@patch('secrets.randbits')
@patch('bitcoin_trading.datetime')
def test_simulate_bitcoin_prices_determinism(mock_datetime, mock_randbits):
    """Test that setting a specific random seed makes the output deterministic."""
    # Mock secrets.randbits to return a fixed seed
    mock_randbits.return_value = 42

    # Mock datetime to return a fixed start date so 'Date' column matches
    fixed_date = datetime(2023, 1, 1, 12, 0, 0)
    mock_datetime.now.return_value = fixed_date

    df1 = simulate_bitcoin_prices(days=30, initial_price=50000, volatility=0.04, drift=0.001)
    df2 = simulate_bitcoin_prices(days=30, initial_price=50000, volatility=0.04, drift=0.001)

    # The mocked seed should make the generated prices identical
    pd.testing.assert_frame_equal(df1, df2)
    assert mock_randbits.call_count == 2

def test_simulate_bitcoin_prices_volatility_zero():
    """Test that when volatility is 0, the price change is deterministic based on drift."""
    days = 10
    initial_price = 10000
    drift = 0.005
    volatility = 0.0

    df = simulate_bitcoin_prices(days=days, initial_price=initial_price, volatility=volatility, drift=drift)

    # When volatility is 0: shock is ignored.
    # price_change = np.exp((drift - 0.5 * 0**2) + 0 * shock) = np.exp(drift)
    expected_price_change = np.exp(drift)

    prices = df['Price'].values
    for i in range(1, days):
        assert np.isclose(prices[i], prices[i-1] * expected_price_change)

def test_simulate_bitcoin_prices_dates():
    """Test that the generated dates cover the expected range and are sequential."""
    days = 15
    df = simulate_bitcoin_prices(days=days)

    dates = df['Date'].tolist()

    # Check that we have the right number of dates
    assert len(dates) == days

    # Check that dates are sequential and 1 day apart
    for i in range(1, days):
        assert (dates[i] - dates[i-1]).days == 1

    # Check that the last date is approximately today (within a day)
    # The function uses datetime.now() for the latest date.
    now = datetime.now()
    assert (now - dates[-1]).total_seconds() < 86400  # less than 24 hours difference

def test_simulate_bitcoin_prices_negative_days():
    """Test behavior with invalid input for days."""
    # Actually, we should test what it actually does. If it raises ValueError, we catch it.
    with pytest.raises(ValueError, match="All arrays must be of the same length"):
        simulate_bitcoin_prices(days=0)

def test_simulate_bitcoin_prices_one_day():
    """Test behavior with just 1 day."""
    df = simulate_bitcoin_prices(days=1)

    assert len(df) == 1
    assert df.iloc[0]['Price'] == 50000
