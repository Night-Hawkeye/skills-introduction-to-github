import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import unittest.mock

# Import functions from the target script
from bitcoin_task import simulate_bitcoin_prices, calculate_moving_averages, run_trading_algorithm

def test_simulate_bitcoin_prices():
    """Test that simulate_bitcoin_prices returns a DataFrame with expected shape and columns."""
    days = 60
    df = simulate_bitcoin_prices(days=days)
    assert isinstance(df, pd.DataFrame)
    assert 'Date' in df.columns
    assert 'Price' in df.columns
    assert len(df) == days

def test_calculate_moving_averages():
    """Test that moving averages are calculated correctly."""
    # Create dummy data
    prices = np.array([10, 12, 15, 14, 18, 20, 22, 24, 25, 23])
    dates = pd.date_range(start=datetime.now(), periods=10)
    df = pd.DataFrame({'Date': dates, 'Price': prices})

    result_df = calculate_moving_averages(df)

    assert 'MA7' in result_df.columns
    assert 'MA30' in result_df.columns

    # Check that first 6 values of MA7 are NaN (window=7)
    assert result_df['MA7'].iloc[:6].isna().all()
    # Check that 7th value is the mean of first 7 elements
    assert np.isclose(result_df['MA7'].iloc[6], prices[:7].mean())

    # MA30 should be all NaN since len(df) < 30
    assert result_df['MA30'].isna().all()

def test_run_trading_algorithm_golden_cross():
    """Test that trading algorithm successfully buys when MA7 crosses above MA30."""
    dates = pd.date_range(start=datetime.now(), periods=3)

    # Day 0: MA7 < MA30 (Setup)
    # Day 1: MA7 crosses above MA30 (Golden Cross -> BUY)
    # Day 2: MA7 > MA30 (HOLD)
    data = {
        'Date': dates,
        'Price': [100.0, 100.0, 100.0],
        'MA7':   [40.0, 60.0, 70.0],
        'MA30':  [50.0, 50.0, 50.0]
    }
    df = pd.DataFrame(data)

    ledger = run_trading_algorithm(df)

    assert len(ledger) == 3
    # Day 0 should hold
    assert ledger.iloc[0]['Action'] == "HOLD"
    # Day 1 should BUY
    assert "BUY" in ledger.iloc[1]['Action']
    assert ledger.iloc[1]['Cash'] == 0.0
    assert ledger.iloc[1]['BTC'] == 10000.0 / 100.0 # Initial cash / price
    # Day 2 should HOLD
    assert ledger.iloc[2]['Action'] == "HOLD"
    assert ledger.iloc[2]['Cash'] == 0.0

def test_run_trading_algorithm_death_cross():
    """Test that trading algorithm successfully sells when MA7 crosses below MA30."""
    dates = pd.date_range(start=datetime.now(), periods=4)

    # We need to simulate a BUY first, then a SELL
    # Day 0: MA7 < MA30 (Setup)
    # Day 1: MA7 crosses above MA30 (Golden Cross -> BUY)
    # Day 2: MA7 < MA30 (Death Cross -> SELL)
    # Day 3: MA7 < MA30 (HOLD)
    data = {
        'Date': dates,
        'Price': [100.0, 100.0, 200.0, 200.0],
        'MA7':   [40.0, 60.0, 40.0, 30.0],
        'MA30':  [50.0, 50.0, 50.0, 50.0]
    }
    df = pd.DataFrame(data)

    ledger = run_trading_algorithm(df)

    assert len(ledger) == 4
    # Day 1 should BUY
    assert "BUY" in ledger.iloc[1]['Action']
    # Day 2 should SELL
    assert "SELL" in ledger.iloc[2]['Action']
    assert ledger.iloc[2]['BTC'] == 0.0
    # Cash should be (10000 / 100) * 200 = 20000
    assert ledger.iloc[2]['Cash'] == 20000.0
    # Day 3 should HOLD
    assert ledger.iloc[3]['Action'] == "HOLD"
