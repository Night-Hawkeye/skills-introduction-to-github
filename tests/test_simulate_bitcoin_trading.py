import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytest
from simulate_bitcoin_trading import run_trading_algorithm

def test_run_trading_algorithm_golden_cross():
    """Test that the algorithm buys on a Golden Cross."""
    dates = [datetime.today() - timedelta(days=5 - i) for i in range(5)]

    # Prices and MA are artificially constructed to trigger a BUY
    # Condition: prev_ma7 <= prev_ma30 AND ma7 > ma30
    df = pd.DataFrame({
        'Date': dates,
        'Price': [10000, 11000, 12000, 13000, 14000],
        'MA7': [9000, 9500, 10000, 11500, 13000],
        'MA30': [10000, 10000, 10000, 10000, 10000]
    })

    # At index 2: prev_ma7(9500) <= prev_ma30(10000)
    #             ma7(10000) <= ma30(10000) -> NO BUY
    # At index 3: prev_ma7(10000) <= prev_ma30(10000)
    #             ma7(11500) > ma30(10000) -> BUY triggered here

    ledger_df = run_trading_algorithm(df)

    # Row index 3 should be BUY
    assert "BUY" in ledger_df.loc[3, 'Action']

    # After BUY, BTC balance > 0
    assert ledger_df.loc[3, 'BTC'] > 0
    assert ledger_df.loc[3, 'Cash'] == 0.0

def test_run_trading_algorithm_death_cross():
    """Test that the algorithm sells on a Death Cross."""
    dates = [datetime.today() - timedelta(days=5 - i) for i in range(5)]

    # First, we need to BUY so we have BTC to SELL
    # Then we trigger a Death Cross
    df = pd.DataFrame({
        'Date': dates,
        'Price': [10000, 11000, 12000, 13000, 14000],
        'MA7': [9000, 11000, 11000, 9000, 8000],
        'MA30': [10000, 10000, 10000, 10000, 10000]
    })

    # At index 1: Golden Cross
    # prev_ma7(9000) <= prev_ma30(10000)
    # ma7(11000) > ma30(10000) -> BUY triggered

    # At index 3: Death Cross
    # prev_ma7(11000) >= prev_ma30(10000)
    # ma7(9000) < ma30(10000) -> SELL triggered

    ledger_df = run_trading_algorithm(df)

    # Row index 1 should be BUY
    assert "BUY" in ledger_df.loc[1, 'Action']

    # Row index 3 should be SELL
    assert "SELL" in ledger_df.loc[3, 'Action']

    # After SELL, Cash > 0 and BTC == 0
    assert ledger_df.loc[3, 'Cash'] > 0
    assert ledger_df.loc[3, 'BTC'] == 0.0

def test_run_trading_algorithm_hold():
    """Test that the algorithm holds when there's no cross."""
    dates = [datetime.today() - timedelta(days=3 - i) for i in range(3)]

    df = pd.DataFrame({
        'Date': dates,
        'Price': [10000, 11000, 12000],
        'MA7': [9000, 9500, 9800],
        'MA30': [10000, 10000, 10000]
    })

    ledger_df = run_trading_algorithm(df)

    # Should all be HOLD
    for action in ledger_df['Action']:
        assert action == "HOLD"
