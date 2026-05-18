import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import pytest
from unittest.mock import patch
from bitcoin_algo import simulate_bitcoin_prices, calculate_moving_averages, run_trading_algorithm

def test_simulate_bitcoin_prices():
    """Test simulating Bitcoin prices generates correct dataframe."""
    df = simulate_bitcoin_prices(days=30, initial_price=40000)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 30
    assert list(df.columns) == ['Date', 'Price']

    # Check that initial price is correct
    assert df.iloc[0]['Price'] == 40000

    # Check dates are correctly generated
    assert df.iloc[-1]['Date'].date() == datetime.now(timezone.utc).date()

def test_calculate_moving_averages():
    """Test calculating moving averages."""
    dates = pd.date_range(end=datetime.now(timezone.utc), periods=10)
    prices = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    df = pd.DataFrame({'Date': dates, 'Price': prices})
    df_ma = calculate_moving_averages(df)

    # MA7 starts at index 6
    assert pd.isna(df_ma.iloc[5]['MA7'])
    assert not pd.isna(df_ma.iloc[6]['MA7'])
    assert df_ma.iloc[6]['MA7'] == sum(prices[0:7]) / 7
    assert df_ma.iloc[9]['MA7'] == sum(prices[3:10]) / 7

    # MA30 starts at index 29 (all will be NaN here since len=10)
    assert pd.isna(df_ma.iloc[-1]['MA30'])

def test_run_trading_algorithm_golden_cross():
    """Test that the algorithm buys on a Golden Cross."""
    dates = pd.date_range(end=datetime.now(timezone.utc), periods=5)

    df = pd.DataFrame({
        'Date': dates,
        'Price': [10000, 11000, 12000, 13000, 14000],
        'MA7': [9000, 9500, 10000, 11500, 13000],
        'MA30': [10000, 10000, 10000, 10000, 10000]
    })

    ledger_df = run_trading_algorithm(df)

    assert len(ledger_df) == 5
    # Row index 3 should be BUY
    assert "BUY" in ledger_df.loc[3, 'Action']

    # After BUY, BTC balance > 0
    assert ledger_df.loc[3, 'BTC'] > 0
    assert ledger_df.loc[3, 'Cash'] == 0.0

def test_run_trading_algorithm_death_cross():
    """Test that the algorithm sells on a Death Cross."""
    dates = pd.date_range(end=datetime.now(timezone.utc), periods=5)

    df = pd.DataFrame({
        'Date': dates,
        'Price': [10000, 11000, 12000, 13000, 14000],
        'MA7': [9000, 11000, 11000, 9000, 8000],
        'MA30': [10000, 10000, 10000, 10000, 10000]
    })

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
    dates = pd.date_range(end=datetime.now(timezone.utc), periods=3)

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
