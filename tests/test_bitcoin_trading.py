import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta
from bitcoin_trader import run_trading_algorithm, calculate_moving_averages

def create_mock_df(prices, ma7s, ma30s):
    dates = [pd.Timestamp(2023, 1, 1) + pd.Timedelta(days=i) for i in range(len(prices))]
    return pd.DataFrame({
        'Date': pd.Series(dates),
        'Price': prices,
        'MA7': ma7s,
        'MA30': ma30s
    })

def test_run_trading_algorithm_hold():
    # MA7 is always below MA30
    df = create_mock_df(
        prices=[10000, 10000, 10000],
        ma7s=[9000, 9000, 9000],
        ma30s=[11000, 11000, 11000]
    )
    result = run_trading_algorithm(df)

    assert len(result) == 3
    assert all(result['Action'] == "HOLD")
    assert all(result['Cash'] == 10000.0)
    assert all(result['BTC'] == 0.0)
    assert all(result['Portfolio Value'] == 10000.0)

def test_run_trading_algorithm_golden_cross():
    # MA7 crosses above MA30
    df = create_mock_df(
        prices=[10000, 20000, 30000],
        ma7s=[9000, 12000, 15000],
        ma30s=[11000, 11000, 11000]
    )

    result = run_trading_algorithm(df)

    assert result.iloc[0]['Action'] == "HOLD"
    assert "BUY" in result.iloc[1]['Action']
    assert result.iloc[2]['Action'] == "HOLD"

    assert result.iloc[1]['Cash'] == 0.0
    assert result.iloc[1]['BTC'] == 10000.0 / 20000.0  # 0.5 BTC
    assert result.iloc[2]['Portfolio Value'] == 0.5 * 30000.0

def test_run_trading_algorithm_death_cross():
    # Golden cross then Death cross
    df = create_mock_df(
        prices=[10000, 20000, 10000],
        ma7s=[9000, 12000, 9000],
        ma30s=[11000, 11000, 11000]
    )

    result = run_trading_algorithm(df)

    assert result.iloc[0]['Action'] == "HOLD"
    assert "BUY" in result.iloc[1]['Action']
    assert "SELL" in result.iloc[2]['Action']

    assert result.iloc[2]['BTC'] == 0.0
    # Bought 0.5 BTC at 20000
    # Sold 0.5 BTC at 10000 -> 5000 cash
    assert result.iloc[2]['Cash'] == 5000.0
    assert result.iloc[2]['Portfolio Value'] == 5000.0

def test_run_trading_algorithm_nans():
    # Start with NaNs (like real MA calculation)
    df = create_mock_df(
        prices=[10000, 10000, 20000],
        ma7s=[np.nan, np.nan, 12000],
        ma30s=[np.nan, 11000, 11000]
    )

    result = run_trading_algorithm(df)

    assert all(result['Action'] == "HOLD")
    assert all(result['Cash'] == 10000.0)

def test_run_trading_algorithm_death_cross_no_btc():
    # Death cross when we have no BTC (because we never bought)
    df = create_mock_df(
        prices=[10000, 10000],
        ma7s=[12000, 9000],
        ma30s=[11000, 11000]
    )

    result = run_trading_algorithm(df)

    assert result.iloc[0]['Action'] == "HOLD"
    assert result.iloc[1]['Action'] == "HOLD" # We have no BTC to sell, so action remains HOLD
    assert result.iloc[1]['Cash'] == 10000.0

def test_calculate_moving_averages():
    # Create a deterministic mock DataFrame with 34 days of data
    # Prices increment by 10 each day for easy mental math: [10.0, 20.0, 30.0, ... 340.0]
    prices = [float(i * 10) for i in range(1, 35)]
    dates = [pd.Timestamp('2023-01-01') + pd.Timedelta(days=i) for i in range(34)]
    df = pd.DataFrame({'Date': dates, 'Price': prices})

    result = calculate_moving_averages(df)

    # 1. Verify basic output shape and columns
    assert 'MA7' in result.columns
    assert 'MA30' in result.columns
    assert len(result) == 34

    # 2. Verify NaNs placement (window-1 initial NaNs)
    assert pd.isna(result['MA7'].iloc[:6]).all()
    assert pd.isna(result['MA30'].iloc[:29]).all()

    # 3. Verify correct calculation values
    # The sum of arithmetic progression 10,20...70 is 7/2 * (10+70) = 280. 280/7 = 40.0
    assert result['MA7'].iloc[6] == 40.0

    # 30 day MA at index 29 (day 30) is sum(10..300)/30 = 155.0
    assert result['MA30'].iloc[29] == 155.0

    # Test shifting behavior
    # For index 7 (day 8), prices are 20..80 -> sum is 350. 350/7 = 50.0
    assert result['MA7'].iloc[7] == 50.0
