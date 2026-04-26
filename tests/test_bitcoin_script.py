import pandas as pd
import numpy as np
import pytest
from bitcoin_script import simulate_bitcoin_prices, calculate_moving_averages, run_trading_algorithm

def create_mock_df(prices, ma7s, ma30s):
    dates = pd.date_range(start='2023-01-01', periods=len(prices))
    return pd.DataFrame({
        'Date': dates,
        'Price': prices,
        'MA7': ma7s,
        'MA30': ma30s
    })

def test_simulate_bitcoin_prices():
    """Test simulate_bitcoin_prices for valid output dimensions and columns."""
    days = 10
    initial_price = 45000.0
    df = simulate_bitcoin_prices(days=days, initial_price=initial_price)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (days, 2)
    assert list(df.columns) == ['Date', 'Price']
    assert df.iloc[0]['Price'] == initial_price
    assert pd.api.types.is_datetime64_any_dtype(df['Date'])

def test_calculate_moving_averages():
    """Test calculate_moving_averages logic."""
    prices = list(range(1, 31))
    dates = pd.date_range(start='2023-01-01', periods=len(prices))
    df = pd.DataFrame({'Date': dates, 'Price': prices})

    result_df = calculate_moving_averages(df)

    assert 'MA7' in result_df.columns
    assert 'MA30' in result_df.columns
    assert result_df['MA7'].iloc[:6].isna().all()
    # Average of 1..7 is 4.0
    assert result_df['MA7'].iloc[6] == 4.0
    assert result_df['MA30'].iloc[:29].isna().all()
    # Average of 1..30 is 15.5
    assert result_df['MA30'].iloc[29] == 15.5

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
    assert result.iloc[1]['BTC'] == 10000.0 / 20000.0
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
    # NaNs should result in HOLD
    df = create_mock_df(
        prices=[10000, 10000, 20000],
        ma7s=[np.nan, np.nan, 12000],
        ma30s=[np.nan, 11000, 11000]
    )

    result = run_trading_algorithm(df)

    assert all(result['Action'] == "HOLD")
    assert all(result['Cash'] == 10000.0)
