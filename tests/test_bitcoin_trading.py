import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta
from bitcoin_trader import run_trading_algorithm, simulate_bitcoin_prices, calculate_moving_averages

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

def test_simulate_bitcoin_prices_dimensions():
    """Test simulate_bitcoin_prices for valid output dimensions and columns."""
    days = 10
    initial_price = 45000
    df = simulate_bitcoin_prices(days=days, initial_price=initial_price)

    # Check if result is a DataFrame
    assert isinstance(df, pd.DataFrame)

    # Check dimensions (rows, columns)
    assert df.shape == (days, 2)

    # Check column names
    assert list(df.columns) == ['Date', 'Price']

    # Check first price matches initial_price
    assert df.iloc[0]['Price'] == initial_price

    # Check if 'Date' column contains datetime-like objects
    assert pd.api.types.is_datetime64_any_dtype(df['Date'])

def test_calculate_moving_averages():
    """Test calculate_moving_averages logic."""
    # Create a mock DataFrame with 30 prices (1 to 30)
    prices = list(range(1, 31))
    dates = [pd.Timestamp(2023, 1, 1) + pd.Timedelta(days=i) for i in range(len(prices))]
    df = pd.DataFrame({'Date': pd.Series(dates), 'Price': prices})

    # Run the function
    result_df = calculate_moving_averages(df)

    # Check that columns were added
    assert 'MA7' in result_df.columns
    assert 'MA30' in result_df.columns

    # First 6 elements of MA7 should be NaN
    assert result_df['MA7'].iloc[:6].isna().all()

    # 7th element of MA7 should be average of 1, 2, 3, 4, 5, 6, 7 -> 28 / 7 = 4.0
    assert result_df['MA7'].iloc[6] == 4.0

    # First 29 elements of MA30 should be NaN
    assert result_df['MA30'].iloc[:29].isna().all()

    # 30th element of MA30 should be average of 1 to 30 -> (30 * 31 / 2) / 30 = 15.5
    assert result_df['MA30'].iloc[29] == 15.5
