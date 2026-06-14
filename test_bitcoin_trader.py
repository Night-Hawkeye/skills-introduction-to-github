import pandas as pd
from bitcoin import simulate_bitcoin_prices, calculate_moving_averages

def test_simulate_bitcoin_prices():
    df = simulate_bitcoin_prices(days=10)
    assert len(df) == 10
    assert 'Date' in df.columns
    assert 'Price' in df.columns

def test_calculate_moving_averages():
    # Create a simple dataframe with known values
    dates = pd.date_range(start='2023-01-01', periods=35)
    # Price linearly increasing: 10, 20, 30...
    prices = [float(i * 10) for i in range(1, 36)]
    df = pd.DataFrame({'Date': dates, 'Price': prices})

    df = calculate_moving_averages(df)

    # Check if columns are created
    assert 'MA7' in df.columns
    assert 'MA30' in df.columns

    # Test that the first 6 values of MA7 are NaN (using index 0-5)
    assert pd.isna(df['MA7'].iloc[5])
    # The 7th value (index 6) should be the average of [10, 20, 30, 40, 50, 60, 70] = 40.0
    assert not pd.isna(df['MA7'].iloc[6])
    assert df['MA7'].iloc[6] == 40.0

    # Test that the first 29 values of MA30 are NaN
    assert pd.isna(df['MA30'].iloc[28])
    # The 30th value (index 29) should be the average of [10, ..., 300] = 155.0
    assert not pd.isna(df['MA30'].iloc[29])
    assert df['MA30'].iloc[29] == 155.0

    # Check length
    assert len(df) == 35

def test_calculate_moving_averages_edge_cases():
    # Test with empty dataframe
    df_empty = pd.DataFrame({'Date': [], 'Price': []})
    df_empty = calculate_moving_averages(df_empty)
    assert 'MA7' in df_empty.columns
    assert 'MA30' in df_empty.columns
    assert len(df_empty) == 0

    # Test with too few rows to calculate any moving average
    dates = pd.date_range(start='2023-01-01', periods=5)
    prices = [10.0, 20.0, 30.0, 40.0, 50.0]
    df_short = pd.DataFrame({'Date': dates, 'Price': prices})
    df_short = calculate_moving_averages(df_short)
    assert pd.isna(df_short['MA7']).all()
    assert pd.isna(df_short['MA30']).all()

def test_run_trading_algorithm_empty_df():
    from bitcoin import run_trading_algorithm
    df_empty = pd.DataFrame({'Date': pd.Series([], dtype='datetime64[ns]'), 'Price': pd.Series([], dtype=float), 'MA7': pd.Series([], dtype=float), 'MA30': pd.Series([], dtype=float)})
    result = run_trading_algorithm(df_empty)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0
