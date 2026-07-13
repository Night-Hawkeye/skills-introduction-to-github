import pandas as pd
from bitcoin_trading import simulate_bitcoin_prices, calculate_moving_averages, SimulationConfig

def test_simulate_bitcoin_prices():
    df = simulate_bitcoin_prices(SimulationConfig(days=10))
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
    from bitcoin_trading import run_trading_algorithm
    df_empty = pd.DataFrame({'Date': pd.Series([], dtype='datetime64[ns]'), 'Price': pd.Series([], dtype=float), 'MA7': pd.Series([], dtype=float), 'MA30': pd.Series([], dtype=float)})
    result = run_trading_algorithm(df_empty)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_generate_signals():
    import numpy as np
    import pandas as pd
    from bitcoin_trading import _generate_signals

    # Test empty arrays
    empty_result = _generate_signals(np.array([]), np.array([]), pd.Index([]))
    assert len(empty_result) == 0

    # Test golden cross (MA7 crosses above MA30)
    # Day 0: MA7 < MA30 (Valid: True)
    # Day 1: MA7 > MA30 (Valid: True) -> Buy signal
    # Day 2: MA7 > MA30 (Valid: True) -> Hold
    ma7 = np.array([10.0, 35.0, 40.0])
    ma30 = np.array([20.0, 30.0, 30.0])
    index = pd.Index([0, 1, 2])
    signals = _generate_signals(ma7, ma30, index)

    assert len(signals) == 3
    assert signals[0] == 0.0 # Initial state
    assert signals[1] == 1.0 # Buy signal
    assert signals[2] == 1.0 # Holding

    # Test death cross (MA7 crosses below MA30)
    # Day 0: MA7 < MA30 (Valid: True)
    # Day 1: MA7 > MA30 (Valid: True) -> Buy
    # Day 2: MA7 < MA30 (Valid: True) -> Sell
    ma7 = np.array([10.0, 40.0, 20.0])
    ma30 = np.array([20.0, 30.0, 30.0])
    signals2 = _generate_signals(ma7, ma30, index)

    assert len(signals2) == 3
    assert signals2[0] == 0.0 # Initial
    assert signals2[1] == 1.0 # Buy
    assert signals2[2] == 0.0 # Sell signal

def test_generate_actions_empty():
    import numpy as np
    from bitcoin_trading import _generate_actions

    prices = np.array([])
    position = np.array([])
    portfolio_value = np.array([])
    btc_held = np.array([])
    initial_cash = 10000.0

    result = _generate_actions(position, portfolio_value, btc_held)
    assert len(result) == 0
    assert isinstance(result, np.ndarray)

def test_calculate_strategy_returns():
    import numpy as np
    from bitcoin_trading import _calculate_strategy_returns

    # Test happy path
    btc_returns = np.array([0.0, 0.1, -0.05, 0.2])
    position = np.array([0, 1, 1, 0])
    strat_returns = _calculate_strategy_returns(btc_returns, position)
    assert len(strat_returns) == 4
    np.testing.assert_array_almost_equal(strat_returns, [0.0, 0.0, -0.05, 0.2])

    # Test all zeros
    btc_returns_zero = np.zeros(5)
    position_zero = np.zeros(5)
    strat_returns_zero = _calculate_strategy_returns(btc_returns_zero, position_zero)
    np.testing.assert_array_almost_equal(strat_returns_zero, np.zeros(5))

    # Test all ones
    btc_returns_ones = np.ones(3)
    position_ones = np.ones(3)
    strat_returns_ones = _calculate_strategy_returns(btc_returns_ones, position_ones)
    np.testing.assert_array_almost_equal(strat_returns_ones, [0.0, 1.0, 1.0])

    # Test empty arrays
    btc_returns_empty = np.array([])
    position_empty = np.array([])
    strat_returns_empty = _calculate_strategy_returns(btc_returns_empty, position_empty)
    assert len(strat_returns_empty) == 0
