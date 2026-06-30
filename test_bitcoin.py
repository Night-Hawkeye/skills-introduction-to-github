import pandas as pd
from bitcoin import run_trading_algorithm

def test_run_trading_algorithm_empty():
    df = pd.DataFrame({'Date': pd.Series([], dtype='datetime64[ns]'), 'Price': pd.Series([], dtype=float), 'MA7': pd.Series([], dtype=float), 'MA30': pd.Series([], dtype=float)})
    result = run_trading_algorithm(df)
    assert len(result) == 0

def test_run_trading_algorithm_basic():
    df = pd.DataFrame({
        'Date': pd.date_range(start='2023-01-01', periods=3),
        'Price': [10000, 20000, 30000],
        'MA7': [9000, 12000, 15000],
        'MA30': [11000, 11000, 11000]
    })
    result = run_trading_algorithm(df)
    assert len(result) == 3
    assert result.iloc[0]['Action'] == "HOLD"
