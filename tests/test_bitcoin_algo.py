import pandas as pd
import numpy as np
import pytest
from bitcoin_algo import run_trading_algorithm

def create_mock_df(prices, ma7s, ma30s):
    dates = [pd.Timestamp(2023, 1, 1) + pd.Timedelta(days=i) for i in range(len(prices))]
    return pd.DataFrame({
        'Date': pd.Series(dates),
        'Price': prices,
        'MA7': ma7s,
        'MA30': ma30s
    })

def test_run_trading_algorithm_death_cross_no_btc():
    """Test Death Cross with no BTC (should result in HOLD)."""
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
