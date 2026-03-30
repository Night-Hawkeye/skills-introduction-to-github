import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from bitcoin_trading import run_trading_algorithm

def test_run_trading_algorithm_golden_cross():
    """Test that Golden Cross (MA7 crosses above MA30) triggers a BUY."""
    dates = [datetime(2023, 1, i+1) for i in range(2)]
    prices = [1000, 1000]

    # prev_ma7 <= prev_ma30 and ma7 > ma30
    ma7s = [900, 1100]
    ma30s = [1000, 1000]

    df = pd.DataFrame({
        'Date': dates,
        'Price': prices,
        'MA7': ma7s,
        'MA30': ma30s
    })

    result_df = run_trading_algorithm(df)

    # Row 0: HOLD
    assert result_df.iloc[0]['Action'] == "HOLD"
    assert result_df.iloc[0]['Cash'] == 10000.0
    assert result_df.iloc[0]['BTC'] == 0.0

    # Row 1: BUY
    assert "BUY" in result_df.iloc[1]['Action']
    assert result_df.iloc[1]['Cash'] == 0.0
    assert result_df.iloc[1]['BTC'] == 10.0  # 10000 / 1000
    assert result_df.iloc[1]['Portfolio Value'] == 10000.0

def test_run_trading_algorithm_death_cross():
    """Test that Death Cross (MA7 crosses below MA30) triggers a SELL if we hold BTC."""
    dates = [datetime(2023, 1, i+1) for i in range(3)]
    prices = [1000, 2000, 3000]

    # Row 0: setup
    # Row 1: golden cross -> BUY (at price 2000, cash=10000 -> 5 BTC)
    # Row 2: death cross -> SELL (at price 3000, 5 BTC -> 15000 cash)

    ma7s =  [900,  1100, 900]
    ma30s = [1000, 1000, 1000]

    df = pd.DataFrame({
        'Date': dates,
        'Price': prices,
        'MA7': ma7s,
        'MA30': ma30s
    })

    result_df = run_trading_algorithm(df)

    # Row 1: BUY
    assert "BUY" in result_df.iloc[1]['Action']
    assert result_df.iloc[1]['Cash'] == 0.0
    assert result_df.iloc[1]['BTC'] == 5.0  # 10000 / 2000

    # Row 2: SELL
    assert "SELL" in result_df.iloc[2]['Action']
    assert result_df.iloc[2]['Cash'] == 15000.0  # 5 BTC * 3000
    assert result_df.iloc[2]['BTC'] == 0.0
    assert result_df.iloc[2]['Portfolio Value'] == 15000.0

def test_run_trading_algorithm_hold():
    """Test that the algorithm holds when no crosses happen."""
    dates = [datetime(2023, 1, i+1) for i in range(3)]
    prices = [1000, 1000, 1000]

    # MA7 consistently below MA30
    ma7s =  [900, 900, 900]
    ma30s = [1000, 1000, 1000]

    df = pd.DataFrame({
        'Date': dates,
        'Price': prices,
        'MA7': ma7s,
        'MA30': ma30s
    })

    result_df = run_trading_algorithm(df)

    for i in range(3):
        assert result_df.iloc[i]['Action'] == "HOLD"
        assert result_df.iloc[i]['Cash'] == 10000.0
        assert result_df.iloc[i]['BTC'] == 0.0

def test_run_trading_algorithm_nan_values():
    """Test that NaN values (common at start of rolling averages) are ignored properly."""
    dates = [datetime(2023, 1, i+1) for i in range(3)]
    prices = [1000, 1000, 1000]

    # Missing MA data
    ma7s =  [np.nan, 1100, 1200]
    ma30s = [np.nan, np.nan, 1000]

    df = pd.DataFrame({
        'Date': dates,
        'Price': prices,
        'MA7': ma7s,
        'MA30': ma30s
    })

    result_df = run_trading_algorithm(df)

    # Everything should be HOLD because we need both MA7 and MA30 and their previous values to trigger
    for i in range(3):
        assert result_df.iloc[i]['Action'] == "HOLD"
        assert result_df.iloc[i]['Cash'] == 10000.0
        assert result_df.iloc[i]['BTC'] == 0.0
