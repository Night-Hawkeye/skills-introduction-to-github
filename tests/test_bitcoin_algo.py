import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from bitcoin import run_trading_algorithm

def test_run_trading_algorithm_golden_cross_no_cash():
    """Test that Golden Cross with no cash results in HOLD instead of BUY."""
    dates = pd.date_range(start=datetime.now(), periods=4)

    # Day 0: Setup
    # Day 1: Golden Cross -> BUY
    # Day 2: MA7 drops back down -> Death Cross -> SELL, but at price 0, so cash becomes 0
    # Day 3: Golden Cross again, but cash is 0, should HOLD
    data = {
        'Date': dates,
        'Price': [100.0, 100.0, 0.0, 100.0],
        'MA7':   [40.0, 60.0, 40.0, 60.0],
        'MA30':  [50.0, 50.0, 50.0, 50.0]
    }
    df = pd.DataFrame(data)

    ledger = run_trading_algorithm(df)

    assert len(ledger) == 4
    # Day 0 should hold
    assert ledger.iloc[0]['Action'] == "HOLD"
    # Day 1 should BUY
    assert "BUY" in ledger.iloc[1]['Action']
    assert ledger.iloc[1]['Cash'] == 0.0
    # Day 2 should SELL at 0 price, cash becomes 0
    assert "SELL" in ledger.iloc[2]['Action']
    assert ledger.iloc[2]['Cash'] == 0.0
    # Day 3 triggers Golden Cross but no cash, so action is HOLD
    assert ledger.iloc[3]['Action'] == "HOLD"
    assert ledger.iloc[3]['Cash'] == 0.0
