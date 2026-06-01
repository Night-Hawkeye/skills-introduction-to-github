import pytest
from unittest.mock import patch
import pandas as pd
from reproduce_issue import test_predictability

def test_reproduce_issue_vulnerable(capsys):
    """Test when prices are identical (vulnerable)."""
    df = pd.DataFrame({'Date': pd.date_range(start='2023-01-01', periods=2), 'Price': [100.0, 105.0]})
    with patch('reproduce_issue.simulate_bitcoin_prices', return_value=df):
        test_predictability()
        captured = capsys.readouterr()
        assert "VULNERABLE: Prices are identical across multiple runs." in captured.out

def test_reproduce_issue_secure(capsys):
    """Test when prices differ (secure)."""
    df1 = pd.DataFrame({'Date': pd.date_range(start='2023-01-01', periods=2), 'Price': [100.0, 105.0]})
    df2 = pd.DataFrame({'Date': pd.date_range(start='2023-01-01', periods=2), 'Price': [100.0, 110.0]})

    with patch('reproduce_issue.simulate_bitcoin_prices', side_effect=[df1, df2]):
        test_predictability()
        captured = capsys.readouterr()
        assert "SECURE: Prices differ across multiple runs." in captured.out
