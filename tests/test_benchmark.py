import numpy as np
import pytest
from benchmark import original, optimized

def test_original_basic():
    """Test original benchmark with deterministic seed."""
    prices = original(days=5, initial_price=100.0, volatility=0.1, drift=0.01)
    assert len(prices) == 5
    assert prices[0] == 100.0

def test_original_math(mocker):
    """Test the math logic in original benchmark by mocking the random shock."""
    mocker.patch('numpy.random.normal', return_value=0.5)

    initial_price = 100.0
    volatility = 0.1
    drift = 0.01

    prices = original(days=4, initial_price=initial_price, volatility=volatility, drift=drift)

    assert len(prices) == 4
    assert prices[0] == 100.0

    expected_price_change = np.exp((drift - 0.5 * volatility**2) + volatility * 0.5)
    assert np.isclose(prices[1], prices[0] * expected_price_change)
    assert np.isclose(prices[2], prices[1] * expected_price_change)
    assert np.isclose(prices[3], prices[2] * expected_price_change)

def test_optimized_basic():
    """Test optimized benchmark with deterministic seed."""
    prices = optimized(days=5, initial_price=100.0, volatility=0.1, drift=0.01)
    assert len(prices) == 5
    assert prices[0] == 100.0

def test_optimized_math(mocker):
    """Test the math logic in optimized benchmark by mocking the random shock."""
    mocker.patch('numpy.random.normal', return_value=np.array([0.5, 0.5, 0.5]))

    initial_price = 100.0
    volatility = 0.1
    drift = 0.01

    prices = optimized(days=4, initial_price=initial_price, volatility=volatility, drift=drift)

    assert len(prices) == 4
    assert prices[0] == 100.0

    expected_price_change = np.exp((drift - 0.5 * volatility**2) + volatility * 0.5)
    assert np.isclose(prices[1], prices[0] * expected_price_change)
    assert np.isclose(prices[2], prices[1] * expected_price_change)
    assert np.isclose(prices[3], prices[2] * expected_price_change)

def test_original_vs_optimized():
    """Test that original and optimized return exactly the same results."""
    orig_res = original(days=10, initial_price=50000.0, volatility=0.04, drift=0.001)
    opt_res = optimized(days=10, initial_price=50000.0, volatility=0.04, drift=0.001)

    assert np.allclose(orig_res, opt_res)
