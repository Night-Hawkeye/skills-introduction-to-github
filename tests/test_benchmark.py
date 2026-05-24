import numpy as np
import pytest
from benchmark import original, optimized

def test_original_correctness():
    """Test that the original function returns a list of expected length and starting value."""
    days = 10
    initial_price = 50000.0
    prices = original(days=days, initial_price=initial_price)

    assert isinstance(prices, list)
    assert len(prices) == days
    assert prices[0] == initial_price

    # Check that all elements are positive
    assert all(p > 0 for p in prices)

def test_optimized_correctness():
    """Test that the optimized function returns a list of expected length and starting value."""
    days = 10
    initial_price = 50000.0
    prices = optimized(days=days, initial_price=initial_price)

    assert isinstance(prices, list)
    assert len(prices) == days
    assert prices[0] == initial_price

    # Check that all elements are positive
    assert all(p > 0 for p in prices)

def test_original_vs_optimized_match():
    """Test that original and optimized return exactly the same results given the same seed."""
    days = 100

    res_orig = original(days=days)
    res_opt = optimized(days=days)

    assert np.allclose(res_orig, res_opt)

def test_custom_parameters():
    """Test the functions with non-default parameters."""
    days = 5
    initial = 100.0
    volatility = 0.1
    drift = 0.05

    res_orig = original(days=days, initial_price=initial, volatility=volatility, drift=drift)
    res_opt = optimized(days=days, initial_price=initial, volatility=volatility, drift=drift)

    assert len(res_orig) == days
    assert len(res_opt) == days
    assert res_orig[0] == initial
    assert res_opt[0] == initial
    assert np.allclose(res_orig, res_opt)

def test_edge_cases():
    """Test edge cases like days=1."""
    # When days=1, no loop iterations happen in original()
    # In optimized, days-1 is 0, so shocks is empty array, and cumulative product is just initial_price.
    res_orig = original(days=1, initial_price=10.0)
    res_opt = optimized(days=1, initial_price=10.0)

    assert res_orig == [10.0]
    assert res_opt == [10.0]
