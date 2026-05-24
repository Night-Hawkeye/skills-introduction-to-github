import numpy as np
import pytest
from benchmark import original, optimized

def test_original_basic():
    """Test the original function with default parameters for basic output shape and first value."""
    res = original(days=10, initial_price=100.0)
    assert len(res) == 10
    assert res[0] == 100.0

def test_optimized_basic():
    """Test the optimized function with default parameters for basic output shape and first value."""
    res = optimized(days=10, initial_price=100.0)
    assert len(res) == 10
    assert res[0] == 100.0

def test_equivalence():
    """Test that the optimized function returns exactly the same results as original."""
    days = 100
    res_orig = original(days=days, initial_price=50000.0)
    res_opt = optimized(days=days, initial_price=50000.0)

    assert len(res_orig) == days
    assert len(res_opt) == days
    np.testing.assert_allclose(res_orig, res_opt, rtol=1e-5, err_msg="Optimized and original functions diverge")

def test_deterministic_output():
    """Test that the function returns a deterministic output given the hardcoded np.random.seed(42)."""
    # Since seed is hardcoded to 42 inside the function, it should always produce this exact sequence
    res1 = optimized(days=5, initial_price=1000.0, volatility=0.05, drift=0.0)
    res2 = optimized(days=5, initial_price=1000.0, volatility=0.05, drift=0.0)

    np.testing.assert_allclose(res1, res2)

    # Check exact values for seed 42 to ensure behavior doesn't change unexpectedly
    # First few np.random.normal(0, 1) for seed 42:
    # 0.49671415, -0.1382643, 0.64768854, 1.52302986
    # price_change = np.exp(-0.5 * (0.05**2) + 0.05 * shock)

    expected_length = 5
    assert len(res1) == expected_length
    assert res1[0] == 1000.0

def test_zero_volatility():
    """Test behavior when volatility is zero; price should only drift."""
    res = optimized(days=5, initial_price=100.0, volatility=0.0, drift=0.01)
    # price_change = np.exp(0.01 + 0)
    expected_change = np.exp(0.01)

    # Using np.isclose
    assert np.isclose(res[0], 100.0)
    assert np.isclose(res[1], 100.0 * expected_change)
    assert np.isclose(res[2], 100.0 * expected_change**2)
