import numpy as np
from benchmark import original, optimized, get_shocks
from unittest.mock import MagicMock

def test_original_basic():
    """Test original benchmark with deterministic seed."""
    prices = original(days=5, initial_price=100.0, volatility=0.1, drift=0.01)
    assert len(prices) == 5
    assert prices[0] == 100.0

def test_original_math(mocker):
    """Test the math logic in original benchmark by mocking the random shock."""

    # Create a mock RNG that returns a mock normal distribution
    mock_rng = MagicMock()
    mock_rng.normal.return_value = np.array([0.5, 0.5, 0.5])

    # Patch default_rng to return our mock
    mocker.patch('numpy.random.default_rng', return_value=mock_rng)

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
    # Create a mock RNG that returns a mock normal distribution array
    mock_rng = MagicMock()
    mock_rng.normal.return_value = np.array([0.5, 0.5, 0.5])

    # Patch default_rng to return our mock
    mocker.patch('numpy.random.default_rng', return_value=mock_rng)

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
    # We must use the same seed for both so we don't rely on random matching
    test_seed = 12345
    orig_res = original(days=10, initial_price=50000.0, volatility=0.04, drift=0.001, seed=test_seed)
    opt_res = optimized(days=10, initial_price=50000.0, volatility=0.04, drift=0.001, seed=test_seed)

    assert np.allclose(orig_res, opt_res)

def test_edge_cases():
    """Test that days <= 0 returns an empty list for both implementations."""
    assert original(days=0) == []
    assert optimized(days=0) == []
    assert original(days=-1) == []
    assert optimized(days=-1) == []

def test_seed_none():
    """Test that original and optimized handle seed=None correctly."""
    orig_prices = original(days=5, seed=None)
    opt_prices = optimized(days=5, seed=None)
    assert len(orig_prices) == 5
    assert len(opt_prices) == 5

def test_get_shocks_shape():
    """Test get_shocks returns the correct shape."""
    shocks = get_shocks(days=10, seed=123)
    assert len(shocks) == 9
    assert isinstance(shocks, np.ndarray)

def test_get_shocks_determinism():
    """Test get_shocks is deterministic given a seed."""
    seed = 42
    shocks1 = get_shocks(days=5, seed=seed)
    shocks2 = get_shocks(days=5, seed=seed)
    assert np.allclose(shocks1, shocks2)

def test_get_shocks_different_seeds():
    """Test get_shocks produces different results for different seeds."""
    shocks1 = get_shocks(days=5, seed=42)
    shocks2 = get_shocks(days=5, seed=43)
    assert not np.allclose(shocks1, shocks2)

def test_get_shocks_none_seed():
    """Test get_shocks works when seed is None."""
    shocks = get_shocks(days=5, seed=None)
    assert len(shocks) == 4

def test_get_shocks_mock_secrets(mocker):
    """Test get_shocks uses secrets.randbits when seed is None."""
    mock_randbits = mocker.patch('secrets.randbits', return_value=12345)
    shocks1 = get_shocks(days=5, seed=None)
    shocks2 = get_shocks(days=5, seed=12345)
    mock_randbits.assert_called_once_with(128)
    assert np.allclose(shocks1, shocks2)
