import pytest
from bitcoin_trader import simulate_bitcoin_prices

def test_predictability():
    """Test that simulate_bitcoin_prices produces different output on consecutive runs."""
    df1 = simulate_bitcoin_prices(days=10)
    df2 = simulate_bitcoin_prices(days=10)

    prices1 = df1['Price'].tolist()
    prices2 = df2['Price'].tolist()

    # The simulation uses random walk and should not be exactly identical across runs
    assert prices1 != prices2, "Prices are identical across multiple runs. Simulation is predictable."
