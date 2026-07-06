import numpy as np
from bitcoin_trading import _generate_actions

def test_generate_actions_empty():
    actions = _generate_actions(np.array([]), np.array([]), np.array([]))
    assert len(actions) == 0

def test_generate_actions_hold():
    prices = np.array([50000, 51000, 52000])
    position = np.array([0, 0, 0])
    portfolio_value = np.array([10000.0, 10000.0, 10000.0])
    btc_held = np.array([0.0, 0.0, 0.0])
    initial_cash = 10000.0

    actions = _generate_actions(position, portfolio_value, btc_held)
    assert np.array_equal(actions, np.full(3, "HOLD"))

def test_generate_actions_buy():
    prices = np.array([50000, 50000, 50000])
    position = np.array([0, 1, 1])
    portfolio_value = np.array([10000.0, 10000.0, 10000.0])
    btc_held = np.array([0.0, 0.2, 0.2])
    initial_cash = 10000.0

    actions = _generate_actions(position, portfolio_value, btc_held)
    assert actions[0] == "HOLD"
    assert actions[1] == "BUY 0.2000 BTC"
    assert actions[2] == "HOLD"

def test_generate_actions_sell():
    prices = np.array([50000, 50000, 50000])
    position = np.array([0, 1, 0])
    portfolio_value = np.array([10000.0, 10000.0, 10000.0])
    btc_held = np.array([0.0, 0.2, 0.0])
    initial_cash = 10000.0

    actions = _generate_actions(position, portfolio_value, btc_held)
    assert actions[0] == "HOLD"
    assert actions[1] == "BUY 0.2000 BTC"
    assert actions[2] == "SELL 0.2000 BTC"

def test_generate_actions_sell_initial():
    prices = np.array([50000, 50000, 50000])
    position = np.array([1, 1, 0])
    portfolio_value = np.array([10000.0, 10000.0, 10000.0])
    btc_held = np.array([0.2, 0.2, 0.0])
    initial_cash = 10000.0

    actions = _generate_actions(position, portfolio_value, btc_held)
    assert actions[0] == "BUY 0.2000 BTC"
    assert actions[1] == "HOLD"
    assert actions[2] == "SELL 0.2000 BTC"

def test_generate_actions_zero_prices():
    prices = np.array([0.0, 0.0, 0.0])
    position = np.array([0, 1, 0])
    portfolio_value = np.array([10000.0, 10000.0, 10000.0])
    btc_held = np.array([0.0, 10000.0, 0.0])
    initial_cash = 10000.0

    actions = _generate_actions(position, portfolio_value, btc_held)
    assert actions[0] == "HOLD"
    assert actions[1] == "BUY 10000.0000 BTC"
    assert actions[2] == "SELL 10000.0000 BTC"
