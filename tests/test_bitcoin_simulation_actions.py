import numpy as np
import pytest
from bitcoin_trading import _generate_actions

def test_generate_actions_empty():
    prices = np.array([])
    position = np.array([])
    portfolio_value = np.array([])
    btc_held = np.array([])
    actions = _generate_actions(prices, position, portfolio_value, btc_held, 10000.0)
    assert len(actions) == 0

def test_generate_actions_hold():
    prices = np.array([100.0, 105.0, 102.0])
    position = np.array([0, 0, 0])
    portfolio_value = np.array([10000.0, 10000.0, 10000.0])
    btc_held = np.array([0.0, 0.0, 0.0])
    actions = _generate_actions(prices, position, portfolio_value, btc_held, 10000.0)
    assert list(actions) == ["HOLD", "HOLD", "HOLD"]

def test_generate_actions_buy():
    prices = np.array([100.0, 105.0, 102.0])
    position = np.array([0, 1, 1])
    portfolio_value = np.array([10000.0, 10000.0, 10000.0])
    btc_held = np.array([0.0, 95.238, 95.238])
    actions = _generate_actions(prices, position, portfolio_value, btc_held, 10000.0)
    assert actions[0] == "HOLD"
    assert "BUY" in actions[1]
    assert actions[2] == "HOLD"

def test_generate_actions_sell():
    prices = np.array([100.0, 200.0, 100.0])
    position = np.array([0, 1, 0])
    portfolio_value = np.array([10000.0, 10000.0, 20000.0])
    btc_held = np.array([0.0, 50.0, 0.0])
    actions = _generate_actions(prices, position, portfolio_value, btc_held, 10000.0)
    assert actions[0] == "HOLD"
    assert "BUY" in actions[1]
    assert "SELL" in actions[2]
