import numpy as np
import pytest
from bitcoin import _calculate_portfolio

def test_calculate_portfolio_empty():
    prices = np.array([])
    position = np.array([])
    portfolio_value, cash_held, btc_held = _calculate_portfolio(prices, position, 10000.0)
    assert len(portfolio_value) == 0
    assert len(cash_held) == 0
    assert len(btc_held) == 0

def test_calculate_portfolio_hold_cash():
    prices = np.array([100.0, 105.0, 102.0])
    position = np.array([0, 0, 0])
    portfolio_value, cash_held, btc_held = _calculate_portfolio(prices, position, 10000.0)
    assert np.allclose(portfolio_value, [10000.0, 10000.0, 10000.0])
    assert np.allclose(cash_held, [10000.0, 10000.0, 10000.0])
    assert np.allclose(btc_held, [0.0, 0.0, 0.0])

def test_calculate_portfolio_buy_and_hold_btc():
    prices = np.array([100.0, 110.0, 121.0])
    # Buy on day 0, so position is 1 for day 0, 1, 2
    # But btc returns are applied to the *previous* position
    # prev_position = [0, 1, 1]
    # returns: 0, 10%, 10%
    # strat returns: 0, 10%, 10%
    # port val: 10k, 11k, 12.1k
    position = np.array([1, 1, 1])
    portfolio_value, cash_held, btc_held = _calculate_portfolio(prices, position, 10000.0)
    assert np.allclose(portfolio_value, [10000.0, 11000.0, 12100.0])
    assert np.allclose(cash_held, [0.0, 0.0, 0.0])
    assert np.allclose(btc_held, [100.0, 100.0, 100.0])

def test_calculate_portfolio_buy_then_sell():
    prices = np.array([100.0, 200.0, 100.0])
    # Buy day 0, sell day 2
    # position: 1, 1, 0
    # prev_pos: 0, 1, 1
    # strat returns: 0, 100%, -50%
    # port val: 10k, 20k, 10k
    position = np.array([1, 1, 0])
    portfolio_value, cash_held, btc_held = _calculate_portfolio(prices, position, 10000.0)
    assert np.allclose(portfolio_value, [10000.0, 20000.0, 10000.0])
    assert np.allclose(cash_held, [0.0, 0.0, 10000.0])
    assert np.allclose(btc_held, [100.0, 100.0, 0.0])

def test_calculate_portfolio_zero_prices():
    prices = np.array([100.0, 0.0, 100.0])
    position = np.array([1, 1, 1])
    # prev_pos: 0, 1, 1
    # returns: 0, -100%, +infinity/handled
    # safe_div makes prev_price 1.0 when it is 0
    # prev_prices: 100.0, 0.0 -> 100.0, 1.0
    # returns[1]: (0.0 - 100.0) / 100.0 = -1.0
    # returns[2]: (100.0 - 0.0) / 1.0 = 100.0
    # strat_returns: 0.0, -1.0, 100.0
    # port val: 10k, 0, 0 (since cumprod of 1 + -1 = 0, then 0 * 101 = 0)
    portfolio_value, cash_held, btc_held = _calculate_portfolio(prices, position, 10000.0)
    assert np.allclose(portfolio_value, [10000.0, 0.0, 0.0])

def test_calculate_portfolio_negative_returns_bounded():
    # Force negative portfolio to test np.maximum(portfolio_value, 0.0)
    prices = np.array([100.0, -50.0])
    position = np.array([1, 1])
    portfolio_value, cash_held, btc_held = _calculate_portfolio(prices, position, 10000.0)
    assert np.allclose(portfolio_value, [10000.0, 0.0])
