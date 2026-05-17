import timeit
import numpy as np
import pandas as pd
import secrets
from datetime import datetime, timedelta

def original(days=10000, initial_price=50000.0, volatility=0.04, drift=0.001):
    np.random.seed(42)
    prices = [initial_price]
    for _ in range(1, days):
        shock = np.random.normal(0, 1)
        price_change = np.exp((drift - 0.5 * volatility**2) + volatility * shock)
        prices.append(prices[-1] * price_change)
    return prices

def optimized(days=10000, initial_price=50000.0, volatility=0.04, drift=0.001):
    np.random.seed(42)
    shocks = np.random.normal(0, 1, days - 1)
    price_changes = np.exp((drift - 0.5 * volatility**2) + volatility * shocks)
    prices = np.concatenate(([initial_price], initial_price * np.cumprod(price_changes)))
    return prices.tolist()

if __name__ == '__main__':
    t_orig = timeit.timeit('original()', globals=globals(), number=100)
    t_opt = timeit.timeit('optimized()', globals=globals(), number=100)
    print(f"Original: {t_orig:.4f}s")
    print(f"Optimized: {t_opt:.4f}s")
    print(f"Speedup: {t_orig/t_opt:.2f}x")

    # Assert correctness
    orig_res = original(days=10)
    opt_res = optimized(days=10)
    assert np.allclose(orig_res, opt_res), f"Results do not match!\nOrig: {orig_res}\nOpt: {opt_res}"
