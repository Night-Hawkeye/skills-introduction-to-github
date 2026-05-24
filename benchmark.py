import timeit
import numpy as np

def original(days=10000, initial_price=50000.0, volatility=0.04, drift=0.001, seed=None):
    if seed is None:
        seed = secrets.randbits(128)
    rng = np.random.default_rng(seed)
    prices = [initial_price]
    for _ in range(1, days):
        shock = rng.normal(0, 1)
        price_change = np.exp((drift - 0.5 * volatility**2) + volatility * shock)
        prices.append(prices[-1] * price_change)
    return prices

def optimized(days=10000, initial_price=50000.0, volatility=0.04, drift=0.001, seed=None):
    if seed is None:
        seed = secrets.randbits(128)
    rng = np.random.default_rng(seed)
    shocks = rng.normal(0, 1, days - 1)
    price_changes = np.exp((drift - 0.5 * volatility**2) + volatility * shocks)
    prices = np.concatenate(([initial_price], initial_price * np.cumprod(price_changes)))
    return prices.tolist()

if __name__ == '__main__':
    bench_seed = secrets.randbits(128)
    t_orig = timeit.timeit(f'original(seed={bench_seed})', globals=globals(), number=100)
    t_opt = timeit.timeit(f'optimized(seed={bench_seed})', globals=globals(), number=100)
    print(f"Original: {t_orig:.4f}s")
    print(f"Optimized: {t_opt:.4f}s")
    print(f"Speedup: {t_orig/t_opt:.2f}x")

    # Assert correctness
    test_seed = secrets.randbits(128)
    orig_res = original(days=10, seed=test_seed)
    opt_res = optimized(days=10, seed=test_seed)
    assert np.allclose(orig_res, opt_res), f"Results do not match!\nOrig: {orig_res}\nOpt: {opt_res}"
