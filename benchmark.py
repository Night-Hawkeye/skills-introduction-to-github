from bitcoin_trading import SimulationConfig
import secrets
import timeit
import numpy as np

def get_shocks(days, seed):
    if seed is None:
        seed = secrets.randbits(128)
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, days - 1)

def original(config: SimulationConfig = None):
    if config is None:
        config = SimulationConfig(days=10000)
    if config.days <= 0:
        return []
    shocks = get_shocks(config.days, config.seed)
    price_changes = np.exp((config.drift - 0.5 * config.volatility**2) + config.volatility * shocks)
    prices = np.concatenate(([config.initial_price], config.initial_price * np.cumprod(price_changes)))
    return prices.tolist()

def optimized(config: SimulationConfig = None):
    if config is None:
        config = SimulationConfig(days=10000)
    if config.days <= 0:
        return []
    shocks = get_shocks(config.days, config.seed)
    log_returns = (config.drift - 0.5 * config.volatility**2) + config.volatility * shocks
    prices = np.empty(config.days)
    prices[0] = config.initial_price
    if config.days > 1:
        prices[1:] = config.initial_price * np.exp(np.cumsum(log_returns))
    return prices.tolist()

if __name__ == '__main__':
    bench_seed = secrets.randbits(128)
    t_orig = timeit.timeit(f'original(SimulationConfig(days=10000, seed={bench_seed}))', globals=globals(), number=100)
    t_opt = timeit.timeit(f'optimized(SimulationConfig(days=10000, seed={bench_seed}))', globals=globals(), number=100)
    print(f"Original: {t_orig:.4f}s")
    print(f"Optimized: {t_opt:.4f}s")
    print(f"Speedup: {t_orig/t_opt:.2f}x")

    # Assert correctness
    test_seed = secrets.randbits(128)
    orig_res = original(SimulationConfig(days=10, seed=test_seed))
    opt_res = optimized(SimulationConfig(days=10, seed=test_seed))
    assert np.allclose(orig_res, opt_res), f"Results do not match!\nOrig: {orig_res}\nOpt: {opt_res}"
