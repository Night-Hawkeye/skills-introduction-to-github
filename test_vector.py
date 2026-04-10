import numpy as np

days = 5
drift = 0.001
volatility = 0.04
initial_price = 50000

# Loop version
np.random.seed(42)
prices1 = [initial_price]
for _ in range(1, days):
    shock = np.random.normal(0, 1)
    price_change = np.exp((drift - 0.5 * volatility**2) + volatility * shock)
    prices1.append(prices1[-1] * price_change)

# Vectorized version
np.random.seed(42)
shocks = np.random.normal(0, 1, days - 1)
price_changes = np.exp((drift - 0.5 * volatility**2) + volatility * shocks)
prices2 = np.concatenate(([initial_price], initial_price * np.cumprod(price_changes)))

print("Loop:", prices1)
print("Vectorized:", list(prices2))
print("Equal:", np.allclose(prices1, prices2))
