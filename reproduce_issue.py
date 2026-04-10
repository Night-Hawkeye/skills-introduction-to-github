import pandas as pd
from btc_trader_script import simulate_bitcoin_prices

def test_predictability():
    df1 = simulate_bitcoin_prices(days=10)
    df2 = simulate_bitcoin_prices(days=10)

    prices1 = df1['Price'].tolist()
    prices2 = df2['Price'].tolist()

    print(f"Prices 1: {prices1}")
    print(f"Prices 2: {prices2}")

    if prices1 == prices2:
        print("VULNERABLE: Prices are identical across multiple runs.")
    else:
        print("SECURE: Prices differ across multiple runs.")

if __name__ == "__main__":
    test_predictability()
