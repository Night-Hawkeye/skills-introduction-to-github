import time
import pandas as pd
import numpy as np
from bitcoin_analysis import simulate_bitcoin_prices, calculate_moving_averages, run_trading_algorithm

def benchmark():
    # Increase the number of days to make the difference measurable
    days = 100000
    print(f"Generating data for {days} days...")
    df = simulate_bitcoin_prices(days)
    df = calculate_moving_averages(df)

    print("Running baseline benchmark...")
    start_time = time.time()
    result = run_trading_algorithm(df)
    end_time = time.time()

    duration = end_time - start_time
    print(f"Baseline duration: {duration:.4f} seconds")
    return duration

if __name__ == "__main__":
    benchmark()
