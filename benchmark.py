import time
import timeit
from bitcoin_trader import simulate_bitcoin_prices

def run_benchmark():
    # Number of days to simulate
    days = 100000

    # Run a few iterations to get average time
    setup_code = "from bitcoin_trader import simulate_bitcoin_prices"
    test_code = f"simulate_bitcoin_prices(days={days})"

    times = timeit.repeat(setup=setup_code, stmt=test_code, repeat=5, number=10)

    print(f"Benchmark: simulating {days} days")
    print(f"Min time for 10 runs: {min(times):.4f} seconds")
    print(f"Average time for 10 runs: {sum(times)/len(times):.4f} seconds")

if __name__ == "__main__":
    run_benchmark()
