import pandas as pd
from trading_utils import simulate_bitcoin_prices, calculate_moving_averages, run_trading_algorithm

if __name__ == "__main__":
    print("Simulating 60 days of Bitcoin prices...")
    df = simulate_bitcoin_prices(60, seed=123)
    df = calculate_moving_averages(df)

    print("\nRunning Golden Cross trading algorithm...")
    ledger_df = run_trading_algorithm(df)

    print("\n--- Daily Ledger ---")
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)

    # Format floats for cleaner output
    out_df = ledger_df.copy()
    out_df['Price'] = out_df['Price'].round(2)
    out_df['MA7'] = out_df['MA7'].round(2)
    out_df['MA30'] = out_df['MA30'].round(2)
    out_df['Portfolio Value'] = out_df['Portfolio Value'].round(2)

    print(out_df[['Date', 'Price', 'MA7', 'MA30', 'Action', 'Portfolio Value']])

    final_value = ledger_df.iloc[-1]['Portfolio Value']
    initial_value = 10000.0
    roi = ((final_value - initial_value) / initial_value) * 100

    print("\n--- Final Portfolio Performance ---")
    print(f"Initial Value: ${initial_value:.2f}")
    print(f"Final Value:   ${final_value:.2f}")
    print(f"Return on Investment (ROI): {roi:.2f}%")
