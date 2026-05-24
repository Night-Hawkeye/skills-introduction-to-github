1. **Optimize `run_trading_algorithm` in `bitcoin_trader.py` using NumPy vectorization.**
   - Instead of a slow for-loop evaluating conditions per row, use `np.roll` to construct vectorized lookbehind variables (`prev_ma7`, `prev_ma30`, `prev_valid`).
   - Use vectorized boolean masking (`buy_signals`, `sell_signals`) to evaluate Golden/Death Crosses efficiently.
   - Use `np.where` to find exactly which indices signal an event.
   - Iterate only over these cross events to update `cash`, `btc`, and `action`.
   - Update state dynamically via NumPy broadcasting/slice assignments instead of single-row scalar updates.

2. **Verify changes using `PYTHONPATH=. /home/jules/.local/share/pipx/venvs/pytest/bin/python test_algo8.py`.**
   - Confirm dataframe equality (`pd.testing.assert_frame_equal(orig_out, new_out)`).
   - Ensure it is a net performance increase over the baseline loop method.

3. **Complete pre-commit steps to ensure proper testing, verification, review, and reflection are done.**
   - Call `pre_commit_instructions` tool to evaluate code formatting and test coverage.

4. **Submit PR.**
   - Make a final commit summarizing the optimization and create the PR.
