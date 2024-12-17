# wfa_driver.py
import json
import pandas as pd
import itertools
import os
from asset_allocation import run_asset_allocation
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import numpy as np

def recalc_sr_with_rf(port):
    # Recalculate SR including the risk-free rate (tbill_ret)
    # port['ret_long'] is the portfolio return
    # port['tbill_ret'] is the risk-free rate
    ret_long = port['ret_long'].fillna(0)
    rf = port['tbill_ret'].fillna(0)
    excess = ret_long - rf
    if excess.std() == 0:
        # Avoid division by zero: If std is zero, set SR to a very negative number or zero
        return -9999
    sr_new = (excess.mean() / excess.std()) * np.sqrt(252)
    return sr_new

def evaluate_params(params, train_start_str, train_end_str):
    metrics_train, port_train = run_asset_allocation(
        start_date=train_start_str,
        end_date=train_end_str,
        UP_DAY=params["UP_DAY"],
        DOWN_DAY=params["DOWN_DAY"],
        ADR_VOL_ADJ=params["ADR_VOL_ADJ"],
        KELT_MULT=params["KELT_MULT"],
        AUM_0=params["AUM_0"],
        invest_cash=params["invest_cash"],
        target_vol=params["target_vol"],
        max_leverage=params["max_leverage"],
        max_not_trade=params["max_not_trade"]
    )

    # Recalculate SR with risk-free rate
    if not port_train.empty and 'tbill_ret' in port_train.columns and 'ret_long' in port_train.columns:
        sr_new = recalc_sr_with_rf(port_train)
        metrics_train["sr"] = round(sr_new, 2)
    else:
        # If data is not available, fallback or keep the original sr
        # If original sr not computed, set to very low
        metrics_train["sr"] = metrics_train.get("sr", -9999)

    return params, metrics_train

def run_single_iteration(i, train_start, train_end, test_start, test_end, param_combinations):
    train_start_str = train_start.strftime("%Y-%m-%d")
    train_end_str = train_end.strftime("%Y-%m-%d")
    test_start_str = test_start.strftime("%Y-%m-%d")
    test_end_str = test_end.strftime("%Y-%m-%d")

    print(f"\nDEBUG: Iteration {i}")
    print(f"DEBUG: Training period: {train_start_str} to {train_end_str}")
    print(f"DEBUG: Testing period: {test_start_str} to {test_end_str}")
    print("DEBUG: Training: Evaluating parameter combinations...")

    best_params = None
    best_metric = -999999
    best_train_metrics = None

    with ProcessPoolExecutor(max_workers=max(1, multiprocessing.cpu_count()//2)) as executor:
        futures = {
            executor.submit(evaluate_params, params, train_start_str, train_end_str): params
            for params in param_combinations
        }
        idx = 0
        for future in as_completed(futures):
            idx += 1
            params, metrics_train = future.result()
            sr = metrics_train["sr"]
            if sr > best_metric:
                best_metric = sr
                best_params = params
                best_train_metrics = metrics_train
            if idx % 10 == 0:
                print(f"DEBUG: Checked {idx}/{len(param_combinations)} combinations... Best SR so far: {best_metric}")

    print(f"DEBUG: Best params for iteration {i}: {best_params} with SR={best_metric}")

    train_result_entry = {
        "iteration": i,
        "training_start": train_start_str,
        "training_end": train_end_str,
        "best_params": json.dumps(best_params),
        "best_sr": best_metric
    }
    for k, v in best_train_metrics.items():
        train_result_entry[k] = v

    print("DEBUG: Testing with best parameters...")
    metrics_test, port_test = run_asset_allocation(
        start_date=test_start_str,
        end_date=test_end_str,
        UP_DAY=best_params["UP_DAY"],
        DOWN_DAY=best_params["DOWN_DAY"],
        ADR_VOL_ADJ=best_params["ADR_VOL_ADJ"],
        KELT_MULT=best_params["KELT_MULT"],
        AUM_0=best_params["AUM_0"],
        invest_cash=best_params["invest_cash"],
        target_vol=best_params["target_vol"],
        max_leverage=best_params["max_leverage"],
        max_not_trade=best_params["max_not_trade"]
    )

    # Recalculate SR with risk-free rate for test data
    if not port_test.empty and 'tbill_ret' in port_test.columns and 'ret_long' in port_test.columns:
        sr_new_test = recalc_sr_with_rf(port_test)
        metrics_test["sr"] = round(sr_new_test, 2)
    else:
        metrics_test["sr"] = metrics_test.get("sr", -9999)

    test_result_entry = {
        "iteration": i,
        "training_start": train_start_str,
        "training_end": train_end_str,
        "testing_start": test_start_str,
        "testing_end": test_end_str,
        "used_params": json.dumps(best_params)
    }
    for k, v in metrics_test.items():
        test_result_entry[k] = v

    if port_test.empty:
        print(f"DEBUG: No test data returned for iteration {i}, skipping CSV writing.")
    else:
        port_test["iteration"] = i
        os.makedirs("results", exist_ok=True)
        port_test.to_csv(f"results/test_iteration_{i}.csv", index=False)
        print(f"DEBUG: Testing complete for iteration {i}, saved results to results/test_iteration_{i}.csv")
        print("DEBUG: Test period data summary:")
        if not port_test.empty:
            print(f"DEBUG: Test data start date: {port_test['caldt'].min()} end date: {port_test['caldt'].max()}")
            print(f"DEBUG: Number of rows in test data: {len(port_test)}")
            print(f"DEBUG: Sample of test data:\n{port_test.head()}")
        else:
            print("DEBUG: No test data returned.")

    return train_result_entry, test_result_entry

def run_wfa():
    start_date = "2010-01-01"
    end_date = "2024-01-01"
    training_period_years = 5
    testing_period_years = 2

    param_grid = {
        "UP_DAY": [20, 30, 40],
        "DOWN_DAY": [20, 30, 40],
        "ADR_VOL_ADJ": [1.2, 1.4, 1.6],
        "KELT_MULT": [2.4, 2.8, 3.2],
        "AUM_0": [1],
        "invest_cash": ["NO", "YES"],
        "target_vol": [0.015, 0.02],
        "max_leverage": [2.0],
        "max_not_trade": [0.20, 0.25],
    }

    os.makedirs("results", exist_ok=True)
    keys = list(param_grid.keys())
    values = (param_grid[k] for k in keys)
    param_combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    print("DEBUG: Pre-caching data...")
    run_asset_allocation(start_date="2010-01-01", end_date="2010-01-10")
    print("DEBUG: Pre-caching complete.")

    wfa_iterations = []
    current_train_start = start_dt
    while True:
        training_start = current_train_start
        training_end = training_start + pd.DateOffset(years=training_period_years) - pd.Timedelta(days=1)
        testing_start = training_end + pd.Timedelta(days=1)
        testing_end = testing_start + pd.DateOffset(years=testing_period_years) - pd.Timedelta(days=1)

        if testing_end > end_dt:
            break
        wfa_iterations.append((training_start, training_end, testing_start, testing_end))
        current_train_start = current_train_start + pd.DateOffset(years=testing_period_years)

    print("DEBUG: WFA Iterations:")
    for i, (ts, te, ss, se) in enumerate(wfa_iterations, start=1):
        print(f"DEBUG: Iteration {i}: Train {ts.strftime('%Y-%m-%d')} to {te.strftime('%Y-%m-%d')}, "
              f"Test {ss.strftime('%Y-%m-%d')} to {se.strftime('%Y-%m-%d')}")

    total_iterations = len(wfa_iterations)
    print(f"DEBUG: Starting WFA with {total_iterations} iterations.")

    training_results = []
    testing_results = []

    with ProcessPoolExecutor(max_workers=max(1, multiprocessing.cpu_count()//2)) as executor:
        iteration_futures = {}
        for i, (train_start, train_end, test_start, test_end) in enumerate(wfa_iterations, start=1):
            iteration_futures[executor.submit(run_single_iteration, i, train_start, train_end, test_start, test_end, param_combinations)] = i

        for future in as_completed(iteration_futures):
            i = iteration_futures[future]
            train_result_entry, test_result_entry = future.result()
            training_results.append(train_result_entry)
            testing_results.append(test_result_entry)

    training_df = pd.DataFrame(training_results)
    testing_df = pd.DataFrame(testing_results)
    training_df.to_csv("results/wfa_training_results.csv", index=False)
    testing_df.to_csv("results/wfa_testing_results.csv", index=False)

    print("DEBUG: WFA Complete.")
    print("DEBUG: Training results saved to results/wfa_training_results.csv")
    print("DEBUG: Testing results saved to results/wfa_testing_results.csv")

if __name__ == "__main__":
    run_wfa()
