import os
import time
import tracemalloc
import pandas as pd
# import matplotlib.pyplot as plt
from subsets.ga_subset import SubsetSumGA
from subsets.dp_subsets import subset_sum_exists
from subsets.brute_force import *



def format_time(elapsed_seconds: float) -> str:
    """Format elapsed time into ms, s, or min."""
    if elapsed_seconds < 1:
        return f"{elapsed_seconds*1000:.2f} ms"
    elif elapsed_seconds < 60:
        return f"{elapsed_seconds:.2f} s"
    else:
        minutes = int(elapsed_seconds // 60)
        seconds = elapsed_seconds % 60
        return f"{minutes}m {seconds:.2f}s"


def benchmark_methods(transactions_df, targets_df):

    results = []

    # Clean column names
    transactions_df.columns = transactions_df.columns.map(lambda x: str(x).strip())
    targets_df.columns = targets_df.columns.map(lambda x: str(x).strip())
    transactions_df.columns = transactions_df.columns.str.replace('.', '_', regex=False)
    targets_df.columns = targets_df.columns.str.replace('.', '_', regex=False)

    # Detect amount & ID columns
    try:
        trans_amount_col = next(c for c in transactions_df.columns if 'amount' in c.lower())
        target_amount_col = next(c for c in targets_df.columns if 'amount' in c.lower())
        trans_id_col = next((c for c in transactions_df.columns if 'id' in c.lower()), None)
        target_id_col = next((c for c in targets_df.columns if 'id' in c.lower()), None)
    except StopIteration as e:
        raise KeyError(f"Missing required column: {e}")

    # Ensure ID columns exist and standardize names for brute_force_subset
    if trans_id_col is None:
        transactions_df["Transaction ID"] = [f"TX{i+1:03}" for i in range(len(transactions_df))]
        trans_id_col = "Transaction ID"
    else:
        # Rename to standard name expected by brute_force_subset
        if trans_id_col != "Transaction ID":
            transactions_df["Transaction ID"] = transactions_df[trans_id_col]
    
    if target_id_col is None:
        targets_df["Target ID"] = [f"TG{i+1:03}" for i in range(len(targets_df))]
        target_id_col = "Target ID"
    else:
        # Rename to standard name expected by brute_force_subset
        if target_id_col != "Target ID":
            targets_df["Target ID"] = targets_df[target_id_col]

    # Convert to numeric safely and remove NaN values
    transactions_df[trans_amount_col] = pd.to_numeric(transactions_df[trans_amount_col], errors='coerce')
    targets_df[target_amount_col] = pd.to_numeric(targets_df[target_amount_col], errors='coerce')
    
    transactions_df = transactions_df.dropna(subset=[trans_amount_col])
    targets_df = targets_df.dropna(subset=[target_amount_col])

    # Prepare data structures for algorithms
    trans_amounts = transactions_df[trans_amount_col].tolist()
    trans_ids = transactions_df["Transaction ID"].tolist()
    target_amounts = targets_df[target_amount_col].tolist()
    target_ids = targets_df["Target ID"].tolist()

    # Create transaction tuples for subset_sum_exists function
    transaction_tuples = list(zip(trans_amounts, trans_ids))

    # --- Guard against empty data ---
    if not trans_amounts or not target_amounts:
        raise ValueError("Transaction or target amounts are empty after cleaning.")

    print(f"Benchmarking with {len(transactions_df)} transactions and {len(targets_df)} targets")

    # 1️⃣ Brute Force
    try:
        tracemalloc.start()
        start = time.time()
        
        # Limit brute force to first few targets for performance
        limited_targets = targets_df.head(min(5, len(targets_df))).copy()
        
        brute_results = brute_force_subset(transactions_df, limited_targets)
        
        # Count matches
        brute_force_matches = 0
        if 'Task 2.1 Matches' in brute_results and not brute_results['Task 2.1 Matches'].empty:
            brute_force_matches += len(brute_results['Task 2.1 Matches'])
        if 'Task 2.2 Matches' in brute_results and not brute_results['Task 2.2 Matches'].empty:
            brute_force_matches += len(brute_results['Task 2.2 Matches'])
        
        elapsed = time.time() - start
        elapsed_ms = elapsed * 1000  # Convert to milliseconds for graphs
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        results.append({
            "Method": "Brute Force", 
            "Time (s)": round(elapsed, 3),  # Time in seconds for table
            "Time (ms)": round(elapsed_ms, 2),  # Time in milliseconds for graphs
            "Time Display": format_time(elapsed),  # Formatted time for display
            "Memory (KB)": round(peak / 1024, 2), 
            "Matches": brute_force_matches,
            "Targets Tested": len(limited_targets)
        })
        print(f"Brute Force completed: {elapsed:.3f}s, {brute_force_matches} matches")
    
    except Exception as e:
        print(f"Brute Force method failed: {e}")
        tracemalloc.stop() if tracemalloc.is_tracing() else None
        results.append({
            "Method": "Brute Force", 
            "Time (s)": float('inf'), 
            "Time (ms)": float('inf'),
            "Time Display": "Error",
            "Memory (KB)": 0, 
            "Matches": 0,
            "Targets Tested": 0
        })

    # 2️⃣ Dynamic Programming
    try:
        tracemalloc.start()
        start = time.time()
        dp_matches = 0
        
        for i, target_amount in enumerate(target_amounts):
            try:
                result = subset_sum_exists(
                    transaction_tuples,
                    (target_amount, target_ids[i]),
                    precision=100
                )
                if result and len(result) > 0 and result[0]:  # Check if match was found
                    dp_matches += 1
            except Exception as e:
                print(f"DP error for target {i}: {e}")
                continue
        
        elapsed = time.time() - start
        elapsed_ms = elapsed * 1000  # Convert to milliseconds for graphs
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        results.append({
            "Method": "Dynamic Programming", 
            "Time (s)": round(elapsed, 3),  # Time in seconds for table
            "Time (ms)": round(elapsed_ms, 2),  # Time in milliseconds for graphs
            "Time Display": format_time(elapsed),  # Formatted time for display
            "Memory (KB)": round(peak / 1024, 2), 
            "Matches": dp_matches,
            "Targets Tested": len(target_amounts)
        })
        print(f"Dynamic Programming completed: {elapsed:.3f}s, {dp_matches} matches")
    
    except Exception as e:
        print(f"Dynamic Programming method failed: {e}")
        tracemalloc.stop() if tracemalloc.is_tracing() else None
        results.append({
            "Method": "Dynamic Programming", 
            "Time (s)": float('inf'), 
            "Time (ms)": float('inf'),
            "Time Display": "Error",
            "Memory (KB)": 0, 
            "Matches": 0,
            "Targets Tested": 0
        })

    # 3️⃣ Genetic Algorithm
    try:
        tracemalloc.start()
        start = time.time()
        
        ga_solver = SubsetSumGA(trans_amounts, target_amounts, pop_size=50, gen_count=200)
        ga_results = ga_solver.find_subsets(verbose=False)
        
        # Count successful matches
        ga_matches = 0
        if hasattr(ga_results, '__len__'):
            ga_matches = len([r for r in ga_results if r is not None])
        elif ga_results is not None:
            ga_matches = 1
        
        elapsed = time.time() - start
        elapsed_ms = elapsed * 1000  # Convert to milliseconds for graphs
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        results.append({
            "Method": "Genetic Algorithm", 
            "Time (s)": round(elapsed, 3),  # Time in seconds for table
            "Time (ms)": round(elapsed_ms, 2),  # Time in milliseconds for graphs
            "Time Display": format_time(elapsed),  # Formatted time for display
            "Memory (KB)": round(peak / 1024, 2), 
            "Matches": ga_matches,
            "Targets Tested": len(target_amounts)
        })
        print(f"Genetic Algorithm completed: {elapsed:.3f}s, {ga_matches} matches")
    
    except Exception as e:
        print(f"Genetic Algorithm method failed: {e}")
        tracemalloc.stop() if tracemalloc.is_tracing() else None
        results.append({
            "Method": "Genetic Algorithm", 
            "Time (s)": float('inf'), 
            "Time (ms)": float('inf'),
            "Time Display": "Error",
            "Memory (KB)": 0, 
            "Matches": 0,
            "Targets Tested": 0
        })

    # Build results DataFrame
    df_results = pd.DataFrame(results)
    
    # Clean up infinite values for display
    df_results['Time (s)'] = df_results['Time (s)'].replace(float('inf'), 'Error')
    df_results['Time (ms)'] = df_results['Time (ms)'].replace(float('inf'), 'Error')
    
    print("\nBenchmark Results:")
    print(df_results)

    return df_results


# ---- Example usage ----
if __name__ == "__main__":

    transactions_file = os.path.join(DATA_DIR, "Customer_Ledger_Entries_FULL.xlsx")
    targets_file = os.path.join(DATA_DIR, "KH_Bank.XLSX")

    trans_df, targets_df = prepare_data(transactions_file, targets_file)

    # Use only small samples for testing
    chunk_trans_data = trans_df.head(10)
    chunk_target_data = targets_df.head(1)

    results = benchmark_methods(chunk_trans_data, chunk_target_data)
    print('Results:', results)
