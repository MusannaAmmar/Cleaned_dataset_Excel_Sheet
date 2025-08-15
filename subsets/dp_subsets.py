import pandas as pd
import os
import time
from tqdm import tqdm
from config.settings import DATA_DIR
from subsets.brute_force import prepare_data

def subset_sum_exists(transactions, target, max_sum=None, precision=100):
    """
    Check if a subset of transactions sums to the target value.
    Args:
        transactions: List of tuples [(amount, trans_id), ...]
        target: Tuple (target_amount, target_id)
        max_sum: Maximum sum to consider (optional)
        precision: Scaling factor for floats (e.g., 100 for 2 decimals)
    Returns:
        Tuple (success, target_id, trans_ids):
            - success: Boolean (True if subset exists)
            - target_id: ID of the target
            - trans_ids: List of Transaction IDs forming the subset (empty if none)
    """
    if not transactions:
        return (target[0] == 0, target[1], [])
    
    # Extract target amount and ID
    target_amount = target[0]
    target_id = target[1]
    
    # Convert floats to integers by multiplying by precision
    # Keep both positive and negative transactions
    nums = [(int(amount * precision), trans_id) for amount, trans_id in transactions if amount != 0]
    target_scaled = int(target_amount * precision)
    
    if not nums:
        return (target_scaled == 0, target_id, [])
    
    # Calculate reasonable max_sum to avoid memory issues
    if not max_sum:
        sum_positive = sum(n for n, _ in nums if n > 0)
        sum_negative = sum(n for n, _ in nums if n < 0)
        
        # Set max_sum to be reasonable - consider both directions
        max_sum = min(
            sum_positive - sum_negative,  # Maximum possible range
            abs(target_scaled) * 10,      # 10x target as upper bound
            1000000  # Hard cap at 1M to prevent memory issues
        )
        
        if max_sum <= 0:
            max_sum = 1000000
    
    # Check memory feasibility (more accurate estimation)
    estimated_memory_mb = (max_sum * 2 + 1) * 8 / (1024 * 1024)  # 8 bytes per pointer
    if estimated_memory_mb > 1000:  # More than 1GB
        print(f"Warning: Estimated memory usage: {estimated_memory_mb:.2f} MB")
        print("Consider reducing precision or using a smaller max_sum")
        return (False, target_id, [])
    
    try:
        # Offset for negative indices (shift everything to positive range)
        offset = max_sum
        dp_size = 2 * max_sum + 1
        
        # DP array: stores list of trans_ids for each achievable sum (or None)
        dp = [None] * dp_size
        dp[offset] = []  # Empty subset achieves sum 0 (at index offset)
        
        # Fill DP table
        for num, trans_id in nums:
            if abs(num) > max_sum:
                continue
                
            # Iterate in appropriate direction to avoid using updated values
            if num > 0:
                for s in range(dp_size - 1, num - 1, -1):
                    if dp[s - num] is not None:
                        if dp[s] is None:  # Only set if not already set
                            dp[s] = dp[s - num] + [trans_id]
            else:  # num < 0
                for s in range(-num, dp_size):
                    if dp[s - num] is not None:
                        if dp[s] is None:  # Only set if not already set
                            dp[s] = dp[s - num] + [trans_id]
        
        # Check if target is achievable
        target_index = target_scaled + offset
        if target_index < 0 or target_index >= dp_size:
            return (False, target_id, [])
            
        success = dp[target_index] is not None
        trans_ids = dp[target_index] if success else []
        return (success, target_id, trans_ids)
    
    except MemoryError:
        print(f"MemoryError: max_sum={max_sum} is too large. Try reducing precision or capping max_sum.")
        return (False, target_id, [])

def subset_sum_exists_optimized(transactions, target, precision=100):
    """
    Optimized version that uses a more memory-efficient approach for large datasets.
    Uses a set-based DP approach instead of storing full transaction lists.
    """
    if not transactions:
        return (target[0] == 0, target[1], [])
    
    target_amount = target[0]
    target_id = target[1]
    
    # Convert to integers
    nums = [(int(amount * precision), trans_id) for amount, trans_id in transactions if amount != 0]
    target_scaled = int(target_amount * precision)
    
    if not nums:
        return (target_scaled == 0, target_id, [])
    
    # Use iterative deepening - try small subsets first
    from itertools import combinations
    
    # Try subsets of increasing size
    for subset_size in range(1, min(len(nums) + 1, 20)):  # Limit to reasonable subset sizes
        for combo in combinations(nums, subset_size):
            if sum(amount for amount, _ in combo) == target_scaled:
                trans_ids = [trans_id for _, trans_id in combo]
                return (True, target_id, trans_ids)
    
    return (False, target_id, [])

if __name__ == '__main__':
    # Start timing the whole process
    start_time = time.time()
    
    transactions_file = os.path.join(DATA_DIR, "Customer_Ledger_Entries_FULL.xlsx")
    targets_file = os.path.join(DATA_DIR, "KH_Bank.XLSX")

    print("Loading data files...")
    load_start = time.time()
    try:
        trans_df, target_df = prepare_data(transactions_file, targets_file)
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)
    load_time = time.time() - load_start
    print(f"Data loading completed in {load_time:.2f} seconds")

    # Find amount and ID columns
    trans_amount_col = [col for col in trans_df.columns if 'amount' in col.lower()]
    target_amount_col = [col for col in target_df.columns if 'amount' in col.lower()]
    trans_id_col = [col for col in trans_df.columns if 'transaction id' in col.lower()]
    target_id_col = [col for col in target_df.columns if 'target id' in col.lower()]

    if not trans_amount_col or not target_amount_col or not trans_id_col or not target_id_col:
        print("Error: Missing 'amount' or 'ID' column in one or both DataFrames")
        exit(1)

    trans_amount_col = trans_amount_col[0]
    target_amount_col = target_amount_col[0]
    trans_id_col = trans_id_col[0]
    target_id_col = target_id_col[0]

    # Pair transaction amounts with their IDs
    trans_values = list(zip(
        pd.to_numeric(trans_df[trans_amount_col], errors='coerce').dropna(),
        trans_df[trans_id_col].dropna()
    ))
    
    # Pair target amounts with their IDs
    target_values = list(zip(
        pd.to_numeric(target_df[target_amount_col], errors='coerce').dropna(),
        target_df[target_id_col].dropna()
    ))

    print(f"Loaded {len(trans_values)} transaction values, {len(target_values)} target values")

    # Process targets
    if not target_values:
        print("No valid target values found")
        exit(1)

    # Use more reasonable max_sum calculation
    precision = 100  # Define precision here
    max_target = max((abs(t[0]) for t in target_values if t[0] != 0), default=1000)
    reasonable_max_sum = min(int(max_target * precision * 5), 500000)  # Much more conservative
    
    print(f"Using max_sum: {reasonable_max_sum}, precision: {precision}")
    
    # Prepare results list for CSV output
    results = []
    
    # Process all targets with progress bar
    print(f"\nProcessing {len(target_values)} targets...")
    processing_start = time.time()
    
    for i, t in enumerate(tqdm(target_values, desc="Finding subsets", unit="target")):
        target_amount, target_id = t
        
        # Try optimized version first for large datasets
        if len(trans_values) > 1000:
            success, returned_target_id, trans_ids = subset_sum_exists_optimized(trans_values, t, precision)
        else:
            success, returned_target_id, trans_ids = subset_sum_exists(trans_values, t, max_sum=reasonable_max_sum, precision=precision)
        
        # Calculate sum and collect transaction details if subset found
        if success and trans_ids:
            # Get the actual amounts for verification
            matching_transactions = [(amount, tid) for amount, tid in trans_values if tid in trans_ids]
            subset_sum = sum(amount for amount, tid in matching_transactions)
            subset_trans_ids = [tid for amount, tid in matching_transactions]
            
            # Verify the match
            is_match = abs(subset_sum - target_amount) < 0.01
            
            result_row = {
                'Target_ID': target_id,
                'Target_Amount': target_amount,
                'Subset_Found': True,
                'Subset_Sum': subset_sum,
                'Sum_Match': is_match,
                'Transaction_Count': len(subset_trans_ids),
                'Transaction_IDs': '|'.join(map(str, subset_trans_ids)),  # Use | as separator
                'Processing_Method': 'Optimized' if len(trans_values) > 1000 else 'Standard'
            }
        else:
            result_row = {
                'Target_ID': target_id,
                'Target_Amount': target_amount,
                'Subset_Found': False,
                'Subset_Sum': None,
                'Sum_Match': False,
                'Transaction_Count': 0,
                'Transaction_IDs': '',
                'Processing_Method': 'Optimized' if len(trans_values) > 1000 else 'Standard'
            }
        
        results.append(result_row)
        
        # Print progress for first few and last few
        if i < 5 or i >= len(target_values) - 5:
            status = "FOUND" if success else "NOT FOUND"
            print(f"Target {target_id} ({target_amount}): {status}")
            if success and trans_ids:
                print(f"  -> Sum: {subset_sum}, IDs: {trans_ids[:5]}{'...' if len(trans_ids) > 5 else ''}")
    
    processing_time = time.time() - processing_start
    print(f"\nProcessing completed in {processing_time:.2f} seconds")
    
    # Create DataFrame and save to CSV
    print("\nSaving results to CSV...")
    results_df = pd.DataFrame(results)
    
    # Add summary statistics
    total_targets = len(results_df)
    found_subsets = len(results_df[results_df['Subset_Found'] == True])
    success_rate = (found_subsets / total_targets) * 100 if total_targets > 0 else 0
    
    print(f"\nSUMMARY:")
    print(f"Total targets processed: {total_targets}")
    print(f"Subsets found: {found_subsets}")
    print(f"Success rate: {success_rate:.1f}%")
    
    # Generate output filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(DATA_DIR, f"subset_sum_results_{timestamp}.csv")
    
    # Save to CSV
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    
    # Display sample results
    print(f"\nSample results (first 10 rows):")
    print(results_df.head(10).to_string(index=False))
    
    # Show some statistics
    if found_subsets > 0:
        avg_transaction_count = results_df[results_df['Subset_Found'] == True]['Transaction_Count'].mean()
        max_transaction_count = results_df[results_df['Subset_Found'] == True]['Transaction_Count'].max()
        print(f"\nSubset Statistics:")
        print(f"Average transactions per subset: {avg_transaction_count:.1f}")
        print(f"Maximum transactions in a subset: {max_transaction_count}")
    
    # Total execution time
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    print(f"Time breakdown:")
    print(f"  Data loading: {load_time:.2f}s ({(load_time/total_time)*100:.1f}%)")
    print(f"  Processing: {processing_time:.2f}s ({(processing_time/total_time)*100:.1f}%)")
    print(f"  Other: {(total_time-load_time-processing_time):.2f}s ({((total_time-load_time-processing_time)/total_time)*100:.1f}%)")