from subsets.brute_force import prepare_data
import os
import pandas as pd
import numpy as np
from config.settings import DATA_DIR


def generate_candidates(transactions, targets):
    """
    Generate all possible transaction-target pairs (cross join)
    using the unique IDs from prepare_data().
    """
    transactions = transactions.reset_index(drop=True)
    targets = targets.reset_index(drop=True)

    transactions['Transaction Index'] = transactions.index
    targets['Target Index'] = targets.index

    # Rename before merge to keep IDs distinct
    transactions = transactions.rename(columns={"Transaction ID": "Transaction Identifier"})
    targets = targets.rename(columns={"Target ID": "Target Identifier"})

    transactions['key'] = 1
    targets['key'] = 1

    candidate_pairs = pd.merge(transactions, targets, on='key')
    candidate_pairs.drop(columns='key', inplace=True)

    # Select only the four columns we need
    candidates_df = candidate_pairs[[
        'Transaction Index',
        'Target Index',
        'Transaction Identifier',
        'Target Identifier'
    ]]

    return candidates_df



def feature_engineering(candidates_df, transactions, targets):
    """
    Build features for each candidate transaction-target pair.

    Features:
      - amount_diff: transaction - target
      - abs_amount_diff
      - rel_diff: abs_diff / (abs(target) + eps)
      - txn_amount, targ_amount
      - fuzzy_amount_str_sim: string similarity between amounts
    """
    eps = 1e-9
    rows = []

    for _, r in candidates_df.iterrows():
        # Get row indices from candidates
        ti = int(r['Transaction Index'])
        gi = int(r['Target Index'])

        # Get amounts
        # Detect amount columns dynamically
        trans_amount_col = [col for col in transactions.columns if 'amount' in col.lower()][0]
        targ_amount_col = [col for col in targets.columns if 'amount' in col.lower()][0]

        txn_amt = float(transactions.at[ti, trans_amount_col])
        targ_amt = float(targets.at[gi, targ_amount_col])

        # txn_amt = float(transactions.at[ti, 'Transaction Amount'])
        # targ_amt = float(targets.at[gi, 'Target Amount'])

        # Numeric features
        amount_diff = txn_amt - targ_amt
        abs_amount_diff = abs(amount_diff)
        rel_diff = abs_amount_diff / (abs(targ_amt) + eps)



        # Append feature row
        rows.append({
            'Target Index': gi,
            'Transaction Index': ti,
            'trans_amount': txn_amt,
            'targ_amount': targ_amt,
            'amount_diff': amount_diff,
            'abs_amount_diff': abs_amount_diff,
            'rel_diff': rel_diff,
        })

    # Convert to DataFrame
    feat_df = pd.DataFrame(rows)
    return feat_df


if __name__ == '__main__':
    transactions_file = os.path.join(DATA_DIR, "Customer_Ledger_Entries_FULL.xlsx")
    targets_file = os.path.join(DATA_DIR, "KH_Bank.xlsx")

    trans_df, target_df = prepare_data(transactions_file, targets_file)

    # Limit for quick testing
    trans_df = trans_df.head(10)
    target_df = target_df.head(3)
    candidates_df = generate_candidates(trans_df, target_df)
    features_df = feature_engineering(candidates_df, trans_df, target_df)

    print(features_df.head())
