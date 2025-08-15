import itertools
import time
from config.settings import  DATA_DIR
import os
import pandas as pd
from src.core.format_parser import *


def prepare_data(transaction,target):
    trans_df=pd.read_excel(transaction)
    target_df=pd.read_excel(target)
    parser=FormatParser()

    trans_df["Transaction ID"] = [f"TX{i+1:03}" for i in range(len(trans_df))]
    target_df["Target ID"] = [f"TG{i+1:03}" for i in range(len(target_df))]

    for df in [trans_df,target_df]:
        for col in df.columns:
            if 'amount' in col.lower():
                df[col] = df[col].apply(lambda x: parser.parse_amount(x))


    return trans_df,target_df



def brute_force_subset(transactions, target):
    start_time = time.time()

    trans_amount_col = [col for col in transactions.columns if 'amount' in col.lower()][0]
    target_amount_col = [col for col in target.columns if 'amount' in col.lower()][0]

    direct_matches = []
    results={}

    for _, trans_row in transactions.iterrows():
        for _ ,target_rows in target.iterrows():
            # print(f"Checking Transaction Row {trans_row['Transaction ID']} (value={trans_row[trans_amount_col]}) "
            #       f"against Target Row {target_rows['Target ID']} (value={target_rows[target_amount_col]})")
            if trans_row[trans_amount_col]==target_rows[target_amount_col]:
                # print(f'Match found between {trans_row['Transaction ID']}:{trans_row[trans_amount_col]} and {target_rows['Target ID']}:{target_rows[target_amount_col]}')
                direct_matches.append({
                    "transaction_amount": trans_row[trans_amount_col],
                    "target_amount": target_rows[target_amount_col]
                })

    
    task_2_1_time = time.time() - start_time
    results['Task 2.1 Matches'] = pd.DataFrame(direct_matches)
    results['Task 2.1 Time'] = task_2_1_time

    # print("Time:", time.time() - start_time)

    #------Subset Sum-------

    subset_sum_matches=[]

    trans_amounts = transactions[trans_amount_col].tolist()
    trans_ids = transactions['Transaction ID'].tolist()

    for _, target_row in target.iterrows():
        target_amt = target_row[target_amount_col]
        target_id = target_row['Target ID']
        found = False

        # Brute force: try all subset sizes from 1 to len(transactions)
        for r in range(1, len(trans_amounts) + 1):
            for combo in itertools.combinations(range(len(trans_amounts)), r):
                combo_ids = [trans_ids[i] for i in combo]
                combo_vals = [trans_amounts[i] for i in combo]
                # print(f"Checking Target {target_id} (value={target_amt}) "
                #       f"against Transactions {combo_ids} (values={combo_vals})")
                if sum(trans_amounts[i] for i in combo) == target_amt:
                    subset_sum_matches.append({
                        'Target Identifier': target_id,
                        'Matched Transactions': [trans_ids[i] for i in combo],
                        'Sum': target_amt
                    })
                    found = True
                    break  # Found a match for this target
            if found:
                break 

    task_2_2_time = time.time() - start_time
    results['Task 2.2 Matches'] = pd.DataFrame(subset_sum_matches)
    results['Task 2.2 Time'] = task_2_2_time


    return results




if __name__ == '__main__':
    # Assuming SUPPORTED_FILES contains both file names
    transactions_file = os.path.join(DATA_DIR, "Customer_Ledger_Entries_FULL.xlsx")
    targets_file = os.path.join(DATA_DIR, "KH_Bank.XLSX")


    # Use only first 10 rows for testing
    trans_df,target_df=prepare_data(transactions_file,targets_file)
    chunk_trans_data = trans_df.head(1)
    chunk_target_data = target_df.head(1)

    result = brute_force_subset(chunk_trans_data, chunk_target_data)
    print('Results:', result)
        
