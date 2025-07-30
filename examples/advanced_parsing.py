import pandas as pd
import openpyxl
# from data.sample import 

df = pd.read_excel("data/sample/Customer_Ledger_Entries_FULL.xlsx")  # defaults to first sheet
print(df.head())