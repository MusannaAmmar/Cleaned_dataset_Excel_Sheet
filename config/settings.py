import os

# Get the absolute path to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directory for raw sample files
DATA_DIR = os.path.join(BASE_DIR, "data", "sample")

# Directory to save processed files
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# List of expected input file names
SUPPORTED_FILES = [
    "Customer_Ledger_Entries_FULL.xlsx",
    "KH_Bank.XLSX"  
]

# Supported date formats (regex patterns)
DATE_FORMATS = [
    r'\d{1,2}/\d{1,2}/\d{4}',  # MM/DD/YYYY or DD/MM/YYYY
    r'\d{4}-\d{1,2}-\d{1,2}',  # YYYY-MM-DD
    r'\d{1,2}-\w{3}-\d{4}',    # DD-MON-YYYY
    r'Q[1-4]-\d{2}',           # Q1-24
    r'\w{3}\s\d{4}',           # Mar 2024
]

CURRENCY_SYMBOLS = {
    '$': 'USD',
    '€': 'EUR',
    '£': 'GBP',
    '₹': 'INR',
    'Ft': 'HUF',
}
