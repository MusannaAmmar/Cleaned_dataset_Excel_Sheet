import pandas as pd
import sys
from config.settings import SUPPORTED_FILES, DATA_DIR

import os

class ExcelProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        print(f'Loading Excel file: {self.file_path}')
        self.excel = pd.ExcelFile(self.file_path)

    def get_sheet_info(self):
        info = []
        for sheet in self.excel.sheet_names:
            df = self.excel.parse(sheet)
            sheet_info = {
                "sheet_name": sheet,
                "num_rows": df.shape[0],
                "num_columns": df.shape[1],
                "column_names": df.columns.tolist()
            }
            info.append(sheet_info)
        print(info)
        return info

    def extract_data(self, sheet_name):
        return self.excel.parse(sheet_name)

    def preview_data(self, sheet_name, rows=5):
        return self.excel.parse(sheet_name).head(rows)

if __name__ == "__main__":
    print("\n Running direct test of ExcelProcessor...")
    try:
        test_files = [os.path.join(DATA_DIR,f) for f in SUPPORTED_FILES]
        for test_file in test_files:
            print(f"Testing with file: {test_file}")

            processor = ExcelProcessor(test_file)
            processor.get_sheet_info()
            print("\nTest completed successfully!")
    except Exception as e:
        print(f"\nTest failed with error: {e}")