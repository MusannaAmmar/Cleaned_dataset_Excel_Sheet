# examples/basic_usage.py
from src.core.excel_processor import ExcelProcessor
from config.settings import SUPPORTED_FILES, DATA_DIR
import os

def main():
    processor = ExcelProcessor()
    file_paths = [os.path.join(DATA_DIR, f) for f in SUPPORTED_FILES]
    processor.load_files(file_paths)
    processor.get_sheet_info()
    for file_path in file_paths:
        for sheet in processor.workbooks[file_path].sheetnames:
            processor.preview_data(file_path, sheet)

if __name__ == "__main__":
    main()