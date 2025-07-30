 # examples/performance_demo.py
from src.core.excel_processor import ExcelProcessor
from src.core.data_storage import DataStorage
from src.core.type_detector import DataTypeDetector
from config.settings import SUPPORTED_FILES, DATA_DIR
import os
import cProfile

def main():
    file_paths = [os.path.join(DATA_DIR, f) for f in SUPPORTED_FILES]
    processor = ExcelProcessor(file_paths)
    cProfile.runctx('processor.load_files(file_paths)', globals(), locals(), 'excel_profile.prof')

    storage = DataStorage()
    detector = DataTypeDetector()
    df = processor.extract_data(file_paths[0], processor.get_sheet_info[file_paths[0]].sheetnames[0])
    column_types = {col: detector.analyze_column(df[col])[0] for col in df.columns}
    cProfile.runctx('storage.store_data("test_data", df, column_types)', globals(), locals(), 'storage_profile.prof')

if __name__ == "__main__":
    main()