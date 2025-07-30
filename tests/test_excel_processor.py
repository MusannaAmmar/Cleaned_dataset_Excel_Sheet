import sys
import os

import unittest
from src.core.excel_processor import ExcelProcessor
from config.settings import SUPPORTED_FILES, DATA_DIR


sys.dont_write_bytecode = True


class TestExcelProcessor(unittest.TestCase):
    def setUp(self):
        sample_file = os.path.join(DATA_DIR, SUPPORTED_FILES[1])
        self.processor = ExcelProcessor(sample_file) if sample_file else None
        self.file_paths = [os.path.join(DATA_DIR, f) for f in SUPPORTED_FILES]

    def test_load_files(self):
        if self.processor:
            sheet_info = self.processor.get_sheet_info()
            self.assertGreater(len(sheet_info), 0)

    def test_extract_data(self):
        if self.processor and self.processor.excel.sheet_names:
            sheet_name = self.processor.excel.sheet_names[0]
            df = self.processor.extract_data(sheet_name)
            self.assertIsNotNone(df)

if __name__ == '__main__':
    unittest.main()