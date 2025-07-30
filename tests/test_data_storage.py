# tests/test_data_storage.py
import unittest
import pandas as pd
from src.core.data_storage import DataStorage
from src.core.type_detector import DataTypeDetector

class TestDataStorage(unittest.TestCase):
    def setUp(self):
        self.storage = DataStorage()
        self.detector = DataTypeDetector()
        self.df = pd.DataFrame({
            'date': ['2023-01-15', '2023-02-01'],
            'amount': ['$1,234.56', '2,000'],
            'category': ['Sales', 'Expenses']
        })

    def test_store_data(self):
        column_types = {col: self.detector.analyze_column(self.df[col])[0] for col in self.df.columns}
        self.storage.store_data('test_data', self.df, column_types)
        self.assertIn('test_data', self.storage.data)

    def test_query_by_criteria(self):
        column_types = {col: self.detector.analyze_column(self.df[col])[0] for col in self.df.columns}
        self.storage.store_data('test_data', self.df, column_types)
        result = self.storage.query_by_criteria('test_data', [('amount', '>', 1500)])
        self.assertEqual(len(result), 1)

if __name__ == '__main__':
    unittest.main()