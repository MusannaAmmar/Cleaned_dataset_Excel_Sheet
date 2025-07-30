# tests/test_format_parser.py
import unittest
from src.core.format_parser import FormatParser
from decimal import Decimal
import pandas as pd

class TestFormatParser(unittest.TestCase):
    def setUp(self):
        self.parser = FormatParser()

    def test_parse_amount(self):
        test_cases = [
            ('$1,234.56', Decimal('1234.56')),
            ('(1,234.56)', Decimal('-1234.56')),
            ('1.2k', Decimal('1200')),
            ('€1.234,56', Decimal('1234.56')),
        ]
        for input_val, expected in test_cases:
            self.assertEqual(self.parser.parse_amount(input_val), expected)

    def test_parse_date(self):
        test_cases = [
            ('2023-01-15', pd.to_datetime('2023-01-15')),
            (44927, pd.to_datetime('2023-01-01')),
        ]
        for input_val, expected in test_cases:
            result = self.parser.parse_date(input_val)
            self.assertTrue(pd.isna(result) or result == expected)

if __name__ == '__main__':
    unittest.main()