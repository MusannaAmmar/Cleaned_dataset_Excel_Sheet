# # tests/test_type_detector.py
# import unittest
# import pandas as pd
# from src.core.type_detector import TypeDetector

# class TestTypeDetector(unittest.TestCase):
#     def setUp(self):
#         self.detector = TypeDetector()

#     def test_date_detection(self):
#         data = pd.Series(['2023-01-15', '01/15/2023', 'Q1-24'])
#         type_, score = self.detector.detect(data)
#         self.assertEqual(type_, 'date')

#     def test_number_detection(self):
#         data = pd.Series(['$1,234.56', '1.2k', '-500'])
#         type_, score = self.detector.detect(data)
#         self.assertEqual(type_, 'number')

#     def test_string_detection(self):
#         data = pd.Series(['Account A', 'Transaction XYZ', ''])
#         type_, score = self.detector.detect(data)
#         self.assertEqual(type_, 'string')

# if __name__ == '__main__':
#     unittest.main()

import unittest
import pandas as pd
from src.core.type_detector import TypeDetector

class TestTypeDetector(unittest.TestCase):

    def setUp(self):
        self.detector = TypeDetector()

    def test_detect_datetime_iso(self):
        data = pd.Series(['2023-01-01', '2023-05-12', '2023-12-31'])
        dtype, confidence = self.detector.detect(data)
        self.assertEqual(dtype, 'datetime')
        self.assertGreaterEqual(confidence, 0.95)

    def test_detect_datetime_quarter_format(self):
        data = pd.Series(['Q1-24', 'Q3-22', 'Q4-25'])
        dtype, confidence = self.detector.detect(data)
        self.assertEqual(dtype, 'datetime')
        self.assertGreaterEqual(confidence, 0.95)

    def test_detect_number_with_symbols(self):
        data = pd.Series(['$1,000.50', '€2,345.00', '£3,200'])
        dtype, confidence = self.detector.detect(data)
        self.assertEqual(dtype, 'number')
        self.assertGreaterEqual(confidence, 0.95)

    def test_detect_number_with_suffixes(self):
        data = pd.Series(['1.5K', '2.3M', '500'])
        dtype, confidence = self.detector.detect(data)
        self.assertEqual(dtype, 'number')
        self.assertGreaterEqual(confidence, 0.95)

    def test_detect_string_fallback(self):
        data = pd.Series(['apple', 'banana', 'cherry'])
        dtype, confidence = self.detector.detect(data)
        self.assertEqual(dtype, 'str')
        self.assertEqual(confidence, 1.0)

    def test_detect_empty_series(self):
        data = pd.Series([])
        dtype, confidence = self.detector.detect(data)
        self.assertEqual(dtype, 'str')
        self.assertEqual(confidence, 1.0)

    def test_detect_mixed_types(self):
        data = pd.Series(['2023-01-01', '1000', 'hello'])
        dtype, confidence = self.detector.detect(data)
        self.assertEqual(dtype, 'str')  # Should fall back to string
        self.assertEqual(confidence, 1.0)

if __name__ == '__main__':
    unittest.main()
