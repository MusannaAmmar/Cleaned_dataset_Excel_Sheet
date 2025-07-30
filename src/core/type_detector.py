import re
import warnings
from decimal import Decimal, InvalidOperation
from typing import Dict, Tuple
from src.core.format_parser import*
import pandas as pd
import numpy as np
import os
from config.settings import*
from src.core.excel_processor import*


class TypeDetector:
    """
    Public API:
        detect(df, column) -> ("str"|"number"|"datetime", confidence)
        detect_all(df)     -> dict {col: (dtype, confidence)}
    """

    MIN_CONFIDENCE = 0.6  # threshold for acceptance
    
    # Common date formats for financial data
    COMMON_DATE_FORMATS = [
        # ISO formats
        '%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M',
        # European formats
        '%d/%m/%Y', '%d/%m/%Y %H:%M:%S', '%d/%m/%Y %H:%M',
        '%d.%m.%Y', '%d.%m.%Y %H:%M:%S', '%d %b %Y', '%d %B %Y',
        # US formats
        '%m/%d/%Y', '%m/%d/%Y %H:%M:%S', '%m/%d/%Y %H:%M',
        '%m-%d-%Y', '%b %d, %Y', '%B %d, %Y',
        # Compact formats
        '%Y%m%d', '%d%m%Y', '%m%d%Y',
        # Special financial formats
        'Q%q-%y', 'Quarter %q %Y', '%b-%Y', '%B %Y',
        # Time formats
        '%H:%M:%S', '%H:%M'
    ]

    # ---------- public ----------

    def detect(self,series: pd.Series) -> Tuple[str, float]:
        """
        Detect the most probable type of a pandas Series.
        Returns: (dtype, confidence)
        """
        s = series.dropna()

        if s.empty:
            return "str", 1.0

        # 1. Date attempt -----------------------------------------------------
        dt_succ, dt_conf = self._try_datetime(s)
        if dt_succ:
            return "datetime", dt_conf

        # 2. Number attempt ---------------------------------------------------
        num_succ, num_conf = self._try_number(s)
        if num_succ:
            return "number", num_conf

        # 3. Default ----------------------------------------------------------
        return "str", 1.0
        

    def detect_all(self,df: pd.DataFrame) -> Dict[str, Tuple[str, float]]:
        """Run detect() on every column."""
        return {col: TypeDetector.detect(df[col]) for col in df.columns}


    def _try_datetime(self,s: pd.Series) -> Tuple[bool, float]:
        """Enhanced datetime detection with format inference and Excel support"""
        # Handle Excel serial dates first (numeric values in date range)
        if pd.api.types.is_numeric_dtype(s):
            if s.min() > 1000 and s.max() < 1000000:
                try:
                    # Convert Excel serial numbers to dates
                    converted = pd.to_datetime(
                        s, 
                        unit='D', 
                        origin='1899-12-30', 
                        errors='coerce'
                    )
                    success_rate = converted.notnull().mean()
                    if success_rate >= 0.95:
                        return True, success_rate
                except Exception:
                    pass

        # Try common explicit formats
        max_success = 0.0
        s_str = s.astype(str)
        
        for fmt in TypeDetector.COMMON_DATE_FORMATS:
            try:
                converted = pd.to_datetime(s_str, format=fmt, errors='coerce')
                success_rate = converted.notnull().mean()
                if success_rate >= 0.95:
                    return True, success_rate
                max_success = max(max_success, success_rate)
            except Exception:
                continue
        quarter_regex = re.compile(r"Q([1-4])-(\d{2})", re.IGNORECASE)
        matched = 0

        for val in s.astype(str):
            m = quarter_regex.match(val.strip())
            if m:
                quarter = int(m.group(1))
                year = int(m.group(2))
                if year < 100:
                    year += 2000
                month = (quarter - 1) * 3 + 1
                try:
                    pd.Timestamp(year=year, month=month, day=1)
                    matched += 1
                except Exception:
                    continue

        if matched / len(s) >= 0.95:
            return True, matched / len(s)

        # Final fallback with warnings suppressed
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            try:
                converted = pd.to_datetime(s_str, errors='coerce')
                success_rate = converted.notnull().mean()
                if success_rate >= 0.95:
                    return True, success_rate
                max_success = max(max_success, success_rate)
            except Exception:
                pass
            
        return False, max_success


    def _try_number(self,s: pd.Series) -> Tuple[bool, float]:
        # Remove common currency symbols & thousand separators
        cleaned = (
            s.astype(str)
            .str.replace(r"[€$₹£¥,]", "", regex=True)
            .str.replace(r"\s+", "", regex=True)
        )

        # Allow negative numbers in parentheses  (1 234)  or  1 234-
        cleaned = cleaned.str.replace(r"\(([^)]+)\)", r"-\1", regex=True)
        cleaned = cleaned.str.replace(r"(\d)-$", r"-\1", regex=True)

        # Accept K, M, B suffixes (1.2K → 1200)
        suffix_map = {"K": 1e3, "M": 1e6, "B": 1e9}
        pattern = re.compile(r"^([+-]?\d+\.?\d*)([KkMmBb])$")

        def parse(x: str) -> bool:
            # Abbreviated amounts
            m = pattern.match(x)
            if m:
                return True
            # Exact decimals
            try:
                Decimal(x)
                return True
            except InvalidOperation:
                return False

        success_mask = cleaned.apply(parse)
        conf = success_mask.mean()
        return conf >= TypeDetector.MIN_CONFIDENCE, conf
    

def main():
    print("Starting Financial Data Parser...\n")
    
    file_path = os.path.join(DATA_DIR, SUPPORTED_FILES[0])
    # print(f"📄 Processing file: {file_path}")

    # for file_path in files_path:
    #     # file_path = os.path.join(DATA_DIR, file_name)
    #     print(f"📄 Processing file: {file_path}")

    try:
        processor = ExcelProcessor(file_path)
        parser = FormatParser()
        for sheet_name in processor.excel.sheet_names:
            print(f"\n📑 Analyzing sheet: {sheet_name}")
            df = processor.extract_data(sheet_name)
            if df is not None and isinstance(df, pd.DataFrame):
                # detector = TypeDetector()
                # for column in df.columns:
                #     # dtype, confidence = detector.analyze_column(df[column])
                #     data_type, confidence = TypeDetector.detect(df[column])
                #     print(f" - Column '{column}': Type = {data_type}, Confidence = {confidence:.2f}")
                #     if data_type == 'number':
                #         parsed = df[column].apply(parser.parse_amount)
                #         print(f"   Sample parsed amounts: {parsed.head().tolist()}")
                #     elif data_type == 'date':
                #         parsed = df[column].apply(parser.parse_date)
                #         print(f"   Sample parsed dates: {parsed.head().tolist()}")
                detector = TypeDetector()
                for column in df.columns:
                    data_type, confidence = detector.detect(df[column])
                    print(f" - Column '{column}': Type = {data_type}, Confidence = {confidence:.2f}")
                    if data_type == 'number':
                        parsed = df[column].apply(parser.parse_amount)
                        print(f"   Sample parsed amounts: {parsed.head().tolist()}")
                    elif data_type == 'date':
                        parsed = df[column].apply(parser.parse_date)
                        print(f"   Sample parsed dates: {parsed.head().tolist()}")

            else:
                print(f"⚠️ Failed to extract DataFrame from sheet '{sheet_name}'.")
    except Exception as e:
        print(f"❌ Error processing file {file_path}: {e}")
if __name__ == "__main__":
    main()




