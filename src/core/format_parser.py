import pandas as pd
import re
from decimal import Decimal,InvalidOperation
from config.settings import CURRENCY_SYMBOLS, DATA_DIR, SUPPORTED_FILES,DATE_FORMATS
import os
from typing import Dict, Tuple,Any
import numpy as np


class FormatParser:
    def __init__(self):
        self.suffix_map = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}

    def parse_amount(self,value:any)->Decimal:
        if pd.isna(value):
        # raise ValueError("Cannot parse NaN into amount")
            return None
      
      

        s=str(value).strip()
        sign=1

        if s.startswith('(') and s.endswith(')'):
          sign=-1
          s=s[1:-1].strip()
        elif s.endswith('-'):
          sign=-1
          s=s[:-1].strip()
        elif s.startswith('-'):
          sign=-1
          s=s[1:].strip()

        m=re.fullmatch(r'([+-]?\d+(?:\.\d+)?)\s*([KkMmBb])',s,flags=re.IGNORECASE)
        if m:
          return Decimal(m.group(1)) * self.suffix_map[m.group(2).upper()]
        s = re.sub(r"[^\d.,-]", "", s)

        if '.' in s and ',' in s:
          last_dot= s.rfind('.')
          last_comma=s.rfind(',')

          if last_dot> last_comma:
            s=s.replace(',','')
          else:
            s=s.replace('.','').replace(',','.')
        elif ',' in s:
          if re.search(r',\d{1,2}$',s):
            s=s.replace(',','.',1)
          s=s.replace(',','')
        if s.count('.')>1:
          parts=s.split('.')
          s=parts[0]+ ''.join(parts[1:-1])+'.'+ parts[-1]

        try:
          return sign*Decimal(s)
        except InvalidOperation as exc:
        #  raise ValueError(f'Invalid amount:{value} ') from exc
          return None





    def parse_date(self,value: Any) -> pd.Timestamp:
        # Handle NaN or None values
        if pd.isna(value):
            return None
        # Handle Excel serial dates (numeric values)
        if isinstance(value, (int, float)):
            try:
                timestamp= pd.to_datetime('1899-12-30') + pd.to_timedelta(float(value), unit='D')
                # print(timestamp)
                return timestamp.round('1s')
            except (ValueError, OverflowError):
                return None
        # Handle string dates
        try:
            # Attempt to parse string as datetime
            return pd.to_datetime(value, errors='coerce')
        except (ValueError, TypeError):
            return None



    def drop_na_columns(self,df: pd.DataFrame, threshold: float = 0.6) -> pd.DataFrame:
        """
        Drops columns with more than `threshold` fraction of null (NaN) values.

        Parameters:
            df (pd.DataFrame): Input DataFrame.
            threshold (float): Proportion of missing values above which to drop a column (default is 0.6).

        Returns:
            pd.DataFrame: Cleaned DataFrame with columns dropped.
        """
        # Calculate the fraction of missing values per column
        null_fraction = df.isna().mean()

        # Keep only columns where missing fraction <= threshold
        columns_to_keep = null_fraction[null_fraction <= threshold].index

        # Return cleaned DataFrame
        return df[columns_to_keep]
    

    def parse_amount_vectorized(self, series: pd.Series) -> pd.Series:
        """
        Vectorized version of amount parsing (10-50x faster than apply)
        Returns a Series of Decimal values
        """
        if series.empty:
            return pd.Series([], dtype=object)

        cleaned = series.astype(str).str.strip()

        signs = np.where(
            cleaned.str.startswith('(') & cleaned.str.endswith(')') |
            cleaned.str.endswith('-') |
            cleaned.str.startswith('-'),
            -1, 1
        )

        cleaned = cleaned.str.replace(r"[€$₹£¥,]", "", regex=True)
        cleaned = cleaned.str.replace(r"\s+", "", regex=True)
        cleaned = cleaned.str.replace(r"\(([^)]+)\)", r"\1", regex=True)
        cleaned = cleaned.str.replace(r"(\d)-$", r"\1", regex=True)
        cleaned = cleaned.str.replace(r"^-", "", regex=True)

        suffix_mask = cleaned.str.contains(r"[KMB]$", case=False)
        suffix_values = cleaned[suffix_mask].str.extract(r"([\d.]+)([KMB])", expand=False, flags=re.I)

        def process_suffix(row):
            try:
                num = Decimal(row[0])
                suffix = row[1].upper()
                return num * self.suffix_map[suffix]
            except (TypeError, InvalidOperation):
                return None

        def process_standard(val):
            try:
                if val.count('.') > 1 and ',' in val:
                    parts = val.split('.')
                    val = parts[0] + ''.join(parts[1:-1]) + '.' + parts[-1]
                elif ',' in val and '.' not in val:
                    if re.search(r",\d{1,2}$", val):
                        val = val.replace(',', '.', 1)
                return Decimal(val.replace(',', ''))
            except (TypeError, InvalidOperation):
                return None

        result = np.empty(len(cleaned), dtype=object)

        if not suffix_values.empty:
            suffix_results = suffix_values.apply(process_suffix, axis=1)
            result[suffix_mask] = suffix_results.values

        std_mask = ~suffix_mask
        std_values = cleaned[std_mask].apply(process_standard)
        result[std_mask] = std_values.values

        signed_result = pd.Series(result, index=series.index) * signs

        return signed_result.apply(lambda x: Decimal(x) if pd.notna(x) else None)

    


if __name__=='__main__':
    file_paths = os.path.join(DATA_DIR,SUPPORTED_FILES[1])
    df=pd.read_excel(file_paths)
    parser=FormatParser()
    parse_df=df.copy()
    for col in df.columns:
        if 'date' in col.lower():
            parse_df[col]=df[col].apply(lambda x: parser.parse_date(x))
            print(parse_df)
        elif any(kw in col.lower() for kw in ['amount', 'pmt', 'disc', 'tolerance', 'value', 'balance']):
                        parse_df[col] = df[col].apply(lambda x: parser.parse_amount(x))

        
    parse_df=parser.drop_na_columns(parse_df,threshold=0.6)
    print('-->> Shape after dropped columns',parse_df.shape)
    # parse_df.to_csv('parse_data.txt')


        
    
        


    
    



