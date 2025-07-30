import pandas as pd
import sqlite3
import os
import numpy as np
import threading
import decimal


class DataStorage:

    # def __init__(self, db_file=None):
    #     """
    #     Initialize DataStorage
        
    #     Args:
    #         db_file: Path to SQLite database file. If None, uses in-memory database.
    #     """
    #     self.data = {}
    #     self.indexes = {}
    #     self.metadata = {}
    #     self.boolean_columns = {}  # Track boolean columns
        
    #     # Connect to SQLite database
    #     if db_file:
    #         self.db_file = db_file
    #         self.conn = sqlite3.connect(db_file,check_same_thread=False)
    #         print(f"Connected to SQLite database: {db_file}")
    #     else:
    #         self.db_file = ":memory:"
    #         self.conn = sqlite3.connect(':memory:')
    #         print("Using in-memory SQLite database")
    def __init__(self, db_file=None):
        self.db_file = db_file if db_file else ":memory:"
        self._thread_local = threading.local()
        self.data = {}
        self.indexes = {}
        self.metadata = {}
        self.boolean_columns = {}
        
    @property
    def conn(self):
        """Get or create a thread-local connection"""
        if not hasattr(self._thread_local, "conn"):
            self._thread_local.conn = sqlite3.connect(self.db_file, check_same_thread=False)
            # Enable foreign keys and other pragmas if needed
            self._thread_local.conn.execute("PRAGMA foreign_keys = ON")
        return self._thread_local.conn
    
    def close_connections(self):
        """Close all thread-local connections"""
        if hasattr(self._thread_local, "conn"):
            try:
                self._thread_local.conn.close()
            except Exception as e:
                print(f"Error closing connection: {e}")
            finally:
                del self._thread_local.conn

    def preserve_boolean_columns(self, df):
        """
        Identify and preserve boolean columns before they get converted
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            tuple: (boolean_columns_dict, preserved_df)
        """
        boolean_cols = {}
        preserved_df = df.copy()
        
        for col in df.columns:
            # Check if column contains boolean-like values
            unique_vals = df[col].dropna().unique()
            
            # Case 1: Already boolean dtype
            if df[col].dtype == bool:
                boolean_cols[col] = 'native_bool'
                print(f"  Found native boolean column: {col}")
            
            # Case 2: String boolean values
            elif len(unique_vals) <= 2 and all(str(val).lower() in ['true', 'false', '1', '0', 'yes', 'no'] for val in unique_vals if pd.notna(val)):
                boolean_cols[col] = 'string_bool'
                # Convert to proper boolean
                preserved_df[col] = df[col].map({
                    'True': True, 'true': True, 'TRUE': True,
                    'False': False, 'false': False, 'FALSE': False,
                    'Yes': True, 'yes': True, 'YES': True,
                    'No': False, 'no': False, 'NO': False,
                    '1': True, 1: True,
                    '0': False, 0: False
                })
                print(f"  Converted string boolean column: {col}")
            
            # Case 3: Numeric 0/1 that should be boolean
            elif set(unique_vals).issubset({0, 1, 0.0, 1.0}) and len(unique_vals) <= 2:
                boolean_cols[col] = 'numeric_bool'
                preserved_df[col] = df[col].astype(bool)
                print(f"  Converted numeric boolean column: {col}")
        
        return boolean_cols, preserved_df

    def restore_boolean_columns(self, df, table_name):
        """
        Restore boolean columns after loading from SQLite
        
        Args:
            df: DataFrame loaded from SQLite
            table_name: Name of the table
            
        Returns:
            DataFrame with restored boolean columns
        """
        if table_name not in self.boolean_columns:
            return df
        
        restored_df = df.copy()
        boolean_cols = self.boolean_columns[table_name]
        
        for col, bool_type in boolean_cols.items():
            if col in restored_df.columns:
                # Convert back to boolean
                restored_df[col] = restored_df[col].astype(bool)
                print(f"  Restored boolean column: {col}")
        
        return restored_df

    def analyze_na_columns(self, df, threshold=0.6):
        """
        Analyze columns with high NA percentages
        
        Args:
            df: DataFrame to analyze
            threshold: Threshold for NA percentage (0.5 = 50%)
            
        Returns:
            tuple: (columns_to_drop, na_summary)
        """
        na_counts = df.isnull().sum()
        na_percentages = na_counts / len(df)
        
        columns_to_drop = na_percentages[na_percentages > threshold].index.tolist()
        
        # Create summary
        na_summary = pd.DataFrame({
            'Column': df.columns,
            'NA_Count': na_counts,
            'NA_Percentage': (na_percentages * 100).round(2),
            'Will_Drop': [col in columns_to_drop for col in df.columns]
        }).sort_values('NA_Percentage', ascending=False)
        
        return columns_to_drop, na_summary

    def drop_high_na_columns(self, df, threshold=0.6, show_summary=True):
        """
        Drop columns with high NA percentages
        
        Args:
            df: DataFrame to clean
            threshold: Threshold for NA percentage (0.5 = 50%)
            show_summary: Whether to show summary of dropped columns
            
        Returns:
            Cleaned DataFrame
        """
        columns_to_drop, na_summary = self.analyze_na_columns(df, threshold)
        
        if show_summary:
            print(f"\nNA Analysis (threshold: {threshold*100}%):")
            print(na_summary.to_string(index=False))
        
        if columns_to_drop:
            print(f"\nDropping {len(columns_to_drop)} columns with >{threshold*100}% NA values:")
            for col in columns_to_drop:
                print(f"  - {col}")
            
            cleaned_df = df.drop(columns=columns_to_drop)
            print(f"Shape changed from {df.shape} to {cleaned_df.shape}")
            return cleaned_df
        else:
            print("No columns to drop based on NA threshold")
            return df

    def store_data(self, name, df, column_types, na_threshold=0.6, preserve_booleans=True):
        """
        Store DataFrame in both memory and SQLite database
        
        Args:
            name: Table name
            df: DataFrame to store
            column_types: Dictionary mapping column names to types
            na_threshold: Threshold for dropping NA columns (0.5 = 50%)
            preserve_booleans: Whether to preserve boolean columns
        """
        print(f"\nProcessing table '{name}'...")
        print(f"Original shape: {df.shape}")
        
        # Step 1: Preserve boolean columns
        if preserve_booleans:
            boolean_cols, processed_df = self.preserve_boolean_columns(df)
            self.boolean_columns[name] = boolean_cols
        else:
            processed_df = df.copy()
            self.boolean_columns[name] = {}
        
        # Step 2: Drop high NA columns
        if na_threshold > 0:
            cleaned_df = self.drop_high_na_columns(processed_df, threshold=na_threshold)
        else:
            cleaned_df = processed_df
        
        # Step 3: Update column types for remaining columns
        remaining_columns = cleaned_df.columns
        updated_column_types = {col: column_types.get(col, 'text') for col in remaining_columns}
        
        # Add boolean column types
        for col in self.boolean_columns[name]:
            if col in remaining_columns:
                updated_column_types[col] = 'boolean'
        
        # Step 4: Store in memory and SQLite
        self.data[name] = cleaned_df
        self.metadata[name] = updated_column_types
        
        # Store in SQLite (booleans will be stored as integers, but we'll restore them)
        for col in cleaned_df.columns:
            if cleaned_df[col].apply(lambda x: isinstance(x, decimal.Decimal)).any():
                print(f"Converting Decimal to float in column: {col}")
                cleaned_df[col] = cleaned_df[col].astype(float)
        cleaned_df.to_sql(name, self.conn, if_exists='replace', index=False)
        self.conn.commit()
        
        # Create indexes
        self.create_indexes(name, cleaned_df.columns)
        
        print(f"Final shape: {cleaned_df.shape}")
        print(f"Boolean columns preserved: {list(self.boolean_columns[name].keys())}")
        print(f"Stored table '{name}' successfully")

    def load_csv_data(self, csv_file, table_name=None, column_types=None, na_threshold=0.6, preserve_booleans=True):
        """
        Load CSV file into DataStorage with NA column dropping and boolean preservation
        
        Args:
            csv_file: Path to CSV file
            table_name: Name for the table
            column_types: Dictionary of column types
            na_threshold: Threshold for dropping NA columns (0.5 = 50%)
            preserve_booleans: Whether to preserve boolean columns
        """
        if not os.path.exists(csv_file):
            print(f"File not found: {csv_file}")
            return
        
        # Set table name
        if table_name is None:
            table_name = os.path.splitext(os.path.basename(csv_file))[0]
        
        try:
            # Load CSV
            df = pd.read_csv(csv_file)
            print(f"\nLoaded {csv_file}")
            
            # Auto-detect column types if not provided
            if column_types is None:
                column_types = self.detect_column_types(df)
            
            # Store the data with cleaning
            self.store_data(table_name, df, column_types, na_threshold, preserve_booleans)
            
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")

    def query_by_criteria(self, name, filters):
        """
        Query data with filters and restore boolean columns
        """
        if name not in self.data:
            print(f"Table '{name}' not found")
            return pd.DataFrame()
        
        if not filters:  # No filters - return all data
            query = f'SELECT * FROM "{name}"'
            params = []
        else:
            query = f'SELECT * FROM "{name}" WHERE '
            conditions = []
            params = []
            
            for col, op, val in filters:
                conditions.append(f'"{col}" {op} ?')
                params.append(val)
            
            query += ' AND '.join(conditions)
        
        try:
            result = pd.read_sql(query, self.conn, params=params)
            # Restore boolean columns
            result = self.restore_boolean_columns(result, name)
            print(f"Query returned {len(result)} rows")
            return result
        except Exception as e:
            print(f"Query error: {e}")
            return pd.DataFrame()

    def get_data(self, name):
        """
        Get data from memory with boolean columns properly restored
        
        Args:
            name: Table name
            
        Returns:
            DataFrame with restored boolean columns
        """
        if name not in self.data:
            print(f"Table '{name}' not found")
            return pd.DataFrame()
        
        df = self.data[name].copy()
        
        # Ensure boolean columns are properly typed
        if name in self.boolean_columns:
            for col, bool_type in self.boolean_columns[name].items():
                if col in df.columns:
                    df[col] = df[col].astype(bool)
        
        return df

    def create_indexes(self, name, columns):
        """Create indexes for better query performance"""
        self.indexes[name] = {}
        cursor = self.conn.cursor()
        
        for col in columns:
            try:
                col_type = self.metadata[name].get(col, 'text')
                if col_type in ['date', 'number', 'boolean']:
                    cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_{name}_{col} ON "{name}" ("{col}")')
                    self.indexes[name][col] = f'{col_type}_index'
            except sqlite3.OperationalError as e:
                print(f"Warning: Could not create index for {col}: {e}")
        
        self.conn.commit()

    def detect_column_types(self, df):
        """Auto-detect column types including boolean detection"""
        column_types = {}
        
        for col in df.columns:
            col_lower = col.lower()
            sample_values = df[col].dropna().head(10)
            unique_vals = df[col].dropna().unique()
            
            # Check for boolean columns first
            if (len(unique_vals) <= 2 and 
                all(str(val).lower() in ['true', 'false', '1', '0', 'yes', 'no'] 
                    for val in unique_vals if pd.notna(val))):
                column_types[col] = 'boolean'
            # Check for date columns
            elif 'date' in col_lower or 'quarter' in col_lower or 'time' in col_lower:
                column_types[col] = 'date'
            # Check for numeric columns
            elif any(kw in col_lower for kw in ['amount', 'pmt', 'disc', 'tolerance', 'value', 'balance', 'price', 'cost']):
                column_types[col] = 'number'
            # Try to detect numeric data
            elif self._is_numeric_data(sample_values):
                column_types[col] = 'number'
            # Try to detect date data
            elif self._is_date_data(sample_values):
                column_types[col] = 'date'
            else:
                column_types[col] = 'text'
        
        return column_types

    def _is_numeric_data(self, sample_values):
        """Check if sample values are numeric"""
        if len(sample_values) == 0:
            return False
            
        numeric_count = 0
        total_count = 0
        
        for value in sample_values:
            if pd.isna(value):
                continue
            total_count += 1
            try:
                float(str(value).replace(',', '').replace('$', '').replace('%', ''))
                numeric_count += 1
            except:
                continue
        
        return total_count > 0 and (numeric_count / total_count) > 0.7

    def _is_date_data(self, sample_values):
        """Check if sample values are dates"""
        if len(sample_values) == 0:
            return False
            
        for value in sample_values[:3]:
            if pd.isna(value):
                continue
            try:
                pd.to_datetime(str(value), infer_datetime_format=True)
                return True
            except:
                continue
        return False

    def get_table_info(self, name=None):
        """Get information about stored tables"""
        if name:
            if name in self.data:
                df = self.data[name]
                print(f"\nTable: {name}")
                print(f"Shape: {df.shape}")
                print(f"Columns: {list(df.columns)}")
                print(f"Column types: {self.metadata[name]}")
                print(f"Boolean columns: {list(self.boolean_columns.get(name, {}).keys())}")
                
                # Show data types in the actual DataFrame
                print("Actual DataFrame dtypes:")
                for col in df.columns:
                    print(f"  {col}: {df[col].dtype}")
            else:
                print(f"Table '{name}' not found")
        else:
            print("All stored tables:")
            for table_name in self.data.keys():
                df = self.data[table_name]
                bool_cols = list(self.boolean_columns.get(table_name, {}).keys())
                print(f"  {table_name}: {df.shape[0]} rows, {df.shape[1]} columns, {len(bool_cols)} boolean columns")

    def aggregate_data(self, name, group_by, measures):
        """Aggregate data by grouping with boolean column restoration"""
        if name not in self.data:
            print(f"Table '{name}' not found")
            return pd.DataFrame()
        
        query = f'SELECT "{group_by}", '
        agg_funcs = []
        
        for measure, func in measures:
            agg_funcs.append(f'{func}("{measure}") AS {measure}_{func}')
        
        query += ', '.join(agg_funcs) + f' FROM "{name}" GROUP BY "{group_by}"'
        
        try:
            result = pd.read_sql(query, self.conn)
            print(f"Aggregation returned {len(result)} groups")
            return result
        except Exception as e:
            print(f"Aggregation error: {e}")
            return pd.DataFrame()

    def load_all_csv_files(self, directory='.', pattern=None, na_threshold=0.6, preserve_booleans=True):
        """Load all CSV files from a directory with cleaning options"""
        print(f"Loading CSV files from: {os.path.abspath(directory)}")
        
        csv_files = []
        for file in os.listdir(directory):
            if file.endswith('.csv'):
                if pattern is None or pattern in file:
                    csv_files.append(file)
        
        print(f"Found CSV files: {csv_files}")
        
        if not csv_files:
            print("No CSV files found!")
            return 0
        
        loaded_count = 0
        for csv_file in csv_files:
            file_path = os.path.join(directory, csv_file)
            try:
                self.load_csv_data(file_path, na_threshold=na_threshold, preserve_booleans=preserve_booleans)
                loaded_count += 1
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
        
        print(f"Successfully loaded {loaded_count} files")
        return loaded_count

    def save_database(self, db_file):
        """Save current in-memory database to a file"""
        if self.db_file == ":memory:":
            file_conn = sqlite3.connect(db_file)
            
            for table_name in self.data.keys():
                df = self.data[table_name]
                df.to_sql(table_name, file_conn, if_exists='replace', index=False)
                
                # Recreate indexes
                cursor = file_conn.cursor()
                for col in df.columns:
                    try:
                        col_type = self.metadata[table_name].get(col, 'text')
                        if col_type in ['date', 'number', 'boolean']:
                            cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_{table_name}_{col} ON "{table_name}" ("{col}")')
                    except sqlite3.OperationalError:
                        pass
            
            file_conn.commit()
            file_conn.close()
            print(f"Database saved to {db_file}")
        else:
            print(f"Database is already persistent at {self.db_file}")

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("Database connection closed")


# Example usage
if __name__ == "__main__":
    # Initialize DataStorage
    storage = DataStorage(db_file='cleaned_data.db')
    
    # Load CSV files with 50% NA threshold and boolean preservation
    count = storage.load_all_csv_files(na_threshold=0.6, preserve_booleans=True)
    
    # Show what was loaded
    storage.get_table_info()
    
    # Example: Get data with proper boolean columns
    for table_name in storage.data.keys():
        print(f"\n--- Data from {table_name} ---")
        df = storage.get_data(table_name)
        print(f"Shape: {df.shape}")
        print("Sample data:")
        print(df.head(3))
        
        # Show boolean columns specifically
        bool_cols = [col for col in df.columns if df[col].dtype == bool]
        if bool_cols:
            print(f"Boolean columns: {bool_cols}")
            for col in bool_cols:
                print(f"  {col} values: {df[col].value_counts().to_dict()}")