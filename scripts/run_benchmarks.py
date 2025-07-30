
import time 
import pandas as pd 
import os
from config.settings import DATA_DIR, SUPPORTED_FILES
from src.core.excel_processor import ExcelProcessor 
from src.core.type_detector import TypeDetector 
from src.core.data_storage import DataStorage 
from src.core.format_parser import FormatParser


def run_benchmark(file=None):
    """Run comprehensive benchmark tests for all components"""

    if file is None:
        # Fallback to supported file list
        from config.settings import DATA_DIR, SUPPORTED_FILES
        file_paths = [os.path.join(DATA_DIR, f) for f in SUPPORTED_FILES]
    else:
        file_paths = [file]

    for test_file in file_paths:
        print(f"Running benchmark with file: {test_file}")
        print("=" * 60)

        # 1. Excel Loading Benchmark
        print("\n1. EXCEL LOADING BENCHMARK")
        print("-" * 30)
        start_time = time.time()
        processor = ExcelProcessor(test_file)
        sheet_names = processor.excel.sheet_names
        print(f"Available sheets: {sheet_names}")

        # Use first sheet
        sheet_name = sheet_names[0]
        df = processor.extract_data(sheet_name)
        load_time = time.time() - start_time

        print(f"✓ Loaded sheet '{sheet_name}' in {load_time:.4f} seconds")
        print(f"✓ Dataset shape: {df.shape}")
        print(f"✓ Columns: {list(df.columns)}")

        # 2. Format Parsing Benchmark
        print("\n2. FORMAT PARSING BENCHMARK")
        print("-" * 30)
        parser = FormatParser()
        start_parse = time.time()

        # Parse amounts and dates
        parsed_df = df.copy()
        for col in df.columns:
            if 'date' in col.lower():
                parsed_df[col] = df[col].apply(lambda x: parser.parse_date(x))
            elif any(kw in col.lower() for kw in ['amount', 'pmt', 'disc', 'tolerance', 'value', 'balance']):
                parsed_df[col] = df[col].apply(lambda x: parser.parse_amount(x))

        # Vectorized parsing test
        amount_cols = [col for col in df.columns if any(kw in col.lower() for kw in ['amount', 'value'])]
        if amount_cols:
            start_vec = time.time()
            for col in amount_cols:
                parsed_df[f"{col}_vectorized"] = parser.parse_amount_vectorized(df[col])
            vec_time = time.time() - start_vec
            print(f"Vectorized parsing of {len(amount_cols)} columns took {vec_time:.4f} seconds")

        # Drop NA columns
        parsed_df = parser.drop_na_columns(parsed_df, threshold=0.6)
        parse_time = time.time() - start_parse

        print(f"✓ Parsed and cleaned dataset in {parse_time:.4f} seconds")
        print(f"✓ New shape after NA drop: {parsed_df.shape}")

        # 3. Type Detection Benchmark
        print("\n3. TYPE DETECTION BENCHMARK")
        print("-" * 30)
        start_detect = time.time()
        detector = TypeDetector()

        # Analyze each column and collect results
        column_analysis = {}
        for col in parsed_df.columns:
            data_type, confidence = detector.detect(parsed_df[col])
            column_analysis[col] = (data_type, confidence)
            print(f"  Column '{col}': {data_type} (confidence: {confidence:.2f})")

        detect_time = time.time() - start_detect
        print(f"✓ Detected {len(column_analysis)} column types in {detect_time:.4f} seconds")

        # 4. Data Storage Benchmark
        print("\n4. DATA STORAGE BENCHMARK")
        print("-" * 30)
        start_store = time.time()

        # Create DataStorage instance
        storage = DataStorage()

        # Prepare column types dictionary (extract just the type, not confidence)
        column_types = {col: analysis[0] for col, analysis in column_analysis.items()}

        # Store the data with cleaning
        storage.store_data("ledger", parsed_df, column_types, na_threshold=0.6, preserve_booleans=True)
        store_time = time.time() - start_store

        print(f"✓ Stored dataset in {store_time:.4f} seconds")

        # 5. Query Benchmark
        print("\n5. QUERY BENCHMARK")
        print("-" * 30)

        # Test different query scenarios
        test_queries = []

        # Find a suitable column for testing
        suitable_col = None
        suitable_value = None

        for col in parsed_df.columns:
            non_null_values = parsed_df[col].dropna()
            if len(non_null_values) > 0:
                suitable_col = col
                suitable_value = non_null_values.iloc[0]
                break
            
        if suitable_col and suitable_value is not None:
            # Test 1: Equality query
            start_query = time.time()
            filters = [(suitable_col, '=', suitable_value)]
            query_result = storage.query_by_criteria("ledger", filters)
            query_time = time.time() - start_query

            print(f"✓ Equality query on '{suitable_col}' took {query_time:.4f} seconds")
            print(f"  Returned {len(query_result)} rows")

            # Test 2: No filter query (get all data)
            start_query2 = time.time()
            all_data = storage.query_by_criteria("ledger", [])
            query_time2 = time.time() - start_query2

            print(f"✓ Full data query took {query_time2:.4f} seconds")
            print(f"  Returned {len(all_data)} rows")

            # Test 3: Get data from memory
            start_memory = time.time()
            memory_data = storage.get_data("ledger")
            memory_time = time.time() - start_memory

            print(f"✓ Memory retrieval took {memory_time:.4f} seconds")
            print(f"  Returned {len(memory_data)} rows")

        # 6. Boolean Column Preservation Test
        print("\n6. BOOLEAN COLUMN PRESERVATION TEST")
        print("-" * 40)

        # Check if any boolean columns were detected and preserved
        bool_columns = storage.boolean_columns.get("ledger", {})
        if bool_columns:
            print(f"✓ Found and preserved {len(bool_columns)} boolean columns:")
            for col, bool_type in bool_columns.items():
                print(f"  - {col}: {bool_type}")

            # Verify boolean columns in retrieved data
            retrieved_data = storage.get_data("ledger")
            actual_bool_cols = [col for col in retrieved_data.columns if retrieved_data[col].dtype == bool]
            print(f"✓ Actual boolean columns in data: {actual_bool_cols}")
        else:
            print("✓ No boolean columns detected in this dataset")

        # 7. Performance Summary
        print("\n7. PERFORMANCE SUMMARY")
        print("-" * 30)
        total_time = load_time + parse_time + detect_time + store_time
        print(f"Excel Loading:    {load_time:.4f}s ({load_time/total_time*100:.1f}%)")
        print(f"Format Parsing:   {parse_time:.4f}s ({parse_time/total_time*100:.1f}%)")
        print(f"Type Detection:   {detect_time:.4f}s ({detect_time/total_time*100:.1f}%)")
        print(f"Data Storage:     {store_time:.4f}s ({store_time/total_time*100:.1f}%)")
        print(f"Total Processing: {total_time:.4f}s")

        # 8. Memory Usage Info
        print("\n8. STORAGE INFO")
        print("-" * 20)
        storage.get_table_info("ledger")

        # Cleanup
        storage.close_connections()
        print(f"\n✓ Benchmark completed successfully!")
        return total_time

# def run_multiple_file_benchmark():
#     """Run benchmark on multiple files if available"""
#     print("\n" + "="*60)
#     print("MULTIPLE FILE BENCHMARK")
#     print("="*60)
    
#     available_files = []
#     for file in SUPPORTED_FILES:
#         file_path = os.path.join(DATA_DIR, file)
#         if os.path.exists(file_path):
#             available_files.append(file_path)
    
#     if len(available_files) <= 1:
#         print("Only one file available, skipping multiple file benchmark")
#         return
    
#     storage = DataStorage()
#     parser = FormatParser()
#     detector = TypeDetector()
#     total_start = time.time()
    
#     for i, file_path in enumerate(available_files):
#         print(f"\nProcessing file {i+1}/{len(available_files)}: {os.path.basename(file_path)}")
        
#         try:
#             # Load and process file
#             start = time.time()
#             processor = ExcelProcessor(file_path)
#             sheet_name = processor.excel.sheet_names[0]
#             df = processor.extract_data(sheet_name)
            
#             # Parse data
#             parsed_df = df.copy()
#             for col in df.columns:
#                 if 'date' in col.lower():
#                     parsed_df[col] = df[col].apply(lambda x: parser.parse_date(x))
#                 elif any(kw in col.lower() for kw in ['amount', 'pmt', 'disc', 'tolerance', 'value', 'balance']):
#                     parsed_df[col] = df[col].apply(lambda x: parser.parse_amount(x))
            
#             # Detect types
#             column_types = {}
#             for col in parsed_df.columns:
#                 data_type, confidence = detector.detect(parsed_df[col])
#                 column_types[col] = data_type
            
#             # Store data
#             table_name = f"dataset_{i+1}"
#             storage.store_data(table_name, parsed_df, column_types)
            
#             process_time = time.time() - start
#             print(f"  Processed in {process_time:.4f}s - Shape: {parsed_df.shape}")
            
#         except Exception as e:
#             print(f"  Error processing {file_path}: {e}")
    
#     total_time = time.time() - total_start
#     print(f"\nTotal time for {len(available_files)} files: {total_time:.4f}s")
#     print(f"Average time per file: {total_time/len(available_files):.4f}s")
    
#     # Show all loaded datasets
#     storage.get_table_info()
#     storage.close_connections()

if __name__ == "__main__":
    try:
        # Run single file benchmark
        run_benchmark()
        
        # Run multiple file benchmark if applicable
        # run_multiple_file_benchmark()
        
    except Exception as e:
        print(f"\nBenchmark failed with error: {e}")
        import traceback
        traceback.print_exc()