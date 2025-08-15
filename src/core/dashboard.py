import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from decimal import Decimal
import io
from scripts.run_benchmarks import*
from src.core.type_detector import TypeDetector
from src.core.format_parser import FormatParser
from src.core.excel_processor import ExcelProcessor
from src.core.data_storage import DataStorage
from subsets.run_benchmarking import*
import time



# Page configuration
st.set_page_config(
    page_title="Financial Data Processor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'storage' not in st.session_state:
    st.session_state.storage = DataStorage()
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = {}

# Sidebar
st.sidebar.title("📊 Financial Data Processor")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.selectbox(
    "Select Page",
    ["🏠 Home", "📤 Upload & Process", "🔍 Type Detection", "🔧 Data Parsing", "💾 Data Storage", "📈 Data Analysis","🧪 Performance","💰 Subset Sum"]
)

# Main content
if page == "🏠 Home":
    st.title("Financial Data Processing System")
    st.markdown("""
    Welcome to the Financial Data Processing System! This application helps you:
    
    ### 🚀 Features
    - **Upload Files**: Support for Excel (.xlsx, .xls) and CSV files
    - **Type Detection**: Automatically detect column types (numbers, dates, strings)
    - **Data Parsing**: Parse financial amounts, dates, and other formats
    - **Data Storage**: Store and manage processed data
    - **Analysis**: Explore and visualize your data
    
    ### 📋 How to Use
    1. **Upload & Process**: Start by uploading your financial data files
    2. **Type Detection**: Review automatically detected column types
    3. **Data Parsing**: Apply parsing rules to format your data correctly
    4. **Data Storage**: Store processed data for analysis
    5. **Data Analysis**: Explore patterns and create visualizations
    
    ### 📊 Supported Formats
    - **Excel Files**: .xlsx, .xls (multiple sheets supported)
    - **CSV Files**: Comma-separated values
    - **Financial Data**: Amounts with currency symbols, percentages, K/M/B suffixes
    - **Dates**: Various date formats including Excel serial dates
    """)
    
    # Quick stats
    if st.session_state.processed_files:
        st.markdown("### 📈 Current Session Stats")
        col1, col2, col3 = st.columns(3)
        
        total_files = len(st.session_state.processed_files)
        total_rows = sum(df.shape[0] for df in st.session_state.processed_files.values())
        total_cols = sum(df.shape[1] for df in st.session_state.processed_files.values())
        
        col1.metric("Files Processed", total_files)
        col2.metric("Total Rows", f"{total_rows:,}")
        col3.metric("Total Columns", total_cols)

elif page == "📤 Upload & Process":
    st.title("Upload & Process Files")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Choose your files",
        type=['xlsx', 'xls', 'csv'],
        accept_multiple_files=True,
        help="Upload Excel or CSV files containing financial data"
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.markdown(f"### 📄 Processing: {uploaded_file.name}")
            
            try:
                # Determine file type and process
                file_extension = uploaded_file.name.split('.')[-1].lower()
                
                if file_extension in ['xlsx', 'xls']:
                    # Excel file processing
                    excel_processor = ExcelProcessor(uploaded_file)
                    sheet_info = excel_processor.get_sheet_info()
                    
                    st.write("**Available Sheets:**")
                    
                    # Add "Process All Sheets" button
                    if st.button(f"🚀 Process All Sheets from {uploaded_file.name}", key=f"process_all_{uploaded_file.name}"):
                        for info in sheet_info:
                            df = excel_processor.extract_data(info['sheet_name'])
                            table_name = f"{uploaded_file.name}_{info['sheet_name']}"
                            st.session_state.processed_files[table_name] = df
                            st.success(f"✅ Processed sheet '{info['sheet_name']}' with {df.shape[0]} rows and {df.shape[1]} columns")
                    
                    for info in sheet_info:
                        with st.expander(f"Sheet: {info['sheet_name']} ({info['num_rows']} rows, {info['num_columns']} cols)"):
                            st.write(f"**Columns:** {', '.join(info['column_names'])}")
                            
                            # Show preview
                            df_preview = excel_processor.extract_data(info['sheet_name'])
                            st.dataframe(df_preview.head(3))
                            
                            if st.button(f"Process Sheet: {info['sheet_name']}", key=f"process_{uploaded_file.name}_{info['sheet_name']}"):
                                df = excel_processor.extract_data(info['sheet_name'])
                                table_name = f"{uploaded_file.name}_{info['sheet_name']}"
                                st.session_state.processed_files[table_name] = df
                                st.success(f"✅ Processed sheet '{info['sheet_name']}' with {df.shape[0]} rows and {df.shape[1]} columns")
                                st.rerun()  # Refresh the page to update session state
                
                elif file_extension == 'csv':
                    # CSV file processing - Auto process CSV files
                    df = pd.read_csv(uploaded_file)
                    table_name = uploaded_file.name
                    st.session_state.processed_files[table_name] = df
                    
                    st.success(f"✅ Auto-processed CSV file with {df.shape[0]} rows and {df.shape[1]} columns")
                    st.dataframe(df.head())
                    
            except Exception as e:
                st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
    
    # Show currently processed files
    if st.session_state.processed_files:
        st.markdown("---")
        st.markdown("### 📊 Currently Processed Files")
        
        for name, df in st.session_state.processed_files.items():
            with st.expander(f"📄 {name} ({df.shape[0]} rows, {df.shape[1]} columns)"):
                st.dataframe(df.head())
                
                # Add remove button
                if st.button(f"🗑️ Remove {name}", key=f"remove_{name}"):
                    del st.session_state.processed_files[name]
                    st.success(f"Removed {name}")
                    st.rerun()

elif page == "🔍 Type Detection":
    st.title("Column Type Detection")
    
    if not st.session_state.processed_files:
        st.warning("⚠️ No files processed yet. Please upload files first.")
    else:
        # Select dataset
        selected_file = st.selectbox("Select Dataset", list(st.session_state.processed_files.keys()))
        
        if selected_file:
            df = st.session_state.processed_files[selected_file]
            st.write(f"**Dataset:** {selected_file} ({df.shape[0]} rows, {df.shape[1]} columns)")
            
            # Run type detection
            if st.button("🔍 Detect Column Types"):
                detector = TypeDetector()
                
                with st.spinner("Detecting column types..."):
                    detection_results = {}
                    
                    for column in df.columns:
                        data_type, confidence = detector.detect(df[column])
                        detection_results[column] = {
                            'type': data_type,
                            'confidence': confidence,
                            'sample_values': df[column].dropna().head(3).tolist()
                        }
                
                # Display results
                st.markdown("### 📊 Detection Results")
                
                results_df = pd.DataFrame([
                    {
                        'Column': col,
                        'Detected Type': info['type'],
                        'Confidence': f"{info['confidence']:.2%}",
                        'Sample Values': ', '.join(str(v) for v in info['sample_values'])
                    }
                    for col, info in detection_results.items()
                ])
                
                st.dataframe(results_df, use_container_width=True)
                
                # Confidence visualization
                st.markdown("### 📈 Confidence Levels")
                fig = px.bar(
                    x=list(detection_results.keys()),
                    y=[info['confidence'] for info in detection_results.values()],
                    color=[info['type'] for info in detection_results.values()],
                    title="Type Detection Confidence by Column",
                    labels={'x': 'Columns', 'y': 'Confidence'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Store results
                st.session_state[f"types_{selected_file}"] = detection_results

elif page == "🔧 Data Parsing":
    st.title("Data Parsing & Formatting")
    
    if not st.session_state.processed_files:
        st.warning("⚠️ No files processed yet. Please upload files first.")
    else:
        selected_file = st.selectbox("Select Dataset", list(st.session_state.processed_files.keys()))
        
        if selected_file:
            df = st.session_state.processed_files[selected_file]
            parser = FormatParser()
            
            st.write(f"**Dataset:** {selected_file}")
            
            # Column selection for parsing
            st.markdown("### 🎯 Select Columns to Parse")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📊 Amount Columns**")
                amount_columns = st.multiselect(
                    "Select columns containing amounts/numbers",
                    options=df.columns.tolist(),
                    default=[col for col in df.columns if any(keyword in col.lower() 
                            for keyword in ['amount', 'value', 'price', 'cost', 'balance'])]
                )
            
            with col2:
                st.markdown("**📅 Date Columns**")
                date_columns = st.multiselect(
                    "Select columns containing dates",
                    options=df.columns.tolist(),
                    default=[col for col in df.columns if 'date' in col.lower()]
                )
            
            if st.button("🔧 Apply Parsing"):
                parsed_df = df.copy()
                parsing_results = {}
                
                with st.spinner("Parsing data..."):
                    # Parse amount columns
                    for col in amount_columns:
                        try:
                            original_sample = df[col].dropna().head(3).tolist()
                            parsed_df[col] = df[col].apply(parser.parse_amount)
                            parsed_sample = parsed_df[col].dropna().head(3).tolist()
                            
                            parsing_results[col] = {
                                'type': 'amount',
                                'original': original_sample,
                                'parsed': parsed_sample,
                                'success_rate': parsed_df[col].notna().mean()
                            }
                        except Exception as e:
                            st.error(f"Error parsing amount column '{col}': {str(e)}")
                    
                    # Parse date columns
                    for col in date_columns:
                        try:
                            original_sample = df[col].dropna().head(3).tolist()
                            parsed_df[col] = df[col].apply(parser.parse_date)
                            parsed_sample = parsed_df[col].dropna().head(3).tolist()
                            
                            parsing_results[col] = {
                                'type': 'date',
                                'original': original_sample,
                                'parsed': parsed_sample,
                                'success_rate': parsed_df[col].notna().mean()
                            }
                        except Exception as e:
                            st.error(f"Error parsing date column '{col}': {str(e)}")
                
                # Display parsing results
                st.markdown("### ✅ Parsing Results")
                
                for col, result in parsing_results.items():
                    with st.expander(f"{result['type'].title()} Column: {col} (Success Rate: {result['success_rate']:.1%})"):
                        col_orig, col_parsed = st.columns(2)
                        
                        with col_orig:
                            st.write("**Original Values:**")
                            for val in result['original']:
                                st.code(str(val))
                        
                        with col_parsed:
                            st.write("**Parsed Values:**")
                            for val in result['parsed']:
                                st.code(str(val))
                
                # Update stored data
                st.session_state.processed_files[f"{selected_file}_parsed"] = parsed_df
                st.success("✅ Parsing completed! Parsed data saved as new dataset.")
                
                # Show comparison
                st.markdown("### 📊 Before vs After")
                tab1, tab2 = st.tabs(["Original Data", "Parsed Data"])
                
                with tab1:
                    st.dataframe(df.head(), use_container_width=True)
                
                with tab2:
                    st.dataframe(parsed_df.head(), use_container_width=True)

elif page == "💾 Data Storage":
    st.title("Data Storage & Management")
    
    if not st.session_state.processed_files:
        st.warning("⚠️ No files processed yet. Please upload files first.")
    else:
        # Storage options
        st.markdown("### 💾 Storage Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Load to Storage System"):
                storage = st.session_state.storage
                
                for name, df in st.session_state.processed_files.items():
                    # Detect column types for storage
                    detector = TypeDetector()
                    column_types = {}
                    
                    for col in df.columns:
                        data_type, _ = detector.detect(df[col])
                        column_types[col] = data_type
                    
                    storage.store_data(name, df, column_types)
                
                st.success("✅ All datasets loaded to storage system!")
        
        with col2:
            if st.button("💾 Export All Data"):
                # Create a download for all processed files
                buffer = io.BytesIO()
                
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    for name, df in st.session_state.processed_files.items():
                        # Clean sheet name for Excel
                        sheet_name = name.replace('.xlsx', '').replace('.csv', '')[:31]
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                st.download_button(
                    label="📥 Download All Data as Excel",
                    data=buffer.getvalue(),
                    file_name=f"processed_financial_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        # Show current storage status
        st.markdown("### 📊 Current Storage Status")
        
        if st.session_state.processed_files:
            storage_df = pd.DataFrame([
                {
                    'Dataset': name,
                    'Rows': df.shape[0],
                    'Columns': df.shape[1],
                    'Memory Usage': f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB",
                    'Data Types': len(df.dtypes.unique())
                }
                for name, df in st.session_state.processed_files.items()
            ])
            
            st.dataframe(storage_df, use_container_width=True)
        
        # Data preview
        st.markdown("### 👁️ Data Preview")
        selected_dataset = st.selectbox("Select Dataset to Preview", list(st.session_state.processed_files.keys()))
        
        if selected_dataset:
            df = st.session_state.processed_files[selected_dataset]
            
            tab1, tab2, tab3 = st.tabs(["📊 Data", "📈 Info", "🔍 Statistics"])
            
            with tab1:
                st.dataframe(df, use_container_width=True)
            
            with tab2:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Dataset Information:**")
                    st.write(f"Shape: {df.shape}")
                    st.write(f"Columns: {df.shape[1]}")
                    st.write(f"Rows: {df.shape[0]}")
                
                with col2:
                    st.write("**Data Types:**")
                    for dtype, count in df.dtypes.value_counts().items():
                        st.write(f"{dtype}: {count} columns")
            
            with tab3:
                st.write("**Numerical Statistics:**")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    st.dataframe(df[numeric_cols].describe(), use_container_width=True)
                else:
                    st.write("No numerical columns found.")

elif page == "📈 Data Analysis":
    st.title("Data Analysis & Visualization")
    
    if not st.session_state.processed_files:
        st.warning("⚠️ No files processed yet. Please upload files first.")
    else:
        selected_dataset = st.selectbox("Select Dataset for Analysis", list(st.session_state.processed_files.keys()))
        
        if selected_dataset:
            df = st.session_state.processed_files[selected_dataset]
            
            st.markdown(f"### 📊 Analyzing: {selected_dataset}")
            
            # Basic statistics
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric("Total Rows", f"{df.shape[0]:,}")
            col2.metric("Total Columns", df.shape[1])
            col3.metric("Numerical Columns", len(df.select_dtypes(include=[np.number]).columns))
            col4.metric("Missing Values", f"{df.isnull().sum().sum():,}")
            
            # Column analysis
            st.markdown("### 📊 Column Analysis")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            text_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            if numeric_cols:
                st.markdown("#### 📈 Numerical Columns")
                selected_numeric = st.selectbox("Select column for analysis", numeric_cols)
                
                if selected_numeric:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Histogram
                        fig = px.histogram(df, x=selected_numeric, title=f"Distribution of {selected_numeric}")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Box plot
                        fig = px.box(df, y=selected_numeric, title=f"Box Plot of {selected_numeric}")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistics
                    st.write(f"**Statistics for {selected_numeric}:**")
                    stats = df[selected_numeric].describe()
                    st.dataframe(stats.to_frame().T, use_container_width=True)
            
            if len(numeric_cols) >= 2:
                st.markdown("#### 🔗 Correlation Analysis")
                
                corr_matrix = df[numeric_cols].corr()
                fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Correlation Matrix")
                st.plotly_chart(fig, use_container_width=True)
            
            # Missing values analysis
            st.markdown("### 🕳️ Missing Values Analysis")
            
            missing_data = df.isnull().sum()
            missing_percent = (missing_data / len(df)) * 100
            
            missing_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing Count': missing_data.values,
                'Missing Percentage': missing_percent.values
            }).sort_values('Missing Count', ascending=False)
            
            missing_df = missing_df[missing_df['Missing Count'] > 0]
            
            if not missing_df.empty:
                fig = px.bar(missing_df, x='Column', y='Missing Percentage', 
                           title="Missing Values by Column")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(missing_df, use_container_width=True)
            else:
                st.success("✅ No missing values found!")
elif page == "🧪 Performance":
    st.title("🧪 Performance Testing")

    st.markdown("""
    This section runs internal performance benchmarks for Excel loading, parsing, detection,
    storage, and querying logic.

    ⚠️ Running Performance Test may take a few seconds.
    """)

    if st.button("▶️ Run Performance Test"):
        with st.spinner("Performance Testing"):
            try:
                # Create containers for different sections
                summary_container = st.container()
                details_container = st.container()
                
                # Redirect stdout to capture the benchmark output
                from io import StringIO
                import sys
                
                old_stdout = sys.stdout
                sys.stdout = mystdout = StringIO()
                
                # Run the benchmark
                # run_benchmark()
                # Use one of the uploaded/processed files for benchmark
                if st.session_state.processed_files:
                    selected_file = st.selectbox("Select file for benchmark", list(st.session_state.processed_files.keys()))
                    selected_df = st.session_state.processed_files[selected_file]
                
                    # Convert DataFrame to Excel-like buffer
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        selected_df.to_excel(writer, index=False, sheet_name="Sheet1")
                    buffer.seek(0)
                
                    run_benchmark(file=buffer)
                else:
                    st.warning("No processed file found to run benchmark.")
                
                
                # Restore stdout
                sys.stdout = old_stdout
                
                # Get the output
                output = mystdout.getvalue()
                
                if not output:
                    st.error("Benchmark completed but no output was captured")
                    
                
                st.success("✅ Benchmark completed.")
                
                # Parse the output for visualization
                try:
                    metrics = {
                        "Excel Loading": None,
                        "Format Parsing": None,
                        "Type Detection": None,
                        "Data Storage": None,
                        "Total Processing": None
                    }
                    
                    # Extract the performance metrics from the output
                    for line in output.split('\n'):
                        if "Excel Loading:" in line:
                            parts = line.split(':')
                            if len(parts) > 1:
                                metrics["Excel Loading"] = float(parts[1].split('s')[0].strip())
                        elif "Format Parsing:" in line:
                            parts = line.split(':')
                            if len(parts) > 1:
                                metrics["Format Parsing"] = float(parts[1].split('s')[0].strip())
                        elif "Type Detection:" in line:
                            parts = line.split(':')
                            if len(parts) > 1:
                                metrics["Type Detection"] = float(parts[1].split('s')[0].strip())
                        elif "Data Storage:" in line:
                            parts = line.split(':')
                            if len(parts) > 1:
                                metrics["Data Storage"] = float(parts[1].split('s')[0].strip())
                        elif "Total Processing:" in line:
                            parts = line.split(':')
                            if len(parts) > 1:
                                metrics["Total Processing"] = float(parts[1].split('s')[0].strip())
                    
                    # Show summary only if we got all metrics
                    if all(v is not None for v in metrics.values()):
                        with summary_container:
                            st.markdown("### 📊 Performance Summary")
                            
                            # Create columns for metrics
                            col1, col2, col3, col4, col5 = st.columns(5)
                            
                            with col1:
                                st.metric("Excel Loading", f"{metrics['Excel Loading']:.2f}s", 
                                         f"{metrics['Excel Loading']/metrics['Total Processing']*100:.1f}%")
                            
                            with col2:
                                st.metric("Format Parsing", f"{metrics['Format Parsing']:.2f}s", 
                                         f"{metrics['Format Parsing']/metrics['Total Processing']*100:.1f}%")
                            
                            with col3:
                                st.metric("Type Detection", f"{metrics['Type Detection']:.2f}s", 
                                         f"{metrics['Type Detection']/metrics['Total Processing']*100:.1f}%")
                            
                            with col4:
                                st.metric("Data Storage", f"{metrics['Data Storage']:.2f}s", 
                                         f"{metrics['Data Storage']/metrics['Total Processing']*100:.1f}%")
                            
                            with col5:
                                st.metric("Total Time", f"{metrics['Total Processing']:.2f}s", "")
                            
                            # Create a pie chart of time distribution
                            st.markdown("### ⏱️ Time Distribution")
                            fig = px.pie(
                                names=list(metrics.keys())[:-1],  # Exclude total time
                                values=list(metrics.values())[:-1],
                                title="Time Spent on Each Processing Stage"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Extract and display storage information
                        storage_info = {}
                        storage_section = False
                        for line in output.split('\n'):
                            if "8. STORAGE INFO" in line:
                                storage_section = True
                            elif storage_section and "Table:" in line:
                                parts = line.split(':')
                                if len(parts) > 1:
                                    storage_info["Table"] = parts[1].strip()
                            elif storage_section and "Shape:" in line:
                                parts = line.split(':')
                                if len(parts) > 1:
                                    storage_info["Shape"] = parts[1].strip()
                            elif storage_section and "Boolean columns:" in line:
                                parts = line.split(':')
                                if len(parts) > 1:
                                    storage_info["Boolean Columns"] = parts[1].strip()
                        
                        if storage_info:
                            with summary_container:
                                st.markdown("### 💾 Storage Information")
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Table Name", storage_info.get("Table", "N/A"))
                                
                                with col2:
                                    st.metric("Data Shape", storage_info.get("Shape", "N/A"))
                                
                                with col3:
                                    bool_cols = storage_info.get("Boolean Columns", "")
                                    bool_count = len(bool_cols.split(',')) if bool_cols else 0
                                    st.metric("Boolean Columns", str(bool_count))
                
                except Exception as parse_error:
                    st.warning(f"Couldn't parse benchmark output for visualization: {str(parse_error)}")
                
                # Always show the detailed output in an expandable section
                with details_container:
                    with st.expander("📄 View Detailed Benchmark Output"):
                        st.text(output)
                
            except Exception as e:
                st.error(f"❌ Error during single file benchmark: {str(e)}")
                if 'output' in locals():
                    with st.expander("📄 View Partial Benchmark Output"):
                        st.text(output)


elif page == "💰 Subset Sum":
    st.title("💰 Subset Sum Benchmarking")
    
    st.markdown("""
    Compare different subset sum algorithms for matching transactions from one dataset 
    to target amounts in another dataset.
    """)

    # Ensure processed_files exists in session state
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = {}

    if not st.session_state.processed_files:
        st.warning("⚠️ No files processed yet. Please upload files first.")
    else:
        # Select datasets
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Target Dataset")
            target_file = st.selectbox(
                "Select dataset with target amounts",
                list(st.session_state.processed_files.keys()),
                key="target_dataset"
            )
            target_df = st.session_state.processed_files[target_file].copy()
            
        with col2:
            st.markdown("### 💰 Transaction Dataset")
            transaction_file = st.selectbox(
                "Select dataset with transactions",
                list(st.session_state.processed_files.keys()),
                key="transaction_dataset"
            )
            transaction_df = st.session_state.processed_files[transaction_file].copy()

        # -------------------------------
        # Clean column names
        # -------------------------------
        transaction_df.columns = transaction_df.columns.map(lambda x: str(x).strip())
        target_df.columns = target_df.columns.map(lambda x: str(x).strip())
        transaction_df.columns = transaction_df.columns.str.replace('.', '_', regex=False)
        target_df.columns = target_df.columns.str.replace('.', '_', regex=False)

        # -------------------------------
        # Amount Column Selection (ONLY)
        # -------------------------------
        st.markdown("### 🔍 Amount Column Selection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Transaction Amount Column**")
            trans_amount_col = st.selectbox(
                "Select the amount column from transactions",
                transaction_df.columns,
                index=next((i for i, col in enumerate(transaction_df.columns) 
                          if 'amount' in col.lower()), 0),
                key="trans_amount_col"
            )
            
        with col2:
            st.markdown("**Target Amount Column**")
            target_amount_col = st.selectbox(
                "Select the amount column from targets",
                target_df.columns,
                index=next((i for i, col in enumerate(target_df.columns) 
                          if 'amount' in col.lower()), 0),
                key="target_amount_col"
            )

        # -------------------------------
        # Data Processing and ID Assignment
        # -------------------------------
        st.markdown("### 🔧 Data Processing")
        
        if st.button("📋 Process Data & Preview"):
            try:
                # Extract only the selected amount columns
                transaction_amounts = transaction_df[trans_amount_col].copy()
                target_amounts = target_df[target_amount_col].copy()
                
                # Parse and clean amount data
                parser = FormatParser()
                
                # Clean transaction amounts
                transaction_amounts = transaction_amounts.apply(
                    lambda x: parser.parse_amount(x) if pd.notna(x) else None
                )
                transaction_amounts = pd.to_numeric(transaction_amounts, errors='coerce')
                
                # Clean target amounts  
                target_amounts = target_amounts.apply(
                    lambda x: parser.parse_amount(x) if pd.notna(x) else None
                )
                target_amounts = pd.to_numeric(target_amounts, errors='coerce')
                
                # Remove null values and get valid indices
                valid_trans_mask = transaction_amounts.notna()
                valid_target_mask = target_amounts.notna()
                
                clean_transaction_amounts = transaction_amounts[valid_trans_mask]
                clean_target_amounts = target_amounts[valid_target_mask]
                
                # Assign unique IDs to valid amounts
                transaction_ids = [f"TX_{i+1:04d}" for i in range(len(clean_transaction_amounts))]
                target_ids = [f"TG_{i+1:04d}" for i in range(len(clean_target_amounts))]
                
                # Create processed dataframes with unique IDs
                processed_transactions = pd.DataFrame({
                    'Transaction_ID': transaction_ids,
                    'amount': clean_transaction_amounts.values,
                    'original_index': clean_transaction_amounts.index
                })
                
                processed_targets = pd.DataFrame({
                    'Target_ID': target_ids, 
                    'amount': clean_target_amounts.values,
                    'original_index': clean_target_amounts.index
                })
                
                # Store in session state for benchmarking
                st.session_state.processed_transactions = processed_transactions
                st.session_state.processed_targets = processed_targets
                
                # Display preview
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Processed Transaction Data:**")
                    st.dataframe(processed_transactions.head(10), use_container_width=True)
                    st.info(f"✅ {len(processed_transactions)} valid transactions processed")
                    
                with col2:
                    st.markdown("**Processed Target Data:**")
                    st.dataframe(processed_targets.head(10), use_container_width=True) 
                    st.info(f"✅ {len(processed_targets)} valid targets processed")
                
                # Show data statistics
                st.markdown("### 📊 Data Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Transaction Range", 
                             f"{clean_transaction_amounts.min():.2f} - {clean_transaction_amounts.max():.2f}")
                with col2:
                    st.metric("Target Range",
                             f"{clean_target_amounts.min():.2f} - {clean_target_amounts.max():.2f}")
                with col3:
                    st.metric("Avg Transaction", f"{clean_transaction_amounts.mean():.2f}")
                with col4:
                    st.metric("Avg Target", f"{clean_target_amounts.mean():.2f}")
                    
            except Exception as e:
                st.error(f"Error processing data: {str(e)}")
                st.code(traceback.format_exc())
        
        # -------------------------------
        # Benchmark Settings (only show if data is processed)
        # -------------------------------
        if 'processed_transactions' in st.session_state and 'processed_targets' in st.session_state:
            st.markdown("### ⚙️ Benchmark Settings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Target Data Selection**")
                max_targets = len(st.session_state.processed_targets)
                target_sample_size = st.number_input(
                    "Number of targets to test (0 for all)",
                    min_value=0,
                    max_value=min(50, max_targets),  # Limit for performance
                    value=min(10, max_targets),
                    help="Number of targets to use for benchmarking (smaller = faster)",
                    key="target_sample_size"
                )
            
            with col2:
                st.markdown("**Transaction Data Selection**")
                max_transactions = len(st.session_state.processed_transactions)
                transaction_sample_size = st.number_input(
                    "Number of transactions to use (0 for all)",
                    min_value=0,
                    max_value=min(1000, max_transactions),  # Limit for performance
                    value=min(100, max_transactions),
                    help="Number of transactions to use for matching (smaller = faster)",
                    key="transaction_sample_size"
                )
            
            if st.button("🏃‍♂️ Run Benchmark"):
                try:
                    # Get processed data
                    trans_data = st.session_state.processed_transactions.copy()
                    target_data = st.session_state.processed_targets.copy()
                    
                    # Sample transactions if requested
                    if transaction_sample_size > 0 and transaction_sample_size < len(trans_data):
                        trans_data = trans_data.sample(transaction_sample_size).copy().reset_index(drop=True)
                        st.info(f"🔄 Using {len(trans_data)} sampled transactions out of {len(st.session_state.processed_transactions)} total")
                    
                    # Sample targets if requested
                    if target_sample_size > 0 and target_sample_size < len(target_data):
                        target_data = target_data.sample(target_sample_size).copy().reset_index(drop=True)
                        st.info(f"🎯 Testing {len(target_data)} sampled targets out of {len(st.session_state.processed_targets)} total")
                    
                    # Display final data selection summary
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Selected Transactions", len(trans_data))
                        st.write(f"Amount range: {trans_data['amount'].min():.2f} - {trans_data['amount'].max():.2f}")
                    with col2:
                        st.metric("Selected Targets", len(target_data))
                        st.write(f"Amount range: {target_data['amount'].min():.2f} - {target_data['amount'].max():.2f}")
                    
                    # Rename columns to match backend expectations
                    trans_data = trans_data.rename(columns={'Transaction_ID': 'Transaction ID'})
                    target_data = target_data.rename(columns={'Target_ID': 'Target ID'})
                    
                    with st.spinner("Running benchmark..."):
                        st.info(f"🔄 Running benchmark with {len(trans_data)} transactions and {len(target_data)} targets...")
                        
                        # Run benchmark
                        benchmark_results = benchmark_methods(trans_data, target_data)
                        
                        # Display results
                        st.markdown("### 📊 Benchmark Results")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Transactions Used", len(trans_data))
                        with col2:
                            st.metric("Targets Processed", len(target_data))
                        with col3:
                            if len(trans_data) < len(st.session_state.processed_transactions):
                                st.metric("Transaction Sample %", f"{len(trans_data)/len(st.session_state.processed_transactions)*100:.1f}%")
                            else:
                                st.metric("Transaction Sample %", "100%")
                        with col4:
                            if len(target_data) < len(st.session_state.processed_targets):
                                st.metric("Target Sample %", f"{len(target_data)/len(st.session_state.processed_targets)*100:.1f}%")
                            else:
                                st.metric("Target Sample %", "100%")
                        
                        st.dataframe(benchmark_results, use_container_width=True)
                        
                        # -------------------------------
                        # Visualizations
                        # -------------------------------
                        if not benchmark_results.empty:
                            st.markdown("### 📈 Performance Comparison")
                            
                            # Filter valid results for plotting
                            valid_results = benchmark_results[
                                (benchmark_results['Time (ms)'] != 'Error') & 
                                (benchmark_results['Time (ms)'] != float('inf'))
                            ].copy()
                            
                            if not valid_results.empty:
                                valid_results['Time (ms)'] = pd.to_numeric(valid_results['Time (ms)'], errors='coerce')
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    fig_time = px.bar(
                                        valid_results,
                                        x='Method',
                                        y='Time (ms)',
                                        color='Method',
                                        title="Execution Time"
                                    )
                                    st.plotly_chart(fig_time, use_container_width=True)
                                
                                with col2:
                                    fig_mem = px.bar(
                                        valid_results,
                                        x='Method',
                                        y='Memory (KB)',
                                        color='Method',
                                        title="Memory Usage"
                                    )
                                    st.plotly_chart(fig_mem, use_container_width=True)
                        
                        # -------------------------------
                        # Show detailed matches with unique IDs
                        # -------------------------------
                        st.markdown("### 🔍 Match Details with Unique IDs")
                        
                        detailed_matches = []
                        
                        # Test first few targets for detailed matching
                        for _, target_row in target_data.head(5).iterrows():
                            try:
                                target_amount = target_row['amount']
                                target_id = target_row['Target ID']
                                
                                # Create transaction tuples for subset_sum_exists
                                transaction_tuples = list(zip(
                                    trans_data['amount'].tolist(),
                                    trans_data['Transaction ID'].tolist()
                                ))
                                
                                # Try to find subset sum match
                                result = subset_sum_exists(
                                    transaction_tuples,
                                    (float(target_amount), target_id),
                                    precision=100
                                )
                                
                                if result and len(result) > 0 and result[0]:  # Match found
                                    matching_transaction_ids = result[2] if len(result) > 2 else []
                                    
                                    # Get the matching transactions with their unique IDs
                                    matching_transactions = trans_data[
                                        trans_data['Transaction ID'].isin(matching_transaction_ids)
                                    ]
                                    
                                    total_sum = matching_transactions['amount'].sum()
                                    difference = abs(total_sum - target_amount)
                                    
                                    detailed_matches.append({
                                        'Target_ID': target_id,
                                        'Target_Amount': target_amount,
                                        'Matching_Transaction_IDs': ', '.join(matching_transaction_ids),
                                        'Matching_Amounts': ', '.join([f"{amt:.2f}" for amt in matching_transactions['amount']]),
                                        'Total_Sum': total_sum,
                                        'Difference': difference,
                                        'Match_Count': len(matching_transactions)
                                    })
                                    
                            except Exception as e:
                                st.warning(f"Error processing target {target_row['Target ID']}: {str(e)}")
                                continue
                        
                        if detailed_matches:
                            matches_df = pd.DataFrame(detailed_matches)
                            st.dataframe(matches_df, use_container_width=True)
                            
                            # Show expandable details for each match
                            st.markdown("#### 📝 Detailed Match Breakdown")
                            for idx, match in enumerate(detailed_matches):
                                with st.expander(f"Match {idx+1}: {match['Target_ID']} → {match['Match_Count']} transactions"):
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.write(f"**Target:** {match['Target_ID']}")
                                        st.write(f"**Target Amount:** {match['Target_Amount']:.2f}")
                                        st.write(f"**Difference:** {match['Difference']:.4f}")
                                    with col2:
                                        st.write(f"**Transaction IDs:** {match['Matching_Transaction_IDs']}")
                                        st.write(f"**Transaction Amounts:** {match['Matching_Amounts']}")
                                        st.write(f"**Total Sum:** {match['Total_Sum']:.2f}")
                        else:
                            st.info("🔍 No exact subset matches found in the sample data")
                            
                            # Show some near misses or direct matches
                            st.markdown("#### 🎯 Checking for Direct Amount Matches")
                            direct_matches = []
                            
                            for _, target_row in target_data.head(5).iterrows():
                                target_amount = target_row['amount']
                                target_id = target_row['Target ID']
                                
                                # Find direct matches (within small tolerance)
                                tolerance = 0.01
                                direct_match_mask = abs(trans_data['amount'] - target_amount) <= tolerance
                                direct_matching_trans = trans_data[direct_match_mask]
                                
                                if not direct_matching_trans.empty:
                                    for _, trans_row in direct_matching_trans.iterrows():
                                        direct_matches.append({
                                            'Target_ID': target_id,
                                            'Target_Amount': target_amount,
                                            'Matching_Transaction_ID': trans_row['Transaction ID'],
                                            'Transaction_Amount': trans_row['amount'],
                                            'Difference': abs(trans_row['amount'] - target_amount)
                                        })
                            
                            if direct_matches:
                                st.write("**Direct matches found:**")
                                direct_matches_df = pd.DataFrame(direct_matches)
                                st.dataframe(direct_matches_df, use_container_width=True)
                            else:
                                st.info("No direct matches found either. Try adjusting the data or algorithms.")
                
                except Exception as e:
                    st.error(f"Error during benchmarking: {str(e)}")
                    st.error("Full error details:")
                    st.code(traceback.format_exc())
        
        else:
            st.info("👆 Please process the data first using the 'Process Data & Preview' button above.")

    # if st.button("📂 Run Multiple File Benchmark"):
    #     with st.spinner("Running multiple file benchmark..."):
    #         try:
    #             run_multiple_file_benchmark()
    #             st.success("✅ Multiple file benchmark completed.")
    #         except Exception as e:
    #             st.error(f"❌ Error during multiple file benchmark: {str(e)}")


# Debug section (remove in production)
st.sidebar.markdown("---")
st.sidebar.markdown("### 🐛 Debug Info")
if st.sidebar.checkbox("Show Debug Info"):
    st.sidebar.write(f"Processed files count: {len(st.session_state.processed_files)}")
    if st.session_state.processed_files:
        st.sidebar.write("Files:")
        for name in st.session_state.processed_files.keys():
            st.sidebar.write(f"- {name}")

# Clear all data button
if st.sidebar.button("🗑️ Clear All Data"):
    st.session_state.processed_files = {}
    st.session_state.storage = DataStorage()
    st.sidebar.success("All data cleared!")
    st.rerun()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📝 About")
st.sidebar.info(
    "Financial Data Processing System v1.0\n\n"
    "Built with Streamlit for analyzing and processing financial data files."
)