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
import traceback


# ===== HELPER FUNCTIONS =====

def prepare_sample_data(trans_data, target_data, transaction_sample_size, target_sample_size):
    """Prepare sampled data for benchmarking"""
    # Sample transactions if requested
    if transaction_sample_size > 0 and transaction_sample_size < len(trans_data):
        trans_data = trans_data.sample(transaction_sample_size).copy().reset_index(drop=True)
    
    # Take first N targets if requested
    if target_sample_size > 0 and target_sample_size < len(target_data):
        target_data = target_data.head(target_sample_size).copy().reset_index(drop=True)
    
    return trans_data, target_data

def update_method_results(method_name, result):
    """Update method results in session state"""
    # Remove existing result for this method
    st.session_state.benchmark_results_individual = [
        r for r in st.session_state.benchmark_results_individual 
        if r["Method"] != method_name
    ]
    # Add new result
    st.session_state.benchmark_results_individual.append(result)

def display_method_results(result, show_details=False, show_ga_details=False):
    """Display results for a single method"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("⏱️ Time", result["Time Display"])
    with col2:
        st.metric("💾 Memory", f"{result['Memory (KB)']} KB")
    with col3:
        st.metric("🎯 Matches", result["Matches"])
    with col4:
        success_rate = (result["Matches"] / result["Targets Tested"]) * 100 if result["Targets Tested"] > 0 else 0
        st.metric("📈 Success Rate", f"{success_rate:.1f}%")
    
    # Show detailed results if requested
    if show_details and "Details" in result:
        st.markdown("#### 📋 Detailed Results")
        details = result["Details"]
        
        if "Task 2.1 Matches" in details and not details["Task 2.1 Matches"].empty:
            st.markdown("**Direct Matches:**")
            st.dataframe(details["Task 2.1 Matches"], use_container_width=True)
        
        if "Task 2.2 Matches" in details and not details["Task 2.2 Matches"].empty:
            st.markdown("**Subset Sum Matches:**")
            st.dataframe(details["Task 2.2 Matches"], use_container_width=True)
    
    if show_ga_details and "GA Results" in result:
        st.markdown("#### 🧬 Genetic Algorithm Results")
        ga_results = result["GA Results"]
        
        if ga_results:
            ga_display = []
            for i, (target, subset, achieved_sum, exact_match) in enumerate(ga_results[:10]):  # Show first 10
                ga_display.append({
                    "Target": f"{target:.2f}",
                    "Achieved Sum": f"{achieved_sum:.2f}",
                    "Difference": f"{abs(achieved_sum - target):.4f}",
                    "Exact Match": "✅" if exact_match else "❌",
                    "Subset Size": len(subset)
                })
            
            if ga_display:
                ga_df = pd.DataFrame(ga_display)
                st.dataframe(ga_df, use_container_width=True)
                
                if len(ga_results) > 10:
                    st.info(f"Showing first 10 results out of {len(ga_results)} total results")

def show_detailed_matches(trans_data, target_data, brute_results, method_name):
    """Show detailed matching results with Target IDs and amounts"""
    st.markdown(f"#### 🔍 Detailed {method_name} Matches")
    
    # Show target amounts being tested
    st.markdown("**🎯 Targets Being Tested:**")
    target_display = target_data[['Target ID', 'amount']].copy()
    target_display.columns = ['Target ID', 'Target Amount']
    st.dataframe(target_display, use_container_width=True)
    
    # Show detailed match results
    if 'Task 2.1 Matches' in brute_results and not brute_results['Task 2.1 Matches'].empty:
        st.markdown("**✅ Direct Matches Found:**")
        direct_matches = brute_results['Task 2.1 Matches'].copy()
        st.dataframe(direct_matches, use_container_width=True)
        
        # Enhanced match details with IDs
        st.markdown("**🔗 Match Details with IDs:**")
        enhanced_matches = []
        
        for _, match_row in direct_matches.iterrows():
            target_amount = match_row['target_amount']
            trans_amount = match_row['transaction_amount']
            
            # Find corresponding target and transaction IDs
            matching_targets = target_data[target_data['amount'] == target_amount]['Target ID'].tolist()
            matching_trans = trans_data[trans_data['amount'] == trans_amount]['Transaction ID'].tolist()
            
            for target_id in matching_targets:
                for trans_id in matching_trans:
                    enhanced_matches.append({
                        'Target_ID': target_id,
                        'Target_Amount': target_amount,
                        'Transaction_ID': trans_id,
                        'Transaction_Amount': trans_amount,
                        'Match_Type': 'Direct Match',
                        'Difference': 0.0
                    })
        
        if enhanced_matches:
            enhanced_df = pd.DataFrame(enhanced_matches)
            st.dataframe(enhanced_df, use_container_width=True)
    
    if 'Task 2.2 Matches' in brute_results and not brute_results['Task 2.2 Matches'].empty:
        st.markdown("**🧩 Subset Sum Matches Found:**")
        subset_matches = brute_results['Task 2.2 Matches'].copy()
        st.dataframe(subset_matches, use_container_width=True)
        
        # Show subset details with transaction amounts
        st.markdown("**📊 Subset Match Breakdown:**")
        for _, subset_match in subset_matches.iterrows():
            target_id = subset_match['Target Identifier']
            matched_trans_ids = subset_match['Matched Transactions']
            target_sum = subset_match['Sum']
            
            # Get target amount for this target ID
            target_amount_info = target_data[target_data['Target ID'] == target_id]
            if not target_amount_info.empty:
                target_amount = target_amount_info.iloc[0]['amount']
            else:
                target_amount = target_sum
            
            with st.expander(f"🎯 {target_id} (Target: {target_amount:.2f}) → {len(matched_trans_ids)} transactions"):
                # Show individual transaction details
                subset_details = []
                total_check = 0
                
                for trans_id in matched_trans_ids:
                    trans_info = trans_data[trans_data['Transaction ID'] == trans_id]
                    if not trans_info.empty:
                        trans_amount = trans_info.iloc[0]['amount']
                        total_check += trans_amount
                        subset_details.append({
                            'Transaction_ID': trans_id,
                            'Amount': trans_amount
                        })
                
                if subset_details:
                    subset_df = pd.DataFrame(subset_details)
                    st.dataframe(subset_df, use_container_width=True)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Target Amount", f"{target_amount:.2f}")
                    with col2:
                        st.metric("Subset Sum", f"{total_check:.2f}")
                    with col3:
                        st.metric("Difference", f"{abs(total_check - target_amount):.4f}")
    
    # Show unmatched targets
    if 'Task 2.1 Matches' in brute_results or 'Task 2.2 Matches' in brute_results:
        matched_target_amounts = set()
        
        if 'Task 2.1 Matches' in brute_results and not brute_results['Task 2.1 Matches'].empty:
            matched_target_amounts.update(brute_results['Task 2.1 Matches']['target_amount'].tolist())
        
        if 'Task 2.2 Matches' in brute_results and not brute_results['Task 2.2 Matches'].empty:
            matched_target_amounts.update(brute_results['Task 2.2 Matches']['Sum'].tolist())
        
        unmatched_targets = target_data[~target_data['amount'].isin(matched_target_amounts)].copy()
        
        if not unmatched_targets.empty:
            st.markdown("**❌ Unmatched Targets:**")
            unmatched_display = unmatched_targets[['Target ID', 'amount']].copy()
            unmatched_display.columns = ['Target ID', 'Target Amount']
            st.dataframe(unmatched_display, use_container_width=True)
        else:
            st.success("🎉 All targets were successfully matched!")

def show_dp_detailed_matches(trans_data, target_data, precision, max_sum_limit):
    """Show detailed Dynamic Programming matching results"""
    st.markdown("#### 🔍 Detailed Dynamic Programming Matches")
    
    # Show settings used
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"⚙️ Precision: {precision}")
    with col2:
        st.info(f"🎯 Max Sum Limit: {'Auto' if max_sum_limit == 0 else max_sum_limit}")
    
    # Show target amounts being tested
    st.markdown("**🎯 Targets Being Tested:**")
    target_display = target_data[['Target_ID', 'amount']].copy()
    target_display.columns = ['Target ID', 'Target Amount']
    st.dataframe(target_display, use_container_width=True)
    
    # Run detailed matching with results
    st.markdown("**🔍 Individual Target Results:**")
    
    # transaction_tuples = list(zip(trans_data['amount'].tolist(), trans_data['Target_ID'].tolist()))
    transaction_tuples = list(zip(trans_data['amount'].tolist(), trans_data['Transaction_ID'].tolist()))

    detailed_results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (_, target_row) in enumerate(target_data.iterrows()):
        try:
            target_id = target_row['Target_ID']
            target_amount = target_row['amount']
            
            result = subset_sum_exists(
                transaction_tuples,
                (target_amount, target_id),
                max_sum=max_sum_limit if max_sum_limit > 0 else None,
                precision=precision
            )
            
            if result and len(result) >= 3:
                success = result[0]
                returned_target_id = result[1]
                matching_trans_ids = result[2] if result[2] else []
                
                # Calculate actual sum if match found
                actual_sum = 0
                if success and matching_trans_ids:
                    matching_trans = trans_data[trans_data['Target_ID'].isin(matching_trans_ids)]
                    actual_sum = matching_trans['amount'].sum()
                
                detailed_results.append({
                    'Target_ID': target_id,
                    'Target_Amount': target_amount,
                    'Match_Found': '✅' if success else '❌',
                    'Matching_Transactions': ', '.join(matching_trans_ids) if matching_trans_ids else 'None',
                    'Actual_Sum': actual_sum if success else 0,
                    'Difference': abs(actual_sum - target_amount) if success else 'N/A',
                    'Transaction_Count': len(matching_trans_ids) if matching_trans_ids else 0
                })
            else:
                detailed_results.append({
                    'Target_ID': target_id,
                    'Target_Amount': target_amount,
                    'Match_Found': '❌',
                    'Matching_Transactions': 'None',
                    'Actual_Sum': 0,
                    'Difference': 'N/A',
                    'Transaction_Count': 0
                })
            
            # Update progress
            progress = (i + 1) / len(target_data)
            progress_bar.progress(progress)
            status_text.text(f"Processing target {i+1}/{len(target_data)}: {target_id}")
            
        except Exception as e:
            st.warning(f"Error processing target {target_row['Target_ID']}: {str(e)}")
            detailed_results.append({
                'Target_ID': target_row['Target_ID'],
                'Target_Amount': target_row['amount'],
                'Match_Found': '⚠️ Error',
                'Matching_Transactions': 'Error',
                'Actual_Sum': 0,
                'Difference': 'Error',
                'Transaction_Count': 0
            })
    
    progress_bar.empty()
    status_text.empty()
    
    # Display results
    if detailed_results:
        results_df = pd.DataFrame(detailed_results)
        st.dataframe(results_df, use_container_width=True)
        
        # Summary statistics
        successful_matches = len([r for r in detailed_results if r['Match_Found'] == '✅'])
        failed_matches = len([r for r in detailed_results if r['Match_Found'] == '❌'])
        error_matches = len([r for r in detailed_results if r['Match_Found'] == '⚠️ Error'])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("✅ Successful", successful_matches)
        with col2:
            st.metric("❌ Failed", failed_matches)
        with col3:
            st.metric("⚠️ Errors", error_matches)
        with col4:
            success_rate = (successful_matches / len(detailed_results)) * 100 if detailed_results else 0
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        # Show detailed breakdown for successful matches
        successful_results = [r for r in detailed_results if r['Match_Found'] == '✅']
        if successful_results:
            st.markdown("**🎉 Successful Matches Breakdown:**")
            for result in successful_results[:5]:  # Show first 5 successful matches
                with st.expander(f"🎯 {result['Target_ID']} (Target: {result['Target_Amount']:.2f})"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Target Amount:** {result['Target_Amount']:.2f}")
                        st.write(f"**Actual Sum:** {result['Actual_Sum']:.2f}")
                        st.write(f"**Difference:** {result['Difference']}")
                    with col2:
                        st.write(f"**Transaction Count:** {result['Transaction_Count']}")
                        st.write(f"**Transaction IDs:** {result['Matching_Transactions']}")
                        
                        # Show individual transaction details
                        if result['Matching_Transactions'] != 'None':
                            trans_ids = result['Matching_Transactions'].split(', ')
                            trans_details = trans_data[trans_data['Target_ID'].isin(trans_ids)][['Target_ID', 'amount']]
                            if not trans_details.empty:
                                st.dataframe(trans_details, use_container_width=True)

def show_ga_detailed_matches(trans_data, target_data, ga_results):
    """Show detailed Genetic Algorithm matching results"""
    st.markdown("#### 🧬 Detailed Genetic Algorithm Results")
    
    # Show target amounts being tested
    st.markdown("**🎯 Targets Being Tested:**")
    target_display = target_data[['Target_ID', 'amount']].copy()
    target_display.columns = ['Target ID', 'Target Amount']
    st.dataframe(target_display, use_container_width=True)
    
    if ga_results:
        # Create comprehensive results table
        detailed_ga_results = []
        
        for i, (target_amount, subset, achieved_sum, exact_match) in enumerate(ga_results):
            # Get corresponding target ID
            target_id = f"TG_{i+1:04d}"  # Assuming sequential target IDs
            if i < len(target_data):
                target_id = target_data.iloc[i]['Target_ID']
            
            # Calculate subset details
            subset_transactions = []
            if subset:
                # Find transaction IDs corresponding to subset values
                for subset_amount in subset:
                    matching_trans = trans_data[abs(trans_data['amount'] - subset_amount) < 0.001]
                    if not matching_trans.empty:
                        # subset_transactions.append(matching_trans.iloc[0]['Target_ID'])
                        subset_transactions.append(matching_trans.iloc[0]['Transaction_ID'])
            
            fuzzy_ratio = max(0, 100 * (1 - abs(target_amount - achieved_sum) / max(abs(target_amount), 1)))
            
            detailed_ga_results.append({
                'Target_ID': target_id,
                'Target_Amount': target_amount,
                'Achieved_Sum': achieved_sum,
                'Difference': abs(achieved_sum - target_amount),
                'Exact_Match': '✅' if exact_match else '❌',
                'Fuzzy_Ratio': f"{fuzzy_ratio:.1f}%",
                'Subset_Size': len(subset),
                'Subset_Values': ', '.join([f"{val:.2f}" for val in subset]) if subset else 'None',
                'Transaction_IDs': ', '.join(subset_transactions) if subset_transactions else 'None'
            })
        
        # Display results table
        st.markdown("**📊 GA Results Summary:**")
        ga_df = pd.DataFrame(detailed_ga_results)
        st.dataframe(ga_df, use_container_width=True)
        
        # Summary statistics
        exact_matches = len([r for r in detailed_ga_results if r['Exact_Match'] == '✅'])
        high_fuzzy = len([r for r in detailed_ga_results if float(r['Fuzzy_Ratio'].rstrip('%')) >= 90])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎯 Exact Matches", exact_matches)
        with col2:
            st.metric("📈 High Accuracy (>90%)", high_fuzzy)
        with col3:
            avg_fuzzy = sum(float(r['Fuzzy_Ratio'].rstrip('%')) for r in detailed_ga_results) / len(detailed_ga_results)
            st.metric("📊 Avg Accuracy", f"{avg_fuzzy:.1f}%")
        with col4:
            avg_diff = sum(r['Difference'] for r in detailed_ga_results) / len(detailed_ga_results)
            st.metric("📏 Avg Difference", f"{avg_diff:.4f}")
        
        # Show detailed breakdown for best matches
        st.markdown("**🏆 Best Matches Breakdown:**")
        sorted_results = sorted(detailed_ga_results, key=lambda x: float(x['Fuzzy_Ratio'].rstrip('%')), reverse=True)
        
        for i, result in enumerate(sorted_results[:5]):  # Show top 5 matches
            accuracy = float(result['Fuzzy_Ratio'].rstrip('%'))
            status_emoji = "🎯" if result['Exact_Match'] == '✅' else "📈" if accuracy >= 90 else "📊"
            
            with st.expander(f"{status_emoji} {result['Target_ID']} - {result['Fuzzy_Ratio']} accuracy (Target: {result['Target_Amount']:.2f})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Target ID:** {result['Target_ID']}")
                    st.write(f"**Target Amount:** {result['Target_Amount']:.2f}")
                    st.write(f"**Achieved Sum:** {result['Achieved_Sum']:.2f}")
                    st.write(f"**Difference:** {result['Difference']:.4f}")
                    st.write(f"**Accuracy:** {result['Fuzzy_Ratio']}")
                
                with col2:
                    st.write(f"**Subset Size:** {result['Subset_Size']}")
                    st.write(f"**Exact Match:** {result['Exact_Match']}")
                    if result['Subset_Values'] != 'None':
                        st.write("**Subset Values:**")
                        st.text(result['Subset_Values'])
                    if result['Transaction_IDs'] != 'None':
                        st.write("**Transaction IDs:**")
                        st.text(result['Transaction_IDs'])
                
                # Progress bar for accuracy
                accuracy_val = float(result['Fuzzy_Ratio'].rstrip('%')) / 100
                st.progress(accuracy_val)
        
        # Show histogram of accuracy distribution
        st.markdown("**📈 Accuracy Distribution:**")
        accuracy_values = [float(r['Fuzzy_Ratio'].rstrip('%')) for r in detailed_ga_results]
        
        import plotly.graph_objects as go
        fig = go.Figure(data=[go.Histogram(x=accuracy_values, nbinsx=20)])
        fig.update_layout(
            title="Distribution of GA Match Accuracy",
            xaxis_title="Accuracy (%)",
            yaxis_title="Count",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("No GA results available to display.")

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
        # Benchmark Settings and Method Selection
        # -------------------------------
        if 'processed_transactions' in st.session_state and 'processed_targets' in st.session_state:
            st.markdown("### ⚙️ Benchmark Settings")
            
            # Data sampling settings
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Target Data Selection**")
                max_targets = len(st.session_state.processed_targets)
                target_sample_size = st.number_input(
                    "Number of targets to test (0 for all)",
                    min_value=0,
                    max_value=min(50, max_targets),
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
                    max_value=min(1000, max_transactions),
                    value=min(100, max_transactions),
                    help="Number of transactions to use for matching (smaller = faster)",
                    key="transaction_sample_size"
                )
            
            # Initialize benchmark results in session state if not exists
            if 'benchmark_results_individual' not in st.session_state:
                st.session_state.benchmark_results_individual = []
            
            # Method selection and individual benchmarking
            st.markdown("### 🎯 Select Method to Benchmark")
            
            # Create tabs for each method
            tab1, tab2, tab3, tab4 = st.tabs(["🐌 Brute Force", "⚡ Dynamic Programming", "🧬 Genetic Algorithm", "📊 Compare Results"])
            
            # ===== TAB 1: BRUTE FORCE =====
            with tab1:
                st.markdown("#### 🐌 Brute Force Method")
                st.info("⚠️ Warning: Brute force can be very slow for large datasets. Recommended for small datasets only.")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Current Selection:**")
                    st.write(f"• Transactions: {transaction_sample_size if transaction_sample_size > 0 else len(st.session_state.processed_transactions)}")
                    st.write(f"• Targets: {target_sample_size if target_sample_size > 0 else len(st.session_state.processed_targets)}")
                
                with col2:
                    # Estimated complexity warning
                    actual_trans = transaction_sample_size if transaction_sample_size > 0 else len(st.session_state.processed_transactions)
                    actual_targets = target_sample_size if target_sample_size > 0 else len(st.session_state.processed_targets)
                    complexity_estimate = (2 ** min(actual_trans, 20)) * actual_targets  # Cap at 2^20 for safety
                    
                    if actual_trans > 15:
                        st.warning(f"⚠️ High complexity: ~{complexity_estimate:,.0f} operations")
                        st.write("Consider reducing transaction count to < 15 for reasonable performance")
                    else:
                        st.success(f"✅ Manageable complexity: ~{complexity_estimate:,.0f} operations")
                
                if st.button("🐌 Run Brute Force", key="run_brute_force"):
                    try:
                        trans_data, target_data = prepare_sample_data(
                            st.session_state.processed_transactions.copy(),
                            st.session_state.processed_targets.copy(),
                            transaction_sample_size,
                            target_sample_size
                        )
                        
                        # Run brute force with progress tracking
                        with st.spinner("Running Brute Force method..."):
                            start_time = time.time()
                            tracemalloc.start()
                            
                            # Limit brute force for safety
                            safe_targets = min(len(target_data), 5) if len(st.session_state.processed_transactions) > 100 else len(target_data)
                            limited_target_data = target_data.head(safe_targets).copy()
                            
                            # Rename columns for brute_force_subset function
                            trans_data_bf = trans_data.rename(columns={'Transaction_ID': 'Transaction ID'})
                            target_data_bf = limited_target_data.rename(columns={'Target_ID': 'Target ID'})
                            
                            brute_results = brute_force_subset(trans_data_bf, target_data_bf)
                            
                            # Count matches
                            brute_force_matches = 0
                            if 'Task 2.1 Matches' in brute_results and not brute_results['Task 2.1 Matches'].empty:
                                brute_force_matches += len(brute_results['Task 2.1 Matches'])
                            if 'Task 2.2 Matches' in brute_results and not brute_results['Task 2.2 Matches'].empty:
                                brute_force_matches += len(brute_results['Task 2.2 Matches'])
                            
                            elapsed = time.time() - start_time
                            _, peak = tracemalloc.get_traced_memory()
                            tracemalloc.stop()
                        
                        # Store results
                        method_result = {
                            "Method": "Brute Force",
                            "Time (s)": round(elapsed, 3),
                            "Time Display": format_time(elapsed),
                            "Memory (KB)": round(peak / 1024, 2),
                            "Matches": brute_force_matches,
                            "Targets Tested": len(limited_target_data),
                            "Transactions Used": len(trans_data),
                            "Details": brute_results
                        }
                        
                        # Update session state
                        update_method_results("Brute Force", method_result)
                        
                        # Display results
                        st.success(f"✅ Brute Force completed in {format_time(elapsed)}")
                        display_method_results(method_result, show_details=True)
                        
                        # Show detailed target matching results
                        show_detailed_matches(trans_data_bf, target_data_bf, brute_results, "Brute Force")
                        
                    except Exception as e:
                        st.error(f"❌ Brute Force failed: {str(e)}")
                        st.code(traceback.format_exc())
            
            # ===== TAB 2: DYNAMIC PROGRAMMING =====
            with tab2:
                st.markdown("#### ⚡ Dynamic Programming Method")
                st.info("💡 Good for medium-sized datasets with reasonable precision requirements.")
                
                col1, col2 = st.columns(2)
                with col1:
                    precision = st.selectbox(
                        "Precision (for decimal handling)",
                        [1, 10, 100, 1000],
                        index=2,
                        help="Higher precision = more memory usage"
                    )
                
                with col2:
                    max_sum_limit = st.number_input(
                        "Max Sum Limit (0 for auto)",
                        min_value=0,
                        value=0,
                        help="Limits memory usage by capping the sum range"
                    )
                
                if st.button("⚡ Run Dynamic Programming", key="run_dp"):
                    try:
                        trans_data, target_data = prepare_sample_data(
                            st.session_state.processed_transactions.copy(),
                            st.session_state.processed_targets.copy(),
                            transaction_sample_size,
                            target_sample_size
                        )
                        
                        with st.spinner("Running Dynamic Programming method..."):
                            start_time = time.time()
                            tracemalloc.start()
                            
                            dp_matches = 0
                            transaction_tuples = list(zip(trans_data['amount'].tolist(), trans_data['Transaction_ID'].tolist()))
                            
                            # Progress bar for DP
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for i, (_, target_row) in enumerate(target_data.iterrows()):
                                try:
                                    result = subset_sum_exists(
                                        transaction_tuples,
                                        (target_row['amount'], target_row['Target_ID']),
                                        max_sum=max_sum_limit if max_sum_limit > 0 else None,
                                        precision=precision
                                    )
                                    if result and len(result) > 0 and result[0]:
                                        dp_matches += 1
                                    
                                    # Update progress
                                    progress = (i + 1) / len(target_data)
                                    progress_bar.progress(progress)
                                    status_text.text(f"Processing target {i+1}/{len(target_data)} - Found {dp_matches} matches so far")
                                    
                                except Exception as e:
                                    st.warning(f"DP error for target {target_row['Target_ID']}: {str(e)}")
                                    continue
                            
                            progress_bar.empty()
                            status_text.empty()
                            
                            elapsed = time.time() - start_time
                            _, peak = tracemalloc.get_traced_memory()
                            tracemalloc.stop()
                        
                        # Store results
                        method_result = {
                            "Method": "Dynamic Programming",
                            "Time (s)": round(elapsed, 3),
                            "Time Display": format_time(elapsed),
                            "Memory (KB)": round(peak / 1024, 2),
                            "Matches": dp_matches,
                            "Targets Tested": len(target_data),
                            "Transactions Used": len(trans_data),
                            "Settings": {"Precision": precision, "Max Sum Limit": max_sum_limit}
                        }
                        
                        update_method_results("Dynamic Programming", method_result)
                        
                        st.success(f"✅ Dynamic Programming completed in {format_time(elapsed)}")
                        display_method_results(method_result)
                        
                        # Show detailed DP matching results
                        show_dp_detailed_matches(trans_data, target_data, precision, max_sum_limit)
                        
                    except Exception as e:
                        st.error(f"❌ Dynamic Programming failed: {str(e)}")
                        st.code(traceback.format_exc())
            
            # ===== TAB 3: GENETIC ALGORITHM =====
            with tab3:
                st.markdown("#### 🧬 Genetic Algorithm Method")
                st.info("🔬 Approximation method - good for large datasets, may not find exact matches.")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    pop_size = st.number_input("Population Size", min_value=10, max_value=500, value=100)
                    gen_count = st.number_input("Generations", min_value=10, max_value=1000, value=200)
                
                with col2:
                    mut_rate = st.slider("Mutation Rate", 0.01, 0.2, 0.05, 0.01)
                    elite_size = st.number_input("Elite Size", min_value=1, max_value=20, value=5)
                
                with col3:
                    tourn_size = st.number_input("Tournament Size", min_value=2, max_value=10, value=3)
                    early_stop = st.checkbox("Early Stopping", value=True)
                
                if st.button("🧬 Run Genetic Algorithm", key="run_ga"):
                    try:
                        trans_data, target_data = prepare_sample_data(
                            st.session_state.processed_transactions.copy(),
                            st.session_state.processed_targets.copy(),
                            transaction_sample_size,
                            target_sample_size
                        )
                        
                        with st.spinner("Running Genetic Algorithm method..."):
                            start_time = time.time()
                            tracemalloc.start()
                            
                            ga_solver = SubsetSumGA(
                                trans_data['amount'].tolist(),
                                target_data['amount'].tolist(),
                                pop_size=pop_size,
                                gen_count=gen_count,
                                mut_rate=mut_rate,
                                elite_size=elite_size,
                                tourn_size=tourn_size
                            )
                            
                            # Progress tracking for GA
                            progress_container = st.container()
                            with progress_container:
                                st.write("🧬 Evolution in progress...")
                                ga_progress = st.progress(0)
                                ga_status = st.empty()
                            
                            # Run GA with custom progress callback (if available)
                            ga_results = ga_solver.find_subsets(verbose=False)
                            
                            # Clean up progress indicators
                            ga_progress.empty()
                            ga_status.empty()
                            
                            # Count matches
                            ga_matches = 0
                            exact_matches = 0
                            if hasattr(ga_results, '__len__'):
                                ga_matches = len([r for r in ga_results if r is not None])
                                # Check for exact matches
                                for result in ga_results:
                                    if result and len(result) >= 4 and result[3]:  # exact_match flag
                                        exact_matches += 1
                            
                            elapsed = time.time() - start_time
                            _, peak = tracemalloc.get_traced_memory()
                            tracemalloc.stop()
                        
                        # Store results
                        method_result = {
                            "Method": "Genetic Algorithm",
                            "Time (s)": round(elapsed, 3),
                            "Time Display": format_time(elapsed),
                            "Memory (KB)": round(peak / 1024, 2),
                            "Matches": ga_matches,
                            "Exact Matches": exact_matches,
                            "Targets Tested": len(target_data),
                            "Transactions Used": len(trans_data),
                            "Settings": {
                                "Population Size": pop_size,
                                "Generations": gen_count,
                                "Mutation Rate": mut_rate,
                                "Elite Size": elite_size
                            },
                            "GA Results": ga_results
                        }
                        
                        update_method_results("Genetic Algorithm", method_result)
                        
                        st.success(f"✅ Genetic Algorithm completed in {format_time(elapsed)}")
                        display_method_results(method_result, show_ga_details=True)
                        
                        # Show detailed GA matching results
                        show_ga_detailed_matches(trans_data, target_data, ga_results)
                        
                    except Exception as e:
                        st.error(f"❌ Genetic Algorithm failed: {str(e)}")
                        st.code(traceback.format_exc())
            
            # ===== TAB 4: COMPARE RESULTS =====
            with tab4:
                st.markdown("#### 📊 Compare All Results")
                
                # Show current target amounts being tested
                st.markdown("#### 🎯 Current Target Data Preview")
                if 'processed_targets' in st.session_state:
                    target_preview = st.session_state.processed_targets.copy()
                    # Apply sampling to show what will be tested
                    if target_sample_size > 0 and target_sample_size < len(target_preview):
                        target_preview = target_preview.head(target_sample_size)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Target IDs and Amounts:**")
                        display_targets = target_preview[['Target_ID', 'amount']].copy()
                        display_targets.columns = ['Target ID', 'Target Amount']
                        st.dataframe(display_targets, use_container_width=True)
                    
                    with col2:
                        st.markdown("**Target Statistics:**")
                        st.metric("Total Targets", len(target_preview))
                        st.metric("Amount Range", f"{target_preview['amount'].min():.2f} - {target_preview['amount'].max():.2f}")
                        st.metric("Average Amount", f"{target_preview['amount'].mean():.2f}")
                        st.metric("Median Amount", f"{target_preview['amount'].median():.2f}")
                
                if st.session_state.benchmark_results_individual:
                    st.markdown("#### 📈 Method Performance Comparison")
                    # Create comparison table
                    comparison_data = []
                    for result in st.session_state.benchmark_results_individual:
                        comparison_data.append({
                            "Method": result["Method"],
                            "Time": result["Time Display"],
                            "Memory (KB)": result["Memory (KB)"],
                            "Matches": result["Matches"],
                            "Targets Tested": result["Targets Tested"],
                            "Transactions Used": result["Transactions Used"]
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # Performance charts
                    if len(comparison_data) > 1:
                        st.markdown("### 📈 Performance Comparison")
                        
                        # Extract numeric time data for plotting
                        plot_data = []
                        for result in st.session_state.benchmark_results_individual:
                            if result["Time (s)"] != float('inf'):
                                plot_data.append({
                                    "Method": result["Method"],
                                    "Time (s)": result["Time (s)"],
                                    "Memory (KB)": result["Memory (KB)"],
                                    "Match Rate (%)": (result["Matches"] / result["Targets Tested"]) * 100 if result["Targets Tested"] > 0 else 0
                                })
                        
                        if plot_data:
                            plot_df = pd.DataFrame(plot_data)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                fig_time = px.bar(plot_df, x='Method', y='Time (s)', 
                                               title="Execution Time Comparison")
                                st.plotly_chart(fig_time, use_container_width=True)
                            
                            with col2:
                                fig_mem = px.bar(plot_df, x='Method', y='Memory (KB)', 
                                               title="Memory Usage Comparison")
                                st.plotly_chart(fig_mem, use_container_width=True)
                            
                            # Match rate comparison
                            fig_matches = px.bar(plot_df, x='Method', y='Match Rate (%)', 
                                               title="Match Success Rate")
                            st.plotly_chart(fig_matches, use_container_width=True)
                    
                    # Clear results button
                    if st.button("🗑️ Clear All Results"):
                        st.session_state.benchmark_results_individual = []
                        st.success("✅ Results cleared!")
                        st.rerun()
                        
                else:
                    st.info("🔍 No benchmark results available. Run some methods first!")
        
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