# pages/0_❓_User_Guide.py

import streamlit as st
import logging

try:
    # Use CONCEPTUAL_COLUMNS for the guide now
    from config import APP_TITLE, CONCEPTUAL_COLUMNS, CRITICAL_CONCEPTUAL_COLUMNS, KPI_CONFIG
except ImportError:
    APP_TITLE = "Trading Performance Dashboard"
    CONCEPTUAL_COLUMNS = {
        "date": "Trade Date/Time", "pnl": "Profit or Loss (PnL)",
        "symbol": "Trading Symbol/Ticker", "strategy": "Strategy Name/Identifier",
        # Add other key conceptual columns for the guide's example
    }
    CRITICAL_CONCEPTUAL_COLUMNS = ["date", "pnl"]
    KPI_CONFIG = {}
    print("Warning (UserGuide): Could not import from config. Using fallback values.")

logger = logging.getLogger(APP_TITLE)

def show_user_guide_page():
    st.set_page_config(page_title=f"User Guide - {APP_TITLE}", layout="wide")
    st.title("❓ User Guide & Help")
    logger.info("Rendering User Guide Page.")

    st.markdown("""
    Welcome to the Trading Performance Dashboard! This guide will help you understand how to use the application,
    the data it expects, and how to interpret the various analyses and Key Performance Indicators (KPIs).
    """)

    st.header("1. Getting Started")
    st.subheader("1.1. Uploading Your Trading Journal")
    st.markdown("""
    To begin, upload your trading journal as a CSV file using the file uploader in the sidebar.
    """)

    st.subheader("1.2. Column Mapping - IMPORTANT!")
    st.markdown(f"""
    After uploading your CSV, you will be prompted to **map your CSV columns** to the fields the application expects.
    This step is crucial for the dashboard to understand your data correctly.

    **How it Works:**
    * **Data Preview:** The mapping interface will show the first 5 rows of your uploaded CSV to help you identify your columns.
    * **Application Fields (Left):** On the left, you'll see a list of data fields the application needs (e.g., "{CONCEPTUAL_COLUMNS.get('date', 'Date/Time')}", "{CONCEPTUAL_COLUMNS.get('pnl', 'Profit/Loss')}"). Critical fields required for basic operation are marked with an asterisk (`*`).
    * **Your CSV Columns (Right):** For each application field, select the corresponding column header from *your* CSV file using the dropdown menu.
    * **Auto-Mapping Attempt:** The system will try to automatically suggest mappings based on common names and synonyms. Please review these suggestions carefully.
    * **Data Type Warnings (⚠️):** If the system detects a potential mismatch between the data type of your selected CSV column and what the application expects for a field (e.g., you map a text column to a numeric PnL field), a warning icon (⚠️) will appear. Please pay close attention to these warnings.
    * **Confirmation:** Once you've mapped all necessary columns (especially critical ones), click "Confirm Column Mapping" to proceed.

    **Key Application Fields (Conceptual Columns):**
    The application internally uses standardized names for data fields. You need to map your CSV columns to these concepts:
    """)
    
    # Display key conceptual columns and their descriptions
    with st.expander("View Key Application Data Fields", expanded=False):
        for conceptual_key, description in CONCEPTUAL_COLUMNS.items():
            is_critical = conceptual_key in CRITICAL_CONCEPTUAL_COLUMNS
            st.markdown(f"* **`{conceptual_key}` ({description}){'*' if is_critical else ''}**")
        st.markdown("*Fields marked with `*` are critical and must be mapped.*")

    st.markdown("""
    **Tips for Successful Mapping:**
    * Ensure your CSV is well-formed.
    * Pay close attention to the data preview to correctly identify your columns.
    * Carefully review auto-mapped suggestions.
    * Address any data type mismatch warnings (⚠️) by ensuring the correct CSV column is selected or by cleaning your source data if necessary.
    * If a conceptual field is not present in your CSV (and is not critical), you can leave it unmapped by selecting the blank option in the dropdown.
    """)

    st.subheader("1.3. Using Sidebar Filters")
    st.markdown("""
    Once data is successfully mapped and processed, you can use the filters in the sidebar to refine the dataset for analysis:
    * **Risk-Free Rate:** Set the annual risk-free rate.
    * **Date Range:** Select a specific period.
    * **Symbol Filter:** Filter by trading symbol (if mapped).
    * **Strategy Filter:** Filter by strategy name (if mapped).
    * **Benchmark Selection:** Choose a benchmark for comparison.
    * **Initial Capital:** Set initial capital for percentage return calculations.

    Changes to these filters will dynamically update the analyses.
    """)

    # --- Section 2: Understanding the Pages (Content largely the same, review for column name consistency) ---
    st.header("2. Navigating the Dashboard Pages")
    # ... (Keep existing expanders, ensure any column references in descriptions are conceptual) ...
    # Example for one expander:
    with st.expander("📈 Overview Page", expanded=False):
        st.markdown("""
        Provides a high-level summary. Key Performance Indicators (KPIs) like Total PnL (from your mapped 'pnl' column),
        Win Rate, Sharpe Ratio, etc., are displayed. Also shows the Equity Curve (based on mapped 'date' and 'pnl').
        """)
    # (Add other page descriptions similarly)
    page_descriptions = {
        "📊 Performance Page": "Delve deeper into PnL distributions, performance by time categories (e.g., hour, day of week, month - derived from your mapped 'date' column), and rolling metrics.",
        "🎯 Categorical Analysis": "Analyze performance based on various categories from your data, such as 'strategy', 'symbol', market conditions, etc., if these columns are mapped.",
        "📉 Risk and Duration Page": "Focuses on risk metrics (VaR, CVaR, Max Drawdown), feature correlations, and trade duration analysis (requires a mapped duration column or PnL over time).",
        "⚖️ Strategy Comparison Page": "Compare different trading strategies (from your mapped 'strategy' column) side-by-side.",
        "🔬 Advanced Stats Page": "Explore bootstrap confidence intervals for KPIs, time series decomposition of PnL or equity.",
        "🔮 Stochastic Models Page": "Simulate equity paths using GBM, analyze trade sequences with Markov chains (based on mapped 'pnl').",
        "🤖 AI and ML Page": "Forecast PnL or equity using models like ARIMA or Prophet; perform anomaly detection.",
        "📋 Data View Page": "Inspect the processed trading data after your column mapping and filtering. Download the current view.",
        "📝 Trade Notes Page": "Review and search trade notes (requires a mapped 'notes' column)."
    }
    for page_title, desc in page_descriptions.items():
        with st.expander(page_title, expanded=False):
            st.markdown(desc)


    # --- Section 3: Understanding Key Performance Indicators (KPIs) (Content largely the same) ---
    st.header("3. Understanding Key Performance Indicators (KPIs)")
    st.markdown("The dashboard uses several KPIs. Here are explanations for common ones (calculated from your mapped data):")
    if KPI_CONFIG:
        for kpi_key, kpi_info in KPI_CONFIG.items():
            # ... (KPI descriptions as before) ...
            kpi_name = kpi_info.get("name", kpi_key.replace("_", " ").title())
            description = f"Measures the {kpi_name.lower()}."
            if kpi_key == "total_pnl": description = "The sum of all profits and losses (from your mapped 'pnl' column)."
            # ... other specific descriptions
            with st.expander(f"{kpi_name} ({kpi_info.get('unit', '')})", expanded=False):
                st.markdown(f"**{kpi_name}**: {description}")
                # ... (rest of KPI detail display)
    else:
        st.markdown("*Detailed KPI explanations will be populated based on application configuration.*")


    # --- Section 4: Troubleshooting & FAQ ---
    st.header("4. Troubleshooting & FAQ")
    with st.expander("Column Mapping Issues", expanded=True): # Expanded by default
        st.markdown("""
        * **Mapper Not Appearing:** Ensure your uploaded file is a valid CSV. The mapper appears after headers are successfully read.
        * **Incorrect Auto-Mapping:** Always review auto-suggestions. Use the dropdowns to correct any mismatches. The data preview helps confirm.
        * **Critical Fields Error:** If you see "Critical fields not mapped," ensure you've selected a CSV column for all fields marked with `*` (e.g., Date, PnL).
        * **Duplicate Critical Mapping Error:** Each critical application field (like 'Date', 'PnL') must be mapped to a *unique* column from your CSV. You cannot map the same CSV column to two different critical fields.
        * **Type Mismatch Warnings (⚠️):** This icon indicates the data in your selected CSV column might not be the type expected by the application (e.g., text where a number is needed for PnL).
            * **Action:** Double-check you've selected the correct CSV column. If correct, the issue might be with the data in your CSV file (e.g., text entries in a PnL column). You may need to clean your CSV.
        * **Error after Confirming Mapping:** If an error occurs during data processing *after* you've confirmed the mapping, the error message will now try to indicate which of *your original CSV columns* (that you mapped) caused the problem. This helps pinpoint issues in your source file.
        """)
    with st.expander("Data Upload Issues (General)", expanded=False):
        st.markdown("""
        * **Error reading CSV:** Ensure your file is a valid CSV. Check for encoding issues (UTF-8 is recommended).
        """)
    # ... (other troubleshooting sections as before) ...
    with st.expander("Analysis Not Appearing", expanded=False):
        st.markdown("""
        * **No data after filtering:** If you apply very restrictive filters, there might be no trades matching the criteria. Try adjusting the filters.
        * **Missing Optional Columns:** Some advanced analyses or specific charts rely on optional data fields (e.g., 'trade notes', 'risk amount', specific strategy tags). If you haven't mapped these columns from your CSV, the corresponding features might be disabled or show a message indicating missing data. This is normal if your CSV doesn't contain that specific information.
        * **Insufficient data for specific analyses:** Some advanced analyses require a minimum number of data points.
        """)
    with st.expander("Understanding 'N/A' or 'Inf' in KPIs", expanded=False):
        st.markdown("""
        * **N/A (Not Available):** Typically means the KPI could not be calculated due to insufficient data or missing required input data.
        * **Inf (Infinity):** Can occur in ratios like Profit Factor if Gross Loss is zero.
        """)

    st.markdown("---")
    st.markdown("We hope this guide helps you make the most of the Trading Performance Dashboard!")

if __name__ == "__main__":
    if 'app_initialized' not in st.session_state:
        st.warning("This page is part of a multi-page app. For full functionality, please run the main `app.py` script.")
    show_user_guide_page()
