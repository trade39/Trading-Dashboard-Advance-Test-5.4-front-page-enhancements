# pages/0_❓_User_Guide.py

import streamlit as st
import logging
import os

try:
    from config import APP_TITLE, CONCEPTUAL_COLUMNS, CRITICAL_CONCEPTUAL_COLUMNS, KPI_CONFIG
except ImportError:
    APP_TITLE = "Trading Performance Dashboard"
    CONCEPTUAL_COLUMNS = {
        "date": "Trade Date/Time", "pnl": "Profit or Loss (PnL)",
        "symbol": "Trading Symbol/Ticker", "strategy": "Strategy Name/Identifier",
        "volume": "Trade Volume/Quantity", "commission": "Commission Paid",
        "fees": "Exchange/Broker Fees", "entry_price": "Entry Price",
        "exit_price": "Exit Price", "trade_type": "Trade Type (e.g., Long/Short)",
        "account_id": "Account Identifier", "notes": "Trade Notes/Comments"
    }
    CRITICAL_CONCEPTUAL_COLUMNS = ["date", "pnl"]
    KPI_CONFIG = {
        "total_pnl": {"name": "Total PnL", "unit": "$", "description": "Sum of all profits and losses."},
        "win_rate": {"name": "Win Rate", "unit": "%", "description": "Percentage of profitable trades."},
        "sharpe_ratio": {"name": "Sharpe Ratio", "unit": "", "description": "Risk-adjusted return."},
    }
    print("Warning (UserGuide): Could not import from config. Using fallback values.")

logger = logging.getLogger(APP_TITLE)

def load_css(file_path):
    """Loads a CSS file and returns its content as a string."""
    try:
        with open(file_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        logger.warning(f"CSS file not found at {file_path}. Styles may not be applied.")
        return ""

def show_user_guide_page():
    st.set_page_config(page_title=f"User Guide - {APP_TITLE}", layout="wide")
    
    # Construct the path to style.css relative to this script's location
    # Assuming pages/ is at the same level as the root directory where style.css might be
    # or if style.css is in a known 'static' or 'assets' folder.
    # For direct deployment, style.css is often in the root.
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "style.css") 
    
    # Fallback if the above path is not correct (e.g. if pages is not in root)
    if not os.path.exists(css_path):
        css_path = "style.css" # Assumes style.css is in the same directory as app.py or Streamlit's root

    if os.path.exists(css_path):
        css = load_css(css_path)
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"User Guide: `style.css` not found at expected paths. Custom styles might not be applied. Checked: {css_path} and 'style.css'")


    st.markdown(f"<h1 class='welcome-title' style='text-align: left; margin-bottom: 0.25rem;'>❓ User Guide & Help</h1>", unsafe_allow_html=True)
    st.markdown(f"<p class='page-subtitle' style='text-align: left; margin-top:0;'>Your comprehensive guide to the {APP_TITLE}</p>", unsafe_allow_html=True)

    logger.info("Rendering User Guide Page.")

    # --- Introduction Section ---
    st.markdown("""
    <div class='custom-card' style='margin-bottom: 2rem;'>
        <h2 class='section-header' style='margin-top:0; border-top: none; padding-top:0;'>👋 Welcome!</h2>
        <p>This guide will help you understand how to use the <strong>{APP_TITLE}</strong>,
        the data it expects, and how to interpret the various analyses and Key Performance Indicators (KPIs).
        Navigate through the sections below to get started.</p>
    </div>
    """.format(APP_TITLE=APP_TITLE), unsafe_allow_html=True)

    # --- Section 1: Getting Started ---
    st.markdown("<h2 class='section-header'>🚀 1. Getting Started</h2>", unsafe_allow_html=True)
    
    with st.container(): # Use st.container for better grouping if applying specific background/border
        st.markdown("<h3 class='section-subheader'>📄 1.1. Uploading Your Trading Journal</h3>", unsafe_allow_html=True)
        st.markdown("""
        To begin, upload your trading journal as a CSV file. You can typically find the file uploader
        in the sidebar of the main application pages (e.g., on the "Upload Data" or a similar page).
        Ensure your CSV file is well-formatted for the best results.
        """)

        st.markdown("<h3 class='section-subheader'>🔗 1.2. Column Mapping - IMPORTANT!</h3>", unsafe_allow_html=True)
        st.markdown(f"""
        After uploading your CSV, you will be prompted to **map your CSV columns** to the fields the application expects.
        This step is crucial for the dashboard to understand your data correctly.

        **How it Works:**
        * **Data Preview:** The mapping interface will show the first few rows of your uploaded CSV to help you identify your columns.
        * **Application Fields (Left):** On the left, you'll see a list of data fields the application needs (e.g., "{CONCEPTUAL_COLUMNS.get('date', 'Date/Time')}", "{CONCEPTUAL_COLUMNS.get('pnl', 'Profit/Loss')}"). Critical fields required for basic operation are marked with an asterisk (`*`).
        * **Your CSV Columns (Right):** For each application field, select the corresponding column header from *your* CSV file using the dropdown menu.
        * **Auto-Mapping Attempt:** The system will try to automatically suggest mappings based on common names and synonyms. **Please review these suggestions carefully.**
        * **Data Type Warnings (⚠️):** If the system detects a potential mismatch between the data type of your selected CSV column and what the application expects for a field (e.g., you map a text column to a numeric PnL field), a warning icon (⚠️) will appear. Please pay close attention to these warnings.
        * **Confirmation:** Once you've mapped all necessary columns (especially critical ones), click "Confirm Column Mapping" to proceed.
        """)

        with st.expander("🔑 View Key Application Data Fields (Conceptual Columns)", expanded=False):
            if CONCEPTUAL_COLUMNS:
                st.markdown("The application internally uses standardized names for data fields. You need to map your CSV columns to these concepts:")
                for conceptual_key, description in CONCEPTUAL_COLUMNS.items():
                    is_critical = conceptual_key in CRITICAL_CONCEPTUAL_COLUMNS
                    st.markdown(f"""
                    <div style='padding: 0.5rem 0; border-bottom: 1px solid var(--input-border-color);'>
                        <strong><code>{conceptual_key}</code></strong>{'<span class="critical-marker" style="color: var(--error-color); font-weight: bold;">*</span>' if is_critical else ''}:
                        <em>{description}</em>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("<p style='margin-top: 0.5rem; font-size: 0.9em;'><em>Fields marked with <span class='critical-marker' style='color: var(--error-color); font-weight: bold;'>*</span> are critical and must be mapped.</em></p>", unsafe_allow_html=True)
            else:
                st.info("Conceptual column definitions are not currently loaded.")
        
        st.markdown("""
        <div class="custom-message-box info" style="margin-top: 1.5rem;">
            <span class="custom-message-box-icon">💡</span>
            <div>
                <strong>Tips for Successful Mapping:</strong>
                <ul>
                    <li>Ensure your CSV is well-formed and uses UTF-8 encoding if possible.</li>
                    <li>Pay close attention to the data preview to correctly identify your columns.</li>
                    <li>Carefully review auto-mapped suggestions.</li>
                    <li>Address any data type mismatch warnings (⚠️) by ensuring the correct CSV column is selected or by cleaning your source data if necessary.</li>
                    <li>If a conceptual field is not present in your CSV (and is not critical), you can leave it unmapped by selecting the blank/empty option in the dropdown.</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<h3 class='section-subheader'>⚙️ 1.3. Using Sidebar Filters</h3>", unsafe_allow_html=True)
        st.markdown("""
        Once data is successfully mapped and processed, you can use the filters in the sidebar on various analysis pages to refine the dataset:
        * **Risk-Free Rate:** Set the annual risk-free rate for calculations like the Sharpe Ratio.
        * **Date Range:** Select a specific period for analysis.
        * **Symbol Filter:** Filter by trading symbol (if mapped and available).
        * **Strategy Filter:** Filter by strategy name (if mapped and available).
        * **Benchmark Selection:** Choose a benchmark for comparison (if applicable).
        * **Initial Capital:** Set initial capital for percentage return calculations.

        Changes to these filters will dynamically update the analyses and visualizations.
        """)

    # --- Section 2: Understanding the Pages ---
    st.markdown("<h2 class='section-header'>🗺️ 2. Navigating the Dashboard Pages</h2>", unsafe_allow_html=True)
    st.markdown("The dashboard is organized into several pages, each focusing on different aspects of your trading performance. Here's a brief overview:")

    page_descriptions = {
        "📈 Overview Page": "Provides a high-level summary of your trading performance. Key Performance Indicators (KPIs) like Total PnL (from your mapped 'pnl' column), Win Rate, Sharpe Ratio, etc., are displayed. Also shows the Equity Curve (based on mapped 'date' and 'pnl').",
        "📊 Performance Analysis Page": "Delve deeper into PnL distributions, performance by time categories (e.g., hour, day of week, month - derived from your mapped 'date' column), and rolling metrics.",
        "🎯 Categorical Analysis Page": "Analyze performance based on various categories from your data, such as 'strategy', 'symbol', market conditions, etc., if these columns are mapped.",
        "📉 Risk and Duration Page": "Focuses on risk metrics such as Value at Risk (VaR), Conditional VaR (CVaR), Maximum Drawdown, feature correlations, and trade duration analysis (requires a mapped duration column or PnL over time).",
        "⚖️ Strategy Comparison Page": "Compare different trading strategies (from your mapped 'strategy' column) side-by-side using various metrics and visualizations.",
        "🔬 Advanced Statistics Page": "Explore bootstrap confidence intervals for KPIs, time series decomposition of PnL or equity, and other advanced statistical measures.",
        "🔮 Stochastic Models Page": "Simulate future equity paths using models like Geometric Brownian Motion (GBM), analyze trade sequences with Markov chains (based on mapped 'pnl').",
        "🤖 AI & Machine Learning Page": "Leverage AI/ML models to forecast PnL or equity using techniques like ARIMA or Prophet; perform anomaly detection in trades.",
        "📋 Data View Page": "Inspect the processed trading data after your column mapping and filtering. This page also typically allows you to download the currently viewed dataset.",
        "📝 Trade Notes Page": "Review and search through any trade notes or comments you might have included in your journal (requires a mapped 'notes' column)."
    }
    
    cols_pages = st.columns(2)
    col_idx = 0
    for i, (page_title, desc) in enumerate(page_descriptions.items()):
        with cols_pages[col_idx % 2]:
            with st.expander(page_title, expanded=False):
                st.markdown(desc)
        col_idx +=1


    # --- Section 3: Understanding Key Performance Indicators (KPIs) ---
    st.markdown("<h2 class='section-header'>📊 3. Understanding Key Performance Indicators (KPIs)</h2>", unsafe_allow_html=True)
    st.markdown("The dashboard uses several KPIs to quantify trading performance. Here are explanations for common ones (calculated from your mapped data):")

    if KPI_CONFIG:
        # Create two columns for KPIs
        kpi_cols = st.columns(2)
        kpi_idx = 0
        for kpi_key, kpi_info in KPI_CONFIG.items():
            kpi_name = kpi_info.get("name", kpi_key.replace("_", " ").title())
            description = kpi_info.get("description", f"Measures the {kpi_name.lower()}.")
            unit = kpi_info.get("unit", "")
            
            # Alternate KPIs between columns
            current_col = kpi_cols[kpi_idx % 2]
            with current_col:
                with st.expander(f"{kpi_name} {f'({unit})' if unit else ''}", expanded=False):
                    st.markdown(f"**{kpi_name}**: {description}")
                    # Add more details if available in kpi_info, e.g., formula, interpretation
                    if "formula" in kpi_info:
                        st.markdown(f"<p style='font-size:0.9em; color: var(--text-muted-color);'><em>Formula: {kpi_info['formula']}</em></p>", unsafe_allow_html=True)
            kpi_idx += 1
    else:
        st.markdown("""
        <div class="custom-message-box info">
            <span class="custom-message-box-icon">ℹ️</span>
            <span>Detailed KPI explanations will be populated based on application configuration. Common KPIs include Total PnL, Win Rate, Sharpe Ratio, Max Drawdown, etc.</span>
        </div>
        """, unsafe_allow_html=True)

    # --- Section 4: Troubleshooting & FAQ ---
    st.markdown("<h2 class='section-header'>🛠️ 4. Troubleshooting & FAQ</h2>", unsafe_allow_html=True)
    
    with st.expander("❓ Column Mapping Issues", expanded=True):
        st.markdown("""
        * **Mapper Not Appearing:** Ensure your uploaded file is a valid CSV. The mapper typically appears after the file headers are successfully read.
        * **Incorrect Auto-Mapping:** Always review auto-suggestions. Use the dropdowns to correct any mismatches. The data preview helps confirm.
        * **Critical Fields Error:** If you see an error like "Critical fields not mapped," ensure you've selected a CSV column for all fields marked with `*` (e.g., Date, PnL).
        * **Duplicate Critical Mapping Error:** Each critical application field (like 'Date', 'PnL') must be mapped to a *unique* column from your CSV. You cannot map the same CSV column to two different critical fields.
        * **Type Mismatch Warnings (⚠️):** This icon indicates the data in your selected CSV column might not be the type expected by the application (e.g., text where a number is needed for PnL).
            * **Action:** Double-check you've selected the correct CSV column. If correct, the issue might be with the data in your CSV file (e.g., text entries in a PnL column, incorrect date formats). You may need to clean your CSV or ensure date formats are consistent (e.g., YYYY-MM-DD HH:MM:SS).
        * **Error after Confirming Mapping:** If an error occurs during data processing *after* you've confirmed the mapping, the error message will now try to indicate which of *your original CSV columns* (that you mapped) caused the problem. This helps pinpoint issues in your source file.
        """)

    with st.expander("📂 Data Upload Issues (General)", expanded=False):
        st.markdown("""
        * **Error reading CSV:** Ensure your file is a valid CSV. Check for encoding issues (UTF-8 is recommended). Special characters or inconsistent delimiters can also cause problems.
        * **Large File Timeouts:** For very large files, the upload or processing might take time. If it times out, consider splitting the file or ensuring your internet connection is stable.
        """)

    with st.expander("📉 Analysis Not Appearing or Incorrect", expanded=False):
        st.markdown("""
        * **No data after filtering:** If you apply very restrictive filters, there might be no trades matching the criteria. Try adjusting or resetting the filters.
        * **Missing Optional Columns:** Some advanced analyses or specific charts rely on optional data fields (e.g., 'trade notes', 'risk amount', specific strategy tags). If you haven't mapped these columns from your CSV, the corresponding features might be disabled or show a message indicating missing data. This is normal if your CSV doesn't contain that specific information.
        * **Insufficient data for specific analyses:** Some statistical analyses or models require a minimum number of data points to produce meaningful results.
        * **Date Format Issues:** Ensure your date/time column is consistently formatted and correctly parsed. Ambiguous date formats can lead to incorrect chronological ordering or filtering.
        """)

    with st.expander("📊 Understanding 'N/A' or 'Inf' in KPIs", expanded=False):
        st.markdown("""
        * **N/A (Not Available) / NaN (Not a Number):** Typically means the KPI could not be calculated due to insufficient data, missing required input data (e.g., risk-free rate for Sharpe Ratio), or division by zero where the denominator is legitimately zero (e.g., zero trades).
        * **Inf (Infinity):** Can occur in ratios like Profit Factor if Gross Loss is zero, or if a denominator in a calculation is zero when it shouldn't be.
        """)
    
    st.markdown("---")
    st.markdown(f"""
    <div class='custom-card' style='margin-top: 2rem; background-color: var(--section-background-color);'>
        <p>We hope this guide helps you make the most of the <strong>{APP_TITLE}</strong>!
        If you have further questions or encounter issues not covered here, please refer to any additional support channels provided for the application.</p>
    </div>
    """.format(APP_TITLE=APP_TITLE), unsafe_allow_html=True)

if __name__ == "__main__":
    # This check is useful if running the page directly,
    # but in a multi-page app, st.session_state might be initialized by the main app.
    # if 'app_initialized' not in st.session_state:
    #     # st.warning("This page is part of a multi-page app. For full functionality, please run the main `app.py` script.")
    #     # For standalone testing of the guide, you might initialize critical session state vars here if needed.
    #     pass
    show_user_guide_page()
