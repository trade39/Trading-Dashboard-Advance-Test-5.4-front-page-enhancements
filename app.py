# app.py - Main Entry Point for Multi-Page Trading Performance Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import logging # Keep standard logging
import sys
import os
import datetime
import base64
from io import BytesIO

# --- MODIFICATION START: st.set_page_config() moved to the top ---
from config import APP_TITLE as PAGE_CONFIG_APP_TITLE # This is "Trading Mastery Hub"
LOGO_PATH_FOR_BROWSER_TAB = "assets/Trading_Mastery_Hub_600x600.png"

st.set_page_config(
    page_title=PAGE_CONFIG_APP_TITLE, # Browser tab title remains "Trading Mastery Hub"
    page_icon=LOGO_PATH_FOR_BROWSER_TAB,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/trade39/Trading-Dashboard-Advance-Test-5.4',
        'Report a bug': "https://github.com/trade39/Trading-Dashboard-Advance-Test-5.4/issues",
        'About': f"## {PAGE_CONFIG_APP_TITLE}\n\nA comprehensive dashboard for trading performance analysis."
    }
)
# --- MODIFICATION END ---

# --- Utility Modules ---
try:
    from utils.logger import setup_logger
    from utils.common_utils import load_css, display_custom_message, log_execution_time
except ImportError as e:
    st.error(f"Fatal Error: Could not import utility modules. App cannot start. Details: {e}")
    logging.basicConfig(level=logging.ERROR)
    logging.error(f"Fatal Error importing utils: {e}", exc_info=True)
    st.stop()

# --- Component Modules ---
try:
    from components.sidebar_manager import SidebarManager
    from components.column_mapper_ui import ColumnMapperUI
    from components.scroll_buttons import ScrollButtons
except ImportError as e:
    st.error(f"Fatal Error: Could not import component modules. App cannot start. Details: {e}")
    logging.error(f"Fatal Error importing components: {e}", exc_info=True)
    st.stop()

# --- Service Modules ---
try:
    from services.data_service import DataService, get_benchmark_data_static
    from services.analysis_service import AnalysisService
except ImportError as e:
    st.error(f"Fatal Error: Could not import service modules. App cannot start. Details: {e}")
    logging.error(f"Fatal Error importing services: {e}", exc_info=True)
    st.stop()

# --- Core Application Modules (Configs) ---
try:
    from config import (
        APP_TITLE, CONCEPTUAL_COLUMNS, CRITICAL_CONCEPTUAL_COLUMNS,
        CONCEPTUAL_COLUMN_TYPES, CONCEPTUAL_COLUMN_SYNONYMS,
        CONCEPTUAL_COLUMN_CATEGORIES,
        RISK_FREE_RATE, LOG_FILE, LOG_LEVEL, LOG_FORMAT,
        DEFAULT_BENCHMARK_TICKER, AVAILABLE_BENCHMARKS, EXPECTED_COLUMNS
    )
    from kpi_definitions import KPI_CONFIG
except ImportError as e:
    st.error(f"Fatal Error: Could not import configuration (config.py or kpi_definitions.py). App cannot start. Details: {e}")
    APP_TITLE = "TradingAppError"; LOG_FILE = "logs/error_app.log"; LOG_LEVEL = "ERROR"; LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    RISK_FREE_RATE = 0.02; CONCEPTUAL_COLUMNS = {"date": "Date", "pnl": "PnL"}; CRITICAL_CONCEPTUAL_COLUMNS = ["date", "pnl"]
    CONCEPTUAL_COLUMN_TYPES = {}; CONCEPTUAL_COLUMN_SYNONYMS = {}; KPI_CONFIG = {}; CONCEPTUAL_COLUMN_CATEGORIES = {}
    EXPECTED_COLUMNS = {"date": "date", "pnl": "pnl"}; DEFAULT_BENCHMARK_TICKER = "SPY"; AVAILABLE_BENCHMARKS = {}
    st.stop()

logger = setup_logger(
    logger_name=APP_TITLE, log_file=LOG_FILE, level=LOG_LEVEL, log_format=LOG_FORMAT
)
logger.info(f"Application '{APP_TITLE}' starting. Logger initialized.")

# Theme management
if 'current_theme' not in st.session_state:
    st.session_state.current_theme = "dark"
theme_js = f"""
<script>
    const currentTheme = '{st.session_state.current_theme}';
    document.documentElement.setAttribute('data-theme', currentTheme);
    if (currentTheme === "dark") {{
        document.body.classList.add('dark-mode');
        document.body.classList.remove('light-mode');
    }} else {{
        document.body.classList.add('light-mode');
        document.body.classList.remove('dark-mode');
    }}
</script>
"""
st.components.v1.html(theme_js, height=0)


# Load CSS
try:
    css_file_path = "style.css"
    if os.path.exists(css_file_path): load_css(css_file_path)
    else: logger.error(f"style.css not found at '{css_file_path}'.")
except Exception as e: logger.error(f"Failed to load style.css: {e}", exc_info=True)

# Initialize session state
default_session_state = {
    'app_initialized': True, 'processed_data': None, 'filtered_data': None,
    'kpi_results': None, 'kpi_confidence_intervals': {},
    'risk_free_rate': RISK_FREE_RATE, 'uploaded_file_name': None,
    'uploaded_file_bytes_for_mapper': None, 'last_processed_file_id': None,
    'user_column_mapping': None, 'column_mapping_confirmed': False,
    'csv_headers_for_mapping': None, 'last_uploaded_file_for_mapping_id': None,
    'last_applied_filters': None, 'sidebar_filters': None, 'active_tab': "📈 Overview",
    'selected_benchmark_ticker': DEFAULT_BENCHMARK_TICKER,
    'selected_benchmark_display_name': next((n for n, t in AVAILABLE_BENCHMARKS.items() if t == DEFAULT_BENCHMARK_TICKER), "None"),
    'benchmark_daily_returns': None, 'initial_capital': 100000.0,
    'last_fetched_benchmark_ticker': None, 'last_benchmark_data_filter_shape': None,
    'last_kpi_calc_state_id': None,
    'max_drawdown_period_details': None
}
for key, value in default_session_state.items():
    if key not in st.session_state: st.session_state[key] = value

data_service = DataService()
analysis_service_instance = AnalysisService()

# Logo
LOGO_PATH_SIDEBAR = "assets/Trading_Mastery_Hub_600x600.png"
logo_to_display_path = LOGO_PATH_SIDEBAR
logo_base64 = None

if os.path.exists(LOGO_PATH_SIDEBAR):
    try:
        with open(LOGO_PATH_SIDEBAR, "rb") as image_file:
            logo_base64 = base64.b64encode(image_file.read()).decode()
        logo_to_display_for_st_logo = f"data:image/png;base64,{logo_base64}"
    except Exception as e:
        logger.error(f"Error encoding logo: {e}", exc_info=True)
        logo_to_display_for_st_logo = LOGO_PATH_SIDEBAR
else:
    logger.error(f"Logo file NOT FOUND at {LOGO_PATH_SIDEBAR}")
    logo_to_display_for_st_logo = None


if logo_to_display_for_st_logo:
    try:
        st.logo(logo_to_display_for_st_logo, icon_image=logo_to_display_for_st_logo)
    except Exception as e:
        logger.error(f"Error setting st.logo: {e}", exc_info=True)
        if logo_base64:
             st.sidebar.image(f"data:image/png;base64,{logo_base64}", use_column_width='auto')
        elif os.path.exists(LOGO_PATH_SIDEBAR):
             st.sidebar.image(LOGO_PATH_SIDEBAR, use_column_width='auto')


st.sidebar.header(APP_TITLE) # Sidebar header remains "Trading Mastery Hub"
st.sidebar.markdown("---")
theme_toggle_value = st.session_state.current_theme == "light"

toggle_label = "Switch to Dark Mode" if st.session_state.current_theme == "light" else "Switch to Light Mode"

if st.sidebar.button(toggle_label, key="theme_toggle_button_main_app"):
    st.session_state.current_theme = "dark" if st.session_state.current_theme == "light" else "light"
    st.rerun()
st.sidebar.markdown("---")


uploaded_file = st.sidebar.file_uploader("Upload Trading Journal (CSV)", type=["csv"], key="app_wide_file_uploader")

sidebar_manager = SidebarManager(st.session_state.get('processed_data'))
current_sidebar_filters = sidebar_manager.render_sidebar_controls()
st.session_state.sidebar_filters = current_sidebar_filters

if current_sidebar_filters:
    rfr_from_sidebar = current_sidebar_filters.get('risk_free_rate', RISK_FREE_RATE)
    if st.session_state.risk_free_rate != rfr_from_sidebar:
        st.session_state.risk_free_rate = rfr_from_sidebar; st.session_state.kpi_results = None

    benchmark_ticker_from_sidebar = current_sidebar_filters.get('selected_benchmark_ticker', "")
    if st.session_state.selected_benchmark_ticker != benchmark_ticker_from_sidebar:
        st.session_state.selected_benchmark_ticker = benchmark_ticker_from_sidebar
        st.session_state.selected_benchmark_display_name = next((n for n, t in AVAILABLE_BENCHMARKS.items() if t == benchmark_ticker_from_sidebar), "None")
        st.session_state.benchmark_daily_returns = None; st.session_state.kpi_results = None

    initial_capital_from_sidebar = current_sidebar_filters.get('initial_capital', 100000.0)
    if st.session_state.initial_capital != initial_capital_from_sidebar:
        st.session_state.initial_capital = initial_capital_from_sidebar; st.session_state.kpi_results = None

@log_execution_time
def get_and_process_data_with_profiling(file_obj, mapping, name):
    return data_service.get_processed_trading_data(file_obj, user_column_mapping=mapping, original_file_name=name)

if uploaded_file is not None:
    current_file_id_for_mapping = f"{uploaded_file.name}-{uploaded_file.size}-{uploaded_file.type}-mapping_stage"
    if st.session_state.last_uploaded_file_for_mapping_id != current_file_id_for_mapping:
        logger.info(f"New file '{uploaded_file.name}' for mapping. Resetting state.")
        for key_to_reset in ['column_mapping_confirmed', 'user_column_mapping', 'processed_data', 'filtered_data', 'kpi_results', 'kpi_confidence_intervals', 'benchmark_daily_returns', 'max_drawdown_period_details']:
            st.session_state[key_to_reset] = None
        st.session_state.uploaded_file_name = uploaded_file.name
        st.session_state.last_uploaded_file_for_mapping_id = current_file_id_for_mapping
        try:
            st.session_state.uploaded_file_bytes_for_mapper = BytesIO(uploaded_file.getvalue())
            st.session_state.uploaded_file_bytes_for_mapper.seek(0)
            df_peek = pd.read_csv(BytesIO(st.session_state.uploaded_file_bytes_for_mapper.getvalue()), nrows=5)
            st.session_state.csv_headers_for_mapping = df_peek.columns.tolist()
            st.session_state.uploaded_file_bytes_for_mapper.seek(0)
        except Exception as e_header:
            logger.error(f"Could not read CSV headers/preview: {e_header}", exc_info=True)
            display_custom_message(f"Error reading from '{uploaded_file.name}': {e_header}. Ensure valid CSV.", "error")
            st.session_state.csv_headers_for_mapping = None; st.session_state.uploaded_file_bytes_for_mapper = None; st.stop()

    if st.session_state.csv_headers_for_mapping and not st.session_state.column_mapping_confirmed:
        st.session_state.processed_data = None; st.session_state.filtered_data = None
        column_mapper = ColumnMapperUI(
            uploaded_file_name=st.session_state.uploaded_file_name,
            uploaded_file_bytes=st.session_state.uploaded_file_bytes_for_mapper,
            csv_headers=st.session_state.csv_headers_for_mapping,
            conceptual_columns_map=CONCEPTUAL_COLUMNS,
            conceptual_column_types=CONCEPTUAL_COLUMN_TYPES,
            conceptual_column_synonyms=CONCEPTUAL_COLUMN_SYNONYMS,
            critical_conceptual_cols=CRITICAL_CONCEPTUAL_COLUMNS,
            conceptual_column_categories=CONCEPTUAL_COLUMN_CATEGORIES
        )
        user_mapping_result = column_mapper.render()
        if user_mapping_result is not None:
            st.session_state.user_column_mapping = user_mapping_result
            st.session_state.column_mapping_confirmed = True
            st.rerun()
        else:
            st.stop()

    if st.session_state.column_mapping_confirmed and st.session_state.user_column_mapping:
        current_file_id_proc = f"{st.session_state.uploaded_file_name}-{uploaded_file.size}-{uploaded_file.type}-processing"
        if st.session_state.last_processed_file_id != current_file_id_proc or st.session_state.processed_data is None:
            with st.spinner(f"Processing '{st.session_state.uploaded_file_name}'..."):
                file_obj_for_service = st.session_state.uploaded_file_bytes_for_mapper
                if file_obj_for_service:
                    file_obj_for_service.seek(0)
                    st.session_state.processed_data = get_and_process_data_with_profiling(
                        file_obj_for_service, st.session_state.user_column_mapping, st.session_state.uploaded_file_name
                    )
                else:
                    logger.warning("uploaded_file_bytes_for_mapper was None, attempting to re-read from uploaded_file.")
                    temp_bytes = BytesIO(uploaded_file.getvalue())
                    st.session_state.processed_data = get_and_process_data_with_profiling(
                        temp_bytes, st.session_state.user_column_mapping, st.session_state.uploaded_file_name
                    )

            st.session_state.last_processed_file_id = current_file_id_proc
            for key_to_reset in ['kpi_results', 'kpi_confidence_intervals', 'benchmark_daily_returns', 'max_drawdown_period_details', 'filtered_data']:
                st.session_state[key_to_reset] = None
            st.session_state.filtered_data = st.session_state.processed_data

            if st.session_state.processed_data is not None:
                display_custom_message(f"Successfully processed '{st.session_state.uploaded_file_name}'.", "success", icon="✅")
            else:
                display_custom_message(f"Failed to process '{st.session_state.uploaded_file_name}'. Check logs and column mapping.", "error")
                st.session_state.column_mapping_confirmed = False
                st.session_state.user_column_mapping = None
elif st.session_state.get('uploaded_file_name') and uploaded_file is None:
    if st.session_state.processed_data is not None:
        logger.info("File uploader is empty. Resetting all data-dependent session states.")
        keys_to_reset_on_file_removal = [
            'processed_data', 'filtered_data', 'kpi_results', 'kpi_confidence_intervals',
            'uploaded_file_name', 'uploaded_file_bytes_for_mapper', 'last_processed_file_id',
            'user_column_mapping', 'column_mapping_confirmed', 'csv_headers_for_mapping',
            'last_uploaded_file_for_mapping_id', 'last_applied_filters', 'sidebar_filters',
            'benchmark_daily_returns', 'last_fetched_benchmark_ticker',
            'last_benchmark_data_filter_shape', 'last_kpi_calc_state_id',
            'max_drawdown_period_details'
        ]
        for key_to_reset in keys_to_reset_on_file_removal:
            if key_to_reset in default_session_state:
                 st.session_state[key_to_reset] = default_session_state[key_to_reset]
            else:
                 st.session_state[key_to_reset] = None
        st.rerun()


@log_execution_time
def filter_data_with_profiling(df, filters, col_map):
    return data_service.filter_data(df, filters, col_map)

@log_execution_time
def get_core_kpis_with_profiling(df, rfr, benchmark_returns, capital):
    return analysis_service_instance.get_core_kpis(df, rfr, benchmark_returns, capital)

@log_execution_time
def get_advanced_drawdown_analysis_with_profiling(equity_series):
    return analysis_service_instance.get_advanced_drawdown_analysis(equity_series)


if st.session_state.processed_data is not None and st.session_state.sidebar_filters:
    if st.session_state.filtered_data is None or st.session_state.last_applied_filters != st.session_state.sidebar_filters:
        with st.spinner("Applying filters..."):
            st.session_state.filtered_data = filter_data_with_profiling(
                st.session_state.processed_data, st.session_state.sidebar_filters, EXPECTED_COLUMNS
            )
        st.session_state.last_applied_filters = st.session_state.sidebar_filters.copy()
        for key_to_reset in ['kpi_results', 'kpi_confidence_intervals', 'benchmark_daily_returns', 'max_drawdown_period_details']:
            st.session_state[key_to_reset] = None

# --- Benchmark Data Fetching Logic ---
if st.session_state.filtered_data is not None and not st.session_state.filtered_data.empty:
    selected_ticker = st.session_state.get('selected_benchmark_ticker')
    if selected_ticker and selected_ticker != "" and selected_ticker.upper() != "NONE":
        refetch_benchmark = False
        if st.session_state.benchmark_daily_returns is None: refetch_benchmark = True
        elif st.session_state.last_fetched_benchmark_ticker != selected_ticker: refetch_benchmark = True
        elif st.session_state.last_benchmark_data_filter_shape != st.session_state.filtered_data.shape: refetch_benchmark = True

        if refetch_benchmark:
            date_col_conceptual = EXPECTED_COLUMNS.get('date', 'date')
            min_d_str_to_fetch, max_d_str_to_fetch = None, None

            if date_col_conceptual in st.session_state.filtered_data.columns:
                dates_for_bm_filtered = pd.to_datetime(st.session_state.filtered_data[date_col_conceptual], errors='coerce').dropna()
                if not dates_for_bm_filtered.empty:
                    min_d_filtered, max_d_filtered = dates_for_bm_filtered.min(), dates_for_bm_filtered.max()
                    if pd.notna(min_d_filtered) and pd.notna(max_d_filtered) and (max_d_filtered.date() - min_d_filtered.date()).days >= 0:
                        min_d_str_to_fetch, max_d_str_to_fetch = min_d_filtered.strftime('%Y-%m-%d'), max_d_filtered.strftime('%Y-%m-%d')

            if min_d_str_to_fetch and max_d_str_to_fetch:
                with st.spinner(f"Fetching benchmark data for {selected_ticker}..."):
                    st.session_state.benchmark_daily_returns = get_benchmark_data_static(selected_ticker, min_d_str_to_fetch, max_d_str_to_fetch)
                st.session_state.last_fetched_benchmark_ticker = selected_ticker
                st.session_state.last_benchmark_data_filter_shape = st.session_state.filtered_data.shape
                if st.session_state.benchmark_daily_returns is None or st.session_state.benchmark_daily_returns.empty:
                    display_custom_message(f"Could not fetch benchmark data for {selected_ticker} or no data returned for the period.", "warning")
            else:
                logger.warning(f"Cannot fetch benchmark for {selected_ticker} due to invalid/missing date range in filtered data.")
                st.session_state.benchmark_daily_returns = None
            st.session_state.kpi_results = None
    elif st.session_state.benchmark_daily_returns is not None:
        st.session_state.benchmark_daily_returns = None
        st.session_state.kpi_results = None

# --- KPI Calculation Logic ---
if st.session_state.filtered_data is not None and not st.session_state.filtered_data.empty:
    current_kpi_state_id_parts = [
        st.session_state.filtered_data.shape,
        st.session_state.risk_free_rate,
        st.session_state.initial_capital,
        st.session_state.selected_benchmark_ticker
    ]
    if st.session_state.benchmark_daily_returns is not None and not st.session_state.benchmark_daily_returns.empty:
        try:
            current_kpi_state_id_parts.append(pd.util.hash_pandas_object(st.session_state.benchmark_daily_returns.sort_index(), index=True).sum())
        except Exception as e_hash:
            logger.warning(f"Hashing benchmark data failed: {e_hash}. Using shape as fallback for KPI state.")
            current_kpi_state_id_parts.append(st.session_state.benchmark_daily_returns.shape)
    else:
        current_kpi_state_id_parts.append(None)
    current_kpi_state_id = tuple(current_kpi_state_id_parts)

    if st.session_state.kpi_results is None or st.session_state.last_kpi_calc_state_id != current_kpi_state_id:
        logger.info("Recalculating KPIs, Confidence Intervals, and Max Drawdown Details...")
        with st.spinner("Calculating performance metrics..."):
            kpi_res = get_core_kpis_with_profiling(
                st.session_state.filtered_data,
                st.session_state.risk_free_rate,
                st.session_state.benchmark_daily_returns,
                st.session_state.initial_capital
            )
            if kpi_res and 'error' not in kpi_res:
                st.session_state.kpi_results = kpi_res
                st.session_state.last_kpi_calc_state_id = current_kpi_state_id

                date_col = EXPECTED_COLUMNS.get('date')
                cum_pnl_col = 'cumulative_pnl'
                equity_series_for_dd = pd.Series(dtype=float)
                if date_col and cum_pnl_col and date_col in st.session_state.filtered_data.columns and cum_pnl_col in st.session_state.filtered_data.columns:
                    temp_df_for_equity = st.session_state.filtered_data[[date_col, cum_pnl_col]].copy()
                    temp_df_for_equity[date_col] = pd.to_datetime(temp_df_for_equity[date_col], errors='coerce')
                    temp_df_for_equity.dropna(subset=[date_col], inplace=True)
                    if not temp_df_for_equity.empty:
                        equity_series_for_dd = temp_df_for_equity.set_index(date_col)[cum_pnl_col].sort_index().dropna()

                if not equity_series_for_dd.empty and len(equity_series_for_dd) >= 5:
                    adv_dd_results = get_advanced_drawdown_analysis_with_profiling(equity_series_for_dd)
                    st.session_state.max_drawdown_period_details = adv_dd_results.get('max_drawdown_details') if adv_dd_results and 'error' not in adv_dd_results else None
                    if adv_dd_results and 'error' in adv_dd_results: logger.warning(f"Advanced drawdown error: {adv_dd_results['error']}")
                else: st.session_state.max_drawdown_period_details = None

                pnl_col_for_ci = EXPECTED_COLUMNS.get('pnl')
                if pnl_col_for_ci and pnl_col_for_ci in st.session_state.filtered_data.columns:
                    pnl_series_for_ci = st.session_state.filtered_data[pnl_col_for_ci].dropna()
                    if len(pnl_series_for_ci) >= 10:
                        ci_res = analysis_service_instance.get_bootstrapped_kpi_cis(st.session_state.filtered_data, ['avg_trade_pnl', 'win_rate', 'sharpe_ratio'])
                        st.session_state.kpi_confidence_intervals = ci_res if ci_res and 'error' not in ci_res else {}
                    else: st.session_state.kpi_confidence_intervals = {}
                else: st.session_state.kpi_confidence_intervals = {}
            else:
                error_msg = kpi_res.get('error', 'Unknown error') if kpi_res else 'KPI calculation failed'
                display_custom_message(f"KPI calculation error: {error_msg}", "error")
                st.session_state.kpi_results = None; st.session_state.kpi_confidence_intervals = {}; st.session_state.max_drawdown_period_details = None
elif st.session_state.filtered_data is not None and st.session_state.filtered_data.empty:
    if st.session_state.processed_data is not None and not st.session_state.processed_data.empty:
        display_custom_message("No data matches the current filter criteria. Adjust filters or upload a new file.", "info")
    st.session_state.kpi_results = None; st.session_state.kpi_confidence_intervals = {}; st.session_state.max_drawdown_period_details = None


# --- WELCOME PAGE LAYOUT FUNCTION ---
def main_page_layout():
    """
    Defines and displays the layout for the welcome page.
    """
    st.markdown("<div class='welcome-container'>", unsafe_allow_html=True)
    st.markdown("<div class='hero-section'>", unsafe_allow_html=True)
    
    # --- MODIFICATION START: Updated Title and Added Subtitle ---
    st.markdown("<h1 class='welcome-title'>Trading Dashboard</h1>", unsafe_allow_html=True)
    st.markdown(f"<p class='welcome-subtitle'>Powered by {PAGE_CONFIG_APP_TITLE}</p>", unsafe_allow_html=True) # Use PAGE_CONFIG_APP_TITLE for "Trading Mastery Hub"
    # --- MODIFICATION END ---
    
    st.markdown("<p class='tagline'>Unlock insights from your trading data with powerful analytics and visualizations.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True) # End hero-section
    
    # Removed the st.markdown("---") that creates a horizontal line
    
    st.markdown("<h2 class='features-title' style='text-align: center; color: var(--secondary-color);'>Get Started</h2>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,1,1], gap="large")

    with col1:
        st.markdown("<div class='feature-item'>", unsafe_allow_html=True)
        st.markdown("<h4>📄 Upload Data</h4>", unsafe_allow_html=True)
        st.markdown("<p>Begin by uploading your trade journal (CSV) via the sidebar. Our intelligent mapping assistant will guide you through aligning your columns.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='feature-item'>", unsafe_allow_html=True)
        st.markdown("<h4>📊 Analyze Performance</h4>", unsafe_allow_html=True)
        st.markdown("<p>Dive deep into comprehensive performance metrics, equity curves, and statistical breakdowns once your data is loaded and processed.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='feature-item'>", unsafe_allow_html=True)
        st.markdown("<h4>💡 Discover Insights</h4>", unsafe_allow_html=True)
        st.markdown("<p>Leverage advanced tools like categorical analysis, strategy comparisons, and AI-driven suggestions available in the dashboard pages.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<div style='text-align: center; margin-top: 30px;'>", unsafe_allow_html=True)
    user_guide_page_path = "pages/0_❓_User_Guide.py"
    if os.path.exists(user_guide_page_path):
        if st.button("📘 Read the User Guide", key="welcome_user_guide_button", help="Navigate to the User Guide page"):
            st.switch_page(user_guide_page_path)
    else:
        st.markdown("<p style='text-align: center; font-style: italic;'>User guide page not found.</p>", unsafe_allow_html=True)
        logger.warning(f"User Guide page not found at expected path: {user_guide_page_path}")

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True) # End welcome-container

# --- Page Navigation and Display ---
if not uploaded_file and st.session_state.processed_data is None:
    main_page_layout()
    st.stop()
elif uploaded_file and st.session_state.processed_data is None and not st.session_state.column_mapping_confirmed:
    if st.session_state.csv_headers_for_mapping is None and uploaded_file:
        display_custom_message("Error reading the uploaded file. Please ensure it's a valid CSV and try again.", "error")
        st.stop()
elif st.session_state.processed_data is not None and (st.session_state.filtered_data is None or st.session_state.filtered_data.empty) and not (st.session_state.kpi_results and 'error' not in st.session_state.kpi_results):
    if st.session_state.sidebar_filters and uploaded_file:
        display_custom_message("No data matches the current filter criteria. Please adjust your filters in the sidebar.", "info")
elif st.session_state.processed_data is None and st.session_state.get('uploaded_file_name') and not st.session_state.get('column_mapping_confirmed'):
    pass

scroll_buttons_component = ScrollButtons()
scroll_buttons_component.render()

logger.info(f"App '{APP_TITLE}' run cycle finished.")
