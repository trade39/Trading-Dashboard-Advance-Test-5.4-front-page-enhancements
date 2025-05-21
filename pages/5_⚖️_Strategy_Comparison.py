"""
pages/5_⚖️_Strategy_Comparison.py

This page allows users to compare the performance of different trading strategies
side-by-side, based on various KPIs and visualizations.
"""
import streamlit as st
import pandas as pd
import logging
import plotly.graph_objects as go

# --- Assuming root-level modules are accessible ---
try:
    from config import APP_TITLE, EXPECTED_COLUMNS, DEFAULT_KPI_DISPLAY_ORDER, COLORS
    from utils.common_utils import display_custom_message, format_currency, format_percentage
    from services.analysis_service import AnalysisService
    from plotting import _apply_custom_theme # Assuming this handles plotly theme application
except ImportError as e:
    st.error(f"Strategy Comparison Page Error: Critical module import failed: {e}. Ensure app structure is correct.")
    # Fallback values for critical configurations if imports fail
    APP_TITLE = "TradingDashboard_Error"
    EXPECTED_COLUMNS = {"strategy": "strategy_fallback", "date": "date_fallback", "pnl": "pnl_fallback"}
    DEFAULT_KPI_DISPLAY_ORDER = []
    COLORS = {} # Define a fallback for COLORS
    # Attempt to get a logger, or set up a basic one if setup_logger itself failed
    try:
        logger = logging.getLogger(APP_TITLE if APP_TITLE != "TradingDashboard_Error" else __name__)
    except Exception:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
    logger.error(f"CRITICAL IMPORT ERROR in Strategy Comparison Page: {e}", exc_info=True)
    st.stop()

logger = logging.getLogger(APP_TITLE)
analysis_service = AnalysisService() # Instantiate service

def show_strategy_comparison_page():
    """
    Renders the content for the Strategy Comparison page.
    """
    st.title("⚖️ Strategy Performance Comparison")
    st.markdown("<p class='page-subtitle'>Analyze and compare the performance of multiple trading strategies side-by-side using key metrics and visualizations.</p>", unsafe_allow_html=True)
    logger.info("Rendering Strategy Comparison Page.")

    # --- Check for necessary data in session state ---
    if 'filtered_data' not in st.session_state or st.session_state.filtered_data is None:
        display_custom_message("Please upload and process data via the sidebar to enable strategy comparison.", "info")
        logger.info("StrategyComparisonPage: No filtered_data in session_state.")
        return

    filtered_df = st.session_state.filtered_data
    plot_theme_name = st.session_state.get('current_theme', 'dark') # e.g., 'dark' or 'light'
    risk_free_rate = st.session_state.get('risk_free_rate', 0.02)
    
    logger.debug(f"StrategyComparisonPage: Columns in st.session_state.filtered_data: {filtered_df.columns.tolist()}")
    strategy_col_from_config = EXPECTED_COLUMNS.get('strategy')
    logger.debug(f"StrategyComparisonPage: Expected strategy column from config: '{strategy_col_from_config}'")

    if filtered_df.empty:
        display_custom_message("No data matches the current filters. Cannot perform strategy comparison.", "info")
        logger.info("StrategyComparisonPage: filtered_df is empty.")
        return

    actual_strategy_col_name = strategy_col_from_config 

    if not actual_strategy_col_name or actual_strategy_col_name not in filtered_df.columns:
        err_msg = (
            f"The configured strategy column ('{actual_strategy_col_name}') was not found in your data. "
            f"Please ensure your data includes this column or update the column mapping in `config.py`. "
            f"Available columns: {filtered_df.columns.tolist()}"
        )
        display_custom_message(err_msg, "warning")
        logger.warning(err_msg)
        return

    # --- Strategy Selection in a styled container ---
    with st.container():
        st.markdown("<div class='input-section-container'>", unsafe_allow_html=True) # Custom class for styling
        st.subheader("Select Strategies to Compare")
        try:
            available_strategies = sorted(filtered_df[actual_strategy_col_name].astype(str).dropna().unique())
            
            if not available_strategies:
                display_custom_message("No distinct strategies found in the data to compare.", "info")
                logger.info("StrategyComparisonPage: No distinct strategies found.")
                st.markdown("</div>", unsafe_allow_html=True)
                return
            
            default_selection = available_strategies[:min(2, len(available_strategies))] # Ensure at least one, up to two

            selected_strategies = st.multiselect(
                "Choose strategies (select two or more for meaningful comparison):",
                options=available_strategies,
                default=default_selection,
                key="strategy_comp_select_v3" # Changed key to avoid conflicts if old one persists
            )

            if not selected_strategies:
                display_custom_message("Please select at least one strategy.", "info")
                st.markdown("</div>", unsafe_allow_html=True)
                return
            
            if len(selected_strategies) == 1:
                st.info(f"Displaying performance for strategy: **{selected_strategies[0]}**. Select additional strategies for a side-by-side comparison.", icon="ℹ️")

        except Exception as e:
            logger.error(f"Error during strategy selection setup using column '{actual_strategy_col_name}': {e}", exc_info=True)
            display_custom_message(f"An error occurred setting up strategy selection: {e}", "error")
            st.markdown("</div>", unsafe_allow_html=True)
            return
        st.markdown("</div>", unsafe_allow_html=True) # Close input-section-container

    # Proceed only if strategies are selected
    if not selected_strategies:
        return

    # --- Comparative KPI Table in a styled container ---
    st.markdown("---") # Visual separator
    with st.container():
        st.markdown("<div class='comparison-section-container'>", unsafe_allow_html=True) # Custom class
        st.subheader("Key Performance Indicator Comparison")
        comparison_kpi_data = []
        try:
            for strat_name in selected_strategies:
                strat_df = filtered_df[filtered_df[actual_strategy_col_name].astype(str) == str(strat_name)]
                if not strat_df.empty:
                    # Pass benchmark_daily_returns=None as it's not directly used for strategy-specific core KPIs here
                    # unless your get_core_kpis is designed to fetch/use it internally based on strat_df dates
                    kpis = analysis_service.get_core_kpis(strat_df, risk_free_rate, benchmark_daily_returns=None, initial_capital=st.session_state.get('initial_capital', 100000))
                    if kpis and 'error' not in kpis:
                        comparison_kpi_data.append({"Strategy": strat_name, **kpis})
                    else:
                        logger.warning(f"Could not calculate KPIs for strategy '{strat_name}'. Error: {kpis.get('error') if kpis else 'Unknown'}")
                else:
                    logger.info(f"No data found for strategy '{strat_name}' within the current filters.")
            
            if comparison_kpi_data:
                comp_df = pd.DataFrame(comparison_kpi_data).set_index("Strategy")
                
                kpis_to_show_in_table = [
                    kpi for kpi in DEFAULT_KPI_DISPLAY_ORDER
                    if kpi in comp_df.columns and kpi not in ['trading_days', 'risk_free_rate_used', 'cumulative_pnl_series', 'daily_returns_series'] # Exclude series data
                ]
                if not kpis_to_show_in_table and comp_df.columns.any(): 
                    kpis_to_show_in_table = [col for col in comp_df.columns if col not in ['cumulative_pnl_series', 'daily_returns_series']]


                if kpis_to_show_in_table:
                    # Define highlight properties based on theme for better readability
                    highlight_props_light_bg = 'color: #000000; font-weight: bold;' # Black text for light green/red
                    highlight_props_dark_bg = 'color: #FFFFFF; font-weight: bold;'  # White text for dark green/red (if using darker highlight colors)
                    
                    # For now, let's use a simpler approach that works okay on both themes
                    # Or, you can pass theme-dependent props if your CSS handles highlight colors well
                    # Using a light text color for dark highlights, dark text for light highlights
                    text_color_for_max = '#006400' # Dark green text
                    text_color_for_min = '#8B0000' # Dark red text
                    
                    # Using CSS classes for highlighting might be more robust for theming
                    st.dataframe(
                        comp_df[kpis_to_show_in_table].style.format("{:,.2f}", na_rep="-")
                        .highlight_max(axis=0, props=f'color: {text_color_for_max}; background-color: #d4edda; font-weight: bold;') # Light green bg
                        .highlight_min(axis=0, props=f'color: {text_color_for_min}; background-color: #f8d7da; font-weight: bold;') # Light red bg
                        , use_container_width=True
                    )
                else:
                    display_custom_message("No common KPIs found to display for selected strategies.", "warning")
            elif selected_strategies: 
                display_custom_message(f"No performance data could be calculated for the selected strategies: {', '.join(selected_strategies)}.", "warning")

        except Exception as e:
            logger.error(f"Error generating KPI comparison table: {e}", exc_info=True)
            display_custom_message(f"An error occurred generating the KPI comparison: {e}", "error")
        st.markdown("</div>", unsafe_allow_html=True) # Close comparison-section-container

    # --- Comparative Visualizations (Example: Equity Curves) ---
    st.markdown("---") # Visual separator
    with st.container():
        st.markdown("<div class='comparison-section-container'>", unsafe_allow_html=True) # Custom class
        st.subheader("Comparative Equity Curves")
        if len(selected_strategies) > 0 :
            equity_comp_fig = go.Figure()
            has_data_for_plot = False
            
            date_col_name = EXPECTED_COLUMNS.get('date')
            pnl_col_name = EXPECTED_COLUMNS.get('pnl')

            if not date_col_name or not pnl_col_name:
                display_custom_message("Date or PnL column configuration is missing. Cannot plot equity curves.", "error")
                logger.error("StrategyComparisonPage: Date or PnL column not found in EXPECTED_COLUMNS for equity plot.")
            else:
                for i, strat_name in enumerate(selected_strategies):
                    strat_df = filtered_df[filtered_df[actual_strategy_col_name].astype(str) == str(strat_name)].copy() # Use .copy()
                    
                    if not strat_df.empty and date_col_name in strat_df.columns and pnl_col_name in strat_df.columns:
                        try:
                            strat_df[date_col_name] = pd.to_datetime(strat_df[date_col_name], errors='coerce')
                            strat_df.dropna(subset=[date_col_name], inplace=True)
                            strat_df = strat_df.sort_values(by=date_col_name)
                            
                            if pd.api.types.is_numeric_dtype(strat_df[pnl_col_name]):
                                strat_df['cumulative_pnl'] = strat_df[pnl_col_name].cumsum()
                                equity_comp_fig.add_trace(go.Scatter(
                                    x=strat_df[date_col_name],
                                    y=strat_df['cumulative_pnl'],
                                    mode='lines',
                                    name=strat_name,
                                    line=dict(color=COLORS.get(f"plot_line_{i % len(COLORS)}", None)) # Cycle through COLORS
                                ))
                                has_data_for_plot = True
                            else:
                                logger.warning(f"PnL column '{pnl_col_name}' for strategy '{strat_name}' is not numeric. Skipping for equity curve.")
                        except Exception as plot_err:
                             logger.error(f"Error processing data for equity curve for strategy '{strat_name}': {plot_err}", exc_info=True)
                    else:
                        logger.info(f"No data or missing PnL/Date columns for strategy '{strat_name}' for equity plot.")
                
                if has_data_for_plot:
                    equity_comp_fig.update_layout(
                        title_text="Equity Curve Comparison by Strategy",
                        xaxis_title="Date",
                        yaxis_title="Cumulative PnL",
                        hovermode="x unified",
                        legend_title_text='Strategy'
                    )
                    # Apply custom theme for Plotly chart
                    themed_fig = _apply_custom_theme(equity_comp_fig, theme_name=plot_theme_name) # Pass theme_name
                    st.plotly_chart(themed_fig, use_container_width=True)
                elif selected_strategies:
                    display_custom_message("Not enough valid data to plot comparative equity curves for the selected strategies.", "info")
        st.markdown("</div>", unsafe_allow_html=True) # Close comparison-section-container

# --- Main execution for the page ---
if __name__ == "__main__":
    # This is primarily for direct execution testing, not for multipage app context
    # In a multipage app, Streamlit handles session state initialization.
    # However, for robust direct testing, ensure critical session state items are present.
    if 'app_initialized' not in st.session_state:
        st.session_state.app_initialized = True # Simulate app initialization
        st.session_state.filtered_data = None # Or load sample data
        st.session_state.current_theme = 'dark'
        st.session_state.risk_free_rate = 0.02
        # Add other necessary session state defaults for this page if testing directly
        logger.info("Simulating app initialization for direct page run.")

    if not st.session_state.get('filtered_data'):
         # Create a dummy DataFrame for testing if no data is loaded
        logger.info("Creating dummy data for Strategy Comparison direct run.")
        num_rows = 100
        start_date = pd.to_datetime('2023-01-01')
        dummy_data = {
            EXPECTED_COLUMNS.get('date', 'date_fallback'): [start_date + pd.Timedelta(days=i) for i in range(num_rows)],
            EXPECTED_COLUMNS.get('pnl', 'pnl_fallback'): [(-1)**i * i * 10.5 for i in range(num_rows)],
            EXPECTED_COLUMNS.get('strategy', 'strategy_fallback'): ['StrategyA'] * (num_rows // 2) + ['StrategyB'] * (num_rows - num_rows // 2)
        }
        st.session_state.filtered_data = pd.DataFrame(dummy_data)
        st.session_state.filtered_data[EXPECTED_COLUMNS.get('date', 'date_fallback')] = pd.to_datetime(st.session_state.filtered_data[EXPECTED_COLUMNS.get('date', 'date_fallback')])


    show_strategy_comparison_page()
