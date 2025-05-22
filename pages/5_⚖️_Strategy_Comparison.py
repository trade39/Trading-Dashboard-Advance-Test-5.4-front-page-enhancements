import streamlit as st
import pandas as pd
import logging
import plotly.graph_objects as go

# --- Imports for app config and utilities ---
try:
    from config import APP_TITLE, EXPECTED_COLUMNS, DEFAULT_KPI_DISPLAY_ORDER, COLORS
    from utils.common_utils import display_custom_message #, format_currency, format_percentage # format_currency and format_percentage not used directly here
    from services.analysis_service import AnalysisService
    from plotting import _apply_custom_theme
except ImportError as e:
    # Fallback for critical import errors to allow the page to at least load an error message
    st.error(f"Strategy Comparison Page Error: Critical module import failed: {e}. Ensure app structure is correct and all dependencies are available.")
    # Define minimal fallbacks so the rest of the script doesn't immediately crash
    APP_TITLE = "TradingDashboard_Error"
    EXPECTED_COLUMNS = {"strategy": "strategy_fallback", "date": "date_fallback", "pnl": "pnl_fallback"}
    DEFAULT_KPI_DISPLAY_ORDER = []
    COLORS = {} # Ensure COLORS is defined, even if empty
    # Define dummy functions if they are essential for the script to parse
    def display_custom_message(message, type): st.text(f"{type.upper()}: {message}")
    class AnalysisService:
        def get_core_kpis(self, df, rate): return {"error": "AnalysisService not loaded"}
    def _apply_custom_theme(fig, theme): return fig # No-op
    # It's generally better to st.stop() after a critical error, but for completeness:
    # st.stop() # Uncomment if you want the app to halt here on critical import failure

# Initialize logger, ensuring APP_TITLE is defined
logger = logging.getLogger(APP_TITLE if 'APP_TITLE' in locals() else "TradingDashboard_Default")
analysis_service = AnalysisService() if 'AnalysisService' in locals() else None # Ensure service is available or None

def get_st_theme():
    """Detect Streamlit theme dynamically."""
    try:
        theme_base = st.get_option("theme.base")
    except Exception:
        theme_base = st.session_state.get("current_theme", "dark") # Fallback
    return theme_base if theme_base in {"dark", "light"} else "dark"

def show_strategy_comparison_page():
    st.title("⚖️ Strategy Performance Comparison")
    st.markdown('<p class="page-subtitle">Easily compare performance, risk, and equity curves between your strategies side-by-side.</p>', unsafe_allow_html=True)
    theme = get_st_theme()

    # --- Data Check Section ---
    # This container is for logic, not necessarily for visual card styling
    with st.container():
        if 'filtered_data' not in st.session_state or st.session_state.filtered_data is None:
            display_custom_message("Please upload and process data in the main application to compare strategies.", "info")
            logger.info("StrategyComparisonPage: No filtered_data in session_state.")
            return

        filtered_df = st.session_state.filtered_data
        risk_free_rate = st.session_state.get('risk_free_rate', 0.02)
        strategy_col_from_config = EXPECTED_COLUMNS.get('strategy')

        if filtered_df.empty:
            display_custom_message("No data matches the current filters. Cannot perform strategy comparison.", "info")
            logger.info("StrategyComparisonPage: filtered_df is empty.")
            return

        actual_strategy_col_name = strategy_col_from_config

        if not actual_strategy_col_name or actual_strategy_col_name not in filtered_df.columns:
            err_msg = (
                f"Strategy column ('{actual_strategy_col_name}') not found in the data. Comparison is not possible. "
                f"Available columns in the filtered data: {filtered_df.columns.tolist()}"
            )
            display_custom_message(err_msg, "warning")
            logger.warning(err_msg)
            return

    # --- Strategy Selection Section ---
    st.subheader("⚙️ Strategy Selection")
    st.markdown('<div class="performance-section-container">', unsafe_allow_html=True)
    try:
        available_strategies = sorted(filtered_df[actual_strategy_col_name].astype(str).dropna().unique())
        if not available_strategies:
            display_custom_message("No distinct strategies found in the data to compare.", "info")
            logger.info("StrategyComparisonPage: No distinct strategies found.")
            st.markdown('</div>', unsafe_allow_html=True) # Close container
            return
        if len(available_strategies) < 2:
            display_custom_message(f"Only one strategy ('{available_strategies[0]}') found. At least two strategies are needed for comparison.", "info")
            logger.info(f"StrategyComparisonPage: Only one strategy found: {available_strategies[0]}")
            # Still allow selection for single strategy view if desired
            # If you strictly require 2+ for this section to proceed, you might return or disable multiselect here.

        default_selection = available_strategies[:2] if len(available_strategies) >= 2 else available_strategies
        selected_strategies = st.multiselect(
            "Choose strategies to compare:",
            options=available_strategies,
            default=default_selection,
            key="strategy_comp_select_v2" # Changed key to ensure it's unique if old version existed
        )

        if not selected_strategies:
            display_custom_message("Please select at least one strategy to view its performance, or two or more to compare.", "info")
            st.markdown('</div>', unsafe_allow_html=True) # Close container
            return

        if len(selected_strategies) == 1:
            st.info(f"Displaying performance for strategy: **{selected_strategies[0]}**. Select more strategies for comparison.")
    except Exception as e:
        logger.error(f"Error during strategy selection setup using column '{actual_strategy_col_name}': {e}", exc_info=True)
        display_custom_message(f"An error occurred setting up strategy selection: {e}", "error")
        st.markdown('</div>', unsafe_allow_html=True) # Close container
        return
    st.markdown('</div>', unsafe_allow_html=True) # Close performance-section-container for Strategy Selection

    # Add a divider for better visual separation if desired
    st.markdown("---")


    # --- KPI Comparison Table Section ---
    st.subheader("📊 Key Performance Indicator Comparison")
    st.markdown('<div class="performance-section-container">', unsafe_allow_html=True)
    comparison_kpi_data = []
    try:
        if not analysis_service: # Check if analysis_service was loaded
            display_custom_message("Analysis service is not available. Cannot calculate KPIs.", "error")
            st.markdown('</div>', unsafe_allow_html=True)
            return

        for strat_name in selected_strategies:
            strat_df = filtered_df[filtered_df[actual_strategy_col_name].astype(str) == str(strat_name)]
            if not strat_df.empty:
                kpis = analysis_service.get_core_kpis(strat_df, risk_free_rate)
                if kpis and 'error' not in kpis:
                    comparison_kpi_data.append({"Strategy": strat_name, **kpis})
                else:
                    error_detail = kpis.get('error') if isinstance(kpis, dict) else 'Unknown or service unavailable'
                    logger.warning(f"Could not calculate KPIs for strategy '{strat_name}'. Error: {error_detail}")
            else:
                logger.info(f"No data found for strategy '{strat_name}' within the current filters.")

        if comparison_kpi_data:
            comp_df = pd.DataFrame(comparison_kpi_data).set_index("Strategy")
            
            # Ensure DEFAULT_KPI_DISPLAY_ORDER is available and is a list
            kpi_order = DEFAULT_KPI_DISPLAY_ORDER if isinstance(DEFAULT_KPI_DISPLAY_ORDER, list) else []

            kpis_to_show_in_table = [
                kpi for kpi in kpi_order # Use the kpi_order
                if kpi in comp_df.columns and kpi not in ['trading_days', 'risk_free_rate_used']
            ]
            if not kpis_to_show_in_table and comp_df.columns.any(): # Fallback if order is empty or doesn't match
                kpis_to_show_in_table = [
                    col for col in comp_df.columns 
                    if col not in ['trading_days', 'risk_free_rate_used']
                ]

            # Define explicit highlight background and text colors based on theme
            if theme == "dark":
                max_bg_color = "#3BA55D"   # Dark theme green background for max values
                min_bg_color = "#FF776B"   # Dark theme red background for min values
                highlight_text_color = "#F0F1F6" # Light text for dark theme highlights (contrasts with bg)
            else: # Light theme
                max_bg_color = "#B2F2BB" # Light theme green background for max values
                min_bg_color = "#FFD6D6" # Light theme red background for min values
                highlight_text_color = "#121416" # Dark text for light theme highlights (contrasts with bg)
            

            if kpis_to_show_in_table:
                styled = (
                    comp_df[kpis_to_show_in_table]
                    .style
                    .format("{:,.2f}", na_rep="-")
                    .highlight_max(axis=0, props=f"background-color: {max_bg_color}; color: {highlight_text_color}; font-weight:bold;")
                    .highlight_min(axis=0, props=f"background-color: {min_bg_color}; color: {highlight_text_color}; font-weight:bold;")
                )
                # The st.dataframe should inherit text color for non-highlighted cells from style.css
                # For highlighted cells, the 'color' in props above will take precedence.
                st.dataframe(styled, use_container_width=True)
            else:
                display_custom_message("No common KPIs found to display for selected strategies.", "warning")
        elif selected_strategies: # Only show this if strategies were selected but no data was processed
            display_custom_message(
                f"No performance data could be calculated for the selected strategies: {', '.join(selected_strategies)}.",
                "warning"
            )
    except Exception as e:
        logger.error(f"Error generating KPI comparison table: {e}", exc_info=True)
        display_custom_message(f"An error occurred generating the KPI comparison: {e}", "error")
    st.markdown('</div>', unsafe_allow_html=True) # Close performance-section-container for KPI table

    # Add another divider
    st.markdown("---")

    # --- Comparative Visualizations Section ---
    st.subheader("📈 Comparative Equity Curves")
    st.markdown('<div class="performance-section-container">', unsafe_allow_html=True)
    if len(selected_strategies) > 0:
        equity_comp_fig = go.Figure()
        has_data_for_plot = False

        date_col_name = EXPECTED_COLUMNS.get('date')
        pnl_col_name = EXPECTED_COLUMNS.get('pnl')

        if not date_col_name or not pnl_col_name:
            display_custom_message("Date or PnL column configuration is missing. Cannot plot equity curves.", "error")
            logger.error("StrategyComparisonPage: Date or PnL column not found in EXPECTED_COLUMNS for equity plot.")
        else:
            for i, strat_name in enumerate(selected_strategies):
                strat_df = filtered_df[filtered_df[actual_strategy_col_name].astype(str) == str(strat_name)]
                if not strat_df.empty and date_col_name in strat_df.columns and pnl_col_name in strat_df.columns:
                    # Ensure date column is datetime
                    try:
                        strat_df[date_col_name] = pd.to_datetime(strat_df[date_col_name])
                    except Exception as e:
                        logger.warning(f"Could not convert date column '{date_col_name}' to datetime for strategy '{strat_name}': {e}")
                        continue # Skip this strategy if date conversion fails

                    strat_df = strat_df.sort_values(by=date_col_name)
                    
                    # Ensure PnL column is numeric
                    try:
                        strat_df[pnl_col_name] = pd.to_numeric(strat_df[pnl_col_name], errors='coerce')
                        strat_df.dropna(subset=[pnl_col_name], inplace=True) # Remove rows where PnL became NaN
                    except Exception as e:
                        logger.warning(f"Could not convert PnL column '{pnl_col_name}' to numeric for strategy '{strat_name}': {e}")
                        continue

                    if not strat_df.empty:
                        strat_df['cumulative_pnl'] = strat_df[pnl_col_name].cumsum()
                        # Use COLORS from config if available and strategy name matches
                        line_color = COLORS.get(strat_name, None) if isinstance(COLORS, dict) else None

                        trace_params = {
                            "x": strat_df[date_col_name],
                            "y": strat_df['cumulative_pnl'],
                            "mode": 'lines',
                            "name": strat_name
                        }
                        if line_color:
                            trace_params["line"] = dict(color=line_color)
                        
                        equity_comp_fig.add_trace(go.Scatter(**trace_params))
                        has_data_for_plot = True
                    else:
                        logger.info(f"Strategy '{strat_name}' became empty after PnL processing. Skipping for equity curve.")
                else:
                    logger.info(f"StrategyComparisonPage: No data or missing PnL/Date columns for strategy '{strat_name}' for equity plot after initial filter.")

            if has_data_for_plot:
                equity_comp_fig.update_layout(
                    # title="Equity Curve Comparison by Strategy", # Title is now part of st.subheader
                    xaxis_title="Date",
                    yaxis_title="Cumulative PnL",
                    hovermode="x unified",
                    legend_title_text='Strategy'
                )
                st.plotly_chart(_apply_custom_theme(equity_comp_fig, theme), use_container_width=True)
            elif selected_strategies: # Only show if strategies were selected but no plot generated
                display_custom_message("Not enough valid data to plot comparative equity curves for the selected strategies.", "info")

    else: # No strategies selected, or an earlier return happened
        if not selected_strategies: # Explicitly state if no strategies are selected
             display_custom_message("Select strategies to view equity curves.", "info")

    st.markdown('</div>', unsafe_allow_html=True) # Close performance-section-container for Equity Curves

if __name__ == "__main__":
    # This is primarily for direct execution testing, actual app runs through a main app.py
    st.set_page_config(layout="wide", page_title="Strategy Comparison")
    if 'app_initialized' not in st.session_state: # A simple check
        st.warning("This page is part of a multi-page app. For full functionality, run the main application script (e.g., app.py). Some features or session data might be missing.")
        # You might want to initialize some mock session state data for testing
        # st.session_state.filtered_data = pd.DataFrame(...) 
        # st.session_state.risk_free_rate = 0.02
        # st.session_state.current_theme = "dark" # or "light"
        
    # Ensure critical variables for the page are at least nominally defined for standalone run
    if 'EXPECTED_COLUMNS' not in globals(): EXPECTED_COLUMNS = {"strategy":"Strategy", "date":"Date", "pnl":"PnL"}
    if 'DEFAULT_KPI_DISPLAY_ORDER' not in globals(): DEFAULT_KPI_DISPLAY_ORDER = ['Total Return', 'Sharpe Ratio'] # Example
    if 'COLORS' not in globals(): COLORS = {}
    if 'analysis_service' not in globals() or analysis_service is None:
        class MockAnalysisService:
            def get_core_kpis(self, df, rate): 
                # Simulate some data for KPIs
                import random
                return {
                    "Total Return": random.uniform(-0.5, 2.0), 
                    "Sharpe Ratio": random.uniform(0.1, 3.0),
                    "Win Rate": random.uniform(0.3, 0.7),
                    "Profit Factor": random.uniform(0.5, 5.0),
                    "error": None
                }
        analysis_service = MockAnalysisService()
    if 'display_custom_message' not in globals():
        def display_custom_message(msg, type):
            if type == "error": st.error(msg)
            elif type == "warning": st.warning(msg)
            else: st.info(msg)
    if '_apply_custom_theme' not in globals():
        def _apply_custom_theme(fig, theme): return fig


    show_strategy_comparison_page()

