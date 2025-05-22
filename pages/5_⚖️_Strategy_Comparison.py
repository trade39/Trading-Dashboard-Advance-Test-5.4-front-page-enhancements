import streamlit as st
import pandas as pd
import logging
import plotly.graph_objects as go

# --- Custom CSS for card containers and DataFrame ---
# It's generally better to load CSS from a file, but for self-contained example:
st.markdown("""
<style>
/* Ensure these styles are consistent with your main style.css */
.card-section {
    background: var(--card-background-color, #262730); /* Fallback if CSS var not defined */
    border-radius: var(--border-radius-lg, 1.25rem);
    padding: 1.5rem;
    margin-bottom: 2rem;
    box-shadow: var(--box-shadow-md, 0 4px 8px rgba(0,0,0,0.2));
    border: 1px solid var(--border-color, #3A3C4E);
}
.card-title {
    font-size: 1.5rem; /* Increased slightly for better hierarchy */
    font-weight: 600;
    color: var(--text-heading-color, #F0F1F6);
    margin-bottom: 0.75rem; /* Adjusted margin */
    padding-bottom: 0.5rem; /* Added padding */
    border-bottom: 1px solid var(--border-color, #3A3C4E); /* Subtle separator */
    display: flex; /* For icon alignment */
    align-items: center; /* For icon alignment */
}
.card-title-icon {
    margin-right: 0.5rem; /* Space between icon and title */
    font-size: 1.3rem; /* Icon size */
}
.card-subtitle { /* Reusing this class for the main page subtitle */
    font-size: 1.1rem; /* Adjusted size */
    color: var(--text-muted-color, #A0A2B3);
    margin-top: -0.25rem; /* Pull up slightly */
    margin-bottom: 2.5rem; /* More space after subtitle */
    font-weight: 300;
}
.styled-table td, .styled-table th { /* Ensure this is used or remove if not */
    border: none !important;
    padding: 0.6rem 1rem !important;
}
</style>
""", unsafe_allow_html=True)

# --- Imports for app config and utilities ---
try:
    # Assuming these are in a reachable path (e.g., utils folder, config file)
    # For a real app, ensure PYTHONPATH or relative imports are set up correctly.
    from config import APP_TITLE, EXPECTED_COLUMNS, DEFAULT_KPI_DISPLAY_ORDER 
    # Removed COLORS as it wasn't used in this specific file previously
    from utils.common_utils import display_custom_message #, format_currency, format_percentage (if needed later)
    from services.analysis_service import AnalysisService
    from plotting import _apply_custom_theme # Make sure this function is robust
except ImportError as e:
    st.error(f"Strategy Comparison Page Error: Critical module import failed: {e}. Ensure app structure is correct and modules are in PYTHONPATH.")
    # Define fallbacks for critical variables if imports fail, to allow some UI rendering
    APP_TITLE = "TradingDashboard_Error"
    EXPECTED_COLUMNS = {"strategy": "strategy_fallback", "date": "date_fallback", "pnl": "pnl_fallback"}
    DEFAULT_KPI_DISPLAY_ORDER = []
    # Mock display_custom_message if it's not imported
    def display_custom_message(message, type="info"):
        if type == "error": st.error(message)
        elif type == "warning": st.warning(message)
        else: st.info(message)
    # Mock AnalysisService
    class AnalysisService:
        def get_core_kpis(self, df, risk_free_rate): return {"error": "AnalysisService not loaded"}
    # Mock _apply_custom_theme
    def _apply_custom_theme(fig, theme): return fig # Passthrough

    # Initialize logger even on import error for consistency
    logger = logging.getLogger(APP_TITLE if 'APP_TITLE' in locals() else "StrategyComparison_Fallback")
    logger.error(f"CRITICAL IMPORT ERROR in Strategy Comparison Page: {e}", exc_info=True)
    # st.stop() # Consider if stopping is always desired or if a degraded mode is better

# Initialize logger
logger = logging.getLogger(APP_TITLE if 'APP_TITLE' in locals() else "StrategyComparison") # Ensure APP_TITLE is defined
analysis_service = AnalysisService()

def get_st_theme():
    """Detect Streamlit theme dynamically."""
    try:
        theme_base = st.get_option("theme.base")
    except Exception:
        theme_base = st.session_state.get("current_theme", "dark") # Fallback
    return "dark" if theme_base == "dark" else "light" # Ensure valid return

def show_strategy_comparison_page():
    st.title("⚖️ Strategy Performance Comparison")
    st.markdown('<div class="page-subtitle">Easily compare performance, risk, and equity curves between your strategies side-by-side.</div>', unsafe_allow_html=True)
    theme = get_st_theme()

    # --- Data Check Section ---
    # This section doesn't have a visible header, so no icon needed here.
    # It's more of a pre-flight check.
    if 'filtered_data' not in st.session_state or st.session_state.filtered_data is None:
        display_custom_message("Please upload and process data on the 'Data Upload & Processing' page to compare strategies.", "info")
        logger.info("StrategyComparisonPage: No filtered_data in session_state.")
        return

    filtered_df = st.session_state.filtered_data
    risk_free_rate = st.session_state.get('risk_free_rate', 0.02) # Default risk-free rate
    
    # Ensure EXPECTED_COLUMNS is available
    if 'EXPECTED_COLUMNS' not in globals():
        display_custom_message("Configuration error: EXPECTED_COLUMNS not defined.", "error")
        logger.error("StrategyComparisonPage: EXPECTED_COLUMNS not defined globally.")
        return
        
    strategy_col_from_config = EXPECTED_COLUMNS.get('strategy')

    if filtered_df.empty:
        display_custom_message("No data matches the current filters. Cannot perform strategy comparison.", "info")
        logger.info("StrategyComparisonPage: filtered_df is empty.")
        return

    actual_strategy_col_name = strategy_col_from_config
    if not actual_strategy_col_name or actual_strategy_col_name not in filtered_df.columns:
        err_msg = (
            f"Strategy column ('{actual_strategy_col_name}') not found in the data. Comparison is not possible. "
            f"Available columns: {filtered_df.columns.tolist()}"
        )
        display_custom_message(err_msg, "warning")
        logger.warning(err_msg)
        return
    
    st.markdown("---") # Visual separator

    # --- Strategy Selection Section ---
    with st.container():
        st.markdown('<div class="card-section"><div class="card-title"><span class="card-title-icon">⚙️</span>Strategy Selection</div>', unsafe_allow_html=True)
        try:
            available_strategies = sorted(filtered_df[actual_strategy_col_name].astype(str).dropna().unique())
            if not available_strategies:
                display_custom_message("No distinct strategies found in the data to compare.", "info")
                st.markdown('</div>', unsafe_allow_html=True) # Close card-section
                logger.info("StrategyComparisonPage: No distinct strategies found.")
                return
            
            if len(available_strategies) < 2:
                display_custom_message(f"Only one strategy ('{available_strategies[0]}') found. At least two strategies are needed for comparison.", "info")
                # Still allow selection for single strategy view
            
            default_selection = available_strategies[:2] if len(available_strategies) >= 2 else available_strategies
            selected_strategies = st.multiselect(
                "Choose strategies to compare (select multiple):",
                options=available_strategies,
                default=default_selection,
                key="strategy_comp_select_v3" # Changed key to avoid state issues from previous versions
            )

            if not selected_strategies:
                display_custom_message("Please select at least one strategy to view its performance, or two or more to compare.", "info")
                st.markdown('</div>', unsafe_allow_html=True) # Close card-section
                return

            if len(selected_strategies) == 1:
                st.info(f"Displaying performance for strategy: **{selected_strategies[0]}**. Select additional strategies for a side-by-side comparison.")
            
            st.markdown('</div>', unsafe_allow_html=True) # Close card-section
        except Exception as e:
            logger.error(f"Error during strategy selection setup using column '{actual_strategy_col_name}': {e}", exc_info=True)
            display_custom_message(f"An error occurred setting up strategy selection: {e}", "error")
            st.markdown('</div>', unsafe_allow_html=True) # Close card-section in case of error
            return

    st.markdown("---") # Visual separator

    # --- KPI Comparison Table Section ---
    if selected_strategies: # Only show if strategies are selected
        with st.container():
            st.markdown('<div class="card-section"><div class="card-title"><span class="card-title-icon">📊</span>Key Performance Indicator Comparison</div>', unsafe_allow_html=True)
            comparison_kpi_data = []
            try:
                for strat_name in selected_strategies:
                    # Ensure strat_name is a string for consistent filtering, especially if IDs are numeric
                    strat_df = filtered_df[filtered_df[actual_strategy_col_name].astype(str) == str(strat_name)]
                    if not strat_df.empty:
                        # Ensure analysis_service is available
                        if 'analysis_service' not in globals():
                            display_custom_message("Analysis service not available.", "error")
                            logger.error("StrategyComparisonPage: analysis_service not defined globally.")
                            st.markdown('</div>', unsafe_allow_html=True)
                            return

                        kpis = analysis_service.get_core_kpis(strat_df, risk_free_rate)
                        if kpis and 'error' not in kpis:
                            comparison_kpi_data.append({"Strategy": strat_name, **kpis})
                        else:
                            logger.warning(f"Could not calculate KPIs for strategy '{strat_name}'. Error: {kpis.get('error') if isinstance(kpis, dict) else 'Unknown error or no KPIs returned'}")
                            display_custom_message(f"Could not retrieve KPIs for strategy: {strat_name}. It might have insufficient data or an issue in calculation.", "warning")
                    else:
                        logger.info(f"No data found for strategy '{strat_name}' within the current filters for KPI calculation.")
                        display_custom_message(f"No data available for strategy '{strat_name}' with current filters to calculate KPIs.", "info")


                if comparison_kpi_data:
                    comp_df = pd.DataFrame(comparison_kpi_data).set_index("Strategy")
                    
                    # Ensure DEFAULT_KPI_DISPLAY_ORDER is available
                    if 'DEFAULT_KPI_DISPLAY_ORDER' not in globals():
                        kpis_to_show_in_table = comp_df.columns.tolist() # Fallback
                        logger.warning("DEFAULT_KPI_DISPLAY_ORDER not defined, showing all available KPIs.")
                    else:
                        kpis_to_show_in_table = [
                            kpi for kpi in DEFAULT_KPI_DISPLAY_ORDER
                            if kpi in comp_df.columns and kpi not in ['trading_days', 'risk_free_rate_used'] # Example exclusions
                        ]
                    
                    if not kpis_to_show_in_table and comp_df.columns.any(): # If filter results in empty, show all
                        kpis_to_show_in_table = comp_df.columns.tolist()
                        logger.info("No KPIs matched DEFAULT_KPI_DISPLAY_ORDER, showing all available KPIs.")

                    # Dynamic theme-adaptive colors for table styling
                    if theme == "dark":
                        max_color_bg = "rgba(59, 165, 93, 0.3)"  # Semi-transparent green
                        min_color_bg = "rgba(255, 119, 107, 0.3)" # Semi-transparent red
                        cell_bg = "var(--card-background-color, #22232a)" # Use CSS var
                        text_color = "var(--text-color, #f6f6f6)"
                        header_bg = "var(--section-background-color, #20212F)"
                    else: # Light theme
                        max_color_bg = "rgba(178, 242, 187, 0.5)"
                        min_color_bg = "rgba(255, 214, 214, 0.5)"
                        cell_bg = "var(--card-background-color, #fff)"
                        text_color = "var(--text-color, #24292f)"
                        header_bg = "var(--section-background-color, #F7F8FC)"
                    
                    if kpis_to_show_in_table:
                        # Create a copy for styling to avoid SettingWithCopyWarning if comp_df is used later
                        display_df = comp_df[kpis_to_show_in_table].copy()
                        
                        # Apply formatting to specific known KPI types (extend as needed)
                        for col in display_df.columns:
                            if "Sharpe" in col or "Sortino" in col or "Ratio" in col or "Win Rate" in col or "Profit Factor" in col:
                                display_df[col] = display_df[col].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else "-")
                            elif "Drawdown" in col or "Return" in col: # Percentage based
                                 display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "-")
                            elif "P&L" in col or "Profit" in col or "Loss" in col: # Currency based
                                display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "-") # Basic currency
                            # Add more specific formatting rules here

                        styled_df = (
                            display_df
                            .style
                            # .format("{:,.2f}", na_rep="-") # General formatting, can be overridden by specific column formats
                            .set_properties(**{
                                "background-color": cell_bg,
                                "color": text_color,
                                "border": f"1px solid var(--input-border-color, {cell_bg})", # Subtle border
                                "font-size": "0.95rem" # Slightly smaller for tables
                            })
                            .set_table_styles([
                                {'selector': 'th', 'props': [('background-color', header_bg), ('color', text_color), ('font-weight', 'bold')]},
                                {'selector': 'td, th', 'props': [('padding', '0.5rem 0.75rem')]}
                            ])
                            .highlight_max(axis=0, props=f"background-color:{max_color_bg}; color:{text_color}; font-weight:bold;")
                            .highlight_min(axis=0, props=f"background-color:{min_color_bg}; color:{text_color}; font-weight:bold;")
                        )
                        st.dataframe(styled_df, use_container_width=True) # Auto-height or fixed height
                    else:
                        display_custom_message("No common KPIs found to display for the selected strategies after filtering.", "warning")
                elif selected_strategies: # If strategies were selected but no KPI data was generated
                    display_custom_message(
                        f"No performance data could be calculated for the selected strategies: {', '.join(selected_strategies)}. They might lack sufficient trade data or have issues.",
                        "warning"
                    )
                st.markdown('</div>', unsafe_allow_html=True) # Close card-section
            except Exception as e:
                logger.error(f"Error generating KPI comparison table: {e}", exc_info=True)
                display_custom_message(f"An error occurred generating the KPI comparison: {e}", "error")
                st.markdown('</div>', unsafe_allow_html=True) # Close card-section in case of error
    
    st.markdown("---") # Visual separator

    # --- Comparative Visualizations Section ---
    if selected_strategies: # Only show if strategies are selected
        with st.container():
            st.markdown('<div class="card-section"><div class="card-title"><span class="card-title-icon">📈</span>Comparative Equity Curves</div>', unsafe_allow_html=True)
            
            date_col_name = EXPECTED_COLUMNS.get('date')
            pnl_col_name = EXPECTED_COLUMNS.get('pnl')

            if not date_col_name or not pnl_col_name:
                display_custom_message("Date or PnL column configuration is missing. Cannot plot equity curves.", "error")
                logger.error("StrategyComparisonPage: Date or PnL column not found in EXPECTED_COLUMNS for equity plot.")
                st.markdown('</div>', unsafe_allow_html=True) # Close card-section
            else:
                equity_comp_fig = go.Figure()
                has_data_for_plot = False
                plot_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


                for i, strat_name in enumerate(selected_strategies):
                    strat_df = filtered_df[filtered_df[actual_strategy_col_name].astype(str) == str(strat_name)].copy() # Use .copy()
                    if not strat_df.empty and date_col_name in strat_df.columns and pnl_col_name in strat_df.columns:
                        # Ensure date column is datetime
                        try:
                            strat_df[date_col_name] = pd.to_datetime(strat_df[date_col_name])
                        except Exception as date_err:
                            logger.warning(f"Could not convert date column '{date_col_name}' to datetime for strategy '{strat_name}': {date_err}")
                            display_custom_message(f"Date column for strategy '{strat_name}' could not be processed. Skipping for equity curve.", "warning")
                            continue

                        strat_df = strat_df.sort_values(by=date_col_name)
                        
                        if pd.api.types.is_numeric_dtype(strat_df[pnl_col_name]):
                            strat_df.loc[:, 'cumulative_pnl'] = strat_df[pnl_col_name].cumsum() # Use .loc for assignment
                            equity_comp_fig.add_trace(go.Scatter(
                                x=strat_df[date_col_name],
                                y=strat_df['cumulative_pnl'],
                                mode='lines',
                                name=str(strat_name), # Ensure name is string
                                line=dict(color=plot_colors[i % len(plot_colors)]) # Cycle through colors
                            ))
                            has_data_for_plot = True
                        else:
                            logger.warning(f"StrategyComparisonPage: PnL column '{pnl_col_name}' for strategy '{strat_name}' is not numeric. Skipping for equity curve.")
                            display_custom_message(f"PnL data for strategy '{strat_name}' is not numeric. Skipping for equity curve.", "warning")
                    else:
                        logger.info(f"StrategyComparisonPage: No data or missing PnL/Date columns for strategy '{strat_name}' for equity plot.")

                if has_data_for_plot:
                    # Ensure _apply_custom_theme is available
                    if '_apply_custom_theme' not in globals():
                         final_fig = equity_comp_fig # Fallback
                         logger.warning("_apply_custom_theme not defined globally, using default Plotly theme.")
                    else:
                        final_fig = _apply_custom_theme(equity_comp_fig, theme)

                    final_fig.update_layout(
                        title_text="Equity Curve Comparison by Strategy",
                        xaxis_title="Date",
                        yaxis_title="Cumulative PnL (Normalized or Absolute)", # Clarify if normalized
                        hovermode="x unified",
                        legend_title_text='Strategies'
                    )
                    st.plotly_chart(final_fig, use_container_width=True)
                elif selected_strategies: # If strategies were selected but no data could be plotted
                    display_custom_message("Not enough valid data to plot comparative equity curves for the selected strategies.", "info")
                st.markdown('</div>', unsafe_allow_html=True) # Close card-section
    
    # Add a final closing div if the last section was conditional and might not render
    # This is tricky with Streamlit's execution model.
    # The current structure of closing divs within each conditional block is safer.

if __name__ == "__main__":
    # This check is good for preventing direct execution if it's part of a larger app
    if 'app_initialized' not in st.session_state: # A common way to check if the main app ran
        st.warning("This page is intended to be run as part of the main application. Some features might not work correctly if run directly.")
        # Optionally, set up minimal session_state for standalone testing
        # st.session_state.filtered_data = pd.DataFrame(...) # Mock data
        # st.session_state.risk_free_rate = 0.02
        # APP_TITLE = "StrategyComparison_Standalone" 
        # EXPECTED_COLUMNS = {"strategy": "Strategy", "date": "Date", "pnl": "PnL"}
        # DEFAULT_KPI_DISPLAY_ORDER = ['Total P&L', 'Sharpe Ratio']


    show_strategy_comparison_page()
