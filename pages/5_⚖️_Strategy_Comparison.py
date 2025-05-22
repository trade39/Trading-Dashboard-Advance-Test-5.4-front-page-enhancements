import streamlit as st
import pandas as pd
import logging
import plotly.graph_objects as go

# --- Custom CSS (ensure it's loaded once, typically in main app.py or via st.set_page_config) ---
# For this page, we assume necessary CSS variables are available from a global style.css
st.markdown("""
<style>
/* Ensure these styles are consistent with your main style.css */
.card-section {
    background: var(--card-background-color, #262730);
    border-radius: var(--border-radius-lg, 1.25rem);
    padding: 1.5rem;
    margin-bottom: 2rem;
    box-shadow: var(--box-shadow-md, 0 4px 8px rgba(0,0,0,0.2));
    border: 1px solid var(--border-color, #3A3C4E);
}
.card-title {
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--text-heading-color, #F0F1F6);
    margin-bottom: 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border-color, #3A3C4E);
    display: flex;
    align-items: center;
}
.card-title-icon {
    margin-right: 0.5rem;
    font-size: 1.3rem;
}
.page-subtitle {
    font-size: 1.1rem;
    color: var(--text-muted-color, #A0A2B3);
    margin-top: -0.25rem;
    margin-bottom: 2.5rem;
    font-weight: 300;
}
</style>
""", unsafe_allow_html=True)

# --- Imports for app config and utilities ---
try:
    from config import APP_TITLE, EXPECTED_COLUMNS, DEFAULT_KPI_DISPLAY_ORDER
    from utils.common_utils import display_custom_message
    from services.analysis_service import AnalysisService
except ImportError as e:
    st.error(f"Strategy Comparison Page Error: Critical module import failed: {e}.")
    APP_TITLE = "TradingDashboard_Error"
    EXPECTED_COLUMNS = {"strategy": "strategy_fallback", "date": "date_fallback", "pnl": "pnl_fallback"}
    DEFAULT_KPI_DISPLAY_ORDER = []
    def display_custom_message(message, type="info"):
        if type == "error": st.error(message)
        elif type == "warning": st.warning(message)
        else: st.info(message)
    class AnalysisService:
        def get_core_kpis(self, df, risk_free_rate): return {"error": "AnalysisService not loaded"}
    logger = logging.getLogger(APP_TITLE if 'APP_TITLE' in locals() else "StrategyComparison_Fallback")
    logger.error(f"CRITICAL IMPORT ERROR in Strategy Comparison Page: {e}", exc_info=True)

logger = logging.getLogger(APP_TITLE if 'APP_TITLE' in locals() else "StrategyComparison")
analysis_service = AnalysisService()

def get_st_theme():
    try:
        theme_base = st.get_option("theme.base")
    except Exception:
        theme_base = st.session_state.get("current_theme", "dark") # Default to dark if not found
    return "dark" if theme_base == "dark" else "light"

def show_strategy_comparison_page():
    st.title("⚖️ Strategy Performance Comparison")
    st.markdown('<div class="page-subtitle">Easily compare performance, risk, and equity curves between your strategies side-by-side.</div>', unsafe_allow_html=True)
    theme = get_st_theme()

    if 'filtered_data' not in st.session_state or st.session_state.filtered_data is None:
        display_custom_message("Please upload and process data on the 'Data Upload & Processing' page to compare strategies.", "info")
        return

    filtered_df = st.session_state.filtered_data
    risk_free_rate = st.session_state.get('risk_free_rate', 0.02)
    
    if 'EXPECTED_COLUMNS' not in globals():
        display_custom_message("Configuration error: EXPECTED_COLUMNS not defined.", "error")
        return
        
    strategy_col_from_config = EXPECTED_COLUMNS.get('strategy')

    if filtered_df.empty:
        display_custom_message("No data matches current filters. Cannot perform strategy comparison.", "info")
        return

    actual_strategy_col_name = strategy_col_from_config
    if not actual_strategy_col_name or actual_strategy_col_name not in filtered_df.columns:
        err_msg = (f"Strategy column ('{actual_strategy_col_name}') not found. Available: {filtered_df.columns.tolist()}")
        display_custom_message(err_msg, "warning")
        return
    
    st.markdown("---") # Visual separator

    with st.container():
        st.markdown('<div class="card-section"><div class="card-title"><span class="card-title-icon">⚙️</span>Strategy Selection</div>', unsafe_allow_html=True)
        try:
            available_strategies = sorted(filtered_df[actual_strategy_col_name].astype(str).dropna().unique())
            if not available_strategies:
                display_custom_message("No distinct strategies found in the data to compare.", "info")
                st.markdown('</div>', unsafe_allow_html=True) # Close card-section
                return
            
            default_selection = available_strategies[:2] if len(available_strategies) >= 2 else available_strategies
            selected_strategies = st.multiselect(
                "Choose strategies to compare (select multiple):",
                options=available_strategies,
                default=default_selection,
                key="strategy_comp_select_v5" # Incremented key
            )

            if not selected_strategies:
                display_custom_message("Please select at least one strategy to view.", "info")
                st.markdown('</div>', unsafe_allow_html=True) # Close card-section
                return

            if len(selected_strategies) == 1:
                st.info(f"Displaying performance for strategy: **{selected_strategies[0]}**. Select more for comparison.")
            
            st.markdown('</div>', unsafe_allow_html=True) # Close card-section
        except Exception as e:
            logger.error(f"Error during strategy selection using column '{actual_strategy_col_name}': {e}", exc_info=True)
            display_custom_message(f"An error occurred setting up strategy selection: {e}", "error")
            st.markdown('</div>', unsafe_allow_html=True) # Close card-section
            return

    st.markdown("---") # Visual separator

    if selected_strategies:
        with st.container():
            st.markdown('<div class="card-section"><div class="card-title"><span class="card-title-icon">📊</span>Key Performance Indicator Comparison</div>', unsafe_allow_html=True)
            comparison_kpi_data = []
            try:
                for strat_name in selected_strategies:
                    strat_df = filtered_df[filtered_df[actual_strategy_col_name].astype(str) == str(strat_name)]
                    if not strat_df.empty:
                        if 'analysis_service' not in globals():
                            display_custom_message("Analysis service not available.", "error")
                            st.markdown('</div>', unsafe_allow_html=True)
                            return

                        kpis = analysis_service.get_core_kpis(strat_df, risk_free_rate)
                        if kpis and 'error' not in kpis:
                            comparison_kpi_data.append({"Strategy": strat_name, **kpis})
                        else:
                            logger.warning(f"Could not calculate KPIs for strategy '{strat_name}'. Error: {kpis.get('error') if isinstance(kpis, dict) else 'Unknown error or no KPIs returned'}")
                    else:
                        logger.info(f"No data found for strategy '{strat_name}' within the current filters for KPI calculation.")

                if comparison_kpi_data:
                    comp_df = pd.DataFrame(comparison_kpi_data).set_index("Strategy")
                    
                    kpis_to_show_in_table = ([
                        kpi for kpi in DEFAULT_KPI_DISPLAY_ORDER
                        if kpi in comp_df.columns and kpi not in ['trading_days', 'risk_free_rate_used']
                    ] if ('DEFAULT_KPI_DISPLAY_ORDER' in globals() and DEFAULT_KPI_DISPLAY_ORDER)
                       else comp_df.columns.tolist())
                    
                    if not kpis_to_show_in_table and comp_df.columns.any(): # Fallback if filter results in empty
                        kpis_to_show_in_table = comp_df.columns.tolist()

                    # Theme-adaptive colors for table styling
                    if theme == "dark":
                        max_color_bg = "rgba(76, 175, 80, 0.3)"   # Greenish highlight bg
                        min_color_bg = "rgba(244, 67, 54, 0.3)"   # Reddish highlight bg
                        highlight_text_color = "#FFFFFF"          # Pure white for text on highlights
                        cell_bg = "var(--card-background-color, #242535)"
                        text_color_default = "var(--text-color, #E0E1E6)" # Default text color for cells
                        header_bg = "var(--section-background-color, #20212F)"
                        border_color_table = "var(--border-color, #3A3C4E)"
                    else: # Light theme
                        max_color_bg = "rgba(200, 230, 201, 0.7)"
                        min_color_bg = "rgba(255, 205, 210, 0.7)"
                        highlight_text_color = "#101010" # Dark text for highlights on light theme
                        cell_bg = "var(--card-background-color, #FFFFFF)"
                        text_color_default = "var(--text-color, #212529)"
                        header_bg = "var(--section-background-color, #F7F8FC)"
                        border_color_table = "var(--border-color, #DEE2E6)"
                    
                    if kpis_to_show_in_table:
                        display_df = comp_df[kpis_to_show_in_table].copy()
                        # Apply specific formatting to known KPI types
                        for col in display_df.columns:
                            if any(term in col.lower() for term in ["sharpe", "sortino", "ratio", "win rate", "profit factor", "calmar", "beta"]):
                                display_df[col] = display_df[col].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else None) # Format, keep None for na_rep
                            elif any(term in col.lower() for term in ["drawdown", "return", "alpha"]): # Percentage based
                                 display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else None)
                            elif any(term in col.lower() for term in ["p&l", "pnl", "profit", "loss", "amount", "value", "capital"]): # Currency based
                                display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else None) # Basic currency
                            # Add more specific formatting rules here

                        styled_df = (
                            display_df.style
                            .format(na_rep="-", precision=2) # Global format for NaNs and default float precision
                            .set_properties(**{
                                "background-color": cell_bg,
                                "color": text_color_default, # Apply default text color to all cells
                                "border": f"1px solid {border_color_table}",
                                "font-size": "0.95rem"
                            })
                            .set_table_styles([
                                {'selector': 'th', 'props': [('background-color', header_bg), ('color', text_color_default), ('font-weight', 'bold'), ('border', f"1px solid {border_color_table}")]},
                                {'selector': 'td, th', 'props': [('padding', '0.5rem 0.75rem')]}
                            ])
                            .highlight_max(axis=0, props=f"background-color:{max_color_bg}; color:{highlight_text_color}; font-weight:bold;")
                            .highlight_min(axis=0, props=f"background-color:{min_color_bg}; color:{highlight_text_color}; font-weight:bold;")
                        )
                        st.dataframe(styled_df, use_container_width=True)
                    else:
                        display_custom_message("No common KPIs found to display for selected strategies.", "warning")
                elif selected_strategies:
                    display_custom_message(
                        f"No performance data could be calculated for the selected strategies: {', '.join(selected_strategies)}.",
                        "warning"
                    )
                st.markdown('</div>', unsafe_allow_html=True) # Close card-section
            except Exception as e:
                logger.error(f"Error generating KPI comparison table: {e}", exc_info=True)
                display_custom_message(f"An error occurred generating the KPI comparison: {e}", "error")
                st.markdown('</div>', unsafe_allow_html=True) # Close card-section
    
    st.markdown("---") # Visual separator

    if selected_strategies:
        with st.container():
            st.markdown('<div class="card-section"><div class="card-title"><span class="card-title-icon">📈</span>Comparative Equity Curves</div>', unsafe_allow_html=True)
            
            date_col_name = EXPECTED_COLUMNS.get('date')
            pnl_col_name = EXPECTED_COLUMNS.get('pnl')

            if not date_col_name or not pnl_col_name:
                display_custom_message("Date or PnL column configuration is missing. Cannot plot equity curves.", "error")
                st.markdown('</div>', unsafe_allow_html=True) # Close card-section
            else:
                equity_comp_fig = go.Figure()
                has_data_for_plot = False
                # Define a list of distinct colors for plot lines
                plot_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


                for i, strat_name in enumerate(selected_strategies):
                    strat_df = filtered_df[filtered_df[actual_strategy_col_name].astype(str) == str(strat_name)].copy()
                    if not strat_df.empty and date_col_name in strat_df.columns and pnl_col_name in strat_df.columns:
                        try:
                            strat_df[date_col_name] = pd.to_datetime(strat_df[date_col_name])
                        except Exception as date_err:
                            logger.warning(f"Could not convert date column '{date_col_name}' to datetime for strategy '{strat_name}': {date_err}")
                            continue # Skip this strategy if date conversion fails

                        strat_df = strat_df.sort_values(by=date_col_name)
                        
                        if pd.api.types.is_numeric_dtype(strat_df[pnl_col_name]):
                            strat_df.loc[:, 'cumulative_pnl'] = strat_df[pnl_col_name].cumsum()
                            equity_comp_fig.add_trace(go.Scatter(
                                x=strat_df[date_col_name],
                                y=strat_df['cumulative_pnl'],
                                mode='lines',
                                name=str(strat_name), # Ensure name is string
                                line=dict(color=plot_colors[i % len(plot_colors)], width=2) # Cycle through defined colors
                            ))
                            has_data_for_plot = True
                        else:
                            logger.warning(f"StrategyComparisonPage: PnL column '{pnl_col_name}' for strategy '{strat_name}' is not numeric. Skipping for equity curve.")
                    else:
                        logger.info(f"StrategyComparisonPage: No data or missing PnL/Date columns for strategy '{strat_name}' for equity plot.")

                if has_data_for_plot:
                    # Define colors based on theme for Plotly chart
                    if theme == "dark":
                        font_color = "var(--text-color, #E0E1E6)"                 # Main text/font color
                        axis_title_color = "var(--text-muted-color, #B0B3C3)"     # Slightly brighter muted for axis titles
                        tick_label_color = "var(--text-muted-color, #A0A2B3)"     # Muted for tick labels
                        grid_line_color = "var(--input-border-color, #4A4C5E)"    # Lighter gridlines
                        bg_color_transparent = "rgba(0,0,0,0)"
                        plotly_template = "plotly_dark" # Base dark template
                    else: # Light theme
                        font_color = "var(--text-color, #212529)"
                        axis_title_color = "var(--text-muted-color, #505050)"
                        tick_label_color = "var(--text-muted-color, #6C757D)"
                        grid_line_color = "var(--input-border-color, #DEE2E6)"
                        bg_color_transparent = "rgba(255,255,255,0)" # Or your card background for light
                        plotly_template = "plotly_white" # Base light template

                    layout_updates = {
                        "template": plotly_template, # Apply base template first
                        "paper_bgcolor": bg_color_transparent,
                        "plot_bgcolor": bg_color_transparent,
                        "font_color": font_color, # Overall font color
                        "title_font_color": font_color, # Title font color
                        "xaxis": {
                            "gridcolor": grid_line_color, "linecolor": grid_line_color,
                            "zerolinecolor": grid_line_color, "zerolinewidth": 1,
                            "title_font_color": axis_title_color, "tickfont_color": tick_label_color
                        },
                        "yaxis": {
                            "gridcolor": grid_line_color, "linecolor": grid_line_color,
                            "zerolinecolor": grid_line_color, "zerolinewidth": 1,
                            "title_font_color": axis_title_color, "tickfont_color": tick_label_color
                        },
                        "hovermode": "x unified",
                        "legend": {
                            "bgcolor": bg_color_transparent, # Make legend background transparent
                            "bordercolor": grid_line_color, "borderwidth": 0, # Optional: remove legend border
                            "font_color": tick_label_color, # Legend text color (can be same as ticks or main font)
                            "title_text": 'Strategies', "title_font_color": axis_title_color
                        }
                        # title_text is set directly in Figure for better control if needed
                    }
                    equity_comp_fig.update_layout(**layout_updates)
                    # Explicitly set title if not done globally
                    equity_comp_fig.update_layout(title_text="Equity Curve Comparison by Strategy")


                    st.plotly_chart(equity_comp_fig, use_container_width=True)
                elif selected_strategies:
                    display_custom_message("Not enough valid data to plot comparative equity curves for the selected strategies.", "info")
                st.markdown('</div>', unsafe_allow_html=True) # Close card-section

if __name__ == "__main__":
    if 'app_initialized' not in st.session_state: # A common way to check if the main app ran
        st.warning("This page is intended to be run as part of the main application. Some features might not work correctly if run directly.")
    show_strategy_comparison_page()
