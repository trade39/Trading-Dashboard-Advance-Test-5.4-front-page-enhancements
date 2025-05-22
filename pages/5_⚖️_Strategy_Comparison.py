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
.page-subtitle { /* Reusing this class for the main page subtitle */
    font-size: 1.1rem; /* Adjusted size */
    color: var(--text-muted-color, #A0A2B3);
    margin-top: -0.25rem; /* Pull up slightly */
    margin-bottom: 2.5rem; /* More space after subtitle */
    font-weight: 300;
}
/* Styles for the KPI table (st.dataframe) can be influenced here if needed,
   but most styling is done via pandas Styler */
</style>
""", unsafe_allow_html=True)

# --- Imports for app config and utilities ---
try:
    from config import APP_TITLE, EXPECTED_COLUMNS, DEFAULT_KPI_DISPLAY_ORDER
    from utils.common_utils import display_custom_message
    from services.analysis_service import AnalysisService
    # _apply_custom_theme is removed as we are handling theming directly for now
except ImportError as e:
    st.error(f"Strategy Comparison Page Error: Critical module import failed: {e}. Ensure app structure is correct and modules are in PYTHONPATH.")
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
        theme_base = st.session_state.get("current_theme", "dark")
    return "dark" if theme_base == "dark" else "light"

def show_strategy_comparison_page():
    st.title("⚖️ Strategy Performance Comparison")
    st.markdown('<div class="page-subtitle">Easily compare performance, risk, and equity curves between your strategies side-by-side.</div>', unsafe_allow_html=True)
    theme = get_st_theme()

    if 'filtered_data' not in st.session_state or st.session_state.filtered_data is None:
        display_custom_message("Please upload and process data on the 'Data Upload & Processing' page to compare strategies.", "info")
        logger.info("StrategyComparisonPage: No filtered_data in session_state.")
        return

    filtered_df = st.session_state.filtered_data
    risk_free_rate = st.session_state.get('risk_free_rate', 0.02)
    
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
        err_msg = (f"Strategy column ('{actual_strategy_col_name}') not found. Available: {filtered_df.columns.tolist()}")
        display_custom_message(err_msg, "warning")
        logger.warning(err_msg)
        return
    
    st.markdown("---")

    with st.container():
        st.markdown('<div class="card-section"><div class="card-title"><span class="card-title-icon">⚙️</span>Strategy Selection</div>', unsafe_allow_html=True)
        try:
            available_strategies = sorted(filtered_df[actual_strategy_col_name].astype(str).dropna().unique())
            if not available_strategies:
                display_custom_message("No distinct strategies found.", "info")
                st.markdown('</div>', unsafe_allow_html=True)
                return
            
            default_selection = available_strategies[:2] if len(available_strategies) >= 2 else available_strategies
            selected_strategies = st.multiselect(
                "Choose strategies to compare (select multiple):",
                options=available_strategies,
                default=default_selection,
                key="strategy_comp_select_v4" 
            )

            if not selected_strategies:
                display_custom_message("Please select at least one strategy.", "info")
                st.markdown('</div>', unsafe_allow_html=True)
                return

            if len(selected_strategies) == 1:
                st.info(f"Displaying for: **{selected_strategies[0]}**. Select more for comparison.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            logger.error(f"Err in strategy selection '{actual_strategy_col_name}': {e}", exc_info=True)
            display_custom_message(f"Err setting up strategy selection: {e}", "error")
            st.markdown('</div>', unsafe_allow_html=True)
            return

    st.markdown("---")

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
                            logger.warning(f"No KPIs for '{strat_name}'. Err: {kpis.get('error') if isinstance(kpis, dict) else 'No KPIs'}")
                    else:
                        logger.info(f"No data for '{strat_name}' for KPI calc.")

                if comparison_kpi_data:
                    comp_df = pd.DataFrame(comparison_kpi_data).set_index("Strategy")
                    
                    kpis_to_show_in_table = [
                        kpi for kpi in DEFAULT_KPI_DISPLAY_ORDER
                        if kpi in comp_df.columns and kpi not in ['trading_days', 'risk_free_rate_used']
                    ] if 'DEFAULT_KPI_DISPLAY_ORDER' in globals() else comp_df.columns.tolist()
                    
                    if not kpis_to_show_in_table and comp_df.columns.any():
                        kpis_to_show_in_table = comp_df.columns.tolist()

                    # Theme-adaptive colors for table styling
                    if theme == "dark":
                        max_color_bg = "rgba(76, 175, 80, 0.25)"  # More transparent green
                        min_color_bg = "rgba(244, 67, 54, 0.25)"  # More transparent red
                        highlight_text_color = "#F0F1F6" # Bright text for dark theme highlights (var(--text-heading-color))
                        cell_bg = "#242535" # var(--card-background-color)
                        text_color = "#E0E1E6" # var(--text-color)
                        header_bg = "#20212F" # var(--section-background-color)
                        border_color_table = "#3A3C4E" # var(--border-color)
                    else: # Light theme
                        max_color_bg = "rgba(200, 230, 201, 0.7)" # Light green
                        min_color_bg = "rgba(255, 205, 210, 0.7)" # Light red
                        highlight_text_color = "#121416" # Dark text for light theme highlights (var(--text-heading-color))
                        cell_bg = "#FFFFFF" # var(--card-background-color)
                        text_color = "#212529" # var(--text-color)
                        header_bg = "#F7F8FC" # var(--section-background-color)
                        border_color_table = "#DEE2E6" # var(--border-color)
                    
                    if kpis_to_show_in_table:
                        display_df = comp_df[kpis_to_show_in_table].copy()
                        for col in display_df.columns: # Apply specific formatting
                            if any(term in col for term in ["Sharpe", "Sortino", "Ratio", "Win Rate", "Profit Factor"]):
                                display_df[col] = display_df[col].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else "-")
                            elif any(term in col for term in ["Drawdown", "Return"]):
                                 display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "-")
                            elif any(term in col for term in ["P&L", "Profit", "Loss", "Amount"]):
                                display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "-")
                            # else: keep as is or apply general float format if numeric

                        styled_df = (
                            display_df.style
                            .set_properties(**{
                                "background-color": cell_bg,
                                "color": text_color,
                                "border": f"1px solid {border_color_table}",
                                "font-size": "0.95rem"
                            })
                            .set_table_styles([
                                {'selector': 'th', 'props': [('background-color', header_bg), ('color', text_color), ('font-weight', 'bold'), ('border', f"1px solid {border_color_table}")]},
                                {'selector': 'td, th', 'props': [('padding', '0.5rem 0.75rem')]}
                            ])
                            .highlight_max(axis=0, props=f"background-color:{max_color_bg}; color:{highlight_text_color}; font-weight:bold;")
                            .highlight_min(axis=0, props=f"background-color:{min_color_bg}; color:{highlight_text_color}; font-weight:bold;")
                        )
                        st.dataframe(styled_df, use_container_width=True)
                    else:
                        display_custom_message("No common KPIs to display.", "warning")
                elif selected_strategies:
                    display_custom_message(f"No KPI data for: {', '.join(selected_strategies)}.", "warning")
                st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                logger.error(f"Err generating KPI table: {e}", exc_info=True)
                display_custom_message(f"Err generating KPI table: {e}", "error")
                st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")

    if selected_strategies:
        with st.container():
            st.markdown('<div class="card-section"><div class="card-title"><span class="card-title-icon">📈</span>Comparative Equity Curves</div>', unsafe_allow_html=True)
            
            date_col_name = EXPECTED_COLUMNS.get('date')
            pnl_col_name = EXPECTED_COLUMNS.get('pnl')

            if not date_col_name or not pnl_col_name:
                display_custom_message("Date/PnL column config missing for equity curves.", "error")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                equity_comp_fig = go.Figure()
                has_data_for_plot = False
                plot_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

                for i, strat_name in enumerate(selected_strategies):
                    strat_df = filtered_df[filtered_df[actual_strategy_col_name].astype(str) == str(strat_name)].copy()
                    if not strat_df.empty and date_col_name in strat_df.columns and pnl_col_name in strat_df.columns:
                        try:
                            strat_df[date_col_name] = pd.to_datetime(strat_df[date_col_name])
                        except Exception as date_err:
                            logger.warning(f"Date conversion error for '{strat_name}': {date_err}")
                            continue
                        strat_df = strat_df.sort_values(by=date_col_name)
                        if pd.api.types.is_numeric_dtype(strat_df[pnl_col_name]):
                            strat_df.loc[:, 'cumulative_pnl'] = strat_df[pnl_col_name].cumsum()
                            equity_comp_fig.add_trace(go.Scatter(
                                x=strat_df[date_col_name],
                                y=strat_df['cumulative_pnl'],
                                mode='lines',
                                name=str(strat_name),
                                line=dict(color=plot_colors[i % len(plot_colors)], width=2)
                            ))
                            has_data_for_plot = True
                        else:
                            logger.warning(f"PnL not numeric for '{strat_name}'. Skipping equity curve.")
                    else:
                        logger.info(f"No/incomplete data for '{strat_name}' for equity plot.")

                if has_data_for_plot:
                    # Define colors based on theme for Plotly chart
                    if theme == "dark":
                        font_color = "#E0E1E6"          # --text-color
                        muted_font_color = "#A0A2B3"    # --text-muted-color
                        grid_line_color = "#3A3C4E"     # --border-color
                        bg_color_transparent = "rgba(0,0,0,0)"
                        plotly_template = "plotly_dark"
                    else: # Light theme
                        font_color = "#212529"          # --text-color
                        muted_font_color = "#6C757D"    # --text-muted-color
                        grid_line_color = "#DEE2E6"     # --border-color
                        bg_color_transparent = "rgba(255,255,255,0)"
                        plotly_template = "plotly_white"

                    layout_updates = {
                        "template": plotly_template,
                        "paper_bgcolor": bg_color_transparent,
                        "plot_bgcolor": bg_color_transparent,
                        "font_color": font_color,
                        "title_text": "Equity Curve Comparison by Strategy",
                        "title_font_color": font_color,
                        "xaxis_title": "Date",
                        "yaxis_title": "Cumulative PnL",
                        "xaxis": {
                            "gridcolor": grid_line_color, "linecolor": grid_line_color,
                            "zerolinecolor": grid_line_color, "zerolinewidth": 1,
                            "title_font_color": muted_font_color, "tickfont_color": muted_font_color
                        },
                        "yaxis": {
                            "gridcolor": grid_line_color, "linecolor": grid_line_color,
                            "zerolinecolor": grid_line_color, "zerolinewidth": 1,
                            "title_font_color": muted_font_color, "tickfont_color": muted_font_color
                        },
                        "hovermode": "x unified",
                        "legend": {
                            "bgcolor": bg_color_transparent,
                            "bordercolor": grid_line_color, "borderwidth": 0, # Set borderwidth to 0 if not desired
                            "font_color": font_color,
                            "title_text": 'Strategies', "title_font_color": muted_font_color
                        }
                    }
                    equity_comp_fig.update_layout(**layout_updates)
                    st.plotly_chart(equity_comp_fig, use_container_width=True)
                elif selected_strategies:
                    display_custom_message("Not enough valid data to plot equity curves.", "info")
                st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    if 'app_initialized' not in st.session_state:
        st.warning("This page is part of a multi-page app. Run main app.py.")
    show_strategy_comparison_page()
