"""
pages/4_📉_Risk_and_Duration.py

This page focuses on risk metrics, correlation analysis, trade duration analysis,
and advanced drawdown analysis.
KPIs are now grouped for better readability.
Survival analysis now uses AIModelService.
Advanced drawdown analysis is added.
Icons and "View Data" options added for enhanced UX.
Ensured _apply_custom_theme is called for all plots.
Contextual help and explanations added for charts and data sections.
"""
import streamlit as st
import pandas as pd
import numpy as np
import logging
import plotly.graph_objects as go

try:
    from config import APP_TITLE, EXPECTED_COLUMNS, COLORS, KPI_CONFIG, KPI_GROUPS_RISK_DURATION, CONFIDENCE_LEVEL
    from utils.common_utils import display_custom_message, format_currency, format_percentage
    from plotting import plot_correlation_matrix, _apply_custom_theme, plot_equity_curve_and_drawdown, plot_underwater_analysis
    from services.ai_model_service import AIModelService
    from services.analysis_service import AnalysisService # For advanced drawdown
    from components.kpi_display import KPIClusterDisplay
except ImportError as e:
    st.error(f"Risk & Duration Page Error: Critical module import failed: {e}.")
    # Fallback definitions for critical components if imports fail
    APP_TITLE = "TradingDashboard_Error"; logger = logging.getLogger(APP_TITLE)
    logger.error(f"CRITICAL IMPORT ERROR in 4_📉_Risk_and_Duration.py: {e}", exc_info=True)
    COLORS = {}; KPI_CONFIG = {}; KPI_GROUPS_RISK_DURATION = {}; CONFIDENCE_LEVEL = 0.95; EXPECTED_COLUMNS = {}
    def display_custom_message(msg, type="error"): st.error(msg)
    class KPIClusterDisplay:
        def __init__(self, **kwargs): pass
        def render(self): st.warning("KPI Display Component failed to load.")
    class AIModelService: # Dummy
        def perform_kaplan_meier_analysis(self, *args, **kwargs): return {"error": "Service not loaded"}
    class AnalysisService: # Dummy
        def get_advanced_drawdown_analysis(self, *args, **kwargs): return {"error": "Service not loaded"}
    def plot_correlation_matrix(**kwargs): return go.Figure() # Dummy returning a figure
    def _apply_custom_theme(fig, theme): return fig # Dummy
    def plot_equity_curve_and_drawdown(**kwargs): return go.Figure() # Dummy returning a figure
    def plot_underwater_analysis(**kwargs): return go.Figure() # Dummy returning a figure
    st.stop()

logger = logging.getLogger(APP_TITLE)
ai_model_service = AIModelService()
analysis_service_instance = AnalysisService()

# --- Help Text Dictionary ---
RISK_PAGE_HELP_TEXT = {
    "equity_curve": """
    **Equity Curve with Drawdown Periods:**
    - **Equity Curve (Main Plot):** Tracks the cumulative profit and loss (P&L) of the strategy over time. An upward trend generally indicates profitability.
    - **Drawdown Periods (Shaded Areas/Lower Plot):** Represent periods where the strategy's equity declined from a previous peak. The lower plot typically shows the percentage drawdown from the peak.
    - **Interpretation:** This chart is crucial for visualizing the growth of capital and understanding the magnitude and duration of losses. Key aspects to observe are:
        - The overall slope and smoothness of the equity curve (consistency of growth).
        - The depth of drawdowns (maximum percentage loss from a peak).
        - The duration of drawdowns (how long it takes to reach a new peak after a drawdown).
        - The frequency of drawdowns.
    """,
    "underwater_plot": """
    **Underwater Plot (Equity vs. High Water Mark):**
    - **Equity Value (Blue Line):** Shows the strategy's cumulative P&L over time.
    - **High Water Mark (Green Dashes/Line, if shown):** Represents the highest peak the equity has reached up to that point in time.
    - **Interpretation:** This plot specifically highlights how long and how far the strategy's equity remains below its previous all-time high (i.e., when it's "underwater").
        - The area between the high water mark (or the zero line if drawdowns are plotted directly) and the equity curve visually represents the drawdown percentage or absolute amount.
        - It helps assess the severity and duration of losing periods and the time it takes for the strategy to recover to new equity highs.
    """,
    "correlation_matrix": """
    **Feature Correlation Matrix:**
    - **What it shows:** This heatmap displays the Pearson correlation coefficients between various selected numeric features from your trading data (e.g., P&L, trade duration, risk per trade, signal confidence).
    - **Correlation Coefficient:** A value between -1 and +1 that measures the linear relationship between two variables.
        - `+1`: Perfect positive linear correlation (as one feature increases, the other tends to increase).
        - `-1`: Perfect negative linear correlation (as one feature increases, the other tends to decrease).
        - `0`: No linear correlation (features do not have a clear linear tendency to move together).
    - **Interpretation:**
        - Helps understand interdependencies between different aspects of your trading strategy and outcomes. For example, is higher P&L correlated with longer trade durations or higher initial risk?
        - Strong correlations (values close to +1 or -1) can indicate multicollinearity if these features are used in a predictive model, or they might reveal important trading dynamics.
        - Weak correlations (values close to 0) suggest that the features are largely linearly independent.
        - *Caution:* Correlation does not imply causation.
    """,
    "drawdown_periods_table": """
    **Individual Drawdown Periods Table:**
    This table lists significant drawdown periods experienced by the strategy. Each row details:
    - **Peak Date & Value:** The date and equity value when the drawdown began (the preceding high point).
    - **Trough Date & Value:** The date and equity value at the lowest point of the drawdown.
    - **End Date:** The date when the equity recovered to surpass the Peak Value (if recovered).
    - **Depth (Abs & Pct):** The maximum loss during the drawdown, in absolute currency and as a percentage of the Peak Value.
    - **Duration Days:** The number of days from the Peak Date to the Trough Date.
    - **Recovery Days:** The number of days from the Trough Date to the End Date. "Ongoing" indicates the drawdown has not yet fully recovered.
    - **Total Days:** The total time from Peak Date to End Date (or current date if ongoing).
    """,
    "drawdown_summary_stats": """
    **Drawdown Summary Statistics:**
    These metrics provide an overview of the strategy's drawdown characteristics:
    - **Total Time in Drawdown:** The cumulative number of days the strategy spent below a previous equity peak.
    - **Avg. Drawdown Duration:** The average length of time from the start of a drawdown to its lowest point.
    - **Avg. Recovery Duration:** The average length of time from the lowest point of a drawdown to when the equity recovers to its previous peak.
    """,
    "view_equity_data": "This table shows the daily time series data used for the 'Equity Curve with Drawdown Periods' plot. It typically includes the date, the cumulative profit and loss (equity value), and the calculated percentage drawdown from the peak equity.",
    "view_underwater_data": "This table displays the time series of the strategy's equity value. This data is used to construct the 'Underwater Plot', illustrating periods and magnitudes where the equity is below its previous peak.",
    "view_correlation_data": "This table presents the raw numeric feature values used to calculate the Pearson correlation coefficients displayed in the 'Feature Correlation Matrix' heatmap above. Each row typically represents a trade or an observation period, and columns are the selected numeric features."
}


def show_risk_duration_page():
    # --- Page Title and Initial Checks ---
    st.title("📉 Risk, Duration & Drawdown Analysis")
    logger.info("Rendering Risk & Duration Page.")

    if 'filtered_data' not in st.session_state or st.session_state.filtered_data is None:
        display_custom_message("ℹ️ Upload and process data to view this page.", "info")
        return
    if 'kpi_results' not in st.session_state or st.session_state.kpi_results is None:
        display_custom_message("⚠️ KPI results are not available. Ensure data is processed.", "warning")
        return
    if 'error' in st.session_state.kpi_results:
        display_custom_message(f"❌ Error in KPI calculation: {st.session_state.kpi_results['error']}", "error")
        return

    filtered_df = st.session_state.filtered_data
    kpi_results = st.session_state.kpi_results
    kpi_confidence_intervals = st.session_state.get('kpi_confidence_intervals', {})
    plot_theme = st.session_state.get('current_theme', 'dark')
    benchmark_daily_returns = st.session_state.get('benchmark_daily_returns')

    if filtered_df.empty:
        display_custom_message("ℹ️ No data matches filters for risk and duration analysis.", "info")
        return

    # --- Key Risk Metrics Section ---
    st.header("🔑 Key Risk Metrics")
    # Consider adding an expander here for a general overview of this section if desired.
    # For individual KPI explanations, this should ideally be handled within the KPIClusterDisplay component,
    # possibly by passing KPI_CONFIG (which should contain definitions) to it.
    # Example:
    # with st.expander("ℹ️ About Key Risk Metrics"):
    #     st.markdown("This section displays critical metrics related to portfolio risk, market exposure, and performance relative to benchmarks...")

    cols_per_row_setting = 3
    for group_name, kpi_keys_in_group in KPI_GROUPS_RISK_DURATION.items():
        group_kpi_results = {key: kpi_results[key] for key in kpi_keys_in_group if key in kpi_results}
        
        if group_name == "Market Risk & Relative Performance":
            if benchmark_daily_returns is None or benchmark_daily_returns.empty:
                if all(pd.isna(group_kpi_results.get(key, np.nan)) for key in kpi_keys_in_group):
                    logger.info(f"Skipping '{group_name}' KPI group as no benchmark is selected or data available.")
                    continue
            if not group_kpi_results or all(pd.isna(val) for val in group_kpi_results.values()):
                 logger.info(f"Skipping '{group_name}' KPI group as results are NaN or empty.")
                 continue
        
        if group_kpi_results:
            st.subheader(f"{group_name}")
            # Placeholder for KPI group explanation - could be sourced from a dictionary
            # with st.expander(f"ℹ️ About {group_name}", expanded=False):
            #    st.markdown(RISK_PAGE_HELP_TEXT.get(group_name.lower().replace(' ', '_'), "Details about this KPI group."))
            try:
                kpi_cluster_risk = KPIClusterDisplay(
                    kpi_results=group_kpi_results,
                    kpi_definitions=KPI_CONFIG, # KPI_CONFIG should contain definitions for individual KPIs
                    kpi_order=kpi_keys_in_group,
                    kpi_confidence_intervals=kpi_confidence_intervals,
                    cols_per_row=cols_per_row_setting
                )
                kpi_cluster_risk.render()
                st.markdown("---")
            except Exception as e:
                logger.error(f"Error rendering Key Risk Metrics for group '{group_name}': {e}", exc_info=True)
                display_custom_message(f"❌ An error occurred while displaying Key Risk Metrics for {group_name}: {e}", "error")

    # --- Advanced Drawdown Analysis Section ---
    st.header("🌊 Advanced Drawdown Analysis")
    date_col = EXPECTED_COLUMNS.get('date')
    cum_pnl_col = 'cumulative_pnl'

    if date_col and cum_pnl_col and date_col in filtered_df.columns and cum_pnl_col in filtered_df.columns:
        equity_series_for_dd_prep = filtered_df.set_index(pd.to_datetime(filtered_df[date_col]))[cum_pnl_col].sort_index().dropna()
        
        if not equity_series_for_dd_prep.empty and len(equity_series_for_dd_prep) >= 5:
            with st.spinner("⏳ Performing advanced drawdown analysis..."):
                adv_dd_results = analysis_service_instance.get_advanced_drawdown_analysis(
                    equity_series=equity_series_for_dd_prep
                )

            if adv_dd_results and 'error' not in adv_dd_results:
                st.subheader("📉 Individual Drawdown Periods")
                with st.expander("ℹ️ Learn more about this table", expanded=False):
                    st.markdown(RISK_PAGE_HELP_TEXT.get('drawdown_periods_table', "Explanation unavailable."))

                drawdown_periods_table = adv_dd_results.get("drawdown_periods")
                if drawdown_periods_table is not None and not drawdown_periods_table.empty:
                    display_dd_table = drawdown_periods_table.copy()
                    for col_name_dt in ['Peak Date', 'Trough Date', 'End Date']:
                        if col_name_dt in display_dd_table:
                            display_dd_table[col_name_dt] = pd.to_datetime(display_dd_table[col_name_dt]).dt.strftime('%Y-%m-%d')
                    for col_name_curr in ['Peak Value', 'Trough Value', 'Depth Abs']:
                         if col_name_curr in display_dd_table:
                            display_dd_table[col_name_curr] = display_dd_table[col_name_curr].apply(lambda x: format_currency(x) if pd.notna(x) else "N/A")
                    if 'Depth Pct' in display_dd_table:
                        display_dd_table['Depth Pct'] = display_dd_table['Depth Pct'].apply(lambda x: format_percentage(x/100.0) if pd.notna(x) else "N/A")
                    if 'Duration Days' in display_dd_table:
                         display_dd_table['Duration Days'] = display_dd_table['Duration Days'].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "N/A")
                    if 'Recovery Days' in display_dd_table:
                         display_dd_table['Recovery Days'] = display_dd_table['Recovery Days'].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "Ongoing")
                    
                    st.dataframe(display_dd_table, use_container_width=True, hide_index=True)
                else:
                    display_custom_message("ℹ️ No distinct drawdown periods identified or data was insufficient.", "info")

                st.subheader("📊 Drawdown Summary Statistics")
                with st.expander("ℹ️ Learn more about these statistics", expanded=False):
                    st.markdown(RISK_PAGE_HELP_TEXT.get('drawdown_summary_stats', "Explanation unavailable."))
                dd_summary_cols = st.columns(3)
                with dd_summary_cols[0]:
                    st.metric("⏱️ Total Time in Drawdown", f"{adv_dd_results.get('total_time_in_drawdown_days', 0):.0f} days")
                with dd_summary_cols[1]:
                    st.metric("⏳ Avg. Drawdown Duration", f"{adv_dd_results.get('average_drawdown_duration_days', np.nan):.1f} days")
                with dd_summary_cols[2]:
                    st.metric("📈 Avg. Recovery Duration", f"{adv_dd_results.get('average_recovery_duration_days', np.nan):.1f} days")
                
                st.subheader("💹 Equity Curve with Drawdown Periods")
                with st.expander("ℹ️ Learn more about this chart", expanded=False):
                    st.markdown(RISK_PAGE_HELP_TEXT.get('equity_curve', "Explanation unavailable."))

                drawdown_pct_col_name = 'drawdown_pct'
                equity_fig_shaded = plot_equity_curve_and_drawdown(
                    filtered_df,
                    date_col=date_col,
                    cumulative_pnl_col=cum_pnl_col,
                    drawdown_pct_col=drawdown_pct_col_name if drawdown_pct_col_name in filtered_df.columns else None,
                    drawdown_periods_df=drawdown_periods_table,
                    theme=plot_theme
                )
                if equity_fig_shaded:
                    st.plotly_chart(_apply_custom_theme(equity_fig_shaded, plot_theme), use_container_width=True)
                    with st.expander("👁️ View Underlying Equity Curve Data"):
                        st.caption(RISK_PAGE_HELP_TEXT.get('view_equity_data', "Data for the equity curve plot."))
                        data_for_equity_plot = filtered_df[[date_col, cum_pnl_col]]
                        if drawdown_pct_col_name in filtered_df.columns:
                            data_for_equity_plot = pd.concat([data_for_equity_plot, filtered_df[[drawdown_pct_col_name]]], axis=1)
                        st.dataframe(data_for_equity_plot.reset_index(drop=True), use_container_width=True)
                        if drawdown_periods_table is not None and not drawdown_periods_table.empty:
                            st.markdown("##### Drawdown Periods Data Used for Shading:")
                            st.dataframe(drawdown_periods_table.reset_index(drop=True), use_container_width=True)
                else:
                    display_custom_message("⚠️ Could not generate equity curve with shaded drawdowns.", "warning")

                st.subheader("💧 Underwater Plot")
                with st.expander("ℹ️ Learn more about this chart", expanded=False):
                    st.markdown(RISK_PAGE_HELP_TEXT.get('underwater_plot', "Explanation unavailable."))
                underwater_fig = plot_underwater_analysis(equity_series_for_dd_prep, theme=plot_theme)
                if underwater_fig:
                    st.plotly_chart(_apply_custom_theme(underwater_fig, plot_theme), use_container_width=True)
                    with st.expander("👁️ View Underlying Underwater Plot Data"):
                        st.caption(RISK_PAGE_HELP_TEXT.get('view_underwater_data', "Data for the underwater plot."))
                        st.dataframe(equity_series_for_dd_prep.reset_index().rename(columns={'index': date_col, cum_pnl_col: 'Equity Value'}), use_container_width=True)
                else:
                    display_custom_message("⚠️ Could not generate underwater plot.", "warning")

            elif adv_dd_results and 'error' in adv_dd_results:
                display_custom_message(f"❌ Advanced Drawdown Analysis Error: {adv_dd_results['error']}", "error")
            else:
                display_custom_message("⚠️ Advanced drawdown analysis did not return expected results.", "warning")
        else:
            display_custom_message(f"ℹ️ Not enough data points in equity series for advanced drawdown analysis (need at least 5). Found: {len(equity_series_for_dd_prep)}", "info")
    else:
        display_custom_message(f"⚠️ Required columns ('{date_col}', '{cum_pnl_col}') not found for Advanced Drawdown Analysis.", "warning")
    st.markdown("---")

    st.header("🔗 Other Risk Visualizations")
    st.subheader("🔢 Feature Correlation Matrix")
    with st.expander("ℹ️ Learn more about this chart", expanded=False):
        st.markdown(RISK_PAGE_HELP_TEXT.get('correlation_matrix', "Explanation unavailable."))
    try:
        pnl_col_name = EXPECTED_COLUMNS.get('pnl')
        numeric_cols_for_corr = []
        if pnl_col_name and pnl_col_name in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[pnl_col_name]):
            numeric_cols_for_corr.append(pnl_col_name)
        
        duration_numeric_col = 'duration_minutes_numeric'
        if duration_numeric_col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[duration_numeric_col]):
            numeric_cols_for_corr.append(duration_numeric_col)
        
        risk_numeric_col = 'risk_numeric_internal'
        if risk_numeric_col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[risk_numeric_col]):
            numeric_cols_for_corr.append(risk_numeric_col)
        
        r_r_csv_col_conceptual = 'r_r_csv_num'
        r_r_csv_col_actual = EXPECTED_COLUMNS.get(r_r_csv_col_conceptual)
        if r_r_csv_col_actual and r_r_csv_col_actual in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[r_r_csv_col_actual]):
            numeric_cols_for_corr.append(r_r_csv_col_actual)
        
        reward_risk_ratio_calculated_col = 'reward_risk_ratio_calculated'
        if reward_risk_ratio_calculated_col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[reward_risk_ratio_calculated_col]):
             numeric_cols_for_corr.append(reward_risk_ratio_calculated_col)

        signal_conf_col_conceptual = 'signal_confidence'
        signal_conf_col_actual = EXPECTED_COLUMNS.get(signal_conf_col_conceptual)
        if signal_conf_col_actual and signal_conf_col_actual in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[signal_conf_col_actual]):
            numeric_csols_for_corr.append(signal_conf_col_actual)

        numeric_cols_for_corr = list(set(numeric_cols_for_corr))

        if len(numeric_cols_for_corr) >= 2:
            correlation_fig = plot_correlation_matrix(
                filtered_df, numeric_cols=numeric_cols_for_corr, theme=plot_theme
            )
            if correlation_fig:
                st.plotly_chart(_apply_custom_theme(correlation_fig, plot_theme), use_container_width=True)
                with st.expander("👁️ View Underlying Correlation Data"):
                    st.caption(RISK_PAGE_HELP_TEXT.get('view_correlation_data', "Data for the correlation matrix."))
                    st.dataframe(filtered_df[numeric_cols_for_corr].reset_index(drop=True), use_container_width=True)
            else:
                display_custom_message("⚠️ Could not generate the correlation matrix.", "warning")
        else:
            display_custom_message(f"ℹ️ Not enough numeric features (need at least 2, found {len(numeric_cols_for_corr)}) for correlation matrix. Available for correlation: {numeric_cols_for_corr}", "info")
    except Exception as e:
        logger.error(f"Error rendering Feature Correlation Matrix: {e}", exc_info=True)
        display_custom_message(f"❌ An error displaying Feature Correlation Matrix: {e}", "error")

    st.markdown("---")
    # --- Trade Duration Analysis (Survival Curve) - Currently Commented Out ---
    # ... (If uncommented, add similar st.expander for explanation and st.caption for view data)

if __name__ == "__main__":
    if 'app_initialized' not in st.session_state:
        st.warning("This page is part of a multi-page app. Please run the main app.py script.")
    show_risk_duration_page()
