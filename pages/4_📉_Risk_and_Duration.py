"""
pages/4_📉_Risk_and_Duration.py

This page focuses on risk metrics, correlation analysis, trade duration analysis,
and advanced drawdown analysis.
KPIs are now grouped for better readability.
Survival analysis now uses AIModelService.
Advanced drawdown analysis is added.
Icons and "View Data" options added for enhanced UX.
Ensured _apply_custom_theme is called for all plots.
Contextual help and explanations added and refined for charts and data sections.
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

# --- Refined Help Text Dictionary ---
RISK_PAGE_HELP_TEXT = {
    "equity_curve": """
    **Equity Curve with Drawdown Periods:**

    The **Equity Curve** is a graphical representation of the strategy's cumulative Net Profit and Loss (Net P&L) over the selected period. It provides a visual narrative of capital appreciation or depreciation.
    - **Primary Y-axis (Left):** Cumulative Net P&L.
    - **X-axis:** Time.

    **Drawdown Periods** are visualized to highlight periods of capital erosion from a prior equity peak (High Water Mark - HWM).
    - **Shaded Areas (on Equity Curve):** Typically represent the duration of significant drawdown periods, from peak to recovery (or ongoing).
    - **Lower Subplot (Drawdown Percentage):** Illustrates the percentage decline from the HWM during each drawdown period. The Y-axis (right or separate) shows drawdown as a negative percentage.

    **Key Interpretations for Quantitative Analysis:**
    - **Trend & Slope:** A consistently upward-sloping curve indicates positive expectancy. The steepness reflects the rate of return.
    - **Volatility of Returns:** Fluctuations around the trend line indicate the volatility of returns. Smoother curves are generally preferred, signifying lower volatility.
    - **Drawdown Characteristics:**
        - **Max Drawdown (MDD):** The largest peak-to-trough percentage decline. A critical measure of downside risk.
        - **Drawdown Duration:** The time taken from the start of a drawdown (peak) to its lowest point (trough).
        - **Recovery Period:** The time taken from the trough to reclaim the previous HWM.
        - **Frequency:** How often significant drawdowns occur.
    - **Path Dependency:** The sequence of returns matters. This chart helps assess if the strategy is prone to prolonged periods of underperformance.
    - **Psychological Impact:** Large or lengthy drawdowns can be psychologically challenging for traders and investors.
    """,
    "underwater_plot": """
    **Underwater Plot (Equity vs. High Water Mark):**

    This plot provides a focused visualization of the strategy's drawdown behavior by explicitly charting the equity relative to its High Water Mark (HWM), or by directly plotting the drawdown percentage over time.
    - **Equity Value Line:** The strategy's cumulative Net P&L.
    - **High Water Mark (HWM) Line (Conceptual or Plotted):** The highest equity value achieved to date.
    - **Underwater Area / Drawdown Line:** The region where the equity curve is below the HWM, or a direct plot of the drawdown percentage (typically negative values or positive values on an inverted axis).

    **Key Interpretations for Quantitative Analysis:**
    - **Time Spent Underwater:** Quantifies the proportion of time the strategy is not generating new equity highs. Prolonged underwater periods can indicate issues with strategy adaptation or market regime changes.
    - **Severity of Drawdowns:** The depth of the "underwater" sections directly corresponds to the magnitude of drawdowns.
    - **Recovery Dynamics:** The shape and speed of the equity curve exiting an underwater period illustrate the strategy's recovery capability.
    - **Comparison with MDD:** While MDD gives the single worst drawdown, the underwater plot shows all instances and their durations, offering a more complete picture of downside risk exposure.
    - **Investor Perspective:** This plot is particularly relevant for assessing investor experience, as periods underwater often correlate with investor dissatisfaction or redemption pressure.
    """,
    "correlation_matrix": """
    **Feature Correlation Matrix:**

    This heatmap visualizes the **Pearson correlation coefficients** between selected numerical features of the trading strategy or market data. The Pearson coefficient (ρ) measures the linear relationship between two variables, ranging from -1 (perfect negative linear correlation) to +1 (perfect positive linear correlation), with 0 indicating no linear correlation.

    **Matrix Elements:**
    - Each cell `(i, j)` shows the correlation between feature `i` and feature `j`.
    - The diagonal `(i, i)` is always 1 (correlation of a feature with itself).
    - The matrix is symmetric: `ρ(i, j) = ρ(j, i)`.

    **Key Interpretations for Quantitative Analysis:**
    - **Factor Analysis:** Identify which factors (e.g., trade duration, risk per trade, specific market indicators, signal confidence scores) are linearly related to P&L or other performance metrics.
    - **Multicollinearity Detection:** In predictive modeling, high correlation between independent variables (features) can lead to multicollinearity, making model coefficients unstable and difficult to interpret. This matrix helps identify such relationships.
    - **Strategy Insights:**
        - Correlation between P&L and risk metrics (e.g., `risk_numeric_internal`) can indicate if higher risk taken generally translates to higher returns (or losses).
        - Correlation between P&L and trade characteristics (e.g., `duration_minutes_numeric`) can reveal if holding trades longer/shorter impacts profitability.
    - **Portfolio Construction:** Understanding correlations between different strategies or assets is fundamental for diversification.
    - **Caution:**
        - **Linearity:** Pearson correlation only captures linear relationships. Non-linear associations might be missed.
        - **Causation:** Correlation does not imply causation. A strong correlation between two variables does not mean one causes the other.
        - **Outliers:** Correlation coefficients can be sensitive to outliers in the data.
        - **Spurious Correlations:** Apparent correlations can arise by chance, especially with many variables or limited data.
    """,
    "drawdown_periods_table": """
    **Individual Drawdown Periods Table:**

    This table provides a granular breakdown of distinct drawdown periods experienced by the strategy. A drawdown is defined as a decline in equity from a previous peak (High Water Mark) to a subsequent trough, before a new peak is achieved.

    **Columns Explained:**
    - **Peak Date & Value:** The date and the equity value at the start of the drawdown (the preceding High Water Mark).
    - **Trough Date & Value:** The date and the equity value at the lowest point reached during that specific drawdown.
    - **End Date (Recovery Date):** The date when the equity value first surpassed the 'Peak Value' of this drawdown, signifying full recovery. 'Ongoing' indicates the drawdown is still active or recovery is not yet complete.
    - **Depth (Abs):** The absolute monetary loss from 'Peak Value' to 'Trough Value' (`Peak Value - Trough Value`).
    - **Depth (Pct):** The percentage loss from 'Peak Value' to 'Trough Value' (`(Peak Value - Trough Value) / Peak Value * 100%`). This is a critical measure of downside risk for that period.
    - **Duration Days (Peak to Trough):** The number of calendar or trading days from the 'Peak Date' to the 'Trough Date'.
    - **Recovery Days (Trough to End):** The number of calendar or trading days from the 'Trough Date' to the 'End Date'.
    - **Total Days (Peak to End):** The total duration of the drawdown event, from 'Peak Date' to 'End Date'.

    **Key Interpretations for Quantitative Analysis:**
    - **Risk Profiling:** Identifies the most severe historical loss periods, crucial for setting risk limits and capital allocation.
    - **Strategy Resilience:** Analyzing recovery times helps assess how quickly a strategy bounces back from losses.
    - **Parameter Tuning:** Can be used to evaluate if changes in strategy parameters correlate with changes in drawdown characteristics.
    - **Investor Reporting:** Essential data for transparent communication with investors about historical risk.
    """,
    "drawdown_summary_stats": """
    **Drawdown Summary Statistics:**

    These metrics provide a high-level statistical overview of the strategy's historical drawdown behavior, aggregating information from all identified drawdown periods.

    - **Total Time in Drawdown (Days):** The cumulative number of days the strategy's equity has spent below any previous High Water Mark. This indicates the proportion of time the strategy is in a state of recovering past losses rather than generating new profits.
        - *Calculation Note:* This is often calculated by summing the 'Total Days' for all distinct drawdown periods, or by counting all days where `Equity < HWM`.
    - **Average Drawdown Duration (Days):** The arithmetic mean of the 'Duration Days (Peak to Trough)' for all identified drawdown periods. It indicates, on average, how long it takes for a drawdown to reach its lowest point.
        - *Formula (Conceptual):* `Sum of all (Peak to Trough Durations) / Number of Drawdowns`
    - **Average Recovery Duration (Days):** The arithmetic mean of the 'Recovery Days (Trough to End)' for all *recovered* drawdown periods. It indicates, on average, how long it takes for the strategy to regain its previous peak after hitting a drawdown trough. Drawdowns that are still ongoing are typically excluded from this calculation or handled specifically.
        - *Formula (Conceptual):* `Sum of all (Trough to End Durations for recovered drawdowns) / Number of Recovered Drawdowns`

    **Key Interpretations for Quantitative Analysis:**
    - **Persistence of Losses:** A high 'Total Time in Drawdown' suggests the strategy frequently struggles to make new highs.
    - **Typical Drawdown Lifecycle:** The average duration and recovery metrics help establish expectations for the "pain period" during typical drawdowns.
    - **Strategy Health:** Significant changes in these averages over different time windows could indicate shifts in strategy performance or market conditions.
    """,
    "view_equity_data": """
    **Underlying Data for Equity Curve & Drawdown Plot:**

    This table presents the time-series data used to generate the 'Equity Curve with Drawdown Periods' visualization.
    - **`{date_col}` (or equivalent):** Timestamp for each data point (e.g., daily, hourly).
    - **`{cumulative_pnl_col}` (or equivalent):** The strategy's cumulative Net Profit and Loss at each timestamp, representing the equity value.
    - **`{drawdown_pct_col_name}` (or equivalent, if present):** The calculated percentage drawdown from the prevailing High Water Mark at each timestamp. This is used for the lower drawdown subplot.
    - **Drawdown Periods Data (if shown separately):** The structured table detailing individual peak-to-trough-to-recovery periods, used for shading the equity curve or highlighting significant drawdowns.

    This raw data allows for verification, further custom analysis, or export.
    """,
    "view_underwater_data": """
    **Underlying Data for Underwater Plot:**

    This table displays the core time-series data used to construct the 'Underwater Plot'.
    - **`{date_col}` (or equivalent):** Timestamp for each data point.
    - **`Equity Value` (derived from `{cumulative_pnl_col}`):** The strategy's cumulative Net Profit and Loss at each timestamp.

    The plot itself visualizes this equity series, often in conjunction with its High Water Mark, to highlight periods and magnitudes where the equity is below its previous peak. This data is fundamental for calculating metrics like 'Time Spent Underwater'.
    """,
    "view_correlation_data": """
    **Underlying Data for Feature Correlation Matrix:**

    This table presents the raw numerical values for the features selected for the correlation analysis. Each row typically corresponds to an observation (e.g., a single trade, a daily summary) and each column represents one of the chosen numeric features.
    - **Columns:** Names of the numeric features included in the correlation matrix (e.g., `pnl`, `duration_minutes_numeric`, `risk_numeric_internal`, `signal_confidence`).

    The Pearson correlation coefficients in the heatmap are calculated based on these underlying values. Access to this data allows for:
    - Verification of the correlation calculations.
    - Exploration of scatter plots between specific pairs of features to visually inspect their relationship beyond the linear correlation coefficient.
    - Advanced statistical analysis, such as testing the significance of correlations or exploring non-linear relationships.
    """
}


def show_risk_duration_page():
    # --- Page Title and Initial Checks ---
    st.title("📉 Risk, Duration & Drawdown Analysis")
    logger.info("Rendering Risk & Duration Page.")

    # Replace placeholders in help text with actual column names if they are dynamic
    # This is a bit more advanced and might require passing EXPECTED_COLUMNS or specific column names
    # into the RISK_PAGE_HELP_TEXT dictionary formatting if you want live column names in help.
    # For now, I've used conceptual placeholders like {date_col}.

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
            try:
                kpi_cluster_risk = KPIClusterDisplay(
                    kpi_results=group_kpi_results,
                    kpi_definitions=KPI_CONFIG,
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
    date_col = EXPECTED_COLUMNS.get('date', 'date') # Default to 'date' if not found
    cum_pnl_col = 'cumulative_pnl'
    drawdown_pct_col_name = 'drawdown_pct'


    # Dynamically insert actual column names into help text (example for one entry)
    # A more robust solution would iterate through all help texts or use a dedicated formatting function
    current_equity_help = RISK_PAGE_HELP_TEXT.get('view_equity_data', "Explanation unavailable.")
    formatted_equity_help = current_equity_help.format(
        date_col=date_col,
        cumulative_pnl_col=cum_pnl_col,
        drawdown_pct_col_name=drawdown_pct_col_name
    )

    current_underwater_help = RISK_PAGE_HELP_TEXT.get('view_underwater_data', "Explanation unavailable.")
    formatted_underwater_help = current_underwater_help.format(
        date_col=date_col,
        cumulative_pnl_col=cum_pnl_col
    )


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
                    for col_name_dt_loop in ['Peak Date', 'Trough Date', 'End Date']: # Renamed loop variable
                        if col_name_dt_loop in display_dd_table:
                            display_dd_table[col_name_dt_loop] = pd.to_datetime(display_dd_table[col_name_dt_loop]).dt.strftime('%Y-%m-%d')
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
                        st.caption(formatted_equity_help) # Use formatted help text
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
                        st.caption(formatted_underwater_help) # Use formatted help text
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
            numeric_cols_for_corr.append(signal_conf_col_actual) # Corrected variable name

        numeric_cols_for_corr = list(set(numeric_cols_for_corr))

        if len(numeric_cols_for_corr) >= 2:
            correlation_fig = plot_correlation_matrix(
                filtered_df, numeric_cols=numeric_cols_for_corr, theme=plot_theme
            )
            if correlation_fig:
                st.plotly_chart(_apply_custom_theme(correlation_fig, plot_theme), use_container_width=True)
                with st.expander("👁️ View Underlying Correlation Data"):
                    # For correlation data, the column names are dynamic based on numeric_cols_for_corr
                    # So, a generic message is often best, or one that lists the included columns.
                    view_corr_data_text = RISK_PAGE_HELP_TEXT.get('view_correlation_data', "Data for the correlation matrix.")
                    if numeric_cols_for_corr:
                         view_corr_data_text += f"\n\nFeatures included: `{'`, `'.join(numeric_cols_for_corr)}`."

                    st.caption(view_corr_data_text)
                    st.dataframe(filtered_df[numeric_cols_for_corr].reset_index(drop=True), use_container_width=True)
            else:
                display_custom_message("⚠️ Could not generate the correlation matrix.", "warning")
        else:
            display_custom_message(f"ℹ️ Not enough numeric features (need at least 2, found {len(numeric_cols_for_corr)}) for correlation matrix. Available for correlation: {numeric_cols_for_corr}", "info")
    except Exception as e:
        logger.error(f"Error rendering Feature Correlation Matrix: {e}", exc_info=True)
        display_custom_message(f"❌ An error displaying Feature Correlation Matrix: {e}", "error")

    st.markdown("---")

if __name__ == "__main__":
    if 'app_initialized' not in st.session_state:
        st.warning("This page is part of a multi-page app. Please run the main app.py script.")
    show_risk_duration_page()
