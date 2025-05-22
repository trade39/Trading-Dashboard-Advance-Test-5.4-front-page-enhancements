"""
pages/4_📉_Risk_and_Duration.py

This page focuses on risk metrics, correlation analysis, trade duration analysis,
and advanced drawdown analysis.
KPIs are now grouped for better readability.
Survival analysis now uses AIModelService.
Advanced drawdown analysis is added.
Icons and "View Data" options added for enhanced UX.
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
    def plot_correlation_matrix(**kwargs): return None # Dummy
    def _apply_custom_theme(fig, theme): return fig # Dummy
    def plot_equity_curve_and_drawdown(**kwargs): return None # Dummy
    def plot_underwater_analysis(**kwargs): return None # Dummy
    st.stop()

logger = logging.getLogger(APP_TITLE)
ai_model_service = AIModelService()
analysis_service_instance = AnalysisService()

def show_risk_duration_page():
    # --- Page Title and Initial Checks ---
    st.title("📉 Risk, Duration & Drawdown Analysis") # ICON ADDED
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
    st.header("🔑 Key Risk Metrics") # ICON ADDED
    cols_per_row_setting = 3 # Define how many KPIs per row
    for group_name, kpi_keys_in_group in KPI_GROUPS_RISK_DURATION.items():
        group_kpi_results = {key: kpi_results[key] for key in kpi_keys_in_group if key in kpi_results}
        
        # Special handling for market risk if benchmark is not available
        if group_name == "Market Risk & Relative Performance":
            if benchmark_daily_returns is None or benchmark_daily_returns.empty:
                if all(pd.isna(group_kpi_results.get(key, np.nan)) for key in kpi_keys_in_group):
                    logger.info(f"Skipping '{group_name}' KPI group as no benchmark is selected or data available.")
                    continue
            if not group_kpi_results or all(pd.isna(val) for val in group_kpi_results.values()):
                 logger.info(f"Skipping '{group_name}' KPI group as results are NaN or empty.")
                 continue
        
        if group_kpi_results:
            # ICON can be added to subheader if desired, e.g., st.subheader(f"📊 {group_name}")
            st.subheader(f"{group_name}") # Placeholder for potential icon
            try:
                kpi_cluster_risk = KPIClusterDisplay(
                    kpi_results=group_kpi_results,
                    kpi_definitions=KPI_CONFIG,
                    kpi_order=kpi_keys_in_group,
                    kpi_confidence_intervals=kpi_confidence_intervals,
                    cols_per_row=cols_per_row_setting
                )
                kpi_cluster_risk.render()
                st.markdown("---") # Visual separator
            except Exception as e:
                logger.error(f"Error rendering Key Risk Metrics for group '{group_name}': {e}", exc_info=True)
                display_custom_message(f"❌ An error occurred while displaying Key Risk Metrics for {group_name}: {e}", "error")

    # --- Advanced Drawdown Analysis Section ---
    st.header("🌊 Advanced Drawdown Analysis") # ICON ADDED
    date_col = EXPECTED_COLUMNS.get('date')
    cum_pnl_col = 'cumulative_pnl' # Assuming this is consistently named

    if date_col and cum_pnl_col and date_col in filtered_df.columns and cum_pnl_col in filtered_df.columns:
        # Prepare equity series for drawdown analysis
        equity_series_for_dd_prep = filtered_df.set_index(pd.to_datetime(filtered_df[date_col]))[cum_pnl_col].sort_index().dropna()
        
        if not equity_series_for_dd_prep.empty and len(equity_series_for_dd_prep) >= 5: # Check for sufficient data
            with st.spinner("⏳ Performing advanced drawdown analysis..."):
                adv_dd_results = analysis_service_instance.get_advanced_drawdown_analysis(
                    equity_series=equity_series_for_dd_prep
                )

            if adv_dd_results and 'error' not in adv_dd_results:
                st.subheader("📉 Individual Drawdown Periods") # ICON ADDED
                drawdown_periods_table = adv_dd_results.get("drawdown_periods")
                if drawdown_periods_table is not None and not drawdown_periods_table.empty:
                    display_dd_table = drawdown_periods_table.copy()
                    # Formatting for display
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

                st.subheader("📊 Drawdown Summary Statistics") # ICON ADDED
                dd_summary_cols = st.columns(3)
                with dd_summary_cols[0]:
                    st.metric("⏱️ Total Time in Drawdown", f"{adv_dd_results.get('total_time_in_drawdown_days', 0):.0f} days")
                with dd_summary_cols[1]:
                    st.metric("⏳ Avg. Drawdown Duration", f"{adv_dd_results.get('average_drawdown_duration_days', np.nan):.1f} days")
                with dd_summary_cols[2]:
                    st.metric("📈 Avg. Recovery Duration", f"{adv_dd_results.get('average_recovery_duration_days', np.nan):.1f} days")
                
                st.subheader("💹 Equity Curve with Drawdown Periods") # ICON ADDED
                drawdown_pct_col_name = 'drawdown_pct' # Ensure this column exists or is calculated
                equity_fig_shaded = plot_equity_curve_and_drawdown(
                    filtered_df,
                    date_col=date_col,
                    cumulative_pnl_col=cum_pnl_col,
                    drawdown_pct_col=drawdown_pct_col_name if drawdown_pct_col_name in filtered_df.columns else None,
                    drawdown_periods_df=drawdown_periods_table,
                    theme=plot_theme
                )
                if equity_fig_shaded:
                    st.plotly_chart(equity_fig_shaded, use_container_width=True)
                    # --- VIEW DATA OPTION ---
                    with st.expander("👁️ View Underlying Equity Curve Data"):
                        data_for_equity_plot = filtered_df[[date_col, cum_pnl_col]]
                        if drawdown_pct_col_name in filtered_df.columns:
                            data_for_equity_plot = pd.concat([data_for_equity_plot, filtered_df[[drawdown_pct_col_name]]], axis=1)
                        st.dataframe(data_for_equity_plot.reset_index(drop=True), use_container_width=True)
                        if drawdown_periods_table is not None and not drawdown_periods_table.empty:
                            st.markdown("##### Drawdown Periods Data Used for Shading:")
                            st.dataframe(drawdown_periods_table.reset_index(drop=True), use_container_width=True)
                else:
                    display_custom_message("⚠️ Could not generate equity curve with shaded drawdowns.", "warning")

                st.subheader("💧 Underwater Plot") # ICON ADDED
                underwater_fig = plot_underwater_analysis(equity_series_for_dd_prep, theme=plot_theme)
                if underwater_fig:
                    st.plotly_chart(underwater_fig, use_container_width=True)
                    # --- VIEW DATA OPTION ---
                    with st.expander("👁️ View Underlying Underwater Plot Data"):
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
    st.markdown("---") # Visual separator

    # --- Other Risk Visualizations Section ---
    st.header("🔗 Other Risk Visualizations") # ICON ADDED
    st.subheader("🔢 Feature Correlation Matrix") # ICON ADDED
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
            numeric_cols_for_corr.append(signal_conf_col_actual)

        numeric_cols_for_corr = list(set(numeric_cols_for_corr)) # Ensure unique columns

        if len(numeric_cols_for_corr) >= 2:
            correlation_fig = plot_correlation_matrix(
                filtered_df, numeric_cols=numeric_cols_for_corr, theme=plot_theme
            )
            if correlation_fig:
                st.plotly_chart(correlation_fig, use_container_width=True)
                # --- VIEW DATA OPTION ---
                with st.expander("👁️ View Underlying Correlation Data"):
                    st.dataframe(filtered_df[numeric_cols_for_corr].reset_index(drop=True), use_container_width=True)
            else:
                display_custom_message("⚠️ Could not generate the correlation matrix.", "warning")
        else:
            display_custom_message(f"ℹ️ Not enough numeric features (need at least 2, found {len(numeric_cols_for_corr)}) for correlation matrix. Available for correlation: {numeric_cols_for_corr}", "info")
    except Exception as e:
        logger.error(f"Error rendering Feature Correlation Matrix: {e}", exc_info=True)
        display_custom_message(f"❌ An error displaying Feature Correlation Matrix: {e}", "error")

    st.markdown("---") # Visual separator
    # --- Trade Duration Analysis (Survival Curve) - Currently Commented Out ---
    # st.subheader("⏳ Trade Duration Analysis (Survival Curve)") # ICON ADDED
    # duration_col_for_analysis = 'duration_minutes_numeric'
    # if duration_col_for_analysis in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[duration_col_for_analysis]):
    #     durations = filtered_df[duration_col_for_analysis].dropna()
    #     if not durations.empty:
    #         event_observed = pd.Series([True] * len(durations), index=durations.index) # Assuming all trades are 'closed' for this analysis
            
    #         with st.spinner("⏳ Performing Kaplan-Meier survival analysis..."):
    #             km_service_results = ai_model_service.perform_kaplan_meier_analysis(
    #                 durations, event_observed, confidence_level=CONFIDENCE_LEVEL
    #             )

    #         if km_service_results and 'error' not in km_service_results and 'survival_function_df' in km_service_results:
    #             survival_df = km_service_results['survival_function_df']
    #             km_plot_fig = go.Figure()
    #             km_plot_fig.add_trace(go.Scatter(
    #                 x=survival_df.index, y=survival_df['KM_estimate'],
    #                 mode='lines', name='Survival Probability (KM Estimate)', line_shape='hv',
    #                 line=dict(color=COLORS.get('royal_blue', 'blue'))
    #             ))
    #             if 'confidence_interval_df' in km_service_results and not km_service_results['confidence_interval_df'].empty:
    #                 ci_df = km_service_results['confidence_interval_df']
    #                 conf_level_str = str(CONFIDENCE_LEVEL).replace("0.", "")
    #                 lower_ci_col = f'KM_estimate_lower_{conf_level_str}'
    #                 upper_ci_col = f'KM_estimate_upper_{conf_level_str}'
                    
    #                 # Fallback for slightly different naming conventions if any
    #                 if lower_ci_col not in ci_df.columns and f'KM_estimate_lower_{CONFIDENCE_LEVEL:.2f}' in ci_df.columns:
    #                     lower_ci_col = f'KM_estimate_lower_{CONFIDENCE_LEVEL:.2f}'
    #                 if upper_ci_col not in ci_df.columns and f'KM_estimate_upper_{CONFIDENCE_LEVEL:.2f}' in ci_df.columns:
    #                     upper_ci_col = f'KM_estimate_upper_{CONFIDENCE_LEVEL:.2f}'
                    
    #                 if lower_ci_col in ci_df.columns and upper_ci_col in ci_df.columns:
    #                     km_plot_fig.add_trace(go.Scatter(
    #                         x=ci_df.index, y=ci_df[lower_ci_col], mode='lines',
    #                         line=dict(width=0), showlegend=False, line_shape='hv'
    #                     ))
    #                     km_plot_fig.add_trace(go.Scatter(
    #                         x=ci_df.index, y=ci_df[upper_ci_col], mode='lines',
    #                         line=dict(width=0), fill='tonexty', fillcolor='rgba(65,105,225,0.2)', # Example fill color
    #                         name=f'{int(CONFIDENCE_LEVEL*100)}% Confidence Interval',
    #                         showlegend=True, line_shape='hv'
    #                     ))
                
    #             duration_display_name = EXPECTED_COLUMNS.get('duration_minutes', 'duration_minutes').replace('_', ' ').title()
    #             km_plot_fig.update_layout(
    #                 title_text=f"Trade Survival Curve for {duration_display_name}",
    #                 xaxis_title=f"Duration ({duration_display_name})",
    #                 yaxis_title="Probability of Trade Still Being Open", yaxis_range=[0, 1.05]
    #             )
    #             st.plotly_chart(_apply_custom_theme(km_plot_fig, plot_theme), use_container_width=True)
    #             # --- VIEW DATA OPTION ---
    #             with st.expander("👁️ View Underlying Survival Curve Data"):
    #                 st.markdown("##### Survival Function Estimates:")
    #                 st.dataframe(survival_df.reset_index(), use_container_width=True)
    #                 if 'confidence_interval_df' in km_service_results and not km_service_results['confidence_interval_df'].empty:
    #                     st.markdown("##### Confidence Interval Data:")
    #                     st.dataframe(km_service_results['confidence_interval_df'].reset_index(), use_container_width=True)

    #             median_survival = km_service_results.get('median_survival_time')
    #             st.metric(
    #                 label=f"Median Trade Duration ({duration_display_name})",
    #                 value=f"{median_survival:.2f} mins" if pd.notna(median_survival) else "N/A",
    #                 help="The time at which 50% of trades are expected to have closed."
    #             )
    #         elif km_service_results and 'error' in km_service_results:
    #             display_custom_message(f"❌ Kaplan-Meier Analysis Error: {km_service_results['error']}", "error")
    #         else:
    #             display_custom_message("⚠️ Survival analysis for trade duration did not return expected results.", "warning")
    #     else:
    #          display_custom_message(f"ℹ️ No valid duration data in '{duration_col_for_analysis}' for survival analysis.", "info")
    # else:
    #     duration_config_key = 'duration_minutes'; expected_duration_col_name = EXPECTED_COLUMNS.get(duration_config_key)
    #     available_numeric_cols = filtered_df.select_dtypes(include=np.number).columns.tolist()
    #     error_msg = (f"⚠️ The standardized numeric duration column ('{duration_col_for_analysis}') was not found or is not numeric. "
    #                  f"Check CSV and `config.EXPECTED_COLUMNS['{duration_config_key}']` (expected as '{expected_duration_col_name}'). "
    #                  f"Available numeric columns: {available_numeric_cols}")
    #     display_custom_message(error_msg, "warning")

if __name__ == "__main__":
    # This check helps ensure the page is run as part of the main Streamlit app
    if 'app_initialized' not in st.session_state:
        st.warning("This page is part of a multi-page app. Please run the main app.py script.")
    show_risk_duration_page()
