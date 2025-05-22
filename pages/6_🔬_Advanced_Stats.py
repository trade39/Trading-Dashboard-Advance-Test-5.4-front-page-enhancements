"""
pages/6_🔬_Advanced_Stats.py
Advanced statistical analyses with UI/UX enhancements.
Uses StatisticalAnalysisService for relevant methods.
Incorporates icons for better visual hierarchy.
Sections are now organized into tabs for clarity.
"""
import streamlit as st
import pandas as pd
import numpy as np
import logging
from statsmodels.tsa.seasonal import DecomposeResult # For type hint

try:
    # Assuming these are correctly defined in your project structure
    from config import APP_TITLE, EXPECTED_COLUMNS, BOOTSTRAP_ITERATIONS, CONFIDENCE_LEVEL
    from utils.common_utils import display_custom_message
    from services.statistical_analysis_service import StatisticalAnalysisService
    from plotting import plot_time_series_decomposition, plot_bootstrap_distribution_and_ci
    # statistical_methods.DISTRIBUTIONS_TO_FIT is kept for potential future use
    from statistical_methods import DISTRIBUTIONS_TO_FIT 
except ImportError as e:
    st.error(f"Advanced Stats Page Error: Critical module import failed: {e}. Please ensure all dependencies and project files are correctly placed.")
    # Fallback definitions for critical error
    APP_TITLE = "TradingDashboard_Error"
    logger = logging.getLogger(APP_TITLE)
    logger.error(f"CRITICAL IMPORT ERROR in 6_🔬_Advanced_Stats.py: {e}", exc_info=True)
    EXPECTED_COLUMNS = {"pnl": "pnl", "date": "date"} 
    BOOTSTRAP_ITERATIONS = 1000
    CONFIDENCE_LEVEL = 0.95
    DISTRIBUTIONS_TO_FIT = ['norm'] 
    
    class StatisticalAnalysisService: # Dummy service
        def get_time_series_decomposition(self, *args, **kwargs):
            return {"error": "StatisticalAnalysisService not loaded due to import failure."}
        def calculate_bootstrap_ci(self, *args, **kwargs):
            return {"error": "StatisticalAnalysisService not loaded due to import failure.", "lower_bound": np.nan, "upper_bound": np.nan, "observed_statistic": np.nan, "bootstrap_statistics": []}
            
    def plot_time_series_decomposition(*args, **kwargs):
        logger.error("plot_time_series_decomposition called but not loaded.")
        return None
        
    def plot_bootstrap_distribution_and_ci(*args, **kwargs):
        logger.error("plot_bootstrap_distribution_and_ci called but not loaded.")
        return None

    def display_custom_message(message, type="error"): # Basic fallback
        if type == "error": st.error(message)
        elif type == "warning": st.warning(message)
        else: st.info(message)
        
    st.stop()

logger = logging.getLogger(APP_TITLE)
statistical_analysis_service = StatisticalAnalysisService()

def show_advanced_stats_page():
    st.title("🔬 Advanced Statistical Analysis")
    
    if 'filtered_data' not in st.session_state or st.session_state.filtered_data is None:
        display_custom_message("Please upload and process your data first to access advanced statistical analyses.", "info")
        return

    filtered_df = st.session_state.filtered_data
    plot_theme = st.session_state.get('current_theme', 'dark')
    
    pnl_col = EXPECTED_COLUMNS.get('pnl')
    date_col = EXPECTED_COLUMNS.get('date')

    if filtered_df.empty:
        display_custom_message("The filtered data is empty. Please adjust your filters or upload new data.", "info")
        return
    if not pnl_col or pnl_col not in filtered_df.columns:
        display_custom_message(f"The expected PnL column ('{pnl_col}') was not found in the data.", "error")
        return
    # Date column check is more crucial for decomposition, handled within its tab/section.

    pnl_series_for_adv = filtered_df[pnl_col].dropna()
    if pnl_series_for_adv.empty:
        display_custom_message("PnL data is empty after removing missing values. Cannot perform analysis.", "warning")
        return

    # --- Create Tabs ---
    tab_bs_ci, tab_ts_decomp = st.tabs(["📊 Bootstrap Confidence Intervals", "📉 Time Series Decomposition"])

    # --- Tab 1: Bootstrap Confidence Intervals ---
    with tab_bs_ci:
        st.header("Bootstrap Confidence Intervals") # Using header for tab content title
        with st.expander("Configure & Run Bootstrap Analysis", expanded=True):
            stat_options_bs = {
                "Mean PnL": np.mean,
                "Median PnL": np.median,
                "PnL Standard Deviation": np.std,
                "PnL Skewness": pd.Series.skew,
                "PnL Kurtosis": pd.Series.kurtosis,
                "Win Rate (%)": lambda x: (np.sum(x > 0) / len(x)) * 100 if len(x) > 0 else 0.0
            }
            
            min_data_for_skew = 3
            min_data_for_kurtosis = 4
            available_stat_options = {}
            for name, func in stat_options_bs.items():
                if name == "PnL Skewness" and len(pnl_series_for_adv) < min_data_for_skew:
                    continue
                if name == "PnL Kurtosis" and len(pnl_series_for_adv) < min_data_for_kurtosis:
                    continue
                available_stat_options[name] = func

            if not available_stat_options:
                st.warning("Not enough data points to calculate any of the available bootstrap statistics.")
            else:
                # Unique form key for this tab
                with st.form("bootstrap_form_tab_v1"): 
                    selected_stat_name_bs = st.selectbox(
                        "Select Statistic for Bootstrap CI:",
                        list(available_stat_options.keys()),
                        key="bs_stat_select_tab_v1",
                        help="Choose the metric for which to calculate the confidence interval."
                    )
                    n_iterations_bs = st.number_input(
                        "Number of Bootstrap Iterations:",
                        min_value=100, max_value=10000, value=BOOTSTRAP_ITERATIONS, step=100,
                        key="bs_iterations_tab_v1",
                        help="More iterations lead to more stable CIs but increase computation time."
                    )
                    conf_level_bs_percent = st.slider(
                        "Confidence Level (%):",
                        min_value=80.0, max_value=99.9, value=CONFIDENCE_LEVEL * 100, step=0.1, format="%.1f%%",
                        key="bs_conf_level_tab_v1",
                        help="The desired confidence level for the interval (e.g., 95%)."
                    )
                    run_bs_button_tab_v1 = st.form_submit_button(f"Calculate & Plot CI for {selected_stat_name_bs}")

                if run_bs_button_tab_v1 and selected_stat_name_bs:
                    if len(pnl_series_for_adv) >= 10:
                        stat_func_to_run_bs = available_stat_options[selected_stat_name_bs]
                        actual_conf_level = conf_level_bs_percent / 100.0

                        with st.spinner(f"Bootstrapping CI for {selected_stat_name_bs}... ({n_iterations_bs} iterations)"):
                            bs_results = statistical_analysis_service.calculate_bootstrap_ci(
                                data_series=pnl_series_for_adv,
                                statistic_func=stat_func_to_run_bs,
                                n_iterations=n_iterations_bs,
                                confidence_level=actual_conf_level
                            )

                        if bs_results and 'error' not in bs_results:
                            st.success(f"Bootstrap for {selected_stat_name_bs} complete!")
                            obs_stat = bs_results.get('observed_statistic', np.nan)
                            lower_b = bs_results.get('lower_bound', np.nan)
                            upper_b = bs_results.get('upper_bound', np.nan)
                            bootstrap_samples = bs_results.get('bootstrap_statistics', [])

                            col1_bs, col2_bs = st.columns(2)
                            with col1_bs:
                                st.metric(label=f"Observed {selected_stat_name_bs}", value=f"{obs_stat:.4f}")
                            with col2_bs:
                                st.metric(label=f"{int(actual_conf_level*100)}% CI", value=f"[{lower_b:.4f}, {upper_b:.4f}]")
                            
                            if bootstrap_samples:
                                bs_plot = plot_bootstrap_distribution_and_ci(
                                    bootstrap_statistics=bootstrap_samples,
                                    observed_statistic=obs_stat,
                                    lower_bound=lower_b,
                                    upper_bound=upper_b,
                                    statistic_name=selected_stat_name_bs,
                                    theme=plot_theme
                                )
                                if bs_plot: st.plotly_chart(bs_plot, use_container_width=True)
                                else: display_custom_message("Could not generate bootstrap distribution plot.", "warning")
                            else: display_custom_message("No bootstrap samples returned for plotting.", "warning")
                        elif bs_results: display_custom_message(f"Bootstrap Error for {selected_stat_name_bs}: {bs_results.get('error', 'Unknown error')}", "error")
                        else: display_custom_message(f"Bootstrap analysis for {selected_stat_name_bs} failed to return results.", "error")
                    else:
                        display_custom_message(f"Not enough PnL data (need ≥10, found {len(pnl_series_for_adv)}) for bootstrap CI for {selected_stat_name_bs}.", "warning")

    # --- Tab 2: Time Series Decomposition ---
    with tab_ts_decomp:
        st.header("Time Series Decomposition") # Using header for tab content title
        
        # Check for date column specifically for this section
        if not date_col or date_col not in filtered_df.columns:
            display_custom_message(f"The expected Date column ('{date_col}') is required for Time Series Decomposition and was not found in the data.", "error")
        else:
            # Prepare series for decomposition
            series_options_decomp = {}
            # Ensure date column is datetime
            # Make a copy to avoid SettingWithCopyWarning if filtered_df is a slice
            df_for_decomp = filtered_df.copy()
            if not pd.api.types.is_datetime64_any_dtype(df_for_decomp[date_col]):
                try:
                    df_for_decomp[date_col] = pd.to_datetime(df_for_decomp[date_col])
                except Exception as e:
                    display_custom_message(f"Could not convert date column '{date_col}' to datetime: {e}. Decomposition cannot proceed.", "error")
                    df_for_decomp = None # Invalidate df_for_decomp
            
            if df_for_decomp is not None:
                # Equity Curve (Cumulative PnL)
                if 'cumulative_pnl' in df_for_decomp.columns:
                    equity_series_raw = df_for_decomp.set_index(date_col)['cumulative_pnl'].dropna()
                    if not equity_series_raw.empty:
                        if not equity_series_raw.index.is_monotonic_increasing:
                            equity_series_raw = equity_series_raw.sort_index()
                        series_options_decomp["Equity Curve (Cumulative PnL)"] = equity_series_raw

                # Daily PnL
                if pnl_col in df_for_decomp.columns:
                    try:
                        daily_pnl_raw = df_for_decomp.groupby(df_for_decomp[date_col].dt.normalize())[pnl_col].sum().dropna()
                        if not daily_pnl_raw.empty:
                            if not daily_pnl_raw.index.is_monotonic_increasing:
                                daily_pnl_raw = daily_pnl_raw.sort_index()
                            series_options_decomp["Daily PnL"] = daily_pnl_raw
                    except Exception as e:
                        logger.error(f"Error grouping by date for Daily PnL: {e}", exc_info=True)
                        display_custom_message(f"Could not prepare Daily PnL for decomposition due to date grouping error: {e}", "warning")
            
            if not series_options_decomp:
                st.warning("No suitable time series data (Equity Curve or Daily PnL with valid dates) is available for decomposition.")
            else:
                # Unique form key for this tab
                with st.form("decomposition_form_tab_v1"): 
                    st.markdown("Decompose a selected time series into its trend, seasonal, and residual components.")
                    
                    sel_series_name_dc = st.selectbox("Select Series for Decomposition:", list(series_options_decomp.keys()), key="dc_series_tab_v1")
                    sel_model_dc = st.selectbox("Decomposition Model:", ["additive", "multiplicative"], key="dc_model_tab_v1", help="Additive for constant seasonal variation, Multiplicative for proportional.")
                    
                    data_dc = series_options_decomp[sel_series_name_dc]
                    default_period = 7
                    if isinstance(data_dc.index, pd.DatetimeIndex):
                        inferred_freq = pd.infer_freq(data_dc.index)
                        if inferred_freq:
                            if 'D' in inferred_freq.upper(): default_period = 7
                            elif 'W' in inferred_freq.upper(): default_period = 52
                            elif 'M' in inferred_freq.upper(): default_period = 12
                    
                    min_period_dc = 2
                    max_period_dc = max(min_period_dc, (len(data_dc) // 2) - 1) if len(data_dc) > 4 else min_period_dc
                    
                    # Ensure value is within min/max bounds for number_input
                    current_value_period = min(default_period, max_period_dc) if max_period_dc >= min_period_dc else min_period_dc
                    if max_period_dc < min_period_dc: # Handle very short series where max_period could be less than min_period_dc
                        max_period_dc_input = min_period_dc
                        help_text_max_period = "N/A due to very short series"
                    else:
                        max_period_dc_input = max_period_dc
                        help_text_max_period = str(max_period_dc_input)


                    period_dc = st.number_input(
                        "Seasonal Period (Number of Observations):",
                        min_value=min_period_dc, 
                        max_value=max_period_dc_input,
                        value=current_value_period, 
                        step=1,
                        key="dc_period_tab_v1",
                        help=f"E.g., 7 for daily data with weekly seasonality. Max allowed: {help_text_max_period}."
                    )
                    submit_decomp_tab_v1 = st.form_submit_button(f"Decompose {sel_series_name_dc}")

                if submit_decomp_tab_v1:
                    data_to_decompose = series_options_decomp[sel_series_name_dc]
                    
                    if period_dc < min_period_dc or (max_period_dc < min_period_dc and period_dc > min_period_dc):
                         display_custom_message(f"The chosen seasonal period ({period_dc}) is not valid. Max allowed: {help_text_max_period}.", "error")
                    elif len(data_to_decompose.dropna()) > 2 * period_dc :
                        with st.spinner(f"Decomposing {sel_series_name_dc} ({sel_model_dc}, Period: {period_dc})..."):
                            service_output = statistical_analysis_service.get_time_series_decomposition(
                                data_to_decompose, model=sel_model_dc, period=period_dc
                            )

                        if service_output:
                            if 'error' in service_output:
                                display_custom_message(f"Decomposition Error: {service_output['error']}", "error")
                            elif 'decomposition_result' in service_output:
                                actual_result = service_output['decomposition_result']
                                if actual_result is not None and isinstance(actual_result, DecomposeResult) and hasattr(actual_result, 'observed') and not actual_result.observed.empty:
                                    st.success("Decomposition complete!")
                                    decomp_fig = plot_time_series_decomposition(
                                        actual_result, 
                                        title=f"{sel_series_name_dc} - {sel_model_dc.capitalize()} (P: {period_dc})", 
                                        theme=plot_theme
                                    )
                                    if decomp_fig: st.plotly_chart(decomp_fig, use_container_width=True)
                                    else: display_custom_message("Could not plot decomposition results.", "warning")
                                else:
                                    display_custom_message("Decomposition failed or returned empty/unexpected data. The series might be too short, lack clear patterns, or the period is too large.", "error")
                            else:
                                display_custom_message("Decomposition analysis returned an unexpected structure from service.", "error")
                        else:
                            display_custom_message("Decomposition analysis failed to return any result from service.", "error")
                    else:
                        display_custom_message(f"Not enough data for decomposition with period {period_dc}. Need > {2*period_dc} non-NaN points. Found: {len(data_to_decompose.dropna())}", "warning")

if __name__ == "__main__":
    if 'app_initialized' not in st.session_state:
        st.warning("This page is part of a multi-page app. Please run the main `app.py` script.")
    show_advanced_stats_page()
