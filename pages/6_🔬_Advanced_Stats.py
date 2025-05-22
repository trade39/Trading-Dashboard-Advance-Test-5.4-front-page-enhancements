"""
pages/6_🔬_Advanced_Stats.py
Advanced statistical analyses with UI/UX enhancements.
Uses StatisticalAnalysisService for relevant methods.
Incorporates icons for better visual hierarchy.
Sections are organized into tabs, each with an explanatory collapsible section.
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
    from statistical_methods import DISTRIBUTIONS_TO_FIT 
except ImportError as e:
    st.error(f"Advanced Stats Page Error: Critical module import failed: {e}. Please ensure all dependencies and project files are correctly placed.")
    # Fallback definitions
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

    def display_custom_message(message, type="error"): 
        if type == "error": st.error(message)
        elif type == "warning": st.warning(message)
        else: st.info(message)
        
    st.stop()

logger = logging.getLogger(APP_TITLE)
statistical_analysis_service = StatisticalAnalysisService()

# --- Explanatory Text Content ---
# Moved here for better organization and reusability if needed
BOOTSTRAP_EXPLANATION = """
**Bootstrap Confidence Intervals** are a powerful non-parametric method to estimate the uncertainty of a statistic (like the mean, median, or standard deviation) without making strong assumptions about the underlying distribution of your data.

**How it works:**
1.  **Resampling:** It repeatedly draws random samples *with replacement* from your original dataset. Each of these "bootstrap samples" is the same size as your original data.
2.  **Statistic Calculation:** For each bootstrap sample, the statistic of interest (e.g., mean PnL) is calculated.
3.  **Distribution:** This process generates a distribution of the calculated statistics (the "bootstrap distribution").
4.  **Confidence Interval:** The confidence interval is then derived from the percentiles of this bootstrap distribution. For example, a 95% confidence interval is typically the range between the 2.5th and 97.5th percentiles of the bootstrap statistics.

**Interpretation:**
A 95% confidence interval suggests that if we were to repeat the sampling process many times, 95% of the intervals constructed would contain the true, unknown population parameter. In simpler terms, it gives a plausible range for the true value of the statistic.

**Why use it?**
-   Useful for small sample sizes.
-   Does not assume data follows a specific distribution (e.g., normal).
-   Can be applied to complex statistics where analytical solutions for CIs are difficult.
"""

DECOMPOSITION_EXPLANATION = """
**Time Series Decomposition** is a statistical method that breaks down a time series into several constituent components, typically:

1.  **Trend ($T_t$):** The long-term direction or general movement of the series. It captures whether the series is increasing, decreasing, or remaining relatively stable over time.
2.  **Seasonality ($S_t$):** Patterns that repeat over a fixed period (e.g., daily, weekly, monthly, yearly). For instance, retail sales might peak during holidays. The 'Seasonal Period' parameter defines this cycle length.
3.  **Residuals/Irregularity ($R_t$ or $E_t$):** The random, unpredictable fluctuations or "noise" remaining after the trend and seasonal components have been removed. It represents what's left unexplained by the systematic components.

**Models:**
-   **Additive Model ($Y_t = T_t + S_t + R_t$):** Assumes the seasonal fluctuations are roughly constant in magnitude, regardless of the level of the time series. Suitable when the seasonal variation does not change significantly with the trend.
-   **Multiplicative Model ($Y_t = T_t \cdot S_t \cdot R_t$):** Assumes the seasonal fluctuations are proportional to the level of the time series. Suitable when the seasonal variation increases or decreases as the trend level changes (e.g., percentage-based seasonality).

**Why use it?**
-   **Understand Patterns:** Helps identify and understand the underlying structures in your data.
-   **Forecasting:** By understanding the components, you can potentially forecast them separately and then combine them.
-   **Deseasonalization:** Removing the seasonal component can make it easier to identify the true trend.
-   **Anomaly Detection:** Unusual deviations might be more apparent in the residual component.
"""

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

    pnl_series_for_adv = filtered_df[pnl_col].dropna()
    if pnl_series_for_adv.empty:
        display_custom_message("PnL data is empty after removing missing values. Cannot perform analysis.", "warning")
        return

    tab_bs_ci, tab_ts_decomp = st.tabs(["📊 Bootstrap Confidence Intervals", "📉 Time Series Decomposition"])

    with tab_bs_ci:
        st.header("Bootstrap Confidence Intervals")
        with st.expander("What are Bootstrap Confidence Intervals?", expanded=False):
            st.markdown(BOOTSTRAP_EXPLANATION)
        
        # Main configuration expander for Bootstrap
        with st.expander("⚙️ Configure & Run Bootstrap Analysis", expanded=True):
            stat_options_bs = {
                "Mean PnL": np.mean, "Median PnL": np.median, "PnL Standard Deviation": np.std,
                "PnL Skewness": pd.Series.skew, "PnL Kurtosis": pd.Series.kurtosis,
                "Win Rate (%)": lambda x: (np.sum(x > 0) / len(x)) * 100 if len(x) > 0 else 0.0
            }
            min_data_for_skew, min_data_for_kurtosis = 3, 4
            available_stat_options = {
                name: func for name, func in stat_options_bs.items()
                if not ((name == "PnL Skewness" and len(pnl_series_for_adv) < min_data_for_skew) or \
                        (name == "PnL Kurtosis" and len(pnl_series_for_adv) < min_data_for_kurtosis))
            }

            if not available_stat_options:
                st.warning("Not enough data points to calculate any bootstrap statistics.")
            else:
                with st.form("bootstrap_form_tab_v2"): # Incremented key
                    selected_stat_name_bs = st.selectbox(
                        "Select Statistic:", list(available_stat_options.keys()), key="bs_stat_select_tab_v2"
                    )
                    n_iterations_bs = st.number_input(
                        "Iterations:", 100, 10000, BOOTSTRAP_ITERATIONS, 100, key="bs_iterations_tab_v2"
                    )
                    conf_level_bs_percent = st.slider(
                        "Confidence Level (%):", 80.0, 99.9, CONFIDENCE_LEVEL * 100, 0.1, "%.1f%%", key="bs_conf_level_tab_v2"
                    )
                    run_bs_button_tab_v2 = st.form_submit_button(f"Calculate CI for {selected_stat_name_bs}")

                if run_bs_button_tab_v2 and selected_stat_name_bs:
                    if len(pnl_series_for_adv) >= 10:
                        stat_func_to_run_bs = available_stat_options[selected_stat_name_bs]
                        actual_conf_level = conf_level_bs_percent / 100.0
                        with st.spinner(f"Bootstrapping CI for {selected_stat_name_bs}..."):
                            bs_results = statistical_analysis_service.calculate_bootstrap_ci(
                                pnl_series_for_adv, stat_func_to_run_bs, n_iterations_bs, actual_conf_level
                            )
                        if bs_results and 'error' not in bs_results:
                            st.success(f"Bootstrap for {selected_stat_name_bs} complete!")
                            obs_stat, lower_b, upper_b = bs_results.get('observed_statistic'), bs_results.get('lower_bound'), bs_results.get('upper_bound')
                            bootstrap_samples = bs_results.get('bootstrap_statistics', [])
                            
                            col1, col2 = st.columns(2)
                            col1.metric(f"Observed {selected_stat_name_bs}", f"{obs_stat:.4f}")
                            col2.metric(f"{int(actual_conf_level*100)}% CI", f"[{lower_b:.4f}, {upper_b:.4f}]")
                            
                            if bootstrap_samples:
                                bs_plot = plot_bootstrap_distribution_and_ci(
                                    bootstrap_samples, obs_stat, lower_b, upper_b, selected_stat_name_bs, plot_theme
                                )
                                if bs_plot: st.plotly_chart(bs_plot, use_container_width=True)
                                else: display_custom_message("Could not generate bootstrap plot.", "warning")
                            else: display_custom_message("No bootstrap samples for plotting.", "warning")
                        elif bs_results: display_custom_message(f"Bootstrap Error: {bs_results.get('error', 'Unknown')}", "error")
                        else: display_custom_message("Bootstrap analysis failed.", "error")
                    else:
                        display_custom_message(f"Need ≥10 PnL data points for bootstrap (found {len(pnl_series_for_adv)}).", "warning")

    with tab_ts_decomp:
        st.header("Time Series Decomposition")
        with st.expander("What is Time Series Decomposition?", expanded=False):
            st.markdown(DECOMPOSITION_EXPLANATION)

        if not date_col or date_col not in filtered_df.columns:
            display_custom_message(f"Date column ('{date_col}') required for Decomposition was not found.", "error")
        else:
            df_for_decomp = filtered_df.copy()
            if not pd.api.types.is_datetime64_any_dtype(df_for_decomp[date_col]):
                try: df_for_decomp[date_col] = pd.to_datetime(df_for_decomp[date_col])
                except Exception as e:
                    display_custom_message(f"Could not convert date column '{date_col}' to datetime: {e}.", "error")
                    df_for_decomp = None
            
            series_options_decomp = {}
            if df_for_decomp is not None:
                if 'cumulative_pnl' in df_for_decomp.columns:
                    equity_series = df_for_decomp.set_index(date_col)['cumulative_pnl'].dropna().sort_index()
                    if not equity_series.empty: series_options_decomp["Equity Curve (Cumulative PnL)"] = equity_series
                if pnl_col in df_for_decomp.columns:
                    try:
                        daily_pnl = df_for_decomp.groupby(df_for_decomp[date_col].dt.normalize())[pnl_col].sum().dropna().sort_index()
                        if not daily_pnl.empty: series_options_decomp["Daily PnL"] = daily_pnl
                    except Exception as e: logger.error(f"Error grouping for Daily PnL: {e}", exc_info=True)
            
            if not series_options_decomp:
                st.warning("No suitable time series (Equity Curve or Daily PnL with valid dates) for decomposition.")
            else:
                # Main configuration expander for Decomposition
                with st.expander("⚙️ Configure & Run Decomposition", expanded=True):
                    with st.form("decomposition_form_tab_v2"): # Incremented key
                        sel_series_name_dc = st.selectbox("Select Series:", list(series_options_decomp.keys()), key="dc_series_tab_v2")
                        sel_model_dc = st.selectbox("Model:", ["additive", "multiplicative"], key="dc_model_tab_v2")
                        
                        data_dc = series_options_decomp[sel_series_name_dc]
                        default_period = 7
                        if isinstance(data_dc.index, pd.DatetimeIndex) and (inferred_freq := pd.infer_freq(data_dc.index)):
                            if 'D' in inferred_freq.upper(): default_period = 7
                            elif 'W' in inferred_freq.upper(): default_period = 52
                            elif 'M' in inferred_freq.upper(): default_period = 12
                        
                        min_p, max_p_calc = 2, (len(data_dc) // 2) - 1 if len(data_dc) > 4 else 2
                        max_p_input = max(min_p, max_p_calc)
                        current_val_p = min(default_period, max_p_input) if max_p_input >= min_p else min_p
                        help_max_p = str(max_p_input) if max_p_input >= min_p else "N/A (short series)"

                        period_dc = st.number_input(
                            "Seasonal Period (Observations):", min_p, max_p_input, current_val_p, 1, key="dc_period_tab_v2",
                            help=f"E.g., 7 for daily data (weekly seasonality). Max: {help_max_p}."
                        )
                        submit_decomp_tab_v2 = st.form_submit_button(f"Decompose {sel_series_name_dc}")

                    if submit_decomp_tab_v2:
                        data_to_decompose = series_options_decomp[sel_series_name_dc]
                        if not (min_p <= period_dc <= max_p_input if max_p_input >= min_p else period_dc == min_p):
                            display_custom_message(f"Invalid period ({period_dc}). Max: {help_max_p}.", "error")
                        elif len(data_to_decompose.dropna()) > 2 * period_dc:
                            with st.spinner(f"Decomposing {sel_series_name_dc}..."):
                                service_output = statistical_analysis_service.get_time_series_decomposition(
                                    data_to_decompose, model=sel_model_dc, period=period_dc
                                )
                            if service_output:
                                if 'error' in service_output: display_custom_message(f"Decomposition Error: {service_output['error']}", "error")
                                elif 'decomposition_result' in service_output:
                                    actual_result = service_output['decomposition_result']
                                    if actual_result and isinstance(actual_result, DecomposeResult) and not actual_result.observed.empty:
                                        st.success("Decomposition complete!")
                                        decomp_fig = plot_time_series_decomposition(
                                            actual_result, f"{sel_series_name_dc} - {sel_model_dc.capitalize()} (P: {period_dc})", plot_theme
                                        )
                                        if decomp_fig: st.plotly_chart(decomp_fig, use_container_width=True)
                                        else: display_custom_message("Could not plot decomposition.", "warning")
                                    else: display_custom_message("Decomposition failed or returned empty/unexpected data.", "error")
                                else: display_custom_message("Decomposition: Unexpected service output.", "error")
                            else: display_custom_message("Decomposition: No service result.", "error")
                        else:
                            display_custom_message(f"Need > {2*period_dc} non-NaN points for period {period_dc} (found {len(data_to_decompose.dropna())}).", "warning")

if __name__ == "__main__":
    if 'app_initialized' not in st.session_state:
        st.warning("This page is part of a multi-page app. Please run the main `app.py` script.")
    show_advanced_stats_page()
