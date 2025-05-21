"""
pages/5_🔬_Advanced_Stats.py
Advanced statistical analyses with UI/UX enhancements.
Now uses StatisticalAnalysisService for relevant methods.
"""
import streamlit as st
import pandas as pd
import numpy as np
import logging
from statsmodels.tsa.seasonal import DecomposeResult # For type hint

try:
    from config import APP_TITLE, EXPECTED_COLUMNS, BOOTSTRAP_ITERATIONS, CONFIDENCE_LEVEL
    from utils.common_utils import display_custom_message
    # MODIFICATION: Import StatisticalAnalysisService
    from services.statistical_analysis_service import StatisticalAnalysisService
    # AnalysisService might still be needed if it retains other methods used by this page.
    # For now, assuming methods used here are moved. If not, keep AnalysisService import.
    # from services.analysis_service import AnalysisService
    from plotting import plot_time_series_decomposition, plot_bootstrap_distribution_and_ci
    # statistical_methods.DISTRIBUTIONS_TO_FIT might be directly used or passed via a service method
    from statistical_methods import DISTRIBUTIONS_TO_FIT
except ImportError as e:
    st.error(f"Advanced Stats Page Error: Critical module import failed: {e}.")
    APP_TITLE = "TradingDashboard_Error"; logger = logging.getLogger(APP_TITLE)
    logger.error(f"CRITICAL IMPORT ERROR in 6_🔬_Advanced_Stats.py: {e}", exc_info=True)
    EXPECTED_COLUMNS = {"pnl": "pnl", "date": "date"}
    BOOTSTRAP_ITERATIONS = 1000; CONFIDENCE_LEVEL = 0.95
    DISTRIBUTIONS_TO_FIT = ['norm']
    def display_custom_message(msg, type="error"): st.error(msg)
    class StatisticalAnalysisService: # Dummy service
        def get_time_series_decomposition(self, *args, **kwargs): return {"error": "Service not loaded"}
        def calculate_bootstrap_ci(self, *args, **kwargs): return {"error": "Service not loaded", "lower_bound": np.nan, "upper_bound": np.nan, "observed_statistic": np.nan, "bootstrap_statistics": []} # Corrected dummy method name
    def plot_time_series_decomposition(*args, **kwargs): return None
    def plot_bootstrap_distribution_and_ci(*args, **kwargs): return None
    st.stop()

logger = logging.getLogger(APP_TITLE)
# MODIFICATION: Instantiate StatisticalAnalysisService
statistical_analysis_service = StatisticalAnalysisService()
# analysis_service = AnalysisService() # Keep if other methods from AnalysisService are used

def show_advanced_stats_page():
    st.title("🔬 Advanced Statistical Analysis")
    if 'filtered_data' not in st.session_state or st.session_state.filtered_data is None:
        display_custom_message("Upload data to access advanced statistics.", "info"); return

    filtered_df = st.session_state.filtered_data
    plot_theme = st.session_state.get('current_theme', 'dark')
    pnl_col = EXPECTED_COLUMNS.get('pnl')
    date_col = EXPECTED_COLUMNS.get('date')

    if filtered_df.empty:
        display_custom_message("No data matches filters.", "info"); return
    if not pnl_col or pnl_col not in filtered_df.columns:
        display_custom_message(f"PnL column ('{pnl_col}') not found.", "error"); return

    pnl_series_for_adv = filtered_df[pnl_col].dropna()
    if pnl_series_for_adv.empty:
        display_custom_message("PnL data empty after NaN removal.", "warning"); return

    st.subheader("Bootstrap Confidence Intervals")
    with st.expander("Configure & Run Bootstrap Analysis", expanded=True):
        stat_options_bs = {
            "Mean PnL": np.mean,
            "Median PnL": np.median,
            "PnL Std Dev": np.std,
            "PnL Skewness": pd.Series.skew,
            "PnL Kurtosis": pd.Series.kurtosis,
            "Win Rate (%)": lambda x: (np.sum(x > 0) / len(x)) * 100 if len(x) > 0 else 0.0
        }
        min_data_for_skew = 3
        min_data_for_kurtosis = 4
        available_stat_options = {}
        for name, func in stat_options_bs.items():
            if name == "PnL Skewness" and len(pnl_series_for_adv) < min_data_for_skew: continue
            if name == "PnL Kurtosis" and len(pnl_series_for_adv) < min_data_for_kurtosis: continue
            available_stat_options[name] = func

        if not available_stat_options:
            st.warning("Not enough data for any bootstrap statistic options.")
        else:
            with st.form("bootstrap_form_adv_v4"):
                selected_stat_name_bs = st.selectbox(
                    "Select Statistic for Bootstrap CI:",
                    list(available_stat_options.keys()),
                    key="bs_stat_select_adv_v4"
                )
                n_iterations_bs = st.number_input(
                    "Bootstrap Iterations:",
                    min_value=100, max_value=10000, value=BOOTSTRAP_ITERATIONS, step=100,
                    key="bs_iterations_adv_v4",
                    help="Number of resamples to generate. More iterations give more stable CIs but take longer."
                )
                conf_level_bs_percent = st.slider(
                    "Confidence Level (%):",
                    min_value=80.0, max_value=99.9, value=CONFIDENCE_LEVEL * 100, step=0.1, format="%.1f%%",
                    key="bs_conf_level_adv_v4",
                    help="The confidence level for the interval (e.g., 95%)."
                )
                run_bs_button_v4 = st.form_submit_button(f"Calculate & Plot CI for {selected_stat_name_bs}")

            if run_bs_button_v4 and selected_stat_name_bs:
                if len(pnl_series_for_adv) >= 10:
                    stat_func_to_run_bs = available_stat_options[selected_stat_name_bs]
                    actual_conf_level = conf_level_bs_percent / 100.0

                    with st.spinner(f"Bootstrapping CI for {selected_stat_name_bs}... ({n_iterations_bs} iterations)"):
                        # --- MODIFICATION: Corrected method name ---
                        bs_results = statistical_analysis_service.calculate_bootstrap_ci(
                            data_series=pnl_series_for_adv,
                            statistic_func=stat_func_to_run_bs, # Pass the actual function
                            n_iterations=n_iterations_bs,
                            confidence_level=actual_conf_level
                        )
                        # --- END MODIFICATION ---

                    if bs_results and 'error' not in bs_results:
                        st.success(f"Bootstrap for {selected_stat_name_bs} complete!")
                        obs_stat = bs_results.get('observed_statistic', np.nan)
                        lower_b = bs_results.get('lower_bound', np.nan)
                        upper_b = bs_results.get('upper_bound', np.nan)
                        bootstrap_samples = bs_results.get('bootstrap_statistics', [])

                        st.metric(label=f"Observed {selected_stat_name_bs}", value=f"{obs_stat:.4f}")
                        st.metric(label=f"{int(actual_conf_level*100)}% Confidence Interval", value=f"[{lower_b:.4f}, {upper_b:.4f}]")

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
                    display_custom_message(f"Not enough PnL data points (need at least 10) for reliable bootstrap CI for {selected_stat_name_bs}.", "warning")

    st.markdown("---")
    st.subheader("Time Series Decomposition")
    with st.form("decomposition_form_adv_v4"):
        st.markdown("Decompose a time series into trend, seasonal, and residual components.")
        series_options_decomp = {}
        if 'cumulative_pnl' in filtered_df.columns and date_col and date_col in filtered_df.columns:
            equity_series_raw = filtered_df.set_index(date_col)['cumulative_pnl'].dropna()
            if not equity_series_raw.index.is_monotonic_increasing: equity_series_raw = equity_series_raw.sort_index()
            if not equity_series_raw.empty: series_options_decomp["Equity Curve"] = equity_series_raw

        if pnl_col in filtered_df.columns and date_col and date_col in filtered_df.columns:
            daily_pnl_raw = filtered_df.groupby(filtered_df[date_col].dt.normalize())[pnl_col].sum().dropna()
            if not daily_pnl_raw.index.is_monotonic_increasing: daily_pnl_raw = daily_pnl_raw.sort_index()
            if not daily_pnl_raw.empty: series_options_decomp["Daily PnL"] = daily_pnl_raw

        submit_decomp = False
        if not series_options_decomp:
            st.warning("No suitable time series data (Equity Curve or Daily PnL) found for decomposition.")
        else:
            sel_series_name_dc = st.selectbox("Select Series for Decomposition:", list(series_options_decomp.keys()), key="dc_series_v4")
            sel_model_dc = st.selectbox("Decomposition Model:", ["additive", "multiplicative"], key="dc_model_v4", help="Additive for constant seasonal variation, Multiplicative for proportional.")
            data_dc = series_options_decomp[sel_series_name_dc]
            default_period = 7
            if isinstance(data_dc.index, pd.DatetimeIndex):
                inferred_freq = pd.infer_freq(data_dc.index)
                if inferred_freq:
                    if 'D' in inferred_freq.upper(): default_period = 7
                    elif 'W' in inferred_freq.upper(): default_period = 52
                    elif 'M' in inferred_freq.upper(): default_period = 12
            min_p, max_p = 2, max(2, len(data_dc)//2 -1) if len(data_dc) > 4 else 2
            period_dc = st.number_input(
                "Seasonal Period (number of observations):",
                min_value=min_p, max_value=max_p, value=min(default_period, max_p), step=1,
                key="dc_period_v4",
                help=f"E.g., 7 for daily data with weekly seasonality. Max allowed: {max_p}"
            )
            submit_decomp = st.form_submit_button(f"Decompose {sel_series_name_dc}")

    if 'submit_decomp' in locals() and submit_decomp and series_options_decomp: # Check if submit_decomp is defined
        data_to_decompose = series_options_decomp[sel_series_name_dc]
        if len(data_to_decompose.dropna()) > 2 * period_dc :
            with st.spinner(f"Decomposing {sel_series_name_dc} with period {period_dc}..."):
                # MODIFICATION: Call StatisticalAnalysisService
                service_output = statistical_analysis_service.get_time_series_decomposition(data_to_decompose, model=sel_model_dc, period=period_dc)

            if service_output:
                if 'error' in service_output:
                    display_custom_message(f"Decomposition Error: {service_output['error']}", "error")
                elif 'decomposition_result' in service_output:
                    actual_result = service_output['decomposition_result']
                    if actual_result is not None and isinstance(actual_result, DecomposeResult) and hasattr(actual_result, 'observed') and not actual_result.observed.empty:
                        st.success("Decomposition complete!")
                        decomp_fig = plot_time_series_decomposition(actual_result, title=f"{sel_series_name_dc} Decomposition ({sel_model_dc}, Period: {period_dc})", theme=plot_theme)
                        if decomp_fig: st.plotly_chart(decomp_fig, use_container_width=True)
                        else: display_custom_message("Could not plot decomposition results.", "warning")
                    else:
                        display_custom_message("Decomposition failed or returned empty/unexpected data from service. The series might be too short or lack clear patterns for the chosen period.", "error")
                else:
                    display_custom_message("Decomposition analysis returned an unexpected structure from service.", "error")
            else:
                display_custom_message("Decomposition analysis failed to return any result from service.", "error")
        else:
            display_custom_message(f"Not enough data points for decomposition with period {period_dc}. Need more than {2*period_dc} non-NaN observations. Current valid count: {len(data_to_decompose.dropna())}", "warning")

    # ... (Placeholders for Distribution Fitting and Change Point Detection would also use StatisticalAnalysisService) ...

if __name__ == "__main__":
    if 'app_initialized' not in st.session_state:
        st.warning("This page is part of a multi-page app. Please run the main app.py script.")
    show_advanced_stats_page()
