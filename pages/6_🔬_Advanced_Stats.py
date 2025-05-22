"""
pages/6_🔬_Advanced_Stats.py

Handles the 'Advanced Statistical Analysis' page of the Streamlit application.

This module provides users with tools for more in-depth statistical examination of
their PnL (Profit and Loss) data. Key analyses include:
- Bootstrap Confidence Intervals: For robust estimation of uncertainty.
- Time Series Decomposition: To identify trend, seasonal, and residual components.
- Distribution Fitting: To analyze the underlying distribution of PnL data.
- Change Point Detection: To identify significant structural breaks in time series.

The page is structured using Streamlit tabs for each distinct analysis type.
Each tab's content and logic are encapsulated in dedicated rendering functions
for improved modularity and readability.

Core computations are delegated to the `StatisticalAnalysisService`.
The module relies on configurations and utility functions shared across the application.
"""
import streamlit as st
import pandas as pd
import numpy as np
import logging
from statsmodels.tsa.seasonal import DecomposeResult # For type hint
from typing import List, Dict, Any, Optional, Callable # For type hints
import sys # For checking test mode if needed for specific conditions


try:
    from config import APP_TITLE, EXPECTED_COLUMNS, BOOTSTRAP_ITERATIONS, CONFIDENCE_LEVEL, DISTRIBUTIONS_TO_FIT
    from utils.common_utils import display_custom_message
    from services.statistical_analysis_service import StatisticalAnalysisService
    from plotting import (
        plot_time_series_decomposition, 
        plot_bootstrap_distribution_and_ci,
        plot_distribution_fit, 
        plot_change_points     
    )
except ImportError as e:
    st.error(f"Advanced Stats Page Error: Critical module import failed: {e}. Please ensure all dependencies and project files are correctly placed.")
    APP_TITLE = "TradingDashboard_Error"
    logger = logging.getLogger(APP_TITLE)
    logger.error(f"CRITICAL IMPORT ERROR in 6_🔬_Advanced_Stats.py: {e}", exc_info=True)
    EXPECTED_COLUMNS = {"pnl": "pnl", "date": "date"} 
    BOOTSTRAP_ITERATIONS = 1000
    CONFIDENCE_LEVEL = 0.95
    DISTRIBUTIONS_TO_FIT = ['norm', 't', 'laplace'] 
    
    class StatisticalAnalysisService: 
        def get_time_series_decomposition(self, *args, **kwargs): return {"error": "StatisticalAnalysisService not loaded"}
        def calculate_bootstrap_ci(self, *args, **kwargs): return {"error": "StatisticalAnalysisService not loaded", "lower_bound": np.nan, "upper_bound": np.nan, "observed_statistic": np.nan, "bootstrap_statistics": []}
        def analyze_pnl_distribution_fit(self, *args, **kwargs): return {"error": "Distribution fitting service not loaded."}
        def find_change_points(self, *args, **kwargs): return {"error": "Change point detection service not loaded."}
            
    def plot_time_series_decomposition(*args, **kwargs): return None
    def plot_bootstrap_distribution_and_ci(*args, **kwargs): return None
    def plot_distribution_fit(*args, **kwargs): return None 
    def plot_change_points(*args, **kwargs): return None

    def display_custom_message(message, type="error"): 
        if type == "error": st.error(message)
        elif type == "warning": st.warning(message)
        else: st.info(message)
    st.stop()

logger = logging.getLogger(APP_TITLE)
statistical_analysis_service = StatisticalAnalysisService()

# --- Explanatory Text Content ---
BOOTSTRAP_EXPLANATION = """
**Bootstrap Confidence Intervals** estimate statistic uncertainty via resampling.
*How it works:* Randomly samples data with replacement, calculates statistic for each, forms a distribution, then derives CI from percentiles.
*Interpretation:* A 95% CI suggests 95% of such intervals would contain the true population parameter.
*Why use it:* Good for small samples, no specific distribution assumption, applicable to complex statistics.
"""

DECOMPOSITION_EXPLANATION = """
**Time Series Decomposition** breaks a series into Trend ($T_t$), Seasonality ($S_t$), and Residuals ($R_t$).
*Models:* Additive ($Y_t = T_t + S_t + R_t$) for constant seasonal variation; Multiplicative ($Y_t = T_t \cdot S_t \cdot R_t$) for proportional variation.
*Why use it:* Understand patterns, aid forecasting, deseasonalize data, detect anomalies in residuals.
"""

DISTRIBUTION_FITTING_EXPLANATION = """
**Distribution Fitting** involves finding a mathematical function that best describes the probability distribution of a given dataset (e.g., your PnL returns).

**Why it's useful for PnL data:**
-   **Risk Management:** Understanding the distribution helps in estimating Value at Risk (VaR), Expected Shortfall (ES), and other risk metrics.
-   **Strategy Evaluation:** Comparing the PnL distribution to known theoretical distributions (e.g., Normal, Student's t, Skewed-t) can reveal characteristics like fat tails (leptokurtosis) or skewness, which are crucial for assessing strategy robustness.
-   **Simulation:** A fitted distribution can be used to simulate future PnL scenarios for stress testing or Monte Carlo analysis.

**Process typically involves:**
1.  Selecting candidate distributions.
2.  Estimating parameters (e.g., via Maximum Likelihood Estimation).
3.  Evaluating goodness-of-fit (e.g., Kolmogorov-Smirnov test, Q-Q plots).
"""

CHANGE_POINT_DETECTION_EXPLANATION = """
**Change Point Detection (CPD)** aims to identify time points in a series where its statistical properties (e.g., mean, variance, trend) change significantly.

**Why it's important for trading PnL or Equity Curves:**
-   **Regime Shift Identification:** Detects if a trading strategy's performance characteristics have fundamentally changed.
-   **Strategy Monitoring:** Helps flag periods where a strategy might have stopped working as expected.
-   **Adaptive Modeling:** Identified change points can be used to segment data for training adaptive models.

**Common Approaches:** PELT, Binary Segmentation, Dynamic Programming.
"""

# --- Tab Rendering Functions ---

def render_bootstrap_tab(
    pnl_series: pd.Series, plot_theme: str, service: StatisticalAnalysisService,
    default_iterations: int, default_confidence_level: float
) -> None:
    """Renders the Bootstrap Confidence Intervals tab."""
    st.header("Bootstrap Confidence Intervals")
    with st.expander("What are Bootstrap Confidence Intervals?", expanded=False):
        st.markdown(BOOTSTRAP_EXPLANATION)
    
    with st.expander("⚙️ Configure & Run Bootstrap Analysis", expanded=True):
        stat_options_bs = {
            "Mean PnL": np.mean, "Median PnL": np.median, "PnL Standard Deviation": np.std,
            "PnL Skewness": pd.Series.skew, "PnL Kurtosis": pd.Series.kurtosis,
            "Win Rate (%)": lambda x: (np.sum(x > 0) / len(x)) * 100 if len(x) > 0 else 0.0
        }
        min_data_for_skew, min_data_for_kurtosis = 3, 4
        available_stat_options = {
            name: func for name, func in stat_options_bs.items()
            if not ((name == "PnL Skewness" and len(pnl_series) < min_data_for_skew) or \
                    (name == "PnL Kurtosis" and len(pnl_series) < min_data_for_kurtosis))
        }

        if not available_stat_options:
            st.warning("Not enough data points to calculate any bootstrap statistics for the PnL series.")
            return

        # Using a more unique form key
        with st.form("bootstrap_form_adv_stats_v1"): 
            selected_stat_name_bs = st.selectbox(
                "Select Statistic:", list(available_stat_options.keys()), key="bs_stat_select_adv_stats_v1",
                help="Choose the PnL metric for which to calculate the confidence interval."
            )
            n_iterations_bs = st.number_input(
                "Iterations:", 100, 10000, default_iterations, 100, key="bs_iterations_adv_stats_v1",
                help="Number of resamples. More iterations yield more stable CIs but take longer."
            )
            conf_level_bs_percent = st.slider(
                "Confidence Level (%):", 80.0, 99.9, default_confidence_level * 100, 0.1, "%.1f%%", key="bs_conf_level_adv_stats_v1",
                help="The desired confidence level for the interval (e.g., 95%)."
            )
            run_bs_button = st.form_submit_button(f"Calculate CI for {selected_stat_name_bs}")

        if run_bs_button and selected_stat_name_bs:
            if len(pnl_series) >= 10:
                stat_func_to_run_bs = available_stat_options[selected_stat_name_bs]
                actual_conf_level = conf_level_bs_percent / 100.0
                with st.spinner(f"Bootstrapping CI for {selected_stat_name_bs}... This may take a moment for {n_iterations_bs} iterations."):
                    bs_results = service.calculate_bootstrap_ci(
                        pnl_series, stat_func_to_run_bs, n_iterations_bs, actual_conf_level
                    )
                if bs_results and 'error' not in bs_results:
                    st.success(f"Bootstrap analysis for {selected_stat_name_bs} completed successfully!")
                    obs_stat = bs_results.get('observed_statistic', np.nan)
                    lower_b = bs_results.get('lower_bound', np.nan)
                    upper_b = bs_results.get('upper_bound', np.nan)
                    bootstrap_samples = bs_results.get('bootstrap_statistics', [])
                    
                    col1, col2 = st.columns(2)
                    col1.metric(f"Observed {selected_stat_name_bs}", f"{obs_stat:.4f}")
                    col2.metric(f"{int(actual_conf_level*100)}% Confidence Interval", f"[{lower_b:.4f}, {upper_b:.4f}]")
                    
                    if bootstrap_samples:
                        bs_plot = plot_bootstrap_distribution_and_ci(
                            bootstrap_samples, obs_stat, lower_b, upper_b, selected_stat_name_bs, plot_theme
                        )
                        if bs_plot: st.plotly_chart(bs_plot, use_container_width=True)
                        else: display_custom_message("Could not generate the bootstrap distribution plot.", "warning")
                    else: display_custom_message("No bootstrap samples were returned, so the plot cannot be generated.", "warning")
                elif bs_results: display_custom_message(f"Bootstrap Error for {selected_stat_name_bs}: {bs_results.get('error', 'An unknown error occurred.')}", "error")
                else: display_custom_message(f"Bootstrap analysis for {selected_stat_name_bs} failed to return any results.", "error")
            else:
                display_custom_message(f"Insufficient PnL data points (need at least 10, found {len(pnl_series)}) for a reliable bootstrap CI for {selected_stat_name_bs}.", "warning")


def render_decomposition_tab(
    input_df: pd.DataFrame, pnl_column_name: str, date_column_name: str,
    plot_theme: str, service: StatisticalAnalysisService
) -> None:
    """Renders the Time Series Decomposition tab."""
    st.header("Time Series Decomposition")
    with st.expander("What is Time Series Decomposition?", expanded=False):
        st.markdown(DECOMPOSITION_EXPLANATION)

    if not date_column_name or date_column_name not in input_df.columns:
        display_custom_message(f"The expected Date column ('{date_column_name}') is required for Time Series Decomposition and was not found in the data.", "error")
        return
    
    df_for_decomp = input_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_for_decomp[date_column_name]):
        try: 
            df_for_decomp[date_column_name] = pd.to_datetime(df_for_decomp[date_column_name])
        except Exception as e:
            display_custom_message(f"Could not convert date column '{date_column_name}' to datetime: {e}. Decomposition cannot proceed.", "error")
            return
            
    series_options_decomp = {}
    if 'cumulative_pnl' in df_for_decomp.columns:
        equity_series = df_for_decomp.set_index(date_column_name)['cumulative_pnl'].dropna().sort_index()
        if not equity_series.empty: 
            series_options_decomp["Equity Curve (Cumulative PnL)"] = equity_series
    
    if pnl_column_name in df_for_decomp.columns:
        try:
            daily_pnl = df_for_decomp.groupby(df_for_decomp[date_column_name].dt.normalize())[pnl_column_name].sum().dropna().sort_index()
            if not daily_pnl.empty: 
                series_options_decomp["Daily PnL"] = daily_pnl
        except Exception as e: 
            logger.error(f"Error grouping by date for Daily PnL in decomposition tab: {e}", exc_info=True)
            display_custom_message(f"Could not prepare Daily PnL for decomposition due to a date grouping error: {e}", "warning")
    
    if not series_options_decomp:
        st.warning("No suitable time series (Equity Curve or Daily PnL with valid dates) could be prepared for decomposition from the provided data.")
        return

    with st.expander("⚙️ Configure & Run Decomposition", expanded=True):
        # Using a more unique form key
        with st.form("decomposition_form_adv_stats_v1"): 
            sel_series_name_dc = st.selectbox("Select Series for Decomposition:", list(series_options_decomp.keys()), key="dc_series_adv_stats_v1", help="Choose the time series data to decompose.")
            sel_model_dc = st.selectbox("Decomposition Model:", ["additive", "multiplicative"], key="dc_model_adv_stats_v1", help="Choose 'additive' for constant seasonal variation, or 'multiplicative' if it scales with the series level.")
            
            data_dc = series_options_decomp[sel_series_name_dc]
            default_period = 7
            if isinstance(data_dc.index, pd.DatetimeIndex) and (inferred_freq := pd.infer_freq(data_dc.index)):
                if 'D' in inferred_freq.upper(): default_period = 7
                elif 'W' in inferred_freq.upper(): default_period = 52 
                elif 'M' in inferred_freq.upper(): default_period = 12
            
            min_p, max_p_calc = 2, (len(data_dc) // 2) - 1 if len(data_dc) > 4 else 2
            max_p_input = max(min_p, max_p_calc)
            current_val_p = min(default_period, max_p_input) if max_p_input >= min_p else min_p
            help_max_p = str(max_p_input) if max_p_input >= min_p else "N/A (series too short)"

            period_dc = st.number_input(
                "Seasonal Period (Number of Observations):", 
                min_value=min_p, max_value=max_p_input, value=current_val_p, step=1, key="dc_period_adv_stats_v1",
                help=f"Specify the observations per seasonal cycle (e.g., 7 for daily data with weekly seasonality). Max allowed: {help_max_p}."
            )
            submit_decomp = st.form_submit_button(f"Decompose {sel_series_name_dc}")

        if submit_decomp:
            data_to_decompose = series_options_decomp[sel_series_name_dc]
            is_period_valid = (min_p <= period_dc <= max_p_input) if max_p_input >= min_p else (period_dc == min_p)

            if not is_period_valid:
                display_custom_message(f"The chosen seasonal period ({period_dc}) is not valid for the selected series. Maximum allowed is {help_max_p}.", "error")
            elif len(data_to_decompose.dropna()) > 2 * period_dc:
                with st.spinner(f"Decomposing {sel_series_name_dc} using {sel_model_dc} model with period {period_dc}..."):
                    service_output = service.get_time_series_decomposition(
                        data_to_decompose, model=sel_model_dc, period=period_dc
                    )
                if service_output:
                    if 'error' in service_output: 
                        display_custom_message(f"Decomposition Error: {service_output['error']}", "error")
                    elif 'decomposition_result' in service_output:
                        actual_result = service_output['decomposition_result']
                        if actual_result and isinstance(actual_result, DecomposeResult) and hasattr(actual_result, 'observed') and not actual_result.observed.empty:
                            st.success("Time series decomposition completed successfully!")
                            decomp_fig = plot_time_series_decomposition(
                                actual_result, 
                                title=f"{sel_series_name_dc} - {sel_model_dc.capitalize()} Decomposition (Period: {period_dc})", 
                                theme=plot_theme
                            )
                            if decomp_fig: st.plotly_chart(decomp_fig, use_container_width=True)
                            else: display_custom_message("Could not plot the decomposition results.", "warning")
                        else: 
                            display_custom_message("Decomposition failed or returned empty/unexpected data. The series might be too short, lack clear patterns for the chosen period, or the period might be too large for the dataset.", "error")
                    else: 
                        display_custom_message("Decomposition analysis returned an unexpected structure from the service.", "error")
                else: 
                    display_custom_message("Decomposition analysis failed to return any result from the service.", "error")
            else:
                display_custom_message(f"Not enough data points for decomposition with period {period_dc}. Need more than {2*period_dc} non-NaN observations. Currently have: {len(data_to_decompose.dropna())}.", "warning")


def render_distribution_fitting_tab(
    pnl_series: pd.Series, 
    plot_theme: str, 
    service: StatisticalAnalysisService,
    configured_distributions: List[str] 
) -> None:
    """Renders the UI and logic for the Distribution Fitting tab."""
    st.header("Distribution Fitting")
    with st.expander("What is Distribution Fitting?", expanded=False):
        st.markdown(DISTRIBUTION_FITTING_EXPLANATION)

    if pnl_series.empty:
        st.warning("PnL series is empty. Cannot perform distribution fitting.")
        return

    with st.expander("⚙️ Configure & Run Distribution Fitting", expanded=True):
        # Using a more unique form key
        with st.form("dist_fit_form_adv_stats_v1"):
            st.markdown("Select one or more statistical distributions to fit to your PnL data.")
            distributions_to_attempt = st.multiselect(
                "Select distributions to fit:",
                options=configured_distributions,
                default=[dist for dist in ['norm', 't'] if dist in configured_distributions], 
                key="dist_fit_select_adv_stats_v1",
                help="Choose distributions to model the PnL data. Results will include parameters and goodness-of-fit."
            )
            run_dist_fit_button = st.form_submit_button("Fit Selected Distributions")
        
        if run_dist_fit_button and distributions_to_attempt:
            with st.spinner(f"Fitting distributions: {', '.join(distributions_to_attempt)}..."):
                # The service method `analyze_pnl_distribution_fit` already takes `distributions_to_try`
                fit_results_dict = service.analyze_pnl_distribution_fit(pnl_series, distributions_to_attempt)
            
            if fit_results_dict and 'error' not in fit_results_dict:
                st.success("Distribution fitting complete!")
                results_data = []
                valid_fits_for_plot_selection = []

                for dist_name, result in fit_results_dict.items():
                    if 'error' in result:
                        results_data.append({
                            "Distribution": dist_name, "Parameters": "Error", 
                            "KS Statistic": "N/A", "KS p-value": "N/A", 
                            "Interpretation": result['error']
                        })
                    else:
                        # Ensure param_names are available, provide fallback if not
                        param_names = result.get("param_names", [f'param_{i+1}' for i in range(len(result.get("params",[])))])
                        params_str = ", ".join([f"{pn}={pv:.4f}" for pn, pv in zip(param_names, result.get("params", []))])
                        
                        results_data.append({
                            "Distribution": dist_name,
                            "Parameters": params_str,
                            "KS Statistic": f"{result.get('ks_statistic', np.nan):.4f}",
                            "KS p-value": f"{result.get('ks_p_value', np.nan):.4f}",
                            "Interpretation": result.get("interpretation", "N/A")
                        })
                        valid_fits_for_plot_selection.append(dist_name)
                
                if results_data:
                    st.subheader("Goodness-of-Fit Results")
                    st.dataframe(pd.DataFrame(results_data), use_container_width=True)

                if valid_fits_for_plot_selection:
                    st.subheader("Visualize Fit")
                    # Use a unique key for the selectbox
                    dist_to_plot = st.selectbox(
                        "Select a fitted distribution to visualize:",
                        options=valid_fits_for_plot_selection,
                        key="dist_plot_select_adv_stats_v1" 
                    )
                    if dist_to_plot and dist_to_plot in fit_results_dict and 'params' in fit_results_dict[dist_to_plot]:
                        selected_fit_details = fit_results_dict[dist_to_plot]
                        dist_plot_fig = plot_distribution_fit(
                            pnl_series=pnl_series,
                            dist_name=dist_to_plot,
                            fit_params=tuple(selected_fit_details['params']), # Ensure params is a tuple
                            gof_stats=selected_fit_details, 
                            theme=plot_theme
                        )
                        if dist_plot_fig:
                            st.plotly_chart(dist_plot_fig, use_container_width=True)
                        else:
                            display_custom_message(f"Could not generate plot for {dist_to_plot}.", "warning")
                elif not results_data: # No results data at all
                    st.info("No distribution fitting results to display.")
                else: # Results data exists, but no valid fits for plotting
                    st.warning("No distributions were successfully fitted to allow visualization.")


            elif fit_results_dict and 'error' in fit_results_dict:
                 display_custom_message(f"Distribution fitting service error: {fit_results_dict['error']}", "error")
            else:
                display_custom_message("Distribution fitting failed to return results or an unexpected error occurred.", "error")
        elif run_dist_fit_button and not distributions_to_attempt:
            st.warning("Please select at least one distribution to fit.")


def render_change_point_detection_tab(
    input_df: pd.DataFrame, 
    pnl_column_name: str, 
    date_column_name: str,
    plot_theme: str, 
    service: StatisticalAnalysisService
) -> None:
    """Renders the UI and logic for the Change Point Detection tab."""
    st.header("Change Point Detection")
    with st.expander("What is Change Point Detection?", expanded=False):
        st.markdown(CHANGE_POINT_DETECTION_EXPLANATION)

    cpd_series_options = {}
    df_for_cpd = input_df.copy()
    
    if date_column_name and date_column_name in df_for_cpd.columns:
        if not pd.api.types.is_datetime64_any_dtype(df_for_cpd[date_column_name]):
            try: df_for_cpd[date_column_name] = pd.to_datetime(df_for_cpd[date_column_name])
            except Exception: df_for_cpd = None 

        if df_for_cpd is not None:
            if pnl_column_name and pnl_column_name in df_for_cpd.columns:
                pnl_ts = df_for_cpd.set_index(date_column_name)[pnl_column_name].dropna()
                if not pnl_ts.empty: cpd_series_options[f"PnL ('{pnl_column_name}')"] = pnl_ts
            
            if 'cumulative_pnl' in df_for_cpd.columns:
                equity_ts = df_for_cpd.set_index(date_column_name)['cumulative_pnl'].dropna()
                if not equity_ts.empty: cpd_series_options["Equity Curve (Cumulative PnL)"] = equity_ts
    
    if not cpd_series_options:
        st.warning("No suitable time series (PnL or Equity Curve with valid dates) available for Change Point Detection.")
        return

    with st.expander("⚙️ Configure & Run Change Point Detection", expanded=True):
        # Using a more unique form key
        with st.form("cpd_form_adv_stats_v1"):
            selected_series_name_cpd = st.selectbox(
                "Select time series for analysis:",
                options=list(cpd_series_options.keys()),
                key="cpd_series_select_adv_stats_v1"
            )
            
            cpd_model = st.selectbox(
                "Detection Model (Cost Function):", 
                options=["l1", "l2", "rbf", "linear", "normal", "ar"], 
                index=1, key="cpd_model_adv_stats_v1",
                help="The cost function to detect changes (e.g., 'l2' for changes in mean)."
            )
            
            detection_method = st.radio(
                "Breakpoint Specification:",
                ("Automatic (Penalty-based)", "Fixed Number of Breakpoints"),
                key="cpd_detection_method_adv_stats_v1"
            )

            penalty_value_cpd: Any = "bic" 
            n_breakpoints_cpd: Optional[int] = None

            if detection_method == "Automatic (Penalty-based)":
                penalty_value_cpd = st.selectbox(
                    "Penalty Method:", 
                    options=["bic", "aic", "mbic", "custom_float"], 
                    index=0, key="cpd_penalty_adv_stats_v1",
                    help="BIC/AIC/MBIC are common. 'custom_float' allows manual float input."
                )
                if penalty_value_cpd == "custom_float":
                    penalty_value_cpd = st.number_input("Custom Penalty Value (float):", min_value=0.1, value=10.0, step=0.1, key="cpd_custom_penalty_val_adv_stats")
            else:
                n_breakpoints_cpd = st.number_input(
                    "Number of Change Points to Detect:", 
                    min_value=1, value=3, step=1, key="cpd_n_bkps_adv_stats_v1"
                )
            
            min_segment_size_cpd = st.number_input(
                "Minimum Segment Size:",
                min_value=2, value=5, step=1, key="cpd_min_size_adv_stats_v1",
                help="Minimum number of data points between change points."
            )
            run_cpd_button = st.form_submit_button("Detect Change Points")

        if run_cpd_button and selected_series_name_cpd:
            series_to_analyze_cpd = cpd_series_options[selected_series_name_cpd]

            if series_to_analyze_cpd.empty:
                st.warning(f"The selected series '{selected_series_name_cpd}' is empty after processing.")
                return

            with st.spinner(f"Detecting change points in '{selected_series_name_cpd}'..."):
                cpd_results = service.find_change_points(
                    series=series_to_analyze_cpd,
                    model=cpd_model,
                    penalty=penalty_value_cpd if detection_method == "Automatic (Penalty-based)" else None,
                    n_bkps=n_breakpoints_cpd if detection_method == "Fixed Number of Breakpoints" else None,
                    min_size=min_segment_size_cpd
                )
            
            if cpd_results and 'error' not in cpd_results:
                st.success("Change point detection complete!")
                # The service returns 'change_points_original_indices' which are actual index values
                change_points_locations = cpd_results.get("change_points_original_indices", []) 
                
                if change_points_locations:
                    st.write(f"Detected {len(change_points_locations)} change point(s) at:")
                    # Display dates if index is DatetimeIndex, otherwise display the raw index values
                    if isinstance(series_to_analyze_cpd.index, pd.DatetimeIndex):
                        st.write([loc.strftime('%Y-%m-%d %H:%M:%S') if isinstance(loc, pd.Timestamp) else str(loc) for loc in change_points_locations])
                    else:
                        st.write(change_points_locations)
                    
                    cpd_plot_fig = plot_change_points(
                        time_series_data=cpd_results.get('series_to_plot', series_to_analyze_cpd), # Use series from results if available
                        change_points_locations=change_points_locations, # Pass the actual locations
                        series_name=selected_series_name_cpd,
                        title=f"Change Points in {selected_series_name_cpd}",
                        theme=plot_theme
                    )
                    if cpd_plot_fig:
                        st.plotly_chart(cpd_plot_fig, use_container_width=True)
                    else:
                        display_custom_message("Could not generate change point plot.", "warning")
                else:
                    st.info("No significant change points were detected with the current settings.")
            elif cpd_results and 'error' in cpd_results:
                display_custom_message(f"Change Point Detection Error: {cpd_results['error']}", "error")
            else:
                display_custom_message("Change point detection failed to return results or an unexpected error occurred.", "error")


# --- Main Page Function ---
def show_advanced_stats_page() -> None:
    """Sets up and displays the 'Advanced Statistical Analysis' page."""
    st.title("🔬 Advanced Statistical Analysis")
    
    if 'filtered_data' not in st.session_state or st.session_state.filtered_data is None:
        display_custom_message("Please upload and process your data on the 'Data Upload & Preprocessing' page first to access advanced statistical analyses.", "info")
        return

    filtered_df = st.session_state.filtered_data
    plot_theme = st.session_state.get('current_theme', 'dark')
    
    pnl_col = EXPECTED_COLUMNS.get('pnl')
    date_col = EXPECTED_COLUMNS.get('date')

    if filtered_df.empty:
        display_custom_message("The filtered data is currently empty. Please adjust your filters or upload new data.", "info")
        return
    if not pnl_col or pnl_col not in filtered_df.columns:
        display_custom_message(f"The expected PnL column ('{pnl_col}') as defined in configuration was not found in the uploaded data.", "error")
        return

    pnl_series_for_adv = filtered_df[pnl_col].dropna()
    
    tab_titles = [
        "📊 Bootstrap CI", 
        "📉 Time Series Decomposition",
        "⚙️ Distribution Fitting",
        "⚠️ Change Point Detection"
    ]
    tab_bs_ci, tab_ts_decomp, tab_dist_fit, tab_cpd = st.tabs(tab_titles)

    with tab_bs_ci:
        if pnl_series_for_adv.empty:
             display_custom_message("PnL data is empty. Bootstrap CI cannot be calculated.", "warning")
        else:
            render_bootstrap_tab(
                pnl_series=pnl_series_for_adv, plot_theme=plot_theme, service=statistical_analysis_service,
                default_iterations=BOOTSTRAP_ITERATIONS, default_confidence_level=CONFIDENCE_LEVEL
            )

    with tab_ts_decomp:
        render_decomposition_tab(
            input_df=filtered_df, pnl_column_name=pnl_col, date_column_name=date_col,
            plot_theme=plot_theme, service=statistical_analysis_service
        )

    with tab_dist_fit:
        if pnl_series_for_adv.empty:
             display_custom_message("PnL data is empty. Distribution Fitting cannot be performed.", "warning")
        else:
            render_distribution_fitting_tab(
                pnl_series=pnl_series_for_adv, 
                plot_theme=plot_theme, 
                service=statistical_analysis_service,
                configured_distributions=DISTRIBUTIONS_TO_FIT # Ensure this is correctly imported/defined
            )

    with tab_cpd:
        render_change_point_detection_tab(
            input_df=filtered_df,
            pnl_column_name=pnl_col, 
            date_column_name=date_col,
            plot_theme=plot_theme,
            service=statistical_analysis_service
        )

if __name__ == "__main__":
    if 'app_initialized' not in st.session_state:
        st.warning("This page is designed to be part of a multi-page Streamlit application. Please run the main `app.py` script for the full experience.")
    show_advanced_stats_page()
