"""
pages/6_🔬_Advanced_Stats.py

Handles the 'Advanced Statistical Analysis' page of the Streamlit application.

This module provides users with tools for more in-depth statistical examination of
their PnL (Profit and Loss) data. Key analyses include:
- Bootstrap Confidence Intervals: For robust estimation of uncertainty.
- Time Series Decomposition: To identify trend, seasonal, and residual components.
- Distribution Fitting (Placeholder): To analyze the underlying distribution of PnL data.
- Change Point Detection (Placeholder): To identify significant structural breaks in time series.

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

try:
    from config import APP_TITLE, EXPECTED_COLUMNS, BOOTSTRAP_ITERATIONS, CONFIDENCE_LEVEL
    from utils.common_utils import display_custom_message
    from services.statistical_analysis_service import StatisticalAnalysisService
    from plotting import plot_time_series_decomposition, plot_bootstrap_distribution_and_ci
    # For Distribution Fitting, you might need specific distributions from scipy.stats
    # from scipy import stats as st_scipy # Alias to avoid conflict with streamlit as st
    from statistical_methods import DISTRIBUTIONS_TO_FIT 
except ImportError as e:
    st.error(f"Advanced Stats Page Error: Critical module import failed: {e}. Please ensure all dependencies and project files are correctly placed.")
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
        # Placeholder for new service methods
        # def fit_distributions_to_data(self, *args, **kwargs):
        #     return {"error": "Distribution fitting service not implemented."}
        # def detect_change_points_in_series(self, *args, **kwargs):
        #     return {"error": "Change point detection service not implemented."}
            
    def plot_time_series_decomposition(*args, **kwargs): return None
    def plot_bootstrap_distribution_and_ci(*args, **kwargs): return None
    # def plot_distribution_fit(*args, **kwargs): return None # Placeholder for new plot
    # def plot_change_points(*args, **kwargs): return None # Placeholder for new plot

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
""" # Condensed for brevity in example

DECOMPOSITION_EXPLANATION = """
**Time Series Decomposition** breaks a series into Trend ($T_t$), Seasonality ($S_t$), and Residuals ($R_t$).
*Models:* Additive ($Y_t = T_t + S_t + R_t$) for constant seasonal variation; Multiplicative ($Y_t = T_t \cdot S_t \cdot R_t$) for proportional variation.
*Why use it:* Understand patterns, aid forecasting, deseasonalize data, detect anomalies in residuals.
""" # Condensed

DISTRIBUTION_FITTING_EXPLANATION = """
**Distribution Fitting** involves finding a mathematical function that best describes the probability distribution of a given dataset (e.g., your PnL returns).

**Why it's useful for PnL data:**
-   **Risk Management:** Understanding the distribution helps in estimating Value at Risk (VaR), Expected Shortfall (ES), and other risk metrics.
-   **Strategy Evaluation:** Comparing the PnL distribution to known theoretical distributions (e.g., Normal, Student's t, Skewed-t) can reveal characteristics like fat tails (leptokurtosis) or skewness, which are crucial for assessing strategy robustness.
-   **Simulation:** A fitted distribution can be used to simulate future PnL scenarios for stress testing or Monte Carlo analysis.
-   **Parameter Estimation:** Provides estimates for distribution parameters (e.g., mean, standard deviation, degrees of freedom, skewness parameter).

**Common Distributions for Financial Returns:**
-   Normal (Gaussian)
-   Student's t (captures fatter tails than normal)
-   Skewed Student's t (captures both skewness and fat tails)
-   Generalized Error Distribution (GED)

**Process typically involves:**
1.  Selecting a set of candidate distributions.
2.  Estimating parameters for each candidate distribution using methods like Maximum Likelihood Estimation (MLE).
3.  Evaluating goodness-of-fit using statistical tests (e.g., Kolmogorov-Smirnov, Anderson-Darling) and visual inspection (e.g., Q-Q plots, P-P plots, histograms with PDF overlay).
"""

CHANGE_POINT_DETECTION_EXPLANATION = """
**Change Point Detection (CPD)**, also known as structural break detection, aims to identify time points in a series where its statistical properties (e.g., mean, variance, trend, seasonality) change significantly.

**Why it's important for trading PnL or Equity Curves:**
-   **Regime Shift Identification:** Detects if a trading strategy's performance characteristics have fundamentally changed, possibly due to evolving market conditions, model decay, or external shocks.
-   **Strategy Monitoring:** Helps in automatically flagging periods where a strategy might have stopped working as expected or started behaving differently.
-   **Performance Attribution:** Understanding when and how performance changed can aid in attributing those changes to specific market events or strategy adjustments.
-   **Adaptive Modeling:** Identified change points can be used to segment data for training adaptive models or to adjust strategy parameters.

**Common Approaches:**
-   **Offline Methods:** Analyze the entire historical series at once (e.g., PELT, Binary Segmentation).
-   **Online Methods:** Process data sequentially and detect changes as new data arrives (e.g., CUSUM, Bayesian Online Change Point Detection).

**Considerations:**
-   The choice of method depends on the type of change expected (e.g., change in mean, variance, or more complex model parameters).
-   Sensitivity of the detection algorithm (avoiding too many false positives or missing true change points).
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
        # ... (existing bootstrap UI and logic remains here) ...
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

        with st.form("bootstrap_form_tab_v3"): 
            selected_stat_name_bs = st.selectbox(
                "Select Statistic:", list(available_stat_options.keys()), key="bs_stat_select_tab_v3",
                help="Choose the PnL metric for which to calculate the confidence interval."
            )
            n_iterations_bs = st.number_input(
                "Iterations:", 100, 10000, default_iterations, 100, key="bs_iterations_tab_v3",
                help="Number of resamples. More iterations yield more stable CIs but take longer."
            )
            conf_level_bs_percent = st.slider(
                "Confidence Level (%):", 80.0, 99.9, default_confidence_level * 100, 0.1, "%.1f%%", key="bs_conf_level_tab_v3",
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
    # ... (existing decomposition UI and logic remains here) ...
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
        with st.form("decomposition_form_tab_v3"): 
            sel_series_name_dc = st.selectbox("Select Series for Decomposition:", list(series_options_decomp.keys()), key="dc_series_tab_v3", help="Choose the time series data to decompose.")
            sel_model_dc = st.selectbox("Decomposition Model:", ["additive", "multiplicative"], key="dc_model_tab_v3", help="Choose 'additive' for constant seasonal variation, or 'multiplicative' if it scales with the series level.")
            
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
                min_value=min_p, max_value=max_p_input, value=current_val_p, step=1, key="dc_period_tab_v3",
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
    available_distributions: list # e.g., from statistical_methods.DISTRIBUTIONS_TO_FIT
) -> None:
    """
    Renders the UI and logic for the Distribution Fitting tab. (Placeholder)

    Args:
        pnl_series (pd.Series): The PnL data (cleaned of NaNs) to be analyzed.
        plot_theme (str): The current theme for plot styling.
        service (StatisticalAnalysisService): Service for distribution fitting.
        available_distributions (list): List of distribution names to offer for fitting.
    """
    st.header("Distribution Fitting")
    with st.expander("What is Distribution Fitting?", expanded=False):
        st.markdown(DISTRIBUTION_FITTING_EXPLANATION)

    st.info("📊 **Distribution Fitting Analysis:** This section is under development. Future enhancements will allow you to fit various statistical distributions to your PnL data, helping to understand its underlying characteristics and for risk modeling.")
    
    if pnl_series.empty:
        st.warning("PnL series is empty. Cannot perform distribution fitting.")
        return

    # Placeholder for future UI and logic
    # Example:
    # with st.expander("⚙️ Configure & Run Distribution Fitting", expanded=True):
    #     with st.form("dist_fit_form_tab_v1"):
    #         selected_dists = st.multiselect(
    #             "Select distributions to fit:",
    #             options=available_distributions, # e.g., ['norm', 't', 'skewnorm']
    #             default=[dist for dist in ['norm', 't'] if dist in available_distributions], # Sensible defaults
    #             key="dist_fit_select_v1"
    #         )
    #         run_dist_fit_button = st.form_submit_button("Fit Selected Distributions")
        
    #     if run_dist_fit_button and selected_dists:
    #         with st.spinner(f"Fitting distributions: {', '.join(selected_dists)}..."):
    #             # fit_results = service.fit_distributions_to_data(pnl_series, selected_dists)
    #             # if fit_results and 'error' not in fit_results:
    #             #     st.success("Distribution fitting complete!")
    #             #     # Display results: parameters, goodness-of-fit stats, plots (histogram with PDF, Q-Q plot)
    #             #     for dist_name, params in fit_results.get("fitted_params", {}).items():
    #             #         st.subheader(f"Results for {dist_name}:")
    #             #         st.write(f"Parameters: {params}")
    #             #         # ... display GoF stats ...
    #             #         # ... plot_distribution_fit(pnl_series, dist_name, params, plot_theme) ...
    #             # else:
    #             #     display_custom_message(f"Distribution fitting error: {fit_results.get('error', 'Unknown')}", "error")
    #             display_custom_message("Distribution fitting logic not yet implemented.", "info")


def render_change_point_detection_tab(
    input_df: pd.DataFrame, 
    target_column_name: str, # Could be 'pnl' or 'cumulative_pnl'
    date_column_name: str,
    plot_theme: str, 
    service: StatisticalAnalysisService
) -> None:
    """
    Renders the UI and logic for the Change Point Detection tab. (Placeholder)

    Args:
        input_df (pd.DataFrame): DataFrame containing the series to analyze.
        target_column_name (str): Name of the column for CPD (e.g., 'pnl', 'cumulative_pnl').
        date_column_name (str): Name of the date column for time series context.
        plot_theme (str): The current theme for plot styling.
        service (StatisticalAnalysisService): Service for change point detection.
    """
    st.header("Change Point Detection")
    with st.expander("What is Change Point Detection?", expanded=False):
        st.markdown(CHANGE_POINT_DETECTION_EXPLANATION)

    st.info("⚠️ **Change Point Detection Analysis:** This feature is planned. It will help identify significant structural breaks or regime shifts in your selected time series data (e.g., PnL or Equity Curve).")

    if input_df.empty or target_column_name not in input_df.columns:
        st.warning(f"Required data ('{target_column_name}') not available for Change Point Detection.")
        return
        
    # Placeholder for future UI and logic
    # Example:
    # series_to_analyze = input_df.set_index(date_column_name)[target_column_name].dropna()
    # if series_to_analyze.empty:
    #     st.warning(f"The selected series '{target_column_name}' is empty after processing.")
    #     return

    # with st.expander("⚙️ Configure & Run Change Point Detection", expanded=True):
    #     with st.form("cpd_form_tab_v1"):
    #         # UI elements for selecting CPD method, parameters (e.g., penalty, number of change points)
    #         cpd_method = st.selectbox("Select CPD Method:", ["PELT", "BinSeg", "DynamicProgramming"], key="cpd_method_v1") # Example methods
    #         # ... other parameters ...
    #         run_cpd_button = st.form_submit_button("Detect Change Points")

    #     if run_cpd_button:
    #         with st.spinner(f"Detecting change points using {cpd_method}..."):
    #             # cpd_results = service.detect_change_points_in_series(series_to_analyze, method=cpd_method, ...)
    #             # if cpd_results and 'error' not in cpd_results:
    #             #     st.success("Change point detection complete!")
    #             #     change_points_indices = cpd_results.get("change_points", [])
    #             #     st.write(f"Detected change points at indices: {change_points_indices}")
    #             #     # Convert indices to dates if possible
    #             #     # ... plot_change_points(series_to_analyze, change_points_indices, plot_theme) ...
    #             # else:
    #             #     display_custom_message(f"CPD error: {cpd_results.get('error', 'Unknown')}", "error")
    #             display_custom_message("Change point detection logic not yet implemented.", "info")


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
    if pnl_series_for_adv.empty and not (len(sys.argv) > 1 and sys.argv[1] == 'streamlit_test_mode'): # Allow empty for testing
        display_custom_message("The PnL data series is empty after removing missing values. Cannot perform advanced statistical analysis.", "warning")
        # For some tabs like CPD, we might still want to proceed if other series (e.g. equity curve) are available.
        # The individual tab rendering functions will handle specific data needs.
    
    # Create tabs - Add new analysis tabs here
    tab_titles = [
        "📊 Bootstrap CI", 
        "📉 Time Series Decomposition",
        "⚙️ Distribution Fitting", # New Tab
        "⚠️ Change Point Detection"  # New Tab
    ]
    tab_bs_ci, tab_ts_decomp, tab_dist_fit, tab_cpd = st.tabs(tab_titles)

    with tab_bs_ci:
        if pnl_series_for_adv.empty: # Specific check for this tab
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

    with tab_dist_fit: # New Distribution Fitting Tab
        if pnl_series_for_adv.empty: # Specific check for this tab
             display_custom_message("PnL data is empty. Distribution Fitting cannot be performed.", "warning")
        else:
            render_distribution_fitting_tab(
                pnl_series=pnl_series_for_adv, 
                plot_theme=plot_theme, 
                service=statistical_analysis_service,
                available_distributions=DISTRIBUTIONS_TO_FIT # Pass configured distributions
            )

    with tab_cpd: # New Change Point Detection Tab
        # CPD might analyze PnL or Equity Curve, so pass filtered_df
        render_change_point_detection_tab(
            input_df=filtered_df,
            target_column_name=pnl_col, # Default to PnL, could be selectable later
            date_column_name=date_col,
            plot_theme=plot_theme,
            service=statistical_analysis_service
        )
import sys # For checking test mode if needed for specific conditions

if __name__ == "__main__":
    if 'app_initialized' not in st.session_state:
        st.warning("This page is designed to be part of a multi-page Streamlit application. Please run the main `app.py` script for the full experience.")
    show_advanced_stats_page()
