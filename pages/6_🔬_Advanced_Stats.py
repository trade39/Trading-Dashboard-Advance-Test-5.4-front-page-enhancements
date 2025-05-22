"""
pages/6_🔬_Advanced_Stats.py

Handles the 'Advanced Statistical Analysis' page of the Streamlit application.

This module provides users with tools for more in-depth statistical examination of
their PnL (Profit and Loss) data. Key analyses include:
- Bootstrap Confidence Intervals: For various PnL metrics (mean, median, std dev, etc.),
  allowing for robust estimation of uncertainty without strong distributional assumptions.
- Time Series Decomposition: To break down PnL or equity curve data into trend,
  seasonal, and residual components, aiding in pattern identification.

The page is structured using Streamlit tabs for each distinct analysis type.
Each tab's content and logic are encapsulated in dedicated rendering functions
for improved modularity and readability (e.g., `render_bootstrap_tab`,
`render_decomposition_tab`).

Core computations are delegated to the `StatisticalAnalysisService`.
The module relies on configurations (e.g., `EXPECTED_COLUMNS`) and utility
functions shared across the application.
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
# Instantiate services once
statistical_analysis_service = StatisticalAnalysisService()

# --- Explanatory Text Content ---
# These constants hold markdown text for informational expanders.
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

# --- Tab Rendering Functions ---

def render_bootstrap_tab(
    pnl_series: pd.Series, 
    plot_theme: str, 
    service: StatisticalAnalysisService,
    default_iterations: int,
    default_confidence_level: float
) -> None:
    """
    Renders the UI and logic for the Bootstrap Confidence Intervals tab.

    This function creates an informational expander explaining bootstrap CIs,
    followed by a configuration expander where users can select a statistic,
    set parameters (iterations, confidence level), and run the analysis.
    Results, including observed statistic, CI, and a distribution plot,
    are displayed upon completion.

    Args:
        pnl_series (pd.Series): The PnL data (already cleaned of NaNs) to be used
            for bootstrapping.
        plot_theme (str): The current theme ('light' or 'dark') for plot styling.
        service (StatisticalAnalysisService): An instance of the service class
            responsible for performing the bootstrap calculation.
        default_iterations (int): The default number of bootstrap iterations
            for the number input field.
        default_confidence_level (float): The default confidence level (e.g., 0.95)
            for the slider, which will be converted to a percentage.
    """
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
    input_df: pd.DataFrame, 
    pnl_column_name: str,
    date_column_name: str,
    plot_theme: str, 
    service: StatisticalAnalysisService
) -> None:
    """
    Renders the UI and logic for the Time Series Decomposition tab.

    This function provides an explanation of time series decomposition,
    prepares the necessary time series data (Equity Curve, Daily PnL) from
    the input DataFrame, and then presents a configuration form. Users can
    select the series, decomposition model, and seasonal period.
    The decomposed components (trend, seasonal, residual) are then plotted.

    Args:
        input_df (pd.DataFrame): The filtered DataFrame containing the raw data.
        pnl_column_name (str): The name of the PnL column in `input_df`.
        date_column_name (str): The name of the date column in `input_df`.
            This column will be converted to datetime if not already.
        plot_theme (str): The current theme ('light' or 'dark') for plot styling.
        service (StatisticalAnalysisService): An instance of the service class
            responsible for performing the time series decomposition.
    """
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

# --- Main Page Function ---
def show_advanced_stats_page() -> None:
    """
    Sets up and displays the 'Advanced Statistical Analysis' page.

    This function serves as the entry point for the page. It performs initial
    data validation by checking `st.session_state` for `filtered_data`.
    It retrieves necessary column names from `EXPECTED_COLUMNS` and the current
    plot theme. If data is valid, it prepares the PnL series for analysis.

    The main content is organized into two Streamlit tabs:
    1. Bootstrap Confidence Intervals: Rendered by `render_bootstrap_tab`.
    2. Time Series Decomposition: Rendered by `render_decomposition_tab`.

    Each rendering function is passed the required data, configurations, and
    the `statistical_analysis_service` instance for performing calculations.
    """
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
    if pnl_series_for_adv.empty:
        display_custom_message("The PnL data series is empty after removing missing values. Cannot perform advanced statistical analysis.", "warning")
        return

    tab_bs_ci, tab_ts_decomp = st.tabs(["📊 Bootstrap Confidence Intervals", "📉 Time Series Decomposition"])

    with tab_bs_ci:
        render_bootstrap_tab(
            pnl_series=pnl_series_for_adv, 
            plot_theme=plot_theme, 
            service=statistical_analysis_service,
            default_iterations=BOOTSTRAP_ITERATIONS,
            default_confidence_level=CONFIDENCE_LEVEL
        )

    with tab_ts_decomp:
        render_decomposition_tab(
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
