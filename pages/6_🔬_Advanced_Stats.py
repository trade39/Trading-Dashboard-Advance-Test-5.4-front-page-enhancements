"""
pages/6_🔬_Advanced_Stats.py
Advanced statistical analyses with UI/UX enhancements.
Now uses StatisticalAnalysisService for relevant methods.
Incorporates icons for better visual hierarchy.
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
    from statistical_methods import DISTRIBUTIONS_TO_FIT # Though not used in the current snippet, kept for completeness if used elsewhere
except ImportError as e:
    st.error(f"Advanced Stats Page Error: Critical module import failed: {e}. Please ensure all dependencies and project files are correctly placed.")
    # Fallback definitions for critical error
    APP_TITLE = "TradingDashboard_Error"
    logger = logging.getLogger(APP_TITLE)
    logger.error(f"CRITICAL IMPORT ERROR in 6_🔬_Advanced_Stats.py: {e}", exc_info=True)
    EXPECTED_COLUMNS = {"pnl": "pnl", "date": "date"} # Minimal fallback
    BOOTSTRAP_ITERATIONS = 1000
    CONFIDENCE_LEVEL = 0.95
    DISTRIBUTIONS_TO_FIT = ['norm'] # Minimal fallback
    
    # Dummy service and plotting functions for graceful failure
    class StatisticalAnalysisService:
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
        st.error(message)
        
    st.stop() # Stop execution if critical imports fail

logger = logging.getLogger(APP_TITLE)
statistical_analysis_service = StatisticalAnalysisService()

def show_advanced_stats_page():
    st.title("🔬 Advanced Statistical Analysis")
    
    # Ensure data is loaded and available
    if 'filtered_data' not in st.session_state or st.session_state.filtered_data is None:
        display_custom_message("Please upload and process your data first to access advanced statistical analyses.", "info")
        return

    filtered_df = st.session_state.filtered_data
    plot_theme = st.session_state.get('current_theme', 'dark') # Default to dark if not set
    
    # Ensure essential columns are defined and present
    pnl_col = EXPECTED_COLUMNS.get('pnl')
    date_col = EXPECTED_COLUMNS.get('date')

    if filtered_df.empty:
        display_custom_message("The filtered data is empty. Please adjust your filters or upload new data.", "info")
        return
    if not pnl_col or pnl_col not in filtered_df.columns:
        display_custom_message(f"The expected PnL column ('{pnl_col}') was not found in the data.", "error")
        return
    if not date_col or date_col not in filtered_df.columns:
        display_custom_message(f"The expected Date column ('{date_col}') was not found in the data.", "error")
        # Continue if date_col is missing but not strictly needed for all analyses, or handle appropriately
        # For decomposition, date_col is crucial.

    pnl_series_for_adv = filtered_df[pnl_col].dropna()
    if pnl_series_for_adv.empty:
        display_custom_message("PnL data is empty after removing missing values. Cannot perform analysis.", "warning")
        return

    # --- Bootstrap Confidence Intervals Section ---
    st.subheader("📊 Bootstrap Confidence Intervals") # Icon added
    with st.expander("Configure & Run Bootstrap Analysis", expanded=True):
        # Define statistic options for bootstrap
        stat_options_bs = {
            "Mean PnL": np.mean,
            "Median PnL": np.median,
            "PnL Standard Deviation": np.std,
            "PnL Skewness": pd.Series.skew,
            "PnL Kurtosis": pd.Series.kurtosis,
            "Win Rate (%)": lambda x: (np.sum(x > 0) / len(x)) * 100 if len(x) > 0 else 0.0
        }
        
        # Filter available statistics based on data size
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
            with st.form("bootstrap_form_adv_v5"): # Incremented form key
                selected_stat_name_bs = st.selectbox(
                    "Select Statistic for Bootstrap CI:",
                    list(available_stat_options.keys()),
                    key="bs_stat_select_adv_v5",
                    help="Choose the metric for which to calculate the confidence interval."
                )
                n_iterations_bs = st.number_input(
                    "Number of Bootstrap Iterations:",
                    min_value=100, max_value=10000, value=BOOTSTRAP_ITERATIONS, step=100,
                    key="bs_iterations_adv_v5",
                    help="More iterations lead to more stable CIs but increase computation time."
                )
                conf_level_bs_percent = st.slider(
                    "Confidence Level (%):",
                    min_value=80.0, max_value=99.9, value=CONFIDENCE_LEVEL * 100, step=0.1, format="%.1f%%",
                    key="bs_conf_level_adv_v5",
                    help="The desired confidence level for the interval (e.g., 95%)."
                )
                run_bs_button_v5 = st.form_submit_button(f"Calculate & Plot CI for {selected_stat_name_bs}")

            if run_bs_button_v5 and selected_stat_name_bs:
                if len(pnl_series_for_adv) >= 10: # Minimum data points for meaningful bootstrap
                    stat_func_to_run_bs = available_stat_options[selected_stat_name_bs]
                    actual_conf_level = conf_level_bs_percent / 100.0

                    with st.spinner(f"Bootstrapping CI for {selected_stat_name_bs}... This may take a moment for {n_iterations_bs} iterations."):
                        bs_results = statistical_analysis_service.calculate_bootstrap_ci(
                            data_series=pnl_series_for_adv,
                            statistic_func=stat_func_to_run_bs,
                            n_iterations=n_iterations_bs,
                            confidence_level=actual_conf_level
                        )

                    if bs_results and 'error' not in bs_results:
                        st.success(f"Bootstrap analysis for {selected_stat_name_bs} completed successfully!")
                        obs_stat = bs_results.get('observed_statistic', np.nan)
                        lower_b = bs_results.get('lower_bound', np.nan)
                        upper_b = bs_results.get('upper_bound', np.nan)
                        bootstrap_samples = bs_results.get('bootstrap_statistics', [])

                        # Display metrics
                        col1_bs, col2_bs = st.columns(2)
                        with col1_bs:
                            st.metric(label=f"Observed {selected_stat_name_bs}", value=f"{obs_stat:.4f}")
                        with col2_bs:
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
                            if bs_plot:
                                st.plotly_chart(bs_plot, use_container_width=True)
                            else:
                                display_custom_message("Could not generate the bootstrap distribution plot.", "warning")
                        else:
                            display_custom_message("No bootstrap samples were returned, so the distribution plot cannot be generated.", "warning")
                    elif bs_results: # Handle specific error from service
                        display_custom_message(f"Bootstrap Error for {selected_stat_name_bs}: {bs_results.get('error', 'An unknown error occurred during bootstrap.')}", "error")
                    else: # Handle case where service returns None or empty
                        display_custom_message(f"Bootstrap analysis for {selected_stat_name_bs} failed to return any results.", "error")
                else:
                    display_custom_message(f"Insufficient PnL data points (need at least 10, found {len(pnl_series_for_adv)}) for a reliable bootstrap CI for {selected_stat_name_bs}.", "warning")

    st.markdown("---") # Visual separator

    # --- Time Series Decomposition Section ---
    st.subheader("📉 Time Series Decomposition") # Icon added
    
    # Prepare series for decomposition
    series_options_decomp = {}
    if date_col and date_col in filtered_df.columns: # Date column is essential here
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(filtered_df[date_col]):
            try:
                filtered_df[date_col] = pd.to_datetime(filtered_df[date_col])
            except Exception as e:
                display_custom_message(f"Could not convert date column '{date_col}' to datetime: {e}", "error")
                # Potentially stop or disable decomposition if date conversion fails
        
        # Equity Curve (Cumulative PnL)
        if 'cumulative_pnl' in filtered_df.columns:
            equity_series_raw = filtered_df.set_index(date_col)['cumulative_pnl'].dropna()
            if not equity_series_raw.empty:
                if not equity_series_raw.index.is_monotonic_increasing:
                    equity_series_raw = equity_series_raw.sort_index()
                series_options_decomp["Equity Curve (Cumulative PnL)"] = equity_series_raw

        # Daily PnL
        if pnl_col in filtered_df.columns: # Already checked pnl_col exists
            try:
                # Ensure date column is suitable for groupby (e.g., datetime)
                daily_pnl_raw = filtered_df.groupby(filtered_df[date_col].dt.normalize())[pnl_col].sum().dropna()
                if not daily_pnl_raw.empty:
                    if not daily_pnl_raw.index.is_monotonic_increasing:
                        daily_pnl_raw = daily_pnl_raw.sort_index()
                    series_options_decomp["Daily PnL"] = daily_pnl_raw
            except Exception as e:
                logger.error(f"Error grouping by date for Daily PnL: {e}", exc_info=True)
                display_custom_message(f"Could not prepare Daily PnL for decomposition due to date grouping error: {e}", "warning")
    else:
        display_custom_message(f"Date column ('{date_col}') is required for Time Series Decomposition and was not found or is invalid.", "error")


    if not series_options_decomp:
        st.warning("No suitable time series data (Equity Curve or Daily PnL with valid dates) is available for decomposition.")
    else:
        with st.form("decomposition_form_adv_v5"): # Incremented form key
            st.markdown("Decompose a selected time series into its trend, seasonal, and residual components to understand underlying patterns.")
            
            sel_series_name_dc = st.selectbox(
                "Select Series for Decomposition:",
                list(series_options_decomp.keys()),
                key="dc_series_v5",
                help="Choose the time series data you want to analyze."
            )
            sel_model_dc = st.selectbox(
                "Decomposition Model:",
                ["additive", "multiplicative"],
                key="dc_model_v5",
                help="Choose 'additive' if seasonal variations are roughly constant over time, or 'multiplicative' if they scale with the level of the series."
            )
            
            data_dc = series_options_decomp[sel_series_name_dc] # Selected series
            default_period = 7 # Default for daily data
            if isinstance(data_dc.index, pd.DatetimeIndex):
                inferred_freq = pd.infer_freq(data_dc.index)
                if inferred_freq:
                    if 'D' in inferred_freq.upper(): default_period = 7  # Weekly for daily
                    elif 'W' in inferred_freq.upper(): default_period = 52 # Yearly for weekly
                    elif 'M' in inferred_freq.upper(): default_period = 12 # Yearly for monthly
                    # Add more sophisticated frequency inference if needed
            
            min_period_dc = 2
            # Max period: ensure it's less than half the series length for meaningful decomposition
            max_period_dc = max(min_period_dc, (len(data_dc) // 2) -1 ) if len(data_dc) > 4 else min_period_dc
            
            period_dc = st.number_input(
                "Seasonal Period (Number of Observations):",
                min_value=min_period_dc, 
                max_value=max_period_dc if max_period_dc >= min_period_dc else min_period_dc, # Ensure max is not less than min
                value=min(default_period, max_period_dc) if max_period_dc >= min_period_dc else min_period_dc, 
                step=1,
                key="dc_period_v5",
                help=f"Specify the number of observations per seasonal cycle (e.g., 7 for daily data with weekly seasonality). Max allowed: {max_period_dc if max_period_dc >= min_period_dc else 'N/A due to short series'}."
            )
            submit_decomp_v5 = st.form_submit_button(f"Decompose {sel_series_name_dc}")

        if 'submit_decomp_v5' in locals() and submit_decomp_v5:
            data_to_decompose = series_options_decomp[sel_series_name_dc]
            
            # Check if period is valid given the data length
            if period_dc < min_period_dc or (max_period_dc < min_period_dc and period_dc > min_period_dc): # Second condition for very short series
                 display_custom_message(f"The chosen seasonal period ({period_dc}) is not valid for the current data length or settings. Max allowed: {max_period_dc}.", "error")
            elif len(data_to_decompose.dropna()) > 2 * period_dc : # Ensure enough data for decomposition
                with st.spinner(f"Decomposing {sel_series_name_dc} using {sel_model_dc} model with period {period_dc}..."):
                    service_output = statistical_analysis_service.get_time_series_decomposition(
                        data_to_decompose, 
                        model=sel_model_dc, 
                        period=period_dc
                    )

                if service_output:
                    if 'error' in service_output:
                        display_custom_message(f"Decomposition Error: {service_output['error']}", "error")
                    elif 'decomposition_result' in service_output:
                        actual_result = service_output['decomposition_result']
                        # Validate the result structure
                        if actual_result is not None and isinstance(actual_result, DecomposeResult) and hasattr(actual_result, 'observed') and not actual_result.observed.empty:
                            st.success("Time series decomposition completed successfully!")
                            decomp_fig = plot_time_series_decomposition(
                                actual_result, 
                                title=f"{sel_series_name_dc} - {sel_model_dc.capitalize()} Decomposition (Period: {period_dc})", 
                                theme=plot_theme
                            )
                            if decomp_fig:
                                st.plotly_chart(decomp_fig, use_container_width=True)
                            else:
                                display_custom_message("Could not plot the decomposition results.", "warning")
                        else:
                            display_custom_message("Decomposition failed or returned empty/unexpected data. The series might be too short, lack clear patterns for the chosen period, or the period might be too large for the dataset.", "error")
                    else:
                        display_custom_message("Decomposition analysis returned an unexpected structure from the service.", "error")
                else:
                    display_custom_message("Decomposition analysis failed to return any result from the service.", "error")
            else:
                display_custom_message(f"Not enough data points for decomposition with period {period_dc}. Need at least {2 * period_dc + 1} non-NaN observations. Currently have: {len(data_to_decompose.dropna())}.", "warning")

    # Placeholder for future sections like Distribution Fitting and Change Point Detection
    # st.markdown("---")
    # st.subheader(" distributional_fitting Distribution Fitting")
    # ...
    # st.markdown("---")
    # st.subheader("⚠️ Change Point Detection")
    # ...

if __name__ == "__main__":
    # This check helps if the page is run directly, though it's part of a multi-page app
    if 'app_initialized' not in st.session_state:
        st.warning("This page is designed to be part of a multi-page Streamlit application. Please run the main `app.py` script for the full experience.")
        # Optionally, initialize minimal session state for standalone testing
        # st.session_state.app_initialized = True 
        # st.session_state.filtered_data = pd.DataFrame() # Example
    show_advanced_stats_page()
