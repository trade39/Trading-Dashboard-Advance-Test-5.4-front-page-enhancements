"""
pages/7_🔮_Stochastic_Models.py
Explore stochastic process models. UI/UX enhanced with iconography and improved visual hierarchy.
Now uses StochasticModelService.
"""
import streamlit as st
import pandas as pd
import numpy as np
import logging
import plotly.graph_objects as go

try:
    # Assuming these modules are in the parent directory or configured in PYTHONPATH
    from config import APP_TITLE, EXPECTED_COLUMNS
    from utils.common_utils import display_custom_message
    from services.stochastic_model_service import StochasticModelService
    from plotting import _apply_custom_theme # Assuming this handles theme application
except ImportError as e:
    st.error(f"Stochastic Models Page Error: Critical module import failed: {e}. Ensure 'config.py', 'utils/common_utils.py', 'services/stochastic_model_service.py', and 'plotting.py' are accessible.")
    # Fallback configurations for basic functionality if imports fail
    APP_TITLE = "TradingDashboard_Error"
    EXPECTED_COLUMNS = {"pnl": "pnl", "cumulative_pnl": "cumulative_pnl"}
    # Dummy logger if main logger setup fails
    logger = logging.getLogger(__name__) # Use __name__ for module-specific logger
    logging.basicConfig(level=logging.ERROR) # Basic config for fallback
    logger.error(f"CRITICAL IMPORT ERROR in 7_🔮_Stochastic_Models.py: {e}", exc_info=True)

    # Fallback dummy service and function if imports fail
    class StochasticModelService:
        def run_gbm_simulation(self, *args, **kwargs):
            logger.error("Fallback StochasticModelService.run_gbm_simulation called.")
            return {"error": "StochasticModelService not loaded due to import failure."}
        def analyze_markov_chain_trades(self, *args, **kwargs):
            logger.error("Fallback StochasticModelService.analyze_markov_chain_trades called.")
            return {"error": "StochasticModelService not loaded due to import failure."}

    def _apply_custom_theme(fig, theme):
        logger.info(f"Fallback _apply_custom_theme called with theme: {theme}")
        return fig # No-op

    def display_custom_message(message, type="info"):
        if type == "error": st.error(message)
        elif type == "warning": st.warning(message)
        elif type == "success": st.success(message)
        else: st.info(message)
    # It's critical to stop execution if core components can't load
    st.stop()


# Initialize logger, assuming APP_TITLE is loaded from config
# If APP_TITLE wasn't loaded due to earlier error, this might use the fallback
logger = logging.getLogger(APP_TITLE if 'APP_TITLE' in globals() else "TradingDashboard_Fallback")

# Instantiate StochasticModelService
# This will use the fallback if the import failed
stochastic_model_service = StochasticModelService()


def show_stochastic_models_page():
    """
    Renders the Stochastic Process Modeling page with UI/UX enhancements.
    """
    st.title("🔮 Stochastic Process Modeling")

    # Check for necessary data in session state
    if 'filtered_data' not in st.session_state or st.session_state.filtered_data is None:
        display_custom_message(
            "Please upload and filter your trading data on the 'Data Upload & Filtering' page to enable stochastic modeling.",
            "info"
        )
        return

    filtered_df = st.session_state.filtered_data
    plot_theme = st.session_state.get('current_theme', 'dark') # Default to dark if not set
    pnl_col = EXPECTED_COLUMNS.get('pnl', 'pnl') # Default to 'pnl'
    # Ensure cumulative_pnl is engineered if not present, or handle its absence gracefully
    cum_pnl_col = 'cumulative_pnl' # This is often an engineered column

    if filtered_df.empty:
        display_custom_message("The filtered dataset is empty. Please adjust your filters or upload new data.", "warning")
        return

    # --- Consistent Iconography and Enhanced Spacing/Visual Hierarchy ---
    # Using emojis for iconography as a simple and effective method.
    # st.expander and st.form provide natural spacing. st.markdown("---") can be used for harder breaks.

    # --- Geometric Brownian Motion (GBM) Simulation ---
    st.subheader("📈 Geometric Brownian Motion (GBM) Simulation")
    st.markdown("Simulate future price paths based on historical volatility and drift.") # Brief explanation

    with st.expander("Configure GBM Simulation Parameters", expanded=True):
        # Determine default S0 for GBM from the last cumulative PnL value
        s0_gbm_default = 1000.0 # Default if cum_pnl_col is missing or data is problematic
        if cum_pnl_col in filtered_df.columns:
            # Ensure the column is numeric and handle potential NaNs before taking iloc[-1]
            cum_pnl_series_cleaned = pd.to_numeric(filtered_df[cum_pnl_col], errors='coerce').dropna()
            if not cum_pnl_series_cleaned.empty:
                s0_gbm_default = float(cum_pnl_series_cleaned.iloc[-1])
            else:
                st.caption(f"Cumulative PnL column ('{cum_pnl_col}') is empty or all NaNs. Using default S0 for GBM.")
        else:
            st.caption(f"Cumulative PnL column ('{cum_pnl_col}') not found. Using default S0 for GBM.")

        # Form for GBM parameters
        with st.form("gbm_simulation_form"): # Using a more descriptive key
            s0_gbm_input = st.number_input(
                "Initial Portfolio Value (S0):",
                value=s0_gbm_default,
                min_value=0.01, # S0 should be positive
                format="%.2f",
                key="gbm_s0_input",
                help="Starting value for the simulation, typically your current portfolio value."
            )
            mu_gbm = st.slider(
                "Annualized Drift (μ):",
                min_value=-0.50, max_value=0.50, value=0.05, step=0.01,
                key="gbm_mu_slider", format="%.2f",
                help="Expected annual rate of return (e.g., 0.05 for 5%)."
            )
            sigma_gbm = st.slider(
                "Annualized Volatility (σ):",
                min_value=0.01, max_value=1.00, value=0.20, step=0.01,
                key="gbm_sigma_slider", format="%.2f",
                help="Annualized standard deviation of returns (e.g., 0.20 for 20% volatility)."
            )
            n_steps_gbm = st.number_input(
                "Simulation Horizon (Trading Days):",
                min_value=30, max_value=730, value=252, step=1, # Max ~2 years
                key="gbm_steps_input",
                help="Number of future trading days to simulate."
            )
            n_sims_gbm = st.number_input(
                "Number of Simulated Paths:",
                min_value=1, max_value=1000, value=50, step=10, # Allow more sims
                key="gbm_sims_input",
                help="Number of different random paths to generate."
            )
            submit_gbm_button = st.form_submit_button("🚀 Simulate GBM Paths")

    if submit_gbm_button:
        if s0_gbm_input <= 0:
            display_custom_message("Initial Portfolio Value (S0) for GBM must be positive.", "error")
        else:
            with st.spinner(f"Simulating {n_sims_gbm} GBM paths for {n_steps_gbm} days... This might take a moment."):
                try:
                    gbm_result = stochastic_model_service.run_gbm_simulation(
                        s0=s0_gbm_input,
                        mu=mu_gbm,
                        sigma=sigma_gbm,
                        dt=1/252, # Assuming daily steps, 252 trading days a year
                        n_steps=n_steps_gbm,
                        n_sims=n_sims_gbm
                    )
                except Exception as ex:
                    logger.error(f"Error during GBM simulation call: {ex}", exc_info=True)
                    gbm_result = {"error": f"An unexpected error occurred: {ex}"}


            if gbm_result and 'paths' in gbm_result and gbm_result['paths'] is not None and gbm_result['paths'].size > 0:
                gbm_paths = gbm_result['paths']
                fig_gbm = go.Figure()
                # Plot a subset of paths for clarity, e.g., first 20 or up to n_sims_gbm
                paths_to_plot = min(n_sims_gbm, 20)
                for i in range(paths_to_plot):
                    fig_gbm.add_trace(go.Scatter(y=gbm_paths[i,:], mode='lines', name=f'Path {i+1}', opacity=0.7))

                fig_gbm.update_layout(
                    title=f"GBM Simulated Equity Paths (First {paths_to_plot} of {n_sims_gbm})",
                    xaxis_title=f"Trading Days (from S0: {s0_gbm_input:.2f})",
                    yaxis_title="Simulated Portfolio Value",
                    showlegend=True,
                    legend_title_text='Simulations'
                )
                st.plotly_chart(_apply_custom_theme(fig_gbm, plot_theme), use_container_width=True)
                st.success("GBM simulation complete! Chart displays a subset of paths if many were generated.")
            elif gbm_result and 'error' in gbm_result:
                display_custom_message(f"GBM Simulation Error: {gbm_result.get('error', 'Unknown error from service.')}", "error")
                logger.warning(f"GBM Service Error: {gbm_result.get('error')}")
            else:
                display_custom_message("GBM simulation failed to return valid paths. Please check logs or parameters.", "error")
                logger.warning("GBM simulation returned no paths or an unexpected result.")

    st.markdown("---") # Visual separator

    # --- Markov Chain Analysis for Trade Sequences ---
    st.subheader("🔗 Markov Chain Analysis for Trade Sequences")
    st.markdown("Analyze the probability of transitioning between trade outcomes (e.g., Win, Loss).")

    with st.expander("Configure Markov Chain Analysis", expanded=False): # Default to collapsed
        if not pnl_col or pnl_col not in filtered_df.columns:
            st.warning(f"PnL column ('{pnl_col}') is required for Markov chain analysis but was not found in the data.")
            run_mc_button = False # Disable button if PnL column is missing
        else:
            with st.form("markov_chain_form"): # Using a more descriptive key
                mc_n_states = st.selectbox(
                    "Number of States (Trade Outcomes):",
                    options=[2, 3],
                    index=0, # Default to 2 states
                    format_func=lambda x: f"{x} States (Win/Loss)" if x == 2 else f"{x} States (Win/Loss/Breakeven)",
                    key="mc_states_selector",
                    help="Define states based on trade P&L: 2 for Win/Loss, 3 for Win/Loss/Breakeven."
                )
                run_mc_button = st.form_submit_button("🔍 Analyze Trade Sequence")

    if 'run_mc_button' in locals() and run_mc_button:
        if not pnl_col or pnl_col not in filtered_df.columns:
             display_custom_message(f"Cannot run Markov Chain analysis: PnL column ('{pnl_col}') is missing.", "error")
        else:
            pnl_series_for_mc = filtered_df[pnl_col].dropna()
            if len(pnl_series_for_mc) < 20: # Arbitrary minimum number of trades for meaningful analysis
                display_custom_message(
                    f"Insufficient trade data ({len(pnl_series_for_mc)} trades). At least 20 trades are recommended for Markov chain analysis.",
                    "info"
                )
            else:
                with st.spinner("Fitting Markov chain model to your trade sequence..."):
                    try:
                        mc_results = stochastic_model_service.analyze_markov_chain_trades(
                            pnl_series=pnl_series_for_mc,
                            n_states=mc_n_states
                        )
                    except Exception as ex:
                        logger.error(f"Error during Markov Chain analysis call: {ex}", exc_info=True)
                        mc_results = {"error": f"An unexpected error occurred: {ex}"}


                if mc_results and 'error' not in mc_results:
                    st.success("Markov chain analysis complete!")
                    st.write(f"**State Definitions:** `{mc_results.get('state_labels', ['N/A'])}`")

                    if 'transition_matrix' in mc_results and mc_results['transition_matrix'] is not None:
                        tm_df = pd.DataFrame(mc_results['transition_matrix'],
                                             index=mc_results.get('state_labels'),
                                             columns=mc_results.get('state_labels'))
                        st.write("**Transition Matrix (P(Row State → Column State)):**")

                        # Styling the dataframe for better readability
                        st.dataframe(tm_df.style.format("{:.3%}").background_gradient(cmap='Blues', axis=1)
                                     .set_caption("Probabilities of moving from one state to another."))

                        if not tm_df.empty:
                            fig_tm = go.Figure(data=go.Heatmap(
                                       z=tm_df.values,
                                       x=tm_df.columns,
                                       y=tm_df.index,
                                       colorscale='Blues',
                                       text=tm_df.applymap(lambda x: f"{x:.2%}").values, # Format text in heatmap cells
                                       texttemplate="%{text}", # Display the formatted text
                                       hoverongaps=False,
                                       colorbar_title='Probability'
                                   ))
                            fig_tm.update_layout(
                                title="Transition Matrix Heatmap",
                                xaxis_title="To State",
                                yaxis_title="From State"
                            )
                            st.plotly_chart(_apply_custom_theme(fig_tm, plot_theme), use_container_width=True)
                    else:
                        display_custom_message("Transition matrix could not be generated.", "warning")

                elif mc_results and 'error' in mc_results:
                    display_custom_message(f"Markov Chain Analysis Error: {mc_results.get('error', 'Unknown error from service.')}", "error")
                    logger.warning(f"Markov Service Error: {mc_results.get('error')}")
                else:
                    display_custom_message("Markov chain analysis failed to return valid results.", "error")
                    logger.warning("Markov analysis returned no results or an unexpected result.")

    st.markdown("---")
    st.info("""
    **Further Exploration (Future Features):**
    - **Ornstein-Uhlenbeck Process:** For modeling mean-reverting series.
    - **Merton Jump-Diffusion Model:** To incorporate sudden jumps in prices.
    - **Advanced Markov Chain Properties:** Stationary distribution, expected hitting times.
    """)

if __name__ == "__main__":
    # This block is useful for direct testing of the page,
    # but in a multi-page app, it's usually run via the main app.py.
    st.set_page_config(layout="wide", page_title="Stochastic Models", page_icon="🔮")

    # Mock session state for standalone testing
    if 'app_initialized' not in st.session_state: # To simulate being run as part of an app
        st.session_state.app_initialized = True
        st.session_state.current_theme = 'dark' # Default theme for testing
        # Mock some filtered_data for testing
        mock_dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'] * 10)
        mock_pnl = np.random.randn(50) * 100 # More data points
        mock_df = pd.DataFrame({
            'timestamp': mock_dates,
            'pnl': mock_pnl,
            'cumulative_pnl': mock_pnl.cumsum() + 1000
        })
        st.session_state.filtered_data = mock_df
        st.session_state.expected_columns = EXPECTED_COLUMNS # Use loaded or fallback

    # Check if essential services/configs were loaded
    if 'StochasticModelService' not in globals() or 'APP_TITLE' not in globals():
        st.error("Critical components failed to load. Page functionality will be severely limited. Please check the console for import errors.")
    else:
        show_stochastic_models_page()

    logger.info("Stochastic Models page loaded directly for testing or as main.")
