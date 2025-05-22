import streamlit as st
import pandas as pd
import numpy as np # Added for potential calculations like std dev
import logging
import plotly.graph_objects as go
import plotly.figure_factory as ff # For distribution plots

# --- Imports for app config and utilities ---
try:
    from config import APP_TITLE, EXPECTED_COLUMNS, DEFAULT_KPI_DISPLAY_ORDER, COLORS
    from utils.common_utils import display_custom_message
    from services.analysis_service import AnalysisService # Assuming this will have new methods
    from plotting import _apply_custom_theme # Assuming this can be reused or adapted
    # Placeholder for new plotting functions if they were in plotting.py
    # from plotting import plot_return_distributions, plot_risk_reward_scatter
except ImportError as e:
    st.error(f"Strategy Comparison Page Error: Critical module import failed: {e}. Ensure app structure is correct and all dependencies are available.")
    APP_TITLE = "TradingDashboard_Error"
    EXPECTED_COLUMNS = {"strategy": "strategy_fallback", "date": "date_fallback", "pnl": "pnl_fallback"}
    DEFAULT_KPI_DISPLAY_ORDER = []
    COLORS = {}
    def display_custom_message(message, type): st.text(f"{type.upper()}: {message}")
    class AnalysisService:
        def get_core_kpis(self, df, rate): return {"error": "AnalysisService not loaded"}
        def calculate_daily_returns(self, strategy_df, pnl_col): # Mock implementation
            if pnl_col in strategy_df:
                return strategy_df[pnl_col].pct_change().fillna(0) # Example, assumes pnl is like equity
            return pd.Series(dtype='float64')
        # Mock methods for scatter plot KPIs if not in get_core_kpis
        def get_scatter_plot_kpis(self, strategy_df, risk_free_rate):
             # These would ideally come from a more robust calculation
            pnl_col = EXPECTED_COLUMNS.get('pnl', 'pnl')
            if pnl_col not in strategy_df or strategy_df[pnl_col].empty:
                return {'Annualized Return': 0, 'Annualized Volatility': 0, 'Sharpe Ratio': 0}

            daily_returns = strategy_df[pnl_col].pct_change().fillna(0) # Simplified
            annualized_return = daily_returns.mean() * 252
            annualized_volatility = daily_returns.std() * np.sqrt(252)
            sharpe = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility != 0 else 0
            return {
                'Annualized Return': annualized_return,
                'Annualized Volatility': annualized_volatility,
                'Sharpe Ratio': sharpe # Example
            }

    def _apply_custom_theme(fig, theme): return fig

logger = logging.getLogger(APP_TITLE if 'APP_TITLE' in locals() else "TradingDashboard_Default")
analysis_service = AnalysisService() if 'AnalysisService' in locals() else None

def get_st_theme():
    try:
        theme_base = st.get_option("theme.base")
    except Exception:
        theme_base = st.session_state.get("current_theme", "dark")
    return theme_base if theme_base in {"dark", "light"} else "dark"

def show_strategy_comparison_page():
    st.title("⚖️ Strategy Performance Comparison")
    st.markdown('<p class="page-subtitle">Easily compare performance, risk, and equity curves between your strategies side-by-side.</p>', unsafe_allow_html=True)
    theme = get_st_theme()

    if 'filtered_data' not in st.session_state or st.session_state.filtered_data is None:
        display_custom_message("Please upload and process data in the main application to compare strategies.", "info")
        return
    filtered_df = st.session_state.filtered_data
    risk_free_rate = st.session_state.get('risk_free_rate', 0.02)
    strategy_col = EXPECTED_COLUMNS.get('strategy')
    date_col = EXPECTED_COLUMNS.get('date')
    pnl_col = EXPECTED_COLUMNS.get('pnl')

    if filtered_df.empty or not strategy_col or not date_col or not pnl_col:
        display_custom_message("Data is missing or key columns (Strategy, Date, PnL) are not configured. Cannot proceed.", "warning")
        return
    if strategy_col not in filtered_df.columns:
        display_custom_message(f"Strategy column '{strategy_col}' not found in data.", "error")
        return
    if date_col not in filtered_df.columns:
        display_custom_message(f"Date column '{date_col}' not found in data.", "error")
        return
    if pnl_col not in filtered_df.columns:
        display_custom_message(f"PnL column '{pnl_col}' not found in data.", "error")
        return
    
    # Ensure PnL column is numeric
    try:
        filtered_df[pnl_col] = pd.to_numeric(filtered_df[pnl_col], errors='coerce')
        # Optional: Handle NaNs if appropriate, e.g., filtered_df.dropna(subset=[pnl_col], inplace=True)
    except Exception as e:
        display_custom_message(f"Could not convert PnL column '{pnl_col}' to numeric: {e}", "error")
        logger.error(f"Error converting PnL column to numeric: {e}", exc_info=True)
        return


    st.subheader("⚙️ Strategy Selection")
    st.markdown('<div class="performance-section-container">', unsafe_allow_html=True)
    available_strategies = sorted(filtered_df[strategy_col].astype(str).dropna().unique())
    if not available_strategies:
        display_custom_message("No distinct strategies found.", "info")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    default_selection = available_strategies[:2] if len(available_strategies) >= 2 else available_strategies
    selected_strategies = st.multiselect(
        "Choose strategies to compare:",
        options=available_strategies,
        default=default_selection,
        key="strategy_comp_select_v3" # Incremented key
    )
    if not selected_strategies:
        display_custom_message("Please select strategies.", "info")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")

    # --- KPI Comparison Table Section ---
    st.subheader("📊 Key Performance Indicator Comparison")
    st.markdown('<div class="performance-section-container">', unsafe_allow_html=True)
    kpi_data_for_table = []
    all_kpis_for_scatter = [] # To store KPIs for the scatter plot

    if not analysis_service:
        display_custom_message("Analysis service is not available.", "error")
    else:
        for strat_name in selected_strategies:
            strat_df = filtered_df[filtered_df[strategy_col].astype(str) == str(strat_name)]
            if not strat_df.empty:
                kpis = analysis_service.get_core_kpis(strat_df, risk_free_rate) # Assumes this returns all needed KPIs
                if kpis and 'error' not in kpis:
                    kpi_data_for_table.append({"Strategy": strat_name, **kpis})
                    # For scatter plot, ensure 'Annualized Return' and 'Annualized Volatility' are present
                    # These might come from get_core_kpis or a dedicated method
                    scatter_kpis = {
                        'Strategy': strat_name,
                        'Annualized Return': kpis.get('Annualized Return', kpis.get('annualized_return', 0)), # Check common naming
                        'Annualized Volatility': kpis.get('Annualized Volatility', kpis.get('annualized_volatility', 0)),
                        'Sharpe Ratio': kpis.get('Sharpe Ratio', kpis.get('sharpe_ratio', 'N/A')) # Default to 'N/A' string for scatter
                    }
                    all_kpis_for_scatter.append(scatter_kpis)
                else:
                    logger.warning(f"KPI calculation failed for strategy '{strat_name}'.")
        
        if kpi_data_for_table:
            comp_df = pd.DataFrame(kpi_data_for_table).set_index("Strategy")
            kpi_order = DEFAULT_KPI_DISPLAY_ORDER if isinstance(DEFAULT_KPI_DISPLAY_ORDER, list) else []
            kpis_to_show = [k for k in kpi_order if k in comp_df.columns and k not in ['trading_days', 'risk_free_rate_used']]
            if not kpis_to_show and comp_df.columns.any():
                kpis_to_show = [c for c in comp_df.columns if c not in ['trading_days', 'risk_free_rate_used']]

            if theme == "dark": max_bg, min_bg, text_hl = "#3BA55D", "#FF776B", "#F0F1F6"
            else: max_bg, min_bg, text_hl = "#B2F2BB", "#FFD6D6", "#121416"
            
            if kpis_to_show:
                styled_df = comp_df[kpis_to_show].style.format("{:,.2f}", na_rep="-") \
                    .highlight_max(axis=0, props=f"background-color: {max_bg}; color: {text_hl}; font-weight:bold;") \
                    .highlight_min(axis=0, props=f"background-color: {min_bg}; color: {text_hl}; font-weight:bold;")
                st.dataframe(styled_df, use_container_width=True)
            else: display_custom_message("No common KPIs to display.", "warning")
        elif selected_strategies: display_custom_message("No KPI data for selected strategies.", "warning")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")

    # --- Comparative Equity Curves ---
    st.subheader("📈 Comparative Equity Curves")
    st.markdown('<div class="performance-section-container">', unsafe_allow_html=True)
    equity_fig = go.Figure()
    has_equity_data = False
    for strat_name in selected_strategies:
        strat_df = filtered_df[filtered_df[strategy_col].astype(str) == str(strat_name)].copy()
        if not strat_df.empty and date_col in strat_df.columns and pnl_col in strat_df.columns:
            try:
                strat_df[date_col] = pd.to_datetime(strat_df[date_col])
                strat_df.sort_values(by=date_col, inplace=True)
                strat_df['cumulative_pnl'] = strat_df[pnl_col].cumsum()
                
                line_color = COLORS.get(strat_name) if isinstance(COLORS, dict) else None
                trace_params = {"x": strat_df[date_col], "y": strat_df['cumulative_pnl'], "mode": 'lines', "name": strat_name}
                if line_color: trace_params["line"] = dict(color=line_color)
                
                equity_fig.add_trace(go.Scatter(**trace_params))
                has_equity_data = True
            except Exception as e:
                logger.error(f"Error processing equity curve for {strat_name}: {e}", exc_info=True)
    if has_equity_data:
        equity_fig.update_layout(xaxis_title="Date", yaxis_title="Cumulative PnL", hovermode="x unified", legend_title_text='Strategy')
        st.plotly_chart(_apply_custom_theme(equity_fig, theme), use_container_width=True)
    else:
        display_custom_message("Not enough data for equity curves.", "info")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")

    # --- Return Distribution Comparison ---
    st.subheader("🎛️ Return Distribution Comparison (Daily PnL)")
    st.markdown('<div class="performance-section-container">', unsafe_allow_html=True)
    hist_data = []
    group_labels = []
    has_dist_data = False

    for strat_name in selected_strategies:
        strat_df = filtered_df[filtered_df[strategy_col].astype(str) == str(strat_name)]
        if not strat_df.empty and pnl_col in strat_df.columns:
            daily_pnl_values = strat_df[pnl_col].dropna()
            if not daily_pnl_values.empty:
                hist_data.append(daily_pnl_values.tolist())
                group_labels.append(strat_name)
                has_dist_data = True
        else:
            logger.info(f"No PnL data for distribution plot for strategy '{strat_name}'.")

    if has_dist_data:
        try:
            dist_fig = ff.create_distplot(hist_data, group_labels, bin_size=.2, show_hist=True, show_rug=False) 
            dist_fig.update_layout(
                title_text='Distribution of Daily PnL',
                xaxis_title='Daily PnL',
                yaxis_title='Density',
                legend_title_text='Strategy'
            )
            st.plotly_chart(_apply_custom_theme(dist_fig, theme), use_container_width=True)
        except Exception as e:
            logger.error(f"Error creating distribution plot: {e}", exc_info=True)
            display_custom_message(f"Could not generate return distribution plot: {e}", "error")

    elif selected_strategies:
        display_custom_message("Not enough PnL data to plot return distributions for selected strategies.", "info")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")


    # --- Risk-Reward Scatter Plot ---
    st.subheader("🎯 Risk-Reward Scatter Plot")
    st.markdown('<div class="performance-section-container">', unsafe_allow_html=True)
    if all_kpis_for_scatter:
        scatter_df = pd.DataFrame(all_kpis_for_scatter)
        
        required_scatter_cols = ['Annualized Return', 'Annualized Volatility', 'Strategy']
        missing_cols = [col for col in required_scatter_cols if col not in scatter_df.columns]

        if not missing_cols and pd.api.types.is_numeric_dtype(scatter_df['Annualized Return']) and pd.api.types.is_numeric_dtype(scatter_df['Annualized Volatility']):
            scatter_fig = go.Figure()
            
            for i, row in scatter_df.iterrows():
                strat_name = row['Strategy']
                ann_return = row['Annualized Return']
                ann_vol = row['Annualized Volatility']
                sharpe_ratio_val = row.get('Sharpe Ratio', 'N/A') 
                
                # Prepare Sharpe Ratio for display (ensure it's a string)
                sharpe_display_text = "N/A"
                if isinstance(sharpe_ratio_val, (int, float)) and pd.notna(sharpe_ratio_val):
                    sharpe_display_text = f"{sharpe_ratio_val:.2f}"
                elif pd.notna(sharpe_ratio_val): # Handles cases where it might be 'N/A' string already
                    sharpe_display_text = str(sharpe_ratio_val)

                point_color = COLORS.get(strat_name) if isinstance(COLORS, dict) and COLORS.get(strat_name) else None

                hovertemplate_text = (
                    f"<b>{strat_name}</b><br><br>"
                    f"Annualized Volatility: {ann_vol:.2%}<br>" # Use direct formatting for known floats
                    f"Annualized Return: {ann_return:.2%}<br>"  # Use direct formatting for known floats
                    f"Sharpe Ratio: {sharpe_display_text}" # Use pre-formatted string
                    "<extra></extra>" 
                )

                scatter_fig.add_trace(go.Scatter(
                    x=[ann_vol],
                    y=[ann_return],
                    mode='markers+text', 
                    name=strat_name,
                    text=[strat_name], 
                    textposition="top right",
                    marker=dict(size=12, color=point_color, line=dict(width=1, color='DarkSlateGrey')),
                    hovertemplate=hovertemplate_text
                ))

            scatter_fig.update_layout(
                xaxis_title="Annualized Volatility (Risk)",
                yaxis_title="Annualized Return (Reward)",
                legend_title_text='Strategy',
                hovermode="closest"
            )
            scatter_fig.update_xaxes(tickformat=".1%")
            scatter_fig.update_yaxes(tickformat=".1%")
            st.plotly_chart(_apply_custom_theme(scatter_fig, theme), use_container_width=True)
        else:
            display_custom_message(f"Missing or non-numeric data for Risk-Reward scatter plot. Required: {', '.join(required_scatter_cols)}. Check KPI calculations.", "warning")
            logger.warning(f"Scatter plot data issue. Missing columns: {missing_cols}. Data: {scatter_df.head()}")

    elif selected_strategies:
        display_custom_message("Not enough KPI data (Annualized Return, Annualized Volatility) for Risk-Reward scatter plot.", "info")
    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Strategy Comparison")
    if 'app_initialized' not in st.session_state:
        st.warning("This page is part of a multi-page app. For full functionality, run the main application script. Mock data may be used for standalone testing.")
        # Mock session state for standalone testing
        mock_dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'] * 2)
        # Ensure EXPECTED_COLUMNS is defined before use or use literals
        expected_date_col = EXPECTED_COLUMNS.get('date', 'Date') if 'EXPECTED_COLUMNS' in globals() else 'Date'
        expected_strat_col = EXPECTED_COLUMNS.get('strategy', 'Strategy') if 'EXPECTED_COLUMNS' in globals() else 'Strategy'
        expected_pnl_col = EXPECTED_COLUMNS.get('pnl', 'PnL') if 'EXPECTED_COLUMNS' in globals() else 'PnL'

        st.session_state.filtered_data = pd.DataFrame({
            expected_date_col: mock_dates,
            expected_strat_col: ['Alpha'] * 5 + ['Beta'] * 5,
            expected_pnl_col: np.random.randn(10).cumsum() + np.random.randint(-5, 5, 10)
        })
        st.session_state.risk_free_rate = 0.02
        st.session_state.current_theme = "dark"

    if 'EXPECTED_COLUMNS' not in globals(): EXPECTED_COLUMNS = {"strategy":"Strategy", "date":"Date", "pnl":"PnL"}
    if 'DEFAULT_KPI_DISPLAY_ORDER' not in globals(): DEFAULT_KPI_DISPLAY_ORDER = ['Annualized Return', 'Annualized Volatility', 'Sharpe Ratio']
    if 'COLORS' not in globals(): COLORS = {"Alpha": "blue", "Beta": "green"}
    if 'analysis_service' not in globals() or analysis_service is None:
        class MockAnalysisService:
            def get_core_kpis(self, df, rate):
                pnl_col_name = EXPECTED_COLUMNS.get('pnl','PnL')
                daily_returns = df[pnl_col_name].pct_change().fillna(0) if pnl_col_name in df and not df[pnl_col_name].empty else pd.Series([0]*len(df), dtype=float)
                if daily_returns.empty: # Handle case where df might be empty or pnl_col not found leading to empty series
                    ann_ret, ann_vol, sharpe = 0.0, 0.0, 0.0
                else:
                    ann_ret = daily_returns.mean() * 252
                    ann_vol = daily_returns.std() * np.sqrt(252)
                    sharpe = (ann_ret - rate) / ann_vol if ann_vol != 0 and ann_vol is not np.nan else 0.0
                return {
                    "Annualized Return": ann_ret if pd.notna(ann_ret) else 0.0,
                    "Annualized Volatility": ann_vol if pd.notna(ann_vol) else 0.0,
                    "Sharpe Ratio": sharpe if pd.notna(sharpe) else 0.0,
                }
        analysis_service = MockAnalysisService()

    show_strategy_comparison_page()
