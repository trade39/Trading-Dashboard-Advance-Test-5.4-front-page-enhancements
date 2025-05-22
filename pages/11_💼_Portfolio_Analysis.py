"""
pages/11_💼_Portfolio_Analysis.py

This page provides portfolio-level aggregation and analysis if multiple
accounts are present in the data. It calculates combined P&L, overall risk metrics,
visualizes inter-strategy and inter-account correlations, compares equity curves,
and includes a portfolio optimization section with efficient frontier visualization,
Risk Parity, robust covariance options, per-asset weight constraints, display of
risk contributions, and turnover reporting.
"""
import streamlit as st
import pandas as pd
import numpy as np
import logging
import plotly.graph_objects as go 
import plotly.express as px 
from typing import Dict, Any, Optional, List, Tuple

try:
    from config import APP_TITLE, EXPECTED_COLUMNS, KPI_CONFIG, DEFAULT_KPI_DISPLAY_ORDER, COLORS, RISK_FREE_RATE
    from utils.common_utils import display_custom_message, format_currency, format_percentage, calculate_portfolio_turnover 
    from services.analysis_service import AnalysisService 
    from services.portfolio_analysis import PortfolioAnalysisService 
    from plotting import plot_equity_curve_and_drawdown, _apply_custom_theme, plot_efficient_frontier
    from components.kpi_display import KPIClusterDisplay
except ImportError as e:
    critical_error_message = f"Portfolio Analysis Page Error: Critical module import failed: {e}. This page cannot be loaded."
    if 'st' in globals() and hasattr(st, 'error'): 
        st.error(critical_error_message)
    try:
        page_logger = logging.getLogger(__name__) 
        page_logger.error(f"CRITICAL IMPORT ERROR in Portfolio Analysis Page: {e}", exc_info=True)
    except Exception as log_e:
        print(f"Fallback logging error during Portfolio Analysis Page import: {log_e}")
    if 'st' in globals() and hasattr(st, 'stop'):
        st.stop()
    else:
        raise ImportError(critical_error_message) from e

logger = logging.getLogger(APP_TITLE)
general_analysis_service = AnalysisService() 
portfolio_specific_service = PortfolioAnalysisService() 

def _calculate_drawdown_series_for_aggregated_df(cumulative_pnl_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Helper to calculate absolute and percentage drawdown series for an aggregated PnL series.
    Args:
        cumulative_pnl_series (pd.Series): Series of cumulative PnL values.
    Returns:
        Tuple[pd.Series, pd.Series]: drawdown_abs_series, drawdown_pct_series
    """
    if cumulative_pnl_series.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    high_water_mark = cumulative_pnl_series.cummax()
    drawdown_abs_series = high_water_mark - cumulative_pnl_series
    
    drawdown_pct_series = pd.Series(index=cumulative_pnl_series.index, dtype=float)
    for i in range(len(cumulative_pnl_series)):
        current_hwm = high_water_mark.iloc[i]
        current_dd_abs = drawdown_abs_series.iloc[i]
        if current_hwm > 1e-9: 
            drawdown_pct_series.iloc[i] = (current_dd_abs / current_hwm) * 100.0
        elif current_dd_abs > 1e-9: 
            drawdown_pct_series.iloc[i] = 100.0 
        else: 
            drawdown_pct_series.iloc[i] = 0.0
            
    drawdown_pct_series = drawdown_pct_series.fillna(0)

    return drawdown_abs_series, drawdown_pct_series


def calculate_metrics_for_df(
    df: pd.DataFrame, 
    pnl_col: str, 
    date_col: str, 
    risk_free_rate: float, 
    initial_capital: float
) -> Dict[str, Any]:
    if df.empty:
        return {
            "Total PnL": 0.0, "Total Trades": 0, "Win Rate %": 0.0, 
            "Avg Trade PnL": 0.0, "Max Drawdown %": 0.0, "Sharpe Ratio": 0.0 
        }
    df_copy = df.copy()
    if 'win' not in df_copy.columns and pnl_col in df_copy.columns:
        if pd.api.types.is_numeric_dtype(df_copy[pnl_col]):
            df_copy['win'] = df_copy[pnl_col] > 0
        else:
            logger.warning(f"PnL column '{pnl_col}' is not numeric in calculate_metrics_for_df.")
            return {
                "Total PnL": 0.0, "Total Trades": 0, "Win Rate %": 0.0, 
                "Avg Trade PnL": 0.0, "Max Drawdown %": 0.0, "Sharpe Ratio": 0.0,
                "error": "PnL column not numeric for 'win' creation."}
    if 'cumulative_pnl' not in df_copy.columns and pnl_col in df_copy.columns:
        if pd.api.types.is_numeric_dtype(df_copy[pnl_col]):
            if date_col in df_copy.columns:
                df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
                df_copy.sort_values(by=date_col, inplace=True)
            df_copy['cumulative_pnl'] = df_copy[pnl_col].cumsum()
        else:
            logger.warning(f"PnL column '{pnl_col}' is not numeric for 'cumulative_pnl'.")

    kpis = general_analysis_service.get_core_kpis(df_copy, risk_free_rate, initial_capital=initial_capital)
    if kpis and 'error' not in kpis:
        return {
            "Total PnL": kpis.get("total_pnl", 0.0), 
            "Total Trades": kpis.get("total_trades", 0), 
            "Win Rate %": kpis.get("win_rate", 0.0),
            "Avg Trade PnL": kpis.get("avg_trade_pnl", 0.0),
            "Max Drawdown %": kpis.get("max_drawdown_pct", 0.0),
            "Sharpe Ratio": kpis.get("sharpe_ratio", 0.0)
        }
    error_msg = kpis.get('error', 'Unknown error in KPI calculation') if kpis else 'KPI calculation returned None'
    logger.warning(f"KPI calculation failed in calculate_metrics_for_df: {error_msg}")
    return {
        "Total PnL": 0.0, "Total Trades": 0, "Win Rate %": 0.0, 
        "Avg Trade PnL": 0.0, "Max Drawdown %": 0.0, "Sharpe Ratio": 0.0,
        "error": error_msg
    }

def show_portfolio_analysis_page():
    st.title("💼 Portfolio-Level Analysis")
    logger.info("Rendering Portfolio Analysis Page.")

    if 'processed_data' not in st.session_state or st.session_state.processed_data is None:
        display_custom_message("Please upload and process data to view portfolio analysis.", "info")
        return

    base_df = st.session_state.processed_data 
    plot_theme = st.session_state.get('current_theme', 'dark')
    risk_free_rate = st.session_state.get('risk_free_rate', RISK_FREE_RATE) 
    global_initial_capital = st.session_state.get('initial_capital', 100000.0) 

    account_col_conceptual = 'account_str'
    account_col_actual = EXPECTED_COLUMNS.get(account_col_conceptual)
    pnl_col_actual = EXPECTED_COLUMNS.get('pnl')
    date_col_actual = EXPECTED_COLUMNS.get('date')
    strategy_col_actual = EXPECTED_COLUMNS.get('strategy')

    if not all([account_col_actual, pnl_col_actual, date_col_actual, strategy_col_actual]):
        display_custom_message(f"Essential column configurations missing (account, pnl, date, strategy). Portfolio analysis cannot proceed.", "error")
        return
    if account_col_actual not in base_df.columns:
        display_custom_message(f"Account column '{account_col_actual}' (mapped from '{account_col_conceptual}') not found. Portfolio analysis requires this column.", "warning")
        return
    required_analysis_cols = [pnl_col_actual, date_col_actual, strategy_col_actual]
    if not all(col in base_df.columns for col in required_analysis_cols):
        missing_cols_for_analysis = [col for col in required_analysis_cols if col not in base_df.columns]
        display_custom_message(f"Essential columns for analysis ({', '.join(missing_cols_for_analysis)}) are missing.", "error")
        return

    unique_accounts_all = sorted(base_df[account_col_actual].dropna().astype(str).unique())
    if not unique_accounts_all:
        display_custom_message("No accounts found in the data.", "info")
        return

    st.sidebar.subheader("Portfolio Account Selection")
    if len(unique_accounts_all) > 1:
        selected_accounts_for_portfolio = st.sidebar.multiselect(
            "Select accounts for portfolio view:",
            options=unique_accounts_all,
            default=unique_accounts_all, 
            key="portfolio_view_account_multiselect"
        )
    else:
        selected_accounts_for_portfolio = unique_accounts_all
        st.sidebar.info(f"Displaying portfolio view for the single account: {unique_accounts_all[0]}")

    if not selected_accounts_for_portfolio:
        display_custom_message("Please select at least one account in the sidebar for the portfolio view.", "info")
        return 
    
    portfolio_df = base_df[base_df[account_col_actual].isin(selected_accounts_for_portfolio)].copy()
    if portfolio_df.empty:
        display_custom_message("No data for the selected accounts in the portfolio view.", "info")
        return 

    # --- Overall Performance Section ---
    st.markdown("<div class='performance-section-container'>", unsafe_allow_html=True)
    st.header(f"📈 Overall Performance for Selected Portfolio ({', '.join(selected_accounts_for_portfolio)})")
    portfolio_df[date_col_actual] = pd.to_datetime(portfolio_df[date_col_actual], errors='coerce')
    portfolio_df_cleaned_dates = portfolio_df.dropna(subset=[date_col_actual, pnl_col_actual])
    
    # Define portfolio_daily_trades_df here to ensure it's in scope for "View Data"
    portfolio_daily_trades_df = pd.DataFrame() 

    if portfolio_df_cleaned_dates.empty:
        display_custom_message("No valid P&L or date data after cleaning for selected portfolio.", "warning")
    else:
        portfolio_daily_pnl = portfolio_df_cleaned_dates.groupby(portfolio_df_cleaned_dates[date_col_actual].dt.normalize())[pnl_col_actual].sum()
        if portfolio_daily_pnl.empty:
            display_custom_message("No P&L data after daily aggregation for selected portfolio.", "warning")
        else:
            portfolio_daily_trades_df = pd.DataFrame({date_col_actual: portfolio_daily_pnl.index, pnl_col_actual: portfolio_daily_pnl.values})
            portfolio_daily_trades_df['cumulative_pnl'] = portfolio_daily_trades_df[pnl_col_actual].cumsum()
            portfolio_daily_trades_df['win'] = portfolio_daily_trades_df[pnl_col_actual] > 0 
            
            if 'cumulative_pnl' in portfolio_daily_trades_df.columns and not portfolio_daily_trades_df['cumulative_pnl'].empty:
                drawdown_abs_series, drawdown_pct_series = _calculate_drawdown_series_for_aggregated_df(portfolio_daily_trades_df['cumulative_pnl'])
                portfolio_daily_trades_df['drawdown_abs'] = drawdown_abs_series
                portfolio_daily_trades_df['drawdown_pct'] = drawdown_pct_series
                logger.info("Calculated drawdown_abs and drawdown_pct for aggregated portfolio daily data.")
            else:
                portfolio_daily_trades_df['drawdown_abs'] = pd.Series(dtype=float)
                portfolio_daily_trades_df['drawdown_pct'] = pd.Series(dtype=float)
                logger.warning("Could not calculate drawdown for aggregated portfolio as cumulative_pnl was missing or empty.")

            with st.spinner("Calculating selected portfolio KPIs..."):
                portfolio_kpis = general_analysis_service.get_core_kpis(portfolio_daily_trades_df, risk_free_rate, initial_capital=global_initial_capital)
            
            if portfolio_kpis and 'error' not in portfolio_kpis:
                st.markdown("<div class='kpi-metrics-block'>", unsafe_allow_html=True)
                portfolio_kpi_keys = ["total_pnl", "sharpe_ratio", "sortino_ratio", "calmar_ratio", "max_drawdown_abs", "max_drawdown_pct", "avg_daily_pnl", "pnl_skewness", "pnl_kurtosis"]
                kpis_to_display_portfolio = {key: portfolio_kpis[key] for key in portfolio_kpi_keys if key in portfolio_kpis}
                if kpis_to_display_portfolio:
                    KPIClusterDisplay(kpis_to_display_portfolio, KPI_CONFIG, portfolio_kpi_keys, cols_per_row=3).render()
                else: display_custom_message("Could not retrieve relevant KPIs for selected portfolio.", "warning")
                st.markdown("</div>", unsafe_allow_html=True)
            else: display_custom_message(f"Error calculating KPIs for selected portfolio: {portfolio_kpis.get('error', 'Unknown error') if portfolio_kpis else 'KPI calc failed'}", "error")
            
            st.subheader("📉 Selected Portfolio Combined Equity Curve & Drawdown")
            portfolio_equity_fig = plot_equity_curve_and_drawdown(
                portfolio_daily_trades_df, 
                date_col_actual, 
                'cumulative_pnl', 
                drawdown_pct_col='drawdown_pct',
                theme=plot_theme
            )
            if portfolio_equity_fig: st.plotly_chart(portfolio_equity_fig, use_container_width=True)
            else: display_custom_message("Could not generate equity curve for selected portfolio.", "warning")

            if not portfolio_daily_trades_df.empty:
                with st.expander("View Underlying Equity Curve Data"):
                    st.markdown("<div class='view-data-expander-content'>", unsafe_allow_html=True)
                    st.dataframe(portfolio_daily_trades_df)
                    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True) # End performance-section-container
    st.markdown("---")

    # --- Inter-Connections Section ---
    st.markdown("<div class='performance-section-container'>", unsafe_allow_html=True)
    st.header(f"🔗 Inter-Connections (Selected Portfolio: {', '.join(selected_accounts_for_portfolio)})")
    
    # Define correlation matrices here to ensure scope for "View Data"
    matrix_df_strat_corr = pd.DataFrame()
    matrix_df_acc_corr = pd.DataFrame()

    st.subheader("🔀 Inter-Strategy P&L Correlation")
    if strategy_col_actual not in portfolio_df.columns: 
        display_custom_message(f"Strategy column '{strategy_col_actual}' not found in selected portfolio data.", "warning")
    else:
        unique_strategies_selected_portfolio = portfolio_df[strategy_col_actual].dropna().astype(str).unique()
        if len(unique_strategies_selected_portfolio) < 2:
            st.info("At least two distinct strategies are needed within the selected portfolio for inter-strategy correlations.")
        else:
            cols_for_strat_corr = [strategy_col_actual, pnl_col_actual, date_col_actual]
            df_strat_corr_prep = portfolio_df[cols_for_strat_corr].copy()
            try:
                df_strat_corr_prep[date_col_actual] = pd.to_datetime(df_strat_corr_prep[date_col_actual], errors='coerce')
                df_strat_corr_prep.dropna(subset=[date_col_actual], inplace=True)
                df_strat_corr_prep[strategy_col_actual] = df_strat_corr_prep[strategy_col_actual].astype(str)
                df_strat_corr_prep[pnl_col_actual] = pd.to_numeric(df_strat_corr_prep[pnl_col_actual], errors='coerce')
                df_strat_corr_prep.dropna(subset=[pnl_col_actual], inplace=True)

                if df_strat_corr_prep.empty:
                     correlation_results_strat = {"error": "No valid data for strategy correlation after cleaning."}
                else:
                    df_strat_corr_prep = df_strat_corr_prep.sort_values(by=[date_col_actual, strategy_col_actual]).reset_index(drop=True)
                    with st.spinner("Calculating inter-strategy P&L correlations..."):
                        correlation_results_strat = portfolio_specific_service.get_portfolio_inter_strategy_correlation(
                            df_strat_corr_prep, strategy_col_actual, pnl_col_actual, date_col_actual)
                
                if correlation_results_strat and 'error' not in correlation_results_strat:
                    matrix_df_strat_corr = correlation_results_strat.get('correlation_matrix')
                    if matrix_df_strat_corr is not None and not matrix_df_strat_corr.empty and matrix_df_strat_corr.shape[0] > 1:
                        fig_strat_corr = go.Figure(data=go.Heatmap(z=matrix_df_strat_corr.values, x=matrix_df_strat_corr.columns, y=matrix_df_strat_corr.index, colorscale='RdBu', zmin=-1, zmax=1, text=matrix_df_strat_corr.round(2).astype(str), texttemplate="%{text}", hoverongaps=False))
                        fig_strat_corr.update_layout(title="Inter-Strategy Daily P&L Correlation")
                        st.plotly_chart(_apply_custom_theme(fig_strat_corr, plot_theme), use_container_width=True)
                        
                        with st.expander("View Inter-Strategy Correlation Matrix"):
                            st.markdown("<div class='view-data-expander-content'>", unsafe_allow_html=True)
                            st.dataframe(matrix_df_strat_corr)
                            st.markdown("</div>", unsafe_allow_html=True)
                    else: display_custom_message("Not enough data for inter-strategy correlation matrix.", "info")
                elif correlation_results_strat: display_custom_message(f"Inter-strategy correlation error: {correlation_results_strat.get('error')}", "error")
                else: display_custom_message("Inter-strategy correlation analysis failed.", "error")
            except Exception as e_strat_corr:
                logger.error(f"Error in inter-strategy correlation section: {e_strat_corr}", exc_info=True)
                display_custom_message(f"Error during inter-strategy correlation: {e_strat_corr}", "error")

    st.subheader("🤝 Inter-Account P&L Correlation")
    if len(selected_accounts_for_portfolio) < 2:
        st.info("At least two accounts must be selected in the sidebar for inter-account correlation.")
    else:
        cols_for_acc_corr = [account_col_actual, pnl_col_actual, date_col_actual]
        df_acc_corr_prep = portfolio_df[cols_for_acc_corr].copy() 
        try:
            df_acc_corr_prep[date_col_actual] = pd.to_datetime(df_acc_corr_prep[date_col_actual], errors='coerce')
            df_acc_corr_prep.dropna(subset=[date_col_actual], inplace=True)
            df_acc_corr_prep[account_col_actual] = df_acc_corr_prep[account_col_actual].astype(str)
            df_acc_corr_prep[pnl_col_actual] = pd.to_numeric(df_acc_corr_prep[pnl_col_actual], errors='coerce')
            df_acc_corr_prep.dropna(subset=[pnl_col_actual], inplace=True)

            if df_acc_corr_prep.empty:
                correlation_results_acc = {"error": "No valid data for account correlation after cleaning."}
            else:
                df_acc_corr_prep = df_acc_corr_prep.sort_values(by=[date_col_actual, account_col_actual]).reset_index(drop=True)
                with st.spinner("Calculating inter-account P&L correlations..."):
                    correlation_results_acc = portfolio_specific_service.get_portfolio_inter_account_correlation(
                        df_acc_corr_prep, account_col_actual, pnl_col_actual, date_col_actual)
            
            if correlation_results_acc and 'error' not in correlation_results_acc:
                matrix_df_acc_corr = correlation_results_acc.get('correlation_matrix')
                if matrix_df_acc_corr is not None and not matrix_df_acc_corr.empty and matrix_df_acc_corr.shape[0] > 1:
                    fig_acc_corr = go.Figure(data=go.Heatmap(z=matrix_df_acc_corr.values, x=matrix_df_acc_corr.columns, y=matrix_df_acc_corr.index, colorscale='RdBu', zmin=-1, zmax=1, text=matrix_df_acc_corr.round(2).astype(str), texttemplate="%{text}", hoverongaps=False))
                    fig_acc_corr.update_layout(title="Inter-Account Daily P&L Correlation")
                    st.plotly_chart(_apply_custom_theme(fig_acc_corr, plot_theme), use_container_width=True)

                    with st.expander("View Inter-Account Correlation Matrix"):
                        st.markdown("<div class='view-data-expander-content'>", unsafe_allow_html=True)
                        st.dataframe(matrix_df_acc_corr)
                        st.markdown("</div>", unsafe_allow_html=True)
                else: display_custom_message("Not enough data for inter-account correlation matrix.", "info")
            elif correlation_results_acc: display_custom_message(f"Inter-account correlation error: {correlation_results_acc.get('error')}", "error")
            else: display_custom_message("Inter-account correlation analysis failed.", "error")
        except Exception as e_acc_corr:
            logger.error(f"Error in inter-account correlation section: {e_acc_corr}", exc_info=True)
            display_custom_message(f"Error during inter-account correlation: {e_acc_corr}", "error")
    st.markdown("</div>", unsafe_allow_html=True) # End performance-section-container
    st.markdown("---")

    # --- Account Performance Breakdown Section ---
    st.markdown("<div class='performance-section-container'>", unsafe_allow_html=True)
    st.header(f"📊 Account Performance Breakdown (within Selected Portfolio: {', '.join(selected_accounts_for_portfolio)})")
    
    account_metrics_data = []
    summary_table_df = pd.DataFrame() # Define for "View Data" scope
    pnl_contribution_df_filtered = pd.DataFrame() # Define for "View Data" scope

    for acc_name_loop in selected_accounts_for_portfolio: 
        acc_df_original_trades = base_df[base_df[account_col_actual] == acc_name_loop].copy()
        if not acc_df_original_trades.empty:
            metrics = calculate_metrics_for_df(acc_df_original_trades, pnl_col_actual, date_col_actual, risk_free_rate, global_initial_capital)
            account_metrics_data.append({"Account": acc_name_loop, **metrics})
    
    if account_metrics_data:
        summary_table_df = pd.DataFrame(account_metrics_data)
        display_cols_summary = ["Account", "Total PnL", "Total Trades", "Win Rate %", "Avg Trade PnL", "Max Drawdown %", "Sharpe Ratio"]
        summary_table_df_display = summary_table_df[[col for col in display_cols_summary if col in summary_table_df.columns]]

        formatted_summary_df = summary_table_df_display.copy()
        if "Total PnL" in formatted_summary_df.columns:
            formatted_summary_df["Total PnL"] = formatted_summary_df["Total PnL"].apply(lambda x: format_currency(x) if pd.notna(x) else "N/A")
        if "Avg Trade PnL" in formatted_summary_df.columns:
            formatted_summary_df["Avg Trade PnL"] = formatted_summary_df["Avg Trade PnL"].apply(lambda x: format_currency(x) if pd.notna(x) else "N/A")
        if "Win Rate %" in formatted_summary_df.columns: 
            formatted_summary_df["Win Rate %"] = formatted_summary_df["Win Rate %"].apply(lambda x: format_percentage(x/100.0) if pd.notna(x) else "N/A")
        if "Max Drawdown %" in formatted_summary_df.columns: 
            formatted_summary_df["Max Drawdown %"] = formatted_summary_df["Max Drawdown %"].apply(lambda x: format_percentage(x/100.0) if pd.notna(x) else "N/A")
        if "Sharpe Ratio" in formatted_summary_df.columns:
            formatted_summary_df["Sharpe Ratio"] = formatted_summary_df["Sharpe Ratio"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        
        st.dataframe(formatted_summary_df.set_index("Account"), use_container_width=True)
        if not summary_table_df.empty:
            with st.expander("View Raw Account Performance Data"):
                st.markdown("<div class='view-data-expander-content'>", unsafe_allow_html=True)
                st.dataframe(summary_table_df)
                st.markdown("</div>", unsafe_allow_html=True)

        pnl_contribution_df = summary_table_df.copy() 
        pnl_contribution_df["Total PnL Numeric"] = pd.to_numeric(
            pnl_contribution_df["Total PnL"].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False), 
            errors='coerce'
        )
        pnl_contribution_df_filtered = pnl_contribution_df[pnl_contribution_df["Total PnL Numeric"] != 0].dropna(subset=["Total PnL Numeric"])
        if not pnl_contribution_df_filtered.empty:
            fig_pnl_contrib = px.pie(pnl_contribution_df_filtered, names='Account', values='Total PnL Numeric', title='P&L Contribution by Account (Selected Portfolio)', hole=0.3)
            fig_pnl_contrib.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(_apply_custom_theme(fig_pnl_contrib, plot_theme), use_container_width=True)

            with st.expander("View P&L Contribution Data"):
                st.markdown("<div class='view-data-expander-content'>", unsafe_allow_html=True)
                st.dataframe(pnl_contribution_df_filtered)
                st.markdown("</div>", unsafe_allow_html=True)
        else: st.info("No P&L contribution data to display (all selected accounts have zero or non-numeric P&L).")
    else: display_custom_message("Could not calculate performance metrics for individual accounts.", "warning")
    st.markdown("</div>", unsafe_allow_html=True) # End performance-section-container
    st.markdown("---")

    # --- Portfolio Optimization Section ---
    st.markdown("<div class='performance-section-container'>", unsafe_allow_html=True)
    st.header("⚖️ Portfolio Optimization")

    with st.expander("⚙️ Configure Portfolio Optimization", expanded=True):
        if strategy_col_actual not in portfolio_df.columns:
            st.warning(f"Strategy column ('{strategy_col_actual}') not found in the selected portfolio data. Cannot perform optimization.")
        else:
            optimizable_strategies = sorted(portfolio_df[strategy_col_actual].dropna().astype(str).unique())
            
            if not optimizable_strategies:
                 st.info("No strategies available in the selected portfolio for optimization.")
            else:
                with st.form("portfolio_optimization_form_v4"): 
                    st.markdown("""
                    Select strategies from the current portfolio to include in the optimization.
                    Optimization uses historical daily returns derived from P&L within the **globally filtered date range**.
                    """)
                    selected_strategies_for_opt = st.multiselect(
                        "Select Strategies for Optimization:",
                        options=optimizable_strategies,
                        default=optimizable_strategies[:min(len(optimizable_strategies), 5)], 
                        key="opt_strategies_select_v4"
                    )

                    optimization_objective_options = ["Maximize Sharpe Ratio", "Minimize Volatility", "Risk Parity"]
                    optimization_objective = st.selectbox(
                        "Optimization Objective:",
                        options=optimization_objective_options,
                        index=0,
                        key="opt_objective_v4"
                    )
                    
                    use_ledoit_wolf_covariance = st.checkbox(
                        "Use Ledoit-Wolf Covariance Shrinkage",
                        value=True, 
                        key="opt_ledoit_wolf_v4",
                        help="Shrinks the sample covariance matrix towards a structured estimator, often improving stability."
                    )

                    target_return_input_visibility = (optimization_objective == "Minimize Volatility")
                    target_return_input = None
                    if target_return_input_visibility:
                        target_return_input = st.number_input(
                            "Target Annualized Return (e.g., 0.10 for 10%):",
                            min_value=-1.0, max_value=2.0, value=0.10, step=0.01, format="%.2f", 
                            key="opt_target_return_v4",
                            help="Specify desired annualized portfolio return if minimizing volatility."
                        )
                    
                    lookback_days_opt = st.number_input(
                        "Historical Lookback for Returns/Covariance (days):",
                        min_value=20, 
                        max_value=max(20, len(portfolio_df[date_col_actual].unique()) if date_col_actual in portfolio_df else 20), 
                        value=min(252, max(20, len(portfolio_df[date_col_actual].unique()) if date_col_actual in portfolio_df else 20) ), 
                        step=10,
                        key="opt_lookback_days_v4",
                        help="Number of recent trading days for calculating expected returns and covariance."
                    )
                    
                    num_frontier_points_input = 25 
                    if optimization_objective in ["Maximize Sharpe Ratio", "Minimize Volatility"]:
                        num_frontier_points_input = st.number_input(
                            "Number of Points for Efficient Frontier Plot:",
                            min_value=10, max_value=100, value=25, step=5,
                            key="opt_frontier_points_v4",
                            help="More points provide a smoother frontier but take longer to compute."
                        )
                    
                    st.markdown("##### Per-Strategy Weight Constraints (Min/Max %)")
                    asset_bounds_input = []
                    current_weights_input_for_turnover = {} 

                    if selected_strategies_for_opt:
                        st.markdown("###### Define Current and Min/Max Allocation % for Each Selected Strategy:")
                        num_selected_opt_strats = len(selected_strategies_for_opt)
                        default_current_weight_pct = 100.0 / num_selected_opt_strats if num_selected_opt_strats > 0 else 0.0

                        for strat_name_opt_key in selected_strategies_for_opt:
                            st.write(f"**Strategy: {strat_name_opt_key}**")
                            cols_turnover = st.columns(3)
                            current_w_pct = cols_turnover[0].number_input(f"Current Weight %", min_value=0.0, max_value=100.0, value=default_current_weight_pct, step=1.0, format="%.1f", key=f"curr_w_{strat_name_opt_key}_v4")
                            min_w_pct = cols_turnover[1].number_input(f"Min Constraint %", min_value=0.0, max_value=100.0, value=0.0, step=1.0, format="%.1f", key=f"min_w_{strat_name_opt_key}_v4")
                            max_w_pct = cols_turnover[2].number_input(f"Max Constraint %", min_value=0.0, max_value=100.0, value=100.0, step=1.0, format="%.1f", key=f"max_w_{strat_name_opt_key}_v4")
                            
                            if min_w_pct > max_w_pct:
                                st.warning(f"For {strat_name_opt_key}, min weight ({min_w_pct}%) cannot exceed max weight ({max_w_pct}%). Adjusting max to {min_w_pct}%.")
                                max_w_pct = min_w_pct
                            
                            asset_bounds_input.append((min_w_pct / 100.0, max_w_pct / 100.0)) 
                            current_weights_input_for_turnover[strat_name_opt_key] = current_w_pct / 100.0
                    else:
                        st.caption("Select strategies above to set individual weight constraints and current weights.")

                    submit_optimization_button = st.form_submit_button("Optimize Portfolio")

                if submit_optimization_button and selected_strategies_for_opt:
                    min_strategies_needed = 1 if optimization_objective == "Risk Parity" else 2
                    sum_current_weights = sum(current_weights_input_for_turnover.values())
                    if not (0.999 < sum_current_weights < 1.001) and sum_current_weights != 0: 
                         display_custom_message(f"Sum of 'Current Weight %' ({sum_current_weights*100:.1f}%) should be close to 100% for turnover calculation. Please adjust.", "warning")
                    
                    if len(selected_strategies_for_opt) < min_strategies_needed:
                        display_custom_message(f"Please select at least {min_strategies_needed} strategies for '{optimization_objective}'.", "warning")
                    elif asset_bounds_input and sum(b[0] for b in asset_bounds_input) > 1.0 + 1e-6 : 
                        display_custom_message(f"Sum of minimum weight constraints ({sum(b[0]*100 for b in asset_bounds_input):.1f}%) exceeds 100%. Please adjust.", "error")
                    else:
                        with st.spinner("Preparing data and optimizing portfolio..."):
                            opt_df_filtered_strategies = portfolio_df[portfolio_df[strategy_col_actual].isin(selected_strategies_for_opt)].copy()
                            opt_df_filtered_strategies[date_col_actual] = pd.to_datetime(opt_df_filtered_strategies[date_col_actual])
                            opt_df_filtered_strategies.sort_values(by=date_col_actual, inplace=True)

                            latest_date_in_data = opt_df_filtered_strategies[date_col_actual].max() if not opt_df_filtered_strategies.empty else pd.Timestamp.now()
                            start_date_lookback = latest_date_in_data - pd.Timedelta(days=lookback_days_opt -1) 
                            opt_df_lookback = opt_df_filtered_strategies[opt_df_filtered_strategies[date_col_actual] >= start_date_lookback]
                            
                            if opt_df_lookback.empty:
                                display_custom_message("No data available for the selected strategies within the lookback period.", "warning")
                            else:
                                daily_pnl_pivot = opt_df_lookback.groupby(
                                    [opt_df_lookback[date_col_actual].dt.normalize(), strategy_col_actual]
                                )[pnl_col_actual].sum().unstack(fill_value=0)
                                daily_pnl_pivot = daily_pnl_pivot.reindex(columns=selected_strategies_for_opt, fill_value=0.0)

                                if global_initial_capital <= 0:
                                    display_custom_message("Initial capital must be positive to calculate returns for optimization.", "error")
                                else:
                                    daily_returns_for_opt = daily_pnl_pivot / global_initial_capital
                                    daily_returns_for_opt = daily_returns_for_opt.fillna(0) 

                                    min_hist_points = 20 if optimization_objective != "Risk Parity" or len(selected_strategies_for_opt) > 1 else 2
                                    if daily_returns_for_opt.empty or daily_returns_for_opt.shape[0] < min_hist_points : 
                                        display_custom_message(f"Not enough historical daily return data points ({daily_returns_for_opt.shape[0]}) for reliable optimization. Need at least {min_hist_points}.", "warning")
                                    else:
                                        opt_results = portfolio_specific_service.prepare_and_run_optimization(
                                            daily_returns_df=daily_returns_for_opt,
                                            objective=optimization_objective.lower().replace(" ", "_"),
                                            risk_free_rate=risk_free_rate,
                                            target_return_level=target_return_input if target_return_input_visibility else None,
                                            trading_days=252,
                                            num_frontier_points=num_frontier_points_input if optimization_objective in ["Maximize Sharpe Ratio", "Minimize Volatility"] else 0,
                                            use_ledoit_wolf=use_ledoit_wolf_covariance,
                                            asset_bounds=asset_bounds_input if asset_bounds_input else None
                                        )

                                        if opt_results and 'error' not in opt_results:
                                            st.success(f"Portfolio Optimization ({optimization_objective}) Complete!")
                                            
                                            st.subheader("⚖️ Optimal Portfolio Weights")
                                            optimal_weights_dict = opt_results['optimal_weights']
                                            weights_df = pd.DataFrame.from_dict(optimal_weights_dict, orient='index', columns=['Weight'])
                                            weights_df.index.name = "Strategy"
                                            weights_df["Weight %"] = (weights_df["Weight"] * 100) 
                                            
                                            st.dataframe(weights_df[["Weight %"]].style.format("{:.2f}%", subset=["Weight %"]))
                                            with st.expander("View Optimal Weights Data"):
                                                st.markdown("<div class='view-data-expander-content'>", unsafe_allow_html=True)
                                                st.dataframe(weights_df)
                                                st.markdown("</div>", unsafe_allow_html=True)


                                            fig_weights_pie = px.pie(
                                                weights_df[weights_df['Weight'] > 0.0001], 
                                                values='Weight', names=weights_df[weights_df['Weight'] > 0.0001].index, 
                                                title=f'Optimal Allocation ({optimization_objective})', hole=0.3
                                            )
                                            fig_weights_pie.update_traces(textposition='inside', textinfo='percent+label')
                                            st.plotly_chart(_apply_custom_theme(fig_weights_pie, plot_theme), use_container_width=True)

                                            if current_weights_input_for_turnover:
                                                turnover = calculate_portfolio_turnover(current_weights_input_for_turnover, optimal_weights_dict)
                                                st.metric(label="Portfolio Turnover", value=format_percentage(turnover))
                                            else: 
                                                 st.caption("Current weights not provided; turnover not calculated.")

                                            st.subheader(f"🚀 Optimized Portfolio Performance (Annualized) - {optimization_objective}")
                                            optimized_kpis_data = opt_results['performance']
                                            st.markdown("<div class='kpi-metrics-block'>", unsafe_allow_html=True)
                                            optimized_kpi_order = ["expected_annual_return", "annual_volatility", "sharpe_ratio"]
                                            KPIClusterDisplay(
                                                kpi_results=optimized_kpis_data,
                                                kpi_definitions=KPI_CONFIG, 
                                                kpi_order=optimized_kpi_order,
                                                cols_per_row=3
                                            ).render()
                                            st.markdown("</div>", unsafe_allow_html=True)
                                            
                                            if optimized_kpis_data:
                                                with st.expander("View Optimized Performance Data"):
                                                    st.markdown("<div class='view-data-expander-content'>", unsafe_allow_html=True)
                                                    st.dataframe(pd.DataFrame.from_dict(optimized_kpis_data, orient='index', columns=['Value']))
                                                    st.markdown("</div>", unsafe_allow_html=True)
                                            
                                            if "risk_contributions" in opt_results and opt_results["risk_contributions"]:
                                                st.subheader("🛡️ Risk Contributions to Portfolio Variance")
                                                rc_df = pd.DataFrame.from_dict(opt_results["risk_contributions"], orient='index', columns=['Risk Contribution %'])
                                                rc_df.index.name = "Strategy"
                                                rc_df_sorted = rc_df.sort_values(by="Risk Contribution %", ascending=False)

                                                fig_rc_bar = px.bar(
                                                    rc_df_sorted,
                                                    x=rc_df_sorted.index,
                                                    y="Risk Contribution %",
                                                    title="Percentage Risk Contribution by Strategy",
                                                    labels={"Risk Contribution %": "Risk Contribution (%)", "Strategy": "Strategy"},
                                                    color="Risk Contribution %",
                                                    color_continuous_scale=px.colors.sequential.Oranges_r 
                                                )
                                                fig_rc_bar.update_yaxes(ticksuffix="%")
                                                st.plotly_chart(_apply_custom_theme(fig_rc_bar, plot_theme), use_container_width=True)
                                                
                                                if not rc_df_sorted.empty:
                                                    with st.expander("View Risk Contribution Data"):
                                                        st.markdown("<div class='view-data-expander-content'>", unsafe_allow_html=True)
                                                        st.dataframe(rc_df_sorted)
                                                        st.markdown("</div>", unsafe_allow_html=True)


                                            if optimization_objective in ["Maximize Sharpe Ratio", "Minimize Volatility"]:
                                                st.subheader("🎯 Efficient Frontier")
                                                frontier_data = opt_results.get("efficient_frontier")
                                                if frontier_data and frontier_data.get('volatility') and frontier_data.get('return'):
                                                    max_sharpe_vol_plot, max_sharpe_ret_plot = None, None
                                                    min_vol_portfolio_vol, min_vol_portfolio_ret = None, None

                                                    if optimization_objective == "Maximize Sharpe Ratio":
                                                        max_sharpe_vol_plot = opt_results['performance']['annual_volatility']
                                                        max_sharpe_ret_plot = opt_results['performance']['expected_annual_return']
                                                    else: 
                                                        temp_frontier_df = pd.DataFrame(frontier_data)
                                                        if not temp_frontier_df.empty and 'volatility' in temp_frontier_df and temp_frontier_df['volatility'].gt(1e-9).any() : 
                                                            temp_frontier_df['sharpe'] = (temp_frontier_df['return'] - risk_free_rate) / temp_frontier_df['volatility'].replace(0, np.nan) 
                                                            if not temp_frontier_df['sharpe'].empty and not temp_frontier_df['sharpe'].isnull().all():
                                                                max_s_idx = temp_frontier_df['sharpe'].idxmax()
                                                                max_sharpe_vol_plot = temp_frontier_df.loc[max_s_idx, 'volatility']
                                                                max_sharpe_ret_plot = temp_frontier_df.loc[max_s_idx, 'return']
                                                    
                                                    temp_frontier_df_for_min_vol = pd.DataFrame(frontier_data)
                                                    if not temp_frontier_df_for_min_vol.empty:
                                                        min_vol_idx = temp_frontier_df_for_min_vol['volatility'].idxmin()
                                                        min_vol_portfolio_vol = temp_frontier_df_for_min_vol.loc[min_vol_idx, 'volatility']
                                                        min_vol_portfolio_ret = temp_frontier_df_for_min_vol.loc[min_vol_idx, 'return']

                                                    frontier_fig = plot_efficient_frontier(
                                                        frontier_vols=frontier_data['volatility'],
                                                        frontier_returns=frontier_data['return'],
                                                        max_sharpe_vol=max_sharpe_vol_plot,
                                                        max_sharpe_ret=max_sharpe_ret_plot,
                                                        min_vol_vol=min_vol_portfolio_vol,
                                                        min_vol_ret=min_vol_portfolio_ret,
                                                        theme=plot_theme
                                                    )
                                                    if frontier_fig:
                                                        st.plotly_chart(frontier_fig, use_container_width=True)
                                                        
                                                        frontier_df_display = pd.DataFrame(frontier_data)
                                                        if not frontier_df_display.empty:
                                                            with st.expander("View Efficient Frontier Data"):
                                                                st.markdown("<div class='view-data-expander-content'>", unsafe_allow_html=True)
                                                                st.dataframe(frontier_df_display)
                                                                st.markdown("</div>", unsafe_allow_html=True)
                                                    else:
                                                        display_custom_message("Could not generate the Efficient Frontier plot.", "warning")
                                                else:
                                                    display_custom_message("Efficient Frontier data not available or incomplete for plotting.", "info")
                                        
                                        elif opt_results:
                                            display_custom_message(f"Optimization Error: {opt_results.get('error')}", "error")
                                        else:
                                            display_custom_message("Portfolio optimization failed to return results.", "error")
    st.markdown("</div>", unsafe_allow_html=True) # End performance-section-container
    st.markdown("---")
    
    # --- Compare Equity Curves Section ---
    st.markdown("<div class='performance-section-container'>", unsafe_allow_html=True)
    st.header("↔️ Compare Equity Curves of Any Two Accounts")
    if len(unique_accounts_all) < 2:
        st.info("At least two distinct accounts are needed for this comparison.")
    else:
        col1_comp, col2_comp = st.columns(2)
        with col1_comp: selected_account_1_comp = st.selectbox("Select Account 1 for Comparison:", unique_accounts_all, index=0, key="portfolio_account_select_1_comp")
        with col2_comp: default_idx_acc2_comp = 1 if len(unique_accounts_all) > 1 else 0; selected_account_2_comp = st.selectbox("Select Account 2 for Comparison:", unique_accounts_all, index=default_idx_acc2_comp, key="portfolio_account_select_2_comp")
        
        if selected_account_1_comp == selected_account_2_comp: 
            st.warning("Please select two different accounts.")
        else:
            st.subheader(f"🆚 Equity Curve Comparison: {selected_account_1_comp} vs. {selected_account_2_comp}")
            df_acc1_comp = base_df[base_df[account_col_actual] == selected_account_1_comp].copy()
            df_acc2_comp = base_df[base_df[account_col_actual] == selected_account_2_comp].copy()
            
            # Initialize for "View Data" expander
            combined_equity_comp_df = pd.DataFrame()

            for df_comp_loop, acc_name_comp in zip([df_acc1_comp, df_acc2_comp], [selected_account_1_comp, selected_account_2_comp]): 
                df_comp_loop[date_col_actual] = pd.to_datetime(df_comp_loop[date_col_actual], errors='coerce')
                df_comp_loop.dropna(subset=[date_col_actual, pnl_col_actual], inplace=True)
                df_comp_loop.sort_values(by=date_col_actual, inplace=True)
                if not df_comp_loop.empty: 
                    df_comp_loop['cumulative_pnl'] = df_comp_loop[pnl_col_actual].cumsum()
                    temp_df = df_comp_loop[[date_col_actual, 'cumulative_pnl']].copy()
                    temp_df.rename(columns={'cumulative_pnl': f'Equity_{acc_name_comp}'}, inplace=True)
                    if combined_equity_comp_df.empty:
                        combined_equity_comp_df = temp_df
                    else:
                        combined_equity_comp_df = pd.merge(combined_equity_comp_df, temp_df, on=date_col_actual, how='outer')
            
            if combined_equity_comp_df.empty or df_acc1_comp.empty or df_acc2_comp.empty : 
                display_custom_message(f"One or both accounts ('{selected_account_1_comp}', '{selected_account_2_comp}') lack valid P&L data for comparison.", "warning")
            else:
                combined_equity_comp_df.sort_values(by=date_col_actual, inplace=True)
                combined_equity_comp_df = combined_equity_comp_df.fillna(method='ffill') # Forward fill for plotting alignment

                fig_comp_equity = go.Figure() 
                if f'Equity_{selected_account_1_comp}' in combined_equity_comp_df.columns:
                    fig_comp_equity.add_trace(go.Scatter(x=combined_equity_comp_df[date_col_actual], y=combined_equity_comp_df[f'Equity_{selected_account_1_comp}'], mode='lines', name=f"{selected_account_1_comp} Equity"))
                if f'Equity_{selected_account_2_comp}' in combined_equity_comp_df.columns:
                    fig_comp_equity.add_trace(go.Scatter(x=combined_equity_comp_df[date_col_actual], y=combined_equity_comp_df[f'Equity_{selected_account_2_comp}'], mode='lines', name=f"{selected_account_2_comp} Equity"))
                
                fig_comp_equity.update_layout(title=f"Equity Comparison: {selected_account_1_comp} vs. {selected_account_2_comp}", xaxis_title="Date", yaxis_title="Cumulative PnL", hovermode="x unified")
                st.plotly_chart(_apply_custom_theme(fig_comp_equity, plot_theme), use_container_width=True)

                if not combined_equity_comp_df.empty:
                    with st.expander("View Combined Equity Comparison Data"):
                        st.markdown("<div class='view-data-expander-content'>", unsafe_allow_html=True)
                        st.dataframe(combined_equity_comp_df)
                        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True) # End performance-section-container

if __name__ == "__main__":
    if 'app_initialized' not in st.session_state: 
        st.warning("This page is part of a multi-page app. Please run the main `app.py` script.")
    show_portfolio_analysis_page()
