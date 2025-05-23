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
    # Ensure these modules are in your project structure or PYTHONPATH
    from config import APP_TITLE, EXPECTED_COLUMNS, KPI_CONFIG, COLORS, RISK_FREE_RATE
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
        page_logger_name = "PortfolioAnalysisPage_ImportErrorLogger"
        if 'APP_TITLE' in globals():
            page_logger_name = f"{APP_TITLE}.PortfolioAnalysisPage.ImportError"
        page_logger = logging.getLogger(page_logger_name)
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


def _clean_data_for_analysis(
    df: pd.DataFrame,
    date_col: str,
    pnl_col: Optional[str] = None,
    strategy_col: Optional[str] = None,
    account_col: Optional[str] = None,
    required_cols_to_check_na: Optional[List[str]] = None,
    numeric_cols_to_convert: Optional[List[str]] = None,
    string_cols_to_convert: Optional[List[str]] = None,
    sort_by_date: bool = True
) -> pd.DataFrame:
    """
    Cleans and prepares a DataFrame for analysis.
    """
    if df.empty:
        logger.info("Input DataFrame for cleaning is empty.")
        return pd.DataFrame()

    df_cleaned = df.copy()

    if date_col not in df_cleaned.columns:
        logger.warning(f"Date column '{date_col}' not found in DataFrame for cleaning.")
        return pd.DataFrame()
    try:
        df_cleaned[date_col] = pd.to_datetime(df_cleaned[date_col], errors='coerce')
    except Exception as e:
        logger.error(f"Error converting date column '{date_col}' to datetime: {e}", exc_info=True)
        df_cleaned[date_col] = pd.NaT

    if pnl_col and pnl_col in df_cleaned.columns:
        try:
            df_cleaned[pnl_col] = pd.to_numeric(df_cleaned[pnl_col], errors='coerce')
        except Exception as e:
            logger.error(f"Error converting PnL column '{pnl_col}' to numeric: {e}", exc_info=True)
            df_cleaned[pnl_col] = np.nan

    if numeric_cols_to_convert:
        for col in numeric_cols_to_convert:
            if col in df_cleaned.columns:
                try:
                    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
                except Exception as e:
                    logger.error(f"Error converting column '{col}' to numeric: {e}", exc_info=True)
                    df_cleaned[col] = np.nan
            else:
                logger.debug(f"Numeric column '{col}' for conversion not found in DataFrame.")

    if string_cols_to_convert:
        for col in string_cols_to_convert:
            if col in df_cleaned.columns:
                try:
                    df_cleaned[col] = df_cleaned[col].astype(str)
                except Exception as e:
                    logger.error(f"Error converting column '{col}' to string: {e}", exc_info=True)
            else:
                logger.debug(f"String column '{col}' for conversion not found in DataFrame.")

    cols_for_nan_check = [date_col]
    if pnl_col and pnl_col in df_cleaned.columns: cols_for_nan_check.append(pnl_col)
    if strategy_col and strategy_col in df_cleaned.columns: cols_for_nan_check.append(strategy_col)
    if account_col and account_col in df_cleaned.columns: cols_for_nan_check.append(account_col)
    if required_cols_to_check_na:
        for rc in required_cols_to_check_na:
            if rc in df_cleaned.columns and rc not in cols_for_nan_check:
                cols_for_nan_check.append(rc)
    
    valid_cols_for_nan_check = [col for col in cols_for_nan_check if col in df_cleaned.columns]
    if valid_cols_for_nan_check:
        df_cleaned.dropna(subset=valid_cols_for_nan_check, inplace=True)
    else:
        logger.warning("No valid columns identified for NaN checking after initial processing.")


    if df_cleaned.empty:
        logger.info(f"DataFrame became empty after cleaning and NaN drop for columns: {valid_cols_for_nan_check}.")
        return pd.DataFrame()

    if sort_by_date and date_col in df_cleaned.columns:
        df_cleaned.sort_values(by=date_col, inplace=True)
    return df_cleaned


def _calculate_drawdown_series_for_aggregated_df(cumulative_pnl_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """ Helper to calculate absolute and percentage drawdown series. """
    if cumulative_pnl_series.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    high_water_mark = cumulative_pnl_series.cummax()
    drawdown_abs_series = high_water_mark - cumulative_pnl_series
    drawdown_pct_series = pd.Series(np.where(high_water_mark > 1e-9, (drawdown_abs_series / high_water_mark) * 100.0,
                                             np.where(drawdown_abs_series > 1e-9, 100.0, 0.0)),
                                    index=cumulative_pnl_series.index, dtype=float)
    return drawdown_abs_series.fillna(0), drawdown_pct_series.fillna(0)


@st.cache_data
def calculate_metrics_for_df(
    df_input: pd.DataFrame,
    pnl_col: str,
    date_col: str,
    risk_free_rate: float,
    initial_capital: float
) -> Dict[str, Any]:
    """Calculates core metrics for a given DataFrame of trades/pnl."""
    if df_input.empty:
        logger.info("calculate_metrics_for_df received an empty DataFrame.")
        return {"error": "Input DataFrame is empty."}

    df_copy = df_input.copy()
    if date_col not in df_copy.columns or pnl_col not in df_copy.columns:
        logger.warning(f"Essential columns ('{date_col}', '{pnl_col}') not in DataFrame for metric calculation.")
        return {"error": f"Missing essential columns: {date_col} or {pnl_col}"}

    try:
        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
        df_copy[pnl_col] = pd.to_numeric(df_copy[pnl_col], errors='coerce')
    except Exception as e:
        logger.error(f"Error during type conversion in calculate_metrics_for_df: {e}", exc_info=True)
        return {"error": f"Type conversion failed: {e}"}

    df_copy.dropna(subset=[date_col, pnl_col], inplace=True)
    if df_copy.empty:
        logger.info("DataFrame empty after NaN drop in calculate_metrics_for_df.")
        return {"error": "No valid data after cleaning."}
    df_copy.sort_values(by=date_col, inplace=True)

    kpis = general_analysis_service.get_core_kpis(df_copy, risk_free_rate, initial_capital=initial_capital)
    if kpis and 'error' not in kpis:
        return {
            "Total PnL": kpis.get("total_pnl", 0.0), "Total Trades": kpis.get("total_trades", 0),
            "Win Rate %": kpis.get("win_rate", 0.0), "Avg Trade PnL": kpis.get("avg_trade_pnl", 0.0),
            "Max Drawdown %": kpis.get("max_drawdown_pct", 0.0), "Sharpe Ratio": kpis.get("sharpe_ratio", 0.0)
        }
    error_msg = kpis.get('error', 'Unknown error') if kpis else 'KPI calc failed'
    logger.warning(f"KPI calculation failed in calculate_metrics_for_df: {error_msg}")
    return {"error": error_msg}

@st.cache_data
def _run_portfolio_optimization_logic(
    portfolio_df_data_tuple: Tuple[List[tuple], List[str]],
    strategy_col_actual: str, date_col_actual: str, pnl_col_actual: str,
    selected_strategies_for_opt_tuple: Tuple[str, ...],
    lookback_days_opt: int, global_initial_capital: float,
    optimization_objective_key: str, risk_free_rate: float,
    target_return_val: Optional[float], num_frontier_points: int,
    use_ledoit_wolf: bool, asset_bounds_list_of_tuples: Optional[List[Tuple[float, float]]]
) -> Dict[str, Any]:
    """ Encapsulates the data preparation and optimization call. """
    portfolio_df_data, portfolio_df_columns = portfolio_df_data_tuple
    portfolio_df = pd.DataFrame(data=portfolio_df_data, columns=portfolio_df_columns)
    selected_strategies_for_opt = list(selected_strategies_for_opt_tuple)

    opt_df_filtered_strategies = portfolio_df[portfolio_df[strategy_col_actual].isin(selected_strategies_for_opt)].copy()
    opt_df_filtered_strategies = _clean_data_for_analysis(
        opt_df_filtered_strategies, date_col=date_col_actual, pnl_col=pnl_col_actual,
        strategy_col=strategy_col_actual, required_cols_to_check_na=[pnl_col_actual, strategy_col_actual],
        sort_by_date=True
    )
    if opt_df_filtered_strategies.empty:
        return {"error": "No data for selected strategies after cleaning for optimization."}

    latest_date_in_data = opt_df_filtered_strategies[date_col_actual].max()
    start_date_lookback = latest_date_in_data - pd.Timedelta(days=lookback_days_opt - 1)
    opt_df_lookback = opt_df_filtered_strategies[opt_df_filtered_strategies[date_col_actual] >= start_date_lookback]
    if opt_df_lookback.empty:
        return {"error": "No data within lookback period."}

    try:
        daily_pnl_pivot = opt_df_lookback.groupby(
            [opt_df_lookback[date_col_actual].dt.normalize(), strategy_col_actual]
        )[pnl_col_actual].sum().unstack(fill_value=0)
        daily_pnl_pivot = daily_pnl_pivot.reindex(columns=selected_strategies_for_opt, fill_value=0.0)
    except (KeyError, ValueError, TypeError, AttributeError) as e:
        logger.error(f"Error during P&L pivot for optimization: {e}", exc_info=True)
        return {"error": f"Failed to pivot P&L data: {e}"}

    if global_initial_capital <= 0: return {"error": "Initial capital must be positive."}
    daily_returns_for_opt = (daily_pnl_pivot / global_initial_capital).fillna(0)
    min_hist_points_needed = 20 if not (optimization_objective_key == "risk_parity" and len(selected_strategies_for_opt) <= 1) else 2
    if daily_returns_for_opt.shape[0] < min_hist_points_needed:
        return {"error": f"Need at least {min_hist_points_needed} historical data points, found {daily_returns_for_opt.shape[0]}."}

    try:
        return portfolio_specific_service.prepare_and_run_optimization(
            daily_returns_df=daily_returns_for_opt, objective=optimization_objective_key,
            risk_free_rate=risk_free_rate, target_return_level=target_return_val,
            trading_days=252, num_frontier_points=num_frontier_points,
            use_ledoit_wolf=use_ledoit_wolf, asset_bounds=asset_bounds_list_of_tuples
        )
    except Exception as e:
        logger.error(f"Error in optimization service: {e}", exc_info=True)
        return {"error": f"Optimization service failed: {e}"}


def show_portfolio_analysis_page():
    st.title("💼 Portfolio-Level Analysis")
    logger.info("Rendering Portfolio Analysis Page.")

    if 'processed_data' not in st.session_state or st.session_state.processed_data is None or st.session_state.processed_data.empty:
        display_custom_message("Please upload and process data to view portfolio analysis.", "info")
        logger.info("Portfolio analysis page: No processed data found.")
        return

    base_df = st.session_state.processed_data
    plot_theme = st.session_state.get('current_theme', 'dark') 
    risk_free_rate = st.session_state.get('risk_free_rate', RISK_FREE_RATE)
    global_initial_capital = st.session_state.get('initial_capital', 100000.0)

    account_col_actual = EXPECTED_COLUMNS.get('account_str')
    pnl_col_actual = EXPECTED_COLUMNS.get('pnl')
    date_col_actual = EXPECTED_COLUMNS.get('date')
    strategy_col_actual = EXPECTED_COLUMNS.get('strategy')

    if not all([account_col_actual, pnl_col_actual, date_col_actual, strategy_col_actual]):
        missing_configs = [col_type for col_type, col_val in zip(
            ['account', 'pnl', 'date', 'strategy'],
            [account_col_actual, pnl_col_actual, date_col_actual, strategy_col_actual]) if not col_val]
        msg = f"Essential column configurations missing ({', '.join(missing_configs)}). Analysis cannot proceed."
        display_custom_message(msg, "error"); logger.error(f"Portfolio page: {msg}"); return

    essential_cols_in_df = [account_col_actual, pnl_col_actual, date_col_actual, strategy_col_actual]
    if not all(col in base_df.columns for col in essential_cols_in_df):
        missing_cols_in_df = [col for col in essential_cols_in_df if col not in base_df.columns]
        msg = f"Essential columns ({', '.join(missing_cols_in_df)}) not found in uploaded data."
        display_custom_message(msg, "error"); logger.error(f"Portfolio page: {msg}. Available: {base_df.columns.tolist()}"); return

    unique_accounts_all = sorted(base_df[account_col_actual].dropna().astype(str).unique())
    if not unique_accounts_all:
        display_custom_message("No accounts found in the data.", "info"); return

    st.sidebar.subheader("Portfolio Account Selection")
    selected_accounts_for_portfolio = unique_accounts_all
    if len(unique_accounts_all) > 1:
        selected_accounts_for_portfolio = st.sidebar.multiselect(
            "Select accounts for portfolio view:", options=unique_accounts_all,
            default=unique_accounts_all, key="portfolio_view_account_multiselect_v2" 
        )
    else:
        st.sidebar.info(f"Displaying portfolio view for the single account: {unique_accounts_all[0]}")

    if not selected_accounts_for_portfolio:
        display_custom_message("Please select at least one account for the portfolio view.", "info"); return

    portfolio_df_uncleaned = base_df[base_df[account_col_actual].isin(selected_accounts_for_portfolio)].copy()
    if portfolio_df_uncleaned.empty:
        display_custom_message("No data for the selected accounts.", "info"); return
    
    portfolio_df = _clean_data_for_analysis(
        portfolio_df_uncleaned, date_col=date_col_actual, pnl_col=pnl_col_actual,
        strategy_col=strategy_col_actual, account_col=account_col_actual,
        required_cols_to_check_na=[pnl_col_actual, strategy_col_actual, account_col_actual],
        string_cols_to_convert=[strategy_col_actual, account_col_actual]
    )
    if portfolio_df.empty:
        display_custom_message("No valid data after cleaning for selected accounts.", "warning"); return

    tab_titles = [
        "📈 Overall Performance", "🔗 Inter-Connections", "📊 Account Breakdown",
        "⚖️ Portfolio Optimization", "↔️ Equity Comparison"
    ]
    tab_overall, tab_connections, tab_breakdown, tab_optimization, tab_comparison = st.tabs(tab_titles)

    with tab_overall:
        st.header(f"Overall Performance for Selected Portfolio ({', '.join(selected_accounts_for_portfolio)})")
        portfolio_daily_trades_df = pd.DataFrame()
        try:
            portfolio_daily_pnl = portfolio_df.groupby(portfolio_df[date_col_actual].dt.normalize())[pnl_col_actual].sum()
        except (AttributeError, KeyError, TypeError) as e:
            logger.error(f"Error grouping by date for daily P&L: {e}", exc_info=True)
            display_custom_message(f"Error processing daily P&L: {e}", "error")
            portfolio_daily_pnl = pd.Series(dtype=float)

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
            else:
                portfolio_daily_trades_df['drawdown_abs'] = pd.Series(dtype=float)
                portfolio_daily_trades_df['drawdown_pct'] = pd.Series(dtype=float)

            with st.spinner("Calculating selected portfolio KPIs..."):
                portfolio_kpis = general_analysis_service.get_core_kpis(portfolio_daily_trades_df, risk_free_rate, initial_capital=global_initial_capital)
            
            if portfolio_kpis and 'error' not in portfolio_kpis:
                portfolio_kpi_keys = ["total_pnl", "sharpe_ratio", "sortino_ratio", "calmar_ratio", "max_drawdown_abs", "max_drawdown_pct", "avg_daily_pnl", "pnl_skewness", "pnl_kurtosis"]
                kpis_to_display_portfolio = {key: portfolio_kpis[key] for key in portfolio_kpi_keys if key in portfolio_kpis}
                if kpis_to_display_portfolio:
                    KPIClusterDisplay(kpis_to_display_portfolio, KPI_CONFIG, portfolio_kpi_keys, cols_per_row=3).render()
                else: display_custom_message("Could not retrieve relevant KPIs for selected portfolio.", "warning")
            else: display_custom_message(f"Error calculating KPIs: {portfolio_kpis.get('error', 'Unknown') if portfolio_kpis else 'KPI calc failed'}", "error")
            
            st.subheader("Combined Equity Curve & Drawdown")
            portfolio_equity_fig = plot_equity_curve_and_drawdown(
                portfolio_daily_trades_df, date_col_actual, 'cumulative_pnl', 'drawdown_pct', theme=plot_theme
            )
            if portfolio_equity_fig: st.plotly_chart(portfolio_equity_fig, use_container_width=True)
            else: display_custom_message("Could not generate equity curve.", "warning")

            if not portfolio_daily_trades_df.empty:
                with st.expander("View Underlying Equity Curve Data"):
                    st.dataframe(portfolio_daily_trades_df)

    with tab_connections:
        st.header(f"Inter-Connections (Selected Portfolio: {', '.join(selected_accounts_for_portfolio)})")
        st.subheader("🔀 Inter-Strategy P&L Correlation")
        if strategy_col_actual not in portfolio_df.columns:
            display_custom_message(f"Strategy column '{strategy_col_actual}' not found.", "warning")
        else:
            unique_strategies_sel_portfolio = portfolio_df[strategy_col_actual].dropna().unique()
            if len(unique_strategies_sel_portfolio) < 2:
                st.info("At least two distinct strategies needed for inter-strategy correlations.")
            else:
                df_strat_corr_prep = portfolio_df[[date_col_actual, strategy_col_actual, pnl_col_actual]].copy()
                df_strat_corr_prep = df_strat_corr_prep.sort_values(by=[date_col_actual, strategy_col_actual]).reset_index(drop=True)
                with st.spinner("Calculating inter-strategy P&L correlations..."):
                    try:
                        correlation_results_strat = portfolio_specific_service.get_portfolio_inter_strategy_correlation(
                            df_strat_corr_prep, strategy_col_actual, pnl_col_actual, date_col_actual)
                    except Exception as e:
                        logger.error(f"Inter-strategy correlation service error: {e}", exc_info=True)
                        correlation_results_strat = {"error": f"Service failed: {e}"}

                if correlation_results_strat and 'error' not in correlation_results_strat:
                    matrix_df_strat_corr = correlation_results_strat.get('correlation_matrix')
                    if matrix_df_strat_corr is not None and not matrix_df_strat_corr.empty and matrix_df_strat_corr.shape[0] > 1:
                        fig_strat_corr = go.Figure(data=go.Heatmap(z=matrix_df_strat_corr.values, x=matrix_df_strat_corr.columns, y=matrix_df_strat_corr.index, colorscale='RdBu', zmin=-1, zmax=1, text=matrix_df_strat_corr.round(2).astype(str), texttemplate="%{text}", hoverongaps=False))
                        fig_strat_corr.update_layout(title="Inter-Strategy Daily P&L Correlation")
                        st.plotly_chart(_apply_custom_theme(fig_strat_corr, plot_theme), use_container_width=True)
                        with st.expander("View Inter-Strategy Correlation Matrix"): st.dataframe(matrix_df_strat_corr)
                    else: display_custom_message("Not enough data for inter-strategy correlation matrix.", "info")
                elif correlation_results_strat: display_custom_message(f"Inter-strategy correlation error: {correlation_results_strat.get('error')}", "error")
                else: display_custom_message("Inter-strategy correlation analysis failed.", "error")

        st.subheader("🤝 Inter-Account P&L Correlation")
        if len(selected_accounts_for_portfolio) < 2:
            st.info("At least two accounts needed for inter-account correlation.")
        else:
            df_acc_corr_prep = portfolio_df[[date_col_actual, account_col_actual, pnl_col_actual]].copy()
            df_acc_corr_prep = df_acc_corr_prep.sort_values(by=[date_col_actual, account_col_actual]).reset_index(drop=True)
            with st.spinner("Calculating inter-account P&L correlations..."):
                try:
                    correlation_results_acc = portfolio_specific_service.get_portfolio_inter_account_correlation(
                        df_acc_corr_prep, account_col_actual, pnl_col_actual, date_col_actual)
                except Exception as e:
                    logger.error(f"Inter-account correlation service error: {e}", exc_info=True)
                    correlation_results_acc = {"error": f"Service failed: {e}"}

            if correlation_results_acc and 'error' not in correlation_results_acc:
                matrix_df_acc_corr = correlation_results_acc.get('correlation_matrix')
                if matrix_df_acc_corr is not None and not matrix_df_acc_corr.empty and matrix_df_acc_corr.shape[0] > 1:
                    fig_acc_corr = go.Figure(data=go.Heatmap(z=matrix_df_acc_corr.values, x=matrix_df_acc_corr.columns, y=matrix_df_acc_corr.index, colorscale='RdBu', zmin=-1, zmax=1, text=matrix_df_acc_corr.round(2).astype(str), texttemplate="%{text}", hoverongaps=False))
                    fig_acc_corr.update_layout(title="Inter-Account Daily P&L Correlation")
                    st.plotly_chart(_apply_custom_theme(fig_acc_corr, plot_theme), use_container_width=True)
                    with st.expander("View Inter-Account Correlation Matrix"): st.dataframe(matrix_df_acc_corr)
                else: display_custom_message("Not enough data for inter-account correlation matrix.", "info")
            elif correlation_results_acc: display_custom_message(f"Inter-account correlation error: {correlation_results_acc.get('error')}", "error")
            else: display_custom_message("Inter-account correlation analysis failed.", "error")

    with tab_breakdown:
        st.header(f"Account Performance Breakdown (Portfolio: {', '.join(selected_accounts_for_portfolio)})")
        account_metrics_data = []
        for acc_name_loop in selected_accounts_for_portfolio:
            acc_df_original_trades = base_df[base_df[account_col_actual] == acc_name_loop].copy()
            if not acc_df_original_trades.empty:
                metrics = calculate_metrics_for_df(acc_df_original_trades, pnl_col_actual, date_col_actual, risk_free_rate, global_initial_capital)
                account_metrics_data.append({"Account": acc_name_loop, **metrics})
            else: logger.info(f"No original trade data for account {acc_name_loop} in breakdown.")

        if account_metrics_data:
            summary_table_df = pd.DataFrame(account_metrics_data)
            if "Total PnL" in summary_table_df.columns:
                summary_table_df["Total PnL Numeric"] = pd.to_numeric(summary_table_df["Total PnL"], errors='coerce') 
                pnl_for_chart_df = summary_table_df[
                    summary_table_df["Total PnL Numeric"].notna() & (summary_table_df["Total PnL Numeric"] != 0)
                ][["Account", "Total PnL Numeric"]].copy()

                if not pnl_for_chart_df.empty:
                    fig_pnl_contrib = px.pie(pnl_for_chart_df, names='Account', values='Total PnL Numeric', title='P&L Contribution by Account', hole=0.3)
                    fig_pnl_contrib.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(_apply_custom_theme(fig_pnl_contrib, plot_theme), use_container_width=True)
                    with st.expander("View P&L Contribution Data (Numeric)"): st.dataframe(pnl_for_chart_df.rename(columns={"Total PnL Numeric": "Total PnL"}))
                else: st.info("No P&L contribution data to display (zero, NaN, or non-numeric P&L).")
            else: st.warning("Total PnL column not found for P&L contribution chart.")

            display_cols_summary = ["Account", "Total PnL", "Total Trades", "Win Rate %", "Avg Trade PnL", "Max Drawdown %", "Sharpe Ratio"]
            valid_display_cols = [col for col in display_cols_summary if col in summary_table_df.columns]
            summary_table_df_display = summary_table_df[valid_display_cols].copy()
            
            for col, item_type in [("Total PnL", "currency"), ("Avg Trade PnL", "currency"), 
                                   ("Win Rate %", "percentage"), ("Max Drawdown %", "percentage"),
                                   ("Sharpe Ratio", "float")]:
                if col in summary_table_df_display.columns:
                    summary_table_df_display[col] = pd.to_numeric(summary_table_df_display[col], errors='coerce')
                    if item_type == "currency": summary_table_df_display[col] = summary_table_df_display[col].apply(lambda x: format_currency(x) if pd.notna(x) else "N/A")
                    elif item_type == "percentage": summary_table_df_display[col] = summary_table_df_display[col].apply(lambda x: format_percentage(x/100.0) if pd.notna(x) else "N/A")
                    elif item_type == "float": summary_table_df_display[col] = summary_table_df_display[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
            
            st.dataframe(summary_table_df_display.set_index("Account"), use_container_width=True)
            if not summary_table_df.empty:
                with st.expander("View Raw Account Performance Data (Pre-formatting)"):
                    st.dataframe(summary_table_df.drop(columns=["Total PnL Numeric"], errors='ignore'))
        else: display_custom_message("Could not calculate performance metrics for individual accounts.", "warning")

    with tab_optimization:
        st.header("Portfolio Optimization")
        selected_strategies_for_opt_form = []
        optimization_objective_form_key = "" 
        optimization_objective_display_form = "" 
        use_ledoit_wolf_covariance_form = True
        target_return_input_form = None
        lookback_days_opt_form = 252
        num_frontier_points_input_form = 25
        asset_bounds_input_form = []
        current_weights_input_for_turnover_form = {}

        with st.expander("⚙️ Configure Portfolio Optimization", expanded=True):
            if strategy_col_actual not in portfolio_df.columns:
                st.warning(f"Strategy column ('{strategy_col_actual}') not found. Cannot perform optimization.")
            else:
                optimizable_strategies = sorted(portfolio_df[strategy_col_actual].dropna().unique())
                if not optimizable_strategies:
                     st.info("No strategies available in selected portfolio for optimization.")
                else:
                    with st.form("portfolio_optimization_form_v6"): 
                        st.markdown("Select strategies, objective, and constraints for optimization.")
                        selected_strategies_for_opt_form = st.multiselect(
                            "Select Strategies:", options=optimizable_strategies,
                            default=optimizable_strategies[:min(len(optimizable_strategies), 5)], key="opt_strat_sel_v6"
                        )
                        optimization_objective_options_map = {
                            "Maximize Sharpe Ratio": "maximize_sharpe_ratio", "Minimize Volatility": "minimize_volatility", "Risk Parity": "risk_parity"
                        }
                        optimization_objective_display_form = st.selectbox(
                            "Objective:", options=list(optimization_objective_options_map.keys()), index=0, key="opt_obj_v6"
                        )
                        optimization_objective_form_key = optimization_objective_options_map[optimization_objective_display_form]

                        use_ledoit_wolf_covariance_form = st.checkbox("Use Ledoit-Wolf Covariance", True, key="opt_lw_v6")
                        if optimization_objective_form_key == "minimize_volatility":
                            target_return_input_form = st.number_input("Target Annualized Return (e.g., 0.10 for 10%):", -1.0, 2.0, 0.10, 0.01, "%.2f", key="opt_tr_v6")
                        
                        max_lookback = max(20, len(portfolio_df[date_col_actual].unique())) if date_col_actual in portfolio_df and not portfolio_df.empty else 20
                        lookback_days_opt_form = st.number_input("Lookback (days):", 20, max_lookback, min(252, max_lookback), 10, key="opt_lb_v6")
                        if optimization_objective_form_key in ["maximize_sharpe_ratio", "minimize_volatility"]:
                            num_frontier_points_input_form = st.number_input("Frontier Points:", 10, 100, 25, 5, key="opt_fp_v6")

                        st.markdown("##### Per-Strategy Weight Constraints (Min/Max %)")
                        asset_bounds_input_form = []
                        current_weights_input_for_turnover_form = {}
                        if selected_strategies_for_opt_form:
                            num_sel = len(selected_strategies_for_opt_form)
                            def_w = 100.0 / num_sel if num_sel > 0 else 0.0
                            for strat in selected_strategies_for_opt_form:
                                cols = st.columns(3)
                                cur = cols[0].number_input(f"Cur W % ({strat})", 0.0, 100.0, def_w, 1.0, "%.1f", key=f"cur_w_{strat}_v6")
                                min_ = cols[1].number_input(f"Min W % ({strat})", 0.0, 100.0, 0.0, 1.0, "%.1f", key=f"min_w_{strat}_v6")
                                max_ = cols[2].number_input(f"Max W % ({strat})", 0.0, 100.0, 100.0, 1.0, "%.1f", key=f"max_w_{strat}_v6")
                                if min_ > max_: max_ = min_
                                asset_bounds_input_form.append((min_ / 100.0, max_ / 100.0))
                                current_weights_input_for_turnover_form[strat] = cur / 100.0
                        else: st.caption("Select strategies to set weights.")
                        submit_optimization_button = st.form_submit_button("Optimize Portfolio")

        if submit_optimization_button and selected_strategies_for_opt_form:
            min_strats = 1 if optimization_objective_form_key == "risk_parity" else 2
            sum_cur_w = sum(current_weights_input_for_turnover_form.values())
            if not (0.999 < sum_cur_w < 1.001) and sum_cur_w != 0.0 and current_weights_input_for_turnover_form:
                display_custom_message(f"Sum of 'Current Weight %' ({sum_cur_w*100:.1f}%) should be ~100% or 0%.", "warning")
            if len(selected_strategies_for_opt_form) < min_strats:
                display_custom_message(f"Select at least {min_strats} strategies for '{optimization_objective_display_form}'.", "warning")
            elif asset_bounds_input_form and sum(b[0] for b in asset_bounds_input_form) > 1.0 + 1e-6 :
                display_custom_message(f"Sum of min weight constraints ({sum(b[0]*100 for b in asset_bounds_input_form):.1f}%) > 100%.", "error")
            else:
                with st.spinner("Optimizing portfolio..."):
                    portfolio_df_tuple_for_opt = (portfolio_df.to_records(index=False).tolist(), portfolio_df.columns.tolist())
                    opt_results = _run_portfolio_optimization_logic(
                        portfolio_df_data_tuple=portfolio_df_tuple_for_opt, strategy_col_actual=strategy_col_actual,
                        date_col_actual=date_col_actual, pnl_col_actual=pnl_col_actual,
                        selected_strategies_for_opt_tuple=tuple(selected_strategies_for_opt_form),
                        lookback_days_opt=lookback_days_opt_form, global_initial_capital=global_initial_capital,
                        optimization_objective_key=optimization_objective_form_key, risk_free_rate=risk_free_rate,
                        target_return_val=target_return_input_form, num_frontier_points=num_frontier_points_input_form,
                        use_ledoit_wolf=use_ledoit_wolf_covariance_form, asset_bounds_list_of_tuples=asset_bounds_input_form
                    )
                
                if opt_results and 'error' not in opt_results:
                    st.success(f"Optimization ({optimization_objective_display_form}) Complete!")
                    st.subheader("Optimal Portfolio Weights")
                    optimal_weights_dict = opt_results.get('optimal_weights', {})
                    if optimal_weights_dict:
                        weights_df = pd.DataFrame.from_dict(optimal_weights_dict, orient='index', columns=['Weight'])
                        weights_df["Weight %"] = (weights_df["Weight"] * 100)
                        st.dataframe(weights_df[["Weight %"]].style.format("{:.2f}%"))
                        with st.expander("View Optimal Weights Data (Numeric)"): st.dataframe(weights_df)
                        fig_pie = px.pie(weights_df[weights_df['Weight'] > 1e-5], values='Weight', names=weights_df[weights_df['Weight'] > 1e-5].index, title=f'Optimal Allocation ({optimization_objective_display_form})', hole=0.3)
                        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(_apply_custom_theme(fig_pie, plot_theme), use_container_width=True)
                        if current_weights_input_for_turnover_form:
                            turnover = calculate_portfolio_turnover(current_weights_input_for_turnover_form, optimal_weights_dict)
                            st.metric(label="Portfolio Turnover", value=format_percentage(turnover))
                    else: st.warning("Optimal weights not found in results.")

                    st.subheader(f"Optimized Portfolio Performance (Annualized) - {optimization_objective_display_form}")
                    optimized_kpis = opt_results.get('performance', {})
                    if optimized_kpis:
                        # Corrected KPIClusterDisplay call
                        KPIClusterDisplay(
                            kpi_results=optimized_kpis,
                            kpi_definitions=KPI_CONFIG,
                            kpi_order=["expected_annual_return", "annual_volatility", "sharpe_ratio"],
                            cols_per_row=3 # Explicitly named
                        ).render()
                        with st.expander("View Optimized Performance Data"): st.dataframe(pd.DataFrame.from_dict(optimized_kpis, orient='index', columns=['Value']))
                    else: st.warning("Optimized performance KPIs not found.")

                    if "risk_contributions" in opt_results and opt_results["risk_contributions"]:
                        st.subheader("Risk Contributions to Portfolio Variance")
                        rc_data = opt_results["risk_contributions"]
                        if rc_data:
                            rc_df = pd.DataFrame.from_dict(rc_data, orient='index', columns=['Risk Contribution %']).sort_values(by="Risk Contribution %", ascending=False)
                            fig_rc = px.bar(rc_df, x=rc_df.index, y="Risk Contribution %", title="Percentage Risk Contribution", labels={"Risk Contribution %": "Risk Contrib. (%)"}, color="Risk Contribution %", color_continuous_scale=px.colors.sequential.Oranges_r)
                            fig_rc.update_yaxes(ticksuffix="%")
                            st.plotly_chart(_apply_custom_theme(fig_rc, plot_theme), use_container_width=True)
                            with st.expander("View Risk Contribution Data"): st.dataframe(rc_df)
                        else: st.info("Risk contribution data is empty.")
                    
                    if optimization_objective_form_key in ["maximize_sharpe_ratio", "minimize_volatility"]:
                        st.subheader("Efficient Frontier")
                        frontier_data = opt_results.get("efficient_frontier")
                        if frontier_data and frontier_data.get('volatility') and frontier_data.get('return'):
                            perf_data = opt_results.get('performance', {})
                            max_s_vol, max_s_ret, min_v_vol, min_v_ret = None, None, None, None
                            if optimization_objective_form_key == "maximize_sharpe_ratio":
                                max_s_vol, max_s_ret = perf_data.get('annual_volatility'), perf_data.get('expected_annual_return')
                            else: 
                                temp_f_df = pd.DataFrame(frontier_data)
                                if not temp_f_df.empty and 'volatility' in temp_f_df and temp_f_df['volatility'].gt(1e-9).any():
                                    temp_f_df['sharpe'] = (temp_f_df['return'] - risk_free_rate) / temp_f_df['volatility'].replace(0, np.nan)
                                    if not temp_f_df['sharpe'].isnull().all():
                                        idx = temp_f_df['sharpe'].idxmax()
                                        max_s_vol, max_s_ret = temp_f_df.loc[idx, 'volatility'], temp_f_df.loc[idx, 'return']
                            
                            temp_f_df_min_vol = pd.DataFrame(frontier_data) 
                            if not temp_f_df_min_vol.empty and 'volatility' in temp_f_df_min_vol:
                                idx = temp_f_df_min_vol['volatility'].idxmin()
                                min_v_vol, min_v_ret = temp_f_df_min_vol.loc[idx, 'volatility'], temp_f_df_min_vol.loc[idx, 'return']

                            frontier_fig = plot_efficient_frontier(frontier_data['volatility'], frontier_data['return'], max_s_vol, max_s_ret, min_v_vol, min_v_ret, theme=plot_theme)
                            if frontier_fig:
                                st.plotly_chart(frontier_fig, use_container_width=True)
                                if not pd.DataFrame(frontier_data).empty:
                                    with st.expander("View Efficient Frontier Data"): st.dataframe(pd.DataFrame(frontier_data))
                            else: display_custom_message("Could not generate Efficient Frontier plot.", "warning")
                        else: display_custom_message("Efficient Frontier data not available/incomplete.", "info")
                elif opt_results: display_custom_message(f"Optimization Error: {opt_results.get('error')}", "error")
                else: display_custom_message("Portfolio optimization failed.", "error")

    with tab_comparison:
        st.header("Compare Equity Curves of Any Two Accounts")
        if len(unique_accounts_all) < 2:
            st.info("At least two distinct accounts needed for comparison.")
        else:
            col1, col2 = st.columns(2)
            acc1_comp = col1.selectbox("Select Account 1:", unique_accounts_all, index=0, key="acc_sel_1_comp_v3")
            idx2 = 1 if len(unique_accounts_all) > 1 else 0
            acc2_comp = col2.selectbox("Select Account 2:", unique_accounts_all, index=idx2, key="acc_sel_2_comp_v3")

            if acc1_comp == acc2_comp: st.warning("Please select two different accounts.")
            else:
                st.subheader(f"Equity Curve Comparison: {acc1_comp} vs. {acc2_comp}")
                df_acc1_raw = base_df[base_df[account_col_actual] == acc1_comp]
                df_acc2_raw = base_df[base_df[account_col_actual] == acc2_comp]
                combined_equity_comp_df = pd.DataFrame()

                for df_raw, acc_name in [(df_acc1_raw, acc1_comp), (df_acc2_raw, acc2_comp)]:
                    if df_raw.empty: continue
                    df_cleaned = _clean_data_for_analysis(
                        df_raw, 
                        date_col=date_col_actual, 
                        pnl_col=pnl_col_actual, 
                        required_cols_to_check_na=[pnl_col_actual] 
                    )
                    if not df_cleaned.empty:
                        df_cleaned['cumulative_pnl'] = df_cleaned[pnl_col_actual].cumsum()
                        temp_df = df_cleaned[[date_col_actual, 'cumulative_pnl']].rename(columns={'cumulative_pnl': f'Equity_{acc_name}'})
                        if combined_equity_comp_df.empty: combined_equity_comp_df = temp_df
                        else: combined_equity_comp_df = pd.merge(combined_equity_comp_df, temp_df, on=date_col_actual, how='outer')
                
                if combined_equity_comp_df.empty or not any(f'Equity_{acc}' in combined_equity_comp_df.columns for acc in [acc1_comp, acc2_comp]):
                    display_custom_message(f"One or both accounts lack valid P&L data for comparison.", "warning")
                else:
                    combined_equity_comp_df.sort_values(by=date_col_actual, inplace=True)
                    combined_equity_comp_df = combined_equity_comp_df.ffill().fillna(0) 
                    fig_comp = go.Figure()
                    if f'Equity_{acc1_comp}' in combined_equity_comp_df:
                        fig_comp.add_trace(go.Scatter(x=combined_equity_comp_df[date_col_actual], y=combined_equity_comp_df[f'Equity_{acc1_comp}'], name=f"{acc1_comp} Equity"))
                    if f'Equity_{acc2_comp}' in combined_equity_comp_df:
                        fig_comp.add_trace(go.Scatter(x=combined_equity_comp_df[date_col_actual], y=combined_equity_comp_df[f'Equity_{acc2_comp}'], name=f"{acc2_comp} Equity"))
                    fig_comp.update_layout(title=f"Equity Comparison: {acc1_comp} vs. {acc2_comp}", xaxis_title="Date", yaxis_title="Cumulative PnL", hovermode="x unified")
                    st.plotly_chart(_apply_custom_theme(fig_comp, plot_theme), use_container_width=True)
                    if not combined_equity_comp_df.empty:
                        with st.expander("View Combined Equity Comparison Data"): st.dataframe(combined_equity_comp_df)

if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Portfolio Analysis", initial_sidebar_state="expanded")
    if 'APP_TITLE' not in globals(): APP_TITLE = "PortfolioApp_Standalone" 
    
    if 'EXPECTED_COLUMNS' not in globals():
        EXPECTED_COLUMNS = {
            'account_str': 'Account', 
            'pnl': 'PnL',             
            'date': 'Date',           
            'strategy': 'Strategy'    
        }
    if 'RISK_FREE_RATE' not in globals():
        RISK_FREE_RATE = 0.01
    if 'KPI_CONFIG' not in globals(): 
        KPI_CONFIG = {} # This should ideally be populated with your KPI definitions

    if 'app_initialized' not in st.session_state: 
        # Mock data for standalone testing:
        # sample_data = {
        #     EXPECTED_COLUMNS['date']: pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02', '2023-01-03', '2023-01-03']),
        #     EXPECTED_COLUMNS['pnl']: [10, -5, 20, 10, -15, 25],
        #     EXPECTED_COLUMNS['strategy']: ['StratA', 'StratB', 'StratA', 'StratB', 'StratA', 'StratB'],
        #     EXPECTED_COLUMNS['account_str']: ['Acc1', 'Acc1', 'Acc2', 'Acc2', 'Acc1', 'Acc2']
        # }
        # st.session_state.processed_data = pd.DataFrame(sample_data)
        # st.session_state.initial_capital = 100000
        # st.session_state.risk_free_rate = RISK_FREE_RATE # Use the defined or mocked one
        # st.session_state.current_theme = 'dark'
        # st.session_state.user_column_mapping = { # Mock a basic mapping if your app relies on it early
        #     'date': EXPECTED_COLUMNS['date'],
        #     'pnl': EXPECTED_COLUMNS['pnl'],
        #     'strategy': EXPECTED_COLUMNS['strategy'],
        #     'account_str': EXPECTED_COLUMNS['account_str']
        # }
        st.warning("Running page directly. Full app functionality may be limited. Ensure `config.py` variables are accessible or mocked, and `st.session_state.processed_data` is populated for testing.")
    
    show_portfolio_analysis_page()
