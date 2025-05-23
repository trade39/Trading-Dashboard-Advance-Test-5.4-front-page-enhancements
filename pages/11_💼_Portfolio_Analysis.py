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
    Converts date column to datetime, specified PnL and numeric columns to numeric,
    and specified string columns to string. Drops rows with NaNs in essential columns.
    Sorts by date if specified.
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
        df_cleaned[date_col] = pd.NaT # Set to NaT on error to allow later dropna

    if pnl_col and pnl_col in df_cleaned.columns:
        try:
            df_cleaned[pnl_col] = pd.to_numeric(df_cleaned[pnl_col], errors='coerce')
        except Exception as e:
            logger.error(f"Error converting PnL column '{pnl_col}' to numeric: {e}", exc_info=True)
            df_cleaned[pnl_col] = np.nan # Set to NaN on error

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

    # Define columns that are essential for any row to be valid
    cols_for_nan_check = [date_col]
    if pnl_col and pnl_col in df_cleaned.columns: cols_for_nan_check.append(pnl_col)
    if strategy_col and strategy_col in df_cleaned.columns: cols_for_nan_check.append(strategy_col)
    if account_col and account_col in df_cleaned.columns: cols_for_nan_check.append(account_col)
    if required_cols_to_check_na:
        for rc in required_cols_to_check_na:
            if rc in df_cleaned.columns and rc not in cols_for_nan_check:
                cols_for_nan_check.append(rc)
    
    # Filter out columns that don't actually exist in the dataframe before attempting dropna
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
    """ Helper to calculate absolute and percentage drawdown series from a cumulative P&L series. """
    if cumulative_pnl_series.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    high_water_mark = cumulative_pnl_series.cummax()
    drawdown_abs_series = high_water_mark - cumulative_pnl_series
    # Handle cases where high_water_mark is zero to avoid division by zero
    drawdown_pct_series = pd.Series(np.where(high_water_mark > 1e-9, (drawdown_abs_series / high_water_mark) * 100.0,
                                             np.where(drawdown_abs_series > 1e-9, 100.0, 0.0)), # If HWM is 0 but drawdown exists, it's 100%
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
    """
    Calculates core performance metrics for a given DataFrame of trades or daily P&L.
    The input DataFrame is expected to have a date column and a P&L column.
    """
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

    # Use the analysis service to get KPIs
    kpis = general_analysis_service.get_core_kpis(df_copy, risk_free_rate, initial_capital=initial_capital)
    
    if kpis and 'error' not in kpis:
        # Return a selection of relevant KPIs
        return {
            "Total PnL": kpis.get("total_pnl", 0.0),
            "Total Trades": kpis.get("total_trades", 0), # This might be more relevant if df_input is raw trades
            "Win Rate %": kpis.get("win_rate", 0.0), # Same as above
            "Avg Trade PnL": kpis.get("avg_trade_pnl", 0.0), # Same as above
            "Sharpe Ratio": kpis.get("sharpe_ratio", 0.0),
            "Sortino Ratio": kpis.get("sortino_ratio", 0.0),
            "Calmar Ratio": kpis.get("calmar_ratio", 0.0),
            "Max Drawdown %": kpis.get("max_drawdown_pct", 0.0),
            "Max Drawdown Abs": kpis.get("max_drawdown_abs", 0.0),
            "Avg Daily PnL": kpis.get("avg_daily_pnl", 0.0) # Relevant if df_input is daily PnL
        }
    error_msg = kpis.get('error', 'Unknown error') if kpis else 'KPI calculation failed'
    logger.warning(f"KPI calculation failed in calculate_metrics_for_df: {error_msg}")
    return {"error": error_msg}

@st.cache_data
def _run_portfolio_optimization_logic(
    portfolio_df_data_tuple: Tuple[List[tuple], List[str]], # Tuple of (data, columns) for portfolio_df
    strategy_col_actual: str, date_col_actual: str, pnl_col_actual: str,
    selected_strategies_for_opt_tuple: Tuple[str, ...], # Tuple of selected strategies
    lookback_days_opt: int, global_initial_capital: float,
    optimization_objective_key: str, risk_free_rate: float,
    target_return_val: Optional[float], num_frontier_points: int,
    use_ledoit_wolf: bool, asset_bounds_list_of_tuples: Optional[List[Tuple[float, float]]]
) -> Dict[str, Any]:
    """ 
    Encapsulates the data preparation and optimization call.
    Accepts portfolio_df as a tuple of its data and columns to be cache-friendly.
    Selected strategies are also passed as a tuple.
    """
    portfolio_df_data, portfolio_df_columns = portfolio_df_data_tuple
    portfolio_df = pd.DataFrame(data=portfolio_df_data, columns=portfolio_df_columns)
    selected_strategies_for_opt = list(selected_strategies_for_opt_tuple) # Convert back to list

    # Filter for selected strategies
    opt_df_filtered_strategies = portfolio_df[portfolio_df[strategy_col_actual].isin(selected_strategies_for_opt)].copy()
    
    # Clean this filtered data
    opt_df_filtered_strategies = _clean_data_for_analysis(
        opt_df_filtered_strategies, date_col=date_col_actual, pnl_col=pnl_col_actual,
        strategy_col=strategy_col_actual, required_cols_to_check_na=[pnl_col_actual, strategy_col_actual],
        sort_by_date=True
    )

    if opt_df_filtered_strategies.empty:
        return {"error": "No data for selected strategies after cleaning for optimization."}

    # Apply lookback period
    latest_date_in_data = opt_df_filtered_strategies[date_col_actual].max()
    start_date_lookback = latest_date_in_data - pd.Timedelta(days=lookback_days_opt - 1) # -1 because days includes the start day
    opt_df_lookback = opt_df_filtered_strategies[opt_df_filtered_strategies[date_col_actual] >= start_date_lookback]

    if opt_df_lookback.empty:
        return {"error": "No data within the specified lookback period."}

    # Pivot P&L data: Date by Strategy, values are sum of PnL
    try:
        daily_pnl_pivot = opt_df_lookback.groupby(
            [opt_df_lookback[date_col_actual].dt.normalize(), strategy_col_actual]
        )[pnl_col_actual].sum().unstack(fill_value=0)
        # Ensure all selected strategies are columns, even if they had no P&L in the lookback
        daily_pnl_pivot = daily_pnl_pivot.reindex(columns=selected_strategies_for_opt, fill_value=0.0)
    except (KeyError, ValueError, TypeError, AttributeError) as e:
        logger.error(f"Error during P&L pivot for optimization: {e}", exc_info=True)
        return {"error": f"Failed to pivot P&L data for optimization: {e}"}

    if global_initial_capital <= 0:
        return {"error": "Initial capital must be a positive value for return calculation."}
    
    # Calculate daily returns based on global initial capital
    # This assumes P&L is absolute and can be divided by a common capital base.
    # For more accuracy, per-strategy capital or a returns-based input would be better.
    daily_returns_for_opt = (daily_pnl_pivot / global_initial_capital).fillna(0)

    # Check for sufficient historical data points
    min_hist_points_needed = 20 # Common minimum for covariance estimation
    if optimization_objective_key == "risk_parity" and len(selected_strategies_for_opt) <= 1:
        min_hist_points_needed = 2 # Risk parity with 1 asset is trivial but allow
    if daily_returns_for_opt.shape[0] < min_hist_points_needed:
        return {"error": f"Insufficient historical data: Need at least {min_hist_points_needed} data points for optimization, found {daily_returns_for_opt.shape[0]}."}

    try:
        # Call the portfolio analysis service for optimization
        optimization_results = portfolio_specific_service.prepare_and_run_optimization(
            daily_returns_df=daily_returns_for_opt,
            objective=optimization_objective_key,
            risk_free_rate=risk_free_rate,
            target_return_level=target_return_val,
            trading_days=252, # Standard assumption
            num_frontier_points=num_frontier_points,
            use_ledoit_wolf=use_ledoit_wolf,
            asset_bounds=asset_bounds_list_of_tuples
        )
        return optimization_results
    except Exception as e:
        logger.error(f"Error calling portfolio optimization service: {e}", exc_info=True)
        return {"error": f"Optimization service execution failed: {e}"}

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

    # Get column names from EXPECTED_COLUMNS configuration
    account_col_actual = EXPECTED_COLUMNS.get('account_str')
    pnl_col_actual = EXPECTED_COLUMNS.get('pnl')
    date_col_actual = EXPECTED_COLUMNS.get('date')
    strategy_col_actual = EXPECTED_COLUMNS.get('strategy')

    # Validate that all essential column names are configured
    if not all([account_col_actual, pnl_col_actual, date_col_actual, strategy_col_actual]):
        missing_configs = [col_type for col_type, col_val in zip(
            ['account', 'pnl', 'date', 'strategy'],
            [account_col_actual, pnl_col_actual, date_col_actual, strategy_col_actual]) if not col_val]
        msg = f"Essential column configurations missing ({', '.join(missing_configs)}). Analysis cannot proceed."
        display_custom_message(msg, "error"); logger.error(f"Portfolio page: {msg}"); return

    # Validate that these columns exist in the DataFrame
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
            default=unique_accounts_all, key="portfolio_view_account_multiselect_v3" # Incremented key
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
        string_cols_to_convert=[strategy_col_actual, account_col_actual] # Ensure these are strings
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
        portfolio_daily_trades_df = pd.DataFrame() # Initialize
        try:
            # Group by normalized date and sum P&L for the selected portfolio
            portfolio_daily_pnl = portfolio_df.groupby(portfolio_df[date_col_actual].dt.normalize())[pnl_col_actual].sum()
        except (AttributeError, KeyError, TypeError) as e: # Catch potential errors if date_col_actual is not datetime or pnl_col_actual is missing
            logger.error(f"Error grouping by date for daily P&L: {e}", exc_info=True)
            display_custom_message(f"Error processing daily P&L: {e}", "error")
            portfolio_daily_pnl = pd.Series(dtype=float) # Empty series on error

        if portfolio_daily_pnl.empty:
            display_custom_message("No P&L data after daily aggregation for selected portfolio.", "warning")
        else:
            portfolio_daily_trades_df = pd.DataFrame({date_col_actual: portfolio_daily_pnl.index, pnl_col_actual: portfolio_daily_pnl.values})
            portfolio_daily_trades_df['cumulative_pnl'] = portfolio_daily_trades_df[pnl_col_actual].cumsum()
            # Assuming daily P&L, so 'win' is if daily P&L > 0
            portfolio_daily_trades_df['win'] = portfolio_daily_trades_df[pnl_col_actual] > 0 
            
            # Calculate drawdown series
            if 'cumulative_pnl' in portfolio_daily_trades_df.columns and not portfolio_daily_trades_df['cumulative_pnl'].empty:
                drawdown_abs_series, drawdown_pct_series = _calculate_drawdown_series_for_aggregated_df(portfolio_daily_trades_df['cumulative_pnl'])
                portfolio_daily_trades_df['drawdown_abs'] = drawdown_abs_series
                portfolio_daily_trades_df['drawdown_pct'] = drawdown_pct_series
            else: # Ensure columns exist even if empty
                portfolio_daily_trades_df['drawdown_abs'] = pd.Series(dtype=float)
                portfolio_daily_trades_df['drawdown_pct'] = pd.Series(dtype=float)

            with st.spinner("Calculating selected portfolio KPIs..."):
                # Pass the daily aggregated P&L df for KPI calculation
                portfolio_kpis = general_analysis_service.get_core_kpis(portfolio_daily_trades_df, risk_free_rate, initial_capital=global_initial_capital)
            
            if portfolio_kpis and 'error' not in portfolio_kpis:
                # Define which KPIs to show for the portfolio overview
                portfolio_kpi_keys = ["total_pnl", "sharpe_ratio", "sortino_ratio", "calmar_ratio", "max_drawdown_abs", "max_drawdown_pct", "avg_daily_pnl", "pnl_skewness", "pnl_kurtosis"]
                kpis_to_display_portfolio = {key: portfolio_kpis[key] for key in portfolio_kpi_keys if key in portfolio_kpis}
                if kpis_to_display_portfolio:
                    KPIClusterDisplay(kpis_to_display_portfolio, KPI_CONFIG, portfolio_kpi_keys, cols_per_row=3).render()
                else: display_custom_message("Could not retrieve relevant KPIs for selected portfolio.", "warning")
            else: display_custom_message(f"Error calculating KPIs: {portfolio_kpis.get('error', 'Unknown error') if portfolio_kpis else 'KPI calculation failed'}", "error")
            
            st.subheader("Combined Equity Curve & Drawdown")
            portfolio_equity_fig = plot_equity_curve_and_drawdown(
                portfolio_daily_trades_df, date_col_actual, 'cumulative_pnl', 'drawdown_pct', theme=plot_theme
            )
            if portfolio_equity_fig: st.plotly_chart(portfolio_equity_fig, use_container_width=True)
            else: display_custom_message("Could not generate equity curve for the selected portfolio.", "warning")

            if not portfolio_daily_trades_df.empty:
                with st.expander("View Underlying Equity Curve Data (Selected Portfolio Daily Aggregated)"):
                    st.dataframe(portfolio_daily_trades_df)

    with tab_connections:
        st.header(f"Inter-Connections (Selected Portfolio: {', '.join(selected_accounts_for_portfolio)})")
        
        # Period selection for correlation
        correlation_period_options = {"Full Period": None, "Last 30 days": 30, "Last 90 days": 90, "Last 180 days": 180}
        selected_period_label = st.selectbox(
            "Select Correlation Period:",
            options=list(correlation_period_options.keys()),
            index=0, # Default to "Full Period"
            key="correlation_period_selector_v1"
        )
        selected_period_days = correlation_period_options[selected_period_label]

        # Filter portfolio_df based on selected_period_days
        df_for_correlation = portfolio_df.copy()
        if selected_period_days is not None and date_col_actual in df_for_correlation.columns:
            latest_date_corr = df_for_correlation[date_col_actual].max()
            start_date_corr = latest_date_corr - pd.Timedelta(days=selected_period_days -1)
            df_for_correlation = df_for_correlation[df_for_correlation[date_col_actual] >= start_date_corr]
        
        if df_for_correlation.empty:
            display_custom_message(f"No data available for the selected period: {selected_period_label}", "warning")
        else:
            st.subheader("🔀 Inter-Strategy P&L Correlation")
            if strategy_col_actual not in df_for_correlation.columns:
                display_custom_message(f"Strategy column '{strategy_col_actual}' not found in the data for the selected period.", "warning")
            else:
                unique_strategies_sel_portfolio = df_for_correlation[strategy_col_actual].dropna().unique()
                if len(unique_strategies_sel_portfolio) < 2:
                    st.info("At least two distinct strategies needed in the selected period for inter-strategy correlations.")
                else:
                    df_strat_corr_prep = df_for_correlation[[date_col_actual, strategy_col_actual, pnl_col_actual]].copy()
                    # Sort for consistent pivoting
                    df_strat_corr_prep = df_strat_corr_prep.sort_values(by=[date_col_actual, strategy_col_actual]).reset_index(drop=True)
                    with st.spinner(f"Calculating inter-strategy P&L correlations ({selected_period_label})..."):
                        try:
                            correlation_results_strat = portfolio_specific_service.get_portfolio_inter_strategy_correlation(
                                df_strat_corr_prep, strategy_col_actual, pnl_col_actual, date_col_actual)
                        except Exception as e:
                            logger.error(f"Inter-strategy correlation service error: {e}", exc_info=True)
                            correlation_results_strat = {"error": f"Service failed: {e}"}

                    if correlation_results_strat and 'error' not in correlation_results_strat:
                        matrix_df_strat_corr = correlation_results_strat.get('correlation_matrix')
                        if matrix_df_strat_corr is not None and not matrix_df_strat_corr.empty and matrix_df_strat_corr.shape[0] > 1:
                            fig_strat_corr = go.Figure(data=go.Heatmap(
                                z=matrix_df_strat_corr.values, x=matrix_df_strat_corr.columns, y=matrix_df_strat_corr.index,
                                colorscale='RdBu', zmin=-1, zmax=1, text=matrix_df_strat_corr.round(2).astype(str),
                                texttemplate="%{text}", hoverongaps=False))
                            fig_strat_corr.update_layout(title=f"Inter-Strategy Daily P&L Correlation ({selected_period_label})")
                            st.plotly_chart(_apply_custom_theme(fig_strat_corr, plot_theme), use_container_width=True)
                            with st.expander("View Inter-Strategy Correlation Matrix"): st.dataframe(matrix_df_strat_corr)
                        else: display_custom_message(f"Not enough data for inter-strategy correlation matrix for {selected_period_label}.", "info")
                    elif correlation_results_strat: display_custom_message(f"Inter-strategy correlation error: {correlation_results_strat.get('error')}", "error")
                    else: display_custom_message("Inter-strategy correlation analysis failed.", "error")

            st.subheader("🤝 Inter-Account P&L Correlation")
            if len(selected_accounts_for_portfolio) < 2: # This check is on original selection, not period-filtered
                st.info("At least two accounts needed in the initial selection for inter-account correlation.")
            else:
                # Use the same period-filtered df_for_correlation
                df_acc_corr_prep = df_for_correlation[[date_col_actual, account_col_actual, pnl_col_actual]].copy()
                # Filter for accounts that are actually in the selected_accounts_for_portfolio (in case period filter removed some)
                df_acc_corr_prep = df_acc_corr_prep[df_acc_corr_prep[account_col_actual].isin(selected_accounts_for_portfolio)]

                if len(df_acc_corr_prep[account_col_actual].unique()) < 2:
                    st.info(f"Fewer than two accounts have data in the selected period ({selected_period_label}) for correlation.")
                else:
                    df_acc_corr_prep = df_acc_corr_prep.sort_values(by=[date_col_actual, account_col_actual]).reset_index(drop=True)
                    with st.spinner(f"Calculating inter-account P&L correlations ({selected_period_label})..."):
                        try:
                            correlation_results_acc = portfolio_specific_service.get_portfolio_inter_account_correlation(
                                df_acc_corr_prep, account_col_actual, pnl_col_actual, date_col_actual)
                        except Exception as e:
                            logger.error(f"Inter-account correlation service error: {e}", exc_info=True)
                            correlation_results_acc = {"error": f"Service failed: {e}"}

                    if correlation_results_acc and 'error' not in correlation_results_acc:
                        matrix_df_acc_corr = correlation_results_acc.get('correlation_matrix')
                        if matrix_df_acc_corr is not None and not matrix_df_acc_corr.empty and matrix_df_acc_corr.shape[0] > 1:
                            fig_acc_corr = go.Figure(data=go.Heatmap(
                                z=matrix_df_acc_corr.values, x=matrix_df_acc_corr.columns, y=matrix_df_acc_corr.index,
                                colorscale='RdBu', zmin=-1, zmax=1, text=matrix_df_acc_corr.round(2).astype(str),
                                texttemplate="%{text}", hoverongaps=False))
                            fig_acc_corr.update_layout(title=f"Inter-Account Daily P&L Correlation ({selected_period_label})")
                            st.plotly_chart(_apply_custom_theme(fig_acc_corr, plot_theme), use_container_width=True)
                            with st.expander("View Inter-Account Correlation Matrix"): st.dataframe(matrix_df_acc_corr)
                        else: display_custom_message(f"Not enough data for inter-account correlation matrix for {selected_period_label}.", "info")
                    elif correlation_results_acc: display_custom_message(f"Inter-account correlation error: {correlation_results_acc.get('error')}", "error")
                    else: display_custom_message("Inter-account correlation analysis failed.", "error")

    with tab_breakdown:
        st.header(f"Account Performance Breakdown (Portfolio: {', '.join(selected_accounts_for_portfolio)})")
        account_metrics_data = []
        for acc_name_loop in selected_accounts_for_portfolio:
            # Use original base_df to get all trades for an account, not the portfolio_df which might be filtered/cleaned differently
            acc_df_original_trades = base_df[base_df[account_col_actual] == acc_name_loop].copy()
            if not acc_df_original_trades.empty:
                # Metrics are calculated on the raw trades for that account
                metrics = calculate_metrics_for_df(acc_df_original_trades, pnl_col_actual, date_col_actual, risk_free_rate, global_initial_capital)
                account_metrics_data.append({"Account": acc_name_loop, **metrics})
            else: logger.info(f"No original trade data for account {acc_name_loop} in breakdown.")

        if account_metrics_data:
            summary_table_df = pd.DataFrame(account_metrics_data)
            # For P&L contribution chart, ensure 'Total PnL' is numeric
            if "Total PnL" in summary_table_df.columns:
                summary_table_df["Total PnL Numeric"] = pd.to_numeric(summary_table_df["Total PnL"], errors='coerce') 
                # Filter out NaN or zero P&L for a cleaner pie chart
                pnl_for_chart_df = summary_table_df[
                    summary_table_df["Total PnL Numeric"].notna() & (summary_table_df["Total PnL Numeric"] != 0)
                ][["Account", "Total PnL Numeric"]].copy()

                if not pnl_for_chart_df.empty:
                    fig_pnl_contrib = px.pie(pnl_for_chart_df, names='Account', values='Total PnL Numeric', title='P&L Contribution by Account', hole=0.3)
                    fig_pnl_contrib.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(_apply_custom_theme(fig_pnl_contrib, plot_theme), use_container_width=True)
                    with st.expander("View P&L Contribution Data (Numeric)"): st.dataframe(pnl_for_chart_df.rename(columns={"Total PnL Numeric": "Total PnL"}))
                else: st.info("No P&L contribution data to display (all P&L values are zero, NaN, or non-numeric).")
            else: st.warning("Total PnL column not found for P&L contribution chart.")

            # Select and format columns for the summary display table
            display_cols_summary = ["Account", "Total PnL", "Total Trades", "Win Rate %", "Avg Trade PnL", "Max Drawdown %", "Sharpe Ratio"]
            valid_display_cols = [col for col in display_cols_summary if col in summary_table_df.columns]
            summary_table_df_display = summary_table_df[valid_display_cols].copy()
            
            # Apply formatting
            for col, item_type in [("Total PnL", "currency"), ("Avg Trade PnL", "currency"), 
                                   ("Win Rate %", "percentage"), ("Max Drawdown %", "percentage"),
                                   ("Sharpe Ratio", "float")]:
                if col in summary_table_df_display.columns:
                    # Ensure numeric before formatting
                    summary_table_df_display[col] = pd.to_numeric(summary_table_df_display[col], errors='coerce')
                    if item_type == "currency": summary_table_df_display[col] = summary_table_df_display[col].apply(lambda x: format_currency(x) if pd.notna(x) else "N/A")
                    elif item_type == "percentage": summary_table_df_display[col] = summary_table_df_display[col].apply(lambda x: format_percentage(x/100.0) if pd.notna(x) else "N/A") # Assuming % is 0-100
                    elif item_type == "float": summary_table_df_display[col] = summary_table_df_display[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
            
            st.dataframe(summary_table_df_display.set_index("Account"), use_container_width=True)
            if not summary_table_df.empty:
                with st.expander("View Raw Account Performance Data (Pre-formatting)"):
                    st.dataframe(summary_table_df.drop(columns=["Total PnL Numeric"], errors='ignore')) # Drop the temp numeric column
        else: display_custom_message("Could not calculate performance metrics for individual accounts.", "warning")

    with tab_optimization:
        st.header("Portfolio Optimization")
        # Initialize form state variables (will be updated by form inputs)
        selected_strategies_for_opt_form = []
        optimization_objective_form_key = "" 
        optimization_objective_display_form = "" 
        use_ledoit_wolf_covariance_form = True
        target_return_input_form = None
        lookback_days_opt_form = 252
        num_frontier_points_input_form = 25
        # asset_bounds_input_form will be a list of tuples [(min, max), ...]
        # current_weights_input_for_turnover_form will be a dict {strategy: weight}

        with st.expander("⚙️ Configure Portfolio Optimization", expanded=True):
            if strategy_col_actual not in portfolio_df.columns:
                st.warning(f"Strategy column ('{strategy_col_actual}') not found in the selected portfolio data. Cannot perform optimization.")
            else:
                optimizable_strategies = sorted(portfolio_df[strategy_col_actual].dropna().unique())
                if not optimizable_strategies:
                     st.info("No strategies available in the selected portfolio data for optimization.")
                else:
                    # Using a unique key for the form to ensure state is managed correctly
                    with st.form("portfolio_optimization_form_v7"): # Incremented key
                        st.markdown("Select strategies, objective, and constraints for optimization.")
                        
                        # Strategy Selection
                        selected_strategies_for_opt_form = st.multiselect(
                            "Select Strategies for Optimization:", options=optimizable_strategies,
                            default=optimizable_strategies[:min(len(optimizable_strategies), 5)], # Default to first 5 or all if fewer
                            key="opt_strat_sel_v7"
                        )
                        
                        # Optimization Objective
                        optimization_objective_options_map = {
                            "Maximize Sharpe Ratio": "maximize_sharpe_ratio",
                            "Minimize Volatility": "minimize_volatility",
                            "Risk Parity": "risk_parity"
                            # Add more objectives here if PortfolioAnalysisService supports them
                        }
                        optimization_objective_display_form = st.selectbox(
                            "Optimization Objective:", options=list(optimization_objective_options_map.keys()),
                            index=0, key="opt_obj_v7"
                        )
                        optimization_objective_form_key = optimization_objective_options_map[optimization_objective_display_form]

                        # Covariance & Lookback
                        col_opt_params1, col_opt_params2 = st.columns(2)
                        with col_opt_params1:
                            use_ledoit_wolf_covariance_form = st.checkbox("Use Ledoit-Wolf Covariance", True, key="opt_lw_v7")
                        with col_opt_params2:
                            max_lookback = max(20, len(portfolio_df[date_col_actual].unique())) if date_col_actual in portfolio_df and not portfolio_df.empty else 20
                            lookback_days_opt_form = st.number_input("Lookback Period (days):", 20, max_lookback, min(252, max_lookback), 10, key="opt_lb_v7")

                        # Target Return (conditional) & Frontier Points
                        if optimization_objective_form_key == "minimize_volatility":
                            target_return_input_form = st.number_input(
                                "Target Annualized Return (e.g., 0.10 for 10%):", 
                                min_value=-1.0, max_value=2.0, value=0.10, step=0.01, format="%.2f", key="opt_tr_v7"
                            )
                        if optimization_objective_form_key in ["maximize_sharpe_ratio", "minimize_volatility"]:
                             num_frontier_points_input_form = st.number_input("Number of Frontier Points:", 10, 100, 25, 5, key="opt_fp_v7")
                        
                        st.markdown("---")
                        st.markdown("##### Per-Strategy Weight Constraints")

                        # Preset Constraints
                        preset_options = {
                            "Default (0-100%)": {"min": 0.0, "max": 100.0, "apply_all": True},
                            "Long Only (Max 30%)": {"min": 0.0, "max": 30.0, "apply_all": True},
                            "Diversified (Min 5%, Max 50%)": {"min": 5.0, "max": 50.0, "apply_all": True},
                            "Custom": None # Allows manual override
                        }
                        selected_preset_label = st.selectbox(
                            "Constraint Presets:", options=list(preset_options.keys()), index=0, key="opt_preset_v7"
                        )
                        selected_preset = preset_options[selected_preset_label]

                        # Dynamic Current Weights Calculation (for turnover)
                        current_weights_for_turnover_calc = {}
                        if selected_strategies_for_opt_form and not portfolio_df.empty:
                            # Filter portfolio_df for selected strategies and lookback period
                            temp_df_for_weights = portfolio_df[portfolio_df[strategy_col_actual].isin(selected_strategies_for_opt_form)].copy()
                            if not temp_df_for_weights.empty and date_col_actual in temp_df_for_weights:
                                latest_date_cw = temp_df_for_weights[date_col_actual].max()
                                start_date_cw = latest_date_cw - pd.Timedelta(days=lookback_days_opt_form -1)
                                temp_df_for_weights = temp_df_for_weights[temp_df_for_weights[date_col_actual] >= start_date_cw]

                            if not temp_df_for_weights.empty:
                                # Sum P&L per strategy over the period
                                pnl_sum_per_strategy = temp_df_for_weights.groupby(strategy_col_actual)[pnl_col_actual].sum()
                                # Only consider positive P&L sums for weight calculation
                                positive_pnl_sum = pnl_sum_per_strategy[pnl_sum_per_strategy > 0]
                                
                                if not positive_pnl_sum.empty:
                                    total_positive_pnl = positive_pnl_sum.sum()
                                    if total_positive_pnl > 0: # Avoid division by zero
                                        current_weights_for_turnover_calc = (positive_pnl_sum / total_positive_pnl).to_dict()
                                # Fill missing strategies (if any selected had no positive P&L) with 0 weight
                                for strat in selected_strategies_for_opt_form:
                                    if strat not in current_weights_for_turnover_calc:
                                        current_weights_for_turnover_calc[strat] = 0.0
                                else: # If all P&L sums are <=0, use equal weighting as fallback
                                    num_sel = len(selected_strategies_for_opt_form)
                                    equal_w = 1.0 / num_sel if num_sel > 0 else 0.0
                                    for strat in selected_strategies_for_opt_form:
                                        current_weights_for_turnover_calc[strat] = equal_w
                            else: # Fallback to equal weighting if no data for P&L sum
                                num_sel = len(selected_strategies_for_opt_form)
                                equal_w = 1.0 / num_sel if num_sel > 0 else 0.0
                                for strat in selected_strategies_for_opt_form:
                                    current_weights_for_turnover_calc[strat] = equal_w
                        
                        # Display derived current weights (read-only)
                        if selected_strategies_for_opt_form and current_weights_for_turnover_calc:
                            st.markdown("###### Derived Current Weights (for Turnover Calculation)")
                            cw_df = pd.DataFrame.from_dict(current_weights_for_turnover_calc, orient='index', columns=['Weight %'])
                            cw_df['Weight %'] = cw_df['Weight %'] * 100
                            st.dataframe(cw_df.style.format("{:.2f}%"), use_container_width=True)
                            sum_derived_cw = sum(current_weights_for_turnover_calc.values())
                            if not (0.99 < sum_derived_cw < 1.01) and sum_derived_cw != 0.0:
                                 st.caption(f"Note: Sum of derived current weights is {sum_derived_cw*100:.1f}%. This might occur if P&L data is sparse or negative.")


                        # Per-strategy min/max weight inputs
                        asset_bounds_input_form = [] # List of (min_val, max_val) tuples
                        min_weight_inputs = {}
                        max_weight_inputs = {}

                        if selected_strategies_for_opt_form:
                            st.markdown("###### Individual Strategy Constraints (Min/Max %)")
                            for i, strat_name in enumerate(selected_strategies_for_opt_form):
                                default_min, default_max = 0.0, 100.0 # Overall default
                                if selected_preset and selected_preset.get("apply_all"):
                                    default_min = selected_preset["min"]
                                    default_max = selected_preset["max"]
                                
                                cols_constraints = st.columns([2,1,1]) # Strategy Name | Min Weight | Max Weight
                                with cols_constraints[0]:
                                    st.markdown(f"**{strat_name}**")
                                with cols_constraints[1]:
                                    min_w = st.number_input(f"Min W %", 0.0, 100.0, default_min, 0.1, key=f"min_w_{strat_name}_v7", label_visibility="collapsed")
                                with cols_constraints[2]:
                                    max_w = st.number_input(f"Max W %", 0.0, 100.0, default_max, 0.1, key=f"max_w_{strat_name}_v7", label_visibility="collapsed")
                                
                                # Immediate validation for min_w <= max_w for this strategy
                                if min_w > max_w:
                                    st.warning(f"For {strat_name}: Min weight ({min_w}%) cannot exceed Max weight ({max_w}%). Adjusting Max to {min_w}%.", icon="⚠️")
                                    max_w = min_w # Auto-correct or just warn

                                asset_bounds_input_form.append((min_w / 100.0, max_w / 100.0))
                                min_weight_inputs[strat_name] = min_w / 100.0
                        else:
                            st.caption("Select strategies to configure their weight constraints.")

                        # Visual Feedback for Sum of Min Weights (before submission)
                        if asset_bounds_input_form:
                            sum_min_weights_pct = sum(b[0] * 100 for b in asset_bounds_input_form)
                            if sum_min_weights_pct > 100.0 + 1e-6: # Add tolerance for float precision
                                st.warning(f"Sum of Minimum Weight constraints ({sum_min_weights_pct:.1f}%) exceeds 100%. Optimization might be infeasible.", icon="🚨")
                        
                        submit_optimization_button = st.form_submit_button("Optimize Portfolio")

        # --- End of Form ---

        if submit_optimization_button and selected_strategies_for_opt_form:
            # Final validation before running optimization
            min_strats_required = 1 if optimization_objective_form_key == "risk_parity" else 2 # Risk parity can run on 1, others need >=2 for covariance
            constraint_error = False
            if len(selected_strategies_for_opt_form) < min_strats_required:
                display_custom_message(f"Please select at least {min_strats_required} strategies for the '{optimization_objective_display_form}' objective.", "warning")
                constraint_error = True
            
            sum_min_weights_final = sum(b[0] for b in asset_bounds_input_form) if asset_bounds_input_form else 0.0
            if sum_min_weights_final > 1.0 + 1e-6 : # Check again on submit
                display_custom_message(f"Error: Sum of minimum weight constraints ({sum_min_weights_final*100:.1f}%) exceeds 100%. Please adjust constraints.", "error")
                constraint_error = True
            
            for i, strat_name in enumerate(selected_strategies_for_opt_form):
                 min_b, max_b = asset_bounds_input_form[i]
                 if min_b > max_b:
                     display_custom_message(f"Error for {strat_name}: Min weight ({min_b*100:.1f}%) cannot be greater than Max weight ({max_b*100:.1f}%).", "error")
                     constraint_error = True
                     break
            
            if not constraint_error:
                with st.spinner("Optimizing portfolio... This may take a moment."):
                    # Pass portfolio_df as tuple (data, columns) for caching
                    portfolio_df_tuple_for_opt = (portfolio_df.to_records(index=False).tolist(), portfolio_df.columns.tolist())
                    
                    opt_results = _run_portfolio_optimization_logic(
                        portfolio_df_data_tuple=portfolio_df_tuple_for_opt,
                        strategy_col_actual=strategy_col_actual,
                        date_col_actual=date_col_actual,
                        pnl_col_actual=pnl_col_actual,
                        selected_strategies_for_opt_tuple=tuple(selected_strategies_for_opt_form), # Pass as tuple
                        lookback_days_opt=lookback_days_opt_form,
                        global_initial_capital=global_initial_capital,
                        optimization_objective_key=optimization_objective_form_key,
                        risk_free_rate=risk_free_rate,
                        target_return_val=target_return_input_form,
                        num_frontier_points=num_frontier_points_input_form,
                        use_ledoit_wolf=use_ledoit_wolf_covariance_form,
                        asset_bounds_list_of_tuples=asset_bounds_input_form
                    )
                
                if opt_results and 'error' not in opt_results:
                    st.success(f"Portfolio Optimization ({optimization_objective_display_form}) Completed Successfully!")
                    
                    st.subheader("Optimal Portfolio Weights")
                    optimal_weights_dict = opt_results.get('optimal_weights', {})
                    if optimal_weights_dict:
                        weights_df = pd.DataFrame.from_dict(optimal_weights_dict, orient='index', columns=['Weight'])
                        weights_df["Weight %"] = (weights_df["Weight"] * 100)
                        # Display formatted weights
                        st.dataframe(weights_df[["Weight %"]].style.format("{:.2f}%"))
                        
                        # Pie chart for optimal allocation
                        fig_pie_optimal = px.pie(
                            weights_df[weights_df['Weight'] > 1e-5], # Filter tiny weights for cleaner chart
                            values='Weight', names=weights_df[weights_df['Weight'] > 1e-5].index,
                            title=f'Optimal Allocation ({optimization_objective_display_form})', hole=0.3
                        )
                        fig_pie_optimal.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(_apply_custom_theme(fig_pie_optimal, plot_theme), use_container_width=True)

                        # Calculate and display turnover if current weights were derived
                        if current_weights_for_turnover_calc:
                            turnover = calculate_portfolio_turnover(current_weights_for_turnover_calc, optimal_weights_dict)
                            st.metric(label="Portfolio Turnover (vs. Derived Current Weights)", value=format_percentage(turnover))
                        else:
                            st.caption("Current weights could not be derived for turnover calculation (e.g., no strategies selected or no P&L data).")
                        
                        with st.expander("View Optimal Weights Data (Numeric)"):
                            st.dataframe(weights_df)
                    else:
                        st.warning("Optimal weights could not be determined from the optimization results.")

                    st.subheader(f"Optimized Portfolio Performance (Annualized) - {optimization_objective_display_form}")
                    optimized_kpis = opt_results.get('performance', {})
                    if optimized_kpis:
                        # Ensure KPI keys match what KPIClusterDisplay expects or what's in KPI_CONFIG
                        opt_kpi_order = ["expected_annual_return", "annual_volatility", "sharpe_ratio"] # Example
                        KPIClusterDisplay(
                            kpi_results=optimized_kpis,
                            kpi_definitions=KPI_CONFIG, # Ensure this is well-defined
                            kpi_order=opt_kpi_order,
                            cols_per_row=3
                        ).render()
                        with st.expander("View Full Optimized Performance Data"):
                            st.dataframe(pd.DataFrame.from_dict(optimized_kpis, orient='index', columns=['Value']))
                    else:
                        st.warning("Optimized performance KPIs not found in results.")

                    # Risk Contributions (if available)
                    if "risk_contributions" in opt_results and opt_results["risk_contributions"]:
                        st.subheader("Risk Contributions to Portfolio Variance")
                        rc_data = opt_results["risk_contributions"]
                        if isinstance(rc_data, dict) and rc_data: # Check if it's a non-empty dict
                            rc_df = pd.DataFrame.from_dict(rc_data, orient='index', columns=['Risk Contribution %']).sort_values(by="Risk Contribution %", ascending=False)
                            # Ensure values are numeric for plotting
                            rc_df["Risk Contribution %"] = pd.to_numeric(rc_df["Risk Contribution %"], errors='coerce').fillna(0)
                            
                            fig_rc = px.bar(rc_df, x=rc_df.index, y="Risk Contribution %",
                                            title="Percentage Risk Contribution to Portfolio Variance",
                                            labels={"index": "Strategy", "Risk Contribution %": "Risk Contrib. (%)"},
                                            color="Risk Contribution %", color_continuous_scale=px.colors.sequential.Oranges_r)
                            fig_rc.update_yaxes(ticksuffix="%")
                            st.plotly_chart(_apply_custom_theme(fig_rc, plot_theme), use_container_width=True)
                            with st.expander("View Risk Contribution Data"): st.dataframe(rc_df)
                        elif not rc_data:
                             st.info("Risk contribution data is available but empty (e.g., single asset portfolio or specific optimization types).")
                        else:
                             st.info("Risk contribution data is not in the expected dictionary format.")
                    
                    # Efficient Frontier (if applicable)
                    if optimization_objective_form_key in ["maximize_sharpe_ratio", "minimize_volatility"]:
                        st.subheader("Efficient Frontier Visualization")
                        frontier_data = opt_results.get("efficient_frontier") # Expects {'volatility': [...], 'return': [...]}
                        if frontier_data and isinstance(frontier_data, dict) and \
                           'volatility' in frontier_data and 'return' in frontier_data and \
                           len(frontier_data['volatility']) == len(frontier_data['return']) and \
                           len(frontier_data['volatility']) > 0:
                            
                            perf_data_for_points = opt_results.get('performance', {})
                            # Points for Max Sharpe and Min Volatility on the frontier
                            max_s_vol, max_s_ret, min_v_vol, min_v_ret = None, None, None, None

                            # If current objective is Max Sharpe, use its performance directly
                            if optimization_objective_form_key == "maximize_sharpe_ratio":
                                max_s_vol = perf_data_for_points.get('annual_volatility')
                                max_s_ret = perf_data_for_points.get('expected_annual_return')
                            
                            # If current objective is Min Vol, use its performance for Min Vol point
                            if optimization_objective_form_key == "minimize_volatility":
                                min_v_vol = perf_data_for_points.get('annual_volatility')
                                min_v_ret = perf_data_for_points.get('expected_annual_return')

                            # Attempt to find Max Sharpe and Min Vol points from the frontier data if not directly available
                            # This is useful if the objective was different or to always show these points
                            temp_frontier_df = pd.DataFrame(frontier_data)
                            if not temp_frontier_df.empty:
                                if min_v_vol is None: # Find Min Volatility point from frontier
                                    min_vol_idx = temp_frontier_df['volatility'].idxmin()
                                    min_v_vol = temp_frontier_df.loc[min_vol_idx, 'volatility']
                                    min_v_ret = temp_frontier_df.loc[min_vol_idx, 'return']
                                
                                if max_s_vol is None: # Find Max Sharpe point from frontier
                                    # Calculate Sharpe for each point on frontier
                                    temp_frontier_df['sharpe'] = (temp_frontier_df['return'] - risk_free_rate) / temp_frontier_df['volatility'].replace(0, np.nan) # Avoid div by zero
                                    if not temp_frontier_df['sharpe'].isnull().all():
                                        max_sharpe_idx = temp_frontier_df['sharpe'].idxmax()
                                        max_s_vol = temp_frontier_df.loc[max_sharpe_idx, 'volatility']
                                        max_s_ret = temp_frontier_df.loc[max_sharpe_idx, 'return']
                            
                            frontier_fig = plot_efficient_frontier(
                                frontier_data['volatility'], frontier_data['return'],
                                max_s_vol, max_s_ret, min_v_vol, min_v_ret, theme=plot_theme
                            )
                            if frontier_fig:
                                st.plotly_chart(frontier_fig, use_container_width=True)
                                if not pd.DataFrame(frontier_data).empty: # Check if frontier data itself is non-empty
                                    with st.expander("View Efficient Frontier Data Points"):
                                        st.dataframe(pd.DataFrame(frontier_data))
                            else: display_custom_message("Could not generate the Efficient Frontier plot.", "warning")
                        else:
                            display_custom_message("Efficient Frontier data is not available or is incomplete for the selected objective.", "info")
                elif opt_results and 'error' in opt_results: # Handle errors reported by the optimization logic
                    display_custom_message(f"Portfolio Optimization Error: {opt_results.get('error')}", "error")
                else: # Generic failure if opt_results is None or malformed
                    display_custom_message("Portfolio optimization process failed to return results.", "error")
        # Implicit else for `if submit_optimization_button...`: if not submitted, nothing happens here.

    with tab_comparison:
        st.header("Compare Equity Curves of Any Two Accounts")
        if len(unique_accounts_all) < 2:
            st.info("At least two distinct accounts are needed in the uploaded data for comparison.")
        else:
            col1_comp, col2_comp = st.columns(2)
            # Ensure unique keys for selectboxes
            acc1_comp_sel = col1_comp.selectbox("Select Account 1 for Comparison:", unique_accounts_all, index=0, key="acc_sel_1_comp_v4")
            # Default index for second account to be different if possible
            idx2_comp = 1 if len(unique_accounts_all) > 1 else 0
            acc2_comp_sel = col2_comp.selectbox("Select Account 2 for Comparison:", unique_accounts_all, index=idx2_comp, key="acc_sel_2_comp_v4")

            if acc1_comp_sel == acc2_comp_sel:
                st.warning("Please select two different accounts for a meaningful comparison.")
            else:
                st.subheader(f"Equity Curve Comparison: {acc1_comp_sel} vs. {acc2_comp_sel}")
                # Fetch raw data for each selected account from the base_df
                df_acc1_raw_comp = base_df[base_df[account_col_actual] == acc1_comp_sel]
                df_acc2_raw_comp = base_df[base_df[account_col_actual] == acc2_comp_sel]
                
                combined_equity_comp_df = pd.DataFrame() # Initialize

                for df_raw_loop, acc_name_loop in [(df_acc1_raw_comp, acc1_comp_sel), (df_acc2_raw_comp, acc2_comp_sel)]:
                    if df_raw_loop.empty: 
                        logger.info(f"No raw data for account {acc_name_loop} in comparison tab.")
                        continue # Skip if no data for this account
                    
                    # Clean data specifically for this account's equity curve
                    df_cleaned_loop = _clean_data_for_analysis(
                        df_raw_loop, 
                        date_col=date_col_actual, 
                        pnl_col=pnl_col_actual, 
                        required_cols_to_check_na=[pnl_col_actual] # Date and PnL are key
                    )
                    if not df_cleaned_loop.empty:
                        df_cleaned_loop['cumulative_pnl'] = df_cleaned_loop[pnl_col_actual].cumsum()
                        # Prepare a temporary df with Date and this account's equity
                        temp_equity_df = df_cleaned_loop[[date_col_actual, 'cumulative_pnl']].rename(
                            columns={'cumulative_pnl': f'Equity_{acc_name_loop}'}
                        )
                        # Merge with the combined df
                        if combined_equity_comp_df.empty:
                            combined_equity_comp_df = temp_equity_df
                        else:
                            combined_equity_comp_df = pd.merge(combined_equity_comp_df, temp_equity_df, on=date_col_actual, how='outer')
                
                if combined_equity_comp_df.empty or not any(f'Equity_{acc}' in combined_equity_comp_df.columns for acc in [acc1_comp_sel, acc2_comp_sel]):
                    display_custom_message(f"Could not generate equity data for one or both selected accounts ({acc1_comp_sel}, {acc2_comp_sel}). They might lack valid P&L entries.", "warning")
                else:
                    combined_equity_comp_df.sort_values(by=date_col_actual, inplace=True)
                    # Forward fill to handle days where one account might not have trades, then fill remaining NaNs (e.g. at start) with 0
                    combined_equity_comp_df = combined_equity_comp_df.ffill().fillna(0) 
                    
                    fig_comp_equity = go.Figure()
                    if f'Equity_{acc1_comp_sel}' in combined_equity_comp_df:
                        fig_comp_equity.add_trace(go.Scatter(
                            x=combined_equity_comp_df[date_col_actual],
                            y=combined_equity_comp_df[f'Equity_{acc1_comp_sel}'],
                            mode='lines', name=f"{acc1_comp_sel} Equity"
                        ))
                    if f'Equity_{acc2_comp_sel}' in combined_equity_comp_df:
                         fig_comp_equity.add_trace(go.Scatter(
                            x=combined_equity_comp_df[date_col_actual],
                            y=combined_equity_comp_df[f'Equity_{acc2_comp_sel}'],
                            mode='lines', name=f"{acc2_comp_sel} Equity"
                        ))
                    fig_comp_equity.update_layout(
                        title=f"Equity Curve Comparison: {acc1_comp_sel} vs. {acc2_comp_sel}",
                        xaxis_title="Date", yaxis_title="Cumulative PnL", hovermode="x unified"
                    )
                    st.plotly_chart(_apply_custom_theme(fig_comp_equity, plot_theme), use_container_width=True)
                    
                    if not combined_equity_comp_df.empty:
                        with st.expander("View Combined Equity Comparison Data (Aligned & Filled)"):
                            st.dataframe(combined_equity_comp_df)

# Standalone execution / testing block
if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Portfolio Analysis", initial_sidebar_state="expanded")
    
    # Mock essential configurations if not imported (e.g., running file directly)
    if 'APP_TITLE' not in globals(): APP_TITLE = "PortfolioApp_Standalone_Test" 
    if 'EXPECTED_COLUMNS' not in globals():
        EXPECTED_COLUMNS = {
            'account_str': 'Account', 
            'pnl': 'PnL',             
            'date': 'Date',           
            'strategy': 'Strategy'    
        }
    if 'RISK_FREE_RATE' not in globals(): RISK_FREE_RATE = 0.01 # Default 1%
    if 'KPI_CONFIG' not in globals(): 
        KPI_CONFIG = { # Basic mock for KPI definitions
            "total_pnl": {"label": "Total PnL", "type": "currency", "description": "Total Profit and Loss"},
            "sharpe_ratio": {"label": "Sharpe Ratio", "type": "float", "description": "Risk-adjusted return (annualized)"},
            "sortino_ratio": {"label": "Sortino Ratio", "type": "float", "description": "Downside risk-adjusted return"},
            "calmar_ratio": {"label": "Calmar Ratio", "type": "float", "description": "Return over Max Drawdown"},
            "max_drawdown_abs": {"label": "Max Drawdown (Abs)", "type": "currency", "description": "Largest peak-to-trough decline"},
            "max_drawdown_pct": {"label": "Max Drawdown (%)", "type": "percentage", "description": "Largest peak-to-trough decline in %"},
            "avg_daily_pnl": {"label": "Avg Daily PnL", "type": "currency", "description": "Average daily profit or loss"},
            "expected_annual_return": {"label": "Expected Annual Return", "type": "percentage", "description": "Annualized expected return"},
            "annual_volatility": {"label": "Annual Volatility", "type": "percentage", "description": "Annualized standard deviation of returns"},
        }
    if 'COLORS' not in globals(): COLORS = {"primary": "#007bff", "secondary": "#6c757d"} # Mock colors

    # Initialize session state for standalone testing if not already done
    if 'app_initialized' not in st.session_state: 
        # Create more comprehensive sample data for testing all features
        dates = pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-01', 
                                '2023-01-02', '2023-01-02', '2023-01-02',
                                '2023-01-03', '2023-01-03', '2023-01-03',
                                '2023-01-04', '2023-01-04', '2023-01-04'] * 5) # More data points
        
        strategies = ['StratA', 'StratB', 'StratC'] * (len(dates)//3)
        accounts = ['AccX', 'AccY'] * (len(dates)//2)
        
        np.random.seed(42) # for reproducibility
        pnl_values = np.random.randn(len(dates)) * 100 # Random PnL values

        sample_data_dict = {
            EXPECTED_COLUMNS['date']: dates[:len(pnl_values)], # Ensure lengths match
            EXPECTED_COLUMNS['pnl']: pnl_values,
            EXPECTED_COLUMNS['strategy']: strategies[:len(pnl_values)],
            EXPECTED_COLUMNS['account_str']: accounts[:len(pnl_values)]
        }
        st.session_state.processed_data = pd.DataFrame(sample_data_dict)
        st.session_state.initial_capital = 100000.0
        st.session_state.risk_free_rate = RISK_FREE_RATE 
        st.session_state.current_theme = 'dark' # Default to dark for testing
        # Mock user column mapping if other parts of your app rely on it
        st.session_state.user_column_mapping = { 
            'date': EXPECTED_COLUMNS['date'], 'pnl': EXPECTED_COLUMNS['pnl'],
            'strategy': EXPECTED_COLUMNS['strategy'], 'account_str': EXPECTED_COLUMNS['account_str']
        }
        st.session_state.app_initialized = True # Mark as initialized
        logger.info("Mock data and session state initialized for standalone run.")
        st.sidebar.success("Mock data loaded for testing.")
    
    # Call the main function to display the page
    show_portfolio_analysis_page()
