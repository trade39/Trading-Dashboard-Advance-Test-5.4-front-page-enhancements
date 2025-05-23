"""
pages/11_💼_Portfolio_Analysis.py

This page provides portfolio-level aggregation and analysis. It features a global
date filter, calculates combined P&L, overall risk metrics, visualizes
inter-strategy and inter-account correlations, compares equity curves, and
includes a portfolio optimization section with efficient frontier visualization,
Risk Parity, robust covariance options, per-asset weight constraints, display of
risk contributions, and turnover reporting, all within the selected date range.
"""
import streamlit as st
import pandas as pd
import numpy as np
import logging
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, Optional, List, Tuple
from datetime import date, timedelta

# Attempt to import project-specific modules.
# These modules are expected to be in the PYTHONPATH or project structure.
try:
    from config import APP_TITLE, EXPECTED_COLUMNS, KPI_CONFIG, COLORS, RISK_FREE_RATE
    from utils.common_utils import display_custom_message, format_currency, format_percentage, calculate_portfolio_turnover
    from services.analysis_service import AnalysisService
    from services.portfolio_analysis import PortfolioAnalysisService
    from plotting import plot_equity_curve_and_drawdown, _apply_custom_theme, plot_efficient_frontier
    from components.kpi_display import KPIClusterDisplay # Assuming this component exists
except ImportError as e:
    # Log critical import errors and display an error message in Streamlit.
    critical_error_message = f"Portfolio Analysis Page Error: Critical module import failed: {e}. This page cannot be loaded."
    if 'st' in globals() and hasattr(st, 'error'): # Check if Streamlit context is available
        st.error(critical_error_message)
    
    # Fallback logging if Streamlit's logger isn't fully set up or APP_TITLE is missing
    try:
        page_logger_name = "PortfolioAnalysisPage_ImportErrorLogger"
        if 'APP_TITLE' in globals(): # Check if APP_TITLE was successfully imported
            page_logger_name = f"{APP_TITLE}.PortfolioAnalysisPage.ImportError"
        page_logger = logging.getLogger(page_logger_name)
        page_logger.critical(f"CRITICAL IMPORT ERROR in Portfolio Analysis Page: {e}", exc_info=True)
    except Exception as log_e: # Catch any exception during fallback logging
        print(f"Fallback logging error during Portfolio Analysis Page import: {log_e}")
    
    # Stop script execution if Streamlit context allows
    if 'st' in globals() and hasattr(st, 'stop'):
        st.stop()
    else: # If not in Streamlit context (e.g. direct script run without Streamlit), raise the error
        raise ImportError(critical_error_message) from e

# Initialize logger for this page
logger = logging.getLogger(APP_TITLE if 'APP_TITLE' in globals() else "PortfolioAnalysisPage")
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
    - Converts date column to datetime.
    - Converts specified PnL and numeric columns to numeric types.
    - Converts specified string columns to string types.
    - Drops rows with NaN/NaT in essential columns (date, and optionally PnL, strategy, account).
    - Sorts the DataFrame by the date column if specified.
    """
    if df.empty:
        logger.info("Input DataFrame for cleaning is empty. Returning empty DataFrame.")
        return pd.DataFrame()

    df_cleaned = df.copy()

    if date_col not in df_cleaned.columns:
        logger.warning(f"Date column '{date_col}' not found in DataFrame. Returning empty DataFrame.")
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
        logger.warning("No valid columns identified for NaN checking after initial processing. Skipping dropna.")


    if df_cleaned.empty:
        logger.info(f"DataFrame became empty after cleaning and NaN drop for columns: {valid_cols_for_nan_check}.")
        return pd.DataFrame()

    if sort_by_date and date_col in df_cleaned.columns:
        df_cleaned.sort_values(by=date_col, inplace=True)
    
    return df_cleaned


def _calculate_drawdown_series_for_aggregated_df(cumulative_pnl_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """ 
    Helper to calculate absolute and percentage drawdown series from a cumulative P&L series.
    Handles cases where cumulative P&L might be zero or negative.
    """
    if cumulative_pnl_series.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    
    high_water_mark = cumulative_pnl_series.cummax()
    drawdown_abs_series = high_water_mark - cumulative_pnl_series
    
    drawdown_pct_series = pd.Series(
        np.where(
            high_water_mark.abs() > 1e-9, 
            (drawdown_abs_series / high_water_mark.replace(0, np.nan)) * 100.0, 
            np.where(drawdown_abs_series > 1e-9, 100.0, 0.0) 
        ),
        index=cumulative_pnl_series.index,
        dtype=float
    ).fillna(0) 
    
    return drawdown_abs_series.fillna(0), drawdown_pct_series.fillna(0)


@st.cache_data 
def calculate_metrics_for_df(
    df_input_tuple: Tuple[List[tuple], List[str]], 
    pnl_col: str,
    date_col: str,
    risk_free_rate: float,
    initial_capital: float
) -> Dict[str, Any]:
    """
    Calculates core performance metrics using programmatic keys.
    """
    df_data, df_columns = df_input_tuple
    df_input = pd.DataFrame(data=df_data, columns=df_columns)

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

    kpis_from_service = general_analysis_service.get_core_kpis(df_copy, risk_free_rate, initial_capital=initial_capital)
    
    if kpis_from_service and 'error' not in kpis_from_service:
        # Return using programmatic keys, ensure all expected keys are present or None
        return {
            "total_pnl": kpis_from_service.get("total_pnl"),
            "total_trades": kpis_from_service.get("total_trades"),
            "win_rate": kpis_from_service.get("win_rate"),
            "avg_trade_pnl": kpis_from_service.get("avg_trade_pnl"),
            "sharpe_ratio": kpis_from_service.get("sharpe_ratio"),
            "sortino_ratio": kpis_from_service.get("sortino_ratio"),
            "calmar_ratio": kpis_from_service.get("calmar_ratio"),
            "max_drawdown_pct": kpis_from_service.get("max_drawdown_pct"),
            "max_drawdown_abs": kpis_from_service.get("max_drawdown_abs"),
            "avg_daily_pnl": kpis_from_service.get("avg_daily_pnl"),
            "pnl_skewness": kpis_from_service.get("pnl_skewness"), # Added for completeness
            "pnl_kurtosis": kpis_from_service.get("pnl_kurtosis")  # Added for completeness
        }
    error_msg = kpis_from_service.get('error', 'Unknown error') if kpis_from_service else 'KPI calculation service failed'
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
    """ 
    Encapsulates data preparation and the call to the optimization service.
    """
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
        logger.warning("No data for selected strategies in the specified date range after cleaning for optimization.")
        return {"error": "No data for selected strategies in the specified date range after cleaning."}

    latest_date_in_data = opt_df_filtered_strategies[date_col_actual].max()
    earliest_date_in_filtered_data = opt_df_filtered_strategies[date_col_actual].min()
    calculated_start_date_lookback = latest_date_in_data - pd.Timedelta(days=lookback_days_opt - 1)
    start_date_lookback = max(calculated_start_date_lookback, earliest_date_in_filtered_data)
    
    opt_df_lookback = opt_df_filtered_strategies[opt_df_filtered_strategies[date_col_actual] >= start_date_lookback]

    if opt_df_lookback.empty:
        logger.warning("No data within the specified lookback period (relative to the global date range).")
        return {"error": "No data within the specified lookback period (relative to the global date range)."}

    try:
        daily_pnl_pivot = opt_df_lookback.groupby(
            [opt_df_lookback[date_col_actual].dt.normalize(), strategy_col_actual]
        )[pnl_col_actual].sum().unstack(fill_value=0)
        daily_pnl_pivot = daily_pnl_pivot.reindex(columns=selected_strategies_for_opt, fill_value=0.0)
    except (KeyError, ValueError, TypeError, AttributeError) as e:
        logger.error(f"Error during P&L pivot for optimization: {e}", exc_info=True)
        return {"error": f"Failed to pivot P&L data for optimization: {e}"}

    if global_initial_capital <= 0:
        logger.error("Initial capital must be a positive value for return calculation in optimization.")
        return {"error": "Initial capital must be a positive value for return calculation."}
    
    daily_returns_for_opt = (daily_pnl_pivot / global_initial_capital).fillna(0)

    min_hist_points_needed = 20 
    if optimization_objective_key == "risk_parity" and len(selected_strategies_for_opt) <= 1:
        min_hist_points_needed = 2 
    if daily_returns_for_opt.shape[0] < min_hist_points_needed:
        logger.warning(f"Insufficient historical data: Need {min_hist_points_needed}, found {daily_returns_for_opt.shape[0]}.")
        return {"error": f"Insufficient historical data: Need at least {min_hist_points_needed} data points within lookback, found {daily_returns_for_opt.shape[0]}."}

    try:
        optimization_results = portfolio_specific_service.prepare_and_run_optimization(
            daily_returns_df=daily_returns_for_opt,
            objective=optimization_objective_key,
            risk_free_rate=risk_free_rate,
            target_return_level=target_return_val,
            trading_days=252, 
            num_frontier_points=num_frontier_points,
            use_ledoit_wolf=use_ledoit_wolf,
            asset_bounds=asset_bounds_list_of_tuples
        )
        return optimization_results
    except Exception as e:
        logger.error(f"Error calling portfolio optimization service: {e}", exc_info=True)
        return {"error": f"Optimization service execution failed: {e}"}


def show_portfolio_analysis_page():
    st.set_page_config(layout="wide", page_title="Portfolio Analysis", initial_sidebar_state="expanded")
    st.title("💼 Portfolio-Level Analysis")
    logger.info("Rendering Portfolio Analysis Page.")

    if 'processed_data' not in st.session_state or st.session_state.processed_data is None or st.session_state.processed_data.empty:
        display_custom_message("Please upload and process data on the main page to view portfolio analysis.", "info")
        logger.info("Portfolio analysis page: No processed data found in session state.")
        return

    base_df_original = st.session_state.processed_data.copy() 
    
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
        msg = f"Essential column configurations are missing ({', '.join(missing_configs)}) in EXPECTED_COLUMNS. Analysis cannot proceed."
        display_custom_message(msg, "error"); logger.critical(f"Portfolio page: {msg}"); return
        
    essential_cols_in_df = [account_col_actual, pnl_col_actual, date_col_actual, strategy_col_actual]
    if not all(col in base_df_original.columns for col in essential_cols_in_df):
        missing_cols_in_df = [col for col in essential_cols_in_df if col not in base_df_original.columns]
        msg = f"Essential columns ({', '.join(missing_cols_in_df)}) defined in EXPECTED_COLUMNS are not found in the uploaded data."
        display_custom_message(msg, "error"); logger.error(f"Portfolio page: {msg}. Available columns: {base_df_original.columns.tolist()}"); return

    try:
        base_df_original[date_col_actual] = pd.to_datetime(base_df_original[date_col_actual], errors='coerce')
        base_df_original.dropna(subset=[date_col_actual], inplace=True) 
    except Exception as e:
        msg = f"Failed to convert date column '{date_col_actual}' to datetime for global filter setup: {e}."
        display_custom_message(msg, "error"); logger.error(f"Portfolio page: {msg}", exc_info=True); return

    if base_df_original.empty:
        display_custom_message("No valid date data found after initial conversion. Cannot proceed.", "warning"); return
        
    st.sidebar.subheader("Global Date Range Filter")
    min_data_date = base_df_original[date_col_actual].min().date() 
    max_data_date = base_df_original[date_col_actual].max().date() 

    start_date_val = st.session_state.get("portfolio_global_start_date", min_data_date)
    end_date_val = st.session_state.get("portfolio_global_end_date", max_data_date)
    
    start_date_val = max(min_data_date, start_date_val)
    end_date_val = min(max_data_date, end_date_val)
    if start_date_val > end_date_val: 
        start_date_val = min_data_date
        end_date_val = max_data_date


    start_date_selected = st.sidebar.date_input(
        "Start Date:", 
        value=start_date_val, 
        min_value=min_data_date, 
        max_value=max_data_date, 
        key="global_start_date_portfolio_v2" 
    )
    end_date_selected = st.sidebar.date_input(
        "End Date:", 
        value=end_date_val, 
        min_value=start_date_selected, 
        max_value=max_data_date, 
        key="global_end_date_portfolio_v2" 
    )
    
    st.session_state.portfolio_global_start_date = start_date_selected
    st.session_state.portfolio_global_end_date = end_date_selected


    if start_date_selected > end_date_selected:
        st.sidebar.error("Start date cannot be after end date. Please adjust the selection.")
        logger.warning("Global date filter: Start date is after end date.")
        return 

    start_datetime_filter = pd.to_datetime(start_date_selected)
    end_datetime_filter = pd.to_datetime(end_date_selected) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1) 

    base_df_filtered_by_date = base_df_original[
        (base_df_original[date_col_actual] >= start_datetime_filter) & 
        (base_df_original[date_col_actual] <= end_datetime_filter)
    ].copy() 

    if base_df_filtered_by_date.empty:
        display_custom_message(f"No data available for the selected global date range: {start_date_selected.strftime('%Y-%m-%d')} to {end_date_selected.strftime('%Y-%m-%d')}.", "info")
        logger.info("Global date filter resulted in an empty DataFrame.")
        return

    unique_accounts_in_range = sorted(base_df_filtered_by_date[account_col_actual].dropna().astype(str).unique())
    if not unique_accounts_in_range:
        display_custom_message(f"No accounts found in the data for the selected date range: {start_date_selected.strftime('%Y-%m-%d')} to {end_date_selected.strftime('%Y-%m-%d')}.", "info")
        return

    st.sidebar.subheader("Portfolio Account Selection")
    selected_accounts_for_portfolio = unique_accounts_in_range
    if len(unique_accounts_in_range) > 1:
        selected_accounts_for_portfolio = st.sidebar.multiselect(
            "Select accounts for portfolio view:", options=unique_accounts_in_range,
            default=unique_accounts_in_range, key="portfolio_view_account_multiselect_v5" 
        )
    else:
        st.sidebar.info(f"Displaying portfolio view for the single account: {unique_accounts_in_range[0]}")

    if not selected_accounts_for_portfolio:
        display_custom_message("Please select at least one account for the portfolio view.", "info"); return

    portfolio_df_uncleaned = base_df_filtered_by_date[base_df_filtered_by_date[account_col_actual].isin(selected_accounts_for_portfolio)].copy()
    if portfolio_df_uncleaned.empty:
        display_custom_message("No data for the selected accounts in the selected date range.", "info"); return
    
    portfolio_df = _clean_data_for_analysis(
        portfolio_df_uncleaned, date_col=date_col_actual, pnl_col=pnl_col_actual,
        strategy_col=strategy_col_actual, account_col=account_col_actual,
        required_cols_to_check_na=[pnl_col_actual, strategy_col_actual, account_col_actual],
        string_cols_to_convert=[strategy_col_actual, account_col_actual] 
    )
    if portfolio_df.empty:
        display_custom_message("No valid data after cleaning for selected accounts and date range.", "warning"); return

    tab_titles = [
        "📈 Overall Performance", "🔗 Inter-Connections", "📊 Account Breakdown",
        "⚖️ Portfolio Optimization", "↔️ Equity Comparison"
    ]
    tab_overall, tab_connections, tab_breakdown, tab_optimization, tab_comparison = st.tabs(tab_titles)

    with tab_overall:
        st.header(f"Overall Performance ({start_date_selected.strftime('%Y-%m-%d')} to {end_date_selected.strftime('%Y-%m-%d')})")
        portfolio_daily_pnl_series = pd.Series(dtype=float)
        if not portfolio_df.empty:
            try:
                portfolio_daily_pnl_series = portfolio_df.groupby(portfolio_df[date_col_actual].dt.normalize())[pnl_col_actual].sum()
            except (AttributeError, KeyError, TypeError) as e:
                logger.error(f"Error grouping by date for daily P&L in Overall Performance: {e}", exc_info=True)
                display_custom_message(f"Error processing daily P&L: {e}", "error")
        
        if portfolio_daily_pnl_series.empty:
            display_custom_message("No P&L data after daily aggregation for selected portfolio and dates.", "warning")
        else:
            portfolio_daily_summary_df = pd.DataFrame({date_col_actual: portfolio_daily_pnl_series.index, pnl_col_actual: portfolio_daily_pnl_series.values})
            portfolio_daily_summary_df['cumulative_pnl'] = portfolio_daily_summary_df[pnl_col_actual].cumsum()
            portfolio_daily_summary_df['win'] = portfolio_daily_summary_df[pnl_col_actual] > 0
            
            if 'cumulative_pnl' in portfolio_daily_summary_df.columns and not portfolio_daily_summary_df['cumulative_pnl'].empty:
                drawdown_abs_series, drawdown_pct_series = _calculate_drawdown_series_for_aggregated_df(portfolio_daily_summary_df['cumulative_pnl'])
                portfolio_daily_summary_df['drawdown_abs'] = drawdown_abs_series
                portfolio_daily_summary_df['drawdown_pct'] = drawdown_pct_series
            else: 
                portfolio_daily_summary_df['drawdown_abs'] = pd.Series(dtype=float)
                portfolio_daily_summary_df['drawdown_pct'] = pd.Series(dtype=float)

            with st.spinner("Calculating selected portfolio KPIs..."):
                portfolio_kpis_results = calculate_metrics_for_df(
                    (portfolio_daily_summary_df.to_records(index=False).tolist(), portfolio_daily_summary_df.columns.tolist()),
                    pnl_col_actual, date_col_actual, risk_free_rate, global_initial_capital
                )
            
            if portfolio_kpis_results and 'error' not in portfolio_kpis_results:
                # Use programmatic keys for display order
                portfolio_kpi_display_order = [
                    "total_pnl", "sharpe_ratio", "sortino_ratio", 
                    "calmar_ratio", "max_drawdown_abs", "max_drawdown_pct", 
                    "avg_daily_pnl", "win_rate", "total_trades" 
                ]
                kpis_to_show = {
                    key: portfolio_kpis_results.get(key) 
                    for key in portfolio_kpi_display_order
                } # KPIClusterDisplay should handle None values gracefully by checking KPI_CONFIG

                if any(v is not None for v in kpis_to_show.values()): # Check if there's anything to show
                    KPIClusterDisplay(kpis_to_show, KPI_CONFIG, portfolio_kpi_display_order, cols_per_row=3).render()
                else: 
                    display_custom_message("Could not retrieve relevant KPIs for the selected portfolio (all values are N/A).", "warning")
            else: 
                display_custom_message(f"Error calculating KPIs: {portfolio_kpis_results.get('error', 'Unknown error') if portfolio_kpis_results else 'KPI calculation failed'}", "error")
            
            st.subheader("Combined Equity Curve & Drawdown")
            fig_overall_equity = plot_equity_curve_and_drawdown(
                portfolio_daily_summary_df, date_col_actual, 'cumulative_pnl', 'drawdown_pct', theme=plot_theme
            )
            if fig_overall_equity: st.plotly_chart(fig_overall_equity, use_container_width=True)
            else: display_custom_message("Could not generate the equity curve for the selected portfolio.", "warning")

            if not portfolio_daily_summary_df.empty:
                with st.expander("View Underlying Equity Curve Data (Daily Aggregated)"):
                    st.dataframe(portfolio_daily_summary_df)


    with tab_connections:
        st.header(f"Inter-Connections ({start_date_selected.strftime('%Y-%m-%d')} to {end_date_selected.strftime('%Y-%m-%d')})")
        
        correlation_period_options = {"Full Selected Range": None, "Last 30 days": 30, "Last 90 days": 90, "Last 180 days": 180}
        selected_corr_period_label = st.selectbox(
            "Select Correlation Period (within global range):",
            options=list(correlation_period_options.keys()),
            index=0, 
            key="correlation_period_selector_portfolio_v3" 
        )
        selected_corr_period_days = correlation_period_options[selected_corr_period_label]

        df_for_correlation_analysis = portfolio_df.copy() 
        if selected_corr_period_days is not None and date_col_actual in df_for_correlation_analysis.columns:
            latest_date_for_corr = df_for_correlation_analysis[date_col_actual].max()
            earliest_date_for_corr = df_for_correlation_analysis[date_col_actual].min()
            
            start_date_relative_corr = max(earliest_date_for_corr, latest_date_for_corr - pd.Timedelta(days=selected_corr_period_days -1))
            df_for_correlation_analysis = df_for_correlation_analysis[df_for_correlation_analysis[date_col_actual] >= start_date_relative_corr]
        
        if df_for_correlation_analysis.empty:
            display_custom_message(f"No data available for the selected correlation period: {selected_corr_period_label}", "warning")
        else:
            st.subheader("🔀 Inter-Strategy P&L Correlation")
            if strategy_col_actual not in df_for_correlation_analysis.columns:
                display_custom_message(f"Strategy column '{strategy_col_actual}' not found.", "warning")
            else:
                unique_strats_for_corr = df_for_correlation_analysis[strategy_col_actual].dropna().unique()
                if len(unique_strats_for_corr) < 2:
                    st.info("At least two distinct strategies are needed for inter-strategy correlation analysis.")
                else:
                    strat_corr_prep_df = df_for_correlation_analysis[[date_col_actual, strategy_col_actual, pnl_col_actual]].copy()
                    strat_corr_prep_df.sort_values(by=[date_col_actual, strategy_col_actual], inplace=True)
                    with st.spinner(f"Calculating inter-strategy P&L correlations ({selected_corr_period_label})..."):
                        try:
                            results_strat_corr = portfolio_specific_service.get_portfolio_inter_strategy_correlation(
                                strat_corr_prep_df, strategy_col_actual, pnl_col_actual, date_col_actual)
                        except Exception as e_corr_strat:
                            logger.error(f"Inter-strategy correlation service error: {e_corr_strat}", exc_info=True)
                            results_strat_corr = {"error": f"Service failed: {e_corr_strat}"}

                    if results_strat_corr and 'error' not in results_strat_corr:
                        matrix_strat_corr = results_strat_corr.get('correlation_matrix')
                        if matrix_strat_corr is not None and not matrix_strat_corr.empty and matrix_strat_corr.shape[0] > 1:
                            fig_strat_heatmap = go.Figure(data=go.Heatmap(
                                z=matrix_strat_corr.values, x=matrix_strat_corr.columns, y=matrix_strat_corr.index,
                                colorscale='RdBu', zmin=-1, zmax=1, text=matrix_strat_corr.round(2).astype(str),
                                texttemplate="%{text}", hoverongaps=False))
                            fig_strat_heatmap.update_layout(title=f"Inter-Strategy Daily P&L Correlation ({selected_corr_period_label})")
                            st.plotly_chart(_apply_custom_theme(fig_strat_heatmap, plot_theme), use_container_width=True)
                            with st.expander("View Inter-Strategy Correlation Matrix"): st.dataframe(matrix_strat_corr)
                        else: display_custom_message("Not enough data or strategies for inter-strategy correlation matrix.", "info")
                    elif results_strat_corr: display_custom_message(f"Inter-strategy correlation error: {results_strat_corr.get('error')}", "error")
                    else: display_custom_message("Inter-strategy correlation analysis failed to produce results.", "error")

            st.subheader("🤝 Inter-Account P&L Correlation")
            if len(selected_accounts_for_portfolio) < 2: 
                st.info("At least two accounts must be selected in the sidebar for inter-account correlation.")
            else:
                acc_corr_prep_df = df_for_correlation_analysis[[date_col_actual, account_col_actual, pnl_col_actual]].copy()
                acc_corr_prep_df = acc_corr_prep_df[acc_corr_prep_df[account_col_actual].isin(selected_accounts_for_portfolio)]

                if len(acc_corr_prep_df[account_col_actual].unique()) < 2:
                    st.info(f"Fewer than two selected accounts have data in the chosen period ({selected_corr_period_label}) for correlation.")
                else:
                    acc_corr_prep_df.sort_values(by=[date_col_actual, account_col_actual], inplace=True)
                    with st.spinner(f"Calculating inter-account P&L correlations ({selected_corr_period_label})..."):
                        try:
                            results_acc_corr = portfolio_specific_service.get_portfolio_inter_account_correlation(
                                acc_corr_prep_df, account_col_actual, pnl_col_actual, date_col_actual)
                        except Exception as e_corr_acc:
                            logger.error(f"Inter-account correlation service error: {e_corr_acc}", exc_info=True)
                            results_acc_corr = {"error": f"Service failed: {e_corr_acc}"}

                    if results_acc_corr and 'error' not in results_acc_corr:
                        matrix_acc_corr = results_acc_corr.get('correlation_matrix')
                        if matrix_acc_corr is not None and not matrix_acc_corr.empty and matrix_acc_corr.shape[0] > 1:
                            fig_acc_heatmap = go.Figure(data=go.Heatmap(
                                z=matrix_acc_corr.values, x=matrix_acc_corr.columns, y=matrix_acc_corr.index,
                                colorscale='RdBu', zmin=-1, zmax=1, text=matrix_acc_corr.round(2).astype(str),
                                texttemplate="%{text}", hoverongaps=False))
                            fig_acc_heatmap.update_layout(title=f"Inter-Account Daily P&L Correlation ({selected_corr_period_label})")
                            st.plotly_chart(_apply_custom_theme(fig_acc_heatmap, plot_theme), use_container_width=True)
                            with st.expander("View Inter-Account Correlation Matrix"): st.dataframe(matrix_acc_corr)
                        else: display_custom_message("Not enough data or accounts for inter-account correlation matrix.", "info")
                    elif results_acc_corr: display_custom_message(f"Inter-account correlation error: {results_acc_corr.get('error')}", "error")
                    else: display_custom_message("Inter-account correlation analysis failed to produce results.", "error")

    with tab_breakdown:
        st.header(f"Account Performance Breakdown ({start_date_selected.strftime('%Y-%m-%d')} to {end_date_selected.strftime('%Y-%m-%d')})")
        account_metrics_list = []
        for acc_name_iter in selected_accounts_for_portfolio:
            acc_df_for_breakdown = base_df_filtered_by_date[base_df_filtered_by_date[account_col_actual] == acc_name_iter].copy()
            
            if not acc_df_for_breakdown.empty:
                metrics_result = calculate_metrics_for_df(
                    (acc_df_for_breakdown.to_records(index=False).tolist(), acc_df_for_breakdown.columns.tolist()),
                    pnl_col_actual, date_col_actual, risk_free_rate, global_initial_capital
                )
                if metrics_result and 'error' not in metrics_result:
                    account_metrics_list.append({"Account": acc_name_iter, **metrics_result})
                else:
                    logger.warning(f"Could not calculate metrics for account {acc_name_iter}: {metrics_result.get('error', 'Unknown error')}")
            else: 
                logger.info(f"No data for account {acc_name_iter} in breakdown for selected date range.")

        if account_metrics_list:
            summary_metrics_df = pd.DataFrame(account_metrics_list)
            
            if "total_pnl" in summary_metrics_df.columns: # Use programmatic key
                summary_metrics_df["Total PnL Numeric"] = pd.to_numeric(summary_metrics_df["total_pnl"], errors='coerce')
                pnl_pie_chart_df = summary_metrics_df[
                    summary_metrics_df["Total PnL Numeric"].notna() & (summary_metrics_df["Total PnL Numeric"].abs() > 1e-6) 
                ][["Account", "Total PnL Numeric"]].copy()

                if not pnl_pie_chart_df.empty:
                    fig_pnl_pie = px.pie(pnl_pie_chart_df, names='Account', values='Total PnL Numeric', 
                                         title='P&L Contribution by Account (Selected Period)', hole=0.3)
                    fig_pnl_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(_apply_custom_theme(fig_pnl_pie, plot_theme), use_container_width=True)
                    with st.expander("View P&L Contribution Data (Numeric)"): 
                        st.dataframe(pnl_pie_chart_df.rename(columns={"Total PnL Numeric": "Total PnL"}))
                else: 
                    st.info("No significant P&L contribution data to display in pie chart (all P&L values are zero, NaN, or non-numeric).")
            else: 
                st.warning("Total PnL (total_pnl) key not found for P&L contribution chart.")

            display_cols_for_summary_keys = ["Account", "total_pnl", "total_trades", "win_rate", "avg_trade_pnl", "max_drawdown_pct", "sharpe_ratio"]
            # Create a display df with labels for columns
            summary_display_table_df = pd.DataFrame()
            summary_display_table_df["Account"] = summary_metrics_df["Account"]

            for key in display_cols_for_summary_keys:
                if key == "Account": continue
                if key in summary_metrics_df.columns:
                    label = KPI_CONFIG.get(key, {}).get("label", key.replace("_", " ").title()) # Get label from KPI_CONFIG or generate
                    summary_display_table_df[label] = summary_metrics_df[key]
            
            # Apply formatting for display using KPI_CONFIG types
            for col_label in summary_display_table_df.columns:
                if col_label == "Account": continue
                # Find original key for this label to get its type from KPI_CONFIG
                original_key = next((k for k, v in KPI_CONFIG.items() if v.get("label") == col_label), None)
                if not original_key and col_label.lower().replace(" ", "_") in KPI_CONFIG: # Fallback if label was generated
                    original_key = col_label.lower().replace(" ", "_")

                if original_key and original_key in KPI_CONFIG:
                    item_format_type = KPI_CONFIG[original_key].get("type")
                    # Ensure column is numeric before formatting
                    summary_display_table_df[col_label] = pd.to_numeric(summary_display_table_df[col_label], errors='coerce')
                    if item_format_type == "currency": 
                        summary_display_table_df[col_label] = summary_display_table_df[col_label].apply(lambda x: format_currency(x) if pd.notna(x) else "N/A")
                    elif item_format_type == "percentage": 
                        # Assuming win_rate and max_drawdown_pct are already in % (0-100) from service
                        summary_display_table_df[col_label] = summary_display_table_df[col_label].apply(lambda x: format_percentage(x/100.0 if key not in ["win_rate", "max_drawdown_pct"] else x) if pd.notna(x) else "N/A")
                    elif item_format_type == "float": 
                        summary_display_table_df[col_label] = summary_display_table_df[col_label].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
                    elif item_format_type == "integer":
                         summary_display_table_df[col_label] = summary_display_table_df[col_label].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "N/A")


            st.dataframe(summary_display_table_df.set_index("Account"), use_container_width=True)
            if not summary_metrics_df.empty:
                with st.expander("View Raw Account Performance Data (Programmatic Keys, Pre-formatting)"):
                    st.dataframe(summary_metrics_df.drop(columns=["Total PnL Numeric"], errors='ignore'))
        else: 
            display_custom_message("Could not calculate performance metrics for any of the selected individual accounts in the given date range.", "warning")

    with tab_optimization:
        st.header(f"Portfolio Optimization ({start_date_selected.strftime('%Y-%m-%d')} to {end_date_selected.strftime('%Y-%m-%d')})")
        
        opt_selected_strategies = []
        opt_objective_key = "" 
        opt_objective_display = "" 
        opt_use_ledoit_wolf = True
        opt_target_return = None
        opt_lookback_days = 252 
        opt_num_frontier_points = 25

        with st.expander("⚙️ Configure Portfolio Optimization", expanded=True):
            if strategy_col_actual not in portfolio_df.columns or portfolio_df.empty:
                st.warning(f"Strategy column ('{strategy_col_actual}') not found or no data in the selected portfolio/date range. Cannot perform optimization.")
            else:
                optimizable_strategies_list = sorted(portfolio_df[strategy_col_actual].dropna().astype(str).unique())
                if not optimizable_strategies_list:
                     st.info("No distinct strategies available in the selected portfolio/date range for optimization.")
                else:
                    with st.form("portfolio_optimization_form_v9"): 
                        st.markdown("Select strategies, objective, and constraints for optimization using data from the globally selected date range.")
                        
                        opt_selected_strategies = st.multiselect(
                            "Select Strategies for Optimization:", options=optimizable_strategies_list,
                            default=optimizable_strategies_list[:min(len(optimizable_strategies_list), 5)], 
                            key="opt_strat_sel_v9"
                        )
                        
                        opt_objective_options_map = {
                            "Maximize Sharpe Ratio": "maximize_sharpe_ratio",
                            "Minimize Volatility": "minimize_volatility",
                            "Risk Parity": "risk_parity"
                        }
                        opt_objective_display = st.selectbox(
                            "Optimization Objective:", options=list(opt_objective_options_map.keys()),
                            index=0, key="opt_obj_v9"
                        )
                        opt_objective_key = opt_objective_options_map[opt_objective_display]

                        col_opt_params_1, col_opt_params_2 = st.columns(2)
                        with col_opt_params_1:
                            opt_use_ledoit_wolf = st.checkbox("Use Ledoit-Wolf Covariance", True, key="opt_lw_v9")
                        
                        with col_opt_params_2:
                            num_unique_days_in_portfolio_df = portfolio_df[date_col_actual].nunique()
                            max_allowable_lookback = max(20, num_unique_days_in_portfolio_df) 
                            default_opt_lookback = min(252, max_allowable_lookback) 

                            opt_lookback_days = st.number_input(
                                "Lookback Period (days for optimization):", 
                                min_value=20, max_value=max_allowable_lookback, 
                                value=default_opt_lookback, step=10, key="opt_lb_v9",
                                help=f"Data within the global date range will be used. Max lookback here is {max_allowable_lookback} unique trading days from the filtered data."
                            )

                        if opt_objective_key == "minimize_volatility":
                            opt_target_return = st.number_input(
                                "Target Annualized Return (e.g., 0.10 for 10%):", 
                                min_value=-1.0, max_value=2.0, value=0.10, step=0.01, format="%.2f", key="opt_tr_v9"
                            )
                        if opt_objective_key in ["maximize_sharpe_ratio", "minimize_volatility"]:
                             opt_num_frontier_points = st.number_input("Number of Frontier Points:", 10, 100, 25, 5, key="opt_fp_v9")
                        
                        st.markdown("---")
                        st.markdown("##### Per-Strategy Weight Constraints")

                        opt_preset_options = {
                            "Default (0-100%)": {"min": 0.0, "max": 100.0, "apply_all": True},
                            "Long Only (Max 30%)": {"min": 0.0, "max": 30.0, "apply_all": True},
                            "Diversified (Min 5%, Max 50%)": {"min": 5.0, "max": 50.0, "apply_all": True},
                            "Custom": None 
                        }
                        opt_selected_preset_label = st.selectbox(
                            "Constraint Presets:", options=list(opt_preset_options.keys()), index=0, key="opt_preset_v9"
                        )
                        opt_selected_preset = opt_preset_options[opt_selected_preset_label]

                        opt_current_weights_derived = {}
                        if opt_selected_strategies and not portfolio_df.empty:
                            temp_df_curr_weights = portfolio_df[portfolio_df[strategy_col_actual].isin(opt_selected_strategies)].copy()
                            if not temp_df_curr_weights.empty and date_col_actual in temp_df_curr_weights:
                                latest_date_for_cw = temp_df_curr_weights[date_col_actual].max()
                                earliest_date_for_cw_filtered = temp_df_curr_weights[date_col_actual].min()
                                start_date_for_cw_calc = max(earliest_date_for_cw_filtered, latest_date_for_cw - pd.Timedelta(days=opt_lookback_days -1))
                                temp_df_curr_weights = temp_df_curr_weights[temp_df_curr_weights[date_col_actual] >= start_date_for_cw_calc]

                            if not temp_df_curr_weights.empty:
                                pnl_sum_per_strat_cw = temp_df_curr_weights.groupby(strategy_col_actual)[pnl_col_actual].sum()
                                positive_pnl_sum_cw = pnl_sum_per_strat_cw[pnl_sum_per_strat_cw > 1e-9] 
                                
                                if not positive_pnl_sum_cw.empty:
                                    total_positive_pnl_cw = positive_pnl_sum_cw.sum()
                                    if abs(total_positive_pnl_cw) > 1e-9: 
                                        opt_current_weights_derived = (positive_pnl_sum_cw / total_positive_pnl_cw).to_dict()
                                
                                for strat_cw in opt_selected_strategies:
                                    if strat_cw not in opt_current_weights_derived:
                                        opt_current_weights_derived[strat_cw] = 0.0
                            
                            if not opt_current_weights_derived and opt_selected_strategies:
                                num_sel_cw = len(opt_selected_strategies)
                                equal_w_cw = 1.0 / num_sel_cw if num_sel_cw > 0 else 0.0
                                for strat_cw in opt_selected_strategies:
                                    opt_current_weights_derived[strat_cw] = equal_w_cw
                        
                        if opt_selected_strategies and opt_current_weights_derived:
                            st.markdown("###### Derived Current Weights (for Turnover Calculation - based on P&L over lookback)")
                            cw_display_df = pd.DataFrame.from_dict(opt_current_weights_derived, orient='index', columns=['Weight %'])
                            cw_display_df['Weight %'] = cw_display_df['Weight %'] * 100
                            st.dataframe(cw_display_df.style.format("{:.2f}%"), use_container_width=True)
                            sum_derived_cw_val = sum(opt_current_weights_derived.values())
                            if not (0.99 < sum_derived_cw_val < 1.01) and abs(sum_derived_cw_val) > 1e-6 :
                                 st.caption(f"Note: Sum of derived current weights is {sum_derived_cw_val*100:.1f}%. This can occur if P&L data is sparse, mostly negative, or zero for selected strategies over the lookback.")


                        opt_asset_bounds_list = [] 
                        opt_min_weight_inputs_dict = {}

                        if opt_selected_strategies:
                            st.markdown("###### Individual Strategy Constraints (Min/Max % of Portfolio)")
                            for i, strat_name_form in enumerate(opt_selected_strategies):
                                default_min_w, default_max_w = 0.0, 100.0 
                                if opt_selected_preset and opt_selected_preset.get("apply_all"):
                                    default_min_w = opt_selected_preset["min"]
                                    default_max_w = opt_selected_preset["max"]
                                
                                cols_constraints_form = st.columns([2,1,1]) 
                                with cols_constraints_form[0]:
                                    st.markdown(f"**{strat_name_form}**")
                                with cols_constraints_form[1]:
                                    min_w_input = st.number_input(f"Min W %", 0.0, 100.0, default_min_w, 0.1, key=f"min_w_{strat_name_form}_v9", label_visibility="collapsed")
                                with cols_constraints_form[2]:
                                    max_w_input = st.number_input(f"Max W %", 0.0, 100.0, default_max_w, 0.1, key=f"max_w_{strat_name_form}_v9", label_visibility="collapsed")
                                
                                if min_w_input > max_w_input: 
                                    st.warning(f"For {strat_name_form}: Min weight ({min_w_input}%) cannot exceed Max weight ({max_w_input}%). Adjusting Max to {min_w_input}%.", icon="⚠️")
                                    max_w_input = min_w_input 

                                opt_asset_bounds_list.append((min_w_input / 100.0, max_w_input / 100.0))
                                opt_min_weight_inputs_dict[strat_name_form] = min_w_input / 100.0
                        else:
                            st.caption("Select strategies above to configure their individual weight constraints.")

                        if opt_asset_bounds_list:
                            sum_min_weights_form_pct = sum(b[0] * 100 for b in opt_asset_bounds_list)
                            if sum_min_weights_form_pct > 100.0 + 1e-6: 
                                st.warning(f"Sum of Minimum Weight constraints ({sum_min_weights_form_pct:.1f}%) currently exceeds 100%. Optimization might be infeasible if not adjusted.", icon="🚨")
                        
                        submit_optimization_button = st.form_submit_button("🚀 Optimize Portfolio")

        if submit_optimization_button and opt_selected_strategies:
            min_strats_for_obj = 1 if opt_objective_key == "risk_parity" else 2 
            form_constraint_error = False
            if len(opt_selected_strategies) < min_strats_for_obj:
                display_custom_message(f"Please select at least {min_strats_for_obj} strategies for the '{opt_objective_display}' objective.", "warning")
                form_constraint_error = True
            
            sum_min_weights_final_val = sum(b[0] for b in opt_asset_bounds_list) if opt_asset_bounds_list else 0.0
            if sum_min_weights_final_val > 1.0 + 1e-6 : 
                display_custom_message(f"Error: Sum of minimum weight constraints ({sum_min_weights_final_val*100:.1f}%) exceeds 100%. Please adjust constraints.", "error")
                form_constraint_error = True
            
            for i_val, strat_name_val in enumerate(opt_selected_strategies): 
                 min_b_val, max_b_val = opt_asset_bounds_list[i_val]
                 if min_b_val > max_b_val:
                     display_custom_message(f"Error for strategy {strat_name_val}: Min weight ({min_b_val*100:.1f}%) cannot be greater than Max weight ({max_b_val*100:.1f}%).", "error")
                     form_constraint_error = True
                     break 
            
            if not form_constraint_error:
                with st.spinner("Optimizing portfolio... This may take a moment."):
                    portfolio_df_for_opt_tuple = (portfolio_df.to_records(index=False).tolist(), portfolio_df.columns.tolist())
                    
                    optimization_run_results = _run_portfolio_optimization_logic(
                        portfolio_df_data_tuple=portfolio_df_for_opt_tuple,
                        strategy_col_actual=strategy_col_actual,
                        date_col_actual=date_col_actual,
                        pnl_col_actual=pnl_col_actual,
                        selected_strategies_for_opt_tuple=tuple(opt_selected_strategies), 
                        lookback_days_opt=opt_lookback_days,
                        global_initial_capital=global_initial_capital,
                        optimization_objective_key=opt_objective_key,
                        risk_free_rate=risk_free_rate,
                        target_return_val=opt_target_return,
                        num_frontier_points=opt_num_frontier_points,
                        use_ledoit_wolf=opt_use_ledoit_wolf,
                        asset_bounds_list_of_tuples=opt_asset_bounds_list
                    )
                
                if optimization_run_results and 'error' not in optimization_run_results:
                    st.success(f"Portfolio Optimization ({opt_objective_display}) Completed Successfully!")
                    
                    st.subheader("Optimal Portfolio Weights")
                    final_optimal_weights = optimization_run_results.get('optimal_weights', {})
                    if final_optimal_weights:
                        optimal_weights_display_df = pd.DataFrame.from_dict(final_optimal_weights, orient='index', columns=['Weight'])
                        optimal_weights_display_df["Weight %"] = (optimal_weights_display_df["Weight"] * 100)
                        st.dataframe(optimal_weights_display_df[["Weight %"]].style.format("{:.2f}%"))
                        
                        fig_pie_optimal_alloc = px.pie(
                            optimal_weights_display_df[optimal_weights_display_df['Weight'] > 1e-5], 
                            values='Weight', names=optimal_weights_display_df[optimal_weights_display_df['Weight'] > 1e-5].index,
                            title=f'Optimal Allocation ({opt_objective_display})', hole=0.3
                        )
                        fig_pie_optimal_alloc.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(_apply_custom_theme(fig_pie_optimal_alloc, plot_theme), use_container_width=True)

                        if opt_current_weights_derived: 
                            portfolio_turnover_val = calculate_portfolio_turnover(opt_current_weights_derived, final_optimal_weights)
                            st.metric(label="Portfolio Turnover (vs. Derived Current Weights)", value=format_percentage(portfolio_turnover_val))
                        else:
                            st.caption("Derived current weights for turnover calculation were not available (e.g., no strategies selected or no P&L data in lookback).")
                        
                        with st.expander("View Optimal Weights Data (Numeric Values)"):
                            st.dataframe(optimal_weights_display_df)
                    else:
                        st.warning("Optimal weights could not be determined from the optimization results.")

                    st.subheader(f"Optimized Portfolio Performance (Annualized) - {opt_objective_display}")
                    optimized_kpis_data = optimization_run_results.get('performance', {})
                    if optimized_kpis_data:
                        # Use programmatic keys for optimized KPIs as well
                        opt_kpi_order_display = ["expected_annual_return", "annual_volatility", "sharpe_ratio"] 
                        kpis_to_show_opt = {
                            key: optimized_kpis_data.get(key)
                            for key in opt_kpi_order_display
                        }
                        if any(v is not None for v in kpis_to_show_opt.values()):
                            KPIClusterDisplay(
                                kpi_results=kpis_to_show_opt, # Pass the filtered dict
                                kpi_definitions=KPI_CONFIG, 
                                kpi_order=opt_kpi_order_display, # Pass the order list
                                cols_per_row=3
                            ).render()
                        else:
                             display_custom_message("Optimized performance KPIs are not available.", "warning")

                        with st.expander("View Full Optimized Performance Data (Raw)"):
                            st.dataframe(pd.DataFrame.from_dict(optimized_kpis_data, orient='index', columns=['Value']))
                    else:
                        st.warning("Optimized performance KPIs not found in results.")

                    if "risk_contributions" in optimization_run_results and optimization_run_results["risk_contributions"]:
                        st.subheader("Risk Contributions to Portfolio Variance")
                        risk_contrib_data = optimization_run_results["risk_contributions"]
                        if isinstance(risk_contrib_data, dict) and risk_contrib_data: 
                            risk_contrib_df = pd.DataFrame.from_dict(risk_contrib_data, orient='index', columns=['Risk Contribution %']).sort_values(by="Risk Contribution %", ascending=False)
                            risk_contrib_df["Risk Contribution %"] = pd.to_numeric(risk_contrib_df["Risk Contribution %"], errors='coerce').fillna(0)
                            
                            fig_risk_contrib_bar = px.bar(risk_contrib_df, x=risk_contrib_df.index, y="Risk Contribution %",
                                            title="Percentage Risk Contribution to Portfolio Variance",
                                            labels={"index": "Strategy", "Risk Contribution %": "Risk Contrib. (%)"},
                                            color="Risk Contribution %", color_continuous_scale=px.colors.sequential.Oranges_r)
                            fig_risk_contrib_bar.update_yaxes(ticksuffix="%")
                            st.plotly_chart(_apply_custom_theme(fig_risk_contrib_bar, plot_theme), use_container_width=True)
                            with st.expander("View Risk Contribution Data"): st.dataframe(risk_contrib_df)
                        elif not risk_contrib_data: 
                             st.info("Risk contribution data is available but empty.")
                        else: 
                             st.info("Risk contribution data is not in the expected dictionary format.")
                    
                    if opt_objective_key in ["maximize_sharpe_ratio", "minimize_volatility"]:
                        st.subheader("Efficient Frontier Visualization")
                        eff_frontier_data = optimization_run_results.get("efficient_frontier") 
                        if eff_frontier_data and isinstance(eff_frontier_data, dict) and \
                           'volatility' in eff_frontier_data and 'return' in eff_frontier_data and \
                           len(eff_frontier_data['volatility']) == len(eff_frontier_data['return']) and \
                           len(eff_frontier_data['volatility']) > 0:
                            
                            perf_data_for_points_plot = optimization_run_results.get('performance', {})
                            max_s_vol_plot, max_s_ret_plot, min_v_vol_plot, min_v_ret_plot = None, None, None, None

                            if opt_objective_key == "maximize_sharpe_ratio":
                                max_s_vol_plot = perf_data_for_points_plot.get('annual_volatility')
                                max_s_ret_plot = perf_data_for_points_plot.get('expected_annual_return')
                            if opt_objective_key == "minimize_volatility":
                                min_v_vol_plot = perf_data_for_points_plot.get('annual_volatility')
                                min_v_ret_plot = perf_data_for_points_plot.get('expected_annual_return')

                            temp_frontier_plot_df = pd.DataFrame(eff_frontier_data)
                            if not temp_frontier_plot_df.empty:
                                if min_v_vol_plot is None and 'volatility' in temp_frontier_plot_df: 
                                    min_vol_idx_plot = temp_frontier_plot_df['volatility'].idxmin()
                                    min_v_vol_plot = temp_frontier_plot_df.loc[min_vol_idx_plot, 'volatility']
                                    min_v_ret_plot = temp_frontier_plot_df.loc[min_vol_idx_plot, 'return']
                                if max_s_vol_plot is None and 'return' in temp_frontier_plot_df and 'volatility' in temp_frontier_plot_df: 
                                    temp_frontier_plot_df['sharpe_calc'] = (temp_frontier_plot_df['return'] - risk_free_rate) / temp_frontier_plot_df['volatility'].replace(0, np.nan)
                                    if not temp_frontier_plot_df['sharpe_calc'].isnull().all():
                                        max_sharpe_idx_plot = temp_frontier_plot_df['sharpe_calc'].idxmax()
                                        max_s_vol_plot = temp_frontier_plot_df.loc[max_sharpe_idx_plot, 'volatility']
                                        max_s_ret_plot = temp_frontier_plot_df.loc[max_sharpe_idx_plot, 'return']
                            
                            fig_eff_frontier = plot_efficient_frontier(
                                eff_frontier_data['volatility'], eff_frontier_data['return'],
                                max_s_vol_plot, max_s_ret_plot, min_v_vol_plot, min_v_ret_plot, theme=plot_theme
                            )
                            if fig_eff_frontier:
                                st.plotly_chart(fig_eff_frontier, use_container_width=True)
                                if not pd.DataFrame(eff_frontier_data).empty:
                                    with st.expander("View Efficient Frontier Data Points"):
                                        st.dataframe(pd.DataFrame(eff_frontier_data))
                            else: display_custom_message("Could not generate the Efficient Frontier plot.", "warning")
                        else:
                            display_custom_message("Efficient Frontier data is not available or is incomplete for the selected objective.", "info")
                elif optimization_run_results and 'error' in optimization_run_results: 
                    display_custom_message(f"Portfolio Optimization Error: {optimization_run_results.get('error')}", "error")
                else: 
                    display_custom_message("Portfolio optimization process failed to return valid results.", "error")

    with tab_comparison:
        st.header(f"Compare Equity Curves ({start_date_selected.strftime('%Y-%m-%d')} to {end_date_selected.strftime('%Y-%m-%d')})")
        if len(unique_accounts_in_range) < 2:
            st.info("At least two distinct accounts are needed in the selected date range for comparison.")
        else:
            col1_comp_select, col2_comp_select = st.columns(2)
            acc1_compare_selected = col1_comp_select.selectbox(
                "Select Account 1 for Comparison:", unique_accounts_in_range, index=0, key="acc_sel_1_comp_v6"
            )
            idx2_compare = 1 if len(unique_accounts_in_range) > 1 else 0
            acc2_compare_selected = col2_comp_select.selectbox(
                "Select Account 2 for Comparison:", unique_accounts_in_range, index=idx2_compare, key="acc_sel_2_comp_v6"
            )

            if acc1_compare_selected == acc2_compare_selected:
                st.warning("Please select two different accounts for a meaningful comparison.")
            else:
                st.subheader(f"Equity Curve Comparison: {acc1_compare_selected} vs. {acc2_compare_selected}")
                df_acc1_comp_raw = base_df_filtered_by_date[base_df_filtered_by_date[account_col_actual] == acc1_compare_selected]
                df_acc2_comp_raw = base_df_filtered_by_date[base_df_filtered_by_date[account_col_actual] == acc2_compare_selected]
                
                combined_equity_for_comparison_df = pd.DataFrame() 

                for df_raw_loop_comp, acc_name_loop_comp in [(df_acc1_comp_raw, acc1_compare_selected), (df_acc2_comp_raw, acc2_compare_selected)]:
                    if df_raw_loop_comp.empty: 
                        logger.info(f"No raw data for account {acc_name_loop_comp} in comparison tab for the selected date range.")
                        continue 
                    
                    df_cleaned_loop_comp = _clean_data_for_analysis(
                        df_raw_loop_comp, date_col=date_col_actual, pnl_col=pnl_col_actual, 
                        required_cols_to_check_na=[pnl_col_actual] 
                    )
                    if not df_cleaned_loop_comp.empty:
                        df_cleaned_loop_comp['cumulative_pnl'] = df_cleaned_loop_comp[pnl_col_actual].cumsum()
                        temp_equity_comp_df = df_cleaned_loop_comp[[date_col_actual, 'cumulative_pnl']].rename(
                            columns={'cumulative_pnl': f'Equity_{acc_name_loop_comp}'}
                        )
                        if combined_equity_for_comparison_df.empty:
                            combined_equity_for_comparison_df = temp_equity_comp_df
                        else:
                            combined_equity_for_comparison_df = pd.merge(combined_equity_for_comparison_df, temp_equity_comp_df, on=date_col_actual, how='outer')
                
                if combined_equity_for_comparison_df.empty or not any(f'Equity_{acc}' in combined_equity_for_comparison_df.columns for acc in [acc1_compare_selected, acc2_compare_selected]):
                    display_custom_message(f"Could not generate equity data for one or both selected accounts ({acc1_compare_selected}, {acc2_compare_selected}) in the chosen date range. They might lack valid P&L entries.", "warning")
                else:
                    combined_equity_for_comparison_df.sort_values(by=date_col_actual, inplace=True)
                    combined_equity_for_comparison_df = combined_equity_for_comparison_df.ffill().fillna(0) 
                    
                    fig_equity_compare_plot = go.Figure()
                    if f'Equity_{acc1_compare_selected}' in combined_equity_for_comparison_df:
                        fig_equity_compare_plot.add_trace(go.Scatter(
                            x=combined_equity_for_comparison_df[date_col_actual],
                            y=combined_equity_for_comparison_df[f'Equity_{acc1_compare_selected}'],
                            mode='lines', name=f"{acc1_compare_selected} Equity"
                        ))
                    if f'Equity_{acc2_compare_selected}' in combined_equity_for_comparison_df:
                         fig_equity_compare_plot.add_trace(go.Scatter(
                            x=combined_equity_for_comparison_df[date_col_actual],
                            y=combined_equity_for_comparison_df[f'Equity_{acc2_compare_selected}'],
                            mode='lines', name=f"{acc2_compare_selected} Equity"
                        ))
                    fig_equity_compare_plot.update_layout(
                        title=f"Equity Curve Comparison: {acc1_compare_selected} vs. {acc2_compare_selected}",
                        xaxis_title="Date", yaxis_title="Cumulative PnL", hovermode="x unified"
                    )
                    st.plotly_chart(_apply_custom_theme(fig_equity_compare_plot, plot_theme), use_container_width=True)
                    
                    if not combined_equity_for_comparison_df.empty:
                        with st.expander("View Combined Equity Comparison Data (Aligned & Forward-Filled)"):
                            st.dataframe(combined_equity_for_comparison_df)

if __name__ == "__main__":
    if not logging.getLogger(APP_TITLE if 'APP_TITLE' in globals() else "PortfolioAnalysisPage").hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if 'APP_TITLE' not in globals(): APP_TITLE = "PortfolioApp_Standalone_Test" 
    if 'EXPECTED_COLUMNS' not in globals():
        EXPECTED_COLUMNS = {
            'account_str': 'Account', 'pnl': 'PnL', 'date': 'Date', 'strategy': 'Strategy'    
        }
    if 'RISK_FREE_RATE' not in globals(): RISK_FREE_RATE = 0.01
    if 'KPI_CONFIG' not in globals(): 
        # Ensure KPI_CONFIG uses programmatic keys
        KPI_CONFIG = { 
            "total_pnl": {"label": "Total PnL", "type": "currency", "description": "Total Profit and Loss"},
            "total_trades": {"label": "Total Trades", "type": "integer", "description": "Total number of trades"},
            "win_rate": {"label": "Win Rate %", "type": "percentage", "description": "Percentage of winning trades"},
            "avg_trade_pnl": {"label": "Avg Trade PnL", "type": "currency", "description": "Average PnL per trade"},
            "sharpe_ratio": {"label": "Sharpe Ratio", "type": "float", "description": "Risk-adjusted return (annualized)"},
            "sortino_ratio": {"label": "Sortino Ratio", "type": "float", "description": "Downside risk-adjusted return (annualized)"},
            "calmar_ratio": {"label": "Calmar Ratio", "type": "float", "description": "Return over Max Drawdown (annualized)"},
            "max_drawdown_pct": {"label": "Max Drawdown %", "type": "percentage", "description": "Largest peak-to-trough decline in %"},
            "max_drawdown_abs": {"label": "Max Drawdown (Abs)", "type": "currency", "description": "Largest peak-to-trough decline in absolute terms"},
            "avg_daily_pnl": {"label": "Avg Daily PnL", "type": "currency", "description": "Average daily profit or loss"},
            "pnl_skewness": {"label": "PnL Skewness", "type": "float", "description": "Skewness of PnL distribution"},
            "pnl_kurtosis": {"label": "PnL Kurtosis", "type": "float", "description": "Kurtosis of PnL distribution"},
            "expected_annual_return": {"label": "Expected Annual Return", "type": "percentage", "description": "Annualized expected return from optimization"},
            "annual_volatility": {"label": "Annual Volatility", "type": "percentage", "description": "Annualized standard deviation of returns from optimization"},
        }
    if 'COLORS' not in globals(): COLORS = {"primary": "#007bff", "secondary": "#6c757d"} 

    if 'app_initialized' not in st.session_state: 
        date_rng_start = pd.to_datetime('2022-01-01')
        date_rng_end = pd.to_datetime('2023-12-31')
        trading_dates = pd.bdate_range(start=date_rng_start, end=date_rng_end) 
        
        num_entries_per_day_approx = 3
        total_mock_entries = len(trading_dates) * num_entries_per_day_approx
        
        mock_dates_list = np.random.choice(trading_dates, total_mock_entries, replace=True)
        mock_dates_list.sort()

        mock_strategies = ['MomentumStrat', 'MeanReversionStrat', 'ArbitrageStrat', 'TrendFollowStrat', 'ValueInvestStrat']
        mock_accounts = ['PrimaryAccount', 'HedgeFundAccount', 'AlphaCaptureAccount']
        
        np.random.seed(42) 
        mock_pnl_values = np.random.normal(loc=20, scale=300, size=total_mock_entries) + np.linspace(0, 100, total_mock_entries)

        sample_data_dict_mock = {
            EXPECTED_COLUMNS['date']: mock_dates_list,
            EXPECTED_COLUMNS['pnl']: mock_pnl_values,
            EXPECTED_COLUMNS['strategy']: np.random.choice(mock_strategies, total_mock_entries),
            EXPECTED_COLUMNS['account_str']: np.random.choice(mock_accounts, total_mock_entries)
        }
        st.session_state.processed_data = pd.DataFrame(sample_data_dict_mock)
        st.session_state.initial_capital = 500000.0 
        st.session_state.risk_free_rate = RISK_FREE_RATE 
        st.session_state.current_theme = 'dark' 
        
        st.session_state.user_column_mapping = { 
            key: EXPECTED_COLUMNS[key] for key in EXPECTED_COLUMNS
        }
        st.session_state.app_initialized = True 
        logger.info("Mock data and session state initialized for standalone run of Portfolio Analysis Page.")
        st.sidebar.success("Mock data loaded for testing Portfolio Analysis.")
    
    show_portfolio_analysis_page()
