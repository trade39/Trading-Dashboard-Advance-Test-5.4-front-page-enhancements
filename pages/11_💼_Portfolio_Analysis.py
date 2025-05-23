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
    # Attempt to use Streamlit's error display if available
    if 'st' in globals() and hasattr(st, 'error'):
        st.error(critical_error_message)
    # Fallback logging if Streamlit isn't fully loaded
    try:
        # Use a generic logger name if APP_TITLE isn't available due to import failure
        page_logger_name = "PortfolioAnalysisPage_ImportErrorLogger"
        if 'APP_TITLE' in globals(): # Check if APP_TITLE was imported before the error
            page_logger_name = f"{APP_TITLE}.PortfolioAnalysisPage.ImportError"

        page_logger = logging.getLogger(page_logger_name)
        page_logger.error(f"CRITICAL IMPORT ERROR in Portfolio Analysis Page: {e}", exc_info=True)
    except Exception as log_e:
        print(f"Fallback logging error during Portfolio Analysis Page import: {log_e}")

    if 'st' in globals() and hasattr(st, 'stop'):
        st.stop()
    else:
        # If st.stop() is not available, re-raise to halt execution
        raise ImportError(critical_error_message) from e

logger = logging.getLogger(APP_TITLE) # Main logger for the page
general_analysis_service = AnalysisService()
portfolio_specific_service = PortfolioAnalysisService()


def _clean_data_for_analysis(
    df: pd.DataFrame,
    date_col: str,
    pnl_col: Optional[str] = None,
    strategy_col: Optional[str] = None,
    account_col: Optional[str] = None,
    required_cols_to_check_na: Optional[List[str]] = None, # Specific columns to check for NaNs beyond date/pnl
    numeric_cols_to_convert: Optional[List[str]] = None,
    string_cols_to_convert: Optional[List[str]] = None,
    sort_by_date: bool = True
) -> pd.DataFrame:
    """
    Cleans and prepares a DataFrame for analysis.
    - Converts date column to datetime.
    - Converts specified columns to numeric or string.
    - Drops NaNs in essential columns (date, pnl if provided, and other required_cols).
    - Sorts by date if specified.
    """
    if df.empty:
        logger.info("Input DataFrame for cleaning is empty.")
        return pd.DataFrame()

    df_cleaned = df.copy()

    # Convert date column
    if date_col not in df_cleaned.columns:
        logger.warning(f"Date column '{date_col}' not found in DataFrame for cleaning.")
        return pd.DataFrame() # Essential column missing
    try:
        df_cleaned[date_col] = pd.to_datetime(df_cleaned[date_col], errors='coerce')
    except Exception as e:
        logger.error(f"Error converting date column '{date_col}' to datetime: {e}", exc_info=True)
        df_cleaned[date_col] = pd.NaT # Mark as NaT for later dropna

    # Convert PnL column if specified and exists
    if pnl_col and pnl_col in df_cleaned.columns:
        try:
            df_cleaned[pnl_col] = pd.to_numeric(df_cleaned[pnl_col], errors='coerce')
        except Exception as e:
            logger.error(f"Error converting PnL column '{pnl_col}' to numeric: {e}", exc_info=True)
            df_cleaned[pnl_col] = np.nan # Mark as NaN

    # Convert other specified numeric columns
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

    # Convert specified string columns
    if string_cols_to_convert:
        for col in string_cols_to_convert:
            if col in df_cleaned.columns:
                try:
                    df_cleaned[col] = df_cleaned[col].astype(str)
                except Exception as e:
                    logger.error(f"Error converting column '{col}' to string: {e}", exc_info=True)
            else:
                logger.debug(f"String column '{col}' for conversion not found in DataFrame.")

    # Determine columns to check for NaNs before dropping
    cols_for_nan_check = [date_col]
    if pnl_col and pnl_col in df_cleaned.columns:
        cols_for_nan_check.append(pnl_col)
    if strategy_col and strategy_col in df_cleaned.columns: # For strategy correlation
        cols_for_nan_check.append(strategy_col)
    if account_col and account_col in df_cleaned.columns: # For account correlation
        cols_for_nan_check.append(account_col)

    if required_cols_to_check_na: # For general use cases
        for rc in required_cols_to_check_na:
            if rc in df_cleaned.columns and rc not in cols_for_nan_check:
                cols_for_nan_check.append(rc)
    
    # Ensure all columns in cols_for_nan_check actually exist in df_cleaned before passing to dropna
    valid_cols_for_nan_check = [col for col in cols_for_nan_check if col in df_cleaned.columns]
    if not valid_cols_for_nan_check: # Should at least have date_col
        logger.warning("No valid columns identified for NaN checking after initial processing.")
    else:
        df_cleaned.dropna(subset=valid_cols_for_nan_check, inplace=True)


    if df_cleaned.empty:
        logger.info(f"DataFrame became empty after cleaning and NaN drop for columns: {valid_cols_for_nan_check}.")
        return pd.DataFrame()

    if sort_by_date and date_col in df_cleaned.columns:
        df_cleaned.sort_values(by=date_col, inplace=True)

    return df_cleaned


def _calculate_drawdown_series_for_aggregated_df(cumulative_pnl_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Helper to calculate absolute and percentage drawdown series for an aggregated PnL series.
    """
    if cumulative_pnl_series.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    high_water_mark = cumulative_pnl_series.cummax()
    drawdown_abs_series = high_water_mark - cumulative_pnl_series
    
    # Vectorized approach for drawdown_pct_series
    drawdown_pct_series = pd.Series(np.where(high_water_mark > 1e-9, (drawdown_abs_series / high_water_mark) * 100.0, 
                                             np.where(drawdown_abs_series > 1e-9, 100.0, 0.0)),
                                    index=cumulative_pnl_series.index, dtype=float)
            
    return drawdown_abs_series.fillna(0), drawdown_pct_series.fillna(0)


@st.cache_data
def calculate_metrics_for_df(
    df_input: pd.DataFrame, # Changed from tuple to direct DataFrame
    pnl_col: str,
    date_col: str,
    risk_free_rate: float,
    initial_capital: float
) -> Dict[str, Any]:
    """Calculates core metrics for a given DataFrame of trades/pnl."""
    if df_input.empty:
        logger.info("calculate_metrics_for_df received an empty DataFrame.")
        return {
            "Total PnL": 0.0, "Total Trades": 0, "Win Rate %": 0.0,
            "Avg Trade PnL": 0.0, "Max Drawdown %": 0.0, "Sharpe Ratio": 0.0,
            "error": "Input DataFrame is empty."
        }

    # Minimal cleaning, assuming df_input is somewhat pre-processed or raw trade data
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

    df_copy.sort_values(by=date_col, inplace=True) # Ensure chronological order

    # 'win' and 'cumulative_pnl' are typically calculated by general_analysis_service.get_core_kpis
    # If not, they might need to be prepared here if the service expects them.
    # For simplicity, assuming the service handles their absence or recalculates them.

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

@st.cache_data
def _run_portfolio_optimization_logic(
    portfolio_df_data_tuple: Tuple[List[tuple], List[str]], # (portfolio_df.to_records(index=False).tolist(), portfolio_df.columns.tolist())
    strategy_col_actual: str,
    date_col_actual: str,
    pnl_col_actual: str,
    selected_strategies_for_opt_tuple: Tuple[str, ...],
    lookback_days_opt: int,
    global_initial_capital: float,
    optimization_objective_key: str, # e.g., "maximize_sharpe_ratio"
    risk_free_rate: float,
    target_return_val: Optional[float],
    num_frontier_points: int,
    use_ledoit_wolf: bool,
    asset_bounds_list_of_tuples: Optional[List[Tuple[float, float]]] # Pass as list of tuples
) -> Dict[str, Any]:
    """
    Encapsulates the data preparation and optimization call.
    Uses global portfolio_specific_service.
    """
    portfolio_df_data, portfolio_df_columns = portfolio_df_data_tuple
    portfolio_df = pd.DataFrame(data=portfolio_df_data, columns=portfolio_df_columns)
    selected_strategies_for_opt = list(selected_strategies_for_opt_tuple)

    opt_df_filtered_strategies = portfolio_df[portfolio_df[strategy_col_actual].isin(selected_strategies_for_opt)].copy()

    # Clean data specifically for optimization needs
    opt_df_filtered_strategies = _clean_data_for_analysis(
        opt_df_filtered_strategies,
        date_col=date_col_actual,
        pnl_col=pnl_col_actual,
        strategy_col=strategy_col_actual, # Ensure strategy column is clean for pivot
        required_cols_to_check_na=[pnl_col_actual, strategy_col_actual],
        sort_by_date=True
    )

    if opt_df_filtered_strategies.empty:
        return {"error": "No data available for selected strategies after initial filtering and cleaning for optimization."}

    latest_date_in_data = opt_df_filtered_strategies[date_col_actual].max()
    start_date_lookback = latest_date_in_data - pd.Timedelta(days=lookback_days_opt - 1) # Inclusive of start day
    opt_df_lookback = opt_df_filtered_strategies[opt_df_filtered_strategies[date_col_actual] >= start_date_lookback]

    if opt_df_lookback.empty:
        return {"error": "No data available for the selected strategies within the lookback period."}

    try:
        daily_pnl_pivot = opt_df_lookback.groupby(
            [opt_df_lookback[date_col_actual].dt.normalize(), strategy_col_actual]
        )[pnl_col_actual].sum().unstack(fill_value=0)
        daily_pnl_pivot = daily_pnl_pivot.reindex(columns=selected_strategies_for_opt, fill_value=0.0)
    except (KeyError, ValueError, TypeError) as e: # More specific pandas errors
        logger.error(f"Error during P&L pivot for optimization: {e}", exc_info=True)
        return {"error": f"Failed to pivot P&L data: {e}"}
    except Exception as e: # Fallback for other unexpected errors
        logger.error(f"Unexpected error during P&L pivot for optimization: {e}", exc_info=True)
        return {"error": f"Unexpected error pivoting P&L data: {e}"}


    if global_initial_capital <= 0:
        return {"error": "Initial capital must be positive to calculate returns for optimization."}

    daily_returns_for_opt = daily_pnl_pivot / global_initial_capital
    daily_returns_for_opt = daily_returns_for_opt.fillna(0)

    # Determine min_hist_points based on objective, not its string representation
    min_hist_points_needed = 20
    if optimization_objective_key == "risk_parity" and len(selected_strategies_for_opt) <= 1 :
         min_hist_points_needed = 2 # Risk parity on single asset is trivial but allow if logic permits

    if daily_returns_for_opt.empty or daily_returns_for_opt.shape[0] < min_hist_points_needed:
        return {"error": f"Not enough historical daily return data points ({daily_returns_for_opt.shape[0]}) for reliable optimization. Need at least {min_hist_points_needed}."}

    try:
        opt_results = portfolio_specific_service.prepare_and_run_optimization(
            daily_returns_df=daily_returns_for_opt,
            objective=optimization_objective_key, # Use the key directly
            risk_free_rate=risk_free_rate,
            target_return_level=target_return_val,
            trading_days=252,
            num_frontier_points=num_frontier_points,
            use_ledoit_wolf=use_ledoit_wolf,
            asset_bounds=asset_bounds_list_of_tuples # Already list of tuples
        )
    except Exception as e: # Catch errors from the service call
        logger.error(f"Error during portfolio_specific_service.prepare_and_run_optimization: {e}", exc_info=True)
        return {"error": f"Optimization service failed: {e}"}
        
    return opt_results


def show_portfolio_analysis_page():
    st.title("💼 Portfolio-Level Analysis")
    logger.info("Rendering Portfolio Analysis Page.")

    if 'processed_data' not in st.session_state or st.session_state.processed_data is None or st.session_state.processed_data.empty:
        display_custom_message("Please upload and process data to view portfolio analysis.", "info")
        logger.info("Portfolio analysis page: No processed data found in session state.")
        return

    base_df = st.session_state.processed_data
    plot_theme = st.session_state.get('current_theme', 'dark')
    risk_free_rate = st.session_state.get('risk_free_rate', RISK_FREE_RATE)
    global_initial_capital = st.session_state.get('initial_capital', 100000.0)

    account_col_conceptual = 'account_str' # This should map to the actual column name in EXPECTED_COLUMNS
    account_col_actual = EXPECTED_COLUMNS.get(account_col_conceptual)
    pnl_col_actual = EXPECTED_COLUMNS.get('pnl')
    date_col_actual = EXPECTED_COLUMNS.get('date')
    strategy_col_actual = EXPECTED_COLUMNS.get('strategy')

    if not all([account_col_actual, pnl_col_actual, date_col_actual, strategy_col_actual]):
        missing_configs = [col_type for col_type, col_val in zip(
            ['account', 'pnl', 'date', 'strategy'],
            [account_col_actual, pnl_col_actual, date_col_actual, strategy_col_actual]) if not col_val
        ]
        msg = f"Essential column configurations missing ({', '.join(missing_configs)}) from EXPECTED_COLUMNS. Portfolio analysis cannot proceed."
        display_custom_message(msg, "error")
        logger.error(f"Portfolio analysis page: {msg}")
        return

    essential_cols_in_df = [account_col_actual, pnl_col_actual, date_col_actual, strategy_col_actual]
    if not all(col in base_df.columns for col in essential_cols_in_df):
        missing_cols_in_df = [col for col in essential_cols_in_df if col not in base_df.columns]
        msg = f"Essential columns ({', '.join(missing_cols_in_df)}) not found in the uploaded data. Portfolio analysis requires these columns."
        display_custom_message(msg, "error")
        logger.error(f"Portfolio analysis page: {msg}. Available columns: {base_df.columns.tolist()}")
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

    portfolio_df_uncleaned = base_df[base_df[account_col_actual].isin(selected_accounts_for_portfolio)].copy()
    if portfolio_df_uncleaned.empty:
        display_custom_message("No data for the selected accounts in the portfolio view.", "info")
        return
    
    # Clean the main portfolio_df once for general use
    portfolio_df = _clean_data_for_analysis(
        portfolio_df_uncleaned,
        date_col=date_col_actual,
        pnl_col=pnl_col_actual,
        strategy_col=strategy_col_actual, # ensure strategy is cleaned for later use
        account_col=account_col_actual,   # ensure account is cleaned
        required_cols_to_check_na=[pnl_col_actual, strategy_col_actual, account_col_actual],
        string_cols_to_convert=[strategy_col_actual, account_col_actual] # Ensure these are strings for grouping/pivoting
    )
    if portfolio_df.empty:
        display_custom_message("No valid data after cleaning for the selected accounts in the portfolio view.", "warning")
        return

    # --- Overall Performance Section ---
    st.markdown("<div class='performance-section-container'>", unsafe_allow_html=True)
    st.header(f"📈 Overall Performance for Selected Portfolio ({', '.join(selected_accounts_for_portfolio)})")
    
    portfolio_daily_trades_df = pd.DataFrame()
    if portfolio_df.empty: # Should have been caught above, but as a safeguard
        display_custom_message("No valid P&L or date data after cleaning for selected portfolio.", "warning")
    else:
        try:
            portfolio_daily_pnl = portfolio_df.groupby(portfolio_df[date_col_actual].dt.normalize())[pnl_col_actual].sum()
        except (AttributeError, KeyError, TypeError) as e: # dt might fail if date_col_actual is not datetime after cleaning (shouldn't happen)
            logger.error(f"Error grouping by date for daily P&L: {e}", exc_info=True)
            display_custom_message(f"Error processing daily P&L: {e}", "error")
            portfolio_daily_pnl = pd.Series(dtype=float) # Empty series

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
            else: # Should not happen if portfolio_daily_pnl was not empty
                portfolio_daily_trades_df['drawdown_abs'] = pd.Series(dtype=float)
                portfolio_daily_trades_df['drawdown_pct'] = pd.Series(dtype=float)
                logger.warning("Could not calculate drawdown for aggregated portfolio as cumulative_pnl was missing or empty.")

            with st.spinner("Calculating selected portfolio KPIs..."):
                # Pass the daily aggregated df for portfolio-level KPIs
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
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")

    # --- Inter-Connections Section ---
    st.markdown("<div class='performance-section-container'>", unsafe_allow_html=True)
    st.header(f"🔗 Inter-Connections (Selected Portfolio: {', '.join(selected_accounts_for_portfolio)})")

    st.subheader("🔀 Inter-Strategy P&L Correlation")
    if strategy_col_actual not in portfolio_df.columns:
        display_custom_message(f"Strategy column '{strategy_col_actual}' not found in selected portfolio data.", "warning")
    else:
        unique_strategies_selected_portfolio = portfolio_df[strategy_col_actual].dropna().unique() # Already string from _clean_data
        if len(unique_strategies_selected_portfolio) < 2:
            st.info("At least two distinct strategies are needed within the selected portfolio for inter-strategy correlations.")
        else:
            # portfolio_df is already cleaned. We can use it directly or a relevant subset.
            # For correlation, we need PnL per strategy per day.
            df_strat_corr_prep = portfolio_df[[date_col_actual, strategy_col_actual, pnl_col_actual]].copy()
            # _clean_data_for_analysis was already applied to portfolio_df.
            # No need to re-clean unless specific transformations for correlation are needed.

            if df_strat_corr_prep.empty: # Should not happen if portfolio_df was not empty
                 correlation_results_strat = {"error": "No valid data for strategy correlation after initial cleaning."}
            else:
                # Service expects sorted data
                df_strat_corr_prep = df_strat_corr_prep.sort_values(by=[date_col_actual, strategy_col_actual]).reset_index(drop=True)
                with st.spinner("Calculating inter-strategy P&L correlations..."):
                    try:
                        correlation_results_strat = portfolio_specific_service.get_portfolio_inter_strategy_correlation(
                            df_strat_corr_prep, strategy_col_actual, pnl_col_actual, date_col_actual)
                    except Exception as e_strat_corr_service: # Catch errors from service
                        logger.error(f"Error calling inter-strategy correlation service: {e_strat_corr_service}", exc_info=True)
                        correlation_results_strat = {"error": f"Correlation service failed: {e_strat_corr_service}"}

            if correlation_results_strat and 'error' not in correlation_results_strat:
                matrix_df_strat_corr = correlation_results_strat.get('correlation_matrix')
                if matrix_df_strat_corr is not None and not matrix_df_strat_corr.empty and matrix_df_strat_corr.shape[0] > 1:
                    fig_strat_corr = go.Figure(data=go.Heatmap(z=matrix_df_strat_corr.values, x=matrix_df_strat_corr.columns, y=matrix_df_strat_corr.index, colorscale='RdBu', zmin=-1, zmax=1, text=matrix_df_strat_corr.round(2).astype(str), texttemplate="%{text}", hoverongaps=False))
                    fig_strat_corr.update_layout(title="Inter-Strategy Daily P&L Correlation")
                    st.plotly_chart(_apply_custom_theme(fig_strat_corr, plot_theme), use_container_width=True)
                    with st.expander("View Inter-Strategy Correlation Matrix"):
                        st.dataframe(matrix_df_strat_corr)
                else: display_custom_message("Not enough data or distinct strategies for inter-strategy correlation matrix.", "info")
            elif correlation_results_strat: display_custom_message(f"Inter-strategy correlation error: {correlation_results_strat.get('error')}", "error")
            else: display_custom_message("Inter-strategy correlation analysis failed to return results.", "error")

    st.subheader("🤝 Inter-Account P&L Correlation")
    if len(selected_accounts_for_portfolio) < 2:
        st.info("At least two accounts must be selected in the sidebar for inter-account correlation.")
    else:
        df_acc_corr_prep = portfolio_df[[date_col_actual, account_col_actual, pnl_col_actual]].copy()
        # portfolio_df is already cleaned.

        if df_acc_corr_prep.empty:
            correlation_results_acc = {"error": "No valid data for account correlation after initial cleaning."}
        else:
            df_acc_corr_prep = df_acc_corr_prep.sort_values(by=[date_col_actual, account_col_actual]).reset_index(drop=True)
            with st.spinner("Calculating inter-account P&L correlations..."):
                try:
                    correlation_results_acc = portfolio_specific_service.get_portfolio_inter_account_correlation(
                        df_acc_corr_prep, account_col_actual, pnl_col_actual, date_col_actual)
                except Exception as e_acc_corr_service:
                    logger.error(f"Error calling inter-account correlation service: {e_acc_corr_service}", exc_info=True)
                    correlation_results_acc = {"error": f"Correlation service failed: {e_acc_corr_service}"}

        if correlation_results_acc and 'error' not in correlation_results_acc:
            matrix_df_acc_corr = correlation_results_acc.get('correlation_matrix')
            if matrix_df_acc_corr is not None and not matrix_df_acc_corr.empty and matrix_df_acc_corr.shape[0] > 1:
                fig_acc_corr = go.Figure(data=go.Heatmap(z=matrix_df_acc_corr.values, x=matrix_df_acc_corr.columns, y=matrix_df_acc_corr.index, colorscale='RdBu', zmin=-1, zmax=1, text=matrix_df_acc_corr.round(2).astype(str), texttemplate="%{text}", hoverongaps=False))
                fig_acc_corr.update_layout(title="Inter-Account Daily P&L Correlation")
                st.plotly_chart(_apply_custom_theme(fig_acc_corr, plot_theme), use_container_width=True)
                with st.expander("View Inter-Account Correlation Matrix"):
                    st.dataframe(matrix_df_acc_corr)
            else: display_custom_message("Not enough data or distinct accounts for inter-account correlation matrix.", "info")
        elif correlation_results_acc: display_custom_message(f"Inter-account correlation error: {correlation_results_acc.get('error')}", "error")
        else: display_custom_message("Inter-account correlation analysis failed to return results.", "error")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")

    # --- Account Performance Breakdown Section ---
    st.markdown("<div class='performance-section-container'>", unsafe_allow_html=True)
    st.header(f"📊 Account Performance Breakdown (within Selected Portfolio: {', '.join(selected_accounts_for_portfolio)})")

    account_metrics_data = []
    # Use the original base_df for individual account metrics as it contains all trades, not daily aggregations.
    for acc_name_loop in selected_accounts_for_portfolio:
        acc_df_original_trades = base_df[base_df[account_col_actual] == acc_name_loop].copy() # Use uncleaned base_df slice
        if not acc_df_original_trades.empty:
            # calculate_metrics_for_df handles its own cleaning of the slice
            metrics = calculate_metrics_for_df(acc_df_original_trades, pnl_col_actual, date_col_actual, risk_free_rate, global_initial_capital)
            account_metrics_data.append({"Account": acc_name_loop, **metrics})
        else:
            logger.info(f"No original trade data for account {acc_name_loop} in breakdown.")


    if account_metrics_data:
        summary_table_df = pd.DataFrame(account_metrics_data)

        # For Pie Chart (using raw numeric P&L from summary_table_df)
        # Ensure "Total PnL" is numeric before filtering for the chart
        if "Total PnL" in summary_table_df.columns:
            summary_table_df["Total PnL"] = pd.to_numeric(summary_table_df["Total PnL"], errors='coerce')
            pnl_for_chart_df = summary_table_df[
                summary_table_df["Total PnL"].notna() & (summary_table_df["Total PnL"] != 0)
            ][["Account", "Total PnL"]].copy()

            if not pnl_for_chart_df.empty:
                fig_pnl_contrib = px.pie(pnl_for_chart_df, names='Account', values='Total PnL', title='P&L Contribution by Account (Selected Portfolio)', hole=0.3)
                fig_pnl_contrib.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(_apply_custom_theme(fig_pnl_contrib, plot_theme), use_container_width=True)
                with st.expander("View P&L Contribution Data (Numeric)"):
                    st.dataframe(pnl_for_chart_df)
            else:
                st.info("No P&L contribution data to display (all selected accounts have zero, NaN, or non-numeric P&L).")
        else:
            st.warning("Total PnL column not found for P&L contribution chart.")


        # For Display Table (formatting applied here)
        display_cols_summary = ["Account", "Total PnL", "Total Trades", "Win Rate %", "Avg Trade PnL", "Max Drawdown %", "Sharpe Ratio"]
        # Ensure all display_cols_summary exist in summary_table_df before selection
        valid_display_cols = [col for col in display_cols_summary if col in summary_table_df.columns]
        summary_table_df_display = summary_table_df[valid_display_cols].copy()


        if "Total PnL" in summary_table_df_display.columns: # Already numeric, format for display
            summary_table_df_display["Total PnL"] = summary_table_df_display["Total PnL"].apply(lambda x: format_currency(x) if pd.notna(x) else "N/A")
        if "Avg Trade PnL" in summary_table_df_display.columns:
            summary_table_df_display["Avg Trade PnL"] = pd.to_numeric(summary_table_df_display["Avg Trade PnL"], errors='coerce').apply(lambda x: format_currency(x) if pd.notna(x) else "N/A")
        if "Win Rate %" in summary_table_df_display.columns:
            summary_table_df_display["Win Rate %"] = pd.to_numeric(summary_table_df_display["Win Rate %"], errors='coerce').apply(lambda x: format_percentage(x/100.0) if pd.notna(x) else "N/A")
        if "Max Drawdown %" in summary_table_df_display.columns:
            summary_table_df_display["Max Drawdown %"] = pd.to_numeric(summary_table_df_display["Max Drawdown %"], errors='coerce').apply(lambda x: format_percentage(x/100.0) if pd.notna(x) else "N/A")
        if "Sharpe Ratio" in summary_table_df_display.columns:
            summary_table_df_display["Sharpe Ratio"] = pd.to_numeric(summary_table_df_display["Sharpe Ratio"], errors='coerce').apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")

        st.dataframe(summary_table_df_display.set_index("Account"), use_container_width=True)
        if not summary_table_df.empty: # Show raw (mostly numeric) data
            with st.expander("View Raw Account Performance Data (Pre-formatting)"):
                st.dataframe(summary_table_df)
    else:
        display_custom_message("Could not calculate performance metrics for individual accounts.", "warning")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")

    # --- Portfolio Optimization Section ---
    st.markdown("<div class='performance-section-container'>", unsafe_allow_html=True)
    st.header("⚖️ Portfolio Optimization")

    selected_strategies_for_opt_form = []
    optimization_objective_form = ""
    use_ledoit_wolf_covariance_form = True
    target_return_input_form = None
    lookback_days_opt_form = 252
    num_frontier_points_input_form = 25
    asset_bounds_input_form = []
    current_weights_input_for_turnover_form = {}
    submit_optimization_button = False

    with st.expander("⚙️ Configure Portfolio Optimization", expanded=True):
        if strategy_col_actual not in portfolio_df.columns: # Use the cleaned portfolio_df
            st.warning(f"Strategy column ('{strategy_col_actual}') not found in the selected portfolio data. Cannot perform optimization.")
        else:
            optimizable_strategies = sorted(portfolio_df[strategy_col_actual].dropna().unique()) # Already string

            if not optimizable_strategies:
                 st.info("No strategies available in the selected portfolio for optimization.")
            else:
                with st.form("portfolio_optimization_form_v5"): # Incremented form key
                    st.markdown("""
                    Select strategies from the current portfolio to include in the optimization.
                    Optimization uses historical daily returns derived from P&L within the **globally filtered date range** of the selected portfolio.
                    """)
                    selected_strategies_for_opt_form = st.multiselect(
                        "Select Strategies for Optimization:",
                        options=optimizable_strategies,
                        default=optimizable_strategies[:min(len(optimizable_strategies), 5)],
                        key="opt_strategies_select_v5"
                    )

                    optimization_objective_options_map = {
                        "Maximize Sharpe Ratio": "maximize_sharpe_ratio",
                        "Minimize Volatility": "minimize_volatility",
                        "Risk Parity": "risk_parity"
                    }
                    optimization_objective_display_form = st.selectbox(
                        "Optimization Objective:",
                        options=list(optimization_objective_options_map.keys()),
                        index=0,
                        key="opt_objective_v5"
                    )
                    optimization_objective_form = optimization_objective_options_map[optimization_objective_display_form] # Store key

                    use_ledoit_wolf_covariance_form = st.checkbox(
                        "Use Ledoit-Wolf Covariance Shrinkage", value=True, key="opt_ledoit_wolf_v5",
                        help="Shrinks the sample covariance matrix, often improving stability."
                    )

                    if optimization_objective_form == "minimize_volatility":
                        target_return_input_form = st.number_input(
                            "Target Annualized Return (e.g., 0.10 for 10%):",
                            min_value=-1.0, max_value=2.0, value=0.10, step=0.01, format="%.2f",
                            key="opt_target_return_v5",
                            help="Specify desired annualized portfolio return if minimizing volatility."
                        )

                    max_lookback = max(20, len(portfolio_df[date_col_actual].unique())) if date_col_actual in portfolio_df and not portfolio_df.empty else 20
                    lookback_days_opt_form = st.number_input(
                        "Historical Lookback for Returns/Covariance (days):",
                        min_value=20, max_value=max_lookback,
                        value=min(252, max_lookback), step=10, key="opt_lookback_days_v5",
                        help="Number of recent trading days for calculating expected returns and covariance."
                    )

                    if optimization_objective_form in ["maximize_sharpe_ratio", "minimize_volatility"]:
                        num_frontier_points_input_form = st.number_input(
                            "Number of Points for Efficient Frontier Plot:",
                            min_value=10, max_value=100, value=25, step=5, key="opt_frontier_points_v5",
                            help="More points provide a smoother frontier but take longer to compute."
                        )

                    st.markdown("##### Per-Strategy Weight Constraints (Min/Max %)")
                    asset_bounds_input_form = []
                    current_weights_input_for_turnover_form = {}

                    if selected_strategies_for_opt_form:
                        st.markdown("###### Define Current and Min/Max Allocation % for Each Selected Strategy:")
                        num_sel_opt_strats = len(selected_strategies_for_opt_form)
                        def_curr_w_pct = 100.0 / num_sel_opt_strats if num_sel_opt_strats > 0 else 0.0

                        for strat_name_opt in selected_strategies_for_opt_form:
                            cols_turnover = st.columns(3)
                            curr_w = cols_turnover[0].number_input(f"Current W % ({strat_name_opt})", 0.0, 100.0, def_curr_w_pct, 1.0, "%.1f", key=f"curr_w_{strat_name_opt}_v5")
                            min_w = cols_turnover[1].number_input(f"Min W % ({strat_name_opt})", 0.0, 100.0, 0.0, 1.0, "%.1f", key=f"min_w_{strat_name_opt}_v5")
                            max_w = cols_turnover[2].number_input(f"Max W % ({strat_name_opt})", 0.0, 100.0, 100.0, 1.0, "%.1f", key=f"max_w_{strat_name_opt}_v5")
                            if min_w > max_w: max_w = min_w # Basic validation
                            asset_bounds_input_form.append((min_w / 100.0, max_w / 100.0))
                            current_weights_input_for_turnover_form[strat_name_opt] = curr_w / 100.0
                    else:
                        st.caption("Select strategies above to set weights.")
                    submit_optimization_button = st.form_submit_button("Optimize Portfolio")

    if submit_optimization_button and selected_strategies_for_opt_form:
        min_strategies_needed = 1 if optimization_objective_form == "risk_parity" else 2
        sum_current_weights = sum(current_weights_input_for_turnover_form.values())
        if not (0.999 < sum_current_weights < 1.001) and sum_current_weights != 0.0 and current_weights_input_for_turnover_form:
            display_custom_message(f"Sum of 'Current Weight %' ({sum_current_weights*100:.1f}%) should be close to 100% or 0% for turnover. Please adjust.", "warning")

        if len(selected_strategies_for_opt_form) < min_strategies_needed:
            display_custom_message(f"Please select at least {min_strategies_needed} strategies for '{optimization_objective_display_form}'.", "warning")
        elif asset_bounds_input_form and sum(b[0] for b in asset_bounds_input_form) > 1.0 + 1e-6 :
            display_custom_message(f"Sum of minimum weight constraints ({sum(b[0]*100 for b in asset_bounds_input_form):.1f}%) exceeds 100%. Please adjust.", "error")
        else:
            with st.spinner("Preparing data and optimizing portfolio..."):
                # Prepare inputs for cached function
                # Use the cleaned portfolio_df
                portfolio_df_tuple_for_opt = (portfolio_df.to_records(index=False).tolist(), portfolio_df.columns.tolist())
                selected_strategies_tuple_for_opt = tuple(selected_strategies_for_opt_form)
                
                opt_results = _run_portfolio_optimization_logic(
                    portfolio_df_data_tuple=portfolio_df_tuple_for_opt,
                    strategy_col_actual=strategy_col_actual,
                    date_col_actual=date_col_actual,
                    pnl_col_actual=pnl_col_actual,
                    selected_strategies_for_opt_tuple=selected_strategies_tuple_for_opt,
                    lookback_days_opt=lookback_days_opt_form,
                    global_initial_capital=global_initial_capital,
                    optimization_objective_key=optimization_objective_form, # Pass the key
                    risk_free_rate=risk_free_rate,
                    target_return_val=target_return_input_form,
                    num_frontier_points=num_frontier_points_input_form,
                    use_ledoit_wolf=use_ledoit_wolf_covariance_form,
                    asset_bounds_list_of_tuples=asset_bounds_input_form # Already list of tuples
                )

            if opt_results and 'error' not in opt_results:
                st.success(f"Portfolio Optimization ({optimization_objective_display_form}) Complete!")
                st.subheader("⚖️ Optimal Portfolio Weights")
                optimal_weights_dict = opt_results.get('optimal_weights', {})
                if not optimal_weights_dict:
                     st.warning("Optimal weights not found in optimization results.")
                else:
                    weights_df = pd.DataFrame.from_dict(optimal_weights_dict, orient='index', columns=['Weight'])
                    weights_df.index.name = "Strategy"
                    weights_df["Weight %"] = (weights_df["Weight"] * 100)
                    st.dataframe(weights_df[["Weight %"]].style.format("{:.2f}%", subset=["Weight %"]))
                    with st.expander("View Optimal Weights Data (Numeric)"):
                        st.dataframe(weights_df)

                    fig_weights_pie = px.pie(
                        weights_df[weights_df['Weight'] > 1e-5], # Filter small weights for cleaner pie
                        values='Weight', names=weights_df[weights_df['Weight'] > 1e-5].index,
                        title=f'Optimal Allocation ({optimization_objective_display_form})', hole=0.3
                    )
                    fig_weights_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(_apply_custom_theme(fig_weights_pie, plot_theme), use_container_width=True)

                    if current_weights_input_for_turnover_form:
                        turnover = calculate_portfolio_turnover(current_weights_input_for_turnover_form, optimal_weights_dict)
                        st.metric(label="Portfolio Turnover", value=format_percentage(turnover))
                    else: st.caption("Current weights not provided; turnover not calculated.")

                st.subheader(f"🚀 Optimized Portfolio Performance (Annualized) - {optimization_objective_display_form}")
                optimized_kpis_data = opt_results.get('performance', {})
                if not optimized_kpis_data:
                    st.warning("Optimized performance KPIs not found in results.")
                else:
                    st.markdown("<div class='kpi-metrics-block'>", unsafe_allow_html=True)
                    optimized_kpi_order = ["expected_annual_return", "annual_volatility", "sharpe_ratio"]
                    KPIClusterDisplay(optimized_kpis_data, KPI_CONFIG, optimized_kpi_order, cols_per_row=3).render()
                    st.markdown("</div>", unsafe_allow_html=True)
                    with st.expander("View Optimized Performance Data"):
                        st.dataframe(pd.DataFrame.from_dict(optimized_kpis_data, orient='index', columns=['Value']))

                if "risk_contributions" in opt_results and opt_results["risk_contributions"]:
                    st.subheader("🛡️ Risk Contributions to Portfolio Variance")
                    rc_data = opt_results["risk_contributions"]
                    if not rc_data:
                        st.info("Risk contribution data is empty.")
                    else:
                        rc_df = pd.DataFrame.from_dict(rc_data, orient='index', columns=['Risk Contribution %'])
                        rc_df.index.name = "Strategy"
                        rc_df_sorted = rc_df.sort_values(by="Risk Contribution %", ascending=False)
                        fig_rc_bar = px.bar(rc_df_sorted, x=rc_df_sorted.index, y="Risk Contribution %",
                                            title="Percentage Risk Contribution by Strategy",
                                            labels={"Risk Contribution %": "Risk Contribution (%)"},
                                            color="Risk Contribution %", color_continuous_scale=px.colors.sequential.Oranges_r)
                        fig_rc_bar.update_yaxes(ticksuffix="%")
                        st.plotly_chart(_apply_custom_theme(fig_rc_bar, plot_theme), use_container_width=True)
                        with st.expander("View Risk Contribution Data"):
                            st.dataframe(rc_df_sorted)
                
                if optimization_objective_form in ["maximize_sharpe_ratio", "minimize_volatility"]:
                    st.subheader("🎯 Efficient Frontier")
                    frontier_data = opt_results.get("efficient_frontier")
                    if frontier_data and frontier_data.get('volatility') and frontier_data.get('return'):
                        # ... (Plotting logic for efficient frontier - largely unchanged but ensure it uses opt_results correctly)
                        # This part needs careful checking of keys from opt_results['performance']
                        max_sharpe_vol_plot, max_sharpe_ret_plot = None, None
                        min_vol_portfolio_vol, min_vol_portfolio_ret = None, None

                        perf_data = opt_results.get('performance', {})
                        
                        # Max Sharpe point (either directly from Maximize Sharpe or calculated for Min Vol)
                        if optimization_objective_form == "maximize_sharpe_ratio":
                            max_sharpe_vol_plot = perf_data.get('annual_volatility')
                            max_sharpe_ret_plot = perf_data.get('expected_annual_return')
                        else: # For Minimize Volatility, find Max Sharpe on the frontier
                            temp_frontier_df = pd.DataFrame(frontier_data)
                            if not temp_frontier_df.empty and 'volatility' in temp_frontier_df and temp_frontier_df['volatility'].gt(1e-9).any():
                                temp_frontier_df['sharpe'] = (temp_frontier_df['return'] - risk_free_rate) / temp_frontier_df['volatility'].replace(0, np.nan)
                                if not temp_frontier_df['sharpe'].empty and not temp_frontier_df['sharpe'].isnull().all():
                                    max_s_idx = temp_frontier_df['sharpe'].idxmax()
                                    max_sharpe_vol_plot = temp_frontier_df.loc[max_s_idx, 'volatility']
                                    max_sharpe_ret_plot = temp_frontier_df.loc[max_s_idx, 'return']
                        
                        # Min Volatility point (always present on the frontier)
                        temp_frontier_df_for_min_vol = pd.DataFrame(frontier_data)
                        if not temp_frontier_df_for_min_vol.empty and 'volatility' in temp_frontier_df_for_min_vol:
                            min_vol_idx = temp_frontier_df_for_min_vol['volatility'].idxmin()
                            min_vol_portfolio_vol = temp_frontier_df_for_min_vol.loc[min_vol_idx, 'volatility']
                            min_vol_portfolio_ret = temp_frontier_df_for_min_vol.loc[min_vol_idx, 'return']

                        frontier_fig = plot_efficient_frontier(
                            frontier_vols=frontier_data['volatility'], frontier_returns=frontier_data['return'],
                            max_sharpe_vol=max_sharpe_vol_plot, max_sharpe_ret=max_sharpe_ret_plot,
                            min_vol_vol=min_vol_portfolio_vol, min_vol_ret=min_vol_portfolio_ret,
                            theme=plot_theme
                        )
                        if frontier_fig:
                            st.plotly_chart(frontier_fig, use_container_width=True)
                            frontier_df_display = pd.DataFrame(frontier_data)
                            if not frontier_df_display.empty:
                                with st.expander("View Efficient Frontier Data"):
                                    st.dataframe(frontier_df_display)
                        else: display_custom_message("Could not generate the Efficient Frontier plot.", "warning")
                    else: display_custom_message("Efficient Frontier data not available or incomplete for plotting.", "info")
            elif opt_results: # Error occurred
                display_custom_message(f"Optimization Error: {opt_results.get('error')}", "error")
            else: # opt_results is None or empty
                display_custom_message("Portfolio optimization failed to return results.", "error")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")

    # --- Compare Equity Curves Section ---
    st.markdown("<div class='performance-section-container'>", unsafe_allow_html=True)
    st.header("↔️ Compare Equity Curves of Any Two Accounts")
    if len(unique_accounts_all) < 2:
        st.info("At least two distinct accounts are needed for this comparison.")
    else:
        col1_comp, col2_comp = st.columns(2)
        with col1_comp: selected_account_1_comp = st.selectbox("Select Account 1:", unique_accounts_all, index=0, key="portfolio_acc_sel_1_comp_v2")
        with col2_comp: default_idx_acc2 = 1 if len(unique_accounts_all) > 1 else 0; selected_account_2_comp = st.selectbox("Select Account 2:", unique_accounts_all, index=default_idx_acc2, key="portfolio_acc_sel_2_comp_v2")

        if selected_account_1_comp == selected_account_2_comp:
            st.warning("Please select two different accounts.")
        else:
            st.subheader(f"🆚 Equity Curve Comparison: {selected_account_1_comp} vs. {selected_account_2_comp}")
            # Use base_df for raw trade data for each account
            df_acc1_raw = base_df[base_df[account_col_actual] == selected_account_1_comp]
            df_acc2_raw = base_df[base_df[account_col_actual] == selected_account_2_comp]

            combined_equity_comp_df = pd.DataFrame()
            dfs_to_compare = [(df_acc1_raw, selected_account_1_comp), (df_acc2_raw, selected_account_2_comp)]

            for df_raw, acc_name_comp in dfs_to_compare:
                if df_raw.empty:
                    logger.info(f"No data for account {acc_name_comp} in comparison.")
                    continue
                
                # Clean individual account data for comparison
                df_comp_loop_cleaned = _clean_data_for_analysis(
                    df_raw, date_col=date_col_actual, pnl_col=pnl_col_actual,
                    required_cols_to_check_na=[pnl_col_actual]
                )

                if not df_comp_loop_cleaned.empty:
                    df_comp_loop_cleaned['cumulative_pnl'] = df_comp_loop_cleaned[pnl_col_actual].cumsum()
                    temp_df = df_comp_loop_cleaned[[date_col_actual, 'cumulative_pnl']].copy()
                    temp_df.rename(columns={'cumulative_pnl': f'Equity_{acc_name_comp}'}, inplace=True)

                    if combined_equity_comp_df.empty:
                        combined_equity_comp_df = temp_df
                    else:
                        combined_equity_comp_df = pd.merge(combined_equity_comp_df, temp_df, on=date_col_actual, how='outer')
                else:
                    logger.warning(f"Account {acc_name_comp} had no valid data after cleaning for equity comparison.")


            if combined_equity_comp_df.empty or not any(f'Equity_{acc}' in combined_equity_comp_df.columns for acc in [selected_account_1_comp, selected_account_2_comp]):
                display_custom_message(f"One or both accounts ('{selected_account_1_comp}', '{selected_account_2_comp}') lack valid P&L data for comparison after cleaning.", "warning")
            else:
                combined_equity_comp_df.sort_values(by=date_col_actual, inplace=True)
                combined_equity_comp_df = combined_equity_comp_df.fillna(method='ffill').fillna(0) # Fill NaNs then remaining NaNs (if at start) with 0

                fig_comp_equity = go.Figure()
                if f'Equity_{selected_account_1_comp}' in combined_equity_comp_df.columns:
                    fig_comp_equity.add_trace(go.Scatter(x=combined_equity_comp_df[date_col_actual], y=combined_equity_comp_df[f'Equity_{selected_account_1_comp}'], mode='lines', name=f"{selected_account_1_comp} Equity"))
                if f'Equity_{selected_account_2_comp}' in combined_equity_comp_df.columns:
                    fig_comp_equity.add_trace(go.Scatter(x=combined_equity_comp_df[date_col_actual], y=combined_equity_comp_df[f'Equity_{selected_account_2_comp}'], mode='lines', name=f"{selected_account_2_comp} Equity"))

                fig_comp_equity.update_layout(title=f"Equity Comparison: {selected_account_1_comp} vs. {selected_account_2_comp}", xaxis_title="Date", yaxis_title="Cumulative PnL", hovermode="x unified")
                st.plotly_chart(_apply_custom_theme(fig_comp_equity, plot_theme), use_container_width=True)

                if not combined_equity_comp_df.empty:
                    with st.expander("View Combined Equity Comparison Data"):
                        st.dataframe(combined_equity_comp_df)
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    # This basic check helps if running the page script directly,
    # though full app functionality relies on `app.py` and session state setup.
    st.set_page_config(layout="wide", page_title="Portfolio Analysis") # Basic config
    if 'app_initialized' not in st.session_state:
        # Mock essential session state for standalone testing if needed
        # st.session_state.processed_data = pd.DataFrame(...) # Load sample data
        # st.session_state.initial_capital = 100000
        # st.session_state.risk_free_rate = 0.01
        # st.session_state.current_theme = 'dark'
        st.warning("This page is part of a multi-page app. For full functionality, run the main `app.py` script. Some features might be limited or unavailable.")
    show_portfolio_analysis_page()
