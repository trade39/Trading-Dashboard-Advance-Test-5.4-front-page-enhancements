"""
pages/1_📈_Overview.py

This page provides a high-level overview of trading performance,
focusing on Key Performance Indicators (KPIs) and the equity curve.
It displays the strategy's equity curve with dynamic timeframe selection
and highlights the maximum drawdown period.
If a benchmark is selected, a separate chart for its equity curve is shown.
Benchmark context added to relevant KPIs. Date index handling improved.
Added Current Status Snapshot, Data Scope Indication, Last Updated Timestamp,
and Collapsible KPI groups.

Enhanced for a sleeker and more professional layout using containers and improved sectioning.
Fixed TypeError for KPIClusterDisplay instantiation.
Fixed TypeError for plot_equity_curve_and_drawdown instantiation (initial_capital).
Fixed TypeError for plot_equity_curve_and_drawdown instantiation (currency_symbol).
"""
import streamlit as st
import pandas as pd
import numpy as np
import logging
import plotly.graph_objects as go
import datetime

try:
    # Assuming these modules are in the PYTHONPATH or accessible
    from config import APP_TITLE, EXPECTED_COLUMNS, KPI_CONFIG, KPI_GROUPS_OVERVIEW, AVAILABLE_BENCHMARKS, PLOT_BENCHMARK_LINE_COLOR
    from components.kpi_display import KPIClusterDisplay
    from plotting import plot_equity_curve_and_drawdown, _apply_custom_theme
    from utils.common_utils import display_custom_message, format_currency
    from services.analysis_service import AnalysisService
except ImportError as e:
    st.error(f"Overview Page Error: Critical module import failed: {e}. Ensure app structure is correct and all dependencies are installed.")
    # Fallback definitions for critical variables to allow the page to render an error message
    APP_TITLE = "TradingDashboard_Error"
    logger = logging.getLogger(APP_TITLE) # Initialize logger here for fallback
    logger.error(f"CRITICAL IMPORT ERROR in Overview Page: {e}", exc_info=True)
    EXPECTED_COLUMNS = {"date": "date", "pnl": "pnl"}
    KPI_CONFIG = {}
    KPI_GROUPS_OVERVIEW = {}
    AVAILABLE_BENCHMARKS = {}
    PLOT_BENCHMARK_LINE_COLOR = "#800080" # Default purple for benchmark
    # Fallback for custom components/functions
    class KPIClusterDisplay:
        def __init__(self, **kwargs): pass
        def render(self): st.warning("KPI Display Component failed to load.")
    class AnalysisService:
        def get_core_kpis(self, *args, **kwargs): return {"error": "Analysis Service not loaded"}
    def plot_equity_curve_and_drawdown(**kwargs): return None 
    def _apply_custom_theme(fig, theme): return fig
    def display_custom_message(msg, type="error"): st.error(msg)
    def format_currency(val, currency_symbol="$", **kwargs): return f"{currency_symbol}{val:,.2f}" if pd.notna(val) else "N/A"
    st.stop() # Stop execution if critical imports fail

logger = logging.getLogger(APP_TITLE)
analysis_service_instance = AnalysisService()

def get_timeframe_filtered_df(df: pd.DataFrame, date_col: str, timeframe_option: str) -> pd.DataFrame:
    """Filters a DataFrame based on a selected timeframe relative to the max date in the data."""
    if df.empty or date_col not in df.columns:
        logger.warning("Timeframe filter: DataFrame is empty or date column missing.")
        return pd.DataFrame()

    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except Exception as e:
            logger.error(f"Timeframe filter: Could not convert date column '{date_col}' to datetime: {e}")
            return pd.DataFrame()

    max_date_in_data = df[date_col].max()
    if pd.isna(max_date_in_data):
        logger.warning("Timeframe filter: Max date in data is NaT. Returning original DataFrame.")
        return df 

    today = max_date_in_data

    if timeframe_option == "All Time":
        return df
    elif timeframe_option == "Last 30 Days":
        start_date = today - pd.Timedelta(days=29) 
    elif timeframe_option == "Last 90 Days":
        start_date = today - pd.Timedelta(days=89)
    elif timeframe_option == "Year to Date (YTD)":
        start_date = pd.Timestamp(year=today.year, month=1, day=1)
    elif timeframe_option == "Last 1 Year":
        start_date = today - pd.Timedelta(days=364) 
    else:
        logger.warning(f"Timeframe filter: Unknown timeframe option '{timeframe_option}'. Returning original DataFrame.")
        return df
    
    return df[df[date_col] >= start_date].copy()


def show_overview_page():
    """Renders the Performance Overview page."""
    st.title("📈 Performance Overview")

    if 'filtered_data' not in st.session_state or st.session_state.filtered_data is None or st.session_state.filtered_data.empty:
        display_custom_message(
            "No data loaded or processed. Please upload and process your trading data on the 'Data Upload' page.",
            "info"
        )
        return
    if 'kpi_results' not in st.session_state or st.session_state.kpi_results is None:
        display_custom_message("KPI results are not available. Please ensure data processing was successful.", "warning")
        return
    if 'error' in st.session_state.kpi_results:
        display_custom_message(f"KPI calculation error: {st.session_state.kpi_results['error']}", "error")
        return

    filtered_df_global = st.session_state.filtered_data
    kpi_results_global = st.session_state.kpi_results
    kpi_confidence_intervals = st.session_state.get('kpi_confidence_intervals', {})
    plot_theme = st.session_state.get('current_theme', 'dark') 
    benchmark_daily_returns = st.session_state.get('benchmark_daily_returns')
    selected_benchmark_display_name = st.session_state.get('selected_benchmark_display_name', "Benchmark")
    initial_capital = st.session_state.get('initial_capital', 100000.0) 
    currency_symbol = st.session_state.get("currency_symbol", "$")


    date_col = EXPECTED_COLUMNS.get('date', 'date')
    pnl_col = EXPECTED_COLUMNS.get('pnl', 'pnl')
    cum_pnl_col = 'cumulative_pnl' 
    drawdown_pct_col_name = 'drawdown_pct' 
    max_dd_details_from_state = st.session_state.get('max_drawdown_period_details')

    if filtered_df_global.empty:
        display_custom_message("No data available after applying global filters. Please adjust filters or upload new data.", "info")
        return

    if date_col in filtered_df_global.columns and not filtered_df_global[date_col].dropna().empty:
        try:
            last_trade_date = pd.to_datetime(filtered_df_global[date_col]).max()
            st.caption(f"Analysis based on data up to: **{last_trade_date.strftime('%B %d, %Y %H:%M:%S')}**")
        except Exception as e:
            logger.error(f"Error formatting last trade date: {e}")
            st.caption("Could not determine the last trade date.")
    else:
        st.caption("Date column not found or empty; cannot determine the last trade date.")

    st.markdown("---") 

    with st.container(border=True):
        st.subheader("📊 Current Status Snapshot")
        
        snap_col1, snap_col2, snap_col3 = st.columns(3)
        latest_equity, current_dd, last_day_pnl, last_trading_day_str = np.nan, np.nan, np.nan, "N/A"

        if cum_pnl_col in filtered_df_global.columns and not filtered_df_global[cum_pnl_col].empty:
            latest_equity = filtered_df_global[cum_pnl_col].iloc[-1] + initial_capital 
        
        if drawdown_pct_col_name in filtered_df_global.columns and not filtered_df_global[drawdown_pct_col_name].empty:
            current_dd_raw = filtered_df_global[drawdown_pct_col_name].iloc[-1]
            current_dd = max(0, current_dd_raw) if pd.notna(current_dd_raw) else np.nan


        if date_col in filtered_df_global.columns and pnl_col in filtered_df_global.columns and not filtered_df_global.empty:
            df_temp_dates = filtered_df_global.copy()
            if not pd.api.types.is_datetime64_any_dtype(df_temp_dates[date_col]):
                df_temp_dates[date_col] = pd.to_datetime(df_temp_dates[date_col], errors='coerce')
            
            last_trading_day_ts = df_temp_dates[date_col].max()
            if pd.notna(last_trading_day_ts):
                last_trading_day_str = last_trading_day_ts.normalize().strftime('%Y-%m-%d')
                last_day_pnl_val = df_temp_dates[df_temp_dates[date_col].dt.normalize() == last_trading_day_ts.normalize()][pnl_col].sum()
                if pd.notna(last_day_pnl_val):
                    last_day_pnl = last_day_pnl_val
        
        with snap_col1:
            st.metric(
                "Latest Portfolio Equity",
                format_currency(latest_equity, currency_symbol=currency_symbol),
                help="Total value of the portfolio including initial capital and cumulative P&L."
            )
        with snap_col2:
            st.metric(
                "Current Drawdown",
                f"{current_dd:.2f}%" if pd.notna(current_dd) else "N/A",
                help="Percentage decline from the last peak equity. 'N/A' if not applicable or no drawdown."
            )
        with snap_col3:
            st.metric(
                f"Last Day P&L ({last_trading_day_str})",
                format_currency(last_day_pnl, currency_symbol=currency_symbol),
                help=f"Net profit or loss on {last_trading_day_str}."
            )
        st.markdown("<br>", unsafe_allow_html=True) 

    st.markdown("---")

    st.header("🚀 Key Performance Indicators")
    if date_col in filtered_df_global.columns and not filtered_df_global[date_col].dropna().empty:
        try:
            min_date_global = pd.to_datetime(filtered_df_global[date_col]).min()
            max_date_global = pd.to_datetime(filtered_df_global[date_col]).max()
            st.caption(f"Global KPIs based on filtered data from **{min_date_global.strftime('%b %d, %Y')}** to **{max_date_global.strftime('%b %d, %Y')}**.")
        except Exception as e:
            logger.error(f"Error formatting KPI date range caption: {e}")
            st.caption("Could not determine date range for global KPIs.")
    
    cols_per_row_setting = st.session_state.get('kpi_cols_per_row', 4) 

    for group_name, kpi_keys_in_group in KPI_GROUPS_OVERVIEW.items():
        group_kpi_results = {key: kpi_results_global[key] for key in kpi_keys_in_group if key in kpi_results_global}
        
        default_expanded = True 
        if group_name == "Benchmark Comparison":
            is_benchmark_active = benchmark_daily_returns is not None and not benchmark_daily_returns.empty
            has_benchmark_kpis = any(pd.notna(group_kpi_results.get(key)) for key in ["alpha", "beta", "benchmark_sharpe_ratio"]) 
            default_expanded = is_benchmark_active and has_benchmark_kpis
            if not default_expanded and (not group_kpi_results or all(pd.isna(val) for val in group_kpi_results.values())):
                continue
        
        if not group_kpi_results or all(pd.isna(val) for val in group_kpi_results.values()):
            if group_name != "Benchmark Comparison":
                logger.info(f"KPI group '{group_name}' has all NaN values and will be skipped.")
                continue

        with st.expander(group_name, expanded=default_expanded):
            KPIClusterDisplay( 
                kpi_results=group_kpi_results,
                kpi_definitions=KPI_CONFIG,
                kpi_order=kpi_keys_in_group,
                kpi_confidence_intervals=kpi_confidence_intervals,
                cols_per_row=cols_per_row_setting,
                benchmark_context_name=(selected_benchmark_display_name if group_name == "Benchmark Comparison" and selected_benchmark_display_name not in [None, "None", ""] else None)
            ).render()
    st.markdown("---")
    
    st.header("📈 Strategy Performance Charts")
    
    df_for_plot_base_global_scope = pd.DataFrame() 
    df_for_plot_time_filtered = pd.DataFrame() 

    with st.container(border=True):
        st.subheader("Equity Curve & Drawdown Analysis")
        timeframe_options = ["All Time", "Last 1 Year", "Year to Date (YTD)", "Last 90 Days", "Last 30 Days"]
        selected_timeframe = st.radio(
            "Select Timeframe for Equity Curve:",
            options=timeframe_options,
            index=0, 
            horizontal=True,
            key="overview_equity_timeframe_radio"
        )

        try:
            if date_col not in filtered_df_global.columns:
                display_custom_message(f"Date column ('{date_col}') not found in the data. Cannot plot equity curve.", "error")
            else:
                df_for_plot_base_global_scope = filtered_df_global.copy()
                if not pd.api.types.is_datetime64_any_dtype(df_for_plot_base_global_scope[date_col]):
                    df_for_plot_base_global_scope[date_col] = pd.to_datetime(df_for_plot_base_global_scope[date_col], errors='coerce')
                
                df_for_plot_base_global_scope.dropna(subset=[date_col], inplace=True) 

                if df_for_plot_base_global_scope.empty:
                    display_custom_message("No valid date entries found after processing. Cannot plot equity curve.", "error")
                else:
                    df_for_plot_base_global_scope = df_for_plot_base_global_scope.sort_values(by=date_col)
                    
                    if cum_pnl_col not in df_for_plot_base_global_scope.columns:
                        if pnl_col in df_for_plot_base_global_scope.columns and pd.api.types.is_numeric_dtype(df_for_plot_base_global_scope[pnl_col]):
                            df_for_plot_base_global_scope[cum_pnl_col] = df_for_plot_base_global_scope[pnl_col].cumsum()
                        else:
                            display_custom_message(f"PnL column ('{pnl_col}') is missing or not numeric. Cannot calculate cumulative PnL.", "error")
                            df_for_plot_base_global_scope = pd.DataFrame() 
                    
                    if not df_for_plot_base_global_scope.empty and drawdown_pct_col_name not in df_for_plot_base_global_scope.columns and cum_pnl_col in df_for_plot_base_global_scope.columns:
                        equity_with_initial = df_for_plot_base_global_scope[cum_pnl_col] + initial_capital
                        hwm_equity = equity_with_initial.cummax()
                        dd_abs_equity = hwm_equity - equity_with_initial
                        df_for_plot_base_global_scope[drawdown_pct_col_name] = (dd_abs_equity / hwm_equity.replace(0, np.nan)).fillna(0) * 100

                    if not df_for_plot_base_global_scope.empty:
                         df_for_plot_time_filtered = get_timeframe_filtered_df(df_for_plot_base_global_scope, date_col, selected_timeframe)
                    
                    max_dd_peak_plot, max_dd_trough_plot, max_dd_end_plot = None, None, None
                    if not df_for_plot_time_filtered.empty and selected_timeframe == "All Time" and max_dd_details_from_state:
                        max_dd_peak_plot = max_dd_details_from_state.get('Peak Date')
                        max_dd_trough_plot = max_dd_details_from_state.get('Trough Date')
                        max_dd_end_plot = max_dd_details_from_state.get('End Date')
                        logger.info(f"Using global max DD details for 'All Time' plot: P:{max_dd_peak_plot}, T:{max_dd_trough_plot}, E:{max_dd_end_plot}")
                    elif not df_for_plot_time_filtered.empty and selected_timeframe != "All Time":
                        logger.info(f"Max DD period highlight is based on global data, shown when 'All Time' is selected.")
                        pass
                    
                    if df_for_plot_time_filtered.empty:
                        if not df_for_plot_base_global_scope.empty: 
                            display_custom_message(f"No data available for the selected timeframe: '{selected_timeframe}'.", "info")
                    else:
                        equity_fig = plot_equity_curve_and_drawdown(
                            df=df_for_plot_time_filtered,
                            date_col=date_col,
                            cumulative_pnl_col=cum_pnl_col,
                            drawdown_pct_col=drawdown_pct_col_name if drawdown_pct_col_name in df_for_plot_time_filtered else None,
                            theme=plot_theme,
                            max_dd_peak_date=max_dd_peak_plot if selected_timeframe == "All Time" else None, 
                            max_dd_trough_date=max_dd_trough_plot if selected_timeframe == "All Time" else None,
                            max_dd_recovery_date=max_dd_end_plot if selected_timeframe == "All Time" else None
                            # currency_symbol=currency_symbol # Removed this line
                        )
                        if equity_fig:
                            st.plotly_chart(equity_fig, use_container_width=True)
                        else:
                            display_custom_message(f"Could not generate the equity curve chart for '{selected_timeframe}'.", "info")
        except Exception as e:
            logger.error(f"Error displaying equity curve and drawdown chart: {e}", exc_info=True)
            display_custom_message(f"An unexpected error occurred while generating the equity curve: {e}", "error")
        st.markdown("<br>", unsafe_allow_html=True)


    if not df_for_plot_time_filtered.empty and selected_timeframe != "All Time":
        st.markdown("---") 
        with st.container(border=True):
            st.subheader(f"Key Metrics for Selected Timeframe: {selected_timeframe}")
            with st.spinner(f"Calculating metrics for {selected_timeframe}..."):
                cols_for_timeframe_kpi = [date_col, pnl_col]
                if drawdown_pct_col_name in df_for_plot_time_filtered.columns:
                    cols_for_timeframe_kpi.append(drawdown_pct_col_name)
                
                kpis_timeframe_df = df_for_plot_time_filtered[cols_for_timeframe_kpi].copy()
                
                timeframe_kpis_results = analysis_service_instance.get_core_kpis(
                    kpis_timeframe_df,
                    st.session_state.get('risk_free_rate', 0.0), 
                    None, 
                    st.session_state.get('initial_capital', 100000.0) 
                )

                if timeframe_kpis_results and 'error' not in timeframe_kpis_results:
                    timeframe_kpi_keys_to_show = [
                        "total_pnl", "win_rate", "avg_trade_pnl", 
                        "max_drawdown_pct", "total_trades", "sharpe_ratio", "sortino_ratio"
                    ]
                    focused_kpis = {
                        key: timeframe_kpis_results[key] for key in timeframe_kpi_keys_to_show if key in timeframe_kpis_results
                    }
                    if focused_kpis and not all(pd.isna(v) for v in focused_kpis.values()):
                        KPIClusterDisplay( 
                            kpi_results=focused_kpis,
                            kpi_definitions=KPI_CONFIG, 
                            kpi_order=timeframe_kpi_keys_to_show,
                            cols_per_row=3
                        ).render()
                    else:
                        display_custom_message(f"No focused KPI data available for the timeframe '{selected_timeframe}'.", "info")
                else:
                    error_msg = timeframe_kpis_results.get('error', 'Unknown error') if timeframe_kpis_results else 'Calculation failed'
                    display_custom_message(f"Could not calculate KPIs for timeframe '{selected_timeframe}': {error_msg}", "warning")
            st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("---")

    benchmark_plot_equity = pd.Series(dtype=float) 
    if benchmark_daily_returns is not None and not benchmark_daily_returns.empty:
        bm_series = benchmark_daily_returns.squeeze() if isinstance(benchmark_daily_returns, pd.DataFrame) else benchmark_daily_returns
        if isinstance(bm_series, pd.Series) and not bm_series.empty:
            bm_series_for_plot = bm_series.copy()
            if pd.isna(bm_series_for_plot.iloc[0]):
                bm_series_for_plot.iloc[0] = 0.0
            
            benchmark_cumulative_growth_factor = (1 + bm_series_for_plot).cumprod()
            if not benchmark_cumulative_growth_factor.empty:
                benchmark_plot_equity = benchmark_cumulative_growth_factor * initial_capital
        else:
            logger.warning("Benchmark daily returns are not a valid Series or are empty after processing.")

    if not benchmark_plot_equity.empty:
        with st.container(border=True):
            st.subheader(f"📊 Benchmark Performance: {selected_benchmark_display_name}")
            
            fig_benchmark_only = go.Figure()
            fig_benchmark_only.add_trace(go.Scatter(
                x=benchmark_plot_equity.index,
                y=benchmark_plot_equity,
                mode='lines',
                name=f"{selected_benchmark_display_name} (Equity)",
                line=dict(color=PLOT_BENCHMARK_LINE_COLOR, width=2)
            ))
            
            fig_benchmark_only.update_layout(
                title_text=f"{selected_benchmark_display_name} Equity Curve (Scaled to Initial Capital)",
                xaxis_title="Date",
                yaxis_title=f"Benchmark Value ({currency_symbol})",
                hovermode="x unified"
            )
            st.plotly_chart(_apply_custom_theme(fig_benchmark_only, plot_theme), use_container_width=True)
            st.markdown("<br>", unsafe_allow_html=True)
    elif st.session_state.get('selected_benchmark_ticker') and st.session_state.get('selected_benchmark_ticker') != "":
        with st.container(border=True):
            st.subheader(f"Benchmark Performance: {selected_benchmark_display_name}")
            display_custom_message(f"Equity curve data is not available for the selected benchmark: '{selected_benchmark_display_name}'. Ensure benchmark data was loaded correctly.", "info")
            st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("---")


if __name__ == "__main__":
    if 'app_initialized' not in st.session_state: 
        st.warning("This page is part of a multi-page Streamlit application. Please run the main application script (e.g., app.py) to view this page correctly with all initializations.")
        st.info("If you are developing this page, ensure necessary session state variables (like 'filtered_data', 'kpi_results', 'initial_capital', 'current_theme') are mocked or set for testing.")
    else:
        show_overview_page()
