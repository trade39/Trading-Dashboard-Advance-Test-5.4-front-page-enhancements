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
"""
import streamlit as st
import pandas as pd
import numpy as np
import logging
import plotly.graph_objects as go
import datetime

try:
    from config import APP_TITLE, EXPECTED_COLUMNS, KPI_CONFIG, KPI_GROUPS_OVERVIEW, AVAILABLE_BENCHMARKS, PLOT_BENCHMARK_LINE_COLOR
    from components.kpi_display import KPIClusterDisplay
    from plotting import plot_equity_curve_and_drawdown, _apply_custom_theme
    from utils.common_utils import display_custom_message, format_currency
    from services.analysis_service import AnalysisService
except ImportError as e:
    st.error(f"Overview Page Error: Critical module import failed: {e}. Ensure app structure is correct.")
    APP_TITLE = "TradingDashboard_Error"; logger = logging.getLogger(APP_TITLE)
    logger.error(f"CRITICAL IMPORT ERROR in Overview Page: {e}", exc_info=True)
    EXPECTED_COLUMNS = {"date": "date", "pnl": "pnl"}; KPI_CONFIG = {}; KPI_GROUPS_OVERVIEW = {}; AVAILABLE_BENCHMARKS = {}
    PLOT_BENCHMARK_LINE_COLOR = "#800080"
    class KPIClusterDisplay: __init__ = lambda self, **kwargs: None; render = lambda self: st.warning("KPI Display Component failed to load.")
    class AnalysisService: get_core_kpis = lambda self, *args, **kwargs: {"error": "Service not loaded"}
    plot_equity_curve_and_drawdown = lambda **kwargs: None; _apply_custom_theme = lambda fig, theme: fig
    display_custom_message = lambda msg, type="error": st.error(msg); format_currency = lambda val, **kwargs: f"${val:,.2f}"
    st.stop()

logger = logging.getLogger(APP_TITLE)
analysis_service_instance = AnalysisService()

def get_timeframe_filtered_df(df: pd.DataFrame, date_col: str, timeframe_option: str) -> pd.DataFrame:
    if df.empty or date_col not in df.columns: return pd.DataFrame()
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        try: df[date_col] = pd.to_datetime(df[date_col])
        except: return pd.DataFrame()
    max_date_in_data = df[date_col].max()
    if pd.isna(max_date_in_data): return df
    today = max_date_in_data
    if timeframe_option == "All Time": return df
    if timeframe_option == "Last 30 Days": start_date = today - pd.Timedelta(days=29)
    elif timeframe_option == "Last 90 Days": start_date = today - pd.Timedelta(days=89)
    elif timeframe_option == "Year to Date (YTD)": start_date = pd.Timestamp(year=today.year, month=1, day=1)
    elif timeframe_option == "Last 1 Year": start_date = today - pd.Timedelta(days=364)
    else: return df
    return df[df[date_col] >= start_date]

def show_overview_page():
    st.title("📈 Performance Overview")
    if 'filtered_data' not in st.session_state or st.session_state.filtered_data is None:
        display_custom_message("Please upload and process data.", "info"); return
    if 'kpi_results' not in st.session_state or st.session_state.kpi_results is None:
        display_custom_message("KPI results not available.", "warning"); return
    if 'error' in st.session_state.kpi_results:
        display_custom_message(f"KPI calculation error: {st.session_state.kpi_results['error']}", "error"); return

    filtered_df_global = st.session_state.filtered_data 
    kpi_results_global = st.session_state.kpi_results
    kpi_confidence_intervals = st.session_state.get('kpi_confidence_intervals', {})
    plot_theme = st.session_state.get('current_theme', 'dark')
    benchmark_daily_returns = st.session_state.get('benchmark_daily_returns')
    selected_benchmark_display_name = st.session_state.get('selected_benchmark_display_name', "Benchmark")
    initial_capital = st.session_state.get('initial_capital', 100000.0)
    date_col = EXPECTED_COLUMNS.get('date', 'date')
    pnl_col = EXPECTED_COLUMNS.get('pnl', 'pnl')
    cum_pnl_col = 'cumulative_pnl'
    drawdown_pct_col_name = 'drawdown_pct'
    max_dd_details_from_state = st.session_state.get('max_drawdown_period_details') # Get max DD details

    if filtered_df_global.empty:
        display_custom_message("No data matches global filters.", "info"); return

    if date_col in filtered_df_global.columns and not filtered_df_global[date_col].dropna().empty:
        last_trade_date = pd.to_datetime(filtered_df_global[date_col]).max()
        st.caption(f"Data analysis based on trades up to: {last_trade_date.strftime('%Y-%m-%d %H:%M:%S')}")
    st.markdown("---")

    st.subheader("Current Snapshot")
    snap_col1, snap_col2, snap_col3 = st.columns(3)
    latest_equity, current_dd, last_day_pnl, last_trading_day_str = np.nan, np.nan, np.nan, "N/A"
    # ... (Snapshot logic remains the same)
    if cum_pnl_col in filtered_df_global.columns and not filtered_df_global[cum_pnl_col].empty:
        latest_equity = filtered_df_global[cum_pnl_col].iloc[-1]
    if drawdown_pct_col_name in filtered_df_global.columns and not filtered_df_global[drawdown_pct_col_name].empty:
        if filtered_df_global[drawdown_pct_col_name].iloc[-1] > 0:
            current_dd = filtered_df_global[drawdown_pct_col_name].iloc[-1]
    if date_col in filtered_df_global.columns and pnl_col in filtered_df_global.columns and not filtered_df_global.empty:
        last_trading_day_ts = filtered_df_global[date_col].max()
        if pd.notna(last_trading_day_ts):
            last_trading_day_str = last_trading_day_ts.normalize().strftime('%Y-%m-%d')
            last_day_pnl_val = filtered_df_global[filtered_df_global[date_col].dt.normalize() == last_trading_day_ts.normalize()][pnl_col].sum()
            if pd.notna(last_day_pnl_val): last_day_pnl = last_day_pnl_val
    with snap_col1: st.metric("Latest Equity", format_currency(latest_equity) if pd.notna(latest_equity) else "N/A")
    with snap_col2: st.metric("Current Drawdown", f"{current_dd:.2f}%" if pd.notna(current_dd) else "None")
    with snap_col3: st.metric(f"Last Day PnL ({last_trading_day_str})", format_currency(last_day_pnl) if pd.notna(last_day_pnl) else "N/A")
    st.markdown("---")


    st.header("Key Performance Indicators (Global Filter)")
    if date_col in filtered_df_global.columns and not filtered_df_global[date_col].empty:
        min_date_global_str = filtered_df_global[date_col].min().strftime('%Y-%m-%d')
        max_date_global_str = filtered_df_global[date_col].max().strftime('%Y-%m-%d')
        st.caption(f"Based on globally filtered data from {min_date_global_str} to {max_date_global_str}.")
    
    cols_per_row_setting = 4
    for group_name, kpi_keys_in_group in KPI_GROUPS_OVERVIEW.items():
        # ... (KPI group rendering logic with expanders remains the same) ...
        group_kpi_results = {key: kpi_results_global[key] for key in kpi_keys_in_group if key in kpi_results_global}
        default_expanded = True
        if group_name == "Benchmark Comparison":
            is_benchmark_active = benchmark_daily_returns is not None and not benchmark_daily_returns.empty
            has_benchmark_kpis = any(pd.notna(group_kpi_results.get(key)) for key in ["alpha", "beta"])
            default_expanded = is_benchmark_active and has_benchmark_kpis
            if not default_expanded and (not group_kpi_results or all(pd.isna(val) for val in group_kpi_results.values())): continue
        if not group_kpi_results or all(pd.isna(val) for val in group_kpi_results.values()):
            if group_name != "Benchmark Comparison": logger.info(f"KPI group '{group_name}' has all NaN values.")
        with st.expander(group_name, expanded=default_expanded):
            if not group_kpi_results or all(pd.isna(val) for val in group_kpi_results.values()):
                display_custom_message(f"No data for KPIs in '{group_name}'.", "info")
            else:
                KPIClusterDisplay(kpi_results=group_kpi_results, kpi_definitions=KPI_CONFIG, kpi_order=kpi_keys_in_group, 
                                  kpi_confidence_intervals=kpi_confidence_intervals, cols_per_row=cols_per_row_setting,
                                  benchmark_context_name=(selected_benchmark_display_name if group_name == "Benchmark Comparison" and selected_benchmark_display_name != "None" else None)
                ).render()
    st.markdown("---")
    
    st.header("Strategy Performance Charts")
    plot_area_equity = st.container() 
    df_for_plot_base_global_scope = pd.DataFrame()
    df_for_plot_time_filtered = pd.DataFrame()

    with plot_area_equity: 
        st.subheader("Strategy Equity and Drawdown")
        timeframe_options = ["All Time", "Last 1 Year", "Year to Date (YTD)", "Last 90 Days", "Last 30 Days"]
        selected_timeframe = st.radio("Select Timeframe for Equity Curve:", options=timeframe_options, index=0, horizontal=True, key="overview_equity_timeframe_radio")
        try:
            if date_col not in filtered_df_global.columns: display_custom_message(f"Date column ('{date_col}') not found.", "error")
            else:
                df_for_plot_base_global_scope = filtered_df_global.copy()
                if not pd.api.types.is_datetime64_any_dtype(df_for_plot_base_global_scope[date_col]):
                    df_for_plot_base_global_scope[date_col] = pd.to_datetime(df_for_plot_base_global_scope[date_col], errors='coerce')
                df_for_plot_base_global_scope.dropna(subset=[date_col], inplace=True)
                if df_for_plot_base_global_scope.empty: display_custom_message("No valid date entries for equity curve.", "error")
                else:
                    df_for_plot_base_global_scope = df_for_plot_base_global_scope.sort_values(by=date_col)
                    if cum_pnl_col not in df_for_plot_base_global_scope.columns:
                         if pnl_col in df_for_plot_base_global_scope.columns and pd.api.types.is_numeric_dtype(df_for_plot_base_global_scope[pnl_col]):
                            df_for_plot_base_global_scope[cum_pnl_col] = df_for_plot_base_global_scope[pnl_col].cumsum()
                         else: display_custom_message(f"PnL/CumPnL columns issue.", "error"); df_for_plot_base_global_scope = pd.DataFrame() 
                    if not df_for_plot_base_global_scope.empty:
                        if drawdown_pct_col_name not in df_for_plot_base_global_scope.columns and cum_pnl_col in df_for_plot_base_global_scope.columns:
                            hwm = df_for_plot_base_global_scope[cum_pnl_col].cummax()
                            dd_abs = hwm - df_for_plot_base_global_scope[cum_pnl_col]
                            df_for_plot_base_global_scope[drawdown_pct_col_name] = (dd_abs / hwm.replace(0,np.nan)).fillna(0) * 100
                            df_for_plot_base_global_scope.loc[(hwm == 0) & (dd_abs > 0), drawdown_pct_col_name] = 100.0
                        df_for_plot_time_filtered = get_timeframe_filtered_df(df_for_plot_base_global_scope, date_col, selected_timeframe)
                    
                    # Prepare max drawdown highlight dates for the *selected timeframe*
                    max_dd_peak_plot, max_dd_trough_plot, max_dd_end_plot = None, None, None
                    if not df_for_plot_time_filtered.empty and selected_timeframe == "All Time" and max_dd_details_from_state:
                        # Only use global max DD details if "All Time" is selected
                        max_dd_peak_plot = max_dd_details_from_state.get('Peak Date')
                        max_dd_trough_plot = max_dd_details_from_state.get('Trough Date')
                        max_dd_end_plot = max_dd_details_from_state.get('End Date') # Could be NaT
                        logger.info(f"Using global max DD details for 'All Time' plot: P:{max_dd_peak_plot}, T:{max_dd_trough_plot}, E:{max_dd_end_plot}")
                    elif not df_for_plot_time_filtered.empty and selected_timeframe != "All Time":
                        # For other timeframes, calculate max DD specific to that timeframe
                        # This requires re-running a simplified drawdown analysis on df_for_plot_time_filtered
                        # For now, we'll skip highlighting max DD for sub-timeframes on Overview to keep it simpler,
                        # or one could call analysis_service.get_advanced_drawdown_analysis here.
                        # Let's assume for now we only highlight the *global* max DD when "All Time" is viewed.
                        logger.info(f"Max DD highlight skipped for timeframe '{selected_timeframe}' on Overview page (only global shown).")
                        pass


                    if df_for_plot_time_filtered.empty:
                         if not df_for_plot_base_global_scope.empty: display_custom_message(f"No data for timeframe '{selected_timeframe}'.", "info")
                    else:
                        equity_fig = plot_equity_curve_and_drawdown(
                            df_for_plot_time_filtered, date_col=date_col, 
                            cumulative_pnl_col=cum_pnl_col, 
                            drawdown_pct_col=drawdown_pct_col_name if drawdown_pct_col_name in df_for_plot_time_filtered else None, 
                            theme=plot_theme,
                            max_dd_peak_date=max_dd_peak_plot, # Pass details
                            max_dd_trough_date=max_dd_trough_plot,
                            max_dd_recovery_date=max_dd_end_plot
                        )
                        if equity_fig: st.plotly_chart(equity_fig, use_container_width=True)
                        else: display_custom_message(f"Chart unavailable for '{selected_timeframe}'.", "info")
        except Exception as e:
            logger.error(f"Error displaying equity curve: {e}", exc_info=True); display_custom_message(f"Error with equity curve: {e}", "error")

    if not df_for_plot_time_filtered.empty and selected_timeframe != "All Time":
        # ... (Timeframe-specific KPI logic remains the same) ...
        st.markdown("---")
        st.subheader(f"Key Metrics for Selected Timeframe: {selected_timeframe}")
        with st.spinner(f"Calculating metrics for {selected_timeframe}..."):
            kpis_timeframe_df = df_for_plot_time_filtered[[date_col, pnl_col] + ([drawdown_pct_col_name] if drawdown_pct_col_name in df_for_plot_time_filtered else []) ].copy()
            timeframe_kpis = analysis_service_instance.get_core_kpis(kpis_timeframe_df, 
                st.session_state.risk_free_rate, None, st.session_state.initial_capital)
            if timeframe_kpis and 'error' not in timeframe_kpis:
                timeframe_kpi_keys_to_show = ["total_pnl", "win_rate", "avg_trade_pnl", "max_drawdown_pct", "total_trades", "sharpe_ratio"]
                focused_kpis = {key: timeframe_kpis[key] for key in timeframe_kpi_keys_to_show if key in timeframe_kpis}
                if focused_kpis: KPIClusterDisplay(kpi_results=focused_kpis, kpi_definitions=KPI_CONFIG,
                    kpi_order=timeframe_kpi_keys_to_show, cols_per_row=3).render()
                else: display_custom_message("No focused KPIs for timeframe.", "info")
            else: display_custom_message(f"KPI calc error for timeframe '{selected_timeframe}': {timeframe_kpis.get('error', 'Unknown') if timeframe_kpis else 'Failed'}", "warning")

    st.markdown("---")
    # ... (Benchmark plotting logic remains the same) ...
    benchmark_plot_equity = pd.Series(dtype=float)
    if benchmark_daily_returns is not None and not benchmark_daily_returns.empty:
        bm_daily_returns_series = benchmark_daily_returns.squeeze() if isinstance(benchmark_daily_returns, pd.DataFrame) else benchmark_daily_returns
        if isinstance(bm_daily_returns_series, pd.Series) and not bm_daily_returns_series.empty:
            bm_returns_for_factor = bm_daily_returns_series.copy()
            if not bm_returns_for_factor.empty and pd.isna(bm_returns_for_factor.iloc[0]): bm_returns_for_factor.iloc[0] = 0.0
            benchmark_cumulative_growth_factor = (1 + bm_returns_for_factor).cumprod()
            if not benchmark_cumulative_growth_factor.empty: benchmark_plot_equity = benchmark_cumulative_growth_factor * initial_capital
        else: logger.warning("Benchmark returns not valid Series or empty.")
    plot_area_benchmark = st.container() 
    with plot_area_benchmark: 
        if not benchmark_plot_equity.empty:
            st.subheader(f"Benchmark Equity Curve: {selected_benchmark_display_name}")
            fig_benchmark_only = go.Figure()
            fig_benchmark_only.add_trace(go.Scatter(x=benchmark_plot_equity.index, y=benchmark_plot_equity, mode='lines',
                name=f"{selected_benchmark_display_name} (Scaled Equity)", line=dict(color=PLOT_BENCHMARK_LINE_COLOR, width=2)))
            fig_benchmark_only.update_layout(title_text=f"{selected_benchmark_display_name} Performance (Scaled to Initial Capital)",
                xaxis_title="Date", yaxis_title="Scaled Benchmark Value", hovermode="x unified")
            st.plotly_chart(_apply_custom_theme(fig_benchmark_only, plot_theme), use_container_width=True)
        elif st.session_state.get('selected_benchmark_ticker') and st.session_state.get('selected_benchmark_ticker') != "":
            st.subheader(f"Benchmark Equity Curve: {selected_benchmark_display_name}")
            display_custom_message(f"Chart unavailable for benchmark '{selected_benchmark_display_name}'.", "info")
    st.markdown("---")


if __name__ == "__main__":
    if 'app_initialized' not in st.session_state:
        st.warning("This page is part of a multi-page app. Please run the main app.py script.")
    show_overview_page()
