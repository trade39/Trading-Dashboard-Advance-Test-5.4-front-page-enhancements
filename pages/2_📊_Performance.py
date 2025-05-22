"""
pages/2_📊_Performance.py

This page delves into detailed performance metrics and visualizations,
such as PnL distributions, categorical PnL analysis, win rates by time,
a P&L calendar view, rolling Sharpe ratio, performance by trade duration,
and Day/Hour performance heatmaps.
"""
import streamlit as st
import pandas as pd
import numpy as np
import logging

try:
    from config import APP_TITLE, EXPECTED_COLUMNS, RISK_FREE_RATE, COLORS
    from plotting import (
        plot_pnl_distribution, plot_pnl_by_category,
        plot_win_rate_analysis, plot_rolling_performance,
        plot_box_plot, plot_heatmap
    )
    from components.calendar_view import PnLCalendarComponent
    from utils.common_utils import display_custom_message, format_currency
except ImportError as e:
    st.error(f"Performance Page Error: Critical module import failed: {e}. Ensure app structure is correct.")
    APP_TITLE = "TradingDashboard_Error"
    logger = logging.getLogger(APP_TITLE)
    logger.error(f"CRITICAL IMPORT ERROR in Performance Page: {e}", exc_info=True)
    EXPECTED_COLUMNS = {"pnl": "pnl", "date": "date", "duration_minutes_numeric": "duration_minutes_numeric", "win": "win"}
    RISK_FREE_RATE = 0.02
    COLORS = {}
    # Mock functions for graceful failure
    def plot_pnl_distribution(*args, **kwargs): return None
    def plot_pnl_by_category(*args, **kwargs): return None
    def plot_win_rate_analysis(*args, **kwargs): return None
    def plot_rolling_performance(*args, **kwargs): return None
    def plot_box_plot(*args, **kwargs): return None
    def plot_heatmap(*args, **kwargs): return None
    def format_currency(val): return f"${val:,.2f}" if isinstance(val, (int, float)) else str(val)
    class PnLCalendarComponent:
        def __init__(self, *args, **kwargs): pass
        def render(self): st.warning("Calendar component failed to load due to import errors.")
    def display_custom_message(msg, type="error"): st.error(msg) # type: ignore
    st.stop()

logger = logging.getLogger(APP_TITLE)

@st.cache_data # Cache the calculation
def calculate_rolling_sharpe(pnl_series: pd.Series, window: int, min_periods: int, risk_free_rate_daily: float, periods_per_year: int = 252) -> pd.Series:
    """Calculates rolling Sharpe ratio."""
    if pnl_series.empty or len(pnl_series) < min_periods:
        return pd.Series(dtype=float)
    rolling_mean_pnl = pnl_series.rolling(window=window, min_periods=min_periods).mean()
    rolling_std_pnl = pnl_series.rolling(window=window, min_periods=min_periods).std().replace(0, np.nan) 
    
    excess_returns = rolling_mean_pnl - risk_free_rate_daily
    sharpe_like_ratio = excess_returns / rolling_std_pnl
    annualized_sharpe = (sharpe_like_ratio * np.sqrt(periods_per_year)).fillna(0)
    return annualized_sharpe

def show_performance_page():
    """Renders the detailed performance analysis page."""
    st.title("📊 Detailed Performance Analysis")
    logger.info("Rendering Performance Page.")

    if 'filtered_data' not in st.session_state or st.session_state.filtered_data is None:
        display_custom_message("Please upload and process data on the 'Data Upload' page to view performance details.", "info")
        return
    
    filtered_df = st.session_state.filtered_data
    plot_theme = st.session_state.get('current_theme', 'dark')
    pnl_col = EXPECTED_COLUMNS.get('pnl', 'pnl')
    date_col_main = EXPECTED_COLUMNS.get('date', 'date')
    win_col = EXPECTED_COLUMNS.get('win', 'win') 
    risk_free_rate_annual = st.session_state.get('risk_free_rate', RISK_FREE_RATE)
    risk_free_rate_daily = (1 + risk_free_rate_annual)**(1/252) - 1

    if filtered_df.empty:
        display_custom_message("No data matches the current filters. Please adjust filters or upload new data.", "info")
        return
    if pnl_col not in filtered_df.columns:
        display_custom_message(f"Critical Error: PnL column ('{pnl_col}') not found in the dataset. Please check column mapping.", "error")
        return
    if date_col_main not in filtered_df.columns:
        display_custom_message(f"Warning: Date column ('{date_col_main}') not found. Some features like rolling metrics and calendar view might be affected.", "warning")
    if win_col not in filtered_df.columns:
        display_custom_message(f"Warning: Win column ('{win_col}') not found. Win rate related metrics will not be available.", "warning")

    # --- Key Performance Indicators (KPIs) ---
    st.markdown("<div class='performance-section-container'>", unsafe_allow_html=True)
    st.subheader("🚀 Key Performance Indicators") 
    
    total_pnl = filtered_df[pnl_col].sum()
    total_trades = len(filtered_df)
    
    overall_win_rate = 0.0
    if win_col in filtered_df.columns and not filtered_df[win_col].empty:
        if pd.api.types.is_numeric_dtype(filtered_df[win_col]):
            overall_win_rate = filtered_df[win_col].mean() * 100
        else:
            try:
                numeric_wins = pd.to_numeric(filtered_df[win_col].astype(str).str.lower().map({'true': 1, 'false': 0, '1': 1, '0': 0}), errors='coerce')
                if not numeric_wins.isnull().all():
                    overall_win_rate = numeric_wins.mean() * 100
                else:
                    display_custom_message(f"Could not interpret '{win_col}' column for win rate. Ensure it contains 0/1 or True/False.", "warning")
            except Exception:
                 display_custom_message(f"Error converting '{win_col}' column for win rate. Please check its format.", "warning")

    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    with kpi_col1:
        st.metric(label="💰 Total PnL", value=format_currency(total_pnl), delta_color="normal")
    with kpi_col2:
        if win_col in filtered_df.columns:
             st.metric(label="🎯 Overall Win Rate", value=f"{overall_win_rate:.2f}%" if overall_win_rate is not None else "N/A")
        else:
            st.metric(label="🎯 Overall Win Rate", value="N/A")
    with kpi_col3:
        st.metric(label="🔢 Total Trades", value=f"{total_trades:,}")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")

    # --- Performance Breakdown Section ---
    st.markdown("<div class='performance-section-container'>", unsafe_allow_html=True)
    st.subheader("📈 Performance Breakdown") 
    st.markdown("Visualizing Profit & Loss distributions and categorical performance.")
    try:
        # Data for PnL Distribution
        pnl_dist_data_for_view = filtered_df[[pnl_col]].copy()
        pnl_dist_fig = plot_pnl_distribution(filtered_df, pnl_col=pnl_col, theme=plot_theme)

        # Data for PnL by DOW
        pnl_dow_data_for_view = None
        pnl_dow_fig = None
        if 'trade_day_of_week' in filtered_df.columns:
            pnl_dow_data_for_view = filtered_df.groupby('trade_day_of_week')[pnl_col].agg(['sum', 'mean', 'count']).reset_index()
            pnl_dow_fig = plot_pnl_by_category(filtered_df, 'trade_day_of_week', pnl_col=pnl_col, theme=plot_theme, title_prefix="Total PnL by")
        
        # Data for Win Rate by Hour
        winrate_hour_data_for_view = None
        winrate_hour_fig = None
        if 'trade_hour' in filtered_df.columns and win_col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[win_col]):
            winrate_hour_data_for_view = filtered_df.groupby('trade_hour')[win_col].agg(['mean', 'count']).reset_index()
            winrate_hour_data_for_view['mean'] *= 100 # Convert to percentage
            winrate_hour_fig = plot_win_rate_analysis(filtered_df, 'trade_hour', win_col=win_col, theme=plot_theme, title_prefix="Win Rate by")

        # Data for PnL by Month
        month_col_for_plot = 'trade_month_name' 
        pnl_month_data_for_view = None
        pnl_month_fig = None
        if month_col_for_plot in filtered_df.columns:
            pnl_month_data_for_view = filtered_df.groupby(month_col_for_plot)[pnl_col].agg(['sum', 'mean', 'count']).reset_index()
            pnl_month_fig = plot_pnl_by_category(filtered_df, month_col_for_plot, pnl_col=pnl_col, theme=plot_theme, title_prefix="Total PnL by")

        # Display plots and "View Data" expanders
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h6>PnL Distribution</h6>", unsafe_allow_html=True)
            if pnl_dist_fig: 
                st.plotly_chart(pnl_dist_fig, use_container_width=True)
                with st.expander("👁️ View PnL Distribution Data (Raw)"):
                    st.dataframe(pnl_dist_data_for_view)
            else: 
                display_custom_message("Could not generate PnL distribution plot. Ensure PnL data is valid.", "warning")

            if pnl_dow_fig:
                st.markdown("<h6>PnL by Day of Week</h6>", unsafe_allow_html=True)
                st.plotly_chart(pnl_dow_fig, use_container_width=True)
                if pnl_dow_data_for_view is not None and not pnl_dow_data_for_view.empty:
                    with st.expander("👁️ View PnL by Day of Week Data"):
                        st.dataframe(pnl_dow_data_for_view)
            elif 'trade_day_of_week' not in filtered_df.columns:
                display_custom_message("Column 'trade_day_of_week' not found for PnL by Day of Week plot.", "info")


        with col2:
            if winrate_hour_fig:
                st.markdown("<h6>Win Rate by Hour of Day</h6>", unsafe_allow_html=True)
                st.plotly_chart(winrate_hour_fig, use_container_width=True)
                if winrate_hour_data_for_view is not None and not winrate_hour_data_for_view.empty:
                    with st.expander("👁️ View Win Rate by Hour Data"):
                        st.dataframe(winrate_hour_data_for_view)
            elif 'trade_hour' not in filtered_df.columns:
                 display_custom_message("Column 'trade_hour' not found for Win Rate by Hour plot.", "info")
            elif win_col not in filtered_df.columns:
                 display_custom_message(f"Column '{win_col}' not found for Win Rate by Hour plot.", "info")
            
            if pnl_month_fig:
                st.markdown("<h6>PnL by Month</h6>", unsafe_allow_html=True)
                st.plotly_chart(pnl_month_fig, use_container_width=True)
                if pnl_month_data_for_view is not None and not pnl_month_data_for_view.empty:
                    with st.expander("👁️ View PnL by Month Data"):
                        st.dataframe(pnl_month_data_for_view)
            elif month_col_for_plot not in filtered_df.columns:
                display_custom_message(f"Column '{month_col_for_plot}' not found for PnL by Month plot.", "info")

    except Exception as e:
        logger.error(f"Error rendering performance breakdown section: {e}", exc_info=True)
        display_custom_message(f"An error occurred in the Performance Breakdown section: {e}", "error")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")

    # --- Rolling Performance Metrics ---
    st.markdown("<div class='performance-section-container'>", unsafe_allow_html=True)
    st.subheader("🔄 Rolling Performance Metrics") 
    st.markdown("Track performance consistency over time with rolling window calculations.")
    
    rolling_pnl_sum_data, rolling_win_rate_data, rolling_sharpe_data = None, None, None # Initialize for view data

    try:
        if date_col_main not in filtered_df.columns:
            display_custom_message(f"Date column ('{date_col_main}') is required for rolling metrics.", "warning")
        elif len(filtered_df) < 10: 
            display_custom_message("Not enough data points (minimum 10) for rolling metrics.", "info")
        else:
            with st.expander("⚙️ Customize Rolling Metrics"):
                controls_col1, controls_col2, controls_col3 = st.columns(3)
                max_slider_val_periods = min(100, len(filtered_df) // 2 if len(filtered_df) > 20 else 10) 
                unique_days_count = len(filtered_df[date_col_main].dt.normalize().unique())
                max_slider_val_days = min(200, unique_days_count // 2 if unique_days_count > 20 else 10)

                with controls_col1: 
                    rolling_pnl_window = st.slider("Rolling PnL Sum Window (Periods):", 
                                                   min_value=5, 
                                                   max_value=max(5, max_slider_val_periods), 
                                                   value=min(30, max(5, max_slider_val_periods)), 
                                                   step=5, key="roll_pnl_win_slider")
                with controls_col2: 
                    rolling_win_rate_window = st.slider("Rolling Win Rate Window (Periods):", 
                                                        min_value=10, 
                                                        max_value=max(10, max_slider_val_periods), 
                                                        value=min(50, max(10, max_slider_val_periods)), 
                                                        step=10, key="roll_wr_win_slider")
                with controls_col3: 
                    rolling_sharpe_window = st.slider("Rolling Sharpe Window (Days):", 
                                                      min_value=10, 
                                                      max_value=max(10, max_slider_val_days), 
                                                      value=min(50, max(10, max_slider_val_days)), 
                                                      step=10, key="roll_sharpe_win_slider")
            
            min_periods_pnl = max(5, rolling_pnl_window // 3)
            min_periods_win_rate = max(10, rolling_win_rate_window // 3)
            min_periods_sharpe = max(10, rolling_sharpe_window // 3)

            df_for_rolling = filtered_df.sort_values(by=date_col_main).copy()
            if not pd.api.types.is_numeric_dtype(df_for_rolling[pnl_col]):
                df_for_rolling[pnl_col] = pd.to_numeric(df_for_rolling[pnl_col], errors='coerce')
                df_for_rolling.dropna(subset=[pnl_col], inplace=True)
            
            if df_for_rolling.empty:
                display_custom_message("No valid numeric PnL data for rolling metrics.", "warning")
            else:
                plot_col1, plot_col2, plot_col3 = st.columns([1,1,1]) 
                with plot_col1:
                    if len(df_for_rolling) >= min_periods_pnl:
                        st.markdown("<h6>Rolling PnL Sum</h6>", unsafe_allow_html=True)
                        rolling_pnl_sum_data = df_for_rolling[pnl_col].rolling(window=rolling_pnl_window, min_periods=min_periods_pnl).sum()
                        rolling_pnl_metric_name = f"{rolling_pnl_window}-P Rolling PnL Sum"
                        rolling_pnl_fig = plot_rolling_performance(df_for_rolling, date_col_main, rolling_pnl_sum_data, rolling_pnl_metric_name, rolling_pnl_metric_name, plot_theme)
                        if rolling_pnl_fig: 
                            st.plotly_chart(rolling_pnl_fig, use_container_width=True)
                            if rolling_pnl_sum_data is not None and not rolling_pnl_sum_data.empty:
                                with st.expander("👁️ View Rolling PnL Data"):
                                    st.dataframe(rolling_pnl_sum_data)
                    else: display_custom_message(f"Need {min_periods_pnl} periods for rolling PnL.", "info")
                
                with plot_col2:
                    if win_col in df_for_rolling.columns and pd.api.types.is_numeric_dtype(df_for_rolling[win_col]) and len(df_for_rolling) >= min_periods_win_rate:
                        st.markdown("<h6>Rolling Win Rate</h6>", unsafe_allow_html=True)
                        rolling_win_rate_data = df_for_rolling[win_col].rolling(window=rolling_win_rate_window, min_periods=min_periods_win_rate).mean() * 100
                        rolling_wr_metric_name = f"{rolling_win_rate_window}-P Rolling Win Rate (%)"
                        rolling_wr_fig = plot_rolling_performance(df_for_rolling, date_col_main, rolling_win_rate_data, rolling_wr_metric_name, rolling_wr_metric_name, plot_theme)
                        if rolling_wr_fig: 
                            st.plotly_chart(rolling_wr_fig, use_container_width=True)
                            if rolling_win_rate_data is not None and not rolling_win_rate_data.empty:
                                with st.expander("👁️ View Rolling Win Rate Data"):
                                    st.dataframe(rolling_win_rate_data)
                    elif win_col not in df_for_rolling.columns: display_custom_message(f"'{win_col}' column needed for Rolling Win Rate.", "info")
                    else: display_custom_message(f"Need {min_periods_win_rate} periods for rolling Win Rate.", "info")

                with plot_col3:
                    st.markdown("<h6>Rolling Sharpe Ratio (Annualized)</h6>", unsafe_allow_html=True)
                    daily_pnl_for_sharpe = df_for_rolling.groupby(df_for_rolling[date_col_main].dt.normalize())[pnl_col].sum()
                    if len(daily_pnl_for_sharpe) >= min_periods_sharpe:
                        rolling_sharpe_data = calculate_rolling_sharpe(daily_pnl_for_sharpe, rolling_sharpe_window, min_periods_sharpe, risk_free_rate_daily)
                        rolling_sharpe_metric_name = f"{rolling_sharpe_window}-D Rolling Sharpe Ratio"
                        if not rolling_sharpe_data.empty:
                             sharpe_df_to_plot = pd.DataFrame({date_col_main: rolling_sharpe_data.index, 'sharpe': rolling_sharpe_data.values})
                             rolling_sharpe_fig = plot_rolling_performance(sharpe_df_to_plot, date_col_main, sharpe_df_to_plot['sharpe'], rolling_sharpe_metric_name, rolling_sharpe_metric_name, plot_theme)
                             if rolling_sharpe_fig: 
                                 st.plotly_chart(rolling_sharpe_fig, use_container_width=True)
                                 with st.expander("👁️ View Rolling Sharpe Data"):
                                     st.dataframe(rolling_sharpe_data)
                        else: display_custom_message("Could not calculate Rolling Sharpe Ratio.", "warning")
                    else: display_custom_message(f"Need {min_periods_sharpe} daily PnL points for rolling Sharpe.", "info")
    except Exception as e:
        logger.error(f"Error rendering rolling performance metrics: {e}", exc_info=True)
        display_custom_message(f"An error occurred in the Rolling Performance section: {e}", "error")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")

    # --- Performance by Trade Duration ---
    st.markdown("<div class='performance-section-container'>", unsafe_allow_html=True)
    st.subheader("⏱️ Performance by Trade Duration") 
    st.markdown("Analyze how trade duration impacts profitability and win rates.")
    duration_col_numeric = EXPECTED_COLUMNS.get('duration_minutes_numeric', 'duration_minutes_numeric')
    
    pnl_by_duration_agg_data, winrate_by_duration_agg_data = None, None # Initialize for view data

    if duration_col_numeric in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[duration_col_numeric]):
        cols_for_duration = [duration_col_numeric, pnl_col]
        if win_col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[win_col]): # ensure win_col is numeric if present
            cols_for_duration.append(win_col)
        
        df_for_duration = filtered_df[cols_for_duration].copy()
        df_for_duration.dropna(subset=[duration_col_numeric, pnl_col], inplace=True)

        if not df_for_duration.empty:
            bins = [-float('inf'), 30, 60, 120, 240, 480, float('inf')]
            labels = ["<30 min", "30-60 min", "1-2 hrs", "2-4 hrs", "4-8 hrs", ">8 hrs"]
            df_for_duration['duration_bin'] = pd.cut(df_for_duration[duration_col_numeric], bins=bins, labels=labels, right=False)
            df_for_duration.dropna(subset=['duration_bin'], inplace=True) 

            if not df_for_duration.empty:
                col_dur1, col_dur2 = st.columns(2)
                with col_dur1:
                    st.markdown("<h6>PnL Distribution by Duration</h6>", unsafe_allow_html=True)
                    pnl_by_duration_fig = plot_box_plot(df_for_duration, 'duration_bin', pnl_col, "PnL Distribution by Trade Duration", plot_theme)
                    if pnl_by_duration_fig: 
                        st.plotly_chart(pnl_by_duration_fig, use_container_width=True)
                        pnl_by_duration_agg_data = df_for_duration.groupby('duration_bin')[pnl_col].agg(['sum', 'mean', 'median', 'std', 'count']).reset_index()
                        with st.expander("👁️ View PnL by Duration Summary Data"):
                            st.dataframe(pnl_by_duration_agg_data)
                    else: display_custom_message("Could not generate PnL by duration plot.", "warning")
                
                with col_dur2:
                    if win_col in df_for_duration.columns and pd.api.types.is_numeric_dtype(df_for_duration[win_col]):
                        st.markdown("<h6>Win Rate by Duration</h6>", unsafe_allow_html=True)
                        winrate_duration_fig = plot_win_rate_analysis(df_for_duration, 'duration_bin', win_col, "Win Rate by Trade Duration", plot_theme)
                        if winrate_duration_fig: 
                            st.plotly_chart(winrate_duration_fig, use_container_width=True)
                            winrate_by_duration_agg_data = df_for_duration.groupby('duration_bin')[win_col].agg(['mean', 'count']).reset_index()
                            winrate_by_duration_agg_data['mean'] *= 100 # Convert to percentage
                            with st.expander("👁️ View Win Rate by Duration Data"):
                                st.dataframe(winrate_by_duration_agg_data)
                        else: display_custom_message("Could not generate Win Rate by duration plot.", "warning")
                    elif win_col not in df_for_duration.columns:
                        display_custom_message(f"'{win_col}' column needed for Win Rate by Duration plot.", "info")
            else:
                display_custom_message("No trades with valid duration data after binning.", "info")
        else:
            display_custom_message("No trades with valid numeric duration and PnL data.", "info")
    else:
        display_custom_message(f"Numeric duration column ('{duration_col_numeric}') not found or not numeric.", "warning")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")

    # --- Day & Hour Performance Matrix ---
    st.markdown("<div class='performance-section-container'>", unsafe_allow_html=True)
    st.subheader("📅 Day & Hour Performance Matrix") 
    st.markdown("Identify profitable trading times using heatmaps.")
    
    hour_col = 'trade_hour'        
    dow_col = 'trade_day_of_week'  
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    pnl_pivot_data, metric_pivot_data = None, None # Initialize for view data

    if hour_col in filtered_df.columns and dow_col in filtered_df.columns:
        with st.expander("⚙️ Customize Heatmaps"):
            heatmap_controls_col1, heatmap_controls_col2 = st.columns(2)
            with heatmap_controls_col1:
                pnl_agg_option = st.selectbox("PnL Aggregation for Heatmap:", ["Total PnL", "Average PnL"], key="heatmap_pnl_aggregation_select")
                pnl_agg_func = 'sum' if pnl_agg_option == "Total PnL" else 'mean'
            with heatmap_controls_col2:
                other_metric_option = st.selectbox("Second Heatmap Metric:", ["Win Rate %", "Trade Count"], key="heatmap_other_metric_select")

        heatmap_plot_col1, heatmap_plot_col2 = st.columns(2)
        try:
            with heatmap_plot_col1:
                st.markdown(f"<h6>{pnl_agg_option} by Hour & Day</h6>", unsafe_allow_html=True)
                pnl_pivot_data = filtered_df.pivot_table(index=hour_col, columns=dow_col, values=pnl_col, aggfunc=pnl_agg_func, fill_value=0)
                if not pnl_pivot_data.empty:
                    pnl_pivot_data = pnl_pivot_data.reindex(columns=[day for day in days_order if day in pnl_pivot_data.columns], fill_value=0) 
                    pnl_pivot_data = pnl_pivot_data.sort_index() 
                    pnl_heatmap_fig = plot_heatmap(
                        pnl_pivot_data, title=f"{pnl_agg_option} by Hour and Day",
                        color_scale="RdYlGn" if pnl_agg_option == "Average PnL" else "Blues",
                        text_format="$,.0f" if pnl_agg_option == "Total PnL" else "$,.2f", theme=plot_theme
                    )
                    if pnl_heatmap_fig: 
                        st.plotly_chart(pnl_heatmap_fig, use_container_width=True)
                        with st.expander(f"👁️ View {pnl_agg_option} Heatmap Data"):
                            st.dataframe(pnl_pivot_data)
                    else: display_custom_message(f"Could not generate {pnl_agg_option} heatmap.", "warning")
                else: display_custom_message(f"Not enough data for {pnl_agg_option} heatmap.", "info")

            with heatmap_plot_col2:
                if other_metric_option == "Win Rate %":
                    st.markdown("<h6>Win Rate (%) by Hour & Day</h6>", unsafe_allow_html=True)
                    if win_col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[win_col]):
                        metric_pivot_data = filtered_df.pivot_table(index=hour_col, columns=dow_col, values=win_col, aggfunc='mean', fill_value=0) * 100
                        if not metric_pivot_data.empty:
                            metric_pivot_data = metric_pivot_data.reindex(columns=[day for day in days_order if day in metric_pivot_data.columns], fill_value=0)
                            metric_pivot_data = metric_pivot_data.sort_index()
                            metric_heatmap_fig = plot_heatmap(metric_pivot_data, title="Win Rate (%) by Hour and Day", color_scale="Greens", text_format=".1f", z_min=0, z_max=100, theme=plot_theme)
                            if metric_heatmap_fig: 
                                st.plotly_chart(metric_heatmap_fig, use_container_width=True)
                                with st.expander(f"👁️ View {other_metric_option} Heatmap Data"):
                                    st.dataframe(metric_pivot_data)
                        else: display_custom_message("Not enough data for Win Rate heatmap.", "info")
                    else: display_custom_message(f"'{win_col}' column (numeric) needed for Win Rate heatmap.", "warning")
                
                elif other_metric_option == "Trade Count":
                    st.markdown("<h6>Trade Count by Hour & Day</h6>", unsafe_allow_html=True)
                    metric_pivot_data = filtered_df.pivot_table(index=hour_col, columns=dow_col, values=pnl_col, aggfunc='count', fill_value=0) 
                    if not metric_pivot_data.empty:
                        metric_pivot_data = metric_pivot_data.reindex(columns=[day for day in days_order if day in metric_pivot_data.columns], fill_value=0)
                        metric_pivot_data = metric_pivot_data.sort_index()
                        metric_heatmap_fig = plot_heatmap(metric_pivot_data, title="Trade Count by Hour and Day", color_scale="Blues", text_format=".0f", theme=plot_theme)
                        if metric_heatmap_fig: 
                            st.plotly_chart(metric_heatmap_fig, use_container_width=True)
                            with st.expander(f"👁️ View {other_metric_option} Heatmap Data"):
                                st.dataframe(metric_pivot_data)
                    else: display_custom_message("Not enough data for Trade Count heatmap.", "info")
        except Exception as e_heatmap:
            logger.error(f"Error generating Day/Hour heatmaps: {e_heatmap}", exc_info=True)
            display_custom_message(f"An error occurred while generating Day/Hour heatmaps: {e_heatmap}", "error")
    else:
        missing_cols = [col for col in [hour_col, dow_col] if col not in filtered_df.columns]
        display_custom_message(f"Engineered columns {', '.join(missing_cols)} not found. Cannot generate Day/Hour heatmaps.", "warning")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")

    # --- P&L Calendar View ---
    st.markdown("<div class='performance-section-container'>", unsafe_allow_html=True)
    st.subheader("🗓️ P&L Calendar View") 
    st.markdown("Visualize daily Profit & Loss in a familiar calendar format.")
    
    daily_pnl_df_agg_for_view = None # Initialize for view data

    if date_col_main in filtered_df.columns and pnl_col in filtered_df.columns:
        try:
            if not pd.api.types.is_datetime64_any_dtype(filtered_df[date_col_main]):
                filtered_df[date_col_main] = pd.to_datetime(filtered_df[date_col_main], errors='coerce')
                if filtered_df[date_col_main].isnull().all(): 
                     display_custom_message(f"Could not convert '{date_col_main}' to datetime for P&L Calendar.", "error")
                     st.markdown("</div>", unsafe_allow_html=True) 
                     return 

            daily_pnl_df_agg_for_view = filtered_df.groupby(filtered_df[date_col_main].dt.normalize())[pnl_col].sum().reset_index()
            daily_pnl_df_agg_for_view = daily_pnl_df_agg_for_view.rename(columns={date_col_main: 'date', pnl_col: 'pnl'}) 
            
            available_years = sorted(daily_pnl_df_agg_for_view['date'].dt.year.unique(), reverse=True)
            if available_years:
                selected_year = st.selectbox("Select Year for P&L Calendar:", options=available_years, index=0, key="perf_page_calendar_year_select_box")
                if selected_year:
                    calendar_component = PnLCalendarComponent(daily_pnl_df=daily_pnl_df_agg_for_view, year=selected_year, plot_theme=plot_theme)
                    calendar_component.render() 
                    if daily_pnl_df_agg_for_view is not None and not daily_pnl_df_agg_for_view.empty:
                        with st.expander("👁️ View Daily PnL Data (Calendar)"):
                            st.dataframe(daily_pnl_df_agg_for_view[daily_pnl_df_agg_for_view['date'].dt.year == selected_year])
            else:
                display_custom_message("No yearly data available to display the P&L calendar.", "info")
        except Exception as e_cal:
            logger.error(f"Error generating P&L Calendar View: {e_cal}", exc_info=True)
            display_custom_message(f"An error occurred while generating the P&L Calendar: {e_cal}", "error")
    else:
        missing_cols_cal = [col for col in [date_col_main, pnl_col] if col not in filtered_df.columns]
        display_custom_message(f"Required columns ({', '.join(missing_cols_cal)}) not found for P&L Calendar.", "warning")
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    if 'app_initialized' not in st.session_state:
        st.warning("This page is part of a multi-page app. For full functionality, please run the main `app.py` script.")
    show_performance_page()
