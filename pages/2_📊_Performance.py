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
    from config import APP_TITLE, EXPECTED_COLUMNS, RISK_FREE_RATE, COLORS # Added COLORS
    from plotting import (
        plot_pnl_distribution, plot_pnl_by_category,
        plot_win_rate_analysis, plot_rolling_performance,
        plot_box_plot, plot_heatmap # Added plot_heatmap
    )
    from components.calendar_view import PnLCalendarComponent
    from utils.common_utils import display_custom_message, format_currency # Added format_currency
except ImportError as e:
    st.error(f"Performance Page Error: Critical module import failed: {e}. Ensure app structure is correct.")
    APP_TITLE = "TradingDashboard_Error" 
    logger = logging.getLogger(APP_TITLE)
    logger.error(f"CRITICAL IMPORT ERROR in Performance Page: {e}", exc_info=True)
    EXPECTED_COLUMNS = {"pnl": "pnl", "date": "date", "duration_minutes_numeric": "duration_minutes_numeric"}
    RISK_FREE_RATE = 0.02; COLORS = {}
    def plot_pnl_distribution(*args, **kwargs): return None
    def plot_pnl_by_category(*args, **kwargs): return None
    def plot_win_rate_analysis(*args, **kwargs): return None
    def plot_rolling_performance(*args, **kwargs): return None
    def plot_box_plot(*args, **kwargs): return None
    def plot_heatmap(*args, **kwargs): return None
    def format_currency(val, **kwargs): return f"${val:,.2f}" 
    class PnLCalendarComponent:
        def __init__(self, *args, **kwargs): pass
        def render(self): st.warning("Calendar component failed to load.")
    def display_custom_message(msg, type="error"): st.error(msg)
    st.stop()

logger = logging.getLogger(APP_TITLE)

def calculate_rolling_sharpe(pnl_series: pd.Series, window: int, min_periods: int, risk_free_rate_daily: float, periods_per_year: int = 252) -> pd.Series:
    if pnl_series.empty or len(pnl_series) < min_periods: return pd.Series(dtype=float)
    rolling_mean_pnl = pnl_series.rolling(window=window, min_periods=min_periods).mean()
    rolling_std_pnl = pnl_series.rolling(window=window, min_periods=min_periods).std().replace(0, np.nan)
    sharpe_like_ratio = (rolling_mean_pnl - risk_free_rate_daily) / rolling_std_pnl
    return (sharpe_like_ratio * np.sqrt(periods_per_year)).fillna(0)

def show_performance_page():
    st.title("📊 Detailed Performance Analysis")
    logger.info("Rendering Performance Page.")

    if 'filtered_data' not in st.session_state or st.session_state.filtered_data is None:
        display_custom_message("Please upload and process data to view performance details.", "info"); return
    
    filtered_df = st.session_state.filtered_data
    plot_theme = st.session_state.get('current_theme', 'dark')
    pnl_col = EXPECTED_COLUMNS.get('pnl', 'pnl')
    date_col_main = EXPECTED_COLUMNS.get('date', 'date')
    risk_free_rate_annual = st.session_state.get('risk_free_rate', RISK_FREE_RATE)
    risk_free_rate_daily = (1 + risk_free_rate_annual)**(1/252) - 1

    if filtered_df.empty: display_custom_message("No data matches the current filters."); return
    if pnl_col not in filtered_df.columns: display_custom_message(f"PnL column ('{pnl_col}') not found.", "error"); return
    if date_col_main not in filtered_df.columns: display_custom_message(f"Date column ('{date_col_main}') not found.", "error")

    # --- Performance Breakdown Section ---
    st.subheader("Performance Breakdown")
    # ... (Existing Performance Breakdown section: PnL Dist, PnL by DOW, Win Rate by Hour, PnL by Month) ...
    try:
        col1, col2 = st.columns(2)
        with col1:
            pnl_dist_fig = plot_pnl_distribution(filtered_df, pnl_col=pnl_col, theme=plot_theme)
            if pnl_dist_fig: st.plotly_chart(pnl_dist_fig, use_container_width=True)
            else: display_custom_message("Could not generate PnL distribution plot.", "warning")

            if 'trade_day_of_week' in filtered_df.columns:
                pnl_dow_fig = plot_pnl_by_category(filtered_df, 'trade_day_of_week', pnl_col=pnl_col, theme=plot_theme, title_prefix="Total PnL by")
                if pnl_dow_fig: st.plotly_chart(pnl_dow_fig, use_container_width=True)
        with col2:
            if 'trade_hour' in filtered_df.columns and 'win' in filtered_df.columns:
                winrate_hour_fig = plot_win_rate_analysis(filtered_df, 'trade_hour', win_col='win', theme=plot_theme, title_prefix="Win Rate by")
                if winrate_hour_fig: st.plotly_chart(winrate_hour_fig, use_container_width=True)
            
            month_col_for_plot = 'trade_month_name'
            if month_col_for_plot in filtered_df.columns:
                pnl_month_fig = plot_pnl_by_category(filtered_df, month_col_for_plot, pnl_col=pnl_col, theme=plot_theme, title_prefix="Total PnL by")
                if pnl_month_fig: st.plotly_chart(pnl_month_fig, use_container_width=True)
    except Exception as e:
        logger.error(f"Error rendering performance breakdown: {e}", exc_info=True)
        display_custom_message(f"Error in performance breakdown: {e}", "error")
    st.markdown("---")

    # --- Rolling Performance ---
    # ... (Rolling Performance section remains the same) ...
    st.subheader("Rolling Performance Metrics")
    st.markdown("Rolling metrics ... highlight trends or changes in consistency.")    
    try:
        if date_col_main not in filtered_df.columns: display_custom_message(f"Date column ('{date_col_main}') required for rolling metrics.", "warning")
        else:
            col_roll1, col_roll2, col_roll3 = st.columns(3)
            with col_roll1: rolling_pnl_window = st.slider("PnL Window:", 10, 100, 30, 5, key="roll_pnl_win")
            with col_roll2: rolling_win_rate_window = st.slider("Win Rate Window:", 20, 200, 50, 10, key="roll_wr_win")
            with col_roll3: rolling_sharpe_window = st.slider("Sharpe Window:", 20, 200, 50, 10, key="roll_sharpe_win")
            
            min_periods_pnl = max(5, rolling_pnl_window // 3); min_periods_win_rate = max(10, rolling_win_rate_window // 3); min_periods_sharpe = max(10, rolling_sharpe_window // 3)
            if len(filtered_df) >= 10 :
                df_for_rolling = filtered_df.sort_values(by=date_col_main).copy()
                if not pd.api.types.is_numeric_dtype(df_for_rolling[pnl_col]):
                    df_for_rolling[pnl_col] = pd.to_numeric(df_for_rolling[pnl_col], errors='coerce'); df_for_rolling.dropna(subset=[pnl_col], inplace=True)
                if df_for_rolling.empty: display_custom_message("No valid numeric PnL for rolling metrics.", "warning")
                else:
                    if len(df_for_rolling) >= min_periods_pnl:
                        rolling_pnl_sum = df_for_rolling[pnl_col].rolling(window=rolling_pnl_window, min_periods=min_periods_pnl).sum()
                        rolling_pnl_metric_name = f"{rolling_pnl_window}-Period Rolling PnL Sum"
                        rolling_pnl_fig = plot_rolling_performance(df_for_rolling, date_col_main, rolling_pnl_sum, rolling_pnl_metric_name, rolling_pnl_metric_name, plot_theme)
                        if rolling_pnl_fig: st.plotly_chart(rolling_pnl_fig, use_container_width=True)
                    if 'win' in df_for_rolling.columns and len(df_for_rolling) >= min_periods_win_rate:
                        rolling_win_rate = df_for_rolling['win'].rolling(window=rolling_win_rate_window, min_periods=min_periods_win_rate).mean() * 100
                        rolling_wr_metric_name = f"{rolling_win_rate_window}-Period Rolling Win Rate (%)"
                        rolling_wr_fig = plot_rolling_performance(df_for_rolling, date_col_main, rolling_win_rate, rolling_wr_metric_name, rolling_wr_metric_name, plot_theme)
                        if rolling_wr_fig: st.plotly_chart(rolling_wr_fig, use_container_width=True)
                    daily_pnl_for_sharpe = df_for_rolling.groupby(df_for_rolling[date_col_main].dt.normalize())[pnl_col].sum()
                    if len(daily_pnl_for_sharpe) >= min_periods_sharpe:
                        rolling_sharpe = calculate_rolling_sharpe(daily_pnl_for_sharpe, rolling_sharpe_window, min_periods_sharpe, risk_free_rate_daily)
                        rolling_sharpe_metric_name = f"{rolling_sharpe_window}-Day Rolling Sharpe Ratio (Annualized)"
                        if not rolling_sharpe.empty:
                             rolling_sharpe_fig = plot_rolling_performance(pd.DataFrame(rolling_sharpe.index, columns=[date_col_main]), date_col_main, rolling_sharpe, rolling_sharpe_metric_name, rolling_sharpe_metric_name, plot_theme)
                             if rolling_sharpe_fig: st.plotly_chart(rolling_sharpe_fig, use_container_width=True)
            else: display_custom_message("Not enough data for rolling metrics.", "info")
    except Exception as e: logger.error(f"Error rolling metrics: {e}", exc_info=True); display_custom_message(f"Error in rolling performance: {e}", "error")
    st.markdown("---")

    # --- Performance by Trade Duration ---
    # ... (Performance by Trade Duration section remains the same) ...
    st.subheader("Performance by Trade Duration")
    duration_col_numeric = EXPECTED_COLUMNS.get('duration_minutes_numeric', 'duration_minutes_numeric')
    if duration_col_numeric in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[duration_col_numeric]):
        df_for_duration = filtered_df[[duration_col_numeric, pnl_col, 'win']].copy(); df_for_duration.dropna(subset=[duration_col_numeric], inplace=True)
        if not df_for_duration.empty:
            bins = [-float('inf'), 30, 60, 120, 240, 480, float('inf')]; labels = ["<30 min", "30-60 min", "1-2 hrs", "2-4 hrs", "4-8 hrs", ">8 hrs"]
            df_for_duration['duration_bin'] = pd.cut(df_for_duration[duration_col_numeric], bins=bins, labels=labels, right=False)
            df_for_duration.dropna(subset=['duration_bin'], inplace=True)
            if not df_for_duration.empty:
                col_dur1, col_dur2 = st.columns(2)
                with col_dur1:
                    pnl_by_duration_fig = plot_box_plot(df_for_duration, 'duration_bin', pnl_col, "PnL Distribution by Trade Duration", plot_theme)
                    if pnl_by_duration_fig: st.plotly_chart(pnl_by_duration_fig, use_container_width=True)
                with col_dur2:
                    if 'win' in df_for_duration.columns:
                        winrate_duration_fig = plot_win_rate_analysis(df_for_duration, 'duration_bin', 'win', "Win Rate by", plot_theme)
                        if winrate_duration_fig: st.plotly_chart(winrate_duration_fig, use_container_width=True)
            else: display_custom_message("No trades with valid duration data after binning.", "info")
        else: display_custom_message("No trades with valid duration data.", "info")
    else: display_custom_message(f"Numeric duration column ('{duration_col_numeric}') not found.", "warning")
    st.markdown("---")

    # --- NEW: Day/Hour Performance Matrix ---
    st.subheader("Day & Hour Performance Matrix")
    hour_col = 'trade_hour' # Engineered in data_processing.py
    dow_col = 'trade_day_of_week' # Engineered in data_processing.py
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    if hour_col in filtered_df.columns and dow_col in filtered_df.columns:
        col_matrix1, col_matrix2 = st.columns(2)
        with col_matrix1:
            pnl_agg_option = st.selectbox("PnL Aggregation for Heatmap:", ["Total PnL", "Average PnL"], key="heatmap_pnl_agg")
            pnl_agg_func = 'sum' if pnl_agg_option == "Total PnL" else 'mean'
        with col_matrix2:
            other_metric_option = st.selectbox("Second Heatmap Metric:", ["Win Rate %", "Trade Count"], key="heatmap_other_metric")

        try:
            # PnL Heatmap
            pnl_pivot = filtered_df.pivot_table(index=hour_col, columns=dow_col, values=pnl_col, aggfunc=pnl_agg_func, fill_value=0)
            if not pnl_pivot.empty:
                pnl_pivot = pnl_pivot.reindex(columns=[day for day in days_order if day in pnl_pivot.columns], fill_value=0) # Order columns
                pnl_pivot = pnl_pivot.sort_index() # Sort rows by hour
                pnl_heatmap_fig = plot_heatmap(
                    pnl_pivot, 
                    title=f"{pnl_agg_option} by Hour and Day of Week",
                    color_scale="RdYlGn" if pnl_agg_option == "Average PnL" else "Blues", # Different scales for sum vs avg
                    text_format="$,.0f" if pnl_agg_option == "Total PnL" else "$,.2f",
                    theme=plot_theme
                )
                if pnl_heatmap_fig: st.plotly_chart(pnl_heatmap_fig, use_container_width=True)
                else: display_custom_message(f"Could not generate {pnl_agg_option} heatmap.", "warning")
            else:
                display_custom_message(f"Not enough data for {pnl_agg_option} heatmap.", "info")

            # Second Metric Heatmap (Win Rate or Trade Count)
            if other_metric_option == "Win Rate %":
                if 'win' in filtered_df.columns:
                    win_rate_pivot = filtered_df.pivot_table(index=hour_col, columns=dow_col, values='win', aggfunc='mean', fill_value=0) * 100
                    if not win_rate_pivot.empty:
                        win_rate_pivot = win_rate_pivot.reindex(columns=[day for day in days_order if day in win_rate_pivot.columns], fill_value=0)
                        win_rate_pivot = win_rate_pivot.sort_index()
                        metric_heatmap_fig = plot_heatmap(win_rate_pivot, title="Win Rate (%) by Hour and Day of Week", color_scale="Greens", text_format=".1f", z_min=0, z_max=100, theme=plot_theme)
                        if metric_heatmap_fig: st.plotly_chart(metric_heatmap_fig, use_container_width=True)
                        else: display_custom_message("Could not generate Win Rate heatmap.", "warning")
                    else: display_custom_message("Not enough data for Win Rate heatmap.", "info")
                else: display_custom_message("Engineered 'win' column needed for Win Rate heatmap.", "warning")
            
            elif other_metric_option == "Trade Count":
                trade_count_pivot = filtered_df.pivot_table(index=hour_col, columns=dow_col, values=pnl_col, aggfunc='count', fill_value=0) # Use any column for count
                if not trade_count_pivot.empty:
                    trade_count_pivot = trade_count_pivot.reindex(columns=[day for day in days_order if day in trade_count_pivot.columns], fill_value=0)
                    trade_count_pivot = trade_count_pivot.sort_index()
                    metric_heatmap_fig = plot_heatmap(trade_count_pivot, title="Trade Count by Hour and Day of Week", color_scale="Blues", text_format=".0f", theme=plot_theme)
                    if metric_heatmap_fig: st.plotly_chart(metric_heatmap_fig, use_container_width=True)
                    else: display_custom_message("Could not generate Trade Count heatmap.", "warning")
                else: display_custom_message("Not enough data for Trade Count heatmap.", "info")

        except Exception as e_heatmap:
            logger.error(f"Error generating Day/Hour heatmaps: {e_heatmap}", exc_info=True)
            display_custom_message(f"An error occurred while generating Day/Hour heatmaps: {e_heatmap}", "error")
    else:
        display_custom_message(f"Engineered columns '{hour_col}' or '{dow_col}' not found. Cannot generate Day/Hour heatmaps.", "warning")
    st.markdown("---")

    # --- P&L Calendar View ---
    # ... (P&L Calendar section remains the same) ...
    if date_col_main in filtered_df.columns and pnl_col in filtered_df.columns:
        try:
            daily_pnl_df_agg = filtered_df.groupby(filtered_df[date_col_main].dt.normalize())[pnl_col].sum().reset_index()
            daily_pnl_df_agg = daily_pnl_df_agg.rename(columns={date_col_main: 'date', pnl_col: 'pnl'})
            available_years = sorted(daily_pnl_df_agg['date'].dt.year.unique(), reverse=True)
            if available_years:
                selected_year = st.selectbox("Select Year for P&L Calendar:", options=available_years, index=0, key="perf_page_calendar_year_select")
                if selected_year:
                    calendar_component = PnLCalendarComponent(daily_pnl_df=daily_pnl_df_agg, year=selected_year, plot_theme=plot_theme)
                    calendar_component.render()
            else: display_custom_message("No yearly data for P&L calendar.", "info")
        except Exception as e: logger.error(f"Error for P&L Calendar: {e}", exc_info=True); display_custom_message(f"Could not generate P&L Calendar: {e}", "error")
    else: display_custom_message(f"Required columns ('{date_col_main}', '{pnl_col}') not found for P&L Calendar.", "warning")


if __name__ == "__main__":
    if 'app_initialized' not in st.session_state:
        st.warning("This page is part of a multi-page app. Please run the main app.py script.")
    show_performance_page()
