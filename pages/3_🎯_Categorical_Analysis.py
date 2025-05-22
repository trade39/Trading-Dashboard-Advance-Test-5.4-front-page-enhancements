# pages/3_🎯_Categorical_Analysis.py

import streamlit as st
import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Tuple, Dict, Any, Callable
import plotly.express as px

# --- Configuration and Utility Imports ---
try:
    from config import APP_TITLE, EXPECTED_COLUMNS, COLORS, PLOTLY_THEME_DARK, PLOTLY_THEME_LIGHT, CONFIDENCE_LEVEL, BOOTSTRAP_ITERATIONS
    from utils.common_utils import display_custom_message, format_currency, format_percentage
    from services.statistical_analysis_service import StatisticalAnalysisService
except ImportError as e:
    st.error(f"Categorical Analysis Page Error: Critical config/utils/service import failed: {e}.")
    APP_TITLE = "TradingDashboard_Error" # Fallback
    EXPECTED_COLUMNS = {"pnl": "pnl_fallback", "date": "date_fallback", "strategy": "strategy_fallback", "market_conditions_str": "market_conditions_fallback", "r_r_csv_num": "r_r_fallback", "direction_str": "direction_fallback"}
    COLORS = {"green": "#00FF00", "red": "#FF0000", "gray": "#808080"}
    PLOTLY_THEME_DARK = "plotly_dark"; PLOTLY_THEME_LIGHT = "plotly_white"
    CONFIDENCE_LEVEL = 0.95; BOOTSTRAP_ITERATIONS = 1000
    def display_custom_message(msg, type="error"): st.error(msg) # Fallback
    def format_currency(val): return f"${val:,.2f}" # Fallback
    def format_percentage(val): return f"{val:.2%}" # Fallback
    class StatisticalAnalysisService: # Fallback
        def calculate_bootstrap_ci(self, *args, **kwargs): return {"error": "Bootstrap CI function not loaded in service.", "lower_bound": np.nan, "upper_bound": np.nan, "observed_statistic": np.nan, "bootstrap_statistics": []}
        def run_hypothesis_test(self, *args, **kwargs): return {"error": "Hypothesis test function not loaded in service."}
    logger = logging.getLogger("CategoricalAnalysisPage_Fallback")
    logger.error(f"CRITICAL IMPORT ERROR (Config/Utils/Service) in Categorical Analysis Page: {e}", exc_info=True)
    st.stop()

# --- Plotting and Component Imports ---
try:
    from plotting import (
        _apply_custom_theme, plot_pnl_by_category, plot_stacked_bar_chart, plot_heatmap,
        plot_value_over_time, plot_grouped_bar_chart, plot_box_plot, plot_donut_chart,
        plot_radar_chart, plot_scatter_plot, plot_pnl_distribution, plot_win_rate_analysis
    )
    from components.calendar_view import PnLCalendarComponent
except ImportError as e:
    st.error(f"Categorical Analysis Page Error: Critical plotting/component import failed: {e}.")
    logger = logging.getLogger(APP_TITLE if 'APP_TITLE' in globals() else "FallbackApp") 
    logger.error(f"CRITICAL IMPORT ERROR (Plotting/Components) in Categorical Analysis Page: {e}", exc_info=True)
    def _apply_custom_theme(fig, theme): return fig # Fallback
    def plot_pnl_by_category(*args, **kwargs): return None # Fallback
    def plot_stacked_bar_chart(*args, **kwargs): return None # Fallback
    def plot_heatmap(*args, **kwargs): return None # Fallback
    def plot_value_over_time(*args, **kwargs): return None # Fallback
    def plot_grouped_bar_chart(*args, **kwargs): return None # Fallback
    def plot_box_plot(*args, **kwargs): return None # Fallback
    def plot_donut_chart(*args, **kwargs): return None # Fallback
    def plot_radar_chart(*args, **kwargs): return None # Fallback
    def plot_scatter_plot(*args, **kwargs): return None # Fallback
    def plot_pnl_distribution(*args, **kwargs): return None # Fallback
    def plot_win_rate_analysis(*args, **kwargs): return None # Fallback
    class PnLCalendarComponent:  # Fallback
        def __init__(self, *args, **kwargs): pass
        def render(self): st.warning("Calendar component could not be loaded.")
    st.stop()


logger = logging.getLogger(APP_TITLE)
statistical_service = StatisticalAnalysisService()

def get_column_name(conceptual_key: str, df_columns: Optional[pd.Index] = None) -> Optional[str]:
    """
    Retrieves the actual column name from the DataFrame based on a conceptual key defined in EXPECTED_COLUMNS.
    If the conceptual_key itself is a column, it's returned.
    Logs a warning if the mapped column is not found.
    """
    if df_columns is not None and conceptual_key in df_columns:
        return conceptual_key
    actual_col = EXPECTED_COLUMNS.get(conceptual_key)
    if df_columns is not None and actual_col and actual_col not in df_columns:
        logger.warning(f"Conceptual key '{conceptual_key}' maps to '{actual_col}', but it's not in DataFrame columns: {df_columns.tolist()}")
        return None
    return actual_col

PERFORMANCE_TABLE_SELECTABLE_CATEGORIES: Dict[str, str] = {
    'entry_time_str': 'Entry Time (Raw String)', 'trade_hour': 'Trade Hour',
    'trade_day_of_week': 'Day of Week', 'trade_month_name': 'Month',
    'symbol': 'Symbol', 'strategy': 'Trade Model', 'trade_plan_str': 'Trade Plan',
    'bias_str': 'Bias', 'time_frame_str': 'Time Frame', 'direction_str': 'Direction',
    'r_r_csv_num': 'R:R (from CSV)', 'session_str': 'Session',
    'market_conditions_str': 'Market Conditions', 'events_details_str': 'Events Details',
    'psychological_factors_str': 'Psychological Factors', 'account_str': 'Account',
    'exit_type_csv_str': 'Exit Type',
    'event_type_str': 'Event Type', # Added for Market Context
    'market_sentiment_str': 'Market Sentiment', # Added for Market Context
    'compliance_check_str': 'Compliance Check', # Added for Behavioral
    'initial_balance_num': 'Initial Balance', # Added for Capital/Risk
    'drawdown_value_csv': 'Drawdown Value (CSV)' # Added for Capital/Risk
}

def calculate_performance_summary_by_category(
    df: pd.DataFrame, category_col: str, pnl_col: str, win_col: str,
    calculate_cis_for: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculates a performance summary DataFrame grouped by a specified category.
    Includes metrics like Total PnL, Win Rate, Expectancy, and optional Bootstrap CIs.
    """
    if calculate_cis_for is None:
        calculate_cis_for = []

    if category_col not in df.columns or pnl_col not in df.columns or win_col not in df.columns:
        logger.error(f"Missing required columns for performance summary: category='{category_col}', pnl='{pnl_col}', win='{win_col}'")
        return pd.DataFrame()

    df_copy = df.copy()
    # Ensure win_col is boolean. If not, try to derive it from pnl_col.
    if not pd.api.types.is_bool_dtype(df_copy[win_col]):
        if pd.api.types.is_numeric_dtype(df_copy[pnl_col]):
            logger.info(f"Win column '{win_col}' is not boolean, creating it from '{pnl_col}'.")
            df_copy[win_col] = df_copy[pnl_col] > 0
        else:
            logger.error(f"Cannot create boolean win column as PnL column '{pnl_col}' is not numeric.")
            return pd.DataFrame()

    df_grouped = df_copy.fillna({category_col: 'N/A'}).groupby(category_col, observed=False)
    summary_data = []

    for name_of_group, group_df in df_grouped:
        total_trades = len(group_df)
        if total_trades == 0: continue

        total_pnl = group_df[pnl_col].sum()
        avg_pnl = group_df[pnl_col].mean()

        num_wins = group_df[win_col].sum()
        num_losses = total_trades - num_wins
        win_rate_pct = (num_wins / total_trades) * 100 if total_trades > 0 else 0.0

        avg_pnl_ci_lower, avg_pnl_ci_upper = np.nan, np.nan
        win_rate_ci_lower, win_rate_ci_upper = np.nan, np.nan

        # Calculate CIs only if enough data and requested
        if total_trades >= 10: # Minimum trades for meaningful CI
            try:
                if "Average PnL" in calculate_cis_for:
                    avg_pnl_bs_results = statistical_service.calculate_bootstrap_ci(
                        data_series=group_df[pnl_col], statistic_func=np.mean,
                        n_iterations=BOOTSTRAP_ITERATIONS // 4, # Reduced iterations for faster table load
                        confidence_level=CONFIDENCE_LEVEL
                    )
                    if 'error' not in avg_pnl_bs_results:
                        avg_pnl_ci_lower = avg_pnl_bs_results['lower_bound']
                        avg_pnl_ci_upper = avg_pnl_bs_results['upper_bound']

                if "Win Rate %" in calculate_cis_for:
                    # Statistic function for win rate (percentage)
                    win_rate_stat_func = lambda x_series: (np.sum(x_series > 0) / len(x_series)) * 100 if len(x_series) > 0 else 0.0
                    data_for_win_rate_bs = group_df[pnl_col] # Use PnL to determine win/loss for bootstrapping

                    win_rate_bs_results = statistical_service.calculate_bootstrap_ci(
                        data_series=data_for_win_rate_bs, statistic_func=win_rate_stat_func,
                        n_iterations=BOOTSTRAP_ITERATIONS // 4, # Reduced iterations
                        confidence_level=CONFIDENCE_LEVEL
                    )
                    if 'error' not in win_rate_bs_results:
                        win_rate_ci_lower = win_rate_bs_results['lower_bound']
                        win_rate_ci_upper = win_rate_bs_results['upper_bound']
            except Exception as e_bs:
                logger.warning(f"Error during bootstrapping for group '{name_of_group}': {e_bs}")

        loss_rate_pct = (num_losses / total_trades) * 100 if total_trades > 0 else 0.0
        wins_df = group_df[group_df[win_col]]
        losses_df = group_df[~group_df[win_col] & (group_df[pnl_col] < 0)] # Ensure losses are actually < 0 PnL
        avg_win_amount = wins_df[pnl_col].sum() / num_wins if num_wins > 0 else 0.0
        avg_loss_amount = abs(losses_df[pnl_col].sum()) / num_losses if num_losses > 0 else 0.0
        expectancy = (avg_win_amount * (win_rate_pct / 100.0)) - (avg_loss_amount * (loss_rate_pct / 100.0))

        summary_data.append({
            "Category Group": name_of_group, "Total PnL": total_pnl, "Total Trades": total_trades,
            "Win Rate %": win_rate_pct, "Expectancy $": expectancy, "Average PnL": avg_pnl,
            "Avg PnL CI": f"[{avg_pnl_ci_lower:,.2f}, {avg_pnl_ci_upper:,.2f}]" if pd.notna(avg_pnl_ci_lower) and pd.notna(avg_pnl_ci_upper) else "N/A",
            "Win Rate % CI": f"[{win_rate_ci_lower:.1f}%, {win_rate_ci_upper:.1f}%]" if pd.notna(win_rate_ci_lower) and pd.notna(win_rate_ci_upper) else "N/A"
        })
    summary_df = pd.DataFrame(summary_data)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(by="Total PnL", ascending=False)
    return summary_df


def show_categorical_analysis_page():
    st.title("🎯 Categorical Performance Analysis")
    logger.info("Rendering Categorical Analysis Page.")

    if 'filtered_data' not in st.session_state or st.session_state.filtered_data is None:
        display_custom_message("Please upload and process data to view categorical analysis.", "info")
        return

    df = st.session_state.filtered_data
    plot_theme = st.session_state.get('current_theme', PLOTLY_THEME_DARK) # Default to dark if not set

    # --- Get Actual Column Names ---
    pnl_col_actual = get_column_name('pnl', df.columns)
    win_col_actual = 'win' # This is an engineered column, should exist after processing
    trade_result_col_actual = 'trade_result_processed' # Engineered column
    date_col_actual = get_column_name('date', df.columns)


    # --- Initial Data Checks ---
    if df.empty:
        display_custom_message("No data matches the current filters. Cannot perform categorical analysis.", "info")
        return
    if not pnl_col_actual:
        display_custom_message("Essential PnL column not found. Analysis cannot proceed.", "error")
        return
    if win_col_actual not in df.columns:
        logger.warning(f"Engineered Win column ('{win_col_actual}') not found. Some analyses may be affected or require PnL > 0 logic.")
    if trade_result_col_actual not in df.columns:
        logger.warning(f"Engineered Trade Result column ('{trade_result_col_actual}') not found. Some analyses may be affected.")
    if not date_col_actual:
        logger.warning(f"Date column ('{date_col_actual}') not found. Some temporal analyses may be affected.")

    # --- Define Tab Names ---
    tab_names = [
        "💡 Strategy Insights",
        "⏳ Temporal Analysis",
        "🌍 Market Context",
        "🤔 Behavioral Factors",
        "💰 Capital & Risk",
        "🚪 Exit & Direction",
        "📊 Custom Summary",
        "🔬 Dynamic Visualizer"
    ]
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(tab_names)

    # --- Tab 1: Strategy Performance Insights ---
    with tab1:
        st.subheader("Strategy Performance Metrics")
        st.markdown("<div class='performance-section-container'>", unsafe_allow_html=True)
        with st.expander("Strategy Metrics Details", expanded=False):
            st.markdown("<div class='charts-grid'>", unsafe_allow_html=True)
            col1a, col1b = st.columns(2)
            with col1a:
                strategy_col_key = 'strategy'
                strategy_col_actual = get_column_name(strategy_col_key, df.columns)
                if strategy_col_actual and pnl_col_actual:
                    avg_pnl_strategy_data = df.groupby(strategy_col_actual, observed=False)[pnl_col_actual].mean().reset_index()
                    avg_pnl_strategy_data = avg_pnl_strategy_data.sort_values(by=pnl_col_actual, ascending=False)
                    fig_avg_pnl_strategy = plot_pnl_by_category(
                        df=avg_pnl_strategy_data, 
                        category_col=strategy_col_actual, 
                        pnl_col=pnl_col_actual, 
                        title_prefix="Average PnL by", aggregation_func='mean', 
                        theme=plot_theme, is_data_aggregated=True
                    )
                    if fig_avg_pnl_strategy: st.plotly_chart(fig_avg_pnl_strategy, use_container_width=True)
                    if not avg_pnl_strategy_data.empty:
                        if st.checkbox("View Data: Average PnL by Strategy", key="view_data_s1_c1_tab"):
                            st.dataframe(avg_pnl_strategy_data, use_container_width=True, hide_index=True)
            with col1b:
                trade_plan_col_key = 'trade_plan_str'
                trade_plan_col_actual = get_column_name(trade_plan_col_key, df.columns)
                if trade_plan_col_actual and trade_result_col_actual in df.columns :
                    result_by_plan_data = pd.crosstab(df[trade_plan_col_actual].fillna('N/A'), df[trade_result_col_actual].fillna('N/A'))
                    for res_col in ['WIN', 'LOSS', 'BREAKEVEN']: # Ensure all result columns exist
                        if res_col not in result_by_plan_data.columns: result_by_plan_data[res_col] = 0
                    result_by_plan_data = result_by_plan_data[['WIN', 'LOSS', 'BREAKEVEN']] 

                    fig_result_by_plan = plot_stacked_bar_chart(
                        df=result_by_plan_data.reset_index(), 
                        category_col=trade_plan_col_actual, 
                        stack_cols=['WIN', 'LOSS', 'BREAKEVEN'], 
                        title=f"{trade_result_col_actual.replace('_',' ').title()} by {PERFORMANCE_TABLE_SELECTABLE_CATEGORIES.get(trade_plan_col_key, trade_plan_col_key).replace('_',' ').title()}",
                        theme=plot_theme,
                        color_discrete_map={'WIN': COLORS.get('green'), 'LOSS': COLORS.get('red'), 'BREAKEVEN': COLORS.get('gray')},
                        is_data_aggregated=True 
                    )
                    if fig_result_by_plan: st.plotly_chart(fig_result_by_plan, use_container_width=True)
                    if not result_by_plan_data.empty:
                        if st.checkbox(f"View Data: {trade_result_col_actual.replace('_',' ').title()} by Trade Plan", key="view_data_s1_c2_tab"):
                            st.dataframe(result_by_plan_data.reset_index(), use_container_width=True, hide_index=True)

            st.markdown("</div>", unsafe_allow_html=True) # Close charts-grid
            st.markdown("---") 

            rr_col_key = 'r_r_csv_num'; direction_col_key = 'direction_str'; strategy_col_key_for_rr = 'strategy'
            rr_col_actual = get_column_name(rr_col_key, df.columns)
            direction_col_actual = get_column_name(direction_col_key, df.columns)
            strategy_col_actual_for_rr = get_column_name(strategy_col_key_for_rr, df.columns)
            if all(c is not None and c in df.columns for c in [strategy_col_actual_for_rr, rr_col_actual, direction_col_actual]):
                try:
                    df_rr_heatmap_prep = df[[strategy_col_actual_for_rr, rr_col_actual, direction_col_actual]].copy()
                    df_rr_heatmap_prep[rr_col_actual] = pd.to_numeric(df_rr_heatmap_prep[rr_col_actual], errors='coerce')
                    df_rr_heatmap_cleaned = df_rr_heatmap_prep.dropna(subset=[rr_col_actual, strategy_col_actual_for_rr, direction_col_actual])
                    pivot_rr_data = pd.DataFrame() 
                    if not df_rr_heatmap_cleaned.empty and df_rr_heatmap_cleaned[strategy_col_actual_for_rr].nunique() >= 1 and df_rr_heatmap_cleaned[direction_col_actual].nunique() >= 1:
                        pivot_rr_data = pd.pivot_table(df_rr_heatmap_cleaned, values=rr_col_actual, index=[strategy_col_actual_for_rr, direction_col_actual], aggfunc='mean').unstack(level=-1)
                        if isinstance(pivot_rr_data.columns, pd.MultiIndex): pivot_rr_data.columns = pivot_rr_data.columns.droplevel(0)

                        if not pivot_rr_data.empty:
                            fig_rr_heatmap = plot_heatmap(df_pivot=pivot_rr_data, title=f"Average R:R by Strategy and Direction", color_scale="Viridis", theme=plot_theme, text_format=".2f")
                            if fig_rr_heatmap: st.plotly_chart(fig_rr_heatmap, use_container_width=True)
                            if st.checkbox("View Data: Average R:R by Strategy and Direction", key="view_data_s1_c3_tab"):
                                st.dataframe(pivot_rr_data.reset_index(), use_container_width=True, hide_index=True)
                except Exception as e_rr_heatmap: logger.error(f"Error in R:R Heatmap: {e_rr_heatmap}", exc_info=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Tab 2: Temporal Analysis ---
    with tab2:
        st.subheader("Time-Based Performance")
        st.markdown("<div class='performance-section-container'>", unsafe_allow_html=True)
        with st.expander("Time-Based Metrics Details", expanded=False):
            st.markdown("<div class='charts-grid'>", unsafe_allow_html=True)
            col2a, col2b = st.columns(2)
            with col2a:
                month_num_col_actual = get_column_name('trade_month_num', df.columns)
                month_name_col_actual = get_column_name('trade_month_name', df.columns)
                if month_num_col_actual and month_name_col_actual and win_col_actual in df.columns:
                    try:
                        monthly_win_rate_series = df.groupby(month_num_col_actual)[win_col_actual].mean() * 100
                        month_map_df = df[[month_num_col_actual, month_name_col_actual]].drop_duplicates().sort_values(month_num_col_actual)
                        month_mapping = pd.Series(month_map_df[month_name_col_actual].values, index=month_map_df[month_num_col_actual]).to_dict()
                        # Ensure all months are present, even if no trades, for consistent sorting
                        all_months_num = sorted(month_mapping.keys())
                        monthly_win_rate_data = monthly_win_rate_series.reindex(all_months_num, fill_value=0).rename(index=month_mapping)


                        if not monthly_win_rate_data.empty:
                            fig_monthly_wr = plot_value_over_time(series=monthly_win_rate_data, series_name="Monthly Win Rate", title="Win Rate by Month", x_axis_title="Month", y_axis_title="Win Rate (%)", theme=plot_theme)
                            if fig_monthly_wr: st.plotly_chart(fig_monthly_wr, use_container_width=True)
                            if st.checkbox("View Data: Win Rate by Month", key="view_data_s2_c1_tab"):
                                st.dataframe(monthly_win_rate_data.reset_index().rename(columns={'index': 'Month', month_num_col_actual: "Win Rate (%)"}), use_container_width=True, hide_index=True)
                    except Exception as e_mwr: logger.error(f"Error in Monthly Win Rate: {e_mwr}", exc_info=True)
            with col2b:
                session_col_key = 'session_str'; time_frame_col_key = 'time_frame_str'
                session_col_actual = get_column_name(session_col_key, df.columns)
                time_frame_col_actual = get_column_name(time_frame_col_key, df.columns)
                if session_col_actual and time_frame_col_actual and trade_result_col_actual in df.columns:
                    try:
                        count_df_agg = df.groupby([session_col_actual, time_frame_col_actual, trade_result_col_actual], observed=False).size().reset_index(name='count')
                        # Using trade_result_col_actual for values if you want to sum counts of wins/losses etc.
                        # If just total trades, then 'count' is fine. For heatmap, often total count is good.
                        pivot_session_tf_data = count_df_agg.pivot_table(index=session_col_actual, columns=time_frame_col_actual, values='count', fill_value=0, aggfunc='sum')

                        if not pivot_session_tf_data.empty:
                            fig_session_tf_heatmap = plot_heatmap(df_pivot=pivot_session_tf_data, title=f"Trade Count by Session and Time Frame", color_scale="Blues", theme=plot_theme, text_format=".0f")
                            if fig_session_tf_heatmap: st.plotly_chart(fig_session_tf_heatmap, use_container_width=True)
                            if st.checkbox("View Data: Trade Count by Session and Time Frame", key="view_data_s2_c2_tab"):
                                 st.dataframe(pivot_session_tf_data.reset_index(), use_container_width=True, hide_index=True)
                    except Exception as e_sess_tf: logger.error(f"Error in Session/TF Heatmap: {e_sess_tf}", exc_info=True)
            st.markdown("</div>", unsafe_allow_html=True) 
            st.markdown("---") 

            if date_col_actual and pnl_col_actual: 
                try:
                    # Ensure date_col_actual is datetime
                    df[date_col_actual] = pd.to_datetime(df[date_col_actual], errors='coerce')
                    daily_pnl_df_agg = df.dropna(subset=[date_col_actual]).groupby(df[date_col_actual].dt.normalize())[pnl_col_actual].sum().reset_index()
                    daily_pnl_df_agg = daily_pnl_df_agg.rename(columns={date_col_actual: 'date', pnl_col_actual: 'pnl'})
                    available_years = sorted(daily_pnl_df_agg['date'].dt.year.unique(), reverse=True)
                    if available_years:
                        selected_year = st.selectbox("Select Year for P&L Calendar:", options=available_years, index=0, key="cat_analysis_calendar_year_select_tab")
                        if selected_year:
                            st.markdown("<div class='calendar-display-area'>", unsafe_allow_html=True)
                            calendar_component = PnLCalendarComponent(daily_pnl_df=daily_pnl_df_agg, year=selected_year, plot_theme=plot_theme)
                            calendar_component.render()
                            st.markdown("</div>", unsafe_allow_html=True)
                except Exception as e_cal: logger.error(f"Error in P&L Calendar: {e_cal}", exc_info=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Tab 3: Market Context Impact ---
    with tab3:
        st.subheader("Market Condition Effects")
        st.markdown("<div class='performance-section-container'>", unsafe_allow_html=True)
        with st.expander("Market Context Details", expanded=False):
            st.markdown("<div class='charts-grid'>", unsafe_allow_html=True)
            col3a, col3b = st.columns(2)
            with col3a:
                event_type_col_key = 'event_type_str'
                event_type_col_actual = get_column_name(event_type_col_key, df.columns)
                if event_type_col_actual and trade_result_col_actual in df.columns:
                    result_by_event_data = df.groupby([event_type_col_actual, trade_result_col_actual], observed=False).size().reset_index(name='count')
                    fig_result_by_event = plot_grouped_bar_chart(
                        df=result_by_event_data, 
                        category_col=event_type_col_actual,
                        value_col='count', 
                        group_col=trade_result_col_actual,
                        title=f"{trade_result_col_actual.replace('_',' ').title()} Count by {PERFORMANCE_TABLE_SELECTABLE_CATEGORIES.get(event_type_col_key, event_type_col_key).replace('_',' ').title()}",
                        theme=plot_theme,
                        color_discrete_map={'WIN': COLORS.get('green'), 'LOSS': COLORS.get('red'), 'BREAKEVEN': COLORS.get('gray')},
                        is_data_aggregated=True
                    )
                    if fig_result_by_event: st.plotly_chart(fig_result_by_event, use_container_width=True)
                    if not result_by_event_data.empty:
                        if st.checkbox(f"View Data: {trade_result_col_actual.replace('_',' ').title()} Count by Event Type", key="view_data_s3_c1_tab"):
                            st.dataframe(result_by_event_data, use_container_width=True, hide_index=True)
            with col3b: 
                market_cond_col_key = 'market_conditions_str'
                market_cond_col_actual = get_column_name(market_cond_col_key, df.columns)
                if market_cond_col_actual and pnl_col_actual:
                    fig_pnl_by_market = plot_box_plot(
                        df=df, category_col=market_cond_col_actual, value_col=pnl_col_actual,
                        title=f"PnL Distribution by {PERFORMANCE_TABLE_SELECTABLE_CATEGORIES.get(market_cond_col_key, market_cond_col_key).replace('_',' ').title()}", theme=plot_theme
                    )
                    if fig_pnl_by_market: st.plotly_chart(fig_pnl_by_market, use_container_width=True)
                    if st.checkbox(f"Show Summary Statistics for PnL by Market Condition", key="cb_market_cond_stats_tab"):
                        market_cond_pnl_summary = df.groupby(market_cond_col_actual, observed=False)[pnl_col_actual].describe()
                        st.dataframe(market_cond_pnl_summary, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("---") 

            market_sent_col_key = 'market_sentiment_str'
            market_sent_col_actual = get_column_name(market_sent_col_key, df.columns)
            if market_sent_col_actual and win_col_actual in df.columns:
                try:
                    sentiment_win_rate_data = df.groupby(market_sent_col_actual, observed=False)[win_col_actual].mean().reset_index()
                    sentiment_win_rate_data[win_col_actual] *= 100
                    if not sentiment_win_rate_data.empty:
                        fig_sent_wr = px.bar(sentiment_win_rate_data, x=market_sent_col_actual, y=win_col_actual,
                                             title=f"Win Rate by {PERFORMANCE_TABLE_SELECTABLE_CATEGORIES.get(market_sent_col_key, market_sent_col_key).replace('_',' ').title()}",
                                             labels={win_col_actual: "Win Rate (%)", market_sent_col_actual: PERFORMANCE_TABLE_SELECTABLE_CATEGORIES.get(market_sent_col_key, market_sent_col_key).replace('_',' ').title()},
                                             color=win_col_actual, color_continuous_scale="Greens")
                        if fig_sent_wr: fig_sent_wr.update_yaxes(ticksuffix="%")
                        if fig_sent_wr: st.plotly_chart(_apply_custom_theme(fig_sent_wr, plot_theme), use_container_width=True)
                        if st.checkbox("View Data: Win Rate by Market Sentiment", key="view_data_s3_c3_tab"):
                            st.dataframe(sentiment_win_rate_data.rename(columns={win_col_actual: "Win Rate (%)"}), use_container_width=True, hide_index=True)
                except Exception as e_sent_wr: logger.error(f"Error generating Market Sentiment vs Win Rate: {e_sent_wr}", exc_info=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Tab 4: Behavioral Factors ---
    with tab4:
        st.subheader("Trader Psychology & Compliance")
        st.markdown("<div class='performance-section-container'>", unsafe_allow_html=True)
        with st.expander("Behavioral Metrics Details", expanded=False):
            st.markdown("<div class='charts-grid'>", unsafe_allow_html=True)
            col4a, col4b = st.columns(2)
            with col4a:
                psych_col_key = 'psychological_factors_str'
                psych_col_actual = get_column_name(psych_col_key, df.columns)
                if psych_col_actual and trade_result_col_actual in df.columns:
                    df_psych = df.copy()
                    # Handle potential list-like strings for psychological factors (take first element)
                    if df_psych[psych_col_actual].dtype == 'object':
                        df_psych[psych_col_actual] = df_psych[psych_col_actual].astype(str).str.split(',').str[0].str.strip().fillna('N/A')
                    
                    psych_result_data = pd.crosstab(df_psych[psych_col_actual], df_psych[trade_result_col_actual])
                    for res_col in ['WIN', 'LOSS', 'BREAKEVEN']: # Ensure all result columns exist
                        if res_col not in psych_result_data.columns: psych_result_data[res_col] = 0
                    psych_result_data = psych_result_data[['WIN', 'LOSS', 'BREAKEVEN']]

                    fig_psych_result = plot_stacked_bar_chart(
                        df=psych_result_data.reset_index(), 
                        category_col=psych_col_actual,
                        stack_cols=['WIN', 'LOSS', 'BREAKEVEN'],
                        title=f"{trade_result_col_actual.replace('_',' ').title()} by Dominant {PERFORMANCE_TABLE_SELECTABLE_CATEGORIES.get(psych_col_key, psych_col_key).replace('_',' ').title()}",
                        theme=plot_theme,
                        color_discrete_map={'WIN': COLORS.get('green'), 'LOSS': COLORS.get('red'), 'BREAKEVEN': COLORS.get('gray')},
                        is_data_aggregated=True
                    )
                    if fig_psych_result: st.plotly_chart(fig_psych_result, use_container_width=True)
                    if not psych_result_data.empty:
                        if st.checkbox("View Data: Trade Result by Dominant Psychological Factor", key="view_data_s4_c1_tab"):
                            st.dataframe(psych_result_data.reset_index(), use_container_width=True, hide_index=True)
            with col4b:
                compliance_col_key = 'compliance_check_str'
                compliance_col_actual = get_column_name(compliance_col_key, df.columns)
                if compliance_col_actual:
                    compliance_data = df[compliance_col_actual].fillna('N/A').value_counts().reset_index()
                    compliance_data.columns = [compliance_col_actual, 'count']
                    fig_compliance = plot_donut_chart(
                        df=compliance_data, 
                        category_col=compliance_col_actual,
                        value_col='count',
                        title=f"{PERFORMANCE_TABLE_SELECTABLE_CATEGORIES.get(compliance_col_key, compliance_col_key).replace('_',' ').title()} Outcomes", theme=plot_theme,
                        is_data_aggregated=True
                    )
                    if fig_compliance: st.plotly_chart(fig_compliance, use_container_width=True)
                    if not compliance_data.empty:
                        if st.checkbox("View Data: Compliance Outcomes", key="view_data_s4_c2_tab"):
                            st.dataframe(compliance_data, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Tab 5: Capital & Risk Insights ---
    with tab5:
        st.subheader("Capital Management and Drawdown")
        st.markdown("<div class='performance-section-container'>", unsafe_allow_html=True)
        with st.expander("Capital & Risk Details", expanded=False):
            st.markdown("<div class='charts-grid'>", unsafe_allow_html=True)
            col5a, col5b = st.columns(2)
            with col5a: 
                initial_bal_col_key = 'initial_balance_num'
                drawdown_csv_col_key = 'drawdown_value_csv'
                initial_bal_col_actual = get_column_name(initial_bal_col_key, df.columns)
                drawdown_csv_col_actual = get_column_name(drawdown_csv_col_key, df.columns)
                if initial_bal_col_actual and drawdown_csv_col_actual and trade_result_col_actual in df.columns:
                    scatter_data_cols = [initial_bal_col_actual, drawdown_csv_col_actual, trade_result_col_actual]
                    scatter_df_view = df[scatter_data_cols].dropna()
                    fig_bal_dd = plot_scatter_plot(
                        df=scatter_df_view, x_col=initial_bal_col_actual, y_col=drawdown_csv_col_actual, color_col=trade_result_col_actual,
                        title=f"{PERFORMANCE_TABLE_SELECTABLE_CATEGORIES.get(drawdown_csv_col_key, drawdown_csv_col_key).replace('_',' ').title()} vs. {PERFORMANCE_TABLE_SELECTABLE_CATEGORIES.get(initial_bal_col_key, initial_bal_col_key).replace('_',' ').title()}",
                        theme=plot_theme,
                        color_discrete_map={'WIN': COLORS.get('green'), 'LOSS': COLORS.get('red'), 'BREAKEVEN': COLORS.get('gray')}
                    )
                    if fig_bal_dd: st.plotly_chart(fig_bal_dd, use_container_width=True)
                    if not scatter_df_view.empty:
                        if st.checkbox("View Data: Drawdown vs. Initial Balance", key="view_data_s5_c1_tab"):
                            st.dataframe(scatter_df_view, use_container_width=True, hide_index=True)
            with col5b:
                trade_plan_col_key_dd = 'trade_plan_str'
                drawdown_csv_col_key_avg = 'drawdown_value_csv' 
                trade_plan_col_actual_dd = get_column_name(trade_plan_col_key_dd, df.columns)
                drawdown_csv_col_actual_avg = get_column_name(drawdown_csv_col_key_avg, df.columns)
                if trade_plan_col_actual_dd and drawdown_csv_col_actual_avg:
                    avg_dd_plan_data = df.groupby(trade_plan_col_actual_dd, observed=False)[drawdown_csv_col_actual_avg].mean().reset_index()
                    avg_dd_plan_data = avg_dd_plan_data.sort_values(by=drawdown_csv_col_actual_avg, ascending=True) # Lower drawdown is better
                    fig_avg_dd_plan = plot_pnl_by_category( # Reusing this for a generic bar chart
                        df=avg_dd_plan_data, 
                        category_col=trade_plan_col_actual_dd,
                        pnl_col=drawdown_csv_col_actual_avg, 
                        title_prefix="Average Drawdown by", aggregation_func='mean', # 'mean' here refers to how data was prepared
                        theme=plot_theme, is_data_aggregated=True,
                        y_axis_title="Average Drawdown"
                    )
                    if fig_avg_dd_plan: st.plotly_chart(fig_avg_dd_plan, use_container_width=True)
                    if not avg_dd_plan_data.empty:
                        if st.checkbox("View Data: Average Drawdown by Trade Plan", key="view_data_s5_c2_tab"):
                            st.dataframe(avg_dd_plan_data, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("---") 
            drawdown_csv_col_key_hist = 'drawdown_value_csv'
            drawdown_csv_col_actual_hist = get_column_name(drawdown_csv_col_key_hist, df.columns)
            if drawdown_csv_col_actual_hist:
                df_dd_hist = df[[drawdown_csv_col_actual_hist]].copy() 
                df_dd_hist[drawdown_csv_col_actual_hist] = pd.to_numeric(df_dd_hist[drawdown_csv_col_actual_hist], errors='coerce')
                df_dd_hist.dropna(subset=[drawdown_csv_col_actual_hist], inplace=True)
                if not df_dd_hist.empty:
                    fig_dd_hist = plot_pnl_distribution( # Reusing for drawdown distribution
                        df=df_dd_hist, pnl_col=drawdown_csv_col_actual_hist, title=f"{PERFORMANCE_TABLE_SELECTABLE_CATEGORIES.get(drawdown_csv_col_key_hist, drawdown_csv_col_key_hist).replace('_',' ').title()} Distribution",
                        theme=plot_theme, nbins=30, x_axis_title="Drawdown Value"
                    )
                    if fig_dd_hist: st.plotly_chart(fig_dd_hist, use_container_width=True)
                    if st.checkbox("View Data: Drawdown Distribution (raw values)", key="view_data_s5_c3_tab"):
                        st.dataframe(df_dd_hist.rename(columns={drawdown_csv_col_actual_hist: "Drawdown Value"}), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Tab 6: Exit & Directional Insights ---
    with tab6:
        st.subheader("Trade Exits and Directional Bias")
        st.markdown("<div class='performance-section-container'>", unsafe_allow_html=True)
        with st.expander("Exit & Direction Details", expanded=False):
            st.markdown("<div class='charts-grid'>", unsafe_allow_html=True)
            col6a, col6b = st.columns(2)
            with col6a:
                exit_type_col_key = 'exit_type_csv_str'
                exit_type_col_actual = get_column_name(exit_type_col_key, df.columns)
                if exit_type_col_actual:
                    exit_type_data = df[exit_type_col_actual].fillna('N/A').value_counts().reset_index()
                    exit_type_data.columns = [exit_type_col_actual, 'count']
                    fig_exit_type = plot_donut_chart(
                        df=exit_type_data, 
                        category_col=exit_type_col_actual,
                        value_col='count',
                        title=f"{PERFORMANCE_TABLE_SELECTABLE_CATEGORIES.get(exit_type_col_key, exit_type_col_key).replace('_',' ').title()} Distribution", theme=plot_theme,
                        is_data_aggregated=True
                    )
                    if fig_exit_type: st.plotly_chart(fig_exit_type, use_container_width=True)
                    if not exit_type_data.empty:
                        if st.checkbox("View Data: Exit Type Distribution", key="view_data_s6_c1_tab"):
                            st.dataframe(exit_type_data, use_container_width=True, hide_index=True)
            with col6b:
                direction_col_key_wr = 'direction_str'
                direction_col_actual_wr = get_column_name(direction_col_key_wr, df.columns)
                if direction_col_actual_wr and win_col_actual in df.columns:
                    dir_wr_data = df.groupby(direction_col_actual_wr, observed=False)[win_col_actual].agg(['mean', 'count']).reset_index()
                    dir_wr_data['mean'] *= 100 
                    dir_wr_data.rename(columns={'mean': 'Win Rate (%)', 'count': 'Total Trades'}, inplace=True)
                    fig_dir_wr = plot_win_rate_analysis(
                        df=dir_wr_data, 
                        category_col=direction_col_actual_wr,
                        win_rate_col='Win Rate (%)', 
                        trades_col='Total Trades', 
                        title_prefix="Win Rate by", theme=plot_theme,
                        is_data_aggregated=True
                    )
                    if fig_dir_wr: st.plotly_chart(fig_dir_wr, use_container_width=True)
                    if not dir_wr_data.empty:
                        if st.checkbox("View Data: Win Rate by Direction", key="view_data_s6_c2_tab"):
                            st.dataframe(dir_wr_data, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("---") 
            time_frame_col_key_facet = 'time_frame_str'
            time_frame_col_actual_facet = get_column_name(time_frame_col_key_facet, df.columns)
            direction_col_actual_facet = get_column_name('direction_str', df.columns) # Re-get for this specific section
            if direction_col_actual_facet and time_frame_col_actual_facet and trade_result_col_actual in df.columns:
                unique_time_frames = sorted(df[time_frame_col_actual_facet].astype(str).dropna().unique())
                if not unique_time_frames:
                    display_custom_message(f"No unique values found in '{time_frame_col_actual_facet}' for faceted chart selection.", "info")
                else:
                    default_selected_time_frames = unique_time_frames[:3] if len(unique_time_frames) > 3 else unique_time_frames 
                    selected_time_frames_for_facet = st.multiselect(
                        f"Select Time Frames for Faceted Chart (Max 5 recommended for clarity):",
                        options=unique_time_frames, default=default_selected_time_frames,
                        key="facet_time_frame_select_cat_page_tab"
                    )
                    if not selected_time_frames_for_facet: st.info("Please select at least one time frame to display the faceted chart.")
                    else:
                        df_facet_filtered = df[df[time_frame_col_actual_facet].isin(selected_time_frames_for_facet)]
                        if df_facet_filtered.empty: display_custom_message("No data for the selected time frames.", "info")
                        else:
                            try:
                                df_grouped_facet_data = df_facet_filtered.groupby(
                                    [direction_col_actual_facet, time_frame_col_actual_facet, trade_result_col_actual], observed=False
                                ).size().reset_index(name='count')
                                if not df_grouped_facet_data.empty:
                                    facet_col_wrap_val = min(3, len(selected_time_frames_for_facet)) # Adjust wrap based on selection
                                    fig_result_dir_tf = px.bar(
                                        df_grouped_facet_data, x=direction_col_actual_facet, y='count', color=trade_result_col_actual,
                                        facet_col=time_frame_col_actual_facet, facet_col_wrap=facet_col_wrap_val,
                                        title=f"{trade_result_col_actual.replace('_',' ').title()} by Direction and Selected Time Frames",
                                        labels={'count': "Number of Trades", 
                                                direction_col_actual_facet: PERFORMANCE_TABLE_SELECTABLE_CATEGORIES.get('direction_str', 'Direction'),
                                                time_frame_col_actual_facet: PERFORMANCE_TABLE_SELECTABLE_CATEGORIES.get('time_frame_str', 'Time Frame')
                                                }, 
                                        barmode='group',
                                        color_discrete_map={'WIN': COLORS.get('green'), 'LOSS': COLORS.get('red'), 'BREAKEVEN': COLORS.get('gray')}
                                    )
                                    if fig_result_dir_tf: st.plotly_chart(_apply_custom_theme(fig_result_dir_tf, plot_theme), use_container_width=True)
                                    if st.checkbox("View Data: Faceted Trade Results", key="view_data_s6_c3_tab"):
                                        st.dataframe(df_grouped_facet_data, use_container_width=True, hide_index=True)
                                else: display_custom_message("No data for Trade Result by Direction and selected Time Frames after grouping.", "info")
                            except Exception as e_gbtf: logger.error(f"Error in Trade Result by Direction and Time Frame: {e_gbtf}", exc_info=True)
            else:
                missing_cols_facet = [col for col_key, col_name_func in [('direction_str', direction_col_actual_facet), ('time_frame_str', time_frame_col_actual_facet), ('trade_result_processed', trade_result_col_actual)] if not col_name_func or col_name_func not in df.columns]
                display_custom_message(f"Missing columns for Faceted Chart. Needed: {', '.join(missing_cols_facet)}.", "warning")
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Tab 7: Performance Summary by Custom Category Table (with CIs) ---
    with tab7:
        st.subheader("Performance Summary by Custom Category")
        st.markdown("<div class='performance-section-container'>", unsafe_allow_html=True)
        with st.expander("View Performance Table with Confidence Intervals", expanded=True): # Expanded by default for this tab
            st.markdown("<div class='view-data-expander-content'>", unsafe_allow_html=True)
            available_categories_for_table: Dict[str, str] = {}
            for conceptual_key, display_name in PERFORMANCE_TABLE_SELECTABLE_CATEGORIES.items():
                actual_col = get_column_name(conceptual_key, df.columns)
                # Check if column exists and has non-empty, non-NA string values after stripping
                if actual_col and actual_col in df.columns and not df[actual_col].dropna().astype(str).str.strip().empty:
                     available_categories_for_table[display_name] = actual_col

            if not available_categories_for_table:
                display_custom_message("No suitable categorical columns found for the summary table.", "warning")
            else:
                selected_display_name_table = st.selectbox(
                    "Select Category for Performance Summary:",
                    options=list(available_categories_for_table.keys()),
                    key="custom_category_summary_select_tab"
                )
                metrics_for_ci_options = ["Average PnL", "Win Rate %"]
                selected_cis_to_calculate = st.multiselect(
                    "Calculate Confidence Intervals for:",
                    options=metrics_for_ci_options, default=metrics_for_ci_options,
                    key="ci_metric_select_cat_page_tab"
                )

                if selected_display_name_table:
                    selected_actual_col_for_table = available_categories_for_table[selected_display_name_table]
                    if not pnl_col_actual or not win_col_actual:
                        display_custom_message(f"PnL ('{pnl_col_actual}') or Win ('{win_col_actual}') column not available for summary table.", "error")
                    else:
                        with st.spinner(f"Calculating performance summary for category: {selected_display_name_table}..."):
                            summary_df = calculate_performance_summary_by_category(
                                df.copy(), category_col=selected_actual_col_for_table,
                                pnl_col=pnl_col_actual, win_col=win_col_actual,
                                calculate_cis_for=selected_cis_to_calculate
                            )
                        if not summary_df.empty:
                            st.markdown(f"##### Performance Summary by: {selected_display_name_table}")
                            cols_to_display_summary = ["Category Group", "Total PnL", "Total Trades",
                                                       "Average PnL", "Avg PnL CI",
                                                       "Win Rate %", "Win Rate % CI", "Expectancy $"]
                            summary_df_display = summary_df[[col for col in cols_to_display_summary if col in summary_df.columns]].copy()
                            # Formatting for display
                            if "Total PnL" in summary_df_display.columns: summary_df_display["Total PnL"] = summary_df_display["Total PnL"].apply(lambda x: format_currency(x) if pd.notna(x) else "N/A")
                            if "Average PnL" in summary_df_display.columns: summary_df_display["Average PnL"] = summary_df_display["Average PnL"].apply(lambda x: format_currency(x) if pd.notna(x) else "N/A")
                            if "Win Rate %" in summary_df_display.columns: summary_df_display["Win Rate %"] = summary_df_display["Win Rate %"].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
                            if "Expectancy $" in summary_df_display.columns: summary_df_display["Expectancy $"] = summary_df_display["Expectancy $"].apply(lambda x: format_currency(x) if pd.notna(x) else "N/A")
                            
                            st.dataframe(
                                summary_df_display, use_container_width=True, hide_index=True,
                                column_config={
                                    "Category Group": st.column_config.TextColumn(label=selected_display_name_table, width="medium"),
                                    "Total PnL": st.column_config.TextColumn(label="Total PnL"),
                                    "Total Trades": st.column_config.NumberColumn(label="Total Trades", format="%d"),
                                    "Average PnL": st.column_config.TextColumn(label="Avg PnL"),
                                    "Avg PnL CI": st.column_config.TextColumn(label=f"Avg PnL {CONFIDENCE_LEVEL*100:.0f}% CI"),
                                    "Win Rate %": st.column_config.TextColumn(label="Win Rate %"),
                                    "Win Rate % CI": st.column_config.TextColumn(label=f"Win Rate {CONFIDENCE_LEVEL*100:.0f}% CI"),
                                    "Expectancy $": st.column_config.TextColumn(label="Expectancy $")
                                }
                            )
                        else: display_custom_message(f"No summary data to display for category '{selected_display_name_table}'.", "info")
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Tab 8: Dynamic Category Visualizer (with Top/N and Significance Testing) ---
    with tab8:
        st.subheader("Dynamic Category Exploration")
        st.markdown("<div class='performance-section-container'>", unsafe_allow_html=True)
        with st.expander("Explore Data Dynamically with Statistical Tests", expanded=True): # Expanded by default
            st.markdown("<div class='controls-expander-content'>", unsafe_allow_html=True)
            available_categories_for_dynamic_plot: Dict[str, str] = {}
            for conceptual_key, display_name in PERFORMANCE_TABLE_SELECTABLE_CATEGORIES.items():
                actual_col = get_column_name(conceptual_key, df.columns)
                if actual_col and actual_col in df.columns and not df[actual_col].dropna().astype(str).str.strip().empty:
                    available_categories_for_dynamic_plot[display_name] = actual_col

            if not available_categories_for_dynamic_plot:
                display_custom_message("No suitable categorical columns found for dynamic visualization.", "warning")
            else:
                col_cat_select, col_metric_select, col_chart_select = st.columns(3)

                with col_cat_select:
                    selected_cat_display_name_dynamic = st.selectbox(
                        "Select Category to Analyze:",
                        options=list(available_categories_for_dynamic_plot.keys()),
                        key="dynamic_cat_select_tab"
                    )
                    actual_selected_category_col = available_categories_for_dynamic_plot.get(selected_cat_display_name_dynamic)

                with col_metric_select:
                    metric_options_dynamic = ["Total PnL", "Average PnL", "Win Rate (%)", "Trade Count", "PnL Distribution"]
                    selected_metric_dynamic = st.selectbox(
                        "Select Metric to Visualize:",
                        options=metric_options_dynamic,
                        key="dynamic_metric_select_tab"
                    )

                chart_type_options_dynamic = ["Bar Chart"]
                if selected_metric_dynamic == "Trade Count": chart_type_options_dynamic.append("Donut Chart")
                elif selected_metric_dynamic == "PnL Distribution": chart_type_options_dynamic = ["Box Plot"] 
                elif selected_metric_dynamic in ["Total PnL", "Average PnL"]: chart_type_options_dynamic.append("Box Plot")

                with col_chart_select:
                    selected_chart_type_dynamic = st.selectbox(
                        "Select Chart Type:", options=chart_type_options_dynamic, key="dynamic_chart_type_select_tab"
                    )

                filter_type_dynamic = "Show All"; num_n_dynamic = 5
                sort_metric_for_top_n = selected_metric_dynamic # Default sort metric to selected viz metric
                show_others_dynamic = False

                # Top N / Bottom N controls only if applicable
                if selected_metric_dynamic != "PnL Distribution" and selected_chart_type_dynamic in ["Bar Chart", "Donut Chart"]:
                    filter_type_dynamic = st.radio(
                        "Filter Categories by Metric Value:", ("Show All", "Top N", "Bottom N"), index=0,
                        key="dynamic_filter_type_tab", horizontal=True
                    )
                    if filter_type_dynamic != "Show All":
                        top_n_cols = st.columns([2,1])
                        with top_n_cols[0]:
                            # Filter out "PnL Distribution" from sort options
                            sort_options_for_top_n = [m for m in metric_options_dynamic if m != "PnL Distribution"]
                            default_sort_index = sort_options_for_top_n.index(selected_metric_dynamic) if selected_metric_dynamic in sort_options_for_top_n else 0
                            sort_metric_for_top_n = st.selectbox(
                                "Rank categories by:", options=sort_options_for_top_n,
                                index=default_sort_index,
                                key="dynamic_sort_metric_top_n_tab"
                            )
                        with top_n_cols[1]:
                            num_n_dynamic = st.number_input(
                                f"N:", 1, 50, 5, 1, key="dynamic_num_n_tab"
                            )
                        show_others_dynamic = st.checkbox("Group remaining into 'Others'", key="dynamic_show_others_tab")

                dynamic_plot_df_for_view = pd.DataFrame() # To store data for "View Data" checkbox

                if actual_selected_category_col:
                    df_dynamic_plot_data_source = df.copy() # Start with full filtered data for this section
                    
                    # Apply Top/Bottom N filtering if selected
                    if filter_type_dynamic != "Show All" and selected_metric_dynamic != "PnL Distribution" and selected_chart_type_dynamic in ["Bar Chart", "Donut Chart"]:
                        if not df_dynamic_plot_data_source.empty:
                            # Ensure necessary columns for ranking exist
                            if not pnl_col_actual or pnl_col_actual not in df_dynamic_plot_data_source.columns: 
                                display_custom_message("PnL column missing for ranking.", "error"); return 
                            if sort_metric_for_top_n == "Win Rate (%)" and (not win_col_actual or win_col_actual not in df_dynamic_plot_data_source.columns): 
                                display_custom_message("Win column missing for win rate ranking.", "error"); return 
                            
                            grouped_for_ranking_series = df_dynamic_plot_data_source.groupby(actual_selected_category_col, observed=False)
                            ranked_values_series = pd.Series(dtype=float)

                            if sort_metric_for_top_n == "Total PnL": ranked_values_series = grouped_for_ranking_series[pnl_col_actual].sum()
                            elif sort_metric_for_top_n == "Average PnL": ranked_values_series = grouped_for_ranking_series[pnl_col_actual].mean()
                            elif sort_metric_for_top_n == "Win Rate (%)": ranked_values_series = grouped_for_ranking_series[win_col_actual].mean() * 100
                            elif sort_metric_for_top_n == "Trade Count": ranked_values_series = grouped_for_ranking_series.size()
                            
                            if not ranked_values_series.empty:
                                top_n_cat_names = ranked_values_series.nlargest(num_n_dynamic).index.tolist() if filter_type_dynamic == "Top N" else ranked_values_series.nsmallest(num_n_dynamic).index.tolist()
                                
                                if show_others_dynamic:
                                    df_top_n_plot = df_dynamic_plot_data_source[df_dynamic_plot_data_source[actual_selected_category_col].isin(top_n_cat_names)].copy()
                                    df_others_plot = df_dynamic_plot_data_source[~df_dynamic_plot_data_source[actual_selected_category_col].isin(top_n_cat_names)].copy()
                                    if not df_others_plot.empty:
                                        df_others_plot.loc[:, actual_selected_category_col] = "Others" # Use .loc for assignment
                                        df_dynamic_plot_data = pd.concat([df_top_n_plot, df_others_plot], ignore_index=True)
                                    else: df_dynamic_plot_data = df_top_n_plot
                                else: 
                                    df_dynamic_plot_data = df_dynamic_plot_data_source[df_dynamic_plot_data_source[actual_selected_category_col].isin(top_n_cat_names)].copy()
                            else: 
                                logger.warning(f"Could not rank categories for Top/Bottom N based on {sort_metric_for_top_n}.")
                                df_dynamic_plot_data = pd.DataFrame() # Ensure it's an empty DF if ranking fails
                        else: 
                            df_dynamic_plot_data = pd.DataFrame() # Empty if source is empty
                    else: 
                        df_dynamic_plot_data = df_dynamic_plot_data_source # Use original if no Top/N filter


                    fig_dynamic = None
                    title_dynamic = f"{selected_metric_dynamic} by {selected_cat_display_name_dynamic}"
                    if filter_type_dynamic != "Show All": title_dynamic += f" ({filter_type_dynamic} {num_n_dynamic} by {sort_metric_for_top_n})"
                    if show_others_dynamic and filter_type_dynamic != "Show All": title_dynamic += " with Others"

                    if df_dynamic_plot_data.empty:
                        if filter_type_dynamic != "Show All": display_custom_message(f"No data remains for '{selected_cat_display_name_dynamic}' after applying '{filter_type_dynamic} {num_n_dynamic}' filter.", "info")
                        else: display_custom_message(f"No data available for '{selected_cat_display_name_dynamic}'.", "info")
                    else:
                        logger.debug(f"Dynamic Plot: Category='{actual_selected_category_col}', Metric='{selected_metric_dynamic}', Chart='{selected_chart_type_dynamic}', PlotTheme='{plot_theme}'")
                        try:
                            if selected_metric_dynamic == "Total PnL":
                                # Data for chart (aggregated)
                                chart_data = df_dynamic_plot_data.groupby(actual_selected_category_col, observed=False)[pnl_col_actual].sum().reset_index()
                                dynamic_plot_df_for_view = chart_data.copy() # For view data
                                if selected_chart_type_dynamic == "Bar Chart":
                                    fig_dynamic = plot_pnl_by_category(df=chart_data, category_col=actual_selected_category_col, pnl_col=pnl_col_actual, title_prefix=title_dynamic, aggregation_func='sum', theme=plot_theme, is_data_aggregated=True)
                                elif selected_chart_type_dynamic == "Box Plot": 
                                    fig_dynamic = plot_box_plot(df=df_dynamic_plot_data, category_col=actual_selected_category_col, value_col=pnl_col_actual, title=title_dynamic, theme=plot_theme)
                                    dynamic_plot_df_for_view = df_dynamic_plot_data[[actual_selected_category_col, pnl_col_actual]].copy() # Raw data for boxplot view
                            
                            elif selected_metric_dynamic == "Average PnL":
                                chart_data = df_dynamic_plot_data.groupby(actual_selected_category_col, observed=False)[pnl_col_actual].mean().reset_index()
                                dynamic_plot_df_for_view = chart_data.copy()
                                if selected_chart_type_dynamic == "Bar Chart":
                                    fig_dynamic = plot_pnl_by_category(df=chart_data, category_col=actual_selected_category_col, pnl_col=pnl_col_actual, title_prefix=title_dynamic, aggregation_func='mean', theme=plot_theme, is_data_aggregated=True)
                                elif selected_chart_type_dynamic == "Box Plot":
                                    fig_dynamic = plot_box_plot(df=df_dynamic_plot_data, category_col=actual_selected_category_col, value_col=pnl_col_actual, title=title_dynamic, theme=plot_theme)
                                    dynamic_plot_df_for_view = df_dynamic_plot_data[[actual_selected_category_col, pnl_col_actual]].copy()
                            
                            elif selected_metric_dynamic == "Win Rate (%)" and selected_chart_type_dynamic == "Bar Chart" and win_col_actual in df_dynamic_plot_data.columns:
                                chart_data = df_dynamic_plot_data.groupby(actual_selected_category_col, observed=False)[win_col_actual].agg(['mean', 'count']).reset_index()
                                chart_data['mean'] *= 100
                                chart_data.rename(columns={'mean': 'Win Rate (%)', 'count': 'Total Trades'}, inplace=True)
                                dynamic_plot_df_for_view = chart_data.copy()
                                fig_dynamic = plot_win_rate_analysis(df=chart_data, category_col=actual_selected_category_col, win_rate_col='Win Rate (%)', trades_col='Total Trades', title_prefix=title_dynamic, theme=plot_theme, is_data_aggregated=True)
                            
                            elif selected_metric_dynamic == "Trade Count":
                                chart_data = df_dynamic_plot_data.groupby(actual_selected_category_col, observed=False).size().reset_index(name='count').sort_values(by='count', ascending=False)
                                dynamic_plot_df_for_view = chart_data.copy()
                                if selected_chart_type_dynamic == "Bar Chart":
                                    fig_dynamic = px.bar(chart_data, x=actual_selected_category_col, y='count', title=title_dynamic, color='count', color_continuous_scale=px.colors.sequential.Blues_r)
                                    if fig_dynamic: fig_dynamic = _apply_custom_theme(fig_dynamic, plot_theme)
                                elif selected_chart_type_dynamic == "Donut Chart":
                                    fig_dynamic = plot_donut_chart(df=chart_data, category_col=actual_selected_category_col, value_col='count', title=title_dynamic, theme=plot_theme, is_data_aggregated=True)
                            
                            elif selected_metric_dynamic == "PnL Distribution" and selected_chart_type_dynamic == "Box Plot":
                                fig_dynamic = plot_box_plot(df=df_dynamic_plot_data, category_col=actual_selected_category_col, value_col=pnl_col_actual, title=title_dynamic, theme=plot_theme)
                                dynamic_plot_df_for_view = df_dynamic_plot_data[[actual_selected_category_col, pnl_col_actual]].copy()

                            if fig_dynamic:
                                st.plotly_chart(fig_dynamic, use_container_width=True)
                                if not dynamic_plot_df_for_view.empty:
                                    if st.checkbox(f"View Data for: {title_dynamic}", key=f"view_data_dynamic_{selected_cat_display_name_dynamic}_{selected_metric_dynamic}_tab"):
                                        st.dataframe(dynamic_plot_df_for_view.reset_index(drop=True), use_container_width=True)
                                
                                # Statistical Tests
                                category_groups_for_test = df_dynamic_plot_data[actual_selected_category_col].dropna().unique()
                                # Exclude "Others" from statistical tests if it exists
                                if "Others" in category_groups_for_test: 
                                    category_groups_for_test = [cat for cat in category_groups_for_test if cat != "Others"]
                                
                                if len(category_groups_for_test) >= 2: # Need at least two groups for comparison
                                    if selected_metric_dynamic == "Average PnL" and selected_chart_type_dynamic == "Bar Chart": # ANOVA for Avg PnL
                                        st.markdown("##### ANOVA F-test (Difference in Average PnL across categories)")
                                        # Prepare data for ANOVA: list of arrays, one for each group's PnL
                                        avg_pnl_data_for_anova = [
                                            df_dynamic_plot_data[df_dynamic_plot_data[actual_selected_category_col] == group][pnl_col_actual].dropna().values 
                                            for group in category_groups_for_test
                                        ]
                                        # Filter out groups with less than 2 observations as ANOVA needs variance
                                        avg_pnl_data_for_anova_filtered = [g_data for g_data in avg_pnl_data_for_anova if len(g_data) >= 2]

                                        if len(avg_pnl_data_for_anova_filtered) >= 2: # Still need at least 2 groups after filtering
                                            anova_results = statistical_service.run_hypothesis_test(data1=avg_pnl_data_for_anova_filtered, test_type='anova')
                                            if 'error' in anova_results: st.caption(f"ANOVA Test Error: {anova_results['error']}")
                                            else: st.metric(label="ANOVA P-value", value=f"{anova_results.get('p_value', np.nan):.4f}", help=anova_results.get('interpretation', ''))
                                        else: st.caption("ANOVA Test: Not enough groups with sufficient data (min 2 obs/group).")
                                    
                                    elif selected_metric_dynamic == "Win Rate (%)" and selected_chart_type_dynamic == "Bar Chart": # Chi-squared for Win Rates
                                        st.markdown("##### Chi-squared Test (Difference in Win Rates across categories)")
                                        contingency_table_data = []
                                        valid_groups_for_chi2 = 0
                                        for group in category_groups_for_test:
                                            group_data = df_dynamic_plot_data[df_dynamic_plot_data[actual_selected_category_col] == group]
                                            if not group_data.empty and win_col_actual in group_data.columns:
                                                wins = group_data[win_col_actual].sum()
                                                losses = len(group_data) - wins
                                                if wins + losses >= 5 : # Min observations for Chi-squared validity
                                                    contingency_table_data.append([wins, losses])
                                                    valid_groups_for_chi2 +=1
                                        
                                        if valid_groups_for_chi2 >= 2 and len(contingency_table_data) >=2 : # Need at least 2x2 table
                                            chi2_results = statistical_service.run_hypothesis_test(data1=np.array(contingency_table_data), test_type='chi-squared')
                                            if 'error' in chi2_results: st.caption(f"Chi-squared Test Error: {chi2_results['error']}")
                                            else: st.metric(label="Chi-squared P-value", value=f"{chi2_results.get('p_value', np.nan):.4f}", help=chi2_results.get('interpretation', ''))
                                        else: st.caption("Chi-squared Test: Not enough groups or observations per group for a valid test.")
                            elif selected_metric_dynamic and selected_chart_type_dynamic:
                                # Avoid showing error if win_col is missing for win rate plot (already handled by plot function)
                                if not (selected_metric_dynamic == "Win Rate (%)" and win_col_actual not in df_dynamic_plot_data.columns):
                                    display_custom_message(f"Could not generate '{selected_chart_type_dynamic}' for '{selected_metric_dynamic}' by '{selected_cat_display_name_dynamic}'. Check data and column selections.", "warning")
                        except Exception as e_dynamic_plot:
                            logger.error(f"Error generating dynamic plot for {selected_cat_display_name_dynamic} ({selected_metric_dynamic} / {selected_chart_type_dynamic}): {e_dynamic_plot}", exc_info=True)
                            display_custom_message(f"An error occurred while generating the dynamic chart: {e_dynamic_plot}", "error")
                else:
                    display_custom_message("Please select a valid category to visualize.", "info")
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    # This is primarily for direct execution testing, which is less common for page files.
    # In a multi-page app, Streamlit handles the execution.
    # Setting a default session state for testing if run directly
    if 'app_initialized' not in st.session_state:
        st.warning("This page is part of a multi-page app. Please run the main app.py script for full functionality.")
        # Mock some data for direct testing if needed, or rely on user to navigate from main app
        # For example:
        # if 'filtered_data' not in st.session_state:
        #     st.session_state.filtered_data = pd.DataFrame({
        #         'pnl': np.random.rand(100) * 200 - 100,
        #         'date': pd.to_datetime(pd.date_range(start='2023-01-01', periods=100, freq='D')),
        #         'strategy': np.random.choice(['Alpha', 'Beta', 'Gamma'], 100),
        #         'win': np.random.choice([True, False], 100),
        #         'trade_result_processed': np.random.choice(['WIN', 'LOSS', 'BREAKEVEN'], 100),
        #         # Add other necessary columns for testing
        #     })
        # st.session_state.current_theme = 'dark'


    show_categorical_analysis_page()
