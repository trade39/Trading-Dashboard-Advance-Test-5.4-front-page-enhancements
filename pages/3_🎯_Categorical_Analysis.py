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
    logger = logging.getLogger("CategoricalAnalysisPage_Fallback_Config")
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
    logger = logging.getLogger(APP_TITLE if 'APP_TITLE' in globals() else "FallbackApp_Plotting")
    logger.error(f"CRITICAL IMPORT ERROR (Plotting/Components) in Categorical Analysis Page: {e}", exc_info=True)
    # Fallback plotting functions
    def _apply_custom_theme(fig, theme): return fig
    def plot_pnl_by_category(*args, **kwargs): return None
    def plot_stacked_bar_chart(*args, **kwargs): return None
    def plot_heatmap(*args, **kwargs): return None
    def plot_value_over_time(*args, **kwargs): return None
    def plot_grouped_bar_chart(*args, **kwargs): return None
    def plot_box_plot(*args, **kwargs): return None
    def plot_donut_chart(*args, **kwargs): return None
    def plot_radar_chart(*args, **kwargs): return None
    def plot_scatter_plot(*args, **kwargs): return None
    def plot_pnl_distribution(*args, **kwargs): return None
    def plot_win_rate_analysis(*args, **kwargs): return None
    class PnLCalendarComponent:
        def __init__(self, *args, **kwargs): pass
        def render(self): st.warning("Calendar component could not be loaded.")
    st.stop()


logger = logging.getLogger(APP_TITLE)
statistical_service = StatisticalAnalysisService()

# --- Constants for Conceptual Column Keys ---
PNL_KEY = 'pnl'
DATE_KEY = 'date'
STRATEGY_KEY = 'strategy'
MARKET_CONDITIONS_KEY = 'market_conditions_str'
RR_CSV_KEY = 'r_r_csv_num'
DIRECTION_KEY = 'direction_str'
TRADE_PLAN_KEY = 'trade_plan_str'
ENTRY_TIME_KEY = 'entry_time_str'
TRADE_HOUR_KEY = 'trade_hour'
TRADE_DAY_OF_WEEK_KEY = 'trade_day_of_week'
TRADE_MONTH_NAME_KEY = 'trade_month_name'
TRADE_MONTH_NUM_KEY = 'trade_month_num'
SYMBOL_KEY = 'symbol'
BIAS_KEY = 'bias_str'
TIME_FRAME_KEY = 'time_frame_str'
SESSION_KEY = 'session_str'
EVENTS_DETAILS_KEY = 'events_details_str'
PSYCHOLOGICAL_FACTORS_KEY = 'psychological_factors_str'
ACCOUNT_KEY = 'account_str'
EXIT_TYPE_CSV_KEY = 'exit_type_csv_str'
EVENT_TYPE_KEY = 'event_type_str'
MARKET_SENTIMENT_KEY = 'market_sentiment_str'
COMPLIANCE_CHECK_KEY = 'compliance_check_str'
INITIAL_BALANCE_KEY = 'initial_balance_num'
DRAWDOWN_VALUE_CSV_KEY = 'drawdown_value_csv'


PERFORMANCE_TABLE_SELECTABLE_CATEGORIES: Dict[str, str] = {
    ENTRY_TIME_KEY: 'Entry Time (Raw String)', TRADE_HOUR_KEY: 'Trade Hour',
    TRADE_DAY_OF_WEEK_KEY: 'Day of Week', TRADE_MONTH_NAME_KEY: 'Month',
    SYMBOL_KEY: 'Symbol', STRATEGY_KEY: 'Trade Model', TRADE_PLAN_KEY: 'Trade Plan',
    BIAS_KEY: 'Bias', TIME_FRAME_KEY: 'Time Frame', DIRECTION_KEY: 'Direction',
    RR_CSV_KEY: 'R:R (from CSV)', SESSION_KEY: 'Session',
    MARKET_CONDITIONS_KEY: 'Market Conditions', EVENTS_DETAILS_KEY: 'Events Details',
    PSYCHOLOGICAL_FACTORS_KEY: 'Psychological Factors', ACCOUNT_KEY: 'Account',
    EXIT_TYPE_CSV_KEY: 'Exit Type', EVENT_TYPE_KEY: 'Event Type',
    MARKET_SENTIMENT_KEY: 'Market Sentiment', COMPLIANCE_CHECK_KEY: 'Compliance Check',
    INITIAL_BALANCE_KEY: 'Initial Balance', DRAWDOWN_VALUE_CSV_KEY: 'Drawdown Value (from CSV)'
}


def get_column_name(conceptual_key: str, df_columns: Optional[pd.Index] = None) -> Optional[str]:
    """
    Retrieves the actual column name from DataFrame columns based on a conceptual key.
    It first checks if the conceptual_key itself is a column.
    If not, it uses EXPECTED_COLUMNS to map the conceptual_key to an actual column name.
    Logs a warning if the mapped column is not found in the DataFrame.
    """
    if df_columns is not None and conceptual_key in df_columns:
        return conceptual_key
    
    actual_col = EXPECTED_COLUMNS.get(conceptual_key)
    
    if df_columns is not None and actual_col and actual_col not in df_columns:
        logger.warning(f"Conceptual key '{conceptual_key}' maps to '{actual_col}', but it's not in DataFrame columns: {df_columns.tolist()}")
        return None
    elif not actual_col: # conceptual_key not in EXPECTED_COLUMNS
        logger.warning(f"Conceptual key '{conceptual_key}' not found in EXPECTED_COLUMNS mapping.")
        return None
        
    return actual_col

@st.cache_data # Cache the results of this potentially expensive function
def calculate_performance_summary_by_category(
    df: pd.DataFrame, category_col: str, pnl_col: str, win_col: str,
    calculate_cis_for: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculates a detailed performance summary grouped by a specified category.

    Args:
        df (pd.DataFrame): The input DataFrame containing trade data.
        category_col (str): The name of the column to group by.
        pnl_col (str): The name of the Profit and Loss (PnL) column.
        win_col (str): The name of the boolean column indicating a winning trade.
        calculate_cis_for (Optional[List[str]]): List of metrics for which to calculate 
                                                 bootstrap confidence intervals (e.g., ["Average PnL", "Win Rate %"]).

    Returns:
        pd.DataFrame: A DataFrame summarizing performance metrics for each category group,
                      including Total PnL, Total Trades, Win Rate, Expectancy, Average PnL,
                      and optionally, confidence intervals for Average PnL and Win Rate.
                      Returns an empty DataFrame if essential columns are missing or errors occur.
    """
    if calculate_cis_for is None:
        calculate_cis_for = []

    # Validate essential columns
    if category_col not in df.columns or pnl_col not in df.columns or win_col not in df.columns:
        logger.error(f"Missing required columns for performance summary: category='{category_col}', pnl='{pnl_col}', win='{win_col}'")
        return pd.DataFrame()

    df_copy = df.copy() # Work on a copy to avoid modifying the original DataFrame

    # Ensure win_col is boolean. If not, try to derive it from pnl_col.
    if not pd.api.types.is_bool_dtype(df_copy[win_col]):
        if pd.api.types.is_numeric_dtype(df_copy[pnl_col]):
            logger.info(f"Win column '{win_col}' is not boolean. Creating it from PnL column '{pnl_col}' (PnL > 0).")
            df_copy[win_col] = df_copy[pnl_col] > 0
        else:
            logger.error(f"Cannot create boolean win column: PnL column '{pnl_col}' is not numeric, and win column '{win_col}' is not boolean.")
            return pd.DataFrame()

    # Group data by the specified category, filling NaN categories with 'N/A'
    # observed=False ensures all categories are present even if filtered out in the current view
    df_grouped = df_copy.fillna({category_col: 'N/A'}).groupby(category_col, observed=False)
    
    summary_data = []

    for name_of_group, group_df in df_grouped:
        total_trades = len(group_df)
        if total_trades == 0:  # Skip empty groups
            continue

        total_pnl = group_df[pnl_col].sum()
        avg_pnl = group_df[pnl_col].mean()

        num_wins = group_df[win_col].sum()
        num_losses = total_trades - num_wins
        win_rate_pct = (num_wins / total_trades) * 100 if total_trades > 0 else 0.0

        # Initialize CI variables
        avg_pnl_ci_lower, avg_pnl_ci_upper = np.nan, np.nan
        win_rate_ci_lower, win_rate_ci_upper = np.nan, np.nan

        # Calculate Confidence Intervals if requested and group size is sufficient
        if total_trades >= 10: # Minimum trades for meaningful CI
            try:
                if "Average PnL" in calculate_cis_for:
                    avg_pnl_bs_results = statistical_service.calculate_bootstrap_ci(
                        data_series=group_df[pnl_col], statistic_func=np.mean,
                        n_iterations=BOOTSTRAP_ITERATIONS // 4, confidence_level=CONFIDENCE_LEVEL # Reduce iterations for faster UI
                    )
                    if 'error' not in avg_pnl_bs_results:
                        avg_pnl_ci_lower = avg_pnl_bs_results['lower_bound']
                        avg_pnl_ci_upper = avg_pnl_bs_results['upper_bound']

                if "Win Rate %" in calculate_cis_for:
                    # Statistic function for win rate (as percentage)
                    win_rate_stat_func = lambda x_series: (np.sum(x_series > 0) / len(x_series)) * 100 if len(x_series) > 0 else 0.0
                    # Bootstrap CI for win rate based on PnL values (determining win/loss)
                    data_for_win_rate_bs = group_df[pnl_col] 
                    
                    win_rate_bs_results = statistical_service.calculate_bootstrap_ci(
                        data_series=data_for_win_rate_bs, statistic_func=win_rate_stat_func,
                        n_iterations=BOOTSTRAP_ITERATIONS // 4, confidence_level=CONFIDENCE_LEVEL
                    )
                    if 'error' not in win_rate_bs_results:
                        win_rate_ci_lower = win_rate_bs_results['lower_bound']
                        win_rate_ci_upper = win_rate_bs_results['upper_bound']
            except Exception as e_bs:
                logger.warning(f"Error during bootstrapping for group '{name_of_group}' in category '{category_col}': {e_bs}")

        loss_rate_pct = (num_losses / total_trades) * 100 if total_trades > 0 else 0.0
        
        # Calculate average win/loss amounts
        wins_df = group_df[group_df[win_col]]
        # Ensure losses are actually < 0 for avg_loss_amount calculation
        losses_df = group_df[~group_df[win_col] & (group_df[pnl_col] < 0)] 
        
        avg_win_amount = wins_df[pnl_col].sum() / num_wins if num_wins > 0 else 0.0
        avg_loss_amount = abs(losses_df[pnl_col].sum()) / num_losses if num_losses > 0 else 0.0
        
        # Calculate Expectancy
        expectancy = (avg_win_amount * (win_rate_pct / 100.0)) - (avg_loss_amount * (loss_rate_pct / 100.0))

        summary_data.append({
            "Category Group": name_of_group, "Total PnL": total_pnl, "Total Trades": total_trades,
            "Win Rate %": win_rate_pct, "Expectancy $": expectancy, "Average PnL": avg_pnl,
            "Avg PnL CI": f"[{avg_pnl_ci_lower:,.2f}, {avg_pnl_ci_upper:,.2f}]" if pd.notna(avg_pnl_ci_lower) and pd.notna(avg_pnl_ci_upper) else "N/A",
            "Win Rate % CI": f"[{win_rate_ci_lower:.1f}%, {win_rate_ci_upper:.1f}%]" if pd.notna(win_rate_ci_lower) and pd.notna(win_rate_ci_upper) else "N/A"
        })
    
    summary_df = pd.DataFrame(summary_data)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(by="Total PnL", ascending=False) # Default sort
    return summary_df

def display_data_table_with_expander(
    df_to_display: pd.DataFrame, 
    expander_label: str, 
    unique_key: str, 
    dataframe_kwargs: Optional[Dict] = None,
    expanded_by_default: bool = False
):
    """
    Displays a Streamlit expander that, when expanded, shows a DataFrame.
    Includes an eye emoji in the label.

    Args:
        df_to_display (pd.DataFrame): The DataFrame to display.
        expander_label (str): The label for the expander (without emoji).
        unique_key (str): A unique key for the Streamlit expander widget.
        dataframe_kwargs (Optional[Dict]): Additional keyword arguments to pass to st.dataframe.
                                             Defaults to {'use_container_width': True, 'hide_index': True}.
        expanded_by_default (bool): Whether the expander should be open by default.
    """
    if dataframe_kwargs is None:
        dataframe_kwargs = {'use_container_width': True, 'hide_index': True}

    if not df_to_display.empty:
        # Prepend eye emoji to the label
        label_with_emoji = f"👁️ {expander_label}"
        with st.expander(label_with_emoji, expanded=expanded_by_default):
            st.dataframe(df_to_display, **dataframe_kwargs)


# --- Helper Functions for Rendering Sections ---
def render_strategy_performance_insights(
    df: pd.DataFrame, pnl_col_actual: str, trade_result_col_actual: str, 
    plot_theme: str, section_key_prefix: str, **kwargs 
):
    col1a, col1b = st.columns(2)
    with col1a:
        strategy_col_actual = get_column_name(STRATEGY_KEY, df.columns)
        if strategy_col_actual and pnl_col_actual:
            avg_pnl_strategy_data = df.groupby(strategy_col_actual, observed=False)[pnl_col_actual].mean().reset_index()
            avg_pnl_strategy_data = avg_pnl_strategy_data.sort_values(by=pnl_col_actual, ascending=False)
            fig_avg_pnl_strategy = plot_pnl_by_category(
                df=avg_pnl_strategy_data, category_col=strategy_col_actual, pnl_col=pnl_col_actual,
                title_prefix="Average PnL by", aggregation_func='mean', theme=plot_theme, is_data_aggregated=True
            )
            if fig_avg_pnl_strategy: st.plotly_chart(fig_avg_pnl_strategy, use_container_width=True)
            display_data_table_with_expander( 
                avg_pnl_strategy_data, "View Data: Average PnL by Strategy", 
                f"{section_key_prefix}_exp_avg_pnl_strategy" 
            )
    with col1b:
        trade_plan_col_actual = get_column_name(TRADE_PLAN_KEY, df.columns)
        if trade_plan_col_actual and trade_result_col_actual in df.columns:
            result_by_plan_data = pd.crosstab(df[trade_plan_col_actual].fillna('N/A'), df[trade_result_col_actual].fillna('N/A'))
            for col in ['WIN', 'LOSS', 'BREAKEVEN']: 
                if col not in result_by_plan_data.columns: result_by_plan_data[col] = 0
            result_by_plan_data = result_by_plan_data[['WIN', 'LOSS', 'BREAKEVEN']]
            
            fig_result_by_plan = plot_stacked_bar_chart(
                df=result_by_plan_data.reset_index(), category_col=trade_plan_col_actual,
                stack_cols=['WIN', 'LOSS', 'BREAKEVEN'],
                title=f"{trade_result_col_actual.replace('_',' ').title()} by {PERFORMANCE_TABLE_SELECTABLE_CATEGORIES.get(TRADE_PLAN_KEY, TRADE_PLAN_KEY).replace('_',' ').title()}",
                theme=plot_theme,
                color_discrete_map={'WIN': COLORS.get('green'), 'LOSS': COLORS.get('red'), 'BREAKEVEN': COLORS.get('gray')},
                is_data_aggregated=True
            )
            if fig_result_by_plan: st.plotly_chart(fig_result_by_plan, use_container_width=True)
            display_data_table_with_expander( 
                result_by_plan_data.reset_index(), f"View Data: {trade_result_col_actual.replace('_',' ').title()} by Trade Plan",
                f"{section_key_prefix}_exp_result_by_plan"
            )
    st.markdown("---")
    rr_col_actual = get_column_name(RR_CSV_KEY, df.columns)
    direction_col_actual = get_column_name(DIRECTION_KEY, df.columns)
    strategy_col_actual_for_rr = get_column_name(STRATEGY_KEY, df.columns) 
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
                display_data_table_with_expander( 
                    pivot_rr_data.reset_index(), "View Data: Average R:R by Strategy and Direction",
                    f"{section_key_prefix}_exp_rr_heatmap"
                )
        except Exception as e_rr_heatmap: logger.error(f"Error in R:R Heatmap: {e_rr_heatmap}", exc_info=True)

def render_temporal_analysis(
    df: pd.DataFrame, pnl_col_actual: str, win_col_actual: str, date_col_actual: str, 
    trade_result_col_actual: str, plot_theme: str, section_key_prefix: str, **kwargs 
):
    col2a, col2b = st.columns(2)
    with col2a:
        month_num_col_actual = get_column_name(TRADE_MONTH_NUM_KEY, df.columns)
        month_name_col_actual = get_column_name(TRADE_MONTH_NAME_KEY, df.columns)
        if month_num_col_actual and month_name_col_actual and win_col_actual in df.columns:
            try:
                monthly_win_rate_series = df.groupby(month_num_col_actual, observed=False)[win_col_actual].mean() * 100
                month_map_df = df[[month_num_col_actual, month_name_col_actual]].drop_duplicates().sort_values(month_num_col_actual)
                month_mapping = pd.Series(month_map_df[month_name_col_actual].values, index=month_map_df[month_num_col_actual]).to_dict()
                monthly_win_rate_data = monthly_win_rate_series.sort_index().rename(index=month_mapping)

                if not monthly_win_rate_data.empty:
                    fig_monthly_wr = plot_value_over_time(series=monthly_win_rate_data, series_name="Monthly Win Rate", title="Win Rate by Month", x_axis_title="Month", y_axis_title="Win Rate (%)", theme=plot_theme)
                    if fig_monthly_wr: st.plotly_chart(fig_monthly_wr, use_container_width=True)
                    display_data_table_with_expander( 
                        monthly_win_rate_data.reset_index().rename(columns={'index': 'Month', month_num_col_actual: 'Month Name', win_col_actual: 'Win Rate (%)'}), 
                        "View Data: Win Rate by Month", f"{section_key_prefix}_exp_monthly_wr"
                    )
            except Exception as e_mwr: logger.error(f"Error in Monthly Win Rate: {e_mwr}", exc_info=True)
    with col2b:
        session_col_actual = get_column_name(SESSION_KEY, df.columns)
        time_frame_col_actual = get_column_name(TIME_FRAME_KEY, df.columns)
        if session_col_actual and time_frame_col_actual and trade_result_col_actual in df.columns:
            try:
                count_df_agg = df.groupby([session_col_actual, time_frame_col_actual, trade_result_col_actual], observed=False).size().reset_index(name='count')
                pivot_session_tf_data = count_df_agg.pivot_table(index=session_col_actual, columns=time_frame_col_actual, values='count', fill_value=0, aggfunc='sum')

                if not pivot_session_tf_data.empty:
                    fig_session_tf_heatmap = plot_heatmap(df_pivot=pivot_session_tf_data, title=f"Trade Count by Session and Time Frame", color_scale="Blues", theme=plot_theme, text_format=".0f")
                    if fig_session_tf_heatmap: st.plotly_chart(fig_session_tf_heatmap, use_container_width=True)
                    display_data_table_with_expander( 
                        pivot_session_tf_data.reset_index(), "View Data: Trade Count by Session and Time Frame",
                        f"{section_key_prefix}_exp_session_tf_heatmap"
                    )
            except Exception as e_sess_tf: logger.error(f"Error in Session/TF Heatmap: {e_sess_tf}", exc_info=True)
    st.markdown("---")
    if date_col_actual and pnl_col_actual: 
        try:
            df_calendar = df.copy()
            if not pd.api.types.is_datetime64_any_dtype(df_calendar[date_col_actual]):
                df_calendar[date_col_actual] = pd.to_datetime(df_calendar[date_col_actual], errors='coerce')
            df_calendar = df_calendar.dropna(subset=[date_col_actual])

            daily_pnl_df_agg = df_calendar.groupby(df_calendar[date_col_actual].dt.normalize())[pnl_col_actual].sum().reset_index()
            daily_pnl_df_agg = daily_pnl_df_agg.rename(columns={date_col_actual: 'date', pnl_col_actual: 'pnl'}) 
            available_years = sorted(daily_pnl_df_agg['date'].dt.year.unique(), reverse=True)
            if available_years:
                selected_year = st.selectbox("Select Year for P&L Calendar:", options=available_years, index=0, key=f"{section_key_prefix}_calendar_year_select")
                if selected_year:
                    st.markdown("<div class='calendar-display-area'>", unsafe_allow_html=True)
                    calendar_component = PnLCalendarComponent(daily_pnl_df=daily_pnl_df_agg, year=selected_year, plot_theme=plot_theme)
                    calendar_component.render()
                    st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e_cal: logger.error(f"Error in P&L Calendar: {e_cal}", exc_info=True)

def render_market_context_impact(
    df: pd.DataFrame, pnl_col_actual: str, win_col_actual: str, 
    trade_result_col_actual: str, plot_theme: str, section_key_prefix: str, **kwargs 
):
    col3a, col3b = st.columns(2)
    with col3a:
        event_type_col_actual = get_column_name(EVENT_TYPE_KEY, df.columns)
        if event_type_col_actual and trade_result_col_actual in df.columns:
            result_by_event_data = df.groupby([event_type_col_actual, trade_result_col_actual], observed=False).size().reset_index(name='count')
            fig_result_by_event = plot_grouped_bar_chart(
                df=result_by_event_data, category_col=event_type_col_actual, value_col='count', group_col=trade_result_col_actual,
                title=f"{trade_result_col_actual.replace('_',' ').title()} Count by {PERFORMANCE_TABLE_SELECTABLE_CATEGORIES.get(EVENT_TYPE_KEY, EVENT_TYPE_KEY).replace('_',' ').title()}",
                theme=plot_theme, color_discrete_map={'WIN': COLORS.get('green'), 'LOSS': COLORS.get('red'), 'BREAKEVEN': COLORS.get('gray')},
                is_data_aggregated=True
            )
            if fig_result_by_event: st.plotly_chart(fig_result_by_event, use_container_width=True)
            display_data_table_with_expander( 
                result_by_event_data, f"View Data: {trade_result_col_actual.replace('_',' ').title()} Count by Event Type",
                f"{section_key_prefix}_exp_result_by_event"
            )
    with col3b:
        market_cond_col_actual = get_column_name(MARKET_CONDITIONS_KEY, df.columns)
        if market_cond_col_actual and pnl_col_actual:
            fig_pnl_by_market = plot_box_plot(
                df=df, category_col=market_cond_col_actual, value_col=pnl_col_actual,
                title=f"PnL Distribution by {PERFORMANCE_TABLE_SELECTABLE_CATEGORIES.get(MARKET_CONDITIONS_KEY, MARKET_CONDITIONS_KEY).replace('_',' ').title()}", theme=plot_theme
            )
            if fig_pnl_by_market: st.plotly_chart(fig_pnl_by_market, use_container_width=True)
            
            with st.expander(f"👁️ View Summary Statistics for PnL by Market Condition", expanded=False): # UPDATED with emoji
                 market_cond_pnl_summary = df.groupby(market_cond_col_actual, observed=False)[pnl_col_actual].describe()
                 st.dataframe(market_cond_pnl_summary, use_container_width=True)

    st.markdown("---")
    market_sent_col_actual = get_column_name(MARKET_SENTIMENT_KEY, df.columns)
    if market_sent_col_actual and win_col_actual in df.columns: 
        try:
            sentiment_win_rate_data = df.groupby(market_sent_col_actual, observed=False)[win_col_actual].mean().reset_index()
            sentiment_win_rate_data[win_col_actual] *= 100 
            if not sentiment_win_rate_data.empty:
                fig_sent_wr = px.bar(sentiment_win_rate_data, x=market_sent_col_actual, y=win_col_actual,
                                     title=f"Win Rate by {PERFORMANCE_TABLE_SELECTABLE_CATEGORIES.get(MARKET_SENTIMENT_KEY, MARKET_SENTIMENT_KEY).replace('_',' ').title()}",
                                     labels={win_col_actual: "Win Rate (%)", market_sent_col_actual: PERFORMANCE_TABLE_SELECTABLE_CATEGORIES.get(MARKET_SENTIMENT_KEY, MARKET_SENTIMENT_KEY).replace('_',' ').title()},
                                     color=win_col_actual, color_continuous_scale="Greens")
                if fig_sent_wr: 
                    fig_sent_wr.update_yaxes(ticksuffix="%")
                    st.plotly_chart(_apply_custom_theme(fig_sent_wr, plot_theme), use_container_width=True)
                display_data_table_with_expander( 
                    sentiment_win_rate_data.rename(columns={win_col_actual: "Win Rate (%)"}), "View Data: Win Rate by Market Sentiment",
                    f"{section_key_prefix}_exp_sent_wr_data"
                )
        except Exception as e_sent_wr: logger.error(f"Error generating Market Sentiment vs Win Rate: {e_sent_wr}", exc_info=True)

def render_behavioral_factors(
    df: pd.DataFrame, trade_result_col_actual: str, plot_theme: str, section_key_prefix: str, **kwargs 
):
    col4a, col4b = st.columns(2)
    with col4a:
        psych_col_actual = get_column_name(PSYCHOLOGICAL_FACTORS_KEY, df.columns)
        if psych_col_actual and trade_result_col_actual in df.columns:
            df_psych = df.copy()
            if df_psych[psych_col_actual].dtype == 'object':
                df_psych[psych_col_actual] = df_psych[psych_col_actual].astype(str).str.split(',').str[0].str.strip().fillna('N/A')
            
            psych_result_data = pd.crosstab(df_psych[psych_col_actual], df_psych[trade_result_col_actual])
            for col in ['WIN', 'LOSS', 'BREAKEVEN']: 
                if col not in psych_result_data.columns: psych_result_data[col] = 0
            psych_result_data = psych_result_data[['WIN', 'LOSS', 'BREAKEVEN']]

            fig_psych_result = plot_stacked_bar_chart(
                df=psych_result_data.reset_index(), category_col=psych_col_actual,
                stack_cols=['WIN', 'LOSS', 'BREAKEVEN'],
                title=f"{trade_result_col_actual.replace('_',' ').title()} by Dominant {PERFORMANCE_TABLE_SELECTABLE_CATEGORIES.get(PSYCHOLOGICAL_FACTORS_KEY, PSYCHOLOGICAL_FACTORS_KEY).replace('_',' ').title()}",
                theme=plot_theme, color_discrete_map={'WIN': COLORS.get('green'), 'LOSS': COLORS.get('red'), 'BREAKEVEN': COLORS.get('gray')},
                is_data_aggregated=True
            )
            if fig_psych_result: st.plotly_chart(fig_psych_result, use_container_width=True)
            display_data_table_with_expander( 
                psych_result_data.reset_index(), "View Data: Trade Result by Dominant Psychological Factor",
                f"{section_key_prefix}_exp_psych_result"
            )
    with col4b:
        compliance_col_actual = get_column_name(COMPLIANCE_CHECK_KEY, df.columns)
        if compliance_col_actual:
            compliance_data = df[compliance_col_actual].fillna('N/A').value_counts().reset_index()
            compliance_data.columns = [compliance_col_actual, 'count'] 
            fig_compliance = plot_donut_chart(
                df=compliance_data, category_col=compliance_col_actual, value_col='count',
                title=f"{PERFORMANCE_TABLE_SELECTABLE_CATEGORIES.get(COMPLIANCE_CHECK_KEY, COMPLIANCE_CHECK_KEY).replace('_',' ').title()} Outcomes", theme=plot_theme,
                is_data_aggregated=True
            )
            if fig_compliance: st.plotly_chart(fig_compliance, use_container_width=True)
            display_data_table_with_expander( 
                compliance_data, "View Data: Compliance Outcomes",
                f"{section_key_prefix}_exp_compliance_data"
            )

def render_capital_risk_insights(
    df: pd.DataFrame, trade_result_col_actual: str, plot_theme: str, section_key_prefix: str, **kwargs 
):
    col5a, col5b = st.columns(2)
    with col5a:
        initial_bal_col_actual = get_column_name(INITIAL_BALANCE_KEY, df.columns)
        drawdown_csv_col_actual = get_column_name(DRAWDOWN_VALUE_CSV_KEY, df.columns)
        if initial_bal_col_actual and drawdown_csv_col_actual and trade_result_col_actual in df.columns:
            scatter_data_cols = [initial_bal_col_actual, drawdown_csv_col_actual, trade_result_col_actual]
            scatter_df_view = df[scatter_data_cols].dropna().copy() 
            
            fig_bal_dd = plot_scatter_plot(
                df=scatter_df_view, x_col=initial_bal_col_actual, y_col=drawdown_csv_col_actual, color_col=trade_result_col_actual,
                title=f"{PERFORMANCE_TABLE_SELECTABLE_CATEGORIES.get(DRAWDOWN_VALUE_CSV_KEY, DRAWDOWN_VALUE_CSV_KEY).replace('_',' ').title()} vs. {PERFORMANCE_TABLE_SELECTABLE_CATEGORIES.get(INITIAL_BALANCE_KEY, INITIAL_BALANCE_KEY).replace('_',' ').title()}",
                theme=plot_theme, color_discrete_map={'WIN': COLORS.get('green'), 'LOSS': COLORS.get('red'), 'BREAKEVEN': COLORS.get('gray')}
            )
            if fig_bal_dd: st.plotly_chart(fig_bal_dd, use_container_width=True)
            display_data_table_with_expander( 
                scatter_df_view, "View Data: Drawdown vs. Initial Balance",
                f"{section_key_prefix}_exp_bal_dd_scatter"
            )
    with col5b:
        trade_plan_col_actual_dd = get_column_name(TRADE_PLAN_KEY, df.columns) 
        drawdown_csv_col_actual_avg = get_column_name(DRAWDOWN_VALUE_CSV_KEY, df.columns) 
        if trade_plan_col_actual_dd and drawdown_csv_col_actual_avg:
            avg_dd_plan_data = df.groupby(trade_plan_col_actual_dd, observed=False)[drawdown_csv_col_actual_avg].mean().reset_index()
            avg_dd_plan_data = avg_dd_plan_data.sort_values(by=drawdown_csv_col_actual_avg, ascending=True)
            
            fig_avg_dd_plan = plot_pnl_by_category( 
                df=avg_dd_plan_data, category_col=trade_plan_col_actual_dd, pnl_col=drawdown_csv_col_actual_avg,
                title_prefix="Average Drawdown by", aggregation_func='mean', theme=plot_theme, is_data_aggregated=True
            )
            if fig_avg_dd_plan: st.plotly_chart(fig_avg_dd_plan, use_container_width=True)
            display_data_table_with_expander( 
                avg_dd_plan_data, "View Data: Average Drawdown by Trade Plan",
                f"{section_key_prefix}_exp_avg_dd_plan"
            )
    st.markdown("---")
    drawdown_csv_col_actual_hist = get_column_name(DRAWDOWN_VALUE_CSV_KEY, df.columns) 
    if drawdown_csv_col_actual_hist:
        df_dd_hist = df[[drawdown_csv_col_actual_hist]].copy()
        df_dd_hist[drawdown_csv_col_actual_hist] = pd.to_numeric(df_dd_hist[drawdown_csv_col_actual_hist], errors='coerce')
        df_dd_hist.dropna(subset=[drawdown_csv_col_actual_hist], inplace=True)
        if not df_dd_hist.empty:
            fig_dd_hist = plot_pnl_distribution( 
                df=df_dd_hist, pnl_col=drawdown_csv_col_actual_hist, 
                title=f"{PERFORMANCE_TABLE_SELECTABLE_CATEGORIES.get(DRAWDOWN_VALUE_CSV_KEY, DRAWDOWN_VALUE_CSV_KEY).replace('_',' ').title()} Distribution",
                theme=plot_theme, nbins=30
            )
            if fig_dd_hist: st.plotly_chart(fig_dd_hist, use_container_width=True)
            display_data_table_with_expander( 
                df_dd_hist.rename(columns={drawdown_csv_col_actual_hist: "Drawdown Value"}), 
                "View Data: Drawdown Distribution (raw values)",
                f"{section_key_prefix}_exp_dd_hist_raw"
            )

def render_exit_directional_insights(
    df: pd.DataFrame, win_col_actual: str, trade_result_col_actual: str, 
    plot_theme: str, section_key_prefix: str, **kwargs 
):
    col6a, col6b = st.columns(2)
    with col6a:
        exit_type_col_actual = get_column_name(EXIT_TYPE_CSV_KEY, df.columns)
        if exit_type_col_actual:
            exit_type_data = df[exit_type_col_actual].fillna('N/A').value_counts().reset_index()
            exit_type_data.columns = [exit_type_col_actual, 'count']
            fig_exit_type = plot_donut_chart(
                df=exit_type_data, category_col=exit_type_col_actual, value_col='count',
                title=f"{PERFORMANCE_TABLE_SELECTABLE_CATEGORIES.get(EXIT_TYPE_CSV_KEY, EXIT_TYPE_CSV_KEY).replace('_',' ').title()} Distribution", theme=plot_theme,
                is_data_aggregated=True
            )
            if fig_exit_type: st.plotly_chart(fig_exit_type, use_container_width=True)
            display_data_table_with_expander( 
                exit_type_data, "View Data: Exit Type Distribution",
                f"{section_key_prefix}_exp_exit_type_dist"
            )
    with col6b:
        direction_col_actual_wr = get_column_name(DIRECTION_KEY, df.columns)
        if direction_col_actual_wr and win_col_actual in df.columns: 
            dir_wr_data = df.groupby(direction_col_actual_wr, observed=False)[win_col_actual].agg(['mean', 'count']).reset_index()
            dir_wr_data['mean'] *= 100 
            dir_wr_data.rename(columns={'mean': 'Win Rate (%)', 'count': 'Total Trades'}, inplace=True)
            fig_dir_wr = plot_win_rate_analysis(
                df=dir_wr_data, category_col=direction_col_actual_wr, win_rate_col='Win Rate (%)', trades_col='Total Trades',
                title_prefix="Win Rate by", theme=plot_theme, is_data_aggregated=True
            )
            if fig_dir_wr: st.plotly_chart(fig_dir_wr, use_container_width=True)
            display_data_table_with_expander( 
                dir_wr_data, "View Data: Win Rate by Direction",
                f"{section_key_prefix}_exp_dir_wr_data"
            )
    st.markdown("---")
    time_frame_col_actual_facet = get_column_name(TIME_FRAME_KEY, df.columns)
    direction_col_actual_facet = get_column_name(DIRECTION_KEY, df.columns) 
    if direction_col_actual_facet and time_frame_col_actual_facet and trade_result_col_actual in df.columns:
        unique_time_frames = sorted(df[time_frame_col_actual_facet].astype(str).dropna().unique())
        if not unique_time_frames:
            display_custom_message(f"No unique values found in '{time_frame_col_actual_facet}' for faceted chart selection.", "info")
        else:
            default_selected_time_frames = unique_time_frames[:3] if len(unique_time_frames) > 3 else unique_time_frames
            selected_time_frames_for_facet = st.multiselect(
                f"Select Time Frames for Faceted Chart (Max 5 recommended for clarity):",
                options=unique_time_frames, default=default_selected_time_frames,
                key=f"{section_key_prefix}_facet_time_frame_select"
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
                            facet_col_wrap_val = min(3, len(selected_time_frames_for_facet))
                            fig_result_dir_tf = px.bar(
                                df_grouped_facet_data, x=direction_col_actual_facet, y='count', color=trade_result_col_actual,
                                facet_col=time_frame_col_actual_facet, facet_col_wrap=facet_col_wrap_val,
                                title=f"{trade_result_col_actual.replace('_',' ').title()} by Direction and Selected Time Frames",
                                labels={'count': "Number of Trades"}, barmode='group',
                                color_discrete_map={'WIN': COLORS.get('green'), 'LOSS': COLORS.get('red'), 'BREAKEVEN': COLORS.get('gray')}
                            )
                            if fig_result_dir_tf: st.plotly_chart(_apply_custom_theme(fig_result_dir_tf, plot_theme), use_container_width=True)
                            display_data_table_with_expander( 
                                df_grouped_facet_data, "View Data: Faceted Trade Results",
                                f"{section_key_prefix}_exp_faceted_results"
                            )
                        else: display_custom_message("No data for Trade Result by Direction and selected Time Frames after grouping.", "info")
                    except Exception as e_gbtf: logger.error(f"Error in Trade Result by Direction and Time Frame: {e_gbtf}", exc_info=True)
    else:
        display_custom_message(f"Missing columns for Trade Result by Direction & Time Frame. Needed: '{direction_col_actual_facet}', '{time_frame_col_actual_facet}', '{trade_result_col_actual}'.", "warning")

def render_performance_summary_table(
    df: pd.DataFrame, pnl_col_actual: str, win_col_actual: str, section_key_prefix: str
):
    available_categories_for_table: Dict[str, str] = {}
    for conceptual_key, display_name in PERFORMANCE_TABLE_SELECTABLE_CATEGORIES.items():
        actual_col = get_column_name(conceptual_key, df.columns)
        if actual_col and actual_col in df.columns and not df[actual_col].dropna().astype(str).str.strip().empty:
            available_categories_for_table[display_name] = actual_col
    
    if not available_categories_for_table:
        display_custom_message("No suitable categorical columns found for the summary table.", "warning")
    else:
        selected_display_name_table = st.selectbox(
            "Select Category for Performance Summary:",
            options=list(available_categories_for_table.keys()),
            key=f"{section_key_prefix}_custom_category_summary_select"
        )
        metrics_for_ci_options = ["Average PnL", "Win Rate %"]
        selected_cis_to_calculate = st.multiselect(
            "Calculate Confidence Intervals for:",
            options=metrics_for_ci_options, default=metrics_for_ci_options,
            key=f"{section_key_prefix}_ci_metric_select"
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

def render_dynamic_category_visualizer(
    df: pd.DataFrame, pnl_col_actual: str, win_col_actual: str, 
    plot_theme: str, section_key_prefix: str
):
    available_categories_for_dynamic_plot: Dict[str, str] = {}
    for conceptual_key, display_name in PERFORMANCE_TABLE_SELECTABLE_CATEGORIES.items():
        actual_col = get_column_name(conceptual_key, df.columns)
        if actual_col and actual_col in df.columns and not df[actual_col].dropna().astype(str).str.strip().empty:
            available_categories_for_dynamic_plot[display_name] = actual_col

    if not available_categories_for_dynamic_plot:
        display_custom_message("No suitable categorical columns found for dynamic visualization.", "warning")
        return

    col_cat_select, col_metric_select, col_chart_select = st.columns(3)
    with col_cat_select:
        selected_cat_display_name_dynamic = st.selectbox(
            "Select Category to Analyze:", options=list(available_categories_for_dynamic_plot.keys()),
            key=f"{section_key_prefix}_dynamic_cat_select"
        )
        actual_selected_category_col = available_categories_for_dynamic_plot.get(selected_cat_display_name_dynamic)
    with col_metric_select:
        metric_options_dynamic = ["Total PnL", "Average PnL", "Win Rate (%)", "Trade Count", "PnL Distribution"]
        selected_metric_dynamic = st.selectbox(
            "Select Metric to Visualize:", options=metric_options_dynamic,
            key=f"{section_key_prefix}_dynamic_metric_select"
        )
    
    chart_type_options_dynamic = ["Bar Chart"]
    if selected_metric_dynamic == "Trade Count": chart_type_options_dynamic.append("Donut Chart")
    elif selected_metric_dynamic == "PnL Distribution": chart_type_options_dynamic = ["Box Plot"]
    elif selected_metric_dynamic in ["Total PnL", "Average PnL"]: chart_type_options_dynamic.append("Box Plot")
    
    with col_chart_select:
        selected_chart_type_dynamic = st.selectbox(
            "Select Chart Type:", options=chart_type_options_dynamic, 
            key=f"{section_key_prefix}_dynamic_chart_type_select"
        )

    filter_type_dynamic = "Show All"; num_n_dynamic = 5
    sort_metric_for_top_n = selected_metric_dynamic
    show_others_dynamic = False

    if selected_metric_dynamic != "PnL Distribution" and selected_chart_type_dynamic in ["Bar Chart", "Donut Chart"]:
        filter_type_dynamic = st.radio(
            "Filter Categories by Metric Value:", ("Show All", "Top N", "Bottom N"), index=0,
            key=f"{section_key_prefix}_dynamic_filter_type", horizontal=True
        )
        if filter_type_dynamic != "Show All":
            top_n_cols = st.columns([2,1])
            with top_n_cols[0]:
                sort_metric_for_top_n = st.selectbox(
                    "Rank categories by:", options=[m for m in metric_options_dynamic if m != "PnL Distribution"],
                    index=metric_options_dynamic.index(selected_metric_dynamic) if selected_metric_dynamic in metric_options_dynamic[:-1] else 0,
                    key=f"{section_key_prefix}_dynamic_sort_metric_top_n"
                )
            with top_n_cols[1]:
                num_n_dynamic = st.number_input(
                    f"N:", 1, 50, 5, 1, key=f"{section_key_prefix}_dynamic_num_n"
                )
            show_others_dynamic = st.checkbox("Group remaining into 'Others'", key=f"{section_key_prefix}_dynamic_show_others")
    
    dynamic_plot_df_for_view = pd.DataFrame()

    if actual_selected_category_col:
        df_dynamic_plot_data_source = df.copy()
        if filter_type_dynamic != "Show All" and selected_metric_dynamic != "PnL Distribution" and selected_chart_type_dynamic in ["Bar Chart", "Donut Chart"]:
            if not df_dynamic_plot_data_source.empty:
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
                            df_others_plot[actual_selected_category_col] = "Others"
                            df_dynamic_plot_data = pd.concat([df_top_n_plot, df_others_plot], ignore_index=True)
                        else: df_dynamic_plot_data = df_top_n_plot
                    else: df_dynamic_plot_data = df_dynamic_plot_data_source[df_dynamic_plot_data_source[actual_selected_category_col].isin(top_n_cat_names)].copy()
                else: 
                    logger.warning(f"Could not rank categories for Top/Bottom N based on {sort_metric_for_top_n}.")
                    df_dynamic_plot_data = pd.DataFrame() 
            else: df_dynamic_plot_data = pd.DataFrame()
        else: df_dynamic_plot_data = df_dynamic_plot_data_source 

        fig_dynamic = None
        title_dynamic = f"{selected_metric_dynamic} by {selected_cat_display_name_dynamic}"
        if filter_type_dynamic != "Show All": title_dynamic += f" ({filter_type_dynamic} {num_n_dynamic} by {sort_metric_for_top_n})"
        if show_others_dynamic and filter_type_dynamic != "Show All": title_dynamic += " with Others"

        if df_dynamic_plot_data.empty:
            if filter_type_dynamic != "Show All": display_custom_message(f"No data remains for '{selected_cat_display_name_dynamic}' after applying '{filter_type_dynamic} {num_n_dynamic}' filter.", "info")
            else: display_custom_message(f"No data available for '{selected_cat_display_name_dynamic}'.", "info")
        else:
            logger.debug(f"Dynamic Plot: Category='{actual_selected_category_col}', Metric='{selected_metric_dynamic}', Chart='{selected_chart_type_dynamic}'")
            try:
                if selected_metric_dynamic == "Total PnL":
                    dynamic_plot_df_for_view = df_dynamic_plot_data.groupby(actual_selected_category_col, observed=False)[pnl_col_actual].sum().reset_index()
                    if selected_chart_type_dynamic == "Bar Chart":
                        fig_dynamic = plot_pnl_by_category(df=dynamic_plot_df_for_view, category_col=actual_selected_category_col, pnl_col=pnl_col_actual, title_prefix=title_dynamic, aggregation_func='sum', theme=plot_theme, is_data_aggregated=True)
                    elif selected_chart_type_dynamic == "Box Plot":
                        fig_dynamic = plot_box_plot(df=df_dynamic_plot_data, category_col=actual_selected_category_col, value_col=pnl_col_actual, title=title_dynamic, theme=plot_theme)
                        dynamic_plot_df_for_view = df_dynamic_plot_data[[actual_selected_category_col, pnl_col_actual]].copy()
                
                elif selected_metric_dynamic == "Average PnL":
                    dynamic_plot_df_for_view = df_dynamic_plot_data.groupby(actual_selected_category_col, observed=False)[pnl_col_actual].mean().reset_index()
                    if selected_chart_type_dynamic == "Bar Chart":
                        fig_dynamic = plot_pnl_by_category(df=dynamic_plot_df_for_view, category_col=actual_selected_category_col, pnl_col=pnl_col_actual, title_prefix=title_dynamic, aggregation_func='mean', theme=plot_theme, is_data_aggregated=True)
                    elif selected_chart_type_dynamic == "Box Plot":
                        fig_dynamic = plot_box_plot(df=df_dynamic_plot_data, category_col=actual_selected_category_col, value_col=pnl_col_actual, title=title_dynamic, theme=plot_theme)
                        dynamic_plot_df_for_view = df_dynamic_plot_data[[actual_selected_category_col, pnl_col_actual]].copy()

                elif selected_metric_dynamic == "Win Rate (%)" and selected_chart_type_dynamic == "Bar Chart" and win_col_actual in df_dynamic_plot_data.columns:
                    dynamic_plot_df_for_view = df_dynamic_plot_data.groupby(actual_selected_category_col, observed=False)[win_col_actual].agg(['mean', 'count']).reset_index()
                    dynamic_plot_df_for_view['mean'] *= 100
                    dynamic_plot_df_for_view.rename(columns={'mean': 'Win Rate (%)', 'count': 'Total Trades'}, inplace=True)
                    fig_dynamic = plot_win_rate_analysis(df=dynamic_plot_df_for_view, category_col=actual_selected_category_col, win_rate_col='Win Rate (%)', trades_col='Total Trades', title_prefix=title_dynamic, theme=plot_theme, is_data_aggregated=True)
                
                elif selected_metric_dynamic == "Trade Count":
                    dynamic_plot_df_for_view = df_dynamic_plot_data.groupby(actual_selected_category_col, observed=False).size().reset_index(name='count').sort_values(by='count', ascending=False)
                    if selected_chart_type_dynamic == "Bar Chart":
                        fig_dynamic = px.bar(dynamic_plot_df_for_view, x=actual_selected_category_col, y='count', title=title_dynamic, color='count', color_continuous_scale=px.colors.sequential.Blues_r)
                        if fig_dynamic: fig_dynamic = _apply_custom_theme(fig_dynamic, plot_theme)
                    elif selected_chart_type_dynamic == "Donut Chart":
                        fig_dynamic = plot_donut_chart(df=dynamic_plot_df_for_view, category_col=actual_selected_category_col, value_col='count', title=title_dynamic, theme=plot_theme, is_data_aggregated=True)
                
                elif selected_metric_dynamic == "PnL Distribution" and selected_chart_type_dynamic == "Box Plot":
                    fig_dynamic = plot_box_plot(df=df_dynamic_plot_data, category_col=actual_selected_category_col, value_col=pnl_col_actual, title=title_dynamic, theme=plot_theme)
                    dynamic_plot_df_for_view = df_dynamic_plot_data[[actual_selected_category_col, pnl_col_actual]].copy()

                if fig_dynamic:
                    st.plotly_chart(fig_dynamic, use_container_width=True)
                    display_data_table_with_expander( 
                        dynamic_plot_df_for_view.reset_index(drop=True), 
                        f"View Data for: {title_dynamic}", 
                        f"{section_key_prefix}_exp_data_dynamic_{selected_cat_display_name_dynamic}_{selected_metric_dynamic}"
                    )
                
                category_groups_for_test = df_dynamic_plot_data[actual_selected_category_col].dropna().unique()
                if "Others" in category_groups_for_test: category_groups_for_test = [cat for cat in category_groups_for_test if cat != "Others"]
                
                if len(category_groups_for_test) >= 2:
                    if selected_metric_dynamic == "Average PnL" and selected_chart_type_dynamic == "Bar Chart": 
                        st.markdown("##### ANOVA F-test (Difference in Average PnL across categories)")
                        avg_pnl_data_for_anova = [df_dynamic_plot_data[df_dynamic_plot_data[actual_selected_category_col] == group][pnl_col_actual].dropna().values for group in category_groups_for_test]
                        avg_pnl_data_for_anova_filtered = [g_data for g_data in avg_pnl_data_for_anova if len(g_data) >= 2] 
                        if len(avg_pnl_data_for_anova_filtered) >= 2: 
                            anova_results = statistical_service.run_hypothesis_test(data1=avg_pnl_data_for_anova_filtered, test_type='anova')
                            if 'error' in anova_results: st.caption(f"ANOVA Test Error: {anova_results['error']}")
                            else: st.metric(label="ANOVA P-value", value=f"{anova_results.get('p_value', np.nan):.4f}", help=anova_results.get('interpretation', 'Lower p-value suggests significant difference in means.'))
                        else: st.caption("ANOVA Test: Not enough groups with sufficient data (min 2 groups, 2 obs/group).")
                    
                    elif selected_metric_dynamic == "Win Rate (%)" and selected_chart_type_dynamic == "Bar Chart": 
                        st.markdown("##### Chi-squared Test (Difference in Win Rates across categories)")
                        contingency_table_data = []
                        valid_groups_for_chi2 = 0
                        for group in category_groups_for_test:
                            group_data = df_dynamic_plot_data[df_dynamic_plot_data[actual_selected_category_col] == group]
                            if not group_data.empty and win_col_actual in group_data.columns:
                                wins = group_data[win_col_actual].sum()
                                losses = len(group_data) - wins
                                if wins + losses >= 5 : 
                                    contingency_table_data.append([wins, losses])
                                    valid_groups_for_chi2 +=1
                        if valid_groups_for_chi2 >= 2 and len(contingency_table_data) >=2 :
                            chi2_results = statistical_service.run_hypothesis_test(data1=np.array(contingency_table_data), test_type='chi-squared')
                            if 'error' in chi2_results: st.caption(f"Chi-squared Test Error: {chi2_results['error']}")
                            else: st.metric(label="Chi-squared P-value", value=f"{chi2_results.get('p_value', np.nan):.4f}", help=chi2_results.get('interpretation', 'Lower p-value suggests significant difference in win rates.'))
                        else: st.caption("Chi-squared Test: Not enough groups or observations per group for a reliable test.")
            
            except Exception as e_dynamic_plot:
                logger.error(f"Error generating dynamic plot for {selected_cat_display_name_dynamic} ({selected_metric_dynamic} / {selected_chart_type_dynamic}): {e_dynamic_plot}", exc_info=True)
                display_custom_message(f"An error occurred while generating the dynamic chart: {e_dynamic_plot}", "error")
    else:
        display_custom_message("Please select a valid category to visualize.", "info")


# --- Main Page Function ---
def show_categorical_analysis_page():
    st.title("🎯 Categorical Performance Analysis")
    logger.info("Rendering Categorical Analysis Page.")

    if 'filtered_data' not in st.session_state or st.session_state.filtered_data is None:
        display_custom_message("Please upload and process data to view categorical analysis.", "info")
        return

    df = st.session_state.filtered_data
    plot_theme = st.session_state.get('current_theme', PLOTLY_THEME_DARK if st.session_state.get('theme', 'dark') == 'dark' else PLOTLY_THEME_LIGHT)


    pnl_col_actual = get_column_name(PNL_KEY, df.columns)
    win_col_actual = 'win' 
    trade_result_col_actual = 'trade_result_processed' 
    date_col_actual = get_column_name(DATE_KEY, df.columns)

    if df.empty:
        display_custom_message("No data matches the current filters. Cannot perform categorical analysis.", "info")
        return
    if not pnl_col_actual:
        display_custom_message(f"Essential PnL column ('{EXPECTED_COLUMNS.get(PNL_KEY, PNL_KEY)}') not found. Analysis cannot proceed.", "error")
        logger.error(f"Essential PnL column ('{EXPECTED_COLUMNS.get(PNL_KEY, PNL_KEY)}') not found in DataFrame columns: {df.columns.tolist()}")
        return
    if win_col_actual not in df.columns:
        logger.warning(f"Engineered Win column ('{win_col_actual}') not found. Some analyses may be affected or default to PnL > 0.")
        if pnl_col_actual and pd.api.types.is_numeric_dtype(df[pnl_col_actual]):
            df[win_col_actual] = df[pnl_col_actual] > 0
            logger.info(f"Created '{win_col_actual}' column based on '{pnl_col_actual}' > 0.")
        else:
            display_custom_message(f"Engineered Win column ('{win_col_actual}') not found and could not be created. Some analyses will be impacted.", "warning")

    if trade_result_col_actual not in df.columns:
        logger.warning(f"Engineered Trade Result column ('{trade_result_col_actual}') not found. Some analyses may be affected.")
        if pnl_col_actual and win_col_actual in df.columns : 
            df[trade_result_col_actual] = np.select(
                [df[pnl_col_actual] > 0, df[pnl_col_actual] < 0],
                ['WIN', 'LOSS'],
                default='BREAKEVEN'
            )
            logger.info(f"Created '{trade_result_col_actual}' column based on PnL.")
        else:
            display_custom_message(f"Engineered Trade Result column ('{trade_result_col_actual}') not found and could not be created. Some analyses will be impacted.", "warning")

    if not date_col_actual:
        logger.warning(f"Date column ('{EXPECTED_COLUMNS.get(DATE_KEY, DATE_KEY)}') not found. Some temporal analyses may be affected.")

    common_render_args = {
        "pnl_col_actual": pnl_col_actual,
        "win_col_actual": win_col_actual,
        "trade_result_col_actual": trade_result_col_actual,
        "date_col_actual": date_col_actual,
        "plot_theme": plot_theme,
    }

    tab_titles = [
        "💡 Strategy Insights",
        "⏳ Temporal Analysis",
        "🌍 Market Context",
        "🤔 Behavioral Factors",
        "💰 Capital & Risk",
        "🚪 Exit & Directional",
        "📊 Performance Summary",
        "🔬 Dynamic Visualizer"
    ]
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(tab_titles)

    with tab1:
        st.subheader("Strategy Performance Insights")
        render_strategy_performance_insights(df, **common_render_args, section_key_prefix="s1")
    
    with tab2:
        st.subheader("Temporal Analysis")
        render_temporal_analysis(df, **common_render_args, section_key_prefix="s2")

    with tab3:
        st.subheader("Market Context Impact")
        render_market_context_impact(df, **common_render_args, section_key_prefix="s3")

    with tab4:
        st.subheader("Behavioral Factors")
        render_behavioral_factors(df, **common_render_args, section_key_prefix="s4")

    with tab5:
        st.subheader("Capital & Risk Insights")
        render_capital_risk_insights(df, **common_render_args, section_key_prefix="s5")

    with tab6:
        st.subheader("Exit & Directional Insights")
        render_exit_directional_insights(df, **common_render_args, section_key_prefix="s6")

    with tab7:
        st.subheader("Performance Summary by Custom Category")
        if pnl_col_actual and win_col_actual in df.columns : 
            render_performance_summary_table(df, pnl_col_actual, win_col_actual, section_key_prefix="s7")
        else:
            display_custom_message("Cannot render Performance Summary Table due to missing PnL or Win columns.", "warning")

    with tab8:
        st.subheader("Dynamic Category Visualizer")
        if pnl_col_actual and win_col_actual in df.columns : 
            render_dynamic_category_visualizer(df, pnl_col_actual, win_col_actual, plot_theme, section_key_prefix="s8")
        else:
            display_custom_message("Cannot render Dynamic Category Visualizer due to missing PnL or Win columns.", "warning")


if __name__ == "__main__":
    if 'app_initialized' not in st.session_state: 
        st.warning("This page is part of a multi-page app. Please run the main app.py script for full functionality.")
    show_categorical_analysis_page()
