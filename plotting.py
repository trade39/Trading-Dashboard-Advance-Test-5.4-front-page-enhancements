"""
plotting.py

Contains functions to generate various interactive Plotly visualizations
for the Trading Performance Dashboard.
Includes advanced drawdown visualizations and highlighting for max drawdown.
Heatmap text formatting for currency is corrected.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Optional, Any, Union
from scipy import stats as scipy_stats # For distribution fitting plots


# Assuming config.py and utils.common_utils are in a path accessible by Python
try:
    from config import (
        COLORS, PLOTLY_THEME_DARK, PLOTLY_THEME_LIGHT,
        PLOT_BG_COLOR_DARK, PLOT_PAPER_BG_COLOR_DARK, PLOT_FONT_COLOR_DARK,
        PLOT_BG_COLOR_LIGHT, PLOT_PAPER_BG_COLOR_LIGHT, PLOT_FONT_COLOR_LIGHT,
        PLOT_LINE_COLOR, PLOT_MARKER_PROFIT_COLOR, PLOT_MARKER_LOSS_COLOR,
        PLOT_BENCHMARK_LINE_COLOR,
        EXPECTED_COLUMNS, APP_TITLE
    )
    from utils.common_utils import format_currency, format_percentage
except ImportError:
    # Fallback for environments where these might not be directly available
    print("Warning: Could not import from config or utils.common_utils. Using placeholder values.")
    COLORS = {'red': '#FF0000', 'green': '#00FF00', 'royal_blue': '#4169E1', 'gray': '#808080', 'orange': '#FFA500',
              'card_background_dark': '#273334', 'card_background_light': '#F0F2F6'}
    PLOTLY_THEME_DARK = 'plotly_dark'
    PLOTLY_THEME_LIGHT = 'plotly_white'
    PLOT_BG_COLOR_DARK = '#1E1E1E'
    PLOT_PAPER_BG_COLOR_DARK = '#1E1E1E'
    PLOT_FONT_COLOR_DARK = '#FFFFFF'
    PLOT_BG_COLOR_LIGHT = '#FFFFFF'
    PLOT_PAPER_BG_COLOR_LIGHT = '#FFFFFF'
    PLOT_FONT_COLOR_LIGHT = '#000000'
    PLOT_LINE_COLOR = COLORS.get('royal_blue')
    PLOT_BENCHMARK_LINE_COLOR = COLORS.get('orange')
    EXPECTED_COLUMNS = {'date': 'Date', 'pnl': 'PnL'}
    APP_TITLE = "TradingApp"
    def format_currency(value, currency_symbol='$', decimals=2): return f"{currency_symbol}{value:,.{decimals}f}"
    def format_percentage(value, decimals=2): return f"{value:.{decimals}%}"


import logging
logger = logging.getLogger(APP_TITLE if 'APP_TITLE' in globals() else "PlottingModule")


def _apply_custom_theme(fig: go.Figure, theme: str = 'dark') -> go.Figure:
    """
    Applies a custom theme (dark or light) to a Plotly figure.
    """
    plotly_theme_template = PLOTLY_THEME_DARK if theme == 'dark' else PLOTLY_THEME_LIGHT
    bg_color = PLOT_BG_COLOR_DARK if theme == 'dark' else PLOT_BG_COLOR_LIGHT
    paper_bg_color = PLOT_PAPER_BG_COLOR_DARK if theme == 'dark' else PLOT_PAPER_BG_COLOR_LIGHT
    font_color = PLOT_FONT_COLOR_DARK if theme == 'dark' else PLOT_FONT_COLOR_LIGHT
    grid_color = COLORS.get('gray', '#808080') if theme == 'dark' else '#e0e0e0'

    fig.update_layout(
        template=plotly_theme_template,
        plot_bgcolor=bg_color, paper_bgcolor=paper_bg_color, font_color=font_color,
        margin=dict(l=50, r=50, t=60, b=50),
        xaxis=dict(showgrid=True, gridcolor=grid_color, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor=grid_color, zeroline=False),
        hoverlabel=dict(
            bgcolor=COLORS.get('card_background_dark', '#273334') if theme == 'dark' else COLORS.get('card_background_light', '#F0F2F6'),
            font_size=12, font_family="Inter, sans-serif", bordercolor=COLORS.get('royal_blue')
        )
    )
    return fig

def plot_heatmap(
    df_pivot: pd.DataFrame,
    title: str = "Heatmap",
    x_axis_title: Optional[str] = None,
    y_axis_title: Optional[str] = None,
    color_scale: str = "RdBu",
    z_min: Optional[float] = None,
    z_max: Optional[float] = None,
    show_text: bool = True,
    text_format: str = ".2f",
    theme: str = 'dark'
) -> Optional[go.Figure]:
    """
    Generates an interactive heatmap from a pivot DataFrame.
    """
    if df_pivot is None or df_pivot.empty:
        logger.warning("Heatmap: Input pivot DataFrame is empty.")
        return None

    formatted_text_values = None
    if show_text:
        def format_cell_value(val):
            if pd.isna(val):
                return ""
            is_currency = text_format.startswith('$')
            is_percentage = text_format.endswith('%')
            numeric_format_part = text_format
            prefix = ""
            suffix = ""
            if is_currency:
                prefix = "$"
                numeric_format_part = numeric_format_part[1:]
            if is_percentage:
                suffix = "%"
                numeric_format_part = numeric_format_part[:-1]
            try:
                formatted_num = f"{val:{numeric_format_part}}"
                return f"{prefix}{formatted_num}{suffix}"
            except ValueError:
                logger.warning(f"Heatmap: Could not apply format '{text_format}' to value '{val}'. Returning raw value.")
                return str(val)
        formatted_text_values = df_pivot.map(format_cell_value).values

    fig = go.Figure(data=go.Heatmap(
        z=df_pivot.values,
        x=df_pivot.columns,
        y=df_pivot.index,
        colorscale=color_scale,
        zmin=z_min,
        zmax=z_max,
        text=formatted_text_values if show_text else None,
        texttemplate="%{text}" if show_text and formatted_text_values is not None else None,
        hoverongaps=False,
        hovertemplate="<b>%{y}</b><br>%{x}: %{z}<extra></extra>"
    ))
    fig.update_layout(
        title_text=title,
        xaxis_title=x_axis_title if x_axis_title else df_pivot.columns.name,
        yaxis_title=y_axis_title if y_axis_title else df_pivot.index.name
    )
    return _apply_custom_theme(fig, theme)


def _add_max_dd_shading_to_plot(
    fig: go.Figure,
    df_dates: pd.Series,
    max_dd_peak_date: Optional[Any],
    max_dd_trough_date: Optional[Any],
    max_dd_recovery_date: Optional[Any],
    row: int,
    col: int
) -> None:
    """Internal helper for adding max drawdown shading."""
    if not (max_dd_peak_date and max_dd_trough_date):
        return

    try:
        peak_dt = pd.to_datetime(max_dd_peak_date)
        trough_dt = pd.to_datetime(max_dd_trough_date)
        end_shade_dt = None

        if pd.notna(max_dd_recovery_date):
            end_shade_dt = pd.to_datetime(max_dd_recovery_date)
        elif not df_dates.empty:
            end_shade_dt = df_dates.max()

        if end_shade_dt is None or peak_dt >= end_shade_dt:
            if peak_dt < trough_dt:
                 end_shade_dt = trough_dt
            else:
                return

        if peak_dt < end_shade_dt:
            annotation_text_val = "Max Drawdown Period"
            if pd.notna(max_dd_recovery_date) and pd.to_datetime(max_dd_recovery_date) == end_shade_dt:
                pass 
            elif trough_dt == end_shade_dt:
                 annotation_text_val = "Max DD (Peak to Trough)"


            fig.add_vrect(
                x0=peak_dt,
                x1=end_shade_dt,
                fillcolor=COLORS.get('red', 'red'),
                opacity=0.25,
                layer="below",
                line_width=1,
                line_color=COLORS.get('red', 'red'),
                annotation_text=annotation_text_val,
                annotation_position="top left",
                annotation_font_size=10,
                annotation_font_color=COLORS.get('red', 'red'),
                row=row, col=col
            )
    except Exception as e_vrect:
        logger.error(f"Error adding max drawdown vrect: {e_vrect}", exc_info=True)


def _add_drawdown_period_shading_to_plot(
    fig: go.Figure,
    df_dates: pd.Series,
    drawdown_periods_df: Optional[pd.DataFrame],
    max_dd_peak_date_for_exclusion: Optional[Any],
    row: int,
    col: int
) -> None:
    """Internal helper for adding general drawdown period shading."""
    if drawdown_periods_df is None or drawdown_periods_df.empty:
        return

    for _, dd_period in drawdown_periods_df.iterrows():
        try:
            peak_date = pd.to_datetime(dd_period.get('Peak Date'))
            end_date_for_shading = pd.to_datetime(dd_period.get('End Date'))

            if pd.isna(peak_date):
                continue

            if pd.isna(end_date_for_shading):
                if not df_dates.empty:
                    last_data_date = pd.to_datetime(df_dates.iloc[-1])
                    if last_data_date > peak_date:
                        end_date_for_shading = last_data_date
                    else:
                        continue
                else:
                    continue

            if peak_date < end_date_for_shading:
                is_max_dd_period = False
                if max_dd_peak_date_for_exclusion:
                    try:
                        if pd.to_datetime(max_dd_peak_date_for_exclusion) == peak_date:
                            is_max_dd_period = True
                    except Exception: 
                        pass

                if not is_max_dd_period:
                    fig.add_vrect(
                        x0=peak_date,
                        x1=end_date_for_shading,
                        fillcolor=COLORS.get('red', 'red'),
                        opacity=0.10,
                        layer="below",
                        line_width=0,
                        row=row, col=col
                    )
        except Exception as e:
            logger.error(f"Error adding generic drawdown period shading for peak {dd_period.get('Peak Date')}: {e}", exc_info=True)


def plot_equity_curve_and_drawdown(
    df: pd.DataFrame,
    date_col: str = EXPECTED_COLUMNS['date'],
    cumulative_pnl_col: str = 'cumulative_pnl',
    drawdown_pct_col: Optional[str] = 'drawdown_pct',
    drawdown_periods_df: Optional[pd.DataFrame] = None,
    theme: str = 'dark',
    max_dd_peak_date: Optional[Any] = None,
    max_dd_trough_date: Optional[Any] = None,
    max_dd_recovery_date: Optional[Any] = None
) -> Optional[go.Figure]:
    """
    Generates a plot with the equity curve and optionally the drawdown percentage over time.
    """
    if df is None or df.empty or date_col not in df.columns or cumulative_pnl_col not in df.columns:
        logger.warning("Equity curve plot: Input DataFrame is invalid or missing required columns.")
        return None

    df_copy = df.copy() 
    try:
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    except Exception as e:
        logger.error(f"Equity curve plot: Could not convert date column '{date_col}' to datetime: {e}")
        return None

    has_drawdown_data_series = drawdown_pct_col and drawdown_pct_col in df_copy.columns and not df_copy[drawdown_pct_col].dropna().empty
    fig_rows, row_heights = (2, [0.7, 0.3]) if has_drawdown_data_series else (1, [1.0])
    subplot_titles_list = ["Equity Curve"] + (["Drawdown (%)"] if has_drawdown_data_series else [])

    fig = make_subplots(
        rows=fig_rows, cols=1, shared_xaxes=True,
        vertical_spacing=0.05, row_heights=row_heights,
        subplot_titles=subplot_titles_list
    )

    fig.add_trace(
        go.Scatter(
            x=df_copy[date_col], y=df_copy[cumulative_pnl_col],
            mode='lines', name='Strategy Equity',
            line=dict(color=PLOT_LINE_COLOR, width=2)
        ),
        row=1, col=1
    )
    fig.update_yaxes(title_text="Cumulative PnL ($)", row=1, col=1)

    _add_max_dd_shading_to_plot(
        fig, df_dates=df_copy[date_col],
        max_dd_peak_date=max_dd_peak_date,
        max_dd_trough_date=max_dd_trough_date,
        max_dd_recovery_date=max_dd_recovery_date,
        row=1, col=1
    )

    _add_drawdown_period_shading_to_plot(
        fig, df_dates=df_copy[date_col],
        drawdown_periods_df=drawdown_periods_df,
        max_dd_peak_date_for_exclusion=max_dd_peak_date,
        row=1, col=1
    )

    if has_drawdown_data_series:
        fig.add_trace(
            go.Scatter(
                x=df_copy[date_col], y=df_copy[drawdown_pct_col],
                mode='lines', name='Drawdown',
                line=dict(color=COLORS.get('red', '#FF0000'), width=1.5),
                fill='tozeroy', fillcolor='rgba(255,0,0,0.2)'
            ),
            row=2, col=1
        )
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1, tickformat=".2f") 
        min_dd_val = df_copy[drawdown_pct_col].min(skipna=True)
        max_dd_val = df_copy[drawdown_pct_col].max(skipna=True)
        if pd.isna(min_dd_val) or pd.isna(max_dd_val) or (min_dd_val == 0 and max_dd_val == 0) :
            fig.update_yaxes(range=[-1, 1], row=2, col=1)

    fig.update_layout(title_text='Strategy Equity and Drawdown Periods', hovermode='x unified')
    return _apply_custom_theme(fig, theme)

def plot_underwater_analysis(
    equity_series: pd.Series,
    theme: str = 'dark',
    title: str = "Underwater Plot (Equity vs. High Water Mark)"
) -> Optional[go.Figure]:
    """
    Generates an underwater plot showing the equity curve against its high water mark.
    """
    if equity_series is None or equity_series.empty:
        logger.warning("Underwater plot: Equity series is empty.")
        return None
    if not isinstance(equity_series.index, pd.DatetimeIndex):
        logger.warning("Underwater plot: Equity series index must be DatetimeIndex.")
        return None
    if len(equity_series.dropna()) < 2:
        logger.warning("Underwater plot: Not enough data points in equity series.")
        return None

    equity = equity_series.dropna()
    high_water_mark = equity.cummax()

    fig_filled = go.Figure()
    fig_filled.add_trace(go.Scatter(
        x=high_water_mark.index, y=high_water_mark,
        mode='lines', name='High Water Mark',
        line=dict(color=COLORS.get('green', 'green'), dash='dash')
    ))
    fig_filled.add_trace(go.Scatter(
        x=equity.index, y=equity,
        mode='lines', name='Equity Curve',
        line=dict(color=PLOT_LINE_COLOR),
        fill='tonexty', fillcolor='rgba(255, 0, 0, 0.2)'
    ))
    fig_filled.update_layout(
        title_text=title,
        xaxis_title="Date",
        yaxis_title="Equity Value",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return _apply_custom_theme(fig_filled, theme)

def plot_equity_vs_benchmark(
    strategy_equity: pd.Series,
    benchmark_cumulative_returns: pd.Series,
    strategy_name: str = "Strategy",
    benchmark_name: str = "Benchmark",
    theme: str = 'dark'
) -> Optional[go.Figure]:
    """
    Plots strategy equity against benchmark cumulative returns.
    """
    if strategy_equity.empty and benchmark_cumulative_returns.empty:
        logger.warning("Equity vs Benchmark: Both strategy and benchmark series are empty.")
        return None

    fig = go.Figure()
    if not strategy_equity.empty:
        fig.add_trace(go.Scatter(
            x=strategy_equity.index, y=strategy_equity,
            mode='lines', name=strategy_name,
            line=dict(color=PLOT_LINE_COLOR, width=2)
        ))
    if not benchmark_cumulative_returns.empty:
        fig.add_trace(go.Scatter(
            x=benchmark_cumulative_returns.index, y=benchmark_cumulative_returns,
            mode='lines', name=benchmark_name,
            line=dict(color=PLOT_BENCHMARK_LINE_COLOR, width=2, dash='dash')
        ))

    fig.update_layout(
        title_text=f'{strategy_name} vs. {benchmark_name} Performance',
        xaxis_title="Date",
        yaxis_title="Normalized Value / Cumulative Return",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return _apply_custom_theme(fig, theme)

def plot_pnl_distribution(
    df: pd.DataFrame,
    pnl_col: str = EXPECTED_COLUMNS['pnl'],
    title: str = "PnL Distribution (per Trade)",
    theme: str = 'dark',
    nbins: int = 50
) -> Optional[go.Figure]:
    """
    Generates a histogram of Profit and Loss (PnL) per trade.
    """
    if df is None or df.empty or pnl_col not in df.columns or df[pnl_col].dropna().empty:
        logger.warning("PnL Distribution: Input DataFrame is invalid or PnL column is empty.")
        return None

    fig = px.histogram(
        df, x=pnl_col, nbins=nbins, title=title,
        marginal="box", color_discrete_sequence=[PLOT_LINE_COLOR]
    )
    fig.update_layout(xaxis_title="PnL per Trade", yaxis_title="Frequency")
    return _apply_custom_theme(fig, theme)

def plot_time_series_decomposition(
    decomposition_result: Any,
    title: str = "Time Series Decomposition",
    theme: str = 'dark'
) -> Optional[go.Figure]:
    """
    Plots components of a time series decomposition result.
    """
    if decomposition_result is None:
        logger.warning("Time Series Decomposition: Input decomposition_result is None.")
        return None

    try:
        observed = getattr(decomposition_result, 'observed', pd.Series(dtype=float))
        trend = getattr(decomposition_result, 'trend', pd.Series(dtype=float))
        seasonal = getattr(decomposition_result, 'seasonal', pd.Series(dtype=float))
        resid = getattr(decomposition_result, 'resid', pd.Series(dtype=float))

        if observed.dropna().empty:
            logger.warning("Time Series Decomposition: Observed series is empty after dropna.")
            return None

        x_axis = observed.index

        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=("Observed", "Trend", "Seasonal", "Residual")
        )

        fig.add_trace(go.Scatter(x=x_axis, y=observed, mode='lines', name='Observed', line=dict(color=PLOT_LINE_COLOR)), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_axis, y=trend, mode='lines', name='Trend', line=dict(color=COLORS.get('green', '#00FF00'))), row=2, col=1)
        fig.add_trace(go.Scatter(x=x_axis, y=seasonal, mode='lines', name='Seasonal', line=dict(color=COLORS.get('royal_blue', '#4169E1'))), row=3, col=1)
        fig.add_trace(go.Scatter(x=x_axis, y=resid, mode='lines+markers', name='Residual', line=dict(color=COLORS.get('gray', '#808080')), marker=dict(size=3)), row=4, col=1)

        fig.update_layout(title_text=title, height=700, showlegend=False)
        return _apply_custom_theme(fig, theme)
    except Exception as e:
        logger.error(f"Error plotting time series decomposition: {e}", exc_info=True)
        return None

def plot_value_over_time(
    series: pd.Series,
    series_name: str,
    title: Optional[str] = None,
    x_axis_title: str = "Date / Time",
    y_axis_title: Optional[str] = None,
    theme: str = 'dark',
    line_color: str = PLOT_LINE_COLOR
) -> Optional[go.Figure]:
    """
    Plots a single series of values over time.
    """
    if series is None or series.empty:
        logger.warning(f"Plot Value Over Time ('{series_name}'): Input series is empty.")
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode='lines', name=series_name, line=dict(color=line_color)))
    fig.update_layout(
        title_text=title if title else series_name,
        xaxis_title=x_axis_title,
        yaxis_title=y_axis_title if y_axis_title else series_name
    )
    return _apply_custom_theme(fig, theme)

def plot_pnl_by_category(
    df: pd.DataFrame,
    category_col: str,
    pnl_col: str = EXPECTED_COLUMNS['pnl'],
    title_prefix: str = "Total PnL by",
    theme: str = 'dark',
    aggregation_func: str = 'sum',
    is_data_aggregated: bool = False
) -> Optional[go.Figure]:
    """
    Generates a bar chart of PnL aggregated by a category.
    Can accept raw data (is_data_aggregated=False) or pre-aggregated data.
    """
    if df is None or df.empty or category_col not in df.columns or pnl_col not in df.columns:
        logger.warning("PnL by Category: Input DataFrame is invalid or missing required columns.")
        return None

    if is_data_aggregated:
        # If data is already aggregated, df is expected to have category_col and pnl_col (which contains aggregated values)
        grouped_pnl = df.sort_values(by=pnl_col, ascending=False)
    else:
        # Perform internal aggregation if data is raw
        try:
            grouped_pnl = df.groupby(category_col, observed=False)[pnl_col].agg(aggregation_func).reset_index().sort_values(by=pnl_col, ascending=False)
        except Exception as e:
            logger.error(f"PnL by Category: Error during aggregation '{aggregation_func}' on column '{pnl_col}' grouped by '{category_col}': {e}")
            return None

    if grouped_pnl.empty:
        logger.info(f"PnL by Category: No data to plot for category '{category_col}'.")
        return None

    # Determine y-axis title based on aggregation type (relevant if not pre-aggregated, but good for consistency)
    yaxis_title_agg_text = aggregation_func.title() if aggregation_func != 'sum' else "Total"
    if is_data_aggregated: # If pre-aggregated, title_prefix might already reflect the aggregation
        # Heuristic: if title_prefix contains 'Average' or 'Mean', assume mean. If 'Total' or 'Sum', assume sum.
        if "average" in title_prefix.lower() or "mean" in title_prefix.lower():
            yaxis_title_agg_text = "Average"
        elif "total" in title_prefix.lower() or "sum" in title_prefix.lower():
             yaxis_title_agg_text = "Total"
        # else, it might be a generic title_prefix, or pnl_col name itself is descriptive.

    plot_title = f"{title_prefix} {category_col.replace('_', ' ').title()}"
    y_axis_label = f"{yaxis_title_agg_text} {pnl_col.replace('_', ' ').title() if pnl_col != EXPECTED_COLUMNS['pnl'] else 'PnL'}"
    if is_data_aggregated: # If data is pre-aggregated, pnl_col is the value column
        y_axis_label = pnl_col.replace('_', ' ').title()


    fig = px.bar(
        grouped_pnl, x=category_col, y=pnl_col, title=plot_title,
        color=pnl_col,
        color_continuous_scale=[COLORS.get('red', '#FF0000'), COLORS.get('gray', '#808080'), COLORS.get('green', '#00FF00')]
    )
    fig.update_layout(
        xaxis_title=category_col.replace('_', ' ').title(),
        yaxis_title=y_axis_label
    )
    return _apply_custom_theme(fig, theme)

def plot_win_rate_analysis(
    df: pd.DataFrame,
    category_col: str,
    win_col: str = 'win', # Used if is_data_aggregated is False
    title_prefix: str = "Win Rate by",
    theme: str = 'dark',
    is_data_aggregated: bool = False,
    win_rate_col: str = 'win_rate_pct', # Used if is_data_aggregated is True (column with pre-calculated win rates)
    trades_col: Optional[str] = None # Optional: column with total trades for hover if data is pre-aggregated
) -> Optional[go.Figure]:
    """
    Generates a bar chart of win rates by category.
    Can accept raw data or pre-aggregated data.
    """
    if df is None or df.empty or category_col not in df.columns:
        logger.warning("Win Rate Analysis: Input DataFrame is invalid or category column is missing.")
        return None

    if is_data_aggregated:
        if win_rate_col not in df.columns:
            logger.error(f"Win Rate Analysis (Aggregated): Pre-aggregated data must contain win rate column '{win_rate_col}'.")
            return None
        win_rate_df = df.sort_values(by=win_rate_col, ascending=False).copy()
        # Ensure win_rate_col is numeric for plotting
        try:
            win_rate_df[win_rate_col] = pd.to_numeric(win_rate_df[win_rate_col])
        except ValueError:
            logger.error(f"Win Rate Analysis (Aggregated): Win rate column '{win_rate_col}' could not be converted to numeric.")
            return None
        y_col_for_plot = win_rate_col
    else:
        if win_col not in df.columns:
            logger.error(f"Win Rate Analysis (Raw): Raw data must contain win column '{win_col}'.")
            return None
        if not pd.api.types.is_bool_dtype(df[win_col]) and not pd.api.types.is_numeric_dtype(df[win_col]):
            logger.warning(f"Win Rate Analysis (Raw): Win column '{win_col}' must be boolean or numeric (0 or 1).")
            # Attempt conversion if possible, or return None if strict typing is required
            # For now, proceed assuming it can be cast to int for sum
            # return None 
        try:
            df_calc = df.copy()
            df_calc[win_col] = df_calc[win_col].astype(int) # Ensure it's int for sum
            category_counts = df_calc.groupby(category_col, observed=False).size().rename('total_trades_in_cat')
            category_wins = df_calc.groupby(category_col, observed=False)[win_col].sum().rename('wins_in_cat')
        except Exception as e:
            logger.error(f"Win Rate Analysis (Raw): Error during grouping or conversion: {e}")
            return None

        win_rate_df = pd.concat([category_counts, category_wins], axis=1).fillna(0)
        win_rate_df['win_rate_pct'] = 0.0 # Initialize column
        non_zero_trades_mask = win_rate_df['total_trades_in_cat'] > 0
        win_rate_df.loc[non_zero_trades_mask, 'win_rate_pct'] = \
            (win_rate_df.loc[non_zero_trades_mask, 'wins_in_cat'] / win_rate_df.loc[non_zero_trades_mask, 'total_trades_in_cat'] * 100)
        win_rate_df = win_rate_df.reset_index().sort_values(by='win_rate_pct', ascending=False)
        y_col_for_plot = 'win_rate_pct'


    if win_rate_df.empty:
        logger.info(f"Win Rate Analysis: No data to plot for category '{category_col}'.")
        return None

    hover_data = None
    if is_data_aggregated and trades_col and trades_col in win_rate_df.columns:
        hover_data = {trades_col: True}


    fig = px.bar(
        win_rate_df, x=category_col, y=y_col_for_plot,
        title=f"{title_prefix} {category_col.replace('_', ' ').title()}",
        color=y_col_for_plot, color_continuous_scale=px.colors.sequential.Greens,
        hover_data=hover_data
    )
    fig.update_layout(
        xaxis_title=category_col.replace('_', ' ').title(),
        yaxis_title="Win Rate (%)",
        yaxis_ticksuffix="%"
    )
    return _apply_custom_theme(fig, theme)

def plot_rolling_performance(
    df: Optional[pd.DataFrame],
    date_col: Optional[str],
    metric_series: pd.Series,
    metric_name: str,
    title: Optional[str] = None,
    theme: str = 'dark'
) -> Optional[go.Figure]:
    """
    Plots a rolling performance metric over time or by period number.
    """
    if metric_series.empty:
        logger.warning(f"Rolling Performance ('{metric_name}'): Metric series is empty.")
        return None

    plot_x_data = metric_series.index
    x_axis_title_text = metric_series.index.name if metric_series.index.name else "Period / Index"


    if df is not None and not df.empty and date_col and date_col in df.columns:
        if len(df[date_col]) == len(metric_series):
            try:
                plot_x_data_candidate = pd.to_datetime(df[date_col])
                # Check if metric_series.index is already DatetimeIndex and matches
                if isinstance(metric_series.index, pd.DatetimeIndex) and metric_series.index.equals(plot_x_data_candidate):
                    plot_x_data = plot_x_data_candidate # or metric_series.index, they are same
                    x_axis_title_text = "Date"
                # If metric_series.index is not DatetimeIndex, but df[date_col] is convertible and matches length
                elif not isinstance(metric_series.index, pd.DatetimeIndex): 
                    plot_x_data = plot_x_data_candidate
                    x_axis_title_text = "Date"
                # If metric_series.index is DatetimeIndex but doesn't match (e.g. resampled), prefer original metric_series.index
                elif isinstance(metric_series.index, pd.DatetimeIndex):
                     plot_x_data = metric_series.index 
                     x_axis_title_text = "Date"

            except Exception:
                logger.warning(f"Rolling Performance ('{metric_name}'): Could not convert '{date_col}' to datetime or align. Using metric series index.")
        else:
             logger.info(f"Rolling Performance ('{metric_name}'): Length mismatch between df['{date_col}'] and metric_series. Using metric_series index.")
    # If df/date_col not provided, but metric_series.index is already DatetimeIndex
    elif isinstance(metric_series.index, pd.DatetimeIndex): 
        plot_x_data = metric_series.index
        x_axis_title_text = "Date"


    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_x_data, y=metric_series,
        mode='lines', name=metric_name,
        line=dict(color=PLOT_LINE_COLOR)
    ))
    fig.update_layout(
        title_text=title if title else f"Rolling {metric_name}",
        xaxis_title=x_axis_title_text,
        yaxis_title=metric_name
    )
    return _apply_custom_theme(fig, theme)

def plot_correlation_matrix(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    title: str = "Correlation Matrix of Numeric Features",
    theme: str = 'dark'
) -> Optional[go.Figure]:
    """
    Generates a heatmap of the correlation matrix for numeric columns.
    """
    if df is None or df.empty:
        logger.warning("Correlation Matrix: Input DataFrame is empty.")
        return None

    if numeric_cols:
        df_numeric = df[numeric_cols].copy()
        non_numeric_selected = [col for col in numeric_cols if not pd.api.types.is_numeric_dtype(df_numeric[col])]
        if non_numeric_selected:
            logger.warning(f"Correlation Matrix: Specified columns {non_numeric_selected} are not numeric and will be excluded or cause errors.")
            df_numeric = df_numeric.select_dtypes(include=np.number)
    else:
        df_numeric = df.select_dtypes(include=np.number)

    if df_numeric.empty or df_numeric.shape[1] < 2:
        logger.warning("Correlation Matrix: No numeric columns or less than 2 numeric columns found for correlation.")
        return None

    corr_matrix = df_numeric.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu', zmin=-1, zmax=1,
        text=corr_matrix.round(2).astype(str), # Show text on heatmap cells
        texttemplate="%{text}",
        hoverongaps=False
    ))
    fig.update_layout(title_text=title)
    return _apply_custom_theme(fig, theme)

def plot_bootstrap_distribution_and_ci(
    bootstrap_statistics: List[float],
    observed_statistic: float,
    lower_bound: float,
    upper_bound: float,
    statistic_name: str,
    theme: str = 'dark'
) -> Optional[go.Figure]:
    """
    Plots the bootstrap distribution of a statistic, observed value, and CI bounds.
    """
    if not bootstrap_statistics or pd.isna(observed_statistic) or pd.isna(lower_bound) or pd.isna(upper_bound):
        logger.warning("Bootstrap Distribution Plot: Invalid input data (empty statistics or NaN bounds).")
        return None

    fig = go.Figure()
    # Plot histogram of bootstrap statistics
    fig.add_trace(go.Histogram(
        x=bootstrap_statistics, name='Bootstrap<br>Distribution',
        marker_color=COLORS.get('royal_blue', '#4169E1'), # Use a theme color
        opacity=0.75, histnorm='probability density' # Normalize to density for comparison with PDF if needed
    ))
    # Add vertical line for observed statistic
    fig.add_vline(
        x=observed_statistic, line_width=2, line_dash="dash",
        line_color=COLORS.get('green', '#00FF00'), # Use a theme color
        name=f'Observed<br>{statistic_name}<br>({observed_statistic:.4f})' # Include value in legend
    )
    # Add vertical lines for CI bounds
    fig.add_vline(
        x=lower_bound, line_width=2, line_dash="dot",
        line_color=COLORS.get('orange', '#FFA500'), # Use a theme color
        name=f'Lower CI<br>({lower_bound:.4f})' # Shortened name, include value
    )
    fig.add_vline(
        x=upper_bound, line_width=2, line_dash="dot",
        line_color=COLORS.get('orange', '#FFA500'), # Use a theme color
        name=f'Upper CI<br>({upper_bound:.4f})' # Shortened name, include value
    )
    fig.update_layout(
        title_text=f'Bootstrap Distribution for {statistic_name}',
        xaxis_title=statistic_name,
        yaxis_title='Density',
        bargap=0.1, # Adjust gap between bars if needed
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return _apply_custom_theme(fig, theme)

def plot_stacked_bar_chart(
    df: pd.DataFrame,
    category_col: str,
    stack_col: Optional[str] = None, # Used if is_data_aggregated is False
    value_col: Optional[str] = None, # Used if is_data_aggregated is False and not counting
    title: Optional[str] = None,
    theme: str = 'dark',
    color_discrete_map: Optional[Dict[str, str]] = None,
    is_data_aggregated: bool = False,
    stack_cols: Optional[List[str]] = None # Used if is_data_aggregated is True and df is pre-pivoted
) -> Optional[go.Figure]:
    """
    Generates a stacked bar chart.
    If is_data_aggregated is True, assumes df is already shaped with category_col as index/column
    and stack_cols (or a single stack_col if value_col is also given for y) as value columns.
    If False, performs aggregation based on category_col, stack_col, and value_col (for sum) or counts.
    """
    if df is None or df.empty or category_col not in df.columns:
        logger.warning("Stacked Bar Chart: Input DataFrame is invalid or category column is missing.")
        return None

    y_values_col_name = 'count' # Default for non-aggregated counts
    y_axis_title_text = "Count"
    plot_df = df # Use df directly if aggregated, otherwise it will be replaced by grouped_df
    color_arg_for_px = None # Let Plotly infer if not set
    legend_title_text = "Metric" # Default legend title

    if is_data_aggregated:
        if not stack_cols and not (stack_col and value_col): # Need columns for y-axis if aggregated
             logger.error("Stacked Bar Chart (Aggregated): Must provide stack_cols (for multiple y) or stack_col and value_col (for single y from pre-agg).")
             return None
        # If stack_cols are provided, these are the y-values for px.bar
        # If stack_col and value_col are provided, it means df has category_col, stack_col (for color), value_col (for y)
        if stack_cols: # df is like: category | stack_val1 | stack_val2 | ...
            y_values_for_plot = stack_cols
            # color argument for px.bar will be handled by plotly if y is a list.
            # We'd need to melt the df to use a single color column if that's desired.
            # For now, let px.bar handle multiple y values.
            color_arg_for_px = None # Let Plotly handle colors for multiple y columns
        elif stack_col and value_col: # df is like: category | stack_group (for color) | value (for y)
            y_values_for_plot = value_col
            color_arg_for_px = stack_col
            legend_title_text = stack_col.replace('_', ' ').title()
        else: # Should not happen due to earlier check
             logger.error("Stacked Bar Chart (Aggregated): Invalid combination of stack_col/value_col/stack_cols.")
             return None
        y_axis_title_text = "Value" # Generic for aggregated data
        plot_df = df # df is already aggregated and shaped

    else: # Original logic: perform aggregation
        if not stack_col:
            logger.error("Stacked Bar Chart (Raw): stack_col must be provided for raw data aggregation.")
            return None
        if value_col and value_col not in df.columns:
            logger.warning(f"Stacked Bar Chart (Raw): Value column '{value_col}' not found. Will use counts.")
            value_col = None # Fallback to counts
        try:
            if value_col:
                grouped_df = df.groupby([category_col, stack_col], observed=False, as_index=False)[value_col].sum()
                y_values_col_name = value_col # Use the actual value column name for y
                y_axis_title_text = f"Sum of {value_col.replace('_', ' ').title()}"
            else: # Counting occurrences
                grouped_df = df.groupby([category_col, stack_col], observed=False, as_index=False).size()
                # Ensure the count column is named consistently, e.g., 'count'
                if 'size' in grouped_df.columns and 'count' not in grouped_df.columns:
                     grouped_df = grouped_df.rename(columns={'size': 'count'})
                elif 0 in grouped_df.columns and 'count' not in grouped_df.columns: # Handle unnamed size column
                     grouped_df = grouped_df.rename(columns={0: 'count'})
                y_values_col_name = 'count' # Already set as default
            plot_df = grouped_df
            y_values_for_plot = y_values_col_name
            color_arg_for_px = stack_col
            legend_title_text = stack_col.replace('_', ' ').title()
        except Exception as e:
            logger.error(f"Stacked Bar Chart (Raw): Error during grouping/aggregation: {e}")
            return None

    if plot_df.empty:
        logger.warning("Stacked Bar Chart: Data for plotting is empty.")
        return None

    fig_title = title if title else f"{ (stack_col or 'Values').replace('_', ' ').title()} by {category_col.replace('_', ' ').title()}"

    fig = px.bar(
        plot_df, x=category_col, y=y_values_for_plot,
        color=color_arg_for_px, title=fig_title,
        barmode='stack', color_discrete_map=color_discrete_map
    )
    fig.update_layout(
        xaxis_title=category_col.replace('_', ' ').title(),
        yaxis_title=y_axis_title_text,
        legend_title_text=legend_title_text
    )
    return _apply_custom_theme(fig, theme)


def plot_grouped_bar_chart(
    df: pd.DataFrame,
    category_col: str,
    value_col: str, # If aggregated, this is the Y value. If raw, used for aggregation.
    group_col: str, # Column to group by for colors
    title: Optional[str] = None,
    aggregation_func: str = 'mean', # Used only if is_data_aggregated is False
    theme: str = 'dark',
    color_discrete_map: Optional[Dict[str, str]] = None,
    is_data_aggregated: bool = False
) -> Optional[go.Figure]:
    """
    Generates a grouped bar chart.
    If is_data_aggregated is True, df is assumed to have category_col, group_col (for color),
    and value_col (for y-values).
    If False, performs aggregation based on aggregation_func.
    """
    if df is None or df.empty or not all(c in df.columns for c in [category_col, value_col, group_col]):
        logger.warning("Grouped Bar Chart: Input DataFrame is invalid or missing required columns.")
        return None

    plot_df = df
    y_col_for_plot = value_col
    y_axis_title_text = value_col.replace('_', ' ').title()

    if is_data_aggregated:
        # Data is already aggregated. value_col is the column with y-values.
        # aggregation_func is ignored.
        y_axis_title_text = value_col.replace('_', ' ').title() # Use value_col directly for title
    else:
        # Perform internal aggregation
        try:
            if aggregation_func == 'mean':
                grouped_df = df.groupby([category_col, group_col], observed=False, as_index=False)[value_col].mean()
                y_axis_title_text = f"Average {value_col.replace('_', ' ').title()}"
            elif aggregation_func == 'sum':
                grouped_df = df.groupby([category_col, group_col], observed=False, as_index=False)[value_col].sum()
                y_axis_title_text = f"Total {value_col.replace('_', ' ').title()}"
            elif aggregation_func == 'count':
                # For count, value_col might not be the one being counted, but rather the one that determines presence.
                # The actual y-values will be in a 'count' column.
                grouped_df = df.groupby([category_col, group_col], observed=False, as_index=False).size()
                if 'size' in grouped_df.columns and 'count' not in grouped_df.columns:
                    grouped_df = grouped_df.rename(columns={'size': 'count'})
                elif 0 in grouped_df.columns and 'count' not in grouped_df.columns: # Handle unnamed size column
                    grouped_df = grouped_df.rename(columns={0: 'count'})
                y_col_for_plot = 'count'
                y_axis_title_text = "Count"
            else:
                logger.error(f"Grouped Bar Chart: Invalid aggregation function '{aggregation_func}'.")
                return None
            plot_df = grouped_df
        except Exception as e:
            logger.error(f"Grouped Bar Chart: Error during grouping/aggregation: {e}")
            return None

    if plot_df.empty:
        logger.warning("Grouped Bar Chart: Data for plotting is empty.")
        return None

    fig_title_val_part = value_col.replace('_', ' ').title() if not (not is_data_aggregated and aggregation_func == 'count') else "Count"
    fig_title = title if title else f"{fig_title_val_part} by {category_col.replace('_', ' ').title()}, Grouped by {group_col.replace('_', ' ').title()}"

    fig = px.bar(
        plot_df, x=category_col, y=y_col_for_plot,
        color=group_col, title=fig_title,
        barmode='group', color_discrete_map=color_discrete_map
    )
    fig.update_layout(
        xaxis_title=category_col.replace('_', ' ').title(),
        yaxis_title=y_axis_title_text,
        legend_title_text=group_col.replace('_', ' ').title()
    )
    return _apply_custom_theme(fig, theme)


def plot_box_plot(
    df: pd.DataFrame,
    category_col: str,
    value_col: str,
    title: Optional[str] = None,
    theme: str = 'dark',
    color_discrete_map: Optional[Dict[str, str]] = None
) -> Optional[go.Figure]:
    """
    Generates a box plot.
    """
    if df is None or df.empty or not all(c in df.columns for c in [category_col, value_col]):
        logger.warning("Box Plot: Input DataFrame is invalid or missing required columns.")
        return None
    if not pd.api.types.is_numeric_dtype(df[value_col]):
        logger.warning(f"Box Plot: Value column '{value_col}' must be numeric.")
        return None


    fig_title = title if title else f"{value_col.replace('_', ' ').title()} Distribution by {category_col.replace('_', ' ').title()}"

    fig = px.box(
        df, x=category_col, y=value_col,
        color=category_col if color_discrete_map or theme == 'dark' else None, # Apply color by category if map provided or dark theme
        title=fig_title,
        points="outliers", # Show outliers
        color_discrete_map=color_discrete_map
    )
    fig.update_layout(
        xaxis_title=category_col.replace('_', ' ').title(),
        yaxis_title=value_col.replace('_', ' ').title()
    )
    return _apply_custom_theme(fig, theme)

def plot_donut_chart(
    df: pd.DataFrame,
    category_col: str,
    value_col: str = 'count', # Default value column name if data is pre-aggregated
    title: Optional[str] = None,
    theme: str = 'dark',
    color_discrete_map: Optional[Dict[str, str]] = None,
    is_data_aggregated: bool = False
) -> Optional[go.Figure]:
    """
    Generates a donut chart.
    If is_data_aggregated is True, assumes df has category_col and value_col (with counts/values).
    If False, performs value_counts on category_col.
    """
    if df is None or df.empty or category_col not in df.columns:
        logger.warning("Donut Chart: Input DataFrame is invalid or category column is missing.")
        return None

    plot_df = df
    val_col_name_for_plot = value_col # Use provided value_col if aggregated

    if is_data_aggregated:
        if value_col not in df.columns:
            logger.error(f"Donut Chart (Aggregated): Pre-aggregated data must contain value column '{value_col}'.")
            return None
        # Ensure value_col is numeric for plotting
        try:
            plot_df[value_col] = pd.to_numeric(plot_df[value_col])
        except ValueError:
            logger.error(f"Donut Chart (Aggregated): Value column '{value_col}' could not be converted to numeric.")
            return None
    else: # Perform aggregation (counts)
        counts = df[category_col].value_counts().reset_index()
        # Pandas < 2.0 might name the columns differently after reset_index if original was Series.name
        if len(counts.columns) == 2: # Common case
            counts.columns = [category_col, 'count'] # Ensure correct column names
        else: # Fallback if column naming is unexpected
            logger.warning(f"Donut Chart: Unexpected columns after value_counts: {counts.columns}. Attempting to use first two.")
            if len(counts.columns) >= 2:
                 counts = counts.iloc[:, :2]
                 counts.columns = [category_col, 'count']
            else:
                 logger.error("Donut Chart: Could not determine category and count columns after value_counts.")
                 return None

        plot_df = counts
        val_col_name_for_plot = 'count' # Ensure value_col is 'count' for non-aggregated data

    if plot_df.empty or val_col_name_for_plot not in plot_df.columns:
        logger.warning("Donut Chart: No data to plot or value column missing after processing.")
        return None

    fig_title = title if title else f"Distribution of {category_col.replace('_', ' ').title()}"

    fig = px.pie(
        plot_df, names=category_col, values=val_col_name_for_plot,
        title=fig_title, hole=0.4, # Creates the donut effect
        color_discrete_map=color_discrete_map
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return _apply_custom_theme(fig, theme)


def plot_radar_chart(
    df_radar: pd.DataFrame,
    categories_col: str,
    value_cols: List[str],
    title: Optional[str] = None,
    fill: str = 'toself', # 'toself' fills area under trace, 'tonext' fills area between traces
    theme: str = 'dark',
    color_discrete_sequence: Optional[List[str]] = None # For custom colors per trace
) -> Optional[go.Figure]:
    """
    Generates a radar chart (spider chart).
    """
    if df_radar is None or df_radar.empty or categories_col not in df_radar.columns or \
       not value_cols or not all(col in df_radar.columns for col in value_cols):
        logger.warning("Radar Chart: Input DataFrame is invalid or missing required columns.")
        return None

    fig = go.Figure()
    category_labels = df_radar[categories_col].tolist()

    if not category_labels: # Should not happen if categories_col exists and df_radar not empty, but good check
        logger.warning("Radar Chart: Category labels are empty.")
        return None

    for i, val_col in enumerate(value_cols):
        trace_color = None
        if color_discrete_sequence and i < len(color_discrete_sequence):
            trace_color = color_discrete_sequence[i]
        
        fig.add_trace(go.Scatterpolar(
            r=df_radar[val_col].tolist(),
            theta=category_labels,
            fill=fill,
            name=val_col.replace('_', ' ').title(), # Use column name as trace name
            line_color=trace_color # Apply custom color if provided
        ))

    fig_title = title if title else "Radar Chart Comparison"
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, # range=[0, 5] # Optionally set range
        )),
        showlegend=True,
        title=fig_title
    )
    return _apply_custom_theme(fig, theme)

def plot_scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    size_col: Optional[str] = None,
    title: Optional[str] = None,
    theme: str = 'dark',
    color_discrete_map: Optional[Dict[str, str]] = None
) -> Optional[go.Figure]:
    """
    Generates a scatter plot.
    """
    if df is None or df.empty or not all(c in df.columns for c in [x_col, y_col]):
        logger.warning("Scatter Plot: Input DataFrame is invalid or missing x/y columns.")
        return None
    if color_col and color_col not in df.columns:
        logger.warning(f"Scatter Plot: Color column '{color_col}' not found. Ignoring.")
        color_col = None
    if size_col and size_col not in df.columns:
        logger.warning(f"Scatter Plot: Size column '{size_col}' not found. Ignoring.")
        size_col = None
    
    # Ensure x and y columns are numeric
    if not pd.api.types.is_numeric_dtype(df[x_col]) or not pd.api.types.is_numeric_dtype(df[y_col]):
        logger.warning(f"Scatter Plot: X column '{x_col}' and Y column '{y_col}' must be numeric.")
        return None
    # Ensure size column is numeric if provided
    if size_col and not pd.api.types.is_numeric_dtype(df[size_col]):
        logger.warning(f"Scatter Plot: Size column '{size_col}' must be numeric. Ignoring.")
        size_col = None


    fig_title = title if title else f"{y_col.replace('_', ' ').title()} vs. {x_col.replace('_', ' ').title()}"

    fig = px.scatter(
        df, x=x_col, y=y_col,
        color=color_col, size=size_col,
        title=fig_title,
        color_discrete_map=color_discrete_map
    )
    fig.update_layout(
        xaxis_title=x_col.replace('_', ' ').title(),
        yaxis_title=y_col.replace('_', ' ').title(),
        legend_title_text=color_col.replace('_', ' ').title() if color_col else None
    )
    return _apply_custom_theme(fig, theme)

def plot_efficient_frontier(
    frontier_vols: List[float],
    frontier_returns: List[float],
    max_sharpe_vol: Optional[float] = None,
    max_sharpe_ret: Optional[float] = None,
    min_vol_vol: Optional[float] = None,
    min_vol_ret: Optional[float] = None,
    title: str = "Efficient Frontier",
    theme: str = 'dark'
) -> Optional[go.Figure]:
    """
    Plots the efficient frontier.
    """
    if not frontier_vols or not frontier_returns or len(frontier_vols) != len(frontier_returns):
        logger.warning("Efficient Frontier: Invalid input data for frontier points (empty lists or mismatched lengths).")
        return None

    fig = go.Figure()
    # Efficient Frontier line
    fig.add_trace(go.Scatter(
        x=frontier_vols, y=frontier_returns,
        mode='lines', name='Efficient Frontier',
        line=dict(color=COLORS.get('royal_blue', '#4169E1'), width=2),
        hovertemplate='Volatility: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
    ))

    # Max Sharpe Ratio Portfolio point
    if max_sharpe_vol is not None and max_sharpe_ret is not None:
        fig.add_trace(go.Scatter(
            x=[max_sharpe_vol], y=[max_sharpe_ret],
            mode='markers', name='Max Sharpe Ratio Portfolio',
            marker=dict(color=COLORS.get('green', '#00FF00'), size=10, symbol='star'),
            hovertemplate='Max Sharpe<br>Volatility: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
        ))

    # Minimum Volatility Portfolio point
    if min_vol_vol is not None and min_vol_ret is not None:
        # Check if it's distinct from Max Sharpe to avoid overlaying identical points
        is_distinct_from_max_sharpe = True
        if max_sharpe_vol is not None and max_sharpe_ret is not None:
            if abs(min_vol_vol - max_sharpe_vol) < 1e-6 and abs(min_vol_ret - max_sharpe_ret) < 1e-6: # Tolerance for float comparison
                is_distinct_from_max_sharpe = False
        
        if is_distinct_from_max_sharpe:
            fig.add_trace(go.Scatter(
                x=[min_vol_vol], y=[min_vol_ret],
                mode='markers', name='Minimum Volatility Portfolio',
                marker=dict(color=COLORS.get('orange', '#FFA500'), size=10, symbol='diamond'),
                hovertemplate='Min Volatility<br>Volatility: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
            ))

    fig.update_layout(
        title_text=title,
        xaxis_title="Annualized Volatility (Standard Deviation)",
        yaxis_title="Annualized Expected Return",
        xaxis_tickformat=".2%", yaxis_tickformat=".2%", # Format axes as percentages
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return _apply_custom_theme(fig, theme)


# --- NEW PLOTTING FUNCTIONS ---

def plot_distribution_fit(
    pnl_series: pd.Series,
    dist_name: str,
    fit_params: tuple,
    gof_stats: Dict[str, float], # e.g., {"ks_statistic": D, "ks_p_value": p_value}
    theme: str = 'dark',
    nbins: int = 50
) -> Optional[go.Figure]:
    """
    Plots the histogram of PnL data overlaid with the PDF of a fitted distribution,
    and includes a Q-Q plot for visual goodness-of-fit assessment.

    Args:
        pnl_series (pd.Series): The PnL data.
        dist_name (str): The name of the distribution (e.g., 'norm', 't').
        fit_params (tuple): Parameters of the fitted distribution from scipy.stats.fit().
        gof_stats (Dict[str, float]): Goodness-of-fit statistics, like KS p-value.
        theme (str): Plotting theme ('light' or 'dark').
        nbins (int): Number of bins for the histogram.

    Returns:
        Optional[go.Figure]: A Plotly figure object or None if an error occurs.
    """
    if pnl_series.empty:
        logger.warning(f"Distribution Fit Plot for '{dist_name}': PnL series is empty.")
        return None
    if not hasattr(scipy_stats, dist_name):
        logger.error(f"Distribution Fit Plot: Distribution '{dist_name}' not found in scipy.stats.")
        return None

    dist_obj = getattr(scipy_stats, dist_name)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"Histogram vs. Fitted {dist_name.capitalize()} PDF", f"Q-Q Plot vs. {dist_name.capitalize()}"),
        column_widths=[0.6, 0.4]
    )

    # 1. Histogram and PDF overlay
    fig.add_trace(
        go.Histogram(
            x=pnl_series, nbinsx=nbins, name='PnL Data', histnorm='probability density',
            marker_color=PLOT_LINE_COLOR, opacity=0.7
        ), row=1, col=1
    )
    x_pdf = np.linspace(pnl_series.min(), pnl_series.max(), 500)
    try:
        y_pdf = dist_obj.pdf(x_pdf, *fit_params)
    except Exception as e:
        logger.error(f"Error calculating PDF for {dist_name} with params {fit_params}: {e}")
        y_pdf = np.zeros_like(x_pdf) 
    fig.add_trace(
        go.Scatter(
            x=x_pdf, y=y_pdf, mode='lines', name=f'Fitted {dist_name.capitalize()} PDF',
            line=dict(color=COLORS.get('green', 'green'), width=2)
        ), row=1, col=1
    )

    # 2. Q-Q Plot
    try:
        # For probplot, sparams are shape parameters. loc and scale are handled by fit=False if already in fit_params.
        # The parameters from dist.fit() are usually (shape_params..., loc, scale).
        # So, if dist_obj.shapes is not None, fit_params[:-2] are the shape parameters.
        shape_params_for_probplot = ()
        if dist_obj.shapes: # If the distribution has shape parameters
            num_shape_params = len(dist_obj.shapes.split(','))
            if len(fit_params) >= num_shape_params + 2: # loc and scale are last two
                 shape_params_for_probplot = fit_params[:num_shape_params]
            else: # Fallback if param count doesn't match expectation (e.g. fixed shape params)
                 logger.warning(f"Q-Q Plot for {dist_name}: Parameter count mismatch for shape params. Expected {num_shape_params} shape params, got {len(fit_params)-2}. Proceeding without explicit sparams for probplot.")
        
        # Using stats.probplot which expects data and a distribution object.
        # If `fit=True` (default), it fits loc and scale. If `fit=False`, it uses provided loc/scale from `sparams` or defaults.
        # Since we already have all params from `dist.fit()`, we want to use them.
        # `scipy.stats.probplot` with `dist=dist_obj` and `sparams=fit_params` (if dist takes them)
        # or `dist=dist_obj(*fit_params)` if we want to create a frozen distribution.
        # For simplicity and to use the `fit_params` directly as understood by the `dist_obj` methods:
        
        # Create a frozen distribution with the fitted parameters
        frozen_dist = dist_obj(*fit_params)
        qq_results = scipy_stats.probplot(pnl_series, dist=frozen_dist, plot=None) # plot=None returns arrays

        osm, osr = qq_results[0] 
        slope, intercept, r_value = qq_results[1] 

        fig.add_trace(
            go.Scatter(x=osm, y=osr, mode='markers', name='Data Quantiles', marker=dict(color=PLOT_LINE_COLOR)),
            row=1, col=2
        )
        # For the theoretical line, it's y = x for a perfect fit if quantiles are directly comparable
        # or use the regression line from probplot.
        fig.add_trace(
            go.Scatter(x=osm, y=slope * osm + intercept, mode='lines', name='Fit Line (OLS)', line=dict(color=COLORS.get('red', 'red'), dash='dash')),
            row=1, col=2
        )
        fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
        fig.update_yaxes(title_text="Sample Quantiles", row=1, col=2)
        fig.layout.annotations += (dict(
            text=f"R-squared: {r_value**2:.4f}", 
            align='left', showarrow=False, xref='paper', yref='paper', 
            x=0.98, y=0.02, # Position in the Q-Q subplot
            xanchor='right', yanchor='bottom',
            font=dict(size=10), row=1, col=2
        ),)


    except Exception as e:
        logger.error(f"Error generating Q-Q plot for {dist_name}: {e}", exc_info=True)
        fig.add_annotation(text="Q-Q plot generation failed.", xref="paper", yref="paper",
                           x=0.8, y=0.5, showarrow=False, row=1, col=2)


    ks_p_value = gof_stats.get('ks_p_value', np.nan)
    title_text = f"PnL vs. Fitted {dist_name.capitalize()} (KS p-value: {ks_p_value:.4f})"
    
    fig.update_layout(
        title_text=title_text,
        xaxis_title="PnL Value", yaxis_title="Density", # For subplot 1,1
        bargap=0.1, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1)
    )
    return _apply_custom_theme(fig, theme)

def plot_change_points(
    time_series_data: pd.Series,
    change_points_locations: List[Any], # List of actual index values (timestamps or numbers)
    series_name: str = "Time Series",
    title: str = "Time Series with Detected Change Points",
    theme: str = 'dark'
) -> Optional[go.Figure]:
    """
    Plots a time series with vertical lines indicating detected change points.

    Args:
        time_series_data (pd.Series): The original time series data.
        change_points_locations (List[Any]): A list of actual index values
            (timestamps or numeric indices from the original series)
            where change points are detected.
        series_name (str): Name of the time series for the legend.
        title (str): Title of the plot.
        theme (str): Plotting theme ('light' or 'dark').

    Returns:
        Optional[go.Figure]: A Plotly figure object or None if an error occurs.
    """
    if time_series_data.empty:
        logger.warning("Change Point Plot: Time series data is empty.")
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time_series_data.index, y=time_series_data.values, mode='lines',
            name=series_name, line=dict(color=PLOT_LINE_COLOR)
        )
    )
    for i, cp_loc in enumerate(change_points_locations):
        try:
            # cp_loc should already be a valid index value from the original series
            if cp_loc not in time_series_data.index:
                 # This might happen if indices are slightly off due to float precision or if they are from a resampled series.
                 # Attempt to find the nearest valid index if it's a DatetimeIndex.
                if isinstance(time_series_data.index, pd.DatetimeIndex):
                    try:
                        # Ensure cp_loc is a timestamp if it's not already
                        cp_timestamp = pd.to_datetime(cp_loc)
                        # Find the closest index in the series
                        nearest_idx_pos = time_series_data.index.get_indexer([cp_timestamp], method='nearest')[0]
                        actual_cp_loc_for_plot = time_series_data.index[nearest_idx_pos]
                        logger.warning(f"Change Point Plot: Location {cp_loc} not found exactly in index. Using nearest: {actual_cp_loc_for_plot}.")
                    except Exception as date_err:
                        logger.warning(f"Change Point Plot: Could not interpret or find nearest for location {cp_loc}: {date_err}. Skipping.")
                        continue
                else:
                    logger.warning(f"Change Point Plot: Location {cp_loc} not in series index. Skipping.")
                    continue
            else:
                actual_cp_loc_for_plot = cp_loc

            fig.add_vline(
                x=actual_cp_loc_for_plot, line_width=2, line_dash="dash",
                line_color=COLORS.get('red', 'red'),
                name=f'Change Point {i+1}' if i == 0 else None, 
                showlegend= i == 0 
            )
        except Exception as e:
            logger.error(f"Error adding vline for change point {cp_loc}: {e}", exc_info=True)

    fig.update_layout(
        title_text=title,
        xaxis_title="Time / Index",
        yaxis_title=series_name if series_name != "Time Series" else "Value",
        hovermode="x unified", showlegend=True
    )
    return _apply_custom_theme(fig, theme)

