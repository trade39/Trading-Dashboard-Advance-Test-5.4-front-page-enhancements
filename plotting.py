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

# Assuming config.py and utils.common_utils are in a path accessible by Python
# For example, they could be in the same directory or in a package.
# If running this script directly and they are in parent/sibling dirs,
# sys.path adjustments might be needed, but for a package structure, it's usually fine.
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
    # This is more for making the script runnable in isolation if needed,
    # in a real project structure, these imports should work.
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

    This internal helper function standardizes the appearance of plots
    by applying predefined colors and layout settings based on the chosen theme.

    Args:
        fig (go.Figure): The Plotly figure object to be themed.
        theme (str, optional): The theme to apply. Can be 'dark' or 'light'.
            Defaults to 'dark'.

    Returns:
        go.Figure: The themed Plotly figure object.

    Example:
        >>> fig = go.Figure(data=[go.Scatter(x=[1, 2], y=[1, 2])])
        >>> themed_fig = _apply_custom_theme(fig, theme='light')
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

    This function creates a heatmap visualization where cell colors represent
    values from the input pivot table. It supports custom text formatting
    for cell values, including currency and percentages.

    Args:
        df_pivot (pd.DataFrame): Pivot DataFrame with values for the heatmap.
            Index will be used for y-axis, columns for x-axis.
        title (str, optional): Title of the heatmap. Defaults to "Heatmap".
        x_axis_title (Optional[str], optional): Title for the x-axis.
            If None, uses the name of the pivot table's columns. Defaults to None.
        y_axis_title (Optional[str], optional): Title for the y-axis.
            If None, uses the name of the pivot table's index. Defaults to None.
        color_scale (str, optional): Plotly colorscale for the heatmap.
            Defaults to "RdBu".
        z_min (Optional[float], optional): Minimum value for the color scale.
            Defaults to None (auto-scaled).
        z_max (Optional[float], optional): Maximum value for the color scale.
            Defaults to None (auto-scaled).
        show_text (bool, optional): Whether to display text values on heatmap cells.
            Defaults to True.
        text_format (str, optional): Format string for the text values
            (e.g., ".2f", "$,.0f", "$.2%"). Supports currency ($) prefix
            and percentage (%) suffix. Defaults to ".2f".
        theme (str, optional): The theme to apply ('dark' or 'light').
            Defaults to 'dark'.

    Returns:
        Optional[go.Figure]: A Plotly Figure object for the heatmap,
            or None if the input DataFrame is empty or invalid.

    Example:
        >>> data = {'Month': ['Jan', 'Jan', 'Feb', 'Feb'],
        ...         'Category': ['A', 'B', 'A', 'B'],
        ...         'Value': [10, 20, 15, 25]}
        >>> df = pd.DataFrame(data)
        >>> df_pivot_example = df.pivot(index='Month', columns='Category', values='Value')
        >>> fig = plot_heatmap(df_pivot_example, title="Monthly Sales by Category",
        ...                    text_format="$,.0f", theme='light')
        >>> # To show the figure (if in an interactive environment):
        >>> # if fig: fig.show()
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
    """
    Adds a shaded vertical rectangle (vrect) to a plot to highlight the maximum drawdown period.

    This is an internal helper for `plot_equity_curve_and_drawdown`.

    Args:
        fig (go.Figure): The Plotly figure object to modify.
        df_dates (pd.Series): Series of dates from the primary data, used as a
            fallback for the end date of the shading if recovery is not complete.
        max_dd_peak_date (Optional[Any]): The peak date of the maximum drawdown.
            Can be a datetime object or a string convertible to datetime.
        max_dd_trough_date (Optional[Any]): The trough date of the maximum drawdown.
            Can be a datetime object or a string convertible to datetime.
        max_dd_recovery_date (Optional[Any]): The recovery date of the maximum drawdown.
            Can be a datetime object or a string convertible to datetime.
        row (int): The subplot row to add the vrect to.
        col (int): The subplot column to add the vrect to.
    """
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
                pass # Default text is fine
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
    """
    Adds shaded vertical rectangles (vrects) to a plot for general drawdown periods,
    optionally excluding the maximum drawdown period to avoid double shading.

    This is an internal helper for `plot_equity_curve_and_drawdown`.

    Args:
        fig (go.Figure): The Plotly figure object to modify.
        df_dates (pd.Series): Series of dates from the primary data, used as a
            fallback for end dates of ongoing drawdowns.
        drawdown_periods_df (Optional[pd.DataFrame]): DataFrame containing drawdown
            period details. Expected columns: 'Peak Date', 'End Date'.
        max_dd_peak_date_for_exclusion (Optional[Any]): Peak date of the maximum
            drawdown. If a drawdown period in `drawdown_periods_df` matches this
            peak date, it will not be shaded by this function (as it's handled
            by `_add_max_dd_shading_to_plot`).
        row (int): The subplot row to add the vrects to.
        col (int): The subplot column to add the vrects to.
    """
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
                    except Exception: # Handle potential conversion error for max_dd_peak_date
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
    Highlights maximum drawdown and other drawdown periods using shaded areas.

    Args:
        df (pd.DataFrame): DataFrame containing the time series data.
            Must include date and cumulative PnL columns.
        date_col (str, optional): Name of the date column in `df`.
            Defaults to `EXPECTED_COLUMNS['date']`.
        cumulative_pnl_col (str, optional): Name of the cumulative PnL column in `df`.
            Defaults to 'cumulative_pnl'.
        drawdown_pct_col (Optional[str], optional): Name of the drawdown percentage
            column in `df`. If provided, a second subplot shows drawdown.
            Defaults to 'drawdown_pct'.
        drawdown_periods_df (Optional[pd.DataFrame], optional): DataFrame with details
            of individual drawdown periods. Expected columns: 'Peak Date', 'End Date'.
            Used for shading general drawdown periods. Defaults to None.
        theme (str, optional): The theme to apply ('dark' or 'light').
            Defaults to 'dark'.
        max_dd_peak_date (Optional[Any], optional): Peak date of the maximum drawdown.
            Used for highlighting the max drawdown period. Defaults to None.
        max_dd_trough_date (Optional[Any], optional): Trough date of the maximum drawdown.
            Used for highlighting the max drawdown period. Defaults to None.
        max_dd_recovery_date (Optional[Any], optional): Recovery date of the maximum drawdown.
            Used for highlighting the max drawdown period. Defaults to None.

    Returns:
        Optional[go.Figure]: A Plotly Figure object containing the equity curve
            and drawdown plot, or None if input data is insufficient or invalid.

    Example:
        >>> dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'])
        >>> pnl = pd.Series([100, 120, 90, 110, 130])
        >>> dd_pct = pd.Series([0, 0, -0.25, -0.083, 0]) # Example drawdown percentages
        >>> data_df = pd.DataFrame({
        ...     'Date': dates,
        ...     'cumulative_pnl': pnl,
        ...     'drawdown_pct': dd_pct * 100 # Assuming percentages are 0-100
        ... })
        >>> # Example drawdown periods (simplified)
        >>> dd_periods = pd.DataFrame({
        ...     'Peak Date': [pd.to_datetime('2023-01-02')],
        ...     'End Date': [pd.to_datetime('2023-01-04')] # Assuming it recovered by 4th
        ... })
        >>> fig = plot_equity_curve_and_drawdown(
        ...     data_df, date_col='Date', cumulative_pnl_col='cumulative_pnl',
        ...     drawdown_pct_col='drawdown_pct', drawdown_periods_df=dd_periods,
        ...     max_dd_peak_date='2023-01-02', max_dd_trough_date='2023-01-03',
        ...     max_dd_recovery_date='2023-01-04'
        ... )
        >>> # if fig: fig.show()
    """
    if df is None or df.empty or date_col not in df.columns or cumulative_pnl_col not in df.columns:
        logger.warning("Equity curve plot: Input DataFrame is invalid or missing required columns.")
        return None

    df_copy = df.copy() # Work on a copy to avoid modifying original DataFrame
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
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1, tickformat=".2f") # Assuming drawdown_pct is already in percent, e.g., -10 for -10%
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
    The area between the high water mark and the equity curve (when equity is lower)
    is shaded to represent drawdown periods.

    Args:
        equity_series (pd.Series): Pandas Series with a DatetimeIndex and equity values.
        theme (str, optional): The theme to apply ('dark' or 'light').
            Defaults to 'dark'.
        title (str, optional): Title of the plot.
            Defaults to "Underwater Plot (Equity vs. High Water Mark)".

    Returns:
        Optional[go.Figure]: A Plotly Figure object for the underwater plot,
            or None if the input equity series is invalid or insufficient.

    Example:
        >>> dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'])
        >>> equity_values = pd.Series([100, 120, 90, 110, 130], index=dates)
        >>> fig = plot_underwater_analysis(equity_values, theme='light')
        >>> # if fig: fig.show()
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
    Plots strategy equity (or normalized values) against benchmark cumulative returns.
    Both series should ideally be indexed by datetime and represent comparable values
    (e.g., normalized to start at 100 or representing cumulative percentage returns).

    Args:
        strategy_equity (pd.Series): Pandas Series for strategy equity or
            normalized value, with a DatetimeIndex.
        benchmark_cumulative_returns (pd.Series): Pandas Series for benchmark
            cumulative returns or normalized value, with a DatetimeIndex.
        strategy_name (str, optional): Name for the strategy trace.
            Defaults to "Strategy".
        benchmark_name (str, optional): Name for the benchmark trace.
            Defaults to "Benchmark".
        theme (str, optional): The theme to apply ('dark' or 'light').
            Defaults to 'dark'.

    Returns:
        Optional[go.Figure]: A Plotly Figure object comparing the two series,
            or None if both input series are empty.

    Example:
        >>> dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
        >>> strategy = pd.Series([100, 102, 101], index=dates)
        >>> benchmark = pd.Series([100, 101, 100.5], index=dates)
        >>> fig = plot_equity_vs_benchmark(strategy, benchmark,
        ...                                strategy_name="My Algo", benchmark_name="S&P 500 Index")
        >>> # if fig: fig.show()
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
    Generates a histogram of Profit and Loss (PnL) per trade, with a box plot marginal.

    Args:
        df (pd.DataFrame): DataFrame containing PnL data for individual trades.
        pnl_col (str, optional): Name of the PnL column in `df`.
            Defaults to `EXPECTED_COLUMNS['pnl']`.
        title (str, optional): Title of the plot.
            Defaults to "PnL Distribution (per Trade)".
        theme (str, optional): The theme to apply ('dark' or 'light').
            Defaults to 'dark'.
        nbins (int, optional): Number of bins for the histogram.
            Defaults to 50.

    Returns:
        Optional[go.Figure]: A Plotly Figure object showing the PnL distribution,
            or None if input data is invalid or the PnL column is empty.

    Example:
        >>> trades_data = pd.DataFrame({'PnL': np.random.normal(0, 50, 100)})
        >>> fig = plot_pnl_distribution(trades_data, pnl_col='PnL', nbins=30)
        >>> # if fig: fig.show()
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
    Plots the observed, trend, seasonal, and residual components of a time series
    decomposition result (e.g., from `statsmodels.tsa.seasonal.seasonal_decompose`).

    Args:
        decomposition_result (Any): The result object from a time series
            decomposition function. Expected to have attributes `observed`,
            `trend`, `seasonal`, and `resid`, each being a Pandas Series.
        title (str, optional): Title of the plot.
            Defaults to "Time Series Decomposition".
        theme (str, optional): The theme to apply ('dark' or 'light').
            Defaults to 'dark'.

    Returns:
        Optional[go.Figure]: A Plotly Figure object with subplots for each component,
            or None if the decomposition result is invalid or components are empty.

    Example:
        >>> # This example requires statsmodels
        >>> # from statsmodels.tsa.seasonal import seasonal_decompose
        >>> # s = pd.Series(np.random.rand(100), index=pd.date_range(start='2023-01-01', periods=100, freq='D'))
        >>> # result = seasonal_decompose(s, model='additive', period=7)
        >>> # fig = plot_time_series_decomposition(result)
        >>> # if fig: fig.show()
        >>> # Placeholder example if statsmodels is not available:
        >>> class MockDecomposition:
        ...     def __init__(self):
        ...         idx = pd.date_range(start='2023-01-01', periods=30)
        ...         self.observed = pd.Series(np.sin(np.linspace(0, 3 * np.pi, 30)) + np.random.rand(30)*0.5 + 5, index=idx)
        ...         self.trend = pd.Series(np.linspace(5, 5.5, 30), index=idx)
        ...         self.seasonal = pd.Series(np.sin(np.linspace(0, 3 * np.pi, 30) * (30/7)), index=idx) # approx weekly
        ...         self.resid = self.observed - self.trend - self.seasonal
        >>> result_mock = MockDecomposition()
        >>> fig = plot_time_series_decomposition(result_mock)
        >>> # if fig: fig.show()

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
    Plots a single series of values over time (e.g., rolling Sharpe ratio, account balance).

    Args:
        series (pd.Series): Pandas Series with values to plot. The index of the
            series will be used for the x-axis (typically datetime or time period).
        series_name (str): Name of the series, used for the legend and as the
            default y-axis title if `y_axis_title` is not provided.
        title (Optional[str], optional): Title for the plot. If None, `series_name`
            is used as the title. Defaults to None.
        x_axis_title (str, optional): Title for the x-axis.
            Defaults to "Date / Time".
        y_axis_title (Optional[str], optional): Title for the y-axis. If None,
            `series_name` is used. Defaults to None.
        theme (str, optional): The theme to apply ('dark' or 'light').
            Defaults to 'dark'.
        line_color (str, optional): Color of the plot line.
            Defaults to `PLOT_LINE_COLOR`.

    Returns:
        Optional[go.Figure]: A Plotly Figure object for the time series plot,
            or None if the input series is empty.

    Example:
        >>> dates = pd.to_datetime(['2023-01-01', '2023-01-08', '2023-01-15'])
        >>> values = pd.Series([1.0, 1.2, 1.1], index=dates)
        >>> fig = plot_value_over_time(values, series_name="Rolling Sharpe Ratio",
        ...                            title="Weekly Rolling Sharpe Ratio")
        >>> # if fig: fig.show()
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
    aggregation_func: str = 'sum'
) -> Optional[go.Figure]:
    """
    Generates a bar chart of Profit and Loss (PnL) aggregated by a specified category
    (e.g., PnL by asset, PnL by day of the week).

    Args:
        df (pd.DataFrame): DataFrame containing the data, including category and PnL columns.
        category_col (str): Name of the column in `df` to group by (e.g., 'Asset', 'DayOfWeek').
        pnl_col (str, optional): Name of the PnL column in `df`.
            Defaults to `EXPECTED_COLUMNS['pnl']`.
        title_prefix (str, optional): Prefix for the plot title. The final title will be
            like "{title_prefix} {Category Column Name}". Defaults to "Total PnL by".
        theme (str, optional): The theme to apply ('dark' or 'light').
            Defaults to 'dark'.
        aggregation_func (str, optional): Aggregation function to apply to PnL
            values within each category (e.g., 'sum', 'mean', 'median').
            Defaults to 'sum'.

    Returns:
        Optional[go.Figure]: A Plotly Figure object for the bar chart,
            or None if input data is invalid or required columns are missing.

    Example:
        >>> data = {'Asset': ['EURUSD', 'GBPUSD', 'EURUSD', 'GBPUSD'],
        ...         'PnL': [10, -5, 20, 15],
        ...         'Day': ['Mon', 'Mon', 'Tue', 'Tue']}
        >>> trades_df = pd.DataFrame(data)
        >>> fig_asset = plot_pnl_by_category(trades_df, category_col='Asset', pnl_col='PnL',
        ...                                  title_prefix="Aggregate PnL by")
        >>> fig_day_avg = plot_pnl_by_category(trades_df, category_col='Day', pnl_col='PnL',
        ...                                    aggregation_func='mean', title_prefix="Average PnL by")
        >>> # if fig_asset: fig_asset.show()
        >>> # if fig_day_avg: fig_day_avg.show()
    """
    if df is None or df.empty or category_col not in df.columns or pnl_col not in df.columns:
        logger.warning("PnL by Category: Input DataFrame is invalid or missing required columns.")
        return None

    try:
        grouped_pnl = df.groupby(category_col)[pnl_col].agg(aggregation_func).reset_index().sort_values(by=pnl_col, ascending=False)
    except Exception as e:
        logger.error(f"PnL by Category: Error during aggregation '{aggregation_func}' on column '{pnl_col}' grouped by '{category_col}': {e}")
        return None


    yaxis_title_agg_text = aggregation_func.title() if aggregation_func != 'sum' else "Total"
    plot_title = f"{title_prefix.replace('Total', yaxis_title_agg_text)} {category_col.replace('_', ' ').title()}"
    y_axis_label = f"{yaxis_title_agg_text} PnL"

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
    win_col: str = 'win',
    title_prefix: str = "Win Rate by",
    theme: str = 'dark'
) -> Optional[go.Figure]:
    """
    Generates a bar chart of win rates (in percentage) by a specified category.

    Args:
        df (pd.DataFrame): DataFrame containing trade data. Must include the
            category column and a 'win' column.
        category_col (str): Name of the column in `df` to group by for win rate
            calculation (e.g., 'Asset', 'HourOfDay').
        win_col (str, optional): Name of the boolean or numeric column in `df`
            indicating a win (typically 1 for win, 0 for loss/breakeven).
            Defaults to 'win'.
        title_prefix (str, optional): Prefix for the plot title. The final title will be
            like "{title_prefix} {Category Column Name}". Defaults to "Win Rate by".
        theme (str, optional): The theme to apply ('dark' or 'light').
            Defaults to 'dark'.

    Returns:
        Optional[go.Figure]: A Plotly Figure object for the win rate bar chart,
            or None if input data is invalid or required columns are missing/invalid.

    Example:
        >>> data = {'Strategy': ['A', 'B', 'A', 'B', 'A'],
        ...         'win': [1, 0, 1, 1, 0]} # 1 for win, 0 for loss
        >>> trades_df = pd.DataFrame(data)
        >>> fig = plot_win_rate_analysis(trades_df, category_col='Strategy', win_col='win')
        >>> # if fig: fig.show()
    """
    if df is None or df.empty or category_col not in df.columns or win_col not in df.columns:
        logger.warning("Win Rate Analysis: Input DataFrame is invalid or missing required columns.")
        return None
    if not pd.api.types.is_bool_dtype(df[win_col]) and not pd.api.types.is_numeric_dtype(df[win_col]):
        logger.warning(f"Win Rate Analysis: Win column '{win_col}' must be boolean or numeric (0 or 1).")
        return None

    try:
        # Ensure win_col is numeric for sum()
        df_calc = df.copy()
        df_calc[win_col] = df_calc[win_col].astype(int)

        category_counts = df_calc.groupby(category_col).size().rename('total_trades_in_cat')
        category_wins = df_calc.groupby(category_col)[win_col].sum().rename('wins_in_cat')
    except Exception as e:
        logger.error(f"Win Rate Analysis: Error during grouping or conversion of win column: {e}")
        return None


    win_rate_df = pd.concat([category_counts, category_wins], axis=1).fillna(0)

    # Avoid division by zero if total_trades_in_cat is 0
    win_rate_df['win_rate_pct'] = 0.0 # Initialize
    non_zero_trades_mask = win_rate_df['total_trades_in_cat'] > 0
    win_rate_df.loc[non_zero_trades_mask, 'win_rate_pct'] = \
        (win_rate_df.loc[non_zero_trades_mask, 'wins_in_cat'] / win_rate_df.loc[non_zero_trades_mask, 'total_trades_in_cat'] * 100)

    win_rate_df = win_rate_df.reset_index().sort_values(by='win_rate_pct', ascending=False)

    fig = px.bar(
        win_rate_df, x=category_col, y='win_rate_pct',
        title=f"{title_prefix} {category_col.replace('_', ' ').title()}",
        color='win_rate_pct', color_continuous_scale=px.colors.sequential.Greens
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

    The x-axis can be datetime objects (if `df` and `date_col` are provided and
    align with `metric_series`) or the index of `metric_series` (e.g., trade number,
    period number).

    Args:
        df (Optional[pd.DataFrame]): The main DataFrame. If provided and `date_col`
            is valid and its length matches `metric_series`, `df[date_col]` will be
            used as the x-axis. Can be None if `metric_series` has a meaningful index.
        date_col (Optional[str]): The name of the date column in `df`.
            Required if `df` is used for x-axis dates.
        metric_series (pd.Series): Pandas Series containing the rolling metric values.
            Its index will be used for the x-axis if `df`/`date_col` are not suitable.
        metric_name (str): Name of the metric (e.g., "Rolling Sharpe", "Rolling Sortino").
            Used for legend and default y-axis title.
        title (Optional[str], optional): Title for the plot. If None, defaults to
            "Rolling {metric_name}". Defaults to None.
        theme (str, optional): The theme to apply ('dark' or 'light').
            Defaults to 'dark'.

    Returns:
        Optional[go.Figure]: A Plotly Figure object for the rolling performance plot,
            or None if the metric series is empty.

    Example:
        >>> # Example 1: Metric series with its own period index
        >>> rolling_sharpe = pd.Series([0.5, 0.6, 0.55, 0.7], index=[10, 20, 30, 40]) # Index is window size
        >>> fig1 = plot_rolling_performance(df=None, date_col=None,
        ...                                metric_series=rolling_sharpe, metric_name="Sharpe Ratio by Window")
        >>> # if fig1: fig1.show()

        >>> # Example 2: Metric series aligned with a DataFrame's date column
        >>> dates = pd.to_datetime(['2023-01-31', '2023-02-28', '2023-03-31'])
        >>> main_df = pd.DataFrame({'Date': dates, 'OtherData': [1,2,3]})
        >>> monthly_returns = pd.Series([0.02, 0.01, 0.03], index=dates) # Index matches main_df['Date']
        >>> fig2 = plot_rolling_performance(df=main_df, date_col='Date',
        ...                                 metric_series=monthly_returns, metric_name="Monthly Return")
        >>> # if fig2: fig2.show()
    """
    if metric_series.empty:
        logger.warning(f"Rolling Performance ('{metric_name}'): Metric series is empty.")
        return None

    plot_x_data = metric_series.index
    x_axis_title_text = metric_series.index.name if metric_series.index.name else "Period / Index"


    if df is not None and not df.empty and date_col and date_col in df.columns:
        if len(df[date_col]) == len(metric_series):
            try:
                # Attempt to use the date column from df if lengths match
                # This assumes metric_series is implicitly aligned with df[date_col]
                plot_x_data_candidate = pd.to_datetime(df[date_col])
                # Further check if metric_series index can be aligned or if it's already datetime
                if isinstance(metric_series.index, pd.DatetimeIndex) and metric_series.index.equals(plot_x_data_candidate):
                    plot_x_data = plot_x_data_candidate
                    x_axis_title_text = "Date"
                elif not isinstance(metric_series.index, pd.DatetimeIndex): # If metric_series index is not datetime, use df's date
                    plot_x_data = plot_x_data_candidate
                    x_axis_title_text = "Date"
                # If both are datetime but different, it's ambiguous, stick to metric_series.index
                # Or, if metric_series.index is preferred and already datetime
                elif isinstance(metric_series.index, pd.DatetimeIndex):
                     plot_x_data = metric_series.index # Keep metric_series's own datetime index
                     x_axis_title_text = "Date"


            except Exception:
                logger.warning(f"Rolling Performance ('{metric_name}'): Could not convert '{date_col}' to datetime or align. Using metric series index.")
        else:
             logger.info(f"Rolling Performance ('{metric_name}'): Length mismatch between df['{date_col}'] and metric_series. Using metric_series index.")
    elif isinstance(metric_series.index, pd.DatetimeIndex): # If df not usable, but series index is datetime
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
    Generates a heatmap of the correlation matrix for numeric columns in a DataFrame.
    Displays correlation coefficients as text on the heatmap cells.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        numeric_cols (Optional[List[str]], optional): A list of numeric column names
            from `df` to include in the correlation matrix. If None, the function
            will attempt to select all columns with numeric data types from `df`.
            Defaults to None.
        title (str, optional): Title of the plot.
            Defaults to "Correlation Matrix of Numeric Features".
        theme (str, optional): The theme to apply ('dark' or 'light').
            Defaults to 'dark'.

    Returns:
        Optional[go.Figure]: A Plotly Figure object for the correlation heatmap,
            or None if the input DataFrame is empty, no numeric columns are found,
            or fewer than two numeric columns are available for correlation.

    Example:
        >>> data = {'FeatureA': np.random.rand(20),
        ...         'FeatureB': np.random.rand(20) * 2,
        ...         'FeatureC': np.random.rand(20) - 0.5,
        ...         'Category': ['X']*10 + ['Y']*10}
        >>> data_df = pd.DataFrame(data)
        >>> # Plot correlation for all numeric features
        >>> fig1 = plot_correlation_matrix(data_df, title="Overall Correlation")
        >>> # if fig1: fig1.show()
        >>> # Plot correlation for specific numeric features
        >>> fig2 = plot_correlation_matrix(data_df, numeric_cols=['FeatureA', 'FeatureC'])
        >>> # if fig2: fig2.show()
    """
    if df is None or df.empty:
        logger.warning("Correlation Matrix: Input DataFrame is empty.")
        return None

    if numeric_cols:
        df_numeric = df[numeric_cols].copy()
        # Verify that selected columns are indeed numeric
        non_numeric_selected = [col for col in numeric_cols if not pd.api.types.is_numeric_dtype(df_numeric[col])]
        if non_numeric_selected:
            logger.warning(f"Correlation Matrix: Specified columns {non_numeric_selected} are not numeric and will be excluded or cause errors.")
            # Option: filter them out, or let .corr() handle it (it usually ignores non-numeric)
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
        text=corr_matrix.round(2).astype(str),
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
    Plots the bootstrap distribution of a statistic, the observed statistic value,
    and its confidence interval (CI) bounds.

    Args:
        bootstrap_statistics (List[float]): A list of statistic values, where each
            value is calculated from a bootstrap resample of the original data.
        observed_statistic (float): The value of the statistic calculated from the
            original, full sample of data.
        lower_bound (float): The lower bound of the confidence interval for the statistic.
        upper_bound (float): The upper bound of the confidence interval for the statistic.
        statistic_name (str): Name of the statistic being plotted (e.g., "Mean PnL",
            "Sharpe Ratio"). Used for titles and labels.
        theme (str, optional): The theme to apply ('dark' or 'light').
            Defaults to 'dark'.

    Returns:
        Optional[go.Figure]: A Plotly Figure object showing the histogram of bootstrap
            statistics with lines for observed value and CI, or None if input data
            is invalid (e.g., empty list or NaN values for bounds).

    Example:
        >>> # Assume these values are derived from a bootstrap procedure
        >>> bootstrapped_means = np.random.normal(loc=10, scale=2, size=1000).tolist()
        >>> observed_mean = 10.1
        >>> ci_lower = 9.5
        >>> ci_upper = 10.5
        >>> fig = plot_bootstrap_distribution_and_ci(
        ...     bootstrap_statistics=bootstrapped_means,
        ...     observed_statistic=observed_mean,
        ...     lower_bound=ci_lower,
        ...     upper_bound=ci_upper,
        ...     statistic_name="Average Profit"
        ... )
        >>> # if fig: fig.show()
    """
    if not bootstrap_statistics or pd.isna(observed_statistic) or pd.isna(lower_bound) or pd.isna(upper_bound):
        logger.warning("Bootstrap Distribution Plot: Invalid input data (empty statistics or NaN bounds).")
        return None

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=bootstrap_statistics, name='Bootstrap<br>Distribution',
        marker_color=COLORS.get('royal_blue', '#4169E1'),
        opacity=0.75, histnorm='probability density'
    ))
    fig.add_vline(
        x=observed_statistic, line_width=2, line_dash="dash",
        line_color=COLORS.get('green', '#00FF00'),
        name=f'Observed<br>{statistic_name}<br>({observed_statistic:.4f})'
    )
    fig.add_vline(
        x=lower_bound, line_width=2, line_dash="dot",
        line_color=COLORS.get('orange', '#FFA500'),
        name=f'Lower 95% CI<br>({lower_bound:.4f})'
    )
    fig.add_vline(
        x=upper_bound, line_width=2, line_dash="dot",
        line_color=COLORS.get('orange', '#FFA500'),
        name=f'Upper 95% CI<br>({upper_bound:.4f})'
    )
    fig.update_layout(
        title_text=f'Bootstrap Distribution for {statistic_name}',
        xaxis_title=statistic_name,
        yaxis_title='Density',
        bargap=0.1,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return _apply_custom_theme(fig, theme)

def plot_stacked_bar_chart(
    df: pd.DataFrame,
    category_col: str,
    stack_col: str,
    value_col: Optional[str] = None,
    title: Optional[str] = None,
    theme: str = 'dark',
    color_discrete_map: Optional[Dict[str, str]] = None
) -> Optional[go.Figure]:
    """
    Generates a stacked bar chart. Each bar represents a category from `category_col`.
    Within each bar, segments are stacked based on unique values from `stack_col`.
    The height of segments can represent counts or sums of `value_col`.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        category_col (str): Name of the column for the x-axis categories
            (e.g., 'Month', 'AssetClass').
        stack_col (str): Name of the column whose unique values will form the
            stacked segments within each bar (e.g., 'StrategyType', 'Region').
        value_col (Optional[str], optional): Numeric column whose values will be
            summed for the height of each segment. If None, the function counts
            occurrences to determine segment heights. Defaults to None (counts).
        title (Optional[str], optional): Title for the plot. If None, a default title
            is generated based on `stack_col` and `category_col`. Defaults to None.
        theme (str, optional): The theme to apply ('dark' or 'light').
            Defaults to 'dark'.
        color_discrete_map (Optional[Dict[str, str]], optional): A dictionary mapping
            unique values from `stack_col` to specific colors. Defaults to None.

    Returns:
        Optional[go.Figure]: A Plotly Figure object for the stacked bar chart,
            or None if input data is invalid or results in an empty chart.

    Example:
        >>> data = {'Month': ['Jan', 'Jan', 'Feb', 'Feb', 'Jan'],
        ...         'Product': ['A', 'B', 'A', 'B', 'A'],
        ...         'Sales': [100, 150, 120, 180, 80]}
        >>> sales_df = pd.DataFrame(data)
        >>> # Stacked bar chart of total sales per month, stacked by product
        >>> fig = plot_stacked_bar_chart(sales_df, category_col='Month',
        ...                              stack_col='Product', value_col='Sales',
        ...                              title="Monthly Sales by Product")
        >>> # if fig: fig.show()
        >>> # Stacked bar chart of transaction counts per month, stacked by product
        >>> fig_counts = plot_stacked_bar_chart(sales_df, category_col='Month',
        ...                                     stack_col='Product',
        ...                                     title="Monthly Transactions by Product")
        >>> # if fig_counts: fig_counts.show()
    """
    if df is None or df.empty or category_col not in df.columns or stack_col not in df.columns:
        logger.warning("Stacked Bar Chart: Input DataFrame is invalid or missing required columns.")
        return None
    if value_col and value_col not in df.columns:
        logger.warning(f"Stacked Bar Chart: Value column '{value_col}' not found. Will use counts instead.")
        value_col = None

    y_values_col_name = 'count'
    y_axis_title_text = "Count"

    try:
        if value_col:
            grouped_df = df.groupby([category_col, stack_col], as_index=False)[value_col].sum()
            y_values_col_name = value_col
            y_axis_title_text = f"Sum of {value_col.replace('_', ' ').title()}"
        else:
            grouped_df = df.groupby([category_col, stack_col], as_index=False).size()
            # pandas >= 2.0.0 names the size column 'size', older versions '0' or 'count'
            # We rename it consistently to 'count' for px.bar
            if 'size' in grouped_df.columns and 'count' not in grouped_df.columns:
                 grouped_df = grouped_df.rename(columns={'size': 'count'})
            elif 0 in grouped_df.columns and 'count' not in grouped_df.columns: # Older pandas might use 0
                 grouped_df = grouped_df.rename(columns={0: 'count'})


    except Exception as e:
        logger.error(f"Stacked Bar Chart: Error during grouping/aggregation: {e}")
        return None


    if grouped_df.empty:
        logger.warning("Stacked Bar Chart: Grouped data is empty.")
        return None

    fig_title = title if title else f"{stack_col.replace('_', ' ').title()} Distribution by {category_col.replace('_', ' ').title()}"

    fig = px.bar(
        grouped_df, x=category_col, y=y_values_col_name,
        color=stack_col, title=fig_title,
        barmode='stack', color_discrete_map=color_discrete_map
    )
    fig.update_layout(
        xaxis_title=category_col.replace('_', ' ').title(),
        yaxis_title=y_axis_title_text,
        legend_title_text=stack_col.replace('_', ' ').title()
    )
    return _apply_custom_theme(fig, theme)

def plot_grouped_bar_chart(
    df: pd.DataFrame,
    category_col: str,
    value_col: str,
    group_col: str,
    title: Optional[str] = None,
    aggregation_func: str = 'mean',
    theme: str = 'dark',
    color_discrete_map: Optional[Dict[str, str]] = None
) -> Optional[go.Figure]:
    """
    Generates a grouped bar chart. Bars for each `category_col` value are grouped
    by unique values from `group_col`. The height of bars is determined by
    aggregating `value_col`.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        category_col (str): Name of the column for the primary x-axis categories
            (e.g., 'Quarter', 'AssetType').
        value_col (str): Numeric column whose values will be aggregated for bar heights
            (e.g., 'Profit', 'Return'). If `aggregation_func` is 'count', this
            column is used for titling but not for y-values directly.
        group_col (str): Name of the column whose unique values will form the
            groups of bars for each category (e.g., 'Year', 'Strategy').
        title (Optional[str], optional): Title for the plot. If None, a default title
            is generated. Defaults to None.
        aggregation_func (str, optional): Aggregation function ('mean', 'sum', 'count')
            to apply to `value_col` within each category and group.
            Defaults to 'mean'.
        theme (str, optional): The theme to apply ('dark' or 'light').
            Defaults to 'dark'.
        color_discrete_map (Optional[Dict[str, str]], optional): A dictionary mapping
            unique values from `group_col` to specific colors. Defaults to None.

    Returns:
        Optional[go.Figure]: A Plotly Figure object for the grouped bar chart,
            or None if input data is invalid or results in an empty chart.

    Example:
        >>> data = {'Quarter': ['Q1', 'Q1', 'Q2', 'Q2', 'Q1', 'Q2'],
        ...         'Year': [2022, 2023, 2022, 2023, 2022, 2023],
        ...         'Revenue': [1000, 1200, 1100, 1300, 900, 1150],
        ...         'Region': ['North', 'North', 'South', 'South', 'South', 'North']}
        >>> revenue_df = pd.DataFrame(data)
        >>> # Grouped bar chart of average revenue per quarter, grouped by year
        >>> fig = plot_grouped_bar_chart(revenue_df, category_col='Quarter',
        ...                              value_col='Revenue', group_col='Year',
        ...                              aggregation_func='mean', title="Avg Revenue: Quarter by Year")
        >>> # if fig: fig.show()
        >>> # Grouped bar chart of transaction counts per region, grouped by quarter
        >>> fig_counts = plot_grouped_bar_chart(revenue_df, category_col='Region',
        ...                                     value_col='Revenue', # value_col still needed for title context
        ...                                     group_col='Quarter',
        ...                                     aggregation_func='count', title="Transactions: Region by Quarter")
        >>> # if fig_counts: fig_counts.show()
    """
    if df is None or df.empty or not all(c in df.columns for c in [category_col, value_col, group_col]):
        logger.warning("Grouped Bar Chart: Input DataFrame is invalid or missing required columns.")
        return None

    y_col_for_plot = value_col

    try:
        if aggregation_func == 'mean':
            grouped_df = df.groupby([category_col, group_col], as_index=False)[value_col].mean()
            y_axis_title = f"Average {value_col.replace('_', ' ').title()}"
        elif aggregation_func == 'sum':
            grouped_df = df.groupby([category_col, group_col], as_index=False)[value_col].sum()
            y_axis_title = f"Total {value_col.replace('_', ' ').title()}"
        elif aggregation_func == 'count':
            grouped_df = df.groupby([category_col, group_col], as_index=False).size()
            # Rename 'size' column to 'count' for consistency if present (pandas >= 2.0)
            if 'size' in grouped_df.columns and 'count' not in grouped_df.columns:
                grouped_df = grouped_df.rename(columns={'size': 'count'})
            elif 0 in grouped_df.columns and 'count' not in grouped_df.columns: # Older pandas
                grouped_df = grouped_df.rename(columns={0: 'count'})

            y_col_for_plot = 'count'
            y_axis_title = "Count"
        else:
            logger.error(f"Grouped Bar Chart: Invalid aggregation function '{aggregation_func}'.")
            return None
    except Exception as e:
        logger.error(f"Grouped Bar Chart: Error during grouping/aggregation: {e}")
        return None

    if grouped_df.empty:
        logger.warning("Grouped Bar Chart: Grouped data is empty.")
        return None

    fig_title_val_part = value_col.replace('_', ' ').title() if aggregation_func != 'count' else "Count"
    fig_title = title if title else f"{fig_title_val_part} by {category_col.replace('_', ' ').title()}, Grouped by {group_col.replace('_', ' ').title()}"

    fig = px.bar(
        grouped_df, x=category_col, y=y_col_for_plot,
        color=group_col, title=fig_title,
        barmode='group', color_discrete_map=color_discrete_map
    )
    fig.update_layout(
        xaxis_title=category_col.replace('_', ' ').title(),
        yaxis_title=y_axis_title,
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
    Generates a box plot to show the distribution of `value_col` for each
    category in `category_col`.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        category_col (str): Name of the column for x-axis categories. Each unique
            value in this column will correspond to a box in the plot.
        value_col (str): Numeric column whose distribution is to be plotted for
            each category. This will be the y-axis.
        title (Optional[str], optional): Title for the plot. If None, a default
            title is generated. Defaults to None.
        theme (str, optional): The theme to apply ('dark' or 'light').
            Defaults to 'dark'.
        color_discrete_map (Optional[Dict[str, str]], optional): A dictionary mapping
            unique values from `category_col` to specific colors for the boxes.
            If None, Plotly's default color sequence is used. Defaults to None.

    Returns:
        Optional[go.Figure]: A Plotly Figure object for the box plot,
            or None if input data is invalid or required columns are missing.

    Example:
        >>> data = {'Group': ['A', 'A', 'B', 'B', 'A', 'B', 'C', 'C'],
        ...         'Score': [70, 75, 80, 82, 65, 78, 90, 88]}
        >>> scores_df = pd.DataFrame(data)
        >>> fig = plot_box_plot(scores_df, category_col='Group', value_col='Score',
        ...                     title="Score Distribution by Group")
        >>> # if fig: fig.show()
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
        color=category_col if color_discrete_map or theme == 'dark' else None,
        title=fig_title,
        points="outliers",
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
    title: Optional[str] = None,
    theme: str = 'dark',
    color_discrete_map: Optional[Dict[str, str]] = None
) -> Optional[go.Figure]:
    """
    Generates a donut chart showing the distribution of categories in `category_col`.
    The size of each slice corresponds to the frequency of each category.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        category_col (str): Name of the column whose unique value counts will
            determine the slices of the donut chart.
        title (Optional[str], optional): Title for the plot. If None, a default
            title is generated. Defaults to None.
        theme (str, optional): The theme to apply ('dark' or 'light').
            Defaults to 'dark'.
        color_discrete_map (Optional[Dict[str, str]], optional): A dictionary mapping
            unique values from `category_col` to specific colors for the slices.
            Defaults to None.

    Returns:
        Optional[go.Figure]: A Plotly Figure object for the donut chart,
            or None if input data is invalid or results in no data to plot.

    Example:
        >>> data = {'AssetType': ['Stock', 'Bond', 'Stock', 'Stock', 'Forex', 'Bond']}
        >>> assets_df = pd.DataFrame(data)
        >>> fig = plot_donut_chart(assets_df, category_col='AssetType',
        ...                        title="Asset Type Distribution")
        >>> # if fig: fig.show()
    """
    if df is None or df.empty or category_col not in df.columns:
        logger.warning("Donut Chart: Input DataFrame is invalid or category column is missing.")
        return None

    counts = df[category_col].value_counts().reset_index()
    counts.columns = [category_col, 'count'] # Ensure correct column names for px.pie

    if counts.empty:
        logger.warning("Donut Chart: No data to plot after value counts.")
        return None

    fig_title = title if title else f"Distribution of {category_col.replace('_', ' ').title()}"

    fig = px.pie(
        counts, names=category_col, values='count',
        title=fig_title, hole=0.4,
        color_discrete_map=color_discrete_map
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return _apply_custom_theme(fig, theme)

def plot_radar_chart(
    df_radar: pd.DataFrame,
    categories_col: str,
    value_cols: List[str],
    title: Optional[str] = None,
    fill: str = 'toself',
    theme: str = 'dark',
    color_discrete_sequence: Optional[List[str]] = None
) -> Optional[go.Figure]:
    """
    Generates a radar chart (spider chart) to compare multiple quantitative variables
    represented on axes starting from the same point.

    The `df_radar` should be structured such that `categories_col` contains the labels
    for each axis of the radar, and each column in `value_cols` represents a
    different trace (e.g., a different item or time period being compared).

    Args:
        df_radar (pd.DataFrame): DataFrame where one column (`categories_col`) holds
            category labels for radar axes, and other columns (`value_cols`) hold
            the corresponding values for each trace.
        categories_col (str): Name of the column in `df_radar` containing the
            category labels for the radar axes (theta values).
        value_cols (List[str]): List of column names from `df_radar`, each
            representing a distinct trace (entity/metric set) on the radar.
            These provide the radial (r) values.
        title (Optional[str], optional): Title for the plot. If None, defaults to
            "Radar Chart Comparison". Defaults to None.
        fill (str, optional): Fill type for `go.Scatterpolar` traces (e.g., 'toself',
            'tonext'). Defaults to 'toself'.
        theme (str, optional): The theme to apply ('dark' or 'light').
            Defaults to 'dark'.
        color_discrete_sequence (Optional[List[str]], optional): A list of colors
            to use for the different traces defined by `value_cols`. If None,
            Plotly's default color sequence is used. Defaults to None.

    Returns:
        Optional[go.Figure]: A Plotly Figure object for the radar chart,
            or None if input data is invalid or insufficient.

    Example:
        >>> data = {
        ...     'Metric': ['Speed', 'Reliability', 'Cost', 'Support', 'Features'],
        ...     'ProductA': [4, 5, 3, 4, 5],
        ...     'ProductB': [5, 4, 4, 3, 4]
        ... }
        >>> radar_df = pd.DataFrame(data)
        >>> fig = plot_radar_chart(radar_df, categories_col='Metric',
        ...                        value_cols=['ProductA', 'ProductB'],
        ...                        title="Product Comparison")
        >>> # if fig: fig.show()
    """
    if df_radar is None or df_radar.empty or categories_col not in df_radar.columns or \
       not value_cols or not all(col in df_radar.columns for col in value_cols):
        logger.warning("Radar Chart: Input DataFrame is invalid or missing required columns.")
        return None

    fig = go.Figure()
    category_labels = df_radar[categories_col].tolist()

    if not category_labels:
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
            name=val_col.replace('_', ' ').title(),
            line_color=trace_color
        ))

    fig_title = title if title else "Radar Chart Comparison"
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True,)),
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
    Generates a scatter plot to visualize the relationship between two numeric variables
    (`x_col` and `y_col`). Optionally, points can be colored by `color_col` and
    sized by `size_col`.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        x_col (str): Name of the column for x-axis values.
        y_col (str): Name of the column for y-axis values.
        color_col (Optional[str], optional): Name of the column to use for coloring
            the scatter points. Can be categorical or continuous. Defaults to None.
        size_col (Optional[str], optional): Name of the numeric column to use for
            sizing the scatter points. Defaults to None.
        title (Optional[str], optional): Title for the plot. If None, a default title
            is generated based on `x_col` and `y_col`. Defaults to None.
        theme (str, optional): The theme to apply ('dark' or 'light').
            Defaults to 'dark'.
        color_discrete_map (Optional[Dict[str, str]], optional): A dictionary mapping
            unique values from `color_col` (if categorical) to specific colors.
            Defaults to None.

    Returns:
        Optional[go.Figure]: A Plotly Figure object for the scatter plot,
            or None if input data is invalid or required columns are missing.

    Example:
        >>> data = {'Efficiency': np.random.rand(50) * 10,
        ...         'Cost': np.random.rand(50) * 100 + 50,
        ...         'Category': np.random.choice(['X', 'Y', 'Z'], 50),
        ...         'Scale': np.random.rand(50) * 10}
        >>> scatter_df = pd.DataFrame(data)
        >>> fig = plot_scatter_plot(scatter_df, x_col='Efficiency', y_col='Cost',
        ...                         color_col='Category', size_col='Scale',
        ...                         title="Efficiency vs. Cost by Category and Scale")
        >>> # if fig: fig.show()
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
    # Ensure x_col and y_col are numeric
    if not pd.api.types.is_numeric_dtype(df[x_col]) or not pd.api.types.is_numeric_dtype(df[y_col]):
        logger.warning(f"Scatter Plot: X column '{x_col}' and Y column '{y_col}' must be numeric.")
        return None
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
    Plots the efficient frontier for a portfolio optimization problem.
    Optionally marks points for the maximum Sharpe ratio portfolio and the
    minimum volatility portfolio.

    Args:
        frontier_vols (List[float]): List of annualized volatilities (standard deviations)
            for portfolios on the efficient frontier.
        frontier_returns (List[float]): List of annualized expected returns for
            portfolios on the efficient frontier, corresponding to `frontier_vols`.
        max_sharpe_vol (Optional[float], optional): Volatility of the portfolio
            with the maximum Sharpe ratio. If provided with `max_sharpe_ret`,
            this point is marked. Defaults to None.
        max_sharpe_ret (Optional[float], optional): Return of the portfolio
            with the maximum Sharpe ratio. Defaults to None.
        min_vol_vol (Optional[float], optional): Volatility of the portfolio
            with the minimum volatility. If provided with `min_vol_ret`,
            this point is marked. Defaults to None.
        min_vol_ret (Optional[float], optional): Return of the portfolio
            with the minimum volatility. Defaults to None.
        title (str, optional): Title of the plot.
            Defaults to "Efficient Frontier".
        theme (str, optional): The theme to apply ('dark' or 'light').
            Defaults to 'dark'.

    Returns:
        Optional[go.Figure]: A Plotly Figure object for the efficient frontier,
            or None if input data for frontier points is invalid or empty.

    Example:
        >>> # Example data for an efficient frontier
        >>> vols = [0.1, 0.12, 0.15, 0.2, 0.25]
        >>> rets = [0.05, 0.07, 0.09, 0.11, 0.12]
        >>> # Optimal portfolios (example values)
        >>> ms_vol, ms_ret = 0.15, 0.09 # Max Sharpe
        >>> mv_vol, mv_ret = 0.1, 0.05  # Min Volatility
        >>> fig = plot_efficient_frontier(
        ...     frontier_vols=vols, frontier_returns=rets,
        ...     max_sharpe_vol=ms_vol, max_sharpe_ret=ms_ret,
        ...     min_vol_vol=mv_vol, min_vol_ret=mv_ret,
        ...     title="Sample Efficient Frontier"
        ... )
        >>> # if fig: fig.show()
    """
    if not frontier_vols or not frontier_returns or len(frontier_vols) != len(frontier_returns):
        logger.warning("Efficient Frontier: Invalid input data for frontier points (empty lists or mismatched lengths).")
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frontier_vols, y=frontier_returns,
        mode='lines', name='Efficient Frontier',
        line=dict(color=COLORS.get('royal_blue', '#4169E1'), width=2),
        hovertemplate='Volatility: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
    ))

    if max_sharpe_vol is not None and max_sharpe_ret is not None:
        fig.add_trace(go.Scatter(
            x=[max_sharpe_vol], y=[max_sharpe_ret],
            mode='markers', name='Max Sharpe Ratio Portfolio',
            marker=dict(color=COLORS.get('green', '#00FF00'), size=10, symbol='star'),
            hovertemplate='Max Sharpe<br>Volatility: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
        ))

    if min_vol_vol is not None and min_vol_ret is not None:
        is_distinct_from_max_sharpe = True
        if max_sharpe_vol is not None and max_sharpe_ret is not None:
            # Check if min_vol point is too close to max_sharpe to be considered distinct
            if abs(min_vol_vol - max_sharpe_vol) < 1e-6 and abs(min_vol_ret - max_sharpe_ret) < 1e-6:
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
        xaxis_tickformat=".2%", yaxis_tickformat=".2%",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return _apply_custom_theme(fig, theme)
