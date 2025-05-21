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

from config import (
    COLORS, PLOTLY_THEME_DARK, PLOTLY_THEME_LIGHT,
    PLOT_BG_COLOR_DARK, PLOT_PAPER_BG_COLOR_DARK, PLOT_FONT_COLOR_DARK,
    PLOT_BG_COLOR_LIGHT, PLOT_PAPER_BG_COLOR_LIGHT, PLOT_FONT_COLOR_LIGHT,
    PLOT_LINE_COLOR, PLOT_MARKER_PROFIT_COLOR, PLOT_MARKER_LOSS_COLOR,
    PLOT_BENCHMARK_LINE_COLOR,
    EXPECTED_COLUMNS, APP_TITLE
)
from utils.common_utils import format_currency, format_percentage 

import logging
logger = logging.getLogger(APP_TITLE)


def _apply_custom_theme(fig: go.Figure, theme: str = 'dark') -> go.Figure:
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
    text_format: str = ".2f", # e.g., ".2f", "$,.0f", "$.2%"
    theme: str = 'dark'
) -> Optional[go.Figure]:
    if df_pivot is None or df_pivot.empty:
        logger.warning("Heatmap: Input pivot DataFrame is empty.")
        return None

    # Prepare text for heatmap cells based on text_format
    formatted_text_values = None
    if show_text:
        def format_cell_value(val):
            if pd.isna(val):
                return ""
            
            # Check for currency or percentage in text_format
            is_currency = text_format.startswith('$')
            is_percentage = text_format.endswith('%')
            
            numeric_format_part = text_format
            prefix = ""
            suffix = ""

            if is_currency:
                prefix = "$"
                numeric_format_part = numeric_format_part[1:] # Remove $
            
            if is_percentage:
                suffix = "%"
                numeric_format_part = numeric_format_part[:-1] # Remove %
                # For percentages, the value itself might need to be multiplied by 100
                # if it's a decimal (e.g., 0.75 for 75%).
                # However, if the data is already scaled (e.g., 75 for 75%), no multiplication needed.
                # Let's assume for heatmap text, if % is in format, value is already scaled.
                # If text_format was e.g. ".1%", val is 75.0, numeric_format_part becomes ".1"
                # f"{val:.1f}%"
            
            try:
                # Apply the core numeric formatting
                formatted_num = f"{val:{numeric_format_part}}"
                return f"{prefix}{formatted_num}{suffix}"
            except ValueError:
                logger.warning(f"Heatmap: Could not apply format '{text_format}' to value '{val}'. Returning raw value.")
                return str(val) # Fallback to string representation

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
        # Ensure hover text also respects formatting if possible, or use default
        hovertemplate="<b>%{y}</b><br>%{x}: %{z}<extra></extra>" # Default hover
    ))
    fig.update_layout(
        title_text=title,
        xaxis_title=x_axis_title if x_axis_title else df_pivot.columns.name,
        yaxis_title=y_axis_title if y_axis_title else df_pivot.index.name
    )
    return _apply_custom_theme(fig, theme)

# ... (rest of plotting.py, including plot_equity_curve_and_drawdown and other functions) ...
# Ensure all other functions are preserved below this point.

def plot_equity_curve_and_drawdown(
    df: pd.DataFrame, date_col: str = EXPECTED_COLUMNS['date'], cumulative_pnl_col: str = 'cumulative_pnl',
    drawdown_pct_col: Optional[str] = 'drawdown_pct', drawdown_periods_df: Optional[pd.DataFrame] = None,
    theme: str = 'dark', max_dd_peak_date: Optional[Any] = None, max_dd_trough_date: Optional[Any] = None,
    max_dd_recovery_date: Optional[Any] = None
) -> Optional[go.Figure]:
    if df is None or df.empty or date_col not in df.columns or cumulative_pnl_col not in df.columns: return None
    try: df[date_col] = pd.to_datetime(df[date_col])
    except Exception as e: logger.error(f"Equity curve plot: Could not convert date column '{date_col}' to datetime: {e}"); return None
    has_drawdown_data_series = drawdown_pct_col and drawdown_pct_col in df.columns and not df[drawdown_pct_col].dropna().empty
    fig_rows, row_heights = (2, [0.7, 0.3]) if has_drawdown_data_series else (1, [1.0])
    subplot_titles_list = ["Equity Curve"] + (["Drawdown (%)"] if has_drawdown_data_series else [])
    fig = make_subplots(rows=fig_rows, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=row_heights, subplot_titles=subplot_titles_list)
    fig.add_trace(go.Scatter(x=df[date_col], y=df[cumulative_pnl_col], mode='lines', name='Strategy Equity', line=dict(color=PLOT_LINE_COLOR, width=2)), row=1, col=1)
    fig.update_yaxes(title_text="Cumulative PnL ($)", row=1, col=1)
    if max_dd_peak_date and max_dd_trough_date:
        try:
            peak_dt, end_shade_dt = pd.to_datetime(max_dd_peak_date), None
            if pd.notna(max_dd_recovery_date): end_shade_dt = pd.to_datetime(max_dd_recovery_date)
            elif not df[date_col].empty: end_shade_dt = df[date_col].max()
            if end_shade_dt and peak_dt < end_shade_dt:
                 fig.add_vrect(x0=peak_dt, x1=end_shade_dt, fillcolor=COLORS.get('red', 'red'), opacity=0.25, layer="below", line_width=1, line_color=COLORS.get('red', 'red'), annotation_text="Max Drawdown Period", annotation_position="top left", annotation_font_size=10, annotation_font_color=COLORS.get('red', 'red'), row=1, col=1)
            elif peak_dt:
                trough_dt_for_shade = pd.to_datetime(max_dd_trough_date)
                if peak_dt < trough_dt_for_shade : fig.add_vrect(x0=peak_dt, x1=trough_dt_for_shade, fillcolor=COLORS.get('red', 'red'), opacity=0.25, layer="below", line_width=1, line_color=COLORS.get('red', 'red'), annotation_text="Max DD (Peak to Trough)", annotation_position="top left", annotation_font_size=10, annotation_font_color=COLORS.get('red', 'red'), row=1, col=1)
        except Exception as e_vrect: logger.error(f"Error adding max drawdown vrect: {e_vrect}", exc_info=True)
    if drawdown_periods_df is not None and not drawdown_periods_df.empty:
        for _, dd_period in drawdown_periods_df.iterrows():
            peak_date, end_date_for_shading = pd.to_datetime(dd_period.get('Peak Date')), pd.to_datetime(dd_period.get('End Date'))
            if pd.isna(end_date_for_shading) and not df[date_col].empty:
                last_data_date = pd.to_datetime(df[date_col].iloc[-1])
                if pd.notna(peak_date) and last_data_date > peak_date: end_date_for_shading = last_data_date
                else: continue 
            if pd.notna(peak_date) and pd.notna(end_date_for_shading) and peak_date < end_date_for_shading:
                is_max_dd_period = max_dd_peak_date and pd.to_datetime(max_dd_peak_date) == peak_date
                if not is_max_dd_period: fig.add_vrect(x0=peak_date, x1=end_date_for_shading, fillcolor=COLORS.get('red', 'red'), opacity=0.10, layer="below", line_width=0, row=1, col=1)
    if has_drawdown_data_series:
        fig.add_trace(go.Scatter(x=df[date_col], y=df[drawdown_pct_col], mode='lines', name='Drawdown', line=dict(color=COLORS.get('red', '#FF0000'), width=1.5), fill='tozeroy', fillcolor='rgba(255,0,0,0.2)'), row=2, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1, tickformat=".2f")
        min_dd_val, max_dd_val = df[drawdown_pct_col].min(skipna=True), df[drawdown_pct_col].max(skipna=True)
        if pd.isna(min_dd_val) or pd.isna(max_dd_val) or (min_dd_val == 0 and max_dd_val == 0) : fig.update_yaxes(range=[-1, 1], row=2, col=1)
    fig.update_layout(title_text='Strategy Equity and Drawdown Periods', hovermode='x unified'); return _apply_custom_theme(fig, theme)

def plot_underwater_analysis(equity_series: pd.Series, theme: str = 'dark', title: str = "Underwater Plot (Equity vs. High Water Mark)") -> Optional[go.Figure]:
    if equity_series is None or equity_series.empty or not isinstance(equity_series.index, pd.DatetimeIndex): return None
    if len(equity_series.dropna()) < 2: return None
    equity, high_water_mark = equity_series.dropna(), equity_series.dropna().cummax()
    fig_filled = go.Figure()
    fig_filled.add_trace(go.Scatter(x=high_water_mark.index, y=high_water_mark, mode='lines', name='High Water Mark', line=dict(color=COLORS.get('green', 'green'), dash='dash')))
    fig_filled.add_trace(go.Scatter(x=equity.index, y=equity, mode='lines', name='Equity Curve', line=dict(color=PLOT_LINE_COLOR), fill='tonexty', fillcolor='rgba(255, 0, 0, 0.2)'))
    fig_filled.update_layout(title_text=title, xaxis_title="Date", yaxis_title="Equity Value", hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)); return _apply_custom_theme(fig_filled, theme)

def plot_equity_vs_benchmark(strategy_equity: pd.Series, benchmark_cumulative_returns: pd.Series, strategy_name: str = "Strategy", benchmark_name: str = "Benchmark", theme: str = 'dark') -> Optional[go.Figure]:
    if strategy_equity.empty and benchmark_cumulative_returns.empty: return None
    fig = go.Figure()
    if not strategy_equity.empty: fig.add_trace(go.Scatter(x=strategy_equity.index, y=strategy_equity, mode='lines', name=strategy_name, line=dict(color=PLOT_LINE_COLOR, width=2)))
    if not benchmark_cumulative_returns.empty: fig.add_trace(go.Scatter(x=benchmark_cumulative_returns.index, y=benchmark_cumulative_returns, mode='lines', name=benchmark_name, line=dict(color=PLOT_BENCHMARK_LINE_COLOR, width=2, dash='dash')))
    fig.update_layout(title_text=f'{strategy_name} vs. {benchmark_name} Performance', xaxis_title="Date", yaxis_title="Normalized Value / Cumulative Return (%)", hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)); return _apply_custom_theme(fig, theme)

def plot_pnl_distribution(df: pd.DataFrame, pnl_col: str = EXPECTED_COLUMNS['pnl'], title: str = "PnL Distribution (per Trade)", theme: str = 'dark', nbins: int = 50) -> Optional[go.Figure]:
    if df is None or df.empty or pnl_col not in df.columns or df[pnl_col].dropna().empty: return None
    fig = px.histogram(df, x=pnl_col, nbins=nbins, title=title, marginal="box", color_discrete_sequence=[PLOT_LINE_COLOR])
    fig.update_layout(xaxis_title="PnL per Trade", yaxis_title="Frequency"); return _apply_custom_theme(fig, theme)

def plot_time_series_decomposition(decomposition_result: Any, title: str = "Time Series Decomposition", theme: str = 'dark') -> Optional[go.Figure]:
    if decomposition_result is None: return None
    try:
        observed, trend, seasonal, resid = getattr(decomposition_result, 'observed', pd.Series(dtype=float)), getattr(decomposition_result, 'trend', pd.Series(dtype=float)), getattr(decomposition_result, 'seasonal', pd.Series(dtype=float)), getattr(decomposition_result, 'resid', pd.Series(dtype=float))
        if observed.empty: return None
        x_axis = observed.index
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=("Observed", "Trend", "Seasonal", "Residual"))
        fig.add_trace(go.Scatter(x=x_axis, y=observed, mode='lines', name='Observed', line=dict(color=PLOT_LINE_COLOR)), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_axis, y=trend, mode='lines', name='Trend', line=dict(color=COLORS.get('green', '#00FF00'))), row=2, col=1)
        fig.add_trace(go.Scatter(x=x_axis, y=seasonal, mode='lines', name='Seasonal', line=dict(color=COLORS.get('royal_blue', '#4169E1'))), row=3, col=1)
        fig.add_trace(go.Scatter(x=x_axis, y=resid, mode='lines+markers', name='Residual', line=dict(color=COLORS.get('gray', '#808080')), marker=dict(size=3)), row=4, col=1)
        fig.update_layout(title_text=title, height=700, showlegend=False); return _apply_custom_theme(fig, theme)
    except Exception as e: logger.error(f"Error plotting decomposition: {e}", exc_info=True); return None

def plot_value_over_time(series: pd.Series, series_name: str, title: Optional[str] = None, x_axis_title: str = "Date / Time", y_axis_title: Optional[str] = None, theme: str = 'dark', line_color: str = PLOT_LINE_COLOR) -> Optional[go.Figure]:
    if series is None or series.empty: return None
    fig = go.Figure(); fig.add_trace(go.Scatter(x=series.index, y=series.values, mode='lines', name=series_name, line=dict(color=line_color)))
    fig.update_layout(title_text=title if title else series_name, xaxis_title=x_axis_title, yaxis_title=y_axis_title if y_axis_title else series_name); return _apply_custom_theme(fig, theme)

def plot_pnl_by_category(df: pd.DataFrame, category_col: str, pnl_col: str = EXPECTED_COLUMNS['pnl'], title_prefix: str = "Total PnL by", theme: str = 'dark', aggregation_func: str = 'sum' ) -> Optional[go.Figure]:
    if df is None or df.empty or category_col not in df.columns or pnl_col not in df.columns: return None
    grouped_pnl = df.groupby(category_col)[pnl_col].agg(aggregation_func).reset_index().sort_values(by=pnl_col, ascending=False)
    yaxis_title, title = (f"{aggregation_func.title()} PnL", f"{title_prefix.replace('Total', aggregation_func.title())} {category_col.replace('_', ' ').title()}") if aggregation_func != 'sum' else ("Total PnL", f"{title_prefix} {category_col.replace('_', ' ').title()}")
    fig = px.bar(grouped_pnl, x=category_col, y=pnl_col, title=title, color=pnl_col, color_continuous_scale=[COLORS.get('red', '#FF0000'), COLORS.get('gray', '#808080'), COLORS.get('green', '#00FF00')])
    fig.update_layout(xaxis_title=category_col.replace('_', ' ').title(), yaxis_title=yaxis_title); return _apply_custom_theme(fig, theme)

def plot_win_rate_analysis(df: pd.DataFrame, category_col: str, win_col: str = 'win', title_prefix: str = "Win Rate by", theme: str = 'dark') -> Optional[go.Figure]:
    if df is None or df.empty or category_col not in df.columns or win_col not in df.columns: return None
    if not pd.api.types.is_bool_dtype(df[win_col]) and not pd.api.types.is_numeric_dtype(df[win_col]): return None
    category_counts, category_wins = df.groupby(category_col).size().rename('total_trades_in_cat'), df.groupby(category_col)[win_col].sum().rename('wins_in_cat')
    win_rate_df = pd.concat([category_counts, category_wins], axis=1).fillna(0)
    win_rate_df['win_rate_pct'] = (win_rate_df['wins_in_cat'] / win_rate_df['total_trades_in_cat'] * 100).fillna(0)
    win_rate_df = win_rate_df.reset_index().sort_values(by='win_rate_pct', ascending=False)
    fig = px.bar(win_rate_df, x=category_col, y='win_rate_pct', title=f"{title_prefix} {category_col.replace('_', ' ').title()}", color='win_rate_pct', color_continuous_scale=px.colors.sequential.Greens)
    fig.update_layout(xaxis_title=category_col.replace('_', ' ').title(), yaxis_title="Win Rate (%)", yaxis_ticksuffix="%"); return _apply_custom_theme(fig, theme)

def plot_rolling_performance(df: pd.DataFrame, date_col: str, metric_series: pd.Series, metric_name: str, title: Optional[str] = None, theme: str = 'dark') -> Optional[go.Figure]:
    if df is None or df.empty or date_col not in df.columns or metric_series.empty: return None
    plot_x_data = df[date_col] if len(df[date_col]) == len(metric_series) else metric_series.index
    fig = go.Figure(); fig.add_trace(go.Scatter(x=plot_x_data, y=metric_series, mode='lines', name=metric_name, line=dict(color=PLOT_LINE_COLOR)))
    fig.update_layout(title_text=title if title else f"Rolling {metric_name}", xaxis_title="Date" if date_col in df.columns and len(df[date_col]) == len(metric_series) else "Trade Number / Period", yaxis_title=metric_name); return _apply_custom_theme(fig, theme)

def plot_correlation_matrix(df: pd.DataFrame, numeric_cols: Optional[List[str]] = None, title: str = "Correlation Matrix of Numeric Features", theme: str = 'dark') -> Optional[go.Figure]:
    if df is None or df.empty: return None
    df_numeric = df[numeric_cols].copy() if numeric_cols else df.select_dtypes(include=np.number)
    if df_numeric.empty or df_numeric.shape[1] < 2: return None
    corr_matrix = df_numeric.corr()
    fig = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns, colorscale='RdBu', zmin=-1, zmax=1, text=corr_matrix.round(2).astype(str), texttemplate="%{text}", hoverongaps=False ))
    fig.update_layout(title_text=title); return _apply_custom_theme(fig, theme)

def plot_bootstrap_distribution_and_ci(bootstrap_statistics: List[float], observed_statistic: float, lower_bound: float, upper_bound: float, statistic_name: str, theme: str = 'dark') -> Optional[go.Figure]:
    if not bootstrap_statistics or pd.isna(observed_statistic) or pd.isna(lower_bound) or pd.isna(upper_bound): return None
    fig = go.Figure(); fig.add_trace(go.Histogram(x=bootstrap_statistics, name='Bootstrap<br>Distribution', marker_color=COLORS.get('royal_blue', '#4169E1'), opacity=0.75, histnorm='probability density'))
    fig.add_vline(x=observed_statistic, line_width=2, line_dash="dash", line_color=COLORS.get('green', '#00FF00'), name=f'Observed<br>{statistic_name}<br>({observed_statistic:.4f})')
    fig.add_vline(x=lower_bound, line_width=2, line_dash="dot", line_color=COLORS.get('orange', '#FFA500'), name=f'Lower 95% CI<br>({lower_bound:.4f})')
    fig.add_vline(x=upper_bound, line_width=2, line_dash="dot", line_color=COLORS.get('orange', '#FFA500'), name=f'Upper 95% CI<br>({upper_bound:.4f})')
    fig.update_layout(title_text=f'Bootstrap Distribution for {statistic_name}', xaxis_title=statistic_name, yaxis_title='Density', bargap=0.1, showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)); return _apply_custom_theme(fig, theme)

def plot_stacked_bar_chart(df: pd.DataFrame, category_col: str, stack_col: str, value_col: Optional[str] = None, title: Optional[str] = None, theme: str = 'dark', color_discrete_map: Optional[Dict[str, str]] = None) -> Optional[go.Figure]:
    if df is None or df.empty or category_col not in df.columns or stack_col not in df.columns: return None
    if value_col and value_col not in df.columns: value_col = None 
    y_values, y_axis_title = (value_col, f"Sum of {value_col.replace('_', ' ').title()}") if value_col else ('count', "Count")
    grouped_df = df.groupby([category_col, stack_col])[value_col].sum().reset_index() if value_col else df.groupby([category_col, stack_col]).size().reset_index(name='count')
    if grouped_df.empty: return None
    fig_title = title if title else f"{stack_col.replace('_', ' ').title()} Distribution by {category_col.replace('_', ' ').title()}"
    fig = px.bar(grouped_df, x=category_col, y=y_values, color=stack_col, title=fig_title, barmode='stack', color_discrete_map=color_discrete_map)
    fig.update_layout(xaxis_title=category_col.replace('_', ' ').title(), yaxis_title=y_axis_title, legend_title_text=stack_col.replace('_', ' ').title()); return _apply_custom_theme(fig, theme)

def plot_grouped_bar_chart(df: pd.DataFrame, category_col: str, value_col: str, group_col: str, title: Optional[str] = None, aggregation_func: str = 'mean', theme: str = 'dark', color_discrete_map: Optional[Dict[str, str]] = None) -> Optional[go.Figure]:
    if df is None or df.empty or not all(c in df.columns for c in [category_col, value_col, group_col]): return None
    if aggregation_func == 'mean': grouped_df, y_axis_title = df.groupby([category_col, group_col])[value_col].mean().reset_index(), f"Average {value_col.replace('_', ' ').title()}"
    elif aggregation_func == 'sum': grouped_df, y_axis_title = df.groupby([category_col, group_col])[value_col].sum().reset_index(), f"Total {value_col.replace('_', ' ').title()}"
    elif aggregation_func == 'count': grouped_df, value_col, y_axis_title = df.groupby([category_col, group_col]).size().reset_index(name='count'), 'count', "Count"
    else: return None
    if grouped_df.empty: return None
    fig_title = title if title else f"{value_col.replace('_', ' ').title()} by {category_col.replace('_', ' ').title()}, Grouped by {group_col.replace('_', ' ').title()}"
    fig = px.bar(grouped_df, x=category_col, y=value_col, color=group_col, title=fig_title, barmode='group', color_discrete_map=color_discrete_map)
    fig.update_layout(xaxis_title=category_col.replace('_', ' ').title(), yaxis_title=y_axis_title, legend_title_text=group_col.replace('_', ' ').title()); return _apply_custom_theme(fig, theme)

def plot_box_plot(df: pd.DataFrame, category_col: str, value_col: str, title: Optional[str] = None, theme: str = 'dark', color_discrete_map: Optional[Dict[str, str]] = None) -> Optional[go.Figure]:
    if df is None or df.empty or not all(c in df.columns for c in [category_col, value_col]): return None
    fig_title = title if title else f"{value_col.replace('_', ' ').title()} Distribution by {category_col.replace('_', ' ').title()}"
    fig = px.box(df, x=category_col, y=value_col, color=category_col if color_discrete_map else None, title=fig_title, points="outliers", color_discrete_map=color_discrete_map)
    fig.update_layout(xaxis_title=category_col.replace('_', ' ').title(), yaxis_title=value_col.replace('_', ' ').title()); return _apply_custom_theme(fig, theme)

def plot_donut_chart(df: pd.DataFrame, category_col: str, title: Optional[str] = None, theme: str = 'dark', color_discrete_map: Optional[Dict[str, str]] = None) -> Optional[go.Figure]:
    if df is None or df.empty or category_col not in df.columns: return None
    counts = df[category_col].value_counts().reset_index(); counts.columns = [category_col, 'count']
    if counts.empty: return None
    fig_title = title if title else f"Distribution of {category_col.replace('_', ' ').title()}"
    fig = px.pie(counts, names=category_col, values='count', title=fig_title, hole=0.4, color_discrete_map=color_discrete_map)
    fig.update_traces(textposition='inside', textinfo='percent+label'); return _apply_custom_theme(fig, theme)

def plot_radar_chart(df_radar: pd.DataFrame, categories_col: str, value_cols: List[str], title: Optional[str] = None, fill: str = 'toself', theme: str = 'dark', color_discrete_sequence: Optional[List[str]] = None ) -> Optional[go.Figure]:
    if df_radar is None or df_radar.empty or categories_col not in df_radar.columns or not value_cols or not all(col in df_radar.columns for col in value_cols): return None
    fig = go.Figure(); category_labels = df_radar[categories_col].tolist()
    if not category_labels: return None
    for i, val_col in enumerate(value_cols):
        trace_color = color_discrete_sequence[i % len(color_discrete_sequence)] if color_discrete_sequence else None
        fig.add_trace(go.Scatterpolar(r=df_radar[val_col].tolist(), theta=category_labels, fill=fill, name=val_col.replace('_', ' ').title(), line_color=trace_color))
    fig_title = title if title else "Radar Chart Comparison"
    fig.update_layout(polar=dict(radialaxis=dict(visible=True,)), showlegend=True, title=fig_title); return _apply_custom_theme(fig, theme)

def plot_scatter_plot(df: pd.DataFrame, x_col: str, y_col: str, color_col: Optional[str] = None, size_col: Optional[str] = None, title: Optional[str] = None, theme: str = 'dark', color_discrete_map: Optional[Dict[str, str]] = None) -> Optional[go.Figure]:
    if df is None or df.empty or not all(c in df.columns for c in [x_col, y_col]): return None
    if color_col and color_col not in df.columns: color_col = None 
    if size_col and size_col not in df.columns: size_col = None 
    fig_title = title if title else f"{y_col.replace('_', ' ').title()} vs. {x_col.replace('_', ' ').title()}"
    fig = px.scatter(df, x=x_col, y=y_col, color=color_col, size=size_col, title=fig_title, color_discrete_map=color_discrete_map)
    fig.update_layout(xaxis_title=x_col.replace('_', ' ').title(), yaxis_title=y_col.replace('_', ' ').title(), legend_title_text=color_col.replace('_', ' ').title() if color_col else None); return _apply_custom_theme(fig, theme)

def plot_efficient_frontier(frontier_vols: List[float], frontier_returns: List[float], max_sharpe_vol: Optional[float] = None, max_sharpe_ret: Optional[float] = None, min_vol_vol: Optional[float] = None, min_vol_ret: Optional[float] = None, title: str = "Efficient Frontier", theme: str = 'dark') -> Optional[go.Figure]:
    if not frontier_vols or not frontier_returns or len(frontier_vols) != len(frontier_returns): return None
    fig = go.Figure(); fig.add_trace(go.Scatter(x=frontier_vols, y=frontier_returns, mode='lines', name='Efficient Frontier', line=dict(color=COLORS.get('royal_blue', '#4169E1'), width=2), hovertemplate='Volatility: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'))
    if max_sharpe_vol is not None and max_sharpe_ret is not None:
        fig.add_trace(go.Scatter(x=[max_sharpe_vol], y=[max_sharpe_ret], mode='markers', name='Max Sharpe Ratio Portfolio', marker=dict(color=COLORS.get('green', '#00FF00'), size=10, symbol='star'), hovertemplate='Max Sharpe<br>Volatility: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'))
    if min_vol_vol is not None and min_vol_ret is not None:
        is_distinct = not (max_sharpe_vol is not None and abs(min_vol_vol - max_sharpe_vol) < 1e-4 and abs(min_vol_ret - max_sharpe_ret) < 1e-4)
        if is_distinct: fig.add_trace(go.Scatter(x=[min_vol_vol], y=[min_vol_ret], mode='markers', name='Minimum Volatility Portfolio', marker=dict(color=COLORS.get('orange', '#FFA500'), size=10, symbol='diamond'), hovertemplate='Min Volatility<br>Volatility: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'))
    fig.update_layout(title_text=title, xaxis_title="Annualized Volatility (Standard Deviation)", yaxis_title="Annualized Expected Return", xaxis_tickformat=".2%", yaxis_tickformat=".2%", hovermode="closest", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)); return _apply_custom_theme(fig, theme)
