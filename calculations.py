"""
calculations.py

Implements calculations for all specified Key Performance Indicators (KPIs)
and provides functions for their qualitative interpretation and color-coding
based on thresholds defined in config.py.
Includes benchmark-relative metrics like Alpha and Beta, and advanced drawdown.
"""
import pandas as pd
import numpy as np
from scipy import stats # Retained for potential future use
from typing import Dict, Any, Tuple, List, Optional, Union

import logging

# Assuming config.py is in the root directory
# KPI_CONFIG will be imported from kpi_definitions (via config.py)
from config import RISK_FREE_RATE, KPI_CONFIG, COLORS, EXPECTED_COLUMNS, APP_TITLE

try:
    from utils.common_utils import format_currency, format_percentage
except ImportError:
    # Fallback definitions if common_utils is not found
    print("Warning (calculations.py): Could not import formatting functions from utils.common_utils. Using basic fallbacks.")
    def format_currency(value: float, currency_symbol: str = "$", decimals: int = 2) -> str:
        if pd.isna(value) or np.isinf(value): return "N/A"
        return f"{currency_symbol}{value:,.{decimals}f}"

    def format_percentage(value: float, decimals: int = 2) -> str:
        if pd.isna(value) or np.isinf(value): return "N/A"
        return f"{value * 100:.{decimals}f}%"

logger = logging.getLogger(APP_TITLE)
if not logger.handlers and not logging.getLogger().handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

# --- Define Minimum Data Points Thresholds ---
MIN_TRADES_FOR_RATIOS = 5
MIN_TRADES_FOR_DISTRIBUTION = 5
MIN_DAILY_RETURNS_FOR_RATIOS = 10
MIN_ALIGNED_POINTS_FOR_BENCHMARK = 10
MIN_LOSSES_FOR_VAR = 10
MIN_POINTS_FOR_DRAWDOWN_ANALYSIS = 5

def _calculate_returns(pnl_series: pd.Series, initial_capital: Optional[float] = None) -> pd.Series:
    if pnl_series.empty:
        return pd.Series(dtype=float)
    if initial_capital and initial_capital != 0:
        return pnl_series / initial_capital
    return pnl_series

def _calculate_drawdowns(cumulative_pnl: pd.Series) -> Tuple[pd.Series, float, float, pd.Series]:
    if cumulative_pnl.empty or len(cumulative_pnl) < 2:
        return pd.Series(dtype=float), 0.0, 0.0, pd.Series(dtype=float)
    high_water_mark = cumulative_pnl.cummax()
    absolute_drawdown_series_values = high_water_mark - cumulative_pnl
    max_drawdown_abs_val = absolute_drawdown_series_values.max() if not absolute_drawdown_series_values.empty else 0.0
    hwm_for_pct = high_water_mark.replace(0, np.nan)
    percentage_drawdown_series_values = (absolute_drawdown_series_values / hwm_for_pct).fillna(0) * 100
    mask_hwm_zero_loss = (high_water_mark == 0) & (absolute_drawdown_series_values > 0)
    percentage_drawdown_series_values[mask_hwm_zero_loss] = 100.0
    max_drawdown_pct_val = percentage_drawdown_series_values.max() if not percentage_drawdown_series_values.empty else 0.0
    if max_drawdown_abs_val > 0 and max_drawdown_pct_val == 0 and (high_water_mark <= 0).all():
        max_drawdown_pct_val = 100.0
    return absolute_drawdown_series_values, max_drawdown_abs_val, max_drawdown_pct_val, percentage_drawdown_series_values

def _calculate_streaks(pnl_series: pd.Series) -> Tuple[int, int]:
    if pnl_series.empty or len(pnl_series) < 1: return 0, 0
    wins = pnl_series > 0; losses = pnl_series < 0
    max_win_streak = current_win_streak = 0
    for w in wins: current_win_streak = current_win_streak + 1 if w else 0; max_win_streak = max(max_win_streak, current_win_streak)
    max_loss_streak = current_loss_streak = 0
    for l_val in losses: current_loss_streak = current_loss_streak + 1 if l_val else 0; max_loss_streak = max(max_loss_streak, current_loss_streak)
    return int(max_win_streak), int(max_loss_streak)

def analyze_detailed_drawdowns(equity_series: pd.Series) -> Dict[str, Any]:
    results = {
        "drawdown_periods": pd.DataFrame(),
        "max_drawdown_details": {},
        "total_time_in_drawdown_days": 0.0,
        "average_drawdown_duration_days": np.nan,
        "average_recovery_duration_days": np.nan,
    }
    if not isinstance(equity_series.index, pd.DatetimeIndex):
        logger.error("analyze_detailed_drawdowns: equity_series must have a DatetimeIndex.")
        results["error"] = "Equity series must have a DatetimeIndex."
        return results
    if equity_series.empty or len(equity_series) < MIN_POINTS_FOR_DRAWDOWN_ANALYSIS:
        logger.warning(f"analyze_detailed_drawdowns: Not enough data points (need >={MIN_POINTS_FOR_DRAWDOWN_ANALYSIS}).")
        results["error"] = f"Not enough data points (need at least {MIN_POINTS_FOR_DRAWDOWN_ANALYSIS})."
        return results

    equity = equity_series.dropna()
    if equity.empty or len(equity) < MIN_POINTS_FOR_DRAWDOWN_ANALYSIS:
        results["error"] = "Not enough valid data points after NaN removal."
        return results

    high_water_mark_series = equity.cummax()
    drawdown_series_abs = high_water_mark_series - equity
    hwm_for_pct = high_water_mark_series.replace(0, np.nan) 
    drawdown_series_pct = (drawdown_series_abs / hwm_for_pct).fillna(0) * 100
    mask_hwm_zero_loss = (high_water_mark_series == 0) & (drawdown_series_abs > 0)
    drawdown_series_pct[mask_hwm_zero_loss] = 100.0

    drawdowns = []
    in_drawdown = False
    current_peak_date = None
    current_peak_value = -np.inf

    for i in range(len(equity)):
        current_date = equity.index[i]
        current_value = equity.iloc[i]
        if not in_drawdown:
            if current_value >= current_peak_value:
                current_peak_value = current_value
                current_peak_date = current_date
            else: 
                in_drawdown = True
                drawdown_start_date = current_peak_date
                drawdown_peak_value = current_peak_value
                current_trough_value = current_value
                current_trough_date = current_date
        if in_drawdown:
            if current_value < current_trough_value:
                current_trough_value = current_value
                current_trough_date = current_date
            if current_value >= drawdown_peak_value:
                in_drawdown = False
                drawdown_end_date = current_date
                depth_abs = drawdown_peak_value - current_trough_value
                depth_pct = (depth_abs / drawdown_peak_value) * 100 if drawdown_peak_value > 1e-9 else (100.0 if depth_abs > 1e-9 else 0.0)
                duration_days = (current_trough_date - drawdown_start_date).days if drawdown_start_date and current_trough_date else 0
                recovery_days = (drawdown_end_date - current_trough_date).days if drawdown_end_date and current_trough_date else 0
                drawdowns.append({
                    "Peak Date": drawdown_start_date, "Peak Value": drawdown_peak_value,
                    "Trough Date": current_trough_date, "Trough Value": current_trough_value,
                    "End Date": drawdown_end_date, "Depth Abs": depth_abs, "Depth Pct": depth_pct,
                    "Duration Days": duration_days, "Recovery Days": recovery_days
                })
                current_peak_value = current_value 
                current_peak_date = current_date
    if in_drawdown:
        depth_abs = drawdown_peak_value - current_trough_value
        depth_pct = (depth_abs / drawdown_peak_value) * 100 if drawdown_peak_value > 1e-9 else (100.0 if depth_abs > 1e-9 else 0.0)
        duration_days = (current_trough_date - drawdown_start_date).days if drawdown_start_date and current_trough_date else 0
        drawdowns.append({
            "Peak Date": drawdown_start_date, "Peak Value": drawdown_peak_value,
            "Trough Date": current_trough_date, "Trough Value": current_trough_value,
            "End Date": pd.NaT, "Depth Abs": depth_abs, "Depth Pct": depth_pct,
            "Duration Days": duration_days, "Recovery Days": np.nan
        })
    if drawdowns:
        results["drawdown_periods"] = pd.DataFrame(drawdowns)
        if not results["drawdown_periods"].empty:
            max_dd_period = results["drawdown_periods"].loc[results["drawdown_periods"]["Depth Abs"].idxmax()]
            results["max_drawdown_details"] = max_dd_period.to_dict()
            total_duration = 0; total_recovery = 0; num_recovered_drawdowns = 0
            for _, row in results["drawdown_periods"].iterrows():
                start = row["Peak Date"]; end = row["End Date"]; trough = row["Trough Date"]
                if pd.notna(end) and pd.notna(start):
                    total_duration += (end - start).days
                    if pd.notna(trough): total_recovery += (end - trough).days; num_recovered_drawdowns +=1
                elif pd.notna(start): total_duration += (equity.index[-1] - start).days
            results["total_time_in_drawdown_days"] = total_duration
            if not results["drawdown_periods"].empty: results["average_drawdown_duration_days"] = results["drawdown_periods"]["Duration Days"].mean()
            if num_recovered_drawdowns > 0: results["average_recovery_duration_days"] = total_recovery / num_recovered_drawdowns
    return results

# THIS FUNCTION MUST BE DEFINED *BEFORE* calculate_all_kpis
def calculate_benchmark_metrics(
    strategy_daily_returns: Union[pd.Series, pd.DataFrame],
    benchmark_daily_returns: Union[pd.Series, pd.DataFrame],
    risk_free_rate: float,
    periods_per_year: int = 252
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "alpha": np.nan, "beta": np.nan, "benchmark_correlation": np.nan,
        "tracking_error": np.nan, "information_ratio": np.nan
    }

    sdr = strategy_daily_returns
    bdr = benchmark_daily_returns

    if isinstance(sdr, pd.DataFrame):
        if sdr.shape[1] == 1: sdr = sdr.iloc[:, 0].copy()
        else: logger.error("Strategy daily returns is a multi-column DataFrame for benchmark metrics."); return metrics
    if isinstance(bdr, pd.DataFrame):
        if bdr.shape[1] == 1: bdr = bdr.iloc[:, 0].copy()
        else: logger.error("Benchmark daily returns is a multi-column DataFrame for benchmark metrics."); return metrics
    
    if not isinstance(sdr, pd.Series) or not isinstance(bdr, pd.Series):
        logger.error("Strategy or benchmark returns are not Series after processing for benchmark metrics.")
        return metrics

    if sdr.empty or bdr.empty:
        logger.warning("Cannot calculate benchmark metrics: strategy or benchmark returns are empty.")
        return metrics

    aligned_df = pd.DataFrame({'strategy': sdr, 'benchmark': bdr}).dropna()

    if len(aligned_df) < MIN_ALIGNED_POINTS_FOR_BENCHMARK:
        logger.warning(f"Not enough overlapping data points (<{MIN_ALIGNED_POINTS_FOR_BENCHMARK}) between strategy and benchmark to calculate metrics. Found: {len(aligned_df)}.")
        return metrics

    strat_returns_1d = aligned_df['strategy']
    bench_returns_1d = aligned_df['benchmark']
    
    covariance = strat_returns_1d.cov(bench_returns_1d)
    benchmark_variance = bench_returns_1d.var()
    if benchmark_variance != 0 and not np.isnan(benchmark_variance) and not np.isinf(benchmark_variance):
        metrics['beta'] = covariance / benchmark_variance
    else:
        metrics['beta'] = np.nan

    daily_rfr = (1 + risk_free_rate)**(1/periods_per_year) - 1
    avg_strat_return_period = strat_returns_1d.mean()
    avg_bench_return_period = bench_returns_1d.mean()

    if not np.isnan(metrics['beta']):
        alpha_period = (avg_strat_return_period - daily_rfr) - metrics['beta'] * (avg_bench_return_period - daily_rfr)
        metrics['alpha'] = alpha_period * periods_per_year * 100
    else:
        metrics['alpha'] = np.nan

    metrics['benchmark_correlation'] = strat_returns_1d.corr(bench_returns_1d)

    difference_returns = strat_returns_1d - bench_returns_1d
    tracking_error_period_std = difference_returns.std()
    if not np.isnan(tracking_error_period_std):
        metrics['tracking_error'] = tracking_error_period_std * np.sqrt(periods_per_year) * 100
    else:
        metrics['tracking_error'] = np.nan
        
    if tracking_error_period_std != 0 and not np.isnan(tracking_error_period_std) and not np.isinf(tracking_error_period_std):
        avg_excess_return_period = difference_returns.mean()
        metrics['information_ratio'] = (avg_excess_return_period / tracking_error_period_std) * np.sqrt(periods_per_year)
    else:
        metrics['information_ratio'] = np.nan
        
    return metrics

def calculate_all_kpis(
    df: pd.DataFrame,
    risk_free_rate: float = RISK_FREE_RATE,
    benchmark_daily_returns: Optional[pd.Series] = None,
    initial_capital: Optional[float] = None
) -> Dict[str, Any]:
    kpis: Dict[str, Any] = {kpi_key: np.nan for kpi_key in KPI_CONFIG.keys()}
    pnl_col = EXPECTED_COLUMNS.get('pnl', 'pnl')
    date_col = EXPECTED_COLUMNS.get('date', 'date')

    if df is None or df.empty:
        logger.warning("KPI calculation skipped: DataFrame is None or empty.")
        return kpis
    if pnl_col not in df.columns:
        logger.warning(f"KPI calculation skipped: PnL column '{pnl_col}' not found.")
        return kpis
    if date_col not in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        logger.warning(f"KPI calculation skipped: Date column '{date_col}' not found and index is not DatetimeIndex.")

    pnl_series = df[pnl_col].dropna()
    if pnl_series.empty:
        logger.warning("KPI calculation skipped: PnL series is empty after dropping NaNs.")
        return kpis

    kpis['total_pnl'] = pnl_series.sum()
    kpis['total_trades'] = len(pnl_series)
    
    wins = pnl_series[pnl_series > 0]
    losses = pnl_series[pnl_series < 0]
    num_wins = len(wins)
    num_losses = len(losses)

    if kpis['total_trades'] > 0:
        kpis['win_rate'] = (num_wins / kpis['total_trades']) * 100
        kpis['avg_trade_pnl'] = pnl_series.mean()
        kpis['loss_rate'] = (num_losses / kpis['total_trades']) * 100
    else:
        kpis['win_rate'] = 0.0
        kpis['avg_trade_pnl'] = 0.0
        kpis['loss_rate'] = 0.0

    if kpis['total_trades'] >= MIN_TRADES_FOR_RATIOS:
        total_gross_profit = wins.sum()
        total_gross_loss = abs(losses.sum())
        
        if total_gross_loss > 1e-9:
            kpis['profit_factor'] = total_gross_profit / total_gross_loss
        elif total_gross_profit > 0:
            kpis['profit_factor'] = np.inf
        else:
            kpis['profit_factor'] = 0.0

        kpis['avg_win'] = wins.mean() if num_wins > 0 else 0.0
        kpis['avg_loss'] = abs(losses.mean()) if num_losses > 0 else 0.0
        
        if kpis['avg_loss'] > 1e-9:
            kpis['win_loss_ratio'] = kpis['avg_win'] / kpis['avg_loss']
        elif kpis['avg_win'] > 0:
            kpis['win_loss_ratio'] = np.inf
        else:
            kpis['win_loss_ratio'] = 0.0
    else:
        logger.info(f"Profit factor, avg win/loss, win/loss ratio require at least {MIN_TRADES_FOR_RATIOS} trades. Found {kpis['total_trades']}.")

    temp_cum_pnl_for_dd = pd.Series(dtype=float)
    if 'cumulative_pnl' in df.columns and not df['cumulative_pnl'].dropna().empty:
        cum_pnl_input = pd.to_numeric(df['cumulative_pnl'], errors='coerce').ffill().fillna(0)
        if not cum_pnl_input.empty:
            temp_cum_pnl_for_dd = cum_pnl_input
    
    if temp_cum_pnl_for_dd.empty:
        if date_col in df.columns:
            df_sorted_for_dd = df.sort_values(by=date_col)
            temp_cum_pnl_for_dd = df_sorted_for_dd[pnl_col].cumsum()
        else:
            temp_cum_pnl_for_dd = pnl_series.cumsum()
            
    if not temp_cum_pnl_for_dd.empty and len(temp_cum_pnl_for_dd) >= 2:
        _abs_dd_series, max_abs_dd_val, max_pct_dd_val, _pct_dd_series = _calculate_drawdowns(temp_cum_pnl_for_dd)
        kpis['max_drawdown_abs'] = max_abs_dd_val
        kpis['max_drawdown_pct'] = max_pct_dd_val
    else:
        kpis['max_drawdown_abs'], kpis['max_drawdown_pct'] = 0.0, 0.0
    if np.isinf(kpis['max_drawdown_pct']): kpis['max_drawdown_pct'] = 100.0

    daily_pnl = pd.Series(dtype=float)
    if date_col in df.columns:
        try:
            df_copy_for_daily = df.copy()
            df_copy_for_daily[date_col] = pd.to_datetime(df_copy_for_daily[date_col], errors='coerce')
            df_valid_dates = df_copy_for_daily.dropna(subset=[date_col])
            if not df_valid_dates.empty:
                daily_pnl = df_valid_dates.groupby(df_valid_dates[date_col].dt.normalize())[pnl_col].sum()
                if not isinstance(daily_pnl.index, pd.DatetimeIndex):
                    daily_pnl.index = pd.to_datetime(daily_pnl.index)
            else:
                logger.warning("No valid dates found after attempting conversion for daily PnL aggregation.")
        except Exception as e_date_agg:
            logger.error(f"Error during date processing for daily PnL aggregation: {e_date_agg}")
    elif isinstance(df.index, pd.DatetimeIndex):
         daily_pnl = df.groupby(df.index.normalize())[pnl_col].sum()
    else:
        logger.warning("Date column missing and index is not DatetimeIndex. Cannot calculate daily metrics like Sharpe, Sortino, Calmar.")

    strategy_daily_returns = daily_pnl
    if initial_capital is not None and initial_capital > 1e-9:
        strategy_daily_returns = daily_pnl / initial_capital
    
    if not isinstance(strategy_daily_returns, pd.Series):
        strategy_daily_returns = pd.Series(strategy_daily_returns, dtype=float)

    if not strategy_daily_returns.empty and len(strategy_daily_returns) >= MIN_DAILY_RETURNS_FOR_RATIOS:
        mean_daily_return = strategy_daily_returns.mean()
        std_daily_return = strategy_daily_returns.std()
        periods_per_year = 252
        daily_rfr = (1 + risk_free_rate)**(1/periods_per_year) - 1

        if std_daily_return > 1e-9 and not np.isnan(std_daily_return):
            kpis['sharpe_ratio'] = (mean_daily_return - daily_rfr) / std_daily_return * np.sqrt(periods_per_year)
        else:
            kpis['sharpe_ratio'] = 0.0 if (mean_daily_return - daily_rfr) <= 0 else np.inf

        negative_daily_returns = strategy_daily_returns[strategy_daily_returns < daily_rfr]
        if not negative_daily_returns.empty and len(negative_daily_returns) >= 2:
            downside_std_daily = (negative_daily_returns - daily_rfr).std()
            if downside_std_daily > 1e-9 and not np.isnan(downside_std_daily):
                kpis['sortino_ratio'] = (mean_daily_return - daily_rfr) / downside_std_daily * np.sqrt(periods_per_year)
            else:
                kpis['sortino_ratio'] = 0.0 if (mean_daily_return - daily_rfr) <= 0 else np.inf
        else:
             kpis['sortino_ratio'] = 0.0 if (mean_daily_return - daily_rfr) <= 0 else np.inf
            
        annualized_return_from_daily = mean_daily_return * periods_per_year
        mdd_pct_for_calmar = kpis.get('max_drawdown_pct', 0.0)
        if mdd_pct_for_calmar > 1e-9:
            mdd_decimal = mdd_pct_for_calmar / 100.0
            kpis['calmar_ratio'] = annualized_return_from_daily / mdd_decimal
        else:
            kpis['calmar_ratio'] = np.inf if annualized_return_from_daily > 0 else 0.0
    else:
        if not strategy_daily_returns.empty:
            logger.info(f"Sharpe, Sortino, Calmar ratios require at least {MIN_DAILY_RETURNS_FOR_RATIOS} daily return points. Found {len(strategy_daily_returns)}.")

    if not daily_pnl.empty:
        losses_for_var_daily = -daily_pnl[daily_pnl < 0]
        if len(losses_for_var_daily) >= MIN_LOSSES_FOR_VAR:
            kpis['var_95_loss'] = losses_for_var_daily.quantile(0.95)
            kpis['cvar_95_loss'] = losses_for_var_daily[losses_for_var_daily >= kpis.get('var_95_loss', 0)].mean()
            kpis['var_99_loss'] = losses_for_var_daily.quantile(0.99)
            kpis['cvar_99_loss'] = losses_for_var_daily[losses_for_var_daily >= kpis.get('var_99_loss', 0)].mean()
        else:
            if not losses_for_var_daily.empty:
                logger.info(f"VaR/CVaR require at least {MIN_LOSSES_FOR_VAR} loss observations. Found {len(losses_for_var_daily)}.")
    else:
        logger.info("Daily PnL series is empty, cannot calculate VaR/CVaR.")

    if kpis['total_trades'] >= MIN_TRADES_FOR_DISTRIBUTION:
        kpis['pnl_skewness'] = pnl_series.skew()
        kpis['pnl_kurtosis'] = pnl_series.kurtosis()
    else:
        logger.info(f"PnL Skewness/Kurtosis require at least {MIN_TRADES_FOR_DISTRIBUTION} trades. Found {kpis['total_trades']}.")

    kpis['max_win_streak'], kpis['max_loss_streak'] = _calculate_streaks(pnl_series)

    kpis['trading_days'] = daily_pnl.count() if not daily_pnl.empty else 0
    kpis['avg_daily_pnl'] = daily_pnl.mean() if not daily_pnl.empty and daily_pnl.count() > 0 else 0.0
    kpis['risk_free_rate_used'] = risk_free_rate * 100

    if benchmark_daily_returns is not None and not benchmark_daily_returns.empty:
        if initial_capital is None:
            logger.warning("Calculating benchmark metrics (Alpha, Beta) using absolute daily PnL for strategy as initial_capital was not provided. Results may be hard to interpret against benchmark's percentage returns.")
        
        temp_benchmark_returns = benchmark_daily_returns
        if isinstance(temp_benchmark_returns, pd.DataFrame):
            temp_benchmark_returns = temp_benchmark_returns.squeeze()
        if not isinstance(temp_benchmark_returns, pd.Series):
            logger.error("Benchmark returns provided to calculate_all_kpis is not a Series. Skipping benchmark metrics.")
            temp_benchmark_returns = pd.Series(dtype=float)

        if not strategy_daily_returns.empty and not temp_benchmark_returns.empty:
            # This is where the call happens
            benchmark_kpis_calculated = calculate_benchmark_metrics(
                strategy_daily_returns,
                temp_benchmark_returns,
                risk_free_rate
            )
            kpis.update(benchmark_kpis_calculated)

            aligned_benchmark_returns_for_total = temp_benchmark_returns.reindex(strategy_daily_returns.index).dropna()
            if not aligned_benchmark_returns_for_total.empty and len(aligned_benchmark_returns_for_total) >= 2:
                kpis['benchmark_total_return'] = ((1 + aligned_benchmark_returns_for_total).cumprod().iloc[-1] - 1) * 100
    
    for key in ["profit_factor", "win_loss_ratio", "sortino_ratio", "sharpe_ratio", "calmar_ratio"]:
        if key in kpis and np.isinf(kpis[key]) and kpis[key] < 0:
            kpis[key] = 0.0
            
    if not strategy_daily_returns.empty:
        mean_daily_ret = strategy_daily_returns.mean()
        std_daily_ret = strategy_daily_returns.std()
        trading_days_const = 252
        
        kpis['expected_annual_return'] = mean_daily_ret * trading_days_const * 100
        kpis['annual_volatility'] = std_daily_ret * np.sqrt(trading_days_const) * 100
    else:
        kpis['expected_annual_return'] = np.nan
        kpis['annual_volatility'] = np.nan

    logger.info(f"Calculated KPIs. Result keys: {list(kpis.keys())}")
    return kpis

def get_kpi_interpretation(kpi_key: str, value: float) -> Tuple[str, str]:
    if kpi_key not in KPI_CONFIG or pd.isna(value):
        return "N/A", "KPI not found or value is N/A"

    if np.isinf(value):
        if kpi_key in ["profit_factor", "win_loss_ratio", "sharpe_ratio", "sortino_ratio", "calmar_ratio"]:
            interp_inf = "Extremely High" if value > 0 else "Extremely Low / Undefined"
            return interp_inf, f"Val: {'+Infinity' if value > 0 else '-Infinity'}"
        else:
            return "Undefined (Infinity)", f"Val: {'+Infinity' if value > 0 else '-Infinity'}"

    config = KPI_CONFIG[kpi_key]
    thresholds = config.get("thresholds", [])
    unit = config.get("unit", "")
    interpretation = "N/A"; threshold_desc = "N/A"
    value_for_comparison = value

    for label, min_val, max_val_exclusive in thresholds:
        if min_val <= value_for_comparison < max_val_exclusive:
            interpretation = label
            if min_val == float('-inf'): threshold_desc = f"< {max_val_exclusive:,.1f}{unit if unit != '%' else ''}"
            elif max_val_exclusive == float('inf'): threshold_desc = f">= {min_val:,.1f}{unit if unit != '%' else ''}"
            else: threshold_desc = f"{min_val:,.1f} - {max_val_exclusive:,.1f}{unit if unit != '%' else ''}"
            break

    if interpretation == "N/A" and thresholds:
        last_label, last_min, last_max = thresholds[-1]
        if value_for_comparison >= last_min and last_max == float('inf'):
            interpretation = last_label; threshold_desc = f">= {last_min:,.1f}{unit if unit != '%' else ''}"
        elif value_for_comparison < thresholds[0][1] and thresholds[0][0] != float('-inf'):
             interpretation = thresholds[0][0]; threshold_desc = f"< {thresholds[0][1]:,.1f}{unit if unit != '%' else ''}"

    formatted_value_display = ""
    if unit == "$": formatted_value_display = format_currency(value)
    elif unit == "%":
        scaled_percentage_kpis = [
            "win_rate", "loss_rate", "max_drawdown_pct", "alpha", "tracking_error",
            "benchmark_total_return", "risk_free_rate_used",
            "expected_annual_return", "annual_volatility"
        ]
        if kpi_key in scaled_percentage_kpis:
            formatted_value_display = format_percentage(value / 100.0)
        else: 
            formatted_value_display = format_percentage(value)
    else: formatted_value_display = f"{value:,.2f}{unit}"

    return interpretation, f"Val: {formatted_value_display} (Thr: {threshold_desc})" if interpretation != "N/A" else f"Val: {formatted_value_display}"

def get_kpi_color(kpi_key: str, value: float) -> str:
    if kpi_key not in KPI_CONFIG or pd.isna(value) or np.isinf(value):
        return COLORS.get("gray", "#808080")

    config = KPI_CONFIG[kpi_key]
    color_logic_fn = config.get("color_logic")

    if color_logic_fn and callable(color_logic_fn):
        try:
            return color_logic_fn(value, 0, COLORS)
        except Exception as e:
            logger.error(f"Error in KPI color_logic for '{kpi_key}': {e}", exc_info=True)
            return COLORS.get("gray", "#808080")

    if "pnl" in kpi_key.lower() or "profit" in kpi_key.lower() or "alpha" in kpi_key.lower():
        if value > 0: return COLORS.get("green", "#00FF00")
        if value < 0: return COLORS.get("red", "#FF0000")
    if "drawdown" in kpi_key.lower() or "loss" in kpi_key.lower() or "volatility" in kpi_key.lower() or "tracking_error" in kpi_key.lower():
        if value > 0 : return COLORS.get("red", "#FF0000")

    return COLORS.get("gray", "#808080")
