# services/analysis_service.py
"""
Orchestrates core analytical calculations and model executions.
Specialized statistical, stochastic, and AI/ML analyses are delegated
to their respective services. Now includes advanced drawdown analysis.
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple, Callable

try:
    from config import APP_TITLE, RISK_FREE_RATE, EXPECTED_COLUMNS, CONFIDENCE_LEVEL, BOOTSTRAP_ITERATIONS
    from calculations import calculate_all_kpis, analyze_detailed_drawdowns
    from statistical_methods import bootstrap_confidence_interval
    from plotting import plot_pnl_distribution
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR in AnalysisService module: {e}. Some functionalities may fail.")
    APP_TITLE = "TradingDashboard_ErrorState"; RISK_FREE_RATE = 0.02; EXPECTED_COLUMNS = {"pnl": "pnl", "date": "date"}
    CONFIDENCE_LEVEL = 0.95; BOOTSTRAP_ITERATIONS = 1000
    def calculate_all_kpis(df, rfr, benchmark_daily_returns=None, initial_capital=None): return {"error": "calc_kpis not loaded"}
    def analyze_detailed_drawdowns(equity_series): return {"error": "analyze_detailed_drawdowns not loaded"}
    def bootstrap_confidence_interval(d, _sf, **kw): return {"error": "bootstrap_ci not loaded", "lower_bound":np.nan, "upper_bound":np.nan, "bootstrap_statistics": []}
    def plot_pnl_distribution(*args, **kwargs): return None

import logging
# logger = logging.getLogger(APP_TITLE) # Instance logger will be in the class

# Define constants at the module level if they are used by top-level functions
MIN_DATA_FOR_CORE_KPIS = 1
MIN_DATA_FOR_BOOTSTRAP_CI_KPI = 10
MIN_DATA_FOR_ADV_DRAWDOWN = 10

# --- MODIFICATION START: Top-level cached function ---
@st.cache_data(ttl=3600, show_spinner="Performing advanced drawdown analysis (cached)...")
def _cached_get_advanced_drawdown_analysis_logic(
    equity_values_tuple: Tuple[float, ...],
    equity_index_tuple: Tuple[str, ...], # Explicitly string for index elements
    series_name: Optional[str] = None,
    # Pass APP_TITLE to get the correct logger name if needed, or use a generic logger name
    logger_name_for_cached_func: str = "CachedDrawdownLogic" 
) -> Dict[str, Any]:
    """
    Cached core logic for advanced drawdown analysis.
    Accepts tuples for values and index to ensure hashability for caching.
    """
    _logger = logging.getLogger(logger_name_for_cached_func) # Use a specific logger
    _logger.info(f"Executing _cached_get_advanced_drawdown_analysis_logic for series '{series_name}'.")

    if not equity_values_tuple or not equity_index_tuple:
        _logger.warning("Equity values or index tuple is empty.")
        return {"error": "Equity values or index tuple is empty for advanced drawdown analysis."}
    if len(equity_values_tuple) != len(equity_index_tuple):
        _logger.error("Mismatch in length between equity values and index tuples.")
        return {"error": "Mismatch in length between equity values and index tuples."}

    try:
        # Reconstruct the pandas Series
        reconstructed_equity_series = pd.Series(
            list(equity_values_tuple),
            index=pd.to_datetime(list(equity_index_tuple)), # Convert ISO strings back to DatetimeIndex
            name=series_name
        ).sort_index() # Ensure it's sorted by index for drawdown logic

        if reconstructed_equity_series.empty:
            _logger.warning("Reconstructed equity series is empty.")
            return {"error": "Reconstructed equity series is empty."}
        if not isinstance(reconstructed_equity_series.index, pd.DatetimeIndex):
            _logger.error("Reconstructed equity series index is not DatetimeIndex after conversion.")
            return {"error": "Failed to reconstruct equity series with a DatetimeIndex."}
        
        # Drop NaNs that might have resulted from data issues before tuple conversion,
        # or from the values themselves if they were NaN (represented as float('nan'))
        reconstructed_equity_series_cleaned = reconstructed_equity_series.dropna()

        if len(reconstructed_equity_series_cleaned) < MIN_DATA_FOR_ADV_DRAWDOWN:
            _logger.warning(f"Not enough data points after cleaning (need >= {MIN_DATA_FOR_ADV_DRAWDOWN}). Found {len(reconstructed_equity_series_cleaned)}.")
            return {"error": f"Not enough data points (need at least {MIN_DATA_FOR_ADV_DRAWDOWN}) after reconstruction and cleaning for advanced drawdown analysis."}
        
        # Call the actual analysis function (not cached)
        analysis_results = analyze_detailed_drawdowns(reconstructed_equity_series_cleaned)
        return analysis_results
    except Exception as e:
        _logger.error(f"Error during cached drawdown logic (after reconstruction): {e}", exc_info=True)
        return {"error": f"An unexpected error occurred during cached drawdown logic: {str(e)}"}
# --- MODIFICATION END ---

class AnalysisService:
    def __init__(self):
        self.logger = logging.getLogger(APP_TITLE) # Instance logger
        self.logger.info(f"AnalysisService initialized.")

    def get_core_kpis(
        self,
        trades_df: pd.DataFrame,
        risk_free_rate: Optional[float] = None,
        benchmark_daily_returns: Optional[pd.Series] = None,
        initial_capital: Optional[float] = None
    ) -> Dict[str, Any]:
        if trades_df is None or trades_df.empty:
            self.logger.warning("get_core_kpis: Input data for KPI calculation is empty.")
            return {"error": "Input data for KPI calculation is empty."}
        if len(trades_df) < MIN_DATA_FOR_CORE_KPIS:
            self.logger.warning(f"get_core_kpis: Not enough trades (need >= {MIN_DATA_FOR_CORE_KPIS}). Found {len(trades_df)}.")
            return {"error": f"Not enough trades for KPI calculation (need at least {MIN_DATA_FOR_CORE_KPIS})."}

        rfr = risk_free_rate if risk_free_rate is not None else RISK_FREE_RATE
        try:
            pnl_col_name = EXPECTED_COLUMNS.get('pnl')
            date_col_name = EXPECTED_COLUMNS.get('date')
            if not pnl_col_name or pnl_col_name not in trades_df.columns:
                return {"error": f"Required PnL column ('{pnl_col_name}') not found."}
            if not date_col_name or date_col_name not in trades_df.columns:
                 if not isinstance(trades_df.index, pd.DatetimeIndex):
                    return {"error": f"Required Date column ('{date_col_name}') not found and index is not DatetimeIndex."}
            if trades_df[pnl_col_name].isnull().all():
                 return {"error": f"PnL column ('{pnl_col_name}') contains only NaN values."}

            kpi_results = calculate_all_kpis(
                trades_df,
                risk_free_rate=rfr,
                benchmark_daily_returns=benchmark_daily_returns,
                initial_capital=initial_capital
            )
            if pd.isna(kpi_results.get('total_pnl')) and pd.isna(kpi_results.get('sharpe_ratio')):
                 self.logger.warning("Several critical KPIs are NaN. This might indicate issues with input PnL data or insufficient data for specific ratios.")
            return kpi_results
        except Exception as e:
            self.logger.error(f"Error calculating core KPIs: {e}", exc_info=True)
            return {"error": str(e)}


    def get_bootstrapped_kpi_cis(self, trades_df: pd.DataFrame, kpis_to_bootstrap: Optional[List[str]] = None) -> Dict[str, Any]:
        if trades_df is None or trades_df.empty:
            self.logger.warning("get_bootstrapped_kpi_cis: Input data for CI calculation is empty.")
            return {"error": "Input data for CI calculation is empty."}

        pnl_col_name = EXPECTED_COLUMNS.get('pnl')
        if not pnl_col_name or pnl_col_name not in trades_df.columns:
            self.logger.error(f"get_bootstrapped_kpi_cis: PnL column '{pnl_col_name}' not found.")
            return {"error": f"PnL column ('{pnl_col_name}') not found for CI calculation."}

        pnl_series = trades_df[pnl_col_name].dropna()
        if pnl_series.empty or len(pnl_series) < MIN_DATA_FOR_BOOTSTRAP_CI_KPI:
            self.logger.warning(f"get_bootstrapped_kpi_cis: PnL data insufficient (need >= {MIN_DATA_FOR_BOOTSTRAP_CI_KPI}). Found {len(pnl_series)}.")
            return {"error": f"PnL data insufficient (need at least {MIN_DATA_FOR_BOOTSTRAP_CI_KPI} values) for CI calculation."}

        if kpis_to_bootstrap is None: kpis_to_bootstrap = ['avg_trade_pnl', 'win_rate', 'sharpe_ratio']
        
        confidence_intervals: Dict[str, Any] = {}

        for kpi_key in kpis_to_bootstrap:
            stat_fn: Optional[Callable[[pd.Series], float]] = None
            if kpi_key == 'avg_trade_pnl': stat_fn = np.mean
            elif kpi_key == 'win_rate': stat_fn = lambda x: (np.sum(x > 0) / len(x)) * 100 if len(x) > 0 else 0.0
            elif kpi_key == 'sharpe_ratio':
                def simplified_sharpe_stat_fn(pnl_sample: pd.Series) -> float:
                    if len(pnl_sample) < 2: return 0.0 
                    std_dev = pnl_sample.std()
                    if std_dev == 0 or np.isnan(std_dev): return 0.0 if pnl_sample.mean() <= 0 else np.inf
                    return pnl_sample.mean() / std_dev
                stat_fn = simplified_sharpe_stat_fn

            if stat_fn:
                try:
                    res = bootstrap_confidence_interval(pnl_series, _statistic_func=stat_fn)
                    if 'error' not in res: confidence_intervals[kpi_key] = (res['lower_bound'], res['upper_bound'])
                    else: confidence_intervals[kpi_key] = (np.nan, np.nan); self.logger.warning(f"CI calc error for {kpi_key}: {res['error']}")
                except Exception as e: self.logger.error(f"Exception during bootstrap for {kpi_key}: {e}", exc_info=True); confidence_intervals[kpi_key] = (np.nan, np.nan)
            else: confidence_intervals[kpi_key] = (np.nan, np.nan); self.logger.warning(f"No CI stat_fn for {kpi_key}")
        return confidence_intervals

    # --- MODIFICATION START: Instance method now calls the top-level cached function ---
    def get_advanced_drawdown_analysis(
        self,
        equity_series: pd.Series # Public method still accepts the Series
    ) -> Dict[str, Any]:
        """
        Prepares data and calls the cached logic for advanced drawdown analysis.
        Args:
            equity_series (pd.Series): Equity curve with DatetimeIndex.
        Returns:
            Dict[str, Any]: Results from the cached analysis.
        """
        self.logger.info(f"Preparing data for cached advanced drawdown analysis for series '{equity_series.name if equity_series is not None else 'None'}'.")

        if equity_series is None or equity_series.empty:
            self.logger.warning("get_advanced_drawdown_analysis (instance method): Equity series is empty.")
            return {"error": "Equity series is empty for advanced drawdown analysis."}

        # Ensure the input series has a DatetimeIndex and is sorted
        equity_series_prepared = equity_series.copy()
        if not isinstance(equity_series_prepared.index, pd.DatetimeIndex):
            try:
                equity_series_prepared.index = pd.to_datetime(equity_series_prepared.index)
                if not isinstance(equity_series_prepared.index, pd.DatetimeIndex): # Check again
                    raise ValueError("Index could not be converted to DatetimeIndex.")
            except Exception as e:
                self.logger.error(f"Equity series index is not DatetimeIndex and conversion failed: {e}", exc_info=True)
                return {"error": "Equity series index must be DatetimeIndex for drawdown analysis, and conversion failed."}
        
        equity_series_prepared = equity_series_prepared.sort_index().dropna()

        if len(equity_series_prepared) < MIN_DATA_FOR_ADV_DRAWDOWN:
            self.logger.warning(f"Not enough valid data points (need >= {MIN_DATA_FOR_ADV_DRAWDOWN}). Found {len(equity_series_prepared)}.")
            return {"error": f"Not enough valid data points (need at least {MIN_DATA_FOR_ADV_DRAWDOWN}) for advanced drawdown analysis."}

        # Prepare hashable inputs for the cached function
        # Convert NaN to a specific float value if necessary, or ensure tuple() handles them.
        # float('nan') is hashable. Using list comprehension for explicit float conversion.
        equity_values_list = [float(v) for v in equity_series_prepared.values]
        equity_values_tuple = tuple(equity_values_list)
        
        equity_index_tuple = tuple(dt.isoformat() for dt in equity_series_prepared.index)
        series_name_for_cache = equity_series_prepared.name

        # Call the new top-level cached function
        return _cached_get_advanced_drawdown_analysis_logic(
            equity_values_tuple=equity_values_tuple,
            equity_index_tuple=equity_index_tuple,
            series_name=series_name_for_cache,
            logger_name_for_cached_func=f"{APP_TITLE}.CachedDrawdown" # Pass a logger name
        )
    # --- MODIFICATION END ---

    def generate_pnl_distribution_plot(self, trades_df: pd.DataFrame, theme: str = 'dark') -> Optional[Any]:
        if trades_df is None or trades_df.empty:
            self.logger.warning("generate_pnl_distribution_plot: Input DataFrame is empty.")
            return None 
        pnl_col = EXPECTED_COLUMNS.get('pnl')
        if not pnl_col or pnl_col not in trades_df.columns:
            self.logger.error(f"generate_pnl_distribution_plot: PnL column '{pnl_col}' not found.")
            return None
        if trades_df[pnl_col].dropna().empty:
            self.logger.warning("generate_pnl_distribution_plot: PnL column is empty after dropping NaNs.")
            return None 
        try:
            return plot_pnl_distribution(trades_df, pnl_col=pnl_col, title="PnL Distribution (per Trade)", theme=theme)
        except Exception as e:
            self.logger.error(f"Error generating PnL dist plot: {e}", exc_info=True)
            return {"error": str(e)}
