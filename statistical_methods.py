"""
statistical_methods.py

Implements advanced statistical methods for trading performance analysis,
including hypothesis testing, bootstrapping, distribution fitting,
time series decomposition, and change point detection.
Corrected for ruptures library API changes/usage.
"""
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose, DecomposeResult
import ruptures as rpt # Ensure this import is present
from typing import List, Tuple, Dict, Any, Optional, Union, Callable

try:
    from config import BOOTSTRAP_ITERATIONS, CONFIDENCE_LEVEL, DISTRIBUTIONS_TO_FIT, APP_TITLE
except ImportError:
    APP_TITLE = "TradingDashboard_Default_Stats"
    BOOTSTRAP_ITERATIONS = 1000
    CONFIDENCE_LEVEL = 0.95
    DISTRIBUTIONS_TO_FIT = ['norm', 't']

import logging
logger = logging.getLogger(APP_TITLE)


@st.cache_data(show_spinner="Performing hypothesis test...", ttl=3600)
def perform_hypothesis_test(
    data1: Union[List[float], pd.Series, np.ndarray, pd.DataFrame],
    data2: Optional[Union[List[float], pd.Series]] = None,
    test_type: str = 't-test_ind', alpha: float = 0.05, **kwargs
) -> Dict[str, Any]:
    results: Dict[str, Any] = {"test_type": test_type, "alpha": alpha}
    if test_type != 'chi-squared':
        if isinstance(data1, list) and test_type == 'anova': pass 
        else: data1 = pd.Series(data1).dropna()
    if data2 is not None: data2 = pd.Series(data2).dropna()
    
    try:
        if test_type == 't-test_ind':
            if data2 is None or len(data1) < 2 or len(data2) < 2: return {"error": "Insufficient data for independent t-test."}
            stat, p_value = stats.ttest_ind(data1, data2, equal_var=kwargs.get('equal_var', False), nan_policy='omit')
        elif test_type == 't-test_rel':
            if data2 is None or len(data1) != len(data2) or len(data1) < 2: return {"error": "Data for paired t-test must be of equal length and sufficient size."}
            stat, p_value = stats.ttest_rel(data1, data2, nan_policy='omit')
        elif test_type == 'anova':
            if not isinstance(data1, list) or len(data1) < 2: return {"error": "ANOVA requires a list of at least two groups."}
            valid_groups = [pd.Series(g).dropna() for g in data1 if len(pd.Series(g).dropna()) >=2]
            if len(valid_groups) < 2: return {"error": "ANOVA requires at least two valid groups (min 2 observations each) after NaN removal."}
            stat, p_value = stats.f_oneway(*valid_groups)
        elif test_type == 'chi-squared':
            if not isinstance(data1, (np.ndarray, pd.DataFrame)) or pd.DataFrame(data1).ndim != 2: 
                return {"error": "Chi-squared test requires a 2D contingency table as input."}
            chi2_stat, p_value, dof, expected_freq = stats.chi2_contingency(pd.DataFrame(data1))
            stat = chi2_stat; results['df'] = dof; results['expected_frequencies'] = expected_freq.tolist()
        else: return {"error": f"Unsupported test type: {test_type}"}
        
        results['statistic'] = stat; results['p_value'] = p_value
        results['significant'] = p_value < alpha
        results['interpretation'] = f"Result is {'statistically significant' if results['significant'] else 'not statistically significant'} at alpha = {alpha} (p-value: {p_value:.4f})."
        results['conclusion'] = "Reject null hypothesis." if results['significant'] else "Fail to reject null hypothesis."
    except Exception as e: 
        logger.error(f"Error in hypothesis test '{test_type}': {e}", exc_info=True)
        results['error'] = str(e)
    return results

@st.cache_data(show_spinner="Performing bootstrapping for CIs...", ttl=3600)
def bootstrap_confidence_interval(
    data: Union[List[float], pd.Series],
    _statistic_func: Callable[[pd.Series], float], 
    n_iterations: int = BOOTSTRAP_ITERATIONS,
    confidence_level: float = CONFIDENCE_LEVEL
) -> Dict[str, Any]:
    data_series = pd.Series(data).dropna()
    if len(data_series) < 2:
        logger.warning("Bootstrapping CI: Not enough data points (need at least 2).")
        observed_stat_val = np.nan
        if not data_series.empty:
            try: observed_stat_val = _statistic_func(data_series)
            except: pass 
        return {"lower_bound": np.nan, "upper_bound": np.nan, "observed_statistic": observed_stat_val, "bootstrap_statistics": [], "error": "Insufficient data for bootstrapping."}

    bootstrap_statistics = np.empty(n_iterations)
    n_size = len(data_series)
    data_values = data_series.values 

    for i in range(n_iterations):
        resample_values = np.random.choice(data_values, size=n_size, replace=True)
        bootstrap_statistics[i] = _statistic_func(pd.Series(resample_values))

    observed_statistic = _statistic_func(data_series)
    alpha_percentile = (1 - confidence_level) / 2 * 100
    
    valid_bootstrap_stats = bootstrap_statistics[~np.isnan(bootstrap_statistics)]
    if len(valid_bootstrap_stats) < n_iterations * 0.1: 
        logger.warning(f"Bootstrapping for {_statistic_func.__name__ if hasattr(_statistic_func, '__name__') else 'custom_stat'} resulted in many NaNs ({len(bootstrap_statistics) - len(valid_bootstrap_stats)} NaNs out of {n_iterations}).")
        return {"lower_bound": np.nan, "upper_bound": np.nan, "observed_statistic": observed_statistic, "bootstrap_statistics": bootstrap_statistics.tolist(), "error": "Many NaNs in bootstrap samples, CI unreliable."}
    if len(valid_bootstrap_stats) == 0: 
        return {"lower_bound": np.nan, "upper_bound": np.nan, "observed_statistic": observed_statistic, "bootstrap_statistics": bootstrap_statistics.tolist(), "error": "No valid bootstrap samples generated."}

    lower_bound = np.percentile(valid_bootstrap_stats, alpha_percentile)
    upper_bound = np.percentile(valid_bootstrap_stats, 100 - alpha_percentile)

    return {
        "lower_bound": lower_bound, "upper_bound": upper_bound, 
        "observed_statistic": observed_statistic, "bootstrap_statistics": bootstrap_statistics.tolist()
    }

@st.cache_data(show_spinner="Fitting distributions to PnL data...", ttl=3600)
def fit_distributions_to_pnl(pnl_series: pd.Series, distributions_to_try: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
    if distributions_to_try is None: distributions_to_try = DISTRIBUTIONS_TO_FIT
    pnl_clean = pnl_series.dropna()
    if pnl_clean.empty: return {"error": "PnL series is empty after NaN removal."}
    results = {}
    for dist_name in distributions_to_try:
        try:
            dist = getattr(stats, dist_name)
            params = dist.fit(pnl_clean)
            D, p_value = stats.kstest(pnl_clean, dist_name, args=params, N=len(pnl_clean))
            # Determine param_names more robustly
            param_names_list = []
            if dist.shapes:
                param_names_list.extend(dist.shapes.split(','))
            param_names_list.extend(['loc', 'scale'])
            
            results[dist_name] = {
                "params": params, "param_names": param_names_list,
                "ks_statistic": D, "ks_p_value": p_value,
                "interpretation": f"KS p-value ({p_value:.4f}) suggests data {'may come' if p_value > 0.05 else 'likely does not come'} from {dist_name.capitalize()}."
            }
        except Exception as e: 
            logger.error(f"Error fitting {dist_name}: {e}", exc_info=True)
            results[dist_name] = {"error": str(e)}
    return results

@st.cache_data(show_spinner="Decomposing time series...", ttl=3600)
def decompose_time_series(
    series: pd.Series, model: str = 'additive', period: Optional[int] = None, 
    extrapolate_trend: str = 'freq'
) -> Optional[DecomposeResult]:
    series_clean = series.dropna()
    if not isinstance(series_clean.index, pd.DatetimeIndex):
        try: series_clean.index = pd.to_datetime(series_clean.index)
        except Exception as e: logger.error(f"TS Decomp: Failed to convert index to DatetimeIndex: {e}"); return None
    
    min_len_required = (2 * (period if period is not None and period > 1 else 2))
    if series_clean.empty or len(series_clean) < min_len_required:
        logger.warning(f"TS Decomp: Not enough data (need at least {min_len_required} points for period {period}, got {len(series_clean)}).")
        return None

    if model.lower() == 'multiplicative':
        if not (series_clean > 1e-8).all(): # Check for strictly positive, allowing for tiny positive values
             # Attempt to shift data if it contains zeros or negatives for multiplicative model
            min_val = series_clean.min()
            if min_val <= 1e-8:
                shift = abs(min_val) + 1e-6 # Add a small constant to make all values positive
                series_clean = series_clean + shift
                logger.warning(f"TS Decomp: Multiplicative model requires positive values. Series shifted by {shift:.2e}.")
                if not (series_clean > 1e-8).all(): # Check again after shift
                    logger.error("TS Decomp: Multiplicative model failed even after attempting to shift data to be positive.")
                    raise ValueError("Multiplicative decomposition requires all series values to be strictly positive, even after shift attempt.")


    if series_clean.index.freq is None and period is None:
        inferred_freq = pd.infer_freq(series_clean.index)
        if inferred_freq: series_clean = series_clean.asfreq(inferred_freq)
        else:
            if (series_clean.index.to_series().diff().dt.days == 1).mean() > 0.5:
                try:
                    series_daily_resampled = series_clean.resample('D').mean()
                    if not series_daily_resampled.isnull().all():
                        series_clean = series_daily_resampled.interpolate(method='linear')
                    else: return None
                except Exception as e: logger.error(f"TS Decomp: Error resampling to daily: {e}"); return None
            else: logger.warning("TS Decomp: Could not infer frequency. Decomposition might be unreliable without a specified period.")
    
    try:
        if period is None and series_clean.index.freq is None:
            logger.error("TS Decomp: Cannot perform decomposition without a frequency or an explicit period.")
            return None
        decomposition = seasonal_decompose(series_clean, model=model, period=period, extrapolate_trend=extrapolate_trend)
        return decomposition
    except ValueError as ve: logger.error(f"TS Decomp ValueError: {ve}", exc_info=True); return None
    except Exception as e: logger.error(f"TS Decomp general error: {e}", exc_info=True); return None

@st.cache_data(show_spinner="Detecting change points...", ttl=3600)
def detect_change_points(
    series: pd.Series, 
    model: str = "l2", 
    penalty: Optional[Union[str, float]] = "bic",
    n_bkps: Optional[int] = None, 
    min_size: int = 2
) -> Dict[str, Any]:
    """
    Detects change points in a time series using the ruptures library.

    Switches between rpt.Dynp (for fixed n_bkps) and rpt.Pelt (for penalty-based detection).
    Handles numerical penalty calculation for Pelt based on common interpretations of
    "bic" and "aic" string penalties.

    Args:
        series (pd.Series): The time series data (NaNs will be dropped).
        model (str): The model for change point detection (e.g., "l2", "rbf").
        penalty (Optional[Union[str, float]]): Penalty criterion or custom float value.
            Used only if n_bkps is None. Can be "bic", "aic", "mbic", or a float.
        n_bkps (Optional[int]): The fixed number of breakpoints to detect. If None,
            penalty-based detection is used.
        min_size (int): Minimum segment length.

    Returns:
        Dict[str, Any]: A dictionary containing detected breakpoints and other info,
                        or an error message.
    """
    series_values = series.dropna().values
    n_samples = len(series_values)
    results: Dict[str, Any] = {}

    required_samples = (n_bkps + 1) * min_size if n_bkps is not None else 2 * min_size
    if n_samples < required_samples:
        logger.warning(f"Change Point Detection: Insufficient data (need >= {required_samples}, have {n_samples}).")
        return {"error": f"Insufficient data (need at least {required_samples} points for current settings). Have {n_samples}."}
    
    try:
        if n_bkps is not None: 
            # Use Dynp for a fixed number of breakpoints
            algo = rpt.Dynp(model=model, min_size=min_size, jump=1).fit(series_values)
            # predict method for Dynp takes n_bkps
            breakpoints_algo_indices = algo.predict(n_bkps=n_bkps)
            results['method_used'] = f"Dynp (fixed {n_bkps} breakpoints)"
        else: 
            # Use Pelt for penalty-based automatic detection
            algo = rpt.Pelt(model=model, min_size=min_size).fit(series_values)
            pen_value: Optional[float] = None

            if isinstance(penalty, (float, int)): 
                pen_value = float(penalty)
            elif isinstance(penalty, str):
                penalty_str = penalty.lower()
                if penalty_str == "bic":
                    # A common numerical penalty value that approximates BIC behavior for Pelt.
                    # The multiplier (e.g., 1.0 or 3.0) can be model-dependent or tuned.
                    # sigma_sq_estimate = np.var(np.diff(series_values)) if n_samples > 1 else 1.0
                    # pen_value = 1.0 * np.log(n_samples) * sigma_sq_estimate # More complex
                    pen_value = 1.5 * np.log(n_samples) # Simplified common value, adjust factor as needed
                elif penalty_str == "aic":
                    # A common numerical penalty value that approximates AIC behavior.
                    pen_value = 2.0 # Simplified (2 * number_of_model_parameters, assuming 1 parameter)
                elif penalty_str == "mbic":
                    logger.warning("MBIC penalty string chosen for Pelt; using a BIC-like numerical penalty as an approximation.")
                    pen_value = 1.5 * np.log(n_samples) 
                else: # Unrecognized penalty string
                    logger.warning(f"Unrecognized penalty string '{penalty}'. Using a default numerical penalty for Pelt.")
                    pen_value = np.log(n_samples) 
            else: # Penalty is None or unexpected type
                 logger.warning(f"Invalid penalty type '{type(penalty)}'. Using default numerical penalty for Pelt.")
                 pen_value = np.log(n_samples) 

            if pen_value is None: # Should ideally not be reached with the logic above
                 logger.error("Internal error: Penalty value for Pelt could not be determined.")
                 return {"error": "Internal error: Penalty value for Pelt could not be determined."}

            breakpoints_algo_indices = algo.predict(pen=pen_value)
            results['method_used'] = f"Pelt (penalty: {pen_value:.2f})"

        # `breakpoints_algo_indices` from ruptures usually includes n_samples as the last element.
        # These are end-of-segment indices. Change points are effectively these indices.
        # e.g., if [50, 100] for 100 samples, change is at index 50.
        # We want the actual locations of change, so we exclude the last one if it's n_samples.
        actual_change_point_locations_in_values = [bkp for bkp in breakpoints_algo_indices if bkp < n_samples]

        # Map these integer indices (from the `series_values` numpy array) back to the original series's index
        actual_change_points_original_indices = []
        if isinstance(series.index, pd.DatetimeIndex):
            actual_change_points_original_indices = [series.index[idx] for idx in actual_change_point_locations_in_values]
        else: 
            # For non-datetime index, map back if original index was not simple range
            # If series.index is RangeIndex(start=0, stop=N, step=1), then actual_change_point_locations_in_values are the direct indices.
            # If series.index has its own custom integer values, we need to map.
            # This assumes series.index allows direct integer indexing or iloc.
            try:
                actual_change_points_original_indices = [series.index[idx] for idx in actual_change_point_locations_in_values]
            except IndexError: # Fallback if direct indexing fails (e.g. non-standard int index)
                actual_change_points_original_indices = actual_change_point_locations_in_values
                logger.warning("Could not map change point indices to original series index directly; using numerical indices.")


        results.update({
            'breakpoints_algo_indices': breakpoints_algo_indices, 
            'change_points_original_indices': actual_change_points_original_indices, 
            'series_to_plot': series 
        })

    except Exception as e: 
        method_info = results.get('method_used', 'Unknown method')
        logger.error(f"Change point detection error ({method_info}): {e}", exc_info=True)
        results['error'] = f"Error during {method_info}: {str(e)}"
    return results
