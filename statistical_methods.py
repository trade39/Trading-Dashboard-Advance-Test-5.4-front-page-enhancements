"""
statistical_methods.py

Implements advanced statistical methods for trading performance analysis,
including hypothesis testing, bootstrapping, distribution fitting,
time series decomposition, and change point detection.
Corrected for ruptures library API changes/usage and enhanced for robustness.
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
    # Attempt to import user-defined configurations
    from config import BOOTSTRAP_ITERATIONS, CONFIDENCE_LEVEL, DISTRIBUTIONS_TO_FIT, APP_TITLE
except ImportError:
    # Fallback to default configurations if import fails
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
    """
    Performs various hypothesis tests.

    Features:
    - Checks shape and length of input data.
    - Handles missing data by dropping NaNs.
    - Rejects insufficient or mismatched data with informative error messages.
    - Supports t-tests (independent and related), ANOVA, and Chi-squared tests.
    """
    results: Dict[str, Any] = {"test_type": test_type, "alpha": alpha}

    # Data preprocessing: Handle missing data and convert to Series
    if test_type != 'chi-squared': # Chi-squared expects a contingency table, not a Series
        if isinstance(data1, list) and test_type == 'anova': # ANOVA expects a list of groups
            pass
        else:
            data1 = pd.Series(data1).dropna()
    if data2 is not None:
        data2 = pd.Series(data2).dropna()
    
    try:
        # Perform the selected test
        if test_type == 't-test_ind':
            # Check for sufficient data for independent t-test
            if data2 is None or len(data1) < 2 or len(data2) < 2:
                return {"error": "Insufficient data for independent t-test. Both samples need at least 2 non-NaN values."}
            stat, p_value = stats.ttest_ind(data1, data2, equal_var=kwargs.get('equal_var', False), nan_policy='omit')
        elif test_type == 't-test_rel':
            # Check for matched pairs and sufficient data for paired t-test
            if data2 is None or len(data1) != len(data2):
                return {"error": "Data for paired t-test must be of equal length."}
            if len(data1) < 2:
                 return {"error": "Insufficient data for paired t-test. Paired samples need at least 2 non-NaN pairs."}
            stat, p_value = stats.ttest_rel(data1, data2, nan_policy='omit')
        elif test_type == 'anova':
            # ANOVA requires a list of at least two groups
            if not isinstance(data1, list) or len(data1) < 2:
                return {"error": "ANOVA requires a list of at least two groups as input."}
            # Ensure each group has sufficient data after NaN removal
            valid_groups = [pd.Series(g).dropna() for g in data1 if len(pd.Series(g).dropna()) >=2]
            if len(valid_groups) < 2:
                return {"error": "ANOVA requires at least two valid groups (min 2 observations each) after NaN removal."}
            stat, p_value = stats.f_oneway(*valid_groups)
        elif test_type == 'chi-squared':
            # Chi-squared test requires a 2D contingency table
            if not isinstance(data1, (np.ndarray, pd.DataFrame)) or pd.DataFrame(data1).ndim != 2:
                return {"error": "Chi-squared test requires a 2D contingency table (e.g., pandas DataFrame or NumPy array) as input."}
            # Ensure no empty or all-zero table
            if pd.DataFrame(data1).empty or pd.DataFrame(data1).sum().sum() == 0:
                return {"error": "Chi-squared test input table is empty or contains all zeros."}
            chi2_stat, p_value, dof, expected_freq = stats.chi2_contingency(pd.DataFrame(data1))
            stat = chi2_stat
            results['df'] = dof
            results['expected_frequencies'] = expected_freq.tolist()
        else:
            return {"error": f"Unsupported test type: {test_type}"}
        
        results['statistic'] = stat
        results['p_value'] = p_value
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
    """
    Calculates confidence intervals using bootstrapping.

    Features:
    - Handles insufficient data (requires at least 2 data points).
    - Handles NaNs in input data by dropping them.
    - Supports custom statistic functions.
    - Warns if a large proportion of bootstrap samples result in NaN.
    - Performance: Can be slow for very large n_iterations or data size, this is expected and user-tunable.
    """
    data_series = pd.Series(data).dropna()
    if len(data_series) < 2: # Handle insufficient data
        logger.warning("Bootstrapping CI: Not enough data points (need at least 2).")
        observed_stat_val = np.nan
        if not data_series.empty:
            try:
                observed_stat_val = _statistic_func(data_series)
            except Exception as e:
                logger.warning(f"Bootstrapping CI: Could not compute observed statistic on insufficient data: {e}")
        return {
            "lower_bound": np.nan, "upper_bound": np.nan,
            "observed_statistic": observed_stat_val,
            "bootstrap_statistics": [],
            "error": "Insufficient data for bootstrapping (need at least 2 non-NaN values)."
        }

    bootstrap_statistics = np.empty(n_iterations)
    n_size = len(data_series)
    data_values = data_series.values

    for i in range(n_iterations):
        resample_values = np.random.choice(data_values, size=n_size, replace=True)
        try:
            bootstrap_statistics[i] = _statistic_func(pd.Series(resample_values))
        except Exception as e: # Catch errors from custom statistic function
            logger.warning(f"Bootstrapping CI: Error in statistic_func for resample {i}: {e}")
            bootstrap_statistics[i] = np.nan


    try:
        observed_statistic = _statistic_func(data_series)
    except Exception as e:
        logger.error(f"Bootstrapping CI: Error computing observed statistic: {e}", exc_info=True)
        return {
            "lower_bound": np.nan, "upper_bound": np.nan,
            "observed_statistic": np.nan,
            "bootstrap_statistics": bootstrap_statistics.tolist(),
            "error": f"Error computing observed statistic: {str(e)}"
        }

    alpha_percentile = (1 - confidence_level) / 2 * 100
    
    valid_bootstrap_stats = bootstrap_statistics[~np.isnan(bootstrap_statistics)]
    if len(valid_bootstrap_stats) < n_iterations * 0.1: # Handle too many NaNs in bootstrap samples
        logger.warning(f"Bootstrapping for {_statistic_func.__name__ if hasattr(_statistic_func, '__name__') else 'custom_stat'} resulted in many NaNs ({len(bootstrap_statistics) - len(valid_bootstrap_stats)} NaNs out of {n_iterations}). CI may be unreliable.")
        return {
            "lower_bound": np.nan, "upper_bound": np.nan,
            "observed_statistic": observed_statistic,
            "bootstrap_statistics": bootstrap_statistics.tolist(),
            "error": "Many NaNs in bootstrap samples, confidence interval is unreliable."
        }
    if len(valid_bootstrap_stats) == 0: # Handle no valid bootstrap samples
        return {
            "lower_bound": np.nan, "upper_bound": np.nan,
            "observed_statistic": observed_statistic,
            "bootstrap_statistics": bootstrap_statistics.tolist(),
            "error": "No valid bootstrap samples were generated (all resulted in NaN)."
        }

    lower_bound = np.percentile(valid_bootstrap_stats, alpha_percentile)
    upper_bound = np.percentile(valid_bootstrap_stats, 100 - alpha_percentile)

    return {
        "lower_bound": lower_bound, "upper_bound": upper_bound,
        "observed_statistic": observed_statistic, "bootstrap_statistics": bootstrap_statistics.tolist()
    }

@st.cache_data(show_spinner="Fitting distributions to PnL data...", ttl=3600)
def fit_distributions_to_pnl(pnl_series: pd.Series, distributions_to_try: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
    """
    Fits various continuous distributions to Profit and Loss (PnL) data.

    Features:
    - Catches exceptions during the fitting process for each distribution.
    - Reports errors if fitting fails.
    - Uses Kolmogorov-Smirnov (KS) test for goodness of fit.
    - Note: Only robustly supports distributions where parameter names can be inferred via `dist.shapes`. Some SciPy distributions might behave unexpectedly.
    """
    if distributions_to_try is None:
        distributions_to_try = DISTRIBUTIONS_TO_FIT
    pnl_clean = pnl_series.dropna()
    if pnl_clean.empty:
        return {"error": "PnL series is empty after NaN removal."}
    if len(pnl_clean) < 2: # Need at least 2 points to fit most distributions
        return {"error": "PnL series has insufficient data points (<2) after NaN removal for distribution fitting."}

    results = {}
    for dist_name in distributions_to_try:
        try:
            dist = getattr(stats, dist_name)
            # Fit distribution to data
            params = dist.fit(pnl_clean)
            
            # Perform Kolmogorov-Smirnov test
            D, p_value = stats.kstest(pnl_clean, dist_name, args=params, N=len(pnl_clean))
            
            # Determine parameter names (can be quirky for some distributions)
            param_names_list = []
            if hasattr(dist, 'shapes') and dist.shapes: # Check if 'shapes' attribute exists and is not empty
                param_names_list.extend(dist.shapes.split(','))
            param_names_list.extend(['loc', 'scale']) # Standard for many continuous distributions
            
            # Ensure params tuple matches length of param_names_list, adjust if necessary
            # This is a heuristic as SciPy's fit doesn't always return fixed params like shape if they are not fitted
            actual_params = params
            if len(params) < len(param_names_list): # If fit returned fewer params (e.g., fixed shape)
                # This part is tricky as we don't know which params were fixed.
                # For simplicity, we'll report what `fit` returned and the standard names.
                # A more robust solution would involve inspecting dist.numargs or specific distribution properties.
                logger.debug(f"Distribution {dist_name}: `fit` returned {len(params)} params, expected up to {len(param_names_list)} names. Param names might not perfectly align.")


            results[dist_name] = {
                "params": actual_params,
                "param_names": param_names_list[:len(actual_params)], # Truncate names if fewer params returned
                "ks_statistic": D,
                "ks_p_value": p_value,
                "interpretation": f"KS p-value ({p_value:.4f}) suggests data {'may come' if p_value > 0.05 else 'likely does not come'} from a {dist_name.capitalize()} distribution."
            }
        except AttributeError:
            logger.error(f"Distribution '{dist_name}' not found in scipy.stats or does not support fitting.", exc_info=True)
            results[dist_name] = {"error": f"Distribution '{dist_name}' not found or not fittable."}
        except RuntimeError as e: # Specific catch for fitting errors like "maximum likelihood estimation failed"
            logger.error(f"Runtime error fitting {dist_name}: {e}", exc_info=True)
            results[dist_name] = {"error": f"Fitting failed for {dist_name}: {str(e)}"}
        except Exception as e: # General catch for other errors
            logger.error(f"Error fitting {dist_name}: {e}", exc_info=True)
            results[dist_name] = {"error": str(e)}
    return results

@st.cache_data(show_spinner="Decomposing time series...", ttl=3600)
def decompose_time_series(
    series: pd.Series, model: str = 'additive', period: Optional[int] = None,
    extrapolate_trend: str = 'freq'
) -> Optional[DecomposeResult]:
    """
    Decomposes a time series into trend, seasonal, and residual components.

    Features:
    - Handles missing frequency/period by attempting to infer or resample.
    - Handles non-positive data for multiplicative models by attempting a shift.
    - Implements careful resampling logic if frequency inference fails.
    - Requires a minimum number of data points based on the period.
    """
    series_clean = series.dropna()
    if not isinstance(series_clean.index, pd.DatetimeIndex):
        try:
            series_clean.index = pd.to_datetime(series_clean.index)
        except Exception as e:
            logger.error(f"TS Decomp: Failed to convert index to DatetimeIndex: {e}")
            return None # Cannot proceed without a DatetimeIndex
    
    # Determine minimum length required for decomposition
    # statsmodels requires at least 2 full periods of data
    min_len_required = (2 * (period if period is not None and period > 1 else 2))
    if series_clean.empty or len(series_clean) < min_len_required:
        logger.warning(f"TS Decomp: Not enough data (need at least {min_len_required} points for period {period}, have {len(series_clean)}).")
        return None

    # Handle non-positive data for multiplicative model
    if model.lower() == 'multiplicative':
        if not (series_clean > 1e-8).all(): # Check for strictly positive, allowing for tiny positive values
            min_val = series_clean.min()
            if min_val <= 1e-8: # Includes zero and negative values
                shift = abs(min_val) + 1e-6 # Add a small constant to make all values positive
                series_clean = series_clean + shift
                logger.warning(f"TS Decomp: Multiplicative model requires positive values. Series shifted by {shift:.2e} to make all values positive.")
                if not (series_clean > 1e-8).all(): # Check again after shift
                    logger.error("TS Decomp: Multiplicative model failed even after attempting to shift data to be positive.")
                    # Cannot proceed with multiplicative if data isn't positive
                    st.error("Multiplicative decomposition requires all series values to be strictly positive. Automatic shift failed.")
                    return None # Or raise ValueError("Multiplicative decomposition requires all series values to be strictly positive...")


    # Handle frequency and period
    if series_clean.index.freq is None and period is None:
        inferred_freq = pd.infer_freq(series_clean.index)
        if inferred_freq:
            series_clean = series_clean.asfreq(inferred_freq)
            logger.info(f"TS Decomp: Inferred frequency '{inferred_freq}'.")
        else:
            # Attempt to resample to daily if frequency cannot be inferred and index is DatetimeIndex
            # This is a heuristic and might not be appropriate for all data.
            logger.warning("TS Decomp: Could not infer frequency. Attempting to resample to daily ('D'). This may not be suitable for all data types.")
            if (series_clean.index.to_series().diff().dt.days == 1).mean() > 0.5 : # If mostly daily data
                 try:
                    series_daily_resampled = series_clean.resample('D').mean() # Use mean, or consider ffill/bfill
                    if not series_daily_resampled.isnull().all():
                        series_clean = series_daily_resampled.interpolate(method='linear') # Interpolate missing values
                        logger.info("TS Decomp: Resampled to daily frequency and interpolated.")
                    else:
                        logger.error("TS Decomp: Resampling to daily resulted in all NaNs.")
                        return None
                 except Exception as e:
                    logger.error(f"TS Decomp: Error resampling to daily: {e}")
                    return None
            else:
                 logger.warning("TS Decomp: Could not infer frequency and data does not appear to be daily. Decomposition might be unreliable or fail without a specified period.")
                 # If period is still None here, decomposition will likely fail or be meaningless.
                 # statsmodels will raise error if period is None and freq cannot be inferred.

    try:
        # If period is still None after attempts to infer/set frequency, statsmodels might raise error.
        # It's better to ensure period is sensible if freq is None.
        # However, seasonal_decompose itself tries to infer period from freq if period is None.
        if period is None and series_clean.index.freq is None:
            logger.error("TS Decomp: Cannot perform decomposition without a discernible frequency or an explicit period.")
            st.error("Time series decomposition requires a discernible frequency or an explicit period. Please check your data or provide a period.")
            return None

        decomposition = seasonal_decompose(series_clean, model=model, period=period, extrapolate_trend=extrapolate_trend)
        return decomposition
    except ValueError as ve:
        logger.error(f"TS Decomp ValueError: {ve}", exc_info=True)
        st.error(f"Time series decomposition failed: {ve}. Ensure data has sufficient length and, if no frequency is set, provide an appropriate 'period'.")
        return None
    except Exception as e:
        logger.error(f"TS Decomp general error: {e}", exc_info=True)
        st.error(f"An unexpected error occurred during time series decomposition: {e}")
        return None

@st.cache_data(show_spinner="Detecting change points...", ttl=3600)
def detect_change_points(
    series: pd.Series,
    model: str = "l2",
    penalty: Optional[Union[str, float]] = "bic", # Common penalties: "aic", "bic", "mbic" or a float
    n_bkps: Optional[int] = None, # Number of breakpoints to detect (if known)
    min_size: int = 2
) -> Dict[str, Any]:
    """
    Detects change points in a time series using the ruptures library.

    Features:
    - Properly distinguishes between fixed (Dynp) and penalty-based (Pelt) breakpoint detection.
    - Handles translation of common penalty strings ("aic", "bic", "mbic") to numerical values for Pelt.
    - Robustly maps detected integer indices back to the original series's index (including DatetimeIndex).
    - Warns and falls back if mapping a custom non-integer/non-datetime index fails.
    - Checks for sufficient data points based on min_size and n_bkps.
    """
    series_values = series.dropna().values
    n_samples = len(series_values)
    results: Dict[str, Any] = {}

    # Check for sufficient data
    required_samples = (n_bkps + 1) * min_size if n_bkps is not None else 2 * min_size
    if n_samples < required_samples:
        msg = f"Insufficient data for change point detection. Need at least {required_samples} non-NaN points for current settings (min_size={min_size}, n_bkps={n_bkps}), but have {n_samples}."
        logger.warning(msg)
        return {"error": msg}
    
    try:
        if n_bkps is not None:
            # Use Dynp for a fixed number of breakpoints
            algo = rpt.Dynp(model=model, min_size=min_size, jump=1).fit(series_values)
            breakpoints_algo_indices = algo.predict(n_bkps=n_bkps)
            results['method_used'] = f"Dynp (fixed {n_bkps} breakpoints)"
        else:
            # Use Pelt for penalty-based automatic detection
            algo = rpt.Pelt(model=model, min_size=min_size, jump=1).fit(series_values) # jump=1 for precision
            pen_value: Optional[float] = None

            if isinstance(penalty, (float, int)):
                pen_value = float(penalty)
            elif isinstance(penalty, str):
                penalty_str = penalty.lower()
                # Common penalty translations for Pelt. These are approximations.
                # Optimal penalty values can be data and model dependent.
                # sigma_sq_estimate might be used for more refined BIC/AIC, but adds complexity.
                # Example: sigma_sq_estimate = np.var(np.diff(series_values)) if n_samples > 1 else 1.0
                if penalty_str == "bic":
                    pen_value = 2 * np.log(n_samples) # Common simplified BIC for Pelt
                elif penalty_str == "aic":
                    pen_value = 2.0 # Simplified AIC (assumes 1 parameter per segment for cost)
                elif penalty_str == "mbic":
                    # MBIC is more complex; using a BIC-like approximation.
                    logger.warning("MBIC penalty string chosen for Pelt; using a BIC-like numerical penalty as an approximation.")
                    pen_value = 3 * np.log(n_samples) # Often a stronger penalty than BIC
                else:
                    logger.warning(f"Unrecognized penalty string '{penalty}'. Using a default numerical penalty (log(n_samples)) for Pelt.")
                    pen_value = np.log(n_samples) # Default if string is not recognized
            else: # Penalty is None or unexpected type
                 logger.warning(f"Invalid penalty type '{type(penalty)}'. Using default numerical penalty (log(n_samples)) for Pelt.")
                 pen_value = np.log(n_samples)

            if pen_value is None: # Should not be reached with the logic above
                 logger.error("Internal error: Penalty value for Pelt could not be determined.")
                 return {"error": "Internal error: Penalty value for Pelt could not be determined."}

            breakpoints_algo_indices = algo.predict(pen=pen_value)
            results['method_used'] = f"Pelt (penalty: {penalty_str if isinstance(penalty, str) else pen_value:.2f})"
            results['penalty_value_used'] = pen_value


        # `breakpoints_algo_indices` from ruptures are end-of-segment indices (exclusive).
        # The last element is typically n_samples.
        # Change points are the indices *before* the segment ends.
        # E.g., if [50, 100] for 100 samples, change is at index 49 (data[0]..data[49] is first segment).
        # To get the actual data point index that is the change point:
        actual_change_point_locations_in_values = [bkp -1 for bkp in breakpoints_algo_indices if bkp > 0 and bkp < n_samples]


        # Map these integer indices (from the `series_values` numpy array) back to the original series's index
        original_series_index = series.dropna().index # Use index of the NaN-dropped series
        actual_change_points_original_indices = []

        if not actual_change_point_locations_in_values: # No change points detected
            pass
        elif isinstance(original_series_index, pd.DatetimeIndex):
            actual_change_points_original_indices = [original_series_index[idx] for idx in actual_change_point_locations_in_values if idx < len(original_series_index)]
        elif pd.api.types.is_integer_dtype(original_series_index) or pd.api.types.is_float_dtype(original_series_index):
             # Handles numeric (int, float) custom indices
            try:
                actual_change_points_original_indices = [original_series_index[idx] for idx in actual_change_point_locations_in_values if idx < len(original_series_index)]
            except IndexError:
                logger.warning("Change Point Indexing: Could not map all change point indices to original numeric series index directly (possibly out of bounds after dropna). Using valid numerical indices.")
                actual_change_points_original_indices = [original_series_index[idx] for idx in actual_change_point_locations_in_values if idx < len(original_series_index)]
            except TypeError: # If index is float and cannot be used for direct list-like indexing
                logger.warning("Change Point Indexing: Original index is float and cannot be used for direct positional lookup. Falling back to iloc.")
                actual_change_points_original_indices = [original_series_index.values[idx] for idx in actual_change_point_locations_in_values if idx < len(original_series_index)]

        else: # For other index types (e.g., string/object) or if mapping is complex
            logger.warning(f"Change Point Indexing: Original series index is of type '{original_series_index.dtype}'. Attempting to map indices. If this fails or is incorrect, numerical indices will be used as a fallback.")
            try:
                 # This assumes the integer locations map directly to positions in the original_series_index
                actual_change_points_original_indices = [original_series_index[idx] for idx in actual_change_point_locations_in_values if idx < len(original_series_index)]
            except (IndexError, TypeError) as map_err:
                logger.error(f"Change Point Indexing: Failed to map to original index type '{original_series_index.dtype}': {map_err}. Falling back to numerical indices relative to the start of the (dropped NaN) series.")
                actual_change_points_original_indices = actual_change_point_locations_in_values # Fallback to numerical indices if mapping fails


        results.update({
            'breakpoints_algo_indices': breakpoints_algo_indices, # Raw indices from ruptures (end of segment, includes n_samples)
            'change_points_numerical_indices': actual_change_point_locations_in_values, # 0-based indices into the `series_values` array
            'change_points_original_indices': actual_change_points_original_indices, # Mapped to original series index
            'series_to_plot': series.dropna() # Plot the same data used for detection
        })

    except rpt.exceptions.NotEnoughPoints:
        msg = "Change point detection error: Not enough data points for the chosen model and min_size after NaN removal."
        logger.error(msg, exc_info=True)
        results['error'] = msg
    except Exception as e:
        method_info = results.get('method_used', 'Unknown method')
        logger.error(f"Change point detection error ({method_info}): {e}", exc_info=True)
        results['error'] = f"Error during {method_info}: {str(e)}"
    return results
