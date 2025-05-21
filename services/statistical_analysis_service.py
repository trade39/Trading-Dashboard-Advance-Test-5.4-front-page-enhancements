# services/statistical_analysis_service.py
"""
This service handles more generic statistical analyses, often acting as a
wrapper around functions in the statistical_methods.py module.
Now includes wrappers for hypothesis testing and bootstrapping.
"""
import pandas as pd
import numpy as np # Added for type hints and potential NaN handling
from typing import Dict, Any, Optional, List, Callable, Union # Added Union

try:
    from config import APP_TITLE, CONFIDENCE_LEVEL, BOOTSTRAP_ITERATIONS
    from statistical_methods import (
        bootstrap_confidence_interval as sm_bootstrap_ci, # Renamed for clarity
        fit_distributions_to_pnl,
        decompose_time_series,
        detect_change_points,
        perform_hypothesis_test as sm_perform_hypothesis_test # Renamed for clarity
    )
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR in StatisticalAnalysisService module: {e}. Some functionalities may fail.")
    APP_TITLE = "TradingDashboard_ErrorState"
    CONFIDENCE_LEVEL = 0.95
    BOOTSTRAP_ITERATIONS = 1000
    # Fallback dummy functions if imports fail
    def sm_bootstrap_ci(*args, **kwargs): return {"error": "bootstrap_confidence_interval not loaded", "lower_bound": np.nan, "upper_bound": np.nan, "observed_statistic": np.nan, "bootstrap_statistics": []}
    def fit_distributions_to_pnl(*args, **kwargs): return {"error": "fit_distributions_to_pnl not loaded"}
    def decompose_time_series(*args, **kwargs): return None 
    def detect_change_points(*args, **kwargs): return {"error": "detect_change_points not loaded"}
    def sm_perform_hypothesis_test(*args, **kwargs): return {"error": "perform_hypothesis_test not loaded"}


import logging
# logger = logging.getLogger(APP_TITLE) # Logger will be initialized in the class

# Define Minimum Data Points Thresholds specific to this service's methods
MIN_DATA_FOR_BOOTSTRAP_CI_GENERIC = 10
MIN_DATA_FOR_DIST_FIT = 20
MIN_DATA_FOR_TS_DECOMPOSITION = 20 # This is a general threshold, specific methods might override
MIN_DATA_FOR_CHANGE_POINT = 15
MIN_DATA_FOR_HYPOTHESIS_TEST = 5 # General minimum, specific tests might need more


class StatisticalAnalysisService:
    def __init__(self):
        self.logger = logging.getLogger(APP_TITLE) # Instance logger
        self.logger.info("StatisticalAnalysisService initialized.")

    def get_time_series_decomposition(self, series: pd.Series, model: str = 'additive', period: Optional[int] = None) -> Dict[str, Any]:
        """
        Performs time series decomposition.
        """
        if series is None or series.dropna().empty:
            self.logger.warning("get_time_series_decomposition: Input series is empty or all NaN.")
            return {"error": "Input series for decomposition is empty or all NaN."}
        
        # Use a dynamic min_len_required based on the period for decomposition
        min_len_required_decomp = (2 * (period if period is not None and period > 1 else 2)) +1 # statsmodels needs 2*period
        if len(series.dropna()) < min_len_required_decomp:
            self.logger.warning(f"get_time_series_decomposition: Series too short (need >= {min_len_required_decomp} non-NaN for period {period}). Found {len(series.dropna())}.")
            return {"error": f"Series too short for decomposition (need at least {min_len_required_decomp} non-NaN points for period {period})."}

        try:
            result = decompose_time_series(series.dropna(), model=model, period=period)
            if result is not None:
                if (hasattr(result, 'observed') and result.observed.isnull().all() and
                    hasattr(result, 'trend') and result.trend.isnull().all() and
                    hasattr(result, 'seasonal') and result.seasonal.isnull().all() and
                    hasattr(result, 'resid') and result.resid.isnull().all()):
                    self.logger.warning(f"TS Decomp for series (len {len(series.dropna())}, model {model}, period {period}) resulted in all NaN components.")
                    return {"error": "Decomposition resulted in all NaN components. Series might be unsuitable."}
                return {"decomposition_result": result}
            else:
                self.logger.warning(f"Decomposition returned None for series (len {len(series.dropna())}, model {model}, period {period}). This might be due to an invalid period or very short series.")
                return {"error": "Decomposition failed. The series might be unsuitable for the chosen model/period or too short for the specified period."}
        except ValueError as ve: 
            self.logger.error(f"ValueError in TS decomp service call: {ve}", exc_info=False) 
            return {"error": str(ve)}
        except Exception as e:
            self.logger.error(f"Unexpected error in TS decomp service call: {e}", exc_info=True)
            return {"error": f"An unexpected error occurred during decomposition: {str(e)}"}

    def calculate_bootstrap_ci(
        self,
        data_series: pd.Series,
        statistic_func: Callable[[pd.Series], float],
        n_iterations: int = BOOTSTRAP_ITERATIONS,
        confidence_level: float = CONFIDENCE_LEVEL
    ) -> Dict[str, Any]:
        """
        Performs bootstrapping for a single statistic and returns data.
        Wrapper around statistical_methods.bootstrap_confidence_interval.
        """
        if data_series is None or data_series.dropna().empty:
            self.logger.warning("calculate_bootstrap_ci: Input data series is empty or all NaN.")
            return {"error": "Input data series for bootstrapping is empty or all NaN."}
        if len(data_series.dropna()) < MIN_DATA_FOR_BOOTSTRAP_CI_GENERIC:
            self.logger.warning(f"calculate_bootstrap_ci: Insufficient data (need >= {MIN_DATA_FOR_BOOTSTRAP_CI_GENERIC}). Found {len(data_series.dropna())}.")
            return {"error": f"Insufficient data points (need at least {MIN_DATA_FOR_BOOTSTRAP_CI_GENERIC}) for bootstrapping."}

        try:
            # Call the renamed function from statistical_methods
            results = sm_bootstrap_ci( 
                data=data_series.dropna(),
                _statistic_func=statistic_func, 
                n_iterations=n_iterations,
                confidence_level=confidence_level
            )
            return results 
        except Exception as e:
            self.logger.error(f"Error in calculate_bootstrap_ci service method: {e}", exc_info=True)
            return {"error": str(e)}

    def run_hypothesis_test(
        self,
        data1: Union[List[float], pd.Series, np.ndarray, pd.DataFrame], # Added pd.DataFrame for chi-squared
        data2: Optional[Union[List[float], pd.Series]] = None,
        test_type: str = 't-test_ind', 
        alpha: float = 0.05, 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Performs a specified hypothesis test.
        Wrapper around statistical_methods.perform_hypothesis_test.
        """
        # Basic data check for non-chi-squared tests
        if test_type != 'chi-squared':
            if isinstance(data1, list) and test_type == 'anova':
                if not all(len(pd.Series(g).dropna()) >= MIN_DATA_FOR_HYPOTHESIS_TEST for g in data1 if isinstance(g, (list, pd.Series, np.ndarray))):
                     return {"error": f"ANOVA: At least one group has fewer than {MIN_DATA_FOR_HYPOTHESIS_TEST} valid observations."}
            elif isinstance(data1, (pd.Series, np.ndarray, list)):
                if len(pd.Series(data1).dropna()) < MIN_DATA_FOR_HYPOTHESIS_TEST:
                    return {"error": f"Data1 has fewer than {MIN_DATA_FOR_HYPOTHESIS_TEST} valid observations for {test_type}."}
            if data2 is not None and isinstance(data2, (pd.Series, np.ndarray, list)):
                 if len(pd.Series(data2).dropna()) < MIN_DATA_FOR_HYPOTHESIS_TEST:
                    return {"error": f"Data2 has fewer than {MIN_DATA_FOR_HYPOTHESIS_TEST} valid observations for {test_type}."}
        elif test_type == 'chi-squared':
            if not isinstance(data1, (np.ndarray, pd.DataFrame)) or pd.DataFrame(data1).ndim != 2:
                return {"error": "Chi-squared test requires a 2D contingency table as input."}
            if pd.DataFrame(data1).shape[0] < 2 or pd.DataFrame(data1).shape[1] < 2:
                 return {"error": "Chi-squared contingency table must have at least 2 rows and 2 columns."}


        try:
            # Call the renamed function from statistical_methods
            results = sm_perform_hypothesis_test(
                data1=data1, data2=data2, test_type=test_type, alpha=alpha, **kwargs
            )
            return results
        except Exception as e:
            self.logger.error(f"Error in run_hypothesis_test service method for '{test_type}': {e}", exc_info=True)
            return {"error": str(e)}


    def analyze_pnl_distribution_fit(self, pnl_series: pd.Series, distributions_to_try: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Fits various statistical distributions to the PnL series.
        """
        if pnl_series is None or pnl_series.dropna().empty:
            self.logger.warning("analyze_pnl_distribution_fit: PnL series is empty.")
            return {"error": "PnL series is empty for distribution fitting."}
        if len(pnl_series.dropna()) < MIN_DATA_FOR_DIST_FIT:
            self.logger.warning(f"analyze_pnl_distribution_fit: Insufficient PnL data (need >= {MIN_DATA_FOR_DIST_FIT}). Found {len(pnl_series.dropna())}.")
            return {"error": f"Insufficient PnL data (need at least {MIN_DATA_FOR_DIST_FIT} points) for distribution fitting."}
        try:
            return fit_distributions_to_pnl(pnl_series.dropna(), distributions_to_try=distributions_to_try)
        except Exception as e:
            self.logger.error(f"Error in PnL dist fit: {e}", exc_info=True)
            return {"error": str(e)}

    def find_change_points(self, series: pd.Series, model: str = "l2", penalty: str = "bic", n_bkps: Optional[int] = None, min_size: int = 2) -> Dict[str, Any]:
        """
        Detects change points in a time series.
        """
        if series is None or series.dropna().empty:
            self.logger.warning("find_change_points: Input series is empty.")
            return {"error": "Input series is empty for change point detection."}
        
        if len(series.dropna()) < MIN_DATA_FOR_CHANGE_POINT:
            self.logger.warning(f"find_change_points: Series too short (need >= {MIN_DATA_FOR_CHANGE_POINT}). Found {len(series.dropna())}.")
            return {"error": f"Series too short (need at least {MIN_DATA_FOR_CHANGE_POINT} points) for change point detection."}
        try:
            return detect_change_points(series.dropna(), model=model, penalty=penalty, n_bkps=n_bkps, min_size=min_size)
        except Exception as e:
            self.logger.error(f"Error in change point detect: {e}", exc_info=True)
            return {"error": str(e)}
