# ai_models.py
"""
This module contains functions for training, prediction, forecasting,
and other AI/ML model operations.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import logging

# --- Attempt to import necessary libraries and set availability flags ---
try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.preprocessing import LabelEncoder # For target encoding
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    RandomForestClassifier = None 
    accuracy_score = None
    classification_report = None
    train_test_split = None
    LabelEncoder = None

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    import warnings
    STATSMODELS_ARIMA_AVAILABLE = True
except ImportError:
    STATSMODELS_ARIMA_AVAILABLE = False
    ARIMA = None

try:
    from pmdarima import auto_arima
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False
    auto_arima = None

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    Prophet = None

try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False
    KaplanMeierFitter = None
    CoxPHFitter = None

try:
    from sklearn.ensemble import IsolationForest
    if SKLEARN_AVAILABLE:
        ISOLATION_FOREST_AVAILABLE = True
    else:
        ISOLATION_FOREST_AVAILABLE = False
except ImportError:
    ISOLATION_FOREST_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Classification Models ---
def train_simple_rf_model(
    df: pd.DataFrame,
    features: List[str],
    target: str, # Target column name
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    class_weight: Optional[Union[Dict, str]] = 'balanced' # Added class_weight
) -> Tuple[Optional[Any], Optional[float], Optional[Dict[str, Any]], Optional[pd.Series]]:
    """
    Trains a simple RandomForestClassifier.
    Returns the trained model, accuracy score, classification report, and feature importances.
    """
    if not SKLEARN_AVAILABLE:
        logger.error("train_simple_rf_model: scikit-learn is not available.")
        return None, None, None, None
    try:
        X = df[features].copy() # Work on a copy
        y_raw = df[target].copy()

        # Preprocessing: Handle potential categorical features with one-hot encoding
        # More robust preprocessing might be needed for mixed-type features
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        if not categorical_cols.empty:
            logger.info(f"One-hot encoding categorical features: {list(categorical_cols)}")
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dummy_na=False)
            # Ensure all feature names are strings (get_dummies can create non-string column names)
            X.columns = X.columns.astype(str)
            # Update features list to reflect new one-hot encoded columns
            # This is tricky as the original 'features' list won't match directly.
            # For simplicity, we'll use all columns from X after get_dummies.
            # A more robust solution would track original features to their encoded versions.
            processed_features = X.columns.tolist() 
        else:
            processed_features = features


        # Ensure target is numerically encoded if it's not already (e.g., True/False, 'Win'/'Loss')
        if not pd.api.types.is_numeric_dtype(y_raw):
            logger.info(f"Target column '{target}' is not numeric. Applying LabelEncoder.")
            le = LabelEncoder()
            y = le.fit_transform(y_raw)
            logger.info(f"Target classes after encoding: {le.classes_}")
            if len(le.classes_) < 2:
                logger.error(f"Target column '{target}' has less than 2 unique classes after encoding. Classification requires at least 2 classes.")
                return None, None, None, None
        else:
            y = y_raw

        if X.empty or len(y) == 0 or len(X) != len(y):
            logger.error("Feature set X or target y is empty or mismatched after preprocessing.")
            return None, None, None, None
        
        # Handle potential NaNs introduced by processing or in original data
        # For simplicity, we'll drop rows with any NaNs in features or target.
        # A more sophisticated approach would involve imputation.
        X_df_temp = X.assign(target_for_dropna=y)
        X_df_temp.dropna(inplace=True)
        if X_df_temp.empty:
            logger.error("No data remaining after dropping NaNs from features/target.")
            return None, None, None, None
        
        X_clean = X_df_temp[processed_features]
        y_clean = X_df_temp['target_for_dropna']


        if len(np.unique(y_clean)) < 2:
             logger.error(f"Target column '{target}' has less than 2 unique classes after cleaning. Classification requires at least 2 classes.")
             return None, None, None, None


        X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=test_size, random_state=random_state, stratify=y_clean if len(np.unique(y_clean)) > 1 else None)

        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=random_state,
            class_weight=class_weight # Handle imbalanced classes
        )
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
        
        feature_importances = pd.Series(model.feature_importances_, index=X_clean.columns).sort_values(ascending=False)
        
        logger.info(f"RandomForest model trained. Accuracy: {accuracy:.4f}. Features used for importances: {X_clean.columns.tolist()}")
        return model, accuracy, report, feature_importances
    except Exception as e:
        logger.error(f"Error training RandomForest model: {e}", exc_info=True)
        return None, None, None, None

# ... (predict_with_model, forecast_arima, forecast_prophet, survival_analysis_kaplan_meier, survival_analysis_cox_ph, detect_anomalies functions remain as before) ...

def predict_with_model(
    model: Any,
    new_data_df: pd.DataFrame,
    original_features: List[str] # Original features before potential one-hot encoding
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if model is None: logger.error("predict_with_model: No model provided."); return None, None
    if not SKLEARN_AVAILABLE: logger.error("predict_with_model: scikit-learn not available."); return None, None
    try:
        X_new = new_data_df[original_features].copy()
        
        # Apply the same preprocessing as during training (e.g., one-hot encoding)
        # This part needs to be robust and use the same columns/categories seen during training.
        # For simplicity, assuming the model object or associated preprocessor handles this.
        # If not, you'd need to store and re-apply the encoding steps (e.g., using scikit-learn Pipeline or ColumnTransformer)
        
        # A common issue: if new_data_df has different categorical levels than training data,
        # pd.get_dummies might produce different columns.
        # One way to handle this is to ensure `X_new` has the same columns as the training `X_clean`
        # This is a complex part of ML deployment. For now, we assume `model.feature_names_in_` exists if trained with sklearn >= 1.0
        
        if hasattr(model, 'feature_names_in_'):
            categorical_cols_new = X_new.select_dtypes(include=['object', 'category']).columns
            if not categorical_cols_new.empty:
                X_new = pd.get_dummies(X_new, columns=categorical_cols_new, drop_first=True, dummy_na=False)
                X_new.columns = X_new.columns.astype(str)

            # Reindex to match training feature set
            X_new = X_new.reindex(columns=model.feature_names_in_, fill_value=0)
        else: # Fallback if feature_names_in_ is not available (older sklearn or different model type)
            logger.warning("Model does not have 'feature_names_in_'. Prediction might fail if feature sets differ significantly from training.")
            # Basic one-hot encoding attempt, might not be robust
            categorical_cols_new = X_new.select_dtypes(include=['object', 'category']).columns
            if not categorical_cols_new.empty:
                X_new = pd.get_dummies(X_new, columns=categorical_cols_new, drop_first=True, dummy_na=False)
                X_new.columns = X_new.columns.astype(str)


        predictions = model.predict(X_new)
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_new)
        return predictions, probabilities
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        return None, None

# --- Forecasting Models ---
def forecast_arima(
    series: pd.Series,
    order: Optional[Tuple[int, int, int]] = None,
    seasonal_order: Optional[Tuple[int, int, int, int]] = None,
    n_periods: int = 30
) -> Dict[str, Any]:
# ... (existing forecast_arima code) ...
    if not STATSMODELS_ARIMA_AVAILABLE and order is not None: # Check for manual ARIMA
        return {"error": "statsmodels (for Manual ARIMA) is not installed."}
    if not PMDARIMA_AVAILABLE and order is None: # Check for auto ARIMA
        return {"error": "pmdarima (for Auto ARIMA) is not installed."}
    
    result: Dict[str, Any] = {}
    try:
        series = series.asfreq('D') 
        if order is None: 
            if not PMDARIMA_AVAILABLE: return {"error": "pmdarima not available for Auto ARIMA."}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                auto_model = auto_arima(series.dropna(), start_p=1, start_q=1, max_p=3, max_q=3,
                                        start_P=0, start_Q=0, max_P=2, max_Q=2, m=7,
                                        seasonal=True, stepwise=True, suppress_warnings=True, 
                                        error_action='ignore', trace=False)
            result['model_summary'] = str(auto_model.summary())
            forecast_values, conf_int = auto_model.predict(n_periods=n_periods, return_conf_int=True)
            result['forecast'] = forecast_values
            result['conf_int_lower'] = pd.Series(conf_int[:, 0], index=forecast_values.index)
            result['conf_int_upper'] = pd.Series(conf_int[:, 1], index=forecast_values.index)
        else: 
            if not STATSMODELS_ARIMA_AVAILABLE: return {"error": "statsmodels not available for Manual ARIMA."}
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', ConvergenceWarning)
                warnings.simplefilter('ignore', UserWarning)
                model = ARIMA(series.dropna(), order=order, seasonal_order=seasonal_order)
                model_fit = model.fit()
            result['model_summary'] = str(model_fit.summary())
            forecast_obj = model_fit.get_forecast(steps=n_periods)
            result['forecast'] = forecast_obj.predicted_mean
            conf_int_df = forecast_obj.conf_int()
            result['conf_int_lower'] = conf_int_df.iloc[:, 0]
            result['conf_int_upper'] = conf_int_df.iloc[:, 1]
    except Exception as e:
        result["error"] = f"ARIMA forecast failed: {e}"
        logger.error(result["error"], exc_info=True)
    return result

def forecast_prophet(
    series_df: pd.DataFrame, 
    n_periods: int = 30,
    **prophet_kwargs
) -> Dict[str, Any]:
# ... (existing forecast_prophet code) ...
    if not PROPHET_AVAILABLE: return {"error": "Prophet library not installed."}
    try:
        series_df['ds'] = pd.to_datetime(series_df['ds'])
        model = Prophet(**prophet_kwargs)
        model.fit(series_df.dropna(subset=['y']))
        future = model.make_future_dataframe(periods=n_periods)
        forecast_df = model.predict(future)
        return {"forecast_df": forecast_df, "model": model}
    except Exception as e:
        logger.error(f"Prophet forecast failed: {e}", exc_info=True)
        return {"error": f"Prophet forecast failed: {e}"}


# --- Survival Analysis ---
def survival_analysis_kaplan_meier(
    durations: Union[List[float], pd.Series],
    event_observed: Union[List[bool], pd.Series],
    confidence_level: float = 0.95
) -> Dict[str, Any]:
# ... (existing survival_analysis_kaplan_meier code) ...
    if not LIFELINES_AVAILABLE: return {"error": "Lifelines library not installed."}
    try:
        kmf = KaplanMeierFitter()
        kmf.fit(durations, event_observed=event_observed, alpha=(1 - confidence_level))
        return {"survival_function": kmf.survival_function_, "confidence_interval": kmf.confidence_interval_survival_function_,
                "median_survival_time": kmf.median_survival_time_, "timeline": kmf.timeline, "event_table": kmf.event_table}
    except Exception as e:
        logger.error(f"Kaplan-Meier analysis failed: {e}", exc_info=True)
        return {"error": f"Kaplan-Meier analysis failed: {e}"}

def survival_analysis_cox_ph(
    df_cox: pd.DataFrame,
    duration_col: str,
    event_col: str,
    covariate_cols: Optional[List[str]] = None
) -> Dict[str, Any]:
# ... (existing survival_analysis_cox_ph code) ...
    if not LIFELINES_AVAILABLE: return {"error": "Lifelines library not installed."}
    try:
        cph = CoxPHFitter()
        if covariate_cols:
            for col in covariate_cols:
                if col in df_cox.columns: df_cox[col] = pd.to_numeric(df_cox[col], errors='coerce')
            df_cox_final = df_cox[[duration_col, event_col] + covariate_cols].dropna()
        else:
            df_cox_final = df_cox[[duration_col, event_col]].dropna()
        if df_cox_final.empty: return {"error": "No valid data for Cox PH after NaNs."}
        cph.fit(df_cox_final, duration_col=duration_col, event_col=event_col)
        return {"summary": cph.summary, "baseline_survival": cph.baseline_survival_, 
                "hazard_ratios": np.exp(cph.params_), "concordance_index": cph.concordance_index_,
                "log_likelihood_ratio_test": cph.log_likelihood_ratio_test()}
    except Exception as e:
        logger.error(f"Cox PH analysis failed: {e}", exc_info=True)
        return {"error": f"Cox PH analysis failed: {e}"}

# --- Anomaly Detection ---
def detect_anomalies(
    data: np.ndarray, 
    method: str = 'isolation_forest',
    contamination: Union[str, float] = 'auto',
    **model_kwargs
) -> Dict[str, Any]:
# ... (existing detect_anomalies code) ...
    if method == 'isolation_forest':
        if not ISOLATION_FOREST_AVAILABLE: return {"error": "Isolation Forest library (scikit-learn) not available."}
        if data is None or not isinstance(data, np.ndarray) or data.ndim != 2 or data.shape[0] < 5:
            return {"error": "Invalid or insufficient data for Isolation Forest."}
        try:
            model = IsolationForest(contamination=contamination, random_state=model_kwargs.get('random_state', 42), 
                                    n_estimators=model_kwargs.get('n_estimators', 100), 
                                    max_samples=model_kwargs.get('max_samples', 'auto'))
            predictions = model.fit_predict(data)
            anomalies_indices = np.where(predictions == -1)[0]
            return {"anomalies_indices": anomalies_indices.tolist(), 
                    "model_details": {"method": "Isolation Forest", "contamination_used": contamination, 
                                      "n_estimators": model.n_estimators, "max_samples": model.max_samples_}}
        except Exception as e:
            logger.error(f"Error in Isolation Forest detection: {e}", exc_info=True)
            return {"error": f"Isolation Forest execution failed: {e}"}
    else:
        return {"error": f"Anomaly detection method '{method}' not supported."}

if __name__ == '__main__':
    logger.info("ai_models.py executed directly for testing.")
    # ... (existing test for Anomaly Detection) ...
