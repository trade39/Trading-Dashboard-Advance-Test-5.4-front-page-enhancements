# services/ai_model_service.py
import pandas as pd
import logging 
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np 

from ai_models import (
    train_simple_rf_model, 
    predict_with_model,
    forecast_arima, 
    forecast_prophet,
    survival_analysis_kaplan_meier,
    survival_analysis_cox_ph,
    detect_anomalies
)
try:
    from config import APP_TITLE 
except ImportError:
    APP_TITLE = "TradingDashboard_Default_Service"

logger = logging.getLogger(APP_TITLE)

class AIModelService:
    """
    Service layer for AI model operations.
    """
    def __init__(self):
        self.trained_rf_classifier: Optional[Any] = None # Store the classifier
        self.rf_classifier_features: List[str] = []
        self.rf_classifier_target: str = ''
        self.rf_classifier_feature_importances: Optional[pd.Series] = None # Store feature importances
        logger.info("AIModelService initialized.")

    def train_trade_outcome_classifier( # Renamed for clarity
        self, 
        df: pd.DataFrame, 
        features: List[str], 
        target: str, 
        test_size: float = 0.2, 
        random_state: int = 42,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        class_weight: Optional[Union[Dict, str]] = 'balanced'
    ) -> Dict[str, Any]: # Return a dictionary for more structured results
        """
        Trains a Random Forest model to classify trade outcomes.
        Stores the trained model and its metrics internally.

        Returns:
            Dict[str, Any]: Contains 'model', 'accuracy', 'classification_report', 
                            'feature_importances', or 'error'.
        """
        logger.info(f"AIModelService: Training Trade Outcome Classifier. Features: {features}, Target: {target}")
        try:
            model, accuracy, report, importances = train_simple_rf_model(
                df, features, target, test_size, random_state, n_estimators, max_depth, class_weight
            )
            if model is not None and accuracy is not None:
                self.trained_rf_classifier = model
                self.rf_classifier_features = features # Store original feature list
                self.rf_classifier_target = target
                self.rf_classifier_feature_importances = importances
                
                logger.info(f"AIModelService: Classifier training successful. Accuracy: {accuracy:.4f}")
                return {
                    "model": model, # Though generally don't return model to UI, useful for service layer
                    "accuracy": accuracy,
                    "classification_report": report,
                    "feature_importances": importances
                }
            else:
                error_msg = "Classifier training failed in ai_models module (model or accuracy was None)."
                logger.error(f"AIModelService: {error_msg}")
                return {"error": error_msg}
        except Exception as e:
            logger.error(f"AIModelService: Exception during classifier training: {e}", exc_info=True)
            return {"error": f"Exception during training: {e}"}

    def predict_trade_outcome( # Renamed for clarity
        self, 
        new_data_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Makes predictions using the internally stored trained classifier.
        """
        if self.trained_rf_classifier is None:
            logger.error("AIModelService: Prediction failed. No classifier model has been trained.")
            return {"error": "No classifier trained."}
        if not self.rf_classifier_features: # Use the stored original features for consistency
            logger.error("AIModelService: Prediction failed. Model features not defined.")
            return {"error": "Model features not defined."}
            
        logger.info(f"AIModelService: Predicting with trained classifier using features: {self.rf_classifier_features}")
        try:
            predictions, probabilities = predict_with_model(
                self.trained_rf_classifier, new_data_df, self.rf_classifier_features
            )
            if predictions is not None:
                return {"predictions": predictions, "probabilities": probabilities}
            else:
                return {"error": "Prediction returned None from ai_models."}
        except Exception as e:
            logger.error(f"AIModelService: Exception during classifier prediction: {e}", exc_info=True)
            return {"error": f"Exception during prediction: {e}"}

    def get_classifier_status(self) -> Dict[str, Any]: # Renamed for clarity
        """Returns information about the currently trained classifier."""
        if self.trained_rf_classifier:
            return {
                "trained": True,
                "features": self.rf_classifier_features,
                "target": self.rf_classifier_target,
                "model_type": str(type(self.trained_rf_classifier).__name__),
                "feature_importances": self.rf_classifier_feature_importances.to_dict() if self.rf_classifier_feature_importances is not None else None
            }
        return {"trained": False}

    # --- Forecasting Methods ---
    def get_arima_forecast(self, series: pd.Series, order: Optional[Tuple[int, int, int]] = None, seasonal_order: Optional[Tuple[int, int, int, int]] = None, n_periods: int = 30) -> Dict[str, Any]:
    # ... (existing get_arima_forecast code) ...
        logger.info(f"AIModelService: Requesting ARIMA forecast for {n_periods} periods.")
        result = forecast_arima(series, order=order, seasonal_order=seasonal_order, n_periods=n_periods)
        return result if result is not None else {"error": "ARIMA forecast function returned None."}


    def get_prophet_forecast(self, series_df: pd.DataFrame, n_periods: int = 30, **prophet_kwargs) -> Dict[str, Any]:
    # ... (existing get_prophet_forecast code) ...
        logger.info(f"AIModelService: Requesting Prophet forecast for {n_periods} periods.")
        result = forecast_prophet(series_df, n_periods=n_periods, **prophet_kwargs)
        return result if result is not None else {"error": "Prophet forecast function returned None."}


    # --- Survival Analysis Methods ---
    def perform_kaplan_meier_analysis(self, durations: Union[List[float], pd.Series], event_observed: Union[List[bool], pd.Series], confidence_level: float) -> Dict[str, Any]:
    # ... (existing perform_kaplan_meier_analysis code) ...
        logger.info("AIModelService: Performing Kaplan-Meier survival analysis.")
        result = survival_analysis_kaplan_meier(durations, event_observed, confidence_level)
        return result if result is not None else {"error": "Kaplan-Meier analysis function returned None."}


    def perform_cox_ph_analysis(self, df_cox: pd.DataFrame, duration_col: str, event_col: str, covariate_cols: Optional[List[str]] = None) -> Dict[str, Any]:
    # ... (existing perform_cox_ph_analysis code) ...
        logger.info("AIModelService: Performing Cox Proportional Hazards analysis.")
        result = survival_analysis_cox_ph(df_cox, duration_col, event_col, covariate_cols)
        return result if result is not None else {"error": "Cox PH analysis function returned None."}

    # --- Anomaly Detection ---
    def perform_anomaly_detection(self, data: Union[pd.DataFrame, pd.Series, np.ndarray], method: str = 'isolation_forest', contamination: Union[str, float] = 'auto', **model_kwargs) -> Dict[str, Any]:
    # ... (existing perform_anomaly_detection code, ensure it handles data correctly for ai_models) ...
        logger.info(f"AIModelService: Performing anomaly detection using {method}.")
        # Ensure data is a 2D numpy array if it's a Series for detect_anomalies
        if isinstance(data, pd.Series):
            data_np = data.dropna().values.reshape(-1, 1)
        elif isinstance(data, pd.DataFrame): # Assuming if DataFrame, it's already correctly shaped or handled by detect_anomalies
            data_np = data.dropna().values # Or specific column logic
        elif isinstance(data, np.ndarray):
            data_np = data
        else:
            return {"error": "Invalid data type for anomaly detection."}

        if data_np.ndim == 1: # Reshape if it became 1D after processing
            data_np = data_np.reshape(-1,1)

        result = detect_anomalies(data_np, method=method, contamination=contamination, **model_kwargs)
        return result if result is not None else {"error": "Anomaly detection function returned None."}
