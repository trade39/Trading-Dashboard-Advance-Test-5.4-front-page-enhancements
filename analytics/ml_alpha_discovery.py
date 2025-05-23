# analytics/ml_alpha_discovery.py
"""
Handles Machine Learning based alpha discovery analyses.
"""
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, r2_score, mean_squared_error
import plotly.express as px

from utils.logger import setup_logger # Corrected import

# Corrected logger initialization:
# Each module gets its own logger instance, configured by setup_logger.
# It will use the default settings from setup_logger's signature 
# (e.g. _DEFAULT_LOG_FILE, _DEFAULT_LOG_LEVEL from utils/logger.py)
# unless a logger with this specific name (__name__) was already configured 
# (e.g., by app.py with different parameters for this specific module name, which is less common).
logger = setup_logger(logger_name=__name__)

@st.cache_data(show_spinner="Running Feature Importance Analysis...")
def run_feature_importance_analysis(X: pd.DataFrame, y: pd.Series, problem_type="classification", test_size=0.2, random_state=42):
    """
    Trains a RandomForest model and returns feature importances.

    Args:
        X (pd.DataFrame): DataFrame of feature variables.
        y (pd.Series): Series of the target variable.
        problem_type (str): "classification" or "regression".
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Controls the shuffling applied to the data before applying the split.

    Returns:
        pd.DataFrame: DataFrame with features and their importance scores.
        plotly.graph_objects.Figure: Plotly figure for feature importances.
        str: Classification report or regression metrics.
        object: Trained model.
    """
    logger.info(f"Starting feature importance analysis. Problem type: {problem_type}, X_shape: {X.shape}, y_shape: {y.shape}")

    if X.empty or y.empty:
        logger.warning("Input data X or y is empty for feature importance analysis.")
        return pd.DataFrame(), None, "Input data is empty.", None

    # Preprocessing: Ensure all data is numeric, handle NaNs
    X_processed = X.copy()
    for col in X_processed.columns:
        if X_processed[col].dtype == 'object' or pd.api.types.is_categorical_dtype(X_processed[col]):
            try:
                # Attempt to convert to numeric if possible (e.g., '123', '12.3')
                X_processed[col] = pd.to_numeric(X_processed[col])
                logger.info(f"Column '{col}' converted to numeric.")
            except ValueError:
                # If direct conversion fails, use one-hot encoding for non-binary categorical
                if X_processed[col].nunique() > 2:
                    logger.info(f"Applying one-hot encoding to column '{col}'.")
                    X_processed = pd.get_dummies(X_processed, columns=[col], prefix=col, dummy_na=False)
                else: # Binary categorical, use label encoding
                    logger.info(f"Applying label encoding to binary column '{col}'.")
                    le = LabelEncoder()
                    X_processed[col] = le.fit_transform(X_processed[col].astype(str)) # astype(str) to handle mixed types or NaNs
        elif pd.api.types.is_datetime64_any_dtype(X_processed[col]):
            logger.info(f"Converting datetime column '{col}' to numeric (timestamp).")
            X_processed[col] = X_processed[col].astype(np.int64) // 10**9 # Convert to Unix timestamp

    # Align X and y after potential row changes from one-hot encoding (though less likely here)
    # More critically, handle NaNs after all conversions
    X_processed = X_processed.fillna(X_processed.median(numeric_only=True)) # Impute with median for simplicity

    # Ensure y is numeric for regression or properly encoded for classification
    y_processed = y.copy()
    if problem_type == "classification":
        if not pd.api.types.is_numeric_dtype(y_processed):
            le_y = LabelEncoder()
            y_processed = pd.Series(le_y.fit_transform(y_processed.astype(str)), index=y.index, name=y.name)
            logger.info(f"Target variable '{y.name}' label encoded for classification.")
    elif problem_type == "regression":
        if not pd.api.types.is_numeric_dtype(y_processed):
            try:
                y_processed = pd.to_numeric(y_processed)
                logger.info(f"Target variable '{y.name}' converted to numeric for regression.")
            except ValueError:
                logger.error(f"Target variable '{y.name}' could not be converted to numeric for regression.")
                return pd.DataFrame(), None, f"Target variable '{y.name}' is non-numeric and could not be converted.", None
        y_processed = y_processed.fillna(y_processed.median())


    # Align X and y after preprocessing (drop rows with NaNs in y if any persist)
    common_index = X_processed.index.intersection(y_processed.index)
    X_processed = X_processed.loc[common_index]
    y_processed = y_processed.loc[common_index]
    
    if X_processed.empty or y_processed.empty:
        logger.warning("Data became empty after preprocessing and alignment.")
        return pd.DataFrame(), None, "Data empty after preprocessing.", None

    X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, test_size=test_size, random_state=random_state, stratify=y_processed if problem_type=="classification" and y_processed.nunique() > 1 else None)

    model = None
    report_or_metrics = ""

    try:
        if problem_type == "classification":
            model = RandomForestClassifier(random_state=random_state, n_estimators=100, class_weight='balanced')
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            report_or_metrics = classification_report(y_test, predictions, output_dict=False, zero_division=0)
            logger.info("Classification model trained and evaluated.")
        elif problem_type == "regression":
            model = RandomForestRegressor(random_state=random_state, n_estimators=100)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            r2 = r2_score(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            report_or_metrics = f"R-squared: {r2:.4f}\nMean Squared Error: {mse:.4f}"
            logger.info("Regression model trained and evaluated.")
        else:
            logger.error(f"Unsupported problem type: {problem_type}")
            return pd.DataFrame(), None, f"Unsupported problem type: {problem_type}", None

        importances = model.feature_importances_
        feature_names = X_processed.columns
        
        # Ensure feature_names and importances are aligned if X_processed changed columns
        if len(feature_names) != len(importances):
             logger.error(f"Mismatch in feature names ({len(feature_names)}) and importances ({len(importances)}) count. This can happen if one-hot encoding drastically changes column structure mid-process.")
             # Fallback or re-align if possible. For now, error out.
             return pd.DataFrame(), None, "Feature name and importance count mismatch after model training.", None


        importances_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
        importances_df = importances_df.sort_values(by='importance', ascending=False).reset_index(drop=True)
        
        fig = px.bar(importances_df.head(20), x='importance', y='feature', orientation='h', title='Top 20 Feature Importances')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        
        logger.info(f"Feature importance analysis completed. Top feature: {importances_df.iloc[0]['feature'] if not importances_df.empty else 'N/A'}")
        return importances_df, fig, report_or_metrics, model

    except Exception as e:
        logger.error(f"Error during feature importance analysis: {e}", exc_info=True)
        return pd.DataFrame(), None, f"An error occurred: {str(e)}", None

# Example of another ML analysis function you might add
@st.cache_data(show_spinner="Running Clustering Analysis...")
def run_clustering_analysis(X: pd.DataFrame, n_clusters=3, random_state=42):
    """
    Performs K-Means clustering on the data.

    Args:
        X (pd.DataFrame): DataFrame of features for clustering.
        n_clusters (int): The number of clusters to form.
        random_state (int): Determines random number generation for centroid initialization.

    Returns:
        pd.DataFrame: Original DataFrame with an added 'cluster' column.
        plotly.graph_objects.Figure: Scatter plot of clusters (if 2D/3D feasible).
        object: Trained KMeans model.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    logger.info(f"Starting clustering analysis with {n_clusters} clusters.")

    if X.empty:
        logger.warning("Input data X is empty for clustering analysis.")
        return pd.DataFrame(), None, None
    
    X_processed = X.copy()
    # Select only numeric columns for K-Means
    numeric_cols = X_processed.select_dtypes(include=np.number).columns
    if len(numeric_cols) < 1: # Need at least one numeric column
        logger.warning("No numeric columns found for clustering.")
        return X, None, None # Return original X, no figure, no model
    
    X_numeric = X_processed[numeric_cols]

    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X_numeric)
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_numeric.columns, index=X_numeric.index)

    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
        cluster_labels = kmeans.fit_predict(X_scaled_df)
        
        X_with_clusters = X.copy() # Start with original X to preserve all columns
        X_with_clusters['cluster'] = cluster_labels
        
        fig = None
        if X_scaled_df.shape[1] >= 2: # Can create a 2D scatter plot
            fig = px.scatter(X_scaled_df, x=X_scaled_df.columns[0], y=X_scaled_df.columns[1], 
                             color=cluster_labels, title=f'K-Means Clustering ({n_clusters} clusters)',
                             labels={'color': 'Cluster'})
            if X_scaled_df.shape[1] >= 3: # Add 3rd dimension if available
                 fig = px.scatter_3d(X_scaled_df, x=X_scaled_df.columns[0], y=X_scaled_df.columns[1], z=X_scaled_df.columns[2],
                             color=cluster_labels, title=f'K-Means Clustering ({n_clusters} clusters)',
                             labels={'color': 'Cluster'})
        
        logger.info("Clustering analysis completed.")
        return X_with_clusters, fig, kmeans
        
    except Exception as e:
        logger.error(f"Error during clustering analysis: {e}", exc_info=True)
        return X, None, None # Return original X if error
