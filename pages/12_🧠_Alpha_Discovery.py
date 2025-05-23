# pages/12_🧠_Alpha_Discovery.py
"""
Streamlit page for Alpha Discovery using statistical and ML techniques.
"""
import streamlit as st
import pandas as pd
import numpy as np

from utils.logger import get_logger
from utils.common_utils import (
    ensure_data_loaded,
    get_numeric_columns,
    get_categorical_columns,
    get_non_numeric_columns
)
# Assuming analytics modules are structured as shown previously
from analytics.ml_alpha_discovery import run_feature_importance_analysis, run_clustering_analysis
from analytics.statistical_alpha_analyzer import calculate_correlations, perform_t_test_for_groups, plot_distribution
# from config import ALPHA_DISCOVERY_CONFIG # Example: For future use

logger = get_logger(__name__)

def show_alpha_discovery_page():
    """
    Renders the Alpha Discovery page.
    """
    st.set_page_config(layout="wide", page_title="Alpha Discovery", page_icon="🧠")
    st.title("🧠 Alpha Discovery")
    st.markdown("""
    Explore your trading data to uncover potential patterns and signals (alpha) 
    using statistical analysis and machine learning models.
    """)

    if not ensure_data_loaded():
        return

    df_processed = st.session_state.get('df_processed', pd.DataFrame())
    if df_processed.empty: # Double check, ensure_data_loaded should catch this
        st.warning("No processed data available. Please upload and process data first.")
        return

    # --- Configuration Note ---
    # Advanced parameters for models (e.g., n_estimators, test_size for RandomForest)
    # could be loaded from a `config.py` file or set via `st.sidebar.expander("Advanced Settings")`.
    # Example:
    # from config import RF_N_ESTIMATORS, RF_TEST_SIZE
    # rf_n_estimators = st.sidebar.number_input("RF Estimators", value=RF_N_ESTIMATORS)

    # --- Page Layout ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Exploratory Data Analysis (EDA)",
        "🔗 Correlation Analysis",
        "🔬 Hypothesis Testing",
        "🤖 Machine Learning Insights"
    ])

    # --- 1. Exploratory Data Analysis (EDA) ---
    with tab1:
        st.header("📊 Exploratory Data Analysis")
        st.markdown("Understand the distribution of individual variables.")
        
        eda_cols = get_numeric_columns(df_processed)
        if not eda_cols:
            st.info("No numeric columns available for EDA distribution plots.")
        else:
            eda_col_select = st.selectbox(
                "Select a numeric column for distribution analysis:",
                options=eda_cols,
                key="eda_col_select",
                help="Choose a column to see its statistical distribution."
            )
            if eda_col_select:
                try:
                    dist_fig = plot_distribution(df_processed, eda_col_select)
                    if dist_fig:
                        st.plotly_chart(dist_fig, use_container_width=True)
                    else:
                        st.warning(f"Could not generate distribution plot for {eda_col_select}.")
                except Exception as e:
                    logger.error(f"Error in EDA distribution plot for {eda_col_select}: {e}", exc_info=True)
                    st.error(f"An error occurred while generating the distribution plot: {e}")

    # --- 2. Correlation Analysis ---
    with tab2:
        st.header("🔗 Correlation Analysis")
        st.markdown("Identify linear relationships between numeric variables.")
        
        corr_method = st.selectbox(
            "Select correlation method:",
            options=['pearson', 'kendall', 'spearman'],
            key="corr_method_select",
            help="Pearson: standard correlation. Kendall/Spearman: rank-based, non-parametric."
        )
        
        if st.button("Calculate Correlation Matrix", key="corr_button"):
            try:
                corr_matrix, corr_fig = calculate_correlations(df_processed, method=corr_method)
                if corr_matrix is not None and not corr_matrix.empty:
                    st.subheader("Correlation Matrix")
                    st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', axis=None).format("{:.2f}"))
                    if corr_fig:
                        st.plotly_chart(corr_fig, use_container_width=True)
                    # --- Plotting Note ---
                    # For more complex or customized plots, consider moving logic to `plotting.py`.
                    # Example: from plotting import plot_custom_heatmap
                    # plot_custom_heatmap(corr_matrix)
                elif corr_matrix is not None and corr_matrix.empty and df_processed.select_dtypes(include=np.number).empty:
                     st.info("No numeric columns found in the data to calculate correlations.")
                elif corr_matrix is not None and corr_matrix.empty:
                    st.warning("Correlation matrix is empty. This might be due to no numeric columns with variance.")
                else:
                    st.warning("Could not calculate correlations. Check data or logs.")
            except Exception as e:
                logger.error(f"Error in correlation analysis: {e}", exc_info=True)
                st.error(f"An error occurred during correlation analysis: {e}")

    # --- 3. Hypothesis Testing (Example: T-test) ---
    with tab3:
        st.header("🔬 Hypothesis Testing")
        st.markdown("Test statistical hypotheses about your data. Example: Is there a significant difference in Profit/Loss between two trade types?")

        numeric_cols_ht = get_numeric_columns(df_processed)
        # For group_col, allow selection from categorical or reasonably low-cardinality numeric/object columns
        potential_group_cols = [col for col in df_processed.columns if df_processed[col].nunique() < 20 and col not in numeric_cols_ht] + get_categorical_columns(df_processed)
        potential_group_cols = sorted(list(set(potential_group_cols)))


        if not numeric_cols_ht:
            st.info("No numeric columns available for hypothesis testing value.")
        elif not potential_group_cols:
            st.info("No suitable columns available for grouping in hypothesis tests.")
        else:
            ht_value_col = st.selectbox(
                "Select Value Column (Numeric):",
                options=numeric_cols_ht,
                key="ht_value_col",
                help="The numeric variable you want to test (e.g., 'Profit/Loss')."
            )
            ht_group_col = st.selectbox(
                "Select Group Column (Categorical/Low Cardinality):",
                options=potential_group_cols,
                key="ht_group_col",
                help="The column that defines the groups to compare (e.g., 'Strategy', 'DayOfWeek')."
            )

            if ht_value_col and ht_group_col and ht_value_col != ht_group_col:
                if st.button(f"Perform T-test/ANOVA for '{ht_value_col}' by '{ht_group_col}'", key="ht_button"):
                    try:
                        ht_summary, ht_fig = perform_t_test_for_groups(df_processed, ht_value_col, ht_group_col)
                        if ht_summary:
                            st.subheader("Hypothesis Test Results")
                            st.text(ht_summary)
                            if ht_fig:
                                st.plotly_chart(ht_fig, use_container_width=True)
                        else:
                            st.warning("Could not perform hypothesis test. Check inputs or logs.")
                    except Exception as e:
                        logger.error(f"Error in hypothesis testing: {e}", exc_info=True)
                        st.error(f"An error occurred during hypothesis testing: {e}")
            elif ht_value_col == ht_group_col and ht_value_col is not None:
                 st.warning("Value column and Group column cannot be the same.")


    # --- 4. Machine Learning Insights ---
    with tab4:
        st.header("🤖 Machine Learning Insights")
        st.markdown("Use ML models to find predictive features or segment data.")

        # --- Feature Importance ---
        st.subheader("🔍 Feature Importance Analysis")
        st.markdown("""
        Identifies which features are most predictive of a target variable using a RandomForest model.
        This can help in understanding drivers of P&L, win/loss, etc.
        """)

        all_cols = df_processed.columns.tolist()
        
        # Try to find a default target (e.g., PnL, Profit, Return)
        default_target_candidates = ['profit/loss', 'pnl', 'profit', 'return', 'target', 'outcome', 'gain']
        target_col_ml = None
        for cand_base in default_target_candidates:
            for col in all_cols: # Case-insensitive check
                if cand_base == col.lower():
                    target_col_ml = col
                    break
            if target_col_ml:
                break
        if not target_col_ml and all_cols: # Fallback to last column if no candidate found
            target_col_ml = all_cols[-1] if all_cols else None

        target_col_ml = st.selectbox(
            "Select Target Variable for ML:",
            options=all_cols,
            index=all_cols.index(target_col_ml) if target_col_ml and target_col_ml in all_cols else 0,
            key="ml_target_select",
            help="The variable the ML model will try to predict or explain."
        )

        if target_col_ml:
            potential_features_ml = [col for col in all_cols if col != target_col_ml]
            
            # Attempt to pre-select sensible features (numeric or low-cardinality categorical)
            default_features_ml = get_numeric_columns(df_processed, exclude=[target_col_ml])
            non_numeric_options = get_non_numeric_columns(df_processed, exclude=[target_col_ml])
            for col in non_numeric_options:
                if df_processed[col].nunique() < 10: # Add low-cardinality non-numeric as potential features
                    default_features_ml.append(col)
            default_features_ml = [f for f in default_features_ml if f in potential_features_ml]


            feature_cols_ml = st.multiselect(
                "Select Feature Variables for ML:",
                options=potential_features_ml,
                default=default_features_ml,
                key="ml_features_select",
                help="Variables used by the model to predict the target. Non-numeric categorical features will be one-hot encoded if not binary."
            )

            problem_type_options = ["classification", "regression"]
            # Auto-detect problem type based on target variable
            default_problem_type = "regression"
            if pd.api.types.is_numeric_dtype(df_processed[target_col_ml]):
                if df_processed[target_col_ml].nunique() < 10 or \
                   (df_processed[target_col_ml].dropna().apply(float.is_integer).all() and df_processed[target_col_ml].nunique() < 20) : # Heuristic for few unique numeric values
                    default_problem_type = "classification"
            else: # Non-numeric is likely classification
                default_problem_type = "classification"

            problem_type_ml = st.radio(
                "Select Problem Type:",
                options=problem_type_options,
                index=problem_type_options.index(default_problem_type),
                key="ml_problem_type",
                horizontal=True,
                help="Classification for discrete outcomes (e.g., win/loss), Regression for continuous values (e.g., P&L amount)."
            )

            if feature_cols_ml:
                if st.button("Run Feature Importance Analysis", key="ml_run_button"):
                    X = df_processed[feature_cols_ml]
                    y = df_processed[target_col_ml]
                    try:
                        importances_df, importances_fig, model_report, _ = run_feature_importance_analysis(X, y, problem_type=problem_type_ml)
                        
                        if importances_df is not None and not importances_df.empty:
                            st.write("Model Performance / Report:")
                            st.text(model_report)
                            st.dataframe(importances_df)
                            if importances_fig:
                                st.plotly_chart(importances_fig, use_container_width=True)
                        elif importances_df is not None and importances_df.empty and model_report:
                             st.info(f"Feature importance analysis ran, but no importances were generated. Model report: {model_report}")
                        else:
                            st.warning(f"Could not run feature importance analysis. Report: {model_report}")
                    except Exception as e:
                        logger.error(f"Error in ML feature importance: {e}", exc_info=True)
                        st.error(f"An error occurred during ML analysis: {e}")
            else:
                st.info("Please select at least one feature variable for ML analysis.")
        else:
            st.info("Please select a target variable for ML analysis.")

        # --- Clustering Analysis ---
        st.subheader("🧩 Clustering Analysis (K-Means)")
        st.markdown("""
        Segment your data into distinct groups (clusters) based on selected features.
        This can help identify different trading regimes or behavioral patterns.
        """)
        
        potential_cluster_features = get_numeric_columns(df_processed) # K-Means typically uses numeric features
        
        if not potential_cluster_features:
            st.info("No numeric features available for clustering analysis.")
        else:
            cluster_features_select = st.multiselect(
                "Select Features for Clustering:",
                options=potential_cluster_features,
                default=potential_cluster_features[:min(len(potential_cluster_features), 3)], # Default to first 3 numeric
                key="cluster_features_select",
                help="Numeric features to be used for K-Means clustering."
            )
            
            n_clusters = st.slider(
                "Number of Clusters (K):",
                min_value=2,
                max_value=10,
                value=3,
                step=1,
                key="n_clusters_slider"
            )

            if cluster_features_select and len(cluster_features_select) >=1: # K-Means needs at least 1 feature
                if st.button("Run Clustering Analysis", key="cluster_run_button"):
                    X_cluster = df_processed[cluster_features_select]
                    try:
                        df_with_clusters, cluster_fig, _ = run_clustering_analysis(X_cluster, n_clusters=n_clusters)
                        if df_with_clusters is not None and 'cluster' in df_with_clusters.columns:
                            st.write("Data with Cluster Labels (sample):")
                            st.dataframe(df_with_clusters.head())
                            
                            # Display cluster sizes
                            st.write("Cluster Sizes:")
                            st.dataframe(df_with_clusters['cluster'].value_counts().sort_index().to_frame())

                            if cluster_fig:
                                st.plotly_chart(cluster_fig, use_container_width=True)
                            else:
                                st.info("Clustering analysis complete. No plot generated (requires >=2 features).")
                            
                            # Add clustered data to session state for potential use in other pages or further analysis
                            st.session_state.df_clustered = df_with_clusters 
                            st.success("Clustering complete. `df_clustered` added to session state.")
                        else:
                            st.warning("Clustering analysis did not produce cluster labels. Check selected features or logs.")
                    except Exception as e:
                        logger.error(f"Error in Clustering analysis: {e}", exc_info=True)
                        st.error(f"An error occurred during clustering analysis: {e}")
            else:
                st.info("Please select at least one numeric feature for clustering.")


if __name__ == "__main__":
    # This is for local development/testing of the page
    # You would typically run the main app.py
    # For testing, you might need to mock st.session_state or load sample data
    # Example:
    # if 'df_processed' not in st.session_state:
    #     st.session_state.df_processed = pd.DataFrame(
    #         np.random.rand(100, 5), columns=[f'Feature_{i}' for i in range(5)]
    #     )
    #     st.session_state.df_processed['Category'] = np.random.choice(['A', 'B', 'C'], 100)
    #     st.session_state.df_processed['Target_Class'] = np.random.randint(0, 2, 100)
    #     st.session_state.df_processed['Target_Reg'] = np.random.rand(100) * 10
    
    show_alpha_discovery_page()
