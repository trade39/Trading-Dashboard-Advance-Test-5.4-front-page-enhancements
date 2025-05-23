# analytics/statistical_alpha_analyzer.py
"""
Handles statistical analyses for alpha discovery.
"""
import pandas as pd
import numpy as np
import streamlit as st
from scipy import stats
import plotly.express as px
import plotly.figure_factory as ff

from utils.logger import setup_logger # Corrected import

# Corrected logger initialization
logger = setup_logger(logger_name=__name__)


@st.cache_data(show_spinner="Calculating Correlations...")
def calculate_correlations(df: pd.DataFrame, method='pearson'):
    """
    Calculates the correlation matrix for numeric columns in the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        method (str): Method of correlation ('pearson', 'kendall', 'spearman').

    Returns:
        pd.DataFrame: Correlation matrix.
        plotly.graph_objects.Figure: Heatmap of the correlation matrix.
    """
    logger.info(f"Calculating correlation matrix using method: {method}.")
    if df.empty:
        logger.warning("Input DataFrame is empty for correlation analysis.")
        return pd.DataFrame(), None

    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.empty:
        logger.warning("No numeric columns found for correlation analysis.")
        return pd.DataFrame(), None
        
    # Drop columns with no variance, as they cause issues with correlation calculation
    numeric_df = numeric_df.loc[:, numeric_df.nunique() > 1]
    if numeric_df.empty:
        logger.warning("No numeric columns with variance found for correlation analysis.")
        return pd.DataFrame(), None

    try:
        corr_matrix = numeric_df.corr(method=method)
        
        # Create heatmap
        fig = px.imshow(corr_matrix, text_auto=".2f", aspect="auto",
                        title=f"{method.capitalize()} Correlation Matrix",
                        color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
        fig.update_layout(height=600 if len(corr_matrix) > 10 else 400)

        logger.info("Correlation matrix and heatmap generated successfully.")
        return corr_matrix, fig
    except Exception as e:
        logger.error(f"Error calculating correlations: {e}", exc_info=True)
        return pd.DataFrame(), None


@st.cache_data(show_spinner="Performing Hypothesis Test (t-test)...")
def perform_t_test_for_groups(df: pd.DataFrame, value_column: str, group_column: str):
    """
    Performs an independent two-sample t-test if the group column has two unique values.
    If more than two groups, suggests ANOVA (but doesn't implement it here for simplicity).

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        value_column (str): The numeric column to test.
        group_column (str): The categorical column defining the groups.

    Returns:
        str: A string summarizing the t-test results or a suggestion for ANOVA.
        plotly.graph_objects.Figure: Box plot of the value column by group.
    """
    logger.info(f"Performing hypothesis test for '{value_column}' grouped by '{group_column}'.")
    if df.empty:
        logger.warning("Input DataFrame is empty for hypothesis test.")
        return "Input DataFrame is empty.", None
    
    if value_column not in df.columns or group_column not in df.columns:
        logger.warning(f"Value column '{value_column}' or group column '{group_column}' not in DataFrame.")
        return f"Column '{value_column}' or '{group_column}' not found.", None

    if not pd.api.types.is_numeric_dtype(df[value_column]):
        logger.warning(f"Value column '{value_column}' is not numeric.")
        return f"Value column '{value_column}' must be numeric.", None

    df_cleaned = df[[value_column, group_column]].copy()
    df_cleaned[group_column] = df_cleaned[group_column].astype(str) # Ensure group column is string
    df_cleaned = df_cleaned.dropna()

    if df_cleaned.empty:
        return "Data is empty after cleaning NaNs.", None

    groups = df_cleaned[group_column].unique()
    
    fig = px.box(df_cleaned, x=group_column, y=value_column, points="all",
                 title=f'Distribution of {value_column} by {group_column}',
                 labels={value_column: value_column, group_column: group_column})
    fig.update_layout(height=500)

    if len(groups) == 2:
        group1_data = df_cleaned[df_cleaned[group_column] == groups[0]][value_column]
        group2_data = df_cleaned[df_cleaned[group_column] == groups[1]][value_column]

        if len(group1_data) < 2 or len(group2_data) < 2:
            logger.warning("Not enough data in one or both groups for t-test.")
            return "Not enough data in one or both groups for t-test after NaN removal.", fig
        
        try:
            # Check for normality (optional, t-test is robust for larger samples)
            # shapiro_g1 = stats.shapiro(group1_data)
            # shapiro_g2 = stats.shapiro(group2_data)
            # normality_info = f"Shapiro-Wilk (Group 1: {groups[0]}): p={shapiro_g1.pvalue:.3f}\nShapiro-Wilk (Group 2: {groups[1]}): p={shapiro_g2.pvalue:.3f}\n"
            
            # Check for equal variances (Levene's test)
            levene_stat, levene_p = stats.levene(group1_data, group2_data)
            equal_var = levene_p > 0.05
            variance_info = f"Levene's test for equal variances: p={levene_p:.3f} (Equal variances: {equal_var})\n"
            
            t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=equal_var, nan_policy='omit')
            
            result_summary = (
                f"Independent Two-Sample t-test for '{value_column}' between '{groups[0]}' and '{groups[1]}':\n"
                f"Mean ({groups[0]}): {group1_data.mean():.4f} (n={len(group1_data)})\n"
                f"Mean ({groups[1]}): {group2_data.mean():.4f} (n={len(group2_data)})\n"
                f"{variance_info}"
                f"T-statistic: {t_stat:.4f}\n"
                f"P-value: {p_value:.4f}\n"
                f"Significance: {'Statistically significant difference' if p_value < 0.05 else 'No statistically significant difference'} (at alpha=0.05)"
            )
            logger.info(f"T-test completed. P-value: {p_value:.4f}")
            return result_summary, fig
        except Exception as e:
            logger.error(f"Error during t-test: {e}", exc_info=True)
            return f"An error occurred during t-test: {str(e)}", fig
            
    elif len(groups) > 2:
        logger.info("More than two groups found. Suggesting ANOVA.")
        # For ANOVA, you would typically use stats.f_oneway
        # Example: f_stat, p_value = stats.f_oneway(*(df_cleaned[df_cleaned[group_column] == group][value_column] for group in groups))
        return f"More than two groups ({len(groups)}) found in '{group_column}'. Consider using ANOVA for comparison.", fig
    else:
        logger.warning("Not enough groups (less than 2) for comparison.")
        return "Not enough distinct groups (less than 2) in '{group_column}' for comparison.", fig

@st.cache_data(show_spinner="Generating Distribution Plot...")
def plot_distribution(df: pd.DataFrame, column_name: str):
    """
    Generates a distribution plot (histogram and KDE) for a numeric column.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column_name (str): The numeric column to plot.

    Returns:
        plotly.graph_objects.Figure: Distribution plot.
    """
    logger.info(f"Generating distribution plot for column: {column_name}")
    if df.empty or column_name not in df.columns:
        logger.warning(f"DataFrame empty or column '{column_name}' not found for distribution plot.")
        return None
    
    if not pd.api.types.is_numeric_dtype(df[column_name]):
        logger.warning(f"Column '{column_name}' is not numeric. Cannot generate distribution plot.")
        return None
        
    try:
        # Using plotly.figure_factory for distplot
        fig = ff.create_distplot([df[column_name].dropna()], [column_name], bin_size=.2, show_hist=True, show_rug=False)
        fig.update_layout(title_text=f'Distribution of {column_name}', height=400)
        logger.info(f"Distribution plot for '{column_name}' generated.")
        return fig
    except Exception as e:
        logger.error(f"Error generating distribution plot for '{column_name}': {e}", exc_info=True)
        return None
