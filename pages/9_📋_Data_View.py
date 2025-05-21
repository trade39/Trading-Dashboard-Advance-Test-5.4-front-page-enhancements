"""
pages/9_📋_Data_View.py

This page provides an enhanced view of the currently filtered trade data,
allowing users to inspect the raw numbers, select columns for display,
and download the dataset. It utilizes the DataTableDisplay component
and includes basic conditional visual cues.
"""
import streamlit as st
import pandas as pd
import numpy as np # For np.select
import logging

try:
    from config import APP_TITLE, EXPECTED_COLUMNS, CONCEPTUAL_COLUMNS, CONCEPTUAL_COLUMN_TYPES
    from components.data_table_display import DataTableDisplay
    from utils.common_utils import display_custom_message
except ImportError as e:
    st.error(f"Data View Page Error: Critical module import failed: {e}. Ensure app structure is correct and all dependencies are met.")
    APP_TITLE = "TradingDashboard_Error"
    logger = logging.getLogger(APP_TITLE)
    logger.error(f"CRITICAL IMPORT ERROR in Data View Page: {e}", exc_info=True)
    EXPECTED_COLUMNS = {"date": "date", "pnl": "pnl", "direction_str": "direction_str"}
    CONCEPTUAL_COLUMNS = {"date": "Date", "pnl": "PnL", "direction_str": "Direction"}
    CONCEPTUAL_COLUMN_TYPES = {"date": "datetime", "pnl": "numeric", "direction_str": "text"}
    class DataTableDisplay:
        def __init__(self, *args, **kwargs): pass
        def render(self): st.warning("DataTableDisplay component could not be loaded.")
    def display_custom_message(message, type): st.info(message)
    st.stop()

logger = logging.getLogger(APP_TITLE)

def generate_dynamic_column_config(df_columns_to_configure: list, original_df_for_dtype_check: pd.DataFrame) -> dict:
    """
    Generates column configurations dynamically.
    """
    configs = {}
    for col_key in df_columns_to_configure:
        display_name = CONCEPTUAL_COLUMNS.get(col_key, col_key.replace("_", " ").title())
        # Special display name for our new 'Outcome' column
        if col_key == 'Outcome':
            display_name = "Outcome"
            
        col_type = CONCEPTUAL_COLUMN_TYPES.get(col_key)
        help_text = f"Data for: {display_name}"
        is_boolean_col = False
        if col_key in original_df_for_dtype_check.columns and pd.api.types.is_bool_dtype(original_df_for_dtype_check[col_key]):
            is_boolean_col = True

        if col_key == 'Outcome' or col_key == EXPECTED_COLUMNS.get('direction_str'): # Our emoji columns
            configs[col_key] = st.column_config.TextColumn(display_name, help=help_text, width="small")
        elif col_type == "datetime":
            configs[col_key] = st.column_config.DatetimeColumn(display_name, format="YYYY-MM-DD HH:mm:ss", help=help_text)
        elif col_type == "numeric":
            format_str = "%.2f"
            if "pnl" in col_key.lower() or "price" in col_key.lower() or \
               "balance" in col_key.lower() or "value" in col_key.lower() or \
               "commission" in col_key.lower() or "fees" in col_key.lower() or \
               "drawdown_abs" in col_key.lower():
                format_str = "$%.2f"
            elif "pct" in col_key.lower() or "rate" in col_key.lower() or \
                 (col_key == "ror_pct") or \
                 (col_key == "sharpe_ratio") or (col_key == "sortino_ratio") or (col_key == "calmar_ratio"):
                if col_key == 'drawdown_pct':
                    format_str = "%.2f%%"
                else:
                    format_str = "%.2f"
            configs[col_key] = st.column_config.NumberColumn(display_name, format=format_str, help=help_text)
        elif is_boolean_col:
             configs[col_key] = st.column_config.CheckboxColumn(display_name, help=help_text)
        else: # Default to TextColumn for 'text' or unspecified types
            configs[col_key] = st.column_config.TextColumn(display_name, help=help_text)
    return configs

def show_data_view_page():
    """
    Renders the content for the Filtered Data View page with enhanced UI.
    """
    st.title("📋 Filtered Trade Data Log")
    logger.info("Rendering Data View Page.")

    if 'filtered_data' not in st.session_state or st.session_state.filtered_data is None:
        display_custom_message("Please upload and process your trading journal in the main application to view the data log here.", "info")
        return

    filtered_df_original = st.session_state.filtered_data.copy() # Work on a copy

    if filtered_df_original.empty:
        display_custom_message("No data matches the current filters. The data log is empty. Try adjusting your filter criteria in the sidebar.", "info")
        return

    # --- Apply Visual Enhancements to DataFrame ---
    pnl_col_name = EXPECTED_COLUMNS.get('pnl')
    direction_col_name = EXPECTED_COLUMNS.get('direction_str')

    # 1. Add 'Outcome' column based on PnL
    if pnl_col_name and pnl_col_name in filtered_df_original.columns:
        conditions = [
            filtered_df_original[pnl_col_name] > 0,
            filtered_df_original[pnl_col_name] < 0,
            filtered_df_original[pnl_col_name] == 0
        ]
        choices = ["✅", "❌", "➖"] # Win, Loss, Breakeven
        filtered_df_original['Outcome'] = np.select(conditions, choices, default="❔")
    else:
        filtered_df_original['Outcome'] = "N/A"


    # 2. Modify 'Direction' column with emojis
    if direction_col_name and direction_col_name in filtered_df_original.columns:
        # Ensure the column is string type before applying string methods
        filtered_df_original[direction_col_name] = filtered_df_original[direction_col_name].astype(str).fillna("N/A")
        
        dir_conditions = [
            filtered_df_original[direction_col_name].str.upper().str.contains("LONG", na=False),
            filtered_df_original[direction_col_name].str.upper().str.contains("SHORT", na=False)
        ]
        dir_choices = [
            "⬆️ " + filtered_df_original[direction_col_name],
            "⬇️ " + filtered_df_original[direction_col_name]
        ]
        filtered_df_original[direction_col_name] = np.select(dir_conditions, dir_choices, default=filtered_df_original[direction_col_name])


    # Main container for the page content
    st.markdown("<div class='page-content-container data-view-page-container'>", unsafe_allow_html=True)
    st.markdown("<div class='config-section-container'>", unsafe_allow_html=True)
    st.markdown("<h2 class='section-header'>Customize Data View</h2>", unsafe_allow_html=True)
    
    all_columns_available = filtered_df_original.columns.tolist()
    
    # --- Define default columns, including the new 'Outcome' column ---
    default_display_keys = [
        EXPECTED_COLUMNS.get('date'),
        'Outcome', # Add new Outcome column
        pnl_col_name, # Keep original PnL for sorting/formatting
        EXPECTED_COLUMNS.get('trade_id'),
        EXPECTED_COLUMNS.get('trade_size_num'),
        direction_col_name, # Display modified direction column
        EXPECTED_COLUMNS.get('strategy'),
        EXPECTED_COLUMNS.get('trade_day_str'),
        'drawdown_pct'
    ]
    
    # Filter out None values and ensure they exist in the current DataFrame
    default_selected_cols = [col for col in default_display_keys if col and col in all_columns_available]

    if not default_selected_cols and all_columns_available:
        default_selected_cols = all_columns_available[:min(10, len(all_columns_available))]

    selected_columns = st.multiselect(
        "Select columns to display in the table below:",
        options=all_columns_available,
        default=default_selected_cols,
        key="data_view_column_selector"
    )
    st.markdown("</div>", unsafe_allow_html=True) 

    if not selected_columns:
        display_custom_message("Please select at least one column to display the data table.", "warning")
        st.markdown("</div>", unsafe_allow_html=True) 
        return

    df_to_display = filtered_df_original[selected_columns]

    st.markdown("<div class='table-section-container'>", unsafe_allow_html=True)
    st.markdown(f"<p class='data-count-info'>Displaying <strong>{len(df_to_display)}</strong> trades with <strong>{len(selected_columns)}</strong> selected columns.</p>", unsafe_allow_html=True)

    try:
        column_configs_for_table = generate_dynamic_column_config(selected_columns, filtered_df_original)
        
        data_table_component = DataTableDisplay(
            dataframe=df_to_display,
            title=None, 
            column_config=column_configs_for_table,
            height=600, 
            use_container_width=True,
            download_button=True,
            download_file_name=f"filtered_trades_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv"
        )
        data_table_component.render()

    except Exception as e:
        logger.error(f"Error rendering data table on Data View page: {e}", exc_info=True)
        display_custom_message(f"An error occurred while displaying the data table: {e}", "error")
    
    st.markdown("</div>", unsafe_allow_html=True) 
    st.markdown("</div>", unsafe_allow_html=True) 

if __name__ == "__main__":
    if 'app_initialized' not in st.session_state:
        st.warning("This page is part of a multi-page app. Please run the main app.py script for full functionality.")
    show_data_view_page()
