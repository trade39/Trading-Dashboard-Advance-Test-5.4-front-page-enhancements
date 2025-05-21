"""
pages/10_📝_Trade_Notes.py

This page provides an interface for viewing, searching, and filtering
qualitative trade notes associated with individual trades, displayed in a card layout.
It utilizes the NotesViewerComponent.
"""
import streamlit as st
import pandas as pd
import logging

try:
    from config import APP_TITLE, EXPECTED_COLUMNS, CONCEPTUAL_COLUMNS
    from components.notes_viewer import NotesViewerComponent
    from utils.common_utils import display_custom_message # get_title_with_tooltip_html no longer needed here
except ImportError as e:
    st.error(f"Trade Notes Page Error: Critical module import failed: {e}. Ensure app structure is correct.")
    APP_TITLE = "TradingDashboard_Error_TNP" 
    logger = logging.getLogger(APP_TITLE)
    logger.error(f"CRITICAL IMPORT ERROR in Trade Notes Page: {e}", exc_info=True)
    EXPECTED_COLUMNS = {"date": "date", "notes": "notes", "pnl": "pnl", "symbol": "symbol", "strategy": "strategy"}
    CONCEPTUAL_COLUMNS = {k: k.title() for k in EXPECTED_COLUMNS.keys()}
    class NotesViewerComponent:
        def __init__(self, *args, **kwargs): pass
        def render(self): st.warning("NotesViewerComponent could not be loaded.")
    def display_custom_message(message, type): st.info(message)
    st.stop()

logger = logging.getLogger(APP_TITLE)

def show_trade_notes_page():
    # Simplified page title - removed icon and subtitle
    st.title("📝 Trade Notes & Journal") 
    
    logger.info("Rendering Trade Notes Page.")

    if 'filtered_data' not in st.session_state or st.session_state.filtered_data is None:
        display_custom_message("Please upload and process your trading journal in the main application to view trade notes.", "info")
        return

    filtered_df = st.session_state.filtered_data
    if filtered_df.empty:
        display_custom_message("No data matches the current filters. Cannot display trade notes.", "info")
        return

    date_col = EXPECTED_COLUMNS.get('date')
    notes_col = EXPECTED_COLUMNS.get('notes')
    pnl_col = EXPECTED_COLUMNS.get('pnl')
    symbol_col = EXPECTED_COLUMNS.get('symbol')
    strategy_col = EXPECTED_COLUMNS.get('strategy')

    essential_keys_for_component = {
        'date': date_col, 'notes': notes_col, 'pnl': pnl_col
    }
    missing_essential_cols_details = []
    for conceptual_key, actual_col_name in essential_keys_for_component.items():
        if not actual_col_name or actual_col_name not in filtered_df.columns:
            display_name = CONCEPTUAL_COLUMNS.get(conceptual_key, conceptual_key)
            missing_essential_cols_details.append(f"'{display_name}' (expected as '{actual_col_name}')")
    
    if missing_essential_cols_details:
        msg = (f"Cannot display trade notes. Essential data column(s) are missing from your processed data: "
               f"{', '.join(missing_essential_cols_details)}. Please ensure these fields are mapped correctly from your CSV.")
        display_custom_message(msg, "error")
        logger.error(f"TradeNotesPage: Missing essential columns for NotesViewerComponent: {missing_essential_cols_details}. Available: {filtered_df.columns.tolist()}")
        return
    
    if not notes_col or notes_col not in filtered_df.columns:
        display_name = CONCEPTUAL_COLUMNS.get('notes', 'notes')
        msg = (f"The primary notes column ('{display_name}', expected as '{notes_col}') was not found. "
               "Cannot display trade notes.")
        display_custom_message(msg, "error")
        logger.error(f"TradeNotesPage: Primary notes column '{notes_col}' not found.")
        return

    st.markdown("<div class='page-content-container trade-notes-page-container'>", unsafe_allow_html=True)
    try:
        notes_component = NotesViewerComponent(
            notes_dataframe=filtered_df.copy(),
            date_col=date_col,
            notes_col=notes_col,
            pnl_col=pnl_col,
            symbol_col=symbol_col if symbol_col and symbol_col in filtered_df.columns else None,
            strategy_col=strategy_col if strategy_col and strategy_col in filtered_df.columns else None,
            default_sort_by_display="Date",
            default_sort_ascending=False,
            items_per_page=5
        )
        notes_component.render()
    except Exception as e:
        logger.error(f"Error instantiating or rendering NotesViewerComponent: {e}", exc_info=True)
        display_custom_message(f"An error occurred while preparing to display trade notes: {e}", "error")
    
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    if 'app_initialized' not in st.session_state:
        st.warning("This page is part of a multi-page app. Please run the main app.py script for full functionality.")
    show_trade_notes_page()
