"""
components/notes_viewer.py

This component provides an interactive interface for viewing, searching,
and filtering trade notes, displayed in a card-based layout.
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any

try:
    from config import EXPECTED_COLUMNS, APP_TITLE, CONCEPTUAL_COLUMNS
    from utils.common_utils import format_currency, display_custom_message
except ImportError:
    print("Warning (notes_viewer.py): Could not import from root config/utils. Using placeholders.")
    APP_TITLE = "TradingDashboard_Default_NV"
    EXPECTED_COLUMNS = {
        "date": "date", "notes": "notes", "pnl": "pnl", "symbol": "symbol", "strategy": "strategy"
    }
    CONCEPTUAL_COLUMNS = {
        "date": "Date", "notes": "Notes", "pnl": "PnL", "symbol": "Symbol", "strategy": "Strategy"
    }
    def format_currency(value, currency_symbol="$", decimals=2):
        if pd.isna(value): return "N/A"
        return f"{currency_symbol}{value:.{decimals}f}"
    def display_custom_message(message, type): st.info(message)


import logging
logger = logging.getLogger(APP_TITLE)

class NotesViewerComponent:
    """
    A component for displaying and interacting with trade notes in a card layout.
    """
    def __init__(
        self,
        notes_dataframe: pd.DataFrame,
        date_col: str = EXPECTED_COLUMNS.get('date', 'date'),
        notes_col: str = EXPECTED_COLUMNS.get('notes', 'notes'),
        pnl_col: str = EXPECTED_COLUMNS.get('pnl', 'pnl'),
        symbol_col: Optional[str] = EXPECTED_COLUMNS.get('symbol'),
        strategy_col: Optional[str] = EXPECTED_COLUMNS.get('strategy'), # New: strategy column
        default_sort_by_display: str = "Date",
        default_sort_ascending: bool = False,
        items_per_page: int = 5 # New: for pagination
    ):
        self.base_df = notes_dataframe
        self.date_col_orig = date_col
        self.notes_col_orig = notes_col
        self.pnl_col_orig = pnl_col
        self.symbol_col_orig = symbol_col
        self.strategy_col_orig = strategy_col # Store strategy column name

        # Mapping from original column names to display names for sorting options
        self.sortable_col_map_for_display = {
            self.date_col_orig: "Date",
            self.pnl_col_orig: "PnL"
            # Add more if other direct columns should be sortable
        }
        if self.symbol_col_orig and self.symbol_col_orig in self.base_df.columns:
             self.sortable_col_map_for_display[self.symbol_col_orig] = CONCEPTUAL_COLUMNS.get(self.symbol_col_orig, "Symbol")
        if self.strategy_col_orig and self.strategy_col_orig in self.base_df.columns:
             self.sortable_col_map_for_display[self.strategy_col_orig] = CONCEPTUAL_COLUMNS.get(self.strategy_col_orig, "Strategy")


        self.default_sort_by_display = default_sort_by_display
        self.default_sort_ascending = default_sort_ascending
        self.items_per_page = items_per_page

        self.df_prepared = self._prepare_and_filter_df()
        logger.debug(f"NotesViewerComponent initialized. Prepared DF shape: {self.df_prepared.shape}")

    def _prepare_and_filter_df(self) -> pd.DataFrame:
        if self.base_df is None or self.base_df.empty:
            return pd.DataFrame()

        required_original_cols = [self.date_col_orig, self.notes_col_orig, self.pnl_col_orig]
        optional_cols = [self.symbol_col_orig, self.strategy_col_orig]
        for col in optional_cols:
            if col and col in self.base_df.columns:
                required_original_cols.append(col)
            else: # Ensure original col name is None if not present, to avoid errors later
                if col == self.symbol_col_orig: self.symbol_col_orig = None
                if col == self.strategy_col_orig: self.strategy_col_orig = None


        missing_cols = [col for col in [self.date_col_orig, self.notes_col_orig, self.pnl_col_orig] if not col or col not in self.base_df.columns]
        if missing_cols:
            logger.error(f"NotesViewer: Missing essential original columns: {missing_cols}.")
            return pd.DataFrame()

        df = self.base_df[list(filter(None, required_original_cols))].copy() # Filter out None column names
        df = df.dropna(subset=[self.notes_col_orig])
        df = df[df[self.notes_col_orig].astype(str).str.strip().str.len() > 0] # Ensure notes are not empty strings

        try:
            df[self.pnl_col_orig] = pd.to_numeric(df[self.pnl_col_orig], errors='coerce')
        except Exception as e:
            logger.error(f"NotesViewer: Could not convert PnL column '{self.pnl_col_orig}' to numeric: {e}")
            return pd.DataFrame()
        
        # Add an outcome emoji based on PnL for card display
        if self.pnl_col_orig in df.columns:
            conditions = [ df[self.pnl_col_orig] > 0, df[self.pnl_col_orig] < 0, df[self.pnl_col_orig] == 0]
            choices = ["✅", "❌", "➖"]
            df['pnl_outcome_emoji'] = np.select(conditions, choices, default="❔")
        else:
            df['pnl_outcome_emoji'] = "N/A"

        return df

    def render(self) -> None:
        # This subheader is part of the component, page title is handled by the page script
        # st.subheader("Trade Notes Log") 

        if self.df_prepared.empty:
            display_custom_message("No trade notes available to display (data might be empty, missing required columns, or all notes are blank).", "info")
            logger.info("NotesViewer: No prepared data to render.")
            return

        # --- Controls: Search and Sort ---
        controls_cols = st.columns([0.6, 0.4]) # Adjusted column widths
        with controls_cols[0]:
            search_term = st.text_input(
                "Search Notes Content:",
                placeholder="Enter keyword(s) to filter notes...",
                key="notes_viewer_search_v2" # Ensure unique key
            ).strip().lower() # Lowercase for case-insensitive search
        with controls_cols[1]:
            sort_options_display = list(self.sortable_col_map_for_display.values())
            default_sort_idx = 0
            if self.default_sort_by_display in sort_options_display:
                default_sort_idx = sort_options_display.index(self.default_sort_by_display)

            sort_by_col_display = st.selectbox(
                "Sort Notes By:",
                options=sort_options_display,
                index=default_sort_idx,
                key="notes_viewer_sort_col_v2"
            )
            sort_ascending = not st.toggle("Descending Order", value=not self.default_sort_ascending, key="notes_viewer_sort_desc_toggle_v2")


        # --- Filtering and Sorting ---
        df_processed = self.df_prepared.copy()

        if search_term:
            try:
                # Ensure notes column is string and search case-insensitively
                df_processed = df_processed[
                    df_processed[self.notes_col_orig].astype(str).str.lower().str.contains(search_term, na=False)
                ]
            except Exception as e:
                logger.error(f"Error during notes search: {e}", exc_info=True)
                st.error("An error occurred while searching notes.")
                return

        original_sort_col = next((orig for orig, disp in self.sortable_col_map_for_display.items() if disp == sort_by_col_display), None)
        
        if original_sort_col and original_sort_col in df_processed.columns:
            try:
                if original_sort_col == self.date_col_orig: # Ensure date is datetime for sorting
                    df_processed[original_sort_col] = pd.to_datetime(df_processed[original_sort_col], errors='coerce')
                
                df_processed = df_processed.sort_values(
                    by=original_sort_col, ascending=sort_ascending, na_position='last'
                ).reset_index(drop=True)
            except Exception as e:
                logger.error(f"Error sorting notes by {original_sort_col}: {e}", exc_info=True)
                st.error(f"Could not sort notes by {sort_by_col_display}.")
        
        if df_processed.empty:
            display_custom_message("No notes match your search criteria.", "info")
            return

        # --- Pagination ---
        total_items = len(df_processed)
        total_pages = (total_items + self.items_per_page - 1) // self.items_per_page
        
        if total_pages > 1:
            pagination_cols = st.columns([0.7, 0.3])
            with pagination_cols[1]: # Align to the right
                current_page = st.number_input(
                    f"Page (1-{total_pages})", 
                    min_value=1, max_value=total_pages, value=1, step=1, 
                    key="notes_viewer_page_v2",
                    label_visibility="collapsed" # Hide label, use placeholder or surrounding text
                )
            with pagination_cols[0]:
                 st.markdown(f"<p class='pagination-info'>Page {current_page} of {total_pages} ({total_items} notes total)</p>", unsafe_allow_html=True)
        else:
            current_page = 1
            if total_items > 0:
                 st.markdown(f"<p class='pagination-info'>{total_items} note(s) found.</p>", unsafe_allow_html=True)


        start_idx = (current_page - 1) * self.items_per_page
        end_idx = start_idx + self.items_per_page
        df_paginated = df_processed.iloc[start_idx:end_idx]

        # --- Display Notes as Cards ---
        for index, row in df_paginated.iterrows():
            st.markdown("<div class='note-card'>", unsafe_allow_html=True)
            
            # Card Header - Metadata
            st.markdown("<div class='note-card-header'>", unsafe_allow_html=True)
            meta_cols = st.columns([0.4, 0.3, 0.3]) # Date | Symbol/Strategy | PnL
            
            with meta_cols[0]:
                date_val = pd.to_datetime(row[self.date_col_orig]) if pd.notna(row[self.date_col_orig]) else "N/A"
                st.markdown(f"<span class='note-metadata-item date'><i class='far fa-calendar-alt'></i> {date_val.strftime('%Y-%m-%d %H:%M') if date_val != 'N/A' else 'N/A'}</span>", unsafe_allow_html=True)
            
            with meta_cols[1]:
                symbol_val = row.get(self.symbol_col_orig, "N/A") if self.symbol_col_orig else "N/A"
                strategy_val = row.get(self.strategy_col_orig, "N/A") if self.strategy_col_orig else "N/A"
                
                meta_text_parts = []
                if symbol_val != "N/A": meta_text_parts.append(f"<i class='fas fa-chart-line'></i> {symbol_val}")
                if strategy_val != "N/A": meta_text_parts.append(f"<i class='fas fa-lightbulb'></i> {strategy_val}")
                
                st.markdown(f"<span class='note-metadata-item strategy-symbol'>{' | '.join(meta_text_parts) if meta_text_parts else 'N/A'}</span>", unsafe_allow_html=True)

            with meta_cols[2]:
                pnl_val = row.get(self.pnl_col_orig, 0)
                pnl_emoji = row.get('pnl_outcome_emoji', '')
                formatted_pnl = format_currency(pnl_val) if pd.notna(pnl_val) else "N/A"
                st.markdown(f"<span class='note-metadata-item pnl'>{pnl_emoji} {formatted_pnl}</span>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True) # Close note-card-header

            # Card Body - Note Content
            st.markdown("<div class='note-card-content'>", unsafe_allow_html=True)
            note_text = str(row[self.notes_col_orig]) if pd.notna(row[self.notes_col_orig]) else "No note content."
            
            # Simple way to add subheaders if notes use markdown, otherwise just display
            # For more complex parsing, would need dedicated logic
            if "##" in note_text or "**" in note_text: # Basic check for markdown-like structures
                 st.markdown(note_text, unsafe_allow_html=True) # Render markdown
            else: # Display as plain text, but allow line breaks
                 st.text_area("", value=note_text, height=150, disabled=True, label_visibility="collapsed", key=f"note_text_{index}_{start_idx}")


            st.markdown("</div>", unsafe_allow_html=True) # Close note-card-content
            st.markdown("</div>", unsafe_allow_html=True) # Close note-card
            # st.markdown("---") # Optional separator between cards

        logger.debug("NotesViewer rendering complete.")

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.title("Test Notes Viewer Component (Card Layout)")
    
    # Font Awesome CDN for icons (add to your main app.py if not already there)
    st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">', unsafe_allow_html=True)


    mock_notes_data = {
        EXPECTED_COLUMNS['date']: pd.to_datetime(['2023-01-02 14:00', '2023-01-01 10:00', '2023-01-03 11:45', '2023-01-01 12:30', '2023-01-05 09:15', '2023-01-06 10:00', '2023-01-07 11:00', '2023-01-08 12:00']),
        EXPECTED_COLUMNS['symbol']: ['USDJPY', 'EURUSD', 'AUDUSD', 'GBPUSD', 'EURUSD', 'USDCAD', 'NZDUSD', 'CHFJPY'],
        EXPECTED_COLUMNS['strategy']: ['Scalp V1', 'TrendFollow', 'MeanRev', 'Breakout', 'TrendFollow', 'Scalp V2', 'MeanRev', 'Breakout'],
        EXPECTED_COLUMNS['pnl']: [90.75, 150.25, -30.20, 0, 220.00, -50.00, 75.00, -10.00],
        EXPECTED_COLUMNS['notes']: [
            "Quick scalp during Tokyo session. Good momentum.", 
            "Good entry on pullback. Followed plan. ## Lessons\n* Stick to TP.", 
            "Small loss, market was choppy. **Considered:** Waiting for confirmation.", 
            "Breakeven trade. Exited early due to news announcement.", 
            "Target reached. Excellent R:R on this one. \n\n### Pre-Market\nIdentified key support.",
            "Another scalp, this time less successful. Market reversed quickly.",
            "Mean reversion play worked out. Entered near the band.",
            "Breakout failed, got stopped out. Maybe waited too long."
            ]
    }
    mock_notes_df = pd.DataFrame(mock_notes_data)

    st.write("### Default View (5 items per page)")
    notes_viewer_default = NotesViewerComponent(
        mock_notes_df.copy(),
        strategy_col=EXPECTED_COLUMNS.get('strategy') # Pass strategy column
    )
    notes_viewer_default.render()

    st.write("### Sorted by PnL, Ascending (3 items per page)")
    notes_viewer_pnl = NotesViewerComponent(
        mock_notes_df.copy(),
        strategy_col=EXPECTED_COLUMNS.get('strategy'),
        default_sort_by_display="PnL",
        default_sort_ascending=True,
        items_per_page=3
    )
    notes_viewer_pnl.render()
