# components/column_mapper_ui.py
"""
Component for allowing users to map their uploaded CSV columns
to the application's expected conceptual columns, with data preview,
enhanced auto-mapping, data type validation, categorized display,
and confirm buttons at top and bottom.
"""
import streamlit as st
import pandas as pd
from typing import List, Dict, Optional, Any, OrderedDict
from collections import OrderedDict # To maintain category order
from io import BytesIO
from thefuzz import fuzz
import re

try:
    # Attempt to import from config, provide fallbacks if not found
    from config import (
        APP_TITLE,
        CONCEPTUAL_COLUMNS,
        CONCEPTUAL_COLUMN_TYPES,
        CONCEPTUAL_COLUMN_SYNONYMS,
        CRITICAL_CONCEPTUAL_COLUMNS,
        CONCEPTUAL_COLUMN_CATEGORIES # New: For grouping
    )
except ImportError:
    APP_TITLE = "TradingDashboard_Default"
    CONCEPTUAL_COLUMNS = {
        "date": "Trade Date/Time", "pnl": "Profit or Loss (PnL)",
        "strategy": "Strategy Name", "symbol": "Trading Symbol",
        "r_r_csv_num": "Risk:Reward Ratio", "notes": "Trade Notes/Lessons",
        "duration_minutes": "Duration (Minutes)", "risk_pct": "Risk Percentage",
        "entry_price": "Entry Price", "exit_price": "Exit Price", "quantity": "Quantity/Size",
        "commission": "Commission", "fees": "Fees", "tags": "Tags/Labels"
    }
    CONCEPTUAL_COLUMN_TYPES = {
        "date": "datetime", "pnl": "numeric", "strategy": "text", "symbol": "text",
        "r_r_csv_num": "numeric", "notes": "text", "duration_minutes": "numeric",
        "risk_pct": "numeric", "entry_price": "numeric", "exit_price": "numeric",
        "quantity": "numeric", "commission": "numeric", "fees": "numeric", "tags": "text"
    }
    CONCEPTUAL_COLUMN_SYNONYMS = {
        "strategy": ["trade_model", "system_name"], "r_r_csv_num": ["r_r", "risk_reward"],
        "pnl": ["profit_loss", "net_result"], "date": ["datetime", "trade_time"],
        "notes": ["comments", "journal_entry", "lesson_learned"],
        "duration_minutes": ["trade_duration_min", "holding_time_mins"],
        "risk_pct": ["risk_percent", "pct_risk"], "tags": ["label", "category_tag"],
        "quantity": ["trade_size", "lot_size", "amount", "vol", "volume"] 
    }
    CRITICAL_CONCEPTUAL_COLUMNS = ["date", "pnl", "symbol"]
    # New: Fallback categories if not in config
    CONCEPTUAL_COLUMN_CATEGORIES = OrderedDict([
        ("Core Trade Information", ["date", "symbol", "entry_price", "exit_price", "quantity"]),
        ("Performance & Strategy", ["pnl", "strategy", "r_r_csv_num", "duration_minutes"]),
        ("Risk & Financials", ["risk_pct", "commission", "fees"]),
        ("Qualitative & Categorization", ["notes", "tags"])
    ])
    print("Warning (column_mapper_ui.py): Could not import some/all configurations from config. Using fallback values.")

import logging
logger = logging.getLogger(APP_TITLE)

class ColumnMapperUI:
    def __init__(
        self,
        uploaded_file_name: str,
        uploaded_file_bytes: Optional[BytesIO],
        csv_headers: List[str],
        conceptual_columns_map: Dict[str, str], 
        conceptual_column_types: Dict[str, str],
        conceptual_column_synonyms: Dict[str, List[str]],
        critical_conceptual_cols: List[str],
        conceptual_column_categories: OrderedDict[str, List[str]] 
    ):
        self.uploaded_file_name = uploaded_file_name
        self.uploaded_file_bytes = uploaded_file_bytes
        self.csv_headers = [""] + csv_headers  
        self.raw_csv_headers = csv_headers
        self.conceptual_columns_map = conceptual_columns_map 
        self.conceptual_column_types = conceptual_column_types
        self.conceptual_column_synonyms = conceptual_column_synonyms
        self.critical_conceptual_cols = critical_conceptual_cols if critical_conceptual_cols else []
        self.conceptual_column_categories = conceptual_column_categories 
        self.mapping: Dict[str, Optional[str]] = {}
        self.preview_df: Optional[pd.DataFrame] = None

        if self.uploaded_file_bytes:
            try:
                self.uploaded_file_bytes.seek(0)
                self.preview_df = pd.read_csv(self.uploaded_file_bytes, nrows=5)
                self.uploaded_file_bytes.seek(0) 
            except Exception as e:
                logger.error(f"ColumnMapperUI: Error reading CSV for preview: {e}")
                self.preview_df = None
        
        logger.debug(f"ColumnMapperUI initialized for file: {self.uploaded_file_name}. Raw CSV Headers: {self.raw_csv_headers}")
        if isinstance(self.conceptual_columns_map, OrderedDict):
            logger.debug("conceptual_columns_map is OrderedDict.")
        else:
            logger.warning("conceptual_columns_map is NOT OrderedDict, auto-mapping general loop order may vary.")


    def _normalize_header(self, header: str) -> str:
        if not isinstance(header, str):
            header = str(header)
        normalized = header.strip().lower()
        normalized = normalized.replace(':', '_').replace('%', 'pct')
        normalized = re.sub(r'[\s\-\./\(\)]+', '_', normalized) 
        normalized = re.sub(r'_+', '_', normalized).strip('_') 
        return normalized

    def _attempt_automatic_mapping(self) -> Dict[str, Optional[str]]:
        auto_mapping: Dict[str, Optional[str]] = {}
        normalized_csv_headers_map = {self._normalize_header(h): h for h in self.raw_csv_headers}
        logger.debug(f"Normalized CSV Headers Map: {normalized_csv_headers_map}")
        used_csv_headers = set() 

        specific_csv_header_targets = {
            "trade_model": "strategy", "r_r": "r_r_csv_num", "pnl": "pnl", "date": "date",
            "symbol_1": "symbol", "lesson_learned": "notes", "duration_mins": "duration_minutes",
            "risk_pct": "risk_pct", "entry": "entry_price", "exit": "exit_price", 
            "size": "quantity"  
        }
        logger.debug(f"Starting specific rule mapping. Current auto_mapping: {auto_mapping}, used_csv_headers: {used_csv_headers}")
        for norm_specific_csv, target_conceptual_key in specific_csv_header_targets.items():
            display_name_for_log = self.conceptual_columns_map.get(target_conceptual_key, f"'{target_conceptual_key}' (Key not in conceptual_columns_map!)")
            logger.debug(f"  Specific rule check: Normalized CSV target='{norm_specific_csv}', Conceptual key='{target_conceptual_key}' (Display: '{display_name_for_log}')")
            
            if norm_specific_csv in normalized_csv_headers_map:
                original_csv_header = normalized_csv_headers_map[norm_specific_csv]
                logger.debug(f"    Found '{original_csv_header}' (normalizes to '{norm_specific_csv}') in CSV headers.")
                logger.debug(f"    Current state before check: '{original_csv_header}' in used_csv_headers? {original_csv_header in used_csv_headers}. Conceptual_key '{target_conceptual_key}' in auto_mapping? {target_conceptual_key in auto_mapping}.")

                if original_csv_header not in used_csv_headers and target_conceptual_key not in auto_mapping:
                    auto_mapping[target_conceptual_key] = original_csv_header
                    used_csv_headers.add(original_csv_header)
                    logger.info(f"    SUCCESS: Auto-mapped (Specific Rule '{norm_specific_csv}' -> '{target_conceptual_key}') CSV '{original_csv_header}' to Conceptual '{display_name_for_log}'")
                else:
                    logger.warning(f"    SKIPPED Specific Rule mapping for CSV target '{norm_specific_csv}' (orig: '{original_csv_header}') to Conceptual '{display_name_for_log}'. Reason: CSV header already used OR Conceptual key already mapped.")
            else:
                logger.debug(f"    Normalized CSV target '{norm_specific_csv}' not found in normalized_csv_headers_map for Conceptual '{display_name_for_log}'.")
        
        logger.debug(f"After specific rules. auto_mapping: {auto_mapping}, used_csv_headers: {used_csv_headers}")
        logger.debug("Starting general rule mapping (exact, synonym, fuzzy)...")

        for conceptual_key in self.conceptual_columns_map.keys():
            display_name_for_log = self.conceptual_columns_map.get(conceptual_key, conceptual_key)
            if conceptual_key in auto_mapping: 
                logger.debug(f"  Skipping general map for Conceptual '{display_name_for_log}' (key: {conceptual_key}): already in auto_mapping.")
                continue

            logger.debug(f"  Attempting general map for Conceptual '{display_name_for_log}' (key: {conceptual_key})")
            mapped_csv_header = None
            norm_conceptual_key = self._normalize_header(conceptual_key) 

            if norm_conceptual_key in normalized_csv_headers_map and \
               normalized_csv_headers_map[norm_conceptual_key] not in used_csv_headers:
                mapped_csv_header = normalized_csv_headers_map[norm_conceptual_key]
                logger.debug(f"    Exact match found: CSV '{mapped_csv_header}' for Conceptual '{display_name_for_log}'")
            
            if not mapped_csv_header and conceptual_key in self.conceptual_column_synonyms:
                logger.debug(f"    Checking synonyms for '{display_name_for_log}': {self.conceptual_column_synonyms[conceptual_key]}")
                for synonym in self.conceptual_column_synonyms[conceptual_key]:
                    norm_synonym = self._normalize_header(synonym) 
                    if norm_synonym in normalized_csv_headers_map and \
                       normalized_csv_headers_map[norm_synonym] not in used_csv_headers:
                        mapped_csv_header = normalized_csv_headers_map[norm_synonym]
                        logger.debug(f"    Synonym match found: CSV '{mapped_csv_header}' (from synonym '{synonym}') for Conceptual '{display_name_for_log}'")
                        break 
            
            FUZZY_MATCH_THRESHOLD = 85 
            if not mapped_csv_header:
                logger.debug(f"    No exact or synonym match for '{display_name_for_log}'. Trying fuzzy match (threshold: {FUZZY_MATCH_THRESHOLD}).")
                best_match_score = 0
                potential_header = None
                for norm_csv_h, original_csv_h in normalized_csv_headers_map.items():
                    if original_csv_h in used_csv_headers: continue 
                    score = fuzz.ratio(norm_conceptual_key, norm_csv_h)
                    # logger.debug(f"      Fuzzy: Conceptual '{norm_conceptual_key}' vs CSV norm '{norm_csv_h}' (orig: '{original_csv_h}') -> Score: {score}") # Can be too verbose
                    if score > best_match_score and score >= FUZZY_MATCH_THRESHOLD:
                        best_match_score = score
                        potential_header = original_csv_h
                if potential_header:
                    mapped_csv_header = potential_header
                    logger.debug(f"    Fuzzy match found: CSV '{mapped_csv_header}' for Conceptual '{display_name_for_log}' with score {best_match_score}")
            
            if mapped_csv_header:
                auto_mapping[conceptual_key] = mapped_csv_header
                used_csv_headers.add(mapped_csv_header)
                logger.info(f"    SUCCESS: Auto-mapped (General Rule) CSV '{mapped_csv_header}' to Conceptual '{display_name_for_log}'")
            elif conceptual_key in self.critical_conceptual_cols:
                logger.warning(f"    FAILED: Could not auto-map critical Conceptual column: '{display_name_for_log}' (key: {conceptual_key}) by any general rule.")
            else:
                logger.debug(f"    No general mapping found for optional Conceptual '{display_name_for_log}'.")
        
        logger.info(f"Final auto_mapping result: {auto_mapping}")
        return auto_mapping

    def _infer_column_data_type(self, csv_column_name: str) -> str:
        if self.preview_df is None or csv_column_name not in self.preview_df.columns:
            return "unknown"
        
        column_sample = self.preview_df[csv_column_name].dropna().convert_dtypes()
        if column_sample.empty:
            return "empty" 

        try:
            numeric_sample = pd.to_numeric(column_sample)
            if (numeric_sample % 1 == 0).all():
                return "integer"
            return "float"
        except (ValueError, TypeError):
            pass 

        try:
            pd.to_datetime(column_sample, errors='raise', infer_datetime_format=True)
            return "datetime"
        except (ValueError, TypeError, pd.errors.ParserError):
            pass 

        return "text" 

    def render(self) -> Optional[Dict[str, Optional[str]]]:
        st.markdown("<div class='column-mapper-container'>", unsafe_allow_html=True)
        st.markdown(f"<h3 class='component-subheader'>Map Columns for '{self.uploaded_file_name}'</h3>", unsafe_allow_html=True)
        
        initial_mapping = self._attempt_automatic_mapping()

        st.markdown("<div class='data-preview-container'>", unsafe_allow_html=True)
        if self.preview_df is not None and not self.preview_df.empty:
            df_to_display = self.preview_df 

            try:
                ordered_conceptual_keys = list(self.conceptual_columns_map.keys())
                mapped_csv_cols_ordered = []
                seen_mapped_csv_cols = set()

                for conceptual_key in ordered_conceptual_keys:
                    mapped_csv_header = initial_mapping.get(conceptual_key)
                    if mapped_csv_header and mapped_csv_header in self.preview_df.columns and mapped_csv_header not in seen_mapped_csv_cols:
                        mapped_csv_cols_ordered.append(mapped_csv_header)
                        seen_mapped_csv_cols.add(mapped_csv_header)

                remaining_original_cols = [
                    col for col in self.preview_df.columns if col not in seen_mapped_csv_cols
                ]
                
                final_display_order = mapped_csv_cols_ordered + remaining_original_cols
                
                if set(final_display_order) == set(self.preview_df.columns) and len(final_display_order) == len(self.preview_df.columns):
                    df_to_display = self.preview_df[final_display_order]
                else:
                    logger.warning(
                        f"Column reordering for preview of '{self.uploaded_file_name}' resulted in mismatched column sets. Displaying in original order."
                    )
            except Exception as e:
                logger.error(f"Error during column reordering for preview of '{self.uploaded_file_name}': {e}. Displaying in original order.")

            st.markdown("<p class='data-preview-title'>Data Preview (First 5 Rows):</p>", unsafe_allow_html=True)
            st.dataframe(df_to_display, hide_index=True, use_container_width=True)
        else:
            st.markdown("<div class='data-preview-placeholder'>Data preview is not available or the file is empty.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True) 
        
        st.markdown(
            "<div class='mapper-instructions'>"
            "<span class='instruction-icon'>ℹ️</span> " 
            "Map your CSV columns to the application's expected fields. "
            "Fields marked with a red asterisk (<strong>*</strong>) are critical. " 
            "A warning icon (⚠️) next to a selection indicates a potential data type mismatch."
            "</div>", unsafe_allow_html=True)
        
        form_key = f"column_mapping_form_{self.uploaded_file_name.replace('.', '_').replace(' ', '_')}"
        with st.form(key=form_key):
            cols_top_button = st.columns([0.75, 0.25]) 
            with cols_top_button[1]:
                 submit_button_top = st.form_submit_button("Confirm Mapping", use_container_width=True, type="primary")
            
            st.markdown("<hr class='styled-hr'>", unsafe_allow_html=True)

            if not self.conceptual_column_categories:
                st.warning("Column categories are not defined. Displaying all columns together for mapping.")
                self._render_mapping_selectboxes(list(self.conceptual_columns_map.keys()), initial_mapping)
            else:
                for category_name, conceptual_keys_in_category in self.conceptual_column_categories.items():
                    valid_keys_in_category = [
                        key for key in conceptual_keys_in_category if key in self.conceptual_columns_map
                    ]
                    if not valid_keys_in_category:
                        logger.warning(f"Category '{category_name}' has no valid conceptual columns to display in mapper.")
                        continue

                    has_critical = any(key in self.critical_conceptual_cols for key in valid_keys_in_category)
                    
                    expander_label_html = f"<strong>{category_name}</strong>" if has_critical else category_name
                    if has_critical:
                         expander_label_html += " <span class='critical-marker'>*</span>" 

                    open_by_default = False
                    for key in valid_keys_in_category:
                        if key in self.critical_conceptual_cols and not initial_mapping.get(key):
                            open_by_default = True
                            break
                    
                    st.markdown(f"<h6>{expander_label_html}</h6>", unsafe_allow_html=True)
                    with st.expander("View/Edit Mappings", expanded=open_by_default):
                        self._render_mapping_selectboxes(valid_keys_in_category, initial_mapping)
            
            st.markdown("<hr class='styled-hr'>", unsafe_allow_html=True)
            _, col_btn_mid, _ = st.columns([0.3, 0.4, 0.3]) 
            with col_btn_mid:
                submit_button_bottom = st.form_submit_button("Confirm Column Mapping", use_container_width=True, type="primary")

        if submit_button_top or submit_button_bottom:
            missing_critical = [
                self.conceptual_columns_map.get(k, k) for k in self.critical_conceptual_cols if not self.mapping.get(k)
            ]
            if missing_critical:
                st.error(f"Critical fields not mapped: {', '.join(missing_critical)}. Please map these fields to proceed.")
                st.markdown("</div>", unsafe_allow_html=True); return None
            
            csv_to_critical_map: Dict[str, List[str]] = {}
            for conc_key, csv_header in self.mapping.items():
                if csv_header and conc_key in self.critical_conceptual_cols:
                    csv_to_critical_map.setdefault(csv_header, []).append(self.conceptual_columns_map.get(conc_key, conc_key))
            
            has_critical_duplicates = False
            for csv_h, mapped_fields in csv_to_critical_map.items():
                if len(mapped_fields) > 1:
                    st.error(
                        f"CSV column '{csv_h}' is mapped to multiple critical application fields: {', '.join(mapped_fields)}. "
                        "Each critical field requires a unique CSV column."
                    )
                    has_critical_duplicates = True
            if has_critical_duplicates:
                st.markdown("</div>", unsafe_allow_html=True); return None

            logger.info(f"Column mapping confirmed for '{self.uploaded_file_name}': {self.mapping}")
            st.success(f"Column mapping confirmed for '{self.uploaded_file_name}'.")
            st.markdown("</div>", unsafe_allow_html=True)
            return {k: v for k, v in self.mapping.items() if v}
        
        st.markdown("</div>", unsafe_allow_html=True)
        return None

    def _render_mapping_selectboxes(self, conceptual_keys_to_render: List[str], initial_mapping: Dict[str, Optional[str]]):
        cols_ui = st.columns(2)
        col_idx = 0
        for conceptual_key in conceptual_keys_to_render:
            if conceptual_key not in self.conceptual_columns_map:
                # This log is important if a key from categories is missing in the main map
                logger.warning(f"RenderSelectbox: Conceptual key '{conceptual_key}' from categories NOT FOUND in self.conceptual_columns_map. Skipping selectbox.")
                continue

            conceptual_desc = self.conceptual_columns_map[conceptual_key]
            target_container = cols_ui[col_idx % 2]
            col_idx += 1
            
            with target_container:
                is_critical = conceptual_key in self.critical_conceptual_cols
                label_html = f"{conceptual_desc}"
                if is_critical:
                    label_html += " <span class='critical-marker'>*</span>" 
                
                default_csv_header = initial_mapping.get(conceptual_key)
                
                # --- ADDED DIAGNOSTIC LOG ---
                if "quantity" in conceptual_key.lower() or "size" in conceptual_desc.lower() or conceptual_key == "quantity":
                    logger.debug(
                        f"RenderSelectbox for '{conceptual_desc}' (Key: '{conceptual_key}'): "
                        f"Retrieved default_csv_header='{default_csv_header}' from initial_mapping.get('{conceptual_key}'). "
                        f"Full initial_mapping: {initial_mapping}"
                    )
                # --- END OF DIAGNOSTIC LOG ---

                default_index = 0
                if default_csv_header and default_csv_header in self.csv_headers:
                    try:
                        default_index = self.csv_headers.index(default_csv_header)
                    except ValueError: 
                        logger.error(f"RenderSelectbox: Error finding index for default_csv_header '{default_csv_header}' (for conceptual key '{conceptual_key}') in self.csv_headers. Defaulting to index 0.")
                        default_index = 0
                elif default_csv_header: # Header was in initial_mapping but not in self.csv_headers (should not happen if CSV parsing is correct)
                     logger.warning(f"RenderSelectbox: default_csv_header '{default_csv_header}' for conceptual key '{conceptual_key}' was in initial_mapping but NOT FOUND in self.csv_headers. This is unexpected. CSV Headers: {self.csv_headers}")


                selectbox_key = f"map_{self.uploaded_file_name.replace('.', '_').replace(' ', '_')}_{conceptual_key}"
                
                st.markdown(f"<label class='selectbox-label' for='{selectbox_key}'>{label_html}</label>", unsafe_allow_html=True)
                selected_csv_col = st.selectbox(
                    label="", 
                    options=self.csv_headers, index=default_index,
                    key=selectbox_key,
                    label_visibility="collapsed",
                    help=(
                        f"Select the CSV column for '{conceptual_desc}'. "
                        f"Expected type: '{self.conceptual_column_types.get(conceptual_key, 'any')}'. "
                        f"{'This field is critical.' if is_critical else 'This field is optional.'}"
                    )
                )
                self.mapping[conceptual_key] = selected_csv_col if selected_csv_col else None

                if selected_csv_col:
                    inferred_type = self._infer_column_data_type(selected_csv_col)
                    expected_type = self.conceptual_column_types.get(conceptual_key, "any")
                    type_mismatch = False

                    if expected_type == "numeric" and inferred_type not in ["integer", "float", "empty", "unknown"]:
                        type_mismatch = True
                    elif expected_type == "datetime" and inferred_type not in ["datetime", "empty", "unknown"]:
                        type_mismatch = True
                    
                    if type_mismatch:
                        st.markdown(
                            f"<small class='type-mismatch-warning'>⚠️ Expected '{expected_type}', "
                            f"but data in '{selected_csv_col}' looks like '{inferred_type}'.</small>", 
                            unsafe_allow_html=True
                        )

# --- Main execution block for testing this component ---
if __name__ == "__main__":
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed", page_title="Column Mapper Test")
    
    # Configure logger for testing
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger_test = logging.getLogger(APP_TITLE) 
    logger_test.info("Starting ColumnMapperUI test run...")


    try:
        with open("style.css") as f: 
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("style.css not found. Using default Streamlit styling for component test.")

    st.title("Test Categorized Column Mapper UI")
    st.markdown("This page demonstrates the `ColumnMapperUI` component with categorized fields, styled instructions, and reordered data preview. Check console for detailed logs.")

    _MOCK_CONCEPTUAL_COLUMNS = OrderedDict([
        ("date", "Trade Date/Time"), 
        ("symbol", "Trading Symbol"),
        ("entry_price", "Entry Price"), 
        ("exit_price", "Exit Price"), 
        ("quantity", "Quantity/Size"), # Key is 'quantity', display is 'Quantity/Size'
        ("pnl", "Profit or Loss (PnL)"), 
        ("strategy", "Strategy Name"), 
        ("r_r_csv_num", "Risk:Reward Ratio"), 
        ("duration_minutes", "Duration (Mins)"), 
        ("risk_pct", "Risk %"), 
        ("commission", "Commission Cost"),
        ("fees", "Total Fees"), 
        ("notes", "Trade Notes"),
        ("tags", "Custom Tags")
    ])
    _MOCK_CONCEPTUAL_COLUMN_TYPES = {
        "date": "datetime", "pnl": "numeric", "strategy": "text", "symbol": "text",
        "r_r_csv_num": "numeric", "notes": "text", "duration_minutes": "numeric", "risk_pct": "numeric",
        "entry_price": "numeric", "exit_price": "numeric", "quantity": "numeric", 
        "commission": "numeric", "fees": "numeric", "tags": "text"
    }
    _MOCK_CONCEPTUAL_COLUMN_SYNONYMS = {
        "strategy": ["trade_model", "system"], "r_r_csv_num": ["r_r", "risk_reward_ratio"],
        "pnl": ["profit", "loss", "netpl"], "date": ["trade_date", "timestamp"],
        "notes": ["comment", "lessons", "journal"], "duration_minutes": ["holding_time", "duration_min"],
        "risk_pct": ["risk_percentage", "percent_risk"], "tags": ["label", "trade_category"],
        "quantity": ["trade_size", "lot_size", "amount", "vol", "volume"] 
    }
    _MOCK_CRITICAL_CONCEPTUAL_COLUMNS = ["date", "pnl", "symbol", "entry_price", "exit_price", "quantity"]
    
    _MOCK_CONCEPTUAL_COLUMN_CATEGORIES = OrderedDict([
        ("Core Trade Data", ["date", "symbol", "entry_price", "exit_price", "quantity"]), # 'quantity' is here
        ("Performance Metrics", ["pnl", "strategy", "r_r_csv_num", "duration_minutes"]),
        ("Financials & Risk", ["risk_pct", "commission", "fees"]),
        ("Additional Information", ["notes", "tags"])
    ])

    sample_csv_content_1 = """Trade ID,Date,Entry Time,Size,Entry,Take Profit,Stop Loss,Exit,Candle Count,Exit Type,Trade Model ,PnL,R:R,Duration (mins),Risk %,Symbol 1,Lesson Learned,Tags,Commission,Total Fees
1,2023-01-01 10:00:00,10:00:00,10000,1.1000,1.1050,1.0950,1.1050,5,TP,Scalp V1,50.00,2.0,15,1,EURUSD,Good exit,News,2.50,0.50
"""
    mock_uploaded_file_bytes_1 = BytesIO(sample_csv_content_1.encode('utf-8'))
    mock_csv_headers_1 = sample_csv_content_1.splitlines()[0].split(',')

    st.write("### Scenario 1: Mapping for 'sample_trades_with_Size.csv'")
    st.caption("This CSV includes a 'Size' column, which should map to 'Quantity/Size'.")
    mapper_ui_instance_1 = ColumnMapperUI(
        uploaded_file_name="sample_trades_with_Size.csv", 
        uploaded_file_bytes=mock_uploaded_file_bytes_1, 
        csv_headers=mock_csv_headers_1,
        conceptual_columns_map=_MOCK_CONCEPTUAL_COLUMNS, 
        conceptual_column_types=_MOCK_CONCEPTUAL_COLUMN_TYPES,
        conceptual_column_synonyms=_MOCK_CONCEPTUAL_COLUMN_SYNONYMS, 
        critical_conceptual_cols=_MOCK_CRITICAL_CONCEPTUAL_COLUMNS,
        conceptual_column_categories=_MOCK_CONCEPTUAL_COLUMN_CATEGORIES
    )
    mapping_result_1 = mapper_ui_instance_1.render()
    if mapping_result_1 is not None:
        st.success("Mapping Confirmed (Scenario 1):")
        st.json(mapping_result_1)

    logger_test.info("ColumnMapperUI test run finished.")
