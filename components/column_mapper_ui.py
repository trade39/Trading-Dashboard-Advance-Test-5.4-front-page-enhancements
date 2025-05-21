# components/column_mapper_ui.py
"""
Component for allowing users to map their uploaded CSV columns
to the application's expected conceptual columns, with data preview,
enhanced auto-mapping, data type validation, categorized display,
and confirm buttons at top and bottom.

Updates:
- "Trade Size/Quantity" is auto-mapped to the "Size" CSV column.
- Sleek, professional dark-themed UI inspired by provided screenshot.
- Resolved StreamlitDuplicateElementKey by adding unique keys to form_submit_buttons within the same form.
- Removed unsupported 'type' parameter from st.form_submit_button (from previous version).
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Optional, Any
from collections import OrderedDict
from io import BytesIO
from thefuzz import fuzz
import re
import logging

SESSION_MAPPINGS_KEY = "column_mappings_v2" # Keeps v2 from previous fix

try:
    from config import (
        APP_TITLE,
        CONCEPTUAL_COLUMNS,
        CONCEPTUAL_COLUMN_TYPES,
        CONCEPTUAL_COLUMN_SYNONYMS,
        CRITICAL_CONCEPTUAL_COLUMNS,
        CONCEPTUAL_COLUMN_CATEGORIES
    )
except ImportError:
    APP_TITLE = "TradingDashboard_Default"
    CONCEPTUAL_COLUMNS = {
        "date": "Trade Date/Time", "pnl": "Profit or Loss (PnL)",
        "strategy": "Strategy Name", "symbol": "Trading Symbol",
        "r_r_csv_num": "Risk:Reward Ratio", "notes": "Trade Notes/Lessons",
        "duration_minutes": "Duration (Minutes)", "risk_pct": "Risk Percentage",
        "entry_price": "Entry Price", "exit_price": "Exit Price",
        "quantity": "Trade Size/Quantity",
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
        "quantity": ["size", "trade_quantity", "vol", "volume"]
    }
    CRITICAL_CONCEPTUAL_COLUMNS = ["date", "pnl", "symbol"]
    CONCEPTUAL_COLUMN_CATEGORIES = OrderedDict([
        ("Core Trade Information", ["date", "symbol", "entry_price", "exit_price", "quantity"]),
        ("Performance & Strategy", ["pnl", "strategy", "r_r_csv_num", "duration_minutes"]),
        ("Risk & Financials", ["risk_pct", "commission", "fees"]),
        ("Qualitative & Categorization", ["notes", "tags"])
    ])

VALID_CRITICAL_COLUMNS = [col for col in CRITICAL_CONCEPTUAL_COLUMNS if col in CONCEPTUAL_COLUMNS]
if len(VALID_CRITICAL_COLUMNS) != len(CRITICAL_CONCEPTUAL_COLUMNS):
    logging.warning("Some CRITICAL_CONCEPTUAL_COLUMNS are not in CONCEPTUAL_COLUMNS.")

EXPECTED_COLUMNS = list(VALID_CRITICAL_COLUMNS)
OPTIONAL_COLUMNS = [k for k in CONCEPTUAL_COLUMNS.keys() if k not in EXPECTED_COLUMNS]

@st.cache_data
def get_cached_dataframe_columns(uploaded_file_bytes: BytesIO) -> List[str]:
    if not uploaded_file_bytes:
        return []
    try:
        uploaded_file_bytes.seek(0)
        df = pd.read_csv(uploaded_file_bytes, nrows=0)
        uploaded_file_bytes.seek(0)
        return list(df.columns)
    except Exception as e:
        logging.error(f"Error reading CSV headers for caching: {e}")
        return []

logger = logging.getLogger(APP_TITLE if 'APP_TITLE' in globals() else __name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

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
        conceptual_column_categories: OrderedDict
    ):
        self.uploaded_file_name = uploaded_file_name
        self.uploaded_file_bytes = uploaded_file_bytes
        self.csv_headers = [""] + (csv_headers if csv_headers else [])
        self.raw_csv_headers = csv_headers if csv_headers else []
        self.conceptual_columns_map = conceptual_columns_map
        self.conceptual_column_types = conceptual_column_types
        self.conceptual_column_synonyms = conceptual_column_synonyms
        self.critical_conceptual_cols = critical_conceptual_cols if critical_conceptual_cols else []
        self.conceptual_column_categories = conceptual_column_categories
        self.mapping: Dict[str, Optional[str]] = {}
        self.preview_df: Optional[pd.DataFrame] = None
        # Ensure session_key_prefix is robust against various characters in filenames
        safe_file_name_part = re.sub(r'[^a-zA-Z0-9_]', '_', self.uploaded_file_name)
        self.session_key_prefix = f"col_map_{safe_file_name_part}"


        if self.uploaded_file_bytes:
            try:
                self.uploaded_file_bytes.seek(0)
                self.preview_df = pd.read_csv(self.uploaded_file_bytes, nrows=10)
                self.uploaded_file_bytes.seek(0)
            except pd.errors.EmptyDataError:
                logger.warning(f"Uploaded file '{self.uploaded_file_name}' is empty or has no data to preview.")
                self.preview_df = pd.DataFrame()
            except Exception as e:
                logger.error(f"Error reading CSV for preview ('{self.uploaded_file_name}'): {e}")
                st.warning(f"Could not generate data preview for '{self.uploaded_file_name}'. The file might be corrupted or in an unexpected format.")
                self.preview_df = None

        logger.debug(f"ColumnMapperUI initialized for file: {self.uploaded_file_name} with session prefix: {self.session_key_prefix}")

    def _normalize_header(self, header: Any) -> str:
        if not isinstance(header, str):
            header = str(header)
        normalized = header.strip().lower()
        normalized = normalized.replace(':', '_').replace('%', 'pct')
        normalized = re.sub(r'[\s\-\./\(\)]+', '_', normalized)
        normalized = re.sub(r'_+', '_', normalized).strip('_')
        return normalized

    def _attempt_automatic_mapping(self) -> Dict[str, Optional[str]]:
        auto_mapping: Dict[str, Optional[str]] = {}
        if not self.raw_csv_headers:
            return auto_mapping

        normalized_csv_headers_map = {self._normalize_header(h): h for h in self.raw_csv_headers}
        used_csv_headers = set()

        if "quantity" in self.conceptual_columns_map:
            normalized_target_csv_header = "size"
            if normalized_target_csv_header in normalized_csv_headers_map:
                original_csv_header = normalized_csv_headers_map[normalized_target_csv_header]
                if original_csv_header not in used_csv_headers:
                    auto_mapping["quantity"] = original_csv_header
                    used_csv_headers.add(original_csv_header)
                    logger.info(f"(Specific Rule) Auto-mapped conceptual 'quantity' ({self.conceptual_columns_map.get('quantity', 'Trade Size/Quantity')}) to CSV column '{original_csv_header}'")

        specific_csv_header_targets = {
            "trade_model": "strategy", "r_r": "r_r_csv_num", "pnl": "pnl", "date": "date",
            "datetime": "date", "trade_time": "date",
            "symbol_1": "symbol", "ticker": "symbol",
            "lesson_learned": "notes", "comments": "notes",
            "duration_mins": "duration_minutes", "holding_time_mins": "duration_minutes",
            "risk_pct": "risk_pct", "risk_percent": "risk_pct",
            "entry": "entry_price", "entryprice": "entry_price",
            "exit": "exit_price", "exitprice": "exit_price",
            "trade_quantity": "quantity", "vol": "quantity", "volume": "quantity"
        }

        for norm_specific_csv, target_conceptual_key in specific_csv_header_targets.items():
            if target_conceptual_key not in self.conceptual_columns_map:
                continue
            if norm_specific_csv in normalized_csv_headers_map:
                original_csv_header = normalized_csv_headers_map[norm_specific_csv]
                if original_csv_header not in used_csv_headers and target_conceptual_key not in auto_mapping:
                    auto_mapping[target_conceptual_key] = original_csv_header
                    used_csv_headers.add(original_csv_header)
                    logger.info(f"Auto-mapped (specific target) CSV '{original_csv_header}' to conceptual '{target_conceptual_key}'")

        for conceptual_key in self.conceptual_columns_map.keys():
            if conceptual_key in auto_mapping:
                continue

            mapped_csv_header: Optional[str] = None
            norm_conceptual_key_text = self._normalize_header(self.conceptual_columns_map[conceptual_key])
            norm_conceptual_key_itself = self._normalize_header(conceptual_key)

            for norm_h, orig_h in normalized_csv_headers_map.items():
                if orig_h in used_csv_headers: continue
                if norm_h == norm_conceptual_key_text or norm_h == norm_conceptual_key_itself:
                    mapped_csv_header = orig_h
                    break
            
            if not mapped_csv_header and conceptual_key in self.conceptual_column_synonyms:
                for synonym in self.conceptual_column_synonyms[conceptual_key]:
                    norm_synonym = self._normalize_header(synonym)
                    if norm_synonym in normalized_csv_headers_map:
                        original_csv_header = normalized_csv_headers_map[norm_synonym]
                        if original_csv_header not in used_csv_headers:
                            mapped_csv_header = original_csv_header
                            break
            
            FUZZY_MATCH_THRESHOLD = 85
            if not mapped_csv_header:
                best_match_score = 0
                potential_header = None
                texts_to_match_against = [norm_conceptual_key_text, norm_conceptual_key_itself]
                
                for norm_csv_h, original_csv_h in normalized_csv_headers_map.items():
                    if original_csv_h in used_csv_headers:
                        continue
                    for conceptual_text_variant in texts_to_match_against:
                        score = fuzz.ratio(conceptual_text_variant, norm_csv_h)
                        if score > best_match_score and score >= FUZZY_MATCH_THRESHOLD:
                            best_match_score = score
                            potential_header = original_csv_h
                
                if potential_header:
                    mapped_csv_header = potential_header

            if mapped_csv_header:
                auto_mapping[conceptual_key] = mapped_csv_header
                used_csv_headers.add(mapped_csv_header)
                logger.info(f"Auto-mapped (general) conceptual '{conceptual_key}' to CSV '{mapped_csv_header}'")
            elif conceptual_key in self.critical_conceptual_cols:
                logger.warning(f"Could not auto-map critical conceptual column: '{conceptual_key}' ({self.conceptual_columns_map.get(conceptual_key, '')})")
        
        return auto_mapping

    def _infer_column_data_type(self, csv_column_name: str) -> str:
        if self.preview_df is None or csv_column_name not in self.preview_df.columns:
            return "unknown"
        
        column_sample = self.preview_df[csv_column_name].dropna()
        if column_sample.empty:
            return "empty"

        try:
            numeric_series = pd.to_numeric(column_sample)
            if (numeric_series % 1 == 0).all():
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
        st.markdown(
            """
            <style>
            .column-mapper-container {
                background: #21232b; border-radius: 10px; padding: 1.8em 1.5em 1.3em 1.5em;
                margin-top: 1.5em; margin-bottom: 1.5em; color: #e0e0e0; 
                box-shadow: 0 6px 18px 0 rgba(0,0,0,0.28);
            }
            .column-mapper-container .component-subheader {
                color: #ffffff; font-size: 1.6em; font-weight: 700; margin-bottom: .7em;
            }
            .column-mapper-container .data-preview-title {
                color: #c0c0c0; font-weight: 600; margin-bottom: 0.5em; font-size: 1.1em;
            }
            .column-mapper-container .mapper-instructions {
                color: #f9d47f; font-size: 1.08em; margin-bottom: 1.1em; margin-top: .7em; line-height: 1.5;
            }
            .column-mapper-container .type-mismatch-warning { color: #f4b400!important; font-size:0.9em; display: block; margin-top: 2px;}
            .column-mapper-container .styled-hr { border: 1px solid #2f3341; margin: 1.8em 0; }
            .column-mapper-container .stExpanderHeader { font-size: 1.08em !important; font-weight: 600; color: #d0d0d0;}
            .column-mapper-container .stSelectbox label { color: #e3e3e3 !important; font-weight: 500; font-size: 0.95rem;}
            .column-mapper-container .stDataFrame { background: #181b22; border-radius: 7px; }
            .column-mapper-container .stButton>button {
                background: linear-gradient(90deg,#0059b2,#002e63); color: #fff; border-radius: 7px;
                font-weight: 700; border: none; padding: 0.5em 1em;
            }
            .column-mapper-container .stButton>button:hover { background: #004080; color: #ffe784; }
            html, body, [data-testid="stAppViewContainer"], [data-testid="stReportViewContainer"] {
                background-color: #16181d !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<div class='column-mapper-container'>", unsafe_allow_html=True)
        st.markdown(f"<div class='component-subheader'>Map Columns for '{self.uploaded_file_name}'</div>", unsafe_allow_html=True)

        if self.preview_df is not None and not self.preview_df.empty:
            st.markdown(f"<p class='data-preview-title'>Data Preview (First {len(self.preview_df)} Rows):</p>", unsafe_allow_html=True) # Corrected f-string
            st.dataframe(self.preview_df, hide_index=True, use_container_width=True)
        elif self.preview_df is not None and self.preview_df.empty:
             st.info(f"Data preview for '{self.uploaded_file_name}' is empty (no data rows found). Please check the CSV file.")
        else:
            st.warning(f"Data preview for '{self.uploaded_file_name}' is unavailable. The file might be corrupted or could not be read.")

        st.markdown(
            "<div class='mapper-instructions'>"
            "Please map your CSV columns (right) to the application's expected conceptual fields (left). "
            "<b>Critical fields (<code>*</code>)</b> are essential for the application to function correctly. "
            "A (<span style='color:#f4b400;'>⚠️</span>) indicates a potential data type mismatch between the selected CSV column and the expected type."
            "</div>", unsafe_allow_html=True)

        session_mappings_key = f"{self.session_key_prefix}_{SESSION_MAPPINGS_KEY}"
        if session_mappings_key not in st.session_state:
            st.session_state[session_mappings_key] = self._attempt_automatic_mapping()
        
        current_form_mappings = st.session_state[session_mappings_key].copy()
        form_key = f"form_{self.session_key_prefix}" # Simplified form key, still unique

        with st.form(key=form_key):
            cols_top_button = st.columns([0.75, 0.25])
            with cols_top_button[1]:
                # ADDED UNIQUE KEY
                submit_button_top = st.form_submit_button(
                    "Apply & Validate",
                    use_container_width=True,
                    key=f"{form_key}_submit_top" 
                )

            st.markdown("<hr class='styled-hr'>", unsafe_allow_html=True)

            if not self.conceptual_column_categories:
                st.warning("Column categories are not defined. Displaying all columns together.")
                current_form_mappings = self._render_mapping_selectboxes(list(self.conceptual_columns_map.keys()), current_form_mappings, form_key)
            else:
                for i, (category_name, conceptual_keys_in_category) in enumerate(self.conceptual_column_categories.items()):
                    valid_keys_in_category = [
                        key for key in conceptual_keys_in_category if key in self.conceptual_columns_map
                    ]
                    if not valid_keys_in_category:
                        logger.debug(f"Category '{category_name}' has no valid conceptual columns to display.")
                        continue

                    has_critical = any(key in self.critical_conceptual_cols for key in valid_keys_in_category)
                    expander_label = f"{category_name}{' *' if has_critical else ''}"
                    
                    contains_unmapped_critical = any(
                        key in self.critical_conceptual_cols and not current_form_mappings.get(key)
                        for key in valid_keys_in_category
                    )
                    # Expand if it's the first category, or contains unmapped critical, or is a critical category with no mappings yet
                    open_by_default = i == 0 or contains_unmapped_critical or \
                                      (has_critical and not any(current_form_mappings.get(k) for k in valid_keys_in_category if k in self.critical_conceptual_cols))

                    # Use a unique key for each expander based on category name and form_key
                    expander_key = f"{form_key}_expander_{category_name.replace(' ', '_')}"
                    with st.expander(expander_label, expanded=open_by_default): # Removed key from expander, not typically needed unless state is complex
                        current_form_mappings = self._render_mapping_selectboxes(valid_keys_in_category, current_form_mappings, form_key)
            
            st.markdown("<hr class='styled-hr'>", unsafe_allow_html=True)
            _, col_btn_mid, _ = st.columns([0.3, 0.4, 0.3])
            with col_btn_mid:
                # ADDED UNIQUE KEY
                submit_button_bottom = st.form_submit_button(
                    "Apply & Validate",
                    use_container_width=True,
                    key=f"{form_key}_submit_bottom" 
                )
            
            if submit_button_top or submit_button_bottom:
                st.session_state[session_mappings_key] = current_form_mappings.copy()
                self.mapping = current_form_mappings.copy()

                missing_critical_fields = [
                    self.conceptual_columns_map.get(k, k) for k in EXPECTED_COLUMNS
                    if not self.mapping.get(k)
                ]
                if missing_critical_fields:
                    st.error(f"Required fields not mapped: {', '.join(missing_critical_fields)}. Please map these fields to proceed.")
                    st.markdown("</div>", unsafe_allow_html=True)
                    return None

                csv_to_critical_map: Dict[str, List[str]] = {}
                for conc_key, csv_header in self.mapping.items():
                    if csv_header and conc_key in self.critical_conceptual_cols:
                        csv_to_critical_map.setdefault(csv_header, []).append(self.conceptual_columns_map.get(conc_key, conc_key))
                
                has_critical_duplicates = False
                for csv_h, mapped_critical_fields in csv_to_critical_map.items():
                    if len(mapped_critical_fields) > 1:
                        st.error(f"CSV column '{csv_h}' is mapped to multiple critical application fields: {', '.join(mapped_critical_fields)}. Each critical field requires a unique CSV column.")
                        has_critical_duplicates = True
                
                if has_critical_duplicates:
                    st.markdown("</div>", unsafe_allow_html=True)
                    return None

                self.apply_and_validate_mappings_on_submit()
                st.markdown("</div>", unsafe_allow_html=True)
                return {k: v for k, v in self.mapping.items() if v}

        st.markdown("</div>", unsafe_allow_html=True)
        return None

    def _render_mapping_selectboxes(self, conceptual_keys_to_render: List[str], current_mappings: Dict[str, Optional[str]], form_key_prefix: str) -> Dict[str, Optional[str]]:
        cols_ui = st.columns(2)
        col_idx = 0

        for conceptual_key in conceptual_keys_to_render:
            if conceptual_key not in self.conceptual_columns_map:
                logger.warning(f"Conceptual key '{conceptual_key}' from categories not found in main conceptual_columns_map. Skipping.")
                continue

            conceptual_desc = self.conceptual_columns_map[conceptual_key]
            is_required = conceptual_key in self.critical_conceptual_cols
            label_text = f"{conceptual_desc} {'*' if is_required else ''}"
            
            target_container = cols_ui[col_idx % 2]
            col_idx += 1

            with target_container:
                default_csv_header = current_mappings.get(conceptual_key)
                default_index = 0
                if default_csv_header and default_csv_header in self.csv_headers:
                    try:
                        default_index = self.csv_headers.index(default_csv_header)
                    except ValueError:
                        logger.warning(f"Default CSV header '{default_csv_header}' not found in options for '{conceptual_key}'.")
                        default_index = 0 
                elif default_csv_header:
                    logger.info(f"Previous mapping '{default_csv_header}' for '{conceptual_key}' not in current CSV. Resetting.")
                    current_mappings[conceptual_key] = None
                    default_index = 0

                selectbox_key = f"{form_key_prefix}_map_{conceptual_key}" # Ensure selectbox keys are also unique per form instance

                selected_csv_col = st.selectbox(
                    label_text, options=self.csv_headers, index=default_index,
                    key=selectbox_key,
                    help=f"Select CSV column for '{conceptual_desc}'. Expected type: '{self.conceptual_column_types.get(conceptual_key, 'any')}'. {'This field is required.' if is_required else 'This field is optional.'}"
                )
                
                current_mappings[conceptual_key] = selected_csv_col if selected_csv_col else None

                if selected_csv_col:
                    inferred_type = self._infer_column_data_type(selected_csv_col)
                    expected_app_type = self.conceptual_column_types.get(conceptual_key, "any")
                    
                    type_mismatch = False
                    if expected_app_type == "numeric" and inferred_type not in ["integer", "float", "empty", "unknown"]:
                        type_mismatch = True
                    elif expected_app_type == "datetime" and inferred_type not in ["datetime", "empty", "unknown"]:
                        type_mismatch = True

                    if type_mismatch:
                        st.markdown(
                            f"<small class='type-mismatch-warning'>⚠️ Expected '{expected_app_type}', but CSV column seems '{inferred_type}'. Review data if results are unexpected.</small>",
                            unsafe_allow_html=True)
        return current_mappings

    def apply_and_validate_mappings_on_submit(self):
        if not self.uploaded_file_bytes:
            st.warning("No file uploaded, cannot apply mappings.")
            return

        try:
            self.uploaded_file_bytes.seek(0)
            df = pd.read_csv(self.uploaded_file_bytes)
            self.uploaded_file_bytes.seek(0)

            mapped_data_for_df = {}

            for concept_col, csv_col_name in self.mapping.items():
                if csv_col_name and csv_col_name in df.columns:
                    mapped_data_for_df[concept_col] = df[csv_col_name]
                elif concept_col in self.critical_conceptual_cols:
                    st.warning(f"Critical column '{self.conceptual_columns_map.get(concept_col, concept_col)}' was expected but corresponding CSV column '{csv_col_name}' is missing or not mapped. This column will be empty.")
                    mapped_data_for_df[concept_col] = pd.Series([None] * len(df), name=concept_col)
                elif concept_col in OPTIONAL_COLUMNS and not csv_col_name:
                    logger.info(f"Optional column '{self.conceptual_columns_map.get(concept_col, concept_col)}' was not mapped and will be excluded or empty.")

            mapped_csv_column_names = {v for v in self.mapping.values() if v}
            for original_csv_col_name in df.columns:
                if original_csv_col_name not in mapped_csv_column_names:
                    unmapped_key = f"unmapped_{original_csv_col_name}"
                    mapped_data_for_df[unmapped_key] = df[original_csv_col_name]
                    logger.info(f"Including unmapped original CSV column '{original_csv_col_name}' as '{unmapped_key}'.")

            if not mapped_data_for_df:
                st.error("No data could be mapped. Please check your selections and the CSV file.")
                return

            final_column_order = []
            for col in EXPECTED_COLUMNS:
                if col in mapped_data_for_df:
                    final_column_order.append(col)
            
            for col in self.conceptual_columns_map.keys():
                if col in mapped_data_for_df and col not in final_column_order:
                    final_column_order.append(col)
            
            for col in mapped_data_for_df.keys():
                if col not in final_column_order:
                    final_column_order.append(col)
            
            try:
                renamed_df = pd.DataFrame(mapped_data_for_df)
                if final_column_order and all(c in renamed_df.columns for c in final_column_order):
                    renamed_df = renamed_df[final_column_order]
                else:
                    logger.warning("Could not apply precise column order to the mapped DataFrame. Using default order.")

            except Exception as df_creation_error:
                st.error(f"Failed to create the final mapped DataFrame: {df_creation_error}")
                logger.error(f"DataFrame creation error from mapped_data: {mapped_data_for_df} with order {final_column_order}. Error: {df_creation_error}")
                return

            st.success("Mapping applied successfully!")
            st.caption("Preview of the processed data (first 10 rows):")
            st.dataframe(renamed_df.head(10), use_container_width=True)
            st.caption("Mapped conceptual fields are named as per application standards. Unmapped original CSV columns are included and prefixed with 'unmapped_'.")
            
            # Store processed_df in session state with a unique key for the current file instance
            processed_df_key = f"{self.session_key_prefix}_processed_df"
            st.session_state[processed_df_key] = renamed_df


        except pd.errors.EmptyDataError:
            st.error(f"The uploaded file '{self.uploaded_file_name}' is empty. Cannot apply mappings.")
            logger.error(f"EmptyDataError for file {self.uploaded_file_name} during apply_and_validate_mappings_on_submit.")
        except Exception as e:
            st.error(f"An error occurred while applying column mappings: {e}")
            logger.exception(f"Error applying mappings for {self.uploaded_file_name}:")

if __name__ == "__main__":
    APP_TITLE = "ColumnMapperDemo"
    CONCEPTUAL_COLUMNS = {
        "date": "Trade Date/Time", "pnl": "Profit or Loss (PnL)", "symbol": "Trading Symbol",
        "strategy": "Strategy Name", "quantity": "Trade Size/Quantity", "entry_price": "Entry Price",
        "exit_price": "Exit Price", "notes": "Trade Notes"
    }
    CONCEPTUAL_COLUMN_TYPES = {
        "date": "datetime", "pnl": "numeric", "symbol": "text", "strategy": "text",
        "quantity": "numeric", "entry_price": "numeric", "exit_price": "numeric", "notes": "text"
    }
    CONCEPTUAL_COLUMN_SYNONYMS = {
        "pnl": ["profit", "loss", "net profit"], "date": ["time", "trade_date"],
        "quantity": ["size", "volume", "qty"], "notes": ["comment", "description"]
    }
    CRITICAL_CONCEPTUAL_COLUMNS = ["date", "pnl", "symbol"]
    
    CRITICAL_CONCEPTUAL_COLUMNS = [c for c in CRITICAL_CONCEPTUAL_COLUMNS if c in CONCEPTUAL_COLUMNS]
    EXPECTED_COLUMNS = list(CRITICAL_CONCEPTUAL_COLUMNS)

    CONCEPTUAL_COLUMN_CATEGORIES = OrderedDict([
        ("Core Trade Info", ["date", "symbol", "entry_price", "exit_price", "quantity"]),
        ("Performance", ["pnl", "strategy"]),
        ("Details", ["notes"])
    ])

    st.set_page_config(layout="wide", page_title="Column Mapper Demo", initial_sidebar_state="collapsed")
    st.title("📊 Column Mapper Component Demo")
    st.markdown("---")

    st.subheader("1. Simulate File Upload")
    
    # Use session state to keep track of the uploaded file across reruns if needed for the demo
    if 'demo_uploaded_file_bytes' not in st.session_state:
        st.session_state.demo_uploaded_file_bytes = None
        st.session_state.demo_uploaded_file_name = None

    # Create a dummy CSV in memory for quick testing
    sample_csv_data = """TradeDate,Profit,Stock,Volume,Entry,Exit,Comments
2023-01-01 10:00:00,150.50,AAPL,10,170.00,171.50,Good entry
2023-01-01 11:30:00,-50.20,MSFT,5,300.00,299.00,Bad exit
2023-01-02 09:45:00,200.00,GOOG,20,130.00,131.00,Scalp
    """
    st.download_button(
        label="Download Sample CSV",
        data=sample_csv_data,
        file_name="sample_trades.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Upload a sample CSV file (or use the downloaded sample)", type=["csv"])

    if uploaded_file:
        st.session_state.demo_uploaded_file_bytes = BytesIO(uploaded_file.getvalue())
        st.session_state.demo_uploaded_file_name = uploaded_file.name
    
    if st.session_state.demo_uploaded_file_bytes and st.session_state.demo_uploaded_file_name:
        file_name = st.session_state.demo_uploaded_file_name
        file_bytes_io = st.session_state.demo_uploaded_file_bytes
        file_bytes_io.seek(0) # Ensure pointer is at the beginning
        
        try:
            temp_df_for_headers = pd.read_csv(file_bytes_io, nrows=0)
            csv_headers_list = list(temp_df_for_headers.columns)
            file_bytes_io.seek(0) 
        except Exception as e:
            st.error(f"Could not read headers from uploaded CSV: {e}")
            csv_headers_list = []

        if not csv_headers_list:
            st.warning("Uploaded CSV appears to have no headers or is empty.")
        else:
            st.info(f"Detected CSV Headers: `{', '.join(csv_headers_list)}`")

            st.subheader("2. Initialize and Render Column Mapper")
            mapper_ui = ColumnMapperUI(
                uploaded_file_name=file_name,
                uploaded_file_bytes=file_bytes_io,
                csv_headers=csv_headers_list,
                conceptual_columns_map=CONCEPTUAL_COLUMNS,
                conceptual_column_types=CONCEPTUAL_COLUMN_TYPES,
                conceptual_column_synonyms=CONCEPTUAL_COLUMN_SYNONYMS,
                critical_conceptual_cols=CRITICAL_CONCEPTUAL_COLUMNS,
                conceptual_column_categories=CONCEPTUAL_COLUMN_CATEGORIES
            )

            with st.container():
                 st.markdown("### Mapping Interface")
                 final_mappings = mapper_ui.render()

            st.subheader("3. Mapping Result")
            if final_mappings:
                st.success("Column mapping process completed successfully!")
                st.write("Final Mappings Applied:")
                st.json(final_mappings)
                
                # Retrieve and display the processed DataFrame from session state
                # Use the same session_key_prefix logic as in the class
                safe_file_name_part_demo = re.sub(r'[^a-zA-Z0-9_]', '_', file_name)
                processed_df_key = f"col_map_{safe_file_name_part_demo}_processed_df"

                if processed_df_key in st.session_state:
                    st.write("Processed DataFrame Preview (from session state):")
                    st.dataframe(st.session_state[processed_df_key].head())
                else:
                    st.warning(f"Processed DataFrame not found in session state under key: {processed_df_key}")

            else: # final_mappings is None
                # Check if form was submitted (using one of the button keys)
                # This requires knowing the exact keys used for submit buttons if we want to be very specific
                # For a general message:
                st.info("Mapping process not yet completed, or validation failed. Please check messages above and confirm your mappings.")
    else:
        st.info("Please upload a CSV file to see the Column Mapper in action.")

    st.markdown("---")
    st.caption("This is a demo of the ColumnMapperUI component.")
