# components/column_mapper_ui.py
"""
Component for allowing users to map their uploaded CSV columns
to the application's expected conceptual columns, with data preview,
enhanced auto-mapping, data type validation, categorized display,
and confirm buttons at top and bottom.

Updated: 
- Uses session state constants.
- Explicit expected/optional columns.
- Caching for dataframe columns.
- Inline mapping feedback and validation.
- Clean, categorized UI with expanders and 2-col layout.
- Robust error handling.
- "Not Applicable" skip option.
- Standalone demo and inline CSS example.
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Optional, Any
from collections import OrderedDict
from io import BytesIO
from thefuzz import fuzz
import re
import logging

# --- Constants for session state keys ---
SESSION_MAPPINGS_KEY = "column_mappings"

# --- Fallback config ---
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
        "risk_pct": ["risk_percent", "pct_risk"], "tags": ["label", "category_tag"]
    }
    CRITICAL_CONCEPTUAL_COLUMNS = ["date", "pnl", "symbol"]
    CONCEPTUAL_COLUMN_CATEGORIES = OrderedDict([
        ("Core Trade Information", ["date", "symbol", "entry_price", "exit_price", "quantity"]),
        ("Performance & Strategy", ["pnl", "strategy", "r_r_csv_num", "duration_minutes"]),
        ("Risk & Financials", ["risk_pct", "commission", "fees"]),
        ("Qualitative & Categorization", ["notes", "tags"])
    ])
    print("Warning: Could not import config, using fallback values.")

# --- Expected/Optional column classification ---
EXPECTED_COLUMNS = list(CRITICAL_CONCEPTUAL_COLUMNS)
OPTIONAL_COLUMNS = [k for k in CONCEPTUAL_COLUMNS.keys() if k not in EXPECTED_COLUMNS]

# --- Caching dataframe columns (optimization) ---
@st.cache_data
def get_cached_dataframe_columns(uploaded_file_bytes):
    uploaded_file_bytes.seek(0)
    df = pd.read_csv(uploaded_file_bytes, nrows=0)
    return list(df.columns)

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
        conceptual_column_categories: OrderedDict
    ):
        self.uploaded_file_name = uploaded_file_name
        self.uploaded_file_bytes = uploaded_file_bytes
        self.csv_headers = [""] + csv_headers  # "" = Not Applicable / Skip
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
                logger.error(f"Error reading CSV for preview: {e}")
                st.warning(f"Could not generate data preview for {self.uploaded_file_name}.")
                self.preview_df = None

        logger.debug(f"ColumnMapperUI initialized for file: {self.uploaded_file_name}")

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
        used_csv_headers = set()

        specific_csv_header_targets = {
            "trade_model": "strategy", "r_r": "r_r_csv_num", "pnl": "pnl", "date": "date",
            "symbol_1": "symbol", "lesson_learned": "notes", "duration_mins": "duration_minutes",
            "risk_pct": "risk_pct", "entry": "entry_price", "exit": "exit_price", "size": "quantity"
        }

        for norm_specific_csv, target_conceptual_key in specific_csv_header_targets.items():
            if norm_specific_csv in normalized_csv_headers_map:
                original_csv_header = normalized_csv_headers_map[norm_specific_csv]
                if original_csv_header not in used_csv_headers and target_conceptual_key not in auto_mapping:
                    auto_mapping[target_conceptual_key] = original_csv_header
                    used_csv_headers.add(original_csv_header)
                    logger.info(f"Auto-mapped (specific) CSV '{original_csv_header}' to conceptual '{target_conceptual_key}'")

        for conceptual_key in self.conceptual_columns_map.keys():
            if conceptual_key in auto_mapping:
                continue
            mapped_csv_header = None
            norm_conceptual_key = self._normalize_header(conceptual_key)

            if norm_conceptual_key in normalized_csv_headers_map and normalized_csv_headers_map[norm_conceptual_key] not in used_csv_headers:
                mapped_csv_header = normalized_csv_headers_map[norm_conceptual_key]

            if not mapped_csv_header and conceptual_key in self.conceptual_column_synonyms:
                for synonym in self.conceptual_column_synonyms[conceptual_key]:
                    norm_synonym = self._normalize_header(synonym)
                    if norm_synonym in normalized_csv_headers_map and normalized_csv_headers_map[norm_synonym] not in used_csv_headers:
                        mapped_csv_header = normalized_csv_headers_map[norm_synonym]
                        break

            FUZZY_MATCH_THRESHOLD = 85
            if not mapped_csv_header:
                best_match_score = 0
                potential_header = None
                for norm_csv_h, original_csv_h in normalized_csv_headers_map.items():
                    if original_csv_h in used_csv_headers:
                        continue
                    score = fuzz.ratio(norm_conceptual_key, norm_csv_h)
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
                logger.warning(f"Could not auto-map critical conceptual column: '{conceptual_key}'")
        return auto_mapping

    def _infer_column_data_type(self, csv_column_name: str) -> str:
        if self.preview_df is None or csv_column_name not in self.preview_df.columns:
            return "unknown"
        column_sample = self.preview_df[csv_column_name].dropna().convert_dtypes()
        if column_sample.empty:
            return "empty"
        try:
            pd.to_numeric(column_sample)
            if (column_sample % 1 == 0).all():
                return "integer"
            return "float"
        except Exception:
            pass
        try:
            pd.to_datetime(column_sample, errors='raise', infer_datetime_format=True)
            return "datetime"
        except Exception:
            pass
        return "text"

    def render(self) -> Optional[Dict[str, Optional[str]]]:
        st.markdown(
            """
            <style>
            .type-mismatch-warning { color: orange; }
            .styled-hr { border: 1px solid #eee; margin: 1em 0; }
            .column-mapper-container { background: #fafafa; border-radius: 8px; padding: 1em; }
            .data-preview-title { font-weight: bold; margin-bottom: .5em; }
            .mapper-instructions { color: #888; font-size: 0.95em; margin-bottom: 1em; }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='column-mapper-container'>", unsafe_allow_html=True)
        st.markdown(f"<h3 class='component-subheader'>Map Columns for '{self.uploaded_file_name}'</h3>", unsafe_allow_html=True)

        if self.preview_df is not None and not self.preview_df.empty:
            st.markdown("<p class='data-preview-title'>Data Preview (First 5 Rows):</p>", unsafe_allow_html=True)
            st.dataframe(self.preview_df, hide_index=True, use_container_width=True)
        else:
            st.info("Data preview is not available.")

        st.markdown(
            "<div class='mapper-instructions'>"
            "Map CSV columns to application fields. Required fields are essential. (⚠️) indicates type mismatch."
            "</div>", unsafe_allow_html=True)

        initial_mapping = self._attempt_automatic_mapping()
        self.mapping = st.session_state.get(SESSION_MAPPINGS_KEY, initial_mapping.copy())

        with st.form(key=f"column_mapping_form_{self.uploaded_file_name.replace('.', '_')}"):
            cols_top_button = st.columns([0.75, 0.25])
            with cols_top_button[1]:
                submit_button_top = st.form_submit_button("Apply & Validate", use_container_width=True, type="primary")

            st.markdown("<hr class='styled-hr'>", unsafe_allow_html=True)

            if not self.conceptual_column_categories:
                st.warning("Column categories are not defined. Displaying all columns together.")
                self._render_mapping_selectboxes(self.conceptual_columns_map.keys(), initial_mapping)
            else:
                for category_name, conceptual_keys_in_category in self.conceptual_column_categories.items():
                    valid_keys_in_category = [
                        key for key in conceptual_keys_in_category if key in self.conceptual_columns_map
                    ]
                    if not valid_keys_in_category:
                        logger.warning(f"Category '{category_name}' has no valid conceptual columns to display.")
                        continue

                    has_critical = any(key in self.critical_conceptual_cols for key in valid_keys_in_category)
                    expander_label = f"{category_name}{' *' if has_critical else ''}"

                    open_by_default = any(key in self.critical_conceptual_cols and not self.mapping.get(key)
                                          for key in valid_keys_in_category)

                    with st.expander(expander_label, expanded=open_by_default):
                        self._render_mapping_selectboxes(valid_keys_in_category, initial_mapping)

            st.markdown("<hr class='styled-hr'>", unsafe_allow_html=True)
            _, col_btn_mid, _ = st.columns([0.3, 0.4, 0.3])
            with col_btn_mid:
                submit_button_bottom = st.form_submit_button("Apply & Validate", use_container_width=True, type="primary")

        # Update session state
        st.session_state[SESSION_MAPPINGS_KEY] = self.mapping.copy()

        if submit_button_top or submit_button_bottom:
            missing_critical = [
                self.conceptual_columns_map.get(k, k) for k in EXPECTED_COLUMNS if not self.mapping.get(k)
            ]
            if missing_critical:
                st.error(f"Required fields not mapped: {', '.join(missing_critical)}. Please map these fields.")
                st.markdown("</div>", unsafe_allow_html=True)
                return None

            csv_to_critical_map: Dict[str, List[str]] = {}
            for conc_key, csv_header in self.mapping.items():
                if csv_header and conc_key in self.critical_conceptual_cols:
                    csv_to_critical_map.setdefault(csv_header, []).append(self.conceptual_columns_map.get(conc_key, conc_key))

            has_critical_duplicates = False
            for csv_h, mapped_fields in csv_to_critical_map.items():
                if len(mapped_fields) > 1:
                    st.error(f"CSV column '{csv_h}' mapped to multiple required fields: {', '.join(mapped_fields)}. Each needs a unique CSV column.")
                    has_critical_duplicates = True
            if has_critical_duplicates:
                st.markdown("</div>", unsafe_allow_html=True)
                return None

            # --- Apply and validate mappings ---
            self.apply_and_validate_mappings()
            st.markdown("</div>", unsafe_allow_html=True)
            return {k: v for k, v in self.mapping.items() if v}

        st.markdown("</div>", unsafe_allow_html=True)
        return None

    def _render_mapping_selectboxes(self, conceptual_keys_to_render: List[str], initial_mapping: Dict[str, Optional[str]]):
        cols_ui = st.columns(2)
        col_idx = 0
        for conceptual_key in conceptual_keys_to_render:
            if conceptual_key not in self.conceptual_columns_map:
                logger.warning(f"Conceptual key '{conceptual_key}' from categories not found in main conceptual_columns_map. Skipping.")
                continue

            conceptual_desc = self.conceptual_columns_map[conceptual_key]
            is_required = conceptual_key in EXPECTED_COLUMNS
            label_text = f"{conceptual_desc} {'(Required)' if is_required else '(Optional)'}"
            target_container = cols_ui[col_idx % 2]
            col_idx += 1

            with target_container:
                default_csv_header = self.mapping.get(conceptual_key) or initial_mapping.get(conceptual_key)
                default_index = 0
                if default_csv_header and default_csv_header in self.csv_headers:
                    try:
                        default_index = self.csv_headers.index(default_csv_header)
                    except ValueError:
                        default_index = 0

                selected_csv_col = st.selectbox(
                    label_text, options=self.csv_headers, index=default_index,
                    key=f"map_{self.uploaded_file_name.replace('.', '_')}_{conceptual_key}",
                    help=f"CSV for '{conceptual_desc}'. Expected: '{self.conceptual_column_types.get(conceptual_key, 'any')}'. {'Required.' if is_required else ''}"
                )
                self.mapping[conceptual_key] = selected_csv_col if selected_csv_col else None

                if selected_csv_col:
                    inferred_type = self._infer_column_data_type(selected_csv_col)
                    expected_type = self.conceptual_column_types.get(conceptual_key, "any")
                    type_mismatch = False
                    if expected_type == "numeric" and inferred_type not in ["numeric", "integer", "float", "empty", "unknown"]:
                        type_mismatch = True
                    elif expected_type == "datetime" and inferred_type not in ["datetime", "empty", "unknown"]:
                        type_mismatch = True
                    if type_mismatch:
                        st.markdown(
                            f"<small class='type-mismatch-warning'>⚠️ Expected '{expected_type}', seems '{inferred_type}'.</small>",
                            unsafe_allow_html=True)

    def apply_and_validate_mappings(self):
        """
        Applies the mappings, creates a new DataFrame with standard columns, provides feedback,
        handles missing required columns, logs unmapped optionals, and previews the mapped data.
        """
        if not self.uploaded_file_bytes:
            st.warning("No file uploaded, cannot apply mappings.")
            return
        try:
            self.uploaded_file_bytes.seek(0)
            df = pd.read_csv(self.uploaded_file_bytes)
            mapped_cols = {}
            # Build new DataFrame with standardized columns
            for concept_col, csv_col in self.mapping.items():
                if csv_col and csv_col in df.columns:
                    mapped_cols[concept_col] = df[csv_col]
                elif concept_col in EXPECTED_COLUMNS:
                    st.warning(f"Required column '{CONCEPTUAL_COLUMNS[concept_col]}' missing from file. Column will be empty.")
                    mapped_cols[concept_col] = pd.Series([None] * len(df))
                elif concept_col in OPTIONAL_COLUMNS:
                    st.caption(f"Optional column '{CONCEPTUAL_COLUMNS[concept_col]}' was not mapped.")

            # Add unmapped original columns (prefixed)
            mapped_csv_cols = set(self.mapping.values())
            for orig_col in df.columns:
                if orig_col not in mapped_csv_cols:
                    mapped_cols[f"unmapped_{orig_col}"] = df[orig_col]

            # Build final DataFrame, reorder: required first, then optional, then unmapped
            std_col_order = EXPECTED_COLUMNS + [k for k in OPTIONAL_COLUMNS if k in mapped_cols]
            ordered_cols = std_col_order + [c for c in mapped_cols if c not in std_col_order]
            renamed_df = pd.DataFrame(mapped_cols)[ordered_cols]

            st.success("Mapping applied and validated!")
            st.dataframe(renamed_df.head(10), use_container_width=True)
            st.caption("Preview above: mapped and unmapped columns included. Unmapped original columns are prefixed with 'unmapped_'.")
        except Exception as e:
            st.error(f"Error applying or previewing column mappings: {e}")

# --- Standalone Demo/Test ---
if __name__ == "__main__":
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
    try:
        st.markdown(f"""<style>{open("style.css").read()}</style>""", unsafe_allow_html=True)
    except Exception:
        pass

    st.title("Test Categorized Column Mapper UI")

    # Mock data (ensure these match the fallback/actual config structure)
    _MOCK_CONCEPTUAL_COLUMNS = {
        "date": "Trade Date/Time", "pnl": "Profit or Loss (PnL)", "strategy": "Strategy Name",
        "symbol": "Trading Symbol", "r_r_csv_num": "Risk:Reward Ratio", "notes": "Trade Notes",
        "duration_minutes": "Duration (Mins)", "risk_pct": "Risk %", "entry_price": "Entry Price",
        "exit_price": "Exit Price", "quantity": "Quantity", "commission": "Commission Cost",
        "fees": "Total Fees", "tags": "Custom Tags"
    }
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
        "risk_pct": ["risk_percentage", "percent_risk"], "tags": ["label", "trade_category"]
    }
    _MOCK_CRITICAL_CONCEPTUAL_COLUMNS = ["date", "pnl", "symbol", "entry_price", "exit_price", "quantity"]
    _MOCK_CONCEPTUAL_COLUMN_CATEGORIES = OrderedDict([
        ("Core Trade Data*", ["date", "symbol", "entry_price", "exit_price", "quantity"]),
        ("Performance Metrics", ["pnl", "strategy", "r_r_csv_num", "duration_minutes"]),
        ("Financials & Risk", ["risk_pct", "commission", "fees"]),
        ("Additional Information", ["notes", "tags"])
    ])

    mock_csv_headers_from_file = [
        "Trade ID", "Date", "Entry Time", "Size", "Entry", "Take Profit", "Stop Loss", "Exit",
        "Candle Count", "Exit Type", "Trade Model ", "PnL", "R:R", "Duration (mins)", "Risk %", "Symbol 1",
        "Lesson Learned", "Tags", "Commission", "Total Fees"
    ]

    sample_data_for_preview = {
        "Date": ["2023-01-01 10:00", "2023-01-02 11:00"], "PnL": [100.50, -50.25],
        "Trade Model ": ["Scalp V1", "Swing V2"], "Symbol 1": ["EURUSD", "GBPUSD"],
        "Entry": [1.1000, 1.2500], "Exit": [1.1050, 1.2400], "Size": [10000, 5000],
        "R:R": [2.0, 1.5], "Duration (mins)": [15, 120], "Risk %": [1, 0.5],
        "Lesson Learned": ["Good exit timing", "Held too long"], "Tags": ["News Trade", "Trend Follow"],
        "Commission": [2.5, 1.5], "Total Fees": [0.5, 0.2]
    }
    df_preview_test = pd.DataFrame(sample_data_for_preview)
    for header in mock_csv_headers_from_file:
        if header not in df_preview_test.columns:
            df_preview_test[header] = "N/A"

    output_bytes = BytesIO()
    df_preview_test.to_csv(output_bytes, index=False)
    output_bytes.seek(0)

    st.write("### Scenario: Categorized Mapping")
    mapper_ui = ColumnMapperUI(
        uploaded_file_name="categorized_test.csv",
        uploaded_file_bytes=output_bytes,
        csv_headers=list(df_preview_test.columns),
        conceptual_columns_map=_MOCK_CONCEPTUAL_COLUMNS,
        conceptual_column_types=_MOCK_CONCEPTUAL_COLUMN_TYPES,
        conceptual_column_synonyms=_MOCK_CONCEPTUAL_COLUMN_SYNONYMS,
        critical_conceptual_cols=_MOCK_CRITICAL_CONCEPTUAL_COLUMNS,
        conceptual_column_categories=_MOCK_CONCEPTUAL_COLUMN_CATEGORIES
    )
    mapping_result = mapper_ui.render()

    if mapping_result is not None:
        st.success("Mapping Confirmed (Categorized):")
        st.json(mapping_result)
        st.write("#### Auto-mapping decisions (review):")
        test_auto_map = mapper_ui._attempt_automatic_mapping()
        st.json({k: v for k, v in test_auto_map.items() if v})

    logger.info("ColumnMapperUI v8 (Categorized) test complete.")
