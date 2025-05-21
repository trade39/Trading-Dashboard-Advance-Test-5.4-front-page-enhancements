# data_processing.py

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
import logging
import re
import chardet # For encoding detection
import csv # For delimiter sniffing
from io import StringIO # For reading BytesIO as text for sniffer

try:
    from config import APP_TITLE, CONCEPTUAL_COLUMNS, CRITICAL_CONCEPTUAL_COLUMNS, CONCEPTUAL_COLUMN_TYPES
except ImportError:
    APP_TITLE = "TradingDashboard_Default_DP"
    CONCEPTUAL_COLUMNS = {"date": "Trade Date/Time", "pnl": "Profit or Loss (PnL)", "entry_time_str": "Entry Time"}
    CRITICAL_CONCEPTUAL_COLUMNS = ["date", "pnl"]
    CONCEPTUAL_COLUMN_TYPES = {"date": "datetime", "pnl": "numeric", "entry_time_str": "text"}
    print("Warning (data_processing.py): Could not import from config. Using fallback values.")

logger = logging.getLogger(APP_TITLE)

def _calculate_drawdown_series_for_df(cumulative_pnl: pd.Series) -> Tuple[pd.Series, pd.Series]:
    if cumulative_pnl.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    high_water_mark = cumulative_pnl.cummax()
    drawdown_abs_series = high_water_mark - cumulative_pnl
    hwm_for_pct = high_water_mark.replace(0, np.nan) # Avoid division by zero if HWM is 0
    drawdown_pct_series = (drawdown_abs_series / hwm_for_pct).fillna(0) * 100
    
    # Handle cases where HWM is 0 and PnL is negative (should be 100% drawdown of the loss amount from 0)
    # This logic might need refinement based on exact definition of drawdown % from zero HWM
    mask_hwm_zero_loss = (high_water_mark == 0) & (drawdown_abs_series > 0)
    drawdown_pct_series[mask_hwm_zero_loss] = 100.0

    return drawdown_abs_series, drawdown_pct_series

def clean_text_column(text_series: pd.Series) -> pd.Series:
    if not isinstance(text_series, pd.Series):
        return pd.Series(text_series, dtype=str)
    processed_series = text_series.astype(str).fillna('').str.strip()
    url_pattern = r"\(?https?://[^\s\)\"]+\)?|www\.[^\s\)\"]+"
    notion_link_pattern = r"\(https://www\.notion\.so/[^)]+\)"
    empty_parens_pattern = r"^\(''\)$"

    def clean_element(text: str) -> Any:
        if pd.isna(text) or text.lower() == 'nan': return pd.NA
        cleaned_text = text
        cleaned_text = re.sub(notion_link_pattern, '', cleaned_text)
        cleaned_text = re.sub(url_pattern, '', cleaned_text)
        cleaned_text = re.sub(empty_parens_pattern, '', cleaned_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        return cleaned_text if cleaned_text else pd.NA
    return processed_series.apply(clean_element)

@st.cache_data(ttl=3600, show_spinner="Loading and processing trade data...")
def load_and_process_data(
    uploaded_file_obj: Any,
    user_column_mapping: Optional[Dict[str, str]] = None,
    on_bad_lines_config: str = 'warn' # 'error', 'warn', or 'skip'
) -> Optional[pd.DataFrame]:
    if uploaded_file_obj is None:
        logger.info("No file object provided.")
        return None

    df = None
    detected_encoding = None
    detected_delimiter = None

    try:
        if hasattr(uploaded_file_obj, 'seek'):
            uploaded_file_obj.seek(0)
        logger.info("Attempting to read CSV with UTF-8 and default delimiter sniffing.")
        df = pd.read_csv(uploaded_file_obj, on_bad_lines=on_bad_lines_config)
        logger.info(f"Successfully loaded CSV with UTF-8. Shape: {df.shape}. Original headers: {df.columns.tolist()}")

    except UnicodeDecodeError:
        logger.warning("UTF-8 decoding failed. Attempting to detect encoding with chardet.")
        if hasattr(uploaded_file_obj, 'seek'):
            uploaded_file_obj.seek(0)
        raw_data_sample = uploaded_file_obj.read(100000) 
        if hasattr(uploaded_file_obj, 'seek'):
            uploaded_file_obj.seek(0)

        if raw_data_sample:
            chardet_result = chardet.detect(raw_data_sample)
            detected_encoding = chardet_result.get('encoding')
            confidence = chardet_result.get('confidence', 0)
            logger.info(f"Chardet detected encoding: {detected_encoding} with confidence: {confidence:.2f}")

            if detected_encoding and confidence > 0.7:
                try:
                    logger.info(f"Retrying CSV read with detected encoding: {detected_encoding}")
                    df = pd.read_csv(uploaded_file_obj, encoding=detected_encoding, on_bad_lines=on_bad_lines_config)
                    logger.info(f"Successfully loaded CSV with encoding '{detected_encoding}'. Shape: {df.shape}.")
                except Exception as e_enc:
                    logger.error(f"Error reading CSV with detected encoding '{detected_encoding}': {e_enc}", exc_info=True)
                    st.error(f"Error reading CSV with auto-detected encoding '{detected_encoding}': {e_enc}. Please ensure the file is a valid CSV.")
                    return None
            else:
                logger.error("Could not confidently detect encoding or chardet failed.")
                st.error("Could not automatically determine the file encoding. Please ensure your CSV is UTF-8 encoded or try converting it.")
                return None
        else:
            logger.error("Could not read sample data for encoding detection.")
            st.error("File appears to be empty or unreadable for encoding detection.")
            return None

    except pd.errors.ParserError as pe:
        logger.warning(f"Pandas ParserError: {pe}. This might be due to an incorrect delimiter. Attempting to sniff delimiter.")
        if hasattr(uploaded_file_obj, 'seek'):
            uploaded_file_obj.seek(0)
        try:
            sample_text_for_sniffer = ""
            if detected_encoding:
                 sample_text_for_sniffer = uploaded_file_obj.read(2048).decode(detected_encoding)
            else:
                 sample_text_for_sniffer = uploaded_file_obj.read(2048).decode('utf-8')

            if hasattr(uploaded_file_obj, 'seek'):
                uploaded_file_obj.seek(0)

            if sample_text_for_sniffer:
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(sample_text_for_sniffer)
                detected_delimiter = dialect.delimiter
                logger.info(f"CSV Sniffer detected delimiter: '{detected_delimiter}'")
                try:
                    logger.info(f"Retrying CSV read with detected delimiter '{detected_delimiter}' and encoding '{detected_encoding or 'UTF-8'}'.")
                    df = pd.read_csv(uploaded_file_obj,
                                     delimiter=detected_delimiter,
                                     encoding=detected_encoding if detected_encoding else 'utf-8',
                                     on_bad_lines=on_bad_lines_config)
                    logger.info(f"Successfully loaded CSV with detected delimiter '{detected_delimiter}'. Shape: {df.shape}.")
                except Exception as e_delim:
                    logger.error(f"Error reading CSV with detected delimiter '{detected_delimiter}': {e_delim}", exc_info=True)
                    st.error(f"Error reading CSV with auto-detected delimiter '{detected_delimiter}': {e_delim}. The file structure might be complex.")
                    return None
            else:
                logger.error("Could not read sample data for delimiter sniffing.")
                st.error("File appears to be empty or unreadable for delimiter sniffing.")
                return None
        except Exception as e_sniff_outer:
            logger.error(f"Error during delimiter sniffing process: {e_sniff_outer}", exc_info=True)
            st.error(f"Failed to parse CSV. Error: {pe}. Automatic delimiter detection also failed.")
            return None

    except Exception as e:
        logger.error(f"General error reading CSV: {e}", exc_info=True)
        st.error(f"Error reading CSV: {e}. Please ensure it's a valid CSV file and the encoding is standard (e.g., UTF-8).")
        return None

    if df is None:
        logger.error("DataFrame is None after all parsing attempts.")
        st.error("Could not load the CSV file after multiple attempts with different parsing strategies.")
        return None
    
    if on_bad_lines_config == 'warn' and not df.empty:
        logger.info(f"CSV loaded with on_bad_lines='{on_bad_lines_config}'. Check console/logs for pandas warnings if any bad lines were encountered.")

    original_csv_headers = df.columns.tolist()
    conceptual_to_original_csv_map = {
        conceptual_key: csv_header
        for conceptual_key, csv_header in (user_column_mapping or {}).items()
    }

    if user_column_mapping:
        rename_map = {csv_col: conceptual_key
                      for conceptual_key, csv_col in user_column_mapping.items()
                      if csv_col in df.columns}
        df.rename(columns=rename_map, inplace=True)
        logger.info(f"Applied user column mapping. Renamed columns based on: {rename_map}")
        logger.info(f"DataFrame columns after user mapping: {df.columns.tolist()}")
    else:
        logger.warning("No user column mapping provided. Applying default header cleaning.")
        df.columns = [str(col).strip().lower().replace(' ', '_') for col in original_csv_headers]
        logger.info(f"Cleaned DataFrame headers (default): {df.columns.tolist()}")

    missing_critical_mapped_cols = [
        key for key in CRITICAL_CONCEPTUAL_COLUMNS if key not in df.columns
    ]
    if missing_critical_mapped_cols:
        error_message_parts = [
            f"Critical data fields are missing after mapping: {', '.join(missing_critical_mapped_cols)}."
        ]
        for missing_key in missing_critical_mapped_cols:
            original_csv_col = conceptual_to_original_csv_map.get(missing_key)
            if original_csv_col:
                error_message_parts.append(
                    f"Conceptual field '{CONCEPTUAL_COLUMNS.get(missing_key, missing_key)}' was expected to be mapped from your CSV column '{original_csv_col}', but this column was not found or the mapping was incomplete."
                )
            else:
                error_message_parts.append(
                    f"Conceptual field '{CONCEPTUAL_COLUMNS.get(missing_key, missing_key)}' was not mapped from any CSV column."
                )
        error_message_parts.append(f"Available columns after mapping/renaming: {df.columns.tolist()}")
        final_error_msg = " ".join(error_message_parts)
        logger.error(final_error_msg)
        st.error(final_error_msg)
        return None

    # --- MODIFICATION START: Combine Date and Entry Time ---
    # Check if both 'date' (for date part) and 'entry_time_str' (for time part) are mapped and exist
    date_conceptual_key = 'date' # This is the target conceptual key for the final datetime
    time_conceptual_key = 'entry_time_str'

    # actual_col_name_in_df for 'date' will be 'date' if mapping was successful.
    # actual_col_name_in_df for 'entry_time_str' will be 'entry_time_str'.
    if date_conceptual_key in df.columns and time_conceptual_key in df.columns:
        logger.info(f"Found both '{date_conceptual_key}' and '{time_conceptual_key}' columns. Attempting to combine them.")
        
        # Ensure both columns are string type for concatenation
        date_series_str = df[date_conceptual_key].astype(str)
        time_series_str = df[time_conceptual_key].astype(str)
        
        # Combine, handling potential NaNs or empty strings gracefully
        # NaNs in either part will result in NaT after pd.to_datetime if not handled carefully before.
        # A simple space concatenation is usually fine if pd.to_datetime can handle various formats.
        combined_datetime_str_series = date_series_str.str.strip() + " " + time_series_str.str.strip()
        
        # Replace the original 'date' column content with these combined strings
        # This combined series will then be processed by pd.to_datetime in the loop below.
        df[date_conceptual_key] = combined_datetime_str_series
        logger.info(f"Combined '{date_conceptual_key}' and '{time_conceptual_key}' into the '{date_conceptual_key}' column for datetime parsing.")
        # We no longer need the separate 'entry_time_str' column for primary date processing,
        # but it might be kept if other conceptual types use it.
        # For clarity, the 'date' conceptual column is now expected to hold the combined string.
    elif date_conceptual_key in df.columns:
        logger.info(f"Only '{date_conceptual_key}' column found. Proceeding with it for datetime parsing. Time component might be missing or default to 00:00:00 if not included in this column.")
    # --- MODIFICATION END ---

    try:
        for conceptual_key, _ in CONCEPTUAL_COLUMNS.items():
            actual_col_name_in_df = conceptual_key # After renaming, conceptual_key is the column name
            original_csv_col_name = conceptual_to_original_csv_map.get(conceptual_key, "N/A (Not Mapped or Direct)")

            if actual_col_name_in_df not in df.columns:
                # Handle specific default creations if a column is entirely missing
                if conceptual_key == 'risk_pct': # Example, was 'risk' before, changed to match config
                    df['risk_numeric_internal'] = 0.0 # Default for risk calculation
                elif conceptual_key == 'duration_minutes':
                    df['duration_minutes_numeric'] = pd.NA # Default for duration
                # logger.debug(f"Conceptual column '{conceptual_key}' not found in DataFrame. Skipping its processing.")
                continue

            logger.debug(f"Processing mapped conceptual column '{conceptual_key}' (from CSV '{original_csv_col_name}')")
            series = df[actual_col_name_in_df]
            original_series_sample = series.dropna().head().tolist()

            try:
                expected_type = CONCEPTUAL_COLUMN_TYPES.get(conceptual_key)
                
                if expected_type == 'datetime': # This will now use the potentially combined series
                    df[actual_col_name_in_df] = pd.to_datetime(series, errors='coerce')
                    if df[actual_col_name_in_df].isnull().sum() > 0:
                        logger.warning(f"{df[actual_col_name_in_df].isnull().sum()} invalid date formats in mapped '{conceptual_key}' (CSV: '{original_csv_col_name}'). Rows with invalid dates will be dropped.")
                        df.dropna(subset=[actual_col_name_in_df], inplace=True)
                    if df.empty: logger.error("DataFrame empty after dropping invalid dates."); return None
                
                elif expected_type == 'numeric':
                    converted_series = pd.to_numeric(series, errors='coerce')
                    num_failed_conversions = series[converted_series.isnull() & series.notnull()].count()
                    if num_failed_conversions > 0:
                        logger.warning(
                            f"Could not convert {num_failed_conversions} values in your CSV column '{original_csv_col_name}' "
                            f"(mapped to '{CONCEPTUAL_COLUMNS.get(conceptual_key, conceptual_key)}') to numbers. "
                            f"Sample original values that failed: {series[converted_series.isnull() & series.notnull()].dropna().head().tolist()}. "
                            "These will be treated as NaN (missing)."
                        )
                    df[actual_col_name_in_df] = converted_series
                    if conceptual_key == 'pnl' and df[actual_col_name_in_df].isnull().all() and not series.dropna().empty:
                         error_msg = f"The mapped PnL column ('{CONCEPTUAL_COLUMNS.get(conceptual_key, conceptual_key)}' from CSV '{original_csv_col_name}') contains no valid numeric data after conversion. All values became NaN."
                         logger.error(error_msg + f" Original series sample: {original_series_sample}")
                         st.error(error_msg)
                         return None
                
                elif expected_type == 'text':
                    df[actual_col_name_in_df] = clean_text_column(series).fillna('N/A')
                
                else: # Fallback for types not explicitly 'datetime', 'numeric', 'text' or if expected_type is None
                    df[actual_col_name_in_df] = series.astype(str).fillna('N/A')

            except ValueError as ve: # Catch errors from pd.to_datetime or pd.to_numeric specifically
                error_msg = f"Error converting data for field '{CONCEPTUAL_COLUMNS.get(conceptual_key, conceptual_key)}' (from your CSV column '{original_csv_col_name}'): {ve}. Original sample: {original_series_sample}. Please check the data in this column."
                logger.error(error_msg, exc_info=True)
                st.error(error_msg)
                return None
            except Exception as e_inner: # Catch any other unexpected errors during conversion
                error_msg = f"Unexpected error processing field '{CONCEPTUAL_COLUMNS.get(conceptual_key, conceptual_key)}' (CSV: '{original_csv_col_name}'): {e_inner}. Original sample: {original_series_sample}."
                logger.error(error_msg, exc_info=True)
                st.error(error_msg)
                return None
        
        # Post-loop specific handling for engineered numeric columns
        # Risk Percentage
        risk_pct_conceptual_key = 'risk_pct' # From config.CONCEPTUAL_COLUMNS
        if risk_pct_conceptual_key in df.columns and pd.api.types.is_numeric_dtype(df[risk_pct_conceptual_key]):
            # Assuming risk_pct is already a percentage, convert to decimal for internal calculations if needed,
            # or ensure it's stored as is if calculations expect percentage.
            # For now, let's assume it's stored as the number (e.g., 0.29 for 0.29%)
            # If it was 'Risk %' in CSV and became 'risk_pct', it might be 0.29.
            # If it was '0.29%', it would have failed to_numeric unless cleaned.
            # Let's create 'risk_numeric_internal' based on 'risk_pct' if available, assuming 'risk_pct' is the value (e.g. 0.29 for 0.29% risk)
            # This 'risk_numeric_internal' can then be used for calculations like R:R
            # If 'risk_pct' is intended to be like 2 for 2%, then it should be divided by 100.
            # The CSV has "Risk %" as "0.29" meaning 0.29%. So, this is already a decimal representation of percentage.
            # For R:R calculation, if PnL is $100 and Risk % is 0.29 (meaning 0.29% of capital),
            # then actual risk amount needs to be calculated first (e.g. 0.0029 * capital).
            # The current R:R in calculations.py uses 'risk_numeric_internal' which is not clearly defined yet.
            # Let's assume 'risk_pct' IS the percentage value like 0.29.
            # For now, we'll just ensure it's numeric. The R:R calculation might need adjustment.
            df['risk_numeric_internal'] = df[risk_pct_conceptual_key].fillna(0.0) / 100.0 # Convert to decimal, e.g., 0.29 -> 0.0029
        else:
            df['risk_numeric_internal'] = 0.0 # Default if 'risk_pct' is not available or not numeric
            if risk_pct_conceptual_key not in df.columns:
                logger.warning(f"Conceptual column '{risk_pct_conceptual_key}' not mapped or found. Defaulting 'risk_numeric_internal' to 0.0.")
            else:
                logger.warning(f"Mapped '{risk_pct_conceptual_key}' (from CSV '{conceptual_to_original_csv_map.get(risk_pct_conceptual_key, 'N/A')}') is not numeric. Defaulting 'risk_numeric_internal' to 0.0.")

        # Duration Minutes
        duration_conceptual_key = 'duration_minutes'
        if duration_conceptual_key in df.columns and pd.api.types.is_numeric_dtype(df[duration_conceptual_key]):
            df['duration_minutes_numeric'] = df[duration_conceptual_key].copy().fillna(pd.NA)
        else:
            df['duration_minutes_numeric'] = pd.NA
            if duration_conceptual_key not in df.columns:
                logger.warning(f"Conceptual column '{duration_conceptual_key}' not mapped. Defaulting 'duration_minutes_numeric' to NA.")
            else:
                 logger.warning(f"Mapped '{duration_conceptual_key}' (from CSV '{conceptual_to_original_csv_map.get(duration_conceptual_key, 'N/A')}') not numeric. Defaulting 'duration_minutes_numeric' to NA.")


    except Exception as e: # Outer try-except for the main processing loop
        logger.error(f"Error during main type conversion/cleaning loop: {e}", exc_info=True)
        st.error(f"An unexpected error occurred during data type processing: {e}")
        return None

    # --- Feature Engineering ---
    # Ensure 'date' column exists and is primary datetime after potential combination
    if 'date' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['date']):
        st.error("Critical 'date' column is missing or not in datetime format after processing. Cannot proceed with feature engineering.")
        logger.error("Feature engineering skipped: 'date' column missing or not datetime.")
        return None # Return None if date column is problematic
        
    df.sort_values(by='date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    try:
        # Ensure 'pnl' column exists and is numeric for these calculations
        if 'pnl' not in df.columns or not pd.api.types.is_numeric_dtype(df['pnl']):
            st.error("PnL column is missing or not numeric. Cannot perform PnL-based feature engineering.")
            logger.error("Feature engineering skipped: PnL column missing or not numeric.")
            # Return df as is, or None if this is critical
            return df # Or return None if PnL is absolutely essential for further steps

        df['cumulative_pnl'] = df['pnl'].cumsum()
        df['win'] = df['pnl'] > 0

        # trade_result_processed: Based on 'win' or 'trade_outcome_csv_str' if available
        if 'trade_outcome_csv_str' in df.columns:
            df['trade_result_processed'] = df['trade_outcome_csv_str'].astype(str).str.upper()
            valid_outcomes = ['WIN', 'LOSS', 'BREAKEVEN', 'BE'] # BE for Breakeven
            df.loc[~df['trade_result_processed'].isin(valid_outcomes), 'trade_result_processed'] = 'UNKNOWN'
            df.loc[df['trade_result_processed'] == 'BE', 'trade_result_processed'] = 'BREAKEVEN'
        else:
            df['trade_result_processed'] = np.select([df['pnl'] > 0, df['pnl'] < 0], ['WIN', 'LOSS'], default='BREAKEVEN')
        
        # Date/Time derived features (rely on 'date' column being correctly populated with time)
        df['trade_hour'] = df['date'].dt.hour
        df['trade_day_of_week'] = df['date'].dt.day_name()
        df['trade_month_num'] = df['date'].dt.month
        df['trade_month_name'] = df['date'].dt.strftime('%B') # Full month name
        df['trade_year'] = df['date'].dt.year
        df['trade_date_only'] = df['date'].dt.date # Date part only

        # Drawdown calculation
        if 'cumulative_pnl' in df.columns and not df['cumulative_pnl'].empty:
            df['drawdown_abs'], df['drawdown_pct'] = _calculate_drawdown_series_for_df(df['cumulative_pnl'])
        else:
            df['drawdown_abs'] = pd.Series(dtype=float)
            df['drawdown_pct'] = pd.Series(dtype=float)
            logger.warning("Could not calculate drawdown series as 'cumulative_pnl' was missing or empty.")

        # Reward:Risk Ratio (Calculated)
        # This uses 'risk_numeric_internal' which should be the decimal risk (e.g., 0.0029 for 0.29% of capital)
        # The PnL used here is the absolute PnL of the trade.
        # If 'risk_numeric_internal' represents risk amount in currency, then PnL / RiskAmount.
        # If 'risk_numeric_internal' represents risk as % of capital, then PnL / (Risk % * Capital).
        # The current CSV has "Risk %" which is 0.29 for 0.29%. 'risk_numeric_internal' becomes 0.0029.
        # The 'Stop Distance' column in CSV (e.g., 8.467) seems to be the risk in points/price.
        # If 'Size' is 0.3, then actual currency risk would be Stop Distance * Multiplier * Size.
        # The R:R in the CSV is already calculated (e.g., 3.01).
        # For now, 'reward_risk_ratio_calculated' will be PnL / (some risk value).
        # Let's assume for now 'risk_numeric_internal' is meant to be the actual currency risked on the trade.
        # This part needs clarification on how 'risk_numeric_internal' is derived or what it represents.
        # If 'risk_pct' from CSV is used, and it's a percentage of capital, we need capital.
        # If 'Stop Distance' is used, we need a multiplier/value per point.
        # Given the CSV has "R:R", we should prioritize using that if mapped.
        # The 'r_r_csv_num' conceptual column maps to "R:R" from CSV.
        
        # Let's assume 'risk_numeric_internal' is the absolute risk amount for now.
        # If not, this calculation will be incorrect. The user has 'Stop Distance' and 'Size'
        # and 'Multiplier' in their CSV. A better risk amount would be:
        # risk_amount = df['stop_distance_num'] * df['multiplier_value'] * df['trade_size_num'] (if these are mapped)

        if 'pnl' in df.columns and 'risk_numeric_internal' in df.columns and pd.api.types.is_numeric_dtype(df['risk_numeric_internal']):
            df['reward_risk_ratio_calculated'] = df.apply(
                lambda row: row['pnl'] / abs(row['risk_numeric_internal'])
                            if pd.notna(row['pnl']) and pd.notna(row['risk_numeric_internal']) and abs(row['risk_numeric_internal']) > 1e-9 # Avoid division by zero
                            else pd.NA, axis=1
            )
        else:
            df['reward_risk_ratio_calculated'] = pd.NA
            logger.warning("Could not calculate 'reward_risk_ratio_calculated' due to missing PnL or valid 'risk_numeric_internal'.")


        df['trade_number'] = range(1, len(df) + 1)
        logger.info("Feature engineering complete using mapped column names.")
    except Exception as e:
        logger.error(f"Error in feature engineering after mapping: {e}", exc_info=True)
        st.error(f"Feature engineering error after mapping: {e}")
        return df # Return df as is, or None if critical features failed

    if df.empty:
        st.warning("No valid trade data found after processing and mapping."); return None

    logger.info(f"Data processing complete. Final DataFrame shape: {df.shape}. Final columns: {df.columns.tolist()}")
    return df
