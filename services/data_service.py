# services/data_service.py

import streamlit as st
import pandas as pd
from typing import Optional, Any, Dict
import yfinance as yf # Added for benchmark data
import datetime # Added for benchmark data
import logging

try:
    from config import APP_TITLE, EXPECTED_COLUMNS
    from data_processing import load_and_process_data
except ImportError:
    print("Warning (data_service.py): Could not import from root config or data_processing. Using placeholders.")
    APP_TITLE = "TradingDashboard_Default_Service"
    EXPECTED_COLUMNS = {"date": "date", "pnl": "pnl", "symbol": "symbol", "strategy": "strategy"}
    def load_and_process_data(uploaded_file_obj: Any, user_column_mapping: Optional[Dict[str, str]] = None) -> Optional[pd.DataFrame]:
        if uploaded_file_obj:
            try:
                df = pd.read_csv(uploaded_file_obj)
                df.columns = [col.lower().replace(' ', '_') for col in df.columns]
                if 'date' in df.columns: df['date'] = pd.to_datetime(df['date'], errors='coerce')
                if 'pnl' in df.columns: df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce')
                return df
            except Exception as e:
                print(f"Placeholder load_and_process_data error: {e}")
        return None

logger = logging.getLogger(APP_TITLE)

# --- MODIFICATION START: Moved get_benchmark_data_static here ---
@st.cache_data(ttl=3600)
def get_benchmark_data_static(
    ticker: str,
    start_date_str: str,
    end_date_str: str
) -> Optional[pd.Series]:
    """
    Fetches historical market data for a given ticker and calculates daily returns.
    This function is cached to avoid redundant API calls.

    Args:
        ticker (str): The stock ticker symbol (e.g., "SPY").
        start_date_str (str): Start date in "YYYY-MM-DD" format.
        end_date_str (str): End date in "YYYY-MM-DD" format.

    Returns:
        Optional[pd.Series]: A Series of daily returns for the benchmark, or None if fetching fails.
    """
    logger_static_func = logging.getLogger(f"{APP_TITLE}.get_benchmark_data_static") # Use main APP_TITLE
    logger_static_func.info(f"Executing get_benchmark_data_static (caching active) for {ticker} from {start_date_str} to {end_date_str}")

    if not ticker:
        logger_static_func.info("No benchmark ticker provided. Skipping data fetch.")
        return None
    try:
        logger_static_func.debug("Attempting to convert date strings to datetime objects...")
        start_dt = pd.to_datetime(start_date_str)
        end_dt = pd.to_datetime(end_date_str)
        logger_static_func.debug(f"Converted dates: start_dt={start_dt}, end_dt={end_dt}")

        if start_dt > end_dt:
            logger_static_func.warning(f"Benchmark start date {start_date_str} is after end date {end_date_str}. Cannot fetch data.")
            return None
        
        fetch_end_dt = end_dt + pd.Timedelta(days=1)
        if start_dt == end_dt: # If fetching for a single day, yfinance needs a slightly larger range
            fetch_end_dt = end_dt + pd.Timedelta(days=2)


        logger_static_func.info(f"Attempting yf.download for {ticker} from {start_dt.date()} to {end_dt.date()} (fetching up to {fetch_end_dt.date()})")

        data = yf.download(ticker, start=start_dt, end=fetch_end_dt, progress=False, auto_adjust=True, actions=False)
        logger_static_func.debug(f"yf.download result for {ticker}:\n{data.head() if not data.empty else 'Empty DataFrame'}")

        if data.empty or 'Close' not in data.columns:
            logger_static_func.warning(f"No data or 'Close' (adjusted) not found for benchmark {ticker} in period {start_date_str} - {end_date_str}.")
            return None

        daily_adj_close = data['Close'].dropna()
        if len(daily_adj_close) < 2: 
            logger_static_func.warning(f"Not enough benchmark data points for {ticker} to calculate returns (<2). Fetched {len(daily_adj_close)} points.")
            return None

        daily_returns = daily_adj_close.pct_change().dropna()
        if daily_returns.empty:
            logger_static_func.warning(f"Calculated daily returns for benchmark {ticker} are empty. This can happen if all fetched prices were identical or only one valid price point after pct_change.")
            return None
            
        daily_returns.name = f"{ticker}_returns" # Assign a name to the series

        logger_static_func.info(f"Successfully fetched and processed benchmark returns for {ticker}. Shape: {daily_returns.shape}")
        return daily_returns
    except Exception as e:
        logger_static_func.error(f"Error fetching benchmark data for {ticker}: {e}", exc_info=True)
        return None
# --- MODIFICATION END ---

class DataService:
    def __init__(self):
        logger.info("DataService initialized.")

    def get_processed_trading_data(
        self,
        uploaded_file_obj: Any, # Can be Streamlit's UploadedFile or BytesIO
        user_column_mapping: Optional[Dict[str, str]] = None,
        original_file_name: Optional[str] = None 
    ) -> Optional[pd.DataFrame]:
        """
        Loads and processes trading data from an uploaded file object, applying user column mapping.
        """
        if uploaded_file_obj is None:
            logger.debug("DataService: No file object provided.")
            return None

        file_name_for_log = original_file_name
        if not file_name_for_log and hasattr(uploaded_file_obj, 'name'):
            file_name_for_log = uploaded_file_obj.name
        elif not file_name_for_log:
            file_name_for_log = "In-memory CSV data (BytesIO)"

        try:
            logger.info(f"DataService: Attempting to process file: '{file_name_for_log}' with user mapping.")
            processed_df = load_and_process_data(
                uploaded_file_obj,
                user_column_mapping=user_column_mapping
            )

            if processed_df is not None:
                logger.info(f"DataService: File '{file_name_for_log}' processed successfully. Shape: {processed_df.shape}")
            else:
                logger.warning(f"DataService: Processing of file '{file_name_for_log}' returned None.")
            return processed_df
        except Exception as e:
            logger.error(f"DataService: Unexpected error during data processing for '{file_name_for_log}': {e}", exc_info=True)
            return None

    def filter_data(
        self,
        df: pd.DataFrame,
        filters: Dict[str, Any],
        column_map: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Applies a set of filters to the DataFrame.
        Assumes df has columns named with conceptual keys after mapping.
        """
        if df is None or df.empty:
            logger.debug("DataService: DataFrame for filtering is None or empty.")
            return pd.DataFrame()

        effective_column_map = column_map if column_map is not None else EXPECTED_COLUMNS

        filtered_df = df.copy()
        logger.info(f"DataService: Applying filters. Initial shape: {filtered_df.shape}. Filters: {filters}")

        date_col_conceptual = effective_column_map.get('date', 'date')
        date_range = filters.get('selected_date_range')
        if date_col_conceptual in filtered_df.columns and date_range and len(date_range) == 2:
            try:
                start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
                # Ensure comparison is done on date part only if filter is date-only
                filtered_df = filtered_df[
                    (pd.to_datetime(filtered_df[date_col_conceptual]).dt.date >= start_date.date()) &
                    (pd.to_datetime(filtered_df[date_col_conceptual]).dt.date <= end_date.date())
                ]
                logger.debug(f"Applied date filter on '{date_col_conceptual}'. Shape after: {filtered_df.shape}")
            except Exception as e:
                logger.error(f"Error applying date filter on '{date_col_conceptual}': {e}", exc_info=True)

        symbol_col_conceptual = effective_column_map.get('symbol', 'symbol')
        selected_symbol = filters.get('selected_symbol')
        if symbol_col_conceptual in filtered_df.columns and selected_symbol and selected_symbol != "All":
            try:
                filtered_df = filtered_df[filtered_df[symbol_col_conceptual].astype(str) == str(selected_symbol)]
                logger.debug(f"Applied symbol filter '{selected_symbol}' on '{symbol_col_conceptual}'. Shape: {filtered_df.shape}")
            except Exception as e:
                logger.error(f"Error applying symbol filter on '{symbol_col_conceptual}': {e}", exc_info=True)

        strategy_col_conceptual = effective_column_map.get('strategy', 'strategy')
        selected_strategy = filters.get('selected_strategy')
        if strategy_col_conceptual in filtered_df.columns and selected_strategy and selected_strategy != "All":
            try:
                filtered_df = filtered_df[filtered_df[strategy_col_conceptual].astype(str) == str(selected_strategy)]
                logger.debug(f"Applied strategy filter '{selected_strategy}' on '{strategy_col_conceptual}'. Shape: {filtered_df.shape}")
            except Exception as e:
                logger.error(f"Error applying strategy filter on '{strategy_col_conceptual}': {e}", exc_info=True)

        logger.info(f"DataService: Filtering complete. Final shape: {filtered_df.shape}")
        return filtered_df

if __name__ == "__main__":
    logger.info("--- Testing DataService (with benchmark function) ---")
    # Example usage (requires yfinance and network)
    # test_returns = get_benchmark_data_static("AAPL", "2023-01-01", "2023-01-10")
    # if test_returns is not None:
    #     print("Fetched AAPL returns:")
    #     print(test_returns.head())
    # else:
    #     print("Failed to fetch AAPL returns.")
    pass
