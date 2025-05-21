# utils/common_utils.py

import streamlit as st
import pandas as pd
import numpy as np
from typing import Any, Optional, Tuple, Dict, List, Callable
import logging
import sys
import time

try:
    from config import COLORS, APP_TITLE, CONCEPTUAL_COLUMNS
except ImportError:
    print("Warning (common_utils): Could not import from config. Using default UI colors/logger name.", file=sys.stderr)
    APP_TITLE = "TradingDashboard_Default"
    COLORS = {
        "royal_blue": "#4169E1", "green": "#00FF00", "red": "#FF0000",
        "gray": "#808080", "dark_background": "#1C2526",
        "text_dark": "#FFFFFF", "text_muted_color": "#A0A0A0",
        "card_border_dark": "#4169E1",
        "card_background_color": "#2C2C2C", # For tooltip background
        "border_color": "#4A4A4A" # For tooltip border
    }
    CONCEPTUAL_COLUMNS = {}

logger = logging.getLogger(APP_TITLE)

def log_execution_time(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Function '{func.__name__}' executed in {execution_time:.4f} seconds.")
        return result
    return wrapper

def load_css(file_name: str) -> None:
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        logger.info(f"Successfully loaded CSS file: {file_name}")
    except FileNotFoundError:
        logger.warning(f"CSS file not found: {file_name}. Custom styles may not be applied.")
    except Exception as e:
        logger.error(f"Error loading CSS file {file_name}: {e}", exc_info=True)

def display_kpi_card(
    title: str, value: Any, unit: str = "", interpretation: str = "",
    interpretation_desc: str = "", color: str = COLORS.get("gray", "#808080"),
    confidence_interval: Optional[Tuple[float, float]] = None, key_suffix: str = ""
) -> None:
    try:
        formatted_value = "N/A"
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)): formatted_value = "N/A"
        elif pd.isna(value): formatted_value = "N/A"
        elif isinstance(value, (int, float)):
            if unit == "$": formatted_value = format_currency(value)
            elif unit == "%":
                scaled_percentage_kpis = [
                    "win_rate", "loss_rate", "max_drawdown_pct", "alpha", "tracking_error",
                    "benchmark_total_return", "risk_free_rate_used",
                    "expected_annual_return", "annual_volatility"
                ]
                if any(scaled_kpi_part in title.lower().replace(" ", "_") for scaled_kpi_part in scaled_percentage_kpis):
                    formatted_value = format_percentage(value / 100.0) if value is not None else "N/A"
                else:
                    formatted_value = format_percentage(value) if value is not None else "N/A"
            else: formatted_value = f"{value:,.2f}{unit}"
        else: formatted_value = f"{str(value)}{unit}"

        color_class = "neutral"
        if color.upper() == COLORS.get("green", "#00FF00").upper(): color_class = "positive"
        elif color.upper() == COLORS.get("red", "#FF0000").upper(): color_class = "negative"

        card_html = f"""
        <div class="kpi-card {color_class}" key="kpi-card-{title.replace(' ','_')}-{key_suffix}">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value" style="color:{color};">{formatted_value}</div>
            <div class="kpi-interpretation">{interpretation}</div>
            <div class="kpi-interpretation" style="font-size:0.7rem; color:{COLORS.get('text_muted_color', COLORS.get('gray', '#808080'))};">{interpretation_desc}</div>
        """
        if confidence_interval and not any(pd.isna(ci_val) for ci_val in confidence_interval):
            lower_ci_val, upper_ci_val = confidence_interval[0], confidence_interval[1]
            if unit == "%":
                if any(scaled_kpi_part in title.lower().replace(" ", "_") for scaled_kpi_part in ["win_rate", "loss_rate", "drawdown_pct", "alpha", "tracking_error", "benchmark_total_return", "risk_free_rate", "annual_return", "annual_volatility"]):
                     ci_text = f"95% CI: [{format_percentage(lower_ci_val / 100.0)}, {format_percentage(upper_ci_val / 100.0)}]"
                else:
                     ci_text = f"95% CI: [{format_percentage(lower_ci_val)}, {format_percentage(upper_ci_val)}]"
            elif unit == "$":
                ci_text = f"95% CI: [{format_currency(lower_ci_val)}, {format_currency(upper_ci_val)}]"
            else:
                 ci_text = f"95% CI: [{lower_ci_val:.2f}{unit}, {upper_ci_val:.2f}{unit}]"
            card_html += f"""<div class="kpi-confidence-interval">{ci_text}</div>"""
        card_html += "</div>"
        st.markdown(card_html, unsafe_allow_html=True)
    except Exception as e:
        logger.error(f"Error displaying KPI card for '{title}': {e}", exc_info=True)

def display_custom_message(message: str, message_type: str = "info", icon: Optional[str] = None) -> None:
    icon_map = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌"}
    display_icon = icon if icon else icon_map.get(message_type, "")
    message_html = f"""<div class="message-box {message_type.lower()}">{display_icon} {message}</div>"""
    st.markdown(message_html, unsafe_allow_html=True)

def format_currency(value: float, currency_symbol: str = "$", decimals: int = 2) -> str:
    if pd.isna(value) or np.isinf(value): return "N/A"
    return f"{currency_symbol}{value:,.{decimals}f}"

def format_percentage(value: float, decimals: int = 2) -> str:
    if pd.isna(value) or np.isinf(value): return "N/A"
    return f"{value * 100:.{decimals}f}%"

def calculate_portfolio_turnover(current_weights: Dict[str, float], new_weights: Dict[str, float]) -> float:
    if not isinstance(current_weights, dict) or not isinstance(new_weights, dict):
        logger.error("Turnover calculation requires current_weights and new_weights to be dictionaries.")
        return np.nan
    all_assets = set(current_weights.keys()) | set(new_weights.keys())
    turnover_sum = sum(abs(new_weights.get(asset, 0.0) - current_weights.get(asset, 0.0)) for asset in all_assets)
    return 0.5 * turnover_sum

def check_and_display_column_warning(
    df_columns: pd.Index,
    required_conceptual_keys: List[str],
    feature_name: str,
    is_critical: bool = False
) -> bool:
    missing_cols = [
        key for key in required_conceptual_keys if key not in df_columns
    ]
    if missing_cols:
        missing_cols_desc = [CONCEPTUAL_COLUMNS.get(key, key) for key in missing_cols]
        message = (
            f"Cannot generate {feature_name}. Required data field(s) "
            f"'{', '.join(missing_cols_desc)}' "
            f"were not found or not mapped from your CSV. "
            f"Please ensure these fields are present and correctly mapped in the column mapping step."
        )
        if is_critical:
            display_custom_message(message, "error")
            logger.error(f"Critical columns missing for {feature_name}: {missing_cols}")
        else:
            display_custom_message(message, "warning")
            logger.warning(f"Optional columns missing for {feature_name}: {missing_cols}")
        return False
    return True

def get_title_with_tooltip_html(
    title_text: str,
    tooltip_text: str,
    header_level: int = 3,
    title_id: Optional[str] = None,
    container_style: Optional[str] = None
) -> str:
    default_id_from_title = "".join(filter(str.isalnum, title_text)).lower()
    final_title_id = title_id if title_id else default_id_from_title
    title_id_attribute = f"id='{final_title_id}'" if final_title_id else ""
    sanitized_tooltip_text = tooltip_text.replace("'", "&apos;").replace("\"", "&quot;")
    container_style_attr = f"style='{container_style}'" if container_style else ""

    # Add a class to the container based on the header level
    container_class = f"title-with-tooltip-container title-container-h{header_level}"

    html_content = f"""
    <div class="{container_class}" {container_style_attr}>
        <h{header_level} {title_id_attribute} style="margin-right: 0px;">{title_text}</h{header_level}>
        <span class="tooltip-icon-container">
            <span class="tooltip-icon" data-tooltip="{sanitized_tooltip_text}">&#9432;</span>
        </span>
    </div>
    """
    return html_content

if __name__ == "__main__":
    # ... (rest of the __main__ block for testing common_utils.py) ...
    st.markdown("### Test Tooltip Function (v2)")
    tooltip_html_h2 = get_title_with_tooltip_html(
        title_text="My Awesome Chart (H2)",
        tooltip_text="This chart visualizes important financial trends.",
        header_level=2,
        title_id="awesome-chart-section-h2"
    )
    st.markdown(tooltip_html_h2, unsafe_allow_html=True)
    
    tooltip_html_h3 = get_title_with_tooltip_html(
        title_text="Another Metric (H3)",
        tooltip_text="This metric is key. It's calculated using X and Y.",
        header_level=3
    )
    st.markdown(tooltip_html_h3, unsafe_allow_html=True)
