"""
components/kpi_display.py

This component is responsible for orchestrating the display of multiple
Key Performance Indicators (KPIs) in a structured layout, typically using cards.
It leverages the `display_kpi_card` utility for individual card rendering
and now explicitly handles confidence intervals and benchmark context for titles.
"""
import streamlit as st
from typing import Dict, Any, List, Optional, Tuple

try:
    from config import KPI_CONFIG, DEFAULT_KPI_DISPLAY_ORDER, APP_TITLE
    from utils.common_utils import display_kpi_card
    from calculations import get_kpi_interpretation, get_kpi_color
except ImportError:
    print("Warning (kpi_display.py): Could not import from root config/utils/calculations. Using placeholders.")
    APP_TITLE = "TradingDashboard_Default"
    DEFAULT_KPI_DISPLAY_ORDER = ["total_pnl", "win_rate", "profit_factor", "sharpe_ratio"]
    KPI_CONFIG = { 
        "total_pnl": {"name": "Total PnL", "unit": "$"}, "win_rate": {"name": "Win Rate", "unit": "%"},
        "profit_factor": {"name": "Profit Factor"}, "sharpe_ratio": {"name": "Sharpe Ratio"},
        "alpha": {"name": "Alpha", "unit": "%"}, "beta": {"name": "Beta"} # Added for testing
    }
    def display_kpi_card(title, value, unit, interpretation, interpretation_desc, color, confidence_interval=None, key_suffix=""):
        ci_text = f" CI: [{confidence_interval[0]:.2f}-{confidence_interval[1]:.2f}]" if confidence_interval else ""
        st.metric(label=title, value=f"{value}{unit}{ci_text}", delta=interpretation_desc if interpretation_desc else interpretation)

    def get_kpi_interpretation(k_key, val): return "N/A", f"Val: {val}"
    def get_kpi_color(k_key, val): return "#808080" 

import logging
logger = logging.getLogger(APP_TITLE) 

class KPIClusterDisplay:
    def __init__(
        self,
        kpi_results: Dict[str, Any],
        kpi_definitions: Dict[str, Dict] = KPI_CONFIG,
        kpi_order: List[str] = DEFAULT_KPI_DISPLAY_ORDER,
        kpi_confidence_intervals: Optional[Dict[str, Tuple[float, float]]] = None,
        cols_per_row: int = 4,
        benchmark_context_name: Optional[str] = None # New parameter
    ):
        self.kpi_results = kpi_results if kpi_results else {}
        self.kpi_definitions = kpi_definitions
        self.kpi_order = kpi_order
        self.kpi_confidence_intervals = kpi_confidence_intervals if kpi_confidence_intervals else {}
        self.cols_per_row = max(1, cols_per_row)
        self.benchmark_context_name = benchmark_context_name # Store benchmark name
        logger.debug(f"KPIClusterDisplay initialized. Benchmark context: {self.benchmark_context_name}")

    def render(self) -> None:
        if not self.kpi_results:
            logger.info("KPIClusterDisplay: No KPI results to display.")
            return

        st_cols = st.columns(self.cols_per_row)
        current_col_idx = 0

        # List of KPIs that are benchmark-relative
        benchmark_relative_kpis = ["alpha", "beta", "benchmark_correlation", "tracking_error", "information_ratio"]

        for kpi_key in self.kpi_order:
            if kpi_key in self.kpi_results:
                value = self.kpi_results[kpi_key]
                kpi_conf = self.kpi_definitions.get(kpi_key, {})

                name = kpi_conf.get("name", kpi_key.replace("_", " ").title())
                unit = kpi_conf.get("unit", "")

                # --- Add benchmark context to title if applicable ---
                if kpi_key in benchmark_relative_kpis and self.benchmark_context_name and self.benchmark_context_name != "None":
                    name = f"{name} (vs. {self.benchmark_context_name})"
                # --- End benchmark context modification ---


                interpretation, desc = get_kpi_interpretation(kpi_key, value)
                color = get_kpi_color(kpi_key, value)
                ci_data = self.kpi_confidence_intervals.get(kpi_key)

                with st_cols[current_col_idx % self.cols_per_row]:
                    display_kpi_card(
                        title=name,
                        value=value,
                        unit=unit,
                        interpretation=interpretation,
                        interpretation_desc=desc,
                        color=color,
                        confidence_interval=ci_data,
                        key_suffix=f"cluster_{kpi_key}"
                    )
                current_col_idx += 1
            else:
                logger.debug(f"KPIClusterDisplay: KPI key '{kpi_key}' from order not found in results. Skipping.")
        logger.debug("KPIClusterDisplay rendering complete.")

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.title("Test KPI Cluster Display (Benchmark Context)")

    mock_kpi_results_bench = {
        "total_pnl": 10500.75, "win_rate": 65.5, "profit_factor": 2.15,
        "sharpe_ratio": 1.8, "max_drawdown_pct": 15.2, "avg_trade_pnl": 50.20,
        "alpha": 5.5, "beta": 1.1, "benchmark_correlation": 0.75
    }
    try:
        from calculations import get_kpi_interpretation, get_kpi_color
    except ImportError:
        def get_kpi_interpretation(kpi_key, value): return f"Interp for {kpi_key}", f"Val: {value:.2f}"
        def get_kpi_color(kpi_key, value): return "#00FF00" if value > 0 else "#FF0000"

    mock_kpi_order_bench = ["total_pnl", "win_rate", "alpha", "beta", "benchmark_correlation", "sharpe_ratio"]
    mock_cis_bench = {"alpha": (2.0, 8.0), "beta": (0.9, 1.3)}

    st.subheader("KPI Cluster with Benchmark Context (SPY)")
    kpi_cluster_bench = KPIClusterDisplay(
        kpi_results=mock_kpi_results_bench,
        kpi_definitions=KPI_CONFIG,
        kpi_order=mock_kpi_order_bench,
        kpi_confidence_intervals=mock_cis_bench,
        cols_per_row=3,
        benchmark_context_name="S&P 500 (SPY)" # Provide context
    )
    kpi_cluster_bench.render()

    st.subheader("KPI Cluster without Benchmark Context (None)")
    kpi_cluster_no_bench = KPIClusterDisplay(
        kpi_results=mock_kpi_results_bench,
        kpi_definitions=KPI_CONFIG,
        kpi_order=mock_kpi_order_bench,
        kpi_confidence_intervals=mock_cis_bench,
        cols_per_row=3,
        benchmark_context_name="None" # Simulate "None" selected
    )
    kpi_cluster_no_bench.render()

    logger.info("KPIClusterDisplay benchmark context test complete.")

