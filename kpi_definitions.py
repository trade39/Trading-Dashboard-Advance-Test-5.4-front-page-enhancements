# kpi_definitions.py
"""
This file contains the detailed configuration for Key Performance Indicators (KPIs),
including their names, units, interpretation thresholds, and color logic.
"""

from typing import Dict, List, Tuple, Callable # Added Callable
import numpy as np # For np.nan, np.isinf if used in color_logic or thresholds

# Note: The COLORS dictionary itself is defined in config.py and will be passed
# into these lambdas by the get_kpi_color function in calculations.py.

KPI_CONFIG: Dict[str, Dict] = {
    "total_pnl": {
        "name": "Total PnL", "unit": "$", "interpretation_type": "higher_is_better",
        "thresholds": [
            ("Negative", float('-inf'), 0),
            ("Slightly Positive", 0, 1000),
            ("Moderately Positive", 1000, 10000),
            ("Highly Positive", 10000, float('inf'))
        ],
        "color_logic": lambda v, t, colors: colors.get("green", "#00FF00") if v > 0 else (colors.get("red", "#FF0000") if v < 0 else colors.get("gray", "#808080"))
    },
    "total_trades": {
        "name": "Total Trades", "unit": "", "interpretation_type": "neutral",
        "thresholds": [
            ("Low", 0, 50), ("Moderate", 50, 200), ("High", 200, float('inf'))
        ],
        "color_logic": lambda v, t, colors: colors.get("gray", "#808080")
    },
    "win_rate": {
        "name": "Win Rate", "unit": "%", "interpretation_type": "higher_is_better",
        "thresholds": [
            ("Very Low", 0, 30),("Low", 30, 40),("Acceptable", 40, 50),
            ("Good", 50, 60),("Very Good", 60, 70),("Excellent", 70, 80),
            ("Exceptional", 80, 101)
        ],
        "color_logic": lambda v, t, colors: colors.get("green", "#00FF00") if v >= 50 else colors.get("red", "#FF0000")
    },
    "loss_rate": {
        "name": "Loss Rate", "unit": "%", "interpretation_type": "lower_is_better",
        "thresholds": [
            ("Exceptional", 0, 20),("Excellent", 20, 30),("Very Good", 30, 40),
            ("Good", 40, 50),("Acceptable", 50, 60),("High", 60, 70),
            ("Very High", 70, 101)
        ],
        "color_logic": lambda v, t, colors: colors.get("red", "#FF0000") if v > 50 else colors.get("green", "#00FF00")
    },
    "profit_factor": {
        "name": "Profit Factor", "unit": "", "interpretation_type": "higher_is_better",
        "thresholds": [
            ("Negative", float('-inf'), 1.0), ("Break-even", 1.0, 1.01),
            ("Acceptable", 1.01, 1.5),("Good", 1.5, 2.0),("Very Good", 2.0, 3.0),
            ("Exceptional", 3.0, float('inf'))
        ],
        "color_logic": lambda v, t, colors: colors.get("green", "#00FF00") if v > 1 else colors.get("red", "#FF0000")
    },
    "avg_trade_pnl": {
        "name": "Average Trade PnL", "unit": "$", "interpretation_type": "higher_is_better",
        "thresholds": [
            ("Negative", float('-inf'), 0),("Neutral", 0, 1),("Positive", 1, float('inf'))
        ],
        "color_logic": lambda v, t, colors: colors.get("green", "#00FF00") if v > 0 else (colors.get("red", "#FF0000") if v < 0 else colors.get("gray", "#808080"))
    },
    "avg_win": {
        "name": "Average Win", "unit": "$", "interpretation_type": "higher_is_better",
        "thresholds": [
            ("Low", 0, 50), ("Moderate", 50, 200),("High", 200, float('inf'))
        ],
        "color_logic": lambda v, t, colors: colors.get("green", "#00FF00") if v > 0 else colors.get("gray", "#808080")
    },
    "avg_loss": {
        "name": "Average Loss", "unit": "$", "interpretation_type": "lower_is_better",
        "thresholds": [
            ("Low", 0, 50),("Moderate", 50, 200),("High", 200, float('inf'))
        ],
        "color_logic": lambda v, t, colors: colors.get("red", "#FF0000") if v > 0 else colors.get("gray", "#808080") # Note: avg_loss is absolute value
    },
    "win_loss_ratio": {
        "name": "Win/Loss Ratio", "unit": "", "interpretation_type": "higher_is_better",
        "thresholds": [
            ("Poor", 0, 1.0),("Acceptable", 1.0, 1.5),("Good", 1.5, 2.0),
            ("Very Good", 2.0, 3.0),("Exceptional", 3.0, float('inf'))
        ],
        "color_logic": lambda v, t, colors: colors.get("green", "#00FF00") if v > 1 else colors.get("red", "#FF0000")
    },
    "max_drawdown_abs": {
        "name": "Max Drawdown Abs", "unit": "$", "interpretation_type": "lower_is_better",
        "thresholds": [
            ("Low", 0, 1000),("Moderate", 1000, 5000),("High", 5000, float('inf'))
        ],
        "color_logic": lambda v, t, colors: colors.get("red", "#FF0000") if v > 0 else colors.get("gray", "#808080") # Absolute DD is always positive if non-zero
    },
    "max_drawdown_pct": {
        "name": "Max Drawdown %", "unit": "%", "interpretation_type": "lower_is_better",
        "thresholds": [
            ("Very Low", 0, 5),("Low", 5, 10),("Moderate", 10, 20),
            ("High (Caution)", 20, 30),("Very High (Danger)", 30, 101)
        ],
        "color_logic": lambda v, t, colors: colors.get("red", "#FF0000") if v >= 20 else (colors.get("green", "#00FF00") if v < 10 else colors.get("gray", "#808080"))
    },
    "sharpe_ratio": {
        "name": "Sharpe Ratio", "unit": "", "interpretation_type": "higher_is_better",
        "thresholds": [
            ("Poor", float('-inf'), 0),("Subpar", 0, 1.0),("Good", 1.0, 2.0),
            ("Excellent", 2.0, 3.0),("Exceptional", 3.0, float('inf'))
        ],
        "color_logic": lambda v, t, colors: colors.get("green", "#00FF00") if v > 1 else (colors.get("red", "#FF0000") if v < 0 else colors.get("gray", "#808080"))
    },
    "sortino_ratio": {
        "name": "Sortino Ratio", "unit": "", "interpretation_type": "higher_is_better",
        "thresholds": [
            ("Poor", float('-inf'), 0),("Subpar", 0, 1.0),("Good", 1.0, 2.0),
            ("Excellent", 2.0, 3.0),("Exceptional", 3.0, float('inf'))
        ],
        "color_logic": lambda v, t, colors: colors.get("green", "#00FF00") if v > 1 else (colors.get("red", "#FF0000") if v < 0 else colors.get("gray", "#808080"))
    },
    "calmar_ratio": {
        "name": "Calmar Ratio", "unit": "", "interpretation_type": "higher_is_better",
        "thresholds": [
            ("Poor", float('-inf'), 0),("Subpar", 0, 0.5),("Acceptable", 0.5, 1.0),
            ("Good", 1.0, 2.0),("Excellent", 2.0, float('inf'))
        ],
        "color_logic": lambda v, t, colors: colors.get("green", "#00FF00") if v > 1 else (colors.get("red", "#FF0000") if v < 0 else colors.get("gray", "#808080"))
    },
    "var_95_loss": {
        "name": "VaR 95% (Loss)", "unit": "$", "interpretation_type": "lower_is_better",
        "thresholds": [
            ("Low Risk", 0, 500), ("Moderate Risk", 500, 2000), ("High Risk", 2000, float('inf'))
        ],
        "color_logic": lambda v, t, colors: colors.get("red", "#FF0000") if v > 1000 else (colors.get("gray", "#808080") if v == 0 else colors.get("orange", "#FFA500"))
    },
    "cvar_95_loss": {
        "name": "CVaR 95% (Loss)", "unit": "$", "interpretation_type": "lower_is_better",
        "thresholds": [
            ("Low Risk", 0, 500), ("Moderate Risk", 500, 2000), ("High Risk", 2000, float('inf'))
        ],
        "color_logic": lambda v, t, colors: colors.get("red", "#FF0000") if v > 1000 else (colors.get("gray", "#808080") if v == 0 else colors.get("orange", "#FFA500"))
    },
    "var_99_loss": {
        "name": "VaR 99% (Loss)", "unit": "$", "interpretation_type": "lower_is_better",
        "thresholds": [
            ("Low Risk", 0, 750), ("Moderate Risk", 750, 3000), ("High Risk", 3000, float('inf'))
        ],
        "color_logic": lambda v, t, colors: colors.get("red", "#FF0000") if v > 1500 else (colors.get("gray", "#808080") if v == 0 else colors.get("orange", "#FFA500"))
    },
    "cvar_99_loss": {
        "name": "CVaR 99% (Loss)", "unit": "$", "interpretation_type": "lower_is_better",
        "thresholds": [
            ("Low Risk", 0, 750), ("Moderate Risk", 750, 3000), ("High Risk", 3000, float('inf'))
        ],
        "color_logic": lambda v, t, colors: colors.get("red", "#FF0000") if v > 1500 else (colors.get("gray", "#808080") if v == 0 else colors.get("orange", "#FFA500"))
    },
    "pnl_skewness": {
        "name": "PnL Skewness", "unit": "", "interpretation_type": "neutral",
        "thresholds": [
            ("Highly Negative", float('-inf'), -1.0), ("Moderately Negative", -1.0, -0.5),
            ("Symmetric", -0.5, 0.5), ("Moderately Positive", 0.5, 1.0),
            ("Highly Positive", 1.0, float('inf'))
        ],
        "color_logic": lambda v, t, colors: colors.get("green", "#00FF00") if v > 0.5 else (colors.get("red", "#FF0000") if v < -0.5 else colors.get("gray", "#808080"))
    },
    "pnl_kurtosis": {
        "name": "PnL Kurtosis (Excess)", "unit": "", "interpretation_type": "neutral",
        "thresholds": [
            ("Platykurtic (Thin)", float('-inf'), -0.5),("Mesokurtic (Normal)", -0.5, 0.5),
            ("Leptokurtic (Fat)", 0.5, 3.0),("Highly Leptokurtic (Very Fat)", 3.0, float('inf'))
        ],
        "color_logic": lambda v, t, colors: colors.get("red", "#FF0000") if v > 1 else colors.get("gray", "#808080") # High kurtosis often means tail risk
    },
    "max_win_streak": {
        "name": "Max Win Streak", "unit": " trades", "interpretation_type": "higher_is_better",
        "thresholds": [
            ("Low", 0, 3),("Moderate", 3, 7),("High", 7, float('inf'))
        ],
        "color_logic": lambda v, t, colors: colors.get("green", "#00FF00") if v >= 3 else colors.get("gray", "#808080")
    },
    "max_loss_streak": {
        "name": "Max Loss Streak", "unit": " trades", "interpretation_type": "lower_is_better",
        "thresholds": [
            ("Low", 0, 3),("Moderate", 3, 7),("High", 7, float('inf'))
        ],
        "color_logic": lambda v, t, colors: colors.get("red", "#FF0000") if v >= 5 else colors.get("gray", "#808080")
    },
    "avg_daily_pnl": {
        "name": "Average Daily PnL", "unit": "$", "interpretation_type": "higher_is_better",
        "thresholds": [
            ("Negative", float('-inf'), 0),("Neutral", 0, 1),("Positive", 1, float('inf'))
        ],
        "color_logic": lambda v, t, colors: colors.get("green", "#00FF00") if v > 0 else (colors.get("red", "#FF0000") if v < 0 else colors.get("gray", "#808080"))
    },
    "trading_days": {
        "name": "Trading Days", "unit": "", "interpretation_type": "neutral",
        "thresholds": [
            ("Short Period", 0, 21), ("Medium Period", 21, 63),
            ("Sufficient Period", 63, 252),("Long Period", 252, float('inf'))
        ],
        "color_logic": lambda v, t, colors: colors.get("gray", "#808080")
    },
    "risk_free_rate_used": {
        "name": "Risk-Free Rate Used", "unit": "%", "interpretation_type": "neutral",
        "thresholds": [("Standard Setting", 0, float('inf'))],
        "color_logic": lambda v, t, colors: colors.get("gray", "#808080")
    },
    "benchmark_total_return": {
        "name": "Benchmark Total Return", "unit": "%", "interpretation_type": "neutral",
        "thresholds": [
            ("Negative", float('-inf'), 0), ("Positive", 0, float('inf'))
        ],
        "color_logic": lambda v, t, colors: colors.get("green", "#00FF00") if v > 0 else (colors.get("red", "#FF0000") if v < 0 else colors.get("gray", "#808080"))
    },
    "alpha": {
        "name": "Alpha (Annualized)", "unit": "%", "interpretation_type": "higher_is_better",
        "thresholds": [
            ("Negative Alpha", float('-inf'), 0), ("Positive Alpha", 0, float('inf'))
        ],
        "color_logic": lambda v, t, colors: colors.get("green", "#00FF00") if v > 0 else (colors.get("red", "#FF0000") if v < 0 else colors.get("gray", "#808080"))
    },
    "beta": {
        "name": "Beta", "unit": "", "interpretation_type": "neutral",
        "thresholds": [
            ("Low Volatility (vs Benchmark)", 0, 0.8),
            ("Market Volatility (vs Benchmark)", 0.8, 1.2),
            ("High Volatility (vs Benchmark)", 1.2, float('inf'))
        ],
        "color_logic": lambda v, t, colors: colors.get("gray", "#808080")
    },
    "benchmark_correlation": {
        "name": "Correlation to Benchmark", "unit": "", "interpretation_type": "neutral",
        "thresholds": [
            ("Negative", -1.0, -0.5), ("Low", -0.5, 0.5), ("Positive", 0.5, 1.01)
        ],
        "color_logic": lambda v, t, colors: colors.get("purple", "#8A2BE2")
    },
    "tracking_error": {
        "name": "Tracking Error (Annualized)", "unit": "%", "interpretation_type": "lower_is_better",
        "thresholds": [
            ("Low", 0, 5), ("Moderate", 5, 15), ("High", 15, float('inf'))
        ],
        "color_logic": lambda v, t, colors: colors.get("green", "#00FF00") if v < 5 else (colors.get("orange", "#FFA500") if v < 15 else colors.get("red", "#FF0000"))
    },
    "information_ratio": {
        "name": "Information Ratio", "unit": "", "interpretation_type": "higher_is_better",
        "thresholds": [
            ("Poor", float('-inf'), 0.0), ("Fair", 0.0, 0.5), ("Good", 0.5, 1.0),
            ("Excellent", 1.0, float('inf'))
        ],
        "color_logic": lambda v, t, colors: colors.get("green", "#00FF00") if v > 0.5 else (colors.get("orange", "#FFA500") if v > 0 else colors.get("red", "#FF0000"))
    },
    "expected_annual_return": {
        "name": "Expected Annual Return", "unit": "%", "interpretation_type": "higher_is_better",
        "thresholds": [
            ("Negative", float('-inf'), 0), ("Low", 0, 5), # Thresholds are direct percentages
            ("Moderate", 5, 15), ("Good", 15, 25),
            ("Excellent", 25, float('inf'))
        ],
        # v here is expected to be a raw decimal (e.g., 0.1 for 10%)
        "color_logic": lambda v, t, colors: colors.get("green", "#00FF00") if v >= 0.10 else (colors.get("orange", "#FFA500") if v >= 0 else colors.get("red", "#FF0000"))
    },
    "annual_volatility": {
        "name": "Annual Volatility", "unit": "%", "interpretation_type": "lower_is_better",
        "thresholds": [ # Thresholds are direct percentages
            ("Very Low", 0, 5), ("Low", 5, 10),
            ("Moderate", 10, 20), ("High", 20, 30),
            ("Very High", 30, float('inf'))
        ],
        # v here is expected to be a raw decimal (e.g., 0.1 for 10%)
        "color_logic": lambda v, t, colors: colors.get("green", "#00FF00") if v < 0.10 else (colors.get("orange", "#FFA500") if v < 0.20 else colors.get("red", "#FF0000"))
    }
}
