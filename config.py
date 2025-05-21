# config.py

from typing import Dict, List, Tuple
from collections import OrderedDict # Added for CONCEPTUAL_COLUMN_CATEGORIES

# --- Import KPI Definitions ---
try:
    from kpi_definitions import KPI_CONFIG
except ImportError:
    print("Warning (config.py): Could not import KPI_CONFIG from kpi_definitions.py. KPI functionality may be limited.")
    KPI_CONFIG = {} # Fallback to an empty dict

# --- General Settings ---
APP_TITLE: str = "Trading Mastery Hub"
RISK_FREE_RATE: float = 0.02 # Default Risk-Free Rate
FORECAST_HORIZON: int = 30

# --- Benchmark Configuration ---
DEFAULT_BENCHMARK_TICKER: str = "SPY"
AVAILABLE_BENCHMARKS: Dict[str, str] = {
    "S&P 500 (SPY)": "SPY",
    "Nasdaq 100 (QQQ)": "QQQ",
    "Dow Jones (DIA)": "DIA",
    "Russell 2000 (IWM)": "IWM",
    "Gold (GLD)": "GLD",
    "None": ""
}

# --- Conceptual Column Definitions ---
# Keys are the internal "conceptual" column names the application uses.
# Values are user-friendly descriptions/names for these concepts, used in the UI.
CONCEPTUAL_COLUMNS: Dict[str, str] = {
    "date": "Trade Date/Time (Primary)",
    "pnl": "Profit or Loss (PnL)",
    "symbol": "Trading Symbol/Ticker",
    "strategy": "Strategy Name/Identifier",
    "trade_id": "Unique Trade ID",
    "entry_price": "Entry Price",
    "exit_price": "Exit Price",
    "risk_pct": "Risk Percentage (%)",
    "notes": "Trade Notes/Comments",
    "signal_confidence": "Signal Confidence Score",
    "duration_minutes": "Trade Duration (minutes)",
    "entry_time_str": "Entry Time (Raw String, if separate)",
    "trade_month_str": "Trade Month (Raw String)",
    "trade_day_str": "Trade Day of Week (Raw String)",
    "trade_plan_str": "Trade Plan/Setup",
    "bias_str": "Market Bias",
    "tag_str": "Custom Tags",
    "time_frame_str": "Trading Time Frame",
    "direction_str": "Trade Direction (Long/Short)",
    "trade_size_num": "Trade Size/Quantity",
    "r_r_csv_num": "Risk:Reward Ratio (from CSV)",
    "session_str": "Trading Session",
    "market_conditions_str": "Market Conditions Description",
    "event_type_str": "Economic Event Type",
    "events_details_str": "Economic Event Details", # Retained as the primary field for event details
    "market_sentiment_str": "Market Sentiment",
    "psychological_factors_str": "Psychological Factors",
    "compliance_check_str": "Compliance Check Status",
    "account_str": "Trading Account ID/Name",
    "initial_balance_num": "Account Initial Balance",
    "current_balance_num": "Account Current Balance",
    "drawdown_value_csv": "Drawdown Value (from CSV)",
    "trade_outcome_csv_str": "Trade Outcome (Win/Loss/BE from CSV)",
    "exit_type_csv_str": "Exit Type/Reason",
    "loss_indicator_num": "Loss Indicator (Numeric)",
    "win_indicator_num": "Win Indicator (Numeric)",
    "stop_distance_num": "Stop Loss Distance",
    "candle_count_num": "Candle Count for Trade Duration",
    "cumulative_equity_csv": "Cumulative Equity (from CSV)",
    "absolute_daily_pnl_csv": "Absolute Daily PnL (from CSV)",
    "error_exit_type_related_str": "Error Type (Exit Related)",
    "profit_value_csv": "Gross Profit Value (from CSV)",
    "loss_value_csv": "Gross Loss Value (from CSV)",
    "duration_hrs_csv": "Trade Duration (hours, from CSV)",
    "peak_value_csv": "Peak Excursion/Value (from CSV)",
    "take_profit_price": "Take Profit Price Level",
    "stop_loss_price": "Stop Loss Price Level",
    "trade_result_screenshot": "Trade Result Screenshot URL/Path",
    "event_release_time": "Event Release Time", # Retained as the primary field for event time
    "htf_key_level_focus": "HTF Key Level Focus",
    "market_reaction_on_news": "Market Reaction on News Release",
    "planned_entry_price": "Planned Entry Price",
    "planned_rrr": "Planned Risk:Reward Ratio",
    "planned_sl_price": "Planned Stop Loss Price",
    "planned_tp_price": "Planned Take Profit Price",
    "pre_market_emotions": "Pre-Market Emotions",
    "pre_market_notes": "Pre-Market Notes",
    "trade_plan_screenshot": "Trade Plan Screenshot URL/Path",
    "actionable_steps_notes": "Actionable Steps (Post-Trade)",
    "ror_pct": "Return on Risk (%)",
    "trade_performance_summary_text": "Trade Performance Summary (Text Block)",
    "multiplier_value": "Multiplier Value"
}

# Define expected data types for conceptual columns.
CONCEPTUAL_COLUMN_TYPES: Dict[str, str] = {
    "date": "datetime", "pnl": "numeric", "symbol": "text", "strategy": "text",
    "trade_id": "text", "entry_price": "numeric", "exit_price": "numeric",
    "risk_pct": "numeric", "notes": "text", "signal_confidence": "numeric",
    "duration_minutes": "numeric", "entry_time_str": "text", "trade_month_str": "text",
    "trade_day_str": "text", "trade_plan_str": "text", "bias_str": "text",
    "tag_str": "text", "time_frame_str": "text", "direction_str": "text",
    "trade_size_num": "numeric", "r_r_csv_num": "numeric", "session_str": "text",
    "market_conditions_str": "text", "event_type_str": "text", "events_details_str": "text",
    "market_sentiment_str": "text", "psychological_factors_str": "text",
    "compliance_check_str": "text", "account_str": "text", "initial_balance_num": "numeric",
    "current_balance_num": "numeric", "drawdown_value_csv": "numeric",
    "trade_outcome_csv_str": "text", "exit_type_csv_str": "text",
    "loss_indicator_num": "numeric", "win_indicator_num": "numeric",
    "stop_distance_num": "numeric", "candle_count_num": "numeric",
    "cumulative_equity_csv": "numeric", "absolute_daily_pnl_csv": "numeric",
    "error_exit_type_related_str": "text", "profit_value_csv": "numeric",
    "loss_value_csv": "numeric", "duration_hrs_csv": "numeric", "peak_value_csv": "numeric",
    "take_profit_price": "numeric", "stop_loss_price": "numeric",
    "trade_result_screenshot": "text", "event_release_time": "text", # Could be datetime if parsed
    "htf_key_level_focus": "text", "market_reaction_on_news": "text",
    "planned_entry_price": "numeric", "planned_rrr": "numeric",
    "planned_sl_price": "numeric", "planned_tp_price": "numeric",
    "pre_market_emotions": "text", "pre_market_notes": "text",
    "trade_plan_screenshot": "text", "actionable_steps_notes": "text",
    "ror_pct": "numeric", "trade_performance_summary_text": "text",
    "multiplier_value": "numeric"
}

# Synonyms for conceptual column keys to improve auto-mapping.
CONCEPTUAL_COLUMN_SYNONYMS: Dict[str, List[str]] = {
    "date": ["trade_date", "timestamp", "time", "date"],
    "pnl": ["profit_loss", "net_profit", "profit", "loss", "net_p_l", "realized_pnl", "pnl"],
    "symbol": ["ticker", "instrument", "market", "asset", "symbol_1"],
    "strategy": ["strategy_name", "model_name", "system", "trade_model"],
    "trade_id": ["order_id", "ticket", "deal_id", "transaction_id", "trade_id"],
    "entry_price": ["open_price", "entry"],
    "exit_price": ["close_price", "exit"],
    "risk_pct": ["risk_percentage", "risk_%", "risk_percent"],
    "notes": ["comment", "description", "trade_notes", "lesson_learned"],
    "duration_minutes": ["duration_mins", "trade_duration_min"],
    "entry_time_str": ["entry_time"],
    "trade_month_str": ["month"],
    "trade_day_str": ["day", "day_of_week"],
    "trade_plan_str": ["trade_plan"],
    "bias_str": ["bias", "market_bias_at_trade"],
    "tag_str": ["tag", "tags", "category"],
    "time_frame_str": ["time_frame", "chart_period"],
    "direction_str": ["long_short", "position_type", "side", "type", "direction"],
    "trade_size_num": ["quantity", "volume", "lots", "contracts", "size"],
    "r_r_csv_num": ["r_r", "rrr", "risk_reward"],
    "session_str": ["session", "market_session"],
    "market_conditions_str": ["market_conditions"],
    "event_type_str": ["event_type", "news_event_type"],
    "events_details_str": ["events", "event_details", "news_details", "ref_events"], # Added "ref_events" as synonym here
    "market_sentiment_str": ["market_sentiment", "sentiment"],
    "psychological_factors_str": ["psychological_factors", "emotions", "psychology"],
    "compliance_check_str": ["compliance_check", "plan_compliance"],
    "account_str": ["account_id", "account_number", "login", "account"],
    "initial_balance_num": ["initial_balance", "start_balance"],
    "current_balance_num": ["current_balance", "end_balance"],
    "drawdown_value_csv": ["drawdown", "max_dd_value"],
    "trade_outcome_csv_str": ["trade_result", "outcome", "result"],
    "exit_type_csv_str": ["exit_type", "reason_for_exit"],
    "loss_indicator_num": ["loss_indicator"],
    "win_indicator_num": ["win_indicator"],
    "stop_distance_num": ["stop_distance", "sl_distance"],
    "candle_count_num": ["candle_count", "bars_held"],
    "cumulative_equity_csv": ["cumulative_equity", "equity_curve"],
    "absolute_daily_pnl_csv": ["absolute_daily_pnl", "daily_pnl_abs"],
    "error_exit_type_related_str": ["error", "trade_error_type"],
    "profit_value_csv": ["profit_value", "gross_profit"],
    "loss_value_csv": ["loss_value", "gross_loss"],
    "duration_hrs_csv": ["duration_hrs", "trade_duration_hours"],
    "peak_value_csv": ["peak_value", "max_favorable_excursion", "max_adverse_excursion"],
    "take_profit_price": ["take_profit", "tp_level", "target_price"],
    "stop_loss_price": ["stop_loss", "sl_level"],
    "trade_result_screenshot": ["trade_result_screenshot", "result_image"],
    "event_release_time": ["event_release_time", "news_time", "ref_event_release_time"], # Added "ref_event_release_time" as synonym here
    "htf_key_level_focus": ["htf_key_level_focus", "higher_time_frame_level"],
    "market_reaction_on_news": ["market_reaction_on_news_release", "news_reaction"],
    "planned_entry_price": ["planned_entry"],
    "planned_rrr": ["planned_r_r", "target_rrr"],
    "planned_sl_price": ["planned_sl", "planned_stop_loss"],
    "planned_tp_price": ["planned_tp", "planned_take_profit"],
    "pre_market_emotions": ["pre_market_emotions", "pre_trade_feelings"],
    "pre_market_notes": ["pre_market_notes", "pre_trade_analysis"],
    "trade_plan_screenshot": ["trade_plan_screenshot", "plan_image"],
    "actionable_steps_notes": ["actionable_steps", "post_trade_actions"],
    "ror_pct": ["ror", "return_on_risk_percent"],
    "trade_performance_summary_text": ["trade_performance_summary"],
    "multiplier_value": ["multiplier"]
}

# Critical conceptual columns that MUST be mapped by the user.
CRITICAL_CONCEPTUAL_COLUMNS: List[str] = ["date", "pnl"] # Keep this concise for essential operation

# --- NEW: Conceptual Column Categories for UI Grouping ---
# Defines how columns are grouped in the ColumnMapperUI.
# Uses OrderedDict to maintain the display order of categories.
CONCEPTUAL_COLUMN_CATEGORIES: OrderedDict[str, List[str]] = OrderedDict([
    ("Core Trade Identifiers & Execution", [
        "date", "symbol", "trade_id", "strategy", "direction_str", "entry_price",
        "exit_price", "trade_size_num", "entry_time_str", "session_str",
        "account_str", "time_frame_str"
    ]),
    ("Performance & Profitability", [
        "pnl", "r_r_csv_num", "ror_pct", "profit_value_csv", "loss_value_csv",
        "cumulative_equity_csv", "absolute_daily_pnl_csv", "trade_outcome_csv_str",
        "win_indicator_num", "loss_indicator_num", "multiplier_value"
    ]),
    ("Risk Management & Planning", [
        "risk_pct", "take_profit_price", "stop_loss_price", "stop_distance_num",
        "planned_entry_price", "planned_sl_price", "planned_tp_price", "planned_rrr",
        "drawdown_value_csv", "peak_value_csv", "initial_balance_num", "current_balance_num"
    ]),
    ("Trade Context & Qualitative Analysis", [
        "notes", "trade_plan_str", "bias_str", "tag_str", "market_conditions_str",
        "market_sentiment_str", "psychological_factors_str", "pre_market_emotions",
        "pre_market_notes", "actionable_steps_notes", "trade_performance_summary_text",
        "error_exit_type_related_str", "exit_type_csv_str", "signal_confidence"
    ]),
    ("Duration & Detailed Timing", [
        "duration_minutes", "duration_hrs_csv", "candle_count_num", "trade_month_str",
        "trade_day_str"
    ]),
    ("External Factors & Economic Events", [
        "event_type_str", "events_details_str", "event_release_time", # "ref_events" and "ref_event_release_time" removed from this list
        "market_reaction_on_news", "htf_key_level_focus"
    ]),
    ("Compliance & Visuals/Attachments", [
        "compliance_check_str", "trade_result_screenshot", "trade_plan_screenshot"
    ])
])

# --- UI and Plotting Colors ---
COLORS: Dict[str, str] = {
    "royal_blue": "#4169E1", "green": "#00FF00", "red": "#FF0000",
    "gray": "#808080", "orange": "#FFA500", "purple": "#8A2BE2",
    "dark_background": "#1C2526", "light_background": "#FFFFFF",
    "text_dark": "#E0E0E0", "text_light": "#333333",
    "text_muted_color": "#A0A0A0", "card_background_dark": "#273334",
    "card_border_dark": "#4169E1", "card_background_light": "#F0F2F6",
    "card_border_light": "#4169E1"
}

# --- KPI Groupings for Display ---
KPI_GROUPS_OVERVIEW: Dict[str, List[str]] = {
    "Overall Performance": ["total_pnl", "total_trades", "trading_days", "avg_daily_pnl"],
    "Profitability & Efficiency": ["win_rate", "loss_rate", "profit_factor", "avg_trade_pnl", "avg_win", "avg_loss", "win_loss_ratio"],
    "Risk-Adjusted Returns": ["sharpe_ratio", "sortino_ratio", "calmar_ratio"],
    "Drawdown & Streaks": ["max_drawdown_abs", "max_drawdown_pct", "max_win_streak", "max_loss_streak"],
    "Benchmark Comparison": ["benchmark_total_return", "alpha", "beta", "benchmark_correlation", "tracking_error", "information_ratio"],
    "Distributional Properties": ["pnl_skewness", "pnl_kurtosis"]
}

KPI_GROUPS_RISK_DURATION: Dict[str, List[str]] = {
    "Drawdown Metrics": ["max_drawdown_abs", "max_drawdown_pct"],
    "Value at Risk (VaR & CVaR)": ["var_95_loss", "cvar_95_loss", "var_99_loss", "cvar_99_loss"],
    "Risk-Adjusted Ratios (vs. Self)": ["sharpe_ratio", "sortino_ratio", "calmar_ratio"],
    "Market Risk & Relative Performance": ["beta", "alpha", "benchmark_correlation", "tracking_error", "information_ratio"],
    "Return Distribution Risk": ["pnl_skewness", "pnl_kurtosis"]
}

# --- Default Display Order for KPIs (if not grouped) ---
DEFAULT_KPI_DISPLAY_ORDER: List[str] = [
    "total_pnl", "total_trades", "win_rate", "loss_rate", "profit_factor", "avg_trade_pnl",
    "sharpe_ratio", "sortino_ratio", "calmar_ratio", "max_drawdown_pct", "max_drawdown_abs",
    "benchmark_total_return", "alpha", "beta", "benchmark_correlation", "tracking_error", "information_ratio",
    "avg_win", "avg_loss", "win_loss_ratio",
    "var_95_loss", "cvar_95_loss", "var_99_loss", "cvar_99_loss",
    "pnl_skewness", "pnl_kurtosis",
    "max_win_streak", "max_loss_streak", "avg_daily_pnl", "trading_days", "risk_free_rate_used",
    "expected_annual_return", "annual_volatility"
]

# --- Plotting Themes and Colors ---
PLOTLY_THEME_DARK: str = "plotly_dark"
PLOTLY_THEME_LIGHT: str = "plotly_white"
PLOT_BG_COLOR_DARK: str = COLORS["dark_background"]
PLOT_PAPER_BG_COLOR_DARK: str = COLORS["dark_background"]
PLOT_FONT_COLOR_DARK: str = COLORS["text_dark"]
PLOT_BG_COLOR_LIGHT: str = COLORS["light_background"]
PLOT_PAPER_BG_COLOR_LIGHT: str = COLORS["light_background"]
PLOT_FONT_COLOR_LIGHT: str = COLORS["text_light"]
PLOT_LINE_COLOR: str = COLORS["royal_blue"]
PLOT_BENCHMARK_LINE_COLOR: str = COLORS["purple"]
PLOT_MARKER_PROFIT_COLOR: str = COLORS["green"]
PLOT_MARKER_LOSS_COLOR: str = COLORS["red"]

# --- Analysis Settings ---
BOOTSTRAP_ITERATIONS: int = 1000
CONFIDENCE_LEVEL: float = 0.95
DISTRIBUTIONS_TO_FIT: List[str] = ['norm', 't', 'laplace', 'johnsonsu', 'genextreme']
MARKOV_MAX_LAG: int = 1

# --- Logging Configuration ---
LOG_FILE: str = "logs/trading_dashboard_app.log"
LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] - %(message)s"

# --- EXPECTED_COLUMNS: Internal names used after mapping ---
# This is dynamically generated based on CONCEPTUAL_COLUMNS keys.
# No need to manually list them here if the generation logic is sound.
EXPECTED_COLUMNS: Dict[str, str] = {
    key: key for key in CONCEPTUAL_COLUMNS.keys()
}
