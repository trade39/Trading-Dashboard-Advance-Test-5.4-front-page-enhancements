"""
services/__init__.py

This file makes the 'services' directory a Python package.
It allows modules within this directory to be imported using dot notation
(e.g., from services.data_service import DataService).
"""
from .data_service import DataService, get_benchmark_data_static
from .analysis_service import AnalysisService
from .portfolio_analysis import PortfolioAnalysisService
from .statistical_analysis_service import StatisticalAnalysisService
from .stochastic_model_service import StochasticModelService
from .ai_model_service import AIModelService # MODIFIED: Added import

__all__ = [
    "DataService",
    "get_benchmark_data_static",
    "AnalysisService",
    "PortfolioAnalysisService",
    "StatisticalAnalysisService",
    "StochasticModelService",
    "AIModelService", # MODIFIED: Added to __all__
]
