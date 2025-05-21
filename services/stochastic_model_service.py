# services/stochastic_model_service.py
"""
This service handles analyses and simulations related to stochastic models,
acting as a wrapper around functions in the stochastic_models.py module.
"""
import pandas as pd
from typing import Dict, Any, Optional

try:
    from config import APP_TITLE
    from stochastic_models import (
        simulate_gbm, fit_ornstein_uhlenbeck,
        simulate_merton_jump_diffusion, fit_markov_chain_trade_sequence
    )
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR in StochasticModelService module: {e}. Some functionalities may fail.")
    APP_TITLE = "TradingDashboard_ErrorState"
    # Fallback dummy functions
    def simulate_gbm(*args, **kwargs): return {"error": "simulate_gbm not loaded"}
    def fit_ornstein_uhlenbeck(*args, **kwargs): return {"error": "fit_ornstein_uhlenbeck not loaded"}
    def simulate_merton_jump_diffusion(*args, **kwargs): return {"error": "simulate_merton_jump_diffusion not loaded"}
    def fit_markov_chain_trade_sequence(*args, **kwargs): return {"error": "fit_markov_chain_trade_sequence not loaded"}

import logging
logger = logging.getLogger(APP_TITLE)

# Define Minimum Data Points Thresholds specific to this service's methods
MIN_DATA_FOR_GBM_SIM_PARAMS = 1 # Not data dependent for series, but parameter check
MIN_DATA_FOR_OU_FIT = 20
MIN_DATA_FOR_MARKOV_CHAIN = 15
MIN_DATA_FOR_MERTON_SIM_PARAMS = 1 # Parameter check

class StochasticModelService:
    def __init__(self):
        self.logger = logging.getLogger(APP_TITLE)
        self.logger.info("StochasticModelService initialized.")

    def run_gbm_simulation(self, s0: float, mu: float, sigma: float, dt: float, n_steps: int, n_sims: int = 1) -> Dict[str, Any]:
        """
        Runs Geometric Brownian Motion simulation.
        """
        if s0 <= 0 or sigma < 0 or dt <= 0 or n_steps < MIN_DATA_FOR_GBM_SIM_PARAMS or n_sims <=0 :
            self.logger.error(f"run_gbm_simulation: Invalid parameters for GBM simulation (s0={s0}, sigma={sigma}, dt={dt}, n_steps={n_steps}, n_sims={n_sims}).")
            return {"error": "Invalid parameters for GBM simulation (e.g., s0, sigma, dt, steps, sims must be positive)."}
        try:
            paths = simulate_gbm(s0, mu, sigma, dt, n_steps, n_sims)
            return {"paths": paths} if (paths is not None and paths.size > 0) else {"error": "GBM simulation returned empty or invalid paths."}
        except Exception as e:
            self.logger.error(f"Error in GBM sim: {e}", exc_info=True)
            return {"error": str(e)}

    def estimate_ornstein_uhlenbeck(self, series: pd.Series) -> Dict[str, Any]:
        """
        Estimates parameters for an Ornstein-Uhlenbeck process.
        """
        if series is None or series.dropna().empty:
            self.logger.warning("estimate_ornstein_uhlenbeck: Input series is empty.")
            return {"error": "Input series is empty for OU fitting."}
        if len(series.dropna()) < MIN_DATA_FOR_OU_FIT:
            self.logger.warning(f"estimate_ornstein_uhlenbeck: Series too short (need >= {MIN_DATA_FOR_OU_FIT}). Found {len(series.dropna())}.")
            return {"error": f"Series too short (need at least {MIN_DATA_FOR_OU_FIT} points) for OU fitting."}
        try:
            result = fit_ornstein_uhlenbeck(series.dropna())
            return result if result is not None else {"error": "OU fitting returned None."}
        except Exception as e:
            self.logger.error(f"Error in OU fit: {e}", exc_info=True)
            return {"error": str(e)}

    def analyze_markov_chain_trades(self, pnl_series: pd.Series, n_states: int = 2) -> Dict[str, Any]:
        """
        Fits a Markov chain to a sequence of trade outcomes.
        """
        if pnl_series is None or pnl_series.dropna().empty:
            self.logger.warning("analyze_markov_chain_trades: PnL series is empty.")
            return {"error": "PnL series is empty for Markov chain analysis."}
        if len(pnl_series.dropna()) < MIN_DATA_FOR_MARKOV_CHAIN:
            self.logger.warning(f"analyze_markov_chain_trades: PnL series too short (need >= {MIN_DATA_FOR_MARKOV_CHAIN}). Found {len(pnl_series.dropna())}.")
            return {"error": f"PnL series too short (need at least {MIN_DATA_FOR_MARKOV_CHAIN} trades) for Markov chain analysis."}
        try:
            result = fit_markov_chain_trade_sequence(pnl_series.dropna(), n_states=n_states)
            return result if result is not None else {"error": "Markov chain analysis returned None."}
        except Exception as e:
            self.logger.error(f"Error in Markov chain: {e}", exc_info=True)
            return {"error": str(e)}

    def run_merton_jump_diffusion_simulation(
        self, s0: float, mu: float, sigma: float,
        lambda_jump: float, mu_jump: float, sigma_jump: float,
        dt: float, n_steps: int, n_sims: int = 1
    ) -> Dict[str, Any]:
        """
        Runs Merton Jump-Diffusion model simulation.
        """
        if s0 <=0 or sigma < 0 or lambda_jump < 0 or sigma_jump < 0 or dt <= 0 or n_steps < MIN_DATA_FOR_MERTON_SIM_PARAMS or n_sims <=0:
            self.logger.error(f"run_merton_jump_diffusion_simulation: Invalid parameters.")
            return {"error": "Invalid parameters for Merton Jump-Diffusion simulation."}
        try:
            paths = simulate_merton_jump_diffusion(s0, mu, sigma, lambda_jump, mu_jump, sigma_jump, dt, n_steps, n_sims)
            return {"paths": paths} if (paths is not None and paths.size > 0) else {"error": "Merton simulation returned empty paths."}
        except Exception as e:
            self.logger.error(f"Error in Merton sim: {e}", exc_info=True)
            return {"error": str(e)}

