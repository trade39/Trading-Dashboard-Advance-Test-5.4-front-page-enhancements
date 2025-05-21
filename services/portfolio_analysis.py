"""
services/portfolio_analysis.py

Handles portfolio-level analytical calculations, such as aggregating performance
across multiple accounts or strategies, calculating portfolio-specific metrics,
and performing portfolio optimization including efficient frontier, Risk Parity,
robust covariance estimation, per-asset weight constraints, and risk contribution analysis.
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf # For robust covariance estimation

try:
    from config import APP_TITLE, EXPECTED_COLUMNS, RISK_FREE_RATE
except ImportError:
    print(f"CRITICAL IMPORT ERROR in PortfolioAnalysisService module. Some functionalities may fail.")
    APP_TITLE = "TradingDashboard_ErrorState"
    EXPECTED_COLUMNS = {"pnl": "pnl", "date": "date", "strategy": "strategy", "account_str": "account"}
    RISK_FREE_RATE = 0.02


import logging
logger = logging.getLogger(APP_TITLE)

class PortfolioAnalysisService:
    def __init__(self):
        self.logger = logging.getLogger(APP_TITLE)
        self.logger.info("PortfolioAnalysisService initialized.")

    @st.cache_data(ttl=3600, show_spinner="Calculating inter-strategy correlations (portfolio)...")
    def _get_portfolio_inter_strategy_correlation_cached(
        _self, 
        data_values_tuple: Tuple[Tuple[Any, ...], ...], 
        data_columns: List[str],
        strategy_col: str,
        pnl_col: str,
        date_col: str
    ) -> Dict[str, Any]:
        # ... (content as before, no changes needed here for risk contributions) ...
        _self.logger.info(f"Executing _get_portfolio_inter_strategy_correlation_cached. Strategy: '{strategy_col}', PnL: '{pnl_col}', Date: '{date_col}'.")
        try:
            df_reconstructed = pd.DataFrame(list(data_values_tuple), columns=data_columns)
            if df_reconstructed.empty:
                return {"error": "Reconstructed DataFrame is empty for inter-strategy correlation."}

            df_reconstructed[date_col] = pd.to_datetime(df_reconstructed[date_col])
            df_reconstructed[pnl_col] = pd.to_numeric(df_reconstructed[pnl_col])
            df_reconstructed[strategy_col] = df_reconstructed[strategy_col].astype(str)
            
            daily_item_pnl = df_reconstructed.groupby(
                [df_reconstructed[date_col].dt.normalize(), strategy_col]
            )[pnl_col].sum().reset_index()
            
            pivot_table = daily_item_pnl.pivot_table(
                index=date_col, 
                columns=strategy_col, 
                values=pnl_col
            ).fillna(0) 

            if pivot_table.empty or pivot_table.shape[1] < 2:
                _self.logger.warning(f"Pivot table for strategy P&L is empty or has < 2 strategies. Shape: {pivot_table.shape}")
                return {"error": "Not enough strategies with overlapping P&L data to calculate correlation."}

            correlation_matrix = pivot_table.corr()
            _self.logger.info(f"Successfully calculated inter-strategy correlation matrix. Shape: {correlation_matrix.shape}")
            return {"correlation_matrix": correlation_matrix, "strategy_pnl_pivot": pivot_table}
        except Exception as e:
            _self.logger.error(f"Error in _get_portfolio_inter_strategy_correlation_cached: {e}", exc_info=True)
            return {"error": f"An unexpected error occurred during cached correlation: {str(e)}"}

    def get_portfolio_inter_strategy_correlation(
        self, 
        df: pd.DataFrame, 
        strategy_col: str, 
        pnl_col: str, 
        date_col: str
    ) -> Dict[str, Any]:
        # ... (content as before, no changes needed here for risk contributions) ...
        self.logger.info(f"Public get_portfolio_inter_strategy_correlation called. Input DF shape: {df.shape if df is not None else 'None'}")
        if df is None or df.empty:
            return {"error": "Input DataFrame is empty for inter-strategy correlation."}
        
        cols_needed = [date_col, strategy_col, pnl_col]
        if not all(c in df.columns for c in cols_needed):
            missing = [c for c in cols_needed if c not in df.columns]
            return {"error": f"Missing one or more required columns for correlation: {', '.join(missing)}"}
            
        df_for_hash = df[cols_needed].copy()
        try:
            df_for_hash[date_col] = pd.to_datetime(df_for_hash[date_col], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S.%f')
            df_for_hash.dropna(subset=[date_col], inplace=True) 
            df_for_hash[pnl_col] = pd.to_numeric(df_for_hash[pnl_col], errors='coerce')
            df_for_hash.dropna(subset=[pnl_col], inplace=True)
            df_for_hash[strategy_col] = df_for_hash[strategy_col].astype(str)
            if df_for_hash.empty:
                return {"error": "No valid data remains after cleaning for correlation calculation."}
            data_values_tuple = tuple(map(tuple, df_for_hash.values))
            data_columns_list = df_for_hash.columns.tolist()
        except Exception as e_prep:
            self.logger.error(f"Error preparing DataFrame for caching in correlation: {e_prep}", exc_info=True)
            return {"error": f"Data preparation error for caching: {e_prep}"}

        return self._get_portfolio_inter_strategy_correlation_cached(
            data_values_tuple=data_values_tuple,
            data_columns=data_columns_list,
            strategy_col=strategy_col,
            pnl_col=pnl_col,
            date_col=date_col
        )


    @st.cache_data(ttl=3600, show_spinner="Calculating inter-account correlations (portfolio)...")
    def _get_portfolio_inter_account_correlation_cached(
        _self, 
        data_values_tuple: Tuple[Tuple[Any, ...], ...], 
        data_columns: List[str],
        account_col: str, 
        pnl_col: str,
        date_col: str
    ) -> Dict[str, Any]:
        # ... (content as before, no changes needed here for risk contributions) ...
        _self.logger.info(f"Executing _get_portfolio_inter_account_correlation_cached. Account: '{account_col}', PnL: '{pnl_col}', Date: '{date_col}'.")
        try:
            df_reconstructed = pd.DataFrame(list(data_values_tuple), columns=data_columns)
            if df_reconstructed.empty:
                return {"error": "Reconstructed DataFrame is empty for inter-account correlation."}

            df_reconstructed[date_col] = pd.to_datetime(df_reconstructed[date_col])
            df_reconstructed[pnl_col] = pd.to_numeric(df_reconstructed[pnl_col])
            df_reconstructed[account_col] = df_reconstructed[account_col].astype(str)
            
            daily_item_pnl = df_reconstructed.groupby(
                [df_reconstructed[date_col].dt.normalize(), account_col] 
            )[pnl_col].sum().reset_index()
            
            pivot_table = daily_item_pnl.pivot_table(
                index=date_col, 
                columns=account_col, 
                values=pnl_col
            ).fillna(0)

            if pivot_table.empty or pivot_table.shape[1] < 2:
                _self.logger.warning(f"Pivot table for account P&L is empty or has < 2 accounts. Shape: {pivot_table.shape}")
                return {"error": "Not enough accounts with overlapping P&L data to calculate correlation."}

            correlation_matrix = pivot_table.corr()
            _self.logger.info(f"Successfully calculated inter-account correlation matrix. Shape: {correlation_matrix.shape}")
            return {"correlation_matrix": correlation_matrix, "account_pnl_pivot": pivot_table}
        except Exception as e:
            _self.logger.error(f"Error in _get_portfolio_inter_account_correlation_cached: {e}", exc_info=True)
            return {"error": f"An unexpected error occurred during cached account correlation: {str(e)}"}

    def get_portfolio_inter_account_correlation(
        self, 
        df: pd.DataFrame, 
        account_col: str, 
        pnl_col: str, 
        date_col: str
    ) -> Dict[str, Any]:
        # ... (content as before, no changes needed here for risk contributions) ...
        self.logger.info(f"Public get_portfolio_inter_account_correlation called. Input DF shape: {df.shape if df is not None else 'None'}")
        if df is None or df.empty:
            return {"error": "Input DataFrame is empty for inter-account correlation."}
        
        cols_needed = [date_col, account_col, pnl_col] 
        if not all(c in df.columns for c in cols_needed):
            missing = [c for c in cols_needed if c not in df.columns]
            return {"error": f"Missing one or more required columns for account correlation: {', '.join(missing)}"}
            
        df_for_hash = df[cols_needed].copy()
        try:
            df_for_hash[date_col] = pd.to_datetime(df_for_hash[date_col], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S.%f')
            df_for_hash.dropna(subset=[date_col], inplace=True)
            df_for_hash[pnl_col] = pd.to_numeric(df_for_hash[pnl_col], errors='coerce')
            df_for_hash.dropna(subset=[pnl_col], inplace=True)
            df_for_hash[account_col] = df_for_hash[account_col].astype(str) 
            if df_for_hash.empty:
                return {"error": "No valid data remains after cleaning for account correlation calculation."}
            data_values_tuple = tuple(map(tuple, df_for_hash.values))
            data_columns_list = df_for_hash.columns.tolist()
        except Exception as e_prep:
            self.logger.error(f"Error preparing DataFrame for caching in account correlation: {e_prep}", exc_info=True)
            return {"error": f"Data preparation error for caching (account correlation): {e_prep}"}

        return self._get_portfolio_inter_account_correlation_cached(
            data_values_tuple=data_values_tuple,
            data_columns=data_columns_list,
            account_col=account_col, 
            pnl_col=pnl_col,
            date_col=date_col
        )


    def _calculate_portfolio_performance_metrics(self, weights: np.ndarray, mean_returns: pd.Series, cov_matrix: pd.DataFrame, risk_free_rate: float, trading_days: int = 252) -> Dict[str, float]:
        # ... (content as before) ...
        if not isinstance(weights, np.ndarray): weights = np.array(weights)
        
        portfolio_return_annual = np.sum(mean_returns.values * weights) * trading_days
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix.values, weights)) * trading_days
        portfolio_volatility_annual = np.sqrt(portfolio_variance)

        if portfolio_volatility_annual < 1e-9: 
            sharpe_ratio = np.inf if (portfolio_return_annual - risk_free_rate) > 0 else 0.0
        else:
            sharpe_ratio = (portfolio_return_annual - risk_free_rate) / portfolio_volatility_annual
        
        return {
            "expected_annual_return": float(portfolio_return_annual),
            "annual_volatility": float(portfolio_volatility_annual),
            "sharpe_ratio": float(sharpe_ratio)
        }

    def _calculate_risk_contributions(self, weights: np.ndarray, cov_matrix: pd.DataFrame, asset_names: List[str]) -> Dict[str, float]:
        """Helper to calculate percentage risk contributions of each asset."""
        if not isinstance(weights, np.ndarray): weights = np.array(weights)
        
        portfolio_variance_daily = np.dot(weights.T, np.dot(cov_matrix.values, weights)) # Daily variance
        
        if portfolio_variance_daily < 1e-12: # Avoid division by zero if variance is ~0
            # If variance is zero, risk contributions are ill-defined or can be considered equal if weights are non-zero
            # For simplicity, return zero contributions or distribute based on weights if variance is truly zero
            if np.any(weights > 1e-6): # If there are any non-zero weights
                 return {name: (w / np.sum(weights)) * 100.0 if np.sum(weights) > 1e-6 else 0.0 for name, w in zip(asset_names, weights)}
            return {name: 0.0 for name in asset_names}

        # Marginal Contribution to Risk (MCTR_i) = (Cov * w)_i / sqrt(w' * Cov * w)
        # Risk Contribution (RC_i) = w_i * MCTR_i
        # More directly: RC_i (absolute) = w_i * (Cov * w)_i (this is component to portfolio variance)
        
        # Component standard deviation (CSD) is not additive.
        # Percentage contribution to portfolio variance is additive.
        # RC_i_pct_variance = w_i * (Cov * w)_i / portfolio_variance
        
        marginal_risk_contributions_to_variance = weights * (np.dot(cov_matrix.values, weights))
        
        # Ensure no negative contributions if weights are non-negative (shouldn't happen with positive semi-definite cov)
        marginal_risk_contributions_to_variance = np.maximum(marginal_risk_contributions_to_variance, 0)

        total_contribution_to_variance = np.sum(marginal_risk_contributions_to_variance)
        
        if total_contribution_to_variance < 1e-12: # If sum of contributions is near zero
            # Fallback to weight-based if total contribution is negligible
            if np.any(weights > 1e-6):
                return {name: (w / np.sum(weights)) * 100.0 if np.sum(weights) > 1e-6 else 0.0 for name, w in zip(asset_names, weights)}
            return {name: 0.0 for name in asset_names}

        risk_contributions_pct = (marginal_risk_contributions_to_variance / total_contribution_to_variance) * 100.0
        
        return dict(zip(asset_names, risk_contributions_pct))


    @st.cache_data(ttl=3600, show_spinner="Optimizing portfolio...")
    def perform_portfolio_optimization(
        _self, 
        daily_returns_df_tuple: Tuple[Tuple[Any, ...], ...], 
        asset_names: List[str], 
        objective: str = "maximize_sharpe_ratio",
        risk_free_rate: float = RISK_FREE_RATE,
        target_return_level: Optional[float] = None,
        trading_days: int = 252,
        num_frontier_points: int = 20,
        use_ledoit_wolf: bool = True,
        asset_bounds: Optional[List[Tuple[float, float]]] = None
    ) -> Dict[str, Any]:
        _self.logger.info(f"Executing perform_portfolio_optimization (cached). Objective: {objective}, Assets: {asset_names}, Target Return: {target_return_level}, Trading Days: {trading_days}, Ledoit-Wolf: {use_ledoit_wolf}, Asset Bounds: {asset_bounds}")
        
        try:
            daily_returns_df = pd.DataFrame(list(daily_returns_df_tuple), columns=asset_names)
            if daily_returns_df.empty or daily_returns_df.shape[1] < 1:
                return {"error": "Insufficient data or assets for optimization after reconstruction."}
            if objective != "risk_parity" and daily_returns_df.shape[1] < 2:
                 return {"error": "Mean-Variance Optimization requires at least two assets."}

            mean_daily_returns = daily_returns_df.mean()
            num_assets = len(asset_names)

            if use_ledoit_wolf:
                try:
                    lw = LedoitWolf()
                    if daily_returns_df.shape[0] > daily_returns_df.shape[1] and daily_returns_df.shape[1] > 1:
                        lw.fit(daily_returns_df.values)
                        cov_matrix_daily = pd.DataFrame(lw.covariance_, index=asset_names, columns=asset_names)
                        _self.logger.info("Using Ledoit-Wolf shrunk covariance matrix.")
                    else:
                        _self.logger.warning("Not enough samples or features for Ledoit-Wolf. Using sample covariance + regularization.")
                        cov_matrix_daily_raw = daily_returns_df.cov()
                        cov_matrix_daily = cov_matrix_daily_raw + np.eye(num_assets) * 1e-8
                except Exception as e_lw:
                    _self.logger.error(f"Error using LedoitWolf, falling back to sample covariance: {e_lw}")
                    cov_matrix_daily_raw = daily_returns_df.cov()
                    cov_matrix_daily = cov_matrix_daily_raw + np.eye(num_assets) * 1e-8
            else:
                cov_matrix_daily_raw = daily_returns_df.cov()
                cov_matrix_daily = cov_matrix_daily_raw + np.eye(num_assets) * 1e-8
            
            if mean_daily_returns.isnull().any() or cov_matrix_daily.isnull().any().any():
                return {"error": "NaN values found in mean returns or covariance matrix."}

            args_mvo = (mean_daily_returns.values, cov_matrix_daily.values, risk_free_rate, trading_days)
            args_rp = (cov_matrix_daily.values,) 

            current_bounds = asset_bounds if asset_bounds and len(asset_bounds) == num_assets else tuple((0.0, 1.0) for _ in range(num_assets))
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            initial_weights = np.array(num_assets * [1. / num_assets])
            for i in range(num_assets):
                if initial_weights[i] < current_bounds[i][0]: initial_weights[i] = current_bounds[i][0]
                if initial_weights[i] > current_bounds[i][1]: initial_weights[i] = current_bounds[i][1]
            if abs(np.sum(initial_weights) - 1.0) > 1e-6 and np.sum(initial_weights) > 1e-9 : 
                initial_weights = initial_weights / np.sum(initial_weights)
            elif np.sum(initial_weights) < 1e-9 and num_assets > 0: # Handle case where all initial weights are zeroed by bounds
                _self.logger.warning("Initial weights sum to zero due to tight bounds. Resetting to a feasible point if possible or default.")
                # Try to find a feasible point or use a simple average respecting bounds
                # This part can be complex; for now, we rely on optimizer to handle it or fail.
                # A simple approach: set to lower bounds and normalize if sum < 1, or error if sum of lower bounds > 1
                sum_lower_bounds = sum(b[0] for b in current_bounds)
                if sum_lower_bounds > 1.0 + 1e-6: # Check if sum of min weights already violates sum-to-one
                    return {"error": "Sum of minimum weight constraints exceeds 100%. Cannot initialize weights."}
                initial_weights = np.array([b[0] for b in current_bounds])
                if np.sum(initial_weights) < 1.0 - 1e-6 and np.sum(initial_weights) > 1e-9: # If there's room to add more weight
                     # Distribute remaining weight, e.g., equally or proportionally, respecting max bounds
                     # This is a simplified heuristic
                     remaining_weight = 1.0 - np.sum(initial_weights)
                     addable_weights = np.array([b[1] - b[0] for b in current_bounds])
                     if np.sum(addable_weights) > 1e-9:
                         initial_weights += remaining_weight * (addable_weights / np.sum(addable_weights))
                     initial_weights = initial_weights / np.sum(initial_weights) # Final normalization
                elif np.sum(initial_weights) < 1e-9 : # Still zero
                    initial_weights = np.array(num_assets * [1. / num_assets]) # Fallback to simple equal weight if all else fails

            def portfolio_volatility_fn(weights, mean_returns_arr, cov_matrix_arr, rf, td):
                port_var = np.dot(weights.T, np.dot(cov_matrix_arr, weights)) * td
                return np.sqrt(port_var)

            def neg_sharpe_ratio_fn(weights, mean_returns_arr, cov_matrix_arr, rf, td):
                port_ret = np.sum(mean_returns_arr * weights) * td
                port_var = np.dot(weights.T, np.dot(cov_matrix_arr, weights)) * td
                port_vol = np.sqrt(port_var)
                if port_vol < 1e-9: return -np.inf 
                return - (port_ret - rf) / port_vol
            
            def risk_parity_objective_fn(weights, cov_matrix_arr):
                weights = np.maximum(weights, 1e-8)
                portfolio_variance = np.dot(weights.T, np.dot(cov_matrix_arr, weights))
                if portfolio_variance < 1e-12: return 1e12 
                marginal_risk_contributions = weights * (np.dot(cov_matrix_arr, weights))
                risk_contributions_pct = marginal_risk_contributions / portfolio_variance
                if len(risk_contributions_pct) > 1:
                    return np.sum((risk_contributions_pct - np.mean(risk_contributions_pct))**2)
                return 0.0 

            optimized_result_single_point = None
            if objective == "maximize_sharpe_ratio":
                optimized_result_single_point = minimize(neg_sharpe_ratio_fn, initial_weights, args=args_mvo, method='SLSQP', bounds=current_bounds, constraints=constraints, options={'ftol': 1e-9, 'disp': False})
            elif objective == "minimize_volatility":
                # ... (feasibility check as before) ...
                mean_annual_returns_individual = mean_daily_returns * trading_days
                min_achievable_ret_individual = mean_annual_returns_individual.min()
                max_achievable_ret_individual = mean_annual_returns_individual.max()
                
                if target_return_level is not None:
                    if not (min_achievable_ret_individual <= target_return_level <= max_achievable_ret_individual):
                        error_msg = (f"Target annualized return {target_return_level:.2%} is not achievable. "
                                     f"Feasible range for these assets is approx. [{min_achievable_ret_individual:.2%}, {max_achievable_ret_individual:.2%}].")
                        _self.logger.warning(error_msg)
                        return {"error": error_msg} 

                current_constraints_min_vol = [constraints] 
                if target_return_level is not None: 
                    def return_constraint_fn(weights, mean_returns_arr, td, target_ret):
                        return np.sum(mean_returns_arr * weights) * td - target_ret
                    current_constraints_min_vol.append({'type': 'eq', 'fun': lambda w: return_constraint_fn(w, mean_daily_returns.values, trading_days, target_return_level)})
                optimized_result_single_point = minimize(portfolio_volatility_fn, initial_weights, args=args_mvo, method='SLSQP', bounds=current_bounds, constraints=tuple(current_constraints_min_vol), options={'ftol': 1e-9, 'disp': False})
            elif objective == "risk_parity":
                optimized_result_single_point = minimize(risk_parity_objective_fn, initial_weights, args=args_rp, method='SLSQP', bounds=current_bounds, constraints=constraints, options={'ftol': 1e-9, 'disp': False})
            else:
                return {"error": f"Unsupported optimization objective: {objective}"}

            if not optimized_result_single_point or not optimized_result_single_point.success:
                 detailed_error = f"Optimization failed for objective '{objective}': {optimized_result_single_point.message if optimized_result_single_point else 'Optimizer did not run'}."
                 if optimized_result_single_point and "Positive directional derivative" in optimized_result_single_point.message and objective == "minimize_volatility" and target_return_level is not None:
                     detailed_error += " This can occur if the target return is at the edge of or outside the feasible return range for the selected assets, or if custom bounds are too restrictive."
                 return {"error": detailed_error, "details": optimized_result_single_point}

            optimal_weights_single_point = optimized_result_single_point.x
            optimal_weights_single_point[optimal_weights_single_point < 1e-4] = 0 
            if np.sum(optimal_weights_single_point) > 1e-9 : 
                optimal_weights_single_point = optimal_weights_single_point / np.sum(optimal_weights_single_point)
            else:
                optimal_weights_single_point = initial_weights 
            performance_single_point = _self._calculate_portfolio_performance_metrics(optimal_weights_single_point, mean_daily_returns, cov_matrix_daily, risk_free_rate, trading_days)
            risk_contributions_single_point = _self._calculate_risk_contributions(optimal_weights_single_point, cov_matrix_daily, asset_names) # Calculate for the single optimal point
            
            efficient_frontier_data = None
            if objective in ["maximize_sharpe_ratio", "minimize_volatility"] and num_frontier_points > 0:
                # ... (Efficient frontier calculation as before) ...
                frontier_returns = []
                frontier_volatilities = []
                frontier_weights_list = []

                min_vol_for_range_args = (mean_daily_returns.values, cov_matrix_daily.values, risk_free_rate, trading_days)
                min_vol_res_for_range = minimize(portfolio_volatility_fn, initial_weights, args=min_vol_for_range_args, method='SLSQP', bounds=current_bounds, constraints=constraints, options={'ftol': 1e-9, 'disp': False})
                
                min_frontier_ret = 0.0
                if min_vol_res_for_range.success:
                    min_ret_weights = min_vol_res_for_range.x
                    min_frontier_ret = np.sum(mean_daily_returns.values * min_ret_weights) * trading_days
                else: 
                    min_frontier_ret = (mean_daily_returns * trading_days).min() 
                
                max_frontier_ret = (mean_daily_returns * trading_days).max()
                
                if min_frontier_ret < max_frontier_ret: 
                    target_returns_for_frontier = np.linspace(min_frontier_ret, max_frontier_ret, num_frontier_points)
                    for target_ret_ef in target_returns_for_frontier:
                        def ef_return_constraint_fn(weights, mean_returns_arr, td, current_target_ret):
                            return np.sum(mean_returns_arr * weights) * td - current_target_ret
                        ef_constraints = [
                            constraints, 
                            {'type': 'eq', 'fun': lambda w: ef_return_constraint_fn(w, mean_daily_returns.values, trading_days, target_ret_ef)}
                        ]
                        ef_result = minimize(portfolio_volatility_fn, initial_weights, args=args_mvo, method='SLSQP', bounds=current_bounds, constraints=tuple(ef_constraints), options={'ftol': 1e-9, 'disp': False})
                        
                        if ef_result.success:
                            current_weights = ef_result.x
                            current_weights[current_weights < 1e-4] = 0.0
                            if np.sum(current_weights) > 1e-9: current_weights = current_weights / np.sum(current_weights)
                            else: current_weights = initial_weights / np.sum(initial_weights)
                            frontier_weights_list.append(current_weights)
                            perf_metrics = _self._calculate_portfolio_performance_metrics(current_weights, mean_daily_returns, cov_matrix_daily, risk_free_rate, trading_days)
                            frontier_returns.append(perf_metrics["expected_annual_return"])
                            frontier_volatilities.append(perf_metrics["annual_volatility"])
                
                if frontier_returns and frontier_volatilities:
                    efficient_frontier_data = pd.DataFrame({
                        'volatility': frontier_volatilities,
                        'return': frontier_returns,
                        'weights': frontier_weights_list 
                    }).sort_values(by='volatility').reset_index(drop=True)
                    if not efficient_frontier_data.empty: 
                        efficient_frontier_data = efficient_frontier_data.loc[efficient_frontier_data.groupby('volatility')['return'].idxmax()]


            return {
                "optimal_weights": dict(zip(asset_names, optimal_weights_single_point)),
                "performance": performance_single_point,
                "risk_contributions": risk_contributions_single_point, # Added risk contributions
                "optimizer_result_summary": {"message": optimized_result_single_point.message, "status": optimized_result_single_point.status, "success": optimized_result_single_point.success} if optimized_result_single_point else None,
                "efficient_frontier": efficient_frontier_data.to_dict('list') if efficient_frontier_data is not None and not efficient_frontier_data.empty else None
            }

        except Exception as e:
            _self.logger.error(f"Error during portfolio optimization: {e}", exc_info=True)
            return {"error": f"An unexpected error occurred during optimization: {str(e)}"}

    def prepare_and_run_optimization(
        self,
        daily_returns_df: pd.DataFrame,
        objective: str = "maximize_sharpe_ratio",
        risk_free_rate: float = RISK_FREE_RATE,
        target_return_level: Optional[float] = None,
        trading_days: int = 252,
        num_frontier_points: int = 20,
        use_ledoit_wolf: bool = True, 
        asset_bounds: Optional[List[Tuple[float, float]]] = None
    ) -> Dict[str, Any]:
        self.logger.info(f"Public prepare_and_run_optimization called. Input DF shape: {daily_returns_df.shape if daily_returns_df is not None else 'None'}, Objective: {objective}, Ledoit-Wolf: {use_ledoit_wolf}, Asset Bounds: {asset_bounds}")
        if daily_returns_df is None or daily_returns_df.empty:
            return {"error": "Input DataFrame for optimization is empty."}
        
        if objective != "risk_parity" and daily_returns_df.shape[1] < 2:
             return {"error": "Mean-Variance Optimization (Maximize Sharpe, Minimize Volatility) requires at least two assets/strategies."}
        if objective == "risk_parity" and daily_returns_df.shape[1] < 1: 
             return {"error": "Risk Parity requires at least one asset."}

        try:
            for col in daily_returns_df.columns:
                daily_returns_df[col] = pd.to_numeric(daily_returns_df[col], errors='coerce')
            
            daily_returns_df_cleaned = daily_returns_df.dropna()
            
            min_points_needed = 20 
            if objective == "risk_parity" and daily_returns_df.shape[1] == 1: 
                min_points_needed = 2 
            
            actual_points = daily_returns_df_cleaned.shape[0] if not daily_returns_df_cleaned.empty else 0
            if actual_points < min_points_needed : 
                return {"error": f"Not enough valid (non-NaN) historical return data points ({actual_points}) for optimization after cleaning. Need at least {min_points_needed}."}

            asset_names_list = daily_returns_df_cleaned.columns.tolist()
            data_values_tuple_for_cache = tuple(map(tuple, daily_returns_df_cleaned.values))

        except Exception as e_prep:
            self.logger.error(f"Error preparing DataFrame for optimization caching: {e_prep}", exc_info=True)
            return {"error": f"Data preparation error for optimization: {e_prep}"}

        return self.perform_portfolio_optimization(
            daily_returns_df_tuple=data_values_tuple_for_cache,
            asset_names=asset_names_list,
            objective=objective,
            risk_free_rate=risk_free_rate,
            target_return_level=target_return_level,
            trading_days=trading_days,
            num_frontier_points=num_frontier_points,
            use_ledoit_wolf=use_ledoit_wolf,
            asset_bounds=asset_bounds
        )
