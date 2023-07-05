import pandas as pd
import numpy as np
from piesta.data.asset import Universe
from abc import ABC, abstractmethod
from typing import List, Callable, Dict, Optional, Tuple
from scipy.optimize import minimize, LinearConstraint

class Optimizer(ABC):
    @abstractmethod
    def optimize(self, data: pd.DataFrame) -> pd.Series:
        pass        

class MeanVarianceOptimizer(Optimizer):
    def optimize(self, data: pd.DataFrame, constraints) -> pd.Series:
        cov_matrix = data.cov()
        mean_returns = data.mean()
        num_assets = len(mean_returns)
        initial_weights = np.random.random(num_assets)
        initial_weights /= np.sum(initial_weights)

        def neg_sharpe_ratio(weights: np.ndarray) -> float:
            portfolio_return = np.dot(mean_returns, weights)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = portfolio_return / portfolio_volatility
            return -sharpe_ratio

        bounds = [(0, 1) for _ in range(num_assets)]
        optimized = minimize(neg_sharpe_ratio, initial_weights, bounds = bounds, constraints = constraints)
        return dict(zip(data.columns, optimized.x))

class NaiveOptimizer(Optimizer):
    def optimize(self, data: pd.DataFrame) -> pd.Series:
        num_assets = len(data.columns)
        weights = np.ones(num_assets) / num_assets
        return pd.Series(weights, index=data.columns)


class BlackLittermanOptimizer(Optimizer):
    def __init__(self, tau: float = 0.05, P: np.ndarray = None, Q: np.ndarray = None, omega: np.ndarray = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau = tau
        self.P = P
        self.Q = Q
        self.omega = omega

    def optimize(self, data: pd.DataFrame) -> Dict[str, float]:
        expected_returns, cov_matrix = self._calculate_statistics(data)
        posterior_expected_returns, posterior_cov_matrix = self._black_litterman(expected_returns, cov_matrix)
        weights = self._optimize_portfolio(posterior_expected_returns, posterior_cov_matrix)
        return dict(zip(data.columns, weights))

    def _black_litterman(self, prior_returns: pd.Series, cov_matrix: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        if self.P is None or self.Q is None or self.omega is None:
            raise ValueError("P, Q, and Omega matrices are required for the Black-Litterman model")

        # Compute the Black-Litterman posterior expected returns and covariance matrix
        pi = self.tau * cov_matrix.dot(prior_returns)
        omega_inv = np.linalg.inv(self.omega)
        cov_matrix_inv = np.linalg.inv(cov_matrix)

        # Compute the posterior expected returns
        posterior_expected_returns = np.linalg.inv(cov_matrix_inv + self.P.T.dot(omega_inv).dot(self.P)).dot(cov_matrix_inv.dot(pi) + self.P.T.dot(omega_inv).dot(self.Q))

        # Compute the posterior covariance matrix
        posterior_cov_matrix = cov_matrix + np.linalg.inv(cov_matrix_inv + self.P.T.dot(omega_inv).dot(self.P))

        return pd.Series(posterior_expected_returns, index=prior_returns.index), posterior_cov_matrix


class GoalBasedOptimizer(Optimizer):
    def __init__(self, goals: List[Dict], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.goals = goals

    def optimize(self, data: pd.DataFrame) -> Dict[str, float]:
        expected_returns, cov_matrix = self._calculate_statistics(data)
        weights = self._goal_based_investing(expected_returns, cov_matrix)
        return dict(zip(data.columns, weights))

    def _goal_based_investing(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame) -> np.ndarray:
        # Sort goals by priority
        sorted_goals = sorted(self.goals, key=lambda goal: goal['priority'])

        # Initialize portfolio weights
        weights = np.zeros(len(expected_returns))

        for goal in sorted_goals:
            goal_horizon = goal['horizon']
            goal_amount = goal['amount']

            # Calculate the weights for the current goal
            goal_weights = self._optimize_portfolio(expected_returns, cov_matrix, goal_horizon)

            # Allocate assets to meet the goal
            weights += goal_weights * goal_amount

        # Normalize weights
        weights /= np.sum(weights)

        return weights


class RiskParityOptimizer(Optimizer):
    def __init__(self, risk_aversion: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.risk_aversion = risk_aversion

    def optimize(self, data: pd.DataFrame) -> Dict[str, float]:
        expected_returns, cov_matrix = self._calculate_statistics(data)
        weights = self._risk_parity_allocation(cov_matrix)
        return dict(zip(data.columns, weights))

    def _risk_parity_allocation(self, cov_matrix: pd.DataFrame) -> np.ndarray:
        # Initialize portfolio weights
        num_assets = len(cov_matrix)
        init_weights = np.ones(num_assets) / num_assets
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]

        # Define the objective function for risk parity
        def risk_parity_objective_function(weights, cov_matrix, risk_aversion):
            portfolio_volatility = np.sqrt(weights.T.dot(cov_matrix).dot(weights))
            risk_contributions = self._calculate_risk_contributions(weights, cov_matrix)
            risk_diffs = risk_contributions - risk_aversion * portfolio_volatility / num_assets
            return np.sum(np.square(risk_diffs))

        # Optimize the portfolio weights
        optimized_result = minimize(risk_parity_objective_function, init_weights, args=(cov_matrix, self.risk_aversion),
                                    method='SLSQP', constraints=constraints, bounds=[(0, 1) for _ in range(num_assets)])

        return optimized_result.x

    def _calculate_risk_contributions(self, weights, cov_matrix):
        portfolio_volatility = np.sqrt(weights.T.dot(cov_matrix).dot(weights))
        marginal_risk_contributions = cov_matrix.dot(weights)
        risk_contributions = np.multiply(marginal_risk_contributions, weights.T) / portfolio_volatility
        return risk_contributions



class CustomOptimizer(Optimizer):
    def __init__(self,
                 custom_objective_function: Callable[[np.ndarray], float],
                 custom_constraints: List[Dict],
                 custom_bounds: Optional[Tuple[float, float]] = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_objective_function = custom_objective_function
        self.custom_constraints = custom_constraints
        self.custom_bounds = custom_bounds

    def optimize(self, data: pd.DataFrame) -> Dict[str, float]:
        expected_returns, cov_matrix = self._calculate_statistics(data)
        weights = self._custom_allocation(expected_returns, cov_matrix)
        return dict(zip(data.columns, weights))

    def _custom_allocation(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame) -> np.ndarray:
        num_assets = len(cov_matrix)
        init_weights = np.ones(num_assets) / num_assets
        bounds = self.custom_bounds or [(0, 1) for _ in range(num_assets)]

        def wrapped_objective_function(weights):
            return self.custom_objective_function(weights, expected_returns, cov_matrix)

        optimized_result = minimize(wrapped_objective_function, init_weights, method='SLSQP',
                                    constraints=self.custom_constraints, bounds=bounds)
        return optimized_result.x


def mean_variance_objective_function(weights, expected_returns, cov_matrix):
    portfolio_volatility = np.sqrt(weights.T.dot(cov_matrix).dot(weights))
    portfolio_return = expected_returns.dot(weights)
    return -portfolio_return / portfolio_volatility

'''
custom_constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
custom_bounds = (0, 1)

custom_mean_variance_optimizer = CustomOptimizer(mean_variance_objective_function,
                                                 custom_constraints,
                                                 custom_bounds)

data = pd.DataFrame()  # load data
allocation = custom_mean_variance_optimizer.optimize(data)
'''


"""
# Sample usage
universe = Universe()  # Assuming you're using the Universe class provided earlier
strategies = [mean_variance_strategy]  # List of strategy functions

pipeline = AssetAllocationPipeline(universe, strategies)
data = pd.DataFrame()  # Load your data here
allocation_results = pipeline.run_pipeline(data)

print(allocation_results)"""
