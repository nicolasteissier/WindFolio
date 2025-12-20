from typing import Literal

import owtp.config
from pathlib import Path
import numpy as np
import cvxpy as cp
import pandas as pd
class MeanVariancePortfolio:
    """
    Comment
    """

    def __init__(self, freq: Literal['hourly', '6minute'] = 'hourly'):
        self.config = owtp.config.load_yaml_config()
        # might change the path, need to ask nico where he will store the returns
        #self.input_dir = Path(self.config['paths_local']['intermediate_data']) / "parquet" / "returns" / str(freq)
        #self.output_dir = Path(self.config['paths_local']['intermediate_data']) / "parquet" / "weights" / str(freq)
        self.returns = np.array([
            [-0.01, 0.02, -0.03, -0.03],
            [0.03, -0.01, 0.00, 0.00],
            [-0.02, 0.01, 0.02, 0.02],
            [0.01, 0.00, 0.01, 0.01]
        ])
        #self.returns = pd.read_parquet(self.input_dir / "returns.parquet").to_numpy()

    
    def covariance_matrix(self):
        Sigma = np.cov(self.returns, rowvar=False)
        return Sigma

    def mean(self):
        mu = self.returns.mean(axis=0)
        return mu

    def mean_variance_portfolio(self):
        
        Sigma = self.covariance_matrix()
        mu = self.mean()

        N = len(mu)

        weights = cp.Variable(N)

        lam = 1.0

        objective = cp.Maximize(
            mu @ weights - lam * cp.quad_form(weights, Sigma)
        )

        constraints = [
            weights >= 0,
            cp.sum(weights) <= 10  # total budget !!!!
        ]

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.CLARABEL)

        weights_opt = weights.value
        weights_opt_int = np.round(weights_opt)

        print(f"Continuous solution: {weights_opt}")
        print(f"Rounded integer solution: {weights_opt_int}")
        return weights_opt, weights_opt_int

    def portfolio_returns(self, weights):
        """
        Calculate portfolio returns given asset returns and weights.
        
        Args:
            returns: (T, N) array where T=time periods, N=assets
            weights: (N,) array of portfolio weights
        
        Returns:
            (T,) array of portfolio returns for each time period
        """
        return self.returns @ weights  


if __name__ == "__main__":
    mvp = MeanVariancePortfolio(freq='hourly')
    weights_opt, weights_opt_int = mvp.mean_variance_portfolio()

    portfolio_rets = mvp.portfolio_returns(weights_opt_int)
    print(f"Portfolio returns over time: {portfolio_rets}")
    print(f"Mean portfolio return: {portfolio_rets.mean():.4f}")
    print(f"Portfolio volatility: {portfolio_rets.std():.4f}")