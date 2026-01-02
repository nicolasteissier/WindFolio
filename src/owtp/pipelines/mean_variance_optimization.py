import owtp.config
from pathlib import Path
from typing import Literal
import numpy as np
import cvxpy as cp
import pandas as pd


class MeanVariancePortfolioOptimizer:
    """
    Mean-variance portfolio optimization for wind farm locations.
    """

    def __init__(self, target: Literal["paths", "paths_local"] = "paths_local", adjusted_height: bool = True):
        self.config = owtp.config.load_yaml_config()
        
        self.mean_revenue_path = Path(self.config[target]['processed_data']) / "parquet" / "revenues" / "mean" / "mean_revenue.parquet"
        self.mean_revenue_100m_path = Path(self.config[target]['processed_data']) / "parquet" / "revenues_100m" / "mean" / "mean_revenue.parquet"
        
        self.covariance_csv_path = Path(self.config[target]['processed_data']) / "csv" / "covariance_matrix" / "covariance_matrix_pivoted.csv"
        self.covariance_parquet_path = Path(self.config[target]['processed_data']) / "parquet" / "covariance_matrix" / "covariance_matrix_pivoted.parquet"
        self.covariance_100m_csv_path = Path(self.config[target]['processed_data']) / "csv" / "covariance_matrix_100m" / "covariance_matrix_pivoted.csv"
        self.covariance_100m_parquet_path = Path(self.config[target]['processed_data']) / "parquet" / "covariance_matrix_100m" / "covariance_matrix_pivoted.parquet"
        
        self.output_parquet_dir = Path(self.config[target]['processed_data']) / "parquet" / "portfolio_weights"
        self.output_parquet_dir.mkdir(parents=True, exist_ok=True)
        self.output_100m_parquet_dir = Path(self.config[target]['processed_data']) / "parquet" / "portfolio_weights_100m"
        self.output_100m_parquet_dir.mkdir(parents=True, exist_ok=True)
        
        self.output_csv_dir = Path(self.config[target]['processed_data']) / "csv" / "portfolio_weights"
        self.output_csv_dir.mkdir(parents=True, exist_ok=True)
        self.output_100m_csv_dir = Path(self.config[target]['processed_data']) / "csv" / "portfolio_weights_100m"
        self.output_100m_csv_dir.mkdir(parents=True, exist_ok=True)
        
        self.adjusted_height = adjusted_height
        self.mean_revenue_df = None
        self.covariance_matrix = None
        self.location_to_idx = None
        self.idx_to_location = None

    def load_data(self, verbose=True):
        """Load mean revenues and covariance matrix."""
        
        if self.adjusted_height:
            mean_revenue_path = self.mean_revenue_100m_path
            covariance_parquet_path = self.covariance_100m_parquet_path
            covariance_csv_path = self.covariance_100m_csv_path
        else:
            mean_revenue_path = self.mean_revenue_path
            covariance_parquet_path = self.covariance_parquet_path
            covariance_csv_path = self.covariance_csv_path
        
        if verbose:
            if self.adjusted_height:
                print(f"Loading mean revenues from {mean_revenue_path} (adjusted height)...")
            else:
                print(f"Loading mean revenues from {mean_revenue_path}...")
        
        self.mean_revenue_df = pd.read_parquet(mean_revenue_path)
        
        self.location_to_idx = {loc: idx for idx, loc in enumerate(self.mean_revenue_df['location'])}
        self.idx_to_location = {idx: loc for loc, idx in self.location_to_idx.items()}
        
        if verbose:
            print(f"\nLoading covariance matrix...")
        
        if covariance_parquet_path.exists():
            cov_df = pd.read_parquet(covariance_parquet_path)
            if verbose:
                print(f"Loaded pivoted covariance matrix from parquet")
        elif covariance_csv_path.exists():
            cov_df = pd.read_csv(covariance_csv_path, index_col=0)
            if verbose:
                print(f"Loaded pivoted covariance matrix from CSV")
        else:
            raise FileNotFoundError(f"Pivoted covariance matrix not found at {covariance_parquet_path} or {covariance_csv_path}")
        
        # Align covariance matrix with mean revenue locations
        locations_revenue = self.mean_revenue_df['location'].tolist()
        locations_cov = cov_df.index.tolist()
        
        common_locations = sorted(set(locations_revenue) & set(locations_cov))
        
        if verbose:
            print(f"  - Mean revenue range: {self.mean_revenue_df['mean_revenue'].min():.2f} - {self.mean_revenue_df['mean_revenue'].max():.2f}")
            print(f"  - Covariance matrix shape: {cov_df.shape}")
            print(f"  - Mean revenue locations: {len(locations_revenue)}")
            print(f"  - Covariance locations: {len(locations_cov)}")
            print(f"  - Common locations: {len(common_locations)} = {len(common_locations) / len(locations_revenue) * 100:.2f} % of mean revenue locations")
            print(f"                           = {len(common_locations) / len(locations_cov) * 100:.2f} % of covariance locations")
        
        # Filter mean revenue to common locations and reindex
        self.mean_revenue_df = self.mean_revenue_df[self.mean_revenue_df['location'].isin(common_locations)].copy()
        self.mean_revenue_df = self.mean_revenue_df.set_index('location').loc[common_locations].reset_index()
        
        # Update location mappings
        self.location_to_idx = {loc: idx for idx, loc in enumerate(common_locations)}
        self.idx_to_location = {idx: loc for loc, idx in self.location_to_idx.items()}
        
        # Extract aligned covariance matrix
        self.covariance_matrix = cov_df.loc[common_locations, common_locations].values
        
        if verbose:
            print(f"Aligned covariance matrix:")
            print(f"  - Final shape: {self.covariance_matrix.shape}")
            print(f"  - Diagonal range: {np.diag(self.covariance_matrix).min():.4f} - {np.diag(self.covariance_matrix).max():.4f}")
            print(f"  - Mean covariance: {self.covariance_matrix.mean():.4f}")

    def optimize(
        self, 
        total_turbines=10, 
        lambda_risk=1.0,
        min_revenue_threshold=None,
        max_locations=None,
        verbose=True
    ):
        """
        Solve mean-variance portfolio optimization.
        
        Args:
            total_turbines: Total number of turbines to allocate
            lambda_risk: Risk aversion parameter (higher = more conservative)
            min_revenue_threshold: Only consider locations with mean revenue above this threshold, None = no filter
            max_locations: Maximum number of locations to invest in (for sparsity), None = no limit
            verbose: Print optimization details
        
        Returns:
            Tuple of (continuous_weights, integer_weights, optimization_results)
        """
        
        if self.mean_revenue_df is None or self.covariance_matrix is None:
            self.load_data(verbose=verbose)
        
        # Filter locations by minimum revenue if specified
        if min_revenue_threshold is not None:
            # Get diagonal variances for filtering
            location_variances = np.diag(self.covariance_matrix)
            
            # Filter by both mean revenue AND variance (avoid zero/low variance issues)
            eligible_mask = (self.mean_revenue_df['mean_revenue'] >= min_revenue_threshold) & (location_variances > 1e-6)
            eligible_indices = eligible_mask.values
            
            if verbose:
                n_eligible = eligible_mask.sum()
                n_low_variance = ((self.mean_revenue_df['mean_revenue'] >= min_revenue_threshold) & (location_variances <= 1e-6)).sum()
                print(f"\nFiltering locations with mean revenue >= {min_revenue_threshold}")
                print(f"Eligible locations: {n_eligible} / {len(self.mean_revenue_df)}")
                if n_low_variance > 0:
                    print(f"  (excluded {n_low_variance} locations with sufficient revenue but near-zero variance)")
            
            mu = self.mean_revenue_df.loc[eligible_mask, 'mean_revenue'].values
            Sigma = self.covariance_matrix[np.ix_(eligible_indices, eligible_indices)]
            eligible_locs = self.mean_revenue_df.loc[eligible_mask, 'location'].values
        else:
            mu = self.mean_revenue_df['mean_revenue'].values
            Sigma = self.covariance_matrix
            eligible_locs = self.mean_revenue_df['location'].values
            eligible_indices = np.ones(len(mu), dtype=bool)
        
        N = len(mu)
        
        if verbose:
            print(f"\nOptimizing portfolio with:")
            print(f"  - {N} locations")
            print(f"  - {total_turbines} total turbines")
            print(f"  - Risk aversion λ = {lambda_risk}")
        
        # Add small regularization to ensure PSD (handles numerical issues)
        epsilon = 1e-6
        Sigma_reg = Sigma + epsilon * np.eye(N)
        
        # Wrap with psd_wrap to skip CVXPY's numerical PSD check (which can fail even for valid matrices)
        Sigma_wrapped = cp.psd_wrap(Sigma_reg)
        
        # Decision variable: number of turbines at each location
        weights = cp.Variable(N)
        
        # Objective: maximize expected return - risk penalty
        expected_return = mu @ weights
        portfolio_variance = cp.quad_form(weights, Sigma_wrapped)
        objective = cp.Maximize(expected_return - lambda_risk * portfolio_variance)
        
        # Constraints
        constraints = [
            weights >= 0,                      # No short selling (non-negative turbines)
            cp.sum(weights) <= total_turbines  # Budget constraint
        ]
        
        # Optional: limit number of locations with turbines (sparsity)
        if max_locations is not None:
            # TODO: implement cardinality constraint ?
            if verbose:
                print(f"  - Note: max_locations constraint not yet implemented, ignoring this constraint.")
        
        # Solve optimization problem
        prob = cp.Problem(objective, constraints)
        
        try:
            if verbose:
                print(f"\nSolving optimization problem...")
            prob.solve(solver=cp.CLARABEL, verbose=False) # True for detailed solver output
        except:
            if verbose:
                print(f"\nCLARABEL solver failed, trying SCS solver...")
            prob.solve(solver=cp.SCS, verbose=False)
        
        if verbose:
            print(f"✓ Solver finished with status: {prob.status}")
        
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError(f"Optimization failed with status: {prob.status}")
        
        # Extract results
        weights_opt = weights.value
        # Round to nearest integer (with round half to even TODO: check if pertinent)
        weights_opt_int = np.round(weights_opt).astype(int)
        
        # Create full-size arrays (including filtered-out locations)
        weights_full = np.zeros(len(self.mean_revenue_df))
        weights_full[eligible_indices] = weights_opt
        
        weights_int_full = np.zeros(len(self.mean_revenue_df), dtype=int)
        weights_int_full[eligible_indices] = weights_opt_int
        
        # Calculate portfolio metrics
        results = {
            'expected_return_continuous': float(mu @ weights_opt),
            'expected_return_integer': float(self.mean_revenue_df['mean_revenue'].values @ weights_int_full),
            'portfolio_variance_continuous': float(weights_opt @ Sigma @ weights_opt),
            'portfolio_variance_integer': float(weights_int_full @ self.covariance_matrix @ weights_int_full),
            'total_turbines_continuous': float(weights_opt.sum()),
            'total_turbines_integer': int(weights_int_full.sum()),
            'num_locations_continuous': int((weights_opt > 1e-6).sum()),
            'num_locations_integer': int((weights_int_full > 0).sum()),
            'objective_value': float(prob.value),
            'regularization': epsilon,
        }
        
        if verbose:
            print(f"\nOptimization Results:")
            print(f"  Continuous solution:")
            print(f"    - Expected return: {results['expected_return_continuous']:.2f}")
            print(f"    - Portfolio std: {np.sqrt(results['portfolio_variance_continuous']):.2f}")
            print(f"    - Total turbines: {results['total_turbines_continuous']:.2f}")
            print(f"    - Active locations: {results['num_locations_continuous']}")
            print(f"\n  Integer solution:")
            print(f"    - Expected return: {results['expected_return_integer']:.2f}")
            print(f"    - Portfolio std: {np.sqrt(results['portfolio_variance_integer']):.2f}")
            print(f"    - Total turbines: {results['total_turbines_integer']}")
            print(f"    - Active locations: {results['num_locations_integer']}")
        
        return weights_full, weights_int_full, results

    def save_weights(self, weights_continuous, weights_integer, results, suffix="", verbose=True):
        """Save portfolio weights to disk."""

        if self.adjusted_height:
            output_parquet_dir = self.output_100m_parquet_dir / suffix
            output_csv_dir = self.output_100m_csv_dir / suffix
        else:
            output_parquet_dir = self.output_parquet_dir / suffix
            output_csv_dir = self.output_csv_dir / suffix
        
        output_parquet_dir.mkdir(parents=True, exist_ok=True)
        output_csv_dir.mkdir(parents=True, exist_ok=True)
        
        weights_df = self.mean_revenue_df[['location', 'latitude', 'longitude', 'mean_revenue']].copy()
        weights_df['weight_continuous'] = weights_continuous
        weights_df['weight_integer'] = weights_integer
        
        # Filter to only show locations with non-zero allocation
        weights_active_df = weights_df[weights_df['weight_integer'] > 0].copy()
        weights_active_df = weights_active_df.sort_values('weight_integer', ascending=False)
        
        parquet_path = output_parquet_dir / f"portfolio_weights{suffix}.parquet"
        csv_path = output_csv_dir / f"portfolio_weights{suffix}.csv"
        
        weights_df.to_parquet(parquet_path, index=False)
        weights_df.to_csv(csv_path, index=False)
        
        # Save active weights only
        parquet_path_active = output_parquet_dir / f"portfolio_weights_active{suffix}.parquet"
        csv_path_active = output_csv_dir / f"portfolio_weights_active{suffix}.csv"
        
        weights_active_df.to_parquet(parquet_path_active, index=False)
        weights_active_df.to_csv(csv_path_active, index=False)
        
        # Save results summary
        results_df = pd.DataFrame([results])
        results_path = output_csv_dir / f"optimization_results{suffix}.csv"
        results_df.to_csv(results_path, index=False)
        
        if verbose:
            print(f"\nSaved portfolio weights to:")
            print(f"  - {parquet_path}")
            print(f"  - {csv_path}")
            print(f"  - {parquet_path_active} (active only)")
            print(f"  - {results_path} (summary)")


if __name__ == "__main__":
    optimizer = MeanVariancePortfolioOptimizer(target="paths_local")
    
    # Load data
    optimizer.load_data(verbose=True)

    # PARAMETERS
    total_turbines = 100
    lambda_risk = 4.5e-4
    min_revenue_threshold = 10.0  # EUR/hour
    max_locations = None
    
    # Run optimization
    weights_cont, weights_int, results = optimizer.optimize(
        total_turbines=total_turbines,
        lambda_risk=lambda_risk,
        min_revenue_threshold=min_revenue_threshold,
        max_locations=max_locations,
        verbose=True
    )
    
    suffix = f"_({total_turbines})_({lambda_risk})_({min_revenue_threshold})" # no max_locations for the moment (not implemented)
    optimizer.save_weights(weights_cont, weights_int, results, suffix=suffix, verbose=True)
    
    print("\n" + "="*60)
    print("Top 10 locations by turbine allocation:")
    print("="*60)
    
    top_locations = weights_int.argsort()[::-1][:10]

    for i, idx in enumerate(top_locations):
        if weights_int[idx] > 0:
            loc = optimizer.mean_revenue_df.loc[idx, 'location']
            lat = optimizer.mean_revenue_df.loc[idx, 'latitude']
            lon = optimizer.mean_revenue_df.loc[idx, 'longitude']
            rev = optimizer.mean_revenue_df.loc[idx, 'mean_revenue']
            n_turbines = weights_int[idx]
            print(f"{i+1}. {loc} ({lat:.1f}, {lon:.1f}): {n_turbines} turbines, {rev:.2f} EUR/hour mean revenue")
