"""
Policy analysis utilities for DCMBench models.

This module provides comprehensive policy analysis capabilities for discrete choice models
using the DCMBench adapter system. It includes ASC calibration, scenario simulation,
and sensitivity analysis with visualization support.

Author: DCMBench Team
License: MIT
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union
from copy import deepcopy
import logging

import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Variable

logger = logging.getLogger(__name__)


class PolicyAnalyzer:
    """
    Policy analysis toolkit for DCMBench-adapted models.
    
    This class provides methods for:
    - Calibrating alternative-specific constants (ASCs) to match observed market shares
    - Simulating policy scenarios with modified attributes
    - Conducting systematic sensitivity analyses
    - Visualizing results
    
    The analyzer works with any model that has a UniversalBiogemeAdapter,
    making it compatible with all DCMBench-benchmarked models.
    
    Parameters
    ----------
    adapter : UniversalBiogemeAdapter
        The DCMBench adapter wrapping the estimated model
    original_data : pd.DataFrame
        The original dataset used for model estimation
    choice_column : str, optional
        Name of the choice column in the data. Default is 'CHOICE'
        
    Examples
    --------
    >>> from dcmbench.adapters import UniversalBiogemeAdapter
    >>> from dcmbench.utils.policy_analysis import PolicyAnalyzer
    >>> 
    >>> # After model estimation
    >>> adapter = UniversalBiogemeAdapter(model, results, database)
    >>> analyzer = PolicyAnalyzer(adapter, data)
    >>> 
    >>> # Calibrate ASCs
    >>> calibrated_params = analyzer.calibrate_ascs()
    >>> 
    >>> # Run sensitivity analysis
    >>> results = analyzer.sensitivity_analysis('cost', [0.5, 1.0, 1.5, 2.0])
    """
    
    def __init__(self, 
                 adapter,
                 original_data: pd.DataFrame,
                 choice_column: str = 'CHOICE'):
        """Initialize the policy analyzer."""
        self.adapter = adapter
        self.original_data = original_data.copy()
        self.choice_column = choice_column
        
        # Extract model components
        self.model = adapter.model
        self.results = adapter.results
        self.original_betas = self.results.get_beta_values()
        self.calibrated_betas = None
        
        # Detect alternatives from data
        self._detect_alternatives()
        
        # Validate adapter has necessary components
        self._validate_adapter()
        
        logger.info(f"PolicyAnalyzer initialized for model: {adapter.name}")
        
    def _detect_alternatives(self):
        """Automatically detect available alternatives from the data."""
        if self.choice_column in self.original_data.columns:
            self.alternatives = sorted(self.original_data[self.choice_column].unique())
            self.alternatives = [alt for alt in self.alternatives if pd.notna(alt)]
        else:
            # Try alternative names
            for col_name in ['choice', 'CHOSEN', 'chosen']:
                if col_name in self.original_data.columns:
                    self.choice_column = col_name
                    self.alternatives = sorted(self.original_data[col_name].unique())
                    self.alternatives = [alt for alt in self.alternatives if pd.notna(alt)]
                    break
            else:
                raise ValueError(f"Could not find choice column in data. Tried: {['CHOICE', 'choice', 'CHOSEN', 'chosen']}")
        
        logger.info(f"Detected {len(self.alternatives)} alternatives: {self.alternatives}")
        
    def _validate_adapter(self):
        """Validate that the adapter has necessary components for policy analysis."""
        if not hasattr(self.adapter, 'V') or self.adapter.V is None:
            logger.warning("Adapter missing utility functions (V). Some features may not work.")
        if not hasattr(self.adapter, 'av') or self.adapter.av is None:
            logger.warning("Adapter missing availability conditions (av). Some features may not work.")
            
    def calculate_observed_shares(self, 
                                 data: Optional[pd.DataFrame] = None) -> Dict[int, float]:
        """
        Calculate observed market shares from data.
        
        Parameters
        ----------
        data : pd.DataFrame, optional
            Dataset to calculate shares from. If None, uses original_data
            
        Returns
        -------
        Dict[int, float]
            Market share for each alternative
            
        Examples
        --------
        >>> shares = analyzer.calculate_observed_shares()
        >>> print(f"Car share: {shares[2]:.2%}")
        """
        if data is None:
            data = self.original_data
            
        shares = {}
        total = len(data)
        
        for alt in self.alternatives:
            count = (data[self.choice_column] == alt).sum()
            shares[alt] = count / total if total > 0 else 0
            
        return shares
    
    def simulate_probabilities(self,
                             data: Optional[pd.DataFrame] = None,
                             betas: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Simulate choice probabilities using the model.
        
        This method uses the model's utility functions and parameters to calculate
        choice probabilities for each observation and alternative.
        
        Parameters
        ----------
        data : pd.DataFrame, optional
            Data for simulation. If None, uses original_data
        betas : Dict[str, float], optional
            Parameter values to use. If None, uses original estimates
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns P_1, P_2, etc. for each alternative's probability
            
        Raises
        ------
        ValueError
            If the adapter doesn't have necessary components (V, av)
        """
        if data is None:
            data = self.original_data
        if betas is None:
            betas = self.original_betas
            
        # Check if adapter has extracted utilities
        if self.adapter.V is None or self.adapter.av is None:
            raise ValueError("Model adapter does not have utility functions extracted. "
                           "Ensure model.V and model.av are set before creating adapter.")
        
        # Create Biogeme database
        sim_db = db.Database("simulation", data)
        
        # Build probability formulas for each alternative
        prob_formulas = {}
        
        # Check if utilities contain random draws (MXL case)
        has_draws = False
        if self.adapter.V:
            # Check if any utility contains bioDraws
            for v in self.adapter.V.values():
                if 'bioDraws' in str(v):
                    has_draws = True
                    break
        
        # Build appropriate probability formulas
        if has_draws:
            # For MXL with random parameters, wrap in MonteCarlo
            from biogeme.expressions import MonteCarlo
            for alt in self.adapter.V.keys():
                alt_prob = models.logit(self.adapter.V, self.adapter.av, alt)
                prob_formulas[f'P_{alt}'] = MonteCarlo(alt_prob)
        elif hasattr(self.adapter, 'nests') and self.adapter.nests:
            # For nested logit, use nested probability
            for alt in self.adapter.V.keys():
                prob_formulas[f'P_{alt}'] = models.nested(
                    self.adapter.V, self.adapter.av, self.adapter.nests, alt
                )
        else:
            # Standard MNL
            for alt in self.adapter.V.keys():
                prob_formulas[f'P_{alt}'] = models.logit(self.adapter.V, self.adapter.av, alt)
        
        # Create simulation model
        sim_model = bio.BIOGEME(sim_db, prob_formulas)
        sim_model.modelName = f"{self.adapter.name}_simulation"
        
        # Transfer model-specific settings
        if hasattr(self.adapter, 'number_of_draws') and self.adapter.number_of_draws:
            sim_model.number_of_draws = self.adapter.number_of_draws
            
        # Simulate probabilities
        probabilities = sim_model.simulate(betas)
        
        return probabilities
    
    def calibrate_ascs(self,
                      target_shares: Optional[Dict[int, float]] = None,
                      asc_mapping: Optional[Dict[int, str]] = None,
                      reference_alternative: Optional[int] = None,
                      max_iter: int = 200,
                      tolerance: float = 0.0001,
                      step_size: float = 0.5,
                      verbose: bool = True) -> Dict[str, float]:
        """
        Calibrate alternative-specific constants (ASCs) to match target market shares.
        
        Uses an iterative algorithm to adjust ASCs until the model's predicted
        market shares match the target shares within tolerance.
        
        Parameters
        ----------
        target_shares : Dict[int, float], optional
            Target market shares for each alternative. If None, uses observed shares
        asc_mapping : Dict[int, str], optional
            Mapping from alternative IDs to ASC parameter names.
            Example: {1: 'ASC_TRAIN', 2: 'ASC_CAR', 3: 'ASC_BUS'}
            If None, attempts auto-detection
        reference_alternative : int, optional
            The reference alternative (ASC fixed to 0). If None, auto-detects
        max_iter : int, default=100
            Maximum number of iterations
        tolerance : float, default=0.001
            Convergence tolerance for market share differences
        step_size : float, default=0.5
            Initial step size for ASC adjustments
        verbose : bool, default=True
            Print progress information
            
        Returns
        -------
        Dict[str, float]
            Dictionary of calibrated parameters (includes all parameters, not just ASCs)
            
        Examples
        --------
        >>> # Calibrate to match observed shares
        >>> calibrated = analyzer.calibrate_ascs()
        >>> 
        >>> # Calibrate to custom target shares
        >>> targets = {1: 0.3, 2: 0.5, 3: 0.2}
        >>> calibrated = analyzer.calibrate_ascs(target_shares=targets)
        """
        # Get target shares
        if target_shares is None:
            target_shares = self.calculate_observed_shares()
            
        if verbose:
            print("\nCalibrating ASCs to match market shares...")
            print(f"Target shares: {', '.join([f'Alt {k}: {v:.2%}' for k, v in target_shares.items()])}")
        
        # Auto-detect ASC mapping if not provided
        if asc_mapping is None:
            asc_mapping = self._detect_asc_mapping(reference_alternative)
            
        # Remove None values (reference alternatives)
        asc_mapping = {k: v for k, v in asc_mapping.items() if v is not None}
        
        if verbose and asc_mapping:
            print(f"Adjusting ASCs: {list(asc_mapping.values())}")
        
        # Start with original parameters
        current_betas = self.original_betas.copy()
        
        # Calibration loop
        best_diff = float('inf')
        best_betas = current_betas.copy()
        current_step = step_size
        
        for iteration in range(max_iter):
            # Calculate current probabilities
            try:
                probs = self.simulate_probabilities(betas=current_betas)
            except Exception as e:
                logger.warning(f"Simulation failed in iteration {iteration}: {e}")
                break
            
            # Calculate current market shares
            current_shares = {}
            for alt in self.alternatives:
                col_name = f'P_{alt}'
                if col_name in probs.columns:
                    current_shares[alt] = probs[col_name].mean()
            
            # Calculate convergence metric
            diffs = {alt: target_shares.get(alt, 0) - current_shares.get(alt, 0) 
                    for alt in self.alternatives}
            max_diff = max(abs(d) for d in diffs.values())
            
            # Track best solution
            if max_diff < best_diff:
                best_diff = max_diff
                best_betas = current_betas.copy()
            
            # Check convergence
            if max_diff < tolerance:
                if verbose:
                    print(f"Converged after {iteration + 1} iterations")
                    print(f"Final shares: {', '.join([f'Alt {k}: {current_shares.get(k, 0):.2%}' for k in self.alternatives])}")
                break
            
            # Update ASCs using gradient-based adjustment
            for alt, asc_name in asc_mapping.items():
                if alt in target_shares and alt in current_shares:
                    if current_shares[alt] > 0:
                        # Log-ratio adjustment with adaptive step size
                        log_ratio = np.log(target_shares[alt] / current_shares[alt])
                        adjustment = current_step * log_ratio
                        current_betas[asc_name] = current_betas.get(asc_name, 0) + adjustment
            
            # Reduce step size gradually
            current_step = max(current_step * 0.95, 0.01)
            
            if verbose and iteration % 20 == 0:
                print(f"  Iteration {iteration}: max_diff = {max_diff:.4f}")
        else:
            if verbose:
                print(f"Maximum iterations ({max_iter}) reached")
                print(f"Best difference achieved: {best_diff:.4f}")
            current_betas = best_betas
        
        self.calibrated_betas = current_betas
        return current_betas
    
    def _detect_asc_mapping(self, 
                           reference_alternative: Optional[int] = None) -> Dict[int, str]:
        """
        Auto-detect ASC parameter mapping from model.
        
        Parameters
        ----------
        reference_alternative : int, optional
            The reference alternative. If None, tries to detect
            
        Returns
        -------
        Dict[int, str]
            Mapping of alternative IDs to ASC parameter names
        """
        asc_mapping = {}
        beta_names = list(self.original_betas.keys())
        
        # Common patterns for ASC parameters
        asc_patterns = ['ASC_', 'asc_', 'CONST_', 'const_', 'ASC', 'asc']
        
        # Common mode name mappings
        mode_names = {
            1: ['TRAIN', 'Train', 'train', 'RAIL', 'Rail', 'WALK', 'Walk', 'walk'],
            2: ['CAR', 'Car', 'car', 'AUTO', 'Auto', 'DRIVE', 'Drive', 'SM', 'CYCLE', 'Cycle'],
            3: ['BUS', 'Bus', 'bus', 'TRANSIT', 'Transit', 'PT', 'Public', 'CAR'],
            4: ['AIR', 'Air', 'air', 'PLANE', 'Plane', 'FLY', 'DRIVE', 'Drive']
        }
        
        # Try to match alternatives to ASC parameters
        for alt in self.alternatives:
            found = False
            
            # First try exact numeric match
            for pattern in asc_patterns:
                param_name = f"{pattern}{alt}"
                if param_name in beta_names:
                    asc_mapping[alt] = param_name
                    found = True
                    break
            
            # If not found, try mode names
            if not found and alt in mode_names:
                for mode in mode_names[alt]:
                    for pattern in asc_patterns:
                        param_name = f"{pattern}{mode}"
                        if param_name in beta_names:
                            asc_mapping[alt] = param_name
                            found = True
                            break
                    if found:
                        break
        
        # Detect reference alternative (ASC fixed to 0 or not in parameters)
        if reference_alternative is None:
            # Check which alternative doesn't have an ASC
            for alt in self.alternatives:
                if alt not in asc_mapping:
                    reference_alternative = alt
                    break
        
        # Mark reference alternative
        if reference_alternative in asc_mapping:
            # Check if it's actually fixed
            param_name = asc_mapping[reference_alternative]
            if param_name in self.original_betas:
                # It exists, so maybe another is reference
                pass
            else:
                asc_mapping[reference_alternative] = None
        else:
            asc_mapping[reference_alternative] = None
        
        logger.info(f"Detected ASC mapping: {asc_mapping}")
        return asc_mapping
    
    def simulate_scenario(self,
                         attribute_changes: Dict[str, float],
                         use_calibrated: bool = True) -> Tuple[pd.DataFrame, Dict[int, float]]:
        """
        Simulate a policy scenario with attribute changes.
        
        Parameters
        ----------
        attribute_changes : Dict[str, float]
            Dictionary mapping attribute names to multipliers.
            Example: {'CAR_COST': 1.5, 'TRAIN_TIME': 0.9}
        use_calibrated : bool, default=True
            Use calibrated ASCs if available, otherwise use original parameters
            
        Returns
        -------
        Tuple[pd.DataFrame, Dict[int, float]]
            (Modified data, New market shares for each alternative)
            
        Examples
        --------
        >>> # Simulate 50% increase in car cost
        >>> data, shares = analyzer.simulate_scenario({'CAR_COST': 1.5})
        >>> print(f"New car share: {shares[2]:.2%}")
        """
        # Create modified data
        modified_data = self.original_data.copy()
        
        for attr, multiplier in attribute_changes.items():
            if attr in modified_data.columns:
                modified_data[attr] = modified_data[attr] * multiplier
                logger.info(f"Modified {attr} by factor {multiplier}")
            else:
                logger.warning(f"Attribute {attr} not found in data")
        
        # Choose parameters
        if use_calibrated and self.calibrated_betas is not None:
            betas = self.calibrated_betas
        else:
            betas = self.original_betas
        
        # Simulate probabilities
        probs = self.simulate_probabilities(data=modified_data, betas=betas)
        
        # Calculate market shares
        market_shares = {}
        for alt in self.alternatives:
            col_name = f'P_{alt}'
            if col_name in probs.columns:
                market_shares[alt] = probs[col_name].mean()
        
        return modified_data, market_shares
    
    def sensitivity_analysis(self,
                            attribute: str,
                            multipliers: List[float],
                            use_calibrated: bool = True,
                            verbose: bool = True) -> pd.DataFrame:
        """
        Conduct sensitivity analysis by varying an attribute.
        
        Parameters
        ----------
        attribute : str
            Name of the attribute to vary (must be a column in the data)
        multipliers : List[float]
            List of multipliers to apply to the attribute
        use_calibrated : bool, default=True
            Use calibrated ASCs if available
        verbose : bool, default=True
            Print progress information
            
        Returns
        -------
        pd.DataFrame
            Results with columns: multiplier, alt_1, alt_2, ..., alt_n
            Each row represents market shares for a given multiplier
            
        Examples
        --------
        >>> # Analyze sensitivity to car cost changes
        >>> results = analyzer.sensitivity_analysis(
        ...     'CAR_COST',
        ...     [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        ... )
        >>> # Plot results
        >>> results.plot(x='multiplier', y=['alt_1', 'alt_2', 'alt_3'])
        """
        results = []
        
        if verbose:
            print(f"\nConducting sensitivity analysis for {attribute}")
            print(f"Multipliers: {multipliers}")
        
        for mult in multipliers:
            try:
                # Simulate scenario
                _, shares = self.simulate_scenario(
                    {attribute: mult},
                    use_calibrated=use_calibrated
                )
                
                # Store results
                result = {'multiplier': mult}
                for alt in self.alternatives:
                    result[f'alt_{alt}'] = shares.get(alt, 0)
                results.append(result)
                
                if verbose:
                    # Report on one alternative (usually the most affected)
                    focal_alt = max(shares.keys(), key=lambda k: abs(shares[k] - 1/len(shares)))
                    print(f"  {attribute} × {mult:.2f}: Alt {focal_alt} share = {shares[focal_alt]:.2%}")
                    
            except Exception as e:
                logger.warning(f"Failed to simulate multiplier {mult}: {e}")
                # Add NaN results
                result = {'multiplier': mult}
                for alt in self.alternatives:
                    result[f'alt_{alt}'] = np.nan
                results.append(result)
        
        return pd.DataFrame(results)
    
    def plot_sensitivity(self,
                        results: pd.DataFrame,
                        attribute_name: str,
                        alternative_names: Optional[Dict[int, str]] = None,
                        focal_alternative: Optional[int] = None,
                        figsize: Tuple[float, float] = (14, 6),
                        colors: Optional[Dict[int, str]] = None) -> plt.Figure:
        """
        Plot sensitivity analysis results.
        
        Creates a two-panel plot showing:
        1. Market shares vs attribute multiplier
        2. Percent change from baseline vs attribute multiplier
        
        Parameters
        ----------
        results : pd.DataFrame
            Results from sensitivity_analysis()
        attribute_name : str
            Name of the attribute for axis labels
        alternative_names : Dict[int, str], optional
            Human-readable names for alternatives
        focal_alternative : int, optional
            Alternative to highlight in the plots
        figsize : Tuple[float, float], default=(14, 6)
            Figure size (width, height)
        colors : Dict[int, str], optional
            Color mapping for alternatives
            
        Returns
        -------
        plt.Figure
            The matplotlib figure object
            
        Examples
        --------
        >>> results = analyzer.sensitivity_analysis('COST', [0.5, 1.0, 1.5])
        >>> fig = analyzer.plot_sensitivity(
        ...     results,
        ...     'Cost',
        ...     alternative_names={1: 'Train', 2: 'Car', 3: 'Bus'}
        ... )
        >>> fig.savefig('sensitivity.png')
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Prepare data
        multipliers = results['multiplier'].values
        alt_columns = [col for col in results.columns if col.startswith('alt_')]
        
        # Get baseline (multiplier = 1.0)
        baseline_idx = np.argmin(np.abs(multipliers - 1.0))
        baseline_shares = {}
        
        # Default colors
        if colors is None:
            colors = {alt: f'C{i}' for i, alt in enumerate(self.alternatives)}
        
        # Plot 1: Market shares
        for col in alt_columns:
            alt_id = int(col.split('_')[1])
            shares = results[col].values
            baseline_shares[alt_id] = shares[baseline_idx] if baseline_idx < len(shares) else shares[0]
            
            # Get label
            if alternative_names and alt_id in alternative_names:
                label = alternative_names[alt_id]
            else:
                label = f"Alternative {alt_id}"
            
            # Determine style
            is_focal = (focal_alternative is not None and alt_id == focal_alternative)
            
            ax1.plot(multipliers, shares,
                    marker='o' if not is_focal else 's',
                    linewidth=3 if is_focal else 2,
                    label=label,
                    markersize=8 if is_focal else 6,
                    alpha=1.0 if is_focal else 0.7,
                    color=colors.get(alt_id, f'C{alt_id}'))
        
        ax1.set_xlabel(f'{attribute_name} Multiplier', fontsize=12)
        ax1.set_ylabel('Market Share', fontsize=12)
        ax1.set_title('Market Share Sensitivity', fontsize=14)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=1.0, color='black', linestyle='--', alpha=0.5)
        
        # Add shaded region for common policy range
        ax1.axvspan(0.9, 1.1, alpha=0.1, color='gray')
        
        # Plot 2: Percent changes
        for col in alt_columns:
            alt_id = int(col.split('_')[1])
            shares = results[col].values
            
            if baseline_shares[alt_id] > 0:
                pct_changes = (shares / baseline_shares[alt_id] - 1) * 100
            else:
                pct_changes = np.zeros_like(shares)
            
            # Get label
            if alternative_names and alt_id in alternative_names:
                label = alternative_names[alt_id]
            else:
                label = f"Alternative {alt_id}"
            
            # Determine style
            is_focal = (focal_alternative is not None and alt_id == focal_alternative)
            
            ax2.plot(multipliers, pct_changes,
                    marker='o' if not is_focal else 's',
                    linewidth=3 if is_focal else 2,
                    label=label,
                    markersize=8 if is_focal else 6,
                    alpha=1.0 if is_focal else 0.7,
                    color=colors.get(alt_id, f'C{alt_id}'))
        
        ax2.set_xlabel(f'{attribute_name} Multiplier', fontsize=12)
        ax2.set_ylabel('% Change from Baseline', fontsize=12)
        ax2.set_title('Percent Change from Baseline', fontsize=14)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.axvline(x=1.0, color='black', linestyle='--', alpha=0.5)
        ax2.axvspan(0.9, 1.1, alpha=0.1, color='gray')
        
        plt.tight_layout()
        return fig
    
    def calculate_elasticities(self,
                              results: pd.DataFrame,
                              focal_alternative: int,
                              reference_multiplier: float = 1.0) -> Dict[float, float]:
        """
        Calculate arc elasticities from sensitivity analysis results.
        
        Parameters
        ----------
        results : pd.DataFrame
            Results from sensitivity_analysis()
        focal_alternative : int
            Alternative to calculate elasticities for
        reference_multiplier : float, default=1.0
            Reference point for elasticity calculation
            
        Returns
        -------
        Dict[float, float]
            Elasticity values for each multiplier point
            
        Examples
        --------
        >>> elasticities = analyzer.calculate_elasticities(results, focal_alternative=2)
        >>> print(f"Elasticity at 1.5x price: {elasticities[1.5]:.3f}")
        """
        elasticities = {}
        
        # Find reference index
        ref_idx = np.argmin(np.abs(results['multiplier'].values - reference_multiplier))
        ref_share = results[f'alt_{focal_alternative}'].iloc[ref_idx]
        
        for idx, row in results.iterrows():
            if idx == ref_idx:
                continue
                
            multiplier = row['multiplier']
            share = row[f'alt_{focal_alternative}']
            
            # Arc elasticity formula
            if ref_share > 0 and share > 0:
                dQ = share - ref_share
                Q_mid = (share + ref_share) / 2
                dP = multiplier - reference_multiplier  
                P_mid = (multiplier + reference_multiplier) / 2
                
                if P_mid != 0 and Q_mid != 0:
                    elasticity = (dQ / Q_mid) / (dP / P_mid)
                    elasticities[multiplier] = elasticity
        
        return elasticities


def compare_model_sensitivities(analyzers: Dict[str, PolicyAnalyzer],
                               attribute: str,
                               multipliers: List[float],
                               focal_alternative: int,
                               alternative_names: Optional[Dict[int, str]] = None,
                               use_calibrated: bool = True,
                               figsize: Tuple[float, float] = (14, 6)) -> plt.Figure:
    """
    Compare sensitivity analysis results across multiple models.
    
    This function creates a comparison plot showing how different model types
    (e.g., MNL, NL, MXL) respond to the same attribute changes.
    
    Parameters
    ----------
    analyzers : Dict[str, PolicyAnalyzer]
        Dictionary mapping model names to PolicyAnalyzer instances
    attribute : str
        Attribute to vary in the sensitivity analysis
    multipliers : List[float]
        Multiplier values to test
    focal_alternative : int
        Alternative to focus on in the comparison
    alternative_names : Dict[int, str], optional
        Human-readable names for alternatives
    use_calibrated : bool, default=True
        Whether to use calibrated parameters
    figsize : Tuple[float, float], default=(14, 6)
        Figure size
        
    Returns
    -------
    plt.Figure
        Comparison plot
        
    Examples
    --------
    >>> analyzers = {
    ...     'MNL': mnl_analyzer,
    ...     'NL': nl_analyzer,
    ...     'MXL': mxl_analyzer
    ... }
    >>> fig = compare_model_sensitivities(
    ...     analyzers, 'COST', [0.5, 1.0, 1.5, 2.0], 
    ...     focal_alternative=2
    ... )
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Colors for different models
    model_colors = {'MNL': 'C0', 'NL': 'C1', 'MXL': 'C2', 'CNL': 'C3'}
    
    # Run sensitivity analysis for each model
    all_results = {}
    for model_name, analyzer in analyzers.items():
        results = analyzer.sensitivity_analysis(
            attribute, multipliers, use_calibrated=use_calibrated, verbose=False
        )
        all_results[model_name] = results
    
    # Plot market shares
    for model_name, results in all_results.items():
        shares = results[f'alt_{focal_alternative}'].values
        ax1.plot(multipliers, shares,
                marker='o', label=model_name,
                linewidth=2.5, markersize=7,
                color=model_colors.get(model_name, f'C{len(all_results)}'))
    
    # Get alternative name
    if alternative_names and focal_alternative in alternative_names:
        alt_name = alternative_names[focal_alternative]
    else:
        alt_name = f"Alternative {focal_alternative}"
    
    ax1.set_title(f'{alt_name} Market Share', fontsize=14)
    ax1.set_xlabel(f'{attribute} Multiplier', fontsize=12)
    ax1.set_ylabel('Market Share', fontsize=12)
    ax1.legend(title='Model Type', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=1.0, color='black', linestyle='--', alpha=0.5)
    
    # Plot percent changes
    baseline_idx = np.argmin(np.abs(np.array(multipliers) - 1.0))
    
    for model_name, results in all_results.items():
        shares = results[f'alt_{focal_alternative}'].values
        baseline = shares[baseline_idx] if baseline_idx < len(shares) else shares[0]
        
        if baseline > 0:
            pct_changes = (shares / baseline - 1) * 100
        else:
            pct_changes = np.zeros_like(shares)
            
        ax2.plot(multipliers, pct_changes,
                marker='o', label=model_name,
                linewidth=2.5, markersize=7,
                color=model_colors.get(model_name, f'C{len(all_results)}'))
    
    ax2.set_title(f'{alt_name} % Change from Baseline', fontsize=14)
    ax2.set_xlabel(f'{attribute} Multiplier', fontsize=12)
    ax2.set_ylabel('% Change', fontsize=12)
    ax2.legend(title='Model Type', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.axvline(x=1.0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    return fig