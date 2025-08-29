"""
Generic sensitivity analysis utilities for DCMBench models.

This module provides tools for conducting sensitivity analyses on any model
that implements the PredictionInterface, including parameter modifications
and market share evolution visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Union, Optional, Any

from ..model_benchmarker.model_interface import PredictionInterface


class SensitivityAnalysis:
    """
    Generic sensitivity analysis for DCMBench-compatible models.
    
    This class works with any model that implements PredictionInterface,
    handles dataset modifications, and creates visualizations of market
    share evolution under different scenarios.
    """
    
    def __init__(self, model: PredictionInterface, name: Optional[str] = None):
        """
        Initialize the sensitivity analyzer.
        
        Parameters
        ----------
        model : PredictionInterface
            Model that implements the prediction interface
        name : Optional[str], default=None
            Name for the model (used in plots)
        """
        self.model = model
        self.name = name or getattr(model, 'name', 'Model')
        self.results = []
    
    def analyze_parameter(
        self,
        data: pd.DataFrame,
        parameter_name: str,
        multipliers: List[float],
        mode_filter: Optional[Dict[str, Any]] = None,
        choice_column: str = "CHOICE"
    ) -> pd.DataFrame:
        """
        Analyze sensitivity to a specific parameter.
        
        Parameters
        ----------
        data : pd.DataFrame
            Base dataset
        parameter_name : str
            Name of parameter to modify (column in data)
        multipliers : List[float]
            Multipliers to apply (e.g., [1.0, 1.1, 1.25, 1.5])
        mode_filter : Optional[Dict[str, Any]], default=None
            Filter to apply modifications only to specific modes
            Example: {'mode_column': 'MODE', 'mode_value': 'car'}
        choice_column : str, default="CHOICE"
            Name of the choice column
            
        Returns
        -------
        pd.DataFrame
            Results with columns: multiplier, parameter, market_shares
        """
        results = []
        
        for multiplier in multipliers:
            # Modify dataset
            modified_data = data.copy()
            
            if mode_filter:
                # Apply modification only to specific mode
                mode_col = mode_filter['mode_column']
                mode_val = mode_filter['mode_value']
                mask = modified_data[mode_col] == mode_val
                modified_data.loc[mask, parameter_name] = data.loc[mask, parameter_name] * multiplier
            else:
                # Apply modification to all observations
                modified_data[parameter_name] = data[parameter_name] * multiplier
            
            # Predict with modified data
            probs = self.model.predict_probabilities(modified_data)
            
            # Calculate market shares
            market_shares = {}
            for col in probs.columns:
                market_shares[str(col)] = probs[col].mean()
            
            results.append({
                'multiplier': multiplier,
                'parameter': parameter_name,
                'market_shares': market_shares,
                'model': self.name
            })
        
        return pd.DataFrame(results)
    
    def analyze_multiple_scenarios(
        self,
        data: pd.DataFrame,
        scenarios: List[Dict[str, Any]],
        choice_column: str = "CHOICE"
    ) -> Dict[str, pd.DataFrame]:
        """
        Analyze multiple sensitivity scenarios.
        
        Parameters
        ----------
        data : pd.DataFrame
            Base dataset
        scenarios : List[Dict[str, Any]]
            List of scenario definitions, each containing:
            - 'name': scenario name
            - 'parameter': parameter to modify
            - 'multipliers': list of multipliers
            - 'mode_filter': optional mode filter
        choice_column : str, default="CHOICE"
            Name of the choice column
            
        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary mapping scenario names to result DataFrames
        """
        results = {}
        
        for scenario in scenarios:
            result = self.analyze_parameter(
                data=data,
                parameter_name=scenario['parameter'],
                multipliers=scenario['multipliers'],
                mode_filter=scenario.get('mode_filter'),
                choice_column=choice_column
            )
            results[scenario['name']] = result
        
        return results
    
    def plot_evolution(
        self,
        results: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        modes_to_plot: Optional[List[str]] = None,
        title: str = "Market Share Evolution",
        xlabel: str = "Parameter Multiplier",
        ylabel: str = "Market Share (%)",
        figsize: tuple = (10, 6),
        colors: Optional[Dict[str, str]] = None,
        save_path: Optional[str] = None
    ):
        """
        Create evolution plots for specified modes.
        
        Parameters
        ----------
        results : Union[pd.DataFrame, Dict[str, pd.DataFrame]]
            Results from analyze_parameter or analyze_multiple_scenarios
        modes_to_plot : Optional[List[str]], default=None
            List of mode IDs to plot. If None, plots all modes
        title : str, default="Market Share Evolution"
            Plot title
        xlabel : str, default="Parameter Multiplier"
            X-axis label
        ylabel : str, default="Market Share (%)"
            Y-axis label
        figsize : tuple, default=(10, 6)
            Figure size
        colors : Optional[Dict[str, str]], default=None
            Dictionary mapping mode IDs to colors
        save_path : Optional[str], default=None
            Path to save the plot
        """
        plt.figure(figsize=figsize)
        
        # Handle single result DataFrame
        if isinstance(results, pd.DataFrame):
            results = {'Scenario': results}
        
        # Default colors
        if colors is None:
            colors = {
                '1': 'blue',
                '2': 'red', 
                '3': 'green',
                '4': 'orange',
                '5': 'purple'
            }
        
        # Plot each scenario
        for scenario_name, scenario_results in results.items():
            # Extract data for plotting
            multipliers = scenario_results['multiplier'].values
            
            # Get first row to determine available modes
            first_shares = scenario_results.iloc[0]['market_shares']
            available_modes = list(first_shares.keys())
            
            # Determine which modes to plot
            if modes_to_plot is None:
                modes_to_plot = available_modes
            
            # Plot each mode
            for mode in modes_to_plot:
                if mode in available_modes:
                    shares = [row['market_shares'][mode] * 100 
                             for _, row in scenario_results.iterrows()]
                    
                    color = colors.get(mode, 'black')
                    label = f"{scenario_name} - Mode {mode}" if len(results) > 1 else f"Mode {mode}"
                    
                    plt.plot(multipliers, shares, 
                            marker='o', 
                            color=color,
                            label=label,
                            linewidth=2,
                            markersize=8)
        
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.title(title, fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        
        # Set y-axis limits
        plt.ylim([0, 100])
        
        # Increase tick label sizes
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def create_comparison_table(
        self,
        results: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        baseline_multiplier: float = 1.0,
        modes_to_include: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create a comparison table showing changes from baseline.
        
        Parameters
        ----------
        results : Union[pd.DataFrame, Dict[str, pd.DataFrame]]
            Results from sensitivity analysis
        baseline_multiplier : float, default=1.0
            Multiplier value representing baseline scenario
        modes_to_include : Optional[List[str]], default=None
            Modes to include in the table
            
        Returns
        -------
        pd.DataFrame
            Comparison table with changes from baseline
        """
        # Handle single result DataFrame
        if isinstance(results, pd.DataFrame):
            results = {'Scenario': results}
        
        comparison_data = []
        
        for scenario_name, scenario_results in results.items():
            # Find baseline
            baseline_row = scenario_results[scenario_results['multiplier'] == baseline_multiplier].iloc[0]
            baseline_shares = baseline_row['market_shares']
            
            # Compare each multiplier to baseline
            for _, row in scenario_results.iterrows():
                multiplier = row['multiplier']
                shares = row['market_shares']
                
                for mode, share in shares.items():
                    if modes_to_include is None or mode in modes_to_include:
                        baseline_share = baseline_shares[mode]
                        change = share - baseline_share
                        pct_change = (change / baseline_share * 100) if baseline_share > 0 else 0
                        
                        comparison_data.append({
                            'Scenario': scenario_name,
                            'Multiplier': multiplier,
                            'Mode': mode,
                            'Market Share': share,
                            'Change from Baseline': change,
                            'Percent Change': pct_change
                        })
        
        return pd.DataFrame(comparison_data)


class ModelCalibrator:
    """
    Calibrates model alternative-specific constants to match target market shares.
    
    This class provides a dataset-agnostic way to adjust ASC parameters in discrete
    choice models to reproduce observed or target market shares.
    """
    
    @staticmethod
    def calibrate(
        model: Any,
        target_shares: Dict[int, float],
        asc_mapping: Optional[Dict[int, str]] = None,
        max_iter: int = 50,
        tolerance: float = 1e-3,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Calibrate alternative-specific constants to match target market shares.
        
        Args:
            model: A model object with the following attributes:
                - model_type: str indicating model type ('MNL', 'NL', 'MXL')
                - betas: Dict[str, float] of current parameter values
                - calculate_probabilities(modified_betas): method to compute choice probabilities
            target_shares: Dictionary mapping alternative IDs to target market shares (0-1)
            asc_mapping: Optional mapping from alternative IDs to ASC parameter names.
                        If None, will auto-detect ASC parameters.
            max_iter: Maximum number of calibration iterations
            tolerance: Convergence tolerance (sum of absolute differences)
            verbose: Whether to print progress information
            
        Returns:
            Dictionary of calibrated beta values
        """
        # Copy initial beta values
        current_betas = model.betas.copy()
        
        # Auto-detect ASC parameters if mapping not provided
        if asc_mapping is None:
            # Find all parameters starting with ASC_
            asc_names = [name for name in current_betas.keys() if name.startswith('ASC_')]
            if verbose and asc_names:
                print(f"Auto-detected ASC parameters: {', '.join(asc_names)}")
        else:
            # Use provided mapping to get ASC names
            asc_names = list(asc_mapping.values())
        
        # Get alternative numbers and sort
        alts = sorted(target_shares.keys())
        
        # Initialize tracking variables
        best_diff = float('inf')
        best_betas = current_betas.copy()
        
        if verbose:
            print(f"\nCalibrating constants for {model.model_type} model...")
            print(f"Target shares: {', '.join([f'Alt {alt}: {target_shares[alt]:.2%}' for alt in alts])}")
            if asc_names:
                print(f"Adjusting parameters: {', '.join(asc_names)}")
        
        # Choose calibration method based on model type
        if hasattr(model, 'model_type') and model.model_type == 'MXL':
            # Mixed logit models need special handling
            calibrated_betas = ModelCalibrator._calibrate_mxl(
                model, target_shares, current_betas, asc_names, asc_mapping,
                alts, max_iter, tolerance, verbose, best_diff, best_betas
            )
        else:
            # Standard calibration for MNL and NL models
            calibrated_betas = ModelCalibrator._calibrate_standard(
                model, target_shares, current_betas, asc_names, asc_mapping,
                alts, max_iter, tolerance, verbose, best_diff, best_betas
            )
        
        return calibrated_betas
    
    @staticmethod
    def _calibrate_mxl(
        model, target_shares, current_betas, asc_names, asc_mapping,
        alts, max_iter, tolerance, verbose, best_diff, best_betas
    ):
        """Calibration method for Mixed Logit models using log-ratio adjustment."""
        from tqdm import tqdm
        lambda_factor = 0.3  # Damping factor for MXL convergence
        
        for iteration in range(max_iter):
            # Get predicted probabilities
            probabilities = model.calculate_probabilities(modified_betas=current_betas)
            
            # Calculate market shares
            current_shares = {}
            for alt in alts:
                current_shares[alt] = probabilities[f'P_{alt}'].mean()
            
            # Calculate total difference
            total_diff = sum(abs(target_shares[alt] - current_shares[alt]) for alt in alts)
            
            # Track best solution
            if total_diff < best_diff:
                best_diff = total_diff
                best_betas = current_betas.copy()
            
            # Check convergence
            if total_diff < tolerance:
                if verbose:
                    print(f"Converged after {iteration+1} iterations.")
                    print(f"Final shares: {', '.join([f'Alt {alt}: {current_shares[alt]:.2%}' for alt in alts])}")
                break
            
            # Check max iterations
            if iteration == max_iter - 1:
                if verbose:
                    print(f"Reached maximum iterations ({max_iter}).")
                current_betas = best_betas.copy()
                break
            
            # Adjust ASCs using log-ratio method
            for alt in alts:
                # Find corresponding ASC parameter
                asc_name = None
                if asc_mapping:
                    asc_name = asc_mapping.get(alt)
                else:
                    # Try to find ASC for this alternative
                    for name in asc_names:
                        # This is a simple heuristic - may need adjustment
                        if str(alt) in name or name.endswith(f'_{alt}'):
                            asc_name = name
                            break
                
                if asc_name and asc_name in current_betas:
                    # Skip if fixed parameter
                    if model.betas.get(f"{asc_name}_is_fixed", False) == 1:
                        continue
                    
                    if current_shares[alt] > 0:
                        # Log-ratio adjustment with damping
                        log_ratio = np.log(target_shares[alt] / current_shares[alt])
                        adjustment = lambda_factor * log_ratio
                        current_betas[asc_name] += adjustment
            
            if verbose and (iteration % 5 == 0):
                print(f"Iteration {iteration+1}: Total diff = {total_diff:.4f}")
        
        return current_betas
    
    @staticmethod
    def _calibrate_standard(
        model, target_shares, current_betas, asc_names, asc_mapping,
        alts, max_iter, tolerance, verbose, best_diff, best_betas
    ):
        """Standard calibration method for MNL and NL models."""
        step_size = 0.5
        min_step = 0.05
        
        for iteration in range(max_iter):
            # Get predicted probabilities
            probabilities = model.calculate_probabilities(modified_betas=current_betas)
            
            # Calculate market shares
            current_shares = {}
            for alt in alts:
                current_shares[alt] = probabilities[f'P_{alt}'].mean()
            
            # Calculate differences
            diff_shares = {alt: target_shares[alt] - current_shares[alt] for alt in alts}
            total_diff = sum(abs(diff) for diff in diff_shares.values())
            
            # Track best solution
            if total_diff < best_diff:
                best_diff = total_diff
                best_betas = current_betas.copy()
            
            # Check convergence
            if total_diff < tolerance:
                if verbose:
                    print(f"Converged after {iteration+1} iterations.")
                    print(f"Final shares: {', '.join([f'Alt {alt}: {current_shares[alt]:.2%}' for alt in alts])}")
                break
            
            # Check max iterations
            if iteration == max_iter - 1:
                if verbose:
                    print(f"Reached maximum iterations ({max_iter}).")
                current_betas = best_betas.copy()
                break
            
            # Adjust ASCs based on differences
            for alt in alts:
                # Find corresponding ASC parameter
                asc_name = None
                if asc_mapping:
                    asc_name = asc_mapping.get(alt)
                else:
                    # Try to find ASC for this alternative
                    for name in asc_names:
                        if str(alt) in name or name.endswith(f'_{alt}'):
                            asc_name = name
                            break
                
                if asc_name and asc_name in current_betas:
                    # Skip if fixed parameter
                    if model.betas.get(f"{asc_name}_is_fixed", False) == 1:
                        continue
                    
                    # Linear adjustment
                    adjustment = diff_shares[alt] * step_size
                    current_betas[asc_name] += adjustment
            
            # Decrease step size
            step_size = max(step_size * 0.9, min_step)
            
            if verbose and (iteration % 5 == 0):
                print(f"Iteration {iteration+1}: Total diff = {total_diff:.4f}")
        
        return current_betas


class SensitivityAnalyzer:
    """
    Performs sensitivity analysis by varying model attributes.
    
    This class provides tools to analyze how model predictions change when
    specific attributes (like cost or time) are modified by various factors.
    """
    
    @staticmethod
    def analyze(
        model: Any,
        calibrated_betas: Dict[str, float],
        attribute_name: str,
        multipliers: List[float],
        alternatives: Optional[List[int]] = None,
        verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Run sensitivity analysis by modifying an attribute.
        
        Args:
            model: Model object with database and calculate_probabilities method
            calibrated_betas: Dictionary of calibrated parameter values
            attribute_name: Name of the attribute to modify (e.g., 'CAR_COST')
            multipliers: List of multiplier values to apply
            alternatives: Optional list of alternative IDs to track. If None, tracks all.
            verbose: Whether to print progress information
            
        Returns:
            List of dictionaries containing results for each multiplier:
                - multiplier: float
                - market_shares: Dict[int, float] for each alternative
                - probabilities: Full probability DataFrame
        """
        from tqdm import tqdm
        
        # Verify attribute exists
        if not hasattr(model, attribute_name):
            # Try to find in database columns
            if attribute_name not in model.database.data.columns:
                raise ValueError(f"Attribute {attribute_name} not found in model or database")
        
        results = []
        
        if verbose:
            model_type = getattr(model, 'model_type', 'Unknown')
            print(f"\nRunning sensitivity analysis for {attribute_name} ({model_type} model)...")
            print(f"Using multipliers: {multipliers}")
        
        # Determine which alternatives to track
        if alternatives is None:
            # Try to infer from the model's choice variable
            if hasattr(model, 'database'):
                alternatives = sorted(model.database.data['CHOICE'].unique())
            else:
                # Default to common range
                alternatives = [1, 2, 3, 4]
        
        # Run analysis for each multiplier
        for multiplier in tqdm(multipliers, disable=not verbose):
            # Create modified data
            modified_data = model.database.data.copy()
            modified_data[attribute_name] = modified_data[attribute_name] * multiplier
            
            # Calculate probabilities
            probabilities = model.calculate_probabilities(
                data=modified_data,
                modified_betas=calibrated_betas
            )
            
            # Calculate market shares
            market_shares = {}
            for alt in alternatives:
                col_name = f'P_{alt}'
                if col_name in probabilities.columns:
                    market_shares[alt] = probabilities[col_name].mean()
            
            # Store results
            results.append({
                'multiplier': multiplier,
                'market_shares': market_shares,
                'probabilities': probabilities
            })
        
        if verbose:
            SensitivityAnalyzer._print_summary(results, alternatives)
        
        return results
    
    @staticmethod
    def _print_summary(results: List[Dict], alternatives: List[int]):
        """Print a summary table of sensitivity analysis results."""
        print("\nSensitivity Analysis Results:")
        print("-" * (15 * (len(alternatives) + 1)))
        
        # Header
        header = f"{'Multiplier':<10}"
        for alt in alternatives:
            header += f" {'Alt ' + str(alt) + ' Share':<15}"
        print(header)
        print("-" * (15 * (len(alternatives) + 1)))
        
        # Data rows
        for result in results:
            row = f"{result['multiplier']:<10.2f}"
            for alt in alternatives:
                share = result['market_shares'].get(alt, 0)
                row += f" {share:<15.2%}"
            print(row)


class SensitivityPlotter:
    """
    Creates plots for sensitivity analysis results.
    
    This class provides visualization tools for comparing sensitivity analysis
    results across different model types.
    """
    
    @staticmethod
    def plot_results(
        all_results: Dict[str, List[Dict]],
        attribute_name: str,
        focal_alternative: Optional[int] = None,
        alternative_names: Optional[Dict[int, str]] = None,
        model_colors: Optional[Dict[str, str]] = None,
        model_markers: Optional[Dict[str, str]] = None,
        show_annotations: bool = True,
        figsize: tuple = (10, 8),
        dpi: int = 300
    ) -> tuple:
        """
        Create comparison plots for sensitivity analysis results.
        
        Args:
            all_results: Dictionary mapping model names to sensitivity results
            attribute_name: Name of the attribute that was varied
            focal_alternative: ID of alternative to focus on. If None, auto-selects.
            alternative_names: Optional mapping of alternative IDs to names
            model_colors: Optional custom colors for each model type
            model_markers: Optional custom markers for each model type
            show_annotations: Whether to show difference annotations
            figsize: Figure size (width, height)
            dpi: Figure resolution
            
        Returns:
            Tuple of (market_share_figure, percent_change_figure)
        """
        import matplotlib.pyplot as plt
        import matplotlib.ticker
        
        # Set default colors and markers if not provided
        if model_colors is None:
            model_colors = {
                'MNL': '#1f77b4',  # Blue
                'NL': '#ff7f0e',   # Orange
                'MXL': '#2ca02c',  # Green
            }
        
        if model_markers is None:
            model_markers = {
                'MNL': 'o',  # Circle
                'NL': 's',   # Square
                'MXL': '^',  # Triangle
            }
        
        # Update plot settings
        plt.rcParams.update({
            'font.size': 14,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
        })
        
        # Get model types and multipliers
        model_types = list(all_results.keys())
        multipliers = [result['multiplier'] for result in all_results[model_types[0]]]
        
        # Auto-select focal alternative if not specified
        if focal_alternative is None:
            # Use the alternative with largest baseline share
            baseline_idx = multipliers.index(1.0) if 1.0 in multipliers else 0
            baseline_shares = {}
            for alt in all_results[model_types[0]][baseline_idx]['market_shares']:
                baseline_shares[alt] = all_results[model_types[0]][baseline_idx]['market_shares'][alt]
            focal_alternative = max(baseline_shares, key=baseline_shares.get)
        
        # Get alternative name
        if alternative_names and focal_alternative in alternative_names:
            focal_name = alternative_names[focal_alternative]
        else:
            focal_name = f"Alternative {focal_alternative}"
        
        # Create market share plot
        market_share_fig = SensitivityPlotter._plot_market_shares(
            all_results, multipliers, focal_alternative, focal_name,
            attribute_name, model_types, model_colors, model_markers,
            show_annotations, figsize
        )
        
        # Create percent change plot
        percent_change_fig = SensitivityPlotter._plot_percent_changes(
            all_results, multipliers, focal_alternative, focal_name,
            attribute_name, model_types, model_colors, model_markers,
            show_annotations, figsize
        )
        
        return market_share_fig, percent_change_fig
    
    @staticmethod
    def _plot_market_shares(
        all_results, multipliers, focal_alternative, focal_name,
        attribute_name, model_types, model_colors, model_markers,
        show_annotations, figsize
    ):
        """Create market share vs multiplier plot."""
        import matplotlib.pyplot as plt
        import matplotlib.ticker
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract and plot data for each model
        model_shares = {}
        for model_type in model_types:
            results = all_results[model_type]
            shares = [result['market_shares'][focal_alternative] for result in results]
            model_shares[model_type] = shares
            
            # Use custom colors/markers if available, otherwise use model type as key
            color = model_colors.get(model_type, model_colors.get(list(model_colors.keys())[0]))
            marker = model_markers.get(model_type, model_markers.get(list(model_markers.keys())[0]))
            
            ax.plot(multipliers, shares,
                   color=color,
                   linestyle='-',
                   marker=marker,
                   linewidth=3,
                   markersize=10,
                   label=model_type)
        
        # Add annotations if requested and MNL/MXL models exist
        if show_annotations and 'MNL' in model_shares and 'MXL' in model_shares:
            SensitivityPlotter._add_market_share_annotations(
                ax, multipliers, model_shares, model_colors.get('MXL', '#2ca02c')
            )
        
        # Formatting
        ax.set_title(f'{focal_name} Market Share vs. {attribute_name} Multiplier', fontsize=18)
        ax.set_xlabel(f'{attribute_name} multiplier', fontsize=16)
        ax.set_ylabel('Market Share (%)', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=14)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
        ax.set_xticks(multipliers)
        
        fig.tight_layout()
        return fig
    
    @staticmethod
    def _plot_percent_changes(
        all_results, multipliers, focal_alternative, focal_name,
        attribute_name, model_types, model_colors, model_markers,
        show_annotations, figsize
    ):
        """Create percent change vs multiplier plot."""
        import matplotlib.pyplot as plt
        import matplotlib.ticker
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate and plot percent changes
        model_pct_changes = {}
        for model_type in model_types:
            results = all_results[model_type]
            shares = [result['market_shares'][focal_alternative] for result in results]
            
            # Find baseline
            baseline_idx = multipliers.index(1.0) if 1.0 in multipliers else 0
            baseline_share = shares[baseline_idx]
            
            # Calculate percent changes
            pct_changes = [(share / baseline_share - 1) * 100 for share in shares]
            model_pct_changes[model_type] = pct_changes
            
            # Plot
            color = model_colors.get(model_type, model_colors.get(list(model_colors.keys())[0]))
            marker = model_markers.get(model_type, model_markers.get(list(model_markers.keys())[0]))
            
            ax.plot(multipliers, pct_changes,
                   color=color,
                   linestyle='-',
                   marker=marker,
                   linewidth=3,
                   markersize=10,
                   label=model_type)
        
        # Add annotations if requested
        if show_annotations and 'MNL' in model_pct_changes and 'MXL' in model_pct_changes:
            SensitivityPlotter._add_percent_change_annotations(
                ax, multipliers, model_pct_changes, model_colors.get('MXL', '#2ca02c')
            )
        
        # Formatting
        ax.set_title(f'Percent Change in {focal_name} Market Share', fontsize=18)
        ax.set_xlabel(f'{attribute_name} multiplier', fontsize=16)
        ax.set_ylabel('Percent Change from Baseline (%)', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=14)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:.1f}%'))
        ax.set_xticks(multipliers)
        
        fig.tight_layout()
        return fig
    
    @staticmethod
    def _add_market_share_annotations(ax, multipliers, model_shares, mxl_color):
        """Add percentage difference annotations between MXL and MNL."""
        mnl_shares = model_shares['MNL']
        mxl_shares = model_shares['MXL']
        
        for i, (multiplier, mxl_share) in enumerate(zip(multipliers, mxl_shares)):
            if multiplier == 1.0:
                continue
            
            pct_diff = ((mxl_share - mnl_shares[i]) / mnl_shares[i]) * 100
            ax.annotate(f"{pct_diff:+.1f}%",
                       xy=(multiplier, mxl_share),
                       xytext=(5, 5),
                       textcoords='offset points',
                       fontsize=11,
                       fontweight='bold',
                       color=mxl_color)
    
    @staticmethod
    def _add_percent_change_annotations(ax, multipliers, model_pct_changes, mxl_color):
        """Add percentage point difference annotations between MXL and MNL."""
        mnl_pct = model_pct_changes['MNL']
        mxl_pct = model_pct_changes['MXL']
        
        for i, (multiplier, mxl_pct_val) in enumerate(zip(multipliers, mxl_pct)):
            if multiplier == 1.0:
                continue
            
            diff_pp = mxl_pct_val - mnl_pct[i]
            ax.annotate(f"{diff_pp:+.1f}pp",
                       xy=(multiplier, mxl_pct_val),
                       xytext=(5, 5),
                       textcoords='offset points',
                       fontsize=11,
                       fontweight='bold',
                       color=mxl_color)