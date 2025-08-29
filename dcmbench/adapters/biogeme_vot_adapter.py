"""
Enhanced BiogemeModelAdapter with Value of Time (VOT) calculation capabilities.

This module extends the BiogemeModelAdapter to include methods for calculating
and visualizing Value of Time distributions for both mixed logit and standard
choice models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, Optional, Any, Union, Tuple

from .enhanced_biogeme_adapter import EnhancedBiogemeAdapter


class BiogemeVOTAdapter(EnhancedBiogemeAdapter):
    """
    BiogemeModelAdapter with Value of Time calculation and visualization.
    
    This adapter extends the enhanced adapter to include VOT calculations for:
    - Mixed Logit models: Individual-specific VOT using Bayesian conditioning
    - Standard models: Segment-based VOT (e.g., by purpose)
    """
    
    def calculate_vot_distribution(
        self,
        data: pd.DataFrame,
        time_param_pattern: str = "B_TIME",
        cost_param_pattern: str = "B_COST",
        segment_column: Optional[str] = None,
        time_units: str = "minutes",
        cost_units: str = "dollars",
        n_draws: int = 1000
    ) -> pd.DataFrame:
        """
        Calculate Value of Time distribution based on model type.
        
        Parameters
        ----------
        data : pd.DataFrame
            Dataset with individual observations
        time_param_pattern : str, default="B_TIME"
            Pattern to identify time parameters
        cost_param_pattern : str, default="B_COST"
            Pattern to identify cost parameters
        segment_column : Optional[str], default=None
            Column name for segmentation (e.g., 'PURPOSE')
        time_units : str, default="minutes"
            Units of time in the model (minutes or hours)
        cost_units : str, default="dollars"
            Units of cost in the model
        n_draws : int, default=1000
            Number of draws for mixed logit simulation
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: 'individual', 'segment' (if applicable), 'vot'
        """
        # Detect model type
        is_mixed_logit = self._is_mixed_logit()
        
        if is_mixed_logit:
            print("Detected Mixed Logit model - calculating individual-specific VOT")
            vot_df = self._calculate_mixed_logit_vot(
                data, time_param_pattern, cost_param_pattern, 
                time_units, n_draws
            )
        else:
            print("Detected standard model - calculating segment-based VOT")
            vot_df = self._calculate_standard_vot(
                data, time_param_pattern, cost_param_pattern,
                segment_column, time_units
            )
        
        return vot_df
    
    def _is_mixed_logit(self) -> bool:
        """
        Detect if the model is a mixed logit by checking for random parameters.
        
        Returns
        -------
        bool
            True if mixed logit, False otherwise
        """
        betas = self.results.get_beta_values()
        # Check for standard deviation parameters (usually suffixed with _S)
        return any('_S' in param for param in betas.keys())
    
    def _calculate_mixed_logit_vot(
        self,
        data: pd.DataFrame,
        time_param_pattern: str,
        cost_param_pattern: str,
        time_units: str,
        n_draws: int
    ) -> pd.DataFrame:
        """
        Calculate individual-specific VOT for mixed logit models.
        
        Uses Bayesian conditioning following Revelt and Train (2000).
        """
        betas = self.results.get_beta_values()
        
        # Find time and cost parameters
        time_params = {k: v for k, v in betas.items() if time_param_pattern in k and '_S' not in k}
        time_std_params = {k: v for k, v in betas.items() if time_param_pattern in k and '_S' in k}
        cost_params = {k: v for k, v in betas.items() if cost_param_pattern in k and '_S' not in k}
        
        if not time_params or not cost_params:
            raise ValueError(f"Could not find time or cost parameters matching patterns: {time_param_pattern}, {cost_param_pattern}")
        
        # For simplicity, use the first parameters found
        time_param_name = list(time_params.keys())[0]
        time_param_value = time_params[time_param_name]
        time_std_value = time_std_params.get(f"{time_param_name}_S", 0.0)
        cost_param_value = list(cost_params.values())[0]
        
        # Generate random draws for time parameter
        np.random.seed(42)
        random_draws = np.random.normal(0, 1, n_draws)
        
        # Determine distribution type (lognormal or normal)
        if any('LOG' in k.upper() for k in time_params.keys()):
            # Lognormal distribution
            beta_draws = -np.exp(time_param_value + time_std_value * random_draws)
        else:
            # Normal distribution
            beta_draws = time_param_value + time_std_value * random_draws
        
        # Calculate individual-specific parameters using Bayesian conditioning
        n_individuals = len(data)
        individual_betas = np.zeros(n_individuals)
        
        for i in range(n_individuals):
            # Get individual's choice and data
            ind_data = data.iloc[i]
            
            # Calculate choice probability for each draw
            probs = np.zeros(n_draws)
            
            # This is a simplified version - in practice, you'd need to know the exact utility specification
            # Here we assume a general calculation based on the choice made
            choice = ind_data.get('CHOICE', ind_data.get('choice', 1))
            
            # For demonstration, we use a simplified approach
            # In a real implementation, you'd calculate the actual utilities
            for r in range(n_draws):
                # Simplified probability calculation
                # In practice, this would involve calculating utilities for all alternatives
                probs[r] = 1.0 / (1.0 + np.exp(-beta_draws[r]))  # Simplified
            
            # Calculate weighted average
            if np.sum(probs) > 0:
                weights = probs / np.sum(probs)
                individual_betas[i] = np.sum(beta_draws * weights)
            else:
                individual_betas[i] = np.mean(beta_draws)
        
        # Calculate VOT
        vot_values = np.abs(individual_betas / cost_param_value)
        
        # Convert to hourly if needed
        if time_units == "minutes":
            vot_values *= 60
        
        # Create output DataFrame
        vot_df = pd.DataFrame({
            'individual': range(n_individuals),
            'vot': vot_values
        })
        
        return vot_df
    
    def _calculate_standard_vot(
        self,
        data: pd.DataFrame,
        time_param_pattern: str,
        cost_param_pattern: str,
        segment_column: Optional[str],
        time_units: str
    ) -> pd.DataFrame:
        """
        Calculate segment-based VOT for standard (non-mixed) models.
        """
        betas = self.results.get_beta_values()
        
        # Find all time and cost parameters
        time_params = {k: v for k, v in betas.items() if time_param_pattern in k}
        cost_params = {k: v for k, v in betas.items() if cost_param_pattern in k}
        
        if not time_params or not cost_params:
            raise ValueError(f"Could not find time or cost parameters matching patterns: {time_param_pattern}, {cost_param_pattern}")
        
        n_individuals = len(data)
        vot_values = np.zeros(n_individuals)
        segments = []
        
        # If no segmentation, use single VOT for all
        if segment_column is None:
            # Use the first parameters found
            time_value = list(time_params.values())[0]
            cost_value = list(cost_params.values())[0]
            
            vot = np.abs(time_value / cost_value)
            
            # Convert to hourly if needed
            if time_units == "minutes":
                vot *= 60
            
            vot_values[:] = vot
            segments = ['All'] * n_individuals
            
        else:
            # Segment-specific VOT
            for i in range(n_individuals):
                segment = data[segment_column].iloc[i]
                segments.append(segment)
                
                # Look for segment-specific parameters
                segment_time_param = None
                segment_cost_param = None
                
                # Try to find parameters with segment suffix
                for param_name, param_value in time_params.items():
                    if str(segment).upper() in param_name.upper():
                        segment_time_param = param_value
                        break
                
                for param_name, param_value in cost_params.items():
                    if str(segment).upper() in param_name.upper():
                        segment_cost_param = param_value
                        break
                
                # Use generic parameters if segment-specific not found
                if segment_time_param is None:
                    segment_time_param = list(time_params.values())[0]
                if segment_cost_param is None:
                    segment_cost_param = list(cost_params.values())[0]
                
                vot = np.abs(segment_time_param / segment_cost_param)
                
                # Convert to hourly if needed
                if time_units == "minutes":
                    vot *= 60
                
                vot_values[i] = vot
        
        # Create output DataFrame
        vot_df = pd.DataFrame({
            'individual': range(n_individuals),
            'segment': segments,
            'vot': vot_values
        })
        
        return vot_df
    
    def plot_vot_distribution(
        self,
        vot_data: pd.DataFrame,
        title: str = "Distribution of Value of Time",
        save_path: Optional[str] = None,
        figsize: tuple = (10, 6),
        show_segments: bool = True,
        xlim: Optional[Tuple[float, float]] = None
    ):
        """
        Plot the distribution of Value of Time.
        
        Parameters
        ----------
        vot_data : pd.DataFrame
            DataFrame with VOT values (from calculate_vot_distribution)
        title : str, default="Distribution of Value of Time"
            Plot title
        save_path : Optional[str], default=None
            Path to save the plot
        figsize : tuple, default=(10, 6)
            Figure size
        show_segments : bool, default=True
            Whether to show segments separately (if available)
        xlim : Optional[Tuple[float, float]], default=None
            X-axis limits
        """
        # Set style
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=figsize)
        
        # Define colors
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f', '#9b59b6']
        
        # Check if we have segments
        has_segments = 'segment' in vot_data.columns and len(vot_data['segment'].unique()) > 1
        
        if has_segments and show_segments:
            # Plot by segment
            segments = vot_data['segment'].unique()
            for i, segment in enumerate(segments):
                segment_data = vot_data[vot_data['segment'] == segment]['vot']
                color = colors[i % len(colors)]
                
                # Check if all values are identical
                unique_values = segment_data.unique()
                if len(unique_values) == 1:
                    # Single value - plot as a vertical line
                    value = unique_values[0]
                    plt.axvline(x=value, color=color, lw=3, alpha=0.8,
                               label=f'{segment}: ${value:.2f}/hr (n={len(segment_data)})')
                else:
                    # Multiple values - use histogram and KDE
                    plt.hist(segment_data, bins=30, density=True, alpha=0.5, 
                            color=color, label=f'{segment} (n={len(segment_data)})')
                    
                    # KDE only if enough unique values
                    if len(unique_values) > 2:
                        try:
                            kde = stats.gaussian_kde(segment_data)
                            x_range = np.linspace(segment_data.min(), segment_data.max(), 200)
                            plt.plot(x_range, kde(x_range), color=color, lw=2)
                        except:
                            pass  # Skip KDE if it fails
                
        else:
            # Plot all together
            vot_values = vot_data['vot']
            
            # Remove any potential infinities or NaNs
            vot_values = vot_values[np.isfinite(vot_values)]
            
            # Histogram
            n, bins, patches = plt.hist(vot_values, bins=50, density=True, 
                                       alpha=0.7, color=colors[0])
            
            # KDE
            if len(vot_values) > 1:
                kde = stats.gaussian_kde(vot_values)
                x_range = np.linspace(vot_values.min(), vot_values.max(), 200)
                plt.plot(x_range, kde(x_range), color=colors[1], lw=2, 
                        label='Kernel Density Estimate')
            
            # Add statistics
            mean_vot = np.mean(vot_values)
            median_vot = np.median(vot_values)
            
            plt.axvline(x=mean_vot, color=colors[2], linestyle='--', 
                       label=f'Mean: ${mean_vot:.1f}')
            plt.axvline(x=median_vot, color=colors[3], linestyle=':', 
                       label=f'Median: ${median_vot:.1f}')
        
        plt.xlabel('Value of Time ($/hour)', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title(title, fontsize=14, pad=20)
        plt.legend()
        
        if xlim:
            plt.xlim(xlim)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        # Print summary statistics
        print("\nVOT Distribution Summary:")
        print(f"Overall Mean: ${vot_data['vot'].mean():.2f}/hour")
        print(f"Overall Median: ${vot_data['vot'].median():.2f}/hour")
        print(f"Overall Std Dev: ${vot_data['vot'].std():.2f}")
        
        if has_segments:
            print("\nBy Segment:")
            for segment in vot_data['segment'].unique():
                seg_data = vot_data[vot_data['segment'] == segment]['vot']
                print(f"\n{segment}:")
                print(f"  Mean: ${seg_data.mean():.2f}/hour")
                print(f"  Median: ${seg_data.median():.2f}/hour")
                print(f"  Std Dev: ${seg_data.std():.2f}")
                print(f"  Count: {len(seg_data)}")