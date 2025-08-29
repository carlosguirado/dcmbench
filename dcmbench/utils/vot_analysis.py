"""
Value of Time (VOT) Analysis Module

This module provides dataset-agnostic tools for calculating and visualizing
Value of Time from discrete choice models.

Example usage:
    from dcmbench.utils import VOTCalculator, VOTPlotter, VOTConfig
    
    # For fixed coefficient model (MNL)
    config = VOTConfig.for_swissmetro()
    calculator = FixedVOTCalculator(config.time_params, config.cost_params)
    vot = calculator.calculate_vot(model_results)
    
    # For mixed logit model
    calculator = MixedLogitVOTCalculator(config.time_params, config.cost_params)
    vot_df = calculator.calculate_vot(model_results)
    
    # Plotting
    plotter = VOTPlotter()
    plotter.plot_vot_distribution(vot_df['vot'].values, 'Mixed Logit')
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import warnings

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    warnings.warn("Matplotlib/Seaborn not available. Plotting functions will be disabled.")


class VOTCalculator(ABC):
    """Abstract base class for VOT calculation"""
    
    def __init__(self, 
                 time_param_names: List[str],
                 cost_param_names: List[str],
                 time_unit: str = 'min',
                 cost_unit: str = '$',
                 vot_unit: str = '$/hour'):
        """
        Initialize VOT calculator.
        
        Args:
            time_param_names: List of time parameter names in the model
            cost_param_names: List of cost parameter names in the model
            time_unit: Unit of time in the model ('min', 'hour', 'sec')
            cost_unit: Currency unit ('$', '€', 'CHF', etc.)
            vot_unit: Desired output unit for VOT (e.g., '$/hour')
        """
        self.time_param_names = time_param_names
        self.cost_param_names = cost_param_names
        self.time_unit = time_unit
        self.cost_unit = cost_unit
        self.vot_unit = vot_unit
    
    @abstractmethod
    def calculate_vot(self, model_results: Any) -> Union[float, np.ndarray, pd.DataFrame]:
        """Calculate VOT from model results"""
        pass
    
    def convert_vot_units(self, vot_raw: Union[float, np.ndarray], 
                         from_time: str = 'min', 
                         to_time: str = 'hour') -> Union[float, np.ndarray]:
        """Convert VOT between time units"""
        conversion_factors = {
            ('min', 'hour'): 60,
            ('hour', 'min'): 1/60,
            ('sec', 'hour'): 3600,
            ('hour', 'sec'): 1/3600,
            ('min', 'min'): 1,
            ('hour', 'hour'): 1,
            ('sec', 'sec'): 1
        }
        factor = conversion_factors.get((from_time, to_time))
        if factor is None:
            raise ValueError(f"Unknown time conversion from {from_time} to {to_time}")
        return vot_raw * factor
    
    def _get_coefficient(self, model_results: Any, param_name: str) -> float:
        """Extract coefficient value from model results"""
        # Handle different model result formats
        if hasattr(model_results, 'get_beta_values'):
            # Biogeme style
            return model_results.get_beta_values()[param_name]
        elif hasattr(model_results, 'beta'):
            # Dictionary style
            return model_results.beta[param_name]
        elif isinstance(model_results, dict):
            return model_results.get(param_name, 0.0)
        else:
            raise ValueError(f"Cannot extract coefficient from model results of type {type(model_results)}")


class FixedVOTCalculator(VOTCalculator):
    """For models with fixed coefficients (MNL, NL)"""
    
    def calculate_vot(self, model_results: Any) -> float:
        """
        Calculate single VOT value for entire population.
        
        Args:
            model_results: Model estimation results
            
        Returns:
            float: Value of time in specified units
        """
        time_coef = self._get_coefficient(model_results, self.time_param_names[0])
        cost_coef = self._get_coefficient(model_results, self.cost_param_names[0])
        
        if abs(cost_coef) < 1e-10:
            raise ValueError("Cost coefficient is too close to zero, cannot calculate VOT")
        
        # Both coefficients typically negative, so we need to handle the sign correctly
        # VOT = -(time_coef / cost_coef) when both are negative
        vot_raw = time_coef / cost_coef
        return self.convert_vot_units(vot_raw, self.time_unit, 'hour')


class MixedLogitVOTCalculator(VOTCalculator):
    """For mixed logit models with individual parameters"""
    
    def __init__(self, time_param_names: List[str], cost_param_names: List[str], 
                 distribution_type: str = 'normal', **kwargs):
        """
        Initialize mixed logit VOT calculator.
        
        Args:
            time_param_names: List of time parameter names (base parameters)
            cost_param_names: List of cost parameter names
            distribution_type: Type of distribution ('normal', 'lognormal', 'triangular')
            **kwargs: Additional arguments for parent class
        """
        super().__init__(time_param_names, cost_param_names, **kwargs)
        self.distribution_type = distribution_type.lower()
    
    def calculate_vot(self, model_results: Any, 
                     individual_params: Optional[pd.DataFrame] = None,
                     database: Optional[Any] = None,
                     model_object: Optional[Any] = None,
                     n_individuals: Optional[int] = None) -> pd.DataFrame:
        """
        Calculate individual-specific VOT values.
        
        Args:
            model_results: Model estimation results (Biogeme results or dict)
            individual_params: Pre-calculated individual parameters (optional)
            database: Biogeme database for calculating individual parameters
            model_object: The Biogeme model object (for accessing database if needed)
            n_individuals: Number of individuals for simulation (if database not available)
            
        Returns:
            pd.DataFrame: DataFrame with columns ['individual_id', 'vot']
        """
        if individual_params is None:
            # Try different methods to get individual parameters
            individual_params = self._get_individual_params(
                model_results, database, model_object, n_individuals
            )
        
        # Get cost coefficient (usually fixed)
        cost_coef = self._get_coefficient(model_results, self.cost_param_names[0])
        
        if abs(cost_coef) < 1e-10:
            raise ValueError("Cost coefficient is too close to zero, cannot calculate VOT")
        
        # Find the time parameter column
        time_param_col = self._find_time_param_column(individual_params)
        
        # Calculate VOT
        vot_raw = individual_params[time_param_col] / cost_coef
        vot_values = self.convert_vot_units(vot_raw.values, self.time_unit, 'hour')
        
        return pd.DataFrame({
            'individual_id': individual_params.index,
            'vot': vot_values
        })
    
    def _get_individual_params(self, model_results: Any, database: Any = None,
                              model_object: Any = None, n_individuals: Optional[int] = None) -> pd.DataFrame:
        """
        Get individual parameters using various methods.
        
        Tries in order:
        1. Biogeme's calculateBetaForBayesianEstimation method
        2. Direct simulation based on distribution parameters
        3. Error if no method available
        """
        # Method 1: Try Biogeme's built-in method
        if model_object is not None and hasattr(model_object, 'database'):
            database = model_object.database
        
        if database is not None:
            try:
                # Import biogeme conditionally
                import biogeme.distributions as dist
                
                # Try to use Biogeme's method for individual parameters
                # This varies by Biogeme version
                if hasattr(model_results, 'getBetaForSensitivityAnalysis'):
                    # Newer Biogeme versions
                    beta_values = model_results.getBetaForSensitivityAnalysis(
                        database.data, ['B_TIME_RND']
                    )
                    individual_params = pd.DataFrame(beta_values)
                elif hasattr(model_results, 'calculateBetaForBayesianEstimation'):
                    # Older Biogeme versions
                    individual_params = model_results.calculateBetaForBayesianEstimation(database)
                else:
                    # Try generic conditional parameters
                    individual_params = self._simulate_from_distribution(
                        model_results, database.getSampleSize()
                    )
            except Exception as e:
                warnings.warn(f"Could not calculate individual parameters from Biogeme: {e}")
                # Fall back to simulation
                n_individuals = database.getSampleSize() if hasattr(database, 'getSampleSize') else len(database.data)
                individual_params = self._simulate_from_distribution(model_results, n_individuals)
        else:
            # Method 2: Direct simulation
            if n_individuals is None:
                raise ValueError("Either database or n_individuals must be provided for simulation")
            individual_params = self._simulate_from_distribution(model_results, n_individuals)
        
        return individual_params
    
    def _simulate_from_distribution(self, model_results: Any, n_individuals: int) -> pd.DataFrame:
        """
        Simulate individual parameters based on estimated distribution parameters.
        
        Args:
            model_results: Model results containing distribution parameters
            n_individuals: Number of individuals to simulate
            
        Returns:
            pd.DataFrame: Simulated individual parameters
        """
        import numpy as np
        
        # Get base parameter name
        base_param = self.time_param_names[0]  # e.g., 'B_TIME'
        
        # Try to find standard deviation parameter
        std_param_names = [f'{base_param}_S', f'{base_param}_STD', f'{base_param}_SIGMA']
        std_param = None
        std_value = None
        
        for param_name in std_param_names:
            try:
                std_value = self._get_coefficient(model_results, param_name)
                std_param = param_name
                break
            except:
                continue
        
        if std_value is None:
            raise ValueError(f"Could not find standard deviation parameter for {base_param}")
        
        # Get mean value
        mean_value = self._get_coefficient(model_results, base_param)
        
        # Simulate based on distribution type
        np.random.seed(42)  # For reproducibility
        
        if self.distribution_type == 'normal':
            # Normal distribution
            individual_values = np.random.normal(mean_value, abs(std_value), n_individuals)
            
        elif self.distribution_type == 'lognormal':
            # Lognormal distribution - parameters are in log space
            # Individual values are -exp(mean + std * normal_draw)
            normal_draws = np.random.normal(0, 1, n_individuals)
            individual_values = -np.exp(mean_value + abs(std_value) * normal_draws)
            
        elif self.distribution_type == 'triangular':
            # Triangular distribution (simplified - using uniform between mean ± std)
            individual_values = np.random.triangular(
                mean_value - abs(std_value),
                mean_value,
                mean_value + abs(std_value),
                n_individuals
            )
        else:
            raise ValueError(f"Unknown distribution type: {self.distribution_type}")
        
        # Create DataFrame with appropriate column name
        # Try to match Biogeme's naming convention
        column_name = 'individual_' + base_param.lower()
        if 'time' in base_param.lower():
            column_name = 'individual_time_param'
        
        return pd.DataFrame({
            column_name: individual_values
        }, index=range(n_individuals))
    
    def _find_time_param_column(self, individual_params: pd.DataFrame) -> str:
        """Find the column containing individual time parameters."""
        # First try exact match
        time_param_col = self.time_param_names[0]
        if time_param_col in individual_params.columns:
            return time_param_col
        
        # Try common variations
        variations = [
            'individual_time_param',
            'individual_time_parameter',
            f'individual_{time_param_col.lower()}',
            'B_TIME_RND',
            'b_time_rnd'
        ]
        
        for var in variations:
            if var in individual_params.columns:
                return var
        
        # Try to find any column with 'time' in it
        for col in individual_params.columns:
            if 'time' in col.lower():
                return col
        
        # Last resort - use first column
        if len(individual_params.columns) == 1:
            warnings.warn(f"Using first column '{individual_params.columns[0]}' as time parameter")
            return individual_params.columns[0]
        
        raise ValueError(f"Time parameter column not found. Available columns: {list(individual_params.columns)}")


class SegmentedVOTCalculator(VOTCalculator):
    """For models with segment-specific coefficients"""
    
    def __init__(self, segment_var: str, segment_mapping: Dict[str, Tuple[str, str]], **kwargs):
        """
        Initialize segmented VOT calculator.
        
        Args:
            segment_var: Name of the segmentation variable
            segment_mapping: Dict mapping segment values to (time_param, cost_param) tuples
            **kwargs: Additional arguments for parent class
        """
        # Extract all unique time and cost parameters
        all_time_params = []
        all_cost_params = []
        for time_param, cost_param in segment_mapping.values():
            if time_param not in all_time_params:
                all_time_params.append(time_param)
            if cost_param not in all_cost_params:
                all_cost_params.append(cost_param)
        
        super().__init__(all_time_params, all_cost_params, **kwargs)
        self.segment_var = segment_var
        self.segment_mapping = segment_mapping
    
    def calculate_vot(self, model_results: Any, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate segment-specific VOT values.
        
        Args:
            model_results: Model estimation results
            data: DataFrame containing segment variable
            
        Returns:
            pd.DataFrame: DataFrame with columns ['vot', segment_var]
        """
        vot_by_segment = {}
        
        for segment, (time_param, cost_param) in self.segment_mapping.items():
            time_coef = self._get_coefficient(model_results, time_param)
            cost_coef = self._get_coefficient(model_results, cost_param)
            
            if abs(cost_coef) < 1e-10:
                warnings.warn(f"Cost coefficient for segment {segment} is too close to zero")
                vot_by_segment[segment] = np.nan
            else:
                vot_raw = time_coef / cost_coef
                vot_by_segment[segment] = self.convert_vot_units(vot_raw, self.time_unit, 'hour')
        
        # Map to individuals
        result_df = data[[self.segment_var]].copy()
        result_df['vot'] = result_df[self.segment_var].map(vot_by_segment)
        
        return result_df


@dataclass
class VOTConfig:
    """Configuration for VOT analysis"""
    time_params: List[str]
    cost_params: List[str]
    time_unit: str = 'min'
    cost_unit: str = '$'
    mode_mapping: Optional[Dict[int, str]] = None
    segment_variable: Optional[str] = None
    segment_mapping: Optional[Dict[str, Tuple[str, str]]] = None
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'VOTConfig':
        """Create VOTConfig from dictionary"""
        return cls(**config_dict)
    
    @classmethod
    def for_modecanada(cls) -> 'VOTConfig':
        """Preset configuration for ModeCanada dataset"""
        return cls(
            time_params=['B_TIME'],
            cost_params=['B_COST'],
            time_unit='min',
            cost_unit='$',
            mode_mapping={1: 'Train', 2: 'Car', 3: 'Bus', 4: 'Air'}
        )
    
    @classmethod
    def for_swissmetro(cls) -> 'VOTConfig':
        """Preset configuration for SwissMetro dataset"""
        return cls(
            time_params=['B_TIME'],
            cost_params=['B_COST'],
            time_unit='min',
            cost_unit='CHF',
            mode_mapping={1: 'Train', 2: 'Swissmetro', 3: 'Car'}
        )
    
    @classmethod
    def for_ltds(cls) -> 'VOTConfig':
        """Preset configuration for LTDS dataset"""
        return cls(
            time_params=['B_TIME'],
            cost_params=['B_COST'],
            time_unit='min',
            cost_unit='£',
            mode_mapping={1: 'Walk', 2: 'Cycle', 3: 'PT', 4: 'Drive'}
        )


class VOTPlotter:
    """Generic VOT plotting functionality"""
    
    def __init__(self, figsize: Tuple[int, int] = (10, 6), style: str = 'whitegrid'):
        """
        Initialize VOT plotter.
        
        Args:
            figsize: Figure size for plots
            style: Seaborn style ('whitegrid', 'darkgrid', 'white', 'dark', 'ticks')
        """
        if not HAS_PLOTTING:
            raise ImportError("Plotting libraries not available. Install matplotlib and seaborn.")
        
        self.figsize = figsize
        self.style = style
        sns.set_theme(style=style)
    
    def plot_single_vot(self, vot_value: float, 
                       model_name: str,
                       ax: Optional[Any] = None) -> Any:
        """
        Plot single VOT value (for fixed coefficient models).
        
        Args:
            vot_value: Value of time
            model_name: Name of the model
            ax: Matplotlib axes (optional)
            
        Returns:
            Matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create a simple bar chart for single value
        ax.bar([model_name], [vot_value], color='steelblue', alpha=0.8)
        ax.set_ylabel('Value of Time ($/hour)')
        ax.set_title(f'Value of Time - {model_name}')
        
        # Add value label on bar
        ax.text(0, vot_value + vot_value*0.02, f'${vot_value:.2f}', 
                ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylim(0, vot_value * 1.2)
        
        return ax
    
    def plot_vot_distribution(self, vot_values: np.ndarray,
                            model_name: str,
                            bins: int = 30,
                            ax: Optional[Any] = None,
                            clean_outliers: bool = True,
                            show_statistics: bool = True) -> Any:
        """
        Plot VOT distribution (for mixed logit or segmented models).
        
        Args:
            vot_values: Array of VOT values
            model_name: Name of the model
            bins: Number of histogram bins
            ax: Matplotlib axes (optional)
            clean_outliers: Whether to remove outliers (>3 std)
            show_statistics: Whether to show mean/median lines
            
        Returns:
            Matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        # Clean outliers if requested
        if clean_outliers:
            mean_vot = np.mean(vot_values)
            std_vot = np.std(vot_values)
            vot_clean = vot_values[(vot_values > mean_vot - 3*std_vot) & 
                                  (vot_values < mean_vot + 3*std_vot)]
            if len(vot_clean) < len(vot_values):
                print(f"Removed {len(vot_values) - len(vot_clean)} outliers")
        else:
            vot_clean = vot_values
        
        # Histogram
        n, bins_edges, patches = ax.hist(vot_clean, bins=bins, density=True, 
                                         alpha=0.7, color='skyblue', 
                                         edgecolor='black', linewidth=0.5)
        
        # KDE
        try:
            kde = stats.gaussian_kde(vot_clean)
            x_range = np.linspace(vot_clean.min(), vot_clean.max(), 200)
            ax.plot(x_range, kde(x_range), 'r-', lw=2, label='KDE')
        except:
            warnings.warn("Could not compute KDE")
        
        # Statistics
        if show_statistics:
            mean_vot = np.mean(vot_clean)
            median_vot = np.median(vot_clean)
            ax.axvline(mean_vot, color='green', linestyle='--', 
                      label=f'Mean: ${mean_vot:.2f}', alpha=0.8)
            ax.axvline(median_vot, color='orange', linestyle=':', 
                      label=f'Median: ${median_vot:.2f}', alpha=0.8)
        
        ax.set_xlabel('Value of Time ($/hour)')
        ax.set_ylabel('Density')
        ax.set_title(f'VOT Distribution - {model_name}')
        if show_statistics or 'kde' in locals():
            ax.legend()
        
        return ax
    
    def plot_vot_comparison(self, vot_data: Dict[str, Union[float, np.ndarray]],
                           title: str = "VOT Comparison Across Models",
                           plot_type: str = 'violin') -> Any:
        """
        Compare VOT across multiple models.
        
        Args:
            vot_data: Dict mapping model names to VOT values (float or array)
            title: Plot title
            plot_type: Type of plot ('violin', 'box', 'combined')
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(max(10, len(vot_data)*2), 8))
        
        positions = []
        labels = []
        all_data = []
        
        for i, (model_name, vot) in enumerate(vot_data.items()):
            positions.append(i)
            labels.append(model_name)
            
            if isinstance(vot, (int, float)):
                # Single value - create small distribution for visualization
                all_data.append([vot])
                ax.scatter([i], [vot], s=200, c='red', zorder=10, 
                          marker='D', edgecolor='black', linewidth=2)
                ax.text(i, vot + 2, f'${vot:.2f}', ha='center', va='bottom',
                       fontweight='bold', fontsize=10)
            else:
                # Distribution
                # Clean outliers
                mean_vot = np.mean(vot)
                std_vot = np.std(vot)
                vot_clean = vot[(vot > mean_vot - 3*std_vot) & 
                               (vot < mean_vot + 3*std_vot)]
                all_data.append(vot_clean)
        
        # Create plots based on type
        if plot_type == 'violin':
            parts = ax.violinplot(all_data, positions=positions, widths=0.7,
                                 showmeans=True, showmedians=True)
            for pc in parts['bodies']:
                pc.set_facecolor('skyblue')
                pc.set_alpha(0.7)
                pc.set_edgecolor('black')
        elif plot_type == 'box':
            bp = ax.boxplot(all_data, positions=positions, widths=0.5,
                           patch_artist=True, showmeans=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.8)
        else:  # combined
            # Violin plot with box plot overlay
            parts = ax.violinplot(all_data, positions=positions, widths=0.7,
                                 showmeans=False, showmedians=False)
            for pc in parts['bodies']:
                pc.set_facecolor('skyblue')
                pc.set_alpha(0.5)
            bp = ax.boxplot(all_data, positions=positions, widths=0.3,
                           patch_artist=True, showmeans=True)
            for patch in bp['boxes']:
                patch.set_facecolor('white')
                patch.set_alpha(0.8)
        
        # Add mean values as text
        for i, data in enumerate(all_data):
            if len(data) > 1:
                mean_val = np.mean(data)
                ax.text(i, ax.get_ylim()[1]*0.95, f'μ=${mean_val:.1f}', 
                       ha='center', va='top', fontsize=9)
        
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45 if len(labels) > 5 else 0)
        ax.set_ylabel('Value of Time ($/hour)')
        ax.set_title(title)
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_vot_by_segment(self, vot_df: pd.DataFrame,
                           segment_col: str,
                           vot_col: str = 'vot',
                           title: Optional[str] = None,
                           plot_type: str = 'bar') -> Any:
        """
        Plot VOT distribution by segments.
        
        Args:
            vot_df: DataFrame with VOT and segment columns
            segment_col: Name of segment column
            vot_col: Name of VOT column
            title: Plot title (optional)
            plot_type: Type of plot ('bar', 'violin', 'box')
            
        Returns:
            Matplotlib figure object
        """
        if title is None:
            title = f"VOT by {segment_col}"
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        segments = sorted(vot_df[segment_col].unique())
        
        if plot_type == 'bar':
            # Calculate mean VOT per segment
            segment_means = vot_df.groupby(segment_col)[vot_col].mean()
            segment_stds = vot_df.groupby(segment_col)[vot_col].std()
            
            x_pos = np.arange(len(segments))
            bars = ax.bar(x_pos, [segment_means[s] for s in segments],
                          yerr=[segment_stds[s] for s in segments],
                          capsize=5, alpha=0.8, color='steelblue')
            
            # Add value labels
            for i, (seg, bar) in enumerate(zip(segments, bars)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'${height:.2f}', ha='center', va='bottom')
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(segments)
            ax.set_ylabel('Mean Value of Time ($/hour)')
            
        else:  # violin or box
            data_by_segment = [vot_df[vot_df[segment_col] == seg][vot_col].values 
                              for seg in segments]
            
            if plot_type == 'violin':
                parts = ax.violinplot(data_by_segment, showmeans=True, showmedians=True)
                for pc in parts['bodies']:
                    pc.set_facecolor('skyblue')
                    pc.set_alpha(0.7)
            else:  # box
                bp = ax.boxplot(data_by_segment, patch_artist=True, showmeans=True)
                for patch in bp['boxes']:
                    patch.set_facecolor('lightblue')
                    patch.set_alpha(0.8)
            
            ax.set_xticklabels(segments)
            ax.set_ylabel('Value of Time ($/hour)')
        
        ax.set_xlabel(segment_col)
        ax.set_title(title)
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig


# Convenience functions
def calculate_vot(model_results: Any, 
                 model_type: str = 'fixed',
                 config: Optional[VOTConfig] = None,
                 distribution_type: str = 'normal',
                 **kwargs) -> Union[float, pd.DataFrame]:
    """
    Convenience function to calculate VOT based on model type.
    
    Args:
        model_results: Model estimation results
        model_type: Type of model ('fixed', 'mixed', 'segmented')
        config: VOT configuration (uses defaults if not provided)
        distribution_type: For mixed logit models ('normal', 'lognormal', 'triangular')
        **kwargs: Additional arguments for specific calculators
            - database: Biogeme database (for mixed logit)
            - model_object: Biogeme model object (for mixed logit)
            - n_individuals: Number of individuals for simulation (for mixed logit)
            - data: DataFrame with segment information (for segmented)
        
    Returns:
        float or pd.DataFrame: VOT value(s)
    """
    if config is None:
        config = VOTConfig(time_params=['B_TIME'], cost_params=['B_COST'])
    
    if model_type == 'fixed':
        calculator = FixedVOTCalculator(
            config.time_params, config.cost_params,
            config.time_unit, config.cost_unit
        )
        return calculator.calculate_vot(model_results)
    
    elif model_type == 'mixed':
        calculator = MixedLogitVOTCalculator(
            config.time_params, config.cost_params,
            distribution_type=distribution_type,
            time_unit=config.time_unit,
            cost_unit=config.cost_unit
        )
        return calculator.calculate_vot(model_results, **kwargs)
    
    elif model_type == 'segmented':
        if config.segment_variable is None or config.segment_mapping is None:
            raise ValueError("Segment variable and mapping must be provided for segmented models")
        calculator = SegmentedVOTCalculator(
            config.segment_variable, config.segment_mapping,
            time_param_names=config.time_params,
            cost_param_names=config.cost_params,
            time_unit=config.time_unit,
            cost_unit=config.cost_unit
        )
        return calculator.calculate_vot(model_results, **kwargs)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def plot_vot(vot_data: Union[float, np.ndarray, pd.DataFrame],
            model_name: str = "Model",
            plot_type: str = 'auto',
            **kwargs) -> Any:
    """
    Convenience function to plot VOT data.
    
    Args:
        vot_data: VOT value(s) to plot
        model_name: Name of the model
        plot_type: Type of plot ('auto', 'single', 'distribution')
        **kwargs: Additional arguments for plotter
        
    Returns:
        Matplotlib axes or figure object
    """
    plotter = VOTPlotter()
    
    if plot_type == 'auto':
        if isinstance(vot_data, (int, float)):
            plot_type = 'single'
        else:
            plot_type = 'distribution'
    
    if plot_type == 'single':
        return plotter.plot_single_vot(float(vot_data), model_name)
    else:
        if isinstance(vot_data, pd.DataFrame):
            vot_values = vot_data['vot'].values
        else:
            vot_values = np.array(vot_data)
        return plotter.plot_vot_distribution(vot_values, model_name, **kwargs)