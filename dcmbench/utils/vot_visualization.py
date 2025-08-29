"""
Value of Time (VOT) Visualization Utilities for DCMBench

This module provides standardized visualization functions for VOT analysis,
including distribution comparisons across model types and segmentation analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Union, Optional, Tuple, Any
try:
    from .vot_systematic import SegmentVOT
except ImportError:
    SegmentVOT = None


class VOTVisualization:
    """Utilities for visualizing Value of Time distributions and comparisons."""
    
    @staticmethod
    def create_density_comparison(
        vot_dict: Dict[str, Union[float, np.ndarray]], 
        title: str = "Value of Time Distribution Comparison",
        xlabel: str = "Value of Time ($/hour)",
        output_file: Optional[str] = None,
        xlim: Optional[Tuple[float, float]] = None,
        figsize: Tuple[float, float] = (12, 8),
        show_legend: bool = True,
        colors: Optional[List] = None,
        discrete_groups: Optional[Dict[str, Dict[str, float]]] = None
    ) -> plt.Figure:
        """
        Create density plot comparing VOT distributions across different models.
        
        Parameters
        ----------
        vot_dict : dict
            Dictionary mapping model names to VOT values (single value or array)
        title : str
            Plot title
        xlabel : str
            X-axis label
        output_file : str, optional
            Path to save the figure
        xlim : tuple, optional
            X-axis limits (min, max)
        figsize : tuple
            Figure size (width, height)
        show_legend : bool
            Whether to show legend
        colors : list, optional
            List of colors to use for different models
            
        Returns
        -------
        fig : matplotlib.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Use default colors if not provided
        if colors is None:
            colors = plt.cm.Set1(np.linspace(0, 1, len(vot_dict)))
        
        # Track min/max for auto xlim if not specified
        all_values = []
        
        # Handle discrete groups separately (bar plots)
        if discrete_groups:
            for model_name, group_vots in discrete_groups.items():
                bar_positions = []
                bar_heights = []
                bar_labels = []
                for group_name, (vot_value, proportion) in group_vots.items():
                    bar_positions.append(vot_value)
                    bar_heights.append(proportion)  # Use true proportion as density (0.4-0.5)
                    bar_labels.append(f'{group_name}: ${vot_value:.1f}/hr ({proportion:.1%})')
                    all_values.append(vot_value)
                
                # Plot as bars
                ax.bar(bar_positions, bar_heights, width=3, alpha=0.6, 
                      label=f'{model_name} (discrete groups)')
                
                # Add text labels at fixed position since bars extend way beyond ylim
                for i, (pos, height, label) in enumerate(zip(bar_positions, bar_heights, bar_labels)):
                    # Shift labels to avoid overlap - alternate left and right, and vertically
                    if i == 0:  # First label
                        label_x = pos + 20  # Shift right
                        label_y = 0.013  # Keep at top position
                    else:  # Second label
                        label_x = pos + 30  # Shift further right
                        label_y = 0.011  # Lower position to avoid overlap
                    
                    # Place labels with arrows pointing to bars
                    ax.annotate(label, 
                               xy=(pos, 0.010),  # Arrow points to bar
                               xytext=(label_x, label_y),  # Label position
                               ha='center', va='top',
                               fontsize=9,
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9),
                               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', 
                                             color='black', alpha=0.7, lw=1))
        
        for i, (model_name, vot_values) in enumerate(vot_dict.items()):
            # Convert to array if single value
            if isinstance(vot_values, (int, float)):
                vot_array = np.array([vot_values])
            else:
                vot_array = np.asarray(vot_values)
            
            # Remove NaN and infinite values
            vot_clean = vot_array[np.isfinite(vot_array)]
            
            if len(vot_clean) == 0:
                continue
                
            # For distributions with multiple values
            if len(vot_clean) > 1 and np.std(vot_clean) > 0:
                # Remove extreme outliers (beyond 3 sigma)
                mean_vot = np.mean(vot_clean)
                std_vot = np.std(vot_clean)
                vot_filtered = vot_clean[
                    (vot_clean > mean_vot - 3*std_vot) & 
                    (vot_clean < mean_vot + 3*std_vot)
                ]
                
                # Track values for xlim
                all_values.extend(vot_filtered.tolist())
                
                # Create KDE plot
                try:
                    kde = stats.gaussian_kde(vot_filtered)
                    x_range = np.linspace(
                        vot_filtered.min() - std_vot/2, 
                        vot_filtered.max() + std_vot/2, 
                        300
                    )
                    ax.plot(x_range, kde(x_range), color=colors[i], lw=2.5,
                           label=f'{model_name} (μ=${mean_vot:.1f}, σ=${std_vot:.1f})')
                except:
                    # Fallback to histogram if KDE fails
                    ax.hist(vot_filtered, bins=30, alpha=0.5, density=True, 
                           color=colors[i], label=f'{model_name} (μ=${mean_vot:.1f})')
            else:
                # For fixed/single VOT values, plot as vertical line
                vot_value = vot_clean[0]
                all_values.append(vot_value)
                ax.axvline(x=vot_value, color=colors[i], lw=2.5, linestyle='--',
                          label=f'{model_name} (${vot_value:.1f}/hr)')
        
        # Set xlim
        if xlim is not None:
            ax.set_xlim(xlim)
        elif all_values:
            # Auto set xlim based on data
            min_val = min(0, min(all_values) - 10)
            max_val = max(all_values) + 10
            ax.set_xlim(min_val, max_val)
        
        # Set ylim to clip the chart and show MXL distributions
        ax.set_ylim(0, 0.015)
        
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel('Density', fontsize=14)
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        if show_legend:
            ax.legend(fontsize=11, frameon=True, fancybox=True, shadow=True, loc='best')
        
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if output_file:
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Figure saved as {output_file}")
        
        return fig
    
    @staticmethod
    def plot_vot_comparison_mixed(
        continuous_vot: Dict[str, np.ndarray],
        discrete_segments: Optional[Dict[str, Union[Dict, List]]] = None,
        title: str = "Value of Time Distribution Comparison",
        xlabel: str = "Value of Time ($/hour)",
        output_file: Optional[str] = None,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = (0, 0.02),
        figsize: Tuple[float, float] = (14, 8),
        show_legend: bool = True,
        bar_width: float = 3.0,
        bar_alpha: float = 0.6,
        show_segment_labels: bool = True,
        segment_style: str = 'bar'  # 'bar' or 'line'
    ) -> plt.Figure:
        """
        Create a mixed visualization with continuous distributions and discrete segments.
        
        This is the proper DCMBench way to visualize both MXL distributions and
        systematic heterogeneity segments on the same plot.
        
        Parameters
        ----------
        continuous_vot : dict
            Dictionary mapping model names to VOT arrays or single values
            e.g., {'MNL': 35.5, 'MXL Normal': array([...]), 'MXL Lognormal': array([...])}
        discrete_segments : dict, optional
            Dictionary mapping model names to segment dictionaries or SegmentVOT objects
            e.g., {'MNL Income': {'Low Income': SegmentVOT(...), 'High Income': SegmentVOT(...)}}
        title : str
            Plot title
        xlabel : str
            X-axis label
        output_file : str, optional
            Path to save the figure
        xlim : tuple, optional
            X-axis limits
        ylim : tuple, optional
            Y-axis limits (default: (0, 0.02) to show MXL distributions properly)
        figsize : tuple
            Figure size
        show_legend : bool
            Whether to show legend
        bar_width : float
            Width of bars for discrete segments
        bar_alpha : float
            Transparency of bars
        show_segment_labels : bool
            Whether to show labels on discrete segment bars
        segment_style : str
            'bar' for bars or 'line' for vertical lines
            
        Returns
        -------
        fig : matplotlib.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Color palette
        colors = plt.cm.Set1(np.linspace(0, 1, len(continuous_vot) + (len(discrete_segments) if discrete_segments else 0)))
        color_idx = 0
        
        # Track all values for xlim
        all_values = []
        
        # Plot discrete segments first (bars in background)
        if discrete_segments:
            for model_name, segments in discrete_segments.items():
                model_color = colors[color_idx]
                color_idx += 1
                
                bar_positions = []
                bar_heights = []
                bar_labels = []
                
                # Handle both SegmentVOT objects and dictionaries
                if isinstance(segments, dict):
                    for segment_name, segment_info in segments.items():
                        if SegmentVOT and isinstance(segment_info, SegmentVOT):
                            # SegmentVOT object from vot_systematic
                            vot_value = segment_info.vot_value
                            proportion = segment_info.population_proportion
                            label = f'{segment_info.name}: ${vot_value:.1f}/hr ({proportion:.1%})'
                        elif isinstance(segment_info, tuple) and len(segment_info) == 2:
                            # Tuple format (vot_value, proportion)
                            vot_value, proportion = segment_info
                            label = f'{segment_name}: ${vot_value:.1f}/hr ({proportion:.1%})'
                        else:
                            # Simple value
                            vot_value = float(segment_info)
                            proportion = 0.5  # Default height
                            label = f'{segment_name}: ${vot_value:.1f}/hr'
                        
                        bar_positions.append(vot_value)
                        # Make bars tall to show they represent large population segments
                        # They will extend beyond the y-limit to show ~50% of population
                        bar_heights.append(proportion * 0.045)  # This makes ~50% population = 0.0225 height
                        bar_labels.append(label)
                        all_values.append(vot_value)
                
                # Plot as bars or lines based on style
                if segment_style == 'line':
                    # Plot as vertical lines like homogeneous MNL
                    for i, (pos, label) in enumerate(zip(bar_positions, bar_labels)):
                        linestyle = '-' if i == 0 else '--'  # Different line styles for visibility
                        ax.axvline(x=pos, color=model_color, lw=2.0, linestyle=linestyle,
                                  alpha=0.8, label=label if i == 0 else None)
                        
                        if show_segment_labels:
                            # Add label at top of plot
                            parts = label.split(':')
                            segment_name = parts[0]
                            ax.text(pos, ax.get_ylim()[1] * 0.95, segment_name,
                                   ha='center', va='top', fontsize=9, rotation=0, 
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
                else:
                    # Plot as bars (default) - use light blue for visibility
                    bars = ax.bar(bar_positions, bar_heights, width=bar_width, 
                                 alpha=bar_alpha, color='lightblue', edgecolor='darkblue',
                                 label=f'{model_name} (segments)')
                    
                    # Add text labels if requested
                    if show_segment_labels:
                        for pos, height, label in zip(bar_positions, bar_heights, bar_labels):
                            # Extract segment name and value info
                            parts = label.split(':')
                            segment_name = parts[0]
                            value_info = parts[1].strip() if len(parts) > 1 else ''
                            
                            # Place labels at fixed position near top of plot since bars extend beyond
                            ax.text(pos, 0.018, segment_name, 
                                   ha='center', va='top', fontsize=9, rotation=0, fontweight='bold',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                            # Place value info below segment name
                            ax.text(pos, 0.015, value_info, 
                                   ha='center', va='top', fontsize=8, rotation=0)
        
        # Plot continuous distributions
        for model_name, vot_values in continuous_vot.items():
            model_color = colors[color_idx]
            color_idx += 1
            
            # Convert to array
            if isinstance(vot_values, (int, float)):
                vot_array = np.array([vot_values])
            else:
                vot_array = np.asarray(vot_values)
            
            # Remove invalid values
            vot_clean = vot_array[np.isfinite(vot_array)]
            if len(vot_clean) == 0:
                continue
            
            # Plot based on data type
            if len(vot_clean) > 1 and np.std(vot_clean) > 0:
                # Continuous distribution
                mean_vot = np.mean(vot_clean)
                std_vot = np.std(vot_clean)
                
                # Remove outliers
                vot_filtered = vot_clean[
                    (vot_clean > mean_vot - 3*std_vot) & 
                    (vot_clean < mean_vot + 3*std_vot)
                ]
                all_values.extend(vot_filtered.tolist())
                
                # Create KDE
                try:
                    kde = stats.gaussian_kde(vot_filtered)
                    x_range = np.linspace(
                        vot_filtered.min() - std_vot/2,
                        vot_filtered.max() + std_vot/2,
                        300
                    )
                    ax.plot(x_range, kde(x_range), color=model_color, lw=2.5,
                           label=f'{model_name} (μ=${mean_vot:.1f}, σ=${std_vot:.1f})')
                except:
                    # Fallback to histogram
                    ax.hist(vot_filtered, bins=30, alpha=0.5, density=True,
                           color=model_color, label=f'{model_name} (μ=${mean_vot:.1f})')
            else:
                # Single value - plot as vertical line
                vot_value = vot_clean[0]
                all_values.append(vot_value)
                ax.axvline(x=vot_value, color=model_color, lw=2.5, linestyle='--',
                          label=f'{model_name} (${vot_value:.1f}/hr)')
        
        # Set axis properties
        if xlim:
            ax.set_xlim(xlim)
        elif all_values:
            min_val = min(0, min(all_values) - 10)
            max_val = max(all_values) + 10
            ax.set_xlim(min_val, max_val)
        
        # Set y-axis limit to show MXL distributions properly
        if ylim:
            ax.set_ylim(ylim)
        
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel('Density', fontsize=14)
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        if show_legend:
            ax.legend(fontsize=10, frameon=True, fancybox=True, shadow=True, loc='best')
        
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if output_file:
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Figure saved as {output_file}")
        
        return fig
    
    @staticmethod
    def create_segmented_vot_plot(
        vot_df: pd.DataFrame,
        segment_col: str = 'mode',
        vot_col: str = 'vot',
        model_name: str = '',
        output_file: Optional[str] = None,
        figsize: Tuple[float, float] = (10, 6),
        plot_type: str = 'violin',
        show_means: bool = True
    ) -> plt.Figure:
        """
        Create VOT distribution plot segmented by a categorical variable.
        
        Parameters
        ----------
        vot_df : pd.DataFrame
            DataFrame containing VOT values and segmentation variable
        segment_col : str
            Column name for segmentation (e.g., 'mode', 'income_group')
        vot_col : str
            Column name containing VOT values
        model_name : str
            Name of the model for title
        output_file : str, optional
            Path to save the figure
        figsize : tuple
            Figure size
        plot_type : str
            Type of plot ('violin', 'box', 'strip')
        show_means : bool
            Whether to annotate mean values
            
        Returns
        -------
        fig : matplotlib.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Filter out infinite and NaN values
        df_clean = vot_df[[segment_col, vot_col]].copy()
        df_clean = df_clean[np.isfinite(df_clean[vot_col])]
        
        # Create the plot based on type
        if plot_type == 'violin':
            sns.violinplot(data=df_clean, x=segment_col, y=vot_col, 
                          palette='Set2', ax=ax)
        elif plot_type == 'box':
            sns.boxplot(data=df_clean, x=segment_col, y=vot_col, 
                       palette='Set2', ax=ax)
        elif plot_type == 'strip':
            sns.stripplot(data=df_clean, x=segment_col, y=vot_col, 
                         palette='Set2', alpha=0.5, ax=ax)
        
        # Add mean annotations if requested
        if show_means:
            means = df_clean.groupby(segment_col)[vot_col].mean()
            for i, (seg, mean_val) in enumerate(means.items()):
                ax.text(i, mean_val, f'${mean_val:.1f}', 
                       ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Capitalize first letter of segment names for display
        segment_name = segment_col.replace('_', ' ').title()
        ax.set_xlabel(segment_name, fontsize=14)
        ax.set_ylabel('Value of Time ($/hour)', fontsize=14)
        
        title = f'Value of Time by {segment_name}'
        if model_name:
            title += f' - {model_name}'
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        ax.grid(True, axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if output_file:
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Figure saved as {output_file}")
        
        return fig
    
    @staticmethod
    def create_vot_heterogeneity_plot(
        vot_data: Dict[str, pd.DataFrame],
        segment_vars: List[str],
        output_file: Optional[str] = None,
        figsize: Tuple[float, float] = (15, 10)
    ) -> plt.Figure:
        """
        Create multi-panel plot showing VOT heterogeneity across different dimensions.
        
        Parameters
        ----------
        vot_data : dict
            Dictionary mapping model names to DataFrames with VOT and segment variables
        segment_vars : list
            List of segmentation variables to plot
        output_file : str, optional
            Path to save the figure
        figsize : tuple
            Figure size
            
        Returns
        -------
        fig : matplotlib.Figure
            The created figure
        """
        n_models = len(vot_data)
        n_segments = len(segment_vars)
        
        fig, axes = plt.subplots(n_segments, n_models, figsize=figsize)
        
        # Handle single model or segment case
        if n_models == 1:
            axes = axes.reshape(-1, 1)
        if n_segments == 1:
            axes = axes.reshape(1, -1)
        
        for i, (model_name, df) in enumerate(vot_data.items()):
            for j, segment_var in enumerate(segment_vars):
                ax = axes[j, i] if n_segments > 1 and n_models > 1 else axes[max(i, j)]
                
                if segment_var in df.columns:
                    # Clean data
                    df_clean = df[[segment_var, 'vot']].copy()
                    df_clean = df_clean[np.isfinite(df_clean['vot'])]
                    
                    # Create violin plot
                    sns.violinplot(data=df_clean, x=segment_var, y='vot',
                                  palette='Set2', ax=ax)
                    
                    # Add title
                    if j == 0:
                        ax.set_title(model_name, fontsize=12, fontweight='bold')
                    
                    # Clean up labels
                    if i == 0:
                        ax.set_ylabel('VOT ($/hr)', fontsize=10)
                    else:
                        ax.set_ylabel('')
                    
                    ax.set_xlabel(segment_var.replace('_', ' ').title(), fontsize=10)
                    ax.tick_params(axis='x', rotation=45)
                else:
                    ax.text(0.5, 0.5, f'No {segment_var} data',
                           transform=ax.transAxes, ha='center', va='center')
                    ax.set_xticks([])
                    ax.set_yticks([])
        
        plt.suptitle('Value of Time Heterogeneity Analysis', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if output_file:
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Figure saved as {output_file}")
        
        return fig