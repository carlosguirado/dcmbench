"""
Systematic Heterogeneity VOT Calculations for DCMBench

This module provides utilities for calculating and managing VOT 
for models with systematic (observable) heterogeneity.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, Optional, Union
from dataclasses import dataclass


@dataclass
class SegmentVOT:
    """Container for VOT information of a population segment."""
    name: str
    vot_value: float
    population_count: int
    population_proportion: float
    
    def __repr__(self):
        return f"SegmentVOT(name='{self.name}', vot=${self.vot_value:.2f}/hr, n={self.population_count}, prop={self.population_proportion:.1%})"


class SystematicHeterogeneityVOT:
    """
    Calculator for VOT with systematic heterogeneity.
    
    This class handles models where VOT varies by observable characteristics
    like income groups, urban/rural, trip purpose, etc.
    """
    
    def __init__(self, time_unit: str = 'hour', cost_unit: str = 'dollar'):
        """
        Initialize the calculator.
        
        Parameters
        ----------
        time_unit : str
            Unit of time in the data ('hour' or 'minute')
        cost_unit : str
            Unit of cost in the data ('dollar', 'euro', etc.)
        """
        self.time_unit = time_unit
        self.cost_unit = cost_unit
        self.time_multiplier = 60 if time_unit == 'minute' else 1
    
    def calculate_segment_vot(
        self,
        time_coef: float,
        cost_coef: float,
        segment_name: str,
        population_count: int,
        total_population: int
    ) -> SegmentVOT:
        """
        Calculate VOT for a specific population segment.
        
        Parameters
        ----------
        time_coef : float
            Time coefficient for this segment
        cost_coef : float
            Cost coefficient for this segment
        segment_name : str
            Name of the segment (e.g., "High Income", "Urban")
        population_count : int
            Number of individuals in this segment
        total_population : int
            Total population size
            
        Returns
        -------
        SegmentVOT
            VOT information for this segment
        """
        vot = self.time_multiplier * time_coef / cost_coef
        proportion = population_count / total_population
        
        return SegmentVOT(
            name=segment_name,
            vot_value=vot,
            population_count=population_count,
            population_proportion=proportion
        )
    
    def calculate_from_model_results(
        self,
        model_results: Dict[str, float],
        segment_mapping: Dict[str, Tuple[str, str]],
        population_data: pd.DataFrame,
        segment_column: str
    ) -> Dict[str, SegmentVOT]:
        """
        Calculate VOT for all segments from model estimation results.
        
        Parameters
        ----------
        model_results : dict
            Dictionary of parameter estimates from model
        segment_mapping : dict
            Maps segment names to (time_param, cost_param) tuples
            e.g., {'Low Income': ('B_TIME_LOW', 'B_COST_LOW')}
        population_data : pd.DataFrame
            Data containing segment information
        segment_column : str
            Column name containing segment indicators
            
        Returns
        -------
        dict
            Dictionary mapping segment names to SegmentVOT objects
        """
        segments = {}
        total_pop = len(population_data)
        
        for segment_name, (time_param, cost_param) in segment_mapping.items():
            # Get coefficients
            time_coef = model_results[time_param]
            cost_coef = model_results[cost_param]
            
            # Count population
            if segment_column == 'income':
                # Special handling for income-based segmentation
                if 'Low' in segment_name:
                    pop_count = (population_data['high_income'] == 0).sum()
                else:
                    pop_count = (population_data['high_income'] == 1).sum()
            elif segment_column == 'urban':
                # Urban/rural segmentation
                if 'Urban' in segment_name:
                    pop_count = (population_data['urban'] == 1).sum()
                else:
                    pop_count = (population_data['urban'] == 0).sum()
            else:
                # Generic segmentation
                segment_value = 1 if segment_name in ['High', 'Urban', 'Yes'] else 0
                pop_count = (population_data[segment_column] == segment_value).sum()
            
            segments[segment_name] = self.calculate_segment_vot(
                time_coef, cost_coef, segment_name, pop_count, total_pop
            )
        
        return segments
    
    def get_population_weighted_stats(
        self,
        segments: Dict[str, SegmentVOT]
    ) -> Dict[str, float]:
        """
        Calculate population-weighted statistics.
        
        Parameters
        ----------
        segments : dict
            Dictionary of SegmentVOT objects
            
        Returns
        -------
        dict
            Statistics including mean, std, min, max VOT
        """
        # Create population array
        vot_array = []
        for segment in segments.values():
            vot_array.extend([segment.vot_value] * segment.population_count)
        
        vot_array = np.array(vot_array)
        
        return {
            'mean': np.mean(vot_array),
            'median': np.median(vot_array),
            'std': np.std(vot_array),
            'min': np.min(vot_array),
            'max': np.max(vot_array),
            'range': np.max(vot_array) - np.min(vot_array)
        }
    
    def create_summary_dataframe(
        self,
        segments: Dict[str, SegmentVOT]
    ) -> pd.DataFrame:
        """
        Create a summary DataFrame of all segments.
        
        Parameters
        ----------
        segments : dict
            Dictionary of SegmentVOT objects
            
        Returns
        -------
        pd.DataFrame
            Summary table with segment information
        """
        data = []
        for segment in segments.values():
            data.append({
                'Segment': segment.name,
                'VOT ($/hr)': f"${segment.vot_value:.2f}",
                'Population': segment.population_count,
                'Proportion': f"{segment.population_proportion:.1%}"
            })
        
        return pd.DataFrame(data)


def calculate_systematic_vot(
    model_results: Union[Dict, Any],
    segment_type: str,
    data: pd.DataFrame,
    time_unit: str = 'minute',
    cost_unit: str = 'dollar'
) -> Tuple[Dict[str, SegmentVOT], Dict[str, float]]:
    """
    Convenience function to calculate systematic heterogeneity VOT.
    
    Parameters
    ----------
    model_results : dict or Results object
        Model estimation results
    segment_type : str
        Type of segmentation ('income', 'urban', etc.)
    data : pd.DataFrame
        Population data
    time_unit : str
        Unit of time in the model
    cost_unit : str
        Unit of cost in the model
        
    Returns
    -------
    tuple
        (segments dictionary, statistics dictionary)
    """
    # Extract parameters if results object
    if hasattr(model_results, 'get_beta_values'):
        params = model_results.get_beta_values()
    else:
        params = model_results
    
    # Create calculator
    calculator = SystematicHeterogeneityVOT(time_unit, cost_unit)
    
    # Define segment mappings based on type
    if segment_type == 'income':
        segment_mapping = {
            'Low Income': ('B_TIME_LOW', 'B_COST_LOW'),
            'High Income': ('B_TIME_HIGH', 'B_COST_HIGH')
        }
        # Ensure high_income column exists
        if 'high_income' not in data.columns and 'income' in data.columns:
            data['high_income'] = (data['income'] > data['income'].median()).astype(int)
    elif segment_type == 'urban':
        segment_mapping = {
            'Rural': ('B_TIME_RURAL', 'B_COST_RURAL'),
            'Urban': ('B_TIME_URBAN', 'B_COST_URBAN')
        }
    else:
        raise ValueError(f"Unknown segment type: {segment_type}")
    
    # Calculate segments
    segments = calculator.calculate_from_model_results(
        params, segment_mapping, data, segment_type
    )
    
    # Get statistics
    stats = calculator.get_population_weighted_stats(segments)
    
    return segments, stats