"""
Model extensions for DCMBench compatibility.

This module provides helper functions to ensure Biogeme models work seamlessly
with DCMBench's policy analysis and benchmarking utilities.

Author: DCMBench Team
License: MIT
"""

import logging
from typing import Dict, Optional, Any
import biogeme.biogeme as bio
from biogeme.nests import NestsForNestedLogit

logger = logging.getLogger(__name__)


def ensure_model_compatibility(biogeme_model: bio.BIOGEME,
                              V: Dict[int, Any],
                              av: Dict[int, Any],
                              nests: Optional[NestsForNestedLogit] = None) -> bio.BIOGEME:
    """
    Ensure a Biogeme model stores necessary components for DCMBench policy analysis.
    
    This function adds the utility functions (V), availability conditions (av),
    and optional nesting structure to the Biogeme model object, making it fully
    compatible with DCMBench's UniversalBiogemeAdapter and PolicyAnalyzer.
    
    Parameters
    ----------
    biogeme_model : bio.BIOGEME
        The Biogeme model object
    V : dict
        Dictionary of utility functions for each alternative
    av : dict
        Dictionary of availability conditions for each alternative
    nests : NestsForNestedLogit, optional
        Nesting structure for nested logit models
        
    Returns
    -------
    bio.BIOGEME
        The same model object with components attached
        
    Examples
    --------
    >>> import biogeme.biogeme as bio
    >>> from biogeme import models
    >>> from dcmbench.utils.model_extensions import ensure_model_compatibility
    >>> 
    >>> # After defining utilities and availability
    >>> V = {1: V1, 2: V2, 3: V3}
    >>> av = {1: TRAIN_AV, 2: SM_AV, 3: CAR_AV}
    >>> 
    >>> # Create model
    >>> logprob = models.loglogit(V, av, CHOICE)
    >>> biogeme = bio.BIOGEME(database, logprob)
    >>> 
    >>> # Make it DCMBench-compatible
    >>> biogeme = ensure_model_compatibility(biogeme, V, av)
    """
    # Store utility functions
    biogeme_model.V = V
    biogeme_model.av = av
    
    # Store nesting structure if provided
    if nests is not None:
        biogeme_model.nests = nests
        logger.info("Stored nesting structure in model")
    
    # Store alternative IDs for reference
    biogeme_model.alternatives = list(V.keys())
    
    logger.info(f"Model '{biogeme_model.modelName}' made DCMBench-compatible")
    logger.debug(f"Stored utilities for alternatives: {biogeme_model.alternatives}")
    
    return biogeme_model


def check_model_compatibility(biogeme_model: bio.BIOGEME) -> Dict[str, bool]:
    """
    Check if a Biogeme model has all necessary components for DCMBench analysis.
    
    Parameters
    ----------
    biogeme_model : bio.BIOGEME
        The Biogeme model to check
        
    Returns
    -------
    Dict[str, bool]
        Dictionary indicating which components are present
        
    Examples
    --------
    >>> status = check_model_compatibility(model)
    >>> if not status['has_utilities']:
    ...     print("Model missing utility functions!")
    """
    status = {
        'has_utilities': hasattr(biogeme_model, 'V') and biogeme_model.V is not None,
        'has_availability': hasattr(biogeme_model, 'av') and biogeme_model.av is not None,
        'has_nests': hasattr(biogeme_model, 'nests') and biogeme_model.nests is not None,
        'has_alternatives': hasattr(biogeme_model, 'alternatives') and biogeme_model.alternatives is not None,
        'is_compatible': False
    }
    
    # Model is compatible if it has at least utilities and availability
    status['is_compatible'] = status['has_utilities'] and status['has_availability']
    
    return status


def prepare_swissmetro_data_standard(data):
    """
    Standard data preparation for Swissmetro dataset following Biogeme conventions.
    
    This function prepares the Swissmetro data exactly as in the Biogeme examples,
    ensuring compatibility with documented models.
    
    Parameters
    ----------
    data : pd.DataFrame
        Raw Swissmetro data
        
    Returns
    -------
    pd.DataFrame
        Prepared data with scaled variables and SP-specific availability
    """
    import pandas as pd
    
    # Filter: PURPOSE in [1,3] and CHOICE != 0
    df = data[
        (data.PURPOSE.isin([1, 3]))
        & (data.CHOICE != 0)
    ].copy()
    
    # Define cost variables (accounting for GA discount)
    df['SM_COST'] = df['SM_CO'] * (df['GA'] == 0)
    df['TRAIN_COST'] = df['TRAIN_CO'] * (df['GA'] == 0)
    
    # SP-specific availability
    df['CAR_AV_SP'] = df['CAR_AV'] * (df['SP'] != 0)
    df['TRAIN_AV_SP'] = df['TRAIN_AV'] * (df['SP'] != 0)
    
    # Scaled variables (divide by 100)
    df['TRAIN_TT_SCALED'] = df['TRAIN_TT'] / 100
    df['TRAIN_COST_SCALED'] = df['TRAIN_COST'] / 100
    df['SM_TT_SCALED'] = df['SM_TT'] / 100
    df['SM_COST_SCALED'] = df['SM_COST'] / 100
    df['CAR_TT_SCALED'] = df['CAR_TT'] / 100
    df['CAR_CO_SCALED'] = df['CAR_CO'] / 100
    
    logger.info(f"Prepared Swissmetro data: {df.shape[0]} observations")
    
    return df


def prepare_modecanada_data_standard(data):
    """
    Standard data preparation for ModeCanada dataset.
    
    Converts the long-format ModeCanada data to wide format required by Biogeme.
    
    Parameters
    ----------
    data : pd.DataFrame
        Raw ModeCanada data in long format
        
    Returns
    -------
    pd.DataFrame
        Wide-format data suitable for Biogeme
    """
    import pandas as pd
    
    df = data.copy()
    
    # Define mode mapping (1-based for Biogeme)
    mode_mapping = {
        'train': 1,
        'car': 2,
        'bus': 3,
        'air': 4
    }
    
    # Convert alt column from strings to numeric
    df['alt_num'] = df['alt'].map(mode_mapping)
    
    # Create wide format dataframe
    cases = df['case'].unique()
    wide_data = []
    
    for case_id in cases:
        case_data = df[df['case'] == case_id]
        
        # Get case-specific variables
        row = {
            'case': case_id,
            'income': case_data['income'].iloc[0],
            'urban': case_data['urban'].iloc[0]
        }
        
        # Get the chosen alternative
        chosen = case_data[case_data['choice'] == 1]['alt_num'].iloc[0]
        row['CHOICE'] = chosen
        
        # Get alternative-specific variables
        for alt in ['train', 'car', 'bus', 'air']:
            alt_data = case_data[case_data['alt'] == alt]
            if not alt_data.empty:
                row[f'{alt}_cost'] = alt_data['cost'].iloc[0]
                row[f'{alt}_time'] = alt_data['ivt'].iloc[0] + alt_data['ovt'].iloc[0]
                row[f'{alt}_available'] = 1
                row[f'{alt}_freq'] = alt_data.get('freq', [0]).iloc[0]
            else:
                row[f'{alt}_cost'] = 0
                row[f'{alt}_time'] = 0
                row[f'{alt}_available'] = 0
                row[f'{alt}_freq'] = 0
        
        wide_data.append(row)
    
    wide_df = pd.DataFrame(wide_data)
    logger.info(f"Converted ModeCanada to wide format: {wide_df.shape[0]} observations")
    
    return wide_df


def prepare_ltds_data_standard(data):
    """
    Standard data preparation for LTDS dataset.
    
    Prepares the London Transportation Data Survey for discrete choice modeling.
    
    Parameters
    ----------
    data : pd.DataFrame
        Raw LTDS data
        
    Returns
    -------
    pd.DataFrame
        Prepared data with numeric choice variable and dummy variables
    """
    import pandas as pd
    
    # Map travel modes to numeric values
    mode_mapping = {
        'walk': 1,
        'cycle': 2,
        'pt': 3,
        'drive': 4
    }
    
    df = data.copy()
    df['CHOICE'] = df['travel_mode'].map(mode_mapping)
    
    # Filter out invalid choices
    df_filtered = df[df['CHOICE'].notna()].copy()
    
    # Create availability variables (assume all modes available)
    df_filtered['walk_available'] = 1
    df_filtered['cycle_available'] = 1
    df_filtered['pt_available'] = 1
    df_filtered['drive_available'] = 1
    
    # Create categorical dummy variables
    categorical_columns = ['purpose', 'fueltype', 'faretype']
    
    for col in categorical_columns:
        if col in df_filtered.columns:
            dummies = pd.get_dummies(df_filtered[col], prefix=col).astype(int)
            df_filtered = pd.concat([df_filtered, dummies], axis=1)
            df_filtered = df_filtered.drop(columns=col)
    
    # Drop any remaining object columns
    object_cols = df_filtered.select_dtypes(include=['object']).columns
    if len(object_cols) > 0:
        df_filtered = df_filtered.drop(columns=object_cols)
        logger.info(f"Dropped object columns: {list(object_cols)}")
    
    # Sample if dataset is large
    max_size = 5000
    if len(df_filtered) > max_size:
        df_filtered = df_filtered.sample(n=max_size, random_state=42)
        logger.info(f"Sampled {max_size} observations from LTDS")
    
    logger.info(f"Prepared LTDS data: {df_filtered.shape[0]} observations")
    
    return df_filtered


def get_standard_nesting_structures():
    """
    Get recommended nesting structures for common datasets.
    
    Returns
    -------
    dict
        Dictionary of nesting structures by dataset
        
    Examples
    --------
    >>> nests = get_standard_nesting_structures()
    >>> swissmetro_nest = nests['swissmetro']['existing_modes']
    """
    return {
        'swissmetro': {
            'existing_modes': {
                'description': 'Train and Car in existing modes nest',
                'alternatives': [1, 3],  # Train and Car
                'expected_mu': 2.05,
                'reference': 'Biogeme example b09nested.py'
            },
            'public_transport': {
                'description': 'Train and Swissmetro in public transport nest',
                'alternatives': [1, 2],  # Train and Swissmetro
                'expected_mu': 1.0,  # Usually collapses to MNL
                'reference': 'Common but often not significant'
            }
        },
        'modecanada': {
            'transit': {
                'description': 'Train, Bus, and Air in transit nest',
                'alternatives': [1, 3, 4],  # Train, Bus, Air
                'expected_mu': 3.0,
                'reference': 'High correlation among transit modes'
            },
            'ground': {
                'description': 'Train, Car, and Bus in ground transport nest',
                'alternatives': [1, 2, 3],  # Train, Car, Bus
                'expected_mu': 1.5,
                'reference': 'Ground vs air distinction'
            },
            'public': {
                'description': 'Train and Bus in public transport nest',
                'alternatives': [1, 3],  # Train and Bus
                'expected_mu': 2.0,
                'reference': 'Public vs private distinction'
            }
        },
        'ltds': {
            'active': {
                'description': 'Walk and Cycle in active modes nest',
                'alternatives': [1, 2],  # Walk and Cycle
                'expected_mu': 2.5,
                'reference': 'Non-motorized modes'
            },
            'motorized': {
                'description': 'PT and Drive in motorized modes nest',
                'alternatives': [3, 4],  # PT and Drive
                'expected_mu': 1.8,
                'reference': 'Motorized vs non-motorized'
            }
        }
    }