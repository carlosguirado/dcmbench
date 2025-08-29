"""
Preprocessing utilities for discrete choice modeling.

This module contains functions for preprocessing data for discrete choice models,
extracted from the BaseDiscreteChoiceModel class to allow for more flexibility in data processing.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any

logger = logging.getLogger(__name__)

def debug_check_nans(df: pd.DataFrame, stage: str = "Unknown") -> None:
    """
    Debug helper to check for NaN values in a DataFrame.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to check for NaN values
    stage : str
        Description of the processing stage for logging purposes
    """
    nan_cols = df.columns[df.isna().any()].tolist()
    if nan_cols:
        logger.info(f"\nNaN check at {stage}:")
        for col in nan_cols:
            nan_count = df[col].isna().sum()
            logger.info(f"{col}: {nan_count} NaN values")
            if nan_count > 0:
                logger.info(f"Sample of rows with NaN in {col}:")
                logger.info(df[df[col].isna()].head())

def convert_numeric_columns(df: pd.DataFrame, 
                           numeric_columns: List[str], 
                           default_value: float = 0.0) -> pd.DataFrame:
    """
    Convert columns to numeric (float64) dtype.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the columns to convert
    numeric_columns : List[str]
        List of column names to convert to float64
    default_value : float, default=0.0
        Value to use for filling NaN values after conversion
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with converted numeric columns
    """
    df_result = df.copy()
    
    for col in numeric_columns:
        if col in df_result.columns:
            df_result[col] = pd.to_numeric(df_result[col], errors='coerce').fillna(default_value).astype('float64')
            
    return df_result

def encode_categorical_variables(df: pd.DataFrame, 
                                categorical_columns: Dict[str, Dict[str, int]], 
                                category_defaults: Optional[Dict[str, int]] = None) -> pd.DataFrame:
    """
    Encode categorical variables to numeric values using provided mappings.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the categorical columns
    categorical_columns : Dict[str, Dict[str, int]]
        Dictionary mapping column names to their respective value mappings
        e.g., {'travel_mode': {'walk': 1, 'cycle': 2, 'pt': 3, 'drive': 4}}
    category_defaults : Dict[str, int], optional
        Default values to use for each categorical column when a value is not in the mapping
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with encoded categorical columns
    """
    df_result = df.copy()
    
    # Use empty dict if no defaults are provided
    if category_defaults is None:
        category_defaults = {}
    
    for col, mapping in categorical_columns.items():
        if col in df_result.columns:
            default_value = category_defaults.get(col, 1)  # Default to 1 if not specified
            
            # If column is still object type (string), map it to integers
            if df_result[col].dtype == 'object':
                df_result[col] = df_result[col].map(mapping).fillna(default_value).astype('int64')
            else:
                # If column already exists but is numeric, just ensure it's int64
                df_result[col] = df_result[col].fillna(default_value).astype('int64')
                
    return df_result

def preprocess_data(df: pd.DataFrame, 
                   numeric_columns: Optional[List[str]] = None,
                   categorical_columns: Optional[Dict[str, Dict[str, int]]] = None,
                   category_defaults: Optional[Dict[str, int]] = None,
                   debug: bool = False) -> pd.DataFrame:
    """
    Preprocess data for discrete choice models.
    
    This function handles standard preprocessing tasks for discrete choice modeling:
    1. Converting numeric columns to float64
    2. Encoding categorical variables to integers
    3. Optionally checking for NaN values during processing
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input data to preprocess
    numeric_columns : List[str], optional
        List of columns to convert to float64
    categorical_columns : Dict[str, Dict[str, int]], optional
        Dictionary mapping column names to their respective value mappings
    category_defaults : Dict[str, int], optional
        Default values to use for each categorical column when a value is not in the mapping
    debug : bool, default=False
        Whether to print debug information during processing
        
    Returns
    -------
    pandas.DataFrame
        Preprocessed data with correct dtypes
    """
    result_df = df.copy()
    
    # Default column lists if none provided
    if numeric_columns is None:
        numeric_columns = [
            'dur_walking', 'dur_cycling', 'dur_driving', 
            'dur_pt_access', 'dur_pt_rail', 'dur_pt_bus', 
            'dur_pt_int_total', 'cost_driving_fuel',
            'cost_driving_con_charge', 'cost_transit',
            'driving_traffic_percent'
        ]
    
    if categorical_columns is None:
        categorical_columns = {
            'travel_mode': {'walk': 1, 'cycle': 2, 'pt': 3, 'drive': 4},
            'purpose': {'HBW': 1, 'HBE': 2, 'HBO': 3, 'B': 4, 'NHBO': 5},
            'fueltype': {'Petrol_Car': 1, 'Diesel_Car': 2, 'Hybrid_Car': 3,
                        'Petrol_LGV': 4, 'Diesel_LGV': 5, 'Average_Car': 6},
            'faretype': {'full': 1, '16+': 2, 'child': 3, 'dis': 4, 'free': 5}
        }
    
    if category_defaults is None:
        category_defaults = {
            'travel_mode': 1,  # Default to walk
            'purpose': 1,      # Default to HBW
            'fueltype': 6,     # Default to Average_Car
            'faretype': 1      # Default to full fare
        }
    
    # Debug: Check before any processing
    if debug:
        debug_check_nans(result_df, "Before preprocessing")
    
    # Convert numeric columns to float64
    result_df = convert_numeric_columns(result_df, numeric_columns)
    
    # Debug: Check after numeric conversion
    if debug:
        debug_check_nans(result_df, "After numeric conversion")
    
    # Encode categorical variables
    result_df = encode_categorical_variables(result_df, categorical_columns, category_defaults)
    
    # Debug: Check after categorical conversion
    if debug:
        debug_check_nans(result_df, "After categorical conversion")
    
    # Convert any remaining float columns to float64 to ensure consistency
    float_cols = result_df.select_dtypes(include=['float']).columns
    for col in float_cols:
        result_df[col] = result_df[col].astype('float64')
    
    # Debug: Check after final conversion
    if debug:
        debug_check_nans(result_df, "After final conversion")
    
    return result_df

def needs_preprocessing(df: pd.DataFrame) -> bool:
    """
    Check if data needs preprocessing (if categorical columns are not yet numeric).
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to check
        
    Returns
    -------
    bool
        True if preprocessing is needed, False otherwise
    """
    categorical_columns = ['travel_mode', 'purpose', 'fueltype', 'faretype']
    return any(col in df.columns and df[col].dtype == 'object' for col in categorical_columns)

def create_derived_variables(df: pd.DataFrame, 
                           derived_vars_config: Optional[Dict[str, Dict[str, Any]]] = None) -> pd.DataFrame:
    """
    Create derived variables based on existing variables.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame
    derived_vars_config : Dict[str, Dict[str, Any]], optional
        Configuration for derived variables. Each key is the name of a new variable,
        and the value is a dictionary with the calculation formula and other options.
        If None, no derived variables are created.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with additional derived variables
    """
    if derived_vars_config is None:
        return df
    
    result_df = df.copy()
    
    for new_var, config in derived_vars_config.items():
        if 'formula' in config:
            # Simple formula-based derivation
            formula = config['formula']
            try:
                result_df[new_var] = eval(formula, {"df": result_df, "np": np})
            except Exception as e:
                logger.warning(f"Error creating derived variable {new_var}: {str(e)}")
                
    return result_df

def create_availability_indicators(df: pd.DataFrame, 
                                 availability_configs: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    Create availability indicator variables for different modes.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame
    availability_configs : Dict[str, str], optional
        Dictionary mapping availability variable names to their calculation formulas
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with additional availability indicator variables
    """
    if availability_configs is None:
        return df
    
    result_df = df.copy()
    
    for var_name, formula in availability_configs.items():
        try:
            result_df[var_name] = eval(formula, {"df": result_df, "np": np})
        except Exception as e:
            logger.warning(f"Error creating availability indicator {var_name}: {str(e)}")
            
    return result_df

def clean_dataset(df: pd.DataFrame, 
                 filter_conditions: Optional[List[str]] = None, 
                 handle_outliers: bool = True,
                 max_z_score: float = 3.0) -> pd.DataFrame:
    """
    Clean dataset by filtering rows and handling outliers.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame
    filter_conditions : List[str], optional
        List of filter conditions as strings to be evaluated
    handle_outliers : bool, default=True
        Whether to handle outliers in numeric columns
    max_z_score : float, default=3.0
        Maximum allowed z-score for outlier detection
        
    Returns
    -------
    pandas.DataFrame
        Cleaned DataFrame
    """
    result_df = df.copy()
    
    # Apply filter conditions
    if filter_conditions:
        for condition in filter_conditions:
            try:
                mask = eval(condition, {"df": result_df, "np": np})
                result_df = result_df[~mask]  # Remove rows that match the condition
            except Exception as e:
                logger.warning(f"Error applying filter condition '{condition}': {str(e)}")
    
    # Handle outliers if requested
    if handle_outliers:
        numeric_cols = result_df.select_dtypes(include=['float64', 'int64']).columns
        
        for col in numeric_cols:
            if result_df[col].std() > 0:  # Only process columns with variation
                z_scores = np.abs((result_df[col] - result_df[col].mean()) / result_df[col].std())
                result_df = result_df[z_scores < max_z_score]
    
    return result_df