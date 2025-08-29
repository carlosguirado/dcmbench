"""
Model interface definitions for DCMBench.

This module defines the interfaces that model adapters must implement to be
compatible with DCMBench benchmarking. It provides a clear contract for how models
should expose their prediction capabilities.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Union, Optional, Any

class PredictionInterface(ABC):
    """
    Interface for model prediction across different frameworks.
    
    This abstract class defines the methods that all model adapters
    must implement to provide a consistent interface for prediction
    and evaluation.
    """
    
    @abstractmethod
    def predict_probabilities(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict choice probabilities for each alternative.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns for each alternative and rows for each observation,
            containing the predicted probability of each alternative being chosen
        """
        pass
    
    @abstractmethod
    def predict_choices(self, data: pd.DataFrame) -> pd.Series:
        """
        Predict the most likely choice for each observation.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data
            
        Returns
        -------
        pd.Series
            Series of predicted choices
        """
        pass
    
    @abstractmethod
    def calculate_choice_accuracy(self, data: pd.DataFrame, choice_column: str = "CHOICE") -> float:
        """
        Calculate the choice prediction accuracy.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data
        choice_column : str, default="CHOICE"
            Name of the column containing actual choices
            
        Returns
        -------
        float
            Proportion of correctly predicted choices
        """
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get model metrics for evaluation.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary of model metrics
        """
        pass

def format_probabilities(probabilities: pd.DataFrame) -> pd.DataFrame:
    """
    Format probability predictions to ensure consistent structure.
    
    Parameters
    ----------
    probabilities : pd.DataFrame
        Raw probability predictions
        
    Returns
    -------
    pd.DataFrame
        Formatted probability predictions with consistent column types
    """
    # Ensure column names are strings
    probabilities = probabilities.copy()
    probabilities.columns = [str(col) for col in probabilities.columns]
    
    return probabilities

def format_choices(choices: pd.Series) -> pd.Series:
    """
    Format choice predictions to ensure consistent structure.
    
    Parameters
    ----------
    choices : pd.Series
        Raw choice predictions
        
    Returns
    -------
    pd.Series
        Formatted choice predictions with consistent data type
    """
    # Convert to string
    return choices.astype(str)