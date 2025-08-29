# dcmbench/utils/metrics.py
"""
Metrics calculation utilities for discrete choice models.

This module provides functions for calculating various metrics for discrete choice models,
including choice accuracy, market share accuracy, elasticities, and other common metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple

def calculate_metrics(model_results: Any, actual_choices: Union[np.ndarray, pd.Series], 
                     predicted_probs: Union[np.ndarray, pd.DataFrame]) -> Dict[str, float]:
    """
    Calculate common metrics for mode choice models.
    
    Parameters
    ----------
    model_results : Any
        Results object from the model (can be Biogeme, sklearn, or custom)
    actual_choices : np.ndarray or pd.Series
        Actual choices made by individuals
    predicted_probs : np.ndarray or pd.DataFrame
        Predicted probabilities for each alternative
    
    Returns
    -------
    Dict[str, float]
        Dictionary of calculated metrics
    """
    metrics = {}
    
    # If input is pandas, convert to numpy for consistent processing
    if isinstance(actual_choices, pd.Series):
        actual_choices = actual_choices.values
    if isinstance(predicted_probs, pd.DataFrame):
        predicted_probs = predicted_probs.values
    
    # Basic log-likelihood metrics if available in results
    if hasattr(model_results, 'data'):
        if hasattr(model_results.data, 'finalLogLikelihood'):
            metrics['log_likelihood'] = model_results.data.finalLogLikelihood
        if hasattr(model_results.data, 'nullLogLikelihood'):
            metrics['null_log_likelihood'] = model_results.data.nullLogLikelihood
            # Calculate rho-squared if both log-likelihoods are available
            if 'log_likelihood' in metrics:
                metrics['rho_squared'] = 1 - (metrics['log_likelihood'] / metrics['null_log_likelihood'])
                # If we know the number of parameters, calculate adjusted rho-squared
                if hasattr(model_results.data, 'betaValues'):
                    n_params = len(model_results.data.betaValues)
                    metrics['adj_rho_squared'] = 1 - ((metrics['log_likelihood'] - n_params) / metrics['null_log_likelihood'])
    
    # Prediction metrics
    metrics['choice_accuracy'] = calculate_choice_accuracy(actual_choices, predicted_probs)
    
    # Calculate market shares
    actual_shares = calculate_actual_shares(actual_choices)
    predicted_shares = calculate_predicted_shares(predicted_probs)
    metrics['market_share_accuracy'] = calculate_market_share_accuracy(actual_shares, predicted_shares)
    
    # Add market share data to the metrics
    metrics['actual_shares'] = actual_shares
    metrics['predicted_shares'] = predicted_shares
    
    # Add confusion matrix
    metrics['confusion_matrix'] = calculate_confusion_matrix(actual_choices, predicted_probs)
    
    return metrics

def calculate_choice_accuracy(actual_choices: Union[np.ndarray, pd.Series],
                             predicted_probs: Union[np.ndarray, pd.DataFrame]) -> float:
    """
    Calculate the choice prediction accuracy.
    
    Parameters
    ----------
    actual_choices : np.ndarray or pd.Series
        Actual choices made by individuals
    predicted_probs : np.ndarray or pd.DataFrame
        Predicted probabilities for each alternative
    
    Returns
    -------
    float
        Prediction accuracy (proportion of correctly predicted choices)
    """
    # Convert pandas objects to numpy if needed
    if isinstance(actual_choices, pd.Series):
        actual_choices = actual_choices.values
    if isinstance(predicted_probs, pd.DataFrame):
        # For DataFrame, get the column indices that match the values
        predicted_choices = predicted_probs.idxmax(axis=1).values
    else:
        # For numpy array, get the index of maximum probability
        predicted_choices = np.argmax(predicted_probs, axis=1)
        # If actual_choices are 1-indexed but predictions are 0-indexed, adjust
        if np.min(actual_choices) == 1 and np.min(predicted_choices) == 0:
            predicted_choices += 1
    
    # Calculate accuracy
    return np.mean(actual_choices == predicted_choices)

def calculate_actual_shares(choices: Union[np.ndarray, pd.Series]) -> Dict[int, float]:
    """
    Calculate the actual market shares from observed choices.
    
    Parameters
    ----------
    choices : np.ndarray or pd.Series
        Observed choices
    
    Returns
    -------
    Dict[int, float]
        Dictionary mapping alternatives to their observed market shares
    """
    if isinstance(choices, pd.Series):
        value_counts = choices.value_counts(normalize=True)
        return value_counts.to_dict()
    else:
        # Calculate using numpy
        unique_values, counts = np.unique(choices, return_counts=True)
        shares = counts / len(choices)
        return {int(val): share for val, share in zip(unique_values, shares)}

def calculate_predicted_shares(predicted_probs: Union[np.ndarray, pd.DataFrame]) -> Dict[int, float]:
    """
    Calculate the predicted market shares from choice probabilities.
    
    Parameters
    ----------
    predicted_probs : np.ndarray or pd.DataFrame
        Predicted probabilities for each alternative
    
    Returns
    -------
    Dict[int, float]
        Dictionary mapping alternatives to their predicted market shares
    """
    if isinstance(predicted_probs, pd.DataFrame):
        # For DataFrame, calculate mean of each column
        shares = predicted_probs.mean()
        return {int(alt): share for alt, share in shares.items()}
    else:
        # For numpy array, calculate mean of each column
        shares = predicted_probs.mean(axis=0)
        return {i+1: share for i, share in enumerate(shares)}  # Assume 1-indexed alternatives

def calculate_market_share_accuracy(actual_shares: Dict[int, float], 
                                  predicted_shares: Dict[int, float],
                                  method: str = 'absolute_difference') -> float:
    """
    Calculate the market share prediction accuracy.
    
    Parameters
    ----------
    actual_shares : Dict[int, float]
        Dictionary of actual market shares
    predicted_shares : Dict[int, float]
        Dictionary of predicted market shares
    method : str, default='absolute_difference'
        Method to calculate accuracy:
        - 'absolute_difference': 1 - (sum of absolute differences / 2)
        - 'root_mean_squared': 1 - sqrt(mean squared difference)
        - 'correlation': correlation between actual and predicted shares
    
    Returns
    -------
    float
        Market share accuracy metric
    """
    # Get all alternatives
    all_alternatives = set(actual_shares.keys()) | set(predicted_shares.keys())
    
    if method == 'absolute_difference':
        # Sum of absolute differences, divided by 2 to normalize
        total_abs_error = sum(
            abs(actual_shares.get(alt, 0) - predicted_shares.get(alt, 0))
            for alt in all_alternatives
        )
        return 1 - (total_abs_error / 2)
    
    elif method == 'root_mean_squared':
        # Root mean squared error, scaled to 0-1
        squared_errors = [
            (actual_shares.get(alt, 0) - predicted_shares.get(alt, 0))**2
            for alt in all_alternatives
        ]
        rmse = np.sqrt(np.mean(squared_errors))
        # Scale to 0-1 (assuming max RMSE is 1)
        return 1 - rmse
    
    elif method == 'correlation':
        # Correlation between actual and predicted shares
        actual = np.array([actual_shares.get(alt, 0) for alt in all_alternatives])
        predicted = np.array([predicted_shares.get(alt, 0) for alt in all_alternatives])
        return np.corrcoef(actual, predicted)[0, 1]
    
    else:
        raise ValueError(f"Unknown market share accuracy method: {method}")

def calculate_confusion_matrix(actual_choices: Union[np.ndarray, pd.Series],
                              predicted_probs: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
    """
    Calculate the confusion matrix for model predictions.
    
    Parameters
    ----------
    actual_choices : np.ndarray or pd.Series
        Actual choices made by individuals
    predicted_probs : np.ndarray or pd.DataFrame
        Predicted probabilities for each alternative
    
    Returns
    -------
    pd.DataFrame
        Confusion matrix as a DataFrame
    """
    # Convert pandas objects to numpy if needed
    if isinstance(actual_choices, pd.Series):
        actual_choices = actual_choices.values
    
    if isinstance(predicted_probs, pd.DataFrame):
        # Get the column names for labeling
        alternatives = [str(col) for col in predicted_probs.columns]
        predicted_choices = predicted_probs.idxmax(axis=1).values
    else:
        # For numpy array, get the index of maximum probability
        alternatives = [str(i+1) for i in range(predicted_probs.shape[1])]
        predicted_choices = np.argmax(predicted_probs, axis=1)
        # If actual_choices are 1-indexed but predictions are 0-indexed, adjust
        if np.min(actual_choices) == 1 and np.min(predicted_choices) == 0:
            predicted_choices += 1
            alternatives = [str(i) for i in range(1, predicted_probs.shape[1] + 1)]
    
    # Create labels for the confusion matrix
    unique_actual = np.unique(actual_choices)
    unique_predicted = np.unique(predicted_choices)
    all_labels = sorted(set(unique_actual) | set(unique_predicted))
    
    # Initialize confusion matrix
    confusion = np.zeros((len(all_labels), len(all_labels)), dtype=int)
    
    # Fill confusion matrix
    for i, true_label in enumerate(all_labels):
        for j, pred_label in enumerate(all_labels):
            confusion[i, j] = np.sum((actual_choices == true_label) & (predicted_choices == pred_label))
    
    # Convert to DataFrame with proper labels
    confusion_df = pd.DataFrame(
        confusion,
        index=[f'Actual {label}' for label in all_labels],
        columns=[f'Predicted {label}' for label in all_labels]
    )
    
    return confusion_df

def calculate_value_of_time(time_coeff: float, cost_coeff: float, 
                          time_unit: str = 'minute', cost_unit: str = 'currency') -> float:
    """
    Calculate the Value of Time (VOT).
    
    Parameters
    ----------
    time_coeff : float
        Time coefficient from the model
    cost_coeff : float
        Cost coefficient from the model
    time_unit : str, default='minute'
        Unit of time in the model (minute, hour)
    cost_unit : str, default='currency'
        Unit of cost in the model
    
    Returns
    -------
    float
        Value of Time (cost per hour)
    """
    # Calculate the ratio of coefficients
    vot_ratio = abs(time_coeff / cost_coeff)
    
    # Convert to standard units (cost per hour)
    if time_unit.lower() == 'minute':
        vot = vot_ratio * 60  # Convert to cost per hour
    elif time_unit.lower() == 'hour':
        vot = vot_ratio  # Already in cost per hour
    else:
        raise ValueError(f"Unsupported time unit: {time_unit}")
    
    return vot

def calculate_elasticity(model: Any, variable: str, alternative: int, 
                       data: Optional[pd.DataFrame] = None,
                       percent_change: float = 0.01) -> float:
    """
    Calculate the elasticity of demand with respect to a variable.
    
    Parameters
    ----------
    model : Any
        Model object with predict_probabilities method
    variable : str
        Name of the variable to calculate elasticity for
    alternative : int
        Alternative to calculate elasticity for
    data : pd.DataFrame, optional
        Data to use for calculation. If None, uses the model's data.
    percent_change : float, default=0.01
        Percentage change in the variable (0.01 = 1%)
    
    Returns
    -------
    float
        Elasticity value
    """
    if data is None:
        # Use model's data if available
        if hasattr(model, 'database') and hasattr(model.database, 'data'):
            data = model.database.data
        else:
            raise ValueError("Data must be provided if model doesn't have data attribute")
    
    # Make a copy of the data
    data_modified = data.copy()
    
    # Calculate baseline probabilities
    baseline_probs = model.predict_probabilities(data)
    baseline_share = baseline_probs[str(alternative)].mean()
    
    # Modify the variable by the specified percentage
    data_modified[variable] = data_modified[variable] * (1 + percent_change)
    
    # Calculate new probabilities
    new_probs = model.predict_probabilities(data_modified)
    new_share = new_probs[str(alternative)].mean()
    
    # Calculate percentage change in share
    share_change = (new_share - baseline_share) / baseline_share
    
    # Calculate elasticity (% change in share / % change in variable)
    elasticity = share_change / percent_change
    
    return elasticity

def calculate_cross_elasticities(model: Any, variable: str, 
                               data: Optional[pd.DataFrame] = None,
                               percent_change: float = 0.01) -> Dict[int, Dict[int, float]]:
    """
    Calculate the cross-elasticities of demand with respect to a variable.
    
    Parameters
    ----------
    model : Any
        Model object with predict_probabilities method
    variable : str
        Name of the variable to calculate elasticity for
    data : pd.DataFrame, optional
        Data to use for calculation. If None, uses the model's data.
    percent_change : float, default=0.01
        Percentage change in the variable (0.01 = 1%)
    
    Returns
    -------
    Dict[int, Dict[int, float]]
        Dictionary mapping (alt_i, alt_j) to cross-elasticity value
    """
    if data is None:
        # Use model's data if available
        if hasattr(model, 'database') and hasattr(model.database, 'data'):
            data = model.database.data
        else:
            raise ValueError("Data must be provided if model doesn't have data attribute")
    
    # Get unique alternatives
    baseline_probs = model.predict_probabilities(data)
    alternatives = [int(col) for col in baseline_probs.columns]
    
    # Calculate cross-elasticities for each pair of alternatives
    cross_elasticities = {}
    
    for alt_i in alternatives:
        cross_elasticities[alt_i] = {}
        
        # Make a copy of the data
        data_modified = data.copy()
        
        # Modify the variable for alternative i by the specified percentage
        var_i = f"{variable}_{alt_i}"  # Assume variable names follow this pattern
        if var_i in data_modified.columns:
            data_modified[var_i] = data_modified[var_i] * (1 + percent_change)
            
            # Calculate baseline shares
            baseline_shares = {alt: baseline_probs[str(alt)].mean() for alt in alternatives}
            
            # Calculate new probabilities
            new_probs = model.predict_probabilities(data_modified)
            new_shares = {alt: new_probs[str(alt)].mean() for alt in alternatives}
            
            # Calculate cross-elasticities
            for alt_j in alternatives:
                if alt_i != alt_j:
                    # Calculate percentage change in share of alt_j
                    share_change = (new_shares[alt_j] - baseline_shares[alt_j]) / baseline_shares[alt_j]
                    
                    # Calculate cross-elasticity
                    cross_elasticities[alt_i][alt_j] = share_change / percent_change
    
    return cross_elasticities

def calculate_log_likelihood(actual_choices: Union[np.ndarray, pd.Series],
                           predicted_probs: Union[np.ndarray, pd.DataFrame]) -> float:
    """
    Calculate the log-likelihood of the model.
    
    Parameters
    ----------
    actual_choices : np.ndarray or pd.Series
        Actual choices made by individuals
    predicted_probs : np.ndarray or pd.DataFrame
        Predicted probabilities for each alternative
    
    Returns
    -------
    float
        Log-likelihood value
    """
    # Convert pandas objects to numpy if needed
    if isinstance(actual_choices, pd.Series):
        actual_choices = actual_choices.values
    
    if isinstance(predicted_probs, pd.DataFrame):
        # Convert DataFrame to probabilities array
        prob_array = predicted_probs.values
        # Adjust actual choices to be 0-indexed if needed
        choice_idx = np.array([np.where(predicted_probs.columns == str(choice))[0][0] 
                              for choice in actual_choices])
    else:
        prob_array = predicted_probs
        # Adjust actual choices to be 0-indexed
        if np.min(actual_choices) == 1:
            choice_idx = actual_choices - 1
        else:
            choice_idx = actual_choices
    
    # Get probability of chosen alternative for each observation
    chosen_probs = prob_array[np.arange(len(choice_idx)), choice_idx]
    
    # Calculate log-likelihood
    log_like = np.sum(np.log(chosen_probs))
    
    return log_like

def calculate_null_log_likelihood(actual_choices: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate the null log-likelihood (equal probabilities for all alternatives).
    
    Parameters
    ----------
    actual_choices : np.ndarray or pd.Series
        Actual choices made by individuals
    
    Returns
    -------
    float
        Null log-likelihood value
    """
    # Convert pandas objects to numpy if needed
    if isinstance(actual_choices, pd.Series):
        actual_choices = actual_choices.values
    
    # Get unique alternatives
    unique_alts = np.unique(actual_choices)
    n_alternatives = len(unique_alts)
    
    # Equal probability for each alternative
    equal_prob = 1.0 / n_alternatives
    
    # Calculate null log-likelihood
    null_ll = len(actual_choices) * np.log(equal_prob)
    
    return null_ll