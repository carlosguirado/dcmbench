"""
PyTorchModelAdapter module for DCMBench.

This module provides adapter classes that make PyTorch models compatible with DCMBench
benchmarking by implementing the required prediction interface. It allows users to
create models using PyTorch and then use them with DCMBench without writing boilerplate
prediction code.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Union, Optional, Any, List, Tuple, Callable

from ..model_benchmarker.model_interface import PredictionInterface


class PyTorchModelAdapter:
    """
    Adapter class that makes PyTorch models compatible with DCMBench benchmarking.
    
    This class takes a trained PyTorch model and implements the prediction interface 
    required by the DCMBench benchmarker, allowing users to create models using PyTorch
    and use them with DCMBench without writing boilerplate code.
    
    Parameters
    ----------
    model : torch.nn.Module
        The trained PyTorch model
    n_alternatives : int, optional
        Number of choice alternatives. If None, will be inferred from model.
    alternative_ids : List[Union[str, int]], optional
        Identifiers for the alternatives. If None, will use indices starting from 1.
    preprocess_func : Callable, optional
        Function to preprocess DataFrame to tensors before prediction. 
        If None, a default conversion will be attempted.
    device : str, optional
        Device to use for computation ("cpu" or "cuda"). 
        If None, will use model's device or fallback to "cpu".
    name : str, optional
        Custom name for the model
    choice_column : str, default="CHOICE"
        Name of the column containing the chosen alternatives
    """
    
    def __init__(
        self,
        model: nn.Module,
        n_alternatives: Optional[int] = None,
        alternative_ids: Optional[List[Union[str, int]]] = None,
        preprocess_func: Optional[Callable] = None,
        device: Optional[str] = None,
        name: Optional[str] = None,
        choice_column: str = "CHOICE"
    ):
        self.model = model
        self.name = name or "PyTorchModel"
        self.choice_column = choice_column
        self.preprocess_func = preprocess_func
        
        # Set the device
        if device is not None:
            self.device = torch.device(device)
        elif hasattr(model, 'device'):
            self.device = model.device
        else:
            # Try to find the device from model parameters
            try:
                self.device = next(model.parameters()).device
            except (StopIteration, AttributeError):
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move model to the specified device if needed
        self.model = self.model.to(self.device)
        
        # Set the model to evaluation mode
        self.model.eval()
        
        # Determine the number of alternatives
        self.n_alternatives = n_alternatives
        if self.n_alternatives is None:
            # Try to infer from model attributes
            if hasattr(model, 'n_alternatives'):
                self.n_alternatives = model.n_alternatives
            elif hasattr(model, 'n_classes_'):
                self.n_alternatives = model.n_classes_
            elif hasattr(model, 'n_outputs'):
                self.n_alternatives = model.n_outputs
        
        # Set the alternative IDs
        self.alternative_ids = alternative_ids
        if self.alternative_ids is None and self.n_alternatives is not None:
            # Use default IDs (1-indexed)
            self.alternative_ids = [str(i+1) for i in range(self.n_alternatives)]
        
        # Initialize metrics
        self.choice_accuracy = None
        self.market_share_accuracy = None
        self.actual_shares = None
        self.predicted_shares = None
        
    def _preprocess_data(self, data: pd.DataFrame) -> torch.Tensor:
        """
        Preprocess data from DataFrame to PyTorch tensor.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data for prediction
            
        Returns
        -------
        torch.Tensor
            Preprocessed data as tensor ready for model input
        """
        if self.preprocess_func is not None:
            # Use the provided preprocessing function
            return self.preprocess_func(data).to(self.device)
        
        # Default preprocessing - convert DataFrame to tensor
        try:
            # Try to extract features (exclude the choice column if present)
            if self.choice_column in data.columns:
                features = data.drop(columns=[self.choice_column]).values
            else:
                features = data.values
                
            # Convert to tensor
            return torch.tensor(features, dtype=torch.float32).to(self.device)
        except Exception as e:
            raise ValueError(f"Failed to preprocess data: {e}. "
                            "Please provide a preprocess_func suitable for your model.")
    
    def predict_probabilities(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict choice probabilities for each alternative.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data for prediction
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns for each alternative and rows for each observation,
            containing the predicted probability of each alternative for each observation.
            Column names match the alternative identifiers.
        """
        # Prepare the input tensor
        input_tensor = self._preprocess_data(data)
        
        # Get model predictions
        with torch.no_grad():
            output = self.model(input_tensor)
            
            # Convert to probabilities if needed
            if hasattr(self.model, 'predict_proba'):
                # Model has a dedicated prediction method
                probabilities = self.model.predict_proba(input_tensor)
            elif output.dim() == 2 and output.size(1) > 1:
                # Check if output is logits or log probabilities
                if torch.allclose(torch.sum(torch.exp(output), dim=1), 
                                 torch.ones(output.size(0), device=self.device),
                                 rtol=1e-3, atol=1e-3):
                    # Already log probabilities
                    probabilities = torch.exp(output)
                else:
                    # Logits - convert to probabilities
                    probabilities = F.softmax(output, dim=1)
            else:
                # Unknown format - try to interpret as probabilities
                probabilities = output
        
        # Convert to numpy
        probs_np = probabilities.cpu().numpy()
        
        # Create DataFrame with proper column names
        if self.alternative_ids is None:
            # If n_alternatives wasn't set, infer from output shape
            if probs_np.ndim == 2:
                n_alternatives = probs_np.shape[1]
                self.alternative_ids = [str(i+1) for i in range(n_alternatives)]
            else:
                raise ValueError("Could not determine alternative IDs from model output. "
                               "Please provide alternative_ids explicitly.")
        
        # Create the DataFrame
        probs_df = pd.DataFrame(
            probs_np,
            columns=self.alternative_ids
        )
        
        return probs_df
    
    def predict_choices(self, data: pd.DataFrame) -> pd.Series:
        """
        Predict the most likely choice for each observation.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data for prediction
            
        Returns
        -------
        pd.Series
            Series containing the predicted choice for each observation.
            Values are the alternative identifiers.
        """
        # If the model has a direct predict method, use it
        if hasattr(self.model, 'predict') and callable(getattr(self.model, 'predict')):
            try:
                input_tensor = self._preprocess_data(data)
                with torch.no_grad():
                    predictions = self.model.predict(input_tensor)
                
                # Convert tensor predictions to numpy/series
                if isinstance(predictions, torch.Tensor):
                    predictions = predictions.cpu().numpy()
                
                # Convert to pandas Series with proper indices
                if isinstance(predictions, np.ndarray):
                    # Add 1 to convert from 0-indexed to 1-indexed if needed
                    if np.all(predictions < len(self.alternative_ids)):
                        predictions = pd.Series([self.alternative_ids[int(p)] for p in predictions])
                    else:
                        predictions = pd.Series(predictions)
                
                return predictions
            except Exception:
                # Fall back to probability-based prediction
                pass
        
        # Get probabilities and select most likely alternative
        probs = self.predict_probabilities(data)
        return probs.idxmax(axis=1)
    
    def calculate_choice_accuracy(self, data: pd.DataFrame = None) -> Dict[str, float]:
        """
        Calculate choice prediction accuracy and market shares.
        
        Parameters
        ----------
        data : pd.DataFrame, optional
            Data to calculate metrics for. If None, accuracy cannot be calculated.
            
        Returns
        -------
        Dict[str, float]
            Dictionary with accuracy metrics including choice_accuracy and market_share_accuracy
        """
        if data is None or self.choice_column not in data.columns:
            raise ValueError(f"Cannot calculate accuracy without data containing the choice column '{self.choice_column}'")
            
        # Get actual choices
        actual_choices = data[self.choice_column]
        
        # Calculate predicted choices
        predicted_choices = self.predict_choices(data)
        
        # Ensure predicted choices has the same index as actual choices
        if isinstance(predicted_choices, pd.Series):
            predicted_choices = predicted_choices.reset_index(drop=True)
        else:
            predicted_choices = pd.Series(predicted_choices)
        predicted_choices.index = actual_choices.index
        
        # Convert string indices to proper type if needed
        if predicted_choices.dtype == object and actual_choices.dtype != object:
            try:
                predicted_choices = predicted_choices.astype(actual_choices.dtype)
            except (ValueError, TypeError):
                # Handle case where alternative IDs are strings that can't be converted
                # Convert actual_choices to strings instead
                actual_choices = actual_choices.astype(str)
                
        # Calculate choice accuracy
        self.choice_accuracy = (actual_choices == predicted_choices).mean()
        
        # Calculate market shares
        self.actual_shares = actual_choices.value_counts(normalize=True).to_dict()
        
        # Calculate predicted market shares
        probs = self.predict_probabilities(data)
        self.predicted_shares = {str(col): probs[col].mean() for col in probs.columns}
        
        # Calculate market share accuracy (mean absolute error normalized)
        total_abs_error = sum(
            abs(self.actual_shares.get(str(alt), 0) - self.predicted_shares.get(str(alt), 0))
            for alt in set(str(k) for k in self.actual_shares.keys()) | set(self.predicted_shares.keys())
        )
        self.market_share_accuracy = 1 - (total_abs_error / 2)  # Divide by 2 to normalize
        
        # Calculate Value of Time (VOT) if possible
        vot = None
        try:
            # For PyTorch models, we need to access the beta parameters
            if hasattr(self.model, 'beta'):
                beta_values = self.model.beta.data.cpu().numpy()
                # Usually first coefficient is time, second is cost
                if len(beta_values) >= 2:
                    b_time = beta_values[0]  # Assuming first coefficient is time
                    b_cost = beta_values[1]  # Assuming second coefficient is cost
                    
                    if b_cost != 0:
                        # VOT = -B_TIME / B_COST
                        vot = (b_time / b_cost) * 60
                        self.value_of_time = {
                            'value': vot,
                            'time_coefficient': b_time,
                            'cost_coefficient': b_cost,
                            'units': 'CHF/h'
                        }
        except Exception as e:
            # If VOT calculation fails, we continue without it
            pass
        
        result = {
            'choice_accuracy': self.choice_accuracy,
            'market_share_accuracy': self.market_share_accuracy
        }
        
        if hasattr(self, 'value_of_time'):
            result['value_of_time'] = self.value_of_time
        
        return result
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Return model metrics for benchmarking.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary of model metrics including parameters and accuracy metrics if available
        """
        metrics = {
            'name': self.name,
            'framework': 'pytorch',
            'n_alternatives': self.n_alternatives,
        }
        
        # Include parameters if available
        if hasattr(self.model, 'state_dict'):
            try:
                params = {}
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        params[name] = param.data.cpu().numpy()
                metrics['parameters'] = params
                metrics['n_parameters'] = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            except Exception:
                # Skip if parameters can't be extracted
                pass
        
        # Add metrics if already calculated
        if hasattr(self, 'choice_accuracy') and self.choice_accuracy is not None:
            metrics['choice_accuracy'] = self.choice_accuracy
            
        if hasattr(self, 'market_share_accuracy') and self.market_share_accuracy is not None:
            metrics['market_share_accuracy'] = self.market_share_accuracy
            
        if hasattr(self, 'actual_shares') and self.actual_shares is not None:
            metrics['actual_shares'] = self.actual_shares
            
        if hasattr(self, 'predicted_shares') and self.predicted_shares is not None:
            metrics['predicted_shares'] = self.predicted_shares
            
        # Include Value of Time if available
        if hasattr(self, 'value_of_time'):
            metrics['value_of_time'] = self.value_of_time['value']
            metrics['vot_details'] = self.value_of_time
            
        # Include log likelihood and other statistics if available
        for attr in ['final_ll', 'null_ll', 'rho_squared', 'rho_squared_bar']:
            if hasattr(self.model, attr):
                metrics[attr] = getattr(self.model, attr)
            elif hasattr(self, attr):
                metrics[attr] = getattr(self, attr)
        
        return metrics
    
    def get_parameter_values(self) -> Dict[str, Any]:
        """
        Get the estimated parameter values.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary of parameter names and values
        """
        params = {}
        if hasattr(self.model, 'state_dict'):
            try:
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        if param.numel() == 1:
                            # Single scalar parameter
                            params[name] = param.item()
                        else:
                            # Vector or matrix of parameters
                            params[name] = param.data.cpu().numpy()
            except Exception as e:
                raise RuntimeError(f"Failed to extract parameters: {e}")
                
        return params
    
    def get_model_name(self) -> str:
        """
        Get the model name.
        
        Returns
        -------
        str
            Model name
        """
        return self.name


class PyTorchMultinomialLogitAdapter(PyTorchModelAdapter):
    """
    Adapter specifically for PyTorch Multinomial Logit models.
    
    This adapter is designed for PyTorch models that implement the MNL structure
    with beta parameters for features and alternative-specific constants (ASC).
    It provides direct access to these parameters for analysis.
    
    Parameters
    ----------
    model : torch.nn.Module
        The trained PyTorch MNL model. Expected to have 'beta' and 'asc' parameters.
    n_alternatives : int, optional
        Number of choice alternatives. If None, will be inferred from model.
    alternative_ids : List[Union[str, int]], optional
        Identifiers for the alternatives. If None, will use indices starting from 1.
    preprocess_func : Callable, optional
        Function to preprocess DataFrame to tensors before prediction
    device : str, optional
        Device to use for computation ("cpu" or "cuda")
    name : str, optional
        Custom name for the model
    choice_column : str, default="CHOICE"
        Name of the column containing the chosen alternatives
    """
    
    def __init__(
        self,
        model: nn.Module,
        n_alternatives: Optional[int] = None,
        alternative_ids: Optional[List[Union[str, int]]] = None,
        preprocess_func: Optional[Callable] = None,
        device: Optional[str] = None,
        name: Optional[str] = None,
        choice_column: str = "CHOICE"
    ):
        # Verify this is an MNL model
        if not (hasattr(model, 'beta') and (hasattr(model, 'asc') or hasattr(model, 'bias'))):
            raise ValueError("Model does not appear to be an MNL model with 'beta' and 'asc' parameters")
            
        super().__init__(
            model=model,
            n_alternatives=n_alternatives,
            alternative_ids=alternative_ids,
            preprocess_func=preprocess_func,
            device=device,
            name=name or "PyTorchMultinomialLogit",
            choice_column=choice_column
        )
        
        # Store specific MNL parameters
        self.beta = self.model.beta.data.cpu().numpy() if hasattr(self.model, 'beta') else None
        
        # Get ASC parameters (could be named 'asc' or 'bias')
        if hasattr(self.model, 'asc'):
            self.asc = self.model.asc.data.cpu().numpy()
        elif hasattr(self.model, 'bias'):
            self.asc = self.model.bias.data.cpu().numpy()
        else:
            self.asc = None
    
    def get_mnl_parameters(self) -> Dict[str, np.ndarray]:
        """
        Get MNL-specific parameters (beta coefficients and ASCs).
        
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary with 'beta' and 'asc' parameters
        """
        return {
            'beta': self.beta,
            'asc': self.asc
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Return model metrics including MNL-specific parameters.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary of model metrics including beta and ASC parameters
        """
        metrics = super().get_metrics()
        
        # Add MNL-specific parameters
        metrics['beta'] = self.beta
        metrics['asc'] = self.asc
        metrics['model_type'] = 'multinomial_logit'
        
        return metrics