"""
PyTorch-specific base classes for discrete choice models.

This module contains base classes for implementing discrete choice models
using the PyTorch framework.
"""

import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Dict, List, Optional, Tuple, Union, Any

from .base import BaseDiscreteChoiceModel

logger = logging.getLogger(__name__)

class PyTorchDiscreteChoiceModel(BaseDiscreteChoiceModel):
    """
    Base class for PyTorch-based discrete choice models.
    
    This class extends BaseDiscreteChoiceModel with PyTorch-specific
    functionality for model estimation and prediction.
    
    Attributes
    ----------
    model : torch.nn.Module
        PyTorch model for utility calculation
    optimizer : torch.optim.Optimizer
        Optimizer for model training
    device : torch.device
        Device to use for computation (CPU/GPU)
    training_config : dict
        Configuration for model training
    """
    
    def __init__(self, data, auto_preprocess=True, preprocessing_options=None, debug=False):
        """
        Initialize the PyTorch model.
        
        Parameters
        ----------
        data : pandas.DataFrame
            Input data for the model
        auto_preprocess : bool, default=True
            Whether to automatically preprocess the data
        preprocessing_options : dict, optional
            Options for data preprocessing
        debug : bool, default=False
            Whether to print debug information during initialization
        """
        super().__init__(data, auto_preprocess, preprocessing_options, debug)
        
    def _init_framework_attributes(self):
        """Initialize PyTorch-specific attributes."""
        super()._init_framework_attributes()
        
        # Setup PyTorch attributes
        self.model = None
        self.optimizer = None
        self.loss_fn = F.nll_loss  # Default loss function
        self.training_history = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_tensor = None
        self.choice_tensor = None
        self.choice_column = 'CHOICE'  # Default choice column name
        
        # Default training configuration
        self.training_config = {
            'batch_size': 32,
            'epochs': 100,
            'learning_rate': 0.01,
            'weight_decay': 1e-5,
            'patience': 10,  # For early stopping
            'optimizer': 'adam'
        }
        
    def _prepare_tensors(self, data=None):
        """
        Prepare PyTorch tensors from DataFrame.
        
        Parameters
        ----------
        data : pandas.DataFrame, optional
            Data to convert to tensors. If None, uses the data from initialization.
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Feature tensor and choice tensor
        """
        if data is None:
            data = self.database.data
        
        # Implementation depends on model type and data structure
        # This is a simple implementation that will need to be overridden by subclasses
        
        # For a simple MNL model, we might have features for each alternative
        # and a choice column indicating the chosen alternative
        
        # Default implementation assumes data is already formatted for PyTorch
        # with features in columns and choice in a separate column
        features = data.drop(columns=[self.choice_column]).values
        choices = data[self.choice_column].values - 1  # Convert to 0-indexed
        
        # Convert to PyTorch tensors
        feature_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
        choice_tensor = torch.tensor(choices, dtype=torch.long).to(self.device)
        
        return feature_tensor, choice_tensor
    
    def _create_model(self):
        """
        Create the PyTorch model.
        
        This method should be implemented by subclasses to create
        the appropriate PyTorch model architecture.
        
        Returns
        -------
        torch.nn.Module
            PyTorch model
        """
        raise NotImplementedError("Subclasses must implement _create_model")
    
    def _setup_optimizer(self):
        """
        Create the optimizer for model training.
        
        Returns
        -------
        torch.optim.Optimizer
            PyTorch optimizer
        """
        if self.model is None:
            raise ValueError("Model must be created before setting up optimizer")
        
        optimizer_type = self.training_config.get('optimizer', 'adam').lower()
        lr = self.training_config.get('learning_rate', 0.01)
        weight_decay = self.training_config.get('weight_decay', 1e-5)
        
        if optimizer_type == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
        elif optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(
                self.model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
        elif optimizer_type == 'lbfgs':
            optimizer = torch.optim.LBFGS(
                self.model.parameters(),
                lr=lr
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
            
        return optimizer
    
    def _train_epoch(self, data_loader):
        """
        Train the model for one epoch.
        
        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            DataLoader providing batches of data
            
        Returns
        -------
        float
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0
        
        for features, choices in data_loader:
            features = features.to(self.device)
            choices = choices.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            log_probs = self.model(features)
            loss = self.loss_fn(log_probs, choices)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        # Return average loss
        return total_loss / len(data_loader)
    
    def _validate_model(self, data_loader):
        """
        Validate the model on a validation set.
        
        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            DataLoader providing batches of validation data
            
        Returns
        -------
        dict
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, choices in data_loader:
                features = features.to(self.device)
                choices = choices.to(self.device)
                
                # Forward pass
                log_probs = self.model(features)
                loss = self.loss_fn(log_probs, choices)
                
                # Calculate accuracy
                pred = log_probs.argmax(dim=1)
                correct += (pred == choices).sum().item()
                total += choices.size(0)
                
                total_loss += loss.item()
        
        # Calculate metrics
        validation_loss = total_loss / len(data_loader)
        accuracy = correct / total
        
        return {
            'validation_loss': validation_loss,
            'accuracy': accuracy
        }
    
    def estimate(self):
        """
        Estimate model parameters using PyTorch training.
        
        Returns
        -------
        dict
            Training results and metrics
        """
        # Prepare data tensors if not already prepared
        if self.feature_tensor is None or self.choice_tensor is None:
            self.feature_tensor, self.choice_tensor = self._prepare_tensors()
        
        # Create model if not already created
        if self.model is None:
            self.model = self._create_model()
            self.model.to(self.device)
        
        # Create optimizer if not already created
        if self.optimizer is None:
            self.optimizer = self._setup_optimizer()
        
        # Create dataset and data loader
        dataset = torch.utils.data.TensorDataset(self.feature_tensor, self.choice_tensor)
        batch_size = self.training_config.get('batch_size', 32)
        data_loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=min(batch_size, len(dataset)), 
            shuffle=True
        )
        
        # Training loop
        epochs = self.training_config.get('epochs', 100)
        patience = self.training_config.get('patience', 10)
        best_loss = float('inf')
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(epochs):
            # Train for one epoch
            train_loss = self._train_epoch(data_loader)
            
            # Validate the model
            valid_metrics = self._validate_model(data_loader)
            valid_loss = valid_metrics['validation_loss']
            accuracy = valid_metrics['accuracy']
            
            # Store metrics
            self.training_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'accuracy': accuracy
            })
            
            # Early stopping
            if valid_loss < best_loss:
                best_loss = valid_loss
                patience_counter = 0
                # Save best model state
                self.best_model_state = {
                    key: value.cpu().clone() 
                    for key, value in self.model.state_dict().items()
                }
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        # Calculate final metrics
        self.training_time = time.time() - start_time
        
        # Restore best model
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
        
        # Put model in evaluation mode
        self.model.eval()
        
        # Calculate final log likelihood
        with torch.no_grad():
            log_probs = self.model(self.feature_tensor)
            final_nll = self.loss_fn(log_probs, self.choice_tensor, reduction='sum').item()
            self.final_ll = -final_nll
        
        # Calculate null model log-likelihood (equiprobable choices)
        num_choices = log_probs.size(1)
        null_probs = torch.ones(self.choice_tensor.size(0), num_choices) / num_choices
        null_nll = -torch.sum(torch.log(null_probs[torch.arange(self.choice_tensor.size(0)), self.choice_tensor])).item()
        
        # Calculate rho-squared measures
        self.rho_squared = 1 - self.final_ll / null_nll
        
        # Number of parameters
        n_params = sum(p.numel() for p in self.model.parameters())
        self.rho_squared_bar = 1 - (self.final_ll - n_params) / null_nll
        
        # Calculate choice accuracy and market shares
        self.calculate_choice_accuracy()
        
        # Create results dictionary similar to Biogeme's results
        self.results = self._create_results_dict(n_params)
        
        return self.results
    
    def _create_results_dict(self, n_params):
        """
        Create a results dictionary with model parameters and metrics.
        
        Parameters
        ----------
        n_params : int
            Number of model parameters
            
        Returns
        -------
        dict
            Results dictionary with model parameters and metrics
        """
        # Get parameter values
        params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params[name] = param.data.cpu().numpy()
        
        # Create a simple results object with necessary methods
        class ResultsObject:
            def __init__(self, params, metrics, n_params, n_obs):
                self.params = params
                self.metrics = metrics
                self.data = type('obj', (object,), {
                    'betaValues': params,
                    'numberOfObservations': n_obs,
                    'finalLogLikelihood': metrics['final_ll'],
                    'nullLogLikelihood': metrics['null_ll']
                })
                self.n_params = n_params
                
            def get_beta_values(self):
                return self.params
                
            def getGeneralStatistics(self):
                return {
                    'Final log likelihood': [self.metrics['final_ll']],
                    'Null log likelihood': [self.metrics['null_ll']],
                    'Rho-square for the null model': [self.metrics['rho_squared']],
                    'Rho-square-bar for the null model': [self.metrics['rho_squared_bar']]
                }
                
            def __str__(self):
                return f"PyTorch Model Results (params: {self.n_params}, obs: {self.data.numberOfObservations})"
        
        # Create metrics dictionary
        metrics = {
            'final_ll': self.final_ll,
            'null_ll': -null_nll if 'null_nll' in locals() else None,
            'rho_squared': self.rho_squared,
            'rho_squared_bar': self.rho_squared_bar,
            'training_time': self.training_time,
            'training_history': self.training_history,
            'choice_accuracy': self.choice_accuracy,
            'market_share_accuracy': self.market_share_accuracy,
        }
        
        # Return results object
        return ResultsObject(params, metrics, n_params, len(self.choice_tensor))
    
    def predict_probabilities(self, data=None):
        """
        Calculate choice probabilities for each alternative.
        
        Parameters
        ----------
        data : pandas.DataFrame, optional
            Data to make predictions for. If None, uses the data from initialization.
            
        Returns
        -------
        pandas.DataFrame
            DataFrame containing probabilities for each alternative
        """
        if self.model is None:
            raise RuntimeError("Model must be estimated before making predictions")
            
        # Prepare tensors for prediction
        if data is not None:
            # Preprocess new data if needed
            if getattr(self, "auto_preprocess", True):
                processed_data = self.preprocess_data(data)
            else:
                processed_data = data
                
            # Create tensors from the new data
            features, _ = self._prepare_tensors(processed_data)
        else:
            # Use existing tensors
            features = self.feature_tensor
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            log_probs = self.model(features)
            probabilities = torch.exp(log_probs)
        
        # Convert to DataFrame
        probs_np = probabilities.cpu().numpy()
        probs_df = pd.DataFrame(
            probs_np,
            columns=[str(i+1) for i in range(probs_np.shape[1])]
        )
        
        return probs_df
    
    def predict_choices(self, data=None):
        """
        Predict the most likely choice for each observation.
        
        Parameters
        ----------
        data : pandas.DataFrame, optional
            Data to make predictions for. If None, uses the data from initialization.
            
        Returns
        -------
        pandas.Series
            Series of predicted choices (1-indexed)
        """
        probabilities = self.predict_probabilities(data)
        
        # Get most likely alternative (add 1 to convert from 0-indexed to 1-indexed)
        choices = probabilities.idxmax(axis=1).astype(int)
        
        return choices
    
    def calculate_choice_accuracy(self, data=None):
        """
        Calculate choice prediction accuracy and market shares.
        
        Parameters
        ----------
        data : pandas.DataFrame, optional
            Data to calculate metrics for. If None, uses the data from initialization.
        """
        # Get choice variable
        choice_col = self.choice_column
        
        # Use original data if no new data is provided
        if data is None:
            actual_choices = self.database.data[choice_col]
            probabilities = self.predict_probabilities()
        else:
            # Preprocess new data if needed
            if getattr(self, "auto_preprocess", True):
                data_processed = self.preprocess_data(data)
            else:
                data_processed = data
                
            actual_choices = data_processed[choice_col]
            probabilities = self.predict_probabilities(data_processed)
        
        # Calculate actual shares
        actual_counts = actual_choices.value_counts(normalize=True)
        self.actual_shares = actual_counts.to_dict()
        
        # Calculate predicted shares
        self.predicted_shares = {
            int(alt): probabilities[alt].mean() 
            for alt in probabilities.columns
        }
        
        # Calculate market share accuracy
        total_abs_error = sum(
            abs(self.actual_shares.get(int(alt), 0) - self.predicted_shares.get(int(alt), 0))
            for alt in set(self.actual_shares.keys()) | set(self.predicted_shares.keys())
        )
        self.market_share_accuracy = 1 - (total_abs_error / 2)  # Divide by 2 to normalize
        
        # Calculate choice accuracy - predicted vs actual choices
        predicted_choices = probabilities.idxmax(axis=1).astype(int)
        
        prediction_data = {
            'actual': actual_choices,
            'predicted': predicted_choices
        }
        df_pred = pd.DataFrame(prediction_data)
        
        # Create confusion matrix
        self.confusion_matrix = pd.crosstab(
            df_pred['actual'], 
            df_pred['predicted'], 
            rownames=['Actual'], 
            colnames=['Predicted']
        )
        
        # Calculate accuracy
        self.choice_accuracy = (df_pred['actual'] == df_pred['predicted']).mean()
        
        logger.info(f"Market Share Accuracy: {self.market_share_accuracy:.4f}")
        logger.info(f"Choice Accuracy: {self.choice_accuracy:.4f}")
        
        return {
            'market_share_accuracy': self.market_share_accuracy,
            'choice_accuracy': self.choice_accuracy
        }
    
    def get_parameter_values(self):
        """
        Get the estimated parameter values.
        
        Returns
        -------
        dict
            Dictionary of parameter names and values
        """
        if self.model is None:
            raise RuntimeError("Model must be estimated before getting parameter values")
        
        params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params[name] = param.data.cpu().numpy()
                
        return params


class MultinomialLogitPyTorch(PyTorchDiscreteChoiceModel):
    """
    PyTorch implementation of Multinomial Logit model.
    
    This class provides a simple MNL model implemented in PyTorch.
    """
    
    def __init__(self, data, auto_preprocess=True, preprocessing_options=None, 
                 debug=False, n_features=None, n_alternatives=None):
        """
        Initialize the PyTorch MNL model.
        
        Parameters
        ----------
        data : pandas.DataFrame
            Input data for the model
        auto_preprocess : bool, default=True
            Whether to automatically preprocess the data
        preprocessing_options : dict, optional
            Options for data preprocessing
        debug : bool, default=False
            Whether to print debug information during initialization
        n_features : int, optional
            Number of features per alternative (if None, inferred from data)
        n_alternatives : int, optional
            Number of choice alternatives (if None, inferred from data)
        """
        super().__init__(data, auto_preprocess, preprocessing_options, debug)
        
        # Model architecture configuration
        self.n_features = n_features
        self.n_alternatives = n_alternatives
        
    def _infer_dimensions(self):
        """
        Infer model dimensions from data.
        
        Returns
        -------
        Tuple[int, int]
            Number of features and number of alternatives
        """
        if self.choice_tensor is None:
            self.feature_tensor, self.choice_tensor = self._prepare_tensors()
            
        # Infer number of alternatives from choice column
        n_alternatives = len(self.database.data[self.choice_column].unique())
        
        # Infer number of features from data shape
        if self.feature_tensor.dim() == 3:
            # Data is shaped (batch_size, n_alternatives, n_features)
            _, _, n_features = self.feature_tensor.shape
        else:
            # Data is shaped (batch_size, n_features)
            # We'll assume features are already processed for all alternatives
            _, n_features = self.feature_tensor.shape
            n_features = n_features // n_alternatives
            
        return n_features, n_alternatives
    
    def _create_model(self):
        """
        Create the PyTorch MNL model.
        
        Returns
        -------
        torch.nn.Module
            PyTorch MNL model
        """
        # Infer dimensions if not provided
        if self.n_features is None or self.n_alternatives is None:
            self.n_features, self.n_alternatives = self._infer_dimensions()
            
        # Create model
        model = MNLModel(self.n_features, self.n_alternatives)
        
        return model
    
    def _prepare_tensors(self, data=None):
        """
        Prepare PyTorch tensors for the MNL model.
        
        Parameters
        ----------
        data : pandas.DataFrame, optional
            Data to convert to tensors. If None, uses the data from initialization.
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Feature tensor and choice tensor
        """
        if data is None:
            data = self.database.data
        
        # We need to reshape the data for the MNL model:
        # Each row should contain features for all alternatives
        
        # Assuming data is structured with alternative-specific variables
        # like: TRAIN_TT, TRAIN_CO, SM_TT, SM_CO, CAR_TT, CAR_CO
        # or columns are already organized by alternative
        
        # This is a simplified version that needs to be adapted based on actual data structure
        if 'TRAIN_TT' in data.columns:
            # Example for Swissmetro-like data structure
            x = data[['TRAIN_TT', 'TRAIN_CO', 'SM_TT', 'SM_CO', 'CAR_TT', 'CAR_CO']].values
            x = x.reshape([-1, 3, 2])  # (batch_size, n_alternatives, n_features)
            
            # Convert choice column to 0-indexed
            y = data[self.choice_column].values - 1
            
        else:
            # Generic case - assumes variables are organized by alternative
            # and properly preprocessed
            feature_cols = [col for col in data.columns if col != self.choice_column]
            x = data[feature_cols].values
            
            # Infer n_alternatives from choice column
            n_alternatives = len(data[self.choice_column].unique())
            
            # Reshape if not already in the right shape
            if x.shape[1] % n_alternatives == 0:
                n_features = x.shape[1] // n_alternatives
                x = x.reshape([-1, n_alternatives, n_features])
            
            # Convert choice column to 0-indexed
            y = data[self.choice_column].values - 1
        
        # Convert to PyTorch tensors
        feature_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
        choice_tensor = torch.tensor(y, dtype=torch.long).to(self.device)
        
        return feature_tensor, choice_tensor


class MNLModel(nn.Module):
    """
    PyTorch implementation of Multinomial Logit model.
    
    This is the actual PyTorch module implementing the MNL model architecture.
    
    Attributes
    ----------
    beta : nn.Parameter
        Parameters for the utility function
    asc : nn.Parameter
        Alternative-specific constants
    """
    
    def __init__(self, n_features, n_alternatives):
        """
        Initialize the MNL model.
        
        Parameters
        ----------
        n_features : int
            Number of features per alternative
        n_alternatives : int
            Number of choice alternatives
        """
        super(MNLModel, self).__init__()
        
        self.n_features = n_features
        self.n_alternatives = n_alternatives
        
        # Parameters to be estimated
        # One beta for each feature
        self.beta = nn.Parameter(torch.randn(n_features))
        
        # Alternative-specific constants (one less than n_alternatives for identification)
        self.asc = nn.Parameter(torch.zeros(n_alternatives - 1))
        
    def forward(self, x):
        """
        Forward pass to calculate choice probabilities.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch_size, n_alternatives, n_features)
            
        Returns
        -------
        torch.Tensor
            Log probabilities with shape (batch_size, n_alternatives)
        """
        # x shape: (batch_size, n_alternatives, n_features)
        # beta shape: (n_features)
        
        # Calculate utility for each alternative: u = x @ beta
        utility = torch.matmul(x, self.beta)  # (batch_size, n_alternatives)
        
        # Add alternative-specific constants (last alternative is base)
        utility[:, :-1] += self.asc
        
        # Return log probabilities
        return F.log_softmax(utility, dim=1)