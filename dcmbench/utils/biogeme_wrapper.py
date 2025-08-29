"""Wrapper utility for integrating Biogeme models with the benchmarker."""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# We don't import BaseDiscreteChoiceModel to avoid circular import
# Instead we make a basic wrapper that doesn't inherit from it

class BiogemeModelWrapper:
    """Wrapper to use an existing Biogeme model with the benchmarker.
    
    This wrapper allows easy integration of Biogeme models with the benchmarking system
    by automatically extracting and calculating all required metrics.
    
    Example:
        >>> import biogeme.biogeme as bio
        >>> from biogeme import models
        >>> from dcmbench.utils import BiogemeModelWrapper
        >>> 
        >>> # Your existing Biogeme model setup
        >>> biogeme = bio.BIOGEME(database, logprob)
        >>> 
        >>> # Wrap it for benchmarking
        >>> model = BiogemeModelWrapper(data, biogeme)
        >>> benchmarker = ModelBenchmarker()
        >>> results = benchmarker.run_benchmark(data, [model])
    """
    
    def __init__(self, data, biogeme_model, choice_column='CHOICE'):
        """Initialize wrapper with a Biogeme model.
        
        Args:
            data: DataFrame with choice data
            biogeme_model: Configured Biogeme model ready for estimation
            choice_column: Name of the choice column in data
        """
        self.data = data
        self.database = None  # Will be set when needed
        self.biogeme_model = biogeme_model
        self.choice_column = choice_column
        
        # Setup other fields with default values
        self.results = None
        self.final_ll = None
        self.rho_squared = None
        self.rho_squared_bar = None
        self.actual_shares = {}
        self.predicted_shares = {}
        self.market_share_accuracy = None
        self.confusion_matrix = None
        self.choice_accuracy = None
    
    def estimate(self):
        """Estimate model and extract metrics."""
        # Estimate the model
        self.results = self.biogeme_model.estimate()
        
        # Extract metrics from Biogeme results
        stats = self.results.getGeneralStatistics()
        self.final_ll = stats['Final log likelihood'][0]
        self.rho_squared = stats['Rho-square for the null model'][0]
        self.rho_squared_bar = stats['Rho-square-bar for the null model'][0]
        
        # Calculate market shares and choice accuracy
        self._calculate_shares_and_accuracy()
        
        return self.results
    
    def _calculate_shares_and_accuracy(self):
        """Calculate market shares and choice accuracy from probabilities."""
        # Get simulated probabilities
        prob_vars = [v for v in self.results.data.simulatedValues.columns 
                    if v.startswith('Prob')]
        
        if not prob_vars:  # If no probability columns, try to simulate
            simulatedValues = self.biogeme_model.simulate(
                self.results.get_beta_values()
            )
        else:
            simulatedValues = self.results.data.simulatedValues
        
        # Calculate actual shares
        choices = self.data[self.choice_column]
        total = len(choices)
        self.actual_shares = choices.value_counts(normalize=True).to_dict()
        
        # Calculate predicted shares
        prob_means = simulatedValues.mean()
        self.predicted_shares = {
            i+1: prob_means[f'Prob. {i+1}'] 
            for i in range(len(prob_vars))
        }
        
        # Calculate market share accuracy
        total_abs_error = sum(
            abs(self.actual_shares.get(i, 0) - self.predicted_shares.get(i, 0))
            for i in set(self.actual_shares) | set(self.predicted_shares)
        )
        self.market_share_accuracy = 1 - (total_abs_error / 2)
        
        # Calculate choice accuracy
        predicted_choices = simulatedValues.idxmax(axis=1)
        predicted_choices = predicted_choices.map(
            lambda x: int(x.split()[-1])  # Extract choice number
        )
        
        # Create confusion matrix
        self.confusion_matrix = pd.crosstab(
            choices,
            predicted_choices,
            rownames=['Actual'],
            colnames=['Predicted']
        )
        
        # Calculate choice accuracy
        self.choice_accuracy = (choices == predicted_choices).mean()
        
    # Add required methods for compatibility with the PredictionInterface
    
    def predict_probabilities(self, data):
        """Predict choice probabilities for each alternative.
        
        Args:
            data: DataFrame with features
            
        Returns:
            DataFrame with predicted probabilities
        """
        # Use the Biogeme model to simulate probabilities
        try:
            if self.results is None:
                logger.warning("Model not estimated yet. Estimating now.")
                self.estimate()
                
            simulatedValues = self.biogeme_model.simulate(
                self.results.get_beta_values()
            )
            
            # Extract probability columns
            prob_cols = [col for col in simulatedValues.columns if col.startswith('Prob')]
            probabilities = simulatedValues[prob_cols].copy()
            
            # Rename columns to match the expected format
            probabilities.columns = [int(col.split()[-1]) for col in prob_cols]
            
            return probabilities
            
        except Exception as e:
            logger.error(f"Error predicting probabilities: {str(e)}")
            return pd.DataFrame()
    
    def predict_choices(self, data):
        """Predict discrete choices for each observation.
        
        Args:
            data: DataFrame with features
            
        Returns:
            Series with predicted choices
        """
        # Get probabilities and take the argmax
        try:
            probs = self.predict_probabilities(data)
            return probs.idxmax(axis=1)
        except Exception as e:
            logger.error(f"Error predicting choices: {str(e)}")
            return pd.Series()
    
    def calculate_choice_accuracy(self, data, choice_column=None):
        """Calculate choice accuracy metrics.
        
        Args:
            data: DataFrame with features and choices
            choice_column: Name of the choice column (default: self.choice_column)
        """
        if choice_column is None:
            choice_column = self.choice_column
        
        # Get predicted choices
        predicted = self.predict_choices(data)
        actual = data[choice_column]
        
        # Calculate accuracy
        self.choice_accuracy = (predicted == actual).mean()
        
        # Calculate confusion matrix
        self.confusion_matrix = pd.crosstab(
            actual, 
            predicted,
            rownames=['Actual'],
            colnames=['Predicted']
        )
        
        # Calculate market shares
        self.actual_shares = actual.value_counts(normalize=True).to_dict()
        
        # Calculate predicted shares
        probs = self.predict_probabilities(data)
        self.predicted_shares = probs.mean().to_dict()
        
        # Calculate market share accuracy
        total_abs_error = sum(
            abs(self.actual_shares.get(i, 0) - self.predicted_shares.get(i, 0))
            for i in set(self.actual_shares) | set(self.predicted_shares)
        )
        self.market_share_accuracy = 1 - (total_abs_error / 2)
    
    def get_metrics(self):
        """Get all metrics in a dictionary.
        
        Returns:
            Dict with metrics
        """
        metrics = {
            'choice_accuracy': self.choice_accuracy,
            'market_share_accuracy': self.market_share_accuracy,
            'actual_shares': self.actual_shares,
            'predicted_shares': self.predicted_shares,
            'confusion_matrix': self.confusion_matrix
        }
        
        # Add Biogeme-specific metrics if available
        if self.results is not None:
            metrics.update({
                'final_log_likelihood': self.final_ll,
                'rho_squared': self.rho_squared,
                'rho_squared_bar': self.rho_squared_bar
            })
            
        return metrics
    
    def get_parameter_values(self):
        """Get the estimated parameter values.
        
        Returns:
            Dict mapping parameter names to values
        """
        if self.results is None:
            logger.warning("Model not estimated yet.")
            return {}
            
        return self.results.get_beta_values()