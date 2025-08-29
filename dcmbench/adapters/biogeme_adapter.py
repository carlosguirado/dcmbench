"""
BiogemeModelAdapter module for DCMBench.

This module provides adapter classes that make Biogeme models compatible with DCMBench
benchmarking by implementing the required prediction interface. It allows users to
create models using standard Biogeme syntax and then use them with DCMBench without
writing boilerplate prediction code.
"""

import pandas as pd
import numpy as np
from typing import Dict, Union, Optional, Any, List, Tuple

import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Expression

from ..model_benchmarker.model_interface import PredictionInterface


class BiogemeModelAdapter:
    """
    Adapter class that makes Biogeme models compatible with DCMBench benchmarking.
    
    This class takes an estimated Biogeme model along with its utility functions
    and availability conditions, and implements the prediction interface required
    by the DCMBench benchmarker. This allows users to create models using standard
    Biogeme syntax and then use them with DCMBench without writing boilerplate code.
    
    Parameters
    ----------
    biogeme_model : bio.BIOGEME
        The Biogeme model object
    utility_functions : Dict[Union[str, int], Expression]
        Dictionary mapping alternative IDs to utility expressions
    availability : Dict[Union[str, int], Expression]
        Dictionary mapping alternative IDs to availability expressions
    nests : Optional[Any], default=None
        Nesting structure for nested logit models
    results : Optional[Any], default=None
        Pre-estimated model results. If None, estimation will be performed
    name : Optional[str], default=None
        Custom name for the model
    choice_column : str, default="CHOICE"
        Name of the column containing the chosen alternatives
    """
    
    def __init__(
        self,
        biogeme_model: bio.BIOGEME,
        utility_functions: Dict[Union[str, int], Expression],
        availability: Dict[Union[str, int], Expression],
        nests: Optional[Any] = None,
        results: Optional[Any] = None,
        name: Optional[str] = None,
        choice_column: str = "CHOICE"
    ):
        self.biogeme_model = biogeme_model
        self.utility_functions = utility_functions
        self.availability = availability
        self.nests = nests
        self.choice_column = choice_column
        self.name = name or "BiogemeModel"
        
        # Store the database for later use
        self.database = biogeme_model.database
        self.data_raw = self.database.data.copy() if hasattr(self.database, 'data') else None
        
        # Get or estimate results
        if results is None:
            self.results = self.biogeme_model.estimate()
        else:
            self.results = results
            
        # Extract model statistics
        self._extract_model_statistics()
        
    def _extract_model_statistics(self):
        """Extract key statistics from Biogeme results."""
        if self.results is None:
            return
            
        try:
            stats = self.results.getGeneralStatistics()
            self.final_ll = stats['Final log likelihood'][0]
            self.null_ll = stats['Null log likelihood'][0]
            self.rho_squared = stats['Rho-square for the null model'][0]
            self.rho_squared_bar = stats['Rho-square-bar for the null model'][0]
            self.n_parameters = len(self.results.getBetaValues())
        except (KeyError, AttributeError) as e:
            # Handle the case where some statistics may not be available
            self.final_ll = None
            self.null_ll = None
            self.rho_squared = None
            self.rho_squared_bar = None
            self.n_parameters = None

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
        if self.results is None:
            raise RuntimeError("Model must be estimated before predicting")
            
        # Determine if we need to create a new database for prediction
        is_same_data = False
        if self.data_raw is not None:
            # Check if data is the same as what was used for estimation
            if data is self.data_raw or data.equals(self.data_raw):
                is_same_data = True
        
        # Create choice column with integer type if needed
        if self.choice_column in data.columns and data[self.choice_column].dtype == object:
            data_copy = data.copy()
            data_copy[self.choice_column] = data_copy[self.choice_column].astype(int)
        else:
            data_copy = data
        
        # Create a new database if needed
        if not is_same_data:
            prediction_db = db.Database('prediction', data_copy)
        else:
            prediction_db = self.database
        
        # Get estimated parameters
        betas = self.results.getBetaValues()
        
        # Prepare simulation expressions
        simulate = {}
        alternatives = list(self.utility_functions.keys())
        
        # Calculate probabilities for each alternative
        for i, alt in enumerate(alternatives):
            if self.nests is None:
                # MNL probabilities
                prob_expr = models.logit(self.utility_functions, self.availability, alt)
            else:
                # NL probabilities 
                prob_expr = models.nested(self.utility_functions, self.availability, self.nests, alt)
            
            # Add to simulation dictionary
            simulate[f'Prob_{alt}'] = prob_expr
        
        # Setup and run simulation
        biogeme = bio.BIOGEME(prediction_db, simulate)
        biogeme.modelName = "prediction"
        simulatedValues = biogeme.simulate(betas)
        
        # Convert to DataFrame with the right column names
        result = pd.DataFrame()
        for alt in alternatives:
            alt_str = str(alt)
            result[alt_str] = simulatedValues[f'Prob_{alt}']
        
        return result
    
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
        # Get probabilities and select most likely alternative
        probs = self.predict_probabilities(data)
        
        # Make sure we return choices as strings
        return probs.idxmax(axis=1)
    
    def calculate_choice_accuracy(self, data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Calculate choice prediction accuracy and market shares.
        
        Parameters
        ----------
        data : pd.DataFrame, optional
            Data to calculate metrics for. If None, uses the data from initialization.
            
        Returns
        -------
        Dict[str, float]
            Dictionary with accuracy metrics including choice_accuracy and market_share_accuracy
        """
        # Use provided data or fall back to original data
        if data is None:
            if self.data_raw is None:
                raise ValueError("No data available for accuracy calculation")
            data = self.data_raw
            
        # Get choice column
        if self.choice_column not in data.columns:
            raise ValueError(f"Choice column '{self.choice_column}' not found in data")
            
        # Get actual choices
        actual_choices = data[self.choice_column]
        
        # Calculate predicted choices
        predicted_choices = self.predict_choices(data)
        if isinstance(predicted_choices, pd.Series):
            # Convert string indices to integers if needed
            predicted_choices = predicted_choices.astype(int)
            
        # Calculate choice accuracy
        choice_accuracy = (actual_choices == predicted_choices).mean()
        
        # Calculate market shares
        actual_shares = actual_choices.value_counts(normalize=True).to_dict()
        
        # Calculate predicted market shares
        probs = self.predict_probabilities(data)
        predicted_shares = {int(col): probs[col].mean() for col in probs.columns}
        
        # Calculate market share accuracy
        total_abs_error = sum(
            abs(actual_shares.get(int(alt), 0) - predicted_shares.get(int(alt), 0))
            for alt in set(actual_shares.keys()) | set(predicted_shares.keys())
        )
        market_share_accuracy = 1 - (total_abs_error / 2)  # Divide by 2 to normalize
        
        # Store for later use
        self.choice_accuracy = choice_accuracy
        self.market_share_accuracy = market_share_accuracy
        self.actual_shares = actual_shares
        self.predicted_shares = predicted_shares
        
        # Calculate Value of Time (VOT) if possible
        vot = None
        if self.results is not None:
            try:
                params = self.results.get_beta_values()
                # Look for standard time and cost parameters
                time_params = [k for k in params.keys() if 'TIME' in k.upper() or 'TT' in k.upper()]
                cost_params = [k for k in params.keys() if 'COST' in k.upper() or 'CO' in k.upper()]
                
                # Try to find B_TIME and B_COST specifically
                b_time = None
                b_cost = None
                
                if 'B_TIME' in params:
                    b_time = params['B_TIME']
                elif time_params:
                    b_time = params[time_params[0]]
                    
                if 'B_COST' in params:
                    b_cost = params['B_COST']
                elif cost_params:
                    b_cost = params[cost_params[0]]
                
                if b_time is not None and b_cost is not None and b_cost != 0:
                    # VOT = (B_TIME / B_COST) * 60
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
            'choice_accuracy': choice_accuracy,
            'market_share_accuracy': market_share_accuracy,
            'actual_shares' : actual_shares,
            'predicted_shares' : predicted_shares
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
            Dictionary of model metrics including log-likelihood, rho-squared values,
            parameter counts, and market shares if available.
        """
        if self.results is None:
            raise RuntimeError("Model must be estimated before getting metrics")
            
        metrics = {
            'n_parameters': self.n_parameters,
            'final_ll': self.final_ll,
            'null_ll': self.null_ll,
            'rho_squared': self.rho_squared,
            'rho_squared_bar': self.rho_squared_bar
        }
        
        # Include accuracy metrics if available
        if hasattr(self, 'choice_accuracy'):
            metrics['choice_accuracy'] = self.choice_accuracy
            
        if hasattr(self, 'market_share_accuracy'):
            metrics['market_share_accuracy'] = self.market_share_accuracy
            
        if hasattr(self, 'actual_shares'):
            metrics['actual_shares'] = self.actual_shares
            
        if hasattr(self, 'predicted_shares'):
            metrics['predicted_shares'] = self.predicted_shares
        
        # Include Value of Time if available
        if hasattr(self, 'value_of_time'):
            metrics['value_of_time'] = self.value_of_time['value']
            metrics['vot_details'] = self.value_of_time
        
        # Calculate market shares if not already calculated
        if not hasattr(self, 'actual_shares') and self.data_raw is not None:
            try:
                accuracy_metrics = self.calculate_choice_accuracy()
                metrics.update(accuracy_metrics)
            except Exception as e:
                # If calculation fails, continue without these metrics
                pass
        
        return metrics
    
    def get_parameter_values(self) -> Dict[str, float]:
        """
        Get the estimated parameter values.
        
        Returns
        -------
        Dict[str, float]
            Dictionary of parameter names and values
        """
        if self.results is None:
            raise RuntimeError("Model must be estimated before getting parameter values")
            
        return self.results.getBetaValues()
    
    def get_model_name(self) -> str:
        """
        Get the model name.
        
        Returns
        -------
        str
            Model name
        """
        return self.name


class BiogemeMultinomialLogitAdapter(BiogemeModelAdapter):
    """
    Adapter specifically for Multinomial Logit models.
    
    This is a convenience class that sets nests=None to ensure MNL probabilities.
    
    Parameters are the same as BiogemeModelAdapter.
    """
    
    def __init__(
        self,
        biogeme_model: bio.BIOGEME,
        utility_functions: Dict[Union[str, int], Expression],
        availability: Dict[Union[str, int], Expression],
        results: Optional[Any] = None,
        name: Optional[str] = None,
        choice_column: str = "CHOICE"
    ):
        super().__init__(
            biogeme_model=biogeme_model,
            utility_functions=utility_functions,
            availability=availability,
            nests=None,  # Explicitly set to None for MNL
            results=results,
            name=name or "BiogemeMultinomialLogit",
            choice_column=choice_column
        )


class BiogemeNestedLogitAdapter(BiogemeModelAdapter):
    """
    Adapter specifically for Nested Logit models.
    
    This class requires a nesting structure and ensures NL probabilities are used.
    
    Parameters are the same as BiogemeModelAdapter with nests required.
    """
    
    def __init__(
        self,
        biogeme_model: bio.BIOGEME,
        utility_functions: Dict[Union[str, int], Expression],
        availability: Dict[Union[str, int], Expression],
        nests: Any,  # Required for NL
        results: Optional[Any] = None,
        name: Optional[str] = None,
        choice_column: str = "CHOICE"
    ):
        if nests is None:
            raise ValueError("Nesting structure is required for Nested Logit models")
            
        super().__init__(
            biogeme_model=biogeme_model,
            utility_functions=utility_functions,
            availability=availability,
            nests=nests,
            results=results,
            name=name or "BiogemeNestedLogit",
            choice_column=choice_column
        )


class BiogemeModelFromSpecAdapter(BiogemeModelAdapter):
    """
    Adapter for models created from model specifications.
    
    This class extends BiogemeModelAdapter to work specifically with models
    generated from JSON specifications through the ModelSpecLoader.
    
    Parameters
    ----------
    model : Any
        The model object created from a specification
    spec : Dict[str, Any]
        The model specification data
    """
    
    def __init__(
        self,
        model: Any,
        spec: Dict[str, Any]
    ):
        # Extract relevant components from the specification
        utility_functions = getattr(model, 'formulas', {})
        availability = getattr(model, 'availability_conditions', {})
        nests = getattr(model, 'nests', None) if hasattr(model, 'nests') else None
        results = getattr(model, 'results', None)
        choice_column = spec.get('choice_column', 'CHOICE')
        name = spec.get('name', 'SpecBasedModel')
        
        # Get the Biogeme model if available
        biogeme_model = getattr(model, 'biogeme_instance', None)
        
        # If no Biogeme model instance is available, create one
        if biogeme_model is None:
            # Try to get the database
            database = getattr(model, 'database', None)
            if database is None:
                raise ValueError("Model must have a database attribute")
                
            # Create a dummy Biogeme model just for the adapter
            biogeme_model = bio.BIOGEME(database, None)
        
        # Initialize the parent class
        super().__init__(
            biogeme_model=biogeme_model,
            utility_functions=utility_functions,
            availability=availability,
            nests=nests,
            results=results,
            name=name,
            choice_column=choice_column
        )
        
        # Store the original model and specification
        self.original_model = model
        self.spec = spec