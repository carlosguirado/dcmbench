"""
SklearnModelAdapter module for DCMBench.

This module provides adapter classes that make scikit-learn models compatible with DCMBench
benchmarking by implementing the required prediction interface. It allows users to
create models using scikit-learn and then use them with DCMBench without writing boilerplate
prediction code.
"""

import pandas as pd
import numpy as np
from typing import Dict, Union, Optional, Any, List, Tuple, Callable
import logging

from ..model_benchmarker.model_interface import PredictionInterface

logger = logging.getLogger(__name__)


class SklearnModelAdapter:
    """
    Adapter class that makes scikit-learn models compatible with DCMBench benchmarking.
    
    This class takes a trained scikit-learn classifier and implements the prediction 
    interface required by the DCMBench benchmarker, allowing users to create models using 
    scikit-learn and use them with DCMBench without writing boilerplate code.
    
    Parameters
    ----------
    model : Any
        The trained scikit-learn classifier that implements predict_proba or decision_function
    alternative_ids : List[Union[str, int]], optional
        Identifiers for the alternatives. If None, will use model.classes_ or indices starting from 1.
    preprocess_func : Callable, optional
        Function to preprocess DataFrame to numpy arrays before prediction. 
        If None, a default conversion will be attempted.
    name : str, optional
        Custom name for the model
    choice_column : str, default="CHOICE"
        Name of the column containing the chosen alternatives
    """
    
    def __init__(
        self,
        model: Any,
        alternative_ids: Optional[List[Union[str, int]]] = None,
        preprocess_func: Optional[Callable] = None,
        name: Optional[str] = None,
        choice_column: str = "CHOICE"
    ):
        self.model = model
        self.name = name or "SklearnModel"
        self.choice_column = choice_column
        self.preprocess_func = preprocess_func
        
        # Verify the model has the required methods
        if not hasattr(model, 'predict'):
            raise ValueError("Model must have a 'predict' method")
            
        if not (hasattr(model, 'predict_proba') or hasattr(model, 'decision_function')):
            raise ValueError("Model must have either 'predict_proba' or 'decision_function' method")
        
        # Set the alternative IDs
        self.alternative_ids = alternative_ids
        if self.alternative_ids is None:
            # Try to get alternatives from model classes
            if hasattr(model, 'classes_'):
                self.alternative_ids = [str(cls) for cls in model.classes_]
            else:
                # Try to infer number of alternatives
                n_alternatives = None
                
                if hasattr(model, 'n_classes_'):
                    n_alternatives = model.n_classes_
                elif hasattr(model, 'coef_') and model.coef_.ndim > 1:
                    n_alternatives = model.coef_.shape[0] + 1  # For multi-class
                
                if n_alternatives is not None:
                    self.alternative_ids = [str(i+1) for i in range(n_alternatives)]
                else:
                    logger.warning("Could not determine alternative IDs from model. "
                                 "Will infer from predictions.")
        
        # Initialize metrics
        self.choice_accuracy = None
        self.market_share_accuracy = None
        self.actual_shares = None
        self.predicted_shares = None
        
    def _preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Preprocess data from DataFrame to numpy array for scikit-learn models.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data for prediction
            
        Returns
        -------
        np.ndarray
            Preprocessed data as array ready for model input
        """
        if self.preprocess_func is not None:
            # Use the provided preprocessing function
            return self.preprocess_func(data)
        
        # Default preprocessing - convert DataFrame to numpy array
        try:
            # Try to extract features (exclude the choice column if present)
            if self.choice_column in data.columns:
                features = data.drop(columns=[self.choice_column]).values
            else:
                features = data.values
                
            return features
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
        # Prepare the input features
        X = self._preprocess_data(data)
        
        # Get model predictions
        if hasattr(self.model, 'predict_proba'):
            # Use predict_proba for probability estimates
            probabilities = self.model.predict_proba(X)
        elif hasattr(self.model, 'decision_function'):
            # Use decision_function and convert to probabilities
            decisions = self.model.decision_function(X)
            
            # Handle both binary and multi-class cases
            if decisions.ndim == 1:
                # Binary case - convert to two-column array
                # Apply sigmoid to get probabilities
                probs_positive = 1 / (1 + np.exp(-decisions))
                probabilities = np.column_stack([1 - probs_positive, probs_positive])
            else:
                # Multi-class case - apply softmax
                exp_decisions = np.exp(decisions - np.max(decisions, axis=1, keepdims=True))
                probabilities = exp_decisions / np.sum(exp_decisions, axis=1, keepdims=True)
        else:
            raise ValueError("Model does not support probability predictions")
        
        # If alternative_ids wasn't set, infer from model classes or probabilities shape
        if self.alternative_ids is None:
            if hasattr(self.model, 'classes_'):
                self.alternative_ids = [str(cls) for cls in self.model.classes_]
            else:
                # Create default IDs based on probability shape
                n_alternatives = probabilities.shape[1]
                self.alternative_ids = [str(i+1) for i in range(n_alternatives)]
        
        # Ensure we have the right number of columns
        if len(self.alternative_ids) != probabilities.shape[1]:
            raise ValueError(f"Number of alternative IDs ({len(self.alternative_ids)}) "
                           f"does not match number of probability columns ({probabilities.shape[1]})")
        
        # Create the DataFrame
        probs_df = pd.DataFrame(
            probabilities,
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
        # Prepare the input features
        X = self._preprocess_data(data)
        
        # Use model's predict method
        predictions = self.model.predict(X)
        
        # Convert to pandas Series
        predictions_series = pd.Series(predictions)
        
        # If predictions are already the alternative IDs, we're done
        if set(predictions_series.unique()).issubset(set(self.alternative_ids or [])):
            return predictions_series
        
        # Otherwise, convert from indices to alternative IDs if needed
        if hasattr(self.model, 'classes_') and all(isinstance(p, (int, np.integer)) for p in predictions):
            # Map from class indices to class values
            classes = self.model.classes_
            predictions_series = predictions_series.map(lambda idx: str(classes[idx]))
        elif all(isinstance(p, (int, np.integer)) for p in predictions):
            # Map from indices to alternative IDs
            if self.alternative_ids:
                # Use existing alternative_ids
                mapping = {i: alt_id for i, alt_id in enumerate(self.alternative_ids)}
                predictions_series = predictions_series.map(lambda idx: mapping.get(idx, str(idx+1)))
        
        return predictions_series
    
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
        
        return {
            'choice_accuracy': self.choice_accuracy,
            'market_share_accuracy': self.market_share_accuracy
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Return model metrics for benchmarking.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary of model metrics including coefficients and accuracy metrics if available
        """
        metrics = {
            'name': self.name,
            'framework': 'sklearn',
        }
        
        # Include number of alternatives
        if self.alternative_ids is not None:
            metrics['n_alternatives'] = len(self.alternative_ids)
        elif hasattr(self.model, 'n_classes_'):
            metrics['n_alternatives'] = self.model.n_classes_
        
        # Include parameters if available
        if hasattr(self.model, 'coef_'):
            metrics['coef'] = self.model.coef_
            metrics['n_parameters'] = self.model.coef_.size
            
            if hasattr(self.model, 'intercept_'):
                metrics['intercept'] = self.model.intercept_
                metrics['n_parameters'] += np.size(self.model.intercept_)
        
        # Include feature importances for tree-based models
        if hasattr(self.model, 'feature_importances_'):
            metrics['feature_importances'] = self.model.feature_importances_
        
        # Add metrics if already calculated
        if hasattr(self, 'choice_accuracy') and self.choice_accuracy is not None:
            metrics['choice_accuracy'] = self.choice_accuracy
            
        if hasattr(self, 'market_share_accuracy') and self.market_share_accuracy is not None:
            metrics['market_share_accuracy'] = self.market_share_accuracy
            
        if hasattr(self, 'actual_shares') and self.actual_shares is not None:
            metrics['actual_shares'] = self.actual_shares
            
        if hasattr(self, 'predicted_shares') and self.predicted_shares is not None:
            metrics['predicted_shares'] = self.predicted_shares
        
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
        
        # Extract coefficients
        if hasattr(self.model, 'coef_'):
            if self.model.coef_.ndim == 1:
                # Single coefficient vector
                for i, coef in enumerate(self.model.coef_):
                    params[f'beta_{i}'] = coef
            else:
                # Multiple coefficient vectors (one per class)
                for cls_idx, cls_coef in enumerate(self.model.coef_):
                    for feat_idx, coef in enumerate(cls_coef):
                        params[f'beta_{cls_idx}_{feat_idx}'] = coef
        
        # Add intercept if available
        if hasattr(self.model, 'intercept_'):
            if isinstance(self.model.intercept_, (list, np.ndarray)):
                for i, intercept in enumerate(self.model.intercept_):
                    params[f'intercept_{i}'] = intercept
            else:
                params['intercept'] = self.model.intercept_
                
        # Add feature importances for tree-based models
        if hasattr(self.model, 'feature_importances_'):
            for i, importance in enumerate(self.model.feature_importances_):
                params[f'importance_{i}'] = importance
                
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


class LogisticRegressionAdapter(SklearnModelAdapter):
    """
    Adapter specifically for scikit-learn's LogisticRegression models.
    
    This adapter provides enhanced functionality for scikit-learn's LogisticRegression,
    with direct access to coefficients and intercepts as well as utility calculations.
    
    Parameters
    ----------
    model : sklearn.linear_model.LogisticRegression
        The trained scikit-learn LogisticRegression model
    alternative_ids : List[Union[str, int]], optional
        Identifiers for the alternatives. If None, will use model.classes_.
    preprocess_func : Callable, optional
        Function to preprocess DataFrame to numpy arrays before prediction
    feature_names : List[str], optional
        Names of the features in the order they appear in the input array
    name : str, optional
        Custom name for the model
    choice_column : str, default="CHOICE"
        Name of the column containing the chosen alternatives
    """
    
    def __init__(
        self,
        model: Any,
        alternative_ids: Optional[List[Union[str, int]]] = None,
        preprocess_func: Optional[Callable] = None,
        feature_names: Optional[List[str]] = None,
        name: Optional[str] = None,
        choice_column: str = "CHOICE"
    ):
        # Verify this is a LogisticRegression model
        from sklearn.linear_model import LogisticRegression
        if not isinstance(model, LogisticRegression):
            raise ValueError("Model must be an instance of sklearn.linear_model.LogisticRegression")
            
        super().__init__(
            model=model,
            alternative_ids=alternative_ids,
            preprocess_func=preprocess_func,
            name=name or "LogisticRegressionModel",
            choice_column=choice_column
        )
        
        # Store coefficient information
        self.is_multiclass = hasattr(model, 'multi_class') and model.multi_class == 'multinomial'
        self.feature_names = feature_names
        
        # Format feature names if not provided
        if self.feature_names is None and hasattr(model, 'feature_names_in_'):
            self.feature_names = model.feature_names_in_.tolist()
        elif self.feature_names is None:
            # Create default feature names
            n_features = model.coef_.shape[1] if self.is_multiclass else len(model.coef_)
            self.feature_names = [f'feature_{i}' for i in range(n_features)]
    
    def calculate_utilities(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate utility values for each alternative.
        
        For LogisticRegression, these are the linear combinations before applying
        the logistic/softmax function.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data for utility calculation
            
        Returns
        -------
        pd.DataFrame
            DataFrame with utility values for each alternative
        """
        # Prepare the input features
        X = self._preprocess_data(data)
        
        # Calculate utilities using the model's decision_function
        if hasattr(self.model, 'decision_function'):
            utilities = self.model.decision_function(X)
            
            # Convert to DataFrame
            if utilities.ndim == 1:
                # Binary case, expand to two columns (negative and positive)
                utilities_df = pd.DataFrame({
                    self.alternative_ids[0]: -utilities,
                    self.alternative_ids[1]: utilities
                })
            else:
                # Multiclass case
                utilities_df = pd.DataFrame(
                    utilities,
                    columns=self.alternative_ids
                )
        else:
            # If decision_function not available, calculate manually
            if self.is_multiclass:
                # For multinomial model, apply dot product for each class
                utilities = np.dot(X, self.model.coef_.T) + self.model.intercept_
                utilities_df = pd.DataFrame(
                    utilities,
                    columns=self.alternative_ids
                )
            else:
                # For binary model, calculate manually
                utilities = np.dot(X, self.model.coef_.T) + self.model.intercept_
                utilities_df = pd.DataFrame({
                    self.alternative_ids[0]: -utilities.flatten(),
                    self.alternative_ids[1]: utilities.flatten()
                })
                
        return utilities_df
    
    def get_coefficient_summary(self) -> pd.DataFrame:
        """
        Get a summary of model coefficients with feature names.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing feature names, coefficients, and classes
        """
        if self.is_multiclass:
            # For multinomial model
            coefs = []
            for i, class_idx in enumerate(range(len(self.model.classes_))):
                class_name = str(self.model.classes_[class_idx])
                for j, feat_name in enumerate(self.feature_names):
                    coefs.append({
                        'feature': feat_name,
                        'coefficient': self.model.coef_[class_idx, j],
                        'class': class_name
                    })
                # Add intercept
                coefs.append({
                    'feature': 'intercept',
                    'coefficient': self.model.intercept_[class_idx],
                    'class': class_name
                })
        else:
            # For binary model
            coefs = []
            pos_class = str(self.model.classes_[1])
            for j, feat_name in enumerate(self.feature_names):
                coefs.append({
                    'feature': feat_name,
                    'coefficient': self.model.coef_[0, j],
                    'class': pos_class
                })
            # Add intercept
            coefs.append({
                'feature': 'intercept',
                'coefficient': self.model.intercept_[0],
                'class': pos_class
            })
                
        return pd.DataFrame(coefs)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Return model metrics including LogisticRegression-specific metrics.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary of model metrics including coefficients and summaries
        """
        metrics = super().get_metrics()
        
        # Add LogisticRegression-specific information
        metrics['model_type'] = 'multinomial_logit' if self.is_multiclass else 'binary_logit'
        metrics['feature_names'] = self.feature_names
        
        # Get coefficient_summary as a dict
        coef_summary = self.get_coefficient_summary()
        metrics['coefficient_summary'] = coef_summary.to_dict('records')
        
        return metrics