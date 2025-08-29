"""
scikit-learn-specific base classes for discrete choice models.

This module contains base classes for implementing discrete choice models
using the scikit-learn framework.
"""

import logging
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score

from .base import BaseDiscreteChoiceModel

logger = logging.getLogger(__name__)

class SklearnDiscreteChoiceModel(BaseDiscreteChoiceModel):
    """
    Base class for scikit-learn-based discrete choice models.
    
    This class extends BaseDiscreteChoiceModel with scikit-learn-specific
    functionality for model estimation and prediction.
    
    Attributes
    ----------
    model : sklearn.base.BaseEstimator
        The underlying scikit-learn model
    feature_columns : list
        List of columns to use as features
    """
    
    def __init__(self, data, auto_preprocess=True, preprocessing_options=None, debug=False):
        """
        Initialize the scikit-learn model.
        
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
        """Initialize scikit-learn-specific attributes."""
        super()._init_framework_attributes()
        
        # Initialize attributes
        self.model = None
        self.feature_columns = None
        self.choice_column = 'CHOICE'  # Default choice column name
        self.encoders = {}
        self.label_encoder = None
    
    def _prepare_data_for_sklearn(self, data=None):
        """
        Prepare data for scikit-learn model.
        
        Parameters
        ----------
        data : pandas.DataFrame, optional
            Data to prepare. If None, uses the data from initialization.
            
        Returns
        -------
        tuple
            X (features) and y (target) arrays for scikit-learn
        """
        if data is None:
            data = self.database.data
        
        # If feature columns not specified, use all columns except choice
        if self.feature_columns is None:
            self.feature_columns = [col for col in data.columns 
                                  if col != self.choice_column]
        
        # Extract features and target
        X = data[self.feature_columns].values
        y = data[self.choice_column].values
        
        return X, y
    
    def _create_model(self):
        """
        Create the scikit-learn model.
        
        This method should be implemented by subclasses to create
        the appropriate scikit-learn model.
        
        Returns
        -------
        sklearn.base.BaseEstimator
            scikit-learn model
        """
        raise NotImplementedError("Subclasses must implement _create_model")
    
    def estimate(self):
        """
        Estimate model parameters using scikit-learn.
        
        Returns
        -------
        sklearn.base.BaseEstimator
            Fitted scikit-learn model
        """
        # Prepare data for scikit-learn
        X, y = self._prepare_data_for_sklearn()
        
        # Create model if not already created
        if self.model is None:
            self.model = self._create_model()
        
        # Fit the model
        self.model.fit(X, y)
        
        # Create a results object similar to Biogeme's
        self.results = self._create_results_object()
        
        # Calculate metrics
        self.calculate_choice_accuracy()
        
        return self.results
    
    def _create_results_object(self):
        """
        Create a results object similar to Biogeme's.
        
        Returns
        -------
        object
            Results object with methods and attributes similar to Biogeme's
        """
        # Simple class to mimic Biogeme's results object
        class SklearnResults:
            def __init__(self, model, model_coefs, n_params, n_obs):
                self.model = model
                self.model_coefs = model_coefs
                self.data = type('obj', (object,), {
                    'betaValues': model_coefs,
                    'numberOfObservations': n_obs
                })
                
            def get_beta_values(self):
                return self.model_coefs
                
            def getGeneralStatistics(self):
                return {
                    'Final log likelihood': [-1],  # Not available in sklearn
                    'Rho-square for the null model': [-1],  # Not available in sklearn
                    'Rho-square-bar for the null model': [-1]  # Not available in sklearn
                }
                
        # Extract model coefficients
        if hasattr(self.model, 'coef_'):
            model_coefs = {f'beta_{i}': coef 
                         for i, coef in enumerate(self.model.coef_.flatten())}
        elif hasattr(self.model, 'feature_importances_'):
            model_coefs = {f'importance_{i}': imp 
                         for i, imp in enumerate(self.model.feature_importances_)}
        else:
            model_coefs = {}
            
        # Add intercept if available
        if hasattr(self.model, 'intercept_'):
            if isinstance(self.model.intercept_, (list, np.ndarray)):
                for i, intercept in enumerate(self.model.intercept_):
                    model_coefs[f'intercept_{i}'] = intercept
            else:
                model_coefs['intercept'] = self.model.intercept_
                
        # Count number of parameters
        n_params = len(model_coefs)
        
        # Get number of observations
        n_obs = len(self.database.data)
        
        # Create and return results object
        return SklearnResults(self.model, model_coefs, n_params, n_obs)
    
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
            
        # Use original data if no new data is provided
        if data is None:
            X, _ = self._prepare_data_for_sklearn()
        else:
            # Preprocess new data if needed
            if getattr(self, "auto_preprocess", True):
                processed_data = self.preprocess_data(data)
            else:
                processed_data = data
                
            # Prepare for sklearn
            X, _ = self._prepare_data_for_sklearn(processed_data)
        
        # Make predictions
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)
        else:
            # For models without predict_proba, use decision_function if available
            if hasattr(self.model, 'decision_function'):
                decisions = self.model.decision_function(X)
                if decisions.ndim == 1:
                    # Binary case
                    probabilities = np.column_stack([1 - decisions, decisions])
                else:
                    # Multiclass case
                    probabilities = np.exp(decisions) / np.sum(np.exp(decisions), axis=1, keepdims=True)
            else:
                raise NotImplementedError(
                    "Model does not support probability predictions"
                )
        
        # Get class labels
        if hasattr(self.model, 'classes_'):
            classes = self.model.classes_
        else:
            classes = np.arange(probabilities.shape[1]) + 1
            
        # Convert to DataFrame
        probabilities_df = pd.DataFrame(
            probabilities,
            columns=[str(cls) for cls in classes]
        )
        
        return probabilities_df
    
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
            X, y = self._prepare_data_for_sklearn()
        else:
            # Preprocess new data if needed
            if getattr(self, "auto_preprocess", True):
                processed_data = self.preprocess_data(data)
            else:
                processed_data = data
                
            actual_choices = processed_data[choice_col]
            X, y = self._prepare_data_for_sklearn(processed_data)
        
        # Get predicted probabilities
        probabilities = self.predict_probabilities(data)
        
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
        
        # Extract model coefficients
        params = {}
        
        # Handle different scikit-learn model types
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


class MultinomialLogitSklearn(SklearnDiscreteChoiceModel):
    """
    scikit-learn implementation of Multinomial Logit model.
    
    This class provides a simple MNL model implemented using scikit-learn's
    LogisticRegression model.
    """
    
    def __init__(self, data, auto_preprocess=True, preprocessing_options=None, 
                 debug=False, C=1.0, max_iter=1000, solver='lbfgs'):
        """
        Initialize the scikit-learn MNL model.
        
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
        C : float, default=1.0
            Inverse of regularization strength
        max_iter : int, default=1000
            Maximum number of iterations
        solver : str, default='lbfgs'
            Solver to use for optimization
        """
        super().__init__(data, auto_preprocess, preprocessing_options, debug)
        
        # Model parameters
        self.C = C
        self.max_iter = max_iter
        self.solver = solver
        
    def _create_model(self):
        """
        Create the scikit-learn multinomial logit model.
        
        Returns
        -------
        sklearn.linear_model.LogisticRegression
            scikit-learn logistic regression model
        """
        from sklearn.linear_model import LogisticRegression
        
        # Create model with multinomial loss
        model = LogisticRegression(
            multi_class='multinomial',
            solver=self.solver,
            C=self.C,
            max_iter=self.max_iter
        )
        
        return model


class DiscreteChoiceClassifier(BaseEstimator, ClassifierMixin):
    """
    A scikit-learn compatible classifier for discrete choice modeling.
    
    This class implements a scikit-learn compatible estimator that can be
    used for discrete choice modeling.
    
    Attributes
    ----------
    coef_ : ndarray
        Coefficient matrix
    intercept_ : ndarray
        Intercept vector
    classes_ : ndarray
        Class labels
    """
    
    def __init__(self, solver='lbfgs', C=1.0, max_iter=1000):
        """
        Initialize the classifier.
        
        Parameters
        ----------
        solver : str, default='lbfgs'
            Solver to use for optimization
        C : float, default=1.0
            Inverse of regularization strength
        max_iter : int, default=1000
            Maximum number of iterations
        """
        self.solver = solver
        self.C = C
        self.max_iter = max_iter
    
    def fit(self, X, y):
        """
        Fit the model to the data.
        
        Parameters
        ----------
        X : ndarray
            Training features
        y : ndarray
            Target values
            
        Returns
        -------
        self
            Fitted model
        """
        # Check input
        X, y = check_X_y(X, y)
        
        # Store class labels
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # Use scikit-learn's LogisticRegression for actual fitting
        from sklearn.linear_model import LogisticRegression
        
        # For multi-class, use multinomial loss
        if n_classes > 2:
            self._lr = LogisticRegression(
                multi_class='multinomial',
                solver=self.solver,
                C=self.C,
                max_iter=self.max_iter
            )
        else:
            # For binary, use standard logistic regression
            self._lr = LogisticRegression(
                solver=self.solver,
                C=self.C,
                max_iter=self.max_iter
            )
        
        # Fit the model
        self._lr.fit(X, y)
        
        # Store coefficients and intercept
        self.coef_ = self._lr.coef_
        self.intercept_ = self._lr.intercept_
        
        # Store additional attributes for model interpretation
        self.feature_names_in_ = getattr(self._lr, 'feature_names_in_', None)
        self.n_features_in_ = getattr(self._lr, 'n_features_in_', X.shape[1])
        
        # Calculate log-likelihood (approximation)
        probs = self.predict_proba(X)
        self.log_likelihood_ = np.sum(np.log(probs[np.arange(len(y)), np.searchsorted(self.classes_, y)]))
        
        # Calculate null log-likelihood (model with only intercept)
        null_probs = np.ones((len(y), n_classes)) / n_classes
        self.null_log_likelihood_ = np.sum(np.log(null_probs[np.arange(len(y)), np.searchsorted(self.classes_, y)]))
        
        # Calculate rho-squared
        self.rho_squared_ = 1 - self.log_likelihood_ / self.null_log_likelihood_
        
        # Calculate rho-squared adjusted
        n_params = np.prod(self.coef_.shape) + len(self.intercept_)
        self.rho_squared_adj_ = 1 - (self.log_likelihood_ - n_params) / self.null_log_likelihood_
        
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters
        ----------
        X : ndarray
            Features
            
        Returns
        -------
        ndarray
            Predicted class labels
        """
        check_is_fitted(self, ['coef_', 'intercept_'])
        X = check_array(X)
        
        return self._lr.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        
        Parameters
        ----------
        X : ndarray
            Features
            
        Returns
        -------
        ndarray
            Predicted class probabilities
        """
        check_is_fitted(self, ['coef_', 'intercept_'])
        X = check_array(X)
        
        return self._lr.predict_proba(X)
    
    def decision_function(self, X):
        """
        Decision function for samples in X.
        
        Parameters
        ----------
        X : ndarray
            Features
            
        Returns
        -------
        ndarray
            Decision function values
        """
        check_is_fitted(self, ['coef_', 'intercept_'])
        X = check_array(X)
        
        return self._lr.decision_function(X)
    
    def get_metrics(self):
        """
        Get model metrics.
        
        Returns
        -------
        dict
            Dictionary of model metrics
        """
        check_is_fitted(self, ['coef_', 'intercept_'])
        
        metrics = {
            'log_likelihood': self.log_likelihood_,
            'null_log_likelihood': self.null_log_likelihood_,
            'rho_squared': self.rho_squared_,
            'rho_squared_adj': self.rho_squared_adj_,
            'n_parameters': np.prod(self.coef_.shape) + len(self.intercept_)
        }
        
        return metrics