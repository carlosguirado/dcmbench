"""
Factory module for creating model adapters.

This module provides functions for creating the appropriate adapter
for different modeling frameworks, including an automatic framework detection
function that can determine which adapter to use based on the model object.
"""

import logging
import inspect
from typing import Dict, Union, Optional, Any, List, Tuple, Type, Callable

import pandas as pd

from ..model_benchmarker.model_interface import PredictionInterface
from .registry import (
    get_adapter_class,
    get_available_frameworks,
    is_framework_available
)

logger = logging.getLogger(__name__)

# Type aliases
ModelType = Any
AdapterParameters = Dict[str, Any]

# Framework detection functions
def _detect_biogeme(model: ModelType) -> float:
    """
    Check if the model is a Biogeme model.
    
    Parameters
    ----------
    model : Any
        The model to check
        
    Returns
    -------
    float
        Confidence score between 0.0 and 1.0 that the model is a Biogeme model
    """
    confidence = 0.0
    
    # Strong indicators
    if hasattr(model, 'logprob') and callable(model.logprob):
        confidence = max(confidence, 0.9)
    
    if hasattr(model, 'modelName'):
        confidence = max(confidence, 0.8)
    
    # Medium indicators
    if hasattr(model, 'biogeme_instance'):
        confidence = max(confidence, 0.7)
        
    if hasattr(model, 'formulas'):
        confidence = max(confidence, 0.6)
    
    if hasattr(model, 'utility_functions'):
        confidence = max(confidence, 0.6)
    
    return confidence

def _detect_pytorch(model: ModelType) -> float:
    """
    Check if the model is a PyTorch model.
    
    Parameters
    ----------
    model : Any
        The model to check
        
    Returns
    -------
    float
        Confidence score between 0.0 and 1.0 that the model is a PyTorch model
    """
    confidence = 0.0
    
    # Direct instance check - highest confidence
    try:
        import torch.nn as nn
        if isinstance(model, nn.Module):
            return 1.0
    except ImportError:
        pass
    
    # Check for PyTorch attributes - high confidence
    if hasattr(model, 'parameters') and callable(model.parameters):
        try:
            parameters = list(model.parameters())
            if parameters:
                try:
                    import torch
                    if all(isinstance(p, torch.Tensor) for p in parameters):
                        return 0.95
                except ImportError:
                    # If torch isn't available, we can still make an educated guess
                    confidence = max(confidence, 0.7)
        except TypeError:
            # Not iterable, probably not a PyTorch model
            pass
    
    # Check for PyTorch-specific attributes - medium confidence
    if hasattr(model, 'state_dict') and callable(model.state_dict):
        confidence = max(confidence, 0.6)
    
    if hasattr(model, 'forward') and callable(model.forward):
        confidence = max(confidence, 0.5)
    
    if hasattr(model, 'train') and callable(model.train):
        confidence = max(confidence, 0.4)
    
    if hasattr(model, 'eval') and callable(model.eval):
        confidence = max(confidence, 0.4)
    
    return confidence

def _detect_sklearn(model: ModelType) -> float:
    """
    Check if the model is a scikit-learn model.
    
    Parameters
    ----------
    model : Any
        The model to check
        
    Returns
    -------
    float
        Confidence score between 0.0 and 1.0 that the model is a scikit-learn model
    """
    confidence = 0.0
    
    # Direct instance check - highest confidence
    try:
        from sklearn.base import BaseEstimator
        if isinstance(model, BaseEstimator):
            return 1.0
    except ImportError:
        pass
    
    # Check for sklearn-specific methods - high confidence
    has_predict = hasattr(model, 'predict') and callable(model.predict)
    has_fit = hasattr(model, 'fit') and callable(model.fit)
    
    if has_predict and has_fit:
        confidence = max(confidence, 0.8)
    
    # Check for sklearn-specific attributes - medium confidence
    if hasattr(model, 'predict_proba') and callable(model.predict_proba):
        confidence = max(confidence, 0.7)
    
    if hasattr(model, 'classes_'):
        confidence = max(confidence, 0.6)
    
    if hasattr(model, 'n_features_in_'):
        confidence = max(confidence, 0.6)
    
    if hasattr(model, 'feature_importances_'):
        confidence = max(confidence, 0.5)
    
    return confidence

def _implements_prediction_interface(model: ModelType) -> bool:
    """
    Check if the model already implements the PredictionInterface.
    
    Parameters
    ----------
    model : Any
        The model to check
        
    Returns
    -------
    bool
        True if the model implements the PredictionInterface, False otherwise
    """
    # Check for required methods
    has_predict_probs = hasattr(model, 'predict_probabilities') and callable(model.predict_probabilities)
    has_predict_choices = hasattr(model, 'predict_choices') and callable(model.predict_choices)
    
    # Check if the methods have the correct signature
    if has_predict_probs and has_predict_choices:
        try:
            # Check predict_probabilities signature
            prob_sig = inspect.signature(model.predict_probabilities)
            if len(prob_sig.parameters) != 1:
                return False
            
            # Check predict_choices signature
            choice_sig = inspect.signature(model.predict_choices)
            if len(choice_sig.parameters) != 1:
                return False
            
            return True
        except (ValueError, TypeError):
            # Exception during signature inspection
            return False
    
    return False

def detect_framework(model: ModelType) -> str:
    """
    Detect the modeling framework based on the model object type.
    
    Parameters
    ----------
    model : Any
        The model object to detect the framework for
        
    Returns
    -------
    str
        The detected framework name ('biogeme', 'pytorch', 'sklearn', 'native', or 'unknown')
    """
    # Check if it already implements the prediction interface
    if _implements_prediction_interface(model):
        return "native"
    
    # Collect confidence scores for each framework
    framework_scores = {}
    
    # Only check available frameworks
    if is_framework_available("biogeme"):
        framework_scores["biogeme"] = _detect_biogeme(model)
    
    if is_framework_available("pytorch"):
        framework_scores["pytorch"] = _detect_pytorch(model)
    
    if is_framework_available("sklearn"):
        framework_scores["sklearn"] = _detect_sklearn(model)
    
    # Get the framework with the highest confidence
    if framework_scores:
        # Must have a minimum confidence to be considered
        max_framework, max_score = max(framework_scores.items(), key=lambda x: x[1])
        if max_score >= 0.5:  # Minimum threshold for detection
            return max_framework
    
    # If no framework detected
    logger.warning(f"Could not detect framework for model of type {type(model).__name__}")
    return "unknown"

# Model type detection functions
def _detect_biogeme_model_type(model: ModelType) -> str:
    """
    Detect the type of Biogeme model.
    
    Parameters
    ----------
    model : Any
        The Biogeme model to inspect
        
    Returns
    -------
    str
        The detected model type ('mnl', 'nl', 'from_spec', or 'base')
    """
    # Check for nested logit structure
    if hasattr(model, 'nests') or hasattr(model, 'nesting_structure'):
        return "nl"
    
    # Check for model created from specification
    if hasattr(model, 'spec') or hasattr(model, 'specification'):
        return "from_spec"
    
    # Check for multinomial logit
    if (hasattr(model, 'utility_functions') or hasattr(model, 'formulas')) and not (
        hasattr(model, 'nests') or hasattr(model, 'nesting_structure')):
        return "mnl"
    
    # Default type
    return "base"

def _detect_pytorch_model_type(model: ModelType) -> str:
    """
    Detect the type of PyTorch model.
    
    Parameters
    ----------
    model : Any
        The PyTorch model to inspect
        
    Returns
    -------
    str
        The detected model type ('mnl' or 'base')
    """
    # Check for MNL structure (has beta parameters and asc/bias)
    if hasattr(model, 'beta') and (hasattr(model, 'asc') or hasattr(model, 'bias')):
        return "mnl"
    
    # Default type
    return "base"

def _detect_sklearn_model_type(model: ModelType) -> str:
    """
    Detect the type of scikit-learn model.
    
    Parameters
    ----------
    model : Any
        The scikit-learn model to inspect
        
    Returns
    -------
    str
        The detected model type ('logistic' or 'base')
    """
    # Check for LogisticRegression
    try:
        from sklearn.linear_model import LogisticRegression
        if isinstance(model, LogisticRegression):
            return "logistic"
    except ImportError:
        # If sklearn isn't available, try to infer from attributes
        if (hasattr(model, 'coef_') and hasattr(model, 'classes_') and
            hasattr(model, 'predict_proba') and callable(model.predict_proba)):
            return "logistic"
    
    # Default type
    return "base"

def detect_model_type(model: ModelType, framework: str) -> str:
    """
    Detect the specific model type within a framework.
    
    Parameters
    ----------
    model : Any
        The model to inspect
    framework : str
        The framework of the model ('biogeme', 'pytorch', 'sklearn')
        
    Returns
    -------
    str
        The detected model type
    """
    if framework == "biogeme":
        return _detect_biogeme_model_type(model)
    elif framework == "pytorch":
        return _detect_pytorch_model_type(model)
    elif framework == "sklearn":
        return _detect_sklearn_model_type(model)
    elif framework == "native":
        # Already implements PredictionInterface, no adapter needed
        return "native"
    else:
        return "base"

# Parameter extraction functions
def _extract_biogeme_parameters(model: ModelType, adapter_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract parameters for Biogeme adapters.
    
    Parameters
    ----------
    model : Any
        The Biogeme model
    adapter_type : str
        The type of adapter ('mnl', 'nl', 'from_spec', 'base')
    parameters : Dict[str, Any]
        Existing parameters provided by the user
        
    Returns
    -------
    Dict[str, Any]
        Updated parameters with inferred values
    """
    # Common parameters for all Biogeme adapters
    if 'utility_functions' not in parameters:
        if hasattr(model, 'utility_functions'):
            parameters['utility_functions'] = model.utility_functions
        elif hasattr(model, 'formulas'):
            parameters['utility_functions'] = model.formulas
    
    if 'availability' not in parameters:
        if hasattr(model, 'availability'):
            parameters['availability'] = model.availability
        elif hasattr(model, 'availability_conditions'):
            parameters['availability'] = model.availability_conditions
    
    # Additional parameters for nested logit
    if adapter_type == "nl" and 'nests' not in parameters:
        if hasattr(model, 'nests'):
            parameters['nests'] = model.nests
        elif hasattr(model, 'nesting_structure'):
            parameters['nests'] = model.nesting_structure
    
    # For spec-based adapters
    if adapter_type == "from_spec" and 'spec' not in parameters:
        if hasattr(model, 'spec'):
            parameters['spec'] = model.spec
        elif hasattr(model, 'specification'):
            parameters['spec'] = model.specification
    
    return parameters

def _extract_pytorch_parameters(model: ModelType, adapter_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract parameters for PyTorch adapters.
    
    Parameters
    ----------
    model : Any
        The PyTorch model
    adapter_type : str
        The type of adapter ('mnl', 'base')
    parameters : Dict[str, Any]
        Existing parameters provided by the user
        
    Returns
    -------
    Dict[str, Any]
        Updated parameters with inferred values
    """
    # Extract n_alternatives
    if 'n_alternatives' not in parameters:
        if hasattr(model, 'n_alternatives'):
            parameters['n_alternatives'] = model.n_alternatives
        elif hasattr(model, 'n_classes_'):
            parameters['n_alternatives'] = model.n_classes_
        elif hasattr(model, 'out_features'):
            parameters['n_alternatives'] = model.out_features
    
    # Extract alternative_ids
    if 'alternative_ids' not in parameters:
        if hasattr(model, 'alternative_ids'):
            parameters['alternative_ids'] = model.alternative_ids
        elif hasattr(model, 'classes_'):
            parameters['alternative_ids'] = [str(cls) for cls in model.classes_]
    
    # Extract device
    if 'device' not in parameters:
        if hasattr(model, 'device'):
            parameters['device'] = model.device
    
    return parameters

def _extract_sklearn_parameters(model: ModelType, adapter_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract parameters for scikit-learn adapters.
    
    Parameters
    ----------
    model : Any
        The scikit-learn model
    adapter_type : str
        The type of adapter ('logistic', 'base')
    parameters : Dict[str, Any]
        Existing parameters provided by the user
        
    Returns
    -------
    Dict[str, Any]
        Updated parameters with inferred values
    """
    # Extract alternative_ids
    if 'alternative_ids' not in parameters:
        if hasattr(model, 'classes_'):
            parameters['alternative_ids'] = [str(cls) for cls in model.classes_]
    
    # For logistic regression adapters
    if adapter_type == "logistic" and 'feature_names' not in parameters:
        if hasattr(model, 'feature_names_in_'):
            parameters['feature_names'] = model.feature_names_in_
    
    return parameters

def extract_parameters(model: ModelType, framework: str, adapter_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract required parameters for adapter creation.
    
    Parameters
    ----------
    model : Any
        The model object
    framework : str
        The framework of the model
    adapter_type : str
        The type of adapter to create
    parameters : Dict[str, Any]
        User-provided parameters
        
    Returns
    -------
    Dict[str, Any]
        Complete parameters for adapter creation
    """
    # Create a copy of the parameters to avoid modifying the original
    params = parameters.copy()
    
    # Add 'model' parameter if not already present
    params['model'] = model
    
    # Framework-specific parameter extraction
    if framework == "biogeme":
        # Remove 'model' key for Biogeme adapters which use 'biogeme_model' instead
        if 'model' in params:
            params['biogeme_model'] = params.pop('model')
        params = _extract_biogeme_parameters(model, adapter_type, params)
    elif framework == "pytorch":
        params = _extract_pytorch_parameters(model, adapter_type, params)
    elif framework == "sklearn":
        params = _extract_sklearn_parameters(model, adapter_type, params)
    
    return params

def create_model_adapter(
    model: ModelType, 
    framework: Optional[str] = None,
    adapter_type: Optional[str] = None,
    **kwargs
) -> PredictionInterface:
    """
    Create an appropriate adapter for any model based on framework.
    
    Parameters
    ----------
    model : Any
        The model object to create an adapter for
    framework : str, optional
        The framework name ('biogeme', 'pytorch', 'sklearn').
        If None, the framework will be detected automatically.
    adapter_type : str, optional
        The specific adapter type to use. If None, the type will be detected.
    **kwargs : dict
        Additional keyword arguments to pass to the adapter
        
    Returns
    -------
    PredictionInterface
        An adapter object that implements the PredictionInterface
        
    Raises
    ------
    ValueError
        If the framework is not supported or cannot be detected
    ImportError
        If the required dependencies for the adapter are not available
    RuntimeError
        If adapter creation fails
    """
    # Detect framework if not specified
    if framework is None:
        framework = detect_framework(model)
        
    if framework == "unknown":
        raise ValueError(
            f"Could not detect framework for model of type {type(model).__name__}. "
            "Please specify framework explicitly."
        )
        
    if framework == "native":
        # Model already implements the PredictionInterface
        logger.info("Model already implements PredictionInterface, no adapter needed")
        return model
    
    # Check if the framework is available
    if not is_framework_available(framework):
        raise ImportError(
            f"Framework '{framework}' is not available. Make sure the required "
            f"dependencies are installed for this framework."
        )
    
    # Detect adapter type if not specified
    if adapter_type is None:
        adapter_type = detect_model_type(model, framework)
    
    # Get the adapter class
    adapter_class = get_adapter_class(framework, adapter_type)
    if adapter_class is None:
        raise ValueError(
            f"No adapter found for framework '{framework}' and type '{adapter_type}'. "
            f"Available adapters: {get_available_frameworks()}"
        )
    
    # Extract and prepare parameters
    parameters = extract_parameters(model, framework, adapter_type, kwargs)
    
    # Create and return the adapter
    try:
        adapter = adapter_class(**parameters)
        logger.info(f"Created {adapter_class.__name__} for {framework}/{adapter_type}")
        return adapter
    except Exception as e:
        raise RuntimeError(f"Failed to create adapter: {str(e)}")

class ModelAdapterFactory:
    """
    Factory class for creating models with adapters from specifications.
    """
    
    @staticmethod
    def create_from_spec(
        spec_name: str, 
        data: pd.DataFrame, 
        framework: Optional[str] = None,
        **kwargs
    ) -> PredictionInterface:
        """
        Create a model from a specification and wrap it with the appropriate adapter.
        
        Parameters
        ----------
        spec_name : str
            Name of the model specification
        data : pd.DataFrame
            Input data for the model
        framework : str, optional
            Framework override (if different from specification)
        **kwargs : dict
            Additional parameters for model creation or adapter configuration
            
        Returns
        -------
        PredictionInterface
            An adapter-wrapped model that implements PredictionInterface
            
        Raises
        ------
        ImportError
            If the required dependencies are not available
        ValueError
            If the framework is not supported or the specification doesn't exist
        RuntimeError
            If model creation or adapter creation fails
        """
        # Import model specification tools
        try:
            from ..models.model_spec_loader import fetch_model_spec
            from ..models.model_specification import ModelSpecification
        except ImportError as e:
            raise ImportError(f"Failed to import model specification modules: {str(e)}")
        
        # Load the model specification
        try:
            spec_data = fetch_model_spec(spec_name)
            spec = ModelSpecification(spec_data)
        except Exception as e:
            raise ValueError(f"Failed to load model specification '{spec_name}': {str(e)}")
        
        # Get the framework from the spec if not provided
        if framework is None:
            framework = spec.framework
            
        # Check framework availability
        if not is_framework_available(framework):
            raise ImportError(
                f"Framework '{framework}' is not available. Make sure the required "
                f"dependencies are installed for this framework."
            )
        
        # Create the appropriate model based on framework and spec
        try:
            if framework == "biogeme":
                from ..models.biogeme_model import create_biogeme_model_from_spec
                model = create_biogeme_model_from_spec(spec, data)
            elif framework == "pytorch":
                from .pytorch_model import create_pytorch_model_from_spec
                model = create_pytorch_model_from_spec(spec, data)
            elif framework == "sklearn":
                from .sklearn_model import create_sklearn_model_from_spec
                model = create_sklearn_model_from_spec(spec, data)
            else:
                raise ValueError(f"Unsupported framework: {framework}")
        except Exception as e:
            raise RuntimeError(f"Failed to create model: {str(e)}")
        
        # Create and return the appropriate adapter
        return create_model_adapter(
            model, 
            framework=framework,
            spec=spec,  # Pass the spec to the adapter
            **kwargs
        )