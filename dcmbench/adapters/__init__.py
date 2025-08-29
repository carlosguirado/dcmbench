"""
DCMBench Adapters Module

This module provides adapter classes that make models from different frameworks
compatible with DCMBench benchmarking by implementing the required prediction interface.
Adapters allow users to create models using their preferred framework and then
use them with DCMBench without writing boilerplate prediction code.
"""

import logging

# Set up logging
logger = logging.getLogger(__name__)

# Import base interfaces
from ..model_benchmarker.model_interface import PredictionInterface


# Register Biogeme adapters - always available
from .biogeme_adapter import (
    BiogemeModelAdapter,
    BiogemeMultinomialLogitAdapter,
    BiogemeNestedLogitAdapter,
    BiogemeModelFromSpecAdapter
)


# Import enhanced adapters
try:
    from .enhanced_biogeme_adapter import EnhancedBiogemeAdapter
    from .biogeme_vot_adapter import BiogemeVOTAdapter
    _has_enhanced_adapters = True
except ImportError:
    logger.info("Enhanced adapters not available")
    _has_enhanced_adapters = False

# Biogeme adapters - always available
logger.info("Biogeme adapters available")

# PyTorch adapters if available
try:
    from .pytorch_adapter import (
        PyTorchModelAdapter,
        PyTorchMultinomialLogitAdapter
    )
    _has_pytorch = True
    logger.info("PyTorch adapters available")
except ImportError:
    logger.info("PyTorch not available, skipping PyTorch adapters")
    _has_pytorch = False

# scikit-learn adapters if available
try:
    from .sklearn_adapter import (
        SklearnModelAdapter,
        LogisticRegressionAdapter
    )
    _has_sklearn = True
    logger.info("scikit-learn adapters available")
except ImportError:
    logger.info("scikit-learn not available, skipping scikit-learn adapters")
    _has_sklearn = False


# Define __all__ with essential components
__all__ = [
    # Base interfaces
    'PredictionInterface',
    
    # Factory function
    'create_model_adapter',
    
    # Biogeme adapters - always available
    'BiogemeModelAdapter',
    'BiogemeMultinomialLogitAdapter',
    'BiogemeNestedLogitAdapter',
    'BiogemeModelFromSpecAdapter',
    
]

# Add enhanced adapters to __all__ if available
if _has_enhanced_adapters:
    __all__.extend([
        'EnhancedBiogemeAdapter',
        'BiogemeVOTAdapter'
    ])

# Add PyTorch adapters to __all__ if available
if _has_pytorch:
    __all__.extend([
        'PyTorchModelAdapter',
        'PyTorchMultinomialLogitAdapter'
    ])

# Add scikit-learn adapters to __all__ if available
if _has_sklearn:
    __all__.extend([
        'SklearnModelAdapter',
        'LogisticRegressionAdapter'
    ])

# Log available frameworks
if _has_pytorch:
    logger.debug("PyTorch adapters available")
if _has_sklearn:
    logger.debug("scikit-learn adapters available")
logger.debug("Biogeme adapters available")