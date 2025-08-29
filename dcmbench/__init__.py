# dcmbench/__init__.py

from .datasets import DatasetLoader
from . import datasets
from . import model_benchmarker
from .model_specifications import (
    fetch_model_spec, 
    build_model_from_spec, 
    model_to_spec,
    list_available_specs,
    validate_model_spec
)
# Do not import models to avoid circular imports
# from . import models

__all__ = [
    'DatasetLoader', 
    'datasets', 
    'model_benchmarker',
    'fetch_model_spec',
    'build_model_from_spec',
    'model_to_spec',
    'list_available_specs',
    'validate_model_spec'
]