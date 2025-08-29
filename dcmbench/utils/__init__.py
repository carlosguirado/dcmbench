"""Utility functions and classes for the mode choice benchmarking system."""

from .biogeme_wrapper import BiogemeModelWrapper
from .individual_parameters import *
from .metrics import *
from .preprocessing import (
    preprocess_data, 
    needs_preprocessing,
    convert_numeric_columns,
    encode_categorical_variables,
    create_derived_variables,
    create_availability_indicators,
    clean_dataset,
    debug_check_nans
)

# Import VOT analysis tools
try:
    from .vot_analysis import (
        VOTCalculator,
        FixedVOTCalculator,
        MixedLogitVOTCalculator,
        SegmentedVOTCalculator,
        VOTConfig,
        VOTPlotter,
        calculate_vot,
        plot_vot
    )
    _vot_imports = [
        'VOTCalculator',
        'FixedVOTCalculator',
        'MixedLogitVOTCalculator',
        'SegmentedVOTCalculator',
        'VOTConfig',
        'VOTPlotter',
        'calculate_vot',
        'plot_vot'
    ]
except ImportError:
    _vot_imports = []

# Import sensitivity analysis tools
try:
    from .sensitivity_analysis import (
        SensitivityAnalysis,
        ModelCalibrator,
        SensitivityAnalyzer,
        SensitivityPlotter
    )
    _sensitivity_imports = [
        'SensitivityAnalysis',
        'ModelCalibrator',
        'SensitivityAnalyzer',
        'SensitivityPlotter'
    ]
except ImportError:
    _sensitivity_imports = []

# Import VOT visualization tools
try:
    from .vot_visualization import VOTVisualization
    _vot_viz_imports = ['VOTVisualization']
except ImportError:
    _vot_viz_imports = []

# Import systematic heterogeneity VOT tools
try:
    from .vot_systematic import (
        SegmentVOT,
        SystematicHeterogeneityVOT,
        calculate_systematic_vot
    )
    _vot_systematic_imports = [
        'SegmentVOT',
        'SystematicHeterogeneityVOT',
        'calculate_systematic_vot'
    ]
except ImportError:
    _vot_systematic_imports = []

__all__ = [
    'BiogemeModelWrapper',
    'preprocess_data',
    'needs_preprocessing',
    'convert_numeric_columns',
    'encode_categorical_variables',
    'create_derived_variables',
    'create_availability_indicators',
    'clean_dataset',
    'debug_check_nans'
] + _vot_imports + _sensitivity_imports + _vot_viz_imports + _vot_systematic_imports
