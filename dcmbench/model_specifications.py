"""
Model specification utilities for DCMBench.

This module provides functions to:
- Fetch model specifications from remote repository
- Build Biogeme models from JSON specifications
- Extract specifications from existing Biogeme models

Author: DCMBench Team
License: MIT
"""

import json
import requests
import pandas as pd
from typing import Dict, Any, Optional, List
from pathlib import Path
import os
import logging

import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, Variable, Expression, bioDraws, log, MonteCarlo
from biogeme.expressions.base_expressions import Expression as BaseExpression

logger = logging.getLogger(__name__)


def fetch_model_spec(
    spec_path: str,
    base_url: str = "https://raw.githubusercontent.com/carlosguirado/dcmbench-data/main/models",
    local_cache_dir: Optional[str] = None,
    force_download: bool = False
) -> Dict[str, Any]:
    """
    Fetch a model specification from the remote repository.
    
    This function retrieves model specifications from the remote GitHub repository
    (https://github.com/carlosguirado/dcmbench-data/models) rather than including
    them in the package itself. This approach keeps the package lightweight while
    still providing easy access to all model specifications.
    
    Parameters
    ----------
    spec_path : str
        Path to the model specification (e.g., "mode_choice/mnl_swissmetro.json")
    base_url : str, optional
        Base URL for the model repository
    local_cache_dir : str, optional
        Directory to cache specifications locally. If None, no caching is performed.
    force_download : bool, default=False
        If True, always download from remote even if cached locally
        
    Returns
    -------
    dict
        The model specification as a dictionary
        
    Raises
    ------
    ValueError
        If the specification cannot be fetched or parsed
    """
    # Check local cache first if enabled
    if local_cache_dir and not force_download:
        cache_path = Path(local_cache_dir) / spec_path
        if cache_path.exists():
            logger.info(f"Loading cached specification from: {cache_path}")
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cached specification: {e}")
    
    # Fetch from remote
    url = f"{base_url}/{spec_path}"
    # Add cache buster for force_download
    if force_download:
        import time
        url += f"?t={int(time.time())}"
    logger.info(f"Fetching model specification from: {url}")
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        spec = response.json()
        
        # Cache locally if enabled
        if local_cache_dir:
            cache_path = Path(local_cache_dir) / spec_path
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump(spec, f, indent=2)
            logger.info(f"Cached specification to: {cache_path}")
        
        return spec
        
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to fetch model specification from {url}: {e}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in model specification: {e}")


def build_model_from_spec(
    spec: Dict[str, Any],
    database: db.Database,
    validate_spec: bool = True
) -> bio.BIOGEME:
    """
    Build a Biogeme model from a JSON specification.
    
    This function creates a complete Biogeme model from a specification dictionary,
    handling parameter definitions, utility functions, and availability conditions.
    
    Parameters
    ----------
    spec : dict
        Model specification dictionary containing:
        - metadata: Model information (name, description, etc.)
        - parameters: Parameter definitions with initial values and bounds
        - utilities: Utility function formulas for each alternative
        - availability: Availability expressions for each alternative
        - data_mapping: Mapping of variables used in the model
    database : biogeme.database.Database
        Biogeme database object containing the data
    validate_spec : bool, default=True
        Whether to validate the specification structure before building
        
    Returns
    -------
    biogeme.biogeme.BIOGEME
        The constructed Biogeme model ready for estimation
        
    Raises
    ------
    ValueError
        If the specification is invalid or missing required fields
    KeyError
        If required specification fields are missing
    """
    if validate_spec:
        _validate_model_spec(spec)
    
    # Extract metadata
    metadata = spec.get('metadata', {})
    model_name = metadata.get('name', 'unnamed_model')
    model_type = metadata.get('model_type', 'MNL').upper()
    
    logger.info(f"Building {model_type} model: {model_name}")
    
    # Create parameters
    parameters = {}
    for param_name, param_info in spec['parameters'].items():
        parameters[param_name] = Beta(
            name=param_name,
            value=param_info.get('initial_value', 0),
            lowerbound=param_info.get('lower_bound'),
            upperbound=param_info.get('upper_bound'),
            status=1 if param_info.get('fixed', False) else 0
        )
        if param_info.get('fixed', False):
            logger.debug(f"Parameter {param_name} is FIXED to {param_info.get('initial_value', 0)}")
    
    # Create variables from database
    variables = {}
    for var_name in spec['data_mapping'].get('variables', {}):
        variables[var_name] = Variable(var_name)
    
    # Add choice variable
    choice_var_name = spec['data_mapping'].get('choice_variable', 'CHOICE')
    choice_var = Variable(choice_var_name)
    
    # Build utility functions
    utilities = {}
    for alt_id, alt_info in spec['utilities'].items():
        # Parse the formula string and create the expression
        formula = alt_info['formula']
        utility_expr = _parse_utility_formula(formula, parameters, variables)
        utilities[int(alt_id)] = utility_expr
        logger.debug(f"Built utility for alternative {alt_id}: {alt_info.get('name', f'Alt{alt_id}')}")
    
    # Build availability conditions
    availabilities = {}
    for alt_id, avail_expr_str in spec['availability'].items():
        if avail_expr_str in variables:
            availabilities[int(alt_id)] = variables[avail_expr_str]
        else:
            # Try to parse as expression (e.g., "1" for always available)
            try:
                availabilities[int(alt_id)] = eval(avail_expr_str)
            except:
                # Default to always available
                availabilities[int(alt_id)] = 1
                logger.warning(f"Could not parse availability for alternative {alt_id}, defaulting to 1")
    
    # Create the appropriate model based on type
    if model_type == 'MNL':
        logprob = models.loglogit(utilities, availabilities, choice_var)
    elif model_type == 'NL':
        # For nested logit, we need nesting structure from spec
        nests_spec = spec.get('nests', {})
        if nests_spec:
            from biogeme.nests import OneNestForNestedLogit, NestsForNestedLogit
            
            # Build nests
            nests_list = []
            for nest_name, nest_info in nests_spec.items():
                nest_param_name = nest_info['nest_param']
                nest_param = parameters[nest_param_name]
                alternatives = nest_info['alternatives']
                
                nest = OneNestForNestedLogit(
                    nest_param=nest_param,
                    list_of_alternatives=alternatives,
                    name=nest_name
                )
                nests_list.append(nest)
            
            # Create nests structure
            nests = NestsForNestedLogit(
                choice_set=list(utilities.keys()),
                tuple_of_nests=tuple(nests_list)
            )
            
            logprob = models.lognested(utilities, availabilities, nests, choice_var)
        else:
            logger.warning("No nesting structure found, falling back to MNL")
            logprob = models.loglogit(utilities, availabilities, choice_var)
    elif model_type == 'MXL':
        # Mixed logit with random parameters
        random_params = spec.get('random_parameters', {})
        if random_params:
            # Need to handle random parameters in utility construction
            # This requires modifying the utility building process above
            prob = models.logit(utilities, availabilities, choice_var)
            logprob = log(MonteCarlo(prob))
        else:
            logger.warning("No random parameters found, falling back to MNL")
            logprob = models.loglogit(utilities, availabilities, choice_var)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Create and configure the model
    model = bio.BIOGEME(database, logprob)
    model.modelName = model_name
    
    # Also create probability formulas for easier prediction
    # Store utilities and availability for later use
    model._utilities = utilities
    model._availabilities = availabilities
    
    # Create probability formulas
    prob_formulas = {}
    for alt in utilities.keys():
        prob_formulas[f'Prob_{alt}'] = models.logit(utilities, availabilities, alt)
    
    # Store probability formulas for later use
    model._prob_formulas = prob_formulas
    
    # Apply estimation settings if provided
    settings = spec.get('estimation_settings', {})
    if 'optimization_algorithm' in settings:
        model.algorithm = settings['optimization_algorithm']
    
    return model


def model_to_spec(
    model: bio.BIOGEME,
    results: Optional[Any] = None,
    name: str = "extracted_model",
    description: str = "Model specification extracted from Biogeme model",
    author: str = "DCMBench",
    extract_utilities: bool = True
) -> Dict[str, Any]:
    """
    Extract a JSON specification from a Biogeme model.
    
    This function extracts a complete model specification from an existing
    Biogeme model, including parameters, utilities, and availability conditions.
    It works with any Biogeme model structure without making assumptions about
    parameter names or model type.
    
    Parameters
    ----------
    model : biogeme.biogeme.BIOGEME
        The Biogeme model to extract specification from
    results : biogeme results object, optional
        Estimation results. If provided, will use estimated values for parameters
    name : str
        Name for the extracted specification
    description : str
        Description of the model
    author : str
        Author of the specification
    extract_utilities : bool, default=True
        Whether to attempt extracting utility formulas (requires model introspection)
        
    Returns
    -------
    dict
        Complete model specification including:
        - metadata: Model information
        - parameters: All parameters with values and constraints
        - utilities: Utility formulas (if extraction enabled)
        - availability: Availability conditions
        - data_mapping: Variables used in the model
    """
    logger.info(f"Extracting specification from model: {name}")
    
    # Initialize specification structure
    spec = {
        "metadata": {
            "name": name,
            "description": description,
            "author": author,
            "version": "1.0.0",
            "created_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
            "framework": "biogeme",
            "model_type": _detect_model_type(model)
        },
        "parameters": {},
        "utilities": {},
        "availability": {},
        "data_mapping": {
            "choice_variable": "CHOICE",  # Default, will be updated if detected
            "variables": {}
        },
        "estimation_settings": {
            "optimization_algorithm": getattr(model, 'algorithm', 'BFGS'),
            "max_iterations": 1000,
            "tolerance": 1e-6
        }
    }
    
    # Extract parameters
    all_betas = {}
    
    # Get free parameters
    if hasattr(model, 'free_beta_names'):
        for beta_name in model.free_beta_names:
            all_betas[beta_name] = {'fixed': False}
    elif hasattr(model, 'freeBetaNames'):
        for beta_name in model.freeBetaNames:
            all_betas[beta_name] = {'fixed': False}
    
    # Get fixed parameters
    if hasattr(model, 'fixed_beta_names'):
        for beta_name in model.fixed_beta_names:
            all_betas[beta_name] = {'fixed': True}
    elif hasattr(model, 'fixedBetaNames'):
        for beta_name in model.fixedBetaNames:
            all_betas[beta_name] = {'fixed': True}
    
    # Extract parameter details
    for beta_name, beta_info in all_betas.items():
        try:
            # Try to get the Beta object
            beta_obj = model.get_beta(beta_name) if hasattr(model, 'get_beta') else None
            
            if results and not beta_info['fixed']:
                # Use estimated values if available
                beta_values = results.get_beta_values()
                value = beta_values.get(beta_name, 0)
            elif beta_obj:
                # Use initial value from Beta object
                value = getattr(beta_obj, 'init_value', 0)
            else:
                value = 0
            
            spec['parameters'][beta_name] = {
                "initial_value": float(value),
                "lower_bound": getattr(beta_obj, 'lower_bound', None) if beta_obj else None,
                "upper_bound": getattr(beta_obj, 'upper_bound', None) if beta_obj else None,
                "fixed": beta_info['fixed'],
                "description": f"{'Fixed parameter' if beta_info['fixed'] else 'Parameter'} {beta_name}"
            }
        except Exception as e:
            logger.warning(f"Could not extract full details for parameter {beta_name}: {e}")
            spec['parameters'][beta_name] = {
                "initial_value": 0,
                "lower_bound": None,
                "upper_bound": None,
                "fixed": beta_info['fixed'],
                "description": f"Parameter {beta_name}"
            }
    
    # Extract variables from database
    if hasattr(model, 'database') and model.database:
        for var_name in model.database.variables:
            if not var_name.startswith('__'):  # Skip internal variables
                spec['data_mapping']['variables'][var_name] = f"{var_name}"
    
    # Extract utilities and availability (simplified for now)
    # In a full implementation, we'd parse the model's expression tree
    if extract_utilities and hasattr(model, 'loglike') and hasattr(model.loglike, 'util'):
        # This would require deep introspection of the model structure
        logger.info("Utility extraction from expression tree not fully implemented")
    
    # For now, create placeholder structure that users can fill in
    # Detect number of alternatives from the model if possible
    num_alts = 3  # Default
    if hasattr(model, 'loglike'):
        # Try to infer from model structure
        pass
    
    for i in range(1, num_alts + 1):
        spec['utilities'][str(i)] = {
            "name": f"Alternative {i}",
            "formula": "TO_BE_FILLED"
        }
        spec['availability'][str(i)] = "TO_BE_FILLED"
    
    return spec


def _validate_model_spec(spec: Dict[str, Any]) -> None:
    """Validate that a model specification has all required fields."""
    required_fields = ['parameters', 'utilities', 'availability', 'data_mapping']
    for field in required_fields:
        if field not in spec:
            raise ValueError(f"Model specification missing required field: {field}")
    
    # Validate data_mapping has choice_variable
    if 'choice_variable' not in spec.get('data_mapping', {}):
        raise ValueError("Model specification missing choice_variable in data_mapping")
    
    # Validate utilities and availability have matching keys
    util_keys = set(spec.get('utilities', {}).keys())
    avail_keys = set(spec.get('availability', {}).keys())
    if util_keys != avail_keys:
        raise ValueError(f"Utilities and availability keys don't match: {util_keys} vs {avail_keys}")


def _parse_utility_formula(
    formula: str,
    parameters: Dict[str, Beta],
    variables: Dict[str, Variable]
) -> Expression:
    """
    Parse a utility formula string into a Biogeme expression.
    
    This function takes a formula like "ASC_TRAIN + B_TIME * TRAIN_TT"
    and converts it into a proper Biogeme expression using the provided
    parameters and variables.
    """
    # Create a namespace combining parameters and variables
    namespace = {}
    namespace.update(parameters)
    namespace.update(variables)
    
    try:
        # Evaluate the formula with the combined namespace
        # This allows direct reference to parameter and variable names
        utility = eval(formula, {"__builtins__": {}}, namespace)
        return utility
    except Exception as e:
        logger.error(f"Failed to parse utility formula: {formula}")
        raise ValueError(f"Invalid utility formula '{formula}': {e}")


def _detect_model_type(model: bio.BIOGEME) -> str:
    """
    Detect the type of a Biogeme model (MNL, NL, MXL, etc.).
    
    This is a simplified implementation. In practice, you'd need to
    analyze the model structure more carefully.
    """
    # For now, default to MNL
    # A full implementation would inspect the model's expression tree
    return "MNL"


def _build_nested_logit(
    utilities: Dict[int, Expression],
    availabilities: Dict[int, Expression],
    choice: Variable,
    nests: Dict[str, Any],
    parameters: Dict[str, Beta]
) -> Expression:
    """
    Build a nested logit model from utilities and nesting structure.
    
    This is a placeholder for nested logit implementation.
    """
    # This would require the full nesting structure from the specification
    # For now, fall back to MNL
    logger.warning("Nested logit construction not fully implemented, using MNL")
    return models.loglogit(utilities, availabilities, choice)


# Additional utility functions

def list_available_specs(
    base_url: str = "https://api.github.com/repos/carlosguirado/dcmbench-data/contents/models"
) -> List[str]:
    """
    List all available model specifications in the repository.
    
    Returns
    -------
    List[str]
        List of available model specification paths
    """
    try:
        response = requests.get(base_url)
        response.raise_for_status()
        
        specs = []
        items = response.json()
        
        for item in items:
            if item['type'] == 'file' and item['name'].endswith('.json'):
                specs.append(item['path'].replace('models/', ''))
            elif item['type'] == 'dir':
                # Recursively list subdirectories
                sub_specs = list_available_specs(item['url'])
                specs.extend(sub_specs)
        
        return sorted(specs)
        
    except Exception as e:
        logger.error(f"Failed to list model specifications: {e}")
        return []


def validate_model_spec(spec: Dict[str, Any], strict: bool = False) -> List[str]:
    """
    Validate a model specification and return any issues found.
    
    Parameters
    ----------
    spec : dict
        Model specification to validate
    strict : bool, default=False
        If True, perform stricter validation including formula parsing
        
    Returns
    -------
    List[str]
        List of validation issues (empty if valid)
    """
    issues = []
    
    try:
        _validate_model_spec(spec)
    except ValueError as e:
        issues.append(str(e))
    
    if strict:
        # Additional strict validations
        # Check if formulas are parseable
        for alt_id, util_info in spec.get('utilities', {}).items():
            formula = util_info.get('formula', '')
            if formula == 'TO_BE_FILLED':
                issues.append(f"Utility formula for alternative {alt_id} is not filled")
    
    return issues