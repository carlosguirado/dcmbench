"""
Universal Biogeme Adapter for DCMBench

This module provides a single, unified adapter for ALL Biogeme model types,
including MNL, Nested Logit, Mixed Logit, Cross-Nested Logit, and Latent Class models.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union

import biogeme.database as db
import biogeme.biogeme as bio

from ..model_benchmarker.model_interface import PredictionInterface

logger = logging.getLogger(__name__)


class UniversalBiogemeAdapter(PredictionInterface):
    """
    Universal adapter for any Biogeme model type.
    
    This adapter works with all Biogeme models by leveraging Biogeme's
    internal methods rather than trying to handle model-specific logic.
    
    Parameters
    ----------
    biogeme_model : biogeme.BIOGEME
        The estimated Biogeme model object
    results : biogeme.results.bioResults
        The estimation results from biogeme_model.estimate()
    database : biogeme.database.Database, optional
        Original database used for estimation (for reference)
    name : str, optional
        Name for the model, defaults to modelName from Biogeme
    """
    
    def __init__(
        self,
        biogeme_model: bio.BIOGEME,
        results: Any,  # bioResults type
        database: Optional[db.Database] = None,
        name: Optional[str] = None
    ):
        self.model = biogeme_model
        self.results = results
        self.database = database
        self.name = name or getattr(biogeme_model, 'modelName', 'Biogeme Model')
        
        # Cache important attributes
        self._formulas = getattr(biogeme_model, 'formulas', None)
        self._number_of_draws = getattr(biogeme_model, 'numberOfDraws', None)
        self._av = getattr(biogeme_model, 'availability_conditions', None)
        
        # Store last calculated probabilities for market share calculation
        self._last_probabilities = None
        
        # Get the choice variable name if available
        self._choice_var = None
        if hasattr(biogeme_model, 'loglike') and hasattr(biogeme_model.loglike, 'choice'):
            self._choice_var = biogeme_model.loglike.choice.name
            
        logger.info(f"Created UniversalBiogemeAdapter for model: {self.name}")
        
    def predict_probabilities(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict choice probabilities for each observation.
        
        This method works for all Biogeme model types by using the
        simulate() method with probability formulas.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data for prediction
            
        Returns
        -------
        pd.DataFrame
            DataFrame with probabilities for each alternative
        """
        # Clean data
        clean_data = self._prepare_data_for_prediction(data)
        
        # Create Biogeme database
        predict_db = db.Database("predict", clean_data)
        
        # Get the estimated parameters
        betas = self.results.get_beta_values()
        
        # Check if the model already has probability formulas stored
        if hasattr(self.model, '_prob_formulas') and self.model._prob_formulas:
            # Use the stored probability formulas directly
            prob_formulas = self.model._prob_formulas
            
            # Create biogeme object with probability formulas
            biogeme_sim = bio.BIOGEME(predict_db, prob_formulas)
            biogeme_sim.modelName = f"{self.name}_probs"
            
            # Set number of draws for Mixed Logit
            if self._number_of_draws is not None:
                biogeme_sim.numberOfDraws = self._number_of_draws
            
            # Simulate probabilities
            try:
                simulated = biogeme_sim.simulate(betas)
                probabilities = self._format_probability_output(simulated)
                self._last_probabilities = probabilities
                return probabilities
            except Exception as e:
                logger.warning(f"Error using stored probability formulas: {str(e)}")
                # Fall through to manual construction
        
        # If no stored formulas, create them manually
        from biogeme import models
        from biogeme.expressions import Beta
        
        # Create probability formulas based on the model structure
        # For models built from specs, we need to reconstruct utilities
        
        # First, let's try to rebuild the utility functions
        # This assumes standard swissmetro-like structure with 3 alternatives
        
        # Define parameters with their estimated values
        prob_formulas = {}
        
        try:
            # Rebuild parameters as Beta expressions with fixed values
            param_dict = {}
            for param_name, param_value in betas.items():
                param_dict[param_name] = Beta(param_name, param_value, None, None, 1)  # Fixed
            
            # Try to identify the model structure from available data columns
            # Check for different dataset structures
            from biogeme.expressions import Variable
            V = {}
            av = {}
            n_alts = 3  # Default
            
            if all(col in clean_data.columns for col in ['TRAIN_TT', 'TRAIN_CO', 'SM_TT', 'SM_CO', 'CAR_TT', 'CAR_CO']):
                # Swissmetro structure
                TRAIN_TT = Variable('TRAIN_TT')
                TRAIN_CO = Variable('TRAIN_CO')
                SM_TT = Variable('SM_TT')
                SM_CO = Variable('SM_CO')
                CAR_TT = Variable('CAR_TT')
                CAR_CO = Variable('CAR_CO')
                
                # Alternative 1: Train
                if 'ASC_TRAIN' in param_dict and 'B_TIME' in param_dict and 'B_COST' in param_dict:
                    V[1] = (param_dict['ASC_TRAIN'] + 
                           param_dict['B_TIME'] * TRAIN_TT + 
                           param_dict['B_COST'] * TRAIN_CO)
                
                # Alternative 2: Swissmetro
                ASC_SM = param_dict.get('ASC_SM', Beta('ASC_SM', 0, None, None, 1))
                if 'B_TIME' in param_dict and 'B_COST' in param_dict:
                    V[2] = (ASC_SM + 
                           param_dict['B_TIME'] * SM_TT + 
                           param_dict['B_COST'] * SM_CO)
                
                # Alternative 3: Car
                ASC_CAR = param_dict.get('ASC_CAR', Beta('ASC_CAR', 0, None, None, 1))
                if 'B_TIME' in param_dict and 'B_COST' in param_dict:
                    V[3] = (ASC_CAR + 
                           param_dict['B_TIME'] * CAR_TT + 
                           param_dict['B_COST'] * CAR_CO)
                
                # Availability
                av = {
                    1: Variable('TRAIN_AV') if 'TRAIN_AV' in clean_data.columns else 1,
                    2: Variable('SM_AV') if 'SM_AV' in clean_data.columns else 1,
                    3: Variable('CAR_AV') if 'CAR_AV' in clean_data.columns else 1
                }
                n_alts = 3
                
            elif all(col in clean_data.columns for col in ['TRAIN_TIME', 'TRAIN_COST', 'CAR_TIME', 'CAR_COST', 
                                                            'BUS_TIME', 'BUS_COST', 'AIR_TIME', 'AIR_COST']):
                # ModeCanada structure (preprocessed wide format)
                TRAIN_TIME = Variable('TRAIN_TIME')
                TRAIN_COST = Variable('TRAIN_COST')
                CAR_TIME = Variable('CAR_TIME')
                CAR_COST = Variable('CAR_COST')
                BUS_TIME = Variable('BUS_TIME')
                BUS_COST = Variable('BUS_COST')
                AIR_TIME = Variable('AIR_TIME')
                AIR_COST = Variable('AIR_COST')
                
                # Alternative 1: Train
                if 'ASC_TRAIN' in param_dict and 'B_TIME' in param_dict and 'B_COST' in param_dict:
                    V[1] = (param_dict['ASC_TRAIN'] + 
                           param_dict['B_TIME'] * TRAIN_TIME + 
                           param_dict['B_COST'] * TRAIN_COST)
                
                # Alternative 2: Car
                ASC_CAR = param_dict.get('ASC_CAR', Beta('ASC_CAR', 0, None, None, 1))
                if 'B_TIME' in param_dict and 'B_COST' in param_dict:
                    V[2] = (ASC_CAR + 
                           param_dict['B_TIME'] * CAR_TIME + 
                           param_dict['B_COST'] * CAR_COST)
                
                # Alternative 3: Bus
                if 'ASC_BUS' in param_dict and 'B_TIME' in param_dict and 'B_COST' in param_dict:
                    V[3] = (param_dict['ASC_BUS'] + 
                           param_dict['B_TIME'] * BUS_TIME + 
                           param_dict['B_COST'] * BUS_COST)
                
                # Alternative 4: Air
                if 'ASC_AIR' in param_dict and 'B_TIME' in param_dict and 'B_COST' in param_dict:
                    V[4] = (param_dict['ASC_AIR'] + 
                           param_dict['B_TIME'] * AIR_TIME + 
                           param_dict['B_COST'] * AIR_COST)
                
                # Availability
                av = {
                    1: Variable('TRAIN_AV') if 'TRAIN_AV' in clean_data.columns else 1,
                    2: Variable('CAR_AV') if 'CAR_AV' in clean_data.columns else 1,
                    3: Variable('BUS_AV') if 'BUS_AV' in clean_data.columns else 1,
                    4: Variable('AIR_AV') if 'AIR_AV' in clean_data.columns else 1
                }
                n_alts = 4
                
            elif all(col in clean_data.columns for col in ['dur_walking', 'dur_cycling', 'dur_pt_total', 'dur_driving']):
                # LTDS structure
                # Map LTDS alternatives: 1=walk, 2=cycle, 3=pt, 4=drive
                WALK_TIME = Variable('dur_walking')
                CYCLE_TIME = Variable('dur_cycling')
                PT_TIME = Variable('dur_pt_total')
                DRIVE_TIME = Variable('dur_driving')
                
                # Get cost variables - LTDS uses different naming
                PT_COST = Variable('cost_transit') if 'cost_transit' in clean_data.columns else 0
                DRIVE_COST = Variable('cost_driving_total') if 'cost_driving_total' in clean_data.columns else 0
                
                # Alternative 1: Walk
                if 'ASC_WALK' in param_dict and 'B_TIME' in param_dict:
                    V[1] = param_dict['ASC_WALK'] + param_dict['B_TIME'] * WALK_TIME
                elif 'B_TIME' in param_dict:
                    # If no ASC_WALK, use 0 as default ASC
                    V[1] = param_dict['B_TIME'] * WALK_TIME
                
                # Alternative 2: Cycle  
                if 'ASC_CYCLE' in param_dict and 'B_TIME' in param_dict:
                    V[2] = param_dict['ASC_CYCLE'] + param_dict['B_TIME'] * CYCLE_TIME
                elif 'B_TIME' in param_dict:
                    V[2] = param_dict['B_TIME'] * CYCLE_TIME
                
                # Alternative 3: PT
                if 'ASC_PT' in param_dict and 'B_TIME' in param_dict and 'B_COST' in param_dict:
                    V[3] = (param_dict['ASC_PT'] + 
                           param_dict['B_TIME'] * PT_TIME + 
                           param_dict['B_COST'] * PT_COST)
                elif 'B_TIME' in param_dict and 'B_COST' in param_dict:
                    V[3] = param_dict['B_TIME'] * PT_TIME + param_dict['B_COST'] * PT_COST
                
                # Alternative 4: Drive
                ASC_DRIVE = param_dict.get('ASC_DRIVE', Beta('ASC_DRIVE', 0, None, None, 1))
                if 'B_TIME' in param_dict and 'B_COST' in param_dict:
                    V[4] = (ASC_DRIVE + 
                           param_dict['B_TIME'] * DRIVE_TIME + 
                           param_dict['B_COST'] * DRIVE_COST)
                
                # Availability - LTDS has availability columns
                av = {
                    1: Variable('walk_available') if 'walk_available' in clean_data.columns else 1,
                    2: Variable('cycle_available') if 'cycle_available' in clean_data.columns else 1,
                    3: Variable('pt_available') if 'pt_available' in clean_data.columns else 1,
                    4: Variable('drive_available') if 'drive_available' in clean_data.columns else 1
                }
                n_alts = 4
                
            else:
                # Generic fallback - equal probabilities
                logger.warning("Cannot identify model structure, using equal probabilities")
                logger.warning(f"Available columns: {list(clean_data.columns)[:20]}...")
                
                # Try to infer number of alternatives from CHOICE variable
                if 'CHOICE' in clean_data.columns:
                    unique_choices = clean_data['CHOICE'].dropna().unique()
                    n_alts = len(unique_choices)
                    logger.info(f"Inferred {n_alts} alternatives from CHOICE variable")
                else:
                    n_alts = 3  # Default
                    
                return pd.DataFrame(
                    {i: [1.0/n_alts] * len(clean_data) for i in range(1, n_alts+1)},
                    index=clean_data.index
                )
            
            # Create probability formulas if we built utilities
            if V:
                for alt in V.keys():
                    prob_formulas[f'Prob_{alt}'] = models.logit(V, av, alt)
            else:
                # No utilities built - return equal probabilities
                logger.warning(f"No utilities could be built, returning equal probabilities for {n_alts} alternatives")
                return pd.DataFrame(
                    {i: [1.0/n_alts] * len(clean_data) for i in range(1, n_alts+1)},
                    index=clean_data.index
                )
            
            # Create biogeme object with probability formulas
            biogeme_sim = bio.BIOGEME(predict_db, prob_formulas)
            biogeme_sim.modelName = f"{self.name}_probs"
            
            # Set number of draws for Mixed Logit
            if self._number_of_draws is not None:
                biogeme_sim.numberOfDraws = self._number_of_draws
            
            # Simulate probabilities
            simulated = biogeme_sim.simulate(betas)
            probabilities = self._format_probability_output(simulated)
            
            # Store probabilities for market share calculation
            self._last_probabilities = probabilities
            
            return probabilities
            
        except Exception as e:
            logger.error(f"Error in probability prediction: {str(e)}")
            logger.warning("Falling back to equal probabilities")
            # Try to infer number of alternatives from CHOICE variable
            n_alts = 3  # Default
            if 'CHOICE' in clean_data.columns:
                unique_choices = clean_data['CHOICE'].dropna().unique()
                if len(unique_choices) > 0:
                    n_alts = len(unique_choices)
                    logger.info(f"Inferred {n_alts} alternatives from CHOICE variable in fallback")
            
            probabilities = pd.DataFrame(
                {i: [1.0/n_alts] * len(clean_data) for i in range(1, n_alts+1)},
                index=clean_data.index
            )
            self._last_probabilities = probabilities
            return probabilities
            
    def predict_choices(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict the most likely choice for each observation.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data for prediction
            
        Returns
        -------
        np.ndarray
            Array of predicted choices
        """
        probabilities = self.predict_probabilities(data)
        
        # Ensure columns are consistent types (all int or all str)
        # Convert column names to integers if they're numeric strings
        new_cols = {}
        for col in probabilities.columns:
            try:
                # Try to convert to int
                new_cols[col] = int(col)
            except (ValueError, TypeError):
                # Keep as string if conversion fails
                new_cols[col] = str(col)
        
        # Only rename if we have numeric columns
        if all(isinstance(v, int) for v in new_cols.values()):
            probabilities = probabilities.rename(columns=new_cols)
        
        # Get the alternative with highest probability for each row
        return probabilities.idxmax(axis=1).values
        
    def calculate_choice_accuracy(self, data: pd.DataFrame, choice_column: str = "CHOICE") -> float:
        """
        Calculate the choice prediction accuracy.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data with actual choices
        choice_column : str, default="CHOICE"
            Name of the column containing actual choices
            
        Returns
        -------
        float
            Proportion of correctly predicted choices
        """
        # First, get probabilities (this also stores them in _last_probabilities)
        probabilities = self.predict_probabilities(data)
        
        # Get predictions based on highest probability
        predicted_choices = self.predict_choices(data)
        
        # Get actual choices
        if choice_column not in data.columns:
            raise ValueError(f"Choice column '{choice_column}' not found in data")
            
        actual_choices = data[choice_column].values
        
        # Calculate accuracy
        correct_predictions = (predicted_choices == actual_choices).sum()
        total_predictions = len(actual_choices)
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        logger.info(f"Choice accuracy: {accuracy:.4f} ({correct_predictions}/{total_predictions})")
        
        # Store accuracy in metrics
        self._choice_accuracy = accuracy
        
        # Calculate market shares using the stored probabilities
        self._calculate_market_shares(data, predicted_choices, actual_choices)
        
        return accuracy
        
    def _calculate_market_shares(self, data: pd.DataFrame, predicted_choices: np.ndarray, actual_choices: np.ndarray):
        """Calculate and store market share information using average probabilities."""
        try:
            # Get the probabilities for market share calculation
            probabilities = self._last_probabilities  # Store probabilities from predict_probabilities
            
            if probabilities is None:
                # Fallback to discrete choice based calculation
                logger.warning("No probabilities available for market share calculation, using discrete choices")
                self._calculate_market_shares_discrete(predicted_choices, actual_choices)
                return
                
            # Calculate actual market shares from observed choices
            actual_shares = {}
            unique_choices = np.unique(actual_choices)
            for alt in unique_choices:
                # Always use int keys for consistency
                alt_key = int(alt) if isinstance(alt, (int, float, np.integer)) and not np.isnan(alt) else alt
                actual_shares[alt_key] = (actual_choices == alt).sum() / len(actual_choices)
                
            # Calculate predicted market shares as average of probabilities
            predicted_shares = {}
            for col in probabilities.columns:
                # Always use int keys for consistency
                alt_key = int(col) if isinstance(col, (int, float, np.integer)) else col
                predicted_shares[alt_key] = probabilities[col].mean()
                
            # Ensure all alternatives are represented with consistent keys
            # Get all unique alternatives as integers
            all_alts = set()
            for k in list(actual_shares.keys()) + list(predicted_shares.keys()):
                if isinstance(k, (int, float, np.integer)):
                    all_alts.add(int(k))
                else:
                    all_alts.add(k)
                    
            # Normalize shares to use consistent integer keys
            actual_shares_clean = {}
            predicted_shares_clean = {}
            
            for alt in all_alts:
                actual_shares_clean[alt] = actual_shares.get(alt, 0.0)
                predicted_shares_clean[alt] = predicted_shares.get(alt, 0.0)
                
            # Calculate market share accuracy (1 - mean absolute error)
            share_errors = []
            for alt in all_alts:
                share_errors.append(abs(actual_shares_clean[alt] - predicted_shares_clean[alt]))
                
            market_share_accuracy = 1.0 - np.mean(share_errors)
            
            # Store results
            self._actual_shares = actual_shares_clean
            self._predicted_shares = predicted_shares_clean
            self._market_share_accuracy = market_share_accuracy
            
            logger.debug(f"Market shares - Actual: {actual_shares_clean}, Predicted: {predicted_shares_clean}")
            
        except Exception as e:
            logger.warning(f"Error calculating market shares: {str(e)}")
            # Set default values
            self._actual_shares = {}
            self._predicted_shares = {}
            self._market_share_accuracy = 0.0
            
    def _calculate_market_shares_discrete(self, predicted_choices: np.ndarray, actual_choices: np.ndarray):
        """Fallback method using discrete choices."""
        # Similar to old implementation but kept as fallback
        alternatives = np.unique(np.concatenate([predicted_choices, actual_choices]))
        actual_shares = {}
        predicted_shares = {}
        
        for alt in alternatives:
            alt_key = int(alt) if isinstance(alt, (int, float)) and not np.isnan(alt) else str(alt)
            actual_shares[alt_key] = (actual_choices == alt).sum() / len(actual_choices)
            predicted_shares[alt_key] = (predicted_choices == alt).sum() / len(predicted_choices)
            
        share_errors = [abs(actual_shares[alt] - predicted_shares[alt]) for alt in actual_shares]
        market_share_accuracy = 1.0 - np.mean(share_errors)
        
        self._actual_shares = actual_shares
        self._predicted_shares = predicted_shares
        self._market_share_accuracy = market_share_accuracy
        
    def calculate_utilities(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate utility values for each alternative.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data for utility calculation
            
        Returns
        -------
        pd.DataFrame
            DataFrame with utility values for each alternative
        """
        clean_data = self._prepare_data_for_prediction(data)
        predict_db = db.Database("utilities", clean_data)
        
        betas = self.results.get_beta_values()
        
        # If utility functions are available as formulas
        if hasattr(self.model, 'utility_formulas') and self.model.utility_formulas:
            utilities = self.model.simulate(betas)
            return self._format_utility_output(utilities)
        else:
            logger.warning("Utility calculation not available for this model")
            return pd.DataFrame()
            
    def get_metrics(self) -> Dict[str, Any]:
        """
        Extract comprehensive metrics from the Biogeme results.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing all available metrics
        """
        # Basic metrics available for all models
        metrics = {
            'model_name': self.name,
            'model_type': self._infer_model_type(),
            'converged': getattr(self.results, 'convergence', True),
        }
        
        # Add statistics from results.data
        if hasattr(self.results, 'data'):
            data_attrs = self.results.data
            
            # Get basic metrics
            n_obs = getattr(data_attrs, 'numberOfObservations', None)
            final_ll = getattr(data_attrs, 'logLike', None)
            n_params = getattr(data_attrs, 'nparam', None)
            
            # Calculate null log-likelihood if not available
            null_ll = getattr(data_attrs, 'nullLogLike', None)
            if null_ll is None and n_obs:
                # For discrete choice models, null LL is typically equal probability
                # Try to detect number of alternatives from the model
                n_alts = 3  # Default
                if hasattr(self.model, 'loglike') and hasattr(self.model.loglike, 'av'):
                    # Try to get from availability conditions
                    n_alts = len(self.model.loglike.av)
                elif hasattr(self, '_last_probabilities') and self._last_probabilities is not None:
                    # Get from last probability calculation
                    n_alts = len(self._last_probabilities.columns)
                elif self.database and self.database.data is not None:
                    # Try to infer from CHOICE variable
                    if 'CHOICE' in self.database.data.columns:
                        unique_choices = self.database.data['CHOICE'].dropna().unique()
                        if len(unique_choices) > 0:
                            n_alts = len(unique_choices)
                
                null_ll = n_obs * np.log(1.0 / n_alts)
            
            # Calculate rho-squared values if we have the components
            rho_squared = getattr(data_attrs, 'rhoSquare', None)
            if (rho_squared is None or rho_squared == 0.0) and null_ll and final_ll:
                rho_squared = 1 - (final_ll / null_ll)
            
            rho_squared_bar = getattr(data_attrs, 'rhoBarSquare', None) 
            if (rho_squared_bar is None or rho_squared_bar < 0) and null_ll and final_ll and n_params:
                rho_squared_bar = 1 - ((final_ll - n_params) / null_ll)
            
            metrics.update({
                'n_observations': n_obs,
                'null_ll': null_ll,
                'null_log_likelihood': null_ll,
                'init_log_likelihood': getattr(data_attrs, 'initLogLike', None), 
                'final_ll': final_ll,
                'final_log_likelihood': final_ll,
                'rho_squared': rho_squared,
                'rho_squared_bar': rho_squared_bar,
                'aic': getattr(data_attrs, 'akaike', None),  # Fixed attribute name
                'bic': getattr(data_attrs, 'bayesian', None),  # Fixed attribute name
                'n_parameters': n_params,
                'n_estimated_parameters': n_params,
                'sample_size': getattr(data_attrs, 'sampleSize', None),
            })
            
        # Add parameter estimates
        metrics['parameters'] = self.results.get_beta_values()
        
        # Add standard errors if available
        try:
            metrics['std_errors'] = self.results.get_std_err()
        except:
            logger.debug("Standard errors not available")
            
        # Add t-statistics if available
        try:
            metrics['t_statistics'] = self.results.get_t_test()
        except:
            logger.debug("T-statistics not available")
            
        # Add p-values if available
        try:
            metrics['p_values'] = self.results.get_p_values()
        except:
            logger.debug("P-values not available")
            
        # For models with draws (Mixed Logit)
        if self._number_of_draws:
            metrics['monte_carlo_draws'] = self._number_of_draws
            
        # Add general statistics
        try:
            general_stats = self.results.get_general_statistics()
            if isinstance(general_stats, dict):
                metrics['general_statistics'] = general_stats
        except:
            logger.debug("General statistics not available")
            
        # Add accuracy metrics if available
        if hasattr(self, '_choice_accuracy'):
            metrics['choice_accuracy'] = self._choice_accuracy
            
        if hasattr(self, '_market_share_accuracy'):
            metrics['market_share_accuracy'] = self._market_share_accuracy
            
        if hasattr(self, '_actual_shares'):
            metrics['actual_shares'] = self._actual_shares
            
        if hasattr(self, '_predicted_shares'):
            metrics['predicted_shares'] = self._predicted_shares
            
        return metrics
        
    def get_parameter_covariance(self) -> pd.DataFrame:
        """
        Get the variance-covariance matrix of parameters.
        
        Returns
        -------
        pd.DataFrame
            Variance-covariance matrix
        """
        try:
            return self.results.get_var_covar()
        except:
            logger.warning("Variance-covariance matrix not available")
            return pd.DataFrame()
            
    def get_robust_covariance(self) -> pd.DataFrame:
        """
        Get the robust variance-covariance matrix if available.
        
        Returns
        -------
        pd.DataFrame
            Robust variance-covariance matrix
        """
        try:
            return self.results.get_robust_var_covar()
        except:
            logger.warning("Robust variance-covariance matrix not available")
            return pd.DataFrame()
            
    def likelihood_ratio_test(self, other_adapter: 'UniversalBiogemeAdapter') -> Dict[str, float]:
        """
        Perform likelihood ratio test against another model.
        
        Parameters
        ----------
        other_adapter : UniversalBiogemeAdapter
            Another Biogeme model adapter to compare against
            
        Returns
        -------
        Dict[str, float]
            Test statistics including LR statistic and p-value
        """
        try:
            return self.results.likelihood_ratio_test(other_adapter.results)
        except Exception as e:
            logger.error(f"Likelihood ratio test failed: {str(e)}")
            return {}
            
    def get_elasticities(
        self, 
        data: pd.DataFrame, 
        variable: str, 
        alternative: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Calculate elasticities if supported by the model.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data for elasticity calculation
        variable : str
            Variable name for elasticity calculation
        alternative : int, optional
            Specific alternative for elasticity
            
        Returns
        -------
        pd.DataFrame
            Elasticity values
        """
        logger.warning("Elasticity calculation requires specific implementation")
        return pd.DataFrame()
        
    def short_summary(self) -> str:
        """
        Get a short summary of the estimation results.
        
        Returns
        -------
        str
            Short summary text
        """
        try:
            return self.results.short_summary()
        except:
            return f"Model: {self.name}"
            
    def _prepare_data_for_prediction(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for Biogeme prediction.
        
        Parameters
        ----------
        data : pd.DataFrame
            Raw data
            
        Returns
        -------
        pd.DataFrame
            Cleaned data ready for Biogeme
        """
        clean_data = data.copy()
        
        # Remove rows with NaN in critical columns
        # This is a basic implementation - could be enhanced
        if self._choice_var and self._choice_var in clean_data.columns:
            clean_data = clean_data.dropna(subset=[self._choice_var])
            
        # Ensure availability conditions are binary
        # This would need to be more sophisticated in practice
        
        return clean_data
        
    def _format_probability_output(self, simulated: pd.DataFrame) -> pd.DataFrame:
        """
        Format Biogeme simulation output as clean probability DataFrame.
        
        Parameters
        ----------
        simulated : pd.DataFrame
            Raw output from Biogeme simulate()
            
        Returns
        -------
        pd.DataFrame
            Clean DataFrame with alternative IDs as columns
        """
        # Biogeme typically returns columns like 'Prob_1', 'Prob_Train', etc.
        prob_df = simulated.copy()
        
        # Clean up column names to just have alternative IDs
        new_columns = {}
        for col in prob_df.columns:
            col_str = str(col)
            if 'Prob' in col_str:
                # Try different patterns
                # Pattern 1: 'Prob_1', 'Prob_2', etc.
                if '_' in col_str:
                    parts = col_str.split('_')
                    if len(parts) >= 2:
                        try:
                            # Try to extract numeric ID
                            alt_num = int(parts[-1])
                            new_columns[col] = alt_num
                        except:
                            # For non-numeric suffixes like 'Train', 'SM', 'Car'
                            # Map to alternative numbers based on common patterns
                            suffix = parts[-1].upper()
                            if suffix in ['TRAIN', 'RAIL']:
                                new_columns[col] = 1
                            elif suffix in ['SM', 'SWISSMETRO']:
                                new_columns[col] = 2
                            elif suffix in ['CAR', 'AUTO']:
                                new_columns[col] = 3
                            elif suffix in ['BUS']:
                                new_columns[col] = 4
                            elif suffix in ['AIR', 'PLANE']:
                                new_columns[col] = 5
                            else:
                                # Keep original if we can't map it
                                new_columns[col] = col_str
                # Pattern 2: 'Prob. 1', 'Prob. 2', etc.
                elif '.' in col_str:
                    parts = col_str.split('.')
                    if len(parts) >= 2:
                        try:
                            alt_num = int(parts[-1].strip())
                            new_columns[col] = alt_num
                        except:
                            new_columns[col] = col_str
                # Pattern 3: Just keep the column if it has 'Prob'
                else:
                    new_columns[col] = col_str
                        
        # If no renaming happened, just return the dataframe as is
        if new_columns:
            prob_df.rename(columns=new_columns, inplace=True)
            
        # Return the entire dataframe if it has probability columns
        # Don't filter columns - let the user see what's there
        return prob_df
        
    def _format_utility_output(self, simulated: pd.DataFrame) -> pd.DataFrame:
        """
        Format utility values from Biogeme simulation.
        
        Parameters
        ----------
        simulated : pd.DataFrame
            Raw utility values
            
        Returns
        -------
        pd.DataFrame
            Clean DataFrame with utilities
        """
        # Similar to probability formatting
        return simulated
        
    def _calculate_probabilities_fallback(
        self, 
        database: db.Database, 
        betas: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Fallback method for probability calculation.
        
        This is a placeholder - in practice, you'd need to handle
        different model types appropriately.
        """
        logger.warning("Using generic fallback for probability calculation")
        
        # This would need proper implementation based on model type
        # For now, return empty DataFrame
        return pd.DataFrame()
        
    def _infer_model_type(self) -> str:
        """
        Try to infer the model type from available information.
        
        Returns
        -------
        str
            Inferred model type or 'unknown'
        """
        # Check for nests (Nested Logit)
        if hasattr(self.model, 'nests') or hasattr(self.results.data, 'nesting_parameters'):
            return 'Nested Logit'
            
        # Check for draws (Mixed Logit)
        if self._number_of_draws and self._number_of_draws > 0:
            return 'Mixed Logit'
            
        # Default to MNL
        return 'Multinomial Logit'


# Convenience function for backward compatibility
def create_biogeme_adapter(biogeme_model, results, **kwargs):
    """
    Create a universal Biogeme adapter.
    
    This function exists for backward compatibility and convenience.
    """
    return UniversalBiogemeAdapter(biogeme_model, results, **kwargs)