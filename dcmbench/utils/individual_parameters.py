# dcmbench/utils/individual_parameters.py

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme.expressions import Expression, Beta, bioDraws, exp, Variable

class IndividualParameterCalculator:
    """Calculates individual-specific parameters from mixture model results"""
    
    def __init__(self, 
                 model_results: Any,  # Changed from bio.ModelResults
                 data: pd.DataFrame,
                 choice_col: str,
                 parameter_name: str):
        """
        Initialize calculator with model results and data
        
        Args:
            model_results: Estimated biogeme model results object
            data: DataFrame containing choice and attribute data
            choice_col: Name of column containing choices
            parameter_name: Name of parameter to calculate individual values for
        """
        self.model_results = model_results
        self.database = db.Database("indiv_params", data)
        self.choice_col = choice_col
        self.parameter_name = parameter_name
        
    def calculate_individual_parameters(self, 
                                     n_draws: int = 1000,
                                     seed: int = 42) -> pd.Series:
        """
        Calculate individual-specific parameters using Bayes theorem
        
        Args:
            n_draws: Number of draws to use in Monte Carlo simulation
            seed: Random seed for reproducibility
            
        Returns:
            Series containing individual parameter values indexed by person ID
        """
        # Set random seed
        np.random.seed(seed)
        
        # Get parameter distribution from model results
        betas = self.model_results.getBetaValues()
        std_errs = self.model_results.getStdErrValues()
        param_mean = betas[self.parameter_name]
        param_std = std_errs[self.parameter_name]
        
        # Generate random draws
        random_draws = np.random.normal(param_mean, param_std, n_draws)
        
        # Calculate probabilities for each draw
        probs = []
        for beta_r in random_draws:
            # Update model formulation with this draw
            prob = self._calculate_choice_probability(beta_r)
            probs.append(prob)
            
        probs = np.array(probs)
        
        # Calculate individual parameters using Bayes formula
        beta_weights = random_draws[:, np.newaxis] * probs
        individual_betas = np.sum(beta_weights, axis=0) / np.sum(probs, axis=0)
        
        return pd.Series(individual_betas, 
                        index=self.database.data.index,
                        name=f"{self.parameter_name}_individual")
    
    def _calculate_choice_probability(self, beta_value: float) -> np.ndarray:
        """
        Calculate choice probability for a given parameter value
        
        Args:
            beta_value: Parameter value to evaluate probability for
            
        Returns:
            Array of probabilities
        """
        raise NotImplementedError(
            "Implement specific choice probability calculation")

class SwissmetroIndividualCalculator(IndividualParameterCalculator):
    """Individual parameter calculator specifically for Swissmetro"""
    
    def _calculate_choice_probability(self, beta_value: float) -> np.ndarray:
        """
        Calculate Swissmetro choice probabilities for a parameter value
        """
        # Create database expressions
        CAR_TT = Expression('CAR_TT', self.database.data['CAR_TT'])
        TRAIN_TT = Expression('TRAIN_TT', self.database.data['TRAIN_TT'])
        SM_TT = Expression('SM_TT', self.database.data['SM_TT'])
        
        # Parameters with fixed beta value
        TRAIN_ASC = Beta('TRAIN_ASC', 0, None, None, 0)
        SM_ASC = Beta('SM_ASC', 0, None, None, 0) 
        BETA_TIME = Beta('BETA_TIME', beta_value, None, None, 0)
        
        # Calculate utilities
        V_TRAIN = TRAIN_ASC + BETA_TIME * TRAIN_TT 
        V_SM = SM_ASC + BETA_TIME * SM_TT
        V_CAR = BETA_TIME * CAR_TT
        
        # Calculate exponentials of utilities
        e_train = np.exp(V_TRAIN.getValue())
        e_sm = np.exp(V_SM.getValue())  
        e_car = np.exp(V_CAR.getValue())
        
        # Get denominator
        denom = e_train + e_sm + e_car
        
        # Get probabilities for chosen alternative
        chosen = self.database.data[self.choice_col].values
        probs = np.where(chosen == 1, e_car/denom,
                np.where(chosen == 2, e_train/denom, e_sm/denom))
        
        return probs

def plot_individual_parameters(param_values: pd.Series,
                             chosen_alt: pd.Series,
                             title: str = None):
    """
    Plot histogram of individual parameters colored by chosen alternative
    
    Args:
        param_values: Series of individual parameter values
        chosen_alt: Series of chosen alternatives
        title: Plot title
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10,6))
    
    # Plot histogram for each alternative
    for alt in sorted(chosen_alt.unique()):
        mask = chosen_alt == alt
        plt.hist(param_values[mask], 
                alpha=0.5,
                label=f'Alternative {alt}',
                bins=30)
    
    plt.xlabel('Parameter Value')
    plt.ylabel('Frequency')
    plt.title(title or 'Distribution of Individual Parameters')
    plt.legend()
    plt.grid(True)
    plt.show()


class RandomCoefficientCalculator:
    """Calculator for individual-specific parameters from random coefficient models"""
    
    def __init__(self, 
                 model_results: Any,
                 data: pd.DataFrame,
                 choice_col: str):
        """
        Initialize calculator
        
        Args:
            model_results: Estimated biogeme model results
            data: DataFrame containing choice and attribute data
            choice_col: Name of column containing choices
        """
        self.model_results = model_results
        self.database = db.Database("random_coef", data)
        self.choice_col = choice_col
        
    def calculate_individual_betas(self, 
                                 n_draws: int = 1000,
                                 seed: int = 42) -> pd.Series:
        """
        Calculate individual-specific parameters using Bayes theorem
        
        P(β|i,x) = P(i|x,β)f(β)/P(i|x)
        
        Args:
            n_draws: Number of draws for Monte Carlo simulation
            seed: Random seed for reproducibility
            
        Returns:
            Series containing individual parameter values
        """
        np.random.seed(seed)
        
        # Get parameter estimates
        betas = self.model_results.getBetaValues()
        mu = betas.get('MU', 0)  # Get mu with default if not found
        sigma = betas.get('SIGMA', 0.5)  # Get sigma with default if not found
        
        # Generate random draws from lognormal distribution
        # Shape: (n_draws,) - one value per draw
        beta_draws = -np.exp(mu + sigma * np.random.normal(0, 1, n_draws))
        
        # Calculate choice probabilities for each draw for each individual
        all_probs = np.zeros((n_draws, len(self.database.data)))
        
        for draw_idx in range(n_draws):
            # Calculate probability for this draw's beta value
            probs = self._calculate_choice_probability(beta_draws[draw_idx])
            all_probs[draw_idx, :] = probs
        
        # Calculate individual parameters using Bayes formula
        # E[β|i] = ∑(β_r * P(i|β_r)) / ∑P(i|β_r)
        # For each individual, weight the beta values by their probability
        # Shape: beta_draws[:, np.newaxis] is (n_draws, 1), all_probs is (n_draws, n_individuals)
        weighted_betas = beta_draws[:, np.newaxis] * all_probs
        
        # Sum across draws and divide by sum of probabilities
        # This gives us one beta value per individual
        individual_betas = np.sum(weighted_betas, axis=0) / np.sum(all_probs, axis=0)
        
        return pd.Series(individual_betas, 
                        index=self.database.data.index,
                        name='beta_time_individual')

    def _calculate_choice_probability(self, beta_time: float) -> np.ndarray:
        """
        Calculate choice probability for given time parameter
        """
        # Get other parameters from model
        betas = self.model_results.getBetaValues()
        
        # Parameters
        ASC_CAR = Beta('ASC_CAR', betas['ASC_CAR'], None, None, 0)
        ASC_TRAIN = Beta('ASC_TRAIN', betas['ASC_TRAIN'], None, None, 0)
        ASC_SM = Beta('ASC_SM', betas['ASC_SM'], None, None, 0)
        B_COST = Beta('B_COST', betas['B_COST'], None, None, 0)
        B_TIME = Beta('B_TIME', beta_time, None, None, 0)
        
        # Calculate utilities
        V_TRAIN = ASC_TRAIN + \
                  B_TIME * self.database.TRAIN_TT + \
                  B_COST * self.database.TRAIN_CO
        
        V_CAR = ASC_CAR + \
                B_TIME * self.database.CAR_TT + \
                B_COST * self.database.CAR_CO
        
        V_SM = ASC_SM + \
               B_TIME * self.database.SM_TT + \
               B_COST * self.database.SM_CO
        
        # Calculate exponentials
        e_train = np.exp(V_TRAIN.getValue())
        e_sm = np.exp(V_SM.getValue())  
        e_car = np.exp(V_CAR.getValue())
        
        # Get denominator
        denom = e_train + e_sm + e_car
        
        # Get probabilities for chosen alternative
        chosen = self.database.data[self.choice_col].values
        probs = np.where(chosen == 1, e_car/denom,
                        np.where(chosen == 2, e_train/denom, e_sm/denom))
        
        return probs

def plot_individual_betas_by_mode(betas: pd.Series,
                                chosen_mode: pd.Series,
                                title: str = "Distribution of Individual Time Parameters by Mode"):
    """
    Plot histogram of individual parameters colored by chosen mode
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10,6))
    
    modes = {1: "Car", 2: "Train", 3: "Swissmetro"}
    
    for mode in sorted(chosen_mode.unique()):
        mask = chosen_mode == mode
        plt.hist(betas[mask], 
                alpha=0.5,
                label=modes.get(mode, f"Mode {mode}"),
                bins=30)
    
    plt.xlabel('Time Parameter Value')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

class SwissmetroRandomCoefficientCalculator:
    """
    Combines random coefficient estimation with individual parameter calculation
    for the Swissmetro dataset.
    """
    def __init__(self, model_results: Any, data: pd.DataFrame, choice_col: str = 'CHOICE'):
        """
        Initialize calculator with model results and dataset.
    
        Args:
            model_results: Estimated biogeme model results
            data: DataFrame containing Swissmetro choice data
            choice_col: Name of column containing choices
        """
        self.model_results = model_results
        self.data = data
        self.choice_col = choice_col
        self.database = db.Database("swissmetro_mxl", data)
    
    def calculate_individual_parameters(self, n_draws: int = 1000, seed: int = 42) -> pd.DataFrame:
        """
        Calculate individual-specific parameters using Bayes theorem.
        
        Returns:
            DataFrame with individual parameter estimates (time and cost)
        """
        np.random.seed(seed)
        
        # Get parameter estimates from the model results
        betas = self.model_results.getBetaValues()
        b_time = betas.get('B_TIME', -0.5)  # Mean of time parameter
        b_time_s = betas.get('B_TIME_S', 0.3)  # Std dev of time parameter 
        b_cost = betas.get('B_COST', -1.0)  # Cost parameter (typically fixed)
        
        # For MXL with lognormal distribution for time parameter
        # Generate random draws from the population distribution
        random_draws = np.random.normal(0, 1, n_draws)
        time_param_draws = -np.exp(b_time + b_time_s * random_draws)  # Lognormal, negative
        
        # Initialize array to store individual parameters
        n_individuals = len(self.data)
        individual_time_params = np.zeros(n_individuals)
        
        # Calculate conditional parameters for each individual
        for i in range(n_individuals):
            choice = self.data[self.choice_col].iloc[i]
            
            # Calculate choice probabilities for each draw
            probs = np.zeros(n_draws)
            for r in range(n_draws):
                # Get time parameter value for this draw
                time_param = time_param_draws[r]
                
                # Calculate utilities for each alternative
                utilities = self._calculate_utilities(self.data.iloc[i], time_param, b_cost)
                
                # Calculate probability of the chosen alternative
                probs[r] = self._calculate_choice_probability(utilities, choice)
            
            # Calculate posterior mean using Bayes theorem
            # P(β|choice) ∝ P(choice|β)f(β)
            if np.sum(probs) > 0:  # Avoid division by zero
                weights = probs / np.sum(probs)  # Normalize probabilities
                individual_time_params[i] = np.sum(time_param_draws * weights)
            else:
                # If all probabilities are zero, use population mean
                individual_time_params[i] = np.mean(time_param_draws)
        
        # Return as a pandas Series
        return pd.Series(
            individual_time_params,
            index=self.data.index,
            name='time_param_individual'
        )
    
    def _calculate_utilities(self, row: pd.Series, time_param: float, cost_param: float) -> dict:
        """
        Calculate utilities for each alternative using given parameters.
        
        Args:
            row: Single row of data for an individual
            time_param: Time parameter value
            cost_param: Cost parameter value
            
        Returns:
            Dictionary of utilities for each alternative
        """
        # Get ASC values from model results if available, otherwise use defaults
        betas = self.model_results.getBetaValues()
        asc_car = betas.get('ASC_CAR', 0)
        asc_train = betas.get('ASC_TRAIN', 0)
        asc_sm = betas.get('ASC_SM', 0)
        
        # Calculate utilities
        v_car = asc_car + time_param * row.get('CAR_TT', 0) + cost_param * row.get('CAR_CO', 0)
        v_train = asc_train + time_param * row.get('TRAIN_TT', 0) + cost_param * row.get('TRAIN_CO', 0)
        v_sm = asc_sm + time_param * row.get('SM_TT', 0) + cost_param * row.get('SM_CO', 0)
        
        return {1: v_car, 2: v_train, 3: v_sm}
    
    def _calculate_choice_probability(self, utilities: dict, choice: int) -> float:
        """
        Calculate probability of chosen alternative using logit formula.
        
        Args:
            utilities: Dictionary of utilities for each alternative
            choice: Chosen alternative
            
        Returns:
            Probability of the chosen alternative
        """
        # Extract utilities
        utils = list(utilities.values())
        
        # Subtract maximum utility for numerical stability
        max_util = max(utils)
        exp_utils = [np.exp(u - max_util) for u in utils]
        
        # Sum of exponential utilities
        sum_exp = sum(exp_utils)
        
        # Get probability of chosen alternative
        if sum_exp > 0 and choice in utilities:
            # Convert choice to 0-indexed for list access
            choice_idx = choice - 1 if choice >= 1 else 0
            if 0 <= choice_idx < len(exp_utils):
                return exp_utils[choice_idx] / sum_exp
        
        # Return 0 probability if choice is invalid or sum_exp is 0
        return 0.0
    
    def calculate_vot(self, individual_time_params: pd.Series) -> pd.Series:
        """
        Calculate Value of Time based on individual time parameters.
        
        Args:
            individual_time_params: Series of individual-specific time parameters
            
        Returns:
            Series of individual-specific VOT values ($/hour)
        """
        # Get cost parameter
        betas = self.model_results.getBetaValues()
        b_cost = betas.get('B_COST', -1.0)
        
        # Calculate VOT: (time_param / cost_param) * 60 to convert from minutes to hours
        # Time and cost parameters are typically negative, so we negate to get positive VOT
        vot = -(individual_time_params / b_cost) * 60
        
        return pd.Series(vot, index=individual_time_params.index, name='VOT_$/hour')
    
    def plot_parameter_distributions(self, individual_params: pd.Series, vot: pd.Series = None):
        """
        Create visualizations of parameter distributions
    
        Args:
            individual_params: Series with individual parameter estimates
            vot: Optional Series with Value of Time estimates
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if vot is None and individual_params is not None:
            # Calculate VOT if not provided
            vot = self.calculate_vot(individual_params)
        
        # Set up plots
        if vot is not None and individual_params is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Time parameter distribution
            sns.histplot(individual_params, ax=ax1, kde=True)
            ax1.set_title('Distribution of Individual Time Parameters')
            ax1.set_xlabel('Time Parameter (utils/minute)')
            ax1.axvline(x=individual_params.mean(), color='red', linestyle='--', 
                       label=f'Mean: {individual_params.mean():.3f}')
            ax1.legend()
            
            # VOT distribution
            sns.histplot(vot, ax=ax2, kde=True)
            ax2.set_title('Distribution of Value of Time')
            ax2.set_xlabel('Value of Time ($/hour)')
            ax2.axvline(x=vot.mean(), color='red', linestyle='--', 
                       label=f'Mean: {vot.mean():.2f}')
            ax2.legend()
            
        elif individual_params is not None:
            # Only plot time parameters if VOT not available
            plt.figure(figsize=(8, 5))
            sns.histplot(individual_params, kde=True)
            plt.title('Distribution of Individual Time Parameters')
            plt.xlabel('Time Parameter (utils/minute)')
            plt.axvline(x=individual_params.mean(), color='red', linestyle='--', 
                      label=f'Mean: {individual_params.mean():.3f}')
            plt.legend()
        
        plt.tight_layout()
        plt.show()

