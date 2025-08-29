"""
Enhanced BiogemeModelAdapter with calibration capabilities.

This module provides an enhanced version of the BiogemeModelAdapter that includes
market share calibration functionality, allowing models to be calibrated to match
observed market shares before performing sensitivity analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Union, Optional, Any, List

import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Expression

from .biogeme_adapter import BiogemeModelAdapter


class EnhancedBiogemeAdapter(BiogemeModelAdapter):
    """
    Enhanced BiogemeModelAdapter with calibration capabilities.
    
    This class extends the standard BiogemeModelAdapter to include methods for
    calibrating alternative-specific constants (ASCs) to match observed market shares.
    This is essential for fair comparison across different model types and for
    conducting meaningful sensitivity analyses.
    """
    
    def calculate_actual_shares(self, data: pd.DataFrame, choice_column: str = "CHOICE") -> Dict[int, float]:
        """
        Calculate actual market shares from the data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data containing choice information
        choice_column : str, default="CHOICE"
            Name of the column containing choices
            
        Returns
        -------
        Dict[int, float]
            Dictionary mapping alternative IDs to their market shares
        """
        # Calculate choice counts
        choice_counts = data[choice_column].value_counts()
        total = len(data)
        
        # Convert to shares
        actual_shares = {int(alt): count/total for alt, count in choice_counts.items()}
        
        # Ensure all alternatives are represented
        alternatives = list(self.utility_functions.keys())
        for alt in alternatives:
            if alt not in actual_shares:
                actual_shares[alt] = 0.0
                
        return actual_shares
    
    def simulate_market_shares(self, data: pd.DataFrame, betas: Optional[Dict[str, float]] = None) -> Dict[int, float]:
        """
        Simulate market shares using provided beta values.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data for simulation
        betas : Optional[Dict[str, float]], default=None
            Beta values to use. If None, uses current model betas
            
        Returns
        -------
        Dict[int, float]
            Dictionary mapping alternative IDs to their predicted market shares
        """
        # Use provided betas or get from model
        if betas is None:
            betas = self.results.get_beta_values()
        
        # Get predicted probabilities
        probs_df = self.predict_probabilities(data)
        
        # Calculate mean probabilities (market shares)
        market_shares = {}
        for alt in probs_df.columns:
            market_shares[int(alt)] = probs_df[alt].mean()
            
        return market_shares
    
    def calibrate_constants(
        self,
        data: pd.DataFrame,
        choice_column: str = "CHOICE",
        asc_prefix: str = "ASC_",
        reference_alternative: Optional[int] = None,
        max_iter: int = 20,
        tolerance: float = 1e-3,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Calibrate alternative-specific constants to match observed market shares.
        
        This method iteratively adjusts ASCs using the log-ratio formula to ensure
        the model predicts market shares that match the observed data. This is
        essential for fair model comparison and sensitivity analysis.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data containing actual choices
        choice_column : str, default="CHOICE"
            Name of the column containing choices
        asc_prefix : str, default="ASC_"
            Prefix used for ASC parameters in the model
        reference_alternative : Optional[int], default=None
            Reference alternative (if any) with ASC fixed to 0
        max_iter : int, default=20
            Maximum number of calibration iterations
        tolerance : float, default=1e-3
            Convergence tolerance for market share differences
        verbose : bool, default=True
            Whether to print calibration progress
            
        Returns
        -------
        Dict[str, float]
            Calibrated beta values
        """
        # Calculate actual market shares
        actual_shares = self.calculate_actual_shares(data, choice_column)
        
        # Get current beta values
        beta_values = self.results.get_beta_values()
        
        # Identify ASC parameters
        alternatives = list(self.utility_functions.keys())
        asc_mapping = {}
        
        for alt in alternatives:
            if alt != reference_alternative:
                # Try to find ASC for this alternative
                asc_name = f"{asc_prefix}{alt}"
                alt_name = {1: "TRAIN", 2: "SM", 3: "CAR"}.get(alt, str(alt))
                alt_asc_name = f"{asc_prefix}{alt_name}"
                
                # Check which naming convention is used
                if asc_name in beta_values:
                    asc_mapping[alt] = asc_name
                elif alt_asc_name in beta_values:
                    asc_mapping[alt] = alt_asc_name
                    
        if verbose:
            print("Starting calibration process...")
            print(f"Alternatives to calibrate: {list(asc_mapping.keys())}")
            print("\nInitial market shares:")
            predicted_shares = self.simulate_market_shares(data, beta_values)
            for alt in sorted(actual_shares.keys()):
                print(f"  Alt {alt}: Actual = {actual_shares[alt]:.3f}, Predicted = {predicted_shares[alt]:.3f}")
        
        # Iterative calibration
        for iteration in range(max_iter):
            # Get current predicted shares
            predicted_shares = self.simulate_market_shares(data, beta_values)
            
            # Calculate maximum difference
            max_diff = max(abs(actual_shares[alt] - predicted_shares[alt]) 
                          for alt in actual_shares.keys())
            
            if verbose:
                print(f"\nIteration {iteration + 1}")
                print(f"Max difference: {max_diff:.6f}")
            
            # Check convergence
            if max_diff < tolerance:
                if verbose:
                    print("\nConverged! Calibration complete.")
                    print("\nFinal market shares:")
                    for alt in sorted(actual_shares.keys()):
                        print(f"  Alt {alt}: Actual = {actual_shares[alt]:.3f}, Predicted = {predicted_shares[alt]:.3f}")
                break
            
            # Adjust ASCs
            for alt, asc_name in asc_mapping.items():
                if alt in actual_shares and alt in predicted_shares:
                    if predicted_shares[alt] > 0:  # Avoid log(0)
                        # Handle nested logit case
                        if hasattr(self, 'nests') and self.nests is not None:
                            # Check if alternative is in a nest
                            nest_param = 1.0
                            # Access the list of nests correctly
                            nests_list = []
                            if hasattr(self.nests, 'nests'):
                                nests_list = self.nests.nests
                            elif hasattr(self.nests, 'list_of_nests'):
                                nests_list = self.nests.list_of_nests()
                            
                            for nest in nests_list:
                                if alt in nest.list_of_alternatives:
                                    nest_param_name = nest.nest_param.name
                                    nest_param = beta_values.get(nest_param_name, 1.0)
                                    break
                            adjustment = nest_param * np.log(actual_shares[alt] / predicted_shares[alt])
                        else:
                            # MNL case
                            adjustment = np.log(actual_shares[alt] / predicted_shares[alt])
                        
                        beta_values[asc_name] += adjustment
        
        else:
            if verbose:
                print(f"\nWarning: Maximum iterations ({max_iter}) reached without convergence")
        
        # Update model with calibrated betas
        self.results.betas = beta_values
        
        return beta_values