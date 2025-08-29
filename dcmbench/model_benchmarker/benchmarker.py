"""
A simplified benchmarker for comparing different discrete choice models.
"""

import logging
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

logger = logging.getLogger(__name__)

class SimpleBenchmarker:
    """
    A simplified benchmarker for comparing different discrete choice models.

    This benchmarker focuses on orchestrating model comparison by:
    1. Running model predictions (probabilities and choices)
    2. Collecting metrics from model adapters
    3. Presenting results in a standardized format

    It works with any model adapters that implement:
    - predict_probabilities(data)
    - predict_choices(data)
    - calculate_choice_accuracy(data)
    - get_metrics()
    """

    def __init__(self):
        """Initialize the benchmarker."""
        self.models = {}
        self.results = {}
        self.metrics_df = None

    def register_model(self, model, name=None):
        """
        Register a model with the benchmarker.

        Parameters
        ----------
        model : object
            The model to register. Must implement predict_probabilities, 
            predict_choices, calculate_choice_accuracy, and get_metrics methods.
        name : str, optional
            The name to use for the model. If not provided, will use model.name
            if available, or a default name.
        """
        if name is None:
            name = getattr(model, 'name', f"Model_{len(self.models)}")
        
        # Validate model interface
        required_methods = [
            'predict_probabilities', 
            'predict_choices', 
            'calculate_choice_accuracy', 
            'get_metrics'
        ]
        
        for method in required_methods:
            if not hasattr(model, method) or not callable(getattr(model, method)):
                raise ValueError(f"Model {name} does not implement required method: {method}")
        
        self.models[name] = model
        logger.info(f"Registered model: {name}")
        return name

    def run_benchmark(self, data, choice_column="CHOICE"):
        """
        Run the benchmark on all registered models.

        Parameters
        ----------
        data : pandas.DataFrame
            The data to use for benchmarking.
        choice_column : str, optional
            The name of the column containing choice data.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing benchmark metrics for all models.
        """
        for name, model in self.models.items():
            logger.info(f"Benchmarking {name}...")
            try:
                start_time = time.time()
                
                # Calculate probabilities
                probabilities = model.predict_probabilities(data)
                
                # Verify probabilities sum to 1
                prob_sums = probabilities.sum(axis=1)
                if not np.allclose(prob_sums, 1.0, rtol=1e-5):
                    logger.warning(f"Model {name} probabilities do not sum to 1")
                
                # Predict choices
                predicted_choices = model.predict_choices(data)
                
                # Calculate metrics (including choice accuracy)
                # Check if the calculate_choice_accuracy method accepts a choice_column parameter
                import inspect
                sig = inspect.signature(model.calculate_choice_accuracy)
                if 'choice_column' in sig.parameters:
                    model.calculate_choice_accuracy(data, choice_column=choice_column)
                else:
                    # Use the adapter's default behavior
                    model.calculate_choice_accuracy(data)
                metrics = model.get_metrics()
                
                # Record timing
                metrics['runtime'] = time.time() - start_time
                
                # Store results
                self.results[name] = {
                    'probabilities': probabilities,
                    'predicted_choices': predicted_choices,
                    'metrics': metrics
                }
                
                logger.info(f"Successfully benchmarked {name}")
            except Exception as e:
                import traceback
                logger.error(f"Error benchmarking {name}: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                self.results[name] = {'error': str(e)}
        
        # Compile metrics into DataFrame
        self._compile_metrics()
        
        return self.metrics_df
    
    def _compile_metrics(self):
        """Compile metrics from all models into a DataFrame."""
        metrics_data = []
        
        for name, result in self.results.items():
            if 'error' in result:
                continue
                
            metrics = result['metrics'].copy()
            
            # Filter out dict values (like actual_shares, predicted_shares)
            # but keep their keys so we know they exist
            for key, value in metrics.items():
                if isinstance(value, dict):
                    metrics[key] = "dict"
            
            metrics['model'] = name
            metrics_data.append(metrics)
        
        if metrics_data:
            self.metrics_df = pd.DataFrame(metrics_data)
            # Set model as index
            self.metrics_df = self.metrics_df.set_index('model')
        else:
            self.metrics_df = pd.DataFrame()
            
        return self.metrics_df
    
    def get_comparison_dataframe(self, metrics=None):
        """
        Get the comparison results as a DataFrame.
        
        Parameters
        ----------
        metrics : list, optional
            List of metric names to include in the comparison.
            If None, will include all available metrics.
        
        Returns
        -------
        pandas.DataFrame
            The comparison DataFrame with all numeric values (not formatted).
        """
        if self.metrics_df is None or self.metrics_df.empty:
            return pd.DataFrame()
        
        if metrics is None:
            return self.metrics_df.copy()
        else:
            # Filter to requested metrics that exist
            available_metrics = [m for m in metrics if m in self.metrics_df.columns]
            return self.metrics_df[available_metrics].copy()
    
    def print_comparison(self, metrics=None, save_csv=False, csv_filename="benchmark_results.csv"):
        """
        Print a comparison of model metrics.
        
        Parameters
        ----------
        metrics : list, optional
            List of metric names to include in the comparison.
            If None, will include standard metrics.
        save_csv : bool, optional
            Whether to save the results to a CSV file. Default is False.
        csv_filename : str, optional
            The filename for the CSV file. Default is "benchmark_results.csv".
        
        Returns
        -------
        pandas.DataFrame
            The formatted comparison DataFrame.
        """
        if self.metrics_df is None or self.metrics_df.empty:
            print("No benchmark results available.")
            return None
        
        if metrics is None:
            # Default metrics to display
            metrics = [
                'final_ll', 'null_ll', 'n_parameters',
                'rho_squared', 'rho_squared_bar', 
                'choice_accuracy', 'market_share_accuracy',
                'runtime'
            ]
            # Filter to metrics that exist
            metrics = [m for m in metrics if m in self.metrics_df.columns]
        
        # Print header
        print("\nModel Comparison Results:")
        print("=" * 100)
        
        # Create a copy for display
        df_to_display = self.metrics_df[metrics].copy()
        
        # Save CSV if requested (before formatting for display)
        if save_csv:
            df_to_display.to_csv(csv_filename)
            print(f"\nResults saved to: {csv_filename}")
        
        # Format numeric columns for display
        formatted_df = df_to_display.copy()
        numeric_cols = formatted_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col == 'runtime':
                # Format runtime in seconds with 3 decimal places
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.3f}s")
            elif 'accuracy' in col.lower():
                # Format accuracy as percentage
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x*100:.2f}%")
            else:
                # Format other numeric values with 4 decimal places
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.4f}")
        
        # Use tabulate for better formatting
        print(tabulate(formatted_df, headers='keys', tablefmt='grid', 
                      showindex=True, numalign='right'))
        
        # Determine best model based on criteria
        try:
            best_models = {}
            criteria = ['rho_squared_bar', 'choice_accuracy', 'market_share_accuracy']
            criteria = [c for c in criteria if c in self.metrics_df.columns]
            
            for criterion in criteria:
                if criterion in self.metrics_df.columns:
                    best_idx = self.metrics_df[criterion].astype(float).idxmax()
                    best_models[criterion] = best_idx
            
            if best_models:
                print("\nBest Models:")
                for criterion, model in best_models.items():
                    print(f"  - {criterion}: {model}")
        except Exception as e:
            logger.error(f"Error determining best model: {str(e)}")
            print(f"\nUnable to determine best model: {str(e)}")
        
        return df_to_display
    
    def plot_comparison(self, metrics=None, figsize=(15, 5)):
        """
        Plot a comparison of model metrics.
        
        Parameters
        ----------
        metrics : list, optional
            List of metric names to include in the comparison.
            If None, will include standard metrics.
        figsize : tuple, optional
            Figure size for the plot.
        """
        if self.metrics_df is None or self.metrics_df.empty:
            print("No benchmark results available to plot.")
            return
        
        if metrics is None:
            # Default metrics to plot
            metrics = [
                'rho_squared_bar', 
                'choice_accuracy', 
                'market_share_accuracy'
            ]
            metrics = [m for m in metrics if m in self.metrics_df.columns]
        
        if not metrics:
            print("No metrics available to plot.")
            return
        
        # Create figure
        fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
        if len(metrics) == 1:
            axes = [axes]
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            if metric in self.metrics_df.columns:
                # Convert to float in case it was stored as string
                values = self.metrics_df[metric].astype(float)
                axes[i].bar(self.metrics_df.index, values)
                axes[i].set_title(metric, fontsize=14)
                axes[i].set_ylabel('Value')
                
                # Set y-limits appropriately
                if 'accuracy' in metric.lower():
                    axes[i].set_ylim([0, 1])
                else:
                    axes[i].set_ylim([0, values.max() * 1.1])
                
                # Rotate x-labels if needed
                if len(self.metrics_df.index) > 3:
                    plt.setp(axes[i].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def plot_market_shares(self, figsize=(12, 6)):
        """
        Plot market shares for all models.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure size for the plot.
        """
        # Extract market shares
        share_data = []
        
        # Check if we have actual shares in any model
        has_shares = False
        for name, result in self.results.items():
            if 'error' in result or 'metrics' not in result:
                continue
            
            metrics = result['metrics']
            if 'actual_shares' in metrics and 'predicted_shares' in metrics:
                has_shares = True
                break
        
        if not has_shares:
            print("No market share data available to plot.")
            return
            
        # Collect share data
        for name, result in self.results.items():
            if 'error' in result or 'metrics' not in result:
                continue
                
            metrics = result['metrics']
            if 'actual_shares' not in metrics or 'predicted_shares' not in metrics:
                continue
                
            actual_shares = metrics['actual_shares']
            pred_shares = metrics['predicted_shares']
            
            # Add actual shares (only once)
            if not share_data:
                for mode, share in actual_shares.items():
                    share_data.append({
                        'Model': 'Actual',
                        'Mode': f'Mode {mode}',
                        'Share': share
                    })
            
            # Add predicted shares
            for mode, share in pred_shares.items():
                share_data.append({
                    'Model': name,
                    'Mode': f'Mode {mode}',
                    'Share': share
                })
        
        if not share_data:
            print("No market share data available to plot.")
            return
            
        # Create DataFrame and plot
        share_df = pd.DataFrame(share_data)
        
        plt.figure(figsize=figsize)
        sns.barplot(x='Mode', y='Share', hue='Model', data=share_df)
        plt.title('Market Shares by Model', fontsize=16)
        plt.ylabel('Market Share')
        plt.ylim([0, 1])
        plt.legend(title='Model')
        plt.tight_layout()
        plt.show()
        
        # Create a table of market shares
        market_shares_table = pd.pivot_table(share_df, values='Share', index='Mode', columns='Model')
        print("Market Share Comparison:")
        print(market_shares_table)