"""
Simple dataset fetcher that doesn't depend on models.
This avoids circular imports while providing basic dataset loading functionality.
"""

import os
import pandas as pd
import logging
import urllib.request
from pathlib import Path
import gzip
import shutil

logger = logging.getLogger(__name__)

def fetch_data(dataset_name, use_local_cache=True, local_cache_dir=None):
    """
    Fetch a dataset from the remote repository or local cache.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset to fetch
    use_local_cache : bool, default=True
        Whether to use local cache
    local_cache_dir : str, optional
        Directory to use for caching. If None, uses ~/.dcmbench/datasets
        
    Returns
    -------
    pandas.DataFrame
        The loaded dataset
    """
    # Map dataset name to URL - hardcoded for simplicity
    base_url = "https://raw.githubusercontent.com/carlosguirado/dcmbench-datasets/master/datasets/"
    dataset_map = {
        'swissmetro_dataset': base_url + "swissmetro/swissmetro.csv.gz",
        'ltds_dataset': base_url + "ltds/ltds.csv.gz",
        'modecanada_dataset': base_url + "modecanada/modecanada.csv.gz",
        'chicago_dataset': base_url + "chicago_mode_choice/chicago_mode.csv.gz"
    }
    
    if dataset_name not in dataset_map:
        raise ValueError(f"Dataset {dataset_name} not found. Available datasets: {list(dataset_map.keys())}")
    
    url = dataset_map[dataset_name]
    
    # Set up cache directory
    if local_cache_dir is None:
        home_dir = str(Path.home())
        local_cache_dir = os.path.join(home_dir, '.dcmbench', 'datasets')
    
    # Extract dataset path from URL
    url_parts = url.split('/')
    dataset_parts = url_parts[-2:]  # Get the last two parts (folder/file)
    dataset_path = os.path.join(local_cache_dir, *dataset_parts)
    
    # Check if the dataset exists in cache
    if use_local_cache and os.path.exists(dataset_path):
        logger.info(f"Using cached dataset from {dataset_path}")
    else:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        
        # Download the dataset
        logger.info(f"Local cache disabled. Downloading dataset from remote source.")
        logger.info(f"Downloading dataset from: {url}")
        try:
            urllib.request.urlretrieve(url, dataset_path)
        except Exception as e:
            raise RuntimeError(f"Failed to download dataset: {str(e)}")
    
    # Load the dataset
    try:
        # Assume it's a gzipped CSV file
        if dataset_path.endswith('.gz'):
            df = pd.read_csv(dataset_path, compression='gzip')
        else:
            df = pd.read_csv(dataset_path)
        
        logger.info(f"Successfully loaded dataset '{dataset_name}' with shape {df.shape}")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {str(e)}")