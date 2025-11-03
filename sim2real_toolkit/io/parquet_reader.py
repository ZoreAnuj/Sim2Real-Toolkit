"""Parquet reader utilities"""

from pathlib import Path
from typing import Optional, List
import pandas as pd
import numpy as np


class ParquetReader:
    """Read and manipulate parquet data files"""
    
    def __init__(self, parquet_path: str):
        self.parquet_path = Path(parquet_path)
        self.df = None
    
    def load(self) -> pd.DataFrame:
        """Load parquet file into memory"""
        self.df = pd.read_parquet(self.parquet_path)
        return self.df
    
    def get_columns(self, prefix: str = "") -> List[str]:
        """Get columns matching prefix"""
        if self.df is None:
            self.load()
        
        if not prefix:
            return list(self.df.columns)
        
        return [col for col in self.df.columns if col.startswith(prefix)]
    
    def get_action_data(self) -> Optional[np.ndarray]:
        """Extract action columns as numpy array"""
        if self.df is None:
            self.load()
        
        # Check if action is a single column with array values
        if "action" in self.df.columns:
            return np.stack(self.df["action"].values)
        
        # Fallback: look for action.* columns
        action_cols = self.get_columns("action")
        if not action_cols:
            return None
        
        return self.df[action_cols].values
    
    def get_state_data(self) -> Optional[np.ndarray]:
        """Extract observation.state columns as numpy array"""
        if self.df is None:
            self.load()
        
        # Check if observation.state is a single column with array values
        if "observation.state" in self.df.columns:
            return np.stack(self.df["observation.state"].values)
        
        # Fallback: look for observation.state.* columns
        state_cols = self.get_columns("observation.state")
        if not state_cols:
            return None
        
        return self.df[state_cols].values
    
    def get_timestamps(self) -> Optional[np.ndarray]:
        """Extract timestamp column"""
        if self.df is None:
            self.load()
        
        if "timestamp" in self.df.columns:
            return self.df["timestamp"].values
        return None
    
    def get_episode_indices(self) -> Optional[np.ndarray]:
        """Extract episode_index column"""
        if self.df is None:
            self.load()
        
        if "episode_index" in self.df.columns:
            return self.df["episode_index"].values
        return None
    
    def sample_rows(self, n_samples: int = 100, seed: int = 42) -> pd.DataFrame:
        """Sample N random rows"""
        if self.df is None:
            self.load()
        
        n = min(n_samples, len(self.df))
        return self.df.sample(n=n, random_state=seed)

