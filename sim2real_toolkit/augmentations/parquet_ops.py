"""Parquet/tabular data augmentation operations for sim2real"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List


class ParquetAugmentor:
    """Apply sensor noise, latency, and actuator realism to tabular data"""
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
    
    # ==================== SENSOR NOISE ====================
    
    def add_gaussian_noise(
        self, 
        data: np.ndarray, 
        sigma: float = 0.01
    ) -> np.ndarray:
        """Add Gaussian noise to sensor readings"""
        noise = self.rng.normal(0, sigma, data.shape)
        return data + noise
    
    def add_bias_drift(
        self, 
        data: np.ndarray, 
        bias_std: float = 0.01,
        drift_std: float = 0.001
    ) -> np.ndarray:
        """Add bias and drift to sensor readings"""
        n_samples, n_dims = data.shape
        
        # Per-dimension bias (constant offset)
        bias = self.rng.normal(0, bias_std, n_dims)
        
        # Per-dimension drift (slow random walk)
        drift = np.cumsum(self.rng.normal(0, drift_std, (n_samples, n_dims)), axis=0)
        
        return data + bias + drift
    
    def add_quantization(
        self, 
        data: np.ndarray, 
        step: float = 0.001
    ) -> np.ndarray:
        """Quantize sensor readings"""
        return np.round(data / step) * step
    
    def add_outliers(
        self, 
        data: np.ndarray, 
        prob: float = 0.001,
        scale: float = 5.0
    ) -> np.ndarray:
        """Add random outliers (sensor glitches)"""
        mask = self.rng.random(data.shape) < prob
        outliers = self.rng.normal(0, scale * data.std(axis=0), data.shape)
        result = data.copy()
        result[mask] = outliers[mask]
        return result
    
    def add_dead_zone(
        self, 
        data: np.ndarray, 
        threshold: float = 0.01
    ) -> np.ndarray:
        """Apply dead zone (values below threshold → 0)"""
        result = data.copy()
        result[np.abs(result) < threshold] = 0
        return result
    
    # ==================== TEMPORAL EFFECTS ====================
    
    def add_latency_shift(
        self, 
        df: pd.DataFrame, 
        columns: List[str],
        shift_frames: int = 2,
        mode: str = "constant"  # "constant", "random", "jitter"
    ) -> pd.DataFrame:
        """Shift columns by N frames to simulate latency"""
        result = df.copy()
        
        if mode == "constant":
            for col in columns:
                if col in result.columns:
                    result[col] = result[col].shift(shift_frames, fill_value=0)
        
        elif mode == "random":
            # Random shift per episode
            if "episode_index" in result.columns:
                for ep in result["episode_index"].unique():
                    mask = result["episode_index"] == ep
                    ep_shift = self.rng.randint(-shift_frames, shift_frames + 1)
                    for col in columns:
                        if col in result.columns:
                            result.loc[mask, col] = result.loc[mask, col].shift(ep_shift, fill_value=0)
        
        elif mode == "jitter":
            # Per-row jitter (not practical, but for completeness)
            for col in columns:
                if col in result.columns:
                    shifts = self.rng.randint(-shift_frames, shift_frames + 1, size=len(result))
                    new_col = result[col].copy()
                    for i, s in enumerate(shifts):
                        if 0 <= i + s < len(result):
                            new_col.iloc[i] = result[col].iloc[i + s]
                    result[col] = new_col
        
        return result
    
    def add_packet_loss(
        self, 
        df: pd.DataFrame, 
        columns: List[str],
        loss_prob: float = 0.01
    ) -> pd.DataFrame:
        """Simulate packet loss (set random rows to NaN, forward-fill)"""
        result = df.copy()
        
        loss_mask = self.rng.random(len(result)) < loss_prob
        
        for col in columns:
            if col in result.columns:
                result.loc[loss_mask, col] = np.nan
                result[col] = result[col].fillna(method="ffill").fillna(0)
        
        return result
    
    def duplicate_rows(
        self, 
        df: pd.DataFrame, 
        dup_prob: float = 0.005
    ) -> pd.DataFrame:
        """Duplicate random rows (sensor repeated measurements)"""
        result = df.copy()
        
        dup_mask = self.rng.random(len(result)) < dup_prob
        dup_indices = np.where(dup_mask)[0]
        
        if len(dup_indices) > 0:
            dup_rows = result.iloc[dup_indices]
            result = pd.concat([result, dup_rows], ignore_index=True)
            result = result.sort_index().reset_index(drop=True)
        
        return result
    
    def add_timestamp_jitter(
        self, 
        df: pd.DataFrame, 
        jitter_std: float = 0.01
    ) -> pd.DataFrame:
        """Add jitter to timestamp column"""
        result = df.copy()
        
        if "timestamp" in result.columns:
            jitter = self.rng.normal(0, jitter_std, len(result))
            result["timestamp"] = result["timestamp"] + jitter
            # Keep monotonic
            result["timestamp"] = result["timestamp"].cummax()
        
        return result
    
    # ==================== ACTUATOR DYNAMICS ====================
    
    def apply_saturation(
        self, 
        data: np.ndarray, 
        min_val: float = -1.0,
        max_val: float = 1.0
    ) -> np.ndarray:
        """Saturate actuator commands"""
        return np.clip(data, min_val, max_val)
    
    def apply_rate_limit(
        self, 
        data: np.ndarray, 
        max_delta: float = 0.1
    ) -> np.ndarray:
        """Limit rate of change between consecutive samples"""
        result = data.copy()
        
        for i in range(1, len(result)):
            delta = result[i] - result[i - 1]
            result[i] = result[i - 1] + np.clip(delta, -max_delta, max_delta)
        
        return result
    
    def add_backlash(
        self, 
        data: np.ndarray, 
        backlash: float = 0.01
    ) -> np.ndarray:
        """Add backlash (hysteresis in direction change)"""
        if data.size == 0 or data.shape[0] == 0 or data.shape[1] == 0:
            return data
        result = data.copy()
        for dim in range(data.shape[1]):
            # Safe: we already checked data has at least one row
            direction = np.sign(np.diff(data[:, dim], prepend=data[0, dim]))
            direction_change = np.diff(direction, prepend=direction[0]) != 0
            offset = np.cumsum(direction_change) * backlash * direction
            result[:, dim] += offset
        return result
    
    def add_command_delay(
        self, 
        data: np.ndarray, 
        delay_frames: int = 1
    ) -> np.ndarray:
        """Delay commands by N frames (shift forward, repeat first)"""
        if delay_frames <= 0 or data.size == 0 or data.shape[0] == 0:
            return data
        result = np.zeros_like(data)
        result[:delay_frames] = data[0]
        result[delay_frames:] = data[:-delay_frames]
        return result
    
    # ==================== COMBINED PIPELINE ====================
    
    def apply_to_columns(
        self,
        df: pd.DataFrame,
        columns: List[str],
        params: Dict
    ) -> pd.DataFrame:
        """Apply augmentations to specific columns"""
        result = df.copy()
        
        # Skip if dataframe is empty
        if result.empty:
            return result
        
        # Skip if no columns to augment
        if not columns:
            return result
        
        # Handle array columns (single column containing arrays)
        array_column_mode = False
        if len(columns) == 1 and columns[0] in result.columns:
            col = columns[0]
            # Ensure we have at least one row before accessing iloc[0]
            if len(result) > 0 and hasattr(result[col].iloc[0], '__len__'):
                # Extract array data
                data = np.stack(result[col].values)
                array_column_mode = True
            else:
                return result
        else:
            # Extract column data
            if not all(col in result.columns for col in columns):
                return result
            data = result[columns].values
        
        # Skip if data is empty
        if data.size == 0:
            return result
        
        # Sensor noise
        if params.get("gaussian_noise", 0) > 0:
            data = self.add_gaussian_noise(data, params["gaussian_noise"])
        
        if params.get("bias_std", 0) > 0 or params.get("drift_std", 0) > 0:
            data = self.add_bias_drift(
                data,
                params.get("bias_std", 0),
                params.get("drift_std", 0)
            )
        
        if params.get("quantization", 0) > 0:
            data = self.add_quantization(data, params["quantization"])
        
        if params.get("outliers_prob", 0) > 0:
            data = self.add_outliers(
                data,
                params["outliers_prob"],
                params.get("outliers_scale", 5.0)
            )
        
        if params.get("dead_zone", 0) > 0:
            data = self.add_dead_zone(data, params["dead_zone"])
        
        # Actuator dynamics (for action columns)
        if params.get("saturate", False):
            # Accept both saturate_min/max and min_val/max for compatibility
            min_val = params.get("saturate_min", params.get("min_val", -1.0))
            max_val = params.get("saturate_max", params.get("max_val", 1.0))
            data = self.apply_saturation(
                data,
                min_val,
                max_val
            )
        
        if params.get("rate_limit", 0) > 0:
            data = self.apply_rate_limit(data, params["rate_limit"])
        
        if params.get("backlash", 0) > 0:
            data = self.add_backlash(data, params["backlash"])
        
        if params.get("command_delay", 0) > 0:
            data = self.add_command_delay(data, int(params["command_delay"]))
        
        # Write back
        if array_column_mode and len(columns) == 1 and columns[0] in result.columns:
            # Convert back to list of arrays without relying on iloc
            result[columns[0]] = list(data)
        else:
            result[columns] = data
        
        # Temporal effects (dataframe-level)
        if params.get("latency_shift", 0) != 0:
            result = self.add_latency_shift(
                result,
                columns,
                int(params["latency_shift"]),
                params.get("latency_mode", "constant")
            )
        
        if params.get("packet_loss", 0) > 0:
            result = self.add_packet_loss(result, columns, params["packet_loss"])
        
        return result
    
    def apply_all(
        self,
        df: pd.DataFrame,
        action_cols: List[str],
        state_cols: List[str],
        params: Dict
    ) -> pd.DataFrame:
        """Apply all augmentations to actions and states"""
        result = df.copy()
        
        # Apply to actions (use full params unless action-specific ones are provided)
        if action_cols:
            action_params = {k: v for k, v in params.items() if k.startswith("action_")}
            if action_params:
                # If action-specific params exist, use only those
                action_params = {k.replace("action_", ""): v for k, v in action_params.items()}
            else:
                # Otherwise use all params
                action_params = params.copy()
            result = self.apply_to_columns(result, action_cols, action_params)
        
        # Apply to states (use full params unless state-specific ones are provided)
        if state_cols:
            state_params = {k: v for k, v in params.items() if k.startswith("state_")}
            if state_params:
                # If state-specific params exist, use only those
                state_params = {k.replace("state_", ""): v for k, v in state_params.items()}
            else:
                # Otherwise use all params
                state_params = params.copy()
            result = self.apply_to_columns(result, state_cols, state_params)
        
        # Timestamp jitter
        if params.get("timestamp_jitter", 0) > 0:
            result = self.add_timestamp_jitter(result, params["timestamp_jitter"])
        
        # Row duplication
        if params.get("duplicate_rows", 0) > 0:
            result = self.duplicate_rows(result, params["duplicate_rows"])
        
        return result

