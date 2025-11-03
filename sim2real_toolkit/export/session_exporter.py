"""Export augmented session data to new folder"""

import json
import shutil
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import av
import pandas as pd
from tqdm import tqdm

from ..io.session_reader import SessionReader
from ..io.video_reader import VideoReader
from ..augmentations.video_ops import VideoAugmentor
from ..augmentations.parquet_ops import ParquetAugmentor


class SessionExporter:
    """Export augmented session to new directory"""
    
    def __init__(
        self,
        session_reader: SessionReader,
        output_path: str,
        video_params: Optional[Dict] = None,
        parquet_params: Optional[Dict] = None,
        video_param_ranges: Optional[Dict] = None,
        parquet_param_ranges: Optional[Dict] = None,
        seed: int = 42
    ):
        self.session_reader = session_reader
        self.output_path = Path(output_path)
        
        # Recommended defaults for parquet augmentation
        default_parquet_params = {
            # Sensor Noise
            "gaussian_noise": 0.01,
            "bias_std": 0.005,
            "drift_std": 0.001,
            "quantization": 0.001,
            "outliers_prob": 0.001,
            "outliers_scale": 5.0,
            "dead_zone": 0.005,
            # Temporal Effects
            "latency_shift": 2,
            "latency_mode": "constant",
            "packet_loss": 0.005,
            "timestamp_jitter": 0.01,
            "duplicate_rows": 0.002,
            # Actuator Dynamics
            "saturate": True,
            "saturate_min": -1.0,
            "saturate_max": 1.0,
            "rate_limit": 0.1,
            "backlash": 0.01,
            "command_delay": 1,
        }

        # Support both fixed params and ranges
        self.video_params = video_params or {}
        # Merge provided parquet params over defaults
        self.parquet_params = {**default_parquet_params, **(parquet_params or {})}
        self.video_param_ranges = video_param_ranges or {}
        self.parquet_param_ranges = parquet_param_ranges or {}
        
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        self.video_augmentor = VideoAugmentor(seed=seed)
        self.parquet_augmentor = ParquetAugmentor(seed=seed)
    
    def export(self, copy_meta: bool = True):
        """Export full augmented session"""
        print(f"Exporting augmented session to: {self.output_path}")
        
        # Create output directory structure
        self.output_path.mkdir(parents=True, exist_ok=True)
        (self.output_path / "videos").mkdir(exist_ok=True)
        (self.output_path / "data").mkdir(exist_ok=True)
        (self.output_path / "meta").mkdir(exist_ok=True)
        
        # Copy metadata (optionally)
        if copy_meta:
            self._copy_metadata()
        
        # Export videos
        self._export_videos()
        
        # Export parquet data
        self._export_parquet()
        
        # Write augmentation manifest
        self._write_manifest()
        
        print("Export complete!")
    
    def _copy_metadata(self):
        """Copy metadata files"""
        meta_src = self.session_reader.meta_path
        meta_dst = self.output_path / "meta"
        
        # Copy info and stats
        for filename in ["info.json", "stats.json", "tasks.parquet"]:
            src = meta_src / filename
            if src.exists():
                shutil.copy2(src, meta_dst / filename)
        
        # Copy episodes
        episodes_src = meta_src / "episodes"
        episodes_dst = meta_dst / "episodes"
        if episodes_src.exists():
            shutil.copytree(episodes_src, episodes_dst, dirs_exist_ok=True)
    
    def _export_videos(self):
        """Export augmented videos"""
        cameras = self.session_reader.get_camera_keys()
        
        for camera in cameras:
            print(f"Processing camera: {camera}")
            
            video_path = self.session_reader.get_video_path(camera)
            if not video_path.exists():
                continue
            
            # Create output directory
            output_dir = self.output_path / "videos" / camera / "chunk-000"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / "file-000.mkv"
            
            # Process video
            self._process_video(video_path, output_file)
    
    def _sample_params_from_ranges(self) -> Dict:
        """Sample random parameters from ranges for this frame"""
        params = {}
        
        # Use ranges if provided, otherwise use fixed params
        if self.video_param_ranges:
            for key, (min_val, max_val) in self.video_param_ranges.items():
                if min_val == max_val:
                    params[key] = min_val
                else:
                    params[key] = self.rng.uniform(min_val, max_val)
        else:
            params = self.video_params.copy()
        
        return params
    
    def _process_video(self, input_path: Path, output_path: Path):
        """Process single video with augmentations"""
        # Open input
        input_container = av.open(str(input_path))
        # Guard: skip files with no video streams
        if not input_container.streams.video:
            print(f"Warning: no video streams found in {input_path}, skipping.")
            input_container.close()
            return
        input_stream = input_container.streams.video[0]
        
        # Open output
        output_container = av.open(str(output_path), 'w')
        output_stream = output_container.add_stream('libx264', rate=input_stream.average_rate)
        output_stream.width = input_stream.width
        output_stream.height = input_stream.height
        output_stream.pix_fmt = 'yuv420p'
        
        # Process frames
        frame_idx = 0
        for frame in tqdm(input_container.decode(video=0), desc="Frames"):
            # Convert to numpy
            img = frame.to_ndarray(format='rgb24')
            
            # Sample random parameters from ranges
            params = self._sample_params_from_ranges()
            params["frame_idx"] = frame_idx
            
            # Apply augmentations
            augmented = self.video_augmentor.apply_all(img, params)
            
            # Convert back to video frame
            new_frame = av.VideoFrame.from_ndarray(augmented, format='rgb24')
            
            # Encode
            for packet in output_stream.encode(new_frame):
                output_container.mux(packet)
            
            frame_idx += 1
        
        # Flush
        for packet in output_stream.encode():
            output_container.mux(packet)
        
        input_container.close()
        output_container.close()
    
    def _export_parquet(self):
        """Export augmented parquet data"""
        print("Processing parquet data")
        
        # Load original
        df = self.session_reader.load_parquet_data()
        
        # Get column names (support both array columns and expanded columns)
        action_cols = []
        state_cols = []
        
        if not df.empty and "action" in df.columns and hasattr(df["action"].iloc[0], "__len__"):
            action_cols = ["action"]
        else:
            action_cols = [f"action.{name}" for name in self.session_reader.get_action_columns()]
        
        if not df.empty and "observation.state" in df.columns and hasattr(df["observation.state"].iloc[0], "__len__"):
            state_cols = ["observation.state"]
        else:
            state_cols = [f"observation.state.{name}" for name in self.session_reader.get_state_columns()]
        
        # Apply augmentations
        augmented_df = self.parquet_augmentor.apply_all(
            df,
            action_cols,
            state_cols,
            self.parquet_params
        )
        
        # Write to output
        output_dir = self.output_path / "data" / "chunk-000"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "file-000.parquet"
        
        augmented_df.to_parquet(output_file, index=False)
    
    def _write_manifest(self):
        """Write augmentation manifest"""
        manifest = {
            "version": "0.1.0",
            "seed": self.seed,
            "source_session": str(self.session_reader.session_path),
            "video_augmentations": self.video_params,
            "parquet_augmentations": self.parquet_params,
        }
        
        manifest_path = self.output_path / "augmentation_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

