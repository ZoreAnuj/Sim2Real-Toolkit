"""Session data reader for robot learning datasets"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import pyarrow.parquet as pq


class SessionReader:
    """Read session data including metadata, videos, and parquet files"""
    
    def __init__(self, session_path: str):
        self.session_path = Path(session_path)
        self.meta_path = self.session_path / "meta"
        self.data_path = self.session_path / "data"
        self.videos_path = self.session_path / "videos"
        
        # Load metadata
        self.info = self._load_json(self.meta_path / "info.json")
        self.stats = self._load_json(self.meta_path / "stats.json")
        
        # Parse episode metadata
        self.episodes_df = None
        episodes_parquet = self.meta_path / "episodes" / "chunk-000" / "file-000.parquet"
        if episodes_parquet.exists():
            self.episodes_df = pd.read_parquet(episodes_parquet)
    
    def _load_json(self, path: Path) -> Dict:
        """Load JSON file"""
        with open(path, 'r') as f:
            return json.load(f)
    
    def get_camera_keys(self) -> List[str]:
        """Get list of camera keys from video paths"""
        cameras = []
        if self.videos_path.exists():
            for cam_dir in self.videos_path.iterdir():
                if cam_dir.is_dir():
                    cameras.append(cam_dir.name)
        return sorted(cameras)
    
    def get_video_path(self, camera_key: str, chunk_idx: int = 0, file_idx: int = 0) -> Path:
        """Get path to video file for a specific camera"""
        video_path = (
            self.videos_path / 
            camera_key / 
            f"chunk-{chunk_idx:03d}" / 
            f"file-{file_idx:03d}.mkv"
        )
        return video_path
    
    def get_parquet_path(self, chunk_idx: int = 0, file_idx: int = 0) -> Path:
        """Get path to data parquet file"""
        return self.data_path / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.parquet"
    
    def load_parquet_data(self, chunk_idx: int = 0, file_idx: int = 0) -> pd.DataFrame:
        """Load parquet data as pandas DataFrame"""
        parquet_path = self.get_parquet_path(chunk_idx, file_idx)
        return pd.read_parquet(parquet_path)
    
    def get_action_columns(self) -> List[str]:
        """Get action column names"""
        if "action" in self.info.get("features", {}):
            return self.info["features"]["action"].get("names", [])
        return []
    
    def get_state_columns(self) -> List[str]:
        """Get state column names"""
        if "observation.state" in self.info.get("features", {}):
            return self.info["features"]["observation.state"].get("names", [])
        return []
    
    def get_video_info(self) -> Dict:
        """Get video metadata (fps, resolution, codec)"""
        return {
            "fps": self.info.get("fps", 30),
            "shape": self.info.get("shape", [480, 640, 3]),
            "total_episodes": self.info.get("total_episodes", 0),
            "total_frames": self.info.get("total_frames", 0),
        }
    
    def get_episode_ranges(self) -> List[Tuple[int, int]]:
        """Get (start_frame, end_frame) for each episode"""
        if self.episodes_df is None:
            return []
        
        ranges = []
        for _, row in self.episodes_df.iterrows():
            start = int(row.get("from", 0))
            end = int(row.get("to", 0))
            ranges.append((start, end))
        return ranges

