"""Video reader utilities using PyAV"""

from pathlib import Path
from typing import Optional, Generator, Tuple
import numpy as np
import av


class VideoReader:
    """Read video frames from MKV/MP4 files using PyAV"""
    
    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        self.container = None
        self._total_frames = None
        self._fps = None
        
    def __enter__(self):
        self.container = av.open(str(self.video_path))
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.container:
            self.container.close()
    
    @property
    def fps(self) -> float:
        """Get video FPS"""
        if self._fps is None and self.container:
            stream = self.container.streams.video[0]
            self._fps = float(stream.average_rate)
        return self._fps or 30.0
    
    @property
    def total_frames(self) -> int:
        """Get total frame count"""
        if self._total_frames is None and self.container:
            stream = self.container.streams.video[0]
            self._total_frames = stream.frames
        return self._total_frames or 0
    
    @property
    def resolution(self) -> Tuple[int, int]:
        """Get (width, height)"""
        if self.container:
            stream = self.container.streams.video[0]
            return (stream.width, stream.height)
        return (640, 480)
    
    def read_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """Read a single frame by index (RGB uint8)"""
        if not self.container:
            return None
        
        # Seek to frame
        stream = self.container.streams.video[0]
        self.container.seek(0)
        
        for i, frame in enumerate(self.container.decode(video=0)):
            if i == frame_idx:
                return frame.to_ndarray(format='rgb24')
        
        return None
    
    def read_frames(
        self, 
        start_idx: int = 0, 
        end_idx: Optional[int] = None
    ) -> Generator[np.ndarray, None, None]:
        """Read frames from start to end (generator)"""
        if not self.container:
            return
        
        self.container.seek(0)
        
        for i, frame in enumerate(self.container.decode(video=0)):
            if i < start_idx:
                continue
            if end_idx is not None and i >= end_idx:
                break
            yield frame.to_ndarray(format='rgb24')
    
    def sample_frames(self, n_samples: int = 10, seed: int = 42) -> list:
        """Sample N random frames uniformly"""
        if not self.container:
            return []
        
        np.random.seed(seed)
        total = self.total_frames
        
        if total == 0:
            # Fallback: count frames
            self.container.seek(0)
            total = sum(1 for _ in self.container.decode(video=0))
        
        indices = np.linspace(0, max(0, total - 1), n_samples, dtype=int)
        
        frames = []
        for idx in indices:
            frame = self.read_frame(idx)
            if frame is not None:
                frames.append(frame)
        
        return frames

