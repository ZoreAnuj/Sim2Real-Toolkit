"""Video augmentation tab with real-time preview"""

import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, 
    QScrollArea, QGroupBox, QGridLayout, QPushButton,
    QSpinBox, QComboBox, QSizePolicy
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap

from ..io.session_reader import SessionReader
from ..io.video_reader import VideoReader
from ..augmentations.video_ops import VideoAugmentor


class VideoAugmentationTab(QWidget):
    """Tab for video augmentation with real-time preview"""
    
    def __init__(self):
        super().__init__()
        
        self.session_reader = None
        self.video_reader = None
        self.augmentor = VideoAugmentor(seed=42)
        
        self.original_frame = None
        self.current_frame_idx = 0
        self.camera_key = None
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI components"""
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)
        
        # Left: Controls
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setMaximumWidth(400)
        
        controls_widget = QWidget()
        controls_layout = QVBoxLayout()
        controls_widget.setLayout(controls_layout)
        controls_scroll.setWidget(controls_widget)
        
        main_layout.addWidget(controls_scroll)
        
        # Camera selection
        cam_group = QGroupBox("Video Source")
        cam_layout = QVBoxLayout()
        cam_group.setLayout(cam_layout)
        
        self.camera_combo = QComboBox()
        self.camera_combo.currentTextChanged.connect(self.on_camera_changed)
        cam_layout.addWidget(QLabel("Camera:"))
        cam_layout.addWidget(self.camera_combo)
        
        self.frame_spin = QSpinBox()
        self.frame_spin.setMinimum(0)
        self.frame_spin.setMaximum(1000)
        self.frame_spin.valueChanged.connect(self.on_frame_changed)
        cam_layout.addWidget(QLabel("Frame Index:"))
        cam_layout.addWidget(self.frame_spin)
        
        controls_layout.addWidget(cam_group)
        
        # Add parameter groups
        self.sliders = {}
        
        # Photometric
        photo_group = self._create_param_group("Photometric", [
            ("gaussian_noise", "Gaussian Noise", 0, 100, 0, 0.001),
            ("shot_noise", "Shot Noise", 0, 100, 0, 0.001),
            ("brightness", "Brightness Δ", -50, 50, 0, 0.01),
            ("contrast", "Contrast Δ", -50, 50, 0, 0.01),
            ("saturation", "Saturation Δ", -50, 50, 0, 0.01),
            ("hue", "Hue Δ (deg)", -180, 180, 0, 1.0),
            ("gamma", "Gamma", 10, 300, 100, 0.01),
        ])
        controls_layout.addWidget(photo_group)
        
        # White balance
        wb_group = self._create_param_group("White Balance", [
            ("wb_r", "Red Gain", 50, 200, 100, 0.01),
            ("wb_g", "Green Gain", 50, 200, 100, 0.01),
            ("wb_b", "Blue Gain", 50, 200, 100, 0.01),
        ])
        controls_layout.addWidget(wb_group)
        
        # Blur
        blur_group = self._create_param_group("Blur & Optics", [
            ("motion_blur", "Motion Blur Size", 0, 21, 0, 1.0),
            ("defocus_blur", "Defocus Radius", 0, 15, 0, 1.0),
            ("gaussian_blur", "Gaussian σ", 0, 50, 0, 0.1),
            ("lens_k1", "Lens Distortion k1", -100, 100, 0, 0.001),
            ("lens_k2", "Lens Distortion k2", -100, 100, 0, 0.0001),
            ("chromatic_aberration", "Chromatic Shift", 0, 10, 0, 1.0),
            ("vignetting", "Vignetting", 0, 100, 0, 0.01),
        ])
        controls_layout.addWidget(blur_group)
        
        # Compression & Artifacts
        artifact_group = self._create_param_group("Compression & Artifacts", [
            ("jpeg_quality", "JPEG Quality", 10, 100, 100, 1.0),
            ("fixed_pattern_noise", "Fixed Pattern", 0, 100, 0, 0.001),
            ("flicker", "Temporal Flicker", 0, 100, 0, 0.01),
        ])
        controls_layout.addWidget(artifact_group)
        
        # Reset button
        reset_btn = QPushButton("Reset All")
        reset_btn.clicked.connect(self.reset_params)
        controls_layout.addWidget(reset_btn)
        
        controls_layout.addStretch()
        
        # Right: Preview
        preview_layout = QVBoxLayout()
        main_layout.addLayout(preview_layout, stretch=1)
        
        preview_label = QLabel("Preview")
        preview_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        preview_layout.addWidget(preview_label)
        
        # Image displays
        images_layout = QHBoxLayout()
        preview_layout.addLayout(images_layout)
        
        # Original
        orig_container = QVBoxLayout()
        orig_container.addWidget(QLabel("Original", alignment=Qt.AlignCenter))
        self.original_label = QLabel()
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setStyleSheet("border: 2px solid #ccc;")
        self.original_label.setMinimumSize(640, 480)
        self.original_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        orig_container.addWidget(self.original_label)
        images_layout.addLayout(orig_container)
        
        # Augmented
        aug_container = QVBoxLayout()
        aug_container.addWidget(QLabel("Augmented", alignment=Qt.AlignCenter))
        self.augmented_label = QLabel()
        self.augmented_label.setAlignment(Qt.AlignCenter)
        self.augmented_label.setStyleSheet("border: 2px solid #4CAF50;")
        self.augmented_label.setMinimumSize(640, 480)
        self.augmented_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        aug_container.addWidget(self.augmented_label)
        images_layout.addLayout(aug_container)
    
    def _create_param_group(self, title: str, params: list) -> QGroupBox:
        """Create a parameter group with 3 classic sliders (Min | Preview | Max)"""
        group = QGroupBox(title)
        layout = QGridLayout()
        group.setLayout(layout)
        
        layout.addWidget(QLabel("<b>Param</b>"), 0, 0)
        layout.addWidget(QLabel("<b>Min</b>"), 0, 1)
        layout.addWidget(QLabel("<b>Preview</b>"), 0, 2)
        layout.addWidget(QLabel("<b>Max</b>"), 0, 3)
        layout.addWidget(QLabel("<b>Values</b>"), 0, 4)
        
        for row, (key, label, min_val, max_val, default, scale) in enumerate(params, start=1):
            # Label
            layout.addWidget(QLabel(label), row, 0)
            
            # Min slider (export lower bound)
            min_slider = QSlider(Qt.Horizontal)
            min_slider.setMinimum(min_val)
            min_slider.setMaximum(max_val)
            min_slider.setValue(default)
            min_slider.setMaximumWidth(110)
            min_slider.valueChanged.connect(self.on_param_changed)
            layout.addWidget(min_slider, row, 1)
            
            # Preview slider (real-time)
            preview_slider = QSlider(Qt.Horizontal)
            preview_slider.setMinimum(min_val)
            preview_slider.setMaximum(max_val)
            preview_slider.setValue(default)
            preview_slider.valueChanged.connect(self.on_param_changed)
            layout.addWidget(preview_slider, row, 2)
            
            # Max slider (export upper bound)
            max_slider = QSlider(Qt.Horizontal)
            max_slider.setMinimum(min_val)
            max_slider.setMaximum(max_val)
            max_slider.setValue(default)
            max_slider.setMaximumWidth(110)
            max_slider.valueChanged.connect(self.on_param_changed)
            layout.addWidget(max_slider, row, 3)
            
            # Value label (fixed width to avoid layout jumps)
            value_label = QLabel(f"[{default * scale:.3f}, {default * scale:.3f}, {default * scale:.3f}]")
            value_label.setFixedWidth(170)
            value_label.setStyleSheet("font-family: Consolas, monospace;")
            layout.addWidget(value_label, row, 4)
            
            self.sliders[key] = {
                "min_slider": min_slider,
                "preview_slider": preview_slider,
                "max_slider": max_slider,
                "label": value_label,
                "scale": scale,
                "default": default
            }
        
        return group
    
    def load_session(self, session_reader: SessionReader):
        """Load session data"""
        self.session_reader = session_reader
        
        # Populate camera combo
        cameras = session_reader.get_camera_keys()
        self.camera_combo.clear()
        self.camera_combo.addItems(cameras)
        
        if cameras:
            self.camera_key = cameras[0]
            self.load_video()
    
    def load_video(self):
        """Load current video"""
        if not self.session_reader or not self.camera_key:
            return
        
        video_path = self.session_reader.get_video_path(self.camera_key)
        
        if video_path.exists():
            self.video_reader = VideoReader(str(video_path))
            with self.video_reader:
                self.frame_spin.setMaximum(max(0, self.video_reader.total_frames - 1))
            
            self.load_frame()
    
    def load_frame(self):
        """Load and display current frame"""
        if not self.video_reader:
            return
        
        with self.video_reader:
            frame = self.video_reader.read_frame(self.current_frame_idx)
            
            if frame is not None:
                self.original_frame = frame
                self.display_original(frame)
                self.update_augmented()
    
    def display_original(self, frame: np.ndarray):
        """Display original frame"""
        self._display_frame(frame, self.original_label)
    
    def display_augmented(self, frame: np.ndarray):
        """Display augmented frame"""
        self._display_frame(frame, self.augmented_label)
    
    def _display_frame(self, frame: np.ndarray, label: QLabel):
        """Convert numpy array to QPixmap and display"""
        h, w, c = frame.shape
        bytes_per_line = 3 * w
        
        q_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        # Scale to fit label while preserving a generous preview size
        target_w = max(label.width(), 640)
        target_h = max(label.height(), 480)
        scaled = pixmap.scaled(
            target_w,
            target_h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        label.setPixmap(scaled)
    
    def update_augmented(self):
        """Apply augmentations and update display"""
        if self.original_frame is None:
            return
        
        params = self.get_params()
        params["frame_idx"] = self.current_frame_idx
        
        augmented = self.augmentor.apply_all(self.original_frame, params)
        self.display_augmented(augmented)
    
    def get_params(self) -> dict:
        """Get current parameter values (for preview)"""
        params = {}
        for key, slider_info in self.sliders.items():
            value = slider_info["preview_slider"].value() * slider_info["scale"]
            params[key] = value
        return params
    
    def get_export_ranges(self) -> dict:
        """Get min/max ranges for export (randomized augmentation)"""
        ranges = {}
        for key, slider_info in self.sliders.items():
            min_val = slider_info["min_slider"].value() * slider_info["scale"]
            max_val = slider_info["max_slider"].value() * slider_info["scale"]
            ranges[key] = (min_val, max_val)
        return ranges
    
    def reset_params(self):
        """Reset all parameters to defaults"""
        for key, slider_info in self.sliders.items():
            default = slider_info["default"]
            slider_info["min_slider"].setValue(default)
            slider_info["preview_slider"].setValue(default)
            slider_info["max_slider"].setValue(default)
    
    def on_param_changed(self):
        """Handle parameter slider change"""
        # Update value labels
        for key, slider_info in self.sliders.items():
            scale = slider_info["scale"]
            min_val = slider_info["min_slider"].value() * scale
            preview_val = slider_info["preview_slider"].value() * scale
            max_val = slider_info["max_slider"].value() * scale
            slider_info["label"].setText(f"[{min_val:.3f}, {preview_val:.3f}, {max_val:.3f}]")
        
        # Update preview
        self.update_augmented()
    
    def on_camera_changed(self, camera: str):
        """Handle camera selection change"""
        if camera:
            self.camera_key = camera
            self.load_video()
    
    def on_frame_changed(self, idx: int):
        """Handle frame index change"""
        self.current_frame_idx = idx
        self.load_frame()

