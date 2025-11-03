"""Parquet augmentation tab with real-time preview and plotting"""

import numpy as np
import pandas as pd
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QScrollArea, QGroupBox, QGridLayout, QPushButton,
    QTableWidget, QTableWidgetItem, QComboBox, QSpinBox, QCheckBox
)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ..io.session_reader import SessionReader
from ..io.parquet_reader import ParquetReader
from ..augmentations.parquet_ops import ParquetAugmentor


class ParquetAugmentationTab(QWidget):
    """Tab for parquet data augmentation with table preview"""
    
    def __init__(self):
        super().__init__()
        
        self.session_reader = None
        self.parquet_reader = None
        self.augmentor = ParquetAugmentor(seed=42)
        
        self.original_df = None
        self.last_augmented_df = None
        self.action_cols = []
        self.state_cols = []
        
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
        
        # Data selection
        data_group = QGroupBox("Data Selection")
        data_layout = QVBoxLayout()
        data_group.setLayout(data_layout)
        
        self.sample_size_spin = QSpinBox()
        self.sample_size_spin.setMinimum(10)
        self.sample_size_spin.setMaximum(1000)
        self.sample_size_spin.setValue(50)
        self.sample_size_spin.valueChanged.connect(self.on_sample_size_changed)
        data_layout.addWidget(QLabel("Sample Size:"))
        data_layout.addWidget(self.sample_size_spin)
        
        refresh_btn = QPushButton("Refresh Sample")
        refresh_btn.clicked.connect(self.load_sample)
        data_layout.addWidget(refresh_btn)
        
        controls_layout.addWidget(data_group)
        
        # Add parameter groups
        self.sliders = {}
        
        # Sensor Noise
        sensor_group = self._create_param_group("Sensor Noise", [
            ("gaussian_noise", "Gaussian Noise σ", 0, 100, 0, 0.001),
            ("bias_std", "Bias σ", 0, 100, 0, 0.001),
            ("drift_std", "Drift σ", 0, 100, 0, 0.0001),
            ("quantization", "Quantization Step", 0, 100, 0, 0.0001),
            ("outliers_prob", "Outlier Probability", 0, 100, 0, 0.0001),
            ("outliers_scale", "Outlier Scale", 10, 100, 50, 0.1),
            ("dead_zone", "Dead Zone", 0, 100, 0, 0.001),
        ])
        controls_layout.addWidget(sensor_group)
        
        # Temporal Effects
        temporal_group = self._create_param_group("Temporal Effects", [
            ("latency_shift", "Latency Shift (frames)", -10, 10, 0, 1.0),
            ("packet_loss", "Packet Loss Prob", 0, 100, 0, 0.001),
            ("timestamp_jitter", "Timestamp Jitter σ", 0, 100, 0, 0.001),
            ("duplicate_rows", "Row Duplication Prob", 0, 100, 0, 0.001),
        ])
        controls_layout.addWidget(temporal_group)
        
        # Actuator Dynamics
        actuator_group = self._create_param_group("Actuator Dynamics", [
            ("saturate_min", "Saturation Min", -200, 0, -100, 0.01),
            ("saturate_max", "Saturation Max", 0, 200, 100, 0.01),
            ("rate_limit", "Rate Limit", 0, 100, 0, 0.01),
            ("backlash", "Backlash", 0, 100, 0, 0.001),
            ("command_delay", "Command Delay (frames)", 0, 10, 0, 1.0),
        ])
        controls_layout.addWidget(actuator_group)
        
        # Reset button
        reset_btn = QPushButton("Reset All")
        reset_btn.clicked.connect(self.reset_params)
        controls_layout.addWidget(reset_btn)
        
        controls_layout.addStretch()
        
        # Right: Preview
        preview_layout = QVBoxLayout()
        main_layout.addLayout(preview_layout, stretch=1)
        
        preview_label = QLabel("Data Preview")
        preview_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        preview_layout.addWidget(preview_label)
        
        # Info label
        self.info_label = QLabel("Load a session to preview data")
        preview_layout.addWidget(self.info_label)
        
        # Tables
        tables_layout = QVBoxLayout()
        preview_layout.addLayout(tables_layout)
        
        # Original table
        tables_layout.addWidget(QLabel("Original Data", alignment=Qt.AlignCenter))
        self.original_table = QTableWidget()
        self.original_table.setMaximumHeight(350)
        tables_layout.addWidget(self.original_table)
        
        # Augmented table
        tables_layout.addWidget(QLabel("Augmented Data", alignment=Qt.AlignCenter))
        self.augmented_table = QTableWidget()
        self.augmented_table.setMaximumHeight(350)
        tables_layout.addWidget(self.augmented_table)
        
        # Statistics comparison
        self.stats_label = QLabel("")
        self.stats_label.setStyleSheet("padding: 10px; background-color: #f0f0f0;")
        preview_layout.addWidget(self.stats_label)

        # Plotting controls and canvas
        plot_group = QGroupBox("Plots")
        plot_layout = QVBoxLayout()
        plot_group.setLayout(plot_layout)

        controls_row = QHBoxLayout()
        plot_layout.addLayout(controls_row)

        controls_row.addWidget(QLabel("Type:"))
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(["Time Series", "Histogram"])
        self.plot_type_combo.currentIndexChanged.connect(self.update_plot)
        controls_row.addWidget(self.plot_type_combo)

        controls_row.addWidget(QLabel("Action:"))
        self.action_signal_combo = QComboBox()
        self.action_signal_combo.currentIndexChanged.connect(self.update_plot)
        controls_row.addWidget(self.action_signal_combo)

        controls_row.addWidget(QLabel("State:"))
        self.state_signal_combo = QComboBox()
        self.state_signal_combo.currentIndexChanged.connect(self.update_plot)
        controls_row.addWidget(self.state_signal_combo)

        self.overlay_checkbox = QCheckBox("Overlay Original vs Augmented")
        self.overlay_checkbox.setChecked(True)
        self.overlay_checkbox.stateChanged.connect(self.update_plot)
        controls_row.addWidget(self.overlay_checkbox)

        controls_row.addStretch()

        self.figure = Figure(figsize=(6, 3))
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)

        preview_layout.addWidget(plot_group)
    
    def _create_param_group(self, title: str, params: list) -> QGroupBox:
        """Create a parameter group with sliders"""
        group = QGroupBox(title)
        layout = QGridLayout()
        group.setLayout(layout)
        
        for row, (key, label, min_val, max_val, default, scale) in enumerate(params):
            # Label
            layout.addWidget(QLabel(label), row, 0)
            
            # Slider
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(min_val)
            slider.setMaximum(max_val)
            slider.setValue(default)
            slider.valueChanged.connect(self.on_param_changed)
            layout.addWidget(slider, row, 1)
            
            # Value label
            value_label = QLabel(f"{default * scale:.4f}")
            value_label.setMinimumWidth(70)
            layout.addWidget(value_label, row, 2)
            
            self.sliders[key] = {
                "slider": slider,
                "label": value_label,
                "scale": scale,
                "default": default
            }
        
        return group
    
    def load_session(self, session_reader: SessionReader):
        """Load session data"""
        self.session_reader = session_reader
        
        # Load parquet first to check structure
        parquet_path = session_reader.get_parquet_path()
        if parquet_path.exists():
            self.parquet_reader = ParquetReader(str(parquet_path))
            df = self.parquet_reader.load()
            
            # Check if action/state are array columns
            if "action" in df.columns:
                self.action_cols = ["action"]
            else:
                self.action_cols = [f"action.{name}" for name in session_reader.get_action_columns()]
            
            if "observation.state" in df.columns:
                self.state_cols = ["observation.state"]
            else:
                self.state_cols = [f"observation.state.{name}" for name in session_reader.get_state_columns()]
            
            self.load_sample()
            self._populate_plot_selectors()
    
    def load_sample(self):
        """Load and display sample data"""
        if not self.parquet_reader:
            return
        
        sample_size = self.sample_size_spin.value()
        self.original_df = self.parquet_reader.sample_rows(sample_size)
        
        # Update info
        self.info_label.setText(
            f"Loaded {len(self.original_df)} rows | "
            f"Actions: {len(self.action_cols)} | "
            f"States: {len(self.state_cols)}"
        )
        
        # Display original
        self.display_dataframe(self.original_df, self.original_table)
        
        # Update augmented
        self.update_augmented()
        # Update plot after data load
        self.update_plot()

    def _populate_plot_selectors(self):
        """Populate signal selectors for plots"""
        # Build names
        actions = self.session_reader.get_action_columns() if self.session_reader else []
        states = self.session_reader.get_state_columns() if self.session_reader else []
        
        self.action_signal_combo.clear()
        self.action_signal_combo.addItem("(none)")
        for name in actions:
            self.action_signal_combo.addItem(name)
        
        self.state_signal_combo.clear()
        self.state_signal_combo.addItem("(none)")
        for name in states:
            self.state_signal_combo.addItem(name)
    
    def display_dataframe(self, df: pd.DataFrame, table: QTableWidget):
        """Display dataframe in table widget"""
        # Expand array columns for display
        df_display = df.copy()
        
        # Expand action column if it's an array
        if "action" in df_display.columns and hasattr(df_display["action"].iloc[0], '__len__'):
            action_names = self.session_reader.get_action_columns() if self.session_reader else []
            action_array = np.stack(df_display["action"].values)
            for i, name in enumerate(action_names[:action_array.shape[1]]):
                df_display[f"action.{name}"] = action_array[:, i]
            df_display = df_display.drop(columns=["action"])
        
        # Expand state column if it's an array
        if "observation.state" in df_display.columns and hasattr(df_display["observation.state"].iloc[0], '__len__'):
            state_names = self.session_reader.get_state_columns() if self.session_reader else []
            state_array = np.stack(df_display["observation.state"].values)
            for i, name in enumerate(state_names[:state_array.shape[1]]):
                df_display[f"state.{name}"] = state_array[:, i]
            df_display = df_display.drop(columns=["observation.state"])
        
        # Select display columns
        display_cols = []
        if "timestamp" in df_display.columns:
            display_cols.append("timestamp")
        
        # Add expanded action/state columns
        action_display = [c for c in df_display.columns if c.startswith("action.")]
        state_display = [c for c in df_display.columns if c.startswith("state.")]
        display_cols.extend(action_display[:8])  # Limit to 8 actions
        display_cols.extend(state_display[:8])   # Limit to 8 states
        
        if not display_cols:
            display_cols = list(df_display.columns[:10])
        
        df_show = df_display[display_cols]
        
        # Set table dimensions
        table.setRowCount(min(len(df_show), 100))  # Limit to 100 rows for performance
        table.setColumnCount(len(display_cols))
        table.setHorizontalHeaderLabels(display_cols)
        
        # Populate table
        for i in range(min(len(df_show), 100)):
            row = df_show.iloc[i]
            for j, col in enumerate(display_cols):
                value = row[col]
                if isinstance(value, (int, float, np.integer, np.floating)):
                    text = f"{float(value):.4f}"
                else:
                    text = str(value)[:20]  # Truncate long strings
                table.setItem(i, j, QTableWidgetItem(text))
        
        table.resizeColumnsToContents()
    
    def update_augmented(self):
        """Apply augmentations and update display"""
        if self.original_df is None:
            return
        
        params = self.get_params()
        
        # Convert params
        aug_params = {
            "gaussian_noise": params.get("gaussian_noise", 0),
            "bias_std": params.get("bias_std", 0),
            "drift_std": params.get("drift_std", 0),
            "quantization": params.get("quantization", 0),
            "outliers_prob": params.get("outliers_prob", 0),
            "outliers_scale": params.get("outliers_scale", 5.0),
            "dead_zone": params.get("dead_zone", 0),
            "latency_shift": int(params.get("latency_shift", 0)),
            "latency_mode": "constant",
            "packet_loss": params.get("packet_loss", 0),
            "timestamp_jitter": params.get("timestamp_jitter", 0),
            "duplicate_rows": params.get("duplicate_rows", 0),
            "saturate": True,
            "min_val": params.get("saturate_min", -1.0),
            "max_val": params.get("saturate_max", 1.0),
            "rate_limit": params.get("rate_limit", 0),
            "backlash": params.get("backlash", 0),
            "command_delay": int(params.get("command_delay", 0)),
        }
        
        augmented_df = self.augmentor.apply_all(
            self.original_df,
            self.action_cols,
            self.state_cols,
            aug_params
        )
        
        # Display augmented
        self.display_dataframe(augmented_df, self.augmented_table)
        
        # Update statistics
        self.update_statistics(augmented_df)
        
        # Cache for plotting
        self.last_augmented_df = augmented_df
    
    def update_statistics(self, augmented_df: pd.DataFrame):
        """Update statistics comparison"""
        if self.original_df is None:
            return
        
        # Calculate stats for action columns
        stats_text = "<b>Statistics Comparison (Actions):</b><br>"
        
        # Handle array columns
        if "action" in self.original_df.columns:
            orig_actions = np.stack(self.original_df["action"].values)
            aug_actions = np.stack(augmented_df["action"].values)
            
            orig_mean = orig_actions.mean()
            orig_std = orig_actions.std()
            aug_mean = aug_actions.mean()
            aug_std = aug_actions.std()
            
            stats_text += f"Original: μ={orig_mean:.4f}, σ={orig_std:.4f}<br>"
            stats_text += f"Augmented: μ={aug_mean:.4f}, σ={aug_std:.4f}<br>"
            stats_text += f"Δμ={aug_mean-orig_mean:.4f}, Δσ={aug_std-orig_std:.4f}"
        else:
            action_cols_present = [col for col in self.action_cols if col in self.original_df.columns]
            
            if action_cols_present:
                orig_mean = self.original_df[action_cols_present].mean().mean()
                orig_std = self.original_df[action_cols_present].std().mean()
                aug_mean = augmented_df[action_cols_present].mean().mean()
                aug_std = augmented_df[action_cols_present].std().mean()
                
                stats_text += f"Original: μ={orig_mean:.4f}, σ={orig_std:.4f}<br>"
                stats_text += f"Augmented: μ={aug_mean:.4f}, σ={aug_std:.4f}<br>"
                stats_text += f"Δμ={aug_mean-orig_mean:.4f}, Δσ={aug_std-orig_std:.4f}"
            else:
                stats_text += "No action data available"
        
        self.stats_label.setText(stats_text)

    def _extract_signals(self, df: pd.DataFrame):
        """Return (actions_array, action_names, states_array, state_names) handling array columns."""
        action_names = self.session_reader.get_action_columns() if self.session_reader else []
        state_names = self.session_reader.get_state_columns() if self.session_reader else []
        
        if "action" in df.columns:
            actions = np.stack(df["action"].values)
        else:
            cols = [f"action.{n}"]
            cols = [c for c in [f"action.{n}" for n in action_names] if c in df.columns]
            actions = df[cols].values if cols else None
        
        if "observation.state" in df.columns:
            states = np.stack(df["observation.state"].values)
        else:
            cols = [c for c in [f"observation.state.{n}" for n in state_names] if c in df.columns]
            states = df[cols].values if cols else None
        
        return actions, action_names, states, state_names

    def update_plot(self):
        """Update the plotting area based on selections"""
        if self.original_df is None:
            return
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        plot_type = self.plot_type_combo.currentText()
        overlay = self.overlay_checkbox.isChecked()
        
        # Extract signals
        orig_actions, action_names, orig_states, state_names = self._extract_signals(self.original_df)
        aug_actions, _, aug_states, _ = (None, None, None, None)
        if self.last_augmented_df is not None:
            aug_actions, _, aug_states, _ = self._extract_signals(self.last_augmented_df)
        
        # Which signals
        a_index = self.action_signal_combo.currentIndex() - 1  # -1 => none
        s_index = self.state_signal_combo.currentIndex() - 1
        
        lines = []
        labels = []
        
        if a_index >= 0 and orig_actions is not None and a_index < (orig_actions.shape[1] if orig_actions.ndim == 2 else 0):
            y = orig_actions[:, a_index]
            if plot_type == "Time Series":
                l, = ax.plot(y, color="#1976D2", label=f"action.{action_names[a_index]} (orig)")
                lines.append(l); labels.append(l.get_label())
                if overlay and aug_actions is not None:
                    l2, = ax.plot(aug_actions[:, a_index], color="#E53935", alpha=0.7, label=f"action.{action_names[a_index]} (aug)")
                    lines.append(l2); labels.append(l2.get_label())
            else:
                ax.hist(y, bins=30, color="#1976D2", alpha=0.6, label=f"action.{action_names[a_index]} (orig)")
                if overlay and aug_actions is not None:
                    ax.hist(aug_actions[:, a_index], bins=30, color="#E53935", alpha=0.5, label=f"action.{action_names[a_index]} (aug)")
        
        if s_index >= 0 and orig_states is not None and s_index < (orig_states.shape[1] if orig_states.ndim == 2 else 0):
            y = orig_states[:, s_index]
            if plot_type == "Time Series":
                l, = ax.plot(y, color="#388E3C", label=f"state.{state_names[s_index]} (orig)")
                lines.append(l); labels.append(l.get_label())
                if overlay and aug_states is not None:
                    l2, = ax.plot(aug_states[:, s_index], color="#FDD835", alpha=0.7, label=f"state.{state_names[s_index]} (aug)")
                    lines.append(l2); labels.append(l2.get_label())
            else:
                ax.hist(y, bins=30, color="#388E3C", alpha=0.6, label=f"state.{state_names[s_index]} (orig)")
                if overlay and aug_states is not None:
                    ax.hist(aug_states[:, s_index], bins=30, color="#FDD835", alpha=0.5, label=f"state.{state_names[s_index]} (aug)")
        
        if labels:
            ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("index")
        ax.set_ylabel("value")
        self.figure.tight_layout()
        self.canvas.draw_idle()
    
    def get_params(self) -> dict:
        """Get current parameter values (for preview)"""
        params = {}
        for key, slider_info in self.sliders.items():
            value = slider_info["slider"].value() * slider_info["scale"]
            params[key] = value
        return params
    
    def get_export_ranges(self) -> dict:
        """Get parameter ranges for export"""
        # For parquet, return current values (not ranges for now)
        # Can be extended to ranges like video tab if needed
        return self.get_params()
    
    def reset_params(self):
        """Reset all parameters to defaults"""
        for key, slider_info in self.sliders.items():
            slider_info["slider"].setValue(slider_info["default"])
    
    def on_param_changed(self):
        """Handle parameter slider change"""
        # Update value labels
        for key, slider_info in self.sliders.items():
            value = slider_info["slider"].value() * slider_info["scale"]
            slider_info["label"].setText(f"{value:.4f}")
        
        # Update preview
        self.update_augmented()
    
    def on_sample_size_changed(self):
        """Handle sample size change"""
        self.load_sample()

