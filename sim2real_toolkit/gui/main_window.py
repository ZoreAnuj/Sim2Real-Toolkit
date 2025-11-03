"""Main GUI window with tabbed interface for video and parquet augmentations"""

import os
import sys
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QPushButton, QFileDialog, QLabel, QMessageBox,
    QInputDialog, QLineEdit
)
from PySide6.QtCore import Qt

from .video_tab import VideoAugmentationTab
from .parquet_tab import ParquetAugmentationTab
from ..io.session_reader import SessionReader


class MainWindow(QMainWindow):
    """Main window for Sim2Real Augmentation Toolkit"""
    
    def __init__(self):
        super().__init__()
        
        self.session_reader = None
        self.session_path = None
        self.last_export_path: Path | None = None
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI components"""
        self.setWindowTitle("Sim2Real Augmentation Toolkit")
        self.setGeometry(100, 100, 1600, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Top bar with session loader
        top_bar = self._create_top_bar()
        main_layout.addWidget(top_bar)
        
        # Status label
        self.status_label = QLabel("No session loaded")
        self.status_label.setStyleSheet("padding: 5px; background-color: #f0f0f0;")
        main_layout.addWidget(self.status_label)
        
        # Tab widget
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Video augmentation tab
        self.video_tab = VideoAugmentationTab()
        self.tabs.addTab(self.video_tab, "Video Augmentations")
        
        # Parquet augmentation tab
        self.parquet_tab = ParquetAugmentationTab()
        self.tabs.addTab(self.parquet_tab, "Parquet Augmentations")
        
        # Bottom bar with export controls
        bottom_bar = self._create_bottom_bar()
        main_layout.addWidget(bottom_bar)
    
    def _create_top_bar(self) -> QWidget:
        """Create top bar with session loading controls"""
        widget = QWidget()
        layout = QHBoxLayout()
        widget.setLayout(layout)
        
        # Load session button
        load_btn = QPushButton("Load Session Folder")
        load_btn.clicked.connect(self.load_session)
        layout.addWidget(load_btn)
        
        layout.addStretch()
        
        return widget
    
    def _create_bottom_bar(self) -> QWidget:
        """Create bottom bar with export controls"""
        widget = QWidget()
        layout = QHBoxLayout()
        widget.setLayout(layout)
        
        layout.addStretch()
        
        # Export button
        self.export_btn = QPushButton("Export Augmented Session")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_session)
        layout.addWidget(self.export_btn)

        # Upload button
        self.upload_btn = QPushButton("Export && Upload to Hugging Face")
        self.upload_btn.setEnabled(False)
        self.upload_btn.clicked.connect(self.upload_to_hf)
        layout.addWidget(self.upload_btn)
        
        return widget
    
    def load_session(self):
        """Load a session folder"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Session Folder",
            "",
            QFileDialog.ShowDirsOnly
        )
        
        if not folder:
            return
        
        try:
            self.session_path = Path(folder)
            self.session_reader = SessionReader(str(self.session_path))
            
            # Update status
            info = self.session_reader.get_video_info()
            self.status_label.setText(
                f"Loaded: {self.session_path.name} | "
                f"Episodes: {info['total_episodes']} | "
                f"Frames: {info['total_frames']} | "
                f"FPS: {info['fps']} | "
                f"Resolution: {info['shape'][1]}x{info['shape'][0]}"
            )
            
            # Load data into tabs
            self.video_tab.load_session(self.session_reader)
            self.parquet_tab.load_session(self.session_reader)
            
            # Enable export actions
            self.export_btn.setEnabled(True)
            self.upload_btn.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Loading Session",
                f"Failed to load session:\n{str(e)}"
            )
    
    def export_session(self):
        """Export augmented session to new folder"""
        if not self.session_reader:
            return
        
        # Get output folder
        output_folder = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder",
            str(self.session_path.parent)
        )
        
        if not output_folder:
            return
        
        self._export_and_notify(Path(output_folder))

    def _export_and_notify(self, output_path: Path, *, show_start: bool = True, show_success: bool = True) -> bool:
        """Run exporter and show message boxes. Returns True on success."""

        try:
            from ..export.session_exporter import SessionExporter
        except Exception as exc:  # pragma: no cover - import guard
            QMessageBox.critical(
                self,
                "Exporter Import Error",
                f"Failed to import exporter module.\n{exc}"
            )
            return False

        # Collect parameters
        video_ranges = self.video_tab.get_export_ranges()
        parquet_params = self.parquet_tab.get_export_ranges()

        exporter = SessionExporter(
            self.session_reader,
            str(output_path),
            video_param_ranges=video_ranges,
            parquet_params=parquet_params,
            seed=42,
        )

        try:
            if show_start:
                QMessageBox.information(
                    self,
                    "Export Started",
                    "Export started. This may take several minutes.\nCheck console for progress."
                )

            exporter.export(copy_meta=True)

            self.last_export_path = output_path

            if show_success:
                QMessageBox.information(
                    self,
                    "Export Complete",
                    f"Successfully exported augmented session to:\n{output_path}"
                )

            return True

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to export session:\n{exc}"
            )
            return False

    def upload_to_hf(self):
        """Export augmented session and upload to Hugging Face Hub."""
        if not self.session_reader:
            return

        # Choose output folder
        output_folder = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder",
            str(self.session_path.parent)
        )

        if not output_folder:
            return

        output_path = Path(output_folder)

        # Export first (no dialogs besides errors)
        if not self._export_and_notify(output_path, show_start=True, show_success=False):
            return

        # Import huggingface_hub lazily
        try:
            from huggingface_hub import HfApi, upload_folder
        except ImportError:
            QMessageBox.critical(
                self,
                "Missing Dependency",
                "huggingface_hub is required to upload.\nInstall with: pip install huggingface_hub"
            )
            return

        # Prompt for repo id
        repo_id, ok = QInputDialog.getText(
            self,
            "Hugging Face Repository",
            "Enter dataset repository (e.g. username/session-name):"
        )
        if not ok or not repo_id.strip():
            return

        repo_id = repo_id.strip()

        # Token: try env, else prompt
        token = os.environ.get("HUGGINGFACE_HUB_TOKEN", "").strip()
        if not token:
            token, ok = QInputDialog.getText(
                self,
                "Hugging Face Token",
                "Enter a write token (stored temporarily for this upload only):",
                QLineEdit.Password
            )
            if not ok or not token.strip():
                return
            token = token.strip()

        # Optional branch name
        branch, ok = QInputDialog.getText(
            self,
            "Target Branch",
            "Enter target branch (default: main):",
            text="main"
        )
        if not ok:
            return
        branch = (branch or "main").strip() or "main"

        # Run upload
        try:
            api = HfApi()
            api.create_repo(repo_id, repo_type="dataset", exist_ok=True, token=token)

            QMessageBox.information(
                self,
                "Upload Started",
                "Uploading folder to Hugging Face Hub. This may take a while." 
            )

            upload_folder(
                repo_id=repo_id,
                folder_path=str(output_path),
                repo_type="dataset",
                token=token,
                commit_message=f"Upload augmented session {self.session_path.name}",
                revision=branch,
            )

            QMessageBox.information(
                self,
                "Upload Complete",
                f"Exported session uploaded to https://huggingface.co/datasets/{repo_id}"
            )

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Upload Error",
                f"Failed to upload to Hugging Face:\n{exc}"
            )


def run_gui():
    """Entry point for GUI application"""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

