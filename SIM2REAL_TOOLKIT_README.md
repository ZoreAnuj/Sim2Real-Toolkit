# Sim2Real Augmentation Toolkit

**Interactive toolkit for bridging the sim-to-real gap in robot learning datasets**

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r sim2real_toolkit/requirements.txt
```

### 2. Launch Interactive GUI
```bash
python launch_gui.py
```

### 3. Load Your Session
- Click "Load Session Folder"
- Navigate to `session_2025-10-29_14-56-44` or your session folder
- Use sliders to adjust augmentation parameters
- See real-time preview of changes
- Export when satisfied

## 📦 What's Included

### Complete Augmentation Operations

#### Video Augmentations (20+ operations)
| Category | Operations |
|----------|-----------|
| **Photometric** | Gaussian noise, shot noise, brightness/contrast/saturation/hue jitter, gamma correction |
| **White Balance** | Per-channel RGB gains |
| **Blur & Optics** | Motion blur, defocus blur, Gaussian blur, lens distortion, chromatic aberration, vignetting |
| **Compression** | JPEG artifacts, fixed-pattern noise, temporal flicker |

#### Parquet Data Augmentations (15+ operations)
| Category | Operations |
|----------|-----------|
| **Sensor Noise** | Gaussian noise, bias/drift, quantization, outliers, dead zones |
| **Temporal Effects** | Latency shifts, packet loss, timestamp jitter, row duplication |
| **Actuator Dynamics** | Saturation, rate limits, backlash, command delays |

### Interactive GUI Features
- ✅ Real-time preview with adjustable sliders
- ✅ Side-by-side original vs augmented comparison
- ✅ Separate tabs for video and parquet modalities
- ✅ Multi-camera support
- ✅ Frame-by-frame navigation
- ✅ Export to new session with manifest
- ✅ Statistics comparison for tabular data

### Command-Line Interface
```bash
# Analyze session
python -m sim2real_toolkit analyze ./session_2025-10-29_14-56-44

# Generate config template
python -m sim2real_toolkit generate-config ./my_config.yaml

# Export augmented session
python -m sim2real_toolkit export \
    ./session_2025-10-29_14-56-44 \
    ./session_augmented \
    --config ./my_config.yaml

# Launch GUI
python -m sim2real_toolkit gui
```

### Python API
```python
from sim2real_toolkit import SessionReader, VideoAugmentor, ParquetAugmentor

# See example_usage.py for complete examples
```

## 📁 Project Structure

```
.
├── sim2real_toolkit/              # Main toolkit package
│   ├── io/                        # Session, video, parquet readers
│   ├── augmentations/             # Video & parquet augmentation ops
│   ├── gui/                       # Interactive PySide6 GUI
│   ├── export/                    # Session exporter
│   ├── cli.py                     # Command-line interface
│   ├── requirements.txt           # Dependencies
│   ├── setup.py                   # Installation script
│   └── README.md                  # Detailed documentation
│
├── session_2025-10-29_14-56-44/   # Your input session (example)
│   ├── videos/                    # Camera videos (MKV)
│   ├── data/                      # Action/state parquet
│   └── meta/                      # Metadata (info.json, etc.)
│
├── launch_gui.py                  # Quick GUI launcher
├── example_usage.py               # API examples
└── SIM2REAL_TOOLKIT_README.md     # This file
```

## 🎯 Use Cases

### 1. **Interactive Parameter Tuning** (Recommended)
Use the GUI to visually tune augmentation strength:
- Load session → adjust sliders → see instant results → export

### 2. **Batch Processing**
Use CLI with config files for reproducible augmentation pipelines:
- Create config YAML → run export command → get augmented session

### 3. **Research & Experimentation**
Use Python API to:
- Test different augmentation strategies
- Integrate with training pipelines
- Analyze augmentation effects on model performance

## 📊 Expected Session Format

Your session folder should follow this structure:

```
session_YYYY-MM-DD_HH-MM-SS/
├── videos/
│   ├── observation.images.camera_0/
│   │   └── chunk-000/
│   │       └── file-000.mkv
│   └── observation.images.camera_1/
│       └── chunk-000/
│           └── file-000.mkv
├── data/
│   └── chunk-000/
│       └── file-000.parquet  (columns: action.*, observation.state.*, timestamp, etc.)
└── meta/
    ├── info.json
    ├── stats.json
    └── episodes/
        └── chunk-000/
            └── file-000.parquet
```

## 🔬 Scientific Background

This toolkit implements augmentation strategies from:

- **Domain Randomization**: Tobin et al. (2017), Sadeghi & Levine (2017)
- **Dynamics Randomization**: Peng et al. (2018), OpenAI Dactyl (2019)
- **Sim→Real Translation**: Bousmalis et al. (2017), Zhu et al. (CycleGAN, 2017), Park et al. (CUT, 2020)
- **Robust Augmentation**: Hendrycks et al. (AugMix, 2019)
- **Domain Adaptation**: Ganin et al. (DANN, 2015), Sun et al. (CORAL, 2016)

## 🛠️ Troubleshooting

### Installation Issues
```bash
# Missing PyAV
pip install av

# Missing PySide6 (GUI)
pip install PySide6

# All dependencies
pip install -r sim2real_toolkit/requirements.txt
```

### Runtime Issues
- **GUI doesn't launch**: Check PySide6 installation
- **Video loading slow**: Try smaller frame indices or lower resolution
- **Export takes long**: Normal for large sessions (check console for progress)

## 📝 Configuration Example

```yaml
seed: 42

video:
  gaussian_noise: 0.01      # Photometric noise
  brightness: 0.1           # ±10% brightness variation
  motion_blur: 5            # 5px motion blur kernel
  vignetting: 0.3           # 30% vignetting strength
  jpeg_quality: 85          # JPEG compression quality

parquet:
  gaussian_noise: 0.01      # Sensor noise σ
  latency_shift: 2          # 2-frame action delay
  rate_limit: 0.1           # Max 0.1 change per step
  command_delay: 1          # 1-frame actuator delay
```

Generate template: `python -m sim2real_toolkit generate-config config.yaml`

## 🚧 Future Work / Placeholders

- [ ] **Domain Translation**: CUT/CycleGAN integration (you're working on this separately)
- [ ] **Temporal Consistency**: RecycleGAN-style video smoothing
- [ ] **Real Calibration**: Fit noise parameters from real data
- [ ] **Metrics**: FVD, LPIPS for augmentation quality assessment
- [ ] **Multi-chunk**: Support sessions with multiple chunks

## 📚 Documentation

- Full API docs: `sim2real_toolkit/README.md`
- Code examples: `example_usage.py`
- CLI help: `python -m sim2real_toolkit --help`

---

**Ready to bridge the sim2real gap? Launch the GUI and start experimenting!** 🚀

```bash
python launch_gui.py
```

