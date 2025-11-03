# Sim2Real Augmentation Toolkit

A comprehensive toolkit for bridging the sim-to-real gap in robot learning datasets through realistic augmentations of video and tabular data.

## Features

### Video Augmentations
- **Photometric**: Gaussian/shot noise, brightness/contrast/saturation/hue jitter, gamma, white balance
- **Blur & Optics**: Motion blur, defocus blur, lens distortion, chromatic aberration, vignetting
- **Compression & Artifacts**: JPEG compression, fixed-pattern noise, temporal flicker
- **Domain Translation**: Placeholder for CUT/CycleGAN stylization (work separately)

### Parquet Data Augmentations
- **Sensor Noise**: Gaussian noise, bias/drift, quantization, outliers, dead zones
- **Temporal Effects**: Latency shifts, packet loss, timestamp jitter, row duplication
- **Actuator Dynamics**: Saturation, rate limits, backlash, command delays

### Interactive GUI
- Real-time preview with adjustable sliders for all parameters
- Separate tabs for video and parquet augmentations
- Side-by-side original vs augmented comparison
- Export to new session folder with manifest

## Installation

```bash
# Clone or extract toolkit
cd sim2real_toolkit

# Install dependencies
pip install -r requirements.txt

# Optional: Install in development mode
pip install -e .
```

## Usage

### 1. Interactive GUI (Recommended)

```bash
# Launch GUI
python -m sim2real_toolkit gui

# Or directly
python -m sim2real_toolkit.cli gui
```

**GUI Workflow:**
1. Click "Load Session Folder" and select your session directory
2. Switch between "Video Augmentations" and "Parquet Augmentations" tabs
3. Adjust sliders to tweak parameters in real-time
4. Preview changes immediately
5. Click "Export Augmented Session" when satisfied

### 2. Command-Line Interface

#### Analyze Session
```bash
python -m sim2real_toolkit analyze ./session_2025-10-29_14-56-44
```

#### Generate Example Config
```bash
python -m sim2real_toolkit generate-config ./augmentation_config.yaml
```

#### Export Augmented Session
```bash
python -m sim2real_toolkit export \
    ./session_2025-10-29_14-56-44 \
    ./session_augmented \
    --config ./augmentation_config.yaml
```

### 3. Python API

```python
from sim2real_toolkit import SessionReader, VideoAugmentor, ParquetAugmentor
from sim2real_toolkit.export import SessionExporter

# Load session
session = SessionReader("./session_2025-10-29_14-56-44")

# Define augmentation parameters
video_params = {
    "gaussian_noise": 0.01,
    "brightness": 0.1,
    "motion_blur": 5,
    "jpeg_quality": 85,
}

parquet_params = {
    "gaussian_noise": 0.01,
    "latency_shift": 2,
    "rate_limit": 0.1,
}

# Export augmented session
exporter = SessionExporter(
    session,
    "./session_augmented",
    video_params=video_params,
    parquet_params=parquet_params,
    seed=42
)
exporter.export()
```

## Configuration File Format

Example `augmentation_config.yaml`:

```yaml
seed: 42

video:
  # Photometric
  gaussian_noise: 0.01
  shot_noise: 0.005
  brightness: 0.1
  contrast: 0.1
  saturation: 0.1
  hue: 10.0
  gamma: 1.0
  
  # White balance
  wb_r: 1.0
  wb_g: 1.0
  wb_b: 1.0
  
  # Blur & Optics
  motion_blur: 5
  defocus_blur: 0
  gaussian_blur: 0.5
  lens_k1: 0.0
  lens_k2: 0.0
  chromatic_aberration: 1
  vignetting: 0.3
  
  # Compression & Artifacts
  jpeg_quality: 85
  fixed_pattern_noise: 0.01
  flicker: 0.05

parquet:
  # Sensor Noise
  gaussian_noise: 0.01
  bias_std: 0.005
  drift_std: 0.001
  quantization: 0.001
  outliers_prob: 0.001
  outliers_scale: 5.0
  dead_zone: 0.005
  
  # Temporal Effects
  latency_shift: 2
  latency_mode: "constant"  # "constant", "random", "jitter"
  packet_loss: 0.005
  timestamp_jitter: 0.01
  duplicate_rows: 0.002
  
  # Actuator Dynamics
  saturate: true
  saturate_min: -1.0
  saturate_max: 1.0
  rate_limit: 0.1
  backlash: 0.01
  command_delay: 1
```

## Expected Session Structure

```
session_2025-10-29_14-56-44/
├── videos/
│   ├── observation.images.camera_0/
│   │   └── chunk-000/
│   │       └── file-000.mkv
│   └── observation.images.camera_1/
│       └── chunk-000/
│           └── file-000.mkv
├── data/
│   └── chunk-000/
│       └── file-000.parquet
└── meta/
    ├── info.json
    ├── stats.json
    ├── tasks.parquet
    └── episodes/
        └── chunk-000/
            └── file-000.parquet
```

## Output Structure

Exported sessions maintain the same structure with an additional manifest:

```
session_augmented/
├── videos/          # Augmented videos
├── data/            # Augmented parquet
├── meta/            # Copied metadata
└── augmentation_manifest.json  # Applied parameters
```

## Key References

- **Domain Randomization**: Tobin et al. (2017), Sadeghi & Levine (2017), Tremblay et al. (2018)
- **Dynamics Randomization**: Peng et al. (2018), OpenAI Dactyl (2019)
- **Sim→Real Translation**: Bousmalis et al. (SimGAN, 2017), Zhu et al. (CycleGAN, 2017), Park et al. (CUT, 2020)
- **Robust Augmentation**: Hendrycks et al. (AugMix, 2019), Rusak et al. (DeepAugment, 2020)
- **Domain Adaptation**: Ganin et al. (DANN, 2015), Sun et al. (CORAL, 2016), Li et al. (AdaBN, 2016)

## Architecture

- `io/`: Session, video, and parquet readers
- `augmentations/`: Video and parquet augmentation operations
- `gui/`: Interactive PySide6 GUI with real-time preview
- `export/`: Session exporter with manifest generation
- `cli.py`: Command-line interface

## Development

```bash
# Install in editable mode
pip install -e .

# Run tests (if available)
pytest tests/

# Run GUI in development
python -m sim2real_toolkit gui
```

## Troubleshooting

**Issue**: "No module named 'av'"
**Solution**: Install PyAV: `pip install av`

**Issue**: GUI doesn't launch
**Solution**: Ensure PySide6 is installed: `pip install PySide6`

**Issue**: Video loading is slow
**Solution**: Use smaller frame indices or reduce sample size

## License

MIT License (adjust as needed)

## TODO / Future Work

- [ ] Add domain translation integration (CUT/CycleGAN) once trained
- [ ] Implement temporal consistency for video (RecycleGAN-style)
- [ ] Add real-world calibration utilities (fit noise curves from real data)
- [ ] Add FVD/LPIPS metrics for quality assessment
- [ ] Support multi-chunk sessions
- [ ] Add batch processing mode

