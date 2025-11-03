# Sim2Real Toolkit Parameters Guide

This guide provides comprehensive information about all augmentation parameters available in the Sim2Real Toolkit, including their effects, tradeoffs, advantages, and recommended usage.

## Table of Contents

1. [Video Augmentation Parameters](#video-augmentation-parameters)
   - [Photometric Parameters](#photometric-parameters)
   - [Blur & Optics Parameters](#blur--optics-parameters)
   - [Compression & Artifacts Parameters](#compression--artifacts-parameters)
2. [Parquet Augmentation Parameters](#parquet-augmentation-parameters)
   - [Sensor Noise Parameters](#sensor-noise-parameters)
   - [Temporal Effects Parameters](#temporal-effects-parameters)
   - [Actuator Dynamics Parameters](#actuator-dynamics-parameters)
3. [Usage Recommendations](#usage-recommendations)

---

## Video Augmentation Parameters

### Photometric Parameters

#### Gaussian Noise (`gaussian_noise`)
- **Range**: 0.0 - 0.1 (typical: 0.001 - 0.02)
- **Description**: Adds Gaussian noise to simulate sensor noise and low-light conditions
- **Advantages**:
  - Realistic sensor noise simulation
  - Improves robustness to low-quality image capture
  - Helps models generalize to noisy real-world conditions
- **Drawbacks**:
  - High values can obscure fine details
  - May degrade image quality significantly
  - Can make training harder if too aggressive
- **Tradeoffs**: Balance between realism and clarity. Start with 0.01 for subtle effect, increase to 0.02-0.05 for more aggressive augmentation
- **When to Use**: Essential for sim-to-real transfer. Use moderate values (0.01-0.02) for most applications

#### Shot Noise (`shot_noise`)
- **Range**: 0.0 - 0.1 (typical: 0.001 - 0.01)
- **Description**: Signal-dependent noise (Poisson-like) that increases with brightness
- **Advantages**:
  - More physically accurate than Gaussian noise
  - Realistic simulation of photon shot noise
  - Preserves relative brightness relationships
- **Drawbacks**:
  - More computationally expensive than Gaussian noise
  - Effect is less noticeable in dark regions
- **Tradeoffs**: Often used together with Gaussian noise. Lower values (0.005) provide subtle realism
- **When to Use**: When simulating high-quality sensors or when physics accuracy matters

#### Brightness (`brightness`)
- **Range**: -0.5 to +0.5 (typical: -0.1 to +0.1)
- **Description**: Random brightness adjustment (multiplier factor)
- **Advantages**:
  - Handles varying lighting conditions
  - Simple and effective augmentation
  - Improves robustness to exposure changes
- **Drawbacks**:
  - Extreme values can clip highlights/shadows
  - May lose information in over/under-exposed regions
- **Tradeoffs**: Lower values (±0.1) preserve image quality, higher values (±0.3) provide more diversity
- **When to Use**: Always useful. Keep moderate (±0.1) unless dealing with extreme lighting variations

#### Contrast (`contrast`)
- **Range**: -0.5 to +0.5 (typical: -0.1 to +0.1)
- **Description**: Random contrast adjustment around image mean
- **Advantages**:
  - Simulates different display settings and camera calibrations
  - Maintains mean brightness
  - Useful for domain adaptation
- **Drawbacks**:
  - High positive values can oversaturate colors
  - Low/negative values reduce visual distinction
- **Tradeoffs**: Moderate values (±0.1) preserve visual quality while adding diversity
- **When to Use**: Essential for handling different camera models and settings

#### Saturation (`saturation`)
- **Range**: -0.5 to +0.5 (typical: -0.1 to +0.1)
- **Description**: Random saturation adjustment in HSV color space
- **Advantages**:
  - Handles color calibration differences
  - Simulates different camera color profiles
  - Useful for monochrome/grayscale simulation at extreme negative values
- **Drawbacks**:
  - High values can create unrealistic colors
  - Negative values desaturate, losing color information
- **Tradeoffs**: Small values (±0.1) for subtle variations, larger for extreme conditions
- **When to Use**: Important for color-dependent tasks. Use moderate values unless color variance is critical

#### Hue (`hue`)
- **Range**: -180° to +180° (typical: -10° to +10°)
- **Description**: Random hue shift in HSV color space
- **Advantages**:
  - Handles white balance variations
  - Simulates different color temperatures
  - Useful for domain adaptation
- **Drawbacks**:
  - Large shifts can make objects unrecognizable
  - May confuse color-dependent tasks
- **Tradeoffs**: Small shifts (±10°) are realistic, larger shifts are for extreme augmentation
- **When to Use**: Use sparingly (±5-10°) unless color invariance is explicitly needed

#### Gamma (`gamma`)
- **Range**: 0.5 - 2.0 (typical: 0.8 - 1.2)
- **Description**: Gamma correction for non-linear brightness response
- **Advantages**:
  - Simulates different display characteristics
  - Handles non-linear color spaces
  - Realistic for many real-world scenarios
- **Drawbacks**:
  - Extreme values can significantly alter appearance
  - May require careful calibration
- **Tradeoffs**: Values near 1.0 (0.9-1.1) are most realistic for most displays
- **When to Use**: Useful for matching specific display characteristics or adding subtle variation

#### White Balance (`wb_r`, `wb_g`, `wb_b`)
- **Range**: 0.5 - 2.0 (typical: 0.9 - 1.1)
- **Description**: Per-channel RGB gain adjustment
- **Advantages**:
  - Realistic color temperature simulation
  - Handles different lighting conditions (indoor/outdoor)
  - More control than hue adjustment
- **Drawbacks**:
  - Requires careful tuning of all three channels
  - Can create unrealistic colors if unbalanced
- **Tradeoffs**: Keep channels balanced (small variations ±0.1) for realistic results
- **When to Use**: Important for color-critical applications. Use subtle adjustments unless matching specific lighting conditions

---

### Blur & Optics Parameters

#### Motion Blur (`motion_blur`)
- **Range**: 0 - 21 pixels (typical: 0 - 9)
- **Description**: Simulates camera or object motion blur
- **Advantages**:
  - Realistic for moving cameras/robots
  - Handles fast motion scenarios
  - Important for real-world robustness
- **Drawbacks**:
  - Obscures fine details and edges
  - Can degrade object recognition
  - Computationally expensive at high values
- **Tradeoffs**: Small values (3-5) for subtle motion, larger (7-15) for fast motion. High values (>15) may degrade training
- **When to Use**: Essential for mobile robots or moving cameras. Keep moderate (5-9) unless simulating fast motion

#### Defocus Blur (`defocus_blur`)
- **Range**: 0 - 15 pixels radius (typical: 0 - 5)
- **Description**: Circular (disk) blur simulating out-of-focus optics
- **Advantages**:
  - Realistic depth-of-field effects
  - Handles focus variations
  - Simulates lens imperfections
- **Drawbacks**:
  - High values blur entire image
  - Can make objects unrecognizable
- **Tradeoffs**: Low values (1-3) for subtle defocus, higher (5-10) for significant blur
- **When to Use**: Useful for depth-aware tasks or when simulating autofocus issues

#### Gaussian Blur (`gaussian_blur`)
- **Range**: 0.0 - 5.0 σ (typical: 0.0 - 2.0)
- **Description**: Gaussian blur for general smoothing
- **Advantages**:
  - Smooth, natural-looking blur
  - Less computationally expensive than motion blur
  - Good baseline blur augmentation
- **Drawbacks**:
  - Less realistic than motion/defocus blur
  - Can degrade edge detection
- **Tradeoffs**: Lower values (0.5-1.0) for subtle smoothing, higher (2.0-3.0) for strong blur
- **When to Use**: General-purpose blur when specific motion/defocus is not needed

#### Lens Distortion (`lens_k1`, `lens_k2`)
- **Range**: -0.1 to +0.1 for k1, -0.01 to +0.01 for k2 (typical: -0.05 to +0.05 for k1)
- **Description**: Radial lens distortion (barrel/pincushion)
- **Advantages**:
  - Realistic lens imperfections
  - Handles different camera models
  - Important for geometric accuracy
- **Drawbacks**:
  - Can distort geometric features
  - May confuse geometric tasks
  - Requires careful calibration
- **Tradeoffs**: Small values (±0.02) for subtle effects, larger for wide-angle simulation
- **When to Use**: Important for geometrically accurate tasks. Use conservatively unless matching specific lenses

#### Chromatic Aberration (`chromatic_aberration`)
- **Range**: 0 - 10 pixels shift (typical: 0 - 3)
- **Description**: RGB channel misalignment simulating lens dispersion
- **Advantages**:
  - Realistic lens artifact
  - Common in low-cost cameras
  - Adds visual realism
- **Drawbacks**:
  - Can degrade color accuracy
  - May create visual artifacts
- **Tradeoffs**: Low values (1-2) for subtle effect, higher (3-5) for noticeable aberration
- **When to Use**: Useful for simulating low-cost camera systems or wide-angle lenses

#### Vignetting (`vignetting`)
- **Range**: 0.0 - 1.0 strength (typical: 0.0 - 0.5)
- **Description**: Darkening at image corners/edges
- **Advantages**:
  - Realistic lens artifact
  - Common in many camera systems
  - Can guide attention to center
- **Drawbacks**:
  - Reduces information in corners
  - May bias models toward center
- **Tradeoffs**: Low values (0.1-0.3) for subtle effect, higher (0.5-0.7) for strong vignetting
- **When to Use**: Use sparingly unless matching specific camera characteristics

---

### Compression & Artifacts Parameters

#### JPEG Quality (`jpeg_quality`)
- **Range**: 10 - 100 (typical: 75 - 95)
- **Description**: JPEG compression quality (lower = more artifacts)
- **Advantages**:
  - Realistic compression artifacts
  - Handles bandwidth-limited scenarios
  - Common in real-world systems
- **Drawbacks**:
  - High compression degrades image quality
  - Creates blocking artifacts
  - May confuse fine-detail tasks
- **Tradeoffs**: Values >90 preserve quality, 75-85 add realistic compression, <70 can degrade significantly
- **When to Use**: Essential for simulating real-world transmission/storage. Use 85-90 for most applications

#### Fixed Pattern Noise (`fixed_pattern_noise`)
- **Range**: 0.0 - 0.1 strength (typical: 0.0 - 0.02)
- **Description**: Sensor-specific noise pattern (consistent across frames)
- **Advantages**:
  - Realistic sensor artifacts
  - Simulates CMOS/CCD characteristics
  - More realistic than random noise
- **Drawbacks**:
  - Can create systematic artifacts
  - May be learned by model if overused
- **Tradeoffs**: Low values (0.01-0.02) for subtle effect, higher for noticeable patterns
- **When to Use**: Useful for matching specific sensor characteristics

#### Temporal Flicker (`flicker`)
- **Range**: 0.0 - 0.1 intensity (typical: 0.0 - 0.05)
- **Description**: Time-varying brightness flicker
- **Advantages**:
  - Simulates power supply variations
  - Realistic for some camera systems
  - Adds temporal variation
- **Drawbacks**:
  - Can create distracting artifacts
  - May confuse temporal models
- **Tradeoffs**: Low values (0.02-0.05) for subtle flicker, higher for noticeable variation
- **When to Use**: Use sparingly unless simulating specific hardware issues

---

## Parquet Augmentation Parameters

### Sensor Noise Parameters

#### Gaussian Noise (`gaussian_noise`)
- **Range**: 0.0 - 0.1 σ (typical: 0.001 - 0.02)
- **Description**: Additive Gaussian noise to sensor readings
- **Advantages**:
  - Realistic sensor noise
  - Improves robustness to noisy measurements
  - Essential for sim-to-real transfer
- **Drawbacks**:
  - High values can obscure signal
  - May degrade control performance
- **Tradeoffs**: Start with 0.01 for subtle noise, increase to 0.02-0.05 for more aggressive augmentation
- **When to Use**: Essential for all sensor data. Use moderate values (0.01-0.02) for most sensors

#### Bias (`bias_std`)
- **Range**: 0.0 - 0.1 σ (typical: 0.001 - 0.01)
- **Description**: Constant offset per sensor dimension
- **Advantages**:
  - Realistic sensor calibration errors
  - Handles systematic offsets
  - Important for robustness
- **Drawbacks**:
  - Accumulates over time
  - Can cause drift in control
- **Tradeoffs**: Low values (0.005) simulate small calibration errors, higher (0.02-0.05) for significant bias
- **When to Use**: Important for simulating uncalibrated or poorly calibrated sensors

#### Drift (`drift_std`)
- **Range**: 0.0 - 0.01 σ per timestep (typical: 0.0001 - 0.001)
- **Description**: Slow random walk (temporal drift)
- **Advantages**:
  - Realistic sensor aging/thermal drift
  - Simulates long-term variations
  - Important for temporal robustness
- **Drawbacks**:
  - Can accumulate significantly over time
  - May destabilize long episodes
- **Tradeoffs**: Very low values (0.0001-0.001) for subtle drift, higher can cause significant deviation
- **When to Use**: Use conservatively (0.0001-0.0005) unless simulating specific drift scenarios

#### Quantization (`quantization`)
- **Range**: 0.0 - 0.01 step size (typical: 0.0001 - 0.001)
- **Description**: Round sensor values to discrete steps
- **Advantages**:
  - Realistic ADC quantization
  - Simulates low-resolution sensors
  - Important for digital systems
- **Drawbacks**:
  - Loses precision
  - Can create quantization artifacts
- **Tradeoffs**: Small steps (0.001) preserve precision, larger (0.005-0.01) simulate low-res sensors
- **When to Use**: Essential for simulating digital sensors. Match to actual sensor resolution

#### Outliers (`outliers_prob`, `outliers_scale`)
- **Range**: prob: 0.0 - 0.01, scale: 1.0 - 10.0 (typical: prob: 0.0001 - 0.001, scale: 3.0 - 7.0)
- **Description**: Random sensor glitches (spikes)
- **Advantages**:
  - Realistic sensor failures
  - Handles electromagnetic interference
  - Improves robustness to glitches
- **Drawbacks**:
  - Can confuse learning
  - High probability creates unrealistic data
- **Tradeoffs**: Low probability (0.001) with moderate scale (5.0) for realistic glitches
- **When to Use**: Use sparingly (prob < 0.001) unless simulating noisy environments

#### Dead Zone (`dead_zone`)
- **Range**: 0.0 - 0.05 threshold (typical: 0.001 - 0.01)
- **Description**: Values below threshold set to zero
- **Advantages**:
  - Realistic sensor thresholds
  - Simulates friction/stiction
  - Common in real sensors
- **Drawbacks**:
  - Loses information near zero
  - Can create discontinuities
- **Tradeoffs**: Low thresholds (0.005) preserve most data, higher (0.02-0.05) simulate significant dead zones
- **When to Use**: Important for actuators with friction or sensors with thresholds

---

### Temporal Effects Parameters

#### Latency Shift (`latency_shift`)
- **Range**: -10 to +10 frames (typical: 0 - 3)
- **Description**: Shift sensor readings by N frames
- **Advantages**:
  - Realistic communication delays
  - Handles processing latency
  - Important for real-world systems
- **Drawbacks**:
  - Creates temporal misalignment
  - Can confuse temporal models
  - Negative shifts may not be physically realistic
- **Tradeoffs**: Small shifts (1-2 frames) are realistic, larger shifts can cause significant misalignment
- **When to Use**: Essential for simulating real-world delays. Use 1-3 frames for most systems

#### Latency Mode (`latency_mode`)
- **Options**: "constant", "random", "jitter"
- **Description**: How latency is applied
- **Advantages**:
  - Constant: Simple, predictable
  - Random: More realistic variation
  - Jitter: Most realistic but complex
- **Drawbacks**:
  - Random/jitter can create inconsistent behavior
  - May confuse learning
- **Tradeoffs**: Start with "constant" for simplicity, use "random" for more realism
- **When to Use**: Use "constant" for most cases, "random" for variable delay simulation

#### Packet Loss (`packet_loss`)
- **Range**: 0.0 - 0.1 probability (typical: 0.001 - 0.01)
- **Description**: Randomly drop sensor readings (forward-filled)
- **Advantages**:
  - Realistic network issues
  - Handles communication failures
  - Improves robustness
- **Drawbacks**:
  - High rates create unrealistic data
  - Forward-filling may mask issues
- **Tradeoffs**: Low rates (0.001-0.005) for realistic packet loss, higher for extreme conditions
- **When to Use**: Use sparingly (< 0.01) unless simulating unreliable networks

#### Timestamp Jitter (`timestamp_jitter`)
- **Range**: 0.0 - 0.1 σ (typical: 0.001 - 0.01)
- **Description**: Add noise to timestamps
- **Advantages**:
  - Realistic clock imperfections
  - Handles synchronization issues
  - Important for temporal accuracy
- **Drawbacks**:
  - Can create temporal inconsistencies
  - May confuse temporal models
- **Tradeoffs**: Low values (0.001-0.005) for subtle jitter, higher for significant clock drift
- **When to Use**: Use conservatively unless simulating poor synchronization

#### Duplicate Rows (`duplicate_rows`)
- **Range**: 0.0 - 0.01 probability (typical: 0.001 - 0.005)
- **Description**: Randomly duplicate sensor readings
- **Advantages**:
  - Realistic sensor repeated readings
  - Simulates buffering issues
- **Drawbacks**:
  - Creates temporal artifacts
  - May confuse temporal models
- **Tradeoffs**: Very low rates (0.001-0.002) for subtle effect, higher for noticeable duplication
- **When to Use**: Use sparingly unless simulating specific hardware issues

---

### Actuator Dynamics Parameters

#### Saturation (`saturate`, `saturate_min`, `saturate_max`)
- **Range**: min: -2.0 to 0.0, max: 0.0 to 2.0 (typical: -1.0 to +1.0)
- **Description**: Clip actuator commands to limits
- **Advantages**:
  - Realistic actuator limits
  - Prevents unrealistic commands
  - Essential for safe operation
- **Drawbacks**:
  - May hide learning issues
  - Clipping can cause discontinuities
- **Tradeoffs**: Match to actual actuator limits. Common: ±1.0 for normalized actuators
- **When to Use**: Essential for all actuator commands. Always enable with realistic limits

#### Rate Limit (`rate_limit`)
- **Range**: 0.0 - 1.0 max change per timestep (typical: 0.05 - 0.2)
- **Description**: Limit rate of change between consecutive commands
- **Advantages**:
  - Realistic actuator dynamics
  - Prevents unrealistic jumps
  - Important for smooth control
- **Drawbacks**:
  - May limit performance
  - Can slow response
- **Tradeoffs**: Low values (0.05-0.1) for smooth control, higher (0.2-0.3) for faster response
- **When to Use**: Essential for simulating real actuators. Match to actual actuator capabilities

#### Backlash (`backlash`)
- **Range**: 0.0 - 0.1 offset (typical: 0.001 - 0.01)
- **Description**: Hysteresis when direction changes
- **Advantages**:
  - Realistic mechanical play
  - Simulates gear backlash
  - Important for precision tasks
- **Drawbacks**:
  - Creates non-linearities
  - Can complicate control
- **Tradeoffs**: Low values (0.005) for subtle backlash, higher (0.02-0.05) for significant play
- **When to Use**: Important for gear-driven actuators or high-precision tasks

#### Command Delay (`command_delay`)
- **Range**: 0 - 10 frames (typical: 0 - 3)
- **Description**: Delay actuator commands by N frames
- **Advantages**:
  - Realistic processing delays
  - Handles communication latency
  - Important for real-world systems
- **Drawbacks**:
  - Creates temporal misalignment
  - Can destabilize control
- **Tradeoffs**: Small delays (1-2 frames) are realistic, larger delays can cause instability
- **When to Use**: Use conservatively (1-2 frames) unless simulating specific delays

---

## Usage Recommendations

### General Guidelines

1. **Start Conservative**: Begin with low parameter values and gradually increase based on results
2. **Match Reality**: Calibrate parameters to match your target real-world system when possible
3. **Test Incrementally**: Test one augmentation category at a time to understand effects
4. **Monitor Performance**: Track model performance as you increase augmentation strength
5. **Use GUI Preview**: Always preview augmentations in the GUI before exporting

### Recommended Starting Values

#### Video Augmentations (Moderate Realism)
```yaml
video:
  gaussian_noise: 0.01
  shot_noise: 0.005
  brightness: 0.1
  contrast: 0.1
  saturation: 0.1
  hue: 10.0
  motion_blur: 5
  jpeg_quality: 85
```

#### Parquet Augmentations (Moderate Realism)
```yaml
parquet:
  gaussian_noise: 0.01
  bias_std: 0.005
  drift_std: 0.001
  latency_shift: 2
  saturate: true
  saturate_min: -1.0
  saturate_max: 1.0
  rate_limit: 0.1
```

### When to Increase Augmentation

- **Low Real-World Performance**: Gradually increase noise, blur, and distortion
- **Overfitting**: Increase augmentation diversity
- **Specific Issues**: Target specific failure modes (e.g., motion blur for moving cameras)
- **Domain Gap**: Increase parameters that bridge sim-to-real differences

### When to Decrease Augmentation

- **Training Instability**: Reduce noise and temporal effects
- **Poor Convergence**: Reduce augmentation strength
- **Unrealistic Results**: Ensure parameters match real-world characteristics
- **Performance Degradation**: Too much augmentation can hurt performance

### Parameter Interactions

- **Noise + Blur**: Can compound to reduce image quality significantly
- **Temporal Effects**: Latency shifts, delays, and jitter can interact to create complex temporal misalignments
- **Saturation + Rate Limits**: Together create realistic actuator constraints
- **Color Adjustments**: Multiple color parameters (brightness, contrast, saturation, hue) can interact non-linearly

### Workflow Tips

1. **Use GUI for Exploration**: Start with GUI to explore parameter space visually
2. **Export Configs**: Save successful parameter sets as YAML configs
3. **Incremental Testing**: Export small test batches before full session export
4. **Compare Metrics**: Track model metrics across different augmentation levels
5. **Iterate**: Adjust parameters based on model performance and real-world validation

---

## Quick Reference

| Parameter | Category | Typical Range | Priority |
|-----------|----------|---------------|----------|
| `gaussian_noise` | Video/Parquet | 0.01-0.02 | High |
| `brightness` | Video | ±0.1 | High |
| `motion_blur` | Video | 5-9 | Medium |
| `jpeg_quality` | Video | 85-90 | High |
| `latency_shift` | Parquet | 1-3 | High |
| `saturate` | Parquet | ±1.0 | High |
| `rate_limit` | Parquet | 0.1 | High |

---

## Troubleshooting

**Issue**: Augmented images look unrealistic
- **Solution**: Reduce augmentation strength, especially blur and noise parameters

**Issue**: Model performance degrades with augmentation
- **Solution**: Reduce augmentation intensity or remove problematic parameters

**Issue**: Training becomes unstable
- **Solution**: Reduce temporal effects (latency, delays) and sensor noise

**Issue**: Model doesn't generalize to real world
- **Solution**: Increase augmentation diversity, especially noise, blur, and color adjustments

**Issue**: Export takes too long
- **Solution**: Reduce sample size in GUI, or use CLI with smaller test batches first

---

For more information, see the main [README.md](README.md) or use the interactive GUI:
```bash
python -m sim2real_toolkit gui
```

