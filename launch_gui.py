#!/usr/bin/env python
"""Quick launcher for Sim2Real Toolkit GUI with parameter guidance."""

import sys
from pathlib import Path

# Add toolkit to path
toolkit_path = Path(__file__).parent / "sim2real_toolkit"
sys.path.insert(0, str(toolkit_path.parent))

from sim2real_toolkit.gui.main_window import run_gui


PARAMETER_GUIDE = [
    (
        "Video - Photometric",
        [
            (
                "Gaussian Noise",
                "Adds sensor grain; too high hides fine details and hurts perception.",
                "Preview 0.000-0.010 | Export max ≈0.015",
            ),
            (
                "Shot Noise",
                "Models photon noise; aggressive values create speckle artifacts.",
                "Preview 0.000-0.008 | Export max ≈0.012",
            ),
            (
                "Brightness Δ",
                "Offsets global brightness; large shifts break cross-camera consistency.",
                "Preview -0.15 to 0.15 | Export within ±0.20",
            ),
            (
                "Contrast Δ",
                "Boosts or flattens dynamic range; extremes clip highlights/shadows.",
                "Preview -0.15 to 0.15 | Export within ±0.20",
            ),
            (
                "Saturation Δ",
                "Controls color vividness; high boosts look synthetic, negative values wash out cues.",
                "Preview -0.20 to 0.20 | Export within ±0.25",
            ),
            (
                "Hue Δ (deg)",
                "Rotates color wheel; large shifts confuse semantic color cues.",
                "Preview ±15-25° | Export max ≈±30°",
            ),
            (
                "Gamma",
                "Non-linear tone curve; low gamma brightens shadows, high gamma darkens mid-tones.",
                "Preview 0.90-1.20 | Export 0.85-1.30",
            ),
        ],
    ),
    (
        "Video - White Balance",
        [
            (
                "Red Gain",
                "Balances channel tint; excessive gain oversaturates highlights.",
                "Preview 0.90-1.10 | Export 0.85-1.20",
            ),
            (
                "Green Gain",
                "Fine-tunes neutral axis; keep close to unity to avoid casts.",
                "Preview 0.95-1.05 | Export 0.90-1.15",
            ),
            (
                "Blue Gain",
                "Warms or cools image; large boosts introduce banding in shadows.",
                "Preview 0.90-1.10 | Export 0.85-1.20",
            ),
        ],
    ),
    (
        "Video - Blur & Optics",
        [
            (
                "Motion Blur Size",
                "Simulates motion smear; high values erase edges needed for tracking.",
                "Preview 0-8 px | Export max ≈12 px",
            ),
            (
                "Defocus Radius",
                "Out-of-focus blur; excessive blur destabilizes pose estimators.",
                "Preview 0-4 px | Export max ≈6 px",
            ),
            (
                "Gaussian σ",
                "Generic blur kernel; high sigma softens the frame and lowers SNR.",
                "Preview 0.0-2.0 | Export max ≈3.0",
            ),
            (
                "Lens Distortion k1",
                "Primary radial distortion; large magnitude curves straight edges unrealistically.",
                "Preview -0.02 to 0.02 | Export within ±0.04",
            ),
            (
                "Lens Distortion k2",
                "Higher-order radial term; strong values create wavy geometry.",
                "Preview -0.002 to 0.002 | Export within ±0.004",
            ),
            (
                "Chromatic Shift",
                "Emulates chromatic aberration; too high causes color fringing.",
                "Preview 0-3 px | Export max ≈4 px",
            ),
            (
                "Vignetting",
                "Darkens corners; aggressive values hide peripheral cues.",
                "Preview 0.00-0.35 | Export max ≈0.45",
            ),
        ],
    ),
    (
        "Video - Compression & Artifacts",
        [
            (
                "JPEG Quality",
                "Lower quality adds block artifacts; too low destroys fine structure.",
                "Preview 70-95 | Export 65-98",
            ),
            (
                "Fixed Pattern",
                "Adds stationary sensor noise; high levels create unrealistic stripes.",
                "Preview 0.000-0.030 | Export max ≈0.050",
            ),
            (
                "Temporal Flicker",
                "Frame-to-frame gain variation; excessive jitter confuses temporal models.",
                "Preview 0.00-0.25 | Export max ≈0.35",
            ),
        ],
    ),
    (
        "Parquet - Sensor Noise",
        [
            (
                "Gaussian Noise σ",
                "Adds zero-mean sensor noise; large σ masks subtle control signals.",
                "Preview 0.000-0.020 | Export max ≈0.030",
            ),
            (
                "Bias σ",
                "Random bias per rollout; high variance skews steady-state offsets.",
                "Preview 0.000-0.010 | Export max ≈0.015",
            ),
            (
                "Drift σ",
                "Slow random walk; too high diverges from realistic drift rates.",
                "Preview 0.0000-0.0020 | Export max ≈0.0030",
            ),
            (
                "Quantization Step",
                "Rounds signals; large steps create stair-stepping and lose precision.",
                "Preview 0.0000-0.0020 | Export max ≈0.0030",
            ),
            (
                "Outlier Probability",
                "Chance of spikes; high rates overwhelm filters expecting rare events.",
                "Preview 0.0000-0.0010 | Export max ≈0.0020",
            ),
            (
                "Outlier Scale",
                "Magnitude of injected spikes; keep moderate to avoid saturating actuators.",
                "Preview 1.0-4.0 | Export max ≈5.0",
            ),
            (
                "Dead Zone",
                "Suppresses small signals; too big hides low-amplitude corrections.",
                "Preview 0.000-0.015 | Export max ≈0.020",
            ),
        ],
    ),
    (
        "Parquet - Temporal Effects",
        [
            (
                "Latency Shift",
                "Shifts commands/states in time; large offsets break correspondence.",
                "Preview -2 to +2 frames | Export within ±3",
            ),
            (
                "Packet Loss Prob",
                "Drops rows; high probability yields unrealistic sparsity.",
                "Preview 0.000-0.015 | Export max ≈0.025",
            ),
            (
                "Timestamp Jitter σ",
                "Random timing noise; excessive jitter disrupts differentiation.",
                "Preview 0.000-0.010 | Export max ≈0.015",
            ),
            (
                "Row Duplication Prob",
                "Repeats rows; large values freeze actuators too often.",
                "Preview 0.000-0.020 | Export max ≈0.030",
            ),
        ],
    ),
    (
        "Parquet - Actuator Dynamics",
        [
            (
                "Saturation Min",
                "Lower clamp; aggressive limits remove valid negative commands.",
                "Preview around -1.10 | Export -1.20 to -0.80",
            ),
            (
                "Saturation Max",
                "Upper clamp; too low prevents reaching peak effort.",
                "Preview around 1.10 | Export 0.80-1.20",
            ),
            (
                "Rate Limit",
                "Constrains change per step; high limits slow responses and destabilize control.",
                "Preview 0.00-0.10 | Export max ≈0.20",
            ),
            (
                "Backlash",
                "Adds deadband before motion; excessive values mimic worn hardware.",
                "Preview 0.000-0.015 | Export max ≈0.020",
            ),
            (
                "Command Delay",
                "Delays actuator commands; more than 2-3 frames feels unrealistic.",
                "Preview 0-2 frames | Export max ≈3",
            ),
        ],
    ),
]


def print_parameter_guide() -> None:
    """Print trade-offs and recommended ranges for GUI parameters."""

    def format_line(label: str, text: str) -> str:
        return f"      {label:<12} {text}"

    print("Parameter Guide - Trade-offs & Recommended Ranges")
    print("-" * 60)
    for category, items in PARAMETER_GUIDE:
        print(f"{category}:")
        for name, tradeoff, recommendation in items:
            print(f"  {name}:")
            print(format_line("Trade-off", tradeoff))
            print(format_line("Recommended", recommendation))
        print()
    print("Note: Preview values affect the live view; export ranges drive batch randomization.")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("Sim2Real Augmentation Toolkit - Interactive GUI")
    print("=" * 60)
    print()
    print("Instructions:")
    print("1. Click 'Load Session Folder' to select your session directory")
    print("2. Use sliders to adjust augmentation parameters")
    print("3. Preview changes in real-time")
    print("4. Export augmented session when satisfied")
    print("5. (Optional) Click 'Export & Upload to Hugging Face' to publish")
    print()
    print_parameter_guide()
    print("=" * 60)
    print()

    run_gui()

