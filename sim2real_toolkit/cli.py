"""Command-line interface for sim2real toolkit"""

import argparse
import yaml
from pathlib import Path

from .io.session_reader import SessionReader
from .export.session_exporter import SessionExporter
from .gui.main_window import run_gui


def cmd_analyze(args):
    """Analyze session and print statistics"""
    session = SessionReader(args.session)
    
    print("=" * 60)
    print(f"Session Analysis: {Path(args.session).name}")
    print("=" * 60)
    
    # Video info
    info = session.get_video_info()
    print(f"\nVideo Information:")
    print(f"  Episodes: {info['total_episodes']}")
    print(f"  Frames: {info['total_frames']}")
    print(f"  FPS: {info['fps']}")
    print(f"  Resolution: {info['shape'][1]}x{info['shape'][0]}")
    
    # Cameras
    cameras = session.get_camera_keys()
    print(f"\nCameras ({len(cameras)}):")
    for cam in cameras:
        print(f"  - {cam}")
    
    # Actions
    action_cols = session.get_action_columns()
    print(f"\nActions ({len(action_cols)}):")
    for col in action_cols:
        print(f"  - {col}")
    
    # States
    state_cols = session.get_state_columns()
    print(f"\nStates ({len(state_cols)}):")
    for col in state_cols:
        print(f"  - {col}")
    
    # Episodes
    episodes = session.get_episode_ranges()
    print(f"\nEpisodes ({len(episodes)}):")
    for i, (start, end) in enumerate(episodes[:5]):  # Show first 5
        print(f"  Episode {i}: frames {start}-{end}")
    if len(episodes) > 5:
        print(f"  ... and {len(episodes) - 5} more")


def cmd_export(args):
    """Export augmented session"""
    # Load config
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Load session
    session = SessionReader(args.session)
    
    # Create exporter
    exporter = SessionExporter(
        session,
        args.output,
        video_params=config.get("video", {}),
        parquet_params=config.get("parquet", {}),
        seed=config.get("seed", 42)
    )
    
    # Export
    exporter.export(copy_meta=args.copy_meta)


def cmd_gui(args):
    """Launch GUI"""
    run_gui()


def cmd_generate_config(args):
    """Generate example config file"""
    config = {
        "seed": 42,
        "video": {
            "gaussian_noise": 0.01,
            "shot_noise": 0.005,
            "brightness": 0.1,
            "contrast": 0.1,
            "saturation": 0.1,
            "hue": 10.0,
            "gamma": 1.0,
            "wb_r": 1.0,
            "wb_g": 1.0,
            "wb_b": 1.0,
            "motion_blur": 5,
            "defocus_blur": 0,
            "gaussian_blur": 0.5,
            "lens_k1": 0.0,
            "lens_k2": 0.0,
            "chromatic_aberration": 1,
            "vignetting": 0.3,
            "jpeg_quality": 85,
            "fixed_pattern_noise": 0.01,
            "flicker": 0.05,
        },
        "parquet": {
            "gaussian_noise": 0.01,
            "bias_std": 0.005,
            "drift_std": 0.001,
            "quantization": 0.001,
            "outliers_prob": 0.001,
            "outliers_scale": 5.0,
            "dead_zone": 0.005,
            "latency_shift": 2,
            "latency_mode": "constant",
            "packet_loss": 0.005,
            "timestamp_jitter": 0.01,
            "duplicate_rows": 0.002,
            "saturate": True,
            "saturate_min": -1.0,
            "saturate_max": 1.0,
            "rate_limit": 0.1,
            "backlash": 0.01,
            "command_delay": 1,
        }
    }
    
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Generated example config: {output_path}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Sim2Real Augmentation Toolkit for Robot Learning Datasets"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze session statistics")
    analyze_parser.add_argument("session", help="Path to session folder")
    analyze_parser.set_defaults(func=cmd_analyze)
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export augmented session")
    export_parser.add_argument("session", help="Path to input session folder")
    export_parser.add_argument("output", help="Path to output folder")
    export_parser.add_argument("--config", "-c", help="Path to augmentation config YAML")
    export_parser.add_argument("--copy-meta", action="store_true", default=True,
                               help="Copy metadata files (default: True)")
    export_parser.set_defaults(func=cmd_export)
    
    # GUI command
    gui_parser = subparsers.add_parser("gui", help="Launch interactive GUI")
    gui_parser.set_defaults(func=cmd_gui)
    
    # Generate config command
    gen_config_parser = subparsers.add_parser("generate-config", 
                                               help="Generate example config file")
    gen_config_parser.add_argument("output", help="Output config file path")
    gen_config_parser.set_defaults(func=cmd_generate_config)
    
    # Parse and execute
    args = parser.parse_args()
    
    if not hasattr(args, "func"):
        parser.print_help()
        return
    
    args.func(args)


if __name__ == "__main__":
    main()

