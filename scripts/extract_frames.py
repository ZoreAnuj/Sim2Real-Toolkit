#!/usr/bin/env python3
"""
Extract frames from simulation and real videos using ffmpeg.

Output layout:
  <out_root>/sim/<video-stem>/%06d.jpg
  <out_root>/real/<video-stem>/%06d.jpg

Requirements:
  - ffmpeg must be available on PATH

Example:
  python scripts/extract_frames.py \
    --sim_dir D:\\Pix2Pix\\stretch_sim \
    --real_dir D:\\Pix2Pix\\stretch_real \
    --out_root D:\\Pix2Pix\\frames \
    --fps 2 --size 256 --workers 8
"""

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


def assert_ffmpeg_available() -> None:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        sys.stderr.write(
            "ffmpeg not found on PATH. Please install ffmpeg and ensure 'ffmpeg' is callable.\n"
        )
        sys.exit(1)


def discover_videos(root_dir: Optional[Path], exts: Tuple[str, ...]) -> List[Path]:
    if root_dir is None:
        return []
    videos: List[Path] = []
    for ext in exts:
        videos.extend(root_dir.glob(f"*.{ext}"))
    # Also search one-level subdirectories just in case
    for ext in exts:
        videos.extend(root_dir.glob(f"*/*.{ext}"))
    # Deduplicate while preserving order
    seen = set()
    unique: List[Path] = []
    for v in videos:
        if v.resolve() not in seen:
            unique.append(v)
            seen.add(v.resolve())
    return unique


def build_ffmpeg_cmd(input_path: Path, output_dir: Path, fps: float, size: int) -> List[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Use POSIX-style path for ffmpeg on Windows to avoid backslash issues
    output_pattern = (output_dir / "%06d.jpg").as_posix()
    # Ensure even height with -2 to avoid codec issues; chain fps then scale
    vf = f"fps={fps},scale=w={size}:h=-2"
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-vf",
        vf,
        "-vsync",
        "vfr",
        output_pattern,
    ]


def already_extracted(output_dir: Path, min_count: int = 1) -> bool:
    if not output_dir.exists():
        return False
    try:
        # Quick check for any jpg present
        first = next(output_dir.glob("*.jpg"))
        return any([first])
    except StopIteration:
        return False


def extract_one(input_path: Path, split: str, out_root: Path, fps: float, size: int, force: bool) -> Tuple[str, str, int]:
    stem = input_path.stem
    out_dir = out_root / split / stem
    if already_extracted(out_dir) and not force:
        return (split, stem, 0)
    cmd = build_ffmpeg_cmd(input_path, out_dir, fps, size)
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
        # Count frames
        count = len(list(out_dir.glob("*.jpg")))
        if count == 0:
            raise RuntimeError(f"No frames were written for {input_path}. FFmpeg output: {proc.stderr.strip() or proc.stdout.strip()}\nCommand: {' '.join(cmd)}")
        return (split, stem, count)
    except Exception as e:
        raise RuntimeError(f"Failed on {input_path}: {e}")


def process_many(
    videos: Iterable[Path],
    split: str,
    out_root: Path,
    fps: float,
    size: int,
    workers: int,
    force: bool,
) -> List[Tuple[str, str, int]]:
    results: List[Tuple[str, str, int]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = [
            ex.submit(extract_one, v, split, out_root, fps, size, force) for v in videos
        ]
        for fut in concurrent.futures.as_completed(futures):
            results.append(fut.result())
    return results


def write_manifest(out_root: Path, entries: List[Tuple[str, str, int]]) -> None:
    manifest_path = out_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {"split": s, "video": vid, "num_frames": n} for (s, vid, n) in entries
    ]
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract frames from sim and real videos using ffmpeg")
    p.add_argument("--sim_dir", type=Path, required=False, help="Directory with simulation videos (e.g., .mkv)")
    p.add_argument("--real_dir", type=Path, required=False, help="Directory with real videos (e.g., .mp4)")
    p.add_argument("--out_root", type=Path, required=True, help="Output root directory for frames")
    p.add_argument("--fps", type=float, default=2.0, help="Frames per second to sample")
    p.add_argument("--size", type=int, default=256, help="Short-side resize in pixels")
    p.add_argument("--workers", type=int, default=8, help="Parallel workers (threaded; ffmpeg is external)")
    p.add_argument(
        "--force", action="store_true", help="Re-extract even if output frames already exist"
    )
    p.add_argument(
        "--exts",
        type=str,
        default="mp4,mkv,avi,mov",
        help="Comma-separated list of video extensions to consider",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    assert_ffmpeg_available()

    out_root: Path = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    exts = tuple([e.strip().lower().lstrip(".") for e in args.exts.split(",") if e.strip()])

    sim_dir = args.sim_dir if args.sim_dir is not None else None
    real_dir = args.real_dir if args.real_dir is not None else None

    sim_videos = discover_videos(sim_dir, exts) if sim_dir else []
    real_videos = discover_videos(real_dir, exts) if real_dir else []

    if not sim_videos and not real_videos:
        sys.stderr.write("No videos found. Check --sim_dir/--real_dir and extensions.\n")
        sys.exit(1)

    print(f"Found {len(sim_videos)} sim videos and {len(real_videos)} real videos.")
    all_entries: List[Tuple[str, str, int]] = []

    if sim_videos:
        print("Extracting sim -> frames...")
        entries = process_many(
            sim_videos,
            split="sim",
            out_root=out_root,
            fps=args.fps,
            size=args.size,
            workers=args.workers,
            force=args.force,
        )
        all_entries.extend(entries)
        sim_total = sum(n for (_, _, n) in entries)
        print(f"Sim extraction done: {len(entries)} videos, {sim_total} frames.")

    if real_videos:
        print("Extracting real -> frames...")
        entries = process_many(
            real_videos,
            split="real",
            out_root=out_root,
            fps=args.fps,
            size=args.size,
            workers=args.workers,
            force=args.force,
        )
        all_entries.extend(entries)
        real_total = sum(n for (_, _, n) in entries)
        print(f"Real extraction done: {len(entries)} videos, {real_total} frames.")

    write_manifest(out_root, all_entries)
    print(f"Wrote manifest: {out_root / 'manifest.json'}")


if __name__ == "__main__":
    main()


