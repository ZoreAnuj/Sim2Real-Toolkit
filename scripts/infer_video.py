#!/usr/bin/env python3
"""
Apply domain transfer to a simulation video using trained CUT model.

Creates a side-by-side video: [Simulated | Domain Transferred]

Usage:
    python scripts/infer_video.py \
        --checkpoint runs/cut/cut_final.pth \
        --config configs/cut.yaml \
        --input_video D:/Pix2Pix/stretch_sim/file-000.mkv \
        --output_video output/demo.mp4 \
        --fps 30
"""

import argparse
import sys
from pathlib import Path
import yaml
from typing import Optional

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.cut.networks import ResnetGenerator


def load_config(path: str) -> dict:
    """Load YAML config file"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def denormalize(img: torch.Tensor) -> torch.Tensor:
    """Convert from [-1, 1] to [0, 1]"""
    return (img + 1) / 2


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor [C, H, W] to PIL Image"""
    tensor = denormalize(tensor).clamp(0, 1)
    to_pil = T.ToPILImage()
    return to_pil(tensor.cpu())


def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV format (BGR)"""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def process_frame(frame: np.ndarray, transform: T.Compose, device: torch.device) -> np.ndarray:
    """Process a single frame: BGR -> RGB -> Tensor -> Model -> PIL -> BGR"""
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    
    # Transform to tensor
    img_tensor = transform(pil_img).unsqueeze(0).to(device)
    
    return img_tensor


def load_model(checkpoint_path: Path, config: dict, device: torch.device) -> ResnetGenerator:
    """Load trained generator from checkpoint"""
    print(f"🔄 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    G = ResnetGenerator(
        ngf=int(config["model"]["gen_channels"]),
        n_blocks=int(config["model"]["n_res_blocks"])
    )
    G.load_state_dict(checkpoint["G"])
    G.to(device)
    G.eval()
    
    print("✅ Model loaded successfully")
    return G


def create_transforms(config: dict) -> T.Compose:
    """Create image transforms matching training"""
    image_size = int(config["data"]["image_size"])
    
    transforms = [
        T.Resize(image_size, interpolation=T.InterpolationMode.LANCZOS, antialias=True),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # [-1, 1]
    ]
    
    return T.Compose(transforms)


def resize_frame(frame: np.ndarray, target_size: int) -> np.ndarray:
    """Resize frame maintaining aspect ratio"""
    h, w = frame.shape[:2]
    
    # Calculate new dimensions maintaining aspect ratio
    if h > w:
        new_h = target_size
        new_w = int(w * target_size / h)
    else:
        new_w = target_size
        new_h = int(h * target_size / w)
    
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    return resized


def process_video(
    input_video: Path,
    output_video: Path,
    checkpoint_path: Path,
    config_path: Path,
    fps: Optional[float] = None,
    device: Optional[torch.device] = None,
    batch_size: int = 4,
) -> None:
    """Process video frame by frame with domain transfer"""
    
    # Setup
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    # Load config
    config = load_config(config_path)
    image_size = int(config["data"]["image_size"])
    
    # Load model
    G = load_model(checkpoint_path, config, device)
    
    # Create transforms
    transform = create_transforms(config)
    
    # Open input video
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_video}")
    
    # Get video properties
    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📹 Video info: {original_width}x{original_height} @ {fps:.2f} fps, {total_frames} frames")
    
    # Create output directory
    output_video.parent.mkdir(parents=True, exist_ok=True)
    
    # Setup video writer (will write final concatenated frames)
    # Output will be 2x width (sim | transferred)
    output_width = original_width * 2
    output_height = original_height
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        str(output_video),
        fourcc,
        fps,
        (output_width, output_height)
    )
    
    if not out.isOpened():
        raise RuntimeError(f"Failed to create output video: {output_video}")
    
    print(f"💾 Output video: {output_video} ({output_width}x{output_height})")
    
    # Process frames
    frame_buffer = []
    frame_indices = []
    
    pbar = tqdm(total=total_frames, desc="Processing frames")
    
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame if needed (maintain aspect ratio)
            if original_width != image_size or original_height != image_size:
                frame_resized = resize_frame(frame, image_size)
                # Pad to square if needed
                h, w = frame_resized.shape[:2]
                if h != w:
                    size = max(h, w)
                    frame_padded = np.zeros((size, size, 3), dtype=np.uint8)
                    y_offset = (size - h) // 2
                    x_offset = (size - w) // 2
                    frame_padded[y_offset:y_offset+h, x_offset:x_offset+w] = frame_resized
                    frame_resized = frame_padded
                
                # Resize to exact size
                frame_resized = cv2.resize(frame_resized, (image_size, image_size))
            else:
                frame_resized = frame
            
            # Process frame
            img_tensor = process_frame(frame_resized, transform, device)
            frame_buffer.append((frame, img_tensor))
            frame_indices.append(len(frame_buffer) - 1)
            
            # Process batch when buffer is full
            if len(frame_buffer) >= batch_size:
                # Extract tensors
                batch_tensors = torch.cat([fb[1] for fb in frame_buffer], dim=0)
                
                # Generate transferred frames
                fake_tensors, _ = G(batch_tensors, return_feats=False)
                
                # Process each frame in batch
                for i, (orig_frame, _) in enumerate(frame_buffer):
                    # Get original frame (resize back if needed)
                    if original_width != image_size or original_height != image_size:
                        orig_display = resize_frame(orig_frame, max(original_height, original_width))
                        # Crop to original aspect ratio
                        h, w = orig_display.shape[:2]
                        if h > original_height:
                            y_start = (h - original_height) // 2
                            orig_display = orig_display[y_start:y_start+original_height, :]
                        if w > original_width:
                            x_start = (w - original_width) // 2
                            orig_display = orig_display[:, x_start:x_start+original_width]
                        orig_display = cv2.resize(orig_display, (original_width, original_height))
                    else:
                        orig_display = orig_frame
                    
                    # Get transferred frame
                    fake_tensor = fake_tensors[i]
                    fake_pil = tensor_to_pil(fake_tensor)
                    
                    # Resize transferred frame to match original size
                    fake_np = np.array(fake_pil)
                    fake_resized = cv2.resize(fake_np, (original_width, original_height))
                    fake_bgr = cv2.cvtColor(fake_resized, cv2.COLOR_RGB2BGR)
                    
                    # Concatenate horizontally: [Original | Transferred]
                    concatenated = np.hstack([orig_display, fake_bgr])
                    
                    # Write frame
                    out.write(concatenated)
                
                frame_buffer.clear()
                pbar.update(batch_size)
        
        # Process remaining frames
        if len(frame_buffer) > 0:
            batch_tensors = torch.cat([fb[1] for fb in frame_buffer], dim=0)
            fake_tensors, _ = G(batch_tensors, return_feats=False)
            
            for i, (orig_frame, _) in enumerate(frame_buffer):
                if original_width != image_size or original_height != image_size:
                    orig_display = resize_frame(orig_frame, max(original_height, original_width))
                    h, w = orig_display.shape[:2]
                    if h > original_height:
                        y_start = (h - original_height) // 2
                        orig_display = orig_display[y_start:y_start+original_height, :]
                    if w > original_width:
                        x_start = (w - original_width) // 2
                        orig_display = orig_display[:, x_start:x_start+original_width]
                    orig_display = cv2.resize(orig_display, (original_width, original_height))
                else:
                    orig_display = orig_frame
                
                fake_tensor = fake_tensors[i]
                fake_pil = tensor_to_pil(fake_tensor)
                fake_np = np.array(fake_pil)
                fake_resized = cv2.resize(fake_np, (original_width, original_height))
                fake_bgr = cv2.cvtColor(fake_resized, cv2.COLOR_RGB2BGR)
                
                concatenated = np.hstack([orig_display, fake_bgr])
                out.write(concatenated)
            
            pbar.update(len(frame_buffer))
    
    # Cleanup
    cap.release()
    out.release()
    pbar.close()
    
    print(f"✅ Video saved: {output_video}")


def main():
    parser = argparse.ArgumentParser(description="Apply domain transfer to simulation video")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to checkpoint file (e.g., runs/cut/cut_final.pth)")
    parser.add_argument("--config", type=str, default="configs/cut.yaml",
                       help="Path to config file")
    parser.add_argument("--input_video", type=str, required=True,
                       help="Path to input simulation video")
    parser.add_argument("--output_video", type=str, required=True,
                       help="Path to output video")
    parser.add_argument("--fps", type=float, default=None,
                       help="Output FPS (default: same as input)")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for processing (default: 4)")
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint).resolve()
    config_path = Path(args.config).resolve()
    input_video = Path(args.input_video).resolve()
    output_video = Path(args.output_video).resolve()
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")
    
    process_video(
        input_video=input_video,
        output_video=output_video,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        fps=args.fps,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

