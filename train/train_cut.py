import argparse
import os
import sys
from pathlib import Path
import yaml
from typing import List

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import torchvision.transforms as T
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.unpaired_frames_dataset import UnpairedFramesDataset
from models.cut.networks import ResnetGenerator, NLayerDiscriminator, PatchSampleMLP
from models.cut.losses import (
    HingeGANLoss,
    PatchNCELoss,
    IdentityLoss,
    LuminanceLoss,
    TVLoss,
    SaturationLoss,
    PatchColorLoss,
    ClippedHighlightsLoss,
    r1_regularization,
)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def denormalize(img: torch.Tensor) -> torch.Tensor:
    """Convert from [-1, 1] to [0, 1]"""
    return (img + 1) / 2

def collate_simple(batch):
    # Custom collate to avoid Windows storage resize issues
    if len(batch[0]) == 4:
        sims, reals, sim_paths, real_paths = zip(*batch)
        return torch.stack(sims, 0), torch.stack(reals, 0), list(sim_paths), list(real_paths)
    else:
        sims, reals = zip(*batch)
        return torch.stack(sims, 0), torch.stack(reals, 0)


def save_batch_images(src_img: torch.Tensor, fake_img: torch.Tensor, tgt_img: torch.Tensor, 
                     save_dir: Path, epoch: int, batch_idx: int = 0) -> None:
    """Save a grid of images for visualization"""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Denormalize from [-1, 1] to [0, 1] and clamp to valid range
    src_viz = denormalize(src_img[:4]).clamp(0, 1)  # First 4 images from batch
    fake_viz = denormalize(fake_img[:4]).clamp(0, 1)
    tgt_viz = denormalize(tgt_img[:4]).clamp(0, 1)
    
    # Create a grid: src | fake | tgt
    grid = []
    for i in range(4):
        grid.append(src_viz[i])
        grid.append(fake_viz[i])
        grid.append(tgt_viz[i])
    
    # Save individual comparisons
    for i in range(4):
        sim_pil = T.ToPILImage()(src_viz[i].cpu())
        fake_pil = T.ToPILImage()(fake_viz[i].cpu())
        real_pil = T.ToPILImage()(tgt_viz[i].cpu())
        
        # Concatenate horizontally
        w, h = sim_pil.size
        result = Image.new('RGB', (w * 3, h))
        result.paste(sim_pil, (0, 0))
        result.paste(fake_pil, (w, 0))
        result.paste(real_pil, (w * 2, 0))
        # Save with high quality to retain sharpness
        result.save(
            save_dir / f"epoch_{epoch:03d}_batch_{batch_idx}_sample_{i}.jpg",
            quality=95,
            subsampling=0,
            optimize=True,
        )


def visualize_checkpoint(checkpoint_path: Path, data_loader: DataLoader, device: torch.device,
                         feat_layers: List[int], cfg: dict, save_dir: Path, epoch: int) -> None:
    """Load checkpoint and generate visualization on first batch"""
    print(f"\n📊 Generating visualization for epoch {epoch}...")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    G = ResnetGenerator(
        ngf=int(cfg["model"]["gen_channels"]),
        n_blocks=int(cfg["model"]["n_res_blocks"]),
        up_mode=str(cfg["model"].get("up_mode", "nearest")),
    )
    G.load_state_dict(checkpoint["G"])
    G.to(device)
    G.eval()
    
    # Get first batch
    batch = next(iter(data_loader))
    if len(batch) == 4:
        sim_img, real_img, _, _ = batch
    else:
        sim_img, real_img = batch
    sim_img = sim_img.to(device, non_blocking=True)
    real_img = real_img.to(device, non_blocking=True)
    direction = str(cfg.get("data", {}).get("direction", "sim2real")).lower()
    sim2real = (direction == "sim2real")
    src_img = sim_img if sim2real else real_img
    tgt_img = real_img if sim2real else sim_img
    
    # Generate fake images
    with torch.no_grad():
        fake_img, _ = G(src_img, return_feats=False)
    
    # Save visualization
    save_batch_images(src_img, fake_img, tgt_img, save_dir / "visualizations", epoch)
    print(f"✅ Saved visualization to {save_dir / 'visualizations'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cut.yaml")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from (e.g., runs/cut/cut_epoch_10.pth)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out or cfg["logging"]["out_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    direction = str(cfg.get("data", {}).get("direction", "sim2real")).lower()
    sim2real = (direction == "sim2real")

    temporal_cfg = cfg["loss"].get("temporal", {"enabled": False})
    temporal_enabled = bool(temporal_cfg.get("enabled", False))

    ds = UnpairedFramesDataset(
        sim_root=cfg["data"]["sim_root"],
        real_root=cfg["data"]["real_root"],
        image_size=int(cfg["data"]["image_size"]),
        random_crop=bool(cfg["data"].get("random_crop", True)),
        jitter=bool(cfg["data"].get("jitter", True)),
        horizontal_flip=bool(cfg["data"].get("horizontal_flip", True)),
        return_paths=temporal_enabled,
    )
    loader = DataLoader(
        ds,
        batch_size=int(cfg["data"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["data"]["num_workers"]),
        drop_last=True,
        pin_memory=False,  # Disable pin_memory to avoid Windows/WSL freezing issues
        persistent_workers=True if int(cfg["data"]["num_workers"]) > 0 else False,  # Keep workers alive
        prefetch_factor=2 if int(cfg["data"]["num_workers"]) > 0 else 2,  # Reduce prefetch to save memory
        collate_fn=collate_simple,
    )

    # Models
    G = ResnetGenerator(
        ngf=int(cfg["model"]["gen_channels"]),
        n_blocks=int(cfg["model"]["n_res_blocks"]),
        up_mode=str(cfg["model"].get("up_mode", "nearest")),
    )
    D = NLayerDiscriminator(
        n_layers=int(cfg["model"]["disc_n_layers"]),
        spectral_norm=bool(cfg["model"].get("spectral_norm", True)),
    )
    # choose layers by index in generator's sequential; select some evenly spaced points
    feat_layers = [int(i) for i in cfg["model"]["nce_layers"]]
    nce_dim = int(cfg["model"]["nce_dim"])
    nce_patches = int(cfg["model"]["nce_patches"])

    # To build PatchSampleMLP we need channel dims; we will probe once
    with torch.no_grad():
        probe = torch.randn(1, 3, cfg["data"]["image_size"], cfg["data"]["image_size"])  # type: ignore
        y, feats = G(probe, return_feats=True, feat_layers=feat_layers)
        in_channels = [f.shape[1] for f in feats]
    patch_mlp = PatchSampleMLP(in_channels=in_channels, out_dim=nce_dim)

    G.to(device)
    D.to(device)
    patch_mlp.to(device)

    # Losses and optimizers
    gan_loss = HingeGANLoss()
    nce_loss = PatchNCELoss(tau=float(cfg["loss"]["nce_tau"]))
    identity_loss = IdentityLoss(float(cfg["loss"].get("lambda_idt", 25.0)))
    luma_loss = LuminanceLoss(float(cfg["loss"].get("lambda_luma", 8.0)))
    tv_loss_fn = TVLoss(float(cfg["loss"].get("lambda_tv", 1.0e-5)))
    saturation_loss = SaturationLoss(
        float(cfg["loss"].get("lambda_sat", 1.0)),
        float(cfg["loss"].get("sat_target", 0.5)),
    )
    # Patch-wise color regularizer (optional)
    patch_color_cfg = dict(cfg["loss"].get("patch_color", {})) if "loss" in cfg else {}
    patch_color_enabled = bool(patch_color_cfg.get("enabled", True))
    if patch_color_enabled:
        patch_color_loss = PatchColorLoss(
            patch_size=int(patch_color_cfg.get("patch_size", 32)),
            stride=int(patch_color_cfg.get("stride", 32)),
            sat_weight=float(patch_color_cfg.get("sat_weight", 0.5)),
            sat_max_weight=float(patch_color_cfg.get("sat_max_weight", 0.5)),
            sat_target=float(patch_color_cfg.get("sat_target", 0.6)),
            chroma_weight=float(patch_color_cfg.get("chroma_weight", 1.0)),
            luma_weight=float(patch_color_cfg.get("luma_weight", 0.0)),
        )
        patch_color_loss.to(device)
    else:
        patch_color_loss = None

    # Highlight clipping (pixel-level) to prevent neon artifacts
    highlight_cfg = dict(cfg["loss"].get("highlight", {})) if "loss" in cfg else {}
    highlight_enabled = bool(highlight_cfg.get("enabled", True))
    if highlight_enabled:
        highlight_loss = ClippedHighlightsLoss(
            weight=float(highlight_cfg.get("weight", 0.2)),
            threshold=float(highlight_cfg.get("threshold", 0.985)),
        )
        highlight_loss.to(device)
    else:
        highlight_loss = None

    g_opt = optim.Adam(G.parameters(), lr=float(cfg["optim"]["lr"]), betas=tuple(cfg["optim"]["betas"]))
    pm_opt = optim.Adam(patch_mlp.parameters(), lr=float(cfg["optim"]["lr"]))
    d_opt = optim.Adam(D.parameters(), lr=float(cfg["optim"]["lr_disc"]), betas=tuple(cfg["optim"]["betas_disc"]))

    # Move loss functions to device
    identity_loss.to(device)
    luma_loss.to(device)
    tv_loss_fn.to(device)
    saturation_loss.to(device)

    epochs = int(cfg["optim"]["epochs"])
    log_interval = int(cfg["logging"]["log_interval"]) if "logging" in cfg and "log_interval" in cfg["logging"] else 100

    # Resume from checkpoint if provided
    start_epoch = 1
    step = 0
    if args.resume:
        resume_path = Path(args.resume).resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        print(f"🔄 Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        
        G.load_state_dict(checkpoint["G"])
        D.load_state_dict(checkpoint["D"])
        patch_mlp_state = checkpoint.get("MLP", {})
        if isinstance(patch_mlp_state, dict) and patch_mlp_state:
            current_state = patch_mlp.state_dict()
            matched = 0
            for key, value in patch_mlp_state.items():
                if key in current_state and current_state[key].shape == value.shape:
                    current_state[key] = value
                    matched += 1
            if matched > 0:
                patch_mlp.load_state_dict(current_state)
                if matched < len(patch_mlp_state):
                    print(f"⚠️  PatchSampleMLP: loaded {matched}/{len(patch_mlp_state)} matching tensors; others reinitialized.")
            else:
                print("⚠️  PatchSampleMLP checkpoint tensors incompatible; using fresh initialization.")
        else:
            print("⚠️  No PatchSampleMLP weights found in checkpoint; using freshly initialized weights.")
        
        # Try to load optimizer states if available
        if "g_opt" in checkpoint:
            try:
                g_opt.load_state_dict(checkpoint["g_opt"])
            except ValueError as e:
                print(f"⚠️  Skipping generator optimizer state: {e}")
        if "pm_opt" in checkpoint:
            try:
                pm_opt.load_state_dict(checkpoint["pm_opt"])
            except ValueError as e:
                print(f"⚠️  Skipping PatchMLP optimizer state: {e}")
        if "d_opt" in checkpoint:
            try:
                d_opt.load_state_dict(checkpoint["d_opt"])
            except ValueError as e:
                print(f"⚠️  Skipping discriminator optimizer state: {e}")
        
        # Extract epoch number from checkpoint filename or use config
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
        else:
            # Try to extract from filename, e.g., cut_epoch_10.pth -> start from epoch 11
            filename = resume_path.stem
            if "epoch_" in filename:
                try:
                    epoch_num = int(filename.split("epoch_")[1].split(".")[0])
                    start_epoch = epoch_num + 1
                except:
                    pass
        
        if "step" in checkpoint:
            step = checkpoint["step"]
        
        print(f"✅ Resumed: Starting from epoch {start_epoch}, step {step}")
    
    for epoch in range(start_epoch, epochs + 1):
        G.train(); D.train(); patch_mlp.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", unit="batch")
        for batch_idx, batch in enumerate(pbar):
            if temporal_enabled:
                sim_img, real_img, sim_path, real_path = batch
            else:
                sim_img, real_img = batch
                sim_path = None
                real_path = None
            sim_img = sim_img.to(device, non_blocking=True)
            real_img = real_img.to(device, non_blocking=True)
            src_img = sim_img if sim2real else real_img
            tgt_img = real_img if sim2real else sim_img

            # ============ BATCH VISUALIZATION POINT ============
            # To visualize a batch during training, uncomment the lines below:
            if batch_idx == 0:  # Visualize first batch of epoch
                with torch.no_grad():
                    fake_img_viz, _ = G(src_img, return_feats=False)
                    save_batch_images(src_img, fake_img_viz, tgt_img, 
                                     out_dir / "batch_viz", epoch, batch_idx)
            # ===================================================

            # ------------------ Train Discriminator ------------------
            with torch.no_grad():
                fake_tgt, _ = G(src_img, return_feats=False)
            d_opt.zero_grad(set_to_none=True)
            tgt_img_req = tgt_img.detach().requires_grad_(True)
            real_logits = D(tgt_img_req)
            fake_logits = D(fake_tgt)
            d_loss = gan_loss.d_loss(real_logits, fake_logits)
            # R1 regularization on real images
            r1 = r1_regularization(tgt_img_req, real_logits, gamma=float(cfg["loss"].get("r1_gamma", 10.0)))
            (d_loss + r1).backward()
            d_opt.step()

            # ------------------ Train Generator + PatchNCE ------------------
            g_opt.zero_grad(set_to_none=True)
            pm_opt.zero_grad(set_to_none=True)
            fake_tgt, feats_src = G(src_img, return_feats=True, feat_layers=feat_layers)
            fake_logits = D(fake_tgt)
            g_adv = gan_loss.g_loss(fake_logits)

            # Compute features on generated as targets (reuse fake_real)
            _, feats_tgt = G(fake_tgt, return_feats=True, feat_layers=feat_layers)
            # Sample the same patch positions for src and tgt (critical for CUT)
            src_samples, patch_ids = patch_mlp(feats_src, num_patches=nce_patches)
            tgt_samples, _ = patch_mlp(feats_tgt, num_patches=nce_patches, ids=patch_ids)
            loss_nce = nce_loss(src_samples, tgt_samples)

            # Identity-NCE on target-domain images to preserve colors/structure
            idt_real, feats_idt_in = G(tgt_img, return_feats=True, feat_layers=feat_layers)
            _,        feats_idt_out = G(idt_real,  return_feats=True, feat_layers=feat_layers)
            idt_src, id_ids = patch_mlp(feats_idt_in,  num_patches=nce_patches)
            idt_tgt, _ = patch_mlp(feats_idt_out, num_patches=nce_patches, ids=id_ids)
            lambda_nce_idt = float(cfg["loss"].get("lambda_nce_idt", 1.0))
            loss_nce_idt = nce_loss(idt_src, idt_tgt) * lambda_nce_idt

            # Additional losses: identity on target, luminance preservation vs input source, TV regularizer
            real_identity, _ = G(tgt_img, return_feats=False)
            loss_ident = identity_loss(tgt_img, real_identity)
            loss_luma = luma_loss(src_img, fake_tgt)
            loss_tv = tv_loss_fn(fake_tgt)
            loss_sat = saturation_loss(fake_tgt)
            if patch_color_loss is not None:
                loss_patch_color = patch_color_loss(src_img, fake_tgt)
            else:
                loss_patch_color = torch.tensor(0.0, device=device)
            if highlight_loss is not None:
                loss_highlight = highlight_loss(fake_tgt)
            else:
                loss_highlight = torch.tensor(0.0, device=device)

            # Temporal consistency (lightweight): encourage consecutive outputs to be close
            temp_loss = torch.tensor(0.0, device=device)
            if temporal_enabled and ((sim2real and isinstance(sim_path, list)) or ((not sim2real) and isinstance(real_path, list))):
                # For first item in batch (cheap): compute next frame loss
                try:
                    sp0 = (sim_path[0] if sim2real else real_path[0])
                    next_path = ds.get_neighbor(sp0, delta=int(temporal_cfg.get("delta", 1)))
                    img_next = Image.open(next_path).convert("RGB")
                    # Build a deterministic transform matching dataset (resize to square + toTensor + norm)
                    image_size = int(cfg["data"]["image_size"])
                    trans = T.Compose([
                        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
                        T.ToTensor(),
                        T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
                    ])
                    x_next = trans(img_next).unsqueeze(0).to(device)
                    fake_next, _ = G(x_next, return_feats=False)
                    temp_loss = torch.nn.functional.l1_loss(fake_tgt[:1], fake_next)
                    temp_loss = temp_loss * float(temporal_cfg.get("lambda_temp", 3.0))
                except Exception:
                    temp_loss = torch.tensor(0.0, device=device)

            # Weights for losses (configurable)
            lambda_nce = 1.0
            loss_g_total = g_adv + lambda_nce * loss_nce + loss_nce_idt + loss_ident + loss_luma + loss_tv + temp_loss + loss_sat + loss_patch_color + loss_highlight
            loss_g_total.backward()
            g_opt.step(); pm_opt.step()

            if step % log_interval == 0:
                pbar.set_postfix({
                    "d_loss": f"{d_loss.item():.3f}",
                    "g_adv": f"{g_adv.item():.3f}",
                    "nce": f"{loss_nce.item():.3f}",
                    "idt": f"{loss_ident.item():.3f}",
                    "nce_idt": f"{loss_nce_idt.item():.3f}",
                    "luma": f"{loss_luma.item():.3f}",
                    "tv": f"{loss_tv.item():.4f}",
                    "sat": f"{loss_sat.item():.3f}",
                    "pcolor": f"{loss_patch_color.item():.3f}",
                    "hl": f"{loss_highlight.item():.3f}",
                })
            step += 1

        # Save checkpoints every few epochs and generate visualizations
        save_interval = int(cfg["logging"].get("save_interval_epochs", 5))
        if (epoch % save_interval) == 0:
            ckpt_path = out_dir / f"cut_epoch_{epoch}.pth"
            torch.save({
                "G": G.state_dict(), 
                "D": D.state_dict(), 
                "MLP": patch_mlp.state_dict(),
                "g_opt": g_opt.state_dict(),
                "pm_opt": pm_opt.state_dict(),
                "d_opt": d_opt.state_dict(),
                "epoch": epoch,
                "step": step,
            }, ckpt_path)
            print(f"✅ Checkpoint saved: {ckpt_path}")
            
            # Generate visualization for this checkpoint
            try:
                visualize_checkpoint(ckpt_path, loader, device, feat_layers, cfg, out_dir, epoch)
            except Exception as e:
                print(f"⚠️  Visualization failed for epoch {epoch}: {e}")

    # Final save
    final_ckpt_path = out_dir / "cut_final.pth"
    torch.save({
        "G": G.state_dict(), 
        "D": D.state_dict(), 
        "MLP": patch_mlp.state_dict(),
        "g_opt": g_opt.state_dict(),
        "pm_opt": pm_opt.state_dict(),
        "d_opt": d_opt.state_dict(),
        "epoch": epochs,
        "step": step,
    }, final_ckpt_path)
    print(f"✅ Final checkpoint saved: {final_ckpt_path}")
    print(f"🎉 Training complete! Visualizations saved to {out_dir / 'visualizations'}")


if __name__ == "__main__":
    main()


