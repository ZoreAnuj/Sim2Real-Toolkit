from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd


class HingeGANLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def d_loss(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        loss_real = F.relu(1.0 - real_logits).mean()
        loss_fake = F.relu(1.0 + fake_logits).mean()
        return 0.5 * (loss_real + loss_fake)

    def g_loss(self, fake_logits: torch.Tensor) -> torch.Tensor:
        return -fake_logits.mean()


class PatchNCELoss(nn.Module):
    def __init__(self, tau: float = 0.07) -> None:
        super().__init__()
        self.tau = tau

    def forward(self, src_feats: List[torch.Tensor], tgt_feats: List[torch.Tensor]) -> torch.Tensor:
        # src_feats and tgt_feats are lists of [B, P, D]
        total = 0.0
        cnt = 0
        for s, t in zip(src_feats, tgt_feats):
            b, p, d = s.shape
            s = F.normalize(s, dim=-1)
            t = F.normalize(t, dim=-1)
            # positives: same position, negatives: all others in batch and patches
            logits = torch.matmul(s.view(b * p, d), t.view(b * p, d).t()) / self.tau  # [BP, BP]
            labels = torch.arange(b * p, device=s.device)
            loss = F.cross_entropy(logits, labels)
            total = total + loss
            cnt += 1
        return total / max(cnt, 1)


class IdentityLoss(nn.Module):
    def __init__(self, weight: float = 10.0) -> None:
        super().__init__()
        self.weight = weight
        self.l1 = nn.L1Loss()

    def forward(self, real_img: torch.Tensor, idt_out: torch.Tensor) -> torch.Tensor:
        return self.weight * self.l1(real_img, idt_out)


class LuminanceLoss(nn.Module):
    def __init__(self, weight: float = 5.0) -> None:
        super().__init__()
        self.weight = weight

    @staticmethod
    def rgb_to_luma(x: torch.Tensor) -> torch.Tensor:
        # x in [-1,1]; convert to [0,1]
        y = (x + 1.0) * 0.5
        r, g, b = y[:, 0:1], y[:, 1:2], y[:, 2:3]
        # BT.601 luma approximation
        return 0.299 * r + 0.587 * g + 0.114 * b

    def forward(self, src_img: torch.Tensor, out_img: torch.Tensor) -> torch.Tensor:
        y_src = self.rgb_to_luma(src_img)
        y_out = self.rgb_to_luma(out_img)
        return self.weight * F.l1_loss(y_src, y_out)


class TVLoss(nn.Module):
    def __init__(self, weight: float = 1e-5) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        loss_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
        loss_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
        return self.weight * (loss_h + loss_w)


class SaturationLoss(nn.Module):
    """
    Penalize oversaturated outputs to preserve natural color balance.
    
    Computes saturation as: S = (max(R,G,B) - min(R,G,B)) / (max(R,G,B) + 1e-8)
    Then penalizes when saturation exceeds a threshold.
    """
    def __init__(self, weight: float = 0.5, target_sat: float = 0.5) -> None:
        super().__init__()
        self.weight = weight
        self.target_sat = target_sat  # target saturation level (0-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x in [-1, 1]; convert to [0, 1]
        x_norm = (x + 1.0) * 0.5
        
        # Extract RGB channels
        r, g, b = x_norm[:, 0:1], x_norm[:, 1:2], x_norm[:, 2:3]
        
        # Compute max and min
        max_ch = torch.max(torch.max(r, g), b)
        min_ch = torch.min(torch.min(r, g), b)
        
        # Compute saturation: (max - min) / (max + eps)
        sat = (max_ch - min_ch) / (max_ch + 1e-8)
        
        # Penalize saturation that exceeds target
        excess_sat = torch.clamp(sat - self.target_sat, min=0.0)
        loss = (excess_sat ** 2).mean()
        
        return self.weight * loss


def r1_regularization(real_img: torch.Tensor, real_logits: torch.Tensor, gamma: float = 10.0) -> torch.Tensor:
    grad_real = autograd.grad(outputs=real_logits.sum(), inputs=real_img, create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_penalty = grad_real.view(grad_real.shape[0], -1).pow(2).sum(dim=1).mean()
    return 0.5 * gamma * grad_penalty


class PatchColorLoss(nn.Module):
    """
    Patch-wise color regularization combining:
      - Saturation guard: penalize patches whose mean saturation exceeds a target
      - Chroma consistency: align local chroma (U,V) between source and output
      - Optional luma consistency: align local luminance per patch
    """
    def __init__(
        self,
        patch_size: int = 32,
        stride: int = None,
        sat_weight: float = 0.5,
        sat_max_weight: float = 0.5,
        sat_target: float = 0.6,
        chroma_weight: float = 1.0,
        luma_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch = patch_size
        self.stride = stride if stride is not None else patch_size
        self.sat_weight = sat_weight
        self.sat_max_weight = sat_max_weight
        self.sat_target = sat_target
        self.chroma_weight = chroma_weight
        self.luma_weight = luma_weight

    @staticmethod
    def rgb_to_yuv(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        u = 0.492 * (x[:, 2:3] - y)
        v = 0.877 * (x[:, 0:1] - y)
        return y, u, v

    @staticmethod
    def saturation(x: torch.Tensor) -> torch.Tensor:
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        max_ch = torch.max(torch.max(r, g), b)
        min_ch = torch.min(torch.min(r, g), b)
        sat = (max_ch - min_ch) / (max_ch + 1e-8)
        return sat

    def forward(self, src_img: torch.Tensor, out_img: torch.Tensor) -> torch.Tensor:
        # Convert to [0,1]
        src = (src_img + 1.0) * 0.5
        out = (out_img + 1.0) * 0.5

        # Patch-wise pooling helper
        def pool_mean(x: torch.Tensor) -> torch.Tensor:
            return F.avg_pool2d(x, kernel_size=self.patch, stride=self.stride)

        def pool_max(x: torch.Tensor) -> torch.Tensor:
            return F.max_pool2d(x, kernel_size=self.patch, stride=self.stride)

        # Saturation guard per patch on output
        sat_out = self.saturation(out)
        sat_mean = pool_mean(sat_out)
        sat_excess_mean = torch.clamp(sat_mean - self.sat_target, min=0.0)
        loss_sat_mean = (sat_excess_mean ** 2).mean()
        # Penalize extreme local saturation via max pooling (catches tiny neon blobs)
        sat_max = pool_max(sat_out)
        sat_excess_max = torch.clamp(sat_max - (self.sat_target + 0.05), min=0.0)
        loss_sat_max = (sat_excess_max ** 2).mean()

        # Chroma consistency between src and out (match U,V per patch)
        y_s, u_s, v_s = self.rgb_to_yuv(src)
        y_o, u_o, v_o = self.rgb_to_yuv(out)
        u_s_p, v_s_p = pool_mean(u_s), pool_mean(v_s)
        u_o_p, v_o_p = pool_mean(u_o), pool_mean(v_o)
        loss_chroma = (u_s_p - u_o_p).abs().mean() + (v_s_p - v_o_p).abs().mean()

        # Optional patch-wise luminance alignment
        if self.luma_weight > 0.0:
            y_s_p, y_o_p = pool_mean(y_s), pool_mean(y_o)
            loss_luma = (y_s_p - y_o_p).abs().mean()
        else:
            loss_luma = out.mean() * 0.0

        return (
            self.sat_weight * loss_sat_mean
            + self.sat_max_weight * loss_sat_max
            + self.chroma_weight * loss_chroma
            + self.luma_weight * loss_luma
        )


class ClippedHighlightsLoss(nn.Module):
    """Penalize pixel intensities that approach 1.0 (in [0,1] space)."""
    def __init__(self, weight: float = 0.2, threshold: float = 0.98) -> None:
        super().__init__()
        self.weight = weight
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = (x + 1.0) * 0.5
        excess = torch.clamp(y - self.threshold, min=0.0)
        return self.weight * (excess ** 2).mean()