from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResnetBlock(nn.Module):
    def __init__(self, dim: int, norm_layer=nn.InstanceNorm2d) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, bias=False),
            norm_layer(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, bias=False),
            norm_layer(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9, norm_layer=nn.InstanceNorm2d, up_mode: str = "nearest"):
        super().__init__()
        assert n_blocks >= 0
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, bias=False),
            norm_layer(ngf),
            nn.ReLU(True),
        ]

        # downsampling
        n_downsampling = 2
        in_dim = ngf
        feats_dims: List[int] = [in_dim]
        for i in range(n_downsampling):
            out_dim = in_dim * 2
            model += [
                nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(out_dim),
                nn.ReLU(True),
            ]
            in_dim = out_dim
            feats_dims.append(in_dim)

        # resnet blocks
        for i in range(n_blocks):
            model += [ResnetBlock(in_dim, norm_layer=norm_layer)]
        feats_dims.extend([in_dim] * n_blocks)

        # upsampling: replace transposed conv with upsample + conv to reduce checkerboard artifacts
        for i in range(n_downsampling):
            out_dim = in_dim // 2
            model += [
                nn.Upsample(scale_factor=2, mode=up_mode, align_corners=False if up_mode == "bilinear" else None),
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_dim, out_dim, kernel_size=3, bias=False),
                norm_layer(out_dim),
                nn.ReLU(True),
            ]
            in_dim = out_dim
            feats_dims.append(in_dim)

        model += [nn.ReflectionPad2d(3), nn.Conv2d(in_dim, output_nc, kernel_size=7), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor, return_feats: bool = False, feat_layers: List[int] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if not return_feats:
            return self.model(x), []
        feats: List[torch.Tensor] = []
        out = x
        for i, layer in enumerate(self.model):
            out = layer(out)
            if feat_layers is not None and i in feat_layers:
                feats.append(out)
        return out, feats


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d, spectral_norm: bool = True):
        super().__init__()
        kw = 4
        padw = 1
        conv0 = nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)
        if spectral_norm:
            conv0 = nn.utils.spectral_norm(conv0)
        sequence = [conv0, nn.LeakyReLU(0.2, True)]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            conv = nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False)
            if spectral_norm:
                conv = nn.utils.spectral_norm(conv)
            sequence += [conv, norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        conv_last = nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False)
        if spectral_norm:
            conv_last = nn.utils.spectral_norm(conv_last)
        sequence += [conv_last, norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]

        conv_out = nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        if spectral_norm:
            conv_out = nn.utils.spectral_norm(conv_out)
        sequence += [conv_out]
        self.model = nn.Sequential(*sequence)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class PatchSampleMLP(nn.Module):
    def __init__(self, in_channels: List[int], out_dim: int = 256) -> None:
        super().__init__()
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(c, out_dim),
                nn.ReLU(True),
                nn.Linear(out_dim, out_dim),
            ) for c in in_channels
        ])

    def forward(self, feats: List[torch.Tensor], num_patches: int = 256, ids: List[torch.Tensor] = None) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        samples: List[torch.Tensor] = []
        out_ids: List[torch.Tensor] = []
        for i, (feat, mlp) in enumerate(zip(feats, self.mlps)):
            b, c, h, w = feat.shape
            feat_flat = feat.permute(0, 2, 3, 1).reshape(b, h * w, c)  # [B, HW, C]
            if num_patches > 0:
                if ids is not None and i < len(ids) and ids[i] is not None:
                    sel = ids[i]
                else:
                    sel = torch.randperm(h * w, device=feat.device)[: min(num_patches, h * w)]
                feat_sel = feat_flat[:, sel, :]  # [B, P, C]
            else:
                feat_sel = feat_flat
                sel = torch.arange(h * w, device=feat.device)
            out = mlp(feat_sel)  # [B, P, D]
            samples.append(out)
            out_ids.append(sel)
        return samples, out_ids


