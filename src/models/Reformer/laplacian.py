import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Reformer.SMP import AdaptiveLaplacianPyramid
from models.Reformer.SMP import PyramidSkipAdapter

class SimpleEncoder(nn.Module):
    def __init__(self, in_channels=3, channels=[32, 64, 128, 256]):
        
        super().__init__()
        layers = []
        prev_ch = in_channels
        self.skips = []
        for ch in channels:
            layers.append(nn.Conv2d(prev_ch, ch, 3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(ch, ch, 3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(2))
            prev_ch = ch
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        feats = []
        cur = x
        idx = 0
        for layer in self.encoder:
            cur = layer(cur)
            # Save outputs after each MaxPool layer as skip features
            if isinstance(layer, nn.MaxPool2d):
                feats.append(cur)
        return feats  # list of features at multiple scales


class SimpleDecoder(nn.Module):
    def __init__(self, channels=[256, 128, 64, 32], out_channels=3):
        super().__init__()
        self.channels = channels
        self.num_blocks = len(channels) - 1
        self.upsamples = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            for _ in range(self.num_blocks)
        ])
        self.convs = nn.ModuleList([
            nn.Conv2d(channels[i], channels[i+1], 3, padding=1)
            for i in range(self.num_blocks)
        ])
        self.activations = nn.ModuleList([
            nn.ReLU(inplace=True)
            for _ in range(self.num_blocks)
        ])
        self.out_conv = nn.Conv2d(channels[-1], out_channels, 3, padding=1)

    def forward(self, feats):
        x = feats[-1]
        for i in range(self.num_blocks):
            x = self.upsamples[i](x)
            x = self.convs[i](x)
            x = self.activations[i](x)
            skip_idx = len(feats) - 2 - i
            if skip_idx >= 0:
                skip = feats[skip_idx]
                if skip.shape == x.shape:
                    x = x + skip
                else:
                    x = x + F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
        out = self.out_conv(x)
        return out



class SimplePyramidDenoiser(nn.Module):
    def __init__(self, num_pyramid_levels=5, kernel_size=5, input_channels=3):
        super().__init__()
        self.v_noise = 0.0

        self.encoder = SimpleEncoder(in_channels=input_channels)
        # Output channels match encoder last channels
        self.decoder = SimpleDecoder(channels=[256, 128, 64, 32], out_channels=input_channels)

        self.pyramid = AdaptiveLaplacianPyramid(
            num_levels=num_pyramid_levels, kernel_size=kernel_size, input_channels=input_channels
        )

        # Adapters for skips (length == number of encoder skip features)
        encoder_channels = [32, 64, 128, 256]
        self.pyr_adapters = nn.ModuleList([
            PyramidSkipAdapter(ch) for ch in encoder_channels
        ])

    def forward(self, noisy):
        if self.v_noise > 0.0:
            noisy = noisy + self.v_noise * torch.randn_like(noisy)

        enc_feats = self.encoder(noisy)  # List of features [feat1, feat2, feat3, feat4]

        pyr_levels, adaptive_kernel = self.pyramid.build(noisy)  # Laplacian levels

        fused_feats = []
        for i, feat in enumerate(enc_feats):
            li = min(i, len(pyr_levels) - 1)
            lap = pyr_levels[li]
            lap_resized = F.interpolate(lap, size=feat.shape[2:], mode='bilinear', align_corners=False)
            lap_proj = self.pyr_adapters[i](lap_resized)
            fused_feats.append(feat + lap_proj)

        residual = self.decoder(fused_feats)
        if residual.shape[2:] != noisy.shape[2:]:
            residual = F.interpolate(residual, size=noisy.shape[2:], mode='bilinear', align_corners=False)

        denoised = noisy + residual
        return denoised, adaptive_kernel
