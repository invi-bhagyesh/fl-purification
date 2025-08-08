import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


def depthwise_adaptive_conv(img, kernel):
    """
    Efficient per-channel adaptive convolution.
    img:    [B,C,H,W]
    kernel: [B,C,K,K] (sum to 1 per-channel recommended)
    returns [B,C,H,W]
    """
    B, C, H, W = img.shape
    K = kernel.shape[-1]
    pad = K // 2

    # reflect pad
    x = F.pad(img, (pad, pad, pad, pad), mode='reflect')  # [B,C,H+2p,W+2p]

    # Grouped conv trick over batch*channels
    weight = kernel.view(B * C, 1, K, K)                  # [B*C,1,K,K]
    x = x.view(1, B * C, H + 2 * pad, W + 2 * pad)        # [1,B*C,H',W']

    out = F.conv2d(x, weight, bias=None, stride=1, padding=0, groups=B * C)
    out = out.view(B, C, H, W)
    return out


class AdaptiveKernelHypernet(nn.Module):
    """
    Hypernetwork that predicts per-channel kernels [B, C, K, K]
    Uses softplus + normalization (sum=1) per-channel kernel.
    """
    def __init__(self, kernel_size=5, input_channels=3, base_channels=32):
        super().__init__()
        self.kernel_size = kernel_size
        self.input_channels = input_channels

        c = base_channels
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, c, kernel_size=(3,1), padding=(1,0)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(c, c*2, kernel_size=(1,3), padding=(0,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(c*2, c*4, kernel_size=(3,1), padding=(1,0)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(c*4, c*8, kernel_size=(1,3), padding=(0,1)),
            nn.ReLU(inplace=True),

            nn.Conv2d(c*8, c*8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((kernel_size, kernel_size)),
        )
        self.head = nn.Conv2d(c*8, input_channels, kernel_size=1)
        self.softplus = nn.Softplus(beta=1.0, threshold=20.0)

        # Bias init toward near-uniform kernel for stable start
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, 0.0)

    def forward(self, image):
        x = self.features(image)              # [B, C_feat, K, K]
        w = self.head(x)                      # [B, C_in,  K, K]
        w = self.softplus(w)                  # non-negative
        # normalize per-channel kernel to sum 1
        w_flat = w.flatten(2)                 # [B, C, K*K]
        w_sum = w_flat.sum(-1, keepdim=True)  # [B, C, 1]
        w = (w_flat / (w_sum + 1e-8)).view_as(w)
        return w


class AdaptiveLaplacianPyramid(nn.Module):
    """
    Builds an adaptive Laplacian pyramid using per-channel kernels.
    """
    def __init__(self, num_levels=5, kernel_size=5, input_channels=3):
        super().__init__()
        self.num_levels = num_levels
        self.kernel_size = kernel_size
        self.hypernet = AdaptiveKernelHypernet(kernel_size=kernel_size, input_channels=input_channels)

    def build(self, x):
        """
        Returns:
          levels: list of Laplacian levels (num_levels-1) + base
          kernel: [B,C,K,K]
        """
        B, C, H, W = x.shape
        kernel = self.hypernet(x)  # [B,C,K,K] per-channel

        levels = []
        current = x
        for i in range(self.num_levels - 1):
            # blur -> downsample
            blur = depthwise_adaptive_conv(current, kernel)
            down = blur[:, :, ::2, ::2]

            # upsample -> blur
            up = F.interpolate(down, size=current.shape[2:], mode='bilinear', align_corners=False)
            up = depthwise_adaptive_conv(up, kernel)

            lap = current - up
            levels.append(lap)
            current = down

        levels.append(current)  # base
        return levels, kernel


class PyramidSkipAdapter(nn.Module):
    """
    Adapts 3-channel Laplacian features to match encoder skip channels.
    Uses 1x1 conv; fuse via addition for decoder compatibility.
    """
    def __init__(self, out_channels):
        super().__init__()
        self.proj = nn.Conv2d(3, out_channels, kernel_size=1)

    def forward(self, x):
        return self.proj(x)


class SMPPyramidDenoiser(nn.Module):
    """
    SMP U-Net + Adaptive Laplacian Pyramid as learned skip boosters.
    - Encoder/Decoder from SMP (unchanged).
    - Pyramid is auxiliary; fused into skips via additive 1x1 projections.
    - Per-channel kernels [B,C,K,K] with softplus-normalization.
    """
    def __init__(
        self,
        encoder_name='resnet34',
        encoder_weights='imagenet',
        decoder_channels=(256, 128, 64, 32, 16),
        num_pyramid_levels=5,
        kernel_size=5,
        fuse_mode='add',   # 'add' is default; 'cat' would require decoder channel changes.
    ):
        super().__init__()
        assert fuse_mode in ('add',), "This implementation keeps decoder unchanged; use 'add'."

        self.v_noise = 0.0
        self.fuse_mode = fuse_mode

        # SMP encoder/decoder
        depth = len(decoder_channels)
        self.encoder = smp.encoders.get_encoder(
            encoder_name, in_channels=3, depth=depth, weights=encoder_weights
        )
        encoder_channels = self.encoder.out_channels  # list length depth+1

        from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            n_blocks=depth,
            attention_type=None
        )

        self.segmentation_head = nn.Conv2d(decoder_channels[-1], 3, kernel_size=3, padding=1)

        # Pyramid
        self.pyramid = AdaptiveLaplacianPyramid(
            num_levels=num_pyramid_levels, kernel_size=kernel_size, input_channels=3
        )

        # Adapters for skips (exclude stage 0 which is the input resolution stem)
        self.pyr_adapters = nn.ModuleList([
            PyramidSkipAdapter(ch) for ch in encoder_channels[1:]
        ])

    def forward(self, noisy):
        if self.v_noise and self.v_noise > 0.0:
            noisy = noisy + self.v_noise * torch.randn_like(noisy)

        # SMP encoder features: list [x0, x1, ..., xD] shallow->deep
        enc_feats = self.encoder(noisy)

        # Build adaptive Laplacian pyramid from input
        pyr_levels, adaptive_kernel = self.pyramid.build(noisy)  # [L levels], [B,C,K,K]
        # pyr_levels: [lap1, lap2, ..., base], len=L (num_pyramid_levels)

        # Fuse pyramid features into encoder skips (additive)
        fused_feats = [enc_feats[0]]  # keep stage 0 as is
        for i, feat in enumerate(enc_feats[1:]):
            # choose pyramid level close to this feature's resolution
            li = min(i, len(pyr_levels) - 1)  # clamp
            lap = pyr_levels[li]
            lap_resized = F.interpolate(lap, size=feat.shape[2:], mode='bilinear', align_corners=False)
            lap_proj = self.pyr_adapters[i](lap_resized)
            fused_feats.append(feat + lap_proj)

        # Decode
        dec = self.decoder(fused_feats)
        residual = self.segmentation_head(dec)

        if residual.shape[2:] != noisy.shape[2:]:
            residual = F.interpolate(residual, size=noisy.shape[2:], mode='bilinear', align_corners=False)

        denoised = noisy + residual
        return denoised, adaptive_kernel
