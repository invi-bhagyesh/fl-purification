import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class AdaptiveKernelHypernet(nn.Module):
    """
    Hypernetwork that predicts optimal kernels for Laplacian pyramid decomposition
    Based on HipyrNet architecture but adapted for denoising
    """
    def __init__(self, kernel_size=5, input_channels=3):
        super(AdaptiveKernelHypernet, self).__init__()
        
        self.kernel_size = kernel_size
        self.kernel_params = kernel_size * kernel_size
        
        # Feature extraction with irregular kernels (inspired by HipyrNet)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28 -> 14x14
        
            nn.Conv2d(16, 32, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14 -> 7x7
        
            nn.Conv2d(32, 64, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 7x7 -> 3x3
        
            nn.Conv2d(64, 128, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.ReLU(inplace=True),
            # LAST POOLING REMOVED
        
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((self.kernel_size, self.kernel_size)),
            nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0),
        )

        
        # Normalization layer to ensure kernel sums appropriately
        self.kernel_norm = nn.Softmax(dim=-1)
        
    def forward(self, image):
        """
        Generate adaptive kernel for input image
        Args:
            image: Input image [B, C, H, W]
        Returns:
            kernel: Adaptive kernel [B, 1, kernel_size, kernel_size]
        """
        # Extract features and generate kernel
        x = self.conv_layers(image)  # [B, 1, kernel_size, kernel_size]
        
        # Normalize kernel to ensure proper convolution properties
        batch_size = x.shape[0]
        kernel_flat = x.view(batch_size, -1)
        kernel_normalized = self.kernel_norm(kernel_flat)
        kernel = kernel_normalized.view(batch_size, 1, self.kernel_size, self.kernel_size)
        
        return kernel

class AdaptiveLaplacianPyramidEncoder(nn.Module):
    """
    Laplacian Pyramid Encoder with learnable kernels from hypernetwork
    """
    def __init__(self, num_levels=5, kernel_size=5):
        super(AdaptiveLaplacianPyramidEncoder, self).__init__()
        
        self.num_levels = num_levels
        self.kernel_size = kernel_size
        
        # Hypernetwork for adaptive kernel prediction
        self.hypernet = AdaptiveKernelHypernet(kernel_size=kernel_size)
        
    def _adaptive_conv_gauss(self, img, kernel):
        """Apply adaptive convolution with predicted kernel"""
        batch_size, channels = img.shape[0], img.shape[1]
        
        # Expand kernel to match input channels
        kernel_expanded = kernel.repeat(1, channels, 1, 1)  # [B, C, K, K]
        
        # Apply padding
        pad_size = self.kernel_size // 2
        img_padded = F.pad(img, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
        
        # Apply grouped convolution for each sample in batch
        output_list = []
        for i in range(batch_size):
            sample_img = img_padded[i:i+1]  # [1, C, H, W]
            sample_kernel = kernel_expanded[i:i+1]  # [1, C, K, K]
            sample_kernel = sample_kernel.view(channels, 1, self.kernel_size, self.kernel_size)
            
            conv_result = F.conv2d(sample_img, sample_kernel, groups=channels)
            output_list.append(conv_result)
        
        return torch.cat(output_list, dim=0)
    
    def _downsample(self, x):
        """Downsample by factor of 2"""
        return x[:, :, ::2, ::2]
    
    def _upsample(self, x, target_size, kernel):
        """Upsample to target size with adaptive kernel"""
        upsampled = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return self._adaptive_conv_gauss(upsampled, kernel)
    
    def forward(self, x):
        """
        Decompose image into adaptive Laplacian pyramid
        """
        # Generate adaptive kernel for this specific image
        adaptive_kernel = self.hypernet(x)
        
        pyramid_levels = []
        current = x
        
        # Create pyramid levels using adaptive kernel
        for i in range(self.num_levels - 1):
            # Apply adaptive Gaussian filtering
            gaussian = self._adaptive_conv_gauss(current, adaptive_kernel)
            downsampled = self._downsample(gaussian)
            
            # Create Laplacian level (detail) using adaptive upsampling
            upsampled = self._upsample(downsampled, current.shape[2:], adaptive_kernel)
            detail = current - upsampled
            pyramid_levels.append(detail)
            
            # Move to next level
            current = downsampled
        
        # Add base level
        pyramid_levels.append(current)
        
        return pyramid_levels, adaptive_kernel

class PyramidToEncoderAdapter(nn.Module):
    """Adapts pyramid features to match encoder output channels"""
    def __init__(self, pyramid_channels=3, target_channels=64):
        super(PyramidToEncoderAdapter, self).__init__()
        
        self.adapter = nn.Sequential(
            nn.Conv2d(pyramid_channels, target_channels//2, 3, padding=1),
            nn.BatchNorm2d(target_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(target_channels//2, target_channels, 3, padding=1),
            nn.BatchNorm2d(target_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.adapter(x)

class AdaptiveLaplacianPyramidUNet(nn.Module):
    """
    Adaptive Laplacian Pyramid + U-Net with Hypernetwork-predicted kernels
    """
    def __init__(self, 
                 encoder_name='resnet34',
                 encoder_weights='imagenet',
                 decoder_channels=(256, 128, 64, 32, 16),
                 num_pyramid_levels=5,
                 kernel_size=5):
        super(AdaptiveLaplacianPyramidUNet, self).__init__()
        
        self.num_pyramid_levels = num_pyramid_levels
        self.v_noise = None

        # Adaptive Laplacian Pyramid Encoder with Hypernetwork
        self.pyramid_encoder = AdaptiveLaplacianPyramidEncoder(
            num_levels=num_pyramid_levels, 
            kernel_size=kernel_size
        )
        
        # Get encoder information
        temp_encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=3,
            depth=5,
            weights=encoder_weights
        )
        encoder_channels = temp_encoder.out_channels
        
        print(f"Encoder channels: {encoder_channels}")
        print(f"Decoder channels: {decoder_channels}")
        
        # U-Net decoder
        from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
        
        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            n_blocks=len(decoder_channels),
            #use_batchnorm=True,
            #center=False,
            attention_type=None
        )
        
        # Segmentation head
        self.segmentation_head = nn.Conv2d(
            decoder_channels[-1], 
            3,
            kernel_size=3, 
            padding=1
        )
        
        # Feature adapters
        self.feature_adapters = nn.ModuleList()
        for target_channels in encoder_channels[1:]:
            adapter = PyramidToEncoderAdapter(3, target_channels)
            self.feature_adapters.append(adapter)
        
        print(f"Created {len(self.feature_adapters)} feature adapters")
        print(f"Using adaptive kernels of size {kernel_size}x{kernel_size}")
    
    # def forward(self, x):
    #     z = self.encoder(x)
    #     # Add noise to bottleneck if enable
    #     if self.v_noise > 0.0:
    #         noise = self.v_noise * torch.randn_like(z)
    #         z = z + noise
    #     out = self.decoder(z)
    #     # Crop to (28,28) if shape != (3,28,28)
    #     out = F.nn.center_crop(out, [28, 28]) # check this once 
    #     return out

    def forward(self, noisy_image):
        # Inject input noise if enabled
        if hasattr(self, "v_noise") and self.v_noise > 0.0:
            noise = self.v_noise * torch.randn_like(noisy_image)
            noisy_image = noisy_image + noise

        # Get adaptive pyramid decomposition
        pyramid, adaptive_kernel = self.pyramid_encoder(noisy_image)

        # Convert pyramid to encoder-like features
        features = [noisy_image]
        for i, pyramid_level in enumerate(pyramid):
            if i < len(self.feature_adapters):
                adapted_features = self.feature_adapters[i](pyramid_level)
                features.append(adapted_features)

        # Decoder and segmentation
        decoder_output = self.decoder(features)
        denoised_features = self.segmentation_head(decoder_output)

        # Resize if needed
        if denoised_features.shape[2:] != noisy_image.shape[2:]:
            denoised_features = F.interpolate(
                denoised_features, size=noisy_image.shape[2:], mode='bilinear', align_corners=False
            )

        output = noisy_image + denoised_features
        return output, adaptive_kernel
