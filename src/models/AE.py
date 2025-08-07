# Detector
import torchvision.transforms.functional as F
import torch.nn as nn
import torch

class SimpleAutoencoder(nn.Module):
    def __init__(self, image_shape=(3, 28, 28)):
        """
        Initializes the autoencoder.
        
        Args:
            image_shape (tuple): Input image shape in the format (channels, height, width).
        """
        super().__init__()
        self.image_shape = image_shape
        # Final output should match the original image spatial dimensions.
        self.out_size = image_shape[1:]
        
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(image_shape[0], 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, *image_shape)
            conv_out = self.encoder_conv(dummy)
            self.flattened_dim = conv_out.view(1, -1).shape[1]

        self.encoder = nn.Sequential(
            self.encoder_conv,
            nn.Flatten(),
            nn.Linear(self.flattened_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.Linear(512, self.flattened_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Unflatten(1, conv_out.shape[1:]),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, image_shape[0], 4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        # Crop output to match the target image dimensions
        out = F.center_crop(out, self.out_size)
        return out