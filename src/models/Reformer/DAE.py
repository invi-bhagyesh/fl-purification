import torch
import torch.nn as nn
import torchvision.transforms.functional as Fnn

class DenoisingAutoEncoder(nn.Module):
    def __init__(self, v_noise=0.0, reg_strength=0.0):
        super().__init__()
        self.v_noise = v_noise
        self.reg_strength = reg_strength
        
        # Encoder: (3,28,28) → (128,4,4) → flatten → (512)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),      # 28x28 → 14x14
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),     # 14x14 → 7x7
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,128, 3, stride=2, padding=1),     # 7x7 → 4x4
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(128*4*4, 512),
            nn.ReLU(inplace=True),
        )
        # Decoder: (512) → (128,4,4) → upsampling to (3,28,28)
        self.decoder = nn.Sequential(
            nn.Linear(512, 128*4*4),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), # 4x4 → 8x8
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), # 8x8 → 16x16
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),  # 16x16 → 32x32
            nn.Sigmoid()
        )
        
    def forward(self, x):
        z = self.encoder(x)
        # Add noise to bottleneck if enabled
        if self.v_noise > 0.0:
            noise = self.v_noise * torch.randn_like(z)
            z = z + noise
        out = self.decoder(z)
        # Crop to (28,28) if shape != (3,28,28)
        out = Fnn.center_crop(out, [28, 28])
        return out

    def get_l2_loss(self):
        l2_loss = sum(torch.sum(param ** 2) for param in self.parameters())
        return self.reg_strength * l2_loss