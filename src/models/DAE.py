# Reformer
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

class DenoisingAutoEncoder(nn.Module):
    def __init__(self, image_shape, structure, v_noise=0.1, activation="relu", model_dir="./defensive_models/", reg_strength=0.0):
        super(DenoisingAutoEncoder, self).__init__()
        self.image_shape = image_shape
        self.model_dir = model_dir
        self.v_noise = v_noise
        self.reg_strength = reg_strength

        act_fn = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}[activation]

        encoder_layers = []
        decoder_layers = []

        # Encoder construction
        in_ch = image_shape[0]
        for layer in structure:
            if isinstance(layer, int):
                encoder_layers.append(nn.Conv2d(in_ch, layer, kernel_size=3, stride=1, padding=1))
                encoder_layers.append(act_fn())
                in_ch = layer
            elif layer == "max":
                encoder_layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            elif layer == "average":
                encoder_layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
            else:
                raise ValueError(f'Unknown layer type: {layer}')

        # Decoder construction (reverse structure)
        rev_structure = list(reversed(structure))
        decoder_in_ch = in_ch
        for layer in rev_structure:
            if isinstance(layer, int):
                decoder_layers.append(nn.Conv2d(decoder_in_ch, layer, kernel_size=3, stride=1, padding=1))
                decoder_layers.append(act_fn())
                decoder_in_ch = layer
            elif layer == "max" or layer == "average":
                decoder_layers.append(nn.Upsample(scale_factor=2, mode='nearest'))

        # Final "decoded" layer matches input channels and uses sigmoid
        decoder_layers.append(nn.Conv2d(decoder_in_ch, image_shape[0], kernel_size=3, stride=1, padding=1))
        decoder_layers.append(nn.Sigmoid())

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        # Add noise to bottleneck if enabled
        if self.v_noise > 0.0:
            noise = self.v_noise * torch.randn_like(encoded)
            encoded = encoded + noise
        decoded = self.decoder(encoded)
        return decoded

    def add_noise(self, x):
        if self.v_noise > 0.0:
            noise = self.v_noise * torch.randn_like(x)
            x_noisy = x + noise
            x_noisy = torch.clamp(x_noisy, 0.0, 1.0)
            return x_noisy
        else:
            return x

    def get_l2_loss(self):
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += torch.sum(param ** 2)
        return self.reg_strength * l2_loss

    def save(self, archive_name):
        os.makedirs(self.model_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(self.model_dir, archive_name))

    def load(self, archive_name, model_dir=None):
        if model_dir is None:
            model_dir = self.model_dir
        self.load_state_dict(torch.load(os.path.join(model_dir, archive_name)))