import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Your reformer model ---
reformer = AdaptiveLaplacianPyramidUNet(
    encoder_name='resnet34',
    encoder_weights='imagenet',
    decoder_channels=(256, 128, 64, 32, 16),
    num_pyramid_levels=5,
    kernel_size=5
).to(device)

# --- Loss and Optimizer ---
criterion = nn.MSELoss()
optimizer = optim.Adam(reformer.parameters(), lr=1e-3)  # or your favorite optimizer

# --- Training Loop ---
epochs = 20  # Choose your number of training epochs

for epoch in range(epochs):
    reformer.train()
    running_loss = 0
    for batch in train_loader:
        images = batch[0].to(device, dtype=torch.float)
        # If using actual noisy/attacked images, replace images as needed
        noise_std = 0.2  # You can experiment with this value
        noisy_images = images + noise_std * torch.randn_like(images)
        noisy_images = torch.clamp(noisy_images, 0, 1)
        clean_images = images

        optimizer.zero_grad()
        outputs, _ = reformer(noisy_images)
        loss = criterion(outputs, clean_images)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {running_loss/len(train_loader.dataset):.6f}')

    # --- (Optional) Validation ---
    reformer.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            images = batch[0].to(device, dtype=torch.float)
            outputs, _ = reformer(images)
            loss = criterion(outputs, images)
            val_loss += loss.item() * images.size(0)
        print(f'Validation Loss: {val_loss/len(val_loader.dataset):.6f}')

