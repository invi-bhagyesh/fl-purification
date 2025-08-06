import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO
from utils.utils import batch_psnr_ssim
from tqdm import tqdm
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_hypernet(model, train_loader, val_loader, device, num_epochs=20, config=None):
    """Enhanced hypernet training based on Hypernet_train.py"""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        total_psnr = 0.0
        total_ssim = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for images, pert_labels, true_labels in progress_bar:
            images = images.to(device, dtype=torch.float)
            # If using actual noisy/attacked images, replace images as needed
            noise_std = 0.2  # You can experiment with this value
            noisy_images = images + noise_std * torch.randn_like(images)
            noisy_images = torch.clamp(noisy_images, 0, 1)
            clean_images = images

            optimizer.zero_grad()
            outputs, _ = model(noisy_images)
            loss = criterion(outputs, clean_images)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            
            # Calculate image quality metrics
            with torch.no_grad():
                psnr, ssim = batch_psnr_ssim(clean_images, outputs)
                total_psnr += psnr
                total_ssim += ssim
                num_batches += 1
            
            progress_bar.set_postfix({
                'Loss': f"{running_loss/(num_batches*images.size(0)):.4f}",
                'PSNR': f"{total_psnr/num_batches:.2f}",
                'SSIM': f"{total_ssim/num_batches:.4f}"
            })

        train_loss = running_loss / len(train_loader.dataset)
        avg_psnr = total_psnr / num_batches
        avg_ssim = total_ssim / num_batches

        # Validation
        model.eval()
        val_loss = 0
        val_psnr = 0.0
        val_ssim = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for images,pert_labels, true_labels in val_loader:
                images = images.to(device, dtype=torch.float)
                outputs, _ = model(images)
                loss = criterion(outputs, images)
                val_loss += loss.item() * images.size(0)
                
                psnr, ssim = batch_psnr_ssim(images, outputs)
                val_psnr += psnr
                val_ssim += ssim
                val_batches += 1
                
        val_loss = val_loss / len(val_loader.dataset)
        avg_val_psnr = val_psnr / val_batches
        avg_val_ssim = val_ssim / val_batches
        
        # Update scheduler
        scheduler.step(val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        print(f'PSNR - Train: {avg_psnr:.2f}, Val: {avg_val_psnr:.2f}')
        print(f'SSIM - Train: {avg_ssim:.4f}, Val: {avg_val_ssim:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs('./models', exist_ok=True)
            torch.save(model.state_dict(), './models/best_hypernet.pth')
            print(f'Saved best hypernet model (val_loss: {best_val_loss:.6f})')
    
    return model
