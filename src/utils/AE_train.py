import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from utils.utils import batch_psnr_ssim

def train_autoencoder(model, train_loader, val_loader, device, num_epochs=10, learning_rate=1e-3, config=None):
    """Enhanced autoencoder training based on AE_train.py"""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for images, _ in progress_bar:
            images = images.to(device)
            
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            
            # Calculate image quality metrics
            with torch.no_grad():
                psnr, ssim = batch_psnr_ssim(images, outputs)
                total_psnr += psnr
                total_ssim += ssim
                num_batches += 1
            
            progress_bar.set_postfix({
                'Loss': f"{train_loss/(num_batches*images.size(0)):.4f}",
                'PSNR': f"{total_psnr/num_batches:.2f}",
                'SSIM': f"{total_ssim/num_batches:.4f}"
            })

        train_loss /= len(train_loader.dataset)
        avg_psnr = total_psnr / num_batches
        avg_ssim = total_ssim / num_batches

        # Validation
        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        val_ssim = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for images, pert_labels, true_labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                loss = criterion(outputs, images)
                val_loss += loss.item() * images.size(0)
                
                psnr, ssim = batch_psnr_ssim(images, outputs)
                val_psnr += psnr
                val_ssim += ssim
                val_batches += 1
                
        val_loss /= len(val_loader.dataset)
        avg_val_psnr = val_psnr / val_batches
        avg_val_ssim = val_ssim / val_batches
        
        # Update scheduler
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        print(f'PSNR - Train: {avg_psnr:.2f}, Val: {avg_val_psnr:.2f}')
        print(f'SSIM - Train: {avg_ssim:.4f}, Val: {avg_val_ssim:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs('./models', exist_ok=True)
            torch.save(model.state_dict(), './models/best_autoencoder.pth')
            print(f'Saved best autoencoder model (val_loss: {best_val_loss:.6f})')
    
    return model
