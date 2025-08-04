"""
Enhanced training functions based on existing utils training files
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import os
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
import lpips
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))
from models.DAE import DenoisingAutoEncoder
from models.hypernet import AdaptiveLaplacianPyramidUNet
from models.resnet18 import ResNet18_MedMNIST
from models.AE import SimpleAutoencoder
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
        for images, pert_labels, true_labels in progress_bar:
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

def train_denoising_autoencoder(model, train_loader, val_loader, device, num_epochs=100, reg_strength=1e-9, config=None):
    """Enhanced denoising autoencoder training based on DAE_train.py"""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_psnr = 0.0
        total_ssim = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for images, pert_labels, true_labels in progress_bar:
            images = images.to(device)
            # Forward pass for bottleneck noise (as in your model's forward)
            output = model(images)
            loss = F.mse_loss(output, images) + model.get_l2_loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            
            # Calculate image quality metrics
            with torch.no_grad():
                psnr, ssim = batch_psnr_ssim(images, output)
                total_psnr += psnr
                total_ssim += ssim
                num_batches += 1
            
            progress_bar.set_postfix({
                'Loss': f"{total_loss/(num_batches*images.size(0)):.4f}",
                'PSNR': f"{total_psnr/num_batches:.2f}",
                'SSIM': f"{total_ssim/num_batches:.4f}"
            })

        train_loss = total_loss / len(train_loader.dataset)
        avg_psnr = total_psnr / num_batches
        avg_ssim = total_ssim / num_batches

        # Validation
        model.eval()
        val_loss = 0
        val_psnr = 0.0
        val_ssim = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for images, pert_labels, true_labels in val_loader:
                images = images.to(device)
                val_output = model(images)
                loss = F.mse_loss(val_output, images)
                val_loss += loss.item() * images.size(0)
                
                psnr, ssim = batch_psnr_ssim(images, val_output)
                val_psnr += psnr
                val_ssim += ssim
                val_batches += 1
                
        val_loss = val_loss / len(val_loader.dataset)
        avg_val_psnr = val_psnr / val_batches
        avg_val_ssim = val_ssim / val_batches
        
        # Update scheduler
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        print(f'PSNR - Train: {avg_psnr:.2f}, Val: {avg_val_psnr:.2f}')
        print(f'SSIM - Train: {avg_ssim:.4f}, Val: {avg_val_ssim:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs('./models', exist_ok=True)
            torch.save(model.state_dict(), './models/best_denoising_autoencoder.pth')
            print(f'Saved best denoising autoencoder model (val_loss: {best_val_loss:.6f})')
    
    return model

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
            for images, pert_labels, true_labels in val_loader:
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

def train_resnet18(model, train_loader, val_loader, device, num_epochs=10, learning_rate=0.0001, config=None):
    """Enhanced ResNet18 training based on Resnet18_train.py"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    best_val_f1 = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_train_preds = []
        all_train_labels = []
        total = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for images, pert_labels, true_labels in progress_bar:
            images, labels = images.to(device), true_labels.squeeze().long().to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            all_train_preds.extend(predicted.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())
            total += labels.size(0)
            
            progress_bar.set_postfix({
                'Loss': f"{running_loss/total:.4f}",
                'Acc': f"{100.*accuracy_score(all_train_labels, all_train_preds):.2f}%"
            })
            
        train_loss = running_loss / total
        train_f1 = f1_score(all_train_labels, all_train_preds, average='weighted')
        train_precision = precision_score(all_train_labels, all_train_preds, average='weighted')
        train_recall = recall_score(all_train_labels, all_train_preds, average='weighted')

        # Validation
        model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_labels = []
        val_total = 0
        
        with torch.no_grad():
            for images, pert_labels, true_labels in val_loader:
                images, labels = images.to(device), true_labels.squeeze().long().to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
                val_total += labels.size(0)
                
        val_loss /= val_total
        val_f1 = f1_score(all_val_labels, all_val_preds, average='weighted')
        val_precision = precision_score(all_val_labels, all_val_preds, average='weighted')
        val_recall = recall_score(all_val_labels, all_val_preds, average='weighted')
        
        # Update scheduler
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} | Train loss: {train_loss:.4f} | Train F1: {train_f1:.4f} "
              f"| Val loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
        print(f"Train - Precision: {train_precision:.4f}, Recall: {train_recall:.4f}")
        print(f"Val   - Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            os.makedirs('./models', exist_ok=True)
            torch.save(model.state_dict(), './models/best_resnet18.pth')
            print(f'Saved best ResNet18 model (val_f1: {best_val_f1:.4f})')
    
    return model 