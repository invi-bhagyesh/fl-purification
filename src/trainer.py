import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from utils.misc.metrics import compute_psnr_ssim
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
from tqdm import tqdm

def train_detector(model, train_loader, val_loader=None, epochs=20, lr=1e-3,
                   device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for images, _ in train_loader:
            images = images.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            psnr_all = []
            ssim_all = []
            with torch.no_grad():
                for images, _ in val_loader:
                    images = images.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, images)
                    val_loss += loss.item() * images.size(0)
                    # Compute PSNR and SSIM for this batch
                    batch_psnr, batch_ssim = compute_psnr_ssim(images, outputs)
                    psnr_all.append(batch_psnr)
                    ssim_all.append(batch_ssim)

            val_loss /= len(val_loader.dataset)
            avg_psnr = np.mean(psnr_all)
            avg_ssim = np.mean(ssim_all)
            print(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                  f"Val PSNR: {avg_psnr:.4f}, Val SSIM: {avg_ssim:.4f}")
        else:
            print(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.6f}")

    return model

def train_reformer(model, train_loader, val_loader=None, epochs=20, lr=1e-3,
                   device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for images, _ in train_loader:
            images = images.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            mse_loss = criterion(outputs, images)
            reg_loss = model.get_l2_loss()
            loss = mse_loss + reg_loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            psnr_all = []
            ssim_all = []
            with torch.no_grad():
                for images, _ in val_loader:
                    images = images.to(device)
                    outputs = model(images)
                    mse_loss = criterion(outputs, images)
                    reg_loss = model.get_l2_loss()
                    loss = mse_loss + reg_loss
                    val_loss += loss.item() * images.size(0)
                    # Compute PSNR and SSIM for this batch
                    batch_psnr, batch_ssim = compute_psnr_ssim(images, outputs)
                    psnr_all.append(batch_psnr)
                    ssim_all.append(batch_ssim)

            val_loss /= len(val_loader.dataset)
            avg_psnr = np.mean(psnr_all)
            avg_ssim = np.mean(ssim_all)
            print(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                  f"Val PSNR: {avg_psnr:.4f}, Val SSIM: {avg_ssim:.4f}")
        else:
            print(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.6f}")

    return model

def train_classifier(model, train_loader, val_loader, epochs=20, lr=1e-3, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Train the ResNet18_MedMNIST classifier on the dataset.

    Args:
        model: the neural network model to train.
        train_loader: DataLoader for training dataset.
        val_loader: DataLoader for validation dataset.
        epochs: number of training epochs.
        lr: learning rate for the optimizer.
        device: 'cuda' or 'cpu'.

    Returns:
        trained model
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_total = 0
        train_targets = []
        train_preds = []

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).long().squeeze()

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item() * images.size(0)
            
            preds = torch.argmax(outputs, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())

        avg_train_loss = train_loss_total / len(train_loader.dataset)
        train_f1 = f1_score(train_targets, train_preds, average='weighted')

        # Validation
        model.eval()
        val_loss_total = 0
        val_targets = []
        val_preds = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).long().squeeze()

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss_total += loss.item() * images.size(0)

                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        avg_val_loss = val_loss_total / len(val_loader.dataset)
        val_f1 = f1_score(val_targets, val_preds, average='weighted')

        print(f"Epoch {epoch}/{epochs} --> "
              f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
              f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")

    return model


def train_reformer_hipyrnet(model, train_loader, val_loader=None, epochs=20, lr=1e-3, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Training]")
        for noisy_images, _ in progress_bar:
            noisy_images = noisy_images.to(device)

            optimizer.zero_grad()
            outputs, _ = model(noisy_images)  # model returns (denoised, kernel)
            loss = criterion(outputs, noisy_images)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * noisy_images.size(0)
            progress_bar.set_postfix(loss=loss.item())

        train_loss /= len(train_loader.dataset)

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            psnr_scores = []
            ssim_scores = []

            val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Validation]")
            with torch.no_grad():
                for noisy_images, _ in val_bar:
                    noisy_images = noisy_images.to(device)
                    outputs, _ = model(noisy_images)
                    loss = criterion(outputs, noisy_images)
                    val_loss += loss.item() * noisy_images.size(0)

                    psnr, ssim = compute_psnr_ssim(noisy_images, outputs)
                    psnr_scores.append(psnr)
                    ssim_scores.append(ssim)

                    val_bar.set_postfix(psnr=psnr, ssim=ssim)

            val_loss /= len(val_loader.dataset)
            avg_psnr = np.mean(psnr_scores)
            avg_ssim = np.mean(ssim_scores)

            print(f"Epoch {epoch}/{epochs} Summary: "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f}")
        else:
            print(f"Epoch {epoch}/{epochs} Summary: Train Loss: {train_loss:.6f}")

    return model