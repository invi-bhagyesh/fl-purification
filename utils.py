import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
import torch.nn.functional as Fnn
import numpy as np

def softmax_with_temperature(logits, temperature):
    exp_logits = torch.exp(logits / temperature)
    return exp_logits / exp_logits.sum(dim=1, keepdim=True)

def jsd(p, q, eps=1e-6):
    m = 0.5 * (p + q)
    p = p + eps
    q = q + eps
    m = m + eps
    kld_pm = (p * (p / m).log()).sum(dim=1)
    kld_qm = (q * (q / m).log()).sum(dim=1)
    return 0.5 * (kld_pm + kld_qm)

def batch_psnr_ssim(originals, reconstructions, data_range=1.0):
    """
    Computes average PSNR and SSIM for a batch of images.
    Assumes images are (N, C, H, W) and normalized to [0,1].
    """
    originals = originals.detach().cpu().numpy()
    reconstructions = reconstructions.detach().cpu().numpy()
    batch_size = originals.shape[0]
    psnr_vals = []
    ssim_vals = []
    
    for i in range(batch_size):
        orig = np.transpose(originals[i], (1, 2, 0))  # (H,W,C)
        recon = np.transpose(reconstructions[i], (1, 2, 0))
        
        psnr = compute_psnr(orig, recon, data_range=data_range)
        psnr_vals.append(psnr)
        
        # SSIM for multi-channel
        ssim = compute_ssim(orig, recon, data_range=data_range, channel_axis=2)
        ssim_vals.append(ssim)
        
    return np.mean(psnr_vals), np.mean(ssim_vals)

import matplotlib.pyplot as plt

def plot_rgb_channel_histograms(original_images, reconstructed_images):
    # Both tensors should be of shape (batch, 3, 28, 28)
    channel_names = ['Red', 'Green', 'Blue']
    colors = ['red', 'green', 'blue']

    plt.figure(figsize=(15, 6))

    for idx in range(3):
        plt.subplot(2, 3, idx + 1)
        orig_pixels = original_images[:, idx, :, :].cpu().numpy().reshape(-1)
        plt.hist(orig_pixels, bins=50, color=colors[idx], alpha=0.7)
        plt.title(f'Original {channel_names[idx]} Channel')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')

        plt.subplot(2, 3, idx + 4)
        recon_pixels = reconstructed_images[:, idx, :, :].cpu().numpy().reshape(-1)
        plt.hist(recon_pixels, bins=50, color=colors[idx], alpha=0.7)
        plt.title(f'Reconstructed {channel_names[idx]} Channel')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()
