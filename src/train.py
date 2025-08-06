# train.py
"""
Comprehensive training script for FL-Purification
Supports individual model training, pipeline training, and data preparation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import os
import wandb
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
import lpips

# Import models
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
from models.DAE import DenoisingAutoEncoder
from models.hypernet import AdaptiveLaplacianPyramidUNet
from models.resnet18 import ResNet18_MedMNIST
from models.victim import SimpleCNN
from models.AE import SimpleAutoencoder

# Import utilities
from dataloader import (
    load_kaggle_dataset, 
    create_dataloader_from_kaggle_data,
    create_clean_only_dataloader_from_kaggle_data,  # NEW: Import clean-only dataloader
    get_clean_only_dataloader,  # NEW: Import direct clean dataloader
    load_multiple_attacks,
    get_dataset_info,
    list_available_kaggle_datasets
)
from utils.Attacks import fgsm_attack, pgd_attack, carlini_attack
from utils.utils import batch_psnr_ssim

from utils.Resnet18_train import train_resnet18
from utils.AE_train import train_autoencoder
from utils.DAE_train import train_denoising_autoencoder
from utils.Hypernet_train import train_hypernet


def get_num_classes(dataset_name):
    """Get number of classes for a given MedMNIST dataset"""
    if dataset_name == 'bloodmnist': # doing - invi
        return 8
    elif dataset_name == 'pathmnist': # doing - akshat
        return 9
    elif dataset_name == 'dermamnist': # doing - akshat
        return 7
    elif dataset_name == 'octmnist': # failed 
        return 4
    elif dataset_name == 'pneumoniamnist': # failed
        return 2
    elif dataset_name == 'retinamnist': # doing - akshat
        return 5
    elif dataset_name == 'breastmnist': # failed
        return 2
    elif dataset_name == 'tissuemnist':
        return 8
    elif dataset_name == 'organamnist': # failed - dishita
        return 11
    elif dataset_name == 'organcmnist': # failed - invi p
        return 11
    elif dataset_name == 'organsmnist':
        return 11
    elif dataset_name == 'chestmnist': # failed - invi
        return 14
    elif dataset_name == 'mnist':
        return 10
    elif dataset_name == 'cifar10':
        return 10
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


def train_detector(config, train_loader, val_loader):
    """Train detector model using autoencoder"""
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    
    # Initialize detector model (using autoencoder as detector)
    detector = SimpleAutoencoder()
    
    # Train using enhanced autoencoder training
    trained_model = train_autoencoder(
        model=detector,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=config['epochs'],
        learning_rate=config['lr'],
        config=config
    )
    
    return trained_model

def train_reformer(config, train_loader, val_loader):
    """Train reformer model"""
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    
    # Initialize reformer model based on type
    reformer_type = config.get('reformer_type', 'hypernet')
    if reformer_type == 'hypernet':
        reformer = AdaptiveLaplacianPyramidUNet(
            encoder_name='resnet34',
            encoder_weights='imagenet',
            decoder_channels=(256, 128, 64, 32, 16),
            num_pyramid_levels=5,
            kernel_size=5
        )
        # Train using enhanced hypernet training
        trained_model = train_hypernet(
            model=reformer,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=config['epochs'],
            config=config
        )
    elif reformer_type == 'autoencoder':
        reformer = SimpleAutoencoder()
        # Train using enhanced autoencoder training
        trained_model = train_autoencoder(
            model=reformer,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=config['epochs'],
            learning_rate=config['lr'],
            config=config
        )
    elif reformer_type == 'denoising_autoencoder':
        reformer = DenoisingAutoEncoder(
            image_shape=(3, 28, 28), 
            structure=[64, 128, 256, "max", 128, 64]
        )
        # Train using enhanced denoising autoencoder training
        trained_model = train_denoising_autoencoder(
            model=reformer,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=config['epochs'],
            config=config
        )
    else:
        raise ValueError(f"Unknown reformer type: {reformer_type}")
    
    return trained_model

def train_classifier(config, train_loader, val_loader):
    """Train classifier model"""
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    
    # Get number of classes
    num_classes = get_num_classes(config['dataset_name'])
    
    # Initialize classifier model (always ResNet18)
    classifier = ResNet18_MedMNIST(num_classes=num_classes)
    
    # Train using enhanced ResNet18 training
    trained_model = train_resnet18(
        model=classifier,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=config['epochs'],
        learning_rate=config['lr'],
        config=config
    )
    
    return trained_model

def train_model(config):
    """Main training function - supports individual model training"""
    print(f"Starting training with pipeline: {config['pipeline_type']}")
    print(f"Dataset: {config['dataset_name']}")
    print(f"Attack type: {config['attack_type']}")
    print(f"Train clean only: {config.get('train_clean_only', False)}")  # NEW: Log clean training mode
    
    # Check if we're in Kaggle mode
    kaggle_mode = config.get('kaggle_mode', False)
    train_clean_only = config.get('train_clean_only', False)  # NEW: Get clean training flag
    
    if kaggle_mode:
        print("Using Kaggle dataloader...")
        
        # Check available datasets
        list_available_kaggle_datasets()
        
        # Get dataset info
        dataset_info = get_dataset_info(config['dataset_name'])
        if not dataset_info:
            print(f"No Kaggle dataset found for {config['dataset_name']}")
            return None
        
        print(f"Available attacks: {dataset_info['available_attacks']}")
        print(f"Available strengths: {dataset_info['available_strengths']}")
        
        if train_clean_only:
            # NEW: Train on clean data only - use direct clean dataloader
            print("Training on CLEAN DATA ONLY...")
            try:
                train_loader = get_clean_only_dataloader(
                    config['dataset_name'], 
                    config['batch_size'], 
                    split='train'
                )
                val_loader = get_clean_only_dataloader(
                    config['dataset_name'], 
                    config['batch_size'], 
                    split='val'
                )
                
                print(f"Successfully loaded clean-only data:")
                print(f"  Train batches: {len(train_loader)}")
                print(f"  Val batches: {len(val_loader)}")
                
            except Exception as e:
                print(f"Error loading clean data: {e}")
                return None
        
        else:
            # Original behavior - load adversarial + clean data
            attack_type = config['attack_type']
            strength = config.get('attack_strength', 'weak')
            
            if attack_type == 'none':
                print("No attack specified, using clean data only")
                attack_type = 'fgsm'  # dummy, will be filtered out
            
            try:
                # Load training data
                train_data = load_kaggle_dataset(
                    config['dataset_name'], 
                    attack_type, 
                    strength, 
                    split='train'
                )
                
                # Load validation data
                val_data = load_kaggle_dataset(
                    config['dataset_name'], 
                    attack_type, 
                    strength, 
                    split='val'
                )
                
                # Create dataloaders
                train_loader = create_dataloader_from_kaggle_data(
                    train_data, 
                    batch_size=config['batch_size'], 
                    shuffle=True
                )
                val_loader = create_dataloader_from_kaggle_data(
                    val_data, 
                    batch_size=config['batch_size'], 
                    shuffle=False
                )
                
                print(f"Successfully loaded Kaggle data:")
                print(f"  Train batches: {len(train_loader)}")
                print(f"  Val batches: {len(val_loader)}")
                
            except Exception as e:
                print(f"Error loading Kaggle data: {e}")
                return None
    
    else:
        # Legacy mode - use local data preparation
        print("Using local data preparation...")
        data_dir = config.get('data_dir', './data_cache')
        if os.path.exists(os.path.join(data_dir, config['dataset_name'])):
            print("Loading prepared data...")
            # Load clean data
            clean_train = load_prepared_data(config['dataset_name'], split='train', data_dir=data_dir)
            clean_val = load_prepared_data(config['dataset_name'], split='val', data_dir=data_dir)
            
            if train_clean_only:
                # NEW: Use only clean data for training
                print("Training on CLEAN DATA ONLY...")
                train_images, train_pert_labels, train_true_labels = clean_train['images'], torch.zeros(len(clean_train['images'])), clean_train['labels']
                val_images, val_pert_labels, val_true_labels = clean_val['images'], torch.zeros(len(clean_val['images'])), clean_val['labels']
            else:
                # Original behavior
                # Load attack data if needed
                if config['attack_type'] != 'none':
                    adv_train = load_prepared_data(config['dataset_name'], config['attack_type'], split='train', data_dir=data_dir)
                    adv_val = load_prepared_data(config['dataset_name'], config['attack_type'], split='val', data_dir=data_dir)
                    
                    # Create combined datasets
                    train_images, train_pert_labels, train_true_labels = create_combined_dataset(
                        clean_train, adv_train, config.get('adv_ratio', 0.5)
                    )
                    val_images, val_pert_labels, val_true_labels = create_combined_dataset(
                        clean_val, adv_val, config.get('adv_ratio', 0.5)
                    )
                else:
                    # Use clean data only
                    train_images, train_pert_labels, train_true_labels = clean_train['images'], torch.zeros(len(clean_train['images'])), clean_train['labels']
                    val_images, val_pert_labels, val_true_labels = clean_val['images'], torch.zeros(len(clean_val['images'])), clean_val['labels']
            
            # Create dataloaders
            train_dataset = TensorDataset(train_images, train_pert_labels, train_true_labels)
            val_dataset = TensorDataset(val_images, val_pert_labels, val_true_labels)
            train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
            
        else:
            print("Prepared data not found. Please run with --prepare_data first.")
            return None
    
    # Train individual models based on pipeline type
    if 'detector' in config['pipeline_type']:
        print("Training detector...")
        train_detector(config, train_loader, val_loader)
    
    if 'reformer' in config['pipeline_type']:
        print("Training reformer...")
        train_reformer(config, train_loader, val_loader)
    
    if 'classifier' in config['pipeline_type']:
        print("Training classifier...")
        train_classifier(config, train_loader, val_loader)
    
    print("Training completed!")

def main(config):
    """Main training function that takes a config dictionary"""
    print(f"Training {config['pipeline_type']} model...")
    print(f"Dataset: {config['dataset_name']}")
    print(f"Attack type: {config['attack_type']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['lr']}")
    print(f"Train clean only: {config.get('train_clean_only', False)}")  # NEW: Log clean training flag
    
    # Train model
    train_model(config)