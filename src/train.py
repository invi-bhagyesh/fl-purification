# train.py
"""
Comprehensive training script for FL-Purification
Supports individual model training, pipeline training, and data preparation
Modified to create clean data on-the-fly for training
"""
# Import models
import sys
sys.path.append(
    os.path.join(os.path.dirname(__file__), '..', 'models')
)
import torch
import os
import wandb
from utils.Resnet18_train import train_resnet18
from utils.AE_train import train_autoencoder
from utils.DAE_train import train_denoising_autoencoder
from utils.Hypernet_train import train_hypernet

from models.DAE import DenoisingAutoEncoder
from models.hypernet import AdaptiveLaplacianPyramidUNet
from models.resnet18 import ResNet18_MedMNIST
from models.AE import SimpleAutoencoder

# Import utilities
from dataloader import (
    get_medmnist_dataloader,
    get_torchvision_dataloader
)

class CleanDataWrapper:
    """Wrapper to convert 2-tuple dataloaders to 3-tuple format for training functions"""
    def __init__(self, base_loader):
        self.base_loader = base_loader
        
    def __len__(self):
        return len(self.base_loader)
    
    @property
    def dataset(self):
        return self.base_loader.dataset
        
    def __iter__(self):
        for batch in self.base_loader:
            if len(batch) == 2:
                images, labels = batch
                # Add perturbation labels (0 = clean, 1 = adversarial)
                pert_labels = torch.zeros(len(images))  # All clean data
                yield images, pert_labels, labels
            else:
                # Already in correct format
                yield batch





def get_num_classes(dataset_name):
    """Get number of classes for a given MedMNIST dataset"""
    if dataset_name == 'bloodmnist':
        return 8
    elif dataset_name == 'pathmnist':
        return 9
    elif dataset_name == 'dermamnist':
        return 7
    elif dataset_name == 'octmnist':
        return 4
    elif dataset_name == 'pneumoniamnist':
        return 2
    elif dataset_name == 'retinamnist':
        return 5
    elif dataset_name == 'breastmnist':
        return 2
    elif dataset_name == 'tissuemnist':
        return 8
    elif dataset_name == 'organamnist':
        return 11
    elif dataset_name == 'organcmnist':
        return 11
    elif dataset_name == 'organsmnist':
        return 11
    elif dataset_name == 'chestmnist':
        return 14
    elif dataset_name == 'mnist':
        return 10
    elif dataset_name == 'cifar10':
        return 10
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


def create_clean_dataloaders_onthefly(config):
    """Create clean dataloaders on-the-fly - fast and simple for training"""
    print("Creating clean dataloaders on-the-fly...")
    dataset_name = config['dataset_name']
    batch_size = config['batch_size']
    
    try:
        # Use the existing dataloader functions to get clean data directly
        if dataset_name.lower() in ['mnist', 'cifar10']:
            base_train_loader = get_torchvision_dataloader(dataset_name, batch_size, 'train')
            base_val_loader = get_torchvision_dataloader(dataset_name, batch_size, 'val')
        else:
            # MedMNIST datasets
            base_train_loader = get_medmnist_dataloader(dataset_name, batch_size, 'train')
            base_val_loader = get_medmnist_dataloader(dataset_name, batch_size, 'val')
        
        # Convert to the expected format with perturbation labels
        train_loader = CleanDataWrapper(base_train_loader)
        val_loader = CleanDataWrapper(base_val_loader)
        
        print("Clean dataloaders created on-the-fly:")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        
        return train_loader, val_loader
        
    except Exception as e:
        print(f"Error creating on-the-fly dataloaders: {e}")
        return None, None


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
    """Main training function - supports individual model training with on-the-fly clean data"""
    print(f"Starting training with pipeline: {config['pipeline_type']}")
    print(f"Dataset: {config['dataset_name']}")
    print(f"Attack type: {config['attack_type']}")
    print(f"Train clean only: {config.get('train_clean_only', False)}")
    
    # Check if we're in Kaggle mode and training clean only
    kaggle_mode = config.get('kaggle_mode', False)
    train_clean_only = config.get('train_clean_only', False)
    
    train_loader, val_loader = None, None
    
    
    print("=== TRAINING ON CLEAN DATA ONLY ===")
    

    print("Local mode: Creating clean data on-the-fly...")
    train_loader, val_loader = create_clean_dataloaders_onthefly(config)
    
    if train_loader is None or val_loader is None:
        print("Failed to create dataloaders")
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
    
    if 'all' in config['pipeline_type']:
        print("Training all models in pipeline...")
        train_detector(config, train_loader, val_loader)
        train_reformer(config, train_loader, val_loader)
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
    print(f"Train clean only: {config.get('train_clean_only', False)}")
    
    # Train model
    train_model(config)