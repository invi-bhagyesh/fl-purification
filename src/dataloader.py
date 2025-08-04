"""
Comprehensive Kaggle Dataloader for FL-Purification
Single file that can be run in Kaggle to download datasets, generate attacks,
and prepare them for easy access in training/testing notebooks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO
import numpy as np
import os
import pickle
from tqdm import tqdm
import zipfile

# Kaggle-specific paths
KAGGLE_INPUT_DIR = "/kaggle/input"
KAGGLE_WORKING_DIR = "/kaggle/working"
KAGGLE_DATA_DIR = "/kaggle/working/fl_purification_data"

# Available datasets and configurations
AVAILABLE_DATASETS = {
    'bloodmnist': {'num_classes': 8, 'channels': 3},
    'pathmnist': {'num_classes': 9, 'channels': 3},
    'dermamnist': {'num_classes': 7, 'channels': 3},
    'octmnist': {'num_classes': 4, 'channels': 1},
    'pneumoniamnist': {'num_classes': 2, 'channels': 1},
    'retinamnist': {'num_classes': 5, 'channels': 3},
    'breastmnist': {'num_classes': 2, 'channels': 1},
    'tissuemnist': {'num_classes': 8, 'channels': 1}
}

# Attack configurations
ATTACK_CONFIGS = {
    'fgsm': {
        'weak': {'epsilon': 0.1},
        'medium': {'epsilon': 0.3},
        'strong': {'epsilon': 0.5}
    },
    'pgd': {
        'weak': {'epsilon': 0.1, 'alpha': 0.01, 'iters': 20},
        'medium': {'epsilon': 0.3, 'alpha': 0.01, 'iters': 40},
        'strong': {'epsilon': 0.5, 'alpha': 0.01, 'iters': 60}
    },
    'carlini': {
        'weak': {'c': 1e-3, 'steps': 100, 'lr': 0.01},
        'medium': {'c': 1e-2, 'steps': 200, 'lr': 0.01},
        'strong': {'c': 1e-1, 'steps': 300, 'lr': 0.01}
    }
}

from models.resnet18 import BasicBlock, ResNet18_MedMNIST


def fgsm_attack(model, images, labels, epsilon=0.3):
    """Fast Gradient Sign Method attack"""
    images.requires_grad = True
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    loss.backward()
    adv_images = images + epsilon * images.grad.sign()
    return torch.clamp(adv_images, 0, 1).detach()

def pgd_attack(model, images, labels, epsilon=0.3, alpha=0.01, iters=40):
    """Projected Gradient Descent attack"""
    adv_images = images.clone()
    for _ in range(iters):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        adv_images = adv_images + alpha * adv_images.grad.sign()
        delta = adv_images - images
        delta = torch.clamp(delta, -epsilon, epsilon)
        adv_images = torch.clamp(images + delta, 0, 1).detach()
    return adv_images

def carlini_attack(model, images, labels, c=1e-2, steps=100, lr=0.01):
    """Carlini & Wagner L2 attack (simplified)"""
    adv_images = images.clone()
    for _ in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        target_scores = outputs.gather(1, labels.unsqueeze(1))
        max_other_scores = (outputs - 1000 * torch.eye(outputs.size(1)).to(outputs.device)[labels]).max(1)[0]
        loss = torch.clamp(max_other_scores - target_scores + 50, min=0).sum()
        loss.backward()
        adv_images = adv_images - lr * adv_images.grad.sign()
        adv_images = torch.clamp(adv_images, 0, 1).detach()
    return adv_images

def get_medmnist_dataloader(dataset_name, batch_size, split='train'):
    """Create dataloader for MedMNIST dataset"""
    if dataset_name not in INFO:
        raise ValueError(f"Dataset {dataset_name} not found in MedMNIST")
    
    info = INFO[dataset_name]
    DataClass = getattr(medmnist, info['python_class'])
    
    channels = AVAILABLE_DATASETS[dataset_name]['channels']
    if channels == 1:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    dataset = DataClass(split=split, transform=transform, download=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split=='train'), num_workers=2)

def setup_kaggle_environment():
    """Setup Kaggle environment and directories"""
    print("Setting up Kaggle environment...")
    os.makedirs(KAGGLE_DATA_DIR, exist_ok=True)
    
    for dataset in AVAILABLE_DATASETS.keys():
        dataset_dir = os.path.join(KAGGLE_DATA_DIR, dataset)
        os.makedirs(dataset_dir, exist_ok=True)
        
        for attack_type in ['fgsm', 'pgd', 'carlini']:
            attack_dir = os.path.join(dataset_dir, attack_type)
            os.makedirs(attack_dir, exist_ok=True)
            
            for strength in ['weak', 'medium', 'strong']:
                strength_dir = os.path.join(attack_dir, strength)
                os.makedirs(strength_dir, exist_ok=True)
                
                splits_dir = os.path.join(strength_dir, 'splits')
                os.makedirs(splits_dir, exist_ok=True)
    
    print(f"Kaggle environment setup complete. Data directory: {KAGGLE_DATA_DIR}")

def generate_attacks_for_strength(model, dataloader, attack_type, strength, device='cuda'):
    """Generate adversarial examples for specific attack type and strength"""
    attack_params = ATTACK_CONFIGS[attack_type][strength]
    
    clean_images, clean_labels, adv_images, adv_labels = [], [], [], []
    model.eval()
    
    print(f"Generating {attack_type} attacks with {strength} strength...")
    
    for images, labels in tqdm(dataloader, desc=f"{attack_type}_{strength}"):
        images = images.to(device)
        labels = labels.squeeze().long().to(device)
        
        if attack_type == 'fgsm':
            epsilon = attack_params['epsilon']
            adv_batch = fgsm_attack(model, images, labels, epsilon)
        elif attack_type == 'pgd':
            epsilon = attack_params['epsilon']
            alpha = attack_params['alpha']
            iters = attack_params['iters']
            adv_batch = pgd_attack(model, images, labels, epsilon, alpha, iters)
        elif attack_type == 'carlini':
            c = attack_params['c']
            steps = attack_params['steps']
            lr = attack_params['lr']
            adv_batch = carlini_attack(model, images, labels, c, steps, lr)
        
        clean_images.append(images.cpu())
        clean_labels.append(labels.cpu())
        adv_images.append(adv_batch.cpu())
        adv_labels.append(labels.cpu())
    
    return (torch.cat(clean_images), torch.cat(clean_labels), 
            torch.cat(adv_images), torch.cat(adv_labels))

def prepare_dataset_comprehensive(dataset_name, device='cuda'):
    """Prepare comprehensive dataset with all attack types and strengths, half adversarial half clean."""
    print(f"Preparing comprehensive dataset for {dataset_name}...")
    
    num_classes = AVAILABLE_DATASETS[dataset_name]['num_classes']
    channels = AVAILABLE_DATASETS[dataset_name]['channels']
    model = ResNet18_MedMNIST(num_classes=num_classes).to(device)
    
    splits = ['train', 'val', 'test']
    batch_size = 64
    
    for split in splits:
        print(f"\nProcessing {split} split...")
        
        try:
            dataloader = get_medmnist_dataloader(dataset_name, batch_size, split)
        except Exception as e:
            print(f"Error loading {split} split: {e}")
            continue
        
        all_samples = []
        for batch in dataloader:
            all_samples.append(batch)
        
        attack_types = ['fgsm', 'pgd', 'carlini']
        strengths = ['weak', 'medium', 'strong']
        
        for attack_type in attack_types:
            for strength in strengths:
                print(f"  Generating {attack_type} {strength} attacks...")
                
                try:
                    clean_images, clean_labels, adv_images, adv_labels = generate_attacks_for_strength(
                        model, all_samples, attack_type, strength, device
                    )
                    
                    # Use half of each for the final dataset
                    half_len = len(clean_images) // 2
                    clean_images_half = clean_images[:half_len]
                    clean_labels_half = clean_labels[:half_len]
                    adv_images_half = adv_images[:half_len]
                    adv_labels_half = adv_labels[:half_len]
                    
                    save_path = os.path.join(KAGGLE_DATA_DIR, dataset_name, attack_type, strength, 'splits')
                    os.makedirs(save_path, exist_ok=True)
                    
                    data = {
                        'clean_images': clean_images_half,
                        'clean_labels': clean_labels_half,
                        'adv_images': adv_images_half,
                        'adv_labels': adv_labels_half,
                        'attack_type': attack_type,
                        'strength': strength,
                        'split': split,
                        'dataset_name': dataset_name
                    }
                    
                    with open(os.path.join(save_path, f'{split}.pkl'), 'wb') as f:
                        pickle.dump(data, f)
                    
                    print(f"    Saved {split} data for {attack_type} {strength}: {len(clean_images_half) * 2} samples (half clean, half adversarial)")
                    
                except Exception as e:
                    print(f"    Error generating {attack_type} {strength} attacks: {e}")
                    continue

def create_kaggle_dataset_zip():
    """Create a zip file for Kaggle dataset submission"""
    print("Creating Kaggle dataset zip file...")
    zip_path = os.path.join(KAGGLE_WORKING_DIR, "fl_purification_complete_dataset.zip")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(KAGGLE_DATA_DIR):
            for file in files:
                if file.endswith('.pkl'):
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, KAGGLE_DATA_DIR)
                    zipf.write(file_path, arcname)
    
    print(f"Kaggle dataset created: {zip_path}")
    return zip_path

def detect_kaggle_dataset(dataset_name):
    """Detect available Kaggle datasets"""
    available_datasets = []
    
    if not os.path.exists(KAGGLE_INPUT_DIR):
        return available_datasets
    
    possible_names = [
        f"fl-purification-{dataset_name}",
        f"fl-purification-{dataset_name}-complete",
        f"fl-purification-{dataset_name}-dataset",
        f"{dataset_name}-adversarial-attacks",
        f"fl-purification-complete-{dataset_name}",
        f"fl-purification-{dataset_name}-all-attacks"
    ]
    
    for name in possible_names:
        dataset_path = os.path.join(KAGGLE_INPUT_DIR, name)
        if os.path.exists(dataset_path):
            available_datasets.append(name)
    
    return available_datasets

def load_kaggle_dataset(dataset_name, attack_type, strength, split='train', kaggle_dataset_name=None):
    """Load data from Kaggle dataset"""
    if kaggle_dataset_name is None:
        available_datasets = detect_kaggle_dataset(dataset_name)
        if not available_datasets:
            raise FileNotFoundError(f"No Kaggle dataset found for {dataset_name}")
        kaggle_dataset_name = available_datasets[0]
        print(f"Auto-detected Kaggle dataset: {kaggle_dataset_name}")
    
    data_path = os.path.join(
        KAGGLE_INPUT_DIR, 
        kaggle_dataset_name,
        dataset_name,
        attack_type,
        strength,
        'splits',
        f'{split}.pkl'
    )
    
    if not os.path.exists(data_path):
        alternative_paths = [
            os.path.join(KAGGLE_INPUT_DIR, kaggle_dataset_name, f'{dataset_name}_{attack_type}_{strength}_{split}.pkl'),
            os.path.join(KAGGLE_INPUT_DIR, kaggle_dataset_name, attack_type, strength, f'{split}.pkl'),
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                data_path = alt_path
                break
        else:
            raise FileNotFoundError(f"Data not found at {data_path} or alternative paths")
    
    print(f"Loading data from: {data_path}")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    return data

def create_dataloader_from_kaggle_data(data, batch_size=64, shuffle=True):
    """Create dataloader from Kaggle data"""
    clean_images = data['clean_images']
    clean_labels = data['clean_labels']
    adv_images = data['adv_images']
    adv_labels = data['adv_labels']
    
    all_images = torch.cat([clean_images, adv_images], dim=0)
    all_labels = torch.cat([clean_labels, adv_labels], dim=0)
    
    pert_labels = torch.cat([
        torch.zeros(len(clean_images)),
        torch.ones(len(adv_images))
    ])
    
    dataset = TensorDataset(all_images, pert_labels, all_labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def list_available_kaggle_datasets():
    """List all available Kaggle datasets"""
    if not os.path.exists(KAGGLE_INPUT_DIR):
        print("Kaggle input directory not found. Not running in Kaggle environment.")
        return []
    
    datasets = os.listdir(KAGGLE_INPUT_DIR)
    print("Available Kaggle datasets:")
    for dataset in datasets:
        dataset_path = os.path.join(KAGGLE_INPUT_DIR, dataset)
        if os.path.isdir(dataset_path):
            print(f"  - {dataset}")
            try:
                contents = os.listdir(dataset_path)
                if len(contents) <= 10:
                    print(f"    Contents: {contents}")
                else:
                    print(f"    Contents: {contents[:10]}... (and {len(contents)-10} more)")
            except:
                pass
    return datasets

def get_dataset_info(dataset_name, kaggle_dataset_name=None):
    """Get information about available data in the dataset"""
    if kaggle_dataset_name is None:
        available_datasets = detect_kaggle_dataset(dataset_name)
        if not available_datasets:
            return None
        kaggle_dataset_name = available_datasets[0]
    
    dataset_path = os.path.join(KAGGLE_INPUT_DIR, kaggle_dataset_name)
    
    info = {
        'kaggle_dataset_name': kaggle_dataset_name,
        'dataset_path': dataset_path,
        'available_attacks': [],
        'available_strengths': [],
        'available_splits': []
    }
    
    expected_path = os.path.join(dataset_path, dataset_name)
    if os.path.exists(expected_path):
        for attack_type in ['fgsm', 'pgd', 'carlini']:
            attack_path = os.path.join(expected_path, attack_type)
            if os.path.exists(attack_path):
                info['available_attacks'].append(attack_type)
                
                for strength in ['weak', 'medium', 'strong']:
                    strength_path = os.path.join(attack_path, strength)
                    if os.path.exists(strength_path):
                        if strength not in info['available_strengths']:
                            info['available_strengths'].append(strength)
                        
                        splits_path = os.path.join(strength_path, 'splits')
                        if os.path.exists(splits_path):
                            for split in ['train', 'val', 'test']:
                                split_file = os.path.join(splits_path, f'{split}.pkl')
                                if os.path.exists(split_file):
                                    if split not in info['available_splits']:
                                        info['available_splits'].append(split)
    
    return info

def load_multiple_attacks(dataset_name, attack_types, strength, split='train', kaggle_dataset_name=None):
    """Load data for multiple attack types and combine them"""
    all_data = []
    
    for attack_type in attack_types:
        try:
            data = load_kaggle_dataset(dataset_name, attack_type, strength, split, kaggle_dataset_name)
            all_data.append(data)
            print(f"Loaded {attack_type} attack data: {len(data['clean_images'])} samples")
        except FileNotFoundError as e:
            print(f"Warning: Could not load {attack_type} attack data: {e}")
            continue
    
    if not all_data:
        raise FileNotFoundError(f"No attack data found for {attack_types}")
    
    combined_data = {
        'clean_images': torch.cat([d['clean_images'] for d in all_data], dim=0),
        'clean_labels': torch.cat([d['clean_labels'] for d in all_data], dim=0),
        'adv_images': torch.cat([d['adv_images'] for d in all_data], dim=0),
        'adv_labels': torch.cat([d['adv_labels'] for d in all_data], dim=0),
        'attack_type': 'combined',
        'strength': strength,
        'split': split
    }
    
    return combined_data

def main():
    """Main function for comprehensive data preparation and loading"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Kaggle Dataloader')
    parser.add_argument('--mode', type=str, default='prepare', 
                       choices=['prepare', 'load', 'list', 'info', 'create_zip'],
                       help='Mode to run in')
    parser.add_argument('--dataset', type=str, default='bloodmnist',
                       choices=list(AVAILABLE_DATASETS.keys()),
                       help='Dataset to work with')
    parser.add_argument('--attack_type', type=str, default='fgsm',
                       choices=['fgsm', 'pgd', 'carlini'],
                       help='Attack type to load')
    parser.add_argument('--strength', type=str, default='medium',
                       choices=['weak', 'medium', 'strong'],
                       help='Attack strength')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'val', 'test'],
                       help='Data split')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--max_samples', type=int, default=1000, 
                       help='Maximum samples per split for preparation')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for dataloader')
    
    args = parser.parse_args()
    
    if args.mode == 'prepare':
        setup_kaggle_environment()
        prepare_dataset_comprehensive(args.dataset, args.device, args.max_samples)
        print(f"Dataset {args.dataset} preparation completed!")
        
    elif args.mode == 'load':
        try:
            data = load_kaggle_dataset(args.dataset, args.attack_type, args.strength, args.split)
            print(f"Successfully loaded data:")
            print(f"  Clean images: {len(data['clean_images'])}")
            print(f"  Adversarial images: {len(data['adv_images'])}")
            print(f"  Attack type: {data['attack_type']}")
            print(f"  Strength: {data['strength']}")
            
            dataloader = create_dataloader_from_kaggle_data(data, args.batch_size)
            print(f"  Dataloader batches: {len(dataloader)}")
            
        except Exception as e:
            print(f"Error loading data: {e}")
    
    elif args.mode == 'list':
        list_available_kaggle_datasets()
    
    elif args.mode == 'info':
        info = get_dataset_info(args.dataset)
        if info:
            print(f"Dataset info for {args.dataset}:")
            print(f"  Kaggle dataset: {info['kaggle_dataset_name']}")
            print(f"  Available attacks: {info['available_attacks']}")
            print(f"  Available strengths: {info['available_strengths']}")
            print(f"  Available splits: {info['available_splits']}")
        else:
            print(f"No dataset info found for {args.dataset}")
    
    elif args.mode == 'create_zip':
        create_kaggle_dataset_zip()

if __name__ == '__main__':
    main() 