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
import torchattacks
from torchattacks import CW

# Kaggle-specific paths
KAGGLE_INPUT_DIR = "/kaggle/input"
KAGGLE_WORKING_DIR = "/kaggle/working"
KAGGLE_DATA_DIR = "/kaggle/working/fl_purification_data"

# Simplified dataset listing. We'll read channels & num_classes from medmnist.INFO dynamically.
AVAILABLE_DATASETS = {
    'bloodmnist': {},
    'pathmnist': {},
    'dermamnist': {},
    'octmnist': {},
    'pneumoniamnist': {},
    'retinamnist': {},
    'breastmnist': {},
    'tissuemnist': {},
    'organamnist': {},
    'organcmnist': {},
    'organsmnist': {},
    'chestmnist': {}
}

# Attack configurations - added 'medium' as an intermediate level
ATTACK_CONFIGS = {
    'fgsm': {
        'weak': {'epsilon': 0.1},
        'strong': {'epsilon': 0.5}
    },
    'pgd': {
        'weak': {'epsilon': 0.1, 'alpha': 0.01, 'iters': 20},
        'strong': {'epsilon': 0.5, 'alpha': 0.01, 'iters': 60}
    },
    'carlini': {
        'weak': {'c': 1e-4, 'steps': 1000, 'lr': 0.01},
        'strong': {'c': 1e-1, 'steps': 1000, 'lr': 0.01}
    }
}

from models.resnet18 import BasicBlock, ResNet18_MedMNIST


def fgsm_attack(model, images, labels, epsilon=0.3):
    """Fast Gradient Sign Method attack"""
    images = images.clone().detach().to(images.device)
    images.requires_grad = True
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    model.zero_grad()
    loss.backward()
    adv_images = images + epsilon * images.grad.sign()
    return torch.clamp(adv_images, 0, 1).detach()


def pgd_attack(model, images, labels, epsilon=0.3, alpha=0.01, iters=40):
    """Projected Gradient Descent attack"""
    adv_images = images.clone().detach().to(images.device)
    ori_images = images.clone().detach().to(images.device)
    for _ in range(iters):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()
        adv_images = adv_images + alpha * adv_images.grad.sign()
        delta = torch.clamp(adv_images - ori_images, -epsilon, epsilon)
        adv_images = torch.clamp(ori_images + delta, 0, 1).detach()
    return adv_images


def carlini_attack(model, images, labels, c=1e-2, steps=1000, lr=0.01):
    # Assumes model is in eval() mode, images and labels are on the correct device
    attack = CW(model, c=c, steps=steps, lr=lr)
    adv_images = attack(images, labels)
    return adv_images


def get_medmnist_dataloader(dataset_name, batch_size, split='train', force_rgb=False, num_workers=2):
    """Create dataloader for MedMNIST dataset.
       - Reads channels from medmnist.INFO
       - If force_rgb=True, grayscale images will be repeated to 3 channels
    """
    if dataset_name not in INFO:
        raise ValueError(f"Dataset {dataset_name} not found in MedMNIST")

    info = INFO[dataset_name]
    DataClass = getattr(medmnist, info['python_class'])

    # Read channels and num_classes from INFO
    channels = int(info.get('n_channels', 1))
    # info['label'] is a mapping like {0: '...'}, so number of classes is its length
    num_classes = len(info.get('label', {}))

    # Build transforms depending on channels. If force_rgb, convert single channel -> 3-channel
    tfms = [transforms.ToTensor()]

    if channels == 1 and force_rgb:
        # convert 1-channel to 3-channel so models expecting RGB can accept it
        tfms.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
        # use ImageNet normalization after repeating
        tfms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]))
    elif channels == 1:
        # grayscale normalization (centered around 0.5 if unknown)
        tfms.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    else:
        # rgb
        tfms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]))

    transform = transforms.Compose(tfms)

    dataset = DataClass(split=split, transform=transform, download=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'), num_workers=num_workers)
    # attach metadata so callers can inspect channels/num_classes easily
    loader.dataset_channels = 3 if (channels == 1 and force_rgb) else channels
    loader.dataset_num_classes = num_classes
    return loader


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


def generate_attacks_for_strength(model, dataloader_or_batches, attack_type, strength, device='cuda'):
    """Generate adversarial examples for specific attack type and strength.
       Accepts either a DataLoader or a list/iterable of (images, labels) batches.
    """
    attack_params = ATTACK_CONFIGS[attack_type][strength]

    clean_images, clean_labels, adv_images, adv_labels = [], [], [], []
    model.eval()

    print(f"Generating {attack_type} attacks with {strength} strength...")

    # Allow either a DataLoader or a pre-collected list of batches
    iterator = dataloader_or_batches
    for batch in tqdm(iterator, desc=f"{attack_type}_{strength}"):
        # batch expected either as (images, labels) for medmnist dataset OR the dataset's tuple form
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            images, labels = batch[0], batch[1]
        else:
            raise ValueError("Each batch must be a tuple (images, labels, ...) or similar")

        images = images.to(device)
        # medmnist returns labels as shape (B,1) sometimes; squeeze and convert to long
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
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")

        clean_images.append(images.cpu())
        clean_labels.append(labels.cpu())
        adv_images.append(adv_batch.cpu())
        adv_labels.append(labels.cpu())

    if len(clean_images) == 0:
        return (torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]))

    return (torch.cat(clean_images), torch.cat(clean_labels),
            torch.cat(adv_images), torch.cat(adv_labels))


def prepare_dataset_comprehensive(dataset_name, device='cuda', max_samples=None, force_rgb=False):
    """Prepare comprehensive dataset with all attack types and strengths, half adversarial half clean.
       - Reads num_classes and channels from MEDMNIST INFO
       - max_samples: if provided, truncates each split to max_samples
       - force_rgb: if True will convert grayscale to 3-channel images (useful if model expects 3 channels)
    """
    print(f"Preparing comprehensive dataset for {dataset_name}...")

    if dataset_name not in INFO:
        raise ValueError(f"{dataset_name} not found in medmnist.INFO")

    info = INFO[dataset_name]
    num_classes = len(info.get('label', {}))
    channels = int(info.get('n_channels', 1))

    # create model according to what your ResNet accepts.
    # If ResNet18_MedMNIST supports an `in_channels` argument, pass it; else default to expectation in model
    try:
        model = ResNet18_MedMNIST(num_classes=num_classes, in_channels=(3 if (channels == 1 and force_rgb) else channels)).to(device)
    except TypeError:
        # fallback if ResNet18_MedMNIST only accepts num_classes
        model = ResNet18_MedMNIST(num_classes=num_classes).to(device)

    splits = ['train', 'val', 'test']
    batch_size = 64

    for split in splits:
        print(f"\nProcessing {split} split...")

        try:
            dataloader = get_medmnist_dataloader(dataset_name, batch_size, split, force_rgb=force_rgb)
        except Exception as e:
            print(f"Error loading {split} split: {e}")
            continue

        # optionally limit total samples by max_samples: collect batches until we have enough
        collected_batches = []
        collected_count = 0
        for batch in dataloader:
            imgs, lbls = batch[0], batch[1]
            batch_len = imgs.shape[0]
            if max_samples is not None and collected_count + batch_len > max_samples:
                # take only part of this batch
                remaining = max_samples - collected_count
                if remaining <= 0:
                    break
                imgs = imgs[:remaining]
                lbls = lbls[:remaining]
                collected_batches.append((imgs, lbls))
                collected_count += remaining
                break
            else:
                collected_batches.append((imgs, lbls))
                collected_count += batch_len
            if max_samples is not None and collected_count >= max_samples:
                break

        if collected_count == 0:
            print(f"  No samples found for {split}, skipping")
            continue

        attack_types = ['fgsm', 'pgd', 'carlini']
        strengths = ['weak', 'medium', 'strong']

        for attack_type in attack_types:
            for strength in strengths:
                print(f"  Generating {attack_type} {strength} attacks for {split} (samples: {collected_count})...")

                try:
                    clean_images, clean_labels, adv_images, adv_labels = generate_attacks_for_strength(
                        model, collected_batches, attack_type, strength, device
                    )

                    if clean_images.numel() == 0:
                        print(f"    No generated adversarial samples for {attack_type} {strength} (empty).")
                        continue

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
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum samples per split for preparation')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for dataloader')
    parser.add_argument('--force_rgb', action='store_true', help='If set, convert grayscale to 3-channel RGB by repeating channel')

    args = parser.parse_args()

    if args.mode == 'prepare':
        setup_kaggle_environment()
        prepare_dataset_comprehensive(args.dataset, args.device, args.max_samples, force_rgb=args.force_rgb)
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
