import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
from train import load_model_from_path
from Data_generation import get_dataloaders
from utils.misc.metrics import jsd, compute_jsd_threshold, compute_psnr_ssim, filter_adversarial_images_by_jsd
from dataloader import AdversarialDataset  # Your provided AdversarialDataset class


def parse_args():
    parser = argparse.ArgumentParser(description="Filter adversarial images with JSD threshold")
    parser.add_argument('--base_dir', type=str, default='medmnist', choices=['medmnist', 'others'],
                        help='Base directory (medmnist or others)')
    parser.add_argument('--dataset', type=str, default='bloodmnist', help='Dataset name from MedMNIST')
    parser.add_argument('--attack_type', type=str, default='fgsm', choices=['fgsm', 'pgd', 'cw'],
                        help='Attack type')
    parser.add_argument('--strength', type=str, default='strong', choices=['weak', 'strong', None],
                        help="Attack strength")
    return parser.parse_args()


def main():
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load models
    classifier_model = load_model_from_path('classifier', device)
    detector_model = load_model_from_path('detector', device)
    reformer_model = load_model_from_path('reformer', device)

    # Load clean dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(args.dataset)

    # Compute JSD threshold on clean validation data using detector reconstructions
    jsd_threshold = compute_jsd_threshold(detector_model, val_loader, device=device)
    print(f"Computed JSD threshold: {jsd_threshold}")

    # Create adversarial dataset instance with parameters from parser args
    adversarial_dataset = AdversarialDataset(
        base_dir=args.base_dir,
        dataset_name=args.dataset,
        attack_type=args.attack_type,
        strength=args.strength
    )

    # Filter adversarial images based on JSD threshold
    filtered_loader = filter_adversarial_images_by_jsd(detector_model, adversarial_dataset, jsd_threshold, device=device)

    if filtered_loader is not None:
        print(f"Filtered dataset size: {len(filtered_loader.dataset)} images")
    else:
        print("No images passed the JSD threshold filtering.")


if __name__ == "__main__":
    main()
