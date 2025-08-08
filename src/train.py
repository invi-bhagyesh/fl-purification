import os
import argparse
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO
from trainer import train_classifier, train_detector, train_reformer, train_reformer_hipyrnet
from models.Classifier.Resnet import BasicBlock , ResNet18_MedMNIST
from models.Detector.AE import SimpleAutoencoder
from models.Reformer.DAE import DenoisingAutoEncoder
from models.Reformer.Hypernet import AdaptiveLaplacianPyramidUNet
from models.Reformer.SMP import SMPPyramidDenoiser
from Data_generation import get_dataloaders
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np

def save_model_path(model, model_type):
    base_dir = '/kaggle/working/trained_models'
    save_dir = os.path.join(base_dir, model_type.capitalize())
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{model_type}_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


def load_model_from_path(model_type, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Load a trained model of the specified type from the predefined saved model path.

    Args:
        model_type (str): One of 'classifier', 'detector', 'reformer'.
        device (str): Device to map the model to ('cuda' or 'cpu').

    Returns:
        model: The loaded model with weights.
    """
    base_dir = '/kaggle/working/trained_models'
    model_dir = os.path.join(base_dir, model_type.capitalize())
    model_path = os.path.join(model_dir, f'{model_type}_model.pth')

    # Instantiate model architecture based on model_type
    if model_type == 'classifier':
        model = ResNet18_MedMNIST()
    elif model_type == 'detector':
        model = SimpleAutoencoder()
    elif model_type == 'reformer':
        model = DenoisingAutoEncoder()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Load saved weights
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"Loaded {model_type} model from {model_path}")
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")

    return model

def main():
    parser = argparse.ArgumentParser(description="Train a model on MedMNIST dataset")
    parser.add_argument('--dataset', type=str, default='bloodmnist', help="Dataset name from MedMNIST")
    parser.add_argument('--epochs', type=int, default=20, help="Number of epochs to train")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for training")
    parser.add_argument('--reformer-type', type=str, default="dae", help="set reformer to use")
    parser.add_argument('--model', type=str, default='classifier', choices=['classifier', 'detector', 'reformer'],
                        help="Choose which model training function to use")
    args = parser.parse_args()

    train_loader, val_loader, test_loader = get_dataloaders(args.dataset)

    # Here you need to instantiate your model according to the chosen model type
    # For example:
    if args.model == 'classifier':
        model = ResNet18_MedMNIST()  
        train_func = train_classifier
    elif args.model == 'detector':
        model = SimpleAutoencoder()  
        train_func = train_detector
    elif args.model == 'reformer' and args.reformer_type == "dae":
        model = DenoisingAutoEncoder()  
        train_func = train_reformer
    elif args.model == 'reformer' and args.reformer_type == "hiprnet":
        print("Using Hypernetwork Reformer")
        model = SMPPyramidDenoiser() # Im changing this each time, fix this logix later
        train_func = train_reformer_hipyrnet # have to make modifications to this or create a new function for hiprnet related trianers

    # Train the model
    trained_model = train_func(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr)
    save_model_path(trained_model, args.model)

if __name__ == "__main__":
    main()
