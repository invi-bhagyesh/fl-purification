import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO
import torch.nn.functional as F
from torchattacks import CW
import shutil
from utils.misc.Attacks import fgsm_attack,pgd_attack,carlini_attack

def get_dataloaders(data_flag, batch_size=64, download=True):
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = DataClass(split='train', transform=transform, download=download)
    val_dataset = DataClass(split='val', transform=transform, download=download)
    test_dataset = DataClass(split='test', transform=transform, download=download)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def generate_adversarial_dataset(model, dataloader, attack_type, device, **attack_params):
    """
    Generate adversarially perturbed dataset using the specified attack.

    Args:
        model: Pretrained model for attacks.
        dataloader: DataLoader with original dataset.
        attack_type: String specifying attack - 'fgsm', 'pgd', or 'cw'.
        device: Device to run computations ('cuda' or 'cpu').
        attack_params: Parameters specific to the attack function.

    Returns:
        perturbed_loader: DataLoader with adversarially perturbed images and original labels.
    """
    model.eval()
    perturbed_images = []
    perturbed_labels = []
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device).long().squeeze()
        if attack_type == 'fgsm':
            images.requires_grad = True
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            model.zero_grad()
            loss.backward()
            data_grad = images.grad.data
            epsilon = attack_params.get('epsilon', 0.1)
            adv_images = fgsm_attack(images, epsilon, data_grad)
        elif attack_type == 'pgd':
            epsilon = attack_params.get('epsilon', 0.1)
            alpha = attack_params.get('alpha', 0.01)
            iters = attack_params.get('iters', 40)
            adv_images = pgd_attack(model, images, labels, epsilon, alpha, iters)
        elif attack_type == 'cw':
            c = attack_params.get('c', 1e-2)
            kappa = attack_params.get('kappa', 0)
            steps = attack_params.get('steps', 1000)
            lr = attack_params.get('lr', 0.01)
            adv_images = carlini_attack(model, images, labels, c, kappa, steps, lr)
        else:
            raise ValueError(f"Unsupported attack type: {attack_type}")

        perturbed_images.append(adv_images.cpu())
        perturbed_labels.append(labels.cpu())

    # Concatenate all batches
    perturbed_images = torch.cat(perturbed_images, dim=0)
    perturbed_labels = torch.cat(perturbed_labels, dim=0)

    perturbed_dataset = TensorDataset(perturbed_images, perturbed_labels)
    perturbed_loader = DataLoader(perturbed_dataset, batch_size=dataloader.batch_size, shuffle=False)

    return perturbed_loader

def save_perturbed_dataset(perturbed_loader, base_dir, dataset_name, attack_type, strength=None):
    """
    Save images and labels from the perturbed loader to the specified directory structure.
    """
    attack_folder = attack_type if not strength else f"{attack_type} {strength}"
    dir_path = os.path.join(base_dir, dataset_name, attack_folder)
    os.makedirs(dir_path, exist_ok=True)

    for batch_idx, (images, labels) in enumerate(perturbed_loader):
        save_path = os.path.join(dir_path, f"batch_{batch_idx}.pt")
        torch.save({'images': images, 'labels': labels}, save_path)

    print(f"Saved perturbed dataset under: {dir_path}")
    return dir_path


def zip_directory(dir_path, zip_output_path):
    """
    Zips the entire contents of dir_path into a zip file at zip_output_path.

    Args:
        dir_path (str): Directory to zip.
        zip_output_path (str): Full path including filename where zip will be saved.

    Returns:
        str: Path to the created zip file.
    """
    base_name = zip_output_path.replace('.zip', '')
    shutil.make_archive(base_name=base_name, format='zip', root_dir=dir_path)
    print(f"Files zipped successfully to {zip_output_path}")
    return zip_output_path

#example usage 
# train_loader, val_loader, test_loader = get_dataloaders('bloodmnist')
# perturbed_loader = generate_adversarial_dataset(model, test_loader, 'cw', DEVICE, c=1e-1, kappa=5, steps=1000, lr=0.005)
# save_perturbed_dataset(perturbed_loader, base_dir='medmnist', dataset_name='bloodmnist', attack_type='cw', strength='strong')
# zip_file = zip_directory('/kaggle/working', '/kaggle/working/all_files.zip')