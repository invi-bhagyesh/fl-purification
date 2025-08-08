import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class AdversarialDataset(Dataset):
    def __init__(self, base_dir, dataset_name, attack_type, strength=None):
        """
        Initializes dataset by scanning the directory with saved batches.
        Automatically prepends the Kaggle input root path.

        Args:
            base_dir (str): Base directory (e.g., "medmnist", "Others")
            dataset_name (str): Dataset name (e.g., "bloodmnist")
            attack_type (str): Attack type string ('fgsm', 'pgd', 'cw')
            strength (str or None): strength ('weak', 'strong')
        """
        # Set the Kaggle input root - fixed dataset root folder in Kaggle environment
        kaggle_input_root = '/kaggle/input/purification'  # Set your Kaggle dataset root here

        attack_folder = attack_type if not strength else f"{attack_type} {strength}"

        # Compose the full directory path by joining Kaggle root with the rest
        self.dir_path = os.path.join(kaggle_input_root, base_dir, dataset_name, attack_folder)

        if not os.path.exists(self.dir_path):
            raise FileNotFoundError(f"Directory not found: {self.dir_path}")

        # List all .pt batch files sorted by batch index
        self.batch_files = sorted([
            f for f in os.listdir(self.dir_path)
            if f.endswith('.pt')
        ])

        # Preload all batches into memory for quick access (optional)
        # If dataset is very large, consider lazy loading in __getitem__ instead
        self.data = []
        for batch_file in self.batch_files:
            batch_path = os.path.join(self.dir_path, batch_file)
            batch_data = torch.load(batch_path)  # dict with 'images' and 'labels'
            self.data.append(batch_data)

        self.num_batches = len(self.data)

    def __len__(self):
        """ Returns the number of batches available """
        return self.num_batches

    def __getitem__(self, idx):
        """
        Returns the batch at index idx as a tuple (images, labels).

        Args:
            idx (int): batch index

        Returns:
            tuple: images tensor and labels tensor for the batch
        """
        if idx < 0 or idx >= self.num_batches:
            raise IndexError(f"Index {idx} out of range for {self.num_batches} batches")

        batch_data = self.data[idx]
        images = batch_data['images']
        labels = batch_data['labels']

        return images, labels
