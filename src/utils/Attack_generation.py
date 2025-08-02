clean_images = []
clean_pert_labels = []
clean_true_labels = []
adversarial_images = []
adv_pert_labels = []
adv_true_labels = []

model.eval()

for images, labels in train_loader:
    images, labels = images.to(DEVICE), labels.to(DEVICE)
    batch_size = images.size(0)
    # Compute number of adversarial (5% of batch, at least 1)
    adv_count = max(1, int(batch_size * 0.05))
    # Clean samples: all except last 5%
    clean_part = images[:batch_size - adv_count]
    clean_pert_labels_batch = torch.zeros(clean_part.size(0), dtype=torch.long, device=DEVICE)
    clean_true_labels_batch = labels[:batch_size - adv_count].squeeze()
    
    # Carlini adversarial samples: last 5% only
    carlini_imgs = images[batch_size - adv_count:].clone().detach()
    carlini_labels_src = labels[batch_size - adv_count:].squeeze()
    carlini_adv = carlini_attack(model, carlini_imgs, carlini_labels_src, c=1e-1, kappa=5, steps=1000, lr=0.01)
    carlini_pert_labels = torch.ones(carlini_adv.size(0), dtype=torch.long, device=DEVICE)
    carlini_true_labels = carlini_labels_src
    
    # Collect results
    clean_images.append(clean_part)
    clean_pert_labels.append(clean_pert_labels_batch)
    clean_true_labels.append(clean_true_labels_batch)
    adversarial_images.append(carlini_adv)
    adv_pert_labels.append(carlini_pert_labels)
    adv_true_labels.append(carlini_true_labels)

# Concatenate everything for the full dataset
all_images = torch.cat(clean_images + adversarial_images, dim=0)
all_pert_labels = torch.cat(clean_pert_labels + adv_pert_labels, dim=0)
all_true_labels = torch.cat(clean_true_labels + adv_true_labels, dim=0)

# Triple dataset: (image, pert_0or1, true_class)
from torch.utils.data import TensorDataset, DataLoader
combined_dataset = TensorDataset(all_images, all_pert_labels, all_true_labels)
combined_loader = DataLoader(combined_dataset, batch_size=64, shuffle=True)
