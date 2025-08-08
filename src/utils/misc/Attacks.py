import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchattacks import CW

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def pgd_attack(model, images, labels, epsilon, alpha, iters):
    ori_images = images.data
    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)
        model.zero_grad()
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-epsilon, max=epsilon)
        images = torch.clamp(ori_images + eta, 0, 1).detach_()
    return images


def carlini_attack(model, images, labels, c=1e-2, kappa=0, steps=1000, lr=0.01):
    # Assumes model is in eval() mode, images and labels are on the correct device
    attack = CW(model, c=c, kappa=kappa, steps=steps, lr=lr)
    adv_images = attack(images, labels)
    return adv_images