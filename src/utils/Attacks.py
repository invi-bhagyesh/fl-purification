import torch
import torch.nn as nn
import torch.nn.functional as F

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