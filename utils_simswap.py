"""
Attack utility functions adapted for SimSwap.

Attack scenario:
  X_nat   : [0,1]-normalized img_att [B, 3, H, W]  (victim's face = appearance base, 섭동 대상)
  targets : list of ImageNet-normalized img_id tensors  (face to inject = identity source)
  model   : SimSwapWrapper

model(img_id=target, img_att=X_nat) → swapped face with target's identity on X_nat's appearance.
The attack maximizes MSE(G(target, X_adv), G(target, X_nat)) so the swap is degraded.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms as T
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.transform import resize

from color_space import rgb2lab, lab2rgb

# ── ImageNet normalization helpers ────────────────────────────────────────────

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

normalize_imagenet = T.Normalize(
    mean=[0.485, 0.456, 0.406],
    std =[0.229, 0.224, 0.225],
)


def denorm_imagenet(x: torch.Tensor) -> torch.Tensor:
    """ImageNet-normalized → [0, 1]"""
    mean = _IMAGENET_MEAN.to(x.device)
    std  = _IMAGENET_STD.to(x.device)
    return (x * std + mean).clamp(0, 1)


def norm_imagenet(x: torch.Tensor) -> torch.Tensor:
    """[0, 1] → ImageNet-normalized"""
    mean = _IMAGENET_MEAN.to(x.device)
    std  = _IMAGENET_STD.to(x.device)
    return (x - mean) / std


# ── Metric helper ─────────────────────────────────────────────────────────────

def compare(img1: torch.Tensor, img2: torch.Tensor):
    """Compute SSIM and PSNR between two [0,1] tensors."""
    img1_np = img1.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    img2_np = img2.squeeze(0).cpu().numpy().transpose(1, 2, 0)

    h, w, _ = img1_np.shape
    min_size = 7
    if h < min_size or w < min_size:
        img1_np = resize(img1_np, (max(min_size, h), max(min_size, w)), anti_aliasing=True)
        img2_np = resize(img2_np, (max(min_size, h), max(min_size, w)), anti_aliasing=True)

    win_size = min(img1_np.shape[0], img1_np.shape[1], 7)
    ssim = structural_similarity(img1_np, img2_np, channel_axis=-1,
                                 win_size=win_size, data_range=1.0)
    psnr = peak_signal_noise_ratio(img1_np, img2_np, data_range=1.0)
    return ssim, psnr


# ── Loss helper ───────────────────────────────────────────────────────────────

def _avg_loss_over_targets(X_adv, X_nat, targets, model, criterion):
    """Average MSE loss over all targets: mean_t[ MSE(G(t, X_adv), G(t, X_nat)) ]
    targets: ImageNet-normalized img_id (identity source)
    X_nat/X_adv: [0,1] img_att (appearance base)
    """
    loss = 0.0
    for target in targets:
        with torch.no_grad():
            gen_clean, _ = model(target, X_nat)
        gen_adv, _ = model(target, X_adv)
        loss = loss + criterion(gen_adv, gen_clean)
    return loss / len(targets)


# ── Attack functions ──────────────────────────────────────────────────────────

def lab_attack_simswap(X_nat, targets, model, epsilon=0.05, iter=100):
    """LAB-space iterative attack on SimSwap appearance base (img_att).

    X_nat   : [0,1] CelebA image (img_att, 섭동 대상)
    targets : ImageNet-normalized identity source images (img_id)
    """
    criterion = nn.MSELoss().cuda()
    pert_a = torch.zeros(X_nat.shape[0], 2, X_nat.shape[2], X_nat.shape[3]).cuda().requires_grad_(True)
    optimizer = torch.optim.Adam([pert_a], lr=1e-4, betas=(0.9, 0.999))

    X = X_nat  # already [0, 1]

    for i in range(iter):
        X_lab = rgb2lab(X).cuda()
        pert = torch.clamp(pert_a, min=-epsilon, max=epsilon)
        X_lab[:, 1:, :, :] = X_lab[:, 1:, :, :] + pert
        X_lab = torch.clamp(X_lab, min=-128, max=128)
        X_new = lab2rgb(X_lab)  # [0, 1]

        loss = -_avg_loss_over_targets(X_new, X_nat, targets, model, criterion)
        if torch.isnan(loss):
            print(f"Iteration {i}: NaN in loss. Stopping.")
            break

        optimizer.zero_grad()
        loss.backward()
        if torch.isnan(pert_a.grad).any():
            pert_a.grad = torch.nan_to_num(pert_a.grad, nan=0.0)
        optimizer.step()

    return X_new, X_new - X


def fgsm_lab_attack_simswap(X_nat, targets, model, epsilon=0.05):
    """Single-step FGSM in LAB space for SimSwap. X_nat is [0,1] img_att."""
    criterion = nn.MSELoss().cuda()
    X = X_nat  # already [0, 1]

    pert_a = torch.empty(
        X_nat.shape[0], 2, X_nat.shape[2], X_nat.shape[3]
    ).cuda().uniform_(-epsilon, epsilon).requires_grad_(True)

    X_lab = rgb2lab(X).cuda()
    pert = torch.clamp(pert_a, min=-epsilon, max=epsilon)
    X_lab[:, 1:, :, :] = X_lab[:, 1:, :, :] + pert
    X_lab = torch.clamp(X_lab, min=-128, max=128)
    X_new = lab2rgb(X_lab)  # [0, 1]

    loss = _avg_loss_over_targets(X_new, X_nat, targets, model, criterion)
    loss.backward()

    with torch.no_grad():
        pert_sign  = epsilon * pert_a.grad.sign()
        pert_final = torch.clamp(pert_a.detach() + pert_sign, min=-epsilon, max=epsilon)
        X_lab_final = rgb2lab(X).cuda()
        X_lab_final[:, 1:, :, :] = X_lab_final[:, 1:, :, :] + pert_final
        X_lab_final = torch.clamp(X_lab_final, min=-128, max=128)
        X_adv = lab2rgb(X_lab_final)  # [0, 1]

    return X_adv, X_adv - X


def pgd_lab_attack_simswap(X_nat, targets, model, epsilon=0.05, alpha=0.005, iter=40):
    """Multi-step PGD in LAB space for SimSwap. X_nat is [0,1] img_att."""
    criterion = nn.MSELoss().cuda()
    X = X_nat  # already [0, 1]

    pert_a = torch.zeros(X_nat.shape[0], 2, X_nat.shape[2], X_nat.shape[3]).cuda()

    for i in range(iter):
        pert_a = pert_a.detach().requires_grad_(True)

        X_lab = rgb2lab(X).cuda()
        X_lab[:, 1:, :, :] = X_lab[:, 1:, :, :] + pert_a
        X_lab = torch.clamp(X_lab, min=-128, max=128)
        X_new = lab2rgb(X_lab)  # [0, 1]

        loss = _avg_loss_over_targets(X_new, X_nat, targets, model, criterion)
        if torch.isnan(loss):
            print(f"Iteration {i}: NaN in loss. Stopping.")
            break
        loss.backward()

        with torch.no_grad():
            pert_a = pert_a + alpha * pert_a.grad.sign()
            pert_a = torch.clamp(pert_a, min=-epsilon, max=epsilon)

    with torch.no_grad():
        X_lab_final = rgb2lab(X).cuda()
        X_lab_final[:, 1:, :, :] = X_lab_final[:, 1:, :, :] + pert_a
        X_lab_final = torch.clamp(X_lab_final, min=-128, max=128)
        X_adv = lab2rgb(X_lab_final)  # [0, 1]

    return X_adv, X_adv - X


def fgsm_attack_simswap(X_nat, targets, model, epsilon=0.05):
    """Single-step FGSM in RGB space for SimSwap. X_nat is [0,1] img_att."""
    criterion = nn.MSELoss().cuda()
    X_adv = X_nat.clone().detach().requires_grad_(True)

    loss = _avg_loss_over_targets(X_adv, X_nat, targets, model, criterion)
    loss.backward()

    with torch.no_grad():
        X_adv = (X_nat + epsilon * X_adv.grad.sign()).clamp(0, 1)

    return X_adv, X_adv - X_nat


def pgd_attack_simswap(X_nat, targets, model, epsilon=0.05, alpha=0.005, iter=40):
    """Multi-step PGD in RGB space for SimSwap. X_nat is [0,1] img_att."""
    criterion = nn.MSELoss().cuda()
    X_adv = X_nat.clone().detach()

    for i in range(iter):
        X_adv = X_adv.detach().requires_grad_(True)

        loss = _avg_loss_over_targets(X_adv, X_nat, targets, model, criterion)
        if torch.isnan(loss):
            print(f"Iteration {i}: NaN in loss. Stopping.")
            break
        loss.backward()

        with torch.no_grad():
            X_adv = X_adv + alpha * X_adv.grad.sign()
            X_adv = torch.clamp(X_adv, X_nat - epsilon, X_nat + epsilon).clamp(0, 1)

    return X_adv, X_adv - X_nat
