"""
Attack utility functions adapted for SimSwap.

Attack scenario:
  X_nat   : ImageNet-normalized img_id [B, 3, H, W]  (victim's identity photo = CelebA, 섭동 대상)
  targets : list of [0,1]-normalized img_att tensors  (fixed target body, e.g. 6.jpg)
  model   : SimSwapWrapper

model(img_id=X_nat, img_att=target) → CelebA identity injected onto target body.
The attack maximizes MSE(G(X_adv, target), G(X_nat, target)) to confuse ArcFace identity extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms as T
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.transform import resize

from color_space import rgb2lab, lab2rgb
from mpi4py import MPI
import time

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
    """Average MSE loss over all targets: mean_t[ MSE(G(X_adv,t), G(X_nat,t)) ]
    X_nat/X_adv : ImageNet-normalized img_id (CelebA, 섭동 대상)
    targets     : [0,1] img_att (target body)
    """
    loss = 0.0
    for target in targets:
        with torch.no_grad():
            gen_clean, _ = model(X_nat, target)
        gen_adv, _ = model(X_adv, target)
        loss = loss + criterion(gen_adv, gen_clean)
    return loss / len(targets)


# ── Attack functions ──────────────────────────────────────────────────────────

def lab_attack_simswap(X_nat, targets, model, epsilon=0.05, iter=100):
    """LAB-space iterative attack on SimSwap identity source (img_id = CelebA).

    X_nat   : ImageNet-normalized CelebA (img_id)
    targets : [0,1] target body images (img_att)
    """
    criterion = nn.MSELoss().cuda()
    pert_a = torch.zeros(X_nat.shape[0], 2, X_nat.shape[2], X_nat.shape[3]).cuda().requires_grad_(True)
    optimizer = torch.optim.Adam([pert_a], lr=1e-3, betas=(0.9, 0.999))

    X = denorm_imagenet(X_nat)   # [0, 1]

    for i in range(iter):
        X_lab = rgb2lab(X).cuda()
        pert = torch.clamp(pert_a, min=-epsilon, max=epsilon)
        X_lab[:, 1:, :, :] = X_lab[:, 1:, :, :] + pert
        X_lab = torch.clamp(X_lab, min=-128, max=128)
        X_new_01 = lab2rgb(X_lab)
        X_new = norm_imagenet(X_new_01)

        loss = -_avg_loss_over_targets(X_new, X_nat, targets, model, criterion)
        if torch.isnan(loss):
            print(f"Iteration {i}: NaN in loss. Stopping.")
            break

        optimizer.zero_grad()
        loss.backward()
        if torch.isnan(pert_a.grad).any():
            pert_a.grad = torch.nan_to_num(pert_a.grad, nan=0.0)
        optimizer.step()

    return X_new, X_new_01 - X

def lab_attack_simswap2(X_nat, targets, model, device, epsilon=0.05, iter=100):
    criterion = nn.MSELoss().to(device)
    pert_a = torch.zeros(X_nat.shape[0], 2, X_nat.shape[2], X_nat.shape[3]).to(device).requires_grad_(True)
    optimizer = torch.optim.Adam([pert_a], lr=1e-3, betas=(0.9, 0.999))

    X = denorm_imagenet(X_nat)   # [0, 1]

    for i in range(iter):
        X_lab = rgb2lab(X).to(device)
        pert = torch.clamp(pert_a, min=-epsilon, max=epsilon)
        X_lab[:, 1:, :, :] = X_lab[:, 1:, :, :] + pert
        X_lab = torch.clamp(X_lab, min=-128, max=128)
        X_new_01 = lab2rgb(X_lab)
        X_new = norm_imagenet(X_new_01)

        loss = -_avg_loss_over_targets(X_new, X_nat, targets, model, criterion)
        if torch.isnan(loss):
            print(f"Iteration {i}: NaN in loss. Stopping.")
            break

        optimizer.zero_grad()
        loss.backward()
        if torch.isnan(pert_a.grad).any():
            pert_a.grad = torch.nan_to_num(pert_a.grad, nan=0.0)
        torch.nn.utils.clip_grad_norm_([pert_a], max_norm=1.0)
        optimizer.step()

    return X_new, X_new_01 - X


def lab_attack_simswap3(X_nat, targets, model, device, epsilon=0.05, iter=100):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    criterion = nn.MSELoss().to(device)
    pert_a = torch.zeros(X_nat.shape[0], 2, X_nat.shape[2], X_nat.shape[3]).to(device).requires_grad_(True)
    optimizer = torch.optim.Adam([pert_a], lr=1e-3, betas=(0.9, 0.999))

    X = denorm_imagenet(X_nat)   # [0, 1]

    for i in range(iter):
        if i % size != rank:
            continue

        X_lab = rgb2lab(X).to(device)
        pert = torch.clamp(pert_a, min=-epsilon, max=epsilon)
        X_lab[:, 1:, :, :] = X_lab[:, 1:, :, :] + pert
        X_lab = torch.clamp(X_lab, min=-128, max=128)
        X_new_01 = lab2rgb(X_lab)
        X_new = norm_imagenet(X_new_01)

        loss = -_avg_loss_over_targets(X_new, X_nat, targets, model, criterion)
        if torch.isnan(loss):
            print(f"Iteration {i}: NaN in loss. Stopping.")
            break

        optimizer.zero_grad()
        loss.backward()
        if torch.isnan(pert_a.grad).any():
            pert_a.grad = torch.nan_to_num(pert_a.grad, nan=0.0)
        torch.nn.utils.clip_grad_norm_([pert_a], max_norm=1.0)
        optimizer.step()

    # ── MPI Gather → sum → Bcast ───────────────────────────────────────────────
    pert_a_cpu = pert_a.detach().cpu().numpy()
    all_pert = np.empty((size, *pert_a_cpu.shape), dtype=pert_a_cpu.dtype) if rank == 0 else None

    t_gather = time.time()
    comm.Gather(pert_a_cpu, all_pert, root=0)
    print(f"MPI Gather rank{rank}: {time.time()-t_gather:.3f}s")

    t_bcast = time.time()
    avg_pert = np.sum(all_pert, axis=0) if rank == 0 else None
    avg_pert = comm.bcast(avg_pert, root=0)
    print(f"MPI Bcast  rank{rank}: {time.time()-t_bcast:.3f}s")

    avg_pert = torch.tensor(avg_pert).to(device)

    # Reapply final averaged perturbation
    X_lab_final = rgb2lab(X).to(device)
    pert_final = torch.clamp(avg_pert, min=-epsilon, max=epsilon)
    X_lab_final[:, 1:, :, :] = X_lab_final[:, 1:, :, :] + pert_final
    X_lab_final = torch.clamp(X_lab_final, min=-128, max=128)
    X_new_01_final = lab2rgb(X_lab_final)
    X_new_final = norm_imagenet(X_new_01_final)

    return X_new_final, X_new_01_final - X

