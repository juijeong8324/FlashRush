import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from skimage.transform import resize

from color_space import *
from mpi4py import MPI
import time


def load_model_weights(model, path):
    pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage, weights_only=True)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'preprocessing' not in k}
    model_dict.update(pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)

def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def label2onehot(labels, dim):
    """Convert label indices to one-hot vectors."""
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    return out


def create_labels(c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
    """Generate target domain labels for debugging and testing."""
    # Get hair color indices.
    if dataset == 'CelebA':
        hair_color_indices = []
        for i, attr_name in enumerate(selected_attrs):
            if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                hair_color_indices.append(i)

    c_trg_list = []
    for i in range(c_dim):
        if dataset == 'CelebA':
            c_trg = c_org.clone()
            if i in hair_color_indices:
                c_trg[:, i] = 1
                for j in hair_color_indices:
                    if j != i:
                        c_trg[:, j] = 0
            else:
                c_trg[:, i] = (c_trg[:, i] == 0)
        elif dataset == 'RaFD':
            c_trg = label2onehot(torch.ones(c_org.size(0)) * i, c_dim)

        c_trg_list.append(c_trg.cuda())
    return c_trg_list

def compare(img1, img2):
    """input tensor, translate to np.array"""
    img1_np = img1.squeeze(0).cpu().numpy()
    img2_np = img2.squeeze(0).cpu().numpy()
    img1_np = np.transpose(img1_np, (1, 2, 0))
    img2_np = np.transpose(img2_np, (1, 2, 0))

    h, w, c = img1_np.shape
    min_size = 7
    if h < min_size or w < min_size:
        img1_np = resize(img1_np, (max(min_size, h), max(min_size, w)), anti_aliasing=True)
        img2_np = resize(img2_np, (max(min_size, h), max(min_size, w)), anti_aliasing=True)

    win_size = min(img1_np.shape[0], img1_np.shape[1], 7)

    try:
        ssim = structural_similarity(img1_np, img2_np, channel_axis=-1, win_size=win_size, data_range=1.0)
        psnr = peak_signal_noise_ratio(img1_np, img2_np, data_range=1.0)
    except ValueError as e:
        print(f"Error in SSIM calculation: {e}")
        raise
    return ssim, psnr


def lab_attack(X_nat, c_trg, model, epsilon=0.05, iter=100):

    criterion = nn.MSELoss().cuda()
    pert_a = torch.zeros(X_nat.shape[0], 2, X_nat.shape[2], X_nat.shape[3]).cuda().requires_grad_()
    optimizer = torch.optim.Adam([pert_a], lr=1e-4, betas=(0.9, 0.999))

    X = denorm(X_nat.clone())

    for i in range(iter):
        X_lab = rgb2lab(X).cuda()
        pert = torch.clamp(pert_a, min=-epsilon, max=epsilon)
        X_lab[:, 1:, :, :] = X_lab[:, 1:, :, :] + pert
        X_lab = torch.clamp(X_lab, min=-128, max=128)
        X_new = T.Normalize(mean=[0.5, 0.5, 0.5], std=[
                            0.5, 0.5, 0.5])(lab2rgb(X_lab))

        with torch.no_grad():
            gen_noattack, gen_feats_noattack = model(
                X_nat, c_trg[i % len(c_trg)])

        gen_stargan, gen_feats_stargan = model(X_new, c_trg[i % 5])

        loss = -criterion(gen_stargan, gen_noattack)
        if torch.isnan(loss):
            print(f"Iteration {i}: NaN detected in loss. Exiting loop.")
            break

        optimizer.zero_grad()
        loss.backward()
        if torch.isnan(pert_a.grad).any():
            pert_a.grad = torch.nan_to_num(pert_a.grad, nan=0.0)
        optimizer.step()
    return X_new, X_new - X


def lab_attack2(X_nat, c_trg, device, model, epsilon=0.05, iter=100):
    criterion = nn.MSELoss().cuda()
    pert_a = torch.zeros(X_nat.shape[0], 2, X_nat.shape[2], X_nat.shape[3]).cuda().requires_grad_()
    optimizer = torch.optim.Adam([pert_a], lr=1e-4, betas=(0.9, 0.999))

    X = denorm(X_nat.clone())

    for i in range(iter):
        X_lab = rgb2lab(X).to(device)
        pert = torch.clamp(pert_a, min=-epsilon, max=epsilon)
        X_lab[:, 1:, :, :] = X_lab[:, 1:, :, :] + pert
        X_lab = torch.clamp(X_lab, min=-128, max=128)
        X_new = T.Normalize(mean=[0.5, 0.5, 0.5], std=[
                            0.5, 0.5, 0.5])(lab2rgb(X_lab))


        with torch.no_grad():
            gen_noattack, gen_feats_noattack = model(
                X_nat, c_trg[i % len(c_trg)])

        gen_stargan, gen_feats_stargan = model(X_new, c_trg[i % 5])

        loss = -criterion(gen_stargan, gen_noattack)
        if torch.isnan(loss):
            print(f"Iteration {i}: NaN detected in loss. Exiting loop.")
            break

        optimizer.zero_grad()
        loss.backward()
        if torch.isnan(pert_a.grad).any():
            pert_a.grad = torch.nan_to_num(pert_a.grad, nan=0.0)
        torch.nn.utils.clip_grad_norm_([pert_a], max_norm=1.0)
        optimizer.step()

    return X_new, X_new - X


def lab_attack3(X_nat, c_trg, device, model, epsilon=0.05, iter=100):
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    criterion = nn.MSELoss().to(device)
    pert_a = torch.zeros(X_nat.shape[0], 2, X_nat.shape[2], X_nat.shape[3]).to(
        device).requires_grad_()

    optimizer = torch.optim.Adam([pert_a], lr=1e-4, betas=(0.9, 0.999))

    X = denorm(X_nat.clone()).to(device)
    for i in range(iter):
        if i % size != rank:
            continue
        X_lab = rgb2lab(X).to(device)
        pert = torch.clamp(pert_a, min=-epsilon, max=epsilon)
        X_lab[:, 1:, :, :] = X_lab[:, 1:, :, :] + pert
        X_lab = torch.clamp(X_lab, min=-128, max=128)
        X_new = T.Normalize(mean=[0.5, 0.5, 0.5], std=[
                            0.5, 0.5, 0.5])(lab2rgb(X_lab))

        with torch.no_grad():
            gen_noattack, gen_feats_noattack = model(
                X_nat, c_trg[i % len(c_trg)])

        gen_stargan, gen_feats_stargan = model(X_new, c_trg[i % 5])

        loss = -criterion(gen_stargan, gen_noattack)
        if torch.isnan(loss):
            print(f"Iteration {i}: NaN detected in loss. Exiting loop.")
            break

        optimizer.zero_grad()
        loss.backward()
        if torch.isnan(pert_a.grad).any():
            pert_a.grad = torch.nan_to_num(pert_a.grad, nan=0.0)
        torch.nn.utils.clip_grad_norm_([pert_a], max_norm=1.0)
        optimizer.step()

    # --- Share final pert_a from all nodes and compute average ---
    pert_a_cpu = pert_a.detach().cpu().numpy()
    if rank == 0:
        all_pert = np.empty((size, *pert_a_cpu.shape), dtype=pert_a_cpu.dtype)
    else:
        all_pert = None
    t_mpi = time.time()
    t_gather = time.time()
    comm.Gather(pert_a_cpu, all_pert, root=0)
    print(f"************MPI Gather rank{rank}: {time.time() - t_gather}")

    # Compute average perturbation on Rank 0 and broadcast to all nodes
    t_bcast = time.time()
    if rank == 0:
        avg_pert = np.mean(all_pert, axis=0)
        print("\n--- Averaged Perturbation on Rank 0 ---")
        print(avg_pert)
    else:
        avg_pert = None
    avg_pert = comm.bcast(avg_pert, root=0)
    print(f"************MPI Bcast rank{rank}: {time.time() - t_bcast}")
    print(f"************MPI 전체 overhead rank{rank}: {time.time() - t_mpi}")
    avg_pert = torch.tensor(avg_pert).to(device)

    # Reapply average perturbation to X_nat to compute final perturbed image
    X_lab_final = rgb2lab(X).to(device)
    pert_final = torch.clamp(avg_pert, min=-epsilon, max=epsilon)
    X_lab_final[:, 1:, :, :] = X_lab_final[:, 1:, :, :] + pert_final
    X_new_final = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(
        lab2rgb(X_lab_final, device))

    return X_new_final, X_new_final - X
