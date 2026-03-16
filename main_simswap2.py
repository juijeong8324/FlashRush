"""
Anti-forgery adversarial attack against SimSwap (face swap model).
MPI distributed version (main_simswap2.py) — based on main2.py pattern.

Attack scenario:
  - CelebA 이미지 (img_id) = 섭동 추가 대상 (victim의 identity photo)
  - target 이미지 (img_att) = 고정된 target body (e.g. 6.jpg)
  - 공격: ArcFace가 CelebA에서 잘못된 identity 추출 → swap 결과 degraded

Usage example:
  mpirun -np 2 python main_simswap2.py \
      --arc_path SimSwap/arcface_model/arcface_checkpoint.tar \
      --G_path   SimSwap/checkpoints/people/latest_net_G.pth \
      --celeba_image_dir ./data/celeba/images \
      --attr_path        ./data/celeba/list_attr_celeba.txt \
      --result_dir       ./results_simswap2 \
      --num_id_images    50 \
      --target_image     SimSwap/crop_224/6.jpg \
      --attack_iters     100 \
      --attack_type      lab
"""

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms as T
from torchvision.utils import save_image
from PIL import Image
from torch.utils import data
from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

from utils_simswap import (
    lab_attack_simswap2,
    denorm_imagenet,
    normalize_imagenet,
    compare,
)
from model.simswap_wrapper import SimSwapWrapper

# ── Data loader ───────────────────────────────────────────────────────────────

class CelebASimSwap(data.Dataset):
    """CelebA loader for SimSwap.

    Returns:
        img_id_norm : ImageNet-normalized [3, H, W]  (identity source, 섭동 대상)
        img_id_01   : [0, 1]-normalized   [3, H, W]  (same image, for visualization)

    Same shuffle seed as the original CelebA loader for consistent test split.
    """
    def __init__(self, image_dir, attr_path, image_size=224, mode='test'):
        self.image_dir = image_dir
        self.mode = mode

        self.transform_norm = T.Compose([
            T.CenterCrop(178),
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.transform_01 = T.Compose([
            T.CenterCrop(178),
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

        self.dataset = []
        self._preprocess(attr_path)

    def _preprocess(self, attr_path):
        import random
        lines = [line.rstrip() for line in open(attr_path, 'r')]
        filenames = [line.split()[0] for line in lines[2:]]
        random.seed(1234)
        random.shuffle(filenames)
        self.dataset = filenames[:2000] if self.mode == 'test' else filenames[2000:]
        if rank == 0:
            print(f'CelebASimSwap: loaded {len(self.dataset)} images ({self.mode})')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.image_dir, self.dataset[index])).convert('RGB')
        return self.transform_norm(img), self.transform_01(img)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()

    # Model paths
    parser.add_argument('--arc_path', type=str,
                        default='SimSwap/arcface_model/arcface_checkpoint.tar')
    parser.add_argument('--G_path', type=str,
                        default='SimSwap/checkpoints/people/latest_net_G.pth')
    parser.add_argument('--crop_size', type=int, default=224, choices=[224, 512])

    # Data
    parser.add_argument('--celeba_image_dir', type=str, default='./data/celeba/images')
    parser.add_argument('--attr_path', type=str,
                        default='./data/celeba/list_attr_celeba.txt')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--num_workers', type=int, default=0)

    # Attack
    parser.add_argument('--attack_iters', type=int, default=100)
    parser.add_argument('--epsilon', type=float, default=2.0)
    parser.add_argument('--num_id_images', type=int, default=50,
                        help='Number of CelebA identity images to attack')
    parser.add_argument('--num_target_images', type=int, default=1,
                        help='Number of target body images (used if --target_image not set)')
    parser.add_argument('--target_image', type=str, default=None,
                        help='Path to a single target body image (img_att)')

    # Output
    parser.add_argument('--result_dir', type=str, default='./results_simswap2')

    config = parser.parse_args()
    os.makedirs(config.result_dir, exist_ok=True)

    # ── CUDA 할당 ──────────────────────────────────────────────────────────────
    num_gpus = torch.cuda.device_count()
    gpu_id = rank % num_gpus
    torch.cuda.set_device(gpu_id)

    # ── Load SimSwap model ─────────────────────────────────────────────────────
    if rank == 0:
        print(f'Loading SimSwap... (ArcFace: {config.arc_path}, G: {config.G_path})')
    t = time.time()
    model = SimSwapWrapper.load(config.arc_path, config.G_path, config.crop_size)
    model.to(gpu_id).eval()
    print(f'Rank {rank} using GPU {gpu_id}: Model loaded in {time.time()-t:.1f}s')

    # ── Load dataset ───────────────────────────────────────────────────────────
    t = time.time()
    dataset = CelebASimSwap(
        config.celeba_image_dir, config.attr_path,
        image_size=config.image_size, mode='test',
    )

    # CelebA = identity source (img_id), ImageNet-normalized
    id_subset = data.Subset(dataset, range(config.num_id_images))
    id_loader = data.DataLoader(id_subset, batch_size=1, shuffle=False,
                                num_workers=config.num_workers)

    # Target body (img_att): [0,1]
    if config.target_image is not None:
        transform_att = T.Compose([
            T.Resize((config.image_size, config.image_size)),
            T.ToTensor(),
        ])
        target_img = transform_att(Image.open(config.target_image).convert('RGB'))
        targets = [target_img.unsqueeze(0).cuda(gpu_id)]
    else:
        target_subset = data.Subset(dataset, range(config.num_id_images,
                                                   config.num_id_images + config.num_target_images))
        target_loader = data.DataLoader(target_subset, batch_size=1, shuffle=False,
                                        num_workers=config.num_workers)
        targets = [img_01.cuda(gpu_id) for _, img_01 in target_loader]

    if rank == 0:
        print(f'Dataset loaded in {time.time()-t:.1f}s, '
              f'{len(targets)} target body image(s) collected')

    # ── Attack loop ────────────────────────────────────────────────────────────
    # Metrics:
    #   l2_error     : MSE(gen_clean, gen_adv)                    — output distortion (↑ better)
    #   ssim/psnr    : SSIM/PSNR(gen_clean, gen_adv)              — output distortion (↓ better)
    #   id_sim_clean : cosine_sim(arc(gen_clean), arc(img_id))    — baseline identity transfer
    #   id_sim_adv   : cosine_sim(arc(gen_adv),   arc(img_id))    — identity after attack (↓ better)
    #   id_drop      : id_sim_clean - id_sim_adv                  — identity disruption (↑ better)
    #   ASR(MSE)     : % where MSE(gen_clean, gen_adv) > 0.05     — pixel-level distortion rate
    #   ASR(ID)      : % where id_sim_adv < 0.5                   — identity disruption rate
    l2_error = 0.0
    ssim_total, psnr_total = 0.0, 0.0
    id_sim_clean_total, id_sim_adv_total = 0.0, 0.0
    n_samples, n_asr_mse, n_asr_id = 0, 0, 0
    ASR_MSE_THRESHOLD = 0.05
    ASR_ID_THRESHOLD  = 0.5

    t_loop = time.time()

    for i, (img_id, _) in enumerate(id_loader):
        # Distribute tasks by rank
        if i % size != rank:
            continue

        print(f'Rank {rank} using GPU {gpu_id}: Image {i}')

        img_id = img_id.cuda(gpu_id)  # ImageNet-normalized CelebA (img_id)

        t = time.time()
        img_adv, _ = lab_attack_simswap2(img_id, targets, model, gpu_id, epsilon=config.epsilon, iter=config.attack_iters)

        print(f'Rank {rank} GPU {gpu_id}: [{i+1}] {config.attack_type.upper()} attack: {time.time()-t:.1f}s')

        # ── Precompute ArcFace features for img_id and img_adv ─────────────────
        with torch.no_grad():
            img_id_112  = F.interpolate(img_id,  size=(112, 112), mode='bilinear', align_corners=False)
            img_adv_112 = F.interpolate(img_adv, size=(112, 112), mode='bilinear', align_corners=False)
            feat_id  = model.netArc(img_id_112)
            feat_adv = model.netArc(img_adv_112)
            feat_id  = feat_id  / feat_id.norm(dim=1, keepdim=True)
            feat_adv = feat_adv / feat_adv.norm(dim=1, keepdim=True)

        # ── Evaluate across target bodies ───────────────────────────────────────
        # Result grid: [img_id | img_adv | target | gen_clean | gen_adv]
        frames = [denorm_imagenet(img_id), denorm_imagenet(img_adv)]

        for target in targets:
            with torch.no_grad():
                gen_clean, _ = model(img_id,  target)   # [0, 1]
                gen_adv,   _ = model(img_adv, target)   # [0, 1]

                # Identity similarity: how much of img_id's identity survived in swap
                gen_clean_norm = normalize_imagenet(gen_clean.clamp(0, 1))
                gen_adv_norm   = normalize_imagenet(gen_adv.clamp(0, 1))
                gen_clean_112 = F.interpolate(gen_clean_norm, size=(112, 112), mode='bilinear', align_corners=False)
                gen_adv_112   = F.interpolate(gen_adv_norm,   size=(112, 112), mode='bilinear', align_corners=False)
                feat_clean_out = model.netArc(gen_clean_112)
                feat_adv_out   = model.netArc(gen_adv_112)
                feat_clean_out = feat_clean_out / feat_clean_out.norm(dim=1, keepdim=True)
                feat_adv_out   = feat_adv_out   / feat_adv_out.norm(dim=1, keepdim=True)

                id_sim_clean = (feat_id * feat_clean_out).sum(dim=1).item()
                id_sim_adv   = (feat_id * feat_adv_out).sum(dim=1).item()

            frames.extend([target, gen_clean, gen_adv])

            mse_val = F.mse_loss(gen_adv, gen_clean).item()
            l2_error       += mse_val
            ssim_v, psnr_v  = compare(gen_adv.clamp(0, 1), gen_clean.clamp(0, 1))
            ssim_total     += ssim_v
            psnr_total     += psnr_v
            id_sim_clean_total += id_sim_clean
            id_sim_adv_total   += id_sim_adv
            if mse_val > ASR_MSE_THRESHOLD:
                n_asr_mse += 1
            if id_sim_adv < ASR_ID_THRESHOLD:
                n_asr_id += 1
            n_samples += 1

        # Save: [img_id | img_adv | target | gen_clean | gen_adv]
        x_concat = torch.cat(frames, dim=3)
        save_path = os.path.join(config.result_dir, f'{i+1:04d}-images.jpg')
        save_image(x_concat.data.cpu().clamp(0, 1), save_path, nrow=1, padding=0)

    print(f'Rank {rank} GPU {gpu_id}: Attack loop done in {time.time()-t_loop:.1f}s')

    # ── Merge Metrics via MPI ──────────────────────────────────────────────────
    t_mpi = time.time()
    total_l2_error        = comm.reduce(l2_error,          op=MPI.SUM, root=0)
    total_ssim            = comm.reduce(ssim_total,        op=MPI.SUM, root=0)
    total_psnr            = comm.reduce(psnr_total,        op=MPI.SUM, root=0)
    total_id_sim_clean    = comm.reduce(id_sim_clean_total,op=MPI.SUM, root=0)
    total_id_sim_adv      = comm.reduce(id_sim_adv_total,  op=MPI.SUM, root=0)
    total_n_samples       = comm.reduce(n_samples,         op=MPI.SUM, root=0)
    total_n_asr_mse       = comm.reduce(n_asr_mse,         op=MPI.SUM, root=0)
    total_n_asr_id        = comm.reduce(n_asr_id,          op=MPI.SUM, root=0)

    print(f'Rank {rank}: MPI communication done in {time.time()-t_mpi:.1f}s')

    if rank == 0:
        n = total_n_samples
        print(f'\n{n} pairs | '
              f'L2: {total_l2_error/n:.4f} | '
              f'SSIM: {total_ssim/n:.4f} | '
              f'PSNR: {total_psnr/n:.2f} | '
              f'ID sim (clean): {total_id_sim_clean/n:.4f} | '
              f'ID sim (adv): {total_id_sim_adv/n:.4f} | '
              f'ID drop: {(total_id_sim_clean-total_id_sim_adv)/n:.4f} | '
              f'ASR(MSE): {total_n_asr_mse/n:.3f} | '
              f'ASR(ID): {total_n_asr_id/n:.3f}')


if __name__ == '__main__':
    start = time.time()
    main()
    print(f'Rank {rank}: Total {time.time()-start:.1f}s')
