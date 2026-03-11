import torch
import numpy as np
import argparse
import os
from torchvision.utils import save_image
import torch
import torch.nn.functional as F

from color_space import *
from data_loader import get_loader
from utils import *

from model.stargan import Generator, Discriminator
import time

def main():
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--c_dim', type=int, default=5)
    parser.add_argument('--c2_dim', type=int, default=8)
    parser.add_argument('--celeba_crop_size', type=int, default=178)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)
    parser.add_argument('--g_repeat_num', type=int, default=6)
    parser.add_argument('--d_repeat_num', type=int, default=6)

    # Data configuration.
    parser.add_argument('--batch_size', type=int, default=2, help='mini-batch size')
    parser.add_argument('--attack_iters', type=int, default=100)
    parser.add_argument('--attack_type', type=str, default='lab', choices=['lab', 'fgsm', 'pgd', 'fgsm_rgb', 'pgd_rgb'])

    parser.add_argument('--resume_iters', type=int, default=200000)
    parser.add_argument('--selected_attrs', '--list', nargs='+',
                        default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])
    parser.add_argument('--test_iters', type=int, default=200000)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])

    parser.add_argument('--celeba_image_dir', type=str, default='./data/celeba/images')
    parser.add_argument('--attr_path', type=str, default='./data/celeba/list_attr_celeba.txt')
    parser.add_argument('--model_save_dir', type=str, default='stargan_celeba_256/models')
    parser.add_argument('--result_dir', type=str, default='results')
    parser.add_argument('--num_images', type=int, default=50, help='number of images to process')

    config = parser.parse_args()
    os.makedirs(config.result_dir, exist_ok=True)

    t = time.time()
    celeba_loader = get_loader(config.celeba_image_dir, config.attr_path, config.selected_attrs,
                               config.celeba_crop_size, config.image_size, config.batch_size,
                               'CelebA', config.mode, config.num_workers)
    print(f"************Load dataset: {time.time() - t}")

    t = time.time()
    G = Generator(config.g_conv_dim, config.c_dim, config.g_repeat_num)
    D = Discriminator(config.image_size, config.d_conv_dim, config.c_dim, config.d_repeat_num)
    G.cuda()
    D.cuda()
    print('Loading the trained models from step {}...'.format(config.resume_iters))
    G_path = os.path.join(config.model_save_dir, '{}-G.ckpt'.format(config.resume_iters))
    D_path = os.path.join(config.model_save_dir, '{}-D.ckpt'.format(config.resume_iters))
    load_model_weights(G, G_path)
    D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
    print("loading model successful")
    print(f"************Generation Model: {time.time() - t}")

    l2_error, ssim, psnr = 0.0, 0.0, 0.0
    n_samples, n_dist = 0, 0
    n_images_processed = 0
    t_loop = time.time()

    for i, (x_real, c_org) in enumerate(celeba_loader):
        if n_images_processed >= config.num_images:
            break

        x_real = x_real.cuda()
        c_trg_list = create_labels(c_org, config.c_dim, 'CelebA', config.selected_attrs)

        # generate adv in lab space (batch 단위)
        t = time.time()
        if config.attack_type == 'fgsm':
            x_adv, pert = fgsm_lab_attack(x_real, c_trg_list, G, epsilon=0.05)
        elif config.attack_type == 'pgd':
            x_adv, pert = pgd_lab_attack(x_real, c_trg_list, G, epsilon=0.05)
        elif config.attack_type == 'fgsm_rgb':
            x_adv, pert = fgsm_attack(x_real, c_trg_list, G, epsilon=0.05)
        elif config.attack_type == 'pgd_rgb':
            x_adv, pert = pgd_attack(x_real, c_trg_list, G, epsilon=0.05)
        else:
            x_adv, pert = lab_attack(x_real, c_trg_list, G, iter=config.attack_iters)
        print(f"************{config.attack_type.upper()} Attack batch {i} ({x_real.shape[0]}장): {time.time() - t}")

        # 배치 내 각 이미지별로 메트릭/저장 처리
        B = x_real.shape[0]
        for b in range(B):
            if n_images_processed >= config.num_images:
                break

            x_real_b = x_real[b:b+1]
            x_adv_b  = x_adv[b:b+1]
            c_trg_list_b = [c[b:b+1] for c in c_trg_list]

            x_fake_list = [x_real_b, x_adv_b]

            for idx, c_trg in enumerate(c_trg_list_b):
                with torch.no_grad():
                    gen_noattack, _ = G(x_real_b, c_trg)
                    gen, _          = G(x_adv_b,  c_trg)

                x_fake_list.append(gen_noattack)
                x_fake_list.append(gen)

                l2_error += F.mse_loss(gen, gen_noattack).item()
                ssim_local, psnr_local = compare(denorm(gen), denorm(gen_noattack))
                ssim += ssim_local
                psnr += psnr_local

                if F.mse_loss(gen, gen_noattack) > 0.05:
                    n_dist += 1
                n_samples += 1

            x_concat = torch.cat(x_fake_list, dim=3)
            result_path = os.path.join(config.result_dir, '{}-images.jpg'.format(n_images_processed + 1))
            save_image(denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
            n_images_processed += 1

    print(f"************Batch Attack & Calculate: {time.time() - t_loop}")
    print('{} images. L2 error: {}. ssim: {}. psnr: {}. n_dist: {}'.format(
        n_samples,
        l2_error / n_samples,
        ssim / n_samples,
        psnr / n_samples,
        float(n_dist) / n_samples))

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"************main_batch: {end_time - start_time}")
