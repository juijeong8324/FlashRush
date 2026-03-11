import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

SIMSWAP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'SimSwap')
sys.path.insert(0, SIMSWAP_DIR)


class SimSwapWrapper(nn.Module):
    """Wraps SimSwap into a consistent interface for adversarial attacks.

    Attack scenario: protect img_id (the victim's identity photo).
    forward(img_id, img_att) extracts identity from img_id via ArcFace,
    then generates the swapped face. Perturbing img_id distorts the extracted
    identity → swap is degraded.

    img_id  : ImageNet-normalized [B, 3, H, W]  (identity source = victim's face)
    img_att : [0, 1]-normalized   [B, 3, H, W]  (attribute source = target face)
    returns : (swapped_face [0,1], None)          -- None keeps StarGAN interface compat
    """

    def __init__(self, netG: nn.Module, netArc: nn.Module):
        super().__init__()
        self.netG   = netG
        self.netArc = netArc

    @classmethod
    def load(cls, arc_path: str, G_path: str, crop_size: int = 224) -> "SimSwapWrapper":
        """Load SimSwapWrapper from pretrained weight files.

        arc_path  : path to arcface_checkpoint.tar  (saved as a full model object)
        G_path    : path to latest_net_G.pth         (SimSwap generator state_dict)
        crop_size : 224 (default) or 512
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ArcFace is saved as a full model object (not state_dict)
        netArc = torch.load(arc_path, map_location="cpu", weights_only=False)
        netArc = netArc.to(device).eval()
        for p in netArc.parameters():
            p.requires_grad_(False)

        # Generator
        if crop_size == 512:
            from models.fs_networks_512 import Generator_Adain_Upsample
        else:
            from models.fs_networks import Generator_Adain_Upsample

        netG = Generator_Adain_Upsample(
            input_nc=3, output_nc=3, latent_size=512, n_blocks=9, deep=False
        )
        netG.load_state_dict(torch.load(G_path, map_location="cpu", weights_only=False))
        netG = netG.to(device).eval()

        return cls(netG, netArc)

    def forward(self, img_id: torch.Tensor, img_att: torch.Tensor):
        """
        img_id  : [B, 3, H, W] ImageNet-normalized identity source
        img_att : [B, 3, H, W] [0, 1]-normalized attribute source
        returns : (swapped [0,1], None)
        """
        # ArcFace expects 112×112 ImageNet-normalized input
        img_id_112 = F.interpolate(img_id, size=(112, 112), mode='bilinear', align_corners=False)
        latent_id = self.netArc(img_id_112)
        latent_id = latent_id / latent_id.norm(dim=1, keepdim=True)

        img_fake = self.netG(img_att, latent_id)  # [0, 1] output
        return img_fake, None
