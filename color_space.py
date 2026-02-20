import torch

def rgb2xyz(rgb):
    mask = (rgb > .04045).type(torch.FloatTensor)
    if rgb.is_cuda:
        mask = mask.cuda()
    rgb = (((rgb + .055) / 1.055) ** 2.4) * mask + rgb / 12.92 * (1 - mask)
    x = .412453 * rgb[:, 0, :, :] + .357580 * rgb[:, 1, :, :] + .180423 * rgb[:, 2, :, :]
    y = .212671 * rgb[:, 0, :, :] + .715160 * rgb[:, 1, :, :] + .072169 * rgb[:, 2, :, :]
    z = .019334 * rgb[:, 0, :, :] + .119193 * rgb[:, 1, :, :] + .950227 * rgb[:, 2, :, :]
    out = torch.cat((x[:, None, :, :], y[:, None, :, :], z[:, None, :, :]), dim=1)
    out = torch.clamp(out, min=0, max=1)
    return out

def xyz2rgb(xyz):
    r = 3.24048134 * xyz[:, 0, :, :] - 1.53715152 * xyz[:, 1, :, :] - 0.49853633 * xyz[:, 2, :, :]
    g = -0.96925495 * xyz[:, 0, :, :] + 1.87599 * xyz[:, 1, :, :] + .04155593 * xyz[:, 2, :, :]
    b = .05564664 * xyz[:, 0, :, :] - .20404134 * xyz[:, 1, :, :] + 1.05731107 * xyz[:, 2, :, :]
    rgb = torch.cat((r[:, None, :, :], g[:, None, :, :], b[:, None, :, :]), dim=1)
    rgb = torch.clamp(rgb, min=0, max=1)
    return rgb

def xyz2lab(xyz):
    sc = torch.Tensor((0.95047, 1., 1.08883))[None, :, None, None]
    if xyz.is_cuda:
        sc = sc.cuda()
    xyz_scale = xyz / sc
    mask = (xyz_scale > .008856).type(torch.FloatTensor)
    if xyz_scale.is_cuda:
        mask = mask.cuda()
    xyz_int = xyz_scale ** (1 / 3.) * mask + (7.787 * xyz_scale + 16. / 116.) * (1 - mask)
    L = 116. * xyz_int[:, 1, :, :] - 16.
    a = 500. * (xyz_int[:, 0, :, :] - xyz_int[:, 1, :, :])
    b = 200. * (xyz_int[:, 1, :, :] - xyz_int[:, 2, :, :])
    out = torch.cat((L[:, None, :, :], a[:, None, :, :], b[:, None, :, :]), dim=1)
    return out

def lab2xyz(lab):
    y_int = (lab[:, 0, :, :] + 16.) / 116.
    x_int = (lab[:, 1, :, :] / 500.) + y_int
    z_int = y_int - (lab[:, 2, :, :] / 200.)
    x_int = torch.clamp(x_int, min=0)
    y_int = torch.clamp(y_int, min=0)
    z_int = torch.clamp(z_int, min=0)
    sc = torch.Tensor((0.95047, 1., 1.08883))[None, :, None, None].to(lab.device)
    out = torch.cat((x_int[:, None, :, :], y_int[:, None, :, :], z_int[:, None, :, :]), dim=1) * sc
    return out

def rgb2lab(rgb):
    return xyz2lab(rgb2xyz(rgb))

def lab2rgb(lab):
    return xyz2rgb(lab2xyz(lab))