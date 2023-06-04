import copy
import random

import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import seg_dvgo as dvgo
from lib import seg_dcvgo as dcvgo

from .load_data import load_data
from .masked_adam import MaskedAdam
from torch import Tensor

''' Misc
'''
mse2psnr = lambda x : -10. * torch.log10(x)
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

def seed_everything(args):
    '''Seed everything for better reproducibility.
    NOTE that some pytorch operation is non-deterministic like the backprop of grid_samples
    '''
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


@torch.jit.script
def cal_IoU(a: Tensor, b: Tensor) -> Tensor:
    """Calculates the Intersection over Union (IoU) between two tensors.

    Args:
        a: A tensor of shape (N, H, W).
        b: A tensor of shape (N, H, W).

    Returns:
        A tensor of shape (N,) containing the IoU score between each pair of
        elements in a and b.
    """
    intersection = torch.count_nonzero(torch.logical_and(a == b, a != 0))
    union = torch.count_nonzero(a + b)
    return intersection / union


def load_everything(args, cfg):
    '''Load images / poses / camera settings / data split.
    '''
    data_dict = load_data(cfg.data)

    # remove useless field
    kept_keys = {
            'hwf', 'HW', 'Ks', 'near', 'far', 'near_clip',
            'i_train', 'i_val', 'i_test', 'irregular_shape',
            'poses', 'render_poses', 'images'}
    for k in list(data_dict.keys()):
        if k not in kept_keys:
            data_dict.pop(k)

    # construct data tensor
    if data_dict['irregular_shape']:
        data_dict['images'] = [torch.FloatTensor(im, device='cpu') for im in data_dict['images']]
    else:
        data_dict['images'] = torch.FloatTensor(data_dict['images'], device='cpu')
    data_dict['poses'] = torch.Tensor(data_dict['poses'])
    data_dict['render_poses'] = torch.Tensor(data_dict['render_poses'])

    return data_dict

# semantic nerf is used for reproducing the segmentation of SPIn-NeRF
def load_existed_model(args, cfg, cfg_train, reload_ckpt_path, device):
    model_class = find_model(cfg)
    model = load_model(model_class, reload_ckpt_path).to(device)
    optimizer = create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
    model, optimizer, start = load_checkpoint(
            model, optimizer, reload_ckpt_path, no_reload_optimizer = True)
    return model, optimizer, start
    


def gen_rand_colors(num_obj):
    rand_colors = np.random.rand(num_obj + 1, 3)
    rand_colors[-1,:] = 0
    return rand_colors


def to_cuda(batch, device=torch.device('cuda')):
    if isinstance(batch, tuple) or isinstance(batch, list):
        batch = [to_cuda(b, device) for b in batch]
    elif isinstance(batch, dict):
        batch_ = {}
        for key in batch:
            if key == 'meta':
                batch_[key] = batch[key]
            else:
                batch_[key] = to_cuda(batch[key], device)
        batch = batch_
    elif isinstance(batch, np.ndarray):
        batch = torch.from_numpy(batch).to(device)
    else:
        batch = batch.to(device)
    return batch


def to_tensor(array, device=torch.device('cuda')):
    '''cvt numpy array to cuda tensor, if already tensor, do nothing
    '''
    if isinstance(array, np.ndarray):
        array = torch.from_numpy(array).to(device)
    elif isinstance(array, torch.Tensor) and not array.is_cuda:
        array = array.to(device)
    else:
        pass
    return array.float()


''' optimizer
'''
def create_optimizer_or_freeze_model(model, cfg_train, global_step):
    decay_steps = cfg_train.lrate_decay * 1000
    decay_factor = 0.1 ** (global_step/decay_steps)

    param_group = []
    for k in cfg_train.keys():
        if not k.startswith('lrate_'):
            continue
        k = k[len('lrate_'):]

        if not hasattr(model, k):
            continue

        param = getattr(model, k)
        if param is None:
            print(f'create_optimizer_or_freeze_model: param {k} not exist')
            continue

        lr = getattr(cfg_train, f'lrate_{k}') * decay_factor
        if lr > 0:
            print(f'create_optimizer_or_freeze_model: param {k} lr {lr}')
            if isinstance(param, nn.Module):
                param = param.parameters()
            param_group.append({'params': param, 'lr': lr, 'skip_zero_grad': (k in cfg_train.skip_zero_grad_fields)})
        else:
            print(f'create_optimizer_or_freeze_model: param {k} freeze')
            param.requires_grad = False
    return MaskedAdam(param_group)


def create_segmentation_optimizer(model, cfg_train):

    param_group = []
    for k in cfg_train.keys():
        if not k.startswith('lrate_'):
            continue
        k = k[len('lrate_'):]

        if not hasattr(model, k):
            continue

        param = getattr(model, k)
        if param is None:
            print(f'create_optimizer_or_freeze_model: param {k} not exist')
            continue

        lr = getattr(cfg_train, f'lrate_{k}')
        if lr > 0:
            print(f'create_optimizer_or_freeze_model: param {k} lr {lr}')
            if isinstance(param, nn.Module):
                param = param.parameters()
            param_group.append({'params': param, 'lr': lr})
        else:
            print(f'create_optimizer_or_freeze_model: param {k} freeze')
            param.requires_grad = False
    return torch.optim.SGD(param_group)


''' Checkpoint utils
'''
def load_checkpoint(model, optimizer, ckpt_path, no_reload_optimizer):
    ckpt = torch.load(ckpt_path)
    try:
        start = ckpt['global_step']
    except:
        start = 0
    msg = model.load_state_dict(ckpt['model_state_dict'], strict = False)
    print("NeRF loaded with msg: ", msg)
    if not no_reload_optimizer:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return model, optimizer, start


def find_model(cfg):
    if cfg.data.ndc:
        model_class = dvgo.DirectVoxGO
    elif cfg.data.unbounded_inward:
        model_class = dcvgo.DirectContractedVoxGO
    else:
        model_class = dvgo.DirectVoxGO
    return model_class


def load_model(model_class, ckpt_path):
    ckpt = torch.load(ckpt_path)
    num_objects = 1
    if 'seg_mask_grid.grid' in ckpt['model_state_dict'].keys():
        num_objects = ckpt['model_state_dict']['seg_mask_grid.grid'].shape[1]
        
    print("Load model with num_objects =", num_objects)

    model = model_class(num_objects = num_objects, **ckpt['model_kwargs'])
    msg = model.load_state_dict(ckpt['model_state_dict'], strict = False)
    print("NeRF loaded with msg: ", msg)
    return model


def create_new_model(cfg, cfg_model, cfg_train, xyz_min, xyz_max, stage, coarse_ckpt_path, device=torch.device('cuda')):
    model_kwargs = copy.deepcopy(cfg_model)
    num_voxels = model_kwargs.pop('num_voxels')
    if len(cfg_train.pg_scale):
        num_voxels = int(num_voxels / (2**len(cfg_train.pg_scale)))

    if cfg.data.ndc:
        #print(f'scene_rep_reconstruction ({stage}): \033[96muse multiplane images\033[0m')
        #model = dmpigo.DirectMPIGO(
        #    xyz_min=xyz_min, xyz_max=xyz_max,
        #    num_voxels=num_voxels,
        #    **model_kwargs)
        print(f'scene_rep_reconstruction ({stage}): \033[96muse dense voxel grid\033[0m')
        model = dvgo.DirectVoxGO(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels=num_voxels,
            mask_cache_path=coarse_ckpt_path,
            **model_kwargs)
    elif cfg.data.unbounded_inward:
        print(f'scene_rep_reconstruction ({stage}): \033[96muse contraced voxel grid (covering unbounded)\033[0m')
        model = dcvgo.DirectContractedVoxGO(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels=num_voxels,
            **model_kwargs)
    else:
        print(f'scene_rep_reconstruction ({stage}): \033[96muse dense voxel grid\033[0m')
        model = dvgo.DirectVoxGO(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels=num_voxels,
            mask_cache_path=coarse_ckpt_path,
            **model_kwargs)
    model = model.to(device)
    optimizer = create_optimizer_or_freeze_model(model, cfg_train, global_step=0)
    return model, optimizer


''' Evaluation metrics (ssim, lpips)
'''
def rgb_ssim(img0, img1, max_val,
             filter_size=11,
             filter_sigma=1.5,
             k1=0.01,
             k2=0.03,
             return_map=False):
    # Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    assert len(img0.shape) == 3
    assert img0.shape[-1] == 3
    assert img0.shape == img1.shape

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma)**2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode='valid')

    filt_fn = lambda z: np.stack([
        convolve2d(convolve2d(z[...,i], filt[:, None]), filt[None, :])
        for i in range(z.shape[-1])], -1)
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2) - mu00
    sigma11 = filt_fn(img1**2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0., sigma00)
    sigma11 = np.maximum(0., sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(
        np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = np.mean(ssim_map)
    return ssim_map if return_map else ssim


__LPIPS__ = {}
def init_lpips(net_name, device):
    assert net_name in ['alex', 'vgg']
    import lpips
    print(f'init_lpips: lpips_{net_name}')
    return lpips.LPIPS(net=net_name, version='0.1').eval().to(device)


def rgb_lpips(np_gt, np_im, net_name, device):
    if net_name not in __LPIPS__:
        __LPIPS__[net_name] = init_lpips(net_name, device)
    gt = torch.from_numpy(np_gt).permute([2, 0, 1]).contiguous().to(device)
    im = torch.from_numpy(np_im).permute([2, 0, 1]).contiguous().to(device)
    return __LPIPS__[net_name](gt, im, normalize=True).item()


''' generate rays
'''
def get_rays(H, W, K, c2w, inverse_y, flip_x, flip_y, mode='center'):
    i, j = torch.meshgrid(
        torch.linspace(0, W-1, W, device=c2w.device),
        torch.linspace(0, H-1, H, device=c2w.device))  # pytorch's meshgrid has indexing='ij'
    i = i.t().float()
    j = j.t().float()
    if mode == 'lefttop':
        pass
    elif mode == 'center':
        i, j = i+0.5, j+0.5
    elif mode == 'random':
        i = i+torch.rand_like(i)
        j = j+torch.rand_like(j)
    else:
        raise NotImplementedError

    if flip_x:
        i = i.flip((1,))
    if flip_y:
        j = j.flip((0,))
    if inverse_y:
        dirs = torch.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], torch.ones_like(i)], -1)
    else:
        dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,3].expand(rays_d.shape)
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]

    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)

    return rays_o, rays_d


def get_rays_of_a_view(H, W, K, c2w, ndc, inverse_y, flip_x, flip_y, mode='center'):
    rays_o, rays_d = get_rays(H, W, K, c2w, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y, mode=mode)
    viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
    if ndc:
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)
    return rays_o, rays_d, viewdirs


''' interactive mode TODO'''
def fetch_user_define_points():
    pass

