import json
import os
import sys
import time

import imageio
import matplotlib.pyplot as plt

# remove the dependency on mmcv
# import mmcv
from lib.config_loader import Config

import numpy as np
import torch
import glob
import cv2

from lib import utils
from lib.bbox_utils import *
from lib.configs import config_parser
from lib import sam3d
from lib.gui import Sam3dGUI
from lib.render_utils import render_fn


def train_seg(args, cfg, data_dict):
    '''main training code for Segment Anything in 3D with NeRFs'''
    print('train: start')
    eps_time = time.time()
    os.makedirs(os.path.join(cfg.basedir, cfg.expname), exist_ok=True)

    # save configs
    with open(os.path.join(cfg.basedir, cfg.expname, 'args.txt'), 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    cfg.dump(os.path.join(cfg.basedir, cfg.expname, 'config.py'))

    # start segmentation stage
    eps_coarse = time.time()
    xyz_min_coarse, xyz_max_coarse = compute_bbox_by_cam_frustrm(args=args, cfg=cfg, **data_dict)

    e_flag = args.sp_name if args.sp_name is not None else ''
    coarse_seg_ckpt_path = os.path.join(cfg.basedir, cfg.expname, f'coarse_segmentation'+e_flag+'.tar')

    # coarse stage
    if not os.path.exists(coarse_seg_ckpt_path):
        Seg3d = sam3d.Sam3D(args, cfg, cfg_model=cfg.coarse_model_and_render, cfg_train=cfg.coarse_train,
                xyz_min=xyz_min_coarse, xyz_max=xyz_max_coarse,
                data_dict=data_dict, stage='coarse')
        gui = Sam3dGUI(Seg3d)
        gui.run()
        eps_coarse = time.time() - eps_coarse
        eps_time_str = f'{eps_coarse//3600:02.0f}:{eps_coarse//60%60:02.0f}:{eps_coarse%60:02.0f}'
        print('train: coarse segmentation in', eps_time_str)
    else:
        print('Coarse segmentation has been completed, skip!')

    # fine stage when the mask from the coarse stage is not good enough
    if args.use_fine_stage:
        eps_fine = time.time()
        if cfg.coarse_train.N_iters == 0:
            xyz_min_fine, xyz_max_fine = xyz_min_coarse.clone(), xyz_max_coarse.clone()
        else:
            xyz_min_fine, xyz_max_fine = compute_bbox_by_coarse_geo(
                    model_class=cfg.coarse_model_and_render, model_path=coarse_seg_ckpt_path,
                    thres=cfg.fine_model_and_render.bbox_thres)
        # finetune
        Seg3d = sam3d.Sam3D(args, cfg, cfg_model=cfg.fine_model_and_render, cfg_train=cfg.fine_train,
                xyz_min=xyz_min_fine, xyz_max=xyz_max_fine,
                data_dict=data_dict, stage='fine',
                coarse_ckpt_path=coarse_seg_ckpt_path)
        gui = Sam3dGUI(Seg3d)
        gui.run()
        eps_fine = time.time() - eps_fine
        eps_time_str = f'{eps_fine//3600:02.0f}:{eps_fine//60%60:02.0f}:{eps_fine%60:02.0f}'
        print('train: fine detail segmentation in', eps_time_str)

    eps_time = time.time() - eps_time
    eps_time_str = f'{eps_time//3600:02.0f}:{eps_time//60%60:02.0f}:{eps_time%60:02.0f}'
    print('train: finish (eps time', eps_time_str, ')')


if __name__=='__main__':
    # load setup
    parser = config_parser()
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)

    # init enviroment
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    utils.seed_everything(args)

    # load images / poses / camera settings / data split
    data_dict = utils.load_everything(args=args, cfg=cfg)

    # train
    if not args.render_only:
        train_seg(args, cfg, data_dict)

    # load model for further rendering
    e_flag = args.sp_name if args.sp_name is not None else ''
    if args.render_opt is not None:
        for seg_type in ['seg_img', 'seg_density']:
            if args.ft_path:
                ckpt_path = args.ft_path
            else:
                fine_path = os.path.join(cfg.basedir, cfg.expname, 'fine_segmentation'+e_flag+'.tar')
                coarse_path = os.path.join(cfg.basedir, cfg.expname, 'coarse_segmentation'+e_flag+'.tar')
                ckpt_path = fine_path if os.path.exists(fine_path) else coarse_path
            print("\033[96mRendering with ckpt "+ckpt_path+"\033[0m")
                
            ckpt_name = ckpt_path.split('/')[-1][:-4]
            model_class = utils.find_model(cfg)
            model, optimizer, start = utils.load_existed_model(args, cfg, cfg.fine_train, ckpt_path, device)
            
            stepsize = cfg.fine_model_and_render.stepsize
            render_viewpoints_kwargs = {
                'model': model,
                'ndc': cfg.data.ndc,
                'render_kwargs': {
                    'near': data_dict['near'],
                    'far': data_dict['far'],
                    'bg': 1 if cfg.data.white_bkgd else 0,
                    'stepsize': stepsize,
                    'inverse_y': cfg.data.inverse_y,
                    'flip_x': cfg.data.flip_x,
                    'flip_y': cfg.data.flip_y,
                    'render_depth': True,
                },
            }

            # rendering
            flag = "seg" if args.segment else ""
            if args.segment:
                if seg_type == 'seg_density':
                    render_viewpoints_kwargs['model'].segmentation_to_density()
                elif seg_type == 'seg_img':
                    render_viewpoints_kwargs['model'].segmentation_only()
                else:
                    raise NotImplementedError('seg type {} is not implemented!'.format(seg_type))

            # default: one object    
            num_obj = render_viewpoints_kwargs['model'].seg_mask_grid.grid.shape[1]
            render_viewpoints_kwargs['model'] = render_viewpoints_kwargs['model'].cuda()
            render_fn(args, cfg, ckpt_name, flag, e_flag, num_obj, \
                                   data_dict, render_viewpoints_kwargs, seg_type=seg_type)

    



