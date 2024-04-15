#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene, GaussianModel, FeatureGaussianModel
import os
from tqdm import tqdm
from os import makedirs
# from gaussian_renderer import render, render_feature, render_contrastive_feature, render_xyz
from gaussian_renderer import render
import torchvision

from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args

# MASK_THRESHOLD = 0.1

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, target, MASK_THRESHOLD):
    render_path = os.path.join(model_path, name, f"ours_{iteration}_{str(MASK_THRESHOLD).replace('.', '_')}", "renders")
    gts_path = os.path.join(model_path, name, f"ours_{iteration}_{str(MASK_THRESHOLD).replace('.', '_')}", "gt")
    mask_path = os.path.join(model_path, name, f"ours_{iteration}_{str(MASK_THRESHOLD).replace('.', '_')}", "mask")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(mask_path, exist_ok=True)

    # if target == 'feature':
    #     render_func = render_feature
    # elif target == 'contrastive_feature':
    #     render_func = render_contrastive_feature
    # elif target == 'xyz':
    #     render_func = render_xyz
    render_func = render

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        res = render_func(view, gaussians, pipeline, background)
        rendering = res["render"]
        
        gt = view.original_image[0:3, :, :]
        # print(rendering.shape, mask.shape, gt.shape, "rendering.shape, mask.shape, gt.shape")

        # print("mask render time", time.time() - start_time)
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image((res["depth"] - res["depth"].min()) / (res["depth"].max() - res["depth"].min()), os.path.join(render_path, '{0:05d}_depth'.format(idx) + ".png"))
        if target == 'seg':
            mask = res["mask"]
            mask[mask <= MASK_THRESHOLD] = 0.
            mask[mask > MASK_THRESHOLD] = 1.
            mask = mask[0, :, :]
            # torchvision.utils.save_image(mask, os.path.join(mask_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(mask, os.path.join(mask_path, f"{view.image_name}.png"))
        if target == 'seg' or target == 'scene' or target == 'coarse_seg_everything':
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            # torchvision.utils.save_image(gt * mask[None], os.path.join(render_path, f"{view.image_name}.png"))
        elif 'feature' in target:
            torch.save(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".pt"))
        elif target == 'xyz':
            torch.save(rendering, os.path.join(render_path, 'xyz_{0:05d}'.format(idx) + ".pt"))
        
        
        

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, segment : bool = False, target = 'scene', idx = 0, precomputed_mask = None, MASK_THRESHOLD=0.1):
    dataset.need_features = dataset.need_masks = False
    if segment:
        assert target == 'seg' or target == 'coarse_seg_everything' or precomputed_mask is not None and "Segmentation only works with target seg!"
    gaussians, feature_gaussians = None, None
    with torch.no_grad():
        if target == 'scene' or target == 'seg' or target == 'coarse_seg_everything' or target == 'xyz':
            gaussians = GaussianModel(dataset.sh_degree)
        if target == 'feature' or target == 'coarse_seg_everything' or target == 'contrastive_feature':
            feature_gaussians = FeatureGaussianModel(dataset.feature_dim)

        scene = Scene(dataset, gaussians, feature_gaussians, load_iteration=iteration, shuffle=False, mode='eval', target=target if target != 'xyz' else 'scene')
        scene.save(scene.loaded_iter, target='scene', colored=True)
        if segment:
            if target == 'coarse_seg_everything':
                mask = feature_gaussians.segment(idx=idx)
                gaussians.segment(mask=mask)
                scene.save(scene.loaded_iter, target=f'seg_res_{idx}')
            else:
                if precomputed_mask is None:
                    gaussians.segment()
                    scene.save(scene.loaded_iter, target='seg_res', colored=True)
                else:
                    pre_mask = torch.load(precomputed_mask)
                    gaussians.segment(pre_mask)
                    scene.save(scene.loaded_iter, target='seg_res')

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        if 'feature' in target:
            gaussians = feature_gaussians
            bg_color = [1 for i in range(dataset.feature_dim)] if dataset.white_background else [0 for i in range(dataset.feature_dim)]

        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, target, MASK_THRESHOLD)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, target, MASK_THRESHOLD)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--segment", action="store_true")
    parser.add_argument('--target', default='scene', const='scene', nargs='?', choices=['scene', 'seg', 'feature', 'coarse_seg_everything', 'contrastive_feature', 'xyz'])
    parser.add_argument('--idx', default=0, type=int)
    parser.add_argument('--precomputed_mask', default=None, type=str)
    parser.add_argument('--MASK_THRESHOLD', default=0.1, type=float)

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    if not hasattr(args, 'precomputed_mask'):
        args.precomputed_mask = None
    if args.precomputed_mask is not None:
        print("Using precomputed mask " + args.precomputed_mask)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.segment, args.target, args.idx, args.precomputed_mask, args.MASK_THRESHOLD)