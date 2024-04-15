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

import os
import torch
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args

import torchvision
import numpy as np
import cv2
from matplotlib import pyplot as plt
from segment_anything import SamPredictor

@torch.no_grad()
def _generate_index_matrix(H, W, depth_map):
    '''generate the index matrix, which contains the coordinate of each pixel and cooresponding depth'''
    xs = torch.arange(1, H+1) / H # NOTE, range (1, H) = arange(1, H+1)
    ys = torch.arange(1, W+1) / W
    grid_x, grid_y = torch.meshgrid(xs, ys)
    index_matrix = torch.stack([grid_x, grid_y], dim = -1) # [H, W, 2]
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) # [H, W, 1]
    depth_map = depth_map.squeeze().unsqueeze(-1)
    index_matrix = index_matrix.to(depth_map.device)
    index_matrix = torch.cat([index_matrix, depth_map], dim = -1)
    return index_matrix

@torch.no_grad()
def mask_to_prompt(rendered_mask_score = None, num_prompts = 3, depth = None):
    '''main function for self prompting'''
    _, h, w = rendered_mask_score.shape
    if depth is not None:
        index_matrix = _generate_index_matrix(h, w, depth)

    tmp = torch.max(rendered_mask_score.view(-1), dim = 0)
    if tmp[0] <= 0:
        return np.zeros((0,2)), np.ones((0))

    prompt_points = []
    prompt_points.append([tmp[1].cpu() % w, tmp[1].cpu() // w])

    tmp_mask = rendered_mask_score.clone().detach()
    area = torch.count_nonzero(tmp_mask > 0)
    r = torch.sqrt(area / 3.14)
    masked_r = max(int(r) // 2, 2)
    previous_max_score = 0


    for _ in range(num_prompts - 1):

        l = 0 if prompt_points[-1][0]-masked_r <= 0 else prompt_points[-1][0]-masked_r
        r = w-1 if prompt_points[-1][0]+masked_r >= w-1 else prompt_points[-1][0]+masked_r

        t = 0 if prompt_points[-1][1]-masked_r <= 0 else prompt_points[-1][1]-masked_r
        b = h-1 if prompt_points[-1][1]+masked_r >= h-1 else prompt_points[-1][1]+masked_r
        tmp_mask[:, t:b+1, l:r+1] = -1e5

        previous_point_index = torch.zeros_like(index_matrix)
        previous_point_index[:,:,0] = prompt_points[-1][1] / h
        previous_point_index[:,:,1] = prompt_points[-1][0] / w
        previous_point_index[:,:,2] = index_matrix[int(prompt_points[-1][1]), int(prompt_points[-1][0]), 2]
        distance_matrix = torch.sqrt(((index_matrix - previous_point_index)**2).sum(-1))
        distance_matrix = (distance_matrix.unsqueeze(0) - distance_matrix.min()) / (distance_matrix.max() - distance_matrix.min())

        tmp_mask = tmp_mask - distance_matrix * max(previous_max_score, 0)

        tmp_val_point = torch.max(tmp_mask.view(-1), dim = 0)
        previous_max_score = tmp_val_point[0]

        if tmp_val_point[0] <= 0:
            break
        prompt_points.append([int(tmp_val_point[1].cpu() % w), int(tmp_val_point[1].cpu() // w)])

    prompt_points = np.array(prompt_points)
    input_label = np.ones(len(prompt_points))

    return prompt_points, input_label


    

def training(dataset, opt, pipe, iteration, saving_iterations, checkpoint_iterations, debug_from):
    dataset.need_features = True
    dataset.need_masks = False
    # Initialize SAM predictor
    from segment_anything import sam_model_registry
    sam_checkpoint = "./third_party/segment-anything/ckpt/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to("cuda")
    predictor = SamPredictor(sam)

    gaussians = GaussianModel(dataset.sh_degree)

    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, init_from_3dgs_pcd=dataset.init_from_3dgs_pcd, target='seg', mode='train')

    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)


    optimization_times = opt.optimization_times
    IoU_thresh = opt.IoU_thresh
    IoA_thresh = opt.IoA_thresh
    lamb = opt.lamb

    cams = scene.getTrainCameras() * optimization_times

    gt_list = []
    bitmap = [True]*len(cams)
    for iteration, view in enumerate(cams):

        # Ambiguous Gaussians Removal
        if iteration == len(cams) // optimization_times:
            gaussians.segment()

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Render
        if iteration == debug_from:
            pipe.debug = True

        render_pkg = render(view, gaussians, pipe, background)
        rendered_mask, rendered_depth, viewspace_point_tensor, visibility_filter, radii = render_pkg["mask"], render_pkg["depth"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        rendered_depth = (rendered_depth - rendered_depth.min()) / (rendered_depth.max() - rendered_depth.min())


        sam_features = view.original_features.cuda()
        
        predictor.original_size = (1024, 1024)
        predictor.input_size = (1024, 1024)
        predictor.features = sam_features
        predictor.is_image_set = True

        if len(gt_list) < len(cams) // optimization_times:
            if iteration == 0:
                # [X, Y]

                input_point = np.array([[300, 370]])

                os.makedirs('tmpvis_files', exist_ok=True)
                os.makedirs(f"tmpvis_files/{dataset.source_path.split('/')[-1]}", exist_ok=True)
                tmp_vis_path = f"tmpvis_files/{dataset.source_path.split('/')[-1]}"

                input_label = np.ones(len(input_point))

                with torch.no_grad():
                    masks, scores, logits = predictor.predict(
                        point_coords=input_point,
                        point_labels=input_label,
                        multimask_output=True,
                    )

                plt.figure(figsize=(10,10))
                tmp_image = cv2.resize(view.original_image.permute([1,2,0]).detach().cpu().numpy(), (1024, 1024))
                plt.imshow(tmp_image)
                plt.scatter(input_point[:, 0], input_point[:, 1], c='red', s=10)
                plt.axis('on')
                # plt.savefig('tmp_image.jpg')
                plt.savefig(os.path.join(tmp_vis_path, "tmp_image.jpg"))

                for j, mask in enumerate(masks): 
                    ### for selection
                    plt.figure(figsize=(10,10))
                    plt.imshow(tmp_image*mask[...,None].astype(np.float32))
                    plt.scatter(input_point[:, 0], input_point[:, 1], c='red', s=10)
                    plt.axis('on')
                    # plt.savefig('tmp_mask_'+str(j)+'.jpg')
                    plt.savefig(os.path.join(tmp_vis_path, f"tmp_maskrgb_{j}.jpg"))
                    # clear the figure
                    plt.clf()
                    plt.close()

                    plt.figure(figsize=(10,10))
                    plt.imshow(mask)
                    plt.axis('on')
                    # plt.savefig('tmp_mask_'+str(j)+'.jpg')
                    plt.savefig(os.path.join(tmp_vis_path, f"tmp_mask_{j}.jpg"))

                selected_mask = int(input(f"Please select a mask (check {tmp_vis_path} for masks):"))
                progress_bar = tqdm(range(len(cams)), desc="Segmenting progress")
            else:

                prompt_points, input_label = mask_to_prompt(rendered_mask_score = rendered_mask - 0.5, num_prompts = args.num_prompts, depth = rendered_depth)

                # project the prompt points to the 1024*1024 canvas required by SAM
                prompt_points[:,0] = prompt_points[:,0] * 1024 / rendered_mask.shape[-1]
                prompt_points[:,1] = prompt_points[:,1] * 1024 / rendered_mask.shape[-2]

                if len(prompt_points) != 0:
                    with torch.no_grad():
                        masks, scores, logits = predictor.predict(
                            point_coords=prompt_points,
                            point_labels=input_label,
                            multimask_output=False,
                        )
                    selected_mask = 0
                else:
                    progress_bar.set_postfix({"IoU": f"{0:.{2}f}", "IoA": f"{0:.{2}f}"})
                    progress_bar.update(1)
                    continue

        
            gt_mask = torch.from_numpy(masks[selected_mask:selected_mask+1]).float().cuda()
            gt_mask = torch.nn.functional.interpolate(gt_mask.unsqueeze(0), size=rendered_mask.shape[-2:], mode='bilinear', align_corners=False).squeeze(0).detach()
            gt_list.append(gt_mask)
        else:
            gt_mask = gt_list[iteration % (len(cams) // optimization_times)]


        tmp_rendered_mask = rendered_mask.detach().clone()
        tmp_rendered_mask[tmp_rendered_mask <= 0.5] = 0
        tmp_rendered_mask[tmp_rendered_mask != 0] = 1

        IoU = (gt_mask * tmp_rendered_mask).sum() / ((gt_mask + tmp_rendered_mask).sum() - (gt_mask * tmp_rendered_mask).sum())
        IoA = (gt_mask * tmp_rendered_mask).sum() / gt_mask.sum()
        # print("IoU: ", IoU)

        # torchvision.utils.save_image(gt_mask[0], os.path.join(tmp_vis_path, "sam_pred_{}".format(iteration)+".png"))
        # torchvision.utils.save_image(tmp_rendered_mask[0], os.path.join(tmp_vis_path, "tmp_rendered_{}".format(iteration)+".png"))
        if iteration < len(cams) // optimization_times:
            if iteration != 0 and IoU < IoU_thresh:
                bitmap[iteration % (len(cams) // optimization_times)] = False
                # print("IoU: ", IoU, "Unacceptable SAM prediction. Skipping this iteration.")
                progress_bar.set_postfix({"IoU": f"{IoU.item():.{2}f}", "IoA": f"{IoA.item():.{2}f}"})
                progress_bar.update(1)
                continue
        else:
            if not bitmap[iteration % (len(cams) // optimization_times)]:
                progress_bar.set_postfix({"IoU": f"{IoU.item():.{2}f}", "IoA": f"{IoA.item():.{2}f}"})
                progress_bar.update(1)
                continue
        
        # Loss

        loss = - (gt_mask * rendered_mask).sum() + lamb * ((1-gt_mask) * rendered_mask).sum()
        loss.backward()

        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none = True)

        iter_end.record()
        with torch.no_grad():
            progress_bar.set_postfix({"IoU": f"{IoU.item():.{2}f}", "IoA": f"{IoA.item():.{2}f}"})
            progress_bar.update(1)
            
    progress_bar.close()
                
    
    scene.save(iteration, target='seg')

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "seg_cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6010)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--target", type=str, default = 'seg')
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--num_prompts", default=3, type=int)

    args = get_combined_args(parser, target_cfg_file = 'cfg_args')
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.iteration, args.save_iterations, args.checkpoint_iterations, args.debug_from)

    # All done
    print("\nTraining complete.")
