import json
import os
import time
from abc import ABC
from typing import Optional

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from segment_anything import (SamAutomaticMaskGenerator, SamPredictor,
                              sam_model_registry)
from tqdm import tqdm

from . import utils
# from .scene_property import INPUT_BOX, INPUT_POINT
from .self_prompting import mask_to_prompt
from .prepare_prompts import get_prompt_points
from .render_utils import render_fn


class Sam3D(ABC):
    '''TODO, add discription'''
    def __init__(self, args, cfg, xyz_min, xyz_max, cfg_model, cfg_train, \
                 data_dict, device=torch.device('cuda'), stage='coarse', coarse_ckpt_path=None):
        self.cfg = cfg
        self.args = args
        sam_checkpoint = "./dependencies/sam_ckpt/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
        self.predictor = SamPredictor(self.sam)
        print("SAM initializd.")
        self.step_size = cfg.fine_model_and_render.stepsize
        self.device = device
        self.segment = args.segment
        self.e_flag = args.sp_name if args.sp_name is not None else ''
        self.base_save_dir = os.path.join(cfg.basedir, cfg.expname)
        # for interactive backend
        self.context = {'num_clicks': 0, 'click': []}

        self.cfg_model, self.cfg_train = cfg_model, cfg_train
        self.xyz_min, self.xyz_max = xyz_min, xyz_max
        self.data_dict = data_dict
        self.stage = stage
        self.coarse_ckpt_path = coarse_ckpt_path


    def init_model(self):
        '''TODO, add discription'''
        if abs(self.cfg_model.world_bound_scale - 1) > 1e-9:
            xyz_shift = (xyz_max - xyz_min) * (self.cfg_model.world_bound_scale - 1) / 2
            xyz_min -= xyz_shift
            xyz_max += xyz_shift
        
        # find whether there is existing checkpoint path
        last_ckpt_path = os.path.join(self.base_save_dir, f'fine_last.tar')
        if self.args.no_reload:
            reload_ckpt_path = None
        elif self.args.ft_path:
            reload_ckpt_path = self.args.ft_path
        elif self.coarse_ckpt_path is not None and os.path.isfile(last_ckpt_path):
            reload_ckpt_path = self.coarse_ckpt_path
        elif os.path.isfile(last_ckpt_path):
            reload_ckpt_path = last_ckpt_path
        else:
            reload_ckpt_path = None

        # init model and optimizer
        assert reload_ckpt_path is not None and 'segmentation must based on a pretrained NeRF'
        print(f'scene_rep_reconstruction ({self.stage}): reload from {reload_ckpt_path}')
        model, optimizer, start = utils.load_existed_model(self.args, self.cfg, 
            self.cfg_train, reload_ckpt_path, self.device)

        if self.segment:
            for param in model.named_parameters():
                if ('density' in param[0]) or ('rgbnet' in param[0]) or ('k0' in param[0]):
                    param[1].requires_grad = False

        if self.stage == 'fine':
            model.change_to_fine_mode()
            print("Segmentation model: FINE MODE.")
        else:
            print("Segmentation model: COARSE MODE.")

        # in case OOM
        torch.cuda.empty_cache()

        self.render_viewpoints_kwargs = {
                'model': model,
                'ndc': self.cfg.data.ndc,
                'render_kwargs': {
                    'near': self.data_dict['near'],
                    'far': self.data_dict['far'],
                    'bg': 1 if self.cfg.data.white_bkgd else 0,
                    'stepsize': self.step_size,
                    'inverse_y': self.cfg.data.inverse_y,
                    'flip_x': self.cfg.data.flip_x,
                    'flip_y': self.cfg.data.flip_y,
                    'render_depth': True,
                },
            }
        self.optimizer = utils.create_segmentation_optimizer(model, self.cfg_train)

        with torch.no_grad():
            rgb, _, _, _, _ = self.render_view(idx=0)
            init_image = utils.to8b(rgb.cpu().numpy())
            self.predictor.set_image(init_image)
        
        return init_image


    def render_view(self, idx, cam_params=None, render_fct=0.0):
        # Training seg
        if cam_params is None:
            render_poses, HW, Ks = fetch_seg_poses(self.args.seg_poses, self.data_dict)
            assert(idx < len(render_poses))
        else:
            render_poses, HW, Ks = cam_params

        model = self.render_viewpoints_kwargs['model']
        render_kwargs = self.render_viewpoints_kwargs['render_kwargs']
        # get data
        c2w = render_poses[idx]
        H, W = HW[idx]; K = Ks[idx]
        ndc = self.cfg.data.ndc
        rays_o, rays_d, viewdirs = utils.get_rays_of_a_view(
                H, W, K, c2w, ndc, inverse_y=render_kwargs['inverse_y'],
                flip_x=self.cfg.data.flip_x, flip_y=self.cfg.data.flip_y)
        
        keys = ['rgb_marched', 'depth', 'alphainv_last', 'seg_mask_marched']
        if self.stage == 'fine': keys.append('dual_seg_mask_marched')
        rays_o, rays_d, viewdirs = [arr.flatten(0, -2) for arr in [rays_o, rays_d, viewdirs]]
        render_result_chunks = [
            {k: v for k, v in model(ro, rd, vd, distill_active=False, render_fct=render_fct, **render_kwargs).items() if k in keys}
            for ro, rd, vd in zip(rays_o.split(8192, 0), rays_d.split(8192, 0), viewdirs.split(8192, 0))
        ]
        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H,W,-1)
            for k in render_result_chunks[0].keys()
        }
        rgb = render_result['rgb_marched']
        depth = render_result['depth']
        bgmap = render_result['alphainv_last']
        seg_m = render_result['seg_mask_marched'] if self.segment else None
        dual_seg_m = render_result['dual_seg_mask_marched'] if self.stage == 'fine' else None

        return rgb, depth, bgmap, seg_m, dual_seg_m
    

    def prompt_and_inverse(self, idx, HW, seg_m, dual_seg_m, depth, num_obj=1):
        H, W = HW[idx]
        index_matrix = _generate_index_matrix(H, W, depth.detach().clone())

        if self.stage == 'coarse': # coarse stage, get sam seg
            loss, sam_seg_show = self.prompting_coarse(H, W, seg_m, index_matrix, num_obj)
        elif self.stage == 'fine':
            loss, sam_seg_show, _ = self.prompting_fine(H, W, seg_m, dual_seg_m, index_matrix, num_obj)
        else:
            raise NotImplementedError
        optim(self.optimizer, loss, model=self.render_viewpoints_kwargs['model'])

        return sam_seg_show


    def inverse(self, seg_m, sam_mask):
        loss = seg_loss(sam_mask, None, seg_m)
        optim(self.optimizer, loss, model=self.render_viewpoints_kwargs['model'])


    def train_step(self, idx, sam_mask=None):
        render_poses, HW, Ks = fetch_seg_poses(self.args.seg_poses, self.data_dict)
        assert(idx < len(render_poses))

        rgb, depth, bgmap, seg_m, dual_seg_m = self.render_view(idx, [render_poses, HW, Ks])
        if sam_mask is None:
            self.predictor.set_image(utils.to8b(rgb.cpu().numpy()))
            sam_seg_show = self.prompt_and_inverse(idx, HW, seg_m, dual_seg_m, depth)
        else:
            self.inverse(seg_m, sam_mask)
            sam_seg_show = None
        
        # logging
        rgb = rgb.detach().cpu().numpy()
        seg_m = seg_m.detach().cpu().numpy()
        recolored_img = utils.to8b(0.4 * rgb + 0.6 * (seg_m>0))
        if sam_seg_show is not None: sam_seg_show = utils.to8b(sam_seg_show)
        return recolored_img, sam_seg_show, idx >= len(render_poses)-1


    def save_ckpt(self):
        if self.args.save_ckpt:
            model = self.render_viewpoints_kwargs['model']
            torch.save({
                'model_kwargs': model.get_kwargs(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, os.path.join(self.base_save_dir, f'{self.stage}_segmentation'+self.e_flag+'.tar'))
            print(f'scene_rep_reconstruction ({self.stage}): saved checkpoints at', os.path.join(self.base_save_dir, f'{self.stage}_segmentation'+self.e_flag+'.tar'))
        else:
            print('Did not add --save_ckpt in parser. Therefore, ckpt is not saved.')
    
    def render_test(self):
        if self.args.ft_path:
            ckpt_path = self.args.ft_path
        else:
            fine_path = os.path.join(self.cfg.basedir, self.cfg.expname, 'fine_segmentation'+self.e_flag+'.tar')
            coarse_path = os.path.join(self.cfg.basedir, self.cfg.expname, 'coarse_segmentation'+self.e_flag+'.tar')
            ckpt_path = fine_path if os.path.exists(fine_path) else coarse_path
        # print("\033[96mRendering with ckpt "+ckpt_path+"\033[0m")
        ckpt_name = ckpt_path.split('/')[-1][:-4]
        
        videos = []
        for seg_type in ['seg_img', 'seg_density']:
            # rendering
            flag = "seg" if self.args.segment else ""
            if self.args.segment:
                if seg_type == 'seg_density':
                    self.render_viewpoints_kwargs['model'].segmentation_to_density()
                elif seg_type == 'seg_img':
                    self.render_viewpoints_kwargs['model'].segmentation_only()
                else:
                    raise NotImplementedError('seg type {} is not implemented!'.format(seg_type))

            # default: one object    
            num_obj = self.render_viewpoints_kwargs['model'].seg_mask_grid.grid.shape[1]
            self.render_viewpoints_kwargs['model'] = self.render_viewpoints_kwargs['model'].cuda()
            video = render_fn(self.args, self.cfg, ckpt_name, flag, self.e_flag, num_obj, \
                                   self.data_dict, self.render_viewpoints_kwargs, seg_type=seg_type)
            videos.append(video)
        return videos

    def seg_init_frame_coarse(self):
        '''for coarse stage init, we need to set a prompt for the user to select a mask'''
        with torch.no_grad():
            prompts = get_prompt_points(self.args, sam=self.predictor, 
                    ctx=self.context, init_rgb=self.init_image)
            input_point = prompts['prompt_points']
            input_label = np.ones(len(input_point))

            masks, scores, logits = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )

        if prompts['mask_id'] is None:
            for j, mask in enumerate(masks): 
                ### for selection
                plt.figure(figsize=(10,10))
                plt.imshow(mask)
                plt.axis('on')
                plt.savefig('tmp_mask_'+str(j)+'.jpg')
            selected_mask = int(input("Please select a mask:"))
        else:
            selected_mask = prompts['mask_id']

        # record the selected prompt and mask
        with open(os.path.join(self.base_save_dir, "user-specific-prompt.json"), 'w') as f:
            prompt_dict = {
                "mask_id": selected_mask,
                "prompt_points": input_point.tolist()
            }
            json.dump(prompt_dict, f)
        print(f"Prompt saved in {os.path.join(self.base_save_dir, 'user-specific-prompt.json')}")
        
        sam_seg_show = masks[selected_mask].astype(np.float32)
        sam_seg_show = np.stack([sam_seg_show,sam_seg_show,sam_seg_show], axis = -1)
        for ip, point in enumerate(input_point):
            sam_seg_show[point[1]-3 : point[1]+3, point[0] - 3 : point[0]+3, :] = 0
            if ip < 3:
                sam_seg_show[point[1]-3 : point[1]+3, point[0] - 3 : point[0]+3, ip] = 1
            else:
                sam_seg_show[point[1]-3 : point[1]+3, point[0] - 3 : point[0]+3, 2] = 1

        return masks, scores, logits, selected_mask, sam_seg_show


    def seg_init_frame_fine(self, seg_m, model, dual_seg_m):
        '''for fine stage, we load the user-specific prompt and mask'''
        # get the recorded user-specific prompt and the corresponding mask
        with open(os.path.join(self.base_save_dir, "user-specific-prompt.json"), 'r') as f:
            prompt_dict = json.load(f)
            mask_id = prompt_dict['mask_id']
            input_point = np.array(prompt_dict['prompt_points'])

        with torch.no_grad():
            # input_point = get_prompt_points(self.args)['prompt_points']
            input_label = np.ones(len(input_point))

            masks, scores, logits = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )

        print("user-specific-prompt loaded, the specified prompt mask id is:", mask_id)
        target_mask = torch.as_tensor(masks[mask_id]).float().to(seg_m.device)
        
        # the rendered segmentation result
        tmp_rendered_mask = seg_m[:,:,0].detach().clone()
        tmp_rendered_mask[torch.logical_or(tmp_rendered_mask <= tmp_rendered_mask.mean(), tmp_rendered_mask <= 0)] = 0
        tmp_rendered_mask[tmp_rendered_mask != 0] = 1

        # get the dual segmentation target
        dual_target = torch.zeros_like(tmp_rendered_mask)
        dual_target[(tmp_rendered_mask - target_mask) == 1] = 1
        
        IoU = utils.cal_IoU(tmp_rendered_mask, target_mask)
        print("Current IoU is", IoU)
        if IoU > 0.9:
            print("IoU is larger than 0.9, no refinement is required. Use Ctrl+C to cancel the fine stage training.")
            time.sleep(5)
            print("Begin refinement.")
            
        
        model.seg_mask_grid.grid.data = torch.zeros_like(model.seg_mask_grid.grid)
        model.dual_seg_mask_grid.grid.data = torch.zeros_like(model.seg_mask_grid.grid)

        sam_seg_show = masks[mask_id].astype(np.float32)
        sam_seg_show = np.stack([sam_seg_show,sam_seg_show,sam_seg_show], axis = -1)
        dual_sam_seg_show = dual_target.detach().cpu().numpy().astype(np.float32)
        dual_sam_seg_show = np.stack([dual_sam_seg_show,dual_sam_seg_show,dual_sam_seg_show], axis = -1)

        return target_mask, dual_target, sam_seg_show, dual_sam_seg_show


    def prompting_coarse(self, H, W, seg_m, index_matrix, num_obj):
        '''TODO, for coarse stage, we use the self-prompting method to generate the prompt and mask'''
        seg_m_clone = seg_m.detach().clone()
        seg_m_for_prompt = seg_m_clone
        # kernel_size = 3
        # padding = kernel_size // 2
        # seg_m_for_prompt = torch.nn.functional.avg_pool2d(seg_m_clone.permute([2,0,1]).unsqueeze(0), kernel_size, stride = 1, padding = padding)
        # seg_m_for_prompt = seg_m_for_prompt.squeeze(0).permute([1,2,0])

        loss = 0

        for num in range(num_obj):
            with torch.no_grad():
                # self-prompting
                prompt_points, input_label = mask_to_prompt(predictor = self.predictor, rendered_mask_score = seg_m_for_prompt[:,:,num][:,:,None], 
                                                            index_matrix = index_matrix, num_prompts = self.args.num_prompts)

                masks, selected = None, -1
                if len(prompt_points) != 0:
                    masks, scores, logits = self.predictor.predict(
                        point_coords=prompt_points,
                        point_labels=input_label,
                        multimask_output=False,
                    )
                    selected = np.argmax(scores)

            if num == 0:
                # used for single object only
                sam_seg_show = masks[selected].astype(np.float32) if masks is not None else np.zeros((H,W))
                sam_seg_show = np.stack([sam_seg_show,sam_seg_show,sam_seg_show], axis = -1)
                r = 8
                for ip, point in enumerate(prompt_points):
                    sam_seg_show[point[1]-r : point[1]+r, point[0] - r : point[0]+r, :] = 0
                    if ip < 3:
                        sam_seg_show[point[1]-r : point[1]+r, point[0] - r : point[0]+r, ip] = 1
                    else:
                        sam_seg_show[point[1]-r : point[1]+r, point[0] - r : point[0]+r, -1] = 1
                    

            if masks is not None:
                tmp_seg_m = seg_m[:,:,num]
                tmp_rendered_mask = tmp_seg_m.detach().clone()
                tmp_rendered_mask[torch.logical_or(tmp_rendered_mask <= tmp_rendered_mask.mean(), tmp_rendered_mask <= 0)] = 0
                tmp_rendered_mask[tmp_rendered_mask != 0] = 1
                tmp_IoU = utils.cal_IoU(torch.as_tensor(masks[selected]).float(), tmp_rendered_mask)
                print(f"current IoU is: {tmp_IoU}")
                if tmp_IoU < 0.5:
                    print("SKIP, unacceptable sam prediction, IoU is", tmp_IoU)
                    continue

                loss += seg_loss(masks, selected, tmp_seg_m, self.args.lamb)
                for neg_i in range(seg_m.shape[-1]):
                    if neg_i == num:
                        continue
                    loss += (torch.tensor(masks[selected]).to(seg_m.device) * seg_m[:,:,neg_i]).sum()
        return loss, sam_seg_show


    def prompting_fine(self, H, W, seg_m, dual_seg_m, index_matrix, num_obj):
        '''TODO, for fine stage, we use the self-prompting method to generate the prompt and mask'''
        loss = 0
        # get the prompt of interest
        seg_m_clone = seg_m.detach().clone()
        seg_m_for_prompt = torch.nn.functional.avg_pool2d(seg_m_clone.permute([2,0,1]).unsqueeze(0), 25, stride = 1, padding = 12)
        seg_m_for_prompt = seg_m_for_prompt.squeeze(0).permute([1,2,0])
        # get the dual prompt of interest
        dual_seg_m_clone = dual_seg_m.detach().clone()
        dual_seg_m_for_prompt = torch.nn.functional.avg_pool2d(dual_seg_m_clone.permute([2,0,1]).unsqueeze(0), 25, stride = 1, padding = 12)
        dual_seg_m_for_prompt = dual_seg_m_for_prompt.squeeze(0).permute([1,2,0])
        
        for num in range(num_obj):
            tmp_seg_m = seg_m[:,:,num]
            dual_tmp_seg_m = dual_seg_m[:,:,num]
            
            with torch.no_grad():
                # rendered segmentation mask
                tmp_rendered_mask = tmp_seg_m.detach().clone()
                tmp_rendered_mask[torch.logical_or(tmp_rendered_mask <= tmp_rendered_mask.mean(), tmp_rendered_mask <= 0)] = 0
                tmp_rendered_mask[tmp_rendered_mask != 0] = 1

                # rendered dual segmentation mask
                tmp_rendered_dual_mask = dual_tmp_seg_m.detach().clone()
                tmp_rendered_dual_mask[torch.logical_or(tmp_rendered_dual_mask <= tmp_rendered_dual_mask.mean(), tmp_rendered_dual_mask <= 0)] = 0
                tmp_rendered_dual_mask[tmp_rendered_dual_mask != 0] = 1

            
                # self-prompting
                ori_prompt_points, ori_input_label = mask_to_prompt(predictor = self.predictor, \
                    rendered_mask_score = seg_m_for_prompt[:,:,num].unsqueeze(-1), index_matrix = index_matrix, num_prompts = self.args.num_prompts)
                num_self_prompts = len(ori_prompt_points)

                # dual self-prompting
                dual_prompt_points, dual_input_label = mask_to_prompt(predictor = self.predictor, \
                    rendered_mask_score = dual_seg_m_for_prompt[:,:,num].unsqueeze(-1), index_matrix = index_matrix, num_prompts = self.args.num_prompts)                
                num_dual_self_prompts = len(dual_prompt_points)

                masks, dual_masks = None, None
                # self-prompting
                if num_self_prompts != 0:
                    prompt_points = np.concatenate([ori_prompt_points, dual_prompt_points], axis = 0)
                    input_label = np.concatenate([ori_input_label, 1-dual_input_label], axis = 0)
                    # generate mask
                    masks, scores, logits = self.predictor.predict(
                        point_coords=prompt_points,
                        point_labels=input_label,
                        multimask_output=False,
                    )
                    
                # dual self-prompting
                if num_dual_self_prompts != 0:
                    prompt_points = np.concatenate([ori_prompt_points, dual_prompt_points], axis = 0)
                    input_label = np.concatenate([1-ori_input_label, dual_input_label], axis = 0)
                    # generate dual mask
                    dual_masks, dual_scores, dual_logits = self.predictor.predict(
                        point_coords=prompt_points,
                        point_labels=input_label,
                        multimask_output=False,
                    )

            r = 8
            if num == 0:
                # used for single object only
                sam_seg_show = masks[0].astype(np.float32) if masks is not None else np.zeros((H,W))
                sam_seg_show = np.stack([sam_seg_show,sam_seg_show,sam_seg_show], axis = -1)
                for point in ori_prompt_points:
                    sam_seg_show[point[1]-r : point[1]+r, point[0] - r : point[0]+r, :] = 0
                    sam_seg_show[point[1]-r : point[1]+r, point[0] - r : point[0]+r, 0] = 1
                for point in dual_prompt_points:
                    sam_seg_show[point[1]-r : point[1]+r, point[0] - r : point[0]+r, :] = 0
                    sam_seg_show[point[1]-r : point[1]+r, point[0] - r : point[0]+r, 2] = 1
                
                dual_sam_seg_show = dual_masks[0].astype(np.float32)  if dual_masks is not None else np.zeros((H,W))
                dual_sam_seg_show = np.stack([dual_sam_seg_show,dual_sam_seg_show,dual_sam_seg_show], axis = -1)
                for point in dual_prompt_points:
                    dual_sam_seg_show[point[1]-r : point[1]+r, point[0] - r : point[0]+r, :] = 0
                    dual_sam_seg_show[point[1]-r : point[1]+r, point[0] - r : point[0]+r, 0] = 1
                for point in ori_prompt_points:
                    dual_sam_seg_show[point[1]-r : point[1]+r, point[0] - r : point[0]+r, :] = 0
                    dual_sam_seg_show[point[1]-r : point[1]+r, point[0] - r : point[0]+r, 2] = 1
                
            if masks is not None:
                tmp_IoU = utils.cal_IoU(torch.as_tensor(masks[0]).float(), tmp_rendered_mask)
                print("tmp_IoU:", tmp_IoU)
                if tmp_IoU < 0.5:
                    print("SKIP, unacceptable sam prediction for original seg, IoU is", tmp_IoU)
                else:
                    loss += seg_loss(masks[0], None, tmp_seg_m, self.args.lamb)
                    # loss += -(torch.tensor(masks[0]).to(seg_m.device) * tmp_seg_m).sum() + 0.15 * (torch.tensor(1-masks[0]).to(seg_m.device) * tmp_seg_m).sum()
                    for neg_i in range(seg_m.shape[-1]):
                        if neg_i == num: 
                            continue
                        loss -= seg_loss(masks[0], None, seg_m[:,:,neg_i], 0)
                        # loss += (torch.tensor(masks[0]).to(seg_m.device) * seg_m[:,:,neg_i]).sum()

                if dual_masks is not None:
                    tmp_IoU = utils.cal_IoU(torch.as_tensor(dual_masks[0]).float(), tmp_rendered_dual_mask)
                    print("tmp_dual_IoU:", tmp_IoU)
                    if tmp_IoU < 0.5:
                        print("SKIP, unacceptable sam prediction for dual seg, IoU is", tmp_IoU)
                    else:
                        loss += seg_loss(dual_masks[0], None, dual_tmp_seg_m, self.args.lamb)
                        # loss += -(torch.tensor(dual_masks[0]).to(seg_m.device) * dual_tmp_seg_m).sum() + 0.15 * (torch.tensor(1-dual_masks[0]).to(dual_seg_m.device) * dual_tmp_seg_m).sum()
                        for neg_i in range(dual_seg_m.shape[-1]):
                            if neg_i == num: 
                                continue
                            loss -= seg_loss(dual_masks[0], None, dual_seg_m[:,:,neg_i], 0)
                            # loss += (torch.tensor(dual_masks[0]).to(seg_m.device) * dual_seg_m[:,:,neg_i]).sum()
        
        return loss, sam_seg_show, dual_sam_seg_show


def seg_loss(mask: Tensor, selected_mask: Optional[Tensor], seg_m: Tensor, lamda: float = 5.0) -> Tensor:
    """
    Compute segmentation loss using binary mask and predicted mask.

    Args:
        mask: Binary ground truth segmentation mask tensor.
        selected_mask: Tensor indicating which indices in `mask` to select. Can be `None`.
        seg_m: Predicted segmentation mask tensor.
        lamda: Weighting factor for outside mask loss. Default is 5.0.

    Returns:
        Computed segmentation loss.

    Raises:
        AssertionError: If `seg_m` is `None`.
    """
    assert seg_m is not None, "Segmentation mask is None."
    device = seg_m.device
    if selected_mask is not None:
        mask_loss = -(utils.to_tensor(mask[selected_mask], device) * seg_m.squeeze(-1)).sum()
        out_mask_loss = lamda * (utils.to_tensor(1 - mask[selected_mask], device) * seg_m.squeeze(-1)).sum()
    else:
        mask_loss = -(utils.to_tensor(mask, device) * seg_m.squeeze(-1)).sum()
        out_mask_loss = lamda * ((1 - utils.to_tensor(mask, device)) * seg_m.squeeze(-1)).sum()
    return mask_loss + out_mask_loss


def optim(optimizer, loss, clip=None, model=None):
    """Perform a single optimization step using the given optimizer and loss.

    Args:
        optimizer: PyTorch optimizer to use for the optimization step.
        loss: The loss tensor to optimize.
        clip: Optional gradient clipping value.
        model: Optional PyTorch model whose parameters to clip.

    Raises:
        TypeError: If the loss is not a tensor.
    """
    if isinstance(loss, torch.Tensor):
        optimizer.zero_grad()
        loss.backward()
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        if model is not None:
            with torch.no_grad():
                model.seg_mask_grid.grid *= model.mask_view_counts
                prev_mask_grid = model.seg_mask_grid.grid.detach().clone()
        optimizer.step()
        # average mask score by view counts
        if model is not None:
            with torch.no_grad():
                model.mask_view_counts += (model.seg_mask_grid.grid != prev_mask_grid)
                model.seg_mask_grid.grid /= (model.mask_view_counts + 1e-9)
    else:
        pass


def _generate_index_matrix(H, W, depth_map):
    '''generate the index matrix, which contains the coordinate of each pixel and cooresponding depth'''
    xs = torch.arange(1, H+1) / H # NOTE, range (1, H) = arange(1, H+1)
    ys = torch.arange(1, W+1) / W
    grid_x, grid_y = torch.meshgrid(xs, ys)
    index_matrix = torch.stack([grid_x, grid_y], dim = -1) # [H, W, 2]
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) # [H, W, 1]
    index_matrix = torch.cat([index_matrix, depth_map], dim = -1)
    return index_matrix


def fetch_seg_poses(seg_poses_type, data_dict):
    if seg_poses_type == 'train':
        render_poses=data_dict['poses'][data_dict['i_train']]
        HW=data_dict['HW'][data_dict['i_train']]
        Ks=data_dict['Ks'][data_dict['i_train']]
    elif seg_poses_type == 'video':
        render_poses=data_dict['render_poses']
        HW=data_dict['HW'][data_dict['i_test']][[0]].repeat(len(render_poses), 0)
        Ks=data_dict['Ks'][data_dict['i_test']][[0]].repeat(len(render_poses), 0)
    else:
        raise NotImplementedError

    return render_poses, HW, Ks
