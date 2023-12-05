import json
import os
import time
from abc import ABC
from typing import Optional
import math
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from rich.console import Console
CONSOLE = Console(width=120)

from dataclasses import dataclass, field
from typing import Type
from nerfstudio.configs import base_config as cfg

from segment_anything import (SamAutomaticMaskGenerator, SamPredictor,
                              sam_model_registry)
from .utils import cal_IoU, to8b, to_tensor
from .grounding_dino import GroundingDino


@dataclass
class SAM3DConfig(cfg.InstantiateConfig):
    """Configuration for SAM3D instantiation"""
    num_prompts: int = 10
    '''Number of prompts for cross-view self-prompting'''
    stage: str = 'coarse'
    '''TODO: implement the fine stage'''
    neg_lamda: float = 1.0
    '''The weight of negetive loss'''
    iou_thresh: float = 0.5
    '''Skip when the IoU of SAM mask and rendered mask is below iou_thresh'''
    _target: Type = field(default_factory=lambda: SAM3D)


class SAM3D:
    def __init__(self, config: SAM3DConfig, device: str) -> None:
        self.device = device
        self.num_prompts = config.num_prompts
        self.stage = config.stage
        self.neg_lamda = config.neg_lamda
        self.iou_thresh = config.iou_thresh
        sam_checkpoint = "sa3d/self_prompting/dependencies/sam_ckpt/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
        self.predictor = SamPredictor(sam_model)
        CONSOLE.print("SAM loaded!")

    @torch.no_grad()
    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        if outputs['sam_mask'] is not None:
            metrics_dict['iou'] = cal_IoU(to_tensor(outputs['sam_mask'], self.device), \
                                          batch['mask_scores'].detach().clone()>0).item()
            CONSOLE.print("Current IoU is: {:07f}".format(metrics_dict['iou']))
            if metrics_dict['iou'] <= self.iou_thresh:
                CONSOLE.print("Lower than IoU threshold, Unacceptable!")
        else:
            metrics_dict['iou'] = None
            CONSOLE.print("No Mask from SAM!")
        return metrics_dict
    
    def get_loss_dict(self, outputs, batch, metrics_dict, neg_lamda=1.0):
        loss_dict = {'mask': None}
        if metrics_dict['iou'] is not None:
            if metrics_dict['iou'] > self.iou_thresh:
                mask_loss = -(to_tensor(outputs['sam_mask'], self.device) * batch['mask_scores']).sum()
                out_mask_loss = neg_lamda * ((1 - to_tensor(outputs['sam_mask'], self.device)) * batch['mask_scores']).sum()
                loss_dict['mask'] = mask_loss + out_mask_loss
         
        return loss_dict

    def get_outputs(self, batch, init_prompt=None):
        '''
        Main function. 
        If no init prompt, perform self prompting, get sam masks and calculate loss;
        else perform mask initialization.
        '''
        image = to8b(batch['rgb'].cpu().numpy())
        H, W = image.shape[:2]
        self.predictor.set_image(image)
        if init_prompt is None: # cross-view self prompting
            index_matrix = SAM3D._generate_index_matrix(H, W, batch['depth'].detach().clone())
            if self.stage == 'coarse': # coarse stage, get sam seg
                outputs = self.prompting_coarse(batch['mask_scores'], index_matrix)
            elif self.stage == 'fine':
                raise NotImplementedError
            else:
                raise NotImplementedError
            metrics_dict = self.get_metrics_dict(outputs, batch)
            loss_dict = self.get_loss_dict(outputs, batch, metrics_dict, neg_lamda=self.neg_lamda)
        else: # mask initialization
            metrics_dict = {'iou': 1.}
            outputs = {
                'sam_mask': self.init_mask(image, init_prompt),
                'prompt_points': None
            }
            loss_dict = self.get_loss_dict(outputs, batch, metrics_dict, neg_lamda=self.neg_lamda)
        
        outputs.update({"sam_mask_show": SAM3D.visualize_prompts(outputs, [H, W])})
        # for k in outputs.keys():
        #     outputs[k] = to_tensor(outputs[k])
        return outputs, loss_dict, metrics_dict
    
    def init_mask(self, image, prompt):
        if isinstance(prompt, str):
            CONSOLE.print(f"Use text prompt: {prompt}")
            return self.init_mask_with_text(image, prompt)
        elif isinstance(prompt, np.ndarray):
            CONSOLE.print(f"Use point prompt: {prompt}")
            return self.init_mask_with_points(image, prompt)
        else:
            raise NotImplementedError
    
    def init_mask_with_text(self, image, text):
        text2box = GroundingDino()
        input_boxes = text2box(image, text)
        boxes = torch.tensor(input_boxes)[0:1].to(self.device)
        transformed_boxes = self.predictor.transform.apply_boxes_torch(boxes, image.shape[:2])
        masks, scores, logits = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            # multimask_output=True,
            multimask_output=False,
        )
        masks = masks[0].cpu().numpy()
        return masks[0][..., None] #[H, W, 1]
    
    def init_mask_with_points(self, image, points):
        # TODO
        raise NotImplementedError

    @torch.no_grad()
    def prompting_coarse(self, seg_m, index_matrix):
        '''For coarse stage, we use the self-prompting method to generate the prompt and mask.'''
        seg_m_for_prompt = seg_m.detach().clone()
        # kernel_size = 3
        # padding = kernel_size // 2
        # seg_m_for_prompt = torch.nn.functional.avg_pool2d(seg_m_clone.permute([2,0,1]).unsqueeze(0), kernel_size, stride = 1, padding = padding)
        # seg_m_for_prompt = seg_m_for_prompt.squeeze(0).permute([1,2,0])

        prompt_points, input_label = self.mask_to_prompt(rendered_mask_score = seg_m_for_prompt, 
                                                    index_matrix = index_matrix, num_prompts = self.num_prompts)
        if len(prompt_points) != 0:
            masks, scores, logits = self.predictor.predict(
                point_coords=prompt_points,
                point_labels=input_label,
                multimask_output=False,
            )
            sam_mask = masks[0][..., None] #[H, W, 1]
        else:
            sam_mask = None

        outputs = {
            'sam_mask': sam_mask, 
            'prompt_points': prompt_points,
        }
        return outputs
    
    @torch.no_grad()
    def mask_to_prompt(self, rendered_mask_score, index_matrix, num_prompts = 3):
        '''main function for self prompting'''
        h, w, _ = rendered_mask_score.shape
        tmp = rendered_mask_score.view(-1)
        CONSOLE.print("\nRendered Mask Scores: Min: {:05f}, Max: {:05f}".format(tmp.min().item(), tmp.max().item()))
        rand = torch.ones_like(tmp)
        topk_v, topk_p = torch.topk(tmp*rand, k = 1)
        topk_v, topk_p = topk_v.cpu().numpy(), topk_p.cpu().numpy()

        if topk_v <= 0:
            CONSOLE.print("No prompt is available!")
            return np.zeros((0,2)), np.ones((0))

        prompt_points = []
        prompt_points.append([topk_p[0] % w, topk_p[0] // w])
        # CONSOLE.print(f'Highest score Coords: ({(topk_p[0] % w)}, {(topk_p[0] // w)})')

        tmp_mask = rendered_mask_score.detach().clone().cpu().numpy()
        area = to8b(tmp_mask).sum() / 255
        r = np.sqrt(area / math.pi)
        masked_r = max(int(r) // 2, 2)
        # masked_r = max(int(r) // 3, 2)

        pre_tmp_mask_score = None
        for _ in range(num_prompts - 1):
            # mask out a region around the last prompt point
            input_label = np.ones(len(prompt_points))
            previous_masks, previous_scores, previous_logits = self.predictor.predict(
                point_coords=np.array(prompt_points),
                point_labels=input_label,
                multimask_output=False,
            )

            l = 0 if prompt_points[-1][0]-masked_r <= 0 else prompt_points[-1][0]-masked_r
            r = w-1 if prompt_points[-1][0]+masked_r >= w-1 else prompt_points[-1][0]+masked_r

            t = 0 if prompt_points[-1][1]-masked_r <= 0 else prompt_points[-1][1]-masked_r
            b = h-1 if prompt_points[-1][1]+masked_r >= h-1 else prompt_points[-1][1]+masked_r
            tmp_mask[t:b+1, l:r+1, :] = -1e5

            # bool: H W
            previous_mask_tensor = torch.tensor(previous_masks[0])
            previous_mask_tensor = previous_mask_tensor.unsqueeze(0).unsqueeze(0).float()
            previous_mask_tensor = torch.nn.functional.max_pool2d(previous_mask_tensor, 25, stride = 1, padding = 12)
            previous_mask_tensor = previous_mask_tensor.squeeze(0).permute([1,2,0])
    #         tmp_mask[previous_mask_tensor > 0] = -1e5
            previous_max_score = torch.max(rendered_mask_score[previous_mask_tensor > 0]).cpu().numpy()

            previous_point_index = np.zeros_like(index_matrix)
            previous_point_index[:,:,0] = prompt_points[-1][1] / h
            previous_point_index[:,:,1] = prompt_points[-1][0] / w
            previous_point_index[:,:,2] = index_matrix[int(prompt_points[-1][1]), int(prompt_points[-1][0]), 2]
            distance_matrix = np.sqrt(((index_matrix - previous_point_index)**2).sum(-1, keepdims=True))
            distance_matrix = (distance_matrix - distance_matrix.min()) / (distance_matrix.max() - distance_matrix.min())

            cur_tmp_mask = tmp_mask - distance_matrix * max(previous_max_score, 0)

            if pre_tmp_mask_score is None:
                pre_tmp_mask_score = cur_tmp_mask
            else:
                pre_tmp_mask_score[pre_tmp_mask_score < cur_tmp_mask] = cur_tmp_mask[pre_tmp_mask_score < cur_tmp_mask]
                pre_tmp_mask_score[tmp_mask == -1e5] = -1e5

            tmp_val, tmp_points = pre_tmp_mask_score.max(), pre_tmp_mask_score.flatten().argmax()

            if tmp_val <= 0:
                break
            prompt_points.append([int(tmp_points % w), int(tmp_points // w)])
        
        CONSOLE.print(f"Have selected {len(prompt_points)}/{self.num_prompts} prompts")
        prompt_points = np.array(prompt_points)
        input_label = np.ones(len(prompt_points))

        return prompt_points, input_label
    
    @staticmethod
    def visualize_prompts(outputs, HW):
        H, W = HW
        sam_seg_show = 255*outputs['sam_mask'].astype(np.uint8) if outputs['sam_mask'] is not None else np.zeros((H,W,1))
        sam_seg_show = np.concatenate([sam_seg_show,sam_seg_show,sam_seg_show], axis = -1)
        if outputs['prompt_points'] is not None:
            r = 8
            for ip, point in enumerate(outputs['prompt_points']):
                sam_seg_show[point[1]-r : point[1]+r, point[0] - r : point[0]+r, :] = 0
                if ip < 3:
                    sam_seg_show[point[1]-r : point[1]+r, point[0] - r : point[0]+r, ip] = 255
                else:
                    sam_seg_show[point[1]-r : point[1]+r, point[0] - r : point[0]+r, -1] = 255
        return sam_seg_show

    @staticmethod
    def _generate_index_matrix(H, W, _depth_map):
        '''generate the index matrix, which contains the coordinate of each pixel and cooresponding depth'''
        depth_map = _depth_map.detach().clone().cpu().numpy()
        xs = np.arange(1, H+1) / H # NOTE, range (1, H) = arange(1, H+1)
        ys = np.arange(1, W+1) / W
        grid_x, grid_y = np.meshgrid(xs, ys, indexing='ij')
        index_matrix = np.stack([grid_x, grid_y], axis=-1) # [H, W, 2]
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) # [H, W, 1]
        index_matrix = np.concatenate([index_matrix, depth_map], axis=-1)
        return index_matrix