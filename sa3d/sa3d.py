# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Model for Segment Anything in 3D.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Tuple, Type
from skimage import morphology
import torch
from torch.nn import Parameter
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.losses import (
    L1Loss,
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
)
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from sa3d.sa3d_field import TCNNMaskFieldConfig


@dataclass
class SA3DModelConfig(NerfactoModelConfig):
    """Configuration for the SA3DModel."""
    mask_fields: TCNNMaskFieldConfig = TCNNMaskFieldConfig()
    '''mask field config.'''
    remove_mask_floaters: bool = False
    '''Remove small regions in the mask.'''
    _target: Type = field(default_factory=lambda: SA3DModel)
    

class SA3DModel(NerfactoModel):
    """Model for SA3D."""

    config: SA3DModelConfig

    def populate_modules(self):
        """Required to use L1 Loss."""
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))
        
        self.mask_fields = self.config.mask_fields.setup(
            aabb=self.scene_box.aabb,
            spatial_distortion=scene_contraction
        )

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["mask_fields"] = list(self.mask_fields.parameters())

        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):
        with torch.no_grad():
            ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
            field_outputs = self.field(ray_samples, compute_normals=self.config.predict_normals)
            weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
            weights_list.append(weights)
            ray_samples_list.append(ray_samples)

            rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
            accumulation = self.renderer_accumulation(weights=weights)

            outputs = {
                "rgb": rgb,
                "accumulation": accumulation,
                "depth": depth,
            }

            if self.config.predict_normals:
                normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
                pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
                outputs["normals"] = self.normals_shader(normals)
                outputs["pred_normals"] = self.normals_shader(pred_normals)

            for i in range(self.config.num_proposal_iterations):
                outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        if self.mask_fields.training:
            with torch.enable_grad():
                outputs["mask_scores"] = self.mask_fields.get_outputs(ray_samples, weights)
        else:
            with torch.no_grad():
                outputs["mask_scores"] = self.mask_fields.get_outputs(ray_samples, weights)

        return outputs
    
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            outputs = self.forward(ray_bundle=ray_bundle)
            for output_name, output in outputs.items():  # type: ignore
                outputs_lists[output_name].append(output)
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            if not torch.is_tensor(outputs_list[0]):
                # TODO: handle lists of tensors as well
                continue
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore

        if 'mask_scores' in outputs.keys():
            with torch.no_grad():
                mask = outputs["mask_scores"].detach()>self.config.mask_fields.mask_threshold
                if self.config.remove_mask_floaters:
                    mask = morphology.remove_small_objects(mask.cpu().numpy(), min_size=30, connectivity=1)
                    mask = torch.from_numpy(mask).to(outputs["rgb"].device)
                outputs["rgb_masked"] = 0.6*mask + 0.4*outputs["rgb"].detach()

        return outputs
