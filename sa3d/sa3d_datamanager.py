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
Segment Anything in 3D Datamanager.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Type
import torch
from rich.progress import Console

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.utils.dataloaders import CacheDataloader
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)

CONSOLE = Console(width=120)

@dataclass
class SA3DDataManagerConfig(VanillaDataManagerConfig):
    """Configuration for the SA3DDataManager."""

    _target: Type = field(default_factory=lambda: SA3DDataManager)
    patch_size: int = 1
    """Size of patch to sample from. If >1, patch-based sampling will be used."""

class SA3DDataManager(VanillaDataManager):
    """Data manager for SA3D."""

    config: SA3DDataManagerConfig

    def setup_train(self):
        """Sets up the data loaders for training"""
        assert self.train_dataset is not None
        CONSOLE.print("Setting up training dataset...")
        self.train_image_dataloader = CacheDataloader(
            self.train_dataset,
            num_images_to_sample_from=self.config.train_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.train_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
        )
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)
        # self.train_pixel_sampler = self._get_pixel_sampler(self.train_dataset, self.config.train_num_rays_per_batch)
        self.train_camera_optimizer = self.config.camera_optimizer.setup(
            num_cameras=self.train_dataset.cameras.size, device=self.device
        )
        self.train_ray_generator = RayGenerator(
            self.train_dataset.cameras.to(self.device),
            self.train_camera_optimizer,
        )

        # pre-fetch the image batch (how images are replaced in dataset)
        self.image_batch = next(self.iter_train_image_dataloader)
        self.len_image_batch = len(self.image_batch["image"])

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        current_index = torch.tensor([step % self.len_image_batch])
        # get current camera, include camera transforms from original optimizer
        camera_transforms = self.train_camera_optimizer(current_index)
        current_camera = self.train_dataparser_outputs.cameras[current_index].to(self.device)
        current_ray_bundle = current_camera.generate_rays(torch.tensor(list(range(1))).unsqueeze(-1), camera_opt_to_camera=camera_transforms)
        batch = {"image": self.image_batch["image"][current_index]} # [H, W, C]
        
        return current_ray_bundle, batch
