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
Segment Anything in 3D configuration file.
"""

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.plugins.types import MethodSpecification

from sa3d.sa3d_datamanager import SA3DDataManagerConfig
from sa3d.sa3d import SA3DModelConfig
from sa3d.sa3d_pipeline import SA3DPipelineConfig
from sa3d.sa3d_trainer import SA3DTrainerConfig
from sa3d.sa3d_optimizer import SGDOptimizerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from sa3d.self_prompting.sam3d import SAM3DConfig
from sa3d.sa3d_field import TCNNMaskFieldConfig

sa3d_method = MethodSpecification(
    config=SA3DTrainerConfig(
        method_name="sa3d",
        max_num_iterations=15000,
        save_only_latest_checkpoint=True,
        mixed_precision=False,
        pipeline=SA3DPipelineConfig(
            text_prompt='the center object',
            datamanager=SA3DDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(
                    mode="off"
                ),
            ),
            model=SA3DModelConfig(
                mask_fields=TCNNMaskFieldConfig(
                    base_res=128,
                    num_levels=16,
                    max_res=2048,
                    log2_hashmap_size=19,
                    mask_threshold=1e-1
                ),
                eval_num_rays_per_chunk=1 << 11,
                remove_mask_floaters=True,
                # use_lpips=True,
            ),
            network=SAM3DConfig(
                num_prompts=10,
                neg_lamda=1.0
            )
        ),
        optimizers={
            "mask_fields": {
                "optimizer": SGDOptimizerConfig(lr=1e-1),
                # "optimizer": AdamOptimizerConfig(lr=1, eps=1e-15),
                "scheduler": None,
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Segment Anything in 3D method",
)
