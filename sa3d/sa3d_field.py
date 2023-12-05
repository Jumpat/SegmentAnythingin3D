from typing import Dict, Optional, Tuple, Type
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torchtyping import TensorType
from dataclasses import dataclass, field
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.field_components.spatial_distortions import (
    SceneContraction,
    SpatialDistortion,
)
from nerfstudio.fields.base_field import Field

try:
    import tinycudann as tcnn
except ImportError:
    pass


Cal_log2_grid_elements = lambda b, n, m: math.log(b**3 * (1 - (m/b)**(3*n/(n-1))) / (1 - (m/b)**(3/(n-1))), 2)
'''
    For (n_input_dims, features_per_level) = (3, 1), the whole elements in grids should be:
    $$
        b^3 \frac{1 - (m/b)^{3n/(n-1)}}{1 - (m/b)^{3/(n-1)}}
    $$
    while b refers to base_res, n refers to num_levels, and m refers to max_res.
'''

@dataclass
class TCNNMaskFieldConfig(InstantiateConfig):
    """Configuration for field instantiation"""
    _target: Type = field(default_factory=lambda: TCNNMaskField)
    """target class to instantiate"""
    base_res: int = 16
    '''base resolution'''
    num_levels: int = 16
    '''number of levels of the hashmap for the base mlp'''
    max_res: int = 2048
    '''maximum resolution of the hashmap for the base mlp'''
    log2_hashmap_size: int = 19
    '''size of the hashmap for the base mlp'''
    use_pred_normals: bool = False
    '''whether to use predicted normals'''
    mask_threshold: float = 1e-1
    '''threshold for the rendered mask score'''


class TCNNMaskField(Field):
    """Compound Field that uses TCNN

    Args:
        aabb: parameters of scene aabb bounds
        num_levels: number of levels of the hashmap for the base mlp
        max_res: maximum resolution of the hashmap for the base mlp
        log2_hashmap_size: size of the hashmap for the base mlp
        use_pred_normals: whether to use predicted normals
        spatial_distortion: spatial distortion to apply to the scene
    """
    def __init__(
        self,
        config: TCNNMaskFieldConfig,
        aabb: TensorType,
        spatial_distortion: SpatialDistortion = None,
    ) -> None:
        super().__init__()

        # self.register_buffer("aabb", aabb)
        # self.register_buffer("max_res", torch.tensor(max_res))
        # self.register_buffer("num_levels", torch.tensor(num_levels))
        # self.register_buffer("log2_hashmap_size", torch.tensor(log2_hashmap_size))
        max_res = config.max_res
        base_res = config.base_res
        num_levels = config.num_levels
        log2_hashmap_size = config.log2_hashmap_size
        self.aabb = aabb
        self.spatial_distortion = spatial_distortion
        self.use_pred_normals = config.use_pred_normals

        features_per_level: int = 1
        growth_factor = np.exp((np.log(max_res) - np.log(base_res)) / (num_levels - 1))

        self.mask_grids = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "Grid",
	            "type": "Hash",
                "n_levels": num_levels,
                "n_features_per_level": features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_res,
                "per_level_scale": growth_factor,
                "interpolation": "Linear"
            }
        )
        self.mask_grids.params = torch.nn.Parameter(torch.zeros_like(self.mask_grids.params.data), requires_grad=True)
        level_alphas = growth_factor ** (-torch.arange(num_levels))
        level_alphas /= level_alphas.sum()
        self.level_alphas = level_alphas
        # self.register_buffer("level_alphas", level_alphas)

    def get_outputs(self, ray_samples: RaySamples, weights: TensorType) -> TensorType:
        """Computes and returns the mask scores."""
        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        self._sample_locations = positions
        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True
        weights.requires_grad = positions.requires_grad = False
        positions_flat = positions.view(-1, 3)
        mask_weights = self.mask_grids(positions_flat).reshape(*weights.shape[:2], -1) #[num_rays, num_samples, num_levels]
        mask_weights = (mask_weights * self.level_alphas.view(1,1,-1).to(weights.device)).sum(-1, keepdim=True)
        mask_scores = (mask_weights * weights).sum(1) #[num_rays, 1]

        return mask_scores

    def get_density(self, ray_samples: RaySamples) -> Tuple[TensorType[..., 1], TensorType[..., "num_features"]]:
        raise NotImplementedError()
    
if __name__ == '__main__':
    mask_fields = TCNNMaskField(
        aabb=None,
        spatial_distortion=None
    )
    for name, param in mask_fields.named_parameters():
        print(name, param.size())

    optimizer = torch.optim.SGD(mask_fields.parameters(), lr=1)
    bs, n_input_dims = 2, 3
    input = torch.rand((bs, n_input_dims)).cuda()
    output = mask_fields.mask_grids(input)

    print('input: ', input, '\nshape: ', input.shape)
    print('output: ', output, '\nshape: ', output.shape)

    optimizer.zero_grad()
    output.sum().backward()
    optimizer.step()

    output = mask_fields.mask_grids(input)
    print('input: ', input, '\nshape: ', input.shape)
    print('output: ', output, '\nshape: ', output.shape)