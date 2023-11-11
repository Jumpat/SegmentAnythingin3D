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
Segment Anything in 3D Optimizers class.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Type, Optional

import torch
from nerfstudio.configs import base_config


# Optimizer related configs
@dataclass
class SGDOptimizerConfig(base_config.PrintableConfig):
    """Basic optimizer config with RAdam"""

    _target: Type = torch.optim.SGD
    """The optimizer class to use."""
    lr: float = 1.
    """The learning rate to use."""
    max_norm: Optional[float] = None
    """The max norm to use for gradient clipping."""

    # TODO: somehow make this more generic. i dont like the idea of overriding the setup function
    # but also not sure how to go about passing things into predefined torch objects.
    def setup(self, params) -> torch.optim.Optimizer:
        """Returns the instantiated object using the config."""
        kwargs = vars(self).copy()
        kwargs.pop("_target")
        kwargs.pop("max_norm")
        return self._target(params, **kwargs)