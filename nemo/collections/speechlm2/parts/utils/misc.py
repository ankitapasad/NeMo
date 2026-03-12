# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from dataclasses import dataclass, fields

import torch.nn as nn
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf

from nemo.utils import logging


def to_dataclass(cls: dataclass, data: dict | DictConfig, add_if_missing: bool = False) -> dataclass:
    supported = {f.name for f in fields(cls)}
    if isinstance(data, DictConfig):
        cfg_dict = OmegaConf.to_container(data, resolve=True)
    else:
        cfg_dict = dict(data)
    unsupported = [k for k in cfg_dict if k not in supported]
    if unsupported and not add_if_missing:
        logging.warning(
            f"{cls.__name__}: the following config parameters are not supported and will be ignored: %s",
            unsupported,
        )
    filtered = {k: cfg_dict[k] for k in supported if k in cfg_dict}
    result = cls(**filtered)
    if add_if_missing:
        for k in unsupported:
            setattr(result, k, cfg_dict[k])
    return result


def freeze_module(module: nn.Module | LightningModule) -> None:
    if isinstance(module, LightningModule):
        module.freeze()
    else:
        module.eval()
        for param in module.parameters():
            param.requires_grad = False


def unfreeze_module(module: nn.Module | LightningModule) -> None:
    if isinstance(module, LightningModule):
        module.unfreeze()
    else:
        module.train()
        for param in module.parameters():
            param.requires_grad = True
