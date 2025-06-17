from typing import Any, List, Optional

import torch
from torch import nn
from torchsummary import summary

from src.modules.models.module import (
    BaseModule,
    get_module_attr_by_name_recursively,
    get_module_by_name,
    replace_module_by_identity,
)


class Segmentation(BaseModule):
    def __init__(
        self,
        model_name: str,
        #num_classes: int,
        model_repo: Optional[str] = None,
        freeze_layers: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name, model_repo, freeze_layers, **kwargs)
        # get head module
        # head = get_module_by_name(
        #     self.model, [name for name, _ in self.model.named_children()][-1]
        # )
        # # get in_features to head module
        # in_features = get_module_attr_by_name_recursively(
        #     head, 0, "in_features"
        # )
        # # replace head module to new module
        # replace_module_by_identity(
        #     self.model, head, nn.Linear(in_features, num_classes, bias=True)
        # )
        print(self.model)
        #self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        # if self.num_classes == 1:
        #     x = x.squeeze(dim=1)
        return x


if __name__ == '__main__':
    seg = Segmentation('custom/SwinTransformer', 1, None, None, img_size=(256, 4096), patch_size=(4, 4), in_chans=3,
                 embed_dim=48, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=16, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first")