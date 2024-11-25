from typing import Optional, Tuple

import torch
import torch.nn.functional as F


class AngularPenaltySMLoss(torch.nn.Module):

    def __init__(
        self, alpha=1.0, beta=1.0
    ) -> None:
        """Angular Penalty Softmax Loss Three 'loss_types' available:

        ['arcface', 'sphereface', 'cosface']

        These losses are described in the following papers:
        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
    
    def forward(
        self, input: torch.Tensor, label: torch.Tensor, mask: torch.Tensor, 
    ) -> Tuple[torch.Tensor, ...]:
        return self.alpha*F.mse_loss(input*mask, label*mask) + self.beta*F.mse_loss(input*(1-mask), label*(1-mask))
