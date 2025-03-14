
from typing import Any


import torch
import torch.nn as nn


from .base_postprocessor import BasePostprocessor

class ENT_Postprocessor(BasePostprocessor):

    def __init__(self, config):
        super().__init__(config)
        self.args = None
        self.APS_mode = False
        self.noise_std=0.01
        

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = net(data)
        probabilities=torch.softmax(output,dim=1)
        entropy =-torch.sum(probabilities * torch.log(probabilities+1e-10),dim=1)
        conf = -entropy
        pred = torch.argmax(probabilities,dim=1)


        return pred,conf