"""Adapted from: https://github.com/facebookresearch/odin."""
from typing import Any

import torch
import torch.nn as nn

from .base_postprocessor import BasePostprocessor
from openood.preprocessors.transform import normalization_dict

class PRO_TEMPSCALE_Postprocessor(BasePostprocessor):
    def __init__(self, config):
        self.APS_mode = True
        self.hyperparam_search_done = False
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args

        self.temperature = self.args.temperature
        self.noise_level = self.args.noise_level
        try:
            self.input_std = normalization_dict[self.config.dataset.name][1]
        except KeyError:
            self.input_std = [0.5, 0.5, 0.5]
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.gd_steps=1

    def postprocess(self, net: nn.Module, data: Any):
        #data.requires_grad = True

        tempInputs=data.clone().detach()
        #criterion = nn.CrossEntropyLoss()
        
        for step in range(self.gd_steps):
            tempInputs.requires_grad=True
            output = net(tempInputs)
            score = torch.softmax(output / self.temperature, dim=1)
            conf, pred = torch.max(score, dim=1)
            if step==0:
                unperturbed_pred = pred
            # Calculating the perturbation we need to add, that is,
            # the sign of gradient of cross entropy loss w.r.t. input
            
            loss = conf.mean()
            loss.backward()
            # Normalizing the gradient to binary in {0, 1}
            gradient = tempInputs.grad.data

            # Adding small perturbations to images
            #tempInputs = torch.add(data.detach(), gradient, alpha=-self.noise)# increase msp
            tempInputs = torch.add(tempInputs.detach(), gradient.sign(), alpha=-self.noise_level) # decrease msp

        output = net(tempInputs)
        output = output / self.temperature
        # Calculating the confidence after adding perturbations
        nnOutput = output.detach()
        nnOutput = nnOutput - nnOutput.max(dim=1, keepdims=True).values
        nnOutput = nnOutput.exp() / nnOutput.exp().sum(dim=1, keepdims=True)
        conf, _ = nnOutput.max(dim=1)

        return unperturbed_pred, conf

    def set_hyperparam(self, hyperparam: list):
        
        self.noise_level = hyperparam[0]
        self.gd_steps=hyperparam[1]
        self.temperature = hyperparam[2]

    def get_hyperparam(self):
        return [self.noise_level, self.gd_steps, self.temperature]