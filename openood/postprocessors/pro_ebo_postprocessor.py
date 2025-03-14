from typing import Any
import torch
import torch.nn as nn

from .base_postprocessor import BasePostprocessor
from openood.preprocessors.transform import normalization_dict

class PRO_EBO_Postprocessor(BasePostprocessor):
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
        self.gd_steps = 2  # Gradient descent steps

    def postprocess(self, net: nn.Module, data: Any):
        tempInputs = data.clone().detach()

        # Perform gradient descent on the input to maximize energy score
        for step in range(self.gd_steps):
            tempInputs.requires_grad = True
            # Forward pass through the network
            output = net(tempInputs)
            if step==0:
                pred = output.argmax(dim=1)
            energy_score = self.temperature * torch.logsumexp(output / self.temperature, dim=1)
            # We want to maximize energy (or minimize negative energy)
            loss = energy_score.mean()
            loss.backward()
            gradient = tempInputs.grad.data
            # Modify the input using the gradient (gradient ascent on energy score)
            tempInputs = torch.add(tempInputs.detach(), gradient.sign(), alpha=self.noise_level)

        # Final pass through the network with the perturbed input
        output = net(tempInputs.detach())
        
        # Calculate the energy score for the perturbed input
        energy_score = self.temperature * torch.logsumexp(output / self.temperature, dim=1)
        # Calculate confidence and prediction
        conf = energy_score.detach()

        return pred, conf

    def set_hyperparam(self, hyperparam: list):
        self.noise_level = hyperparam[0]
        self.gd_steps = hyperparam[1]

    def get_hyperparam(self):
        return [self.noise_level, self.gd_steps]
