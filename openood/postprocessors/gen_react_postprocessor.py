from typing import Any
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from openood.postprocessors import BasePostprocessor
from scipy.special import softmax


class GENLocalReactPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(GENLocalReactPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.gamma = self.args.gamma
        self.M = self.args.M
        self.percentile = self.args.percentile
        self.clip_quantile_local = self.args.clip_quantile_local
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.setup_flag = False

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            activation_log = []
            net.eval()
            with torch.no_grad():
                for batch in tqdm(id_loader_dict['val'],
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    data = batch['data'].cuda()
                    data = data.float()

                    _, feature = net(data, return_feature=True)
                    activation_log.append(feature.data.cpu().numpy())

            self.activation_log = np.concatenate(activation_log, axis=0)
            self.setup_flag = True
        else:
            pass
        self.threshold = np.percentile(self.activation_log.flatten(),
                                       self.percentile)

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        _, feature = net(data, return_feature=True)
        feature_clipped = np.clip(feature.data.cpu().numpy(), a_min=None, a_max=self.threshold)
        weights, bias = net.fc.weight.data.cpu().numpy(), net.fc.bias.data.cpu().numpy()
        logits = feature_clipped @ weights.T + bias
        softmax_values = softmax(logits, axis=-1)
        GEN_score = self.generalized_entropy(softmax_values, self.gamma, self.M)
        preds = np.argmax(softmax_values, axis=-1)
        return torch.tensor(preds).to(data.device), torch.tensor(GEN_score).to(data.device)

    def generalized_entropy(self, softmax_vals, gamma=0.1, M=100):
        probs_sorted = np.sort(softmax_vals, axis=1)[:, -M:]
        scores = np.sum(probs_sorted ** gamma * (1 - probs_sorted) ** gamma, axis=1)
        return -scores

    def set_hyperparam(self, hyperparam: list):
        self.percentile = hyperparam[0]
        self.gamma = hyperparam[1]
        self.M = hyperparam[2]
        self.threshold = np.percentile(self.activation_log.flatten(),
                                       self.percentile)
        print('Threshold at percentile {:2d} over id data is: {}'.format(
            self.percentile, self.threshold))

    def get_hyperparam(self):
        return [self.percentile,self.gamma, self.M]