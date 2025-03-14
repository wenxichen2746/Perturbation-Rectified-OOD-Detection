import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Tuple, TypeVar
from torch import Tensor


class ImageNormalizer(nn.Module):

    def __init__(self, mean: Tuple[float, float, float],
                 std: Tuple[float, float, float]) -> None:
        super(ImageNormalizer, self).__init__()

        self.register_buffer('mean', torch.as_tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.as_tensor(std).view(1, 3, 1, 1))

    def forward(self, input: Tensor) -> Tensor:
        return (input - self.mean) / self.std

    def __repr__(self):
        return f'ImageNormalizer(mean={self.mean.squeeze()}, std={self.std.squeeze()})'  # type: ignore

class CusSequentialModel(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super(CusSequentialModel, self).__init__()
        self.model = model

    def forward(self, input: Tensor, return_feature=False, return_feature_list=False) -> Tensor:
        return self.model(input, return_feature=return_feature, return_feature_list=return_feature_list)
    def get_fc(self):
        # Retrieve the weights and biases of the classifier layer
        fc_weight = self.model.fc.weight.cpu().detach().numpy()
        fc_bias = self.model.fc.bias.cpu().detach().numpy() 
        return fc_weight, fc_bias
    def get_fc_layer(self):
        return self.model.fc

def normalize_model(model: nn.Module, mean: Tuple[float, float, float],
                    std: Tuple[float, float, float],justmodel=False) -> nn.Module:
    if justmodel:
        print('abort normalizer from robustbench returning model only')
        return CusSequentialModel(model)
    #robustbench normalizer
    layers = OrderedDict([('normalize', ImageNormalizer(mean, std)),
                          ('model', model)])
    return nn.Sequential(layers)
    
    #return CustomSequentialModel(model, mean, std) # add robust bench to OpenOOD



M = TypeVar('M', bound=nn.Module)


def normalize_timm_model(model: M) -> M:
    return normalize_model(
        model,
        mean=model.default_cfg['mean'],  # type: ignore
        std=model.default_cfg['std'])  # type: ignore


