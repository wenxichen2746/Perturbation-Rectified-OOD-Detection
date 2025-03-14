# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torchvision


class LRR_ResNet(torchvision.models.ResNet):
    expansion = 1

    def __init__(self, block=torchvision.models.resnet.BasicBlock, layers=[2, 2, 2, 2], num_classes=10, width=64):
        """To make it possible to vary the width, we need to override the constructor of the torchvision resnet."""

        torch.nn.Module.__init__(self)  # Skip the parent constructor. This replaces it.
        self._norm_layer = torch.nn.BatchNorm2d
        self.inplanes = width
        self.dilation = 1
        self.groups = 1
        self.base_width = 64

        # The initial convolutional layer.
        self.conv1 = torch.nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = torch.nn.ReLU(inplace=True)

        # The subsequent blocks.
        self.layer1 = self._make_layer(block, width, layers[0])
        self.layer2 = self._make_layer(block, width*2, layers[1], stride=2, dilate=False)
        self.layer3 = self._make_layer(block, width*4, layers[2], stride=2, dilate=False)
        self.layer4 = self._make_layer(block, width*8, layers[3], stride=2, dilate=False)

        # The last layers.
        self.avgpool = torch.nn.AvgPool2d(4)
        self.fc = torch.nn.Linear(width*8*block.expansion, num_classes)

        # Default init.
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
    '''
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    '''
    def forward(self, x, return_feature=False, return_feature_list=False):
        feature1 = self.relu(self.bn1(self.conv1(x)))
        feature2 = self.layer1(feature1)
        feature3 = self.layer2(feature2)
        feature4 = self.layer3(feature3)
        feature5 = self.layer4(feature4)
        x = self.avgpool(feature5)
        feature = torch.flatten(x, 1)
        logits_cls = self.fc(feature)
        feature_list = [feature1, feature2, feature3, feature4, feature5]
        
        if return_feature:
            return logits_cls, feature
        elif return_feature_list:
            return logits_cls, feature_list
        else:
            return logits_cls

    def get_fc(self):
        # Retrieve the weights and biases of the classifier layer
        fc_weight = self.fc.weight.cpu().detach().numpy()
        fc_bias = self.fc.bias.cpu().detach().numpy() 
        return fc_weight, fc_bias
    def get_fc_layer(self):
        return self.fc

# edgepopup
class PreActBasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBasicBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(in_planes, affine=False)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes, affine=False)

        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(self.expansion * planes, affine=False),
            )

    def forward(self, x):
        out = torch.nn.functional.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x # Important: using out instead of x
        out = self.conv1(out)
        out = self.conv2(torch.nn.functional.relu(self.bn2(out)))
        out += shortcut
        return out


class WidePreActResNet(torch.nn.Module):
    def __init__(self, block=PreActBasicBlock, num_blocks=[2, 2, 2, 2], num_classes=10, widen_factor=2):
        super(WidePreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(256*(widen_factor+1) * block.expansion, affine=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64*(widen_factor+1), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128*(widen_factor+1), num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256*(widen_factor+1), num_blocks[3], stride=2)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Conv2d(256*(widen_factor+1) * block.expansion, num_classes, kernel_size=1, bias=False)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return torch.nn.Sequential(*layers)
    '''
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.nn.functional.relu(self.bn1(out))
        out = torch.nn.functional.avg_pool2d(out, 4)
        out = self.fc(out)
        return out.flatten(1)
    '''
    def forward(self, x, return_feature=False, return_feature_list=False):
        feature1 = self.conv1(x)
        feature2 = self.layer1(feature1)
        feature3 = self.layer2(feature2)
        feature4 = self.layer3(feature3)
        feature5 = self.layer4(feature4)

        feature5 = torch.nn.functional.relu(self.bn1(feature5))

        pooled_feature = self.avgpool(feature5)
        feature = pooled_feature.view(pooled_feature.size(0), -1)
        logits_cls = self.fc(pooled_feature).flatten(1)
        feature_list = [feature1, feature2, feature3, feature4, pooled_feature]
        if return_feature:
            return logits_cls, feature
        elif return_feature_list:
            return logits_cls, feature_list
        else:
            return logits_cls
    def get_fc(self):
        # Extract weights and biases from the fully connected layer
        fc = self.fc
        return fc.weight.cpu().detach().numpy(), (fc.bias.cpu().detach().numpy() if fc.bias is not None else None)

    def get_fc_layer(self):
        # Return the fully connected layer directly
        return self.fc