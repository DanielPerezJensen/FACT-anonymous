#!/usr/bin/env python

"""Classes that define our self-created classifier models to aid in
the reproducibility study of
'Generative causal explanations of black-box classifiers'

Retrieved from: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial5/Inception_ResNet_DenseNet.html
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionBlock(nn.Module):
    """InceptionBlock, described in: https://arxiv.org/abs/1409.4842"""

    def __init__(self, c_in, c_out, c_red, act_fn):
        """
        Inputs:
            c_in - Number of input feature maps from the previous layers
            c_red - Dictionary with keys "3x3" and "5x5" specifying the output of the dimensionality reducing 1x1 convolutions
            c_out - Dictionary with keys "1x1", "3x3", "5x5", and "max"
            act_fn - Activation class constructor (e.g. nn.ReLU)
        """
        super().__init__()

        # 1x1 convolution branch
        self.conv_1x1 = nn.Sequential(
                nn.Conv2d(c_in, c_out["1x1"], kernel_size=1),
                nn.BatchNorm2d(c_out["1x1"]),
                act_fn()
            )

        # 3x3 convolution branch
        self.conv_3x3 = nn.Sequential(
                nn.Conv2d(c_in, c_red["3x3"], kernel_size=1),
                nn.BatchNorm2d(c_red["3x3"]),
                act_fn(),
                nn.Conv2d(c_red["3x3"], c_out["3x3"], kernel_size=3, padding=1),
                nn.BatchNorm2d(c_out["3x3"]),
                act_fn()
            )

        # 5x5 convolution branch
        self.conv_5x5 = nn.Sequential(
                nn.Conv2d(c_in, c_red["5x5"], kernel_size=1),
                nn.BatchNorm2d(c_red["5x5"]),
                act_fn(),
                nn.Conv2d(c_red["5x5"], c_out["5x5"], kernel_size=3, padding=1),
                nn.BatchNorm2d(c_out["5x5"]),
                act_fn()
            )

        # Max-pool branch
        self.max_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            nn.Conv2d(c_in, c_out["max"], kernel_size=1),
            nn.BatchNorm2d(c_out["max"]),
            act_fn()
        )

    def forward(self, x):
        x_1x1 = self.conv_1x1(x)
        x_3x3 = self.conv_3x3(x)
        x_5x5 = self.conv_5x5(x)
        x_max = self.max_pool(x)
        x_out = torch.cat([x_1x1, x_3x3, x_5x5, x_max], dim=1)
        return x_out


class InceptionNetDerivative(nn.Module):
    """Derivative of Google net with only one inception block"""

    def __init__(self, num_classes=10, act_fn=nn.ReLU):
        super().__init__()

        self.act_fn = act_fn

        self.input_net = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                self.act_fn()
            )

        self.blocks = nn.Sequential(
                InceptionBlock(64, c_red={"3x3": 32, "5x5": 16},
                               c_out={"1x1": 16, "3x3": 48, "5x5": 8, "max": 8},
                               act_fn=self.act_fn),
                nn.MaxPool2d(3, stride=2, padding=1)
            )

        self.output_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(80, num_classes)
            )

    def forward(self, x):
        x = self.input_net(x)
        x = self.blocks(x)
        y = self.output_net(x)
        prob = F.softmax(y, dim=-1)

        return prob, y


class ResNetBlock(nn.Module):
    """Original ResNetBlock, described in https://arxiv.org/abs/1603.05027"""

    def __init__(self, c_in, act_fn, c_out):
        """
        Inputs:
            c_in - Number of input dims
            act_fn - Activation function (e.g. nn.ReLU)
            c_out - Number of classes
        """
        super().__init__()

        self.net = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=1, bias=False),
                nn.BatchNorm2d(c_out),
                act_fn(),
                nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_out)
            )

        self.act_fn = act_fn()

    def forward(self, x):
        z = self.net(x)

        out = z + x
        out = self.act_fn(out)

        return out


class ResNetDerivative(nn.Module):
    """Derivative of ResNet with only one group of blocks"""

    def __init__(self, num_classes=10, act_fn=nn.ReLU):
        super().__init__()

        self.num_classes = num_classes
        self.act_fn = act_fn

        self.input_net = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(16),
                self.act_fn()
            )

        self.blocks = nn.Sequential(
                ResNetBlock(16, self.act_fn, 16),
                ResNetBlock(16, self.act_fn, 16),
                ResNetBlock(16, self.act_fn, 16)
            )

        self.output_net = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(16, self.num_classes)
            )

    def forward(self, x):
        x = self.input_net(x)
        x = self.blocks(x)
        y = self.output_net(x)
        prob = F.softmax(y, dim=-1)

        return prob, y


class DenseLayer(nn.Module):
    """Dense layer from DenseNet, described in https://arxiv.org/abs/1608.06993"""

    def __init__(self, c_in, bn_size, growth_rate, act_fn):
        """
        Inputs:
            c_in - Number of input channels
            bn_size - Bottleneck size (factor of growth rate) for the output of the 1x1 convolution. Typically between 2 and 4.
            growth_rate - Number of output channels of the 3x3 convolution
            act_fn - Activation class constructor (e.g. nn.ReLU)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(c_in),
            act_fn(),
            nn.Conv2d(c_in, bn_size * growth_rate, kernel_size=1, bias=False),
            nn.BatchNorm2d(bn_size * growth_rate),
            act_fn(),
            nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        out = self.net(x)
        out = torch.cat([out, x], dim=1)
        return out


class DenseBlock(nn.Module):
    """Dense Block from DenseNet, described in https://arxiv.org/abs/1608.06993"""

    def __init__(self, c_in, num_layers, bn_size, growth_rate, act_fn=nn.ReLU):
        """
        Inputs:
            c_in - Number of input channels
            num_layers - Number of dense layers to apply in the block
            bn_size - Bottleneck size to use in the dense layers
            growth_rate - Growth rate to use in the dense layers
            act_fn - Activation function to use in the dense layers
        """
        super().__init__()
        layers = []

        for layer_idx in range(num_layers):
            layers.append(DenseLayer(c_in=c_in + layer_idx * growth_rate,
                                     bn_size=bn_size,
                                     growth_rate=growth_rate,
                                     act_fn=act_fn))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        return out


class TransitionLayer(nn.Module):
    """Transition Layer from DenseNet, described in https://arxiv.org/abs/1608.06993"""

    def __init__(self, c_in, c_out, act_fn):
        super().__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(c_in),
            act_fn(),
            nn.Conv2d(c_in, c_out, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.transition(x)


class DenseNetDerivative(nn.Module):
    """Derivative of ResNet with only one group of blocks"""
    def __init__(self, num_classes=2, num_layers=[3], bn_size=2, growth_rate=16, act_fn=nn.ReLU):
        super().__init__()

        c_hidden = growth_rate * bn_size

        self.input_net = nn.Sequential(
                nn.Conv2d(1, c_hidden, kernel_size=3, padding=1)
            )

        # Creating dense block, including transition layers for the last block
        blocks = []
        for block_idx, num_layer in enumerate(num_layers):
            blocks.append(
                DenseBlock(c_in=c_hidden,
                           num_layers=num_layer,
                           bn_size=bn_size,
                           growth_rate=growth_rate,
                           act_fn=act_fn)
                )
            c_hidden = c_hidden + num_layer * growth_rate

            if block_idx < len(num_layers) - 1:
                blocks.append(
                    TransitionLayer(c_in=c_hidden,
                                    c_out=c_hidden // 2,
                                    act_fn=act_fn)
                    )
                c_hidden = c_hidden // 2

        self.blocks = nn.Sequential(*blocks)

        self.output_net = nn.Sequential(
            nn.BatchNorm2d(c_hidden),
            act_fn()
            ,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(c_hidden, num_classes)
        )

    def forward(self, x):
        x = self.input_net(x)
        x = self.blocks(x)
        out = self.output_net(x)
        prob = F.softmax(out, dim=-1)

        return prob, out
