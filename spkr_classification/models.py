# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn


class LinearNet(nn.Module):
    def __init__(self, num_feature, num_class):
        super(LinearNet, self).__init__()

        self.layer_out = nn.Linear(num_feature, num_class)

    def forward(self, x):
        x = self.layer_out(x)

        return x


class OneHiddenLayerReluNet(nn.Module):
    def __init__(self, num_feature, num_class):
        super(OneHiddenLayerReluNet, self).__init__()

        self.layer_1 = nn.Linear(num_feature, 1024)
        self.layer_out = nn.Linear(1024, num_class)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu(x)

        x = self.layer_out(x)

        return x
