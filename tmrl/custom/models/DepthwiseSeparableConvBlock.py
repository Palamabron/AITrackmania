import torch
from collections import OrderedDict

from cv2 import imshow
from torch import optim, nn

import numpy as np
from torchvision import datasets, transforms
import helper

class DepthwiseSeparableConvBlock(nn.Module):
    def __init__(self,hidden_units: list,output_units: int,use_gpu : bool = False):
        super(DepthwiseSeparableConvBlock, self).__init__()
        self.depthwise = nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1, groups=hidden_units)
        self.pointwise = nn.Conv2d(hidden_units, output_units, kernel_size=1)
        self.hidden_units = hidden_units
        self.output_units = output_units
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.model=createmodel()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = nn.ReLU()(x)
        return x

class DepthwiseSeparableConvNet(nn.Module):
    def init(self, hidden_units: list, output_units: int,num_classes=10, image_size=28, num_blocks=3):
        super(DepthwiseSeparableConvNet, self).init()
        self.hidden_units = hidden_units
        self.output_units= output_units
        self.features = []
        for _ in range(num_blocks):
            self.features.append(DepthwiseSeparableConvBlock(in_channels, out_channels))
            self.features.append(nn.MaxPool2d(2))
            in_channels = out_channels
            out_channels = 2
        self.features = nn.Sequential(self.features)






