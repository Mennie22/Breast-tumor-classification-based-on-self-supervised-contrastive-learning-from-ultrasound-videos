# encoding: utf-8
import time
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os


class Densenet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=32, reduction=0.5, channel=1):
        super(Densenet, self).__init__()
        self.growthrate = growth_rate

        num_planes = 2 * growth_rate
        self.conv1 = nn.Conv2d(channel, num_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm0 = nn.BatchNorm2d(num_planes)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.dense1 = self.make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self.make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self.make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self.make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3] * growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, 1024)

    def make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growthrate))
            in_planes += self.growthrate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool(F.relu(self.norm0(out)))
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.adaptive_avg_pool2d(F.relu(self.bn(out)), 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out).squeeze()
        return out


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growthrate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4 * growthrate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growthrate)
        self.conv2 = nn.Conv2d(4 * growthrate, growthrate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out, x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


def Densenet121():
    return Densenet(Bottleneck, [6, 12, 24, 16], growth_rate=32)


def Densenet169():
    return Densenet(Bottleneck, [6, 12, 32, 32], growth_rate=32)


def Densenet201():
    return Densenet(Bottleneck, [6, 12, 48, 32], growth_rate=32)


def Densenet161():
    return Densenet(Bottleneck, [6, 12, 36, 24], growth_rate=48)


def Densenet264():
    return Densenet(Bottleneck, [6, 12, 64, 48], growth_rate=32)
