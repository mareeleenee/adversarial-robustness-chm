'''ResNet-18 + MeanSparse: post-training mean-centered feature sparsification.

MeanSparse (arXiv:2406.05927): after each ReLU, deviations from the per-channel
calibration mean smaller than alpha*sigma are zeroed:
    y = mu + (x - mu) * 1[|x - mu| > alpha * sigma]
mu/sigma are calibrated on training data with the trained checkpoint frozen;
alpha=0 is the identity, so plain checkpoints reproduce exactly at alpha=0.
Parameter attribute names match models/resnet.py -> load plain RN18 state dicts
with strict=False (MeanSparse adds only buffers).
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanSparse(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.register_buffer('mean', torch.zeros(channels))
        self.register_buffer('var', torch.ones(channels))
        self.register_buffer('count', torch.zeros(1))
        self.alpha = 0.0
        self.calibrating = False

    @torch.no_grad()
    def _update_stats(self, x):
        # x: [B, C, H, W]; accumulate exact running mean/var over all pixels
        b_mean = x.mean(dim=(0, 2, 3))
        b_var = x.var(dim=(0, 2, 3), unbiased=False)
        n = x.numel() / x.size(1)
        tot = self.count + n
        delta = b_mean - self.mean
        self.mean += delta * (n / tot)
        self.var = (self.var * self.count + b_var * n
                    + delta.pow(2) * (self.count * n / tot)) / tot
        self.count.copy_(tot)

    def forward(self, x):
        if self.calibrating:
            self._update_stats(x)
            return x
        if self.alpha <= 0:
            return x
        mu = self.mean.view(1, -1, 1, 1)
        thr = self.alpha * self.var.sqrt().view(1, -1, 1, 1)
        dev = x - mu
        return mu + dev * (dev.abs() > thr)


class BasicBlockMS(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.ms1 = MeanSparse(planes)
        self.ms2 = MeanSparse(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.ms1(F.relu(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.ms2(F.relu(out))
        return out


class ResNetMS(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.ms0 = MeanSparse(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def set_alpha(self, alpha):
        for m in self.modules():
            if isinstance(m, MeanSparse):
                m.alpha = alpha

    def set_calibrating(self, flag):
        for m in self.modules():
            if isinstance(m, MeanSparse):
                m.calibrating = flag

    def forward(self, x):
        out = self.ms0(F.relu(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18MS(num_classes=10):
    return ResNetMS(BasicBlockMS, [2, 2, 2, 2], num_classes)
