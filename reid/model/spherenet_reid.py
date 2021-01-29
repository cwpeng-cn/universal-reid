from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
import torch
import torchvision

__all__ = ['SphereNet', 'spherenet18', 'spherenet34', 'spherenet50', 'spherenet101',
           'spherenet152']


class SphereNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True,
                 num_features=1024, dropout=0.5, num_classes=751, scale=14):
        super(SphereNet, self).__init__()
        resnet = SphereNet.__factory[depth](pretrained=pretrained)
        self.scale = scale

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.bn2 = nn.BatchNorm1d(2048)
        self.dp = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features=2048, out_features=num_features, bias=True)
        self.bn3 = nn.BatchNorm1d(num_features)

        self.W = torch.nn.Parameter(torch.randn(num_features, num_classes),
                                    requires_grad=True)

        nn.init.kaiming_normal_(self.fc.weight, a=1)
        nn.init.constant_(self.fc.bias, 0)

        nn.init.kaiming_normal_(self.W, a=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, x.size()[2:]).view(x.size()[:2])

        x = self.bn2(x)
        x = self.dp(x)
        x = self.fc(x)
        emb = self.bn3(x)

        if self.training:
            emb_norm = torch.norm(emb, 2, 1, True).clamp(min=1e-12).expand_as(emb)
            emb_norm = emb / emb_norm
            w_norm = torch.norm(self.W, 2, 0, True).clamp(min=1e-12).expand_as(self.W)
            w_norm = self.W / w_norm
            cos_th = torch.mm(emb_norm, w_norm)
            s_cos_th = self.scale * cos_th
            return s_cos_th
        else:
            return emb


def spherenet18(**kwargs):
    return SphereNet(18, **kwargs)


def spherenet34(**kwargs):
    return SphereNet(34, **kwargs)


def spherenet50(**kwargs):
    return SphereNet(50, **kwargs)


def spherenet101(**kwargs):
    return SphereNet(101, **kwargs)


def spherenet152(**kwargs):
    return SphereNet(152, **kwargs)
