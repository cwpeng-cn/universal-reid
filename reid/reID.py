import torch
from reid.model.resnet_ibn_b import *

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision


class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()

        self.localization = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=3),
            nn.BatchNorm2d(2048),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(True),
            nn.Linear(512, 2 * 3),
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([0.8, 0, 0, 0, 0.8, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def forward(self, x):
        xs = self.localization(x)
        xs = F.adaptive_avg_pool2d(xs, (1, 1))
        xs = xs.view(xs.size(0), -1)  # N,4096
        theta = self.fc_loc(xs)  # N,6
        theta = theta.view(-1, 2, 3)  # N,2,3

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x


class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth=50, pretrained=True, cut_at_pooling=False,
                 num_features=1024, dropout=0.5, num_classes=0):

        super(ResNet, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling

        # Construct base (pretrained) resnet
        base = resnet50_ibn_b(pretrained=pretrained)
        base_stn = resnet50_ibn_b(pretrained=pretrained)
        self.stn = STN()

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.layer4_stn = base_stn.layer4

        for mo in self.layer4[0].modules():
            if isinstance(mo, nn.Conv2d):
                mo.stride = (1, 1)
        for mo in self.layer4_stn[0].modules():
            if isinstance(mo, nn.Conv2d):
                mo.stride = (1, 1)

        self.mmaxpool = nn.AdaptiveMaxPool2d((1, 1))

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes
            out_planes = base.fc.in_features

            # Append new layers
            if self.has_embedding:
                feat = nn.Linear(out_planes, self.num_features)
                feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(feat.weight, mode='fan_out')
                init.constant_(feat.bias, 0)
                init.normal_(feat_bn.weight, 1, 0.02)
                init.constant_(feat_bn.bias, 0.0)
                embed_layer = [feat, feat_bn]
                self.embed_layer = nn.Sequential(*embed_layer)

                feat = nn.Linear(out_planes, self.num_features)
                feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(feat.weight, mode='fan_out')
                init.constant_(feat.bias, 0)
                init.normal_(feat_bn.weight, 1, 0.02)
                init.constant_(feat_bn.bias, 0.0)
                embed_layer = [feat, feat_bn]
                self.embed_layer_stn = nn.Sequential(*embed_layer)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes

            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)

            if self.num_classes > 0:
                self.last_fc = nn.Linear(self.num_features, self.num_classes)
                init.normal_(self.last_fc.weight, std=0.001)
                init.constant_(self.last_fc.bias, 0.0)

                self.last_fc_stn = nn.Linear(self.num_features, self.num_classes)
                init.normal_(self.last_fc_stn.weight, std=0.001)
                init.constant_(self.last_fc_stn.bias, 0.0)

        if not self.pretrained:
            self.reset_params()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_stn = self.stn(x)
        x = self.layer4(x)
        x_stn = self.layer4_stn(x_stn)

        if self.cut_at_pooling:
            return x

        x = F.max_pool2d(x, x.size()[2:]).view(x.size()[:2])
        x_stn = F.max_pool2d(x_stn, x_stn.size()[2:]).view(x_stn.size()[:2])

        if self.has_embedding:
            triplet_out = self.embed_layer(x)
            triplet_out_stn = self.embed_layer_stn(x_stn)

        if not self.training:
            triplet_out = self.normalize(triplet_out)
            triplet_out_stn = self.normalize(triplet_out_stn)
            return torch.cat((triplet_out, triplet_out_stn), 1)  # N,2C

        if self.num_classes > 0:
            x = self.last_fc(triplet_out)
            x_stn = self.last_fc_stn(triplet_out_stn)
        return triplet_out, x, triplet_out_stn, x_stn

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def normalize(self, x, axis=-1):
        """Normalizing to unit length along the specified dimension.
        Args:
          x: pytorch Variable
        Returns:
          x: pytorch Variable, same shape as input
        """
        x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
        return x
