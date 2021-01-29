from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth=50, pretrained=True, cut_at_pooling=False,
                 num_features=512, dropout=0.5, num_classes=0):
        """
        两种常规用法：
        1. ResNet网络选择avgpool层前面部分，后面加上dropout+输出全连接层
          ResNet(depth,dropout=dropout,num_classes=num_classes)
        2. ZheDong在github上放出来的baseline，ResNet网络选择avgpool层前面部分，
        后面接着全连接隐藏层(512)+BatchNorm+LeakeyReLU(0.1)+Dropout(0.5)+输出全连接层
          ResNet(depth=50,num_features=512,dropout=0.5,num_classes=num_classes)
        :param depth: 选用的ResNet模型（18，34，50，101，152）
        :param pretrained: ResNet是否与训练
        :param cut_at_pooling: 是否在ResNet网络的avgpooling层处截断,默认不截断
        :param num_features: 加入的全连接隐藏层特征数目，默认为0
        :param dropout:dropout参数，默认为0
        :param num_classes:训练数据集类别数
        """
        super(ResNet, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling

        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        base = ResNet.__factory[depth](pretrained=pretrained)

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

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
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.last_fc = nn.Linear(self.num_features, self.num_classes)
                init.normal_(self.last_fc.weight, std=0.001)
                init.constant_(self.last_fc.bias, 0.0)

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
        x = self.layer4(x)

        if self.cut_at_pooling:
            return x

        x = F.avg_pool2d(x, x.size()[2:]).view(x.size()[:2])
        # 如果是测试，则取这一结果作为特征
        if not self.training:
            return x
        if self.has_embedding:
            x = self.embed_layer(x)
        if self.dropout > 0:
            x = self.drop(x)
        if self.num_classes > 0:
            x = self.last_fc(x)
        return x

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


def resnet18(**kwargs):
    return ResNet(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


def resnet50(**kwargs):
    return ResNet(50, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)
