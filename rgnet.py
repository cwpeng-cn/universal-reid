from reid.model.pgs import PGSNet
from reid.utils.model_save_restore import *
from torch import nn
from gait.model.resnet_video import resnet50 as gaitnet


class RGNet(nn.Module):
    def __init__(self, num_class, seq_num):
        super(RGNet, self).__init__()
        self.seq_num = seq_num
        rgb_net = PGSNet(num_classes=num_class, num_features=1024)
        self.rgb_net = restore_network("./", 9, rgb_net).eval()
        self.gait_net = gaitnet(num_classes=num_class, pretrained=True, seq_num=seq_num, droprate=0.1)
        self.fc = self._construct_fc_layer([1024], 4096, 0.1)
        self.classifier = nn.Linear(1024, num_class)
        self._init_params(self.fc)
        self._init_params(self.classifier)

    def forward(self, rgb_seqs, gait_seqs):
        n, s, c, h1, w1 = rgb_seqs.shape
        rgb_seqs = rgb_seqs.view(n * s, c, h1, w1)
        n, s, c, h2, w2 = gait_seqs.shape
        gait_seqs = gait_seqs.view(n * s, c, h2, w2)
        with torch.no_grad():
            self.rgb_net.eval()
            rgb_features = torch.mean(self.rgb_net(rgb_seqs).view(n, s, -1), 1)
        gait_features = self.gait_net(gait_seqs)
        if not self.training:
            return rgb_features
        combined = torch.cat([self.normalize(rgb_features), self.normalize(gait_features)], 1)
        features = self.fc(combined)
        if not self.training:
            return features
        else:
            return self.classifier(features)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        """Constructs fully connected layer

        Args:
            fc_dims (list or tuple): dimensions of fc layers, if None, no fc layers are constructed
            input_dim (int): input dimension
            dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None

        assert isinstance(
            fc_dims, (list, tuple)
        ), 'fc_dims must be either list or tuple, but got {}'.format(
            type(fc_dims)
        )

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)

    def _init_params(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu'
            )
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def normalize(self, x, axis=-1):
        """Normalizing to unit length along the specified dimension.
        Args:
          x: pytorch Variable
        Returns:
          x: pytorch Variable, same shape as input
        """
        x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
        return x
