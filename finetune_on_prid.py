import numpy as np
from reid.model.pgs import PGSNet
from reid.data import transforms
from reid.data.samplers import RandomIdentitySampler
from torch.utils.data import DataLoader
from reid.utils.model_save_restore import *
from reid.evaluation import market_evaluate
from torch.utils.data import Dataset
import csv
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from reid.data.prid2011 import PRID2011
import reid.feature_op_video as FO


class NewDataset(Dataset):
    def __init__(self, image_path, transform, use_onehot=False, categories_num=0):
        """
        :param image_path: 训练图片路径
        :param transform: 转换
        :param use_onehot: 是否使用onehot,默认不使用
        :param categories_num: 类别数
        """
        self.image_path = image_path
        self.transform = transform
        self.use_onehot = use_onehot
        self.categories_num = categories_num

        self.ret = []
        self.preprocess()

    def preprocess(self, relabel=True):
        reader = csv.reader(open(self.image_path))
        for pid, fpaths in enumerate(reader):
            for fpath in fpaths:
                self.ret.append((fpath, pid))

    def __getitem__(self, index):
        image = Image.open(self.ret[index][0])
        id_ = self.ret[index][1]
        return self.transform(image), id_, 0

    def __len__(self):
        return len(self.ret)


class MNet(nn.Module):

    def __init__(self, net, depth=50, pretrained=True, cut_at_pooling=False,
                 num_features=512, dropout=0.5, num_classes=0):

        super(MNet, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling

        # Construct base (pretrained) resnet
        base = net

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.layer4_stn = base.layer4_stn

        self.stn = base.stn

        self.mmaxpool = base.mmaxpool

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes
            out_planes = 2048

            # Append new layers
            if self.has_embedding:
                self.embed_layer = base.embed_layer
                self.embed_layer_stn = base.embed_layer_stn
            else:
                # Change the num_features to CNN output channels
                self.num_features = 2048
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


# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
from bisect import bisect_right
import torch


# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            milestones,
            gamma=0.1,
            warmup_factor=1.0 / 3,
            warmup_iters=500,
            warmup_method="linear",
            last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


def make_optimizer(model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = 3e-4
        weight_decay = 0.0005
        if "bias" in key:
            lr = 3e-4 * 2
            weight_decay = 0
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    optimizer = torch.optim.Adam(params)
    return optimizer


# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
from torch import nn


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class TripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=None):
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(
            dist_mat, labels)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an


def loss(feat, score, feat_stn, score_stn, target):
    return F.cross_entropy(score, target) + F.cross_entropy(score_stn, target) + triplet(feat, target)[0] + \
           triplet(feat_stn, target)[0]


excel_name = "pseudo_label_prid.csv"
num_classes = 144
save_path = './storage'

train_transform = transforms.Compose([
    transforms.Resize((384, 128), interpolation=3),
    transforms.Pad(10),
    transforms.RandomCrop((384, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
new_dataset = NewDataset(image_path=excel_name, transform=train_transform, use_onehot=False)
sampler = RandomIdentitySampler(new_dataset, num_instances=4)
train_loader = DataLoader(dataset=new_dataset, sampler=sampler, batch_size=32, num_workers=4)

test_transform = transforms.Compose([
    transforms.Resize((384, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
query = PRID2011(test_transform, root='../../datasets/', split_id=0, mode="query")
query_loader = DataLoader(query, batch_size=1, shuffle=False)
gallery = PRID2011(test_transform, root='../../datasets/', split_id=0, mode="gallery")
gallery_loader = DataLoader(gallery, batch_size=1, shuffle=False)

net = PGSNet(num_classes=4101, num_features=1024)
net = restore_network("./storage/", 149, net).cuda()
net = MNet(net, num_classes=num_classes, num_features=1024).cuda()

triplet = TripletLoss(0.3)

optimizer = make_optimizer(net)
# scheduler = WarmupMultiStepLR(optimizer, (30, 55), 0.1, 1.0 / 3,500, "linear")
scheduler = WarmupMultiStepLR(optimizer, (18, 30), 0.1, 1.0 / 3, 500, "linear")
net.train()

step = 0
best_map = -1
best_map_epoch = 0
best_rank1 = -1
best_rank1_epoch = 0
print("测试直接迁移结果")
query_feature, query_id, query_camera = FO.extract_cnn_feature(net, loader=query_loader, vis=False)
gallery_feature, gallery_id, gallery_camera = FO.extract_cnn_feature(net, loader=gallery_loader, vis=False)
map, cmc = market_evaluate.evaluate(query_feature, np.array(query_id), np.array(query_camera), gallery_feature,
                                    np.array(gallery_id), np.array(gallery_camera), vis=False)
print("直接迁移结果: rank-1:{},rank-5:{},rank-10:{},rank-20:{}".format(cmc[0], cmc[4], cmc[9], cmc[19]))
print("开始训练>>>")
for epoch in range(40):
    scheduler.step()
    for images, ids, cams in train_loader:
        feat, predict, feat_stn, predict_stn = net(images.cuda())
        loss_value = loss(feat, predict, feat_stn, predict_stn, ids.cuda())
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
        if step % 10 == 0:
            print(step, loss_value.item())
        step += 1
    if (epoch + 1) > 0 and (epoch + 1) % 1 == 0:
        save_network(save_path, net, epoch)
        print("第{}轮效果评估开始>>>".format(epoch + 1))
        query_feature, query_id, query_camera = FO.extract_cnn_feature(net, loader=query_loader, vis=False)
        gallery_feature, gallery_id, gallery_camera = FO.extract_cnn_feature(net, loader=gallery_loader, vis=False)
        map, cmc = market_evaluate.evaluate(query_feature, np.array(query_id), np.array(query_camera), gallery_feature,
                                            np.array(gallery_id), np.array(gallery_camera), vis=False)
        print("第{}轮训练结果: rank-1:{},rank-5:{},rank-10:{},rank-20:{}".format(epoch + 1, cmc[0], cmc[4], cmc[9],
                                                                           cmc[19]))
        if map > best_map:
            best_map = map
            best_map_epoch = epoch
        if cmc[0] > best_rank1:
            best_rank1 = cmc[0]
            best_cmc_epoch = epoch

    print("已经训练了{}个epoch".format(epoch + 1))
print("最佳map:{},最佳rank-1{},最佳map训练轮数:{},最佳cmc训练轮数:{}".format(best_map, best_rank1, best_map_epoch, best_rank1_epoch))
