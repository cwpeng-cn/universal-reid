from combined_prid import COM_PRID2011
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
from gait.model.resnet_video import resnet50 as gaitnet

EPOCH = 30
SEQ_NUM = 10
NUM_CLASS = 144

train_dataset = COM_PRID2011(seq_num=SEQ_NUM)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)

net = PGSNet(num_classes=NUM_CLASS, num_features=1024)
rgb_net = restore_network("./", 9, net).cuda().eval()
gait_net = gaitnet(num_classes=NUM_CLASS, pretrained=True, seq_num=SEQ_NUM, droprate=0.1)

for epoch in range(EPOCH):
    for data in train_loader:
        rgb_seqs, gait_seqs, ids = data
        n, s, c, h1, w1 = rgb_seqs.shape
        rgb_seqs = rgb_seqs.view(n * s, c, h1, w1).cuda()
        n, s, c, h2, w2 = gait_seqs.shape
        gait_seqs = gait_seqs.view(n * s, c, h2, w2).cuda()

        with torch.no_grad():
            rgb_features = torch.mean(rgb_net(rgb_seqs).view(n, s, -1), 1)

        gait_features = gait_net(rgb_seqs)
        print(gait_features.shape, rgb_features.shape)
