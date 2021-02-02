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

EPOCH = 30
SEQ_NUM = 10

train_dataset = COM_PRID2011(seq_num=SEQ_NUM)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)

net = PGSNet(num_classes=144, num_features=1024)
net = restore_network("./", 9, net).cuda().eval()

for epoch in range(EPOCH):
    for data in train_loader:
        rgb_seqs, gait_seqs, ids = data
        n, s, c, h1, w1 = rgb_seqs.shape
        rgb_seqs = rgb_seqs.view(n * s, c, h1, w1).cuda()
        n, s, c, h2, w2 = gait_seqs.shape
        gait_seqs = gait_seqs.view(n * s, c, h2, w2).cuda()

        rgb_features = torch.mean(net(rgb_seqs).view(n, s, -1), 1)
        print(rgb_features.shape)
