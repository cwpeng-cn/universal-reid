from combined_prid import COM_PRID2011
import numpy as np
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
from rgnet import RGNet
from torch import optim

EPOCH = 30
SEQ_NUM = 10
NUM_CLASS = 144

train_dataset = COM_PRID2011(seq_num=SEQ_NUM, mode="train")
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)

query_dataset = COM_PRID2011(mode="query")
query_loader = DataLoader(query_dataset, batch_size=1, shuffle=False)
gallery_dataset = COM_PRID2011(mode="gallery")
gallery_loader = DataLoader(gallery_dataset, batch_size=1, shuffle=False)

net = RGNet(num_class=NUM_CLASS, seq_num=SEQ_NUM).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(net.gait_net.parameters()) + list(net.classifier.parameters()) + list(net.fc.parameters()))

for epoch in range(EPOCH):
    for data in train_loader:
        rgb_seqs, gait_seqs, ids = data
        output = net(rgb_seqs.cuda(), gait_seqs.cuda())
        loss = criterion(output, ids.cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())
