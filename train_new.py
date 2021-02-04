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
from rgnet_new import RGNet
from torch import optim

EPOCH = 30
SEQ_NUM = 10
NUM_CLASS = 144
LR = 0.001

train_dataset = COM_PRID2011(seq_num=SEQ_NUM, mode="train")
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

query_dataset = COM_PRID2011(mode="query")
query_loader = DataLoader(query_dataset, batch_size=1, shuffle=False)
gallery_dataset = COM_PRID2011(mode="gallery")
gallery_loader = DataLoader(gallery_dataset, batch_size=1, shuffle=False)

net = RGNet(num_class=NUM_CLASS, seq_num=SEQ_NUM).cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    [{"params": list(net.gait_net.parameters()) + list(net.classifier.parameters()) + list(net.fc.parameters())}],
    lr=LR)
optimizer2 = optim.Adam(
    [{"params": list(net.gait_net.parameters()) + list(net.classifier.parameters()) + list(net.fc.parameters())},
     {"params": net.rgb_net.parameters(), "lr": LR / 10}], lr=LR)

print("测试直接迁移结果")
query_feature, query_id, query_camera = FO.extract_cnn_feature_combined(net, loader=query_loader, vis=True,
                                                                        mode="query")
gallery_feature, gallery_id, gallery_camera = FO.extract_cnn_feature_combined(net, loader=gallery_loader, vis=True,
                                                                              mode="gallery")
map, cmc = market_evaluate.evaluate(query_feature, np.array(query_id), np.array(query_camera), gallery_feature,
                                    np.array(gallery_id), np.array(gallery_camera), vis=False)
print("直接迁移结果: rank-1:{},rank-5:{},rank-10:{},rank-20:{}".format(cmc[0], cmc[4], cmc[9], cmc[19]))
for epoch in range(EPOCH):
    if epoch == 10:
        optimizer = optimizer2
    for step, data in enumerate(train_loader):
        rgb_seqs, gait_seqs, ids = data
        output = net(rgb_seqs.cuda(), gait_seqs.cuda())
        loss = criterion(output, ids.cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 10 == 0:
            print(loss.item())
    if epoch > 8 or epoch % 2 == 0:
        print("第{}轮效果评估开始>>>".format(epoch + 1))
        query_feature, query_id, query_camera = FO.extract_cnn_feature_combined(net, loader=query_loader, vis=False,
                                                                                mode="query")
        gallery_feature, gallery_id, gallery_camera = FO.extract_cnn_feature_combined(net, loader=gallery_loader,
                                                                                      vis=False, mode="gallery")
        map, cmc = market_evaluate.evaluate(query_feature, np.array(query_id), np.array(query_camera), gallery_feature,
                                            np.array(gallery_id), np.array(gallery_camera), vis=False)
        print("第{}轮训练结果: rank-1:{},rank-5:{},rank-10:{},rank-20:{}".format(epoch + 1, cmc[0], cmc[4], cmc[9],
                                                                           cmc[19]))
