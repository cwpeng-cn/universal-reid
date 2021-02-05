from combined_prid import COM_PRID2011
import numpy as np
from torch.utils.data import DataLoader
from reid.utils.model_save_restore import *
from reid.evaluation import market_evaluate
import reid.feature_op_video as FO
from rgnet_new import RGNet
from reid.utils.seed import set_seed

EPOCH = 30
SEQ_NUM = 10
NUM_CLASS = 144
LR = 0.001
save_path = './storage'

set_seed()

train_dataset = COM_PRID2011(seq_num=SEQ_NUM, mode="train")
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

query_dataset = COM_PRID2011(mode="query")
query_loader = DataLoader(query_dataset, batch_size=1, shuffle=False)
gallery_dataset = COM_PRID2011(mode="gallery")
gallery_loader = DataLoader(gallery_dataset, batch_size=1, shuffle=False)

net = RGNet(num_class=NUM_CLASS, seq_num=SEQ_NUM)
net = restore_network(save_path, epoch=15, network=net).eval().cuda()

print("测试迁移结果")
query_feature, query_id, query_camera = FO.extract_cnn_feature_combined(net, loader=query_loader, vis=True,
                                                                        mode="query")
gallery_feature, gallery_id, gallery_camera = FO.extract_cnn_feature_combined(net, loader=gallery_loader, vis=True,
                                                                              mode="gallery")
map, cmc = market_evaluate.evaluate(query_feature, np.array(query_id), np.array(query_camera), gallery_feature,
                                    np.array(gallery_id), np.array(gallery_camera), vis=False)
print("迁移结果: rank-1:{},rank-5:{},rank-10:{},rank-20:{}".format(cmc[0], cmc[4], cmc[9], cmc[19]))
