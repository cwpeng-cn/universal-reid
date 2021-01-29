import numpy as np
from reid.model.pgs import PGSNet
from reid.data.prid2011 import PRID2011
from torchvision import transforms
from torch.utils.data import DataLoader
from reid.utils.model_save_restore import *
from reid.evaluation import market_evaluate
import reid.feature_op_video as FO

transform = transforms.Compose([
    transforms.Resize((384, 128), interpolation=3),
    # transforms.Pad(10),
    # transforms.RandomCrop((384, 128)),
    transforms.RandomHorizontalFlip(),
    # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

net = PGSNet(num_classes=4101, num_features=1024)
net = restore_network("./storage/", 149, net).cuda()
query = PRID2011(transform, root='../../datasets/', split_id=0, mode="query")
query_loader = DataLoader(query, batch_size=1, shuffle=False)
gallery = PRID2011(transform, root='../../datasets/', split_id=0, mode="gallery")
gallery_loader = DataLoader(gallery, batch_size=1, shuffle=False)

query_feature, query_id, query_camera = FO.extract_cnn_feature(net, loader=query_loader, vis=True)
gallery_feature, gallery_id, gallery_camera = FO.extract_cnn_feature(net, loader=gallery_loader, vis=True)

map, cmc = market_evaluate.evaluate(query_feature, np.array(query_id), np.array(query_camera), gallery_feature,
                                    np.array(gallery_id), np.array(gallery_camera), vis=False)
print("训练结果: map:{},rank-1:{},rank-5:{},rank-10:{}".format(map, cmc[0], cmc[4], cmc[9]))

##训练结果: map:0.8773935619013782,rank-1:0.8539325842696629,rank-5:0.9550561797752809,rank-10:0.9662921348314607
