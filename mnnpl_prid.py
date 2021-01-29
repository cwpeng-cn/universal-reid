import numpy as np
from reid.model.pgs import PGSNet
from reid.data.unsup_prid2011 import UNSUPPRID2011
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

cam1 = UNSUPPRID2011(transform, root='../../datasets/', split_id=0, mode="cam1")
cam1_loader = DataLoader(cam1, batch_size=1, shuffle=False)
cam2 = UNSUPPRID2011(transform, root='../../datasets/', split_id=0, mode="cam2")
cam2_loader = DataLoader(cam2, batch_size=1, shuffle=False)
cam1_feature, cam1_id, cam1_camera = FO.extract_cnn_feature(net, loader=cam1_loader, vis=True)
cam2_feature, cam2_id, cam2_camera = FO.extract_cnn_feature(net, loader=cam2_loader, vis=True)

score1 = torch.matmul(cam1_feature, cam2_feature.T)
score2 = torch.matmul(cam2_feature, cam1_feature.T)
cam1_persons = cam1.data['cam1_persons']
cam2_persons = cam2.data['cam2_persons']

with open("pesudo_label_prid.txt", "w") as fw:
    for i in range(178):
        index = torch.argmax(score1[i])
        index2 = torch.argmax(score2[index])
        if i == index2:
            fw.write(cam1_persons[i] + "\t" + cam2_persons[index.item()])
