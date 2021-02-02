import os.path as osp
from torch.utils.data import Dataset
from PIL import Image
import torch
import os
import numpy as np
from torchvision import transforms

txt_name = "pseudo_label_prid.txt"
dataset_path = "../../datasets/{}/prid_2011/multi_shot/"
gait_dataset_dir = dataset_path.format('prid2011_gait2_cropped')
rgb_dataset_dir = dataset_path.format("prid2011")


class COM_PRID2011(Dataset):

    def __init__(self, seq_num=15):
        self.tracklets = self.load(txt_name)
        self.rgb_transform = transforms.Compose([
            transforms.Resize((384, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.gait_transform = transforms.Compose([
            transforms.Resize((192, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.seq_num = seq_num

    def __getitem__(self, index):
        cam = "cam_a" if np.random.rand() >= 0.5 else 'cam_b'

        rgb_img_paths = np.random.choice(self.tracklets['rgb'][cam][index], size=self.seq_num, replace=True)
        gait_img_paths = np.random.choice(self.tracklets['gait'][cam][index], size=self.seq_num, replace=True)

        rgb_imgs_data = torch.tensor([])
        gait_imgs_data = torch.tensor([])

        for _ in rgb_img_paths:
            img_data = Image.open(_)
            img = torch.unsqueeze(self.rgb_transform(img_data), 0)
            rgb_imgs_data = torch.cat((rgb_imgs_data, img), 0)

        for _ in gait_img_paths:
            img_data = Image.open(_)
            print(img_data.shape)
            img = torch.unsqueeze(self.gait_transform(img_data), 0)
            gait_imgs_data = torch.cat((gait_imgs_data, img), 0)

        return rgb_imgs_data, gait_imgs_data, index

    def __len__(self):
        return len(self.tracklets["rgb"]["cam_a"])

    def load(self, txt_name):
        tracklets = {"rgb": {'cam_a': [], 'cam_b': []}, "gait": {'cam_a': [], 'cam_b': []}}
        with open(txt_name, 'r') as f:
            lines = f.readlines()
            for pid, line in enumerate(lines):
                line = line.split('\n')[0]
                cam_a_person = line.split("\t")[0]
                cam_b_person = line.split("\t")[1]
                person = {"cam_a": cam_a_person, "cam_b": cam_b_person}

                if not (len(os.listdir(gait_dataset_dir + cam_a_person)) and len(
                        os.listdir(gait_dataset_dir + cam_b_person))):
                    continue

                for cam in ["cam_a", "cam_b"]:
                    rgb_tracklet, gait_tracklet = [], []
                    for img_name in os.listdir(gait_dataset_dir + person[cam]):
                        if img_name.endswith(".png"):
                            rgb_tracklet.append(rgb_dataset_dir + person[cam] + "/" + img_name)
                            gait_tracklet.append(gait_dataset_dir + person[cam] + "/" + img_name)
                    tracklets['rgb'][cam].append(rgb_tracklet)
                    tracklets['gait'][cam].append(gait_tracklet)
        return tracklets
