import os.path as osp
from torch.utils.data import Dataset
from PIL import Image
import torch
import os
import numpy as np

txt_name = "pseudo_label_prid.txt"
dataset_path = "../../datasets/{}/prid_2011/multi_shot/"
gait_dataset_dir = dataset_path.format('prid2011_gait2_cropped')
rgb_dataset_dir = dataset_path.format("prid2011")


class COM_PRID2011(Dataset):

    def __init__(self, transform=None, seq_num=15):
        self.tracklets = self.load(txt_name)
        self.transform = transform
        self.seq_num = seq_num

    def __getitem__(self, index):
        rgb_cam_a_imgs = np.random.choice(self.tracklets['rgb']['cam_a'][index], size=self.seq_num, replace=True)
        rgb_cam_b_imgs = np.random.choice(self.tracklets['rgb']['cam_b'][index], size=self.seq_num, replace=True)
        gait_cam_a_imgs = np.random.choice(self.tracklets['gait']['cam_a'][index], size=self.seq_num, replace=True)
        gait_cam_b_imgs = np.random.choice(self.tracklets['gait']['cam_b'][index], size=self.seq_num, replace=True)

        rgb_cam_a_imgs_data=torch.tensor([])
        for path in rgb_cam_a_imgs:
            img_data = Image.open(path)
            img = torch.unsqueeze(self.transform(img_data), 0)
            rgb_cam_a_imgs_data = torch.cat((rgb_cam_a_imgs_data, img), 0)
        for





        for path in sorted(person_tracklet)[:64]:
            img_data = Image.open(path)
            img = torch.unsqueeze(self.transform(img_data), 0)
            imgs = torch.cat((imgs, img), 0)

        np.random.choice(self.tracklets['gait']['cam_a'][index], size=self.seq_num, replace=True)
        np.random.choice(self.tracklets['rgb']['cam_b'])

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
