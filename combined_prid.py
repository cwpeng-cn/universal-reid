from torch.utils.data import Dataset
from PIL import Image
import torch
import os
import numpy as np
from torchvision import transforms
import json

txt_path = "pseudo_label_prid.txt"
dataset_path = "../../datasets/{}/prid_2011/multi_shot/"
json_path = "../../datasets/prid2011/splits_prid2011.json"
gait_dataset_dir = dataset_path.format('prid2011_gait2_cropped')
rgb_dataset_dir = dataset_path.format("prid2011")


class COM_PRID2011(Dataset):

    def __init__(self, seq_num=15, mode="train"):
        self.train_tracklets = self.load_trainset(txt_path)
        self.query_tracklets, self.gallery_tracklets = self.load_testset(json_path)
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
        self.mode = mode

    def __getitem__(self, index):
        if self.mode == "train":
            cam = "cam_a" if np.random.rand() >= 0.5 else 'cam_b'

            rgb_img_paths = np.random.choice(self.train_tracklets['rgb'][cam][index], size=self.seq_num, replace=True)
            gait_img_paths = np.random.choice(self.train_tracklets['gait'][cam][index], size=self.seq_num, replace=True)

            rgb_imgs_data = torch.tensor([])
            gait_imgs_data = torch.tensor([])

            for _ in rgb_img_paths:
                img_data = Image.open(_).convert('RGB')
                img = torch.unsqueeze(self.rgb_transform(img_data), 0)
                rgb_imgs_data = torch.cat((rgb_imgs_data, img), 0)

            for _ in gait_img_paths:
                img_data = Image.open(_).convert('RGB')
                img = torch.unsqueeze(self.gait_transform(img_data), 0)
                gait_imgs_data = torch.cat((gait_imgs_data, img), 0)

            return rgb_imgs_data, gait_imgs_data, index

        if self.mode == "query":
            rgb_imgs_data = torch.tensor([])
            gait_imgs_data = torch.tensor([])

            for _ in sorted(self.query_tracklets[index]["gait"])[:64]:
                img_data = Image.open(_).convert('RGB')
                img = torch.unsqueeze(self.gait_transform(img_data), 0)
                gait_imgs_data = torch.cat((gait_imgs_data, img), 0)

            for _ in sorted(self.query_tracklets[index]['rgb'])[:64]:
                img_data = Image.open(_).convert('RGB')
                img = torch.unsqueeze(self.rgb_transform(img_data), 0)
                rgb_imgs_data = torch.cat((rgb_imgs_data, img), 0)

            return rgb_imgs_data, gait_imgs_data, index

        if self.mode == "gallery":
            rgb_imgs_data = torch.tensor([])
            gait_imgs_data = torch.tensor([])

            for _ in sorted(self.gallery_tracklets[index]["gait"])[:64]:
                img_data = Image.open(_).convert('RGB')
                img = torch.unsqueeze(self.gait_transform(img_data), 0)
                gait_imgs_data = torch.cat((gait_imgs_data, img), 0)

            for _ in sorted(self.gallery_tracklets[index]['rgb'])[:64]:
                img_data = Image.open(_).convert('RGB')
                img = torch.unsqueeze(self.rgb_transform(img_data), 0)
                rgb_imgs_data = torch.cat((rgb_imgs_data, img), 0)

            return rgb_imgs_data, gait_imgs_data, index

    def __len__(self):
        if self.mode == "train":
            return len(self.train_tracklets["rgb"]["cam_a"])
        if self.mode == "query":
            return len(self.query_tracklets)
        if self.mode == "gallery":
            return len(self.gallery_tracklets)

    def load_trainset(self, txt_name):
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

    def load_testset(self, json_path):
        with open(json_path, "r") as f:
            persons = json.load(f)[0]
            test_persons = persons['test']
            query_tracklets = []
            gallery_tracklets = []
            for person in test_persons:
                query_rgb_tracklet, query_gait_tracklet, gallery_rgb_tracklet, gallery_gait_tracklet = [], [], [], []

                for img_name in os.listdir(gait_dataset_dir + "cam_a/" + person):
                    if img_name.endswith("png"):
                        query_rgb_tracklet.append(rgb_dataset_dir + "cam_a/" + person + "/" + img_name)
                        query_gait_tracklet.append(gait_dataset_dir + "cam_a/" + person + "/" + img_name)

                for img_name in os.listdir(gait_dataset_dir + "cam_b/" + person):
                    if img_name.endswith("png"):
                        gallery_rgb_tracklet.append(rgb_dataset_dir + "cam_b/" + person + "/" + img_name)
                        gallery_gait_tracklet.append(gait_dataset_dir + "cam_b/" + person + "/" + img_name)
                query_tracklets.append({"gait": query_gait_tracklet, "rgb": query_rgb_tracklet})
                gallery_tracklets.append({"gait": gallery_gait_tracklet, "rgb": gallery_rgb_tracklet})
            return query_tracklets, gallery_tracklets
