from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
import os.path as osp
from glob import glob
import re
from reid.data.samplers import BalancedSampler
from .bases import BaseImageDataset


class MarketDataset(data.Dataset):
    def __init__(self, image_path, transform, use_onehot=False, categories_num=0):
        """
        :param image_path: 训练图片路径
        :param transform: 转换
        :param use_onehot: 是否使用onehot,默认不使用
        :param categories_num: 类别数
        """
        self.image_path = image_path
        self.transform = transform
        self.use_onehot = use_onehot
        self.categories_num = categories_num

        # 原始的id信息(图片编号)
        self.original_id = []
        # 原始的相机编号
        self.cameras = []
        # 原始id信息与对应索引{id_0:[index_1,index_2,...],...}
        self.person_indexs_dict = {}
        self.ret = []
        self.preprocess()

    def preprocess(self, relabel=True):
        pattern = re.compile(r'([-\d]+)_c(\d)')
        all_pids = {}
        fpaths = sorted(glob(osp.join(self.image_path, '*.jpg')))
        for index, fpath in enumerate(fpaths):
            fname = osp.basename(fpath)
            pid, cam = map(int, pattern.search(fname).groups())
            self.original_id.append(pid)
            self.cameras.append(cam)
            if pid in self.person_indexs_dict.keys():
                self.person_indexs_dict[pid].append(index)
            else:
                self.person_indexs_dict[pid] = [index]
            if relabel:
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
            else:
                if pid not in all_pids:
                    all_pids[pid] = pid
            pid = all_pids[pid]
            cam -= 1
            if self.use_onehot:
                pid = torch.zeros(self.categories_num).scatter_(0, torch.LongTensor([pid]), 1)
            self.ret.append((fpath, pid, cam))

    def __getitem__(self, index):
        image = Image.open(self.ret[index][0])
        id_ = self.ret[index][1]
        cam = self.ret[index][2]
        return self.transform(image), id_, cam

    def __len__(self):
        return len(self.ret)


class DukeMTMCDataset(data.Dataset):
    """
    Dataset statistics:
    # identities: 1404 (train + query)
    # images:16522 (train) + 2228 (query) + 17661 (gallery)
    # cameras: 8
    """

    def __init__(self, image_path, transform, use_onehot=False, categories_num=0):
        self.image_path = image_path
        self.transform = transform
        self.use_onehot = use_onehot
        self.categories_num = categories_num

        # 原始的id信息(图片编号)
        self.original_id = []
        # 原始的相机编号
        self.cameras = []
        # 原始id信息与对应索引{id_0:[index_1,index_2,...],...}
        self.person_indexs_dict = {}
        self.ret = []
        self.preprocess()

    def preprocess(self, relabel=True):
        pattern = re.compile(r'([-\d]+)_c(\d)')
        all_pids = {}
        fpaths = sorted(glob(osp.join(self.image_path, '*.jpg')))
        for index, fpath in enumerate(fpaths):
            fname = osp.basename(fpath)
            pid, cam = map(int, pattern.search(fname).groups())
            self.original_id.append(pid)
            self.cameras.append(cam)
            if pid in self.person_indexs_dict.keys():
                self.person_indexs_dict[pid].append(index)
            else:
                self.person_indexs_dict[pid] = [index]
            if relabel:
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
            else:
                if pid not in all_pids:
                    all_pids[pid] = pid
            pid = all_pids[pid]
            cam -= 1
            if self.use_onehot:
                pid = torch.zeros(self.categories_num).scatter_(0, torch.LongTensor([pid]), 1)
            self.ret.append((fpath, pid, cam))

    def __getitem__(self, index):
        image = Image.open(self.ret[index][0])
        id_ = self.ret[index][1]
        cam = self.ret[index][2]
        return self.transform(image), id_, cam

    def __len__(self):
        return len(self.ret)

class MSMT17(data.Dataset):
    def __init__(self, path,name,transform):
        """
        root: MSMT folder position
        name: train,query or gallery
        """
        helper=MSMT17Helper(path)
        if name=="train":
            self.ret=helper.train
        elif name=="query":
            self.ret=helper.query
        elif name=="gallery":
            self.ret=helper.gallery
        self.transform = transform
        
        self.original_id = [x[1] for x in self.ret]
        self.cameras = [x[2] for x in self.ret]
            
    def __getitem__(self, index):
        image = Image.open(self.ret[index][0])
        id_ = self.ret[index][1]
        cam = self.ret[index][2]
        return self.transform(image), id_, cam

    def __len__(self):
        return len(self.ret)
     

class MSMT17Helper(BaseImageDataset):
    """
    MSMT17

    Reference:
    Wei et al. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. CVPR 2018.

    URL: http://www.pkuvmc.com/publications/msmt17.html

    Dataset statistics:
    # identities: 4101
    # images: 32621 (train) + 11659 (query) + 82161 (gallery)
    # cameras: 15
    """

    def __init__(self,root='../ReidDatasets', verbose=True, **kwargs):
        super(MSMT17Helper, self).__init__()
        self.dataset_dir = root
        self.train_dir = osp.join(self.dataset_dir, 'MSMT17_V2/mask_train_v2')
        self.test_dir = osp.join(self.dataset_dir, 'MSMT17_V2/mask_test_v2')
        self.list_train_path = osp.join(self.dataset_dir, 'MSMT17_V2/list_train.txt')
        self.list_val_path = osp.join(self.dataset_dir, 'MSMT17_V2/list_val.txt')
        self.list_query_path = osp.join(self.dataset_dir, 'MSMT17_V2/list_query.txt')
        self.list_gallery_path = osp.join(self.dataset_dir, 'MSMT17_V2/list_gallery.txt')

        self._check_before_run()
        train = self._process_dir(self.train_dir, self.list_train_path)
        #val, num_val_pids, num_val_imgs = self._process_dir(self.train_dir, self.list_val_path)
        query = self._process_dir(self.test_dir, self.list_query_path)
        gallery = self._process_dir(self.test_dir, self.list_gallery_path)
        if verbose:
            print("=> MSMT17 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))

    def _process_dir(self, dir_path, list_path):
        with open(list_path, 'r') as txt:
            lines = txt.readlines()
        dataset = []
        pid_container = set()
        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.split(' ')
            pid = int(pid)  # no need to relabel
            camid = int(img_path.split('_')[2])
            img_path = osp.join(dir_path, img_path)
            dataset.append((img_path, pid, camid))
            pid_container.add(pid)

        # check if pid starts from 0 and increments with 1
        for idx, pid in enumerate(pid_container):
            assert idx == pid, "See code comment for explanation"
        return dataset