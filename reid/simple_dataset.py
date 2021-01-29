from torch.utils import data
from PIL import Image
import os.path as osp
from glob import glob


class SimpleDataset(data.Dataset):
    def __init__(self, image_path, transform):
        """
        :param image_path: 训练图片路径
        :param transform: 转换
        """
        self.image_path = image_path
        self.transform = transform

        self.paths = []
        self.preprocess()

    def preprocess(self):
        fpaths = sorted(glob(osp.join(self.image_path, '*.jpg')))
        for index, fpath in enumerate(fpaths):
            self.paths.append(fpath)

    def __getitem__(self, index):
        path = self.paths[index]
        image = Image.open(path)
        return self.transform(image), osp.basename(path)

    def __len__(self):
        return len(self.paths)
