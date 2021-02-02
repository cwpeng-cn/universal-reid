import torch
from PIL import Image
import scipy.io
import os
import numpy as np


def extract_cnn_feature(model, loader=None, transforms=None, image_path=None, vis=True, is_normlize=True):
    """
    此函数目前只针对prid数据集
    :param model:
    :param loader:
    :param transforms: loader为None时必须指定
    :param image_path: loader为None时需要的图片地址
    :return: 返回特征或特征列表
    """
    with torch.no_grad():
        cuda_is_available = torch.cuda.is_available
        model.eval()

        features = torch.Tensor().cuda()
        count = 0
        feature_length = 0
        pids, cams = [], []
        for data in loader:
            imgs, pid, cam = data
            imgs = imgs[0]

            n, c, h, w = imgs.size()
            if count == 0:
                feature_length = model(imgs.cuda() if cuda_is_available else imgs).size()[1]
            ff = torch.Tensor(n, feature_length).zero_().cuda()
            for i in range(2):
                if i == 1:
                    imgs = flip_img(imgs.cpu())
                outputs = model(imgs.cuda())
                f = outputs.data
                ff = ff + f

            ff = torch.mean(ff, dim=0, keepdim=True)
            if is_normlize:
                ff = normalize(ff)

            features = torch.cat((features, ff), 0)

            count += 1
            if vis:
                print("已经提取了{}个人".format(count))
            pids.append(pid)
            cams.append(cam)
        model.train()
        return features.cpu(), pids, cams


def extract_cnn_feature_combined(model, loader=None, transforms=None, image_path=None, vis=True, is_normlize=True,
                                 mode="query"):
    """
    此函数目前只针对prid数据集,混合模型
    :param model:
    :param loader:
    :param transforms: loader为None时必须指定
    :param image_path: loader为None时需要的图片地址
    :return: 返回特征或特征列表
    """
    with torch.no_grad():
        model.eval()
        model.gait_net.eval()

        features = torch.Tensor().cuda()
        count = 0
        pids, cams = [], []

        for pid, data in enumerate(loader):
            cam = 0 if mode == "query" else 1
            rgb_seqs, gait_seqs, ids = data

            f = model(rgb_seqs.cuda(), gait_seqs.cuda())
            if is_normlize:
                f = normalize(f)

            features = torch.cat((features, f), 0)

            count += 1
            if vis:
                print("已经提取了{}个人".format(count))
            pids.append(pid)
            cams.append(cam)

        model.train()
        return features.cpu(), pids, cams


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def save_feature(gallery_paths, gallery_feature, path):
    file_name = os.path.join(path, 'feature_result.mat')
    result = {'gallery_feature': gallery_feature.numpy(),
              'gallery_paths': np.array(gallery_paths)}
    scipy.io.savemat(file_name, result)


def restore_feature(path):
    """
    :param path: 特征文件路径
    :return: query特征，gallery特征
    """
    file_name = os.path.join(path, 'feature_result.mat')
    result = scipy.io.loadmat(file_name)
    gallery_paths = result['gallery_paths']
    gallery_feature = result['gallery_feature']
    return gallery_paths, gallery_feature


def flip_img(img):
    """
    flip horizontal
    """
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip
