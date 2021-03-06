import torch
from PIL import Image
import scipy.io
import os


def extract_cnn_feature(model, loader=None, transforms=None, image_path=None,vis=True,is_normlize=True):
    """
    此函数目前只针对market数据集
    :param model:
    :param loader:
    :param transforms: loader为None时必须指定
    :param image_path: loader为None时需要的图片地址
    :return: 返回特征或特征列表
    """
    cuda_is_available = torch.cuda.is_available
    model.eval()
    if loader is None:
        image = Image.open(image_path)
        image = transforms(image).unsqueeze(0)
        feature = model(image.cuda() if cuda_is_available else image)
        feature = feature.data.cpu()[0]
        return feature
    else:
        features = torch.Tensor()
        count = 0
        feature_length = 0
        for data in loader:
            img, label, _ = data
            n, c, h, w = img.size()
            if count == 0:
                feature_length = model(img.cuda() if cuda_is_available else img).size()[1]
            ff = torch.Tensor(n, feature_length).zero_()
            for i in range(2):
                if i == 1:
                    img = flip_img(img.cpu())
                outputs = model(img.cuda())
                f = outputs.data.cpu()
                ff = ff + f
            if is_normlize:
#                 # norm feature
#                 f_norm = torch.norm(ff, p=2, dim=1, keepdim=True)
#                 ff = ff.div(f_norm.expand_as(ff))
                ff = normalize(ff , 1)
            features = torch.cat((features, ff), 0)
            count += n
            if vis:
                print("已经提取了{}张".format(count))
        model.train()
        return features

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x    

def save_feature(query_feature, gallery_feature, path):
    file_name = os.path.join(path, 'feature_result.mat')
    result = {'gallery_f': gallery_feature.numpy(),
              'query_f': query_feature.numpy()}
    scipy.io.savemat(file_name, result)


def restore_feature(path):
    """
    :param path: 特征文件路径
    :return: query特征，gallery特征
    """
    file_name = os.path.join(path, 'feature_result.mat')
    result = scipy.io.loadmat(file_name)
    query_feature = result['query_f']
    gallery_feature = result['gallery_f']
    return query_feature, gallery_feature


def flip_img(img):
    """
    flip horizontal
    """
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip
