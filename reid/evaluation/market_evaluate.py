import numpy as np
import scipy.io
import torch
import numpy as np
#import time
import os

#######################################################################
# Evaluate
# def evaluate(qf,ql,qc,gf,gl,gc):
#     query = qf.view(-1,1)
#     # print(query.shape)
#     score = torch.mm(gf,query)
#     score = score.squeeze(1).cpu()
#     score = score.numpy()
#     # predict index
#     index = np.argsort(score)  #from small to large
#     index = index[::-1]
#     # index = index[0:2000]
#     # good index
#     query_index = np.argwhere(gl==ql)
#     camera_index = np.argwhere(gc==qc)

#     good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
#     junk_index1 = np.argwhere(gl==-1)
#     junk_index2 = np.intersect1d(query_index, camera_index)
#     junk_index = np.append(junk_index2, junk_index1) #.flatten())
    
#     CMC_tmp = compute_mAP(index, good_index, junk_index)
#     return CMC_tmp


# def compute_mAP(index, good_index, junk_index):
#     ap = 0
#     cmc = torch.IntTensor(len(index)).zero_()
#     if good_index.size==0:   # if empty
#         cmc[0] = -1
#         return ap,cmc

#     # remove junk_index
#     mask = np.in1d(index, junk_index, invert=True)
#     index = index[mask]

#     # find good_index index
#     ngood = len(good_index)
#     mask = np.in1d(index, good_index)
#     rows_good = np.argwhere(mask==True)
#     rows_good = rows_good.flatten()
    
#     cmc[rows_good[0]:] = 1
#     for i in range(ngood):
#         d_recall = 1.0/ngood
#         precision = (i+1)*1.0/(rows_good[i]+1)
#         if rows_good[i]!=0:
#             old_precision = i*1.0/rows_good[i]
#         else:
#             old_precision=1.0
#         ap = ap + d_recall*(old_precision + precision)/2

#     return ap, cmc


def evaluate(qf, ql, qc, gf, gl, gc, is_single=False, vis=True):
    """
    :param is_single: 传入的特征是否为一个样本
    :param qf: 查询图片的特征
    :param ql: 查询图片的标签
    :param qc: 查询图片的相机
    :param gf: 图库图片的特征
    :param gl: 图库图片的标签
    :param gc: 图库图片的相机
    :param vis:是否显示计算过程
    :return: （mAP,CMC）
    """
    if is_single:
        query = qf
        score = np.dot(gf, query)
        # predict index
        index = np.argsort(score)  # from small to large
        index = index[::-1]
        # print(index.shape)#(19732,)
        # index = index[0:2000]
        # good index
        # 同一人物标签
        query_index = np.argwhere(gl == ql)
        # 同一相机标签
        camera_index = np.argwhere(gc == qc)
        # 来自不同相机同一人物的图片索引
        good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        # 标签为-1的
        junk_index1 = np.argwhere(gl == -1)
        # 同一相机同一人物拍摄的图片索引
        junk_index2 = np.intersect1d(query_index, camera_index)
        # 结合上述两种
        junk_index = np.append(junk_index2, junk_index1)  # .flatten())

        CMC_tmp = compute_mAP(index, good_index, junk_index)
        return CMC_tmp
    else:
        CMC = np.zeros(len(gl))
        ap = 0.0
        for i in range(len(ql)):
            query = qf[i]
            score = np.dot(gf, query)
            # print(score.shape)  # (19732,)
            # print(qf.shape)  # (2048,)
            # print(gf.shape)  # (19732,2048)
            # print(ql.shape)  # ()
            # print(gl.shape)  # (19732,)
            index = np.argsort(score)  # from small to large
            index = index[::-1]
            query_index = np.argwhere(gl == ql[i])
            camera_index = np.argwhere(gc == qc[i])
            good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
            junk_index1 = np.argwhere(gl == -1)
            junk_index2 = np.intersect1d(query_index, camera_index)
            junk_index = np.append(junk_index2, junk_index1)  # .flatten())
            ap_tmp, CMC_tmp = compute_mAP(index, good_index, junk_index)
            if CMC_tmp[0] == -1:
                continue
            CMC = CMC + CMC_tmp
            ap += ap_tmp
            if vis:
                print(i, 'acc:', CMC_tmp[0])
        CMC = CMC / len(ql)
        map = ap / len(ql)
        return map, CMC


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = np.zeros(len(index))
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    # 移除了那些数据
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc
