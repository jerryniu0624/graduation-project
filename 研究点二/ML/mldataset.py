import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import h5py
import scipy.io
from simplecv.module import SEBlock
import torch.nn.functional as F
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def divisible_pad(image_list, size_divisor=128, to_tensor=True):
    """

    Args:
        image_list: a list of images with shape [channel, height, width]
        size_divisor: int
        to_tensor: whether to convert to tensor
    Returns:
        blob: 4-D ndarray of shape [batch, channel, divisible_max_height, divisible_max_height]
    """
    max_shape = np.array([im.shape for im in image_list]).max(axis=0)

    max_shape[1] = int(np.ceil(max_shape[1] / size_divisor) * size_divisor)
    max_shape[2] = int(np.ceil(max_shape[2] / size_divisor) * size_divisor)

    if to_tensor:
        storage = torch.FloatStorage._new_shared(len(image_list) * np.prod(max_shape))
        out = torch.Tensor(storage).view([len(image_list), max_shape[0], max_shape[1], max_shape[2]])
        out = out.zero_()
    else:
        out = np.zeros([len(image_list), max_shape[0], max_shape[1], max_shape[2]], np.float32)

    for i, resized_im in enumerate(image_list):
        out[i, :, 0:resized_im.shape[1], 0:resized_im.shape[2]] = torch.from_numpy(resized_im)

    return out

class HyperspectralDataset(Dataset):
    def __init__(self, im_path, gt_path, mode='train', num_samples_train=5, num_samples_test=10, seed=42):
        if mode == 'train':
            with h5py.File(im_path, 'r') as h5file:
                images = h5file['data'][:].transpose(0, 3, 1, 2) #[num_pic, h, w, num_bands]
                labels = h5file['gt'][:] #[num_pic, h, w]
        else:
            images = scipy.io.loadmat(im_path)['data'] #[h, w, num_bands]
            labels = scipy.io.loadmat(im_path)['gt'] #[h, w]
            blob = divisible_pad([np.concatenate([images.transpose(2, 0, 1),
                                                labels[None, :, :]], axis=0)], 16, False)
            images = blob[:, :images.shape[-1], :, :]
            labels = blob[:, -1, :, :]
        self.images = torch.tensor(images).float()
        self.labels = torch.tensor(labels)
        self.mode = mode
        self.num_samples_train = num_samples_train
        self.num_samples_test = num_samples_test
        self.seed = seed

        random.seed(self.seed)
        np.random.seed(self.seed)
    
    def __len__(self):
        return len(self.images)

    def generate_mask(self, label):
        h, w = label.shape
        mask = torch.zeros(h, w)
        unique_labels = torch.unique(label)
        for cls in unique_labels:
            if cls == 0:
                continue
            indices = (label == cls).nonzero(as_tuple=False)
            if len(indices) > 0:
                sampled_indices = indices[np.random.choice(len(indices), min(self.num_samples_train, len(indices)), replace=False)]
                mask[sampled_indices[:, 0], sampled_indices[:, 1]] = 1
        return mask

    def generate_reference_points(self, labels):
        """
        为每个类别随机选择 m 个参考点，返回参考点掩码.
        
        :param labels: 标签图像, 形状 [H, W]
        :return: 参考点掩码, 形状 [H, W]，值为 0 和 1
        """
        m = self.num_samples_test  # 每个类别选取的参考点数量
        h, w = labels.shape  # 获取标签图像的高度和宽度
        unique_labels = torch.unique(labels)  # 获取所有类别
        reference_mask = torch.zeros(h, w, dtype=torch.float32)  # 初始化参考点掩码，全 0
        
        for cls in unique_labels:
            if cls == 0:  # 跳过背景类，假设0是背景
                continue
            
            # 获取当前类的所有像素点坐标
            indices = (labels == cls).nonzero(as_tuple=False)
            
            # 如果该类的像素点数小于 m，选择全部；否则随机选择 m 个点
            if len(indices) > 0:
                sampled_indices = indices[torch.randperm(len(indices))[:min(m, len(indices))]]
                
                # 将这些点在参考点掩码中标为1
                for idx in sampled_indices:
                    reference_mask[idx[0], idx[1]] = 1  # 在掩码中将选中的位置置为 1
        
        return reference_mask

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.mode == 'train':
            mask = self.generate_mask(label)
            return image, label, mask
        elif self.mode == 'test':
            mask = self.generate_reference_points(label)
            return self.images[0], self.labels[0], mask