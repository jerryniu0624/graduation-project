import torch.nn as nn
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
from torchvision.transforms import ToPILImage
import numpy as np
from module.mst_plus_plus import MST_Plus_Plus, MSAB
from skimage.metrics import structural_similarity as ssim
from math import log10, sqrt 
import h5py
from matplotlib.gridspec import GridSpec
import time
import math as m
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
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap

def SSIM(x_hsi_bs: torch.Tensor, x_rec: torch.Tensor) -> float:
    """
    计算两个高光谱图像张量之间的平均 SSIM（结构相似性）。

    参数:
    - x_hsi_bs: torch.Tensor, 形状为 [1, 20, h, w] 的重建目标图像。
    - x_rec: torch.Tensor, 形状为 [1, 20, h, w] 的重建结果图像。

    返回:
    - average_ssim: float, 平均 SSIM 值。
    """
    
    # 将张量从 GPU 转移到 CPU，并转换为 numpy 数组
    x_hsi_bs_np = x_hsi_bs.squeeze(0).cpu().detach().numpy()  # 移除 batch 维度，形状变为 [20, h, w]
    x_rec_np = x_rec.squeeze(0).cpu().detach().numpy()  # 移除 batch 维度，形状变为 [20, h, w]

    # 初始化 SSIM 总和变量
    total_ssim = 0

    # 遍历每个波段，计算 SSIM
    for i in range(x_hsi_bs_np.shape[0]):
        band_ssim = ssim(x_hsi_bs_np[i], x_rec_np[i], data_range=x_rec_np[i].max() - x_rec_np[i].min())
        total_ssim += band_ssim

    # 计算平均 SSIM
    average_ssim = total_ssim / x_hsi_bs_np.shape[0]

    return average_ssim
 
def tsne_visualization_and_save(final_feat, y, w, num_per_class=10, save_path=None):
    """
    从特征、标签和掩码中抽取每类指定数量的像素点，计算 t-SNE 并进行可视化或保存。

    参数:
        final_feat: Tensor, [batchsize, num_dim, h, w], 每个像素的特征向量
        y: Tensor, [batchsize, h, w], 每个像素的标签
        w: Tensor, [batchsize, h, w], 标记选中的像素点 (1 表示选中, 0 表示未选中)
        num_per_class: int, 每类选取的像素点数目，默认是 10
        save_path: str, 如果提供路径，则保存 t-SNE 图到该路径 (例如 'tsne_plot.png')，否则显示图

    返回: None
    """
    batch_size, num_dim, height, width = final_feat.shape
    
    # 变换形状
    final_feat = final_feat.view(batch_size, num_dim, -1)  # [batchsize, num_dim, h * w]
    y = y.view(batch_size, -1)  # [batchsize, h * w]
    w = w.view(batch_size, -1)  # [batchsize, h * w]
    
    selected_feats = []
    selected_labels = []
    
    # 遍历每个 batch
    for i in range(batch_size):
        # 选择有效像素（w == 1 的位置）
        mask = w[i].bool()
        selected_feat = final_feat[i][:, mask]  # [num_dim, selected_pixels]
        selected_y = y[i][mask]  # [selected_pixels]
        
        # 转换为 numpy 并准备选择
        selected_feat_np = selected_feat.T.cpu().detach().numpy()  # [selected_pixels, num_dim]
        selected_y_np = selected_y.cpu().detach().numpy()  # [selected_pixels]
        
        # 获取所有的类别
        unique_labels = np.unique(selected_y_np)
        
        # 为每个类别选取 num_per_class 个点
        for label in unique_labels:
            label_mask = (selected_y_np == label)
            label_feats = selected_feat_np[label_mask]  # 当前类的所有特征
            label_indices = np.where(label_mask)[0]  # 当前类的位置索引
            
            # 随机选择 num_per_class 个点
            if len(label_feats) > num_per_class:
                chosen_indices = np.random.choice(label_indices, num_per_class, replace=False)
            else:
                chosen_indices = label_indices  # 如果少于 num_per_class，就选择所有点
            
            # 记录选择的特征和标签
            selected_feats.append(final_feat[i][:, chosen_indices].T)  # 添加特征
            selected_labels.extend(selected_y_np[chosen_indices])  # 添加标签
    
    # 将选中的特征和标签拼接
    selected_feats = torch.cat(selected_feats, dim=0).cpu().detach().numpy()  # 转换为 numpy 数组
    selected_labels = np.array(selected_labels)
    
    # 计算 t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(selected_feats)
    
    # 绘制 t-SNE
    plt.figure(figsize=(16, 16))
    unique_labels = np.unique(selected_labels)
    for label in unique_labels:
        indices = selected_labels == label
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=f'Class {label}')
    
    plt.title('t-SNE visualization of selected pixel features')
    plt.xlabel('t-SNE dim 1')
    plt.ylabel('t-SNE dim 2')
    plt.legend()
    
    # 保存或显示 t-SNE 图
    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        print(f"t-SNE 图已保存至: {save_path}")
    else:
        plt.show()

def conv3x3_gn_relu(in_channel, out_channel, num_group):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 3, 1, 1),
        nn.GroupNorm(num_group, out_channel),
        nn.ReLU(inplace=True),
    )

def conv3x3_gn_conv3x3_relu(in_channel, out_channel, num_group):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 3, 1, 1),
        nn.GroupNorm(num_group, out_channel),
        nn.Conv2d(out_channel, in_channel, 3, 1, 1),
        nn.ReLU(inplace=True),
    )


def downsample2x(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 3, 2, 1),
        nn.ReLU(inplace=True)
    )


def repeat_block(block_channel, r, n):
    layers = [
        nn.Sequential(
            SEBlock(block_channel, r),#SPC32(1, outplane=1, kernel_size=[3,1,1],  padding=[0,0,0]),#Spectral_attention(block_channel, int(block_channel//12), block_channel), ##MSAB(block_channel, 32, int(block_channel/32), 1),#
            conv3x3_gn_relu(block_channel, block_channel, r)
        )
        for _ in range(n)]
    return nn.Sequential(*layers)

# class MetricLearningNet(nn.Module):
#     def __init__(self, num_bands, num_dim):
#         super(MetricLearningNet, self).__init__()
#         # 第一个卷积层：将输入通道数 num_bands 转换为 64
#         self.conv1 = nn.Conv2d(num_bands, 64, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(128, num_dim, kernel_size=3, padding=1)  # 最后一层输出 num_dim 个通道
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         # 前向传播，空间尺寸保持不变
#         x = self.relu(self.conv1(x))  # [batch_size, 64, h, w]
#         x = self.relu(self.conv2(x))  # [batch_size, 128, h, w]
#         x = self.conv3(x)             # [batch_size, num_dim, h, w]
#         return x
class MetricLearningNet(nn.Module):
    def __init__(self, num_bands, num_dim, num_classes):
        super(MetricLearningNet, self).__init__()
        r = int(16 * 1.0)
        self.seed = None
        block1_channels = int(96 * 1.0 / r) * r # (96, 128, 192, 256)
        block2_channels = int(128 * 1.0 / r) * r
        block3_channels = int(192 * 1.0 / r) * r
        block4_channels = int(256 * 1.0 / r) * r
        # print(self.config.in_channels)
        self.feature_ops = nn.ModuleList([
            conv3x3_gn_relu(num_bands, block1_channels, r),

            repeat_block(block1_channels, r, 1),
            nn.Identity(),
            downsample2x(block1_channels, block2_channels),

            repeat_block(block2_channels, r, 1),
            nn.Identity(),
            downsample2x(block2_channels, block3_channels),

            repeat_block(block3_channels, r, 1),
            nn.Identity(),
            downsample2x(block3_channels, block4_channels),

            repeat_block(block4_channels, r, 1),
            nn.Identity(),
        ])
        inner_dim = int(num_dim * 1.0)
        self.reduce_1x1convs = nn.ModuleList([
            nn.Conv2d(block1_channels, inner_dim, 1),
            nn.Conv2d(block2_channels, inner_dim, 1),
            nn.Conv2d(block3_channels, inner_dim, 1),
            nn.Conv2d(block4_channels, inner_dim, 1),
        ])
        self.fuse_3x3convs = nn.ModuleList([
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
        ])
        self.cls_pred_conv = nn.Conv2d(inner_dim, num_classes, 1)  # 添加分类器
        self.time_= 0
        # self.BS = PCA(n_components=self.config.hidden_channels)

    def top_down(self, top, lateral):
        top2x = F.interpolate(top, scale_factor=2.0, mode='nearest')
        return lateral + top2x # 

    def forward(self, x, y=None, w=None, mode='train', **kwargs):
        
        self.time_+= 1 # torch.Size([1, 270, 288, 400])

        # x = x[:, [0, 30, 56, 67, 107, 138, 142, 206],:,:]
        feat_list = []
        for op in self.feature_ops:
            x = op(x)
            if isinstance(op, nn.Identity):
                feat_list.append(x)

        inner_feat_list = [self.reduce_1x1convs[i](feat) for i, feat in enumerate(feat_list)] # [#self.reduce_1x1convs[i](feat) for i, feat in enumerate(feat_list)]
        inner_feat_list.reverse()

        out_feat_list = [self.fuse_3x3convs[0](inner_feat_list[0])]
        for i in range(len(feat_list) - 1):
            inner = self.top_down(out_feat_list[i], inner_feat_list[i+1])
            out = self.fuse_3x3convs[i + 1](inner)
            out_feat_list.append(out)

        final_feat = out_feat_list[-1]
        logit = self.cls_pred_conv(final_feat)  # 测试时输出 logits
        if self.time_ % 10 == 0:
            visualize_classification_with_custom_colormap(logit, self.time_)
        # 在测试阶段输出 logits，训练阶段输出 final_feat
        if mode == 'test':
            
            return logit # cross_entropy_loss(logit, y, w), 
        else:
            return logit # 训练时输出特征 final_feat, 

class LyotFilter(nn.Module):
    def __init__(self, in_dim, out_dim, minBand, maxBand, nBandDataset, dataset):
        super(LyotFilter, self).__init__()
        self.weight_ = nn.Parameter(10*torch.rand((out_dim)))
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.minBand = minBand
        self.maxBand = maxBand
        self.nBandDataset = nBandDataset
        self.dataset = dataset
        # weight =
        # self.weight = nn.Parameter(torch.rand((out_dim, in_dim)).cuda())

    def forward(self, x):
        # self.weight = self.weight.cpu()
        # tmp = 0
        weight = torch.rand((self.out_dim, self.in_dim)).cuda() # srf
        # print(self.weight_) # voltage
        tmp = 0 # 尽量别让反传时找不到梯度，因此不要detach,转到cpu之类的;对weight以一个整体操作，不要分开操作

		# IP
        if self.dataset == 'IP':
            # x = x[:,:,:,:32]
            for n in range(self.nBandDataset):
                if (n >= 104 - 1 and n <= 108 - 1) or (n >= 150 - 1 and n <= 163 - 1) or n == 220 - 1:
                    tmp = tmp + 1
                else:
                    band = self.minBand + (self.maxBand - self.minBand) * n / self.nBandDataset
                    weight[:,n - tmp] = (0.5 - 0.5*torch.cos(2*m.pi*(-0.01)*(pow(self.weight_,-1))/(band*pow(10,-6))))
		# PU
        if self.dataset == 'PU':
            # x = x[:,:,:,:72]
            for n in range(self.nBandDataset):
                if n > self.in_dim - 1 - 12:
                    tmp = tmp + 1
                else:
                    band = self.minBand + (self.maxBand - self.minBand) * n / self.nBandDataset
                    weight[:,n - tmp] = (0.5 - 0.5*torch.cos(2*m.pi*(-0.01)*(pow(self.weight_,-1))/(band*pow(10,-6))))

        # PU
        if self.dataset == 'PC':
            # x = x[:,:,:,:72]
            for n in range(self.nBandDataset):
                if n > self.in_dim - 1 - 13:
                    tmp = tmp + 1
                else:
                    band = self.minBand + (self.maxBand - self.minBand) * n / self.nBandDataset
                    weight[:,n - tmp] = (0.5 - 0.5*torch.cos(2*m.pi*(-0.01)*(pow(self.weight_,-1))/(band*pow(10,-6))))
		# SV
        if self.dataset == 'SV':
            # x = x[:,:,:,:32]
            for n in range(self.nBandDataset):
                if (n >= 108 - 1 and n <= 112 - 1) or (n >= 154 - 1 and n <= 167 - 1) or n == 224 - 1:
                    tmp = tmp + 1
                else:
                    band = self.minBand + (self.maxBand - self.minBand) * n / self.nBandDataset
                    weight[:,n - tmp] = (0.5 - 0.5*torch.cos(2*m.pi*(-0.01)*(pow(self.weight_,-1))/(band*pow(10,-6))))	
		
		# DC
        if self.dataset == 'DC':
            # x = x[:,:,:,:27]
            for n in range(self.nBandDataset):
                band = self.minBand + (self.maxBand - self.minBand) * n / self.nBandDataset
                weight[:,n - tmp] = (0.5 - 0.5*torch.cos(2*m.pi*(-0.01)*(pow(self.weight_,-1))/(band*pow(10,-6))))
        
        # print('dcdcdcdcdcdc')
		# HU2018
        if self.dataset == 'HU18':
            # x = x[:,:,:,1:24]
            for n in range(self.nBandDataset):
                band = self.minBand + (self.maxBand - self.minBand) * n / self.nBandDataset
                weight[:,n - tmp] = (0.5 - 0.5*torch.cos(2*m.pi*(-0.01)*(pow(self.weight_,-1))/(band*pow(10,-6))))
        
        if self.dataset == 'HC':
            for n in range(self.nBandDataset):
                band = self.minBand + (self.maxBand - self.minBand) * n / self.nBandDataset
                weight[:,n - tmp] = (0.5 - 0.5*torch.cos(2*m.pi*(-0.01)*(pow(self.weight_,-1))/(band*pow(10,-6))))

        if self.dataset == 'LK':
            for n in range(self.nBandDataset):
                band = self.minBand + (self.maxBand - self.minBand) * n / self.nBandDataset
                weight[:,n - tmp] = (0.5 - 0.5*torch.cos(2*m.pi*(-0.01)*(pow(self.weight_,-1))/(band*pow(10,-6))))
# IP
        if self.dataset == 'BS':
            # x = x[:,:,:,:32]
            for n in range(self.nBandDataset):
                if (n >= 0 - 1 and n <= 9 - 1) or (n >= 56 - 1 and n <= 81 - 1) or (n >= 98 - 1 and n <= 101 - 1)\
                    or (n >= 120 - 1 and n <= 133 - 1) or (n >= 165 - 1 and n <= 186 - 1) or (n >= 221 - 1 ):
                    tmp = tmp + 1
                else:
                    band = self.minBand + (self.maxBand - self.minBand) * n / self.nBandDataset
                    weight[:,n - tmp] = (0.5 - 0.5*torch.cos(2*m.pi*(-0.01)*(pow(self.weight_,-1))/(band*pow(10,-6))))
        
        return torch.matmul(x, F.relu(weight.transpose(0,1))/weight.transpose(0,1).sum(0))

class MetricLearningLCTFNet(nn.Module):
    def __init__(self, num_bands, num_dim, num_classes):
        super(MetricLearningLCTFNet, self).__init__()
        r = int(16 * 1.0)
        self.seed = None
        block1_channels = int(96 * 1.0 / r) * r # (96, 128, 192, 256)
        block2_channels = int(128 * 1.0 / r) * r
        block3_channels = int(192 * 1.0 / r) * r
        block4_channels = int(256 * 1.0 / r) * r
        # print(self.config.in_channels)
        self.feature_ops = nn.ModuleList([
            conv3x3_gn_relu(num_bands, block1_channels, r),

            repeat_block(block1_channels, r, 1),
            nn.Identity(),
            downsample2x(block1_channels, block2_channels),

            repeat_block(block2_channels, r, 1),
            nn.Identity(),
            downsample2x(block2_channels, block3_channels),

            repeat_block(block3_channels, r, 1),
            nn.Identity(),
            downsample2x(block3_channels, block4_channels),

            repeat_block(block4_channels, r, 1),
            nn.Identity(),
        ])
        inner_dim = int(num_dim * 1.0)
        self.reduce_1x1convs = nn.ModuleList([
            nn.Conv2d(block1_channels, inner_dim, 1),
            nn.Conv2d(block2_channels, inner_dim, 1),
            nn.Conv2d(block3_channels, inner_dim, 1),
            nn.Conv2d(block4_channels, inner_dim, 1),
        ])
        self.fuse_3x3convs = nn.ModuleList([
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
        ])
        self.cls_pred_conv = nn.Conv2d(inner_dim, num_classes, 1)  # 添加分类器
        self.time_= 0
        self.srf = LyotFilter(100, num_bands, 400, 1000, 100, 'LK')
        # self.BS = PCA(n_components=self.config.hidden_channels)

    def top_down(self, top, lateral):
        top2x = F.interpolate(top, scale_factor=2.0, mode='nearest')
        return lateral + top2x # 

    def forward(self, x, y=None, w=None, mode='train', **kwargs):
        x = x.permute(0,2,3,1)
        x = self.srf(x) 
        x = x.permute(0,3,1,2)
        if self.training:
            self.time_+= 1 # torch.Size([1, 270, 288, 400])
        if self.time_ == 998:
            print(self.srf.weight_)
        # self.srf
        # x = x[:, [0, 30, 56, 67, 107, 138, 142, 206],:,:]
        feat_list = []
        for op in self.feature_ops:
            x = op(x)
            if isinstance(op, nn.Identity):
                feat_list.append(x)

        inner_feat_list = [self.reduce_1x1convs[i](feat) for i, feat in enumerate(feat_list)] # [#self.reduce_1x1convs[i](feat) for i, feat in enumerate(feat_list)]
        inner_feat_list.reverse()

        out_feat_list = [self.fuse_3x3convs[0](inner_feat_list[0])]
        for i in range(len(feat_list) - 1):
            inner = self.top_down(out_feat_list[i], inner_feat_list[i+1])
            out = self.fuse_3x3convs[i + 1](inner)
            out_feat_list.append(out)

        final_feat = out_feat_list[-1]
        logit = self.cls_pred_conv(final_feat)  # 测试时输出 logits
        # 在测试阶段输出 logits，训练阶段输出 final_feat
        if mode == 'test':
            return cross_entropy_loss(logit, y, w), logit
        else:
            return final_feat, logit  # 训练时输出特征
    
class MetricLearningRecLCTFNet(nn.Module):
    def __init__(self, num_bands, num_dim, num_classes):
        super(MetricLearningRecLCTFNet, self).__init__()
        r = int(16 * 1.0)
        self.seed = None
        block1_channels = int(96 * 1.0 / r) * r # (96, 128, 192, 256)
        block2_channels = int(128 * 1.0 / r) * r
        block3_channels = int(192 * 1.0 / r) * r
        block4_channels = int(256 * 1.0 / r) * r
        # print(self.config.in_channels)
        self.feature_ops = nn.ModuleList([
            conv3x3_gn_relu(num_bands, block1_channels, r),

            repeat_block(block1_channels, r, 1),
            nn.Identity(),
            downsample2x(block1_channels, block2_channels),

            repeat_block(block2_channels, r, 1),
            nn.Identity(),
            downsample2x(block2_channels, block3_channels),

            repeat_block(block3_channels, r, 1),
            nn.Identity(),
            downsample2x(block3_channels, block4_channels),

            repeat_block(block4_channels, r, 1),
            nn.Identity(),
        ])
        inner_dim = int(num_dim * 1.0)
        self.reduce_1x1convs = nn.ModuleList([
            nn.Conv2d(block1_channels, inner_dim, 1),
            nn.Conv2d(block2_channels, inner_dim, 1),
            nn.Conv2d(block3_channels, inner_dim, 1),
            nn.Conv2d(block4_channels, inner_dim, 1),
        ])
        self.fuse_3x3convs = nn.ModuleList([
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
        ])
        self.cls_pred_conv = nn.Conv2d(inner_dim, num_classes, 1)  # 添加分类器
        self.time_= 0
        self.srf = LyotFilter(100, num_bands, 400, 1000, 100, 'LK')
        # self.BS = PCA(n_components=self.config.hidden_channels)

    def top_down(self, top, lateral):
        top2x = F.interpolate(top, scale_factor=2.0, mode='nearest')
        return lateral + top2x # 

    def forward(self, x, y=None, w=None, mode='train', **kwargs):
        x = x.permute(0,2,3,1)
        x = self.srf(x) 
        x = x.permute(0,3,1,2)
        x_rec = x
        self.time_+= 1 # torch.Size([1, 270, 288, 400])
        # self.srf
        # x = x[:, [0, 30, 56, 67, 107, 138, 142, 206],:,:]
        feat_list = []
        for op in self.feature_ops:
            x = op(x)
            if isinstance(op, nn.Identity):
                feat_list.append(x)

        inner_feat_list = [self.reduce_1x1convs[i](feat) for i, feat in enumerate(feat_list)] # [#self.reduce_1x1convs[i](feat) for i, feat in enumerate(feat_list)]
        inner_feat_list.reverse()

        out_feat_list = [self.fuse_3x3convs[0](inner_feat_list[0])]
        for i in range(len(feat_list) - 1):
            inner = self.top_down(out_feat_list[i], inner_feat_list[i+1])
            out = self.fuse_3x3convs[i + 1](inner)
            out_feat_list.append(out)

        final_feat = out_feat_list[-1]
        logit = self.cls_pred_conv(final_feat)  # 测试时输出 logits
        # 在测试阶段输出 logits，训练阶段输出 final_feat
        if mode == 'test':
            
            return cross_entropy_loss(logit, y, w),  logit
        else:
            return final_feat,  logit  # 训练时输出特征

def set_srf_args(test_data_path):
    if 'LongKou' in test_data_path:
        minBand, maxBand, nBandDataset, dataset = 400, 1000, 100, 'LK'
    elif 'HongHu' in test_data_path:
        minBand, maxBand, nBandDataset, dataset = 400, 1000, 100, 'HH'
    elif 'HanChuan' in test_data_path:
        minBand, maxBand, nBandDataset, dataset = 400, 1000, 100, 'HC'
    elif 'Loukia' in test_data_path:
        minBand, maxBand, nBandDataset, dataset = 400, 1000, 100, 'LKA'
    elif 'Dioni' in test_data_path:
        minBand, maxBand, nBandDataset, dataset = 400, 1000, 100, 'DN'
    elif 'Shanghai' in test_data_path:
        minBand, maxBand, nBandDataset, dataset = 356, 1000, 100, 'SH'
    elif 'Xuzhou' in test_data_path:
        minBand, maxBand, nBandDataset, dataset = 415, 1000, 100, 'XZ'
    elif 'Hangzhou' in test_data_path:
        minBand, maxBand, nBandDataset, dataset = 356, 1000, 100, 'HZ'
    elif 'Houston18' in test_data_path:
        minBand, maxBand, nBandDataset, dataset = 380, 1000, 100, 'HU18'
    elif 'pavia1' in test_data_path:
        minBand, maxBand, nBandDataset, dataset = 430, 860, 100, 'PU'
    elif 'AB1' in test_data_path:
        minBand, maxBand, nBandDataset, dataset = 400, 1000, 100, 'AB'
        # Houston18
    return minBand, maxBand, nBandDataset, dataset
    
class FixedPosLinear(nn.Module):
    def __init__(self, in_dim, out_dim, minBand, maxBand, nBandDataset, dataset):
        super(FixedPosLinear, self).__init__()
        P = scipy.io.loadmat('/mnt/nas/xinjiang/code/nyz/FreeNet-master/P.mat')['P']
        # self.weight = nn.Parameter(torch.tensor(P).unsqueeze(0))
        # self.bias = nn.Parameter(torch.zeros((out_dim,)))
        # IP
        if dataset == 'IP':
            self.weight = F.interpolate(torch.tensor(P).unsqueeze(0), size=32).transpose(1,2).squeeze(0).float()
		# PU
        if dataset == 'PU':
            self.weight = F.interpolate(torch.tensor(P).unsqueeze(0)[:,:,3:], size=72).transpose(1,2).squeeze(0).float()
        # PU
        if dataset == 'PC':
            self.weight = F.interpolate(torch.tensor(P).unsqueeze(0)[:,:,3:], size=72).transpose(1,2).squeeze(0).float()
		# SV
		# SV
        if dataset == 'SV':
            self.weight = F.interpolate(torch.tensor(P).unsqueeze(0), size=32).transpose(1,2).squeeze(0).float()
		# DC
        if dataset == 'DC':
            self.weight = F.interpolate(torch.tensor(P).unsqueeze(0), size=27).transpose(1,2).squeeze(0).float()
        # HU18
        if dataset == 'HU18':
            self.weight = F.interpolate(torch.tensor(P).unsqueeze(0), size=23).transpose(1,2).squeeze(0).float()
        # HU18
        if dataset == 'HC':
            self.weight = torch.tensor(P).transpose(1,0).float()
        # HU18
        if dataset == 'LK':
            self.weight = torch.tensor(P).transpose(1,0).float()
        if dataset == 'LKA':
            self.weight = torch.tensor(P).transpose(1,0).float()
        if dataset == 'DN':
            self.weight = torch.tensor(P).transpose(1,0).float()
        # HU18
        if dataset == 'XZ':
            self.weight = torch.tensor(P).transpose(1,0).float()
        if dataset == 'HZ':
            self.weight = torch.tensor(P).transpose(1,0).float()
        if dataset == 'SH':
            self.weight = torch.tensor(P).transpose(1,0).float()
        # HU18
        if dataset == 'HH':
            self.weight = torch.tensor(P).transpose(1,0).float()
        if dataset == 'BS':
            self.weight = torch.tensor(P)[:,7:].transpose(1,0).float()
        if dataset == 'AB':
            self.weight = torch.tensor(P).transpose(1,0).float()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.minBand = minBand
        self.maxBand = maxBand
        self.nBandDataset = nBandDataset
        self.dataset = dataset
        
    def forward(self, x):
        # self.weight = torch.clamp(self.weight,0,1)
# 		min_vals = torch.sum(torch.abs(torch.tensor(matrix)))
# # max_vals = np.max(matrix, axis=0)
        batch_size, h, w, num_bands = x.shape
        tmp = 0 # 尽量别让反传时找不到梯度，因此不要detach,转到cpu之类的;对weight以一个整体操作，不要分开操作
        # print(self.weight[0,0])
		# IP
        if self.dataset == 'IP':
            x = x[:,:,:,:32]
		# PU
        if self.dataset == 'PU':
            x = x[:,:,:,:72]
        # PU
        if self.dataset == 'PC':
            x = x[:,:,:,:72]
		# SV
        if self.dataset == 'SV':
            x = x[:,:,:,:32]
		# DC
        if self.dataset == 'DC':
            x = x[:,:,:,:27]
        # HU18
        if self.dataset == 'HU18':
            x = x[:,:,:,1:24]
        # HU18
        if self.dataset == 'HC':
            x = F.interpolate(x[:,:,:,:50].reshape(-1, 50).unsqueeze(1), size=31, mode='linear', align_corners=False).squeeze(1).view(batch_size, h, w, 31)
        # HU18
        if self.dataset == 'LK':
            x = F.interpolate(x[:,:,:,:50].reshape(-1, 50).unsqueeze(1), size=31, mode='linear', align_corners=False).squeeze(1).view(batch_size, h, w, 31)
        # HU18
        if self.dataset == 'HH':
            x = F.interpolate(x[:,:,:,:50].reshape(-1, 50).unsqueeze(1), size=31, mode='linear', align_corners=False).squeeze(1).view(batch_size, h, w, 31)
        if self.dataset == 'BS':
            x = x[:,:,:,:24]
        if self.dataset == 'LKA':
            x = F.interpolate(x[:,:,:,:50].reshape(-1, 50).unsqueeze(1), size=31, mode='linear', align_corners=False).squeeze(1).view(batch_size, h, w, 31)
        if self.dataset == 'DN':
            x = F.interpolate(x[:,:,:,:50].reshape(-1, 50).unsqueeze(1), size=31, mode='linear', align_corners=False).squeeze(1).view(batch_size, h, w, 31)
        # HU18
        if self.dataset == 'XZ':
            x = F.interpolate(x[:,:,:,:48].reshape(-1, 48).unsqueeze(1), size=31, mode='linear', align_corners=False).squeeze(1).view(batch_size, h, w, 31)
        if self.dataset == 'HZ':
            x = F.interpolate(x[:,:,:,6:54].reshape(-1, 48).unsqueeze(1), size=31, mode='linear', align_corners=False).squeeze(1).view(batch_size, h, w, 31)
        if self.dataset == 'SH':
            x = F.interpolate(x[:,:,:,6:54].reshape(-1, 48).unsqueeze(1), size=31, mode='linear', align_corners=False).squeeze(1).view(batch_size, h, w, 31)
        if self.dataset == 'AB':
            x = F.interpolate(x[:,:,:,:26].reshape(-1, 26).unsqueeze(1), size=31, mode='linear', align_corners=False).squeeze(1).view(batch_size, h, w, 31)
# # 归一化矩阵
# 			normalized_matrix = torch.abs(torch.tensor(matrix)) / min_vals
        
        return torch.matmul(x, self.weight.cuda())

class LyotFilter(nn.Module):
    def __init__(self, in_dim, out_dim, minBand, maxBand, nBandDataset, dataset):
        super(LyotFilter, self).__init__()
        self.weight_ = nn.Parameter(10*torch.rand((out_dim)))
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.minBand = minBand
        self.maxBand = maxBand
        self.nBandDataset = nBandDataset
        self.dataset = dataset
        # weight =
        # self.weight = nn.Parameter(torch.rand((out_dim, in_dim)).cuda())

    def forward(self, x):
        # self.weight = self.weight.cpu()
        # tmp = 0
        weight = torch.rand((self.out_dim, self.in_dim)).cuda() # srf
        # print(self.weight_) # voltage
        tmp = 0 # 尽量别让反传时找不到梯度，因此不要detach,转到cpu之类的;对weight以一个整体操作，不要分开操作

		# IP
        if self.dataset == 'IP':
            # x = x[:,:,:,:32]
            for n in range(self.nBandDataset):
                if (n >= 104 - 1 and n <= 108 - 1) or (n >= 150 - 1 and n <= 163 - 1) or n == 220 - 1:
                    tmp = tmp + 1
                else:
                    band = self.minBand + (self.maxBand - self.minBand) * n / self.nBandDataset
                    weight[:,n - tmp] = (0.5 - 0.5*torch.cos(2*m.pi*(-0.01)*(pow(self.weight_,-1))/(band*pow(10,-6))))
		# PU
        if self.dataset == 'PU':
            # x = x[:,:,:,:72]
            for n in range(self.nBandDataset):
                if n > self.in_dim - 1 - 12:
                    tmp = tmp + 1
                else:
                    band = self.minBand + (self.maxBand - self.minBand) * n / self.nBandDataset
                    weight[:,n - tmp] = (0.5 - 0.5*torch.cos(2*m.pi*(-0.01)*(pow(self.weight_,-1))/(band*pow(10,-6))))

        # PU
        if self.dataset == 'PC':
            # x = x[:,:,:,:72]
            for n in range(self.nBandDataset):
                if n > self.in_dim - 1 - 13:
                    tmp = tmp + 1
                else:
                    band = self.minBand + (self.maxBand - self.minBand) * n / self.nBandDataset
                    weight[:,n - tmp] = (0.5 - 0.5*torch.cos(2*m.pi*(-0.01)*(pow(self.weight_,-1))/(band*pow(10,-6))))
		# SV
        if self.dataset == 'SV':
            # x = x[:,:,:,:32]
            for n in range(self.nBandDataset):
                if (n >= 108 - 1 and n <= 112 - 1) or (n >= 154 - 1 and n <= 167 - 1) or n == 224 - 1:
                    tmp = tmp + 1
                else:
                    band = self.minBand + (self.maxBand - self.minBand) * n / self.nBandDataset
                    weight[:,n - tmp] = (0.5 - 0.5*torch.cos(2*m.pi*(-0.01)*(pow(self.weight_,-1))/(band*pow(10,-6))))	
		
		# DC
        if self.dataset == 'DC':
            # x = x[:,:,:,:27]
            for n in range(self.nBandDataset):
                band = self.minBand + (self.maxBand - self.minBand) * n / self.nBandDataset
                weight[:,n - tmp] = (0.5 - 0.5*torch.cos(2*m.pi*(-0.01)*(pow(self.weight_,-1))/(band*pow(10,-6))))
        
        # print('dcdcdcdcdcdc')
		# HU2018
        if self.dataset == 'HU18':
            # x = x[:,:,:,1:24]
            for n in range(self.nBandDataset):
                band = self.minBand + (self.maxBand - self.minBand) * n / self.nBandDataset
                weight[:,n - tmp] = (0.5 - 0.5*torch.cos(2*m.pi*(-0.01)*(pow(self.weight_,-1))/(band*pow(10,-6))))

        if self.dataset in ['LK', 'HC', 'LKA', 'DN', 'XZ', 'HZ', 'SH', 'HH']:
            for n in range(self.nBandDataset):
                band = self.minBand + (self.maxBand - self.minBand) * n / self.nBandDataset
                weight[:,n - tmp] = (0.5 - 0.5*torch.cos(2*m.pi*(-0.01)*(pow(self.weight_,-1))/(band*pow(10,-6))))
# IP
        if self.dataset == 'BS':
            # x = x[:,:,:,:32]
            for n in range(self.nBandDataset):
                if (n >= 0 - 1 and n <= 9 - 1) or (n >= 56 - 1 and n <= 81 - 1) or (n >= 98 - 1 and n <= 101 - 1)\
                    or (n >= 120 - 1 and n <= 133 - 1) or (n >= 165 - 1 and n <= 186 - 1) or (n >= 221 - 1 ):
                    tmp = tmp + 1
                else:
                    band = self.minBand + (self.maxBand - self.minBand) * n / self.nBandDataset
                    weight[:,n - tmp] = (0.5 - 0.5*torch.cos(2*m.pi*(-0.01)*(pow(self.weight_,-1))/(band*pow(10,-6))))
        
        return torch.matmul(x, F.relu(weight.transpose(0,1))/weight.transpose(0,1).sum(0))

# MetricLearningRGBDynamicLCTFNet
class MetricLearning1DynamicLCTFNet(nn.Module):
    def __init__(self, num_bands, num_dim, num_classes, minBand, maxBand, nBandDataset, dataset):
        super(MetricLearning1DynamicLCTFNet, self).__init__()
        r = int(16 * 1.0)

        block1_channels = int(96 * 1.0 / r) * r # (96, 128, 192, 256)
        block2_channels = int(128 * 1.0 / r) * r
        block3_channels = int(192 * 1.0 / r) * r
        block4_channels = int(256 * 1.0 / r) * r
        # print(self.config.in_channels)
        self.feature_ops = nn.ModuleList([
            conv3x3_gn_relu(num_bands, block1_channels, r),

            repeat_block(block1_channels, r, 1),
            nn.Identity(),
            downsample2x(block1_channels, block2_channels),

            repeat_block(block2_channels, r, 1),
            nn.Identity(),
            downsample2x(block2_channels, block3_channels),

            repeat_block(block3_channels, r, 1),
            nn.Identity(),
            downsample2x(block3_channels, block4_channels),

            repeat_block(block4_channels, r, 1),
            nn.Identity(),
        ])
        inner_dim = int(num_dim * 1.0)
        self.reduce_1x1convs = nn.ModuleList([
            nn.Conv2d(block1_channels, inner_dim, 1),
            nn.Conv2d(block2_channels, inner_dim, 1),
            nn.Conv2d(block3_channels, inner_dim, 1),
            nn.Conv2d(block4_channels, inner_dim, 1),
        ])
        self.fuse_3x3convs = nn.ModuleList([
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
        ])
        self.cls_pred_conv = nn.Conv2d(inner_dim, num_classes, 1)  # 添加分类器
        self.time_= 0

        self.full_band_srf = LyotFilter(100, 1, minBand, maxBand, nBandDataset, dataset)

        self.srf1 = DynamicLyotFilter(100, 2, 1, minBand, maxBand, nBandDataset, dataset)
        # num_first_shots = num_first_shots + num_second_shots
        self.srf2 = DynamicLyotFilter(100, 3, 2, minBand, maxBand, nBandDataset, dataset)
        # num_first_shots = num_first_shots + num_second_shots + num_third_shots
        self.srf3 = DynamicLyotFilter(100, 4, 3, minBand, maxBand, nBandDataset, dataset)
        self.srf4 = DynamicLyotFilter(100, 5, 4, minBand, maxBand, nBandDataset, dataset)
        self.srf5 = DynamicLyotFilter(100, 6, 5, minBand, maxBand, nBandDataset, dataset)
        self.seed = None

    def top_down(self, top, lateral):
        top2x = F.interpolate(top, scale_factor=2.0, mode='nearest')
        return lateral + top2x # 

    def forward(self, x, y=None, w=None, mode='train', **kwargs):
        # 先拍个RGB图像或者固定的3通道SRF拍出来的图片，然后根据此生成液态水晶的光谱响应曲线[num_bands, 3]额外生成3波段，与原来的的3波段一起进行后面的度量学习和分类网络
        x_hsi = x
        x = x.permute(0,2,3,1)
        x_hsi = x_hsi.permute(0,2,3,1)
        # x_rgb = self.rgb_srf(x) 
        x_srf = self.full_band_srf(x) 
        # x = x.permute(0,3,1,2)
        # x = x.permute(0,2,3,1)


        x_dynamic_srf1 = self.srf1(x_srf, x_hsi) 
        x_dynamic_srf2 = torch.cat((x_srf, x_dynamic_srf1), dim=-1)

        x_dynamic_srf3 = self.srf2(x_dynamic_srf2, x_hsi) 
        x_dynamic_srf4 = torch.cat((x_dynamic_srf2, x_dynamic_srf3), dim=-1)
        x_dynamic_srf5 = self.srf3(x_dynamic_srf4, x_hsi) 
        x_dynamic_srf6 = torch.cat((x_dynamic_srf4, x_dynamic_srf5), dim=-1)
        x_dynamic_srf7 = self.srf4(x_dynamic_srf6, x_hsi) 
        x_dynamic_srf8 = torch.cat((x_dynamic_srf6, x_dynamic_srf7), dim=-1)
        x_dynamic_srf9 = self.srf5(x_dynamic_srf8, x_hsi) 
        x_dynamic_srf10 = torch.cat((x_dynamic_srf8, x_dynamic_srf9), dim=-1)

        x = x_dynamic_srf10
       
        x = x.permute(0,3,1,2) # [4, 6, 512, 512]

        self.time_+= 1 
        # self.srf
        # x = x[:, [0, 30, 56, 67, 107, 138, 142, 206],:,:]
        feat_list = []
        for op in self.feature_ops:
            x = op(x)
            if isinstance(op, nn.Identity):
                feat_list.append(x)

        inner_feat_list = [self.reduce_1x1convs[i](feat) for i, feat in enumerate(feat_list)] # [#self.reduce_1x1convs[i](feat) for i, feat in enumerate(feat_list)]
        inner_feat_list.reverse()

        out_feat_list = [self.fuse_3x3convs[0](inner_feat_list[0])]
        for i in range(len(feat_list) - 1):
            inner = self.top_down(out_feat_list[i], inner_feat_list[i+1])
            out = self.fuse_3x3convs[i + 1](inner)
            out_feat_list.append(out)

        final_feat = out_feat_list[-1] # [4, 64, 512, 512]
        logit = self.cls_pred_conv(final_feat)  # 测试时输出 logits
        # 在测试阶段输出 logits，训练阶段输出 final_feat
        if mode == 'test':
            return cross_entropy_loss(logit, y, w), logit
        else:
            return final_feat, logit # 训练时输出特征
        
# MetricLearningRGBDynamicLCTFNet
class MetricLearningRGBDynamicLCTFNet(nn.Module):
    def __init__(self, num_first_shots, num_second_shots,  num_third_shots, num_bands, num_dim, num_classes, minBand, maxBand, nBandDataset, dataset):
        super(MetricLearningRGBDynamicLCTFNet, self).__init__()
        r = int(16 * 1.0)
        self.num_third_shots = num_third_shots
        self.num_second_shots = num_second_shots
        block1_channels = int(96 * 1.0 / r) * r # (96, 128, 192, 256)
        block2_channels = int(128 * 1.0 / r) * r
        block3_channels = int(192 * 1.0 / r) * r
        block4_channels = int(256 * 1.0 / r) * r
        # print(self.config.in_channels)
        self.feature_ops = nn.ModuleList([
            conv3x3_gn_relu(num_bands, block1_channels, r),

            repeat_block(block1_channels, r, 1),
            nn.Identity(),
            downsample2x(block1_channels, block2_channels),

            repeat_block(block2_channels, r, 1),
            nn.Identity(),
            downsample2x(block2_channels, block3_channels),

            repeat_block(block3_channels, r, 1),
            nn.Identity(),
            downsample2x(block3_channels, block4_channels),

            repeat_block(block4_channels, r, 1),
            nn.Identity(),
        ])
        inner_dim = int(num_dim * 1.0)
        self.reduce_1x1convs = nn.ModuleList([
            nn.Conv2d(block1_channels, inner_dim, 1),
            nn.Conv2d(block2_channels, inner_dim, 1),
            nn.Conv2d(block3_channels, inner_dim, 1),
            nn.Conv2d(block4_channels, inner_dim, 1),
        ])
        self.fuse_3x3convs = nn.ModuleList([
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
        ])
        self.cls_pred_conv = nn.Conv2d(inner_dim, num_classes, 1)  # 添加分类器
        self.time_= 0
        self.rgb_srf = FixedPosLinear(100, 3, minBand, maxBand, nBandDataset, dataset)
        self.dynamic_shot = DynamicLyotFilter(num_first_shots, num_bands - num_first_shots, 100, minBand, maxBand, nBandDataset, dataset)

        self.seed = None

    def top_down(self, top, lateral):
        top2x = F.interpolate(top, scale_factor=2.0, mode='nearest')
        return lateral + top2x # 

    def forward(self, x, y=None, w=None, mode='train', **kwargs):
        # 先拍个RGB图像或者固定的3通道SRF拍出来的图片，然后根据此生成液态水晶的光谱响应曲线[num_bands, 3]额外生成3波段，与原来的的3波段一起进行后面的度量学习和分类网络
        x_hsi = x
        x = x.permute(0,2,3,1)
        x_hsi = x_hsi.permute(0,2,3,1)
        # x_rgb = self.rgb_srf(x) 
        first_shot = self.rgb_srf(x) 
        # x = x.permute(0,3,1,2)
        # x = x.permute(0,2,3,1)
        dynamic_shot = self.dynamic_shot(first_shot, x_hsi) 
        x = torch.cat((dynamic_shot, first_shot), dim=-1)
        x = x.permute(0,3,1,2) # [4, 6, 512, 512]
        if self.training:
            self.time_+= 1 

        # if self.time_ == 998:
        #     print(self.srf1.voltages)
        #     print(self.srf2.voltages)
        
        feat_list = []
        for op in self.feature_ops:
            x = op(x)
            if isinstance(op, nn.Identity):
                feat_list.append(x)

        inner_feat_list = [self.reduce_1x1convs[i](feat) for i, feat in enumerate(feat_list)] # [#self.reduce_1x1convs[i](feat) for i, feat in enumerate(feat_list)]
        inner_feat_list.reverse()

        out_feat_list = [self.fuse_3x3convs[0](inner_feat_list[0])]
        for i in range(len(feat_list) - 1):
            inner = self.top_down(out_feat_list[i], inner_feat_list[i+1])
            out = self.fuse_3x3convs[i + 1](inner)
            out_feat_list.append(out)

        final_feat = out_feat_list[-1] # [4, 64, 512, 512]

        logit = self.cls_pred_conv(final_feat)  # 测试时输出 logits
        metric_loss = metric_learning_loss(logit, y, w)
        cls_loss = cross_entropy_loss(logit, y, w)

        if mode == 'train':
            return metric_loss, logit
        else:
            return cls_loss, logit # 训练时输出特征
        
class MetricLearningDynamicLCTFNet(nn.Module):
    def __init__(self, num_first_shots, num_bands, num_dim, num_classes, minBand, maxBand, nBandDataset, dataset):
        super(MetricLearningDynamicLCTFNet, self).__init__()
        r = int(16 * 1.0)
        block1_channels = int(96 * 1.0 / r) * r # (96, 128, 192, 256)
        block2_channels = int(128 * 1.0 / r) * r
        block3_channels = int(192 * 1.0 / r) * r
        block4_channels = int(256 * 1.0 / r) * r
        # print(self.config.in_channels)
        self.feature_ops = nn.ModuleList([
            conv3x3_gn_relu(num_bands, block1_channels, r),

            repeat_block(block1_channels, r, 1),
            nn.Identity(),
            downsample2x(block1_channels, block2_channels),

            repeat_block(block2_channels, r, 1),
            nn.Identity(),
            downsample2x(block2_channels, block3_channels),

            repeat_block(block3_channels, r, 1),
            nn.Identity(),
            downsample2x(block3_channels, block4_channels),

            repeat_block(block4_channels, r, 1),
            nn.Identity(),
        ])
        inner_dim = int(num_dim * 1.0)
        self.reduce_1x1convs = nn.ModuleList([
            nn.Conv2d(block1_channels, inner_dim, 1),
            nn.Conv2d(block2_channels, inner_dim, 1),
            nn.Conv2d(block3_channels, inner_dim, 1),
            nn.Conv2d(block4_channels, inner_dim, 1),
        ])
        self.fuse_3x3convs = nn.ModuleList([
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
        ])
        self.cls_pred_conv = nn.Conv2d(inner_dim, num_classes, 1)  # 添加分类器
        self.time_= 0
        self.rgb_srf = FixedPosLinear(100, 3, minBand, maxBand, nBandDataset, dataset)
        self.first_shot = LyotFilter(100, num_first_shots, minBand, maxBand, nBandDataset, dataset)
        self.dynamic_shot = DynamicLyotFilter(num_first_shots, num_bands - num_first_shots, 100, minBand, maxBand, nBandDataset, dataset)
        self.seed = None

    def top_down(self, top, lateral):
        top2x = F.interpolate(top, scale_factor=2.0, mode='nearest')
        return lateral + top2x # 

    def forward(self, x, y=None, w=None, mode='train', **kwargs):
        # 先拍个RGB图像或者固定的3通道SRF拍出来的图片，然后根据此生成液态水晶的光谱响应曲线[num_bands, 3]额外生成3波段，与原来的的3波段一起进行后面的度量学习和分类网络
        x_hsi = x
        x = x.permute(0,2,3,1)
        x_hsi = x_hsi.permute(0,2,3,1)
        # x_rgb = self.rgb_srf(x) 
        first_shot = self.first_shot(x) 
        # x = x.permute(0,3,1,2)
        # x = x.permute(0,2,3,1)
        dynamic_shot = self.dynamic_shot(first_shot, x_hsi) 
        x = torch.cat((dynamic_shot, first_shot), dim=-1)
        x = x.permute(0,3,1,2) # [4, 6, 512, 512]
        if self.training:
            self.time_+= 1 

        # self.srf
        # x = x[:, [0, 30, 56, 67, 107, 138, 142, 206],:,:]
        feat_list = []
        for op in self.feature_ops:
            x = op(x)
            if isinstance(op, nn.Identity):
                feat_list.append(x)

        inner_feat_list = [self.reduce_1x1convs[i](feat) for i, feat in enumerate(feat_list)] # [#self.reduce_1x1convs[i](feat) for i, feat in enumerate(feat_list)]
        inner_feat_list.reverse()

        out_feat_list = [self.fuse_3x3convs[0](inner_feat_list[0])]
        for i in range(len(feat_list) - 1):
            inner = self.top_down(out_feat_list[i], inner_feat_list[i+1])
            out = self.fuse_3x3convs[i + 1](inner)
            out_feat_list.append(out)

        final_feat = out_feat_list[-1] # [4, 64, 512, 512]

        logit = self.cls_pred_conv(final_feat)  # 测试时输出 logits
        # 在测试阶段输出 logits，训练阶段输出 final_feat
        # metric_loss = metric_learning_loss(logit, y, w)
        # cls_loss = cross_entropy_loss(logit, y, w)

        if mode == 'train':
            return logit # metric_loss, 
        else:
            return  logit # 训练时输出特征 cls_loss,

class MetricLearningRGBNet(nn.Module):
    def __init__(self, num_bands, num_dim, num_classes, minBand, maxBand, nBandDataset, dataset):
        super(MetricLearningRGBNet, self).__init__()
        r = int(16 * 1.0)
        self.seed = None
        block1_channels = int(96 * 1.0 / r) * r # (96, 128, 192, 256)
        block2_channels = int(128 * 1.0 / r) * r
        block3_channels = int(192 * 1.0 / r) * r
        block4_channels = int(256 * 1.0 / r) * r
        # print(self.config.in_channels)
        self.feature_ops = nn.ModuleList([
            conv3x3_gn_relu(num_bands, block1_channels, r),

            repeat_block(block1_channels, r, 1),
            nn.Identity(),
            downsample2x(block1_channels, block2_channels),

            repeat_block(block2_channels, r, 1),
            nn.Identity(),
            downsample2x(block2_channels, block3_channels),

            repeat_block(block3_channels, r, 1),
            nn.Identity(),
            downsample2x(block3_channels, block4_channels),

            repeat_block(block4_channels, r, 1),
            nn.Identity(),
        ])
        inner_dim = int(num_dim * 1.0)
        self.reduce_1x1convs = nn.ModuleList([
            nn.Conv2d(block1_channels, inner_dim, 1),
            nn.Conv2d(block2_channels, inner_dim, 1),
            nn.Conv2d(block3_channels, inner_dim, 1),
            nn.Conv2d(block4_channels, inner_dim, 1),
        ])
        self.fuse_3x3convs = nn.ModuleList([
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
        ])
        self.cls_pred_conv = nn.Conv2d(inner_dim, num_classes, 1)  # 添加分类器
        self.time_= 0
        self.rgb_srf = FixedPosLinear(100, 3, minBand, maxBand, nBandDataset, dataset)
        # self.BS = PCA(n_components=self.config.hidden_channels)

    def top_down(self, top, lateral):
        top2x = F.interpolate(top, scale_factor=2.0, mode='nearest')
        return lateral + top2x # 

    def forward(self, x, y=None, w=None, mode='train', **kwargs):
        x = x.permute(0,2,3,1)
        x = self.rgb_srf(x) 
        x = x.permute(0,3,1,2)

        # # 去掉 batch 维度，形状变为 (3, 512, 512)
        # image_tensor = x.squeeze(0)  # 去掉第 0 维

        # # 将张量转换为 PIL 图片
        # to_pil = ToPILImage()
        # image = to_pil(image_tensor)

        # # 保存为 RGB 图片
        # image.save("./RGB/Augsburg_RGB.png")
        self.time_+= 1 # torch.Size([1, 270, 288, 400])
        # self.srf
        # x = x[:, [0, 30, 56, 67, 107, 138, 142, 206],:,:]
        feat_list = []
        for op in self.feature_ops:
            x = op(x)
            if isinstance(op, nn.Identity):
                feat_list.append(x)

        inner_feat_list = [self.reduce_1x1convs[i](feat) for i, feat in enumerate(feat_list)] # [#self.reduce_1x1convs[i](feat) for i, feat in enumerate(feat_list)]
        inner_feat_list.reverse()

        out_feat_list = [self.fuse_3x3convs[0](inner_feat_list[0])]
        for i in range(len(feat_list) - 1):
            inner = self.top_down(out_feat_list[i], inner_feat_list[i+1])
            out = self.fuse_3x3convs[i + 1](inner)
            out_feat_list.append(out)

        final_feat = out_feat_list[-1]

        logit = self.cls_pred_conv(final_feat)  # 测试时输出 logits
        # 在测试阶段输出 logits，训练阶段输出 final_feat
        if mode == 'test':
            return cross_entropy_loss(logit, y, w), logit
        else:
            return final_feat, logit # 训练时输出特征

l1loss1 = nn.L1Loss(reduction='mean')

def visualize_hyperspectral_reconstruction(image, time):
    """
    使用GridSpec精确控制子图布局，消除空隙
    """
    image = image[0]
    num_bands = image.shape[0]
    rows, cols = 2, 10
    
    # 获取图像尺寸
    height, width = image.shape[1], image.shape[2]
    
    # 计算图形尺寸
    dpi = 100
    fig_width = cols * width / dpi
    fig_height = rows * height / dpi
    
    # 使用GridSpec创建图形
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    
    # 创建GridSpec，设置子图间距为0
    gs = plt.GridSpec(rows, cols, 
                      left=0, right=1, 
                      top=1, bottom=0, 
                      wspace=0, hspace=0)
    
    for i in range(num_bands):
        row = i // cols
        col = i % cols
        
        ax = fig.add_subplot(gs[row, col])
        
        # 处理图像
        band_img = image[i, :, :]
        band_min = band_img.min()
        band_max = band_img.max()
        
        if torch.isclose(band_max, band_min, atol=1e-6):
            normalized_img = torch.zeros_like(band_img)
        else:
            normalized_img = (band_img - band_min) / (band_max - band_min)
        
        # 显示图像
        ax.imshow(normalized_img.detach().cpu().numpy(), 
                  cmap='gray', 
                  aspect='auto')
        
        # 完全关闭坐标轴
        ax.set_axis_off()
    
    # 保存图片
    plt.savefig('rec_gridspec_' + str(time) + '.jpg', 
                bbox_inches='tight', 
                pad_inches=0, 
                dpi=dpi)
    plt.close(fig)
    
    return fig

def visualize_classification_with_custom_colormap(logits, name):
    """
    使用自定义颜色映射可视化分类结果
    
    参数:
        logits: 形状为[1, num_classes, height, width]的张量
        class_colors: 可选，每个类别的RGB颜色列表
    """
    cmap2 = ListedColormap([
    # '#000000',  # 类别0 - 黑色
    '#FF0000',  # 类别1 - 红色
    '#00FF00',  # 类别2 - 绿色
    '#0000FF',  # 类别3 - 蓝色
    '#FFFF00',  # 类别4 - 黄色
    '#FF00FF',  # 类别5 - 洋红色
    '#00FFFF',  # 类别6 - 青色
    '#800080',  # 类别7 - 紫色
    '#808080',  # 类别8 - 灰色
    # '#FFA500',  # 类别9 - 橙色
    ])

    plt.figure()
    logits = logits.squeeze(0).argmax(dim=0).detach().cpu() + 1
    plt.imshow(logits, cmap=cmap2, interpolation='nearest')
        
        # 删除x轴和y轴坐标
    plt.xticks([])
    plt.yticks([])

    plt.savefig('classification_result_' + str(name) + '.jpg')
    # plt.show()
    
    return 


class MetricLearningDynamicRecLCTFNet(nn.Module): # This is the most updated in 10/12/2025.
    def __init__(self, num_first_shots, num_second_shots, num_bands, num_dim, num_classes, minBand, maxBand, nBandDataset, dataset, rec_bands):
        super(MetricLearningDynamicRecLCTFNet, self).__init__()
        self.num_first_shots = num_first_shots
        self.num_second_shots = num_second_shots
        r = int(16 * 1.0)
        block1_channels = int(96 * 1.0 / r) * r # (96, 128, 192, 256)
        block2_channels = int(128 * 1.0 / r) * r
        block3_channels = int(192 * 1.0 / r) * r
        block4_channels = int(256 * 1.0 / r) * r
        # print(self.config.in_channels)
        self.turn = False
        self.feature_ops = nn.ModuleList([
            conv3x3_gn_relu(rec_bands, block1_channels, r),

            repeat_block(block1_channels, r, 1),
            nn.Identity(),
            downsample2x(block1_channels, block2_channels),

            repeat_block(block2_channels, r, 1),
            nn.Identity(),
            downsample2x(block2_channels, block3_channels),

            repeat_block(block3_channels, r, 1),
            nn.Identity(),
            downsample2x(block3_channels, block4_channels),

            repeat_block(block4_channels, r, 1),
            nn.Identity(),
        ])
        
        inner_dim = int(num_dim * 1.0)
        self.reduce_1x1convs = nn.ModuleList([
            nn.Conv2d(block1_channels, inner_dim, 1),
            nn.Conv2d(block2_channels, inner_dim, 1),
            nn.Conv2d(block3_channels, inner_dim, 1),
            nn.Conv2d(block4_channels, inner_dim, 1),
        ])
        self.fuse_3x3convs = nn.ModuleList([
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
        ])
        self.cls_pred_conv = nn.Conv2d(inner_dim, num_classes, 1)  # 添加分类器
        self.time_ = 0
        self.rgb_srf = FixedPosLinear(100, 3, minBand, maxBand, nBandDataset, dataset)
        self.first_shot = LyotFilter(100, num_first_shots, minBand, maxBand, nBandDataset, dataset)
        # self.full_band_srf2 = LyotFilter(num_first_shots, num_second_shots, minBand, maxBand, num_first_shots, dataset)
        self.dynamic_srf = DynamicLyotFilter(num_first_shots, num_bands - num_first_shots, 100, minBand, maxBand, nBandDataset, dataset)
        self.reconstruction = MST_Plus_Plus(num_bands, rec_bands,  rec_bands, 3)
        self.seed = None
        self.l1_loss_l = []
        self.cls_loss_l = []
        

    def top_down(self, top, lateral):
        top2x = F.interpolate(top, scale_factor=2.0, mode='nearest')
        return lateral + top2x # 

    def forward(self, x, y=None, w=None, mode='train', **kwargs):
        # 先拍个RGB图像或者固定的3通道SRF拍出来的图片，然后根据此生成液态水晶的光谱响应曲线[num_bands, 3]额外生成3波段，与原来的的3波段一起进行后面的度量学习和分类网络
        
        # if self.training:
        #     x.requires_grad_(True)
        x_hsi = x
        x_hsi_bs = x_hsi[:,[0, 5, 10, 15, 20, 26, 31, 36, 41, 46, 52, 57, 62, 67, 72, 78, 83, 88, 93, 99], :, :]
        # 1. 检查输入x的梯度状态
        # print(f"1. 输入x requires_grad: {x.requires_grad}")
        # print(f"2. first_shot grad_fn: {x.grad_fn}")
        
        x = x.permute(0,2,3,1)
        x_hsi = x_hsi.permute(0,2,3,1)
        
        # 2. 检查first_shot的梯度
        first_shot = self.first_shot(x)
        
        # 3. 检查dynamic_srf的梯度
        dynamic_shot = self.dynamic_srf(first_shot, x_hsi)
        
        x = torch.cat((first_shot, dynamic_shot), dim=-1)
        
        x = x.permute(0,3,1,2)
        # 5. 检查reconstruction输入
        
        # 6. 检查reconstruction网络参数状态
        # start_time = time.time()
        x = self.reconstruction(x)
        # end_time = time.time()
        # execution_time = end_time - start_time
        # print(f"Reconstruction time: {execution_time:.6f} seconds")
        x_rec = x
        
        self.time_ += 1 
        # self.srf
        # x = x[:, [0, 30, 56, 67, 107, 138, 142, 206],:,:]
        feat_list = []
        for op in self.feature_ops:
            x = op(x)
            if isinstance(op, nn.Identity):
                feat_list.append(x)

        inner_feat_list = [self.reduce_1x1convs[i](feat) for i, feat in enumerate(feat_list)] # [#self.reduce_1x1convs[i](feat) for i, feat in enumerate(feat_list)]
        inner_feat_list.reverse()

        out_feat_list = [self.fuse_3x3convs[0](inner_feat_list[0])]
        for i in range(len(feat_list) - 1):
            inner = self.top_down(out_feat_list[i], inner_feat_list[i+1])
            out = self.fuse_3x3convs[i + 1](inner)
            out_feat_list.append(out)

        final_feat = out_feat_list[-1] # [4, 64, 512, 512]

        # 在测试阶段输出 logits，训练阶段输出 final_feat
        # if mode == 'test':
        # else:
        #     x_hsi_bs = x_hsi[:,list(map(int, np.linspace(0, 99, 20))), :, :]
        logit = self.cls_pred_conv(final_feat)  # 测试时输出 logits
        # if mode == 'test' and 0 <= self.time_ <= 50:
        #     visualize_classification_with_custom_colormap(logit, self.time_)
        
        metric_loss = metric_learning_loss(logit, y, w)
        cls_loss = cross_entropy_loss(logit, y, w)
        rec_loss = l1loss1(x_hsi_bs, x_rec)
        # total_loss = cross_entropy_loss(logit, y, w)
        self.l1_loss_l.append(rec_loss.item())
        self.cls_loss_l.append(cls_loss.item())

        
            

                
        # else:
        if self.time_ % 10 == 0 and mode == 'test':
            visualize_hyperspectral_reconstruction(x_rec, self.time_)
        #     print('PSNR:' + str(PSNR(x_hsi_bs, x_rec)))
        #     print('SSIM:' + str(SSIM(x_hsi_bs, x_rec)))     

        # total_loss =  rec_loss # + self.loss(logit, y, w) , # + abs(20 - PSNR(x_hsi, x_rec)), # 
        # l1_loss = l1loss1(x_hsi_bs, x_rec)
        # total_loss = cls_loss / 5.0 + l1_loss / 10.0
        # if not self.training:
        # if self.time_ > 800:
        if self.time_ == 1000:
            print('ok')
            plt.figure()
            if mode == 'test':
                loss_show = np.vstack((np.array(self.cls_loss_l)))
            else:
                loss_show = np.vstack((np.array(self.l1_loss_l)))
            plt.axhline(y=0.013, color='r', linestyle='--')
            plt.plot(np.arange(0, 1000, 1), loss_show)
            plt.savefig(str(self.seed) + '_2losses_' + str(mode) + '_' + str(self.time_) + '.jpg')

        # if self.time_ == 998:
        #     print('ok')
        #     plt.figure()
        #     loss_show = np.vstack((np.array(self.l1_loss_l), np.array(self.cls_loss_l)))
        #     plt.axhline(y=0.013, color='r', linestyle='--')
        #     plt.plot(np.arange(0, 998, 1), loss_show.T)
        #     plt.savefig(str(self.seed) + '_2losses_' + str(mode) + '_' + str(self.time_) + '.jpg')

        if mode == 'train':
            total_loss = metric_loss + rec_loss
            return logit # total_loss, 
        else:
            total_loss = cls_loss
            return logit # total_loss, 
        # else:
        #     return final_feat  # 训练时输出特征

def PSNR(original, compressed):
    mse = torch.mean((original - compressed) ** 2)
    if(mse == 0): # MSE is zero means no noise is present in the signal .                   
    # Therefore PSNR have no importance.         
        return 100
    max_pixel = torch.max(original)
    psnr = 10 * log10(max_pixel ** 2 / mse)
    return psnr  
  
def cross_entropy_loss(x, y, weight):
    losses = F.cross_entropy(x, y.long() - 1, weight=None,
                                 ignore_index=-1, reduction='none')

    v = losses.mul_(weight).sum() / weight.sum()
    return v

def metric_learning_loss(final_feat, y, w, margin=1.0):
    """
    计算基于对比损失的度量学习损失

    参数:
        final_feat: Tensor, [batchsize, num_dim, h, w], 每个像素的特征向量
        y: Tensor, [batchsize, h, w], 每个像素的标签
        w: Tensor, [batchsize, h, w], 标记选中的像素点 (1 表示选中, 0 表示未选中)
        margin: float, 不同类别像素之间的最小距离

    返回:
        loss: Tensor, 平均损失
    """
    batch_size, num_dim, height, width = final_feat.shape
    
    # 变换形状
    final_feat = final_feat.view(batch_size, num_dim, -1)  # [batchsize, num_dim, h * w]
    y = y.view(batch_size, -1)  # [batchsize, h * w]
    w = w.view(batch_size, -1)  # [batchsize, h * w]
    
    # 初始化总损失
    total_loss = 0.0
    
    # 遍历每个 batch
    for i in range(batch_size):
        # 选择有效像素（w == 1 的位置）
        mask = w[i].bool()
        selected_feat = final_feat[i][:, mask]  # [num_dim, selected_pixels]
        selected_y = y[i][mask]  # [selected_pixels]
        
        if selected_feat.shape[1] < 2:  # 如果选中的像素点少于2个，则跳过
            continue
        
        # 计算选中像素之间的 L2 距离
        dist_matrix = torch.cdist(selected_feat.T.unsqueeze(0), selected_feat.T.unsqueeze(0), p=2).squeeze(0)  # [selected_pixels, selected_pixels]
        
        # 构建标签距离矩阵
        same_label_mask = (selected_y.unsqueeze(1) == selected_y.unsqueeze(0)).float()  # [selected_pixels, selected_pixels]
        diff_label_mask = 1 - same_label_mask
        
        # 计算同标签损失 (最小化同类像素点之间的距离)
        same_loss = same_label_mask * torch.pow(dist_matrix, 2)
        
        # 计算不同标签损失 (最大化不同类像素点之间的距离)
        diff_loss = diff_label_mask * torch.pow(torch.clamp(margin - dist_matrix, min=0.0), 2)
        
        # 计算总损失
        loss = torch.sum(same_loss + diff_loss)
        total_loss += loss / (selected_feat.shape[1] * selected_feat.shape[1])  # 归一化损失
    
    # 返回平均损失
    return total_loss / batch_size

class VoltageNet4(nn.Module):
    def __init__(self, in_dim=3, out_dim=3):
        super(VoltageNet4, self).__init__()
        
        # 单层卷积 + 全局池化
        self.conv = nn.Conv2d(in_dim, 64, 3, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        
        # 单层全连接
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x):
        # 输入: [batch, h, w, bands]
        x = x.permute(0, 3, 1, 2)  # 维度重排
        
        # 特征提取
        x = torch.relu(self.bn(self.conv(x)))  # 单层卷积
        x = self.pool(x)  # 全局池化
        x = x.view(x.size(0), -1)  # 展平
        
        # 电压生成 (0-10范围)
        voltage = torch.sigmoid(self.fc(x)) * 10
        
        return voltage
    
class VoltageNet0(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(VoltageNet0, self).__init__()
        
        # 简化卷积层设计：使用更少的层但保持特征提取能力
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_dim, 64, 3, padding=1),  # 直接映射到更高维度
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # 全局平均池化替代自适应池化，更简洁
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 简化的全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, out_dim)
        )

    def forward(self, x):
        # 输入形状: [batch_size, h, w, band]
        x = x.permute(0, 3, 1, 2)  # 调整为 [batch_size, band, h, w]
        
        # 简化卷积流程
        x = self.conv_layers(x)  # [batch_size, 128, h, w]
        
        # 全局平均池化
        x = self.global_pool(x)  # [batch_size, 128, 1, 1]
        x = x.view(x.size(0), -1)  # [batch_size, 128]
        
        # 全连接层生成新电压
        voltage = torch.sigmoid(self.fc_layers(x)) * 10  # [batch_size, out_dim]
        
        return voltage
    
class VoltageNet2(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(VoltageNet2, self).__init__()

        # 定义卷积层，增加深度并引入批量归一化
        self.conv1 = nn.Conv2d(in_channels=in_dim, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # 自适应池化层，保留更多的空间信息
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))  # 输出大小为 [batch_size, 128, 2, 2]

        # 定义全连接层
        self.fc1 = nn.Linear(128 * 2 * 2, 128)  # 自适应池化后展平的大小是 128 * 2 * 2
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, out_dim)

        # Dropout 层，防止过拟合
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 输入形状: [batch_size, h, w, band]
        x = x.permute(0, 3, 1, 2)  # torch.Size([1, 1376, 256, 3])

        # 卷积 + 批量归一化 + 激活函数
        x = F.relu(self.bn1(self.conv1(x)))  # [batch_size, 32, h, w]
        x = F.relu(self.bn2(self.conv2(x)))  # [batch_size, 64, h, w]
        x = F.relu(self.bn3(self.conv3(x)))  # [batch_size, 128, h, w]

        # 自适应池化，将特征图调整为固定大小 (2, 2)
        x = self.adaptive_pool(x)  # [batch_size, 128, 2, 2]

        # 将卷积特征展平
        x = x.reshape(x.size(0), -1)  # [batch_size, 128 * 2 * 2]

        # 全连接层 + Dropout + 激活函数
        x = F.relu(self.fc1(x))    # [batch_size, 128]
        x = self.dropout(x)        # 添加 Dropout，防止过拟合
        x = F.relu(self.fc2(x))    # [batch_size, 64]

        # 输出层，使用 sigmoid 激活将输出限制在 [0, 10]
        output = torch.sigmoid(self.fc3(x)) * 10  # [batch_size, 3]

        return output # torch.Size([1, 3])
    
class VoltageNet1(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(VoltageNet1, self).__init__()

        # 定义卷积层
        self.conv1 = nn.Conv2d(in_channels=in_dim, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=in_dim, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        # 定义自适应池化层
        self.adaptive_pool = nn.AdaptiveMaxPool2d((1, 1))  # 将输出调整为 [batch_size, channels, 4, 4]

        # 定义全连接层
        # self.fc1 = nn.Linear(128 * 1 * 1, 128)  # 4*4是自适应池化后的固定大小
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, out_dim)

    def forward(self, x):
        # x 的形状: [batch_size, patch, patch, band]
        x = x.permute(0, 3, 1, 2)  # 调整维度顺序为 [batch_size, band, patch, patch]

        # 卷积 + ReLU 
        # x = F.relu(self.conv1(x))  # [batch_size, 32, patch, patch]
        # x = F.relu(self.conv2(x))  # [batch_size, 64, patch, patch]
        x = F.relu(self.conv3(x))  # [batch_size, 128, patch, patch]
        
        # 自适应池化，将卷积特征图大小调整为固定的4x4
        x = self.adaptive_pool(x)  # [batch_size, 128, 4, 4]

        # 将卷积特征展平
        x = x.reshape(x.size(0), -1)  # [batch_size, 128 * 4 * 4]
        
        # 全连接层
        # x = F.relu(self.fc1(x))  # [batch_size, 128]
        x = F.relu(self.fc2(x))  # [batch_size, 64]
        
        # 输出层，使用 sigmoid 激活函数将输出限制在 [0, 1]，再乘以 10 得到 [0, 10] 的值
        output = torch.sigmoid(self.fc3(x)) * 10  # [batch_size, 3] output = torch.sigmoid(self.fc3(x)) * 10
        
        return output

class VoltageNet3(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(VoltageNet3, self).__init__()
        self.num_voltage = in_dim
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(in_channels=in_dim, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # 动态通道选择层
        self.gate_conv = nn.Linear(64, 1)  # 用于动态选择通道的全连接层
        
        # 全局平均池化层
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 动态的全连接层
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, out_dim)

    def forward(self, x):
        # 调整输入维度为 [batchsize, band, h, w] -> [batchsize, h, w, band] -> [batchsize, band, h, w]
        x = x.permute(0, 3, 1, 2)  # 调整通道维度到第二维度
        
        # 1. 卷积操作
        x = F.relu(self.conv1(x))  # [batchsize, 32, h, w]
        x = F.max_pool2d(x, kernel_size=2)  # [batchsize, 32, h/2, w/2]
        x = F.relu(self.conv2(x))  # [batchsize, 64, h/2, w/2]
        
        # 2. 动态通道选择
        # 全局平均池化
        avg_features = self.global_avg_pool(x).view(x.size(0), -1)  # [batchsize, 64]
        # 使用 gate_conv 生成通道权重
        gate_weights = torch.sigmoid(self.gate_conv(avg_features))  # [batchsize, 1]
        gate_weights = gate_weights.view(-1, 1, 1, 1)  # [batchsize, 1, 1, 1]
        # 根据 gate_weights 调整卷积特征
        x = x * gate_weights  # [batchsize, 64, h/2, w/2]
        
        # 3. 全局平均池化后进入全连接层
        x = self.global_avg_pool(x).view(x.size(0), -1)  # [batchsize, 64]
        x = F.relu(self.fc1(x))  # [batchsize, 128]
        
        # 输出电压值，并使用 softplus 确保电压值为正
        voltages = F.softplus(self.fc2(x))  # [batchsize, num_voltage] 确保输出为正值
        
        return voltages

class VoltageNet5(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, low_quantile=0.05, high_quantile=0.95):
        super(VoltageNet5, self).__init__()
        self.low_quantile = low_quantile
        self.high_quantile = high_quantile
        self.conv = nn.Conv2d(in_dim, 64, 3, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, out_dim + 2)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = torch.relu(self.bn(self.conv(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        raw_output = self.fc(x)

        min_vals = raw_output.min(dim=1, keepdim=True)[0]
        max_vals = raw_output.max(dim=1, keepdim=True)[0]
        normalized = (raw_output - min_vals) / (max_vals - min_vals + 1e-8)
        sorted_values, _ = torch.sort(normalized, dim=1)
        middle_values = sorted_values[:, 1:-1]  # 去掉首尾
        
        voltage_output = middle_values * 10
        return voltage_output
    
class DynamicLyotFilter(nn.Module):
    def __init__(self, in_dim, out_dim, hsi_dim,  minBand, maxBand, nBandDataset, dataset):
        super(DynamicLyotFilter, self).__init__()
        # self.voltages = nn.Parameter(10*torch.rand((in_dim)))
        self.in_dim = in_dim
        self.hsi_dim = hsi_dim
        self.out_dim = out_dim

        self.minBand = minBand
        self.maxBand = maxBand
        self.nBandDataset = nBandDataset
        self.dataset = dataset
        # Define the small network to generate voltages from input x
        self.voltage_net = VoltageNet5(in_dim=in_dim, out_dim=out_dim)  # 输入为平展后的图像
        self.voltages = 0


    def forward(self, x, x_hsi):
        # self.weight = self.weight.cpu()
        # tmp = 0atch_size, patch * patch * band]
        batch_size, h, w, band_size = x.shape
        # 通过 VoltageNet 生成电压 [batch_size, 3]
        voltages = self.voltage_net(x)  # [batch_size, 3]
        self.voltages = voltages
        # print(voltages[:2,:])
        lctf_srf = torch.rand((batch_size, self.out_dim, self.hsi_dim)).cuda()# srf
        # print(self.voltages) # voltage
        tmp = 0 # 尽量别让反传时找不到梯度，因此不要detach,转到cpu之类的;对weight以一个整体操作，不要分开操作

		# IP

        


      
            # for i in range(batch_size):
            #     for n in range(self.nBandDataset):
            #         band = self.minBand + (self.maxBand - self.minBand) * n / self.nBandDataset
            #         lctf_srf[i, :,n - tmp] = (0.5 - 0.5*torch.cos(2*m.pi*(-0.01)*(pow(voltages[i, :],-1))/(band*pow(10,-6))))
        for i in range(batch_size):
                # 生成 n 的张量
            n_indices = torch.arange(self.nBandDataset).cuda()

                # 计算 band 张量，向量化 n 维度
            band_tensor = self.minBand + (self.maxBand - self.minBand) * n_indices / self.nBandDataset

                # 计算 voltage 的倒数，向量化
            voltage_inv = voltages[i, :].pow(-1).repeat(self.hsi_dim, 1).transpose(0,1)  # voltage_inv 的形状为 [nBandDataset,]

                # 计算公式中的 cos 部分，向量化
            cos_part = torch.cos(2 * m.pi * (-0.01) * voltage_inv)

                # 计算最终结果，使用广播机制
            lctf_srf[i, :, n_indices - tmp] = (0.5 - 0.5 * cos_part) / (band_tensor.repeat(self.out_dim, 1) * pow(10, -6))

        return torch.einsum('bca,bpqa->bpqc', lctf_srf, x_hsi)
# def compute_loss(final_feat, y, w):
#     batch_size, num_dim, height, width = final_feat.shape
    
#     # 变换形状
#     final_feat = final_feat.view(batch_size, num_dim, -1)
#     y = y.view(batch_size, -1)
#     w = w.view(batch_size, -1)
    
#     # 检查 NaN
#     assert not torch.isnan(final_feat).any(), "NaN found in final_feat after reshape"
#     assert not torch.isnan(y).any(), "NaN found in y after reshape"
#     assert not torch.isnan(w).any(), "NaN found in w after reshape"

#     # 有效 mask
#     valid_masks = w.sum(dim=-1) >= 2
#     if not valid_masks.any():
#         return torch.tensor(0.0, device=final_feat.device)

#     # 选择的索引
#     selected_indices = w.bool()
#     selected_feat = final_feat.permute(0, 2, 1)[selected_indices]
#     selected_y = y[selected_indices]

#     # 检查 NaN
#     assert not torch.isnan(selected_feat).any(), "NaN found in selected_feat"
#     assert not torch.isnan(selected_y).any(), "NaN found in selected_y"

#     # 计算距离矩阵
#     dist_matrix = torch.cdist(selected_feat.unsqueeze(0), selected_feat.unsqueeze(0), p=2) ** 2
    
#     # 检查 NaN
#     assert not torch.isnan(dist_matrix).any(), "NaN found in dist_matrix"

#     # 计算 sigma_pq
#     sigma_pq = 2 / (1 + torch.exp(dist_matrix))
    
#     # 检查 NaN
#     assert not torch.isnan(sigma_pq).any(), "NaN found in sigma_pq"

#     # 计算实例大小
#     instance_sizes = torch.bincount(selected_y.long(), minlength=int(y.max().item() + 1)).float()
    
#     # 检查 NaN
#     assert not torch.isnan(instance_sizes).any(), "NaN found in instance_sizes"

#     inv_instance_sizes = 1.0 / (instance_sizes[selected_y.long()] + 1e-8)
    
#     # 检查 NaN
#     assert not torch.isnan(inv_instance_sizes).any(), "NaN found in inv_instance_sizes"

#     # 计算权重
#     w_pq = inv_instance_sizes.unsqueeze(1) * inv_instance_sizes.unsqueeze(0)
#     w_pq /= w_pq.sum()
    
#     # 检查 NaN
#     assert not torch.isnan(w_pq).any(), "NaN found in w_pq"

#     # 同标签和不同标签的掩码
#     same_label_mask = (selected_y.unsqueeze(1) == selected_y.unsqueeze(0)).float()
#     diff_label_mask = 1 - same_label_mask
    
#     # 计算相同标签和不同标签的损失
#     same_loss = same_label_mask * torch.log(sigma_pq + 1e-8)
#     diff_loss = diff_label_mask * torch.log(1 - sigma_pq + 1e-8)
    
#     # 检查 NaN
#     assert not torch.isnan(same_loss).any(), "NaN found in same_loss"
#     assert not torch.isnan(diff_loss).any(), "NaN found in diff_loss"

#     # 计算总损失
#     loss = -torch.sum(w_pq * (same_loss + diff_loss))
    
#     # 检查 NaN
#     assert not torch.isnan(loss).any(), "NaN found in loss"

#     return loss / batch_size

# def compute_loss(final_feat, y, w):
#     batch_size, num_dim, height, width = final_feat.shape
    
#     final_feat = final_feat.view(batch_size, num_dim, -1)
#     y = y.view(batch_size, -1)
#     w = w.view(batch_size, -1)
    
#     # 检查 NaN
#     assert not torch.isnan(final_feat).any(), "NaN found in final_feat after reshape"
#     assert not torch.isnan(y).any(), "NaN found in y after reshape"
#     assert not torch.isnan(w).any(), "NaN found in w after reshape"

#     # 有效 mask
#     valid_masks = w.sum(dim=-1) >= 2
#     if not valid_masks.any():
#         return torch.tensor(0.0, device=final_feat.device)

#     # 选择的索引
#     selected_indices = w.bool()
#     selected_feat = final_feat.permute(0, 2, 1)[selected_indices]
#     selected_y = y[selected_indices]

#     # 检查 NaN
#     assert not torch.isnan(selected_feat).any(), "NaN found in selected_feat"
#     assert not torch.isnan(selected_y).any(), "NaN found in selected_y"

#     # 计算距离矩阵并添加数值稳定性
#     dist_matrix = torch.cdist(selected_feat.unsqueeze(0), selected_feat.unsqueeze(0), p=2) ** 2
#     dist_matrix = torch.clamp(dist_matrix, min=1e-8, max=1e8)  # 限制距离矩阵的范围，避免数值过大或过小
    
#     # 检查 NaN
#     assert not torch.isnan(dist_matrix).any(), "NaN found in dist_matrix"

#     # 计算 sigma_pq
#     sigma_pq = 2 / (1 + torch.exp(dist_matrix - dist_matrix.max()))  # 稳定数值，避免 exp() 的大值溢出
    
#     # 检查 NaN
#     assert not torch.isnan(sigma_pq).any(), "NaN found in sigma_pq"

#     # 计算实例大小
#     instance_sizes = torch.bincount(selected_y.long(), minlength=int(y.max().item() + 1)).float()
#     instance_sizes = torch.clamp(instance_sizes, min=1e-8)  # 避免 instance_sizes 过小导致 inf
    
#     # 检查 NaN
#     assert not torch.isnan(instance_sizes).any(), "NaN found in instance_sizes"

#     inv_instance_sizes = 1.0 / (instance_sizes[selected_y.long()] + 1e-8)
    
#     # 检查 NaN
#     assert not torch.isnan(inv_instance_sizes).any(), "NaN found in inv_instance_sizes"

#     # 计算权重 w_pq
#     w_pq = inv_instance_sizes.unsqueeze(1) * inv_instance_sizes.unsqueeze(0)
#     w_pq /= w_pq.sum() + 1e-8  # 防止除以零的情况
    
#     # 检查 NaN
#     assert not torch.isnan(w_pq).any(), "NaN found in w_pq"

#     # 同标签和不同标签的掩码
#     same_label_mask = (selected_y.unsqueeze(1) == selected_y.unsqueeze(0)).float()
#     diff_label_mask = 1 - same_label_mask
    
#     # 计算相同标签和不同标签的损失
#     same_loss = same_label_mask * torch.log(sigma_pq + 1e-8)
#     diff_loss = diff_label_mask * torch.log(1 - sigma_pq + 1e-8)
    
#     # 检查 NaN
#     assert not torch.isnan(same_loss).any(), "NaN found in same_loss"
#     assert not torch.isnan(diff_loss).any(), "NaN found in diff_loss"

#     # 计算总损失
#     loss = -torch.sum(w_pq * (same_loss + diff_loss))
    
#     # 检查 NaN
#     assert not torch.isnan(loss).any(), "NaN found in loss"

#     return loss / batch_size
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

def plot_tsne_with_mask(final_feat, y, w, num_samples=1000, save_path=None):
    """
    使用 t-SNE 对嵌入的特征进行降维，并绘制 2D t-SNE 图，基于掩码 w 选择像素点。
    Args:
        final_feat (torch.Tensor): 嵌入的特征向量，形状为 [batch_size, num_dim, h, w]。
        y (torch.Tensor): 分割标签，形状为 [batch_size, h, w]，值为 0 到 C 之间的整数。
        w (torch.Tensor): 选择的掩码，形状为 [batch_size, h, w]，值为 0 或 1 表示是否选择对应像素。
        num_samples (int): 采样的像素点数量，默认 1000。
        save_path (str): 保存图片的路径（例如 'output/tsne_plot.png'），默认 None 不保存。
    """
    batch_size, num_dim, height, width = final_feat.shape

    for i in range(batch_size):
        # 展开每个 batch 的特征向量、标签和掩码，形状变为 [h * w, num_dim] 和 [h * w]
        final_feat_single = final_feat[i].view(num_dim, -1).permute(1, 0).cpu().detach().numpy()  # [h * w, num_dim]
        y_single = y[i].view(-1).cpu().numpy()  # [h * w]
        w_single = w[i].view(-1).cpu().numpy()  # [h * w]

        # 只选择掩码为 1 的像素点
        selected_indices = np.where(w_single == 1)[0]

        if len(selected_indices) == 0:
            print(f"第 {i + 1} 张图没有选择的像素点")
            continue

        # 选择特征和标签
        final_feat_selected = final_feat_single[selected_indices]
        y_selected = y_single[selected_indices]

        # 随机选择 num_samples 个点进行 t-SNE（如果选择的点超过 num_samples）
        if num_samples < len(final_feat_selected):
            indices = np.random.choice(len(final_feat_selected), num_samples, replace=False)
            feat_sampled = final_feat_selected[indices]
            y_sampled = y_selected[indices]
        else:
            feat_sampled = final_feat_selected
            y_sampled = y_selected

        # 使用 t-SNE 降维到 2D
        tsne = TSNE(n_components=2, random_state=42)
        feat_tsne = tsne.fit_transform(feat_sampled)

        # 使用 seaborn 绘制 t-SNE 图
        plt.figure(figsize=(10, 8))
        palette = sns.color_palette("hsv", len(np.unique(y_sampled)))  # 为每个类别选择一种颜色
        sns.scatterplot(x=feat_tsne[:, 0], y=feat_tsne[:, 1], hue=y_sampled, palette=palette, legend='full', s=8)
        plt.title(f't-SNE of Feature Embeddings for Image {i + 1} (Using Mask w)')

        # 保存图像到指定路径
        if save_path:
            img_save_path = f"{save_path}_image_{i + 1}.png"
            plt.savefig(img_save_path, bbox_inches='tight')  # 保存图片
            print(f"t-SNE 图已保存至 {img_save_path}")
        else:
            plt.show()  # 显示图片

# def plot_tsne_with_mask(final_feat, y, w, num_samples=1000, save_path=None):
#     """
#         使用 t-SNE 对嵌入的特征进行降维，并绘制 2D t-SNE 图，基于掩码 w 选择像素点。
#         Args:
#             final_feat (torch.Tensor): 嵌入的特征向量，形状为 [batch_size, num_dim, h, w]。
#             y (torch.Tensor): 分割标签，形状为 [batch_size, h, w]，值为 0 到 C 之间的整数。
#             w (torch.Tensor): 选择的掩码，形状为 [batch_size, h, w]，值为 0 或 1 表示是否选择对应像素。
#             num_samples (int): 采样的像素点数量，默认 1000。
#             save_path (str): 保存图片的路径（例如 'output/tsne_plot.png'），默认 None 不保存。
#     """
#     batch_size, num_dim, height, width = final_feat.shape

#         # 展开特征向量、标签和掩码，形状变为 [batch_size * h * w, num_dim] 和 [batch_size * h * w]
#     final_feat = final_feat.view(batch_size, num_dim, -1).permute(0, 2, 1).contiguous().view(-1, num_dim).cpu().detach().numpy()
#     y = y.view(-1).cpu().numpy()
#     w = w.view(-1).cpu().numpy()

#         # 只选择掩码为 1 的像素点
#     selected_indices = np.where(w == 1)[0]
        
#     if len(selected_indices) == 0:
#         print("没有选择的像素点")
#         return

#         # 选择特征和标签
#     final_feat_selected = final_feat[selected_indices]
#     y_selected = y[selected_indices]

#         # 随机选择 num_samples 个点进行 t-SNE（如果选择的点超过 num_samples）
#     if num_samples < len(final_feat_selected):
#         indices = np.random.choice(len(final_feat_selected), num_samples, replace=False)
#         feat_sampled = final_feat_selected[indices]
#         y_sampled = y_selected[indices]
#     else:
#         feat_sampled = final_feat_selected
#         y_sampled = y_selected

#         # 使用 t-SNE 降维到 2D
#     tsne = TSNE(n_components=2, random_state=42)
#     feat_tsne = tsne.fit_transform(feat_sampled)

#         # 使用 seaborn 绘制 t-SNE 图
#     plt.figure(figsize=(10, 8))
#     palette = sns.color_palette("hsv", len(np.unique(y_sampled)))  # 为每个类别选择一种颜色
#     sns.scatterplot(x=feat_tsne[:, 0], y=feat_tsne[:, 1], hue=y_sampled, palette=palette, legend='full', s=8)
#     plt.title('t-SNE of Feature Embeddings (Using Mask w)')

#         # 保存图像到指定路径
#     if save_path:
#         plt.savefig(save_path, bbox_inches='tight')  # 保存图片
#         print(f"t-SNE 图已保存至 {save_path}")

def generate_metric_features(net, test_data):
    net.eval()
    with torch.no_grad():
        metric_features = net(test_data)
    return metric_features

class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ClassificationHead, self).__init__()
        # 定义一个简单的全连接层作为分类头
        self.fc = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)

def nearest_neighbor_classification_with_svm(metric_features, reference_mask, labels, use_classification_head=False):
    """
    使用参考点掩码进行分类，基于提取的特征和 SVM 分类器，还可以选择性地在 SVM 上加一个分类头。
    
    :param metric_features: 特征图像, 形状 [1, num_band, H, W]
    :param reference_mask: 参考点的掩码, 形状 [1, H, W], 值为 0 和 1
    :param labels: 标签图像, 形状 [1, H, W]
    :param use_classification_head: 是否使用分类头
    :return: 预测的标签图, 形状 [H, W]
    """
    h, w = metric_features.shape[2], metric_features.shape[3]  # 获取图像的高度和宽度
    
    # 获取参考点的位置
    reference_indices = (reference_mask == 1).nonzero(as_tuple=False)
    
    # 提取参考点的特征和标签
    reference_points = []
    reference_labels = []
    for idx in reference_indices:
        # 获取参考点的特征
        reference_points.append(metric_features[:, :, idx[1], idx[2]].squeeze(0))  # [num_band]
        reference_labels.append(labels[0, idx[1], idx[2]])  # 对应的标签
    
    # 将参考点的特征和标签转换为 numpy 数组
    reference_points = torch.stack(reference_points).cpu().numpy()  # [num_reference, num_band]
    reference_labels = torch.tensor(reference_labels).cpu().numpy()  # [num_reference]
    
    # 使用 SVM 进行分类训练
    svm_classifier = SVC(kernel='linear', probability=True)
    svm_classifier.fit(reference_points, reference_labels)
    
    # 将 metric_features 展平成 [H*W, num_band]，每个像素的特征向量
    pixel_features = metric_features.view(metric_features.shape[1], -1).T  # [H*W, num_band]
    pixel_features_numpy = pixel_features.cpu().numpy()  # 转换为 numpy，适配 SVM
    
    # 使用 SVM 对所有像素进行分类预测
    svm_predictions = svm_classifier.predict(pixel_features_numpy)  # [H*W]
    
    # 如果使用分类头，则将 SVM 的输出作为分类头的输入
    if use_classification_head:
        num_classes = len(set(reference_labels))
        input_dim = metric_features.shape[1]
        print('num_classes :' + str(num_classes))
        # 定义分类头
        classification_head = ClassificationHead(input_dim=input_dim, num_classes=num_classes)
        
        # 将分类头应用于所有像素特征
        pixel_features_torch = pixel_features  # 不需要转换回 numpy
        classification_logits = classification_head(pixel_features_torch)  # [H*W, num_classes]
        svm_predictions = torch.argmax(classification_logits, dim=1).cpu().numpy()  # [H*W]
    
    # 将预测的标签转换为 [H, W] 的形状
    predicted_labels = torch.tensor(svm_predictions).view(h, w)
    
    return predicted_labels
def nearest_neighbor_classification(metric_features, reference_mask, labels):
    """
    使用参考点掩码进行最近邻分类，基于度量学习中的距离计算方式。
    
    :param metric_features: 特征图像, 形状 [1, num_band, H, W]
    :param reference_mask: 参考点的掩码, 形状 [1, H, W], 值为 0 和 1
    :param labels: 标签图像, 形状 [1, H, W]
    :return: 预测的标签图, 形状 [H, W]
    """
    h, w = metric_features.shape[2], metric_features.shape[3]  # 获取图像的高度和宽度
    
    # 获取参考点的位置
    reference_indices = (reference_mask == 1).nonzero(as_tuple=False)
    
    # 提取参考点的特征和标签
    reference_points = []
    reference_labels = []
    for idx in reference_indices:
        # 获取参考点的特征
        reference_points.append(metric_features[:, :, idx[1], idx[2]].unsqueeze(0))  # 使用正确的索引
        reference_labels.append(labels[0, idx[1], idx[2]])  # 获取参考点的标签
    
    # 将参考点的特征和标签转换为张量
    reference_points = torch.cat(reference_points, dim=0)  # [num_reference, num_band]
    reference_labels = torch.tensor(reference_labels)  # [num_reference]
    
    # 将 metric_features 展平成 [H*W, num_band]，每个像素的特征向量
    pixel_features = metric_features.view(metric_features.shape[1], -1).T  # [H*W, num_band]
    
    # 获取所有唯一的标签
    unique_labels = reference_labels.unique()
    
    # 初始化一个张量来存储每个像素与每一类的平均距离（相似度）
    avg_similarity_matrix = torch.zeros(len(unique_labels), pixel_features.shape[0])  # [num_classes, H*W]
    
    # 遍历每一类
    for class_idx, label in enumerate(unique_labels):
        # 获取属于该类的参考点的索引
        class_mask = (reference_labels == label)
        class_reference_points = reference_points[class_mask]  # [num_class_reference, num_band]
        
        # 初始化距离张量来存储该类所有参考点与像素的距离
        class_distance_matrix = torch.zeros(class_reference_points.shape[0], pixel_features.shape[0])  # [num_class_reference, H*W]
        
        # 计算每个参考点与所有像素的 L2 欧氏距离
        for i, ref_point in enumerate(class_reference_points):
            dist_matrix = torch.cdist(ref_point.unsqueeze(0), pixel_features.unsqueeze(0), p=2).squeeze(0)  # [H*W]
            class_distance_matrix[i] = dist_matrix  # 存储距离 [H*W]
        
        # 对该类的所有参考点距离取平均，距离越小越相似
        avg_similarity_matrix[class_idx] = class_distance_matrix.mean(dim=0)  # [H*W]
    
    # 对每个像素，找到平均距离最小的类别（距离越小，相似度越高）
    min_distance_indices = torch.argmin(avg_similarity_matrix, dim=0)  # [H*W]
    
    # 根据最小距离的类别分配标签
    predicted_labels = unique_labels[min_distance_indices]  # [H*W]
    
    # 将预测的标签转换为 [H, W] 的形状
    return predicted_labels.view(h, w)
# def nearest_neighbor_classification(metric_features, reference_mask, labels):
#     """
#     使用参考点掩码进行最近邻分类，根据相似度分配标签。
    
#     :param metric_features: 特征图像, 形状 [1, num_band, H, W]
#     :param reference_mask: 参考点的掩码, 形状 [1, H, W], 值为 0 和 1
#     :param labels: 标签图像, 形状 [1, H, W]
#     :return: 预测的标签图, 形状 [H, W]
#     """
#     h, w = metric_features.shape[2], metric_features.shape[3]  # 获取图像的高度和宽度
    
#     # 获取参考点的位置
#     reference_indices = (reference_mask == 1).nonzero(as_tuple=False)
    
#     # 提取参考点的特征和标签
#     reference_points = []
#     reference_labels = []
#     for idx in reference_indices:
#         # 获取参考点的特征
#         reference_points.append(metric_features[:, :, idx[1], idx[2]].unsqueeze(0))  # 使用正确的索引
#         reference_labels.append(labels[0, idx[1], idx[2]])  # 获取参考点的标签
    
#     # 将参考点的特征和标签转换为张量
#     reference_points = torch.cat(reference_points, dim=0)  # [num_reference, num_band]
#     reference_labels = torch.tensor(reference_labels)  # [num_reference]
    
#     # 将 metric_features 展平成 [H*W, num_band]，每个像素的特征向量
#     pixel_features = metric_features.view(metric_features.shape[1], -1).T  # [H*W, num_band]
    
#     # 获取所有唯一的标签
#     unique_labels = reference_labels.unique()
    
#     # 初始化一个张量来存储每个像素与每一类的平均相似度
#     avg_similarity_matrix = torch.zeros(len(unique_labels), pixel_features.shape[0])  # [num_classes, H*W]
    
#     # 遍历每一类
#     for class_idx, label in enumerate(unique_labels):
#         # 获取属于该类的参考点的索引
#         class_mask = (reference_labels == label)
#         class_reference_points = reference_points[class_mask]  # [num_class_reference, num_band]
        
#         # 初始化相似度张量来存储该类所有参考点的相似度
#         class_similarity = torch.zeros(class_reference_points.shape[0], pixel_features.shape[0])  # [num_class_reference, H*W]
        
#         # 计算每个参考点与所有像素的相似度
#         for i, ref_point in enumerate(class_reference_points):
#             dist_squared = torch.sum((pixel_features - ref_point) ** 2, dim=1)  # [H*W]
#             class_similarity[i] = 2 / (1 + torch.exp(dist_squared))  # [H*W]
        
#         # 对该类的所有参考点相似度取平均
#         avg_similarity_matrix[class_idx] = class_similarity.mean(dim=0)  # [H*W]
    
#     # 对每个像素，找到平均相似度最大的类别
#     max_similarity_indices = torch.argmax(avg_similarity_matrix, dim=0)  # [H*W]
    
#     # 根据最大相似度的类别分配标签
#     predicted_labels = unique_labels[max_similarity_indices]  # [H*W]
    
#     # 将预测的标签转换为 [H, W] 的形状
#     return predicted_labels.view(h, w)