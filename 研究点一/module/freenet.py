import torch.nn as nn
# import statsmodels.api as sm
import sys
from simplecv.util import metric
from scipy.io import savemat

import os
from simplecv import dp_train as train
from sklearn.metrics import precision_recall_fscore_support
from HSI_BandSelection_master.src.HSIBandSelection.Classification.Model import Model
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from simplecv.interface import CVModule
from simplecv.module import SEBlock
from simplecv import registry
from scipy.io import loadmat
import torch
import math
import math as m
from sklearn.decomposition import PCA
# from math import abs
from .mst_plus_plus import MST_Plus_Plus, MSAB
from .loss_mrae import Loss_MRAE, Loss_RMSE
from .BiSRNet import BiSRNet
import scipy
import torch
import torch.nn as nn
import numpy as np
# import matplotlib.pyplot as plt
from .herosnet import HerosNet
from .GST import GST_MODEL
from .RSSAN import RSSAN, RSSAN1, RSSAN2, RSSAN3, RSSAN4, RSSAN5
from .TPPP import SSAN, SSRN
from torch.nn import functional
# from sklearn.preprocessing import MinMaxScaler
# from fvcore.nn import FlopCountAnalysis
from HSI_BandSelection_master.src.HSIBandSelection.Data.readSAT import loadata, createImageCubes
from HSI_BandSelection_master.src.HSIBandSelection.SelectBands import SelectBands
from HSI_BandSelection_master.src.HSIBandSelection.utils import Dataset

from math import log10, sqrt 
import cv2 
import numpy as np
from sklearn.manifold import TSNE
import torch
from simplecv.util.logger import eval_progress, speed
import time
from module import freenet
from simplecv.util import metric
from data import dataloader
import argparse
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import numpy as np
from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
import seaborn as sns

OA_list = []
AA_list = []
KAPPA_list = []
total_var_superpixel_l = []
entropy_minimization_loss_l = []

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
    x_hsi_bs_np = x_hsi_bs.squeeze(0).cpu().numpy()  # 移除 batch 维度，形状变为 [20, h, w]
    x_rec_np = x_rec.squeeze(0).cpu().numpy()  # 移除 batch 维度，形状变为 [20, h, w]

    # 初始化 SSIM 总和变量
    total_ssim = 0

    # 遍历每个波段，计算 SSIM
    for i in range(x_hsi_bs_np.shape[0]):
        band_ssim = ssim(x_hsi_bs_np[i], x_rec_np[i], data_range=x_rec_np[i].max() - x_rec_np[i].min())
        total_ssim += band_ssim

    # 计算平均 SSIM
    average_ssim = total_ssim / x_hsi_bs_np.shape[0]

    return average_ssim

def visualize_and_save_all_bands(image, filename):
    num_bands = image.shape[0]
    rows, cols = 2, 10  # 4行5列的子图

    # 假设单个波段图像的宽高比为 (width / height)
    height, width = image.shape[1], image.shape[2]
    output_dir = '/mnt/sde/niuyuanzhuo/FreeNet-master/'
    # 根据子图的数量和单个图像的宽高比动态调整整体 figsize
    figsize = (cols * width / 100, rows * height / 100)  # 调整比例使波段图像不失真
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    for i in range(num_bands):
        ax = axes[i // cols, i % cols]
        band_img = image[i, :, :]
        band_img = (band_img - band_img.min()) / (band_img.max() - band_img.min())  # 标准化
        ax.imshow(band_img, cmap='gray')
        ax.axis('off')
        ax.set_aspect('auto')  # 保持图像的原始比例

    # 调整子图间距，确保没有空隙
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    # 保存整个图像
    os.makedirs(output_dir, exist_ok=True)  # 确保保存目录存在
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    
    plt.show()


# 可视化RGB图像的函数
def visualize_hsi_rgb(image, title):
    # 选择指定波段并标准化
    img_rgb = (image - image.min()) / (image.max() - image.min())
    
    # 转置为(height, width, channels)
    img_rgb = img_rgb.transpose(1, 2, 0)
    output_dir = '/mnt/sde/niuyuanzhuo/FreeNet-master/'
    plt.figure(figsize=(6, 6))
    plt.imshow(img_rgb)
    # plt.title(title)
    plt.axis('off')
    plt.show()
    # os.makedirs(output_dir, exist_ok=True)
    # 保存整个图像
    save_path = os.path.join(output_dir, f"{title}_all_bands.png")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

def calculate_entropy_minimization_loss(logits, segments):
    """
    Args:
    - logits: model logits (1, num_classes, h, w)
    - segments: superpixel segments (h, w)
    - temperature: temperature parameter for the softmax function

    Returns:
    - loss: clustering loss value
    """
    temperature = 1.0
    num_classes = logits.size(1)
    h, w = logits.size(2), logits.size(3)
    
    # Apply softmax to get probabilities
    probabilities = F.softmax(logits / temperature, dim=1)  # shape: (1, num_classes, h, w)
    probabilities = probabilities.view(num_classes, h * w)  # shape: (num_classes, h * w)
    segments = segments.view(h * w)  # shape: (h * w)
    
    # Ignore segments with value 0
    valid_mask = segments != 0
    valid_probabilities = probabilities[:, valid_mask]
    valid_segments = segments[valid_mask]
    
    # Get unique segment IDs and their indices
    unique_segments, inverse_indices = torch.unique(valid_segments, return_inverse=True)
    
    # Calculate the average probability for each segment (centroid)
    one_hot_segments = F.one_hot(inverse_indices, num_classes=len(unique_segments)).float()
    segment_counts = one_hot_segments.sum(dim=0)  # Number of pixels in each segment
    segment_probs = torch.matmul(valid_probabilities, one_hot_segments) / (segment_counts + 1e-6)
    
    # Calculate the distance of each pixel to its segment centroid
    expanded_segment_probs = segment_probs[:, inverse_indices]  # Expand segment_probs to pixel level
    intra_segment_distance = F.mse_loss(valid_probabilities, expanded_segment_probs, reduction='none')
    intra_segment_loss = intra_segment_distance.sum(dim=0).mean()  # Averaged over all valid pixels
    
    # Calculate the inter-segment separation
    segment_probs = segment_probs.permute(1, 0)  # shape: (num_segments, num_classes)
    pairwise_distances = torch.cdist(segment_probs, segment_probs, p=2)  # Euclidean distance
    eye_mask = torch.eye(len(unique_segments), device=pairwise_distances.device).bool()
    pairwise_distances = pairwise_distances[~eye_mask].view(len(unique_segments), -1)
    inter_segment_loss = pairwise_distances.mean()

    # Combined loss
    loss = intra_segment_loss + inter_segment_loss
    return loss

def calculate_superpixel_variance(logits, segments):
    """
    计算每个超像素内部的logit值方差（无显式for循环版本）。
    
    :param logits: 模型输出的预测类别值，形状为 [h, w]
    :param segments: 超像素分割结果，形状为 [h, w]
    :return: 所有超像素区域的预测类别值方差之和
    """
    h, w = logits.shape

    # 展平logits和segments
    logits_flat = logits.view(-1)  # 形状为 [h*w]
    segments_flat = segments.view(-1).long()  # 形状为 [h*w]

    # 获取所有超像素的唯一标签
    unique_segments = torch.unique(segments_flat)
    num_segments = unique_segments.size(0)

    # 创建用于scatter操作的张量，初始化为0
    segment_sums = torch.zeros(num_segments, device=logits.device)
    segment_counts = torch.zeros(num_segments, device=logits.device)

    # 计算每个超像素区域的logits之和和计数
    segment_sums.scatter_add_(0, segments_flat, logits_flat.float())
    segment_counts.scatter_add_(0, segments_flat, torch.ones_like(segments_flat, dtype=torch.float))

    # 计算每个超像素区域的logits均值
    segment_means = segment_sums / segment_counts

    # 计算每个像素与其所属超像素区域均值的差值
    expanded_segment_means = segment_means[segments_flat]  # 扩展成与logits_flat相同形状
    diffs = logits_flat.float() - expanded_segment_means

    # 计算每个超像素区域的方差
    segment_vars = torch.zeros(num_segments, device=logits.device)
    diffs_squared = diffs ** 2
    segment_vars.scatter_add_(0, segments_flat, diffs_squared)

    # 计算总方差
    total_variance = segment_vars.sum() / segment_vars.shape[0]

    return total_variance



def fcn_evaluate_fn(self, test_dataloader, config):
    if self.checkpoint.global_step < 0:
        return
    self._model.eval()
    total_time = 0.
    with torch.no_grad():
        for idx, (im, mask, w) in enumerate(test_dataloader):
            # start = time.time()
            y_pred = self._model(im).squeeze(0) #, final_feat
            torch.cuda.synchronize()
            y_pred = y_pred.argmax(dim=0).cpu() + 1
            
            w.unsqueeze_(dim=0)

            w = w.byte()
        
            mask = torch.masked_select(mask.view(-1), w.bool().view(-1))
            y_pred = torch.masked_select(y_pred.view(-1), w.bool().view(-1))
            
            
            oa = metric.th_overall_accuracy_score(mask.view(-1), y_pred.view(-1))
            aa, acc_per_class = metric.th_average_accuracy_score(mask.view(-1), y_pred.view(-1),
                                                                 len(np.unique(mask).tolist()),
                                                                 return_accuracys=True)
            kappa = metric.th_cohen_kappa_score(mask.view(-1), y_pred.view(-1), self._model.module.config.num_classes)
            # total_time += time_cost
            # speed(self._logger, time_cost, 'im')

            eval_progress(self._logger, idx + 1, len(test_dataloader))

    # metric_dict = {
    #     'OA': oa.item(),
    #     'AA': aa.item(),
    #     'Kappa': kappa.item()
    # }
    # for i, acc in enumerate(acc_per_class):
    #     metric_dict['acc_{}'.format(i + 1)] = acc.item()
    avg = (oa.item() + aa.item() + kappa.item()) / 3.0           
    self.avg = avg
    # print('ok')
    # for i, acc in enumerate(acc_per_class):
    #     metric_dict['acc_{}'.format(i + 1)] = acc.item()
    # self._logger.eval_log(metric_dict=metric_dict, step=self.checkpoint.global_step)
    # if self.checkpoint.global_step == 999:
    #     print('ok')
    #     # final_feat = torch.masked_select(final_feat.cpu().detach().reshape(final_feat.shape[1], -1), w.bool().view(-1))
    #     X = final_feat.view(128,-1).transpose(1,0).cpu().detach().numpy()
    #     Y = TSNE(n_components=2).fit_transform(X[-10000:,:])
    #     # color = mask.cpu().detach().numpy()
    #     plt.scatter(Y[:, 0], Y[:, 1], c=mask_res.view(-1).cpu().detach().numpy()[-10000:], cmap=plt.cm.Spectral)
    #     plt.show()
    # print('ok')


def register_evaluate_fn(launcher):
    launcher.override_evaluate(fcn_evaluate_fn)

def weight_reset(m):
    """Reset model weights after one epoch"""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        m.reset_parameters()

PARAM_KERNEL_SIZE = 3






    
def normalization_for_RSNR(arr1, arr2):
    return (arr2 - arr2.min()) * (1/(arr2.max() - arr2.min()) * arr1.max())

class BAM2(nn.Module):

    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.fc11 = nn.Linear(in_channels,128)
        self.fc12 = nn.Linear(128,64)
        self.fc13 = nn.Linear(64,hidden_channels)
        self.relu = nn.ReLU()
        self.sig  = nn.Sigmoid()
        self.start_rec = True

    def forward(self,x):
        x = x.permute(0, 2, 3, 1)
        x = self.fc11(x)
        x = self.relu(x)
        x = self.fc12(x)
        x = self.relu(x)
        x = self.fc13(x)
        x = x.permute(0, 3, 1, 2)
        return x

class SpectralAttentionNetwork_Init(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SpectralAttentionNetwork_Init, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_channel = out_channel
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // 4),  # 压缩通道数以减少参数量
            nn.ReLU(),
            nn.Linear(in_channel // 4, in_channel),  # 恢复通道数
            nn.Sigmoid()  # 使用Sigmoid激活函数使输出在0和1之间，可以理解为权重
        )
        self.spectral_attention = None

    def forward(self, x, epoch, bs_init):
        # x: [batch, channel, height, width]
        # 全局平均池化
        batch, channel, _, _ = x.size()
        if epoch < 110:
            spectral_attention = self.global_avg_pool(x)  # [batch, channel, 1, 1]
            
            # 将输出平铺为 [batch, channel]
            spectral_attention = spectral_attention.view(batch, channel)
            
            # 通过全连接层学习通道间的关系
            spectral_attention = self.fc(spectral_attention)  # [batch, channel]

            # 将注意力权重应用于原始图像
            # 注意力权重扩展为 [batch, channel, 1, 1] 以使用广播
            self.spectral_attention = spectral_attention
            spectral_attention = spectral_attention.view(batch, channel, 1, 1)
            print(self.spectral_attention.topk(self.out_channel)[1].sort()[0].squeeze(0))
            return x * spectral_attention, False, self.spectral_attention, self.spectral_attention.topk(self.out_channel + 1)[0][0][-1]
        else:
            print(self.spectral_attention.topk(self.out_channel)[0].squeeze(0))
            print(self.spectral_attention.topk(self.out_channel)[1].sort()[0].squeeze(0))
            return x[:, self.spectral_attention.topk(self.out_channel)[1].sort()[0].squeeze(0), :, :], True, self.spectral_attention, self.spectral_attention.topk(self.out_channel + 1)[0][0][-1]

class SpectralAttentionNetwork(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SpectralAttentionNetwork, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_channel = out_channel
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // 8),  # 压缩通道数以减少参数量
            nn.ReLU(),
            nn.Linear(in_channel // 8, in_channel),  # 恢复通道数
            nn.Sigmoid()  # 使用Sigmoid激活函数使输出在0和1之间，可以理解为权重
        )
        self.spectral_attention = None

    def forward(self, x, epoch):
        # x: [batch, channel, height, width]
        # 全局平均池化
        batch, channel, _, _ = x.size()
        trainx_hsi = x.unsqueeze(0)
        entropies =  [entropy_torch(trainx_hsi[:, :, i, :, :]) for i in range(channel)]
        if epoch < 2000:
            # spectral_attention = self.global_avg_pool(x)  # [batch, channel, 1, 1]
            spectral_attention = torch.tensor(entropies).view(1, channel , 1, 1).cuda()
            # 将输出平铺为 [batch, channel]
            spectral_attention = spectral_attention.view(batch, channel)
            
            # 通过全连接层学习通道间的关系
            spectral_attention = self.fc(spectral_attention)  # [batch, channel]

            # 将注意力权重应用于原始图像
            # 注意力权重扩展为 [batch, channel, 1, 1] 以使用广播
            self.spectral_attention = spectral_attention
            spectral_attention = spectral_attention.view(batch, channel, 1, 1)
            print(self.spectral_attention.sort()[0][0][-40:])
            print(self.spectral_attention.topk(self.out_channel)[1].sort()[0].squeeze(0))
            if self.spectral_attention.topk(self.out_channel)[1].sort()[0].squeeze(0) == [  1,  27,  53,  66,  67,  68,  78,  87, 104, 124, 134, 149, 156, 164, 166, 173, 176, 186, 191, 202]:
                sys.exit()
            return x * spectral_attention, False, self.spectral_attention, self.spectral_attention.topk(self.out_channel + 1)[0][0][-1]
        else:
            # sys.exit()
            print(self.spectral_attention.sort()[0][0][-40:])
            print(self.spectral_attention.topk(self.out_channel)[1].sort()[0].squeeze(0))
            return x[:, self.spectral_attention.topk(self.out_channel)[1].sort()[0].squeeze(0), :, :], True, self.spectral_attention, self.spectral_attention.topk(self.out_channel + 1)[0][0][-1]
    
class BAM(nn.Module):

    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.fc11 = nn.Linear(in_channels, in_channels)
        self.fc12 = nn.Linear(128, 64)
        self.fc13 = nn.Linear(64, in_channels)
        self.relu = nn.ReLU()
        self.sig  = nn.Sigmoid()
        self.start_rec = True

    def forward(self,x):
        x = x.permute(0, 2, 3, 1)
        x = self.fc11(x)
        x = self.relu(x)
        x = self.fc12(x)
        x = self.relu(x)
        x = self.fc13(x)
        x = x.permute(0, 3, 1, 2)
        return self.sig(x)

    # def BAM2(self,x):
    #     x = x.permute(0, 2, 3, 1)
    #     x = self.fc11(x)
    #     x = self.relu(x)
    #     x = self.fc12(x)
    #     x = self.relu(x)
    #     x = self.fc13(x)
    #     x = x.permute(0, 3, 1, 2)
    #     return x
    
class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv2d(
            1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w, t = x.size()

        # feature descriptor on the global spatial information
        # 24, 1, 1, 1
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -3)).transpose(
            -1, -3).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class Residual(nn.Module):  # pytorch
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding,
            use_1x1conv=False,
            stride=1,
            start_block=False,
            end_block=False,
    ):
        super(Residual, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride), nn.ReLU())
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride)
        if use_1x1conv:
            self.conv3 = nn.Conv3d(
                in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

        if not start_block:
            self.bn0 = nn.BatchNorm3d(in_channels)

        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)

        if start_block:
            self.bn2 = nn.BatchNorm3d(out_channels)

        if end_block:
            self.bn2 = nn.BatchNorm3d(out_channels)

        # ECA Attention Layer
        self.ecalayer = eca_layer(out_channels)

        # start and end block initialization
        self.start_block = start_block
        self.end_block = end_block

    def forward(self, X):
        identity = X

        if self.start_block:
            out = self.conv1(X)
        else:
            out = self.bn0(X)
            out = F.relu(out)
            out = self.conv1(out)

        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)


        if self.start_block:
            out = self.bn2(out)

        out = self.ecalayer(out)

        out += identity

        if self.end_block:
            out = self.bn2(out)
            out = F.relu(out)

        return out

class S3KAIResNet(nn.Module):
    def __init__(self, band, classes, reduction):
        super(S3KAIResNet, self).__init__()
        self.name = 'SSRN'
        self.dropout = nn.Dropout(p=0.2)
        self.conv1x1 = nn.Conv3d(
            in_channels=1,
            out_channels=band,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1))
        self.conv3x3 = nn.Conv3d(
            in_channels=1,
            out_channels=band,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1))

        self.batch_norm1x1 = nn.Sequential(
            nn.BatchNorm3d(
                band, eps=0.001, momentum=0.1,
                affine=True),  # 0.1
            nn.ReLU(inplace=True))
        self.batch_norm3x3 = nn.Sequential(
            nn.BatchNorm3d(
                band, eps=0.001, momentum=0.1,
                affine=True),  # 0.1
            nn.ReLU(inplace=True))

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.conv_se = nn.Sequential(
            nn.Conv3d(
                band, band // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True))
        self.conv_ex = nn.Conv3d(
            band // reduction, band, 1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.res_net1 = Residual(
            band,
            band, (1, 1, 7), (0, 0, 3),
            start_block=True)
        self.res_net2 = Residual(band, band,
                                 (1, 1, 7), (0, 0, 3))
        self.res_net3 = Residual(band, band,
                                 (3, 3, 1), (1, 1, 0))
        self.res_net4 = Residual(
            band,
            band, (3, 3, 1), (1, 1, 0),
            end_block=True)

        kernel_3d = band #math.ceil((band - 6) / 2)
        # print(kernel_3d)

        self.conv2 = nn.Conv3d(
            in_channels=band,
            out_channels=128,
            padding=(0, 0, 0),
            kernel_size=(1, 1, kernel_3d),
            stride=(1, 1, 1))
        self.batch_norm2 = nn.Sequential(
            nn.BatchNorm3d(128, eps=0.001, momentum=0.1, affine=True),  # 0.1
            nn.ReLU(inplace=True))
        self.conv3 = nn.Conv3d(
            in_channels=1,
            out_channels=band,
            padding=(0, 0, 0),
            kernel_size=(3, 3, 128),
            stride=(1, 1, 1))
        self.batch_norm3 = nn.Sequential(
            nn.BatchNorm3d(
                band, eps=0.001, momentum=0.1,
                affine=True),  # 0.1
            nn.ReLU(inplace=True))

        # self.avg_pooling = nn.AvgPool3d(kernel_size=(5, 5, 1))
        # self.full_connection = nn.Sequential(
        #     nn.Linear(PARAM_KERNEL_SIZE, classes)
        #     # nn.Softmax()
        # )

    def forward(self, X):
        X = X.unsqueeze(0)
        # X = nn.
        # X = self.dropout(X)
        x_1x1 = self.conv1x1(X)
        # x_1x1 = self.batch_norm1x1(x_1x1)#.unsqueeze(dim=1)
        # x_3x3 = self.conv3x3(X)
        # x_3x3 = self.batch_norm3x3(x_3x3).unsqueeze(dim=1)

        # x1 = torch.cat([x_3x3, x_1x1], dim=1)
        # U = torch.sum(x1, dim=1)
        # S = self.pool(U)
        # Z = self.conv_se(S)
        # attention_vector = torch.cat(
        #     [
        #         self.conv_ex(Z).unsqueeze(dim=1),
        #         self.conv_ex(Z).unsqueeze(dim=1)
        #     ],
        #     dim=1)
        # attention_vector = self.softmax(attention_vector)
        # V = (x1 * attention_vector).sum(dim=1)

        # x2 = self.res_net1(V)
        # x2 = self.res_net2(x2)
        # x2 = self.batch_norm2(self.conv2(x2))
        # x2 = x2.permute(0, 4, 2, 3, 1)
        # x2 = self.batch_norm3(self.conv3(x2))

        # x2 = self.res_net3(x2)
        # x2 = self.res_net4(x2)
        # x4 = self.avg_pooling(x3)
        # x4 = x4.view(x4.size(0), -1)
        return x_1x1.squeeze(0) #self.full_connection(x4)

class VoltageBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(VoltageBlock, self).__init__()
        # self.gap = GlobalAvgPool2D()
        self.seq = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # v = self.gap(x) # [1, 128, 552, 184] PC
        score = self.seq(x)
        # y = x * score.view(score.size(0), score.size(1), 1, 1)
        return score # [1, 128, 552, 184]

class SPC32(nn.Module):
    def __init__(self, msize=24, outplane=49, kernel_size=[7,1,1], stride=[1,1,1], padding=[3,0,0], spa_size=9,  bias=True):
        super(SPC32, self).__init__()
                                                  
        self.convm0 = nn.Conv3d(1, msize, kernel_size=kernel_size, padding=padding)         # generate mask0
        self.bn1 = nn.BatchNorm2d(outplane)
        
        self.convm2 = nn.Conv3d(1, msize, kernel_size=kernel_size, padding=padding)         # generate mask2
        self.bn2 = nn.BatchNorm2d(outplane)


    def forward(self, x, identity=None):
        
        if identity is None:
            identity = x                                                  # NCHW
        n,c,h,w = identity.size()
        
        mask0 = self.convm0(x.unsqueeze(1)).squeeze(2)                    # NCHW ==> NDHW
        mask0 = torch.softmax(mask0.view(n,-1,h*w), -1)                    
        mask0 = mask0.view(n,-1,h,w)
        _,d,_,_ = mask0.size()
        
        fk = torch.einsum('ndhw,nchw->ncd', mask0, x)                     # NCD
        
        out = torch.einsum('ncd,ndhw->ncdhw', fk, mask0)                  # NCDHW
        
        out = F.leaky_relu(out)
        out = out.sum(2)
        
        out = out #+ identity
        
        out0 = self.bn1(out.view(n,-1,h,w))
        
        mask2 = self.convm2(out0.unsqueeze(1)).squeeze(2)                 # NCHW ==> NDHW
        mask2 = torch.softmax(mask2.view(n,-1,h*w), -1)                    
        mask2 = mask2.view(n,-1,h,w)
        
        fk = torch.einsum('ndhw,nchw->ncd', mask2, x)                     # NCD
        
        out = torch.einsum('ncd,ndhw->ncdhw', fk, mask2)                  # NCDHW
        
        out = F.leaky_relu(out)
        out = out.sum(2)
        
        out = out + identity
        
        out = self.bn2(out.view(n,-1,h,w))

        return out 

class Spectral_attention(nn.Module):
    #  batchsize 16 25 200
    def __init__(self, in_features, hidden_features, out_features):
        super(Spectral_attention, self).__init__()
        self.AvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.MaxPool = nn.AdaptiveMaxPool2d((1, 1))
        self.in_features = in_features
        self.SharedMLP = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()  # ！

    def forward(self, X):

        y1 = self.AvgPool(X)
        y2 = self.MaxPool(X)
        y1 = y1.view(y1.size(0), -1)
        y2 = y2.view(y2.size(0), -1)
        # print(y1.shape, y2.shape)
        y1 = self.SharedMLP(y1)
        y2 = self.SharedMLP(y2)
        y = y1 + y2
        y = torch.reshape(y, (self.in_features, self.in_features, 1, 1))
        return self.sigmoid(y) * X #

l1loss1 = nn.L1Loss(reduction='mean')

l1loss2 = Loss_MRAE()
l1loss3 = Loss_RMSE()

def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
 
    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.show()
    return fig

def similarity(original, compressed):
    # mse = torch.mean((original - compressed) ** 2)
    # if(mse == 0): # MSE is zero means no noise is present in the signal .                   
    # # Therefore PSNR have no importance.         
    #     return 100
    # max_pixel = torch.max(original)
    # psnr = 10 * log10(max_pixel ** 2 / mse)
    return (original - compressed).abs().mean() * 1000  

def PSNR(original, compressed):
    mse = torch.mean((original - compressed) ** 2)
    if(mse == 0): # MSE is zero means no noise is present in the signal .                   
    # Therefore PSNR have no importance.         
        return 100
    max_pixel = torch.max(original)
    psnr = 10 * log10(max_pixel ** 2 / mse)
    return psnr  



# def my_summary(test_model, H = 256, W = 256, C = 31, N = 1):
#     model = test_model.cuda()
#     print(model)
#     inputs = torch.randn((N, C, H, W)).cuda()
#     flops = FlopCountAnalysis(model,inputs)
#     n_param = sum([p.nelement() for p in model.parameters()])
#     print(f'GMac:{flops.total()/(1024*1024*1024)}')
#     print(f'Params:{n_param}')

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

class FixedPosLinear(nn.Module):
    def __init__(self, in_dim, out_dim, minBand, maxBand, nBandDataset, dataset):
        super(FixedPosLinear, self).__init__()
        P = scipy.io.loadmat('/mnt/sde/niuyuanzhuo/FreeNet-master/P.mat')['P']
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
        # HU18
        if dataset == 'HH':
            self.weight = torch.tensor(P).transpose(1,0).float()
        if dataset == 'BS':
            self.weight = torch.tensor(P)[:,7:].transpose(1,0).float()
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
        tmp = 0 # 尽量别让反传时找不到梯度，因此不要detach,转到cpu之类的;对weight以一个整体操作，不要分开操作
        print(self.weight[0,0])
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
            x = F.interpolate(x[:,:,:,:138].squeeze(0), size=31).unsqueeze(0)
        # HU18
        if self.dataset == 'LK':
            x = F.interpolate(x[:,:,:,:135].squeeze(0), size=31).unsqueeze(0)
        # HU18
        if self.dataset == 'HH':
            x = F.interpolate(x[:,:,:,:135].squeeze(0), size=31).unsqueeze(0)
        if self.dataset == 'BS':
            x = x[:,:,:,:24]
# # 归一化矩阵
# 			normalized_matrix = torch.abs(torch.tensor(matrix)) / min_vals
        
        return torch.matmul(x, self.weight.cuda())
    
class PosLinear(nn.Module):
    def __init__(self, in_dim, out_dim, minBand, maxBand, nBandDataset, dataset):
        super(PosLinear, self).__init__()
        P = scipy.io.loadmat('/mnt/sde/niuyuanzhuo/FreeNet-master/P.mat')['P']
        # self.weight = nn.Parameter(torch.tensor(P).unsqueeze(0))
        # self.bias = nn.Parameter(torch.zeros((out_dim,)))
        		# IP
        if dataset == 'IP':
            self.weight = nn.Parameter(F.interpolate(torch.tensor(P).unsqueeze(0), size=32).transpose(1,2).squeeze(0).float())
		# PU
        if dataset == 'PU':
            self.weight = nn.Parameter(F.interpolate(torch.tensor(P).unsqueeze(0)[:,:,3:], size=72).transpose(1,2).squeeze(0).float())
		# SV
        if dataset == 'SV':
            self.weight = nn.Parameter(F.interpolate(torch.tensor(P).unsqueeze(0), size=32).transpose(1,2).squeeze(0).float())
		# DC
        if dataset == 'DC':
            self.weight = nn.Parameter(F.interpolate(torch.tensor(P).unsqueeze(0), size=27).transpose(1,2).squeeze(0).float())
        # HU18
        if dataset == 'HU18':
            self.weight = nn.Parameter(F.interpolate(torch.tensor(P).unsqueeze(0), size=23).transpose(1,2).squeeze(0).float())
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
        tmp = 0 # 尽量别让反传时找不到梯度，因此不要detach,转到cpu之类的;对weight以一个整体操作，不要分开操作
        print(self.weight[0,0])
		# IP
        if self.dataset == 'IP':
            x = x[:,:,:,:32]
		# PU
        if self.dataset == 'PU':
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
# # 归一化矩阵
# 			normalized_matrix = torch.abs(torch.tensor(matrix)) / min_vals
        
        return torch.matmul(x, F.relu(self.weight)/self.weight.sum(0))
    
class LyotVisualFilter(nn.Module):
    def __init__(self, in_dim, out_dim, minBand, maxBand, nBandDataset, dataset):
        super(LyotVisualFilter, self).__init__()
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
        weight = torch.rand((self.out_dim, self.nBandDataset)).cuda()
        print(self.weight_)
        tmp = 0 # 尽量别让反传时找不到梯度，因此不要detach,转到cpu之类的;对weight以一个整体操作，不要分开操作

		# IP
        if self.dataset == 'IP':
            x = x[:,:,:,:32]
            for n in range(self.nBandDataset):
                band = self.minBand + (self.maxBand - self.minBand) * n / self.nBandDataset
                weight[:,n - tmp] = (0.5 - 0.5*torch.cos(2*m.pi*(-0.01)*(pow(self.weight_,-1))/(band*pow(10,-6))))
		# PU
        if self.dataset == 'PU':
            x = x[:,:,:,:72]
            for n in range(self.nBandDataset):
                band = self.minBand + (self.maxBand - self.minBand) * n / self.nBandDataset
                weight[:,n - tmp] = (0.5 - 0.5*torch.cos(2*m.pi*(-0.01)*(pow(self.weight_,-1))/(band*pow(10,-6))))
		# SV
        if self.dataset == 'SV':
            x = x[:,:,:,:32]
            for n in range(self.nBandDataset):
                band = self.minBand + (self.maxBand - self.minBand) * n / self.nBandDataset
                weight[:,n - tmp] = (0.5 - 0.5*torch.cos(2*m.pi*(-0.01)*(pow(self.weight_,-1))/(band*pow(10,-6))))
		# DC
        if self.dataset == 'DC':
            x = x[:,:,:,:27]
            for n in range(self.nBandDataset):
                band = self.minBand + (self.maxBand - self.minBand) * n / self.nBandDataset
                weight[:,n - tmp] = (0.5 - 0.5*torch.cos(2*m.pi*(-0.01)*(pow(self.weight_,-1))/(band*pow(10,-6))))
        
        # print('dcdcdcdcdcdc')
		# HU2018
        if self.dataset == 'HU18':
            x = x[:,:,:,1:24]
            for n in range(self.nBandDataset):
                band = self.minBand + (self.maxBand - self.minBand) * n / self.nBandDataset
                weight[:,n - tmp] = (0.5 - 0.5*torch.cos(2*m.pi*(-0.01)*(pow(self.weight_,-1))/(band*pow(10,-6))))


        return torch.matmul(x, F.relu(weight.transpose(0,1))/weight.transpose(0,1).sum(0)) # F.relu(weight.transpose(0,1))/weight.transpose(0,1).sum(0)

# index_l = []
# class BandSelectionLayer(nn.Module):
#     def __init__(self, num_input_bands, num_output_bands):
#         super(BandSelectionLayer, self).__init__()
#         # 初始化选择矩阵，使用softmax进行列归一化
#         self.selection_matrix = nn.Parameter(torch.randn(num_input_bands, num_output_bands))

#     def forward(self, x):
#         # 应用选择矩阵：矩阵乘法
#         # x shape: (batch_size, num_input_bands, H, W)
#         # selection_matrix shape: (num_input_bands, num_output_bands)
#         # 结果 shape: (batch_size, num_output_bands, H, W)
#         selection_weights = torch.softmax(self.selection_matrix, dim=0)  # 列归一化
#         x = torch.einsum('bihw,io->bohw', x, selection_weights)
        
#         return x, selection_weights.max(0)[0].sum()

class E2E_BS(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, in_channels, hidden_channels, dataset):
        super().__init__()
        self.size_in, self.size_out, self.dataset = in_channels, hidden_channels, dataset
        self.fc_l = []
        if self.dataset == 'LK':
            for i in range(self.size_out):
                self.fc_l.append(nn.Linear(1 * 288 * 400, 1, bias=False).cuda())
        self.sigmoid  = nn.Sigmoid().cuda()
        self.softmax = nn.Softmax().cuda()

    def forward(self, x):
        b_, c_, h_, w_ = x.shape
        x = x.reshape(self.size_in, -1)
        weights = torch.rand(self.size_out, self.size_in).cuda()
        for i in range(self.size_out):
            weights[i] = self.sigmoid(self.fc_l[i](x)).squeeze()
        x = x.reshape(b_, h_, w_, c_)
        
        return torch.matmul(x, weights.T).permute(0, 3, 1, 2), weights # w times x + b
    
class MyLinearLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        weights = torch.Tensor(size_out, size_in)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
        bias = torch.Tensor(size_out)
        self.bias = nn.Parameter(bias)

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5)) # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        w_times_x = torch.matmul(x, self.weights.t())
        # print(self.weights[0][:3])
        return w_times_x.permute(0, 3, 1, 2)  # w times x + b
    
class BandSelection(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(BandSelection, self).__init__()
        self.BS1 = nn.Parameter(torch.rand((in_dim)))
        self.BS2 = nn.Parameter(torch.randint(0, in_dim, (out_dim, )).float())
        self.BS3 = nn.Parameter(torch.rand((in_dim, out_dim )))
        self.sig  = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.in_dim = in_dim
        self.out_dim = out_dim
        # weight =
        # self.weight = nn.Parameter(torch.rand((out_dim, in_dim)).cuda())

    def forward(self, x):
        sig = torch.randn((1, self.in_dim, 1, 1)).cuda()
        sig = self.sig(self.BS1).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        # output = torch.randn(self.in_dim)
        # vals, idx = self.BS1.topk(self.out_dim)
        # topk = torch.zeros_like(output)
        # topk[idx] = vals
        # topk.unsqueeze(0)
        # topk.expand_as()
        # BS = torch.rand((self.out_dim)).cuda()
        # BS = self.BS1.topk(self.out_dim)[1]
        # BS = self.BS1.ge(self.BS1.topk(self.out_dim)[0][-1]).float().expand_as(BS)
        # mask = torch.rand_like(self.BS1) > 0.7
        # self.sig(x)
        # x = x.permute(0, 2, 3, 1)
        
        # result = self.BS.argsort()[:self.out_dim].sort(0, False)[0]
        print(self.BS1[:3])
        # return x[:, BS, :, :]
        return x * sig

class ReduceChannelsFilter(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ReduceChannelsFilter, self).__init__()
        self.weight_ = nn.Parameter(torch.rand((in_dim, out_dim))).cuda()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # weight =
        # self.weight = nn.Parameter(torch.rand((out_dim, in_dim)).cuda())

    def forward(self, x):
        # self.weight = self.weight.cpu()
        # tmp = 0
        # weight = torch.rand((self.out_dim, self.in_dim)).cuda() # srf
        # print(self.weight_) # voltage
        
        return torch.matmul(x, self.weight_)
    
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
        print(self.weight_) # voltage
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
                # F.relu(weight.transpose(0,1))/weight.transpose(0,1).sum(0)
                          





class LyotFilterwithoutChangeX(nn.Module):
    def __init__(self, in_dim, out_dim, minBand, maxBand, nBandDataset, dataset):
        super(LyotFilterwithoutChangeX, self).__init__()
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
        weight = torch.rand((self.out_dim, self.nBandDataset)).cuda()
        print(self.weight_)
        tmp = 0 # 尽量别让反传时找不到梯度，因此不要detach,转到cpu之类的;对weight以一个整体操作，不要分开操作

		# IP
        if self.dataset == 'IP':
            # x = x[:,:,:,:32]
            for n in range(self.nBandDataset):
                band = self.minBand + (self.maxBand - self.minBand) * n / self.nBandDataset
                weight[:,n - tmp] = (0.5 - 0.5*torch.cos(2*m.pi*(-0.01)*(pow(self.weight_,-1))/(band*pow(10,-6))))
		# PU
        if self.dataset == 'PU':
            # x = x[:,:,:,:72]
            for n in range(self.nBandDataset):
                band = self.minBand + (self.maxBand - self.minBand) * n / self.nBandDataset
                weight[:,n - tmp] = (0.5 - 0.5*torch.cos(2*m.pi*(-0.01)*(pow(self.weight_,-1))/(band*pow(10,-6))))
		# SV
        if self.dataset == 'SV':
            # x = x[:,:,:,:32]
            for n in range(self.nBandDataset):
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

        return torch.matmul(x, F.relu(weight.transpose(0,1))/weight.transpose(0,1).sum(0))

class PixelwiseMetricLoss(torch.nn.Module):
    def __init__(self, temperature=0.1, pos_margin=0.1, neg_margin=1.0, num_negative=10):
        super().__init__()
        self.temperature = temperature
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.num_negative = num_negative

    def forward(self, features, labels, weights):
        B, C, H, W = features.shape
        
        # 归一化特征
        features = F.normalize(features, p=2, dim=1)
        
        # 重塑张量
        features = features.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
        labels = labels.reshape(-1)  # [B*H*W]
        weights = weights.reshape(-1)  # [B*H*W]
        
        # 只选择权重为1的像素
        valid_mask = weights == 1
        valid_features = features[valid_mask]
        valid_labels = labels[valid_mask]
        
        # 如果没有有效像素，返回零损失
        if valid_features.shape[0] == 0:
            return torch.tensor(0.0, device=features.device)
        
        # 计算有效像素之间的余弦相似度
        sim_matrix = torch.mm(valid_features, valid_features.t()) / self.temperature
        
        # 创建标签掩码
        label_matrix = valid_labels.unsqueeze(1) == valid_labels.unsqueeze(0)
        
        # 移除自身比较
        eye_mask = torch.eye(label_matrix.shape[0], dtype=torch.bool, device=label_matrix.device)
        label_matrix.masked_fill_(eye_mask, False)
        
        # 正样本对
        pos_pairs = sim_matrix[label_matrix]
        
        # 负样本对 (通过随机采样)
        neg_mask = ~label_matrix
        neg_sim = sim_matrix[neg_mask].reshape(sim_matrix.shape[0], -1)
        neg_pairs = neg_sim[:, torch.randperm(neg_sim.shape[1])[:self.num_negative]]
        
        # 计算损失
        pos_loss = F.relu(self.pos_margin - pos_pairs).mean()
        neg_loss = F.relu(neg_pairs - self.neg_margin).mean()
        
        loss = pos_loss + neg_loss
        
        return loss
    
@registry.MODEL.register('FreeNetMetricL')
class FreeNetMetricL(CVModule):
    def __init__(self, config):
        super(FreeNetMetricL, self).__init__(config)
        r = int(16 * self.config.reduction_ratio)
        block1_channels = int(self.config.block_channels[0] * self.config.reduction_ratio / r) * r
        block2_channels = int(self.config.block_channels[1] * self.config.reduction_ratio / r) * r
        block3_channels = int(self.config.block_channels[2] * self.config.reduction_ratio / r) * r
        block4_channels = int(self.config.block_channels[3] * self.config.reduction_ratio / r) * r
        # print(self.config.in_channels)
        self.feature_ops = nn.ModuleList([
            conv3x3_gn_relu(self.config.in_channels, block1_channels, r),

            repeat_block(block1_channels, r, self.config.num_blocks[0]),
            nn.Identity(),
            downsample2x(block1_channels, block2_channels),

            repeat_block(block2_channels, r, self.config.num_blocks[1]),
            nn.Identity(),
            downsample2x(block2_channels, block3_channels),

            repeat_block(block3_channels, r, self.config.num_blocks[2]),
            nn.Identity(),
            downsample2x(block3_channels, block4_channels),

            repeat_block(block4_channels, r, self.config.num_blocks[3]),
            nn.Identity(),
        ])
        inner_dim = int(self.config.inner_dim * self.config.reduction_ratio)
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
        self.cls_pred_conv = nn.Conv2d(inner_dim, self.config.num_classes, 1)
        self.time_= 0
        # 添加一个投影头用于度量学习
        self.projection_head = nn.Conv2d(self.config.num_classes, inner_dim, 1)
        # self.BS = PCA(n_components=self.config.hidden_channels)

    def top_down(self, top, lateral):
        top2x = F.interpolate(top, scale_factor=2.0, mode='nearest')
        return lateral + top2x # 

    def forward(self, x, y=None, w=None, **kwargs):
        
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

        final_feat = out_feat_list[-1] # [1, 128, 288, 400]

        # embedding = self.projection_head(final_feat.mean([-2, -1]))
        logit = self.cls_pred_conv(final_feat) # [1, 6, 288, 400]
        # embeddings = self.projection_head(logit)
        # if self.time_ == 1000:
        if self.training and self.time_ >= 990:
            self.plot_tsne_with_mask(final_feat, y, w, 1000, r'C:\\Users\\17735\\OneDrive\\桌面\\FreeNet-master\\tsne_result.png')

        if self.training:
            loss_dict = {
                # 'cls_loss': self.loss(logit, y, w),
                'metric_loss': self.compute_loss(final_feat, y, w)
            }
            return loss_dict

        return torch.softmax(logit, dim=1)#.squeeze(), final_feat

    def plot_tsne_with_mask(self, final_feat, y, w, num_samples=1000, save_path=None):
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

        # 展开特征向量、标签和掩码，形状变为 [batch_size * h * w, num_dim] 和 [batch_size * h * w]
        final_feat = final_feat.view(batch_size, num_dim, -1).permute(0, 2, 1).contiguous().view(-1, num_dim).cpu().detach().numpy()
        y = y.view(-1).cpu().numpy()
        w = w.view(-1).cpu().numpy()

        # 只选择掩码为 1 的像素点
        selected_indices = np.where(w == 1)[0]
        
        if len(selected_indices) == 0:
            print("没有选择的像素点")
            return

        # 选择特征和标签
        final_feat_selected = final_feat[selected_indices]
        y_selected = y[selected_indices]

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
        plt.title('t-SNE of Feature Embeddings (Using Mask w)')

        # 保存图像到指定路径
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')  # 保存图片
            print(f"t-SNE 图已保存至 {save_path}")
        
        # 显示图像
        # plt.show()


    def compute_loss(self, final_feat, y, w):
        """
        计算嵌入的损失函数，基于选定的像素集合 S。

        Args:
            final_feat (torch.Tensor): 嵌入的特征向量，形状为 [batch_size, num_dim, h, w]。
            y (torch.Tensor): 分割标签，形状为 [batch_size, h, w]，值为 0 到 C 之间的整数。
            w (torch.Tensor): 选择的权重掩码，形状为 [batch_size, h, w]，值为 0 或 1，1 表示选择的点。

        Returns:
            torch.Tensor: 计算得到的损失值。
        """
        batch_size, num_dim, height, width = final_feat.shape

        # 展开所有像素点 [batch_size, num_dim, h*w]
        final_feat = final_feat.view(batch_size, num_dim, -1)
        y = y.view(batch_size, -1)  # 标签展开为 [batch_size, h*w]
        w = w.view(batch_size, -1)  # 掩码展开为 [batch_size, h*w]

        # 确保每个 batch 中至少有 2 个有效像素被选择
        valid_masks = w.sum(dim=-1) >= 2

        if not valid_masks.any():
            return torch.tensor(0.0, device=final_feat.device)

        # 选定的像素点（掩码 w 为 1 的位置）
        selected_indices = w.bool()

        # 选择的嵌入向量、标签和权重
        selected_feat = final_feat.permute(0, 2, 1)[selected_indices]  # [total_selected, num_dim]
        selected_y = y[selected_indices]  # [total_selected]

        # 计算像素间的欧氏距离平方
        dist_matrix = torch.cdist(selected_feat.unsqueeze(0), selected_feat.unsqueeze(0), p=2) ** 2  # [1, total_selected, total_selected]

        # 计算相似度 σ(p, q)
        sigma_pq = 2 / (1 + torch.exp(dist_matrix))

        # 计算每个实例的大小
        instance_sizes = torch.bincount(selected_y.long(), minlength=int(y.max().item())).float()  # [num_classes]

        # 计算权重 w_{pq}，权重与实例大小成反比
        inv_instance_sizes = 1.0 / (instance_sizes[selected_y.long()] + 1e-8)  # [total_selected]
        w_pq = inv_instance_sizes.unsqueeze(1) * inv_instance_sizes.unsqueeze(0)  # [total_selected, total_selected]

        # 归一化 w_pq，确保所有的权重和为 1
        w_pq /= w_pq.sum()

        # 创建标签相等的掩码矩阵和相异掩码矩阵
        same_label_mask = (selected_y.unsqueeze(1) == selected_y.unsqueeze(0)).float()  # [total_selected, total_selected]
        diff_label_mask = 1 - same_label_mask  # [total_selected, total_selected]

        # 计算损失的两部分
        same_loss = same_label_mask * torch.log(sigma_pq + 1e-8)
        diff_loss = diff_label_mask * torch.log(1 - sigma_pq + 1e-8)

        # 计算总损失
        loss = -torch.sum(w_pq * (same_loss + diff_loss))

        return loss / batch_size
    
    def metric_loss(self, final_feat, y, w):
        """
        计算嵌入的损失函数，基于选定的像素集合 S。

        Args:
            final_feat (torch.Tensor): 嵌入的特征向量，形状为 [batch_size, num_dim, h, w]。
            y (torch.Tensor): 分割标签，形状为 [batch_size, h, w]，值为 0 到 C 之间的整数。
            w (torch.Tensor): 选择的权重掩码，形状为 [batch_size, h, w]，值为 0 或 1，1 表示选择的点。

        Returns:
            torch.Tensor: 计算得到的损失值。
        """
        batch_size, num_dim, height, width = final_feat.shape

        # 展开所有像素点
        final_feat = final_feat.view(batch_size, num_dim, -1)  # 形状变为 [batch_size, num_dim, h*w]
        y = y.view(batch_size, -1)  # 形状变为 [batch_size, h*w]
        w = w.view(batch_size, -1)  # 形状变为 [batch_size, h*w]

        # 初始化损失
        loss = 0.0

        # 逐个 batch 处理
        for i in range(batch_size):
            # 选定的像素点，即 w 为 1 的像素点集合 S
            selected_indices = (w[i] == 1).nonzero(as_tuple=True)[0]  # 获取选择的像素点索引
            if len(selected_indices) < 2:  # 如果没有足够的选择点，跳过
                continue

            # 选择的嵌入向量、标签和权重
            selected_feat = final_feat[i][:, selected_indices]  # [num_dim, |S|]
            selected_y = y[i][selected_indices]  # [|S|]
            selected_w = w[i][selected_indices]  # [|S|]

            # 计算像素间的欧氏距离平方
            dist_matrix = torch.cdist(selected_feat.permute(1, 0), selected_feat.permute(1, 0), p=2) ** 2  # [|S|, |S|]

            # 计算相似度 σ(p, q)
            sigma_pq = 2 / (1 + torch.exp(dist_matrix))

            # 计算每个实例的大小
            instance_sizes = torch.bincount(selected_y.long(), minlength=int(y.max().item())).float()  # [num_classes]

            # 计算权重 w_{pq}，权重与实例大小成反比
            # 对每个像素找到它所属的实例，然后计算权重
            inv_instance_sizes = 1.0 / (instance_sizes[selected_y.long()] + 1e-8)  # [|S|]
            w_pq = inv_instance_sizes.unsqueeze(1) * inv_instance_sizes.unsqueeze(0)  # [|S|, |S|]

            # 归一化 w_pq，确保所有的权重和为 1
            w_pq /= w_pq.sum()

            # 创建标签相等的掩码矩阵和相异掩码矩阵
            same_label_mask = (selected_y.unsqueeze(1) == selected_y.unsqueeze(0)).float()  # [|S|, |S|]
            diff_label_mask = 1 - same_label_mask  # [|S|, |S|]

            # 计算损失的两部分
            same_loss = same_label_mask * torch.log(sigma_pq + 1e-8)
            diff_loss = diff_label_mask * torch.log(1 - sigma_pq + 1e-8)

            # 计算总损失
            loss += -torch.sum(w_pq * (same_loss + diff_loss))

        # 平均损失
        loss /= batch_size

        return loss
    
    def loss(self, x, y, weight):
        losses = F.cross_entropy(x, y.long() - 1, weight=None,
                                 ignore_index=-1, reduction='none')

        v = losses.mul_(weight).sum() / weight.sum()
        return v
    
    # def metric_loss(self, final_feat, y, w):
    #     loss_fn = PixelwiseMetricLoss()
    #     return loss_fn(final_feat, y, w)
    
    def set_defalut_config(self):
        self.config.update(dict(
            in_channels=204,
            num_classes=16,
            block_channels=(96, 128, 192, 256),
            num_blocks=(1, 1, 1, 1),
            inner_dim=128,
            reduction_ratio=1.0,
        ))

@registry.MODEL.register('FreeNet')
class FreeNet(CVModule):
    def __init__(self, config):
        super(FreeNet, self).__init__(config)
        r = int(16 * self.config.reduction_ratio)
        block1_channels = int(self.config.block_channels[0] * self.config.reduction_ratio / r) * r
        block2_channels = int(self.config.block_channels[1] * self.config.reduction_ratio / r) * r
        block3_channels = int(self.config.block_channels[2] * self.config.reduction_ratio / r) * r
        block4_channels = int(self.config.block_channels[3] * self.config.reduction_ratio / r) * r
        # print(self.config.in_channels)
        self.feature_ops = nn.ModuleList([
            conv3x3_gn_relu(self.config.in_channels, block1_channels, r),

            repeat_block(block1_channels, r, self.config.num_blocks[0]),
            nn.Identity(),
            downsample2x(block1_channels, block2_channels),

            repeat_block(block2_channels, r, self.config.num_blocks[1]),
            nn.Identity(),
            downsample2x(block2_channels, block3_channels),

            repeat_block(block3_channels, r, self.config.num_blocks[2]),
            nn.Identity(),
            downsample2x(block3_channels, block4_channels),

            repeat_block(block4_channels, r, self.config.num_blocks[3]),
            nn.Identity(),
        ])
        inner_dim = int(self.config.inner_dim * self.config.reduction_ratio)
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
        self.cls_pred_conv = nn.Conv2d(inner_dim, self.config.num_classes, 1)
        self.time_= 0
        # self.BS = PCA(n_components=self.config.hidden_channels)

    def top_down(self, top, lateral):
        top2x = F.interpolate(top, scale_factor=2.0, mode='nearest')
        return lateral + top2x # 

    def forward(self, x, y=None, w=None, **kwargs):
        
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
        # if self.time_== 3:
        #     print('ok')
        #     final_feat = torch.masked_select(final_feat.cpu().detach().reshape(final_feat.shape[1], -1), w.bool().view(-1).cpu().detach())
        #     X = final_feat.squeeze(0).view(final_feat.squeeze(0).shape[0],-1).transpose(1,0).cpu().detach().numpy()
        #     Y = TSNE(n_components=2, perplexity=1).fit_transform(X)
        #     color = y.squeeze(0).view(-1).cpu().detach().numpy()
        #     plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
        # # # plt.show(fig)
        # # # plt.imshow(final_feat_embedded, cmap='hot',interpolation='nearest')
        # # # plt.colorbar()
        #     plt.show()

        logit = self.cls_pred_conv(final_feat)
        if self.training:
            loss_dict = {
                'cls_loss': self.loss(logit, y, w),
            }
            return loss_dict

        return torch.softmax(logit, dim=1)#.squeeze(), final_feat

    def loss(self, x, y, weight):
        losses = F.cross_entropy(x, y.long() - 1, weight=None,
                                 ignore_index=-1, reduction='none')

        v = losses.mul_(weight).sum() / weight.sum()
        return v
    
    def set_defalut_config(self):
        self.config.update(dict(
            in_channels=204,
            num_classes=16,
            block_channels=(96, 128, 192, 256),
            num_blocks=(1, 1, 1, 1),
            inner_dim=128,
            reduction_ratio=1.0,
        ))


@registry.MODEL.register('FreeNetBS_test')
class FreeNetBS_test(CVModule):
    def __init__(self, config):
        super(FreeNetBS_test, self).__init__(config)
        r = int(16 * self.config.reduction_ratio)
        block1_channels = int(self.config.block_channels[0] * self.config.reduction_ratio / r) * r
        block2_channels = int(self.config.block_channels[1] * self.config.reduction_ratio / r) * r
        block3_channels = int(self.config.block_channels[2] * self.config.reduction_ratio / r) * r
        block4_channels = int(self.config.block_channels[3] * self.config.reduction_ratio / r) * r
        # print(self.config.in_channels)
        self.feature_ops = nn.ModuleList([
            conv3x3_gn_relu(self.config.hidden_channels, block1_channels, r),

            repeat_block(block1_channels, r, self.config.num_blocks[0]),
            nn.Identity(),
            downsample2x(block1_channels, block2_channels),

            repeat_block(block2_channels, r, self.config.num_blocks[1]),
            nn.Identity(),
            downsample2x(block2_channels, block3_channels),

            repeat_block(block3_channels, r, self.config.num_blocks[2]),
            nn.Identity(),
            downsample2x(block3_channels, block4_channels),

            repeat_block(block4_channels, r, self.config.num_blocks[3]),
            nn.Identity(),
        ])
        inner_dim = int(self.config.inner_dim * self.config.reduction_ratio)
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
        self.cls_pred_conv = nn.Conv2d(inner_dim, self.config.num_classes, 1)
        self.time_= 0
        self.BS = PCA(n_components=self.config.hidden_channels)

    def top_down(self, top, lateral):
        top2x = F.interpolate(top, scale_factor=2.0, mode='nearest')
        return lateral + top2x # 

    def forward(self, x, y=None, w=None, **kwargs):
        
        self.time_+= 1 # torch.Size([1, 270, 288, 400])

        x = x[:, [  0,  13,  16,  22,  30,  37,  48,  88,  93, 104, 116, 120, 125, 129, 132, 144, 157, 184, 186, 196],:,:] # [  5,  18,  29,  69,  73,  74,  95,  97, 104, 107, 136, 139, 157, 165,
        #169, 176, 182, 188, 193, 201]     # [  5,  18,  29,  69,  73,  74,  95,  97, 104, 107, 136, 139, 157, 165,
        #169, 176, 182, 188, 193, 201]
        
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
        # if self.time_== 3:
        #     print('ok')
        #     final_feat = torch.masked_select(final_feat.cpu().detach().reshape(final_feat.shape[1], -1), w.bool().view(-1).cpu().detach())
        #     X = final_feat.squeeze(0).view(final_feat.squeeze(0).shape[0],-1).transpose(1,0).cpu().detach().numpy()
        #     Y = TSNE(n_components=2, perplexity=1).fit_transform(X)
        #     color = y.squeeze(0).view(-1).cpu().detach().numpy()
        #     plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
        # # plt.show(fig)
        # # plt.imshow(final_feat_embedded, cmap='hot',interpolation='nearest')
        # # plt.colorbar()
        #     plt.show()

        logit = self.cls_pred_conv(final_feat)
        if self.training:
            loss_dict = {
                'cls_loss': self.loss(logit, y, w)
            }
            return loss_dict

        return torch.softmax(logit, dim=1)#.squeeze(), final_feat

    def loss(self, x, y, weight):
        losses = F.cross_entropy(x, y.long() - 1, weight=None,
                                 ignore_index=-1, reduction='none')

        v = losses.mul_(weight).sum() / weight.sum()
        return v

    def set_defalut_config(self):
        self.config.update(dict(
            in_channels=204,
            num_classes=16,
            block_channels=(96, 128, 192, 256),
            num_blocks=(1, 1, 1, 1),
            inner_dim=128,
            reduction_ratio=1.0,
        ))


@registry.MODEL.register('FreeNetTest')
class FreeNetTest(CVModule):
    def __init__(self, config,
                 time = 0,
                selection = None):
        super(FreeNetTest, self).__init__(config)
        r = int(16 * self.config.reduction_ratio)
        block1_channels = int(self.config.block_channels[0] * self.config.reduction_ratio / r) * r
        block2_channels = int(self.config.block_channels[1] * self.config.reduction_ratio / r) * r
        block3_channels = int(self.config.block_channels[2] * self.config.reduction_ratio / r) * r
        block4_channels = int(self.config.block_channels[3] * self.config.reduction_ratio / r) * r
        # print(self.config.in_channels)
        self.feature_ops = nn.ModuleList([
            conv3x3_gn_relu(self.config.hidden_channels, block1_channels, r),

            repeat_block(block1_channels, r, self.config.num_blocks[0]),
            nn.Identity(),
            downsample2x(block1_channels, block2_channels),

            repeat_block(block2_channels, r, self.config.num_blocks[1]),
            nn.Identity(),
            downsample2x(block2_channels, block3_channels),

            repeat_block(block3_channels, r, self.config.num_blocks[2]),
            nn.Identity(),
            downsample2x(block3_channels, block4_channels),

            repeat_block(block4_channels, r, self.config.num_blocks[3]),
            nn.Identity(),
        ])
        inner_dim = int(self.config.inner_dim * self.config.reduction_ratio)
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
        self.cls_pred_conv = nn.Conv2d(inner_dim, self.config.num_classes, 1)
        self.time_= time
        self.BS = PCA(n_components=self.config.hidden_channels)
        self.selection = selection

    def top_down(self, top, lateral):
        top2x = F.interpolate(top, scale_factor=2.0, mode='nearest')
        return lateral + top2x # 

    def forward(self, x, y=None, w=None, selection=None, **kwargs):
        if self.training and selection is not None:
            self.selection = selection
        self.time_+= 1 # torch.Size([1, 270, 288, 400])
        # x = x.permute(1, 0, 2, 3)
        
        # x = x[:, [0, 30, 56, 67, 107, 138, 142, 206],:,:]
        feat_list = []
        if len(x.shape) == 5:
            x = x.squeeze(0)

        x = x[:, self.selection, :, :]
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
        # if self.time_== 3:
        #     print('ok')
        #     final_feat = torch.masked_select(final_feat.cpu().detach().reshape(final_feat.shape[1], -1), w.bool().view(-1).cpu().detach())
        #     X = final_feat.squeeze(0).view(final_feat.squeeze(0).shape[0],-1).transpose(1,0).cpu().detach().numpy()
        #     Y = TSNE(n_components=2, perplexity=1).fit_transform(X)
        #     color = y.squeeze(0).view(-1).cpu().detach().numpy()
        #     plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
        # # plt.show(fig)
        # # plt.imshow(final_feat_embedded, cmap='hot',interpolation='nearest')
        # # plt.colorbar()
        #     plt.show()

        logit = self.cls_pred_conv(final_feat)
        if self.training:
            loss_dict = {
                'cls_loss': self.loss(logit, y, w)
            }
            return loss_dict

        return torch.softmax(logit, dim=1)#.squeeze(), final_feat

    def loss(self, x, y, weight):
        losses = F.cross_entropy(x, y.long() - 1, weight=None,
                                 ignore_index=-1, reduction='none')

        v = losses.mul_(weight).sum() / weight.sum()
        return v

    def set_defalut_config(self):
        self.config.update(dict(
            in_channels=204,
            num_classes=16,
            block_channels=(96, 128, 192, 256),
            num_blocks=(1, 1, 1, 1),
            inner_dim=128,
            reduction_ratio=1.0,
        ))

@registry.MODEL.register('FreeNetVisual')
class FreeNetVisual(CVModule):
    def __init__(self, config):
        super(FreeNetVisual, self).__init__(config)
        r = int(16 * self.config.reduction_ratio)
        block1_channels = int(self.config.block_channels[0] * self.config.reduction_ratio / r) * r
        block2_channels = int(self.config.block_channels[1] * self.config.reduction_ratio / r) * r
        block3_channels = int(self.config.block_channels[2] * self.config.reduction_ratio / r) * r
        block4_channels = int(self.config.block_channels[3] * self.config.reduction_ratio / r) * r
        # print(self.config.in_channels)
        self.feature_ops = nn.ModuleList([
            conv3x3_gn_relu(self.config.in_channels, block1_channels, r),

            repeat_block(block1_channels, r, self.config.num_blocks[0]),
            nn.Identity(),
            downsample2x(block1_channels, block2_channels),

            repeat_block(block2_channels, r, self.config.num_blocks[1]),
            nn.Identity(),
            downsample2x(block2_channels, block3_channels),

            repeat_block(block3_channels, r, self.config.num_blocks[2]),
            nn.Identity(),
            downsample2x(block3_channels, block4_channels),

            repeat_block(block4_channels, r, self.config.num_blocks[3]),
            nn.Identity(),
        ])
        inner_dim = int(self.config.inner_dim * self.config.reduction_ratio)
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
        self.cls_pred_conv = nn.Conv2d(inner_dim, self.config.num_classes, 1)

    def top_down(self, top, lateral):
        top2x = F.interpolate(top, scale_factor=2.0, mode='nearest')
        return lateral + top2x # 

    def forward(self, x, y=None, w=None, **kwargs):
        # print('okokokokokokokokokokokokokokokokokok')
        # torch.save(x.to(torch.device('cpu')), "myTensor.pth")
        # if  self.training is False:
        #     print('now')
        
        # Visible Band
        if self.config.dataset == 'IP':
            x = x[:,1:32,:,:]
        if self.config.dataset == 'PU':
            x = x[:,:72,:,:]
        if self.config.dataset == 'SV':
            x = x[:,:32,:,:]
        if self.config.dataset == 'DC':
            x = x[:,:27,:,:]
        if self.config.dataset == 'HU18':
            x = x[:,1:24,:,:]
        if self.config.dataset == 'HC':
            x = x[:,:138,:,:]
        if self.config.dataset == 'LK':
            x = x[:,:135,:,:]
        if self.config.dataset == 'HH':
            x = x[:,:135,:,:]
        if self.config.dataset == 'PC':
            x = x[:,:72,:,:]
        if self.config.dataset == 'BS':
            x = x[:,:24,:,:]

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

        logit = self.cls_pred_conv(final_feat)
        if self.training:
            loss_dict = {
                'cls_loss': self.loss(logit, y, w)
            }
            return loss_dict

        return torch.softmax(logit, dim=1)

    def loss(self, x, y, weight):
        losses = F.cross_entropy(x, y.long() - 1, weight=None,
                                 ignore_index=-1, reduction='none')

        v = losses.mul_(weight).sum() / weight.sum()
        return v

    def set_defalut_config(self):
        self.config.update(dict(
            in_channels=204,
            num_classes=16,
            block_channels=(96, 128, 192, 256),
            num_blocks=(1, 1, 1, 1),
            inner_dim=128,
            reduction_ratio=1.0,
        ))

@registry.MODEL.register('FreeNetLinearSRF')
class FreeNetLinearSRF(CVModule):
    def __init__(self, config):
        super(FreeNetLinearSRF, self).__init__(config)
        r = int(16 * self.config.reduction_ratio)
        block1_channels = int(self.config.block_channels[0] * self.config.reduction_ratio / r) * r
        block2_channels = int(self.config.block_channels[1] * self.config.reduction_ratio / r) * r
        block3_channels = int(self.config.block_channels[2] * self.config.reduction_ratio / r) * r
        block4_channels = int(self.config.block_channels[3] * self.config.reduction_ratio / r) * r
        # print(self.config.in_channels)
        self.feature_ops = nn.ModuleList([
            conv3x3_gn_relu(self.config.out_channels, block1_channels, r),

            repeat_block(block1_channels, r, self.config.num_blocks[0]),
            nn.Identity(),
            downsample2x(block1_channels, block2_channels),

            repeat_block(block2_channels, r, self.config.num_blocks[1]),
            nn.Identity(),
            downsample2x(block2_channels, block3_channels),

            repeat_block(block3_channels, r, self.config.num_blocks[2]),
            nn.Identity(),
            downsample2x(block3_channels, block4_channels),

            repeat_block(block4_channels, r, self.config.num_blocks[3]),
            nn.Identity(),
        ])
        inner_dim = int(self.config.inner_dim * self.config.reduction_ratio)
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
        self.cls_pred_conv = nn.Conv2d(inner_dim, self.config.num_classes, 1)
        self.srf = PosLinear(self.config.in_channels, self.config.out_channels,  self.config.minBand, self.config.maxBand, self.config.nBandDataset, self.config.dataset)

    def top_down(self, top, lateral):
        top2x = F.interpolate(top, scale_factor=2.0, mode='nearest')
        return lateral + top2x # 

    def forward(self, x, y=None, w=None, **kwargs):
        # print(x.shape) # torch.Size([1, 103, 624, 352])
        x = x.permute(0,2,3,1)
        x = self.srf(x)
        x = x.permute(0,3,1,2)
        feat_list = []
        for op in self.feature_ops:
            x = op(x)
            if isinstance(op, nn.Identity):
                feat_list.append(x)

        inner_feat_list = [self.reduce_1x1convs[i](feat) for i, feat in enumerate(feat_list)]
        inner_feat_list.reverse()

        out_feat_list = [self.fuse_3x3convs[0](inner_feat_list[0])]
        for i in range(len(feat_list) - 1):
            inner = self.top_down(out_feat_list[i], inner_feat_list[i+1])
            out = self.fuse_3x3convs[i + 1](inner)
            out_feat_list.append(out)

        final_feat = out_feat_list[-1]

        logit = self.cls_pred_conv(final_feat)
        if self.training:
            loss_dict = {
                'cls_loss': self.loss(logit, y, w)
            }
            return loss_dict

        return torch.softmax(logit, dim=1)

    def loss(self, x, y, weight):
        losses = F.cross_entropy(x, y.long() - 1, weight=None,
                                 ignore_index=-1, reduction='none')

        v = losses.mul_(weight).sum() / weight.sum()
        return v

    def set_defalut_config(self):
        self.config.update(dict(
            in_channels=204,
            num_classes=16,
            block_channels=(96, 128, 192, 256),
            num_blocks=(1, 1, 1, 1),
            inner_dim=128,
            reduction_ratio=1.0,
        ))


@registry.MODEL.register('FreeNetCaveSRF')
class FreeNetCaveSRF(CVModule):
    def __init__(self, config):
        super(FreeNetCaveSRF, self).__init__(config)
        r = int(16 * self.config.reduction_ratio)
        block1_channels = int(self.config.block_channels[0] * self.config.reduction_ratio / r) * r
        block2_channels = int(self.config.block_channels[1] * self.config.reduction_ratio / r) * r
        block3_channels = int(self.config.block_channels[2] * self.config.reduction_ratio / r) * r
        block4_channels = int(self.config.block_channels[3] * self.config.reduction_ratio / r) * r
        # print(self.config.in_channels)
        self.feature_ops = nn.ModuleList([
            conv3x3_gn_relu(self.config.out_channels, block1_channels, r),

            repeat_block(block1_channels, r, self.config.num_blocks[0]),
            nn.Identity(),
            downsample2x(block1_channels, block2_channels),

            repeat_block(block2_channels, r, self.config.num_blocks[1]),
            nn.Identity(),
            downsample2x(block2_channels, block3_channels),

            repeat_block(block3_channels, r, self.config.num_blocks[2]),
            nn.Identity(),
            downsample2x(block3_channels, block4_channels),

            repeat_block(block4_channels, r, self.config.num_blocks[3]),
            nn.Identity(),
        ])
        inner_dim = int(self.config.inner_dim * self.config.reduction_ratio)
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
        self.cls_pred_conv = nn.Conv2d(inner_dim, self.config.num_classes, 1)
        self.srf = FixedPosLinear(self.config.in_channels, self.config.out_channels,  self.config.minBand, self.config.maxBand, self.config.nBandDataset, self.config.dataset)

    def top_down(self, top, lateral):
        top2x = F.interpolate(top, scale_factor=2.0, mode='nearest')
        return lateral + top2x # 

    def forward(self, x, y=None, w=None, **kwargs):
        # print(x.shape) # torch.Size([1, 103, 624, 352])
        x = x.permute(0,2,3,1)
        x = self.srf(x)
        x = x.permute(0,3,1,2)
        # plt.figure()
        # plt.imshow(x.cpu().numpy().squeeze(0).reshape(x.shape[2],x.shape[3],x.shape[1])[:,:,(2,1,0)])
        # plt.show()
        feat_list = []
        for op in self.feature_ops:
            x = op(x)
            if isinstance(op, nn.Identity):
                feat_list.append(x)

        inner_feat_list = [self.reduce_1x1convs[i](feat) for i, feat in enumerate(feat_list)]
        inner_feat_list.reverse()

        out_feat_list = [self.fuse_3x3convs[0](inner_feat_list[0])]
        for i in range(len(feat_list) - 1):
            inner = self.top_down(out_feat_list[i], inner_feat_list[i+1])
            out = self.fuse_3x3convs[i + 1](inner)
            out_feat_list.append(out)

        final_feat = out_feat_list[-1]

        logit = self.cls_pred_conv(final_feat)
        if self.training:
            loss_dict = {
                'cls_loss': self.loss(logit, y, w)
            }
            return loss_dict

        return torch.softmax(logit, dim=1)#.squeeze(), final_feat

    def loss(self, x, y, weight):
        losses = F.cross_entropy(x, y.long() - 1, weight=None,
                                 ignore_index=-1, reduction='none')

        v = losses.mul_(weight).sum() / weight.sum()
        return v

    def set_defalut_config(self):
        self.config.update(dict(
            in_channels=204,
            num_classes=16,
            block_channels=(96, 128, 192, 256),
            num_blocks=(1, 1, 1, 1),
            inner_dim=128,
            reduction_ratio=1.0,
        ))

@registry.MODEL.register('FreeNetFewChannelsVoltage')
class FreeNetFewChannelsVoltage(CVModule):
    def __init__(self, config):
        super(FreeNetFewChannelsVoltage, self).__init__(config)
        r = int(16 * self.config.reduction_ratio)
        block1_channels = int(self.config.block_channels[0] * self.config.reduction_ratio / r) * r
        block2_channels = int(self.config.block_channels[1] * self.config.reduction_ratio / r) * r
        block3_channels = int(self.config.block_channels[2] * self.config.reduction_ratio / r) * r
        block4_channels = int(self.config.block_channels[3] * self.config.reduction_ratio / r) * r
        # print(self.config.in_channels)
        self.feature_ops = nn.ModuleList([
            conv3x3_gn_relu(self.config.out_channels, block1_channels, r),

            repeat_block(block1_channels, r, self.config.num_blocks[0]),
            nn.Identity(),
            downsample2x(block1_channels, block2_channels),

            repeat_block(block2_channels, r, self.config.num_blocks[1]),
            nn.Identity(),
            downsample2x(block2_channels, block3_channels),

            repeat_block(block3_channels, r, self.config.num_blocks[2]),
            nn.Identity(),
            downsample2x(block3_channels, block4_channels),

            repeat_block(block4_channels, r, self.config.num_blocks[3]),
            nn.Identity(),
        ])
        inner_dim = int(self.config.inner_dim * self.config.reduction_ratio)
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
        self.cls_pred_conv = nn.Conv2d(inner_dim, self.config.num_classes, 1)
        self.srf = LyotFilter(self.config.in_channels, self.config.out_channels,  self.config.minBand, self.config.maxBand, self.config.nBandDataset, self.config.dataset)
    
    def top_down(self, top, lateral):
        top2x = F.interpolate(top, scale_factor=2.0, mode='nearest')

        return lateral + top2x # 

    def forward(self, x, y=None, w=None, **kwargs):
        # print(x.shape) # torch.Size([1, 103, 624, 352])
        
        # torch.save(x.to(torch.device('cpu')), "myTensor.pth")
        x = x.permute(0,2,3,1)
        x = self.srf(x) 
        x = x.permute(0,3,1,2)
        feat_list = []
        for op in self.feature_ops:
            x = op(x)
            if isinstance(op, nn.Identity):
                feat_list.append(x)

        inner_feat_list = [self.reduce_1x1convs[i](feat) for i, feat in enumerate(feat_list)]
        inner_feat_list.reverse()

        out_feat_list = [self.fuse_3x3convs[0](inner_feat_list[0])]
        for i in range(len(feat_list) - 1):
            inner = self.top_down(out_feat_list[i], inner_feat_list[i+1])
            out = self.fuse_3x3convs[i + 1](inner)
            out_feat_list.append(out)

        final_feat = out_feat_list[-1]

        logit = self.cls_pred_conv(final_feat)
        if self.training:
            loss_dict = {
                'cls_loss': self.loss(logit, y, w)
            }
            return loss_dict

        return torch.softmax(logit, dim=1)

    def loss(self, x, y, weight):
        losses = F.cross_entropy(x, y.long() - 1, weight=None,
                                 ignore_index=-1, reduction='none')

        v = losses.mul_(weight).sum() / weight.sum()
        return v

    def set_defalut_config(self):
        self.config.update(dict(
            in_channels=204,
            num_classes=16,
            block_channels=(96, 128, 192, 256),
            num_blocks=(1, 1, 1, 1),
            inner_dim=128,
            reduction_ratio=1.0,
        ))

        
@registry.MODEL.register('FreeNetFewChannels')
class FreeNetFewChannels(CVModule):
    def __init__(self, config):
        super(FreeNetFewChannels, self).__init__(config)
        r = int(16 * self.config.reduction_ratio)
        block1_channels = int(self.config.block_channels[0] * self.config.reduction_ratio / r) * r
        block2_channels = int(self.config.block_channels[1] * self.config.reduction_ratio / r) * r
        block3_channels = int(self.config.block_channels[2] * self.config.reduction_ratio / r) * r
        block4_channels = int(self.config.block_channels[3] * self.config.reduction_ratio / r) * r
        # print(self.config.in_channels)
        self.feature_ops = nn.ModuleList([
            conv3x3_gn_relu(self.config.out_channels, block1_channels, r),

            repeat_block(block1_channels, r, self.config.num_blocks[0]),
            nn.Identity(),
            downsample2x(block1_channels, block2_channels),

            repeat_block(block2_channels, r, self.config.num_blocks[1]),
            nn.Identity(),
            downsample2x(block2_channels, block3_channels),

            repeat_block(block3_channels, r, self.config.num_blocks[2]),
            nn.Identity(),
            downsample2x(block3_channels, block4_channels),

            repeat_block(block4_channels, r, self.config.num_blocks[3]),
            nn.Identity(),
        ])
        inner_dim = int(self.config.inner_dim * self.config.reduction_ratio)
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
        self.cls_pred_conv = nn.Conv2d(inner_dim, self.config.num_classes, 1)
        self.srf = nn.Linear(self.config.in_channels, self.config.out_channels)
    
    def top_down(self, top, lateral):
        top2x = F.interpolate(top, scale_factor=2.0, mode='nearest')

        return lateral + top2x # 

    def forward(self, x, y=None, w=None, **kwargs):
        # print(x.shape) # torch.Size([1, 103, 624, 352])
        
        # torch.save(x.to(torch.device('cpu')), "myTensor.pth")
        x = x.permute(0,2,3,1)
        x = self.srf(x) 
        x = x.permute(0,3,1,2)
        feat_list = []
        for op in self.feature_ops:
            x = op(x)
            if isinstance(op, nn.Identity):
                feat_list.append(x)

        inner_feat_list = [self.reduce_1x1convs[i](feat) for i, feat in enumerate(feat_list)]
        inner_feat_list.reverse()

        out_feat_list = [self.fuse_3x3convs[0](inner_feat_list[0])]
        for i in range(len(feat_list) - 1):
            inner = self.top_down(out_feat_list[i], inner_feat_list[i+1])
            out = self.fuse_3x3convs[i + 1](inner)
            out_feat_list.append(out)

        final_feat = out_feat_list[-1]

        logit = self.cls_pred_conv(final_feat)
        if self.training:
            loss_dict = {
                'cls_loss': self.loss(logit, y, w)
            }
            return loss_dict

        return torch.softmax(logit, dim=1)

    def loss(self, x, y, weight):
        losses = F.cross_entropy(x, y.long() - 1, weight=None,
                                 ignore_index=-1, reduction='none')

        v = losses.mul_(weight).sum() / weight.sum()
        return v

    def set_defalut_config(self):
        self.config.update(dict(
            in_channels=204,
            num_classes=16,
            block_channels=(96, 128, 192, 256),
            num_blocks=(1, 1, 1, 1),
            inner_dim=128,
            reduction_ratio=1.0,
        ))

@registry.MODEL.register('FreeNetVisualFewChannelsVoltageRecAfterDecoder')
class FreeNetVisualFewChannelsVoltageRecAfterDecoder(CVModule):
    def __init__(self, config):
        super(FreeNetVisualFewChannelsVoltageRecAfterDecoder, self).__init__(config)
        r = int(16 * self.config.reduction_ratio)
        block1_channels = int(self.config.block_channels[0] * self.config.reduction_ratio / r) * r
        block2_channels = int(self.config.block_channels[1] * self.config.reduction_ratio / r) * r
        block3_channels = int(self.config.block_channels[2] * self.config.reduction_ratio / r) * r
        block4_channels = int(self.config.block_channels[3] * self.config.reduction_ratio / r) * r
        # print(self.config.in_channels)
        self.feature_ops = nn.ModuleList([
            conv3x3_gn_relu(self.config.out_channels, block1_channels, r),

            repeat_block(block1_channels, r, self.config.num_blocks[0]),
            nn.Identity(),
            downsample2x(block1_channels, block2_channels),

            repeat_block(block2_channels, r, self.config.num_blocks[1]),
            nn.Identity(),
            downsample2x(block2_channels, block3_channels),

            repeat_block(block3_channels, r, self.config.num_blocks[2]),
            nn.Identity(),
            downsample2x(block3_channels, block4_channels),

            repeat_block(block4_channels, r, self.config.num_blocks[3]),
            nn.Identity(),
        ])
        inner_dim = int(self.config.inner_dim * self.config.reduction_ratio)
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
        self.fuse_3x3convs_reconstruct = nn.ModuleList([
            nn.Conv2d(inner_dim, inner_dim, 1),
            # nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            # nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            # nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
        ])
        self.fuse_3x3convs_classify = nn.ModuleList([
            nn.Conv2d(inner_dim, inner_dim, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            # nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
        ])
        self.cls_pred_conv = nn.Conv2d(inner_dim, self.config.num_classes, 1)
        self.reconstruct = nn.Conv2d(inner_dim, self.config.nBandDataset, 1)
        self.srf = LyotFilterwithoutChangeX(self.config.in_channels, self.config.out_channels,  self.config.minBand, self.config.maxBand, self.config.nBandDataset, self.config.dataset)
    
    def top_down(self, top, lateral):
        top2x = F.interpolate(top, scale_factor=2.0, mode='nearest')
   
        return lateral + top2x # 

    def forward(self, x, y=None, w=None, **kwargs):
        # print(x.shape) # torch.Size([1, 103, 624, 352])
        train_length = (int(int(x.shape[2]/2)/16) + 1) * 16
        test_length = x.shape[2] - train_length
        x = x.permute(0,2,3,1)

        if self.config.dataset == 'IP':
            x = x[:,:,:,:32]
        if self.config.dataset == 'PU':
            x = x[:,:,:,:72]
        if self.config.dataset == 'SV':
            x = x[:,:,:,:32]
        if self.config.dataset == 'DC':
            x = x[:,:,:,:27]
        if self.config.dataset == 'HU18':
            x = x[:,:,:,1:24]
        x = x.permute(0,3,1,2)
        x_target_train = x[:,:,:train_length,:] # torch.Size([1, 32, 96, 160])
        x_target_test = x[:,:,train_length:,:]
        # torch.save(x.to(torch.device('cpu')), "myTensor.pth")

        x = x.permute(0,2,3,1)
        x = self.srf(x) 
        x = x.permute(0,3,1,2)

        x_source_train_rec = x[:,:,:train_length,:] # torch.Size([1, 3, 96, 160])
        x_source_test_rec = x[:,:,train_length:,:] # torch.Size([1, 3, 96, 160])

        # classification
        feat_list = []
        for op in self.feature_ops:
            x = op(x)
            if isinstance(op, nn.Identity):
                feat_list.append(x)

        inner_feat_list = [self.reduce_1x1convs[i](feat) for i, feat in enumerate(feat_list)]
        inner_feat_list.reverse()

        out_feat_list = [self.fuse_3x3convs[0](inner_feat_list[0])]
        for i in range(len(feat_list) - 1):
            inner = self.top_down(out_feat_list[i], inner_feat_list[i+1])
            out = self.fuse_3x3convs[i + 1](inner)
            out_feat_list.append(out)

        
        final_feat_cla = out_feat_list[-1]

        

        # rec_feat = final_feat_rec

        for i in range(len(self.fuse_3x3convs_classify)):
            final_feat_cla = self.fuse_3x3convs_classify[i](final_feat_cla)


        
        logit = self.cls_pred_conv(final_feat_cla)
        if self.training:
            # reconstruct
            feat_list_rec = []
            for op in self.feature_ops:
                x_source_train_rec = op(x_source_train_rec)
                if isinstance(op, nn.Identity):
                    feat_list_rec.append(x_source_train_rec)

            inner_feat_rec_list = [self.reduce_1x1convs[i](feat) for i, feat in enumerate(feat_list_rec)]
            inner_feat_rec_list.reverse()

            out_feat_rec_list = [self.fuse_3x3convs[0](inner_feat_rec_list[0])]
            for i in range(len(feat_list) - 1):
                inner = self.top_down(out_feat_rec_list[i], inner_feat_rec_list[i+1])
                out = self.fuse_3x3convs[i + 1](inner)
                out_feat_rec_list.append(out)

            final_feat_rec = out_feat_rec_list[-1]

            for i in range(len(self.fuse_3x3convs_reconstruct)):
                final_feat_rec = self.fuse_3x3convs_reconstruct[i](final_feat_rec)

            rec_res = self.reconstruct(final_feat_rec)
            # print('ok')

            loss_dict = {
                'cls_loss': self.loss(logit, y, w),
                'rex_loss': l1loss1(rec_res, x_target_train)
            }
            return loss_dict

        # reconstruct test
        if not self.training:
            
            # # reconstruct test
            feat_list_reconstruct_test = []
                        # feat_list_rec = []
            for op in self.feature_ops:
                x_source_test_rec = op(x_source_test_rec)
                if isinstance(op, nn.Identity):
                    feat_list_reconstruct_test.append(x_source_test_rec)

            inner_feat_rec_test_list = [self.reduce_1x1convs[i](feat) for i, feat in enumerate(feat_list_reconstruct_test)]
            inner_feat_rec_test_list.reverse()

            out_feat_rec_test_list = [self.fuse_3x3convs[0](inner_feat_rec_test_list[0])]
            for i in range(len(feat_list) - 1):
                inner = self.top_down(out_feat_rec_test_list[i], inner_feat_rec_test_list[i+1])
                out = self.fuse_3x3convs[i + 1](inner)
                out_feat_rec_test_list.append(out)

            final_feat_reconstruct_test = out_feat_rec_test_list[-1]

            for i in range(len(self.fuse_3x3convs_reconstruct)):
                final_feat_reconstruct_test = self.fuse_3x3convs_reconstruct[i](final_feat_reconstruct_test)

            # rec_res = self.reconstruct(final_feat_rec)
            x_reconstruct_test_pred = self.reconstruct(final_feat_reconstruct_test)
        print('Reconstruction PSNR:' + str(PSNR(x_reconstruct_test_pred, x_target_test)))
        return torch.softmax(logit, dim=1)

    def loss(self, x, y, weight):
        losses = F.cross_entropy(x, y.long() - 1, weight=None,
                                 ignore_index=-1, reduction='none')

        v = losses.mul_(weight).sum() / weight.sum()
        return v

    def set_defalut_config(self):
        self.config.update(dict(
            in_channels=204,
            num_classes=16,
            block_channels=(96, 128, 192, 256),
            num_blocks=(1, 1, 1, 1),
            inner_dim=128,
            reduction_ratio=1.0,
        ))

@registry.MODEL.register('FreeNetVisualFewChannelsVoltage')
class FreeNetVisualFewChannelsVoltage(CVModule):
    def __init__(self, config):
        super(FreeNetVisualFewChannelsVoltage, self).__init__(config)
        r = int(16 * self.config.reduction_ratio)
        block1_channels = int(self.config.block_channels[0] * self.config.reduction_ratio / r) * r
        block2_channels = int(self.config.block_channels[1] * self.config.reduction_ratio / r) * r
        block3_channels = int(self.config.block_channels[2] * self.config.reduction_ratio / r) * r
        block4_channels = int(self.config.block_channels[3] * self.config.reduction_ratio / r) * r
        # print(self.config.in_channels)
        self.feature_ops = nn.ModuleList([
            conv3x3_gn_relu(self.config.out_channels, block1_channels, r),

            repeat_block(block1_channels, r, self.config.num_blocks[0]),
            nn.Identity(),
            downsample2x(block1_channels, block2_channels),

            repeat_block(block2_channels, r, self.config.num_blocks[1]),
            nn.Identity(),
            downsample2x(block2_channels, block3_channels),

            repeat_block(block3_channels, r, self.config.num_blocks[2]),
            nn.Identity(),
            downsample2x(block3_channels, block4_channels),

            repeat_block(block4_channels, r, self.config.num_blocks[3]),
            nn.Identity(),
        ])
        inner_dim = int(self.config.inner_dim * self.config.reduction_ratio)
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
        self.cls_pred_conv = nn.Conv2d(inner_dim, self.config.num_classes, 1)
        self.srf = LyotVisualFilter(self.config.in_channels, self.config.out_channels,  self.config.minBand, self.config.maxBand, self.config.nBandDataset, self.config.dataset)
    
    def top_down(self, top, lateral):
        top2x = F.interpolate(top, scale_factor=2.0, mode='nearest')
   
        return lateral + top2x # 

    def forward(self, x, y=None, w=None, **kwargs):
        # print(x.shape) # torch.Size([1, 103, 624, 352])
        
        # torch.save(x.to(torch.device('cpu')), "myTensor.pth")
        x = x.permute(0,2,3,1)
        x = self.srf(x) 
        x = x.permute(0,3,1,2)
        feat_list = []
        for op in self.feature_ops:
            x = op(x)
            if isinstance(op, nn.Identity):
                feat_list.append(x)

        inner_feat_list = [self.reduce_1x1convs[i](feat) for i, feat in enumerate(feat_list)]
        inner_feat_list.reverse()

        out_feat_list = [self.fuse_3x3convs[0](inner_feat_list[0])]
        for i in range(len(feat_list) - 1):
            inner = self.top_down(out_feat_list[i], inner_feat_list[i+1])
            out = self.fuse_3x3convs[i + 1](inner)
            out_feat_list.append(out)

        final_feat = out_feat_list[-1]

        logit = self.cls_pred_conv(final_feat)
        if self.training:
            loss_dict = {
                'cls_loss': self.loss(logit, y, w)
            }
            return loss_dict

        return torch.softmax(logit, dim=1)

    def loss(self, x, y, weight):
        losses = F.cross_entropy(x, y.long() - 1, weight=None,
                                 ignore_index=-1, reduction='none')

        v = losses.mul_(weight).sum() / weight.sum()
        return v

    def set_defalut_config(self):
        self.config.update(dict(
            in_channels=204,
            num_classes=16,
            block_channels=(96, 128, 192, 256),
            num_blocks=(1, 1, 1, 1),
            inner_dim=128,
            reduction_ratio=1.0,
        ))

@registry.MODEL.register('FreeNetFewChannelsVoltageRecAfterDecoder')
class FreeNetFewChannelsVoltageRecAfterDecoder(CVModule):
    def __init__(self, config):
        super(FreeNetFewChannelsVoltageRecAfterDecoder, self).__init__(config)
        r = int(16 * self.config.reduction_ratio)
        block1_channels = int(self.config.block_channels[0] * self.config.reduction_ratio / r) * r
        block2_channels = int(self.config.block_channels[1] * self.config.reduction_ratio / r) * r
        block3_channels = int(self.config.block_channels[2] * self.config.reduction_ratio / r) * r
        block4_channels = int(self.config.block_channels[3] * self.config.reduction_ratio / r) * r
        # print(self.config.in_channels)
        self.feature_ops = nn.ModuleList([
            conv3x3_gn_relu(self.config.out_channels, block1_channels, r),

            repeat_block(block1_channels, r, self.config.num_blocks[0]),
            nn.Identity(),
            downsample2x(block1_channels, block2_channels),

            repeat_block(block2_channels, r, self.config.num_blocks[1]),
            nn.Identity(),
            downsample2x(block2_channels, block3_channels),

            repeat_block(block3_channels, r, self.config.num_blocks[2]),
            nn.Identity(),
            downsample2x(block3_channels, block4_channels),

            repeat_block(block4_channels, r, self.config.num_blocks[3]),
            nn.Identity(),
        ])
        inner_dim = int(self.config.inner_dim * self.config.reduction_ratio)
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
        self.fuse_3x3convs_reconstruct = nn.ModuleList([
            nn.Conv2d(inner_dim, inner_dim, 1),
            # nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            # nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            # nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
        ])
        self.fuse_3x3convs_classify = nn.ModuleList([
            nn.Conv2d(inner_dim, inner_dim, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            # nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
        ])
        self.cls_pred_conv = nn.Conv2d(inner_dim, self.config.num_classes, 1)
        self.reconstruct = nn.Conv2d(inner_dim, self.config.nBandDataset, 1)
        self.srf = LyotFilterwithoutChangeX(self.config.in_channels, self.config.out_channels,  self.config.minBand, self.config.maxBand, self.config.nBandDataset, self.config.dataset)
    
    def top_down(self, top, lateral):
        top2x = F.interpolate(top, scale_factor=2.0, mode='nearest')
   
        return lateral + top2x # 

    def forward(self, x, y=None, w=None, **kwargs):
        # print(x.shape) # torch.Size([1, 103, 624, 352])
        train_length = (int(int(x.shape[2]/2)/16) + 1) * 16
        test_length = x.shape[2] - train_length
        x = x.permute(0,2,3,1)

        # if self.config.dataset == 'IP':
        #     x = x[:,:,:,:32]
        # if self.config.dataset == 'PU':
        #     x = x[:,:,:,:72]
        # if self.config.dataset == 'SV':
        #     x = x[:,:,:,:32]
        # if self.config.dataset == 'DC':
        #     x = x[:,:,:,:27]
        # if self.config.dataset == 'HU18':
        #     x = x[:,:,:,1:24]
        x = x.permute(0,3,1,2)
        x_target_train = x[:,:,:train_length,:] # torch.Size([1, 32, 96, 160])
        x_target_test = x[:,:,train_length:,:]
        # torch.save(x.to(torch.device('cpu')), "myTensor.pth")

        x = x.permute(0,2,3,1)
        x = self.srf(x) 
        x = x.permute(0,3,1,2)

        x_source_train_rec = x[:,:,:train_length,:] # torch.Size([1, 3, 96, 160])
        x_source_test_rec = x[:,:,train_length:,:] # torch.Size([1, 3, 96, 160])

        # classification
        feat_list = []
        for op in self.feature_ops:
            x = op(x)
            if isinstance(op, nn.Identity):
                feat_list.append(x)

        inner_feat_list = [self.reduce_1x1convs[i](feat) for i, feat in enumerate(feat_list)]
        inner_feat_list.reverse()

        out_feat_list = [self.fuse_3x3convs[0](inner_feat_list[0])]
        for i in range(len(feat_list) - 1):
            inner = self.top_down(out_feat_list[i], inner_feat_list[i+1])
            out = self.fuse_3x3convs[i + 1](inner)
            out_feat_list.append(out)

        
        final_feat_cla = out_feat_list[-1]

        

        # rec_feat = final_feat_rec

        for i in range(len(self.fuse_3x3convs_classify)):
            final_feat_cla = self.fuse_3x3convs_classify[i](final_feat_cla)


        
        logit = self.cls_pred_conv(final_feat_cla)
        if self.training:
            # reconstruct
            feat_list_rec = []
            for op in self.feature_ops:
                x_source_train_rec = op(x_source_train_rec)
                if isinstance(op, nn.Identity):
                    feat_list_rec.append(x_source_train_rec)

            inner_feat_rec_list = [self.reduce_1x1convs[i](feat) for i, feat in enumerate(feat_list_rec)]
            inner_feat_rec_list.reverse()

            out_feat_rec_list = [self.fuse_3x3convs[0](inner_feat_rec_list[0])]
            for i in range(len(feat_list) - 1):
                inner = self.top_down(out_feat_rec_list[i], inner_feat_rec_list[i+1])
                out = self.fuse_3x3convs[i + 1](inner)
                out_feat_rec_list.append(out)

            final_feat_rec = out_feat_rec_list[-1]

            for i in range(len(self.fuse_3x3convs_reconstruct)):
                final_feat_rec = self.fuse_3x3convs_reconstruct[i](final_feat_rec)

            rec_res = self.reconstruct(final_feat_rec)
            # print('ok')

            loss_dict = {
                'cls_loss': self.loss(logit, y, w),
                'rex_loss': l1loss1(rec_res, x_target_train)
            }
            return loss_dict

        # reconstruct test
        if not self.training:
            
            # # reconstruct test
            feat_list_reconstruct_test = []
                        # feat_list_rec = []
            for op in self.feature_ops:
                x_source_test_rec = op(x_source_test_rec)
                if isinstance(op, nn.Identity):
                    feat_list_reconstruct_test.append(x_source_test_rec)

            inner_feat_rec_test_list = [self.reduce_1x1convs[i](feat) for i, feat in enumerate(feat_list_reconstruct_test)]
            inner_feat_rec_test_list.reverse()

            out_feat_rec_test_list = [self.fuse_3x3convs[0](inner_feat_rec_test_list[0])]
            for i in range(len(feat_list) - 1):
                inner = self.top_down(out_feat_rec_test_list[i], inner_feat_rec_test_list[i+1])
                out = self.fuse_3x3convs[i + 1](inner)
                out_feat_rec_test_list.append(out)

            final_feat_reconstruct_test = out_feat_rec_test_list[-1]

            for i in range(len(self.fuse_3x3convs_reconstruct)):
                final_feat_reconstruct_test = self.fuse_3x3convs_reconstruct[i](final_feat_reconstruct_test)

            # rec_res = self.reconstruct(final_feat_rec)
            x_reconstruct_test_pred = self.reconstruct(final_feat_reconstruct_test)
        print('Reconstruction PSNR:' + str(PSNR(x_reconstruct_test_pred, x_target_test)))
        return torch.softmax(logit, dim=1)

    def loss(self, x, y, weight):
        losses = F.cross_entropy(x, y.long() - 1, weight=None,
                                 ignore_index=-1, reduction='none')

        v = losses.mul_(weight).sum() / weight.sum()
        return v

    def set_defalut_config(self):
        self.config.update(dict(
            in_channels=204,
            num_classes=16,
            block_channels=(96, 128, 192, 256),
            num_blocks=(1, 1, 1, 1),
            inner_dim=128,
            reduction_ratio=1.0,
        ))

mrae = []
psnr = []
l1_loss_l_1 = []
l1_loss_l = []
cls_loss_l = []


def padWithZeros(Xc, margin=2):
    newX = torch.zeros((Xc.shape[0] + 2 * margin, Xc.shape[1] + 2 * margin, Xc.shape[2])).cuda()
    x_offset = margin
    y_offset = margin
    newX[x_offset:Xc.shape[0] + x_offset, y_offset:Xc.shape[1] + y_offset, :] = Xc
    return newX



def normalize(trainx):
    """Normalize and returns the calculated means and stds for each band"""
    trainxn = trainx
    if trainx.ndim == 5:
        dim = trainx.shape[2]
    else:
        dim = trainx.shape[3]

    means = np.zeros((dim, 1))
    stds = np.zeros((dim, 1))
    for n in range(dim):
        if trainx.ndim == 5:  # Apply normalization to the data that is already in Pytorch format
            means[n,] = np.mean(trainxn[:, :, n, :, :])
            stds[n,] = np.std(trainxn[:, :, n, :, :])
            trainxn[:, :, n, :, :] = (trainxn[:, :, n, :, :] - means[n,]) / (stds[n,])
        elif trainx.ndim == 4:
            means[n,] = np.mean(trainxn[:, :, :, n])
            stds[n,] = np.std(trainxn[:, :, :, n])
            trainxn[:, :, :, n] = (trainxn[:, :, :, n] - means[n,]) / (stds[n,])
    return trainxn, means, stds

def entropy(labels, base=2):
    value, counts = np.unique(labels, return_counts=True)
    norm_counts = counts / counts.sum()
    return -(norm_counts * np.log(norm_counts) / np.log(base)).sum()

def entropy_torch(labels, base=torch.tensor(2)):
    value, counts = torch.unique(labels, return_counts=True)
    norm_counts = counts / counts.sum()
    return -(norm_counts * torch.log(norm_counts) / torch.log(base)).sum()


@registry.MODEL.register('FreeNetBS_SV')
class FreeNetBS_SV(CVModule):
    def __init__(self, config, time_ = 0, start_cls = 0, turn = False, num = 1,\
                    selection = None,
                    bestselection = None,
                    f1base = None,
                    f1 = None,
                    indexes = None,
                    preselected = None,
                    mask = None,
                    changed_f1base = 100):
        
        super(FreeNetBS_SV, self).__init__(config)
        self.time_ = time_
        self.start_cls = start_cls
    
        self.turn = turn
        self.num = num
        self.selection = selection
        self.bestselection = bestselection
        self.f1base = f1base
        self.f1 = f1
        self.indexes = indexes
        self.preselected = preselected
        self.mask = mask
        self.changed_f1base = changed_f1base
        self.X_full = None
        self.Y_full = None
        r = int(16 * self.config.reduction_ratio)
        
        block1_channels = int(self.config.block_channels[0] * self.config.reduction_ratio / r) * r
        block2_channels = int(self.config.block_channels[1] * self.config.reduction_ratio / r) * r
        block3_channels = int(self.config.block_channels[2] * self.config.reduction_ratio / r) * r
        block4_channels = int(self.config.block_channels[3] * self.config.reduction_ratio / r) * r
        # print(self.config.in_channels)
        self.feature_ops = nn.ModuleList([
            conv3x3_gn_relu(self.config.hidden_channels, block1_channels, r),

            repeat_block(block1_channels, r, self.config.num_blocks[0]),
            nn.Identity(),
            downsample2x(block1_channels, block2_channels),

            repeat_block(block2_channels, r, self.config.num_blocks[1]),
            nn.Identity(),
            downsample2x(block2_channels, block3_channels),

            repeat_block(block3_channels, r, self.config.num_blocks[2]),
            nn.Identity(),
            downsample2x(block3_channels, block4_channels),

            repeat_block(block4_channels, r, self.config.num_blocks[3]),
            nn.Identity(),
        ])
        inner_dim = int(self.config.inner_dim * self.config.reduction_ratio)
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
        # self.batch_size = self.config.rec_bs
        self.in_dim = self.config.in_channels
        self.out_dim = self.config.out_channels
        self.resrf = nn.Parameter(torch.rand((self.out_dim, self.in_dim)))
        self.cls_pred_conv = nn.Conv2d(inner_dim, self.config.num_classes, 1)
        
        self.srf = LyotFilter(self.config.in_channels, self.config.out_channels,  self.config.minBand, self.config.maxBand, self.config.nBandDataset, self.config.dataset)
        # self.resrf = reLyotFilter(self.config.in_channels, self.config.out_channels,  self.config.minBand, self.config.maxBand, self.config.nBandDataset, self.config.dataset)
        self.dynamic1 = S3KAIResNet(self.config.out_channels, self.config.num_classes, 2)# nn.Conv2d(self.config.out_channels, self.config.out_channels * self.config.out_channels, 1, 1, 0)#
        self.dynamic2 = nn.Conv2d(self.config.out_channels, self.config.in_channels, 1, 1, 0)#  S3KAIResNet(self.config.out_channels, self.config.num_classes, 2)# 
        self.dynamic3 = RSSAN5(self.config.num_classes, self.config.out_channels, 3, self.config.out_channels, 1, 1, 1, self.config.hidden_channels)
        self.dynamic4 = SEBlock(self.config.out_channels, 2)

        self.reconstruction1 = MST_Plus_Plus(self.config.out_channels, self.config.hidden_channels,  self.config.hidden_channels, 3)
        self.reconstruction2 = BiSRNet(self.config.out_channels, self.config.hidden_channels, self.config.hidden_channels)
        self.reconstruction4 = nn.Conv2d(self.config.out_channels, self.config.hidden_channels, 3, 1, 1)#  S3KAIResNet(self.config.out_channels, self.config.num_classes, 2)# 

        

    # def entropy(labels, base=2):
    #     value, counts = torch.unique(labels, return_counts=True)
    #     norm_counts = counts / counts.sum()
    #     return -(norm_counts * torch.log(norm_counts) / torch.log(base)).sum()

    def top_down(self, top, lateral):
        top2x = F.interpolate(top, scale_factor=2.0, mode='nearest')
        return lateral + top2x # 
    
    def tune_BFS(self, x, mask=None, w=None, **kwargs):
        x = x[:, self.selection, :, :]

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

        logit = self.cls_pred_conv(final_feat)

        y_pred = torch.softmax(logit, dim=1).squeeze(0)

        y_pred = y_pred.argmax(dim=0) + 1

        w = w.byte()

        mask = torch.masked_select(mask.view(-1), w.bool().view(-1))
        y_pred = torch.masked_select(y_pred.view(-1), w.bool().view(-1))
            
        oa = metric.th_overall_accuracy_score(mask.view(-1), y_pred.view(-1))
        aa, acc_per_class = metric.th_average_accuracy_score(mask.view(-1), y_pred.view(-1),
                                                                 len(torch.unique(mask).tolist()),
                                                                 return_accuracys=True)
        kappa = metric.th_cohen_kappa_score(mask.view(-1).detach().cpu(), y_pred.view(-1).detach().cpu(), len(torch.unique(mask).tolist()))

        return (oa.item() + aa.item() + kappa.item()) / 3.0 #.squeeze(), final_feat

    def tune_DFS(self):
        """Get the mean F1 validation score using a set of selected bands"""
        # X, Y = loadata(name='WHLK')
        # print('Initial image shape: ' + str(X.shape))
        # # X, Y = createImageCubes(X, Y, window=5)
        # # print('Processed dataset shape: ' + str(X.shape))

        # dataset = Dataset(train_x=X, train_y=Y, ind=False, name='WHLK')
        # selector = SelectBands(dataset=dataset,  method='GSS', nbands=20)
        # VIF_best, IBRA_best, GSS_best, stats_best = selector.run_selection(init_vf=11, final_vf=11, preselected_bands=X.shape[2], config = self.config_)
        
        # torch.backends.cudnn.benchmark = True
        print('Start testing candidate selection: ' + str(self.selection))
        avg = 0
        SEED = [1, 2, 3, 4, 5]
        # torch.manual_seed(SEED)
        # torch.cuda.manual_seed(SEED)
        for i in range(len(SEED)):
            res = train.run(config_path='freenet.sv.freenet_1_0_salinas_test',
                model_dir='./log/freenet_1_0_salinas_test/freenet/1.0_poly',
                cpu_mode=False,
                after_construct_launcher_callbacks=[register_evaluate_fn],
                opts=None,
                np_seed=SEED[i],
                selection=self.selection) # freenet_1_0_salinas_test
            avg = avg + res['launcher'].avg
            # if res['launcher'].avg < 0.8:
            #     break
        # print('ok')
        return avg / 5.0
        

    def checkMulticollinearity(self, s=None, trainx=None):
        """Calculate the VIF value of each selected band in s"""
        vifV = []
        nbands = len(s)
        trainx, _, _ = normalize(trainx)
        for n, i in enumerate(s):
            y = trainx[:, 0, np.where(np.array(self.indexes) == i)[0][0], :, :]
            x = np.zeros((trainx.shape[0], trainx.shape[3], trainx.shape[4], nbands - 1))
            c = 0
            for nb in s:
                if nb != i:
                    x[:, :, :, c] = trainx[:, 0, np.where(np.array(self.indexes) == nb)[0][0], :, :]
                    c += 1
            x = x.reshape((x.shape[0] * x.shape[1] * x.shape[2], nbands - 1))
            y = y.reshape((y.shape[0] * y.shape[1] * y.shape[2], 1))
            model = sm.OLS(y, x)
            results = model.fit()
            rsq = results.rsquared
            vifV.append(round(1 / (1 - rsq), 2))
            # print("R Square value of {} band is {} keeping all other bands as features".format(s[n],
            #                                                                                    (round(rsq, 4))))
            # print("\t\t\tMulticolinearity analysis. Variance Inflation Factor of band {} is {}".format(s[n], vifV[n]))

        return vifV
    
    def forward(self, x, y=None, w=None, selection=None, **kwargs):
        x_hsi = x
        
        self.time_ += 1

        if self.time_ == 1:
            self.X_full = x.permute(0, 2, 3, 1).squeeze(0)
            self.Y_full = y.squeeze(0)
            # self.w_full = torch.load('/mnt/c/Users/17735/OneDrive/桌面/FreeNet-master/full_w.pth').cuda()
            trainx_hsi, _ = createImageCubes(self.X_full.cpu(), self.Y_full.cpu())
            trainx_hsi = trainx_hsi.reshape(trainx_hsi.shape[0], 1, trainx_hsi.shape[3], trainx_hsi.shape[1], trainx_hsi.shape[2])
            trainx_hsi, _, _ = normalize(trainx_hsi)
            self.indexes = [i for i in range(x.shape[1])]
            entropies =  [entropy(trainx_hsi[:, :, i, :, :]) for i in range(len(self.indexes))]

            # Sort the pre-selected bands according to their entropy (in decreasing order)
            self.preselected = self.indexes.copy()
            pairs = list(tuple(zip(self.preselected, entropies)))
            pairs.sort(key=lambda x: x[1], reverse=True)
            self.preselected, _ = zip(*pairs)

            # Select the first "select" bands
            self.preselected = list(self.preselected)
            self.selection = self.preselected[:self.config.hidden_channels]
            self.selection.sort()
            self.preselected = self.preselected[self.config.hidden_channels:]
            # self.selection = [0, 74, 75, 98, 101, 109, 114, 116, 125, 128, 131, 132, 136, 241, 242, 243, 244, 247, 248, 249]
            ct = 1
            # print("\tAnalyzing candidate combination " + str(ct) + ". 5x2 CV using bands: " + str(selection))
            # self.f1base = self.tune_DFS()
            # print("\tMean F1: " + str(self.f1base))
            
            self.bestselection = self.selection.copy()
            ct += 1
        # if self.training:
        #     # print('Current F1:' + str(self.tune(self.x_full, self.mask_full, self.w_full)))
        #     if self.tune_DFS() < 0.6:
        #         self.turn = True
        #     else:
        #         self.turn = False
    
        # self.selection = self.tune_DFS()
        x = x_hsi[:, self.selection, :, :]
        
        if self.training and self.num > 0: #  and self.turn and self.num > 0 and len(self.preselected) > 0
            self.num = self.num - 1
            
            
            # Try new bands until there is no more elements in the list
            # while len(self.preselected) > 0:
                
            #     # Calculate the maximum VIF of all the band in "selection"

            #     VIF = self.checkMulticollinearity(s=self.selection, trainx=trainx_hsi) # (1, 94890, 1, 1, 20)
            #     # Remove the band with the highest VIF of "selection"
            #     self.selection.remove(self.selection[VIF.index(max(VIF))])
            #     # Pop the next available band from "preselected"
            #     self.selection.append(self.preselected[0])
            #     self.selection.sort()
            #     self.preselected = self.preselected[1:]

            #     # Train using the bands in "selection"
            #     print("\tAnalyzing candidate combination " + str(ct) + ". 5x2 CV using bands: " + str(self.selection))
            #     self.f1 = self.tune_DFS()
            #     with open("/mnt/sdd/niuyuanzhuo/FreeNet-master/log1_SV.txt", encoding="utf-8",mode="a") as file:  
            #         file.write(str(self.selection) + '  ' + str(self.f1) +'\n')
                
            #     print("\tMean F1: " + str(self.f1))
            #     # Check if the new selection has better performance than the previous one. If not, break
            #     if self.f1 > self.f1base:
            #         self.bestselection = self.selection.copy()
            #         self.f1base = self.f1
            #     #     self.changed = 50

            #     # elif self.f1base > 0.9 or self.changed < 0:
            #     #     break
            #     # else:
            #     #     self.changed = self.changed - 1
            #     ct += 1
            #     print("\tBest selection so far: " + str(self.bestselection) + "with an F1 score of " + str(self.f1base))
        
        # if not self.turn:
        # x = x_hsi[:, [0, 74, 75, 98, 101, 109, 114, 116, 125, 128, 131, 132, 136, 241, 242, 243, 244, 247, 248, 249], :, :]
        print("\tBest selection so far: " + str(self.bestselection) + "with an F1 score of " + str(self.f1base))
        x = x_hsi[:, self.bestselection, :, :]
        feat_list = []
        for op in self.feature_ops:
            x = op(x)
            if isinstance(op, nn.Identity):
                feat_list.append(x)

        inner_feat_list = [self.reduce_1x1convs[i](feat) for i, feat in enumerate(feat_list)]
        inner_feat_list.reverse()

        out_feat_list = [self.fuse_3x3convs[0](inner_feat_list[0])]
        for i in range(len(feat_list) - 1):
            inner = self.top_down(out_feat_list[i], inner_feat_list[i+1])
            out = self.fuse_3x3convs[i + 1](inner)
            out_feat_list.append(out)

        final_feat = out_feat_list[-1]

        logit = self.cls_pred_conv(final_feat)
        
        
        if self.training:
            loss_dict = {
                'cls_loss': self.loss(logit, y, w),
            }

            return loss_dict
          

        return torch.softmax(logit, dim=1)
    

    def loss(self, x, y, weight):
        losses = F.cross_entropy(x, y.long() - 1, weight=None,
                                 ignore_index=-1, reduction='none')

        v = losses.mul_(weight).sum() / weight.sum()
        return v

    def set_defalut_config(self):
        self.config.update(dict(
            in_channels=204,
            num_classes=16,
            block_channels=(96, 128, 192, 256),
            num_blocks=(1, 1, 1, 1),
            inner_dim=128,
            reduction_ratio=1.0,
        ))


@registry.MODEL.register('FreeNetBS_LK')
class FreeNetBS_LK(CVModule):
    def __init__(self, config, time_ = 0, start_cls = 0, turn = False, num = 1,\
                    selection = None,
                    bestselection = None,
                    f1base = None,
                    f1 = None,
                    indexes = None,
                    preselected = None,
                    mask = None,
                    changed_f1base = 100):
        
        super(FreeNetBS_LK, self).__init__(config)
        self.time_ = time_
        self.start_cls = start_cls
    
        self.turn = turn
        self.num = num
        self.selection = selection
        self.bestselection = bestselection
        self.f1base = f1base
        self.f1 = f1
        self.indexes = indexes
        self.preselected = preselected
        self.mask = mask
        self.changed = 10
        self.changed_f1base = changed_f1base
        self.X_full = None
        self.Y_full = None
        r = int(16 * self.config.reduction_ratio)
        
        block1_channels = int(self.config.block_channels[0] * self.config.reduction_ratio / r) * r
        block2_channels = int(self.config.block_channels[1] * self.config.reduction_ratio / r) * r
        block3_channels = int(self.config.block_channels[2] * self.config.reduction_ratio / r) * r
        block4_channels = int(self.config.block_channels[3] * self.config.reduction_ratio / r) * r
        # print(self.config.in_channels)
        self.feature_ops = nn.ModuleList([
            conv3x3_gn_relu(self.config.hidden_channels, block1_channels, r),

            repeat_block(block1_channels, r, self.config.num_blocks[0]),
            nn.Identity(),
            downsample2x(block1_channels, block2_channels),

            repeat_block(block2_channels, r, self.config.num_blocks[1]),
            nn.Identity(),
            downsample2x(block2_channels, block3_channels),

            repeat_block(block3_channels, r, self.config.num_blocks[2]),
            nn.Identity(),
            downsample2x(block3_channels, block4_channels),

            repeat_block(block4_channels, r, self.config.num_blocks[3]),
            nn.Identity(),
        ])
        inner_dim = int(self.config.inner_dim * self.config.reduction_ratio)
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
        # self.batch_size = self.config.rec_bs
        self.in_dim = self.config.in_channels
        self.out_dim = self.config.out_channels
        self.resrf = nn.Parameter(torch.rand((self.out_dim, self.in_dim)))
        self.cls_pred_conv = nn.Conv2d(inner_dim, self.config.num_classes, 1)
        
        self.srf = LyotFilter(self.config.in_channels, self.config.out_channels,  self.config.minBand, self.config.maxBand, self.config.nBandDataset, self.config.dataset)
        # self.resrf = reLyotFilter(self.config.in_channels, self.config.out_channels,  self.config.minBand, self.config.maxBand, self.config.nBandDataset, self.config.dataset)
        self.dynamic1 = S3KAIResNet(self.config.out_channels, self.config.num_classes, 2)# nn.Conv2d(self.config.out_channels, self.config.out_channels * self.config.out_channels, 1, 1, 0)#
        self.dynamic2 = nn.Conv2d(self.config.out_channels, self.config.in_channels, 1, 1, 0)#  S3KAIResNet(self.config.out_channels, self.config.num_classes, 2)# 
        self.dynamic3 = RSSAN5(self.config.num_classes, self.config.out_channels, 3, self.config.out_channels, 1, 1, 1, self.config.hidden_channels)
        self.dynamic4 = SEBlock(self.config.out_channels, 2)

        self.reconstruction1 = MST_Plus_Plus(self.config.out_channels, self.config.hidden_channels,  self.config.hidden_channels, 3)
        self.reconstruction2 = BiSRNet(self.config.out_channels, self.config.hidden_channels, self.config.hidden_channels)
        self.reconstruction4 = nn.Conv2d(self.config.out_channels, self.config.hidden_channels, 3, 1, 1)#  S3KAIResNet(self.config.out_channels, self.config.num_classes, 2)# 

        

    # def entropy(labels, base=2):
    #     value, counts = torch.unique(labels, return_counts=True)
    #     norm_counts = counts / counts.sum()
    #     return -(norm_counts * torch.log(norm_counts) / torch.log(base)).sum()

    def top_down(self, top, lateral):
        top2x = F.interpolate(top, scale_factor=2.0, mode='nearest')
        return lateral + top2x # 
    
    def tune_BFS(self, x, mask=None, w=None, **kwargs):
        x = x[:, self.selection, :, :]

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

        logit = self.cls_pred_conv(final_feat)

        y_pred = torch.softmax(logit, dim=1).squeeze(0)

        y_pred = y_pred.argmax(dim=0) + 1

        w = w.byte()

        mask = torch.masked_select(mask.view(-1), w.bool().view(-1))
        y_pred = torch.masked_select(y_pred.view(-1), w.bool().view(-1))
            
        oa = metric.th_overall_accuracy_score(mask.view(-1), y_pred.view(-1))
        aa, acc_per_class = metric.th_average_accuracy_score(mask.view(-1), y_pred.view(-1),
                                                                 len(torch.unique(mask).tolist()),
                                                                 return_accuracys=True)
        kappa = metric.th_cohen_kappa_score(mask.view(-1).detach().cpu(), y_pred.view(-1).detach().cpu(), len(torch.unique(mask).tolist()))

        return (oa.item() + aa.item() + kappa.item()) / 3.0 #.squeeze(), final_feat

    def tune_DFS(self):
        """Get the mean F1 validation score using a set of selected bands"""
        # X, Y = loadata(name='WHLK')
        # print('Initial image shape: ' + str(X.shape))
        # # X, Y = createImageCubes(X, Y, window=5)
        # # print('Processed dataset shape: ' + str(X.shape))

        # dataset = Dataset(train_x=X, train_y=Y, ind=False, name='WHLK')
        # selector = SelectBands(dataset=dataset,  method='GSS', nbands=20)
        # VIF_best, IBRA_best, GSS_best, stats_best = selector.run_selection(init_vf=11, final_vf=11, preselected_bands=X.shape[2], config = self.config_)
        
        # torch.backends.cudnn.benchmark = True
        print('Start testing candidate selection: ' + str(self.selection))
        avg = 0
        SEED = [1, 2, 3, 4, 5]
        # torch.manual_seed(SEED)
        # torch.cuda.manual_seed(SEED)
        for i in range(len(SEED)):
            res = train.run(config_path='freenet.whlk.freenet_1_0_whlk_test',
                model_dir='./log/freenet_1_0_whlk_test/freenet/1.0_poly',
                cpu_mode=False,
                after_construct_launcher_callbacks=[register_evaluate_fn],
                opts=None,
                np_seed=SEED[i],
                selection=self.selection) # freenet_1_0_whlk_test
            avg = avg + res['launcher'].avg
            # if res['launcher'].avg < 0.8:
            #     break
        # print('ok')
        return avg / 5.0
        

    def checkMulticollinearity(self, s=None, trainx=None):
        """Calculate the VIF value of each selected band in s"""
        vifV = []
        nbands = len(s)
        trainx, _, _ = normalize(trainx)
        for n, i in enumerate(s):
            y = trainx[:, 0, np.where(np.array(self.indexes) == i)[0][0], :, :]
            x = np.zeros((trainx.shape[0], trainx.shape[3], trainx.shape[4], nbands - 1))
            c = 0
            for nb in s:
                if nb != i:
                    x[:, :, :, c] = trainx[:, 0, np.where(np.array(self.indexes) == nb)[0][0], :, :]
                    c += 1
            x = x.reshape((x.shape[0] * x.shape[1] * x.shape[2], nbands - 1))
            y = y.reshape((y.shape[0] * y.shape[1] * y.shape[2], 1))
            model = sm.OLS(y, x)
            results = model.fit()
            rsq = results.rsquared
            vifV.append(round(1 / (1 - rsq), 2))
            # print("R Square value of {} band is {} keeping all other bands as features".format(s[n],
            #                                                                                    (round(rsq, 4))))
            # print("\t\t\tMulticolinearity analysis. Variance Inflation Factor of band {} is {}".format(s[n], vifV[n]))

        return vifV
    
    def forward(self, x, y=None, w=None, selection=None, **kwargs):
        x_hsi = x
        
        self.time_ += 1

        if self.time_ == 1:
            self.X_full = x.permute(0, 2, 3, 1).squeeze(0)
            self.Y_full = y.squeeze(0)
            # self.w_full = torch.load('/mnt/c/Users/17735/OneDrive/桌面/FreeNet-master/full_w.pth').cuda()
            trainx_hsi, _ = createImageCubes(self.X_full.cpu(), self.Y_full.cpu())
            trainx_hsi = trainx_hsi.reshape(trainx_hsi.shape[0], 1, trainx_hsi.shape[3], trainx_hsi.shape[1], trainx_hsi.shape[2])
            trainx_hsi, _, _ = normalize(trainx_hsi)
            self.indexes = [i for i in range(x.shape[1])]
            entropies =  [entropy(trainx_hsi[:, :, i, :, :]) for i in range(len(self.indexes))]

            # Sort the pre-selected bands according to their entropy (in decreasing order)
            self.preselected = self.indexes.copy()
            pairs = list(tuple(zip(self.preselected, entropies)))
            pairs.sort(key=lambda x: x[1], reverse=True)
            self.preselected, _ = zip(*pairs)

            # Select the first "select" bands
            self.preselected = list(self.preselected)
            self.selection = self.preselected[:self.config.hidden_channels]
            self.selection.sort()
            self.preselected = self.preselected[self.config.hidden_channels:]
            # self.selection = [0, 74, 75, 98, 101, 109, 114, 116, 125, 128, 131, 132, 136, 241, 242, 243, 244, 247, 248, 249]
            ct = 1
            # print("\tAnalyzing candidate combination " + str(ct) + ". 5x2 CV using bands: " + str(selection))
            # self.f1base = self.tune_DFS()
            # print("\tMean F1: " + str(self.f1base))
            
            self.bestselection = self.selection.copy()
            ct += 1
        # if self.training:
        #     # print('Current F1:' + str(self.tune(self.x_full, self.mask_full, self.w_full)))
        #     if self.tune_DFS() < 0.6:
        #         self.turn = True
        #     else:
        #         self.turn = False
    
        # self.selection = self.tune_DFS()
        x = x_hsi[:, self.selection, :, :]
        
        if self.training and self.num > 0: #  and self.turn and self.num > 0 and len(self.preselected) > 0
            self.num = self.num - 1
            
            
            # Try new bands until there is no more elements in the list
            # while len(self.preselected) > 0:
                
            #     # Calculate the maximum VIF of all the band in "selection"

            #     VIF = self.checkMulticollinearity(s=self.selection, trainx=trainx_hsi) # (1, 94890, 1, 1, 20)
            #     # Remove the band with the highest VIF of "selection"
            #     self.selection.remove(self.selection[VIF.index(max(VIF))])
            #     # Pop the next available band from "preselected"
            #     self.selection.append(self.preselected[0])
            #     self.selection.sort()
            #     self.preselected = self.preselected[1:]

            #     # Train using the bands in "selection"
            #     print("\tAnalyzing candidate combination " + str(ct) + ". 5x2 CV using bands: " + str(self.selection))
            #     self.f1 = self.tune_DFS()
            #     # with open("/mnt/sdd/niuyuanzhuo/FreeNet-master/log2_LK.txt", encoding="utf-8",mode="a") as file:  
            #     #     file.write(str(self.selection) + '  ' + str(self.f1) +'\n')
                
            #     print("\tMean F1: " + str(self.f1))
            #     # Check if the new selection has better performance than the previous one. If not, break
            #     if self.f1 > self.f1base:
            #         self.bestselection = self.selection.copy()
            #         self.f1base = self.f1
            #     #     self.changed = 50

            #     elif self.f1base > 0.9 or self.changed < 0:
            #         break
            #     else:
            #         self.changed = self.changed - 1
            #     ct += 1
            #     print("\tBest selection so far: " + str(self.bestselection) + "with an F1 score of " + str(self.f1base))
        
        # if not self.turn:
        # x = x_hsi[:, [0, 74, 75, 98, 101, 109, 114, 116, 125, 128, 131, 132, 136, 241, 242, 243, 244, 247, 248, 249], :, :]
        print("\tBest selection so far: " + str(self.bestselection) + "with an F1 score of " + str(self.f1base))
        x = x_hsi[:, self.bestselection, :, :]
        feat_list = []
        for op in self.feature_ops:
            x = op(x)
            if isinstance(op, nn.Identity):
                feat_list.append(x)

        inner_feat_list = [self.reduce_1x1convs[i](feat) for i, feat in enumerate(feat_list)]
        inner_feat_list.reverse()

        out_feat_list = [self.fuse_3x3convs[0](inner_feat_list[0])]
        for i in range(len(feat_list) - 1):
            inner = self.top_down(out_feat_list[i], inner_feat_list[i+1])
            out = self.fuse_3x3convs[i + 1](inner)
            out_feat_list.append(out)

        final_feat = out_feat_list[-1]

        logit = self.cls_pred_conv(final_feat)
        
        
        if self.training:
            loss_dict = {
                'cls_loss': self.loss(logit, y, w),
            }

            return loss_dict
          

        return torch.softmax(logit, dim=1)
    

    def loss(self, x, y, weight):
        losses = F.cross_entropy(x, y.long() - 1, weight=None,
                                 ignore_index=-1, reduction='none')

        v = losses.mul_(weight).sum() / weight.sum()
        return v

    def set_defalut_config(self):
        self.config.update(dict(
            in_channels=204,
            num_classes=16,
            block_channels=(96, 128, 192, 256),
            num_blocks=(1, 1, 1, 1),
            inner_dim=128,
            reduction_ratio=1.0,
        ))

@registry.MODEL.register('FreeNetFewDynamicChannelsVoltage2stage')
class FreeNetFewDynamicChannelsVoltage2stage(CVModule):
    def __init__(self, config):
        super(FreeNetFewDynamicChannelsVoltage2stage, self).__init__(config)
        r = int(16 * self.config.reduction_ratio)
        block1_channels = int(self.config.block_channels[0] * self.config.reduction_ratio / r) * r
        block2_channels = int(self.config.block_channels[1] * self.config.reduction_ratio / r) * r
        block3_channels = int(self.config.block_channels[2] * self.config.reduction_ratio / r) * r
        block4_channels = int(self.config.block_channels[3] * self.config.reduction_ratio / r) * r
        # print(self.config.in_channels)
        self.feature_ops = nn.ModuleList([
            conv3x3_gn_relu(self.config.hidden_channels, block1_channels, r),

            repeat_block(block1_channels, r, self.config.num_blocks[0]),
            nn.Identity(),
            downsample2x(block1_channels, block2_channels),

            repeat_block(block2_channels, r, self.config.num_blocks[1]),
            nn.Identity(),
            downsample2x(block2_channels, block3_channels),

            repeat_block(block3_channels, r, self.config.num_blocks[2]),
            nn.Identity(),
            downsample2x(block3_channels, block4_channels),

            repeat_block(block4_channels, r, self.config.num_blocks[3]),
            nn.Identity(),
        ])
        inner_dim = int(self.config.inner_dim * self.config.reduction_ratio)
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
        # self.batch_size = self.config.rec_bs
        self.in_dim = self.config.in_channels
        self.out_dim = self.config.out_channels
        self.resrf = nn.Parameter(torch.rand((self.out_dim, self.in_dim)))
        self.cls_pred_conv = nn.Conv2d(inner_dim, self.config.num_classes, 1)
        
        self.srf = LyotFilter(self.config.in_channels, self.config.out_channels,  self.config.minBand, self.config.maxBand, self.config.nBandDataset, self.config.dataset)
        # self.resrf = reLyotFilter(self.config.in_channels, self.config.out_channels,  self.config.minBand, self.config.maxBand, self.config.nBandDataset, self.config.dataset)
        self.dynamic1 = S3KAIResNet(self.config.out_channels, self.config.num_classes, 2)# nn.Conv2d(self.config.out_channels, self.config.out_channels * self.config.out_channels, 1, 1, 0)#
        self.dynamic2 = nn.Conv2d(self.config.out_channels, self.config.in_channels, 1, 1, 0)#  S3KAIResNet(self.config.out_channels, self.config.num_classes, 2)# 
        self.dynamic3 = RSSAN5(self.config.num_classes, self.config.out_channels, 3, self.config.out_channels, 1, 1, 1, self.config.hidden_channels)
        self.dynamic4 = SEBlock(self.config.out_channels, 2)

        self.reconstruction1 = MST_Plus_Plus(self.config.out_channels, self.config.hidden_channels,  self.config.hidden_channels, 3)
        self.reconstruction2 = BiSRNet(self.config.out_channels, self.config.hidden_channels, self.config.hidden_channels)
        # self.reconstruction3 = nn.Conv3d(
        #             in_channels=self.config.out_channels,
        #             out_channels=self.config.hidden_channels,
        #             kernel_size=(3, 1, 1),
        #             stride=(1, 1, 1),
        #             padding=(1, 0, 0))
        self.reconstruction4 = nn.Conv2d(self.config.out_channels, self.config.hidden_channels, 3, 1, 1)#  S3KAIResNet(self.config.out_channels, self.config.num_classes, 2)# 

        self.herosnet = HerosNet(Ch=self.config.out_channels, stages=8)
        self.GST = GST_MODEL(in_ch=self.config.out_channels,
                           out_ch=self.config.out_channels,
                           noise_mean=0.1,
                           noise_std=0.1,
                           init="normal",
                        #    conv = default_conv,
                           noise_act=nn.Softplus(),
                           inter_channels=10,
                           spatial_scale=4)
        self.time_ = 0
        self.BS = PCA(n_components=self.config.hidden_channels)
        self.start_cls = 0
        self.changed_f1base = 10
        self.turn = False
        self.pre_rec_x = None
        self.pre_x_middle = None
        self.num = 1
        self.num_to_stop = 30
        self.indexes = []
        

    # def entropy(labels, base=2):
    #     value, counts = np.unique(labels, return_counts=True)
    #     norm_counts = counts / counts.sum()
    #     return -(norm_counts * np.log(norm_counts) / np.log(base)).sum()
    def checkMulticollinearity(self, s=None, trainx=None):
        """Calculate the VIF value of each selected band in s"""
        vifV = []
        nbands = len(s)
        trainx, _, _ = normalize(trainx)
        for n, i in enumerate(s):
            y = trainx[:, 0, np.where(np.array(self.indexes) == i)[0][0], :, :]
            x = np.zeros((trainx.shape[0], trainx.shape[3], trainx.shape[4], nbands - 1))
            c = 0
            for nb in s:
                if nb != i:
                    x[:, :, :, c] = trainx[:, 0, np.where(np.array(self.indexes) == nb)[0][0], :, :]
                    c += 1
            x = x.reshape((x.shape[0] * x.shape[1] * x.shape[2], nbands - 1))
            y = y.reshape((y.shape[0] * y.shape[1] * y.shape[2], 1))
            model = sm.OLS(y, x)
            results = model.fit()
            rsq = results.rsquared
            vifV.append(round(1 / (1 - rsq), 2))
            # print("R Square value of {} band is {} keeping all other bands as features".format(s[n],
            #                                                                                    (round(rsq, 4))))
            # print("\t\t\tMulticolinearity analysis. Variance Inflation Factor of band {} is {}".format(s[n], vifV[n]))

        return vifV
    
    def tune_DFS(self):
        """Get the mean F1 validation score using a set of selected bands"""
        # X, Y = loadata(name='WHLK')
        # print('Initial image shape: ' + str(X.shape))
        # # X, Y = createImageCubes(X, Y, window=5)
        # # print('Processed dataset shape: ' + str(X.shape))

        # dataset = Dataset(train_x=X, train_y=Y, ind=False, name='WHLK')
        # selector = SelectBands(dataset=dataset,  method='GSS', nbands=20)
        # VIF_best, IBRA_best, GSS_best, stats_best = selector.run_selection(init_vf=11, final_vf=11, preselected_bands=X.shape[2], config = self.config_)
        
        # torch.backends.cudnn.benchmark = True
        print('Start testing candidate selection: ' + str(self.selection))
        avg = 0
        SEED = [1, 2, 3, 4, 5]
        # torch.manual_seed(SEED)
        # torch.cuda.manual_seed(SEED)
        for i in range(len(SEED)):
            res = train.run(config_path='freenet.whlk.freenet_1_0_whlk_test',
                model_dir='./log/freenet_1_0_whlk_test/freenet/1.0_poly',
                cpu_mode=False,
                after_construct_launcher_callbacks=[register_evaluate_fn],
                opts=None,
                np_seed=SEED[i],
                selection=self.selection) # freenet_1_0_salinas_test
            avg = avg + res['launcher'].avg
            # if res['launcher'].avg < 0.8:
            #     break
        # print('ok')
        return avg / len(SEED)

    def top_down(self, top, lateral):
        top2x = F.interpolate(top, scale_factor=2.0, mode='nearest')
        return lateral + top2x # 
    
    def forward(self, x, y=None, w=None, **kwargs):
        x_hsi = x
        
        self.time_ += 1

        if self.time_ == 1:
            self.X_full = x.permute(0, 2, 3, 1).squeeze(0)
            self.Y_full = y.squeeze(0)
            # self.X_full = torch.load('/mnt/sdd/niuyuanzhuo/FreeNet-master/X_full.pth').cuda()
            # self.Y_full = torch.load('/mnt/sdd/niuyuanzhuo/FreeNet-master/Y_full.pth').cuda()
            # self.w_full = torch.load('/mnt/c/Users/17735/OneDrive/桌面/FreeNet-master/full_w.pth').cuda()
            trainx_hsi, _ = createImageCubes(self.X_full.cpu(), self.Y_full.cpu())
            trainx_hsi = trainx_hsi.reshape(trainx_hsi.shape[0], 1, trainx_hsi.shape[3], trainx_hsi.shape[1], trainx_hsi.shape[2])
            trainx_hsi, _, _ = normalize(trainx_hsi)
            self.indexes = [i for i in range(x.shape[1])]
            entropies =  [entropy(trainx_hsi[:, :, i, :, :]) for i in range(len(self.indexes))]

            # Sort the pre-selected bands according to their entropy (in decreasing order)
            self.preselected = self.indexes.copy()
            pairs = list(tuple(zip(self.preselected, entropies)))
            pairs.sort(key=lambda x: x[1], reverse=True)
            self.preselected, _ = zip(*pairs)

            # Select the first "select" bands
            self.preselected = list(self.preselected)
            self.selection = self.preselected[:self.config.hidden_channels]
            self.selection.sort()
            self.preselected = self.preselected[self.config.hidden_channels:]
            # self.selection = [0, 74, 75, 98, 101, 109, 114, 116, 125, 128, 131, 132, 136, 241, 242, 243, 244, 247, 248, 249]
            ct = 1
            print("\tAnalyzing candidate combination " + str(ct) + ". 5x2 CV using bands: " + str(self.selection))
            self.f1base = self.tune_DFS()
            print("\tMean F1: " + str(self.f1base))
            
            self.bestselection = self.selection.copy()
            ct += 1
        # # if self.training:
        # #     # print('Current F1:' + str(self.tune(self.x_full, self.mask_full, self.w_full)))
        # #     if self.tune_DFS() < 0.6:
        # #         self.turn = True
        # #     else:
        # #         self.turn = False
    
        # # self.selection = self.tune_DFS()
        # x = x_hsi[:, self.selection, :, :]
        
        if self.training and self.num > 0: #  and self.turn and self.num > 0 and len(self.preselected) > 0
            self.num = self.num - 1
            
            # Try new bands until there is no more elements in the list
            # while len(self.preselected) > 0:
            #     if self.f1base > 0.8 or self.changed_f1base < 0:
            #         break
            #     # Calculate the maximum VIF of all the band in "selection"

            #     VIF = self.checkMulticollinearity(s=self.selection, trainx=trainx_hsi) # (1, 94890, 1, 1, 20)
            #     # Remove the band with the highest VIF of "selection"
            #     self.selection.remove(self.selection[VIF.index(max(VIF))])
            #     # Pop the next available band from "preselected"
            #     self.selection.append(self.preselected[0])
            #     self.selection.sort()
            #     self.preselected = self.preselected[1:]

            #     # Train using the bands in "selection"
            #     print("\tAnalyzing candidate combination " + str(ct) + ". 5x2 CV using bands: " + str(self.selection))
            #     self.f1 = self.tune_DFS()
            #     # with open("/mnt/sdd/niuyuanzhuo/FreeNet-master/log1_LK.txt", encoding="utf-8",mode="a") as file:  
            #     #     file.write(str(self.selection) + '  ' + str(self.f1) +'\n')
                
            #     print("\tMean F1: " + str(self.f1))
            #     # Check if the new selection has better performance than the previous one. If not, break
            #     if self.f1 > self.f1base:
            #         self.bestselection = self.selection.copy()
            #         self.f1base = self.f1
            #         self.changed_f1base = 10

            #     elif self.f1base > 0.8 or self.changed_f1base > 0:
            #         break
            #     else:
            #         self.changed_f1base = self.changed_f1base - 1
            #     ct += 1
            #     print("\tBest selection so far: " + str(self.bestselection) + "with an F1 score of " + str(self.f1base))
        
        # # if not self.turn:
        # # x = x_hsi[:, [0, 74, 75, 98, 101, 109, 114, 116, 125, 128, 131, 132, 136, 241, 242, 243, 244, 247, 248, 249], :, :]
        # print("\tBest selection so far: " + str(self.bestselection) + "with an F1 score of " + str(self.f1base))
        
        print("\tBest selection so far: " + str(self.bestselection) + "with an F1 score of " + str(self.f1base))

        if self.turn:

            # self.srf.weight_.requires_grad  = False
 
            # self.reconstruction4.weight.requires_grad = False
            # self.reconstruction4.bias.requires_grad = False

            # MST++
            for name, param in self.reconstruction1.named_parameters():
                if name == 'conv_out.weight' or name == 'conv_in.weight':
                    continue
                else:
                    param.requires_grad = False

        # self.time_ += 1
        x_hsi = x
        b_, c_, h_, w_ = x_hsi.shape
        x = x.permute(0,2,3,1)
        x = self.srf(x) 
        x = x.permute(0,3,1,2)
        # x_middle = x
        # for i in range(self.config.hidden_channels):
        # sv: [23, 24, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 55, 63, 64, 66, 77, 78, 94]
        # wh: [0, 74, 75, 98, 101, 109, 114, 116, 125, 128, 131, 132, 136, 241, 242, 243, 244, 247, 248, 249]
        # entropies = [entropy(x_hsi[:, i, :, :]) for i in range(len(self.indexes))]

        # # Sort the pre-selected bands according to their entropy (in decreasing order)
        # preselected = self.indexes.copy()
        # pairs = list(tuple(zip(preselected, entropies)))
        # pairs.sort(key=lambda x: x[1], reverse=True)
        # preselected, _ = zip(*pairs)

        # # Select the first "select" bands
        # preselected = list(preselected)
        # selection = preselected[:self.config.hidden_channels]
        # selection.sort()
        # preselected = preselected[self.config.hidden_channels:]
        
        x_hsi_bs = x_hsi[:, self.bestselection,:,:]
        # x_hsi_bs = x_hsi[:, 0:self.config.in_channels - 10:int((self.config.in_channels - 10)/self.config.hidden_channels),:,:]
        # x_hsi_bs = torch.tensor(self.BS.fit_transform(x_hsi.reshape(x_hsi.shape[0],\
        #                                     x_hsi.shape[1], x_hsi.shape[2] \
        #                                     * x_hsi.shape[3]).squeeze(0).T\
        #                                     .cpu().detach()).T\
        #                                     .reshape(b_, self.config.hidden_channels, h_, w_))\
        #                                     .cuda()
        # x = x.reshape(self.batch_size * self.batch_size, x.shape[1], int(x.shape[2]/self.batch_size), int(x.shape[3]/self.batch_size))

        x = self.reconstruction1(x) # / 2.0
        # x = x.reshape(b_, x.shape[1], int(x.shape[2] * self.batch_size), int(x.shape[3] * self.batch_size))
        x_rec = x

        feat_list = []
        for op in self.feature_ops:
            x = op(x)
            if isinstance(op, nn.Identity):
                feat_list.append(x)

        inner_feat_list = [self.reduce_1x1convs[i](feat) for i, feat in enumerate(feat_list)]
        inner_feat_list.reverse()

        out_feat_list = [self.fuse_3x3convs[0](inner_feat_list[0])]
        for i in range(len(feat_list) - 1):
            inner = self.top_down(out_feat_list[i], inner_feat_list[i+1])
            out = self.fuse_3x3convs[i + 1](inner)
            out_feat_list.append(out)

        final_feat = out_feat_list[-1]

        logit = self.cls_pred_conv(final_feat)

        if self.time_ == 3998 and self.turn:
            print('ok')
            loss_show = np.vstack((np.array(l1_loss_l),np.array(cls_loss_l)))
            # loss_show = np.array(l1_loss_l)
            plt.axhline(y=0.16, color='r', linestyle='--')
            plt.plot(np.arange(self.start_cls, 1999, 1), loss_show.T)
            plt.savefig('loss.jpg')
            plt.show()

        # if self.time_== 1996:
        #     print('ok')
            
        #     plt.plot(psnr)
        #     plt.savefig('psnr.jpg')
        
        if self.training:
            # l1_loss = l1loss1(x_hsi_bs, x_rec)
            # cls_loss = self.loss(logit, y, w)
           
            # l1_loss_l.append(l1_loss.item())
            # cls_loss_l.append(cls_loss.item())
            # self.time_> 1200 or 
            if (l1loss1(x_rec, x_hsi_bs) < 0.13 or self.turn): # self.time_> 1600 or  0.019 for sv 0.16 for whlk5 0.26 for whlk3 in train3 0.029/0.03 for whhc
                if not self.turn:
                    # print('ok')
                    self.start_cls = int(self.time_/2)
                l1_loss = l1loss1(x_hsi_bs, x_rec)
                cls_loss = self.loss(logit, y, w)

                l1_loss_l.append(l1_loss.item())
                cls_loss_l.append(cls_loss.item())
                # if not self.turn:
                #     rec = x_rec.detach().cpu().numpy()
                
                #     result = {"WHU_Hi_LongKou_train": rec}
                
                #     scipy.io.savemat("WHU_Hi_LongKou_train.mat", result)
                
                loss_dict = {
                    'cls_loss': cls_loss / 5.0 ,
                    # 'l1_loss': l1_loss/ 10.0,
                    }
                if l1_loss > 0.13:
                    loss_dict = {
                    'cls_loss': cls_loss/ 5.0,
                    'l1_loss': l1_loss/ 10.0,
                    }
                # if cls_loss < l1_loss and self.num_to_stop > 0:
                #     # loss_dict = {
                #     # 'cls_loss': cls_loss/ 5.0,
                #     # # 'l1_loss': l1_loss/ 10.0,
                #     # }
                #     # print('ok')
                #     self.num_to_stop = self.num_to_stop - 1
                #     # loss_show = np.vstack((np.array(l1_loss_l),np.array(cls_loss_l)))
                #     # # loss_show = np.array(l1_loss_l)
                #     # plt.axhline(y=0.16, color='r', linestyle='--')
                #     # plt.plot(loss_show.T)
                #     # plt.savefig('loss.jpg')
                #     # plt.show()
                #     sys.exit()
                self.turn = True
                # self.pre_rec_x = x_rec
                # self.pre_x_middle = x_middle

                # MST++
                for name, param in self.reconstruction1.named_parameters():
                    if name == 'conv_out.weight' or name == 'conv_in.weight':
                        continue
                    else:
                        param.requires_grad = False


                
            else:
                        
                loss_dict = {
                    'l1_loss': l1loss1(x_rec, x_hsi_bs),# + self.loss(logit, y, w) , # + abs(20 - PSNR(x_hsi, x_rec)), # 
                # 'cls_loss': self.loss(logit, y, w)
                }
                 
            return loss_dict
        
        if not self.training:
            # psnr.append(PSNR(x_hsi_bs, x_rec))
            print('PSNR:' + str(PSNR(x_hsi_bs, x_rec)))

        return torch.softmax(logit, dim=1)

    def loss(self, x, y, weight):
        losses = F.cross_entropy(x, y.long() - 1, weight=None,
                                 ignore_index=-1, reduction='none')

        v = losses.mul_(weight).sum() / weight.sum()
        return v

    def set_defalut_config(self):
        self.config.update(dict(
            in_channels=204,
            num_classes=16,
            block_channels=(96, 128, 192, 256),
            num_blocks=(1, 1, 1, 1),
            inner_dim=128,
            reduction_ratio=1.0,
        ))


@registry.MODEL.register('FreeNetE2E_BA')
class FreeNetE2E_BA(CVModule):
    def __init__(self, config):
        super(FreeNetE2E_BA, self).__init__(config)
        r = int(16 * self.config.reduction_ratio)
        block1_channels = int(self.config.block_channels[0] * self.config.reduction_ratio / r) * r
        block2_channels = int(self.config.block_channels[1] * self.config.reduction_ratio / r) * r
        block3_channels = int(self.config.block_channels[2] * self.config.reduction_ratio / r) * r
        block4_channels = int(self.config.block_channels[3] * self.config.reduction_ratio / r) * r
        # print(self.config.in_channels)
        self.feature_ops_1 = nn.ModuleList([
            conv3x3_gn_relu(self.config.in_channels, block1_channels, r),

            repeat_block(block1_channels, r, self.config.num_blocks[0]),
            nn.Identity(),
            downsample2x(block1_channels, block2_channels),

            repeat_block(block2_channels, r, self.config.num_blocks[1]),
            nn.Identity(),
            downsample2x(block2_channels, block3_channels),

            repeat_block(block3_channels, r, self.config.num_blocks[2]),
            nn.Identity(),
            downsample2x(block3_channels, block4_channels),

            repeat_block(block4_channels, r, self.config.num_blocks[3]),
            nn.Identity(),
        ])
        self.feature_ops_2 = nn.ModuleList([
            conv3x3_gn_relu(self.config.hidden_channels, block1_channels, r),

            repeat_block(block1_channels, r, self.config.num_blocks[0]),
            nn.Identity(),
            downsample2x(block1_channels, block2_channels),

            repeat_block(block2_channels, r, self.config.num_blocks[1]),
            nn.Identity(),
            downsample2x(block2_channels, block3_channels),

            repeat_block(block3_channels, r, self.config.num_blocks[2]),
            nn.Identity(),
            downsample2x(block3_channels, block4_channels),

            repeat_block(block4_channels, r, self.config.num_blocks[3]),
            nn.Identity(),
        ])
        inner_dim = int(self.config.inner_dim * self.config.reduction_ratio)
        self.reduce_1x1convs_1 = nn.ModuleList([
            nn.Conv2d(block1_channels, inner_dim, 1),
            nn.Conv2d(block2_channels, inner_dim, 1),
            nn.Conv2d(block3_channels, inner_dim, 1),
            nn.Conv2d(block4_channels, inner_dim, 1),
        ])
        self.reduce_1x1convs_2 = nn.ModuleList([
            nn.Conv2d(block1_channels, inner_dim, 1),
            nn.Conv2d(block2_channels, inner_dim, 1),
            nn.Conv2d(block3_channels, inner_dim, 1),
            nn.Conv2d(block4_channels, inner_dim, 1),
        ])
        self.fuse_3x3convs_1 = nn.ModuleList([
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
        ])
        self.fuse_3x3convs_2 = nn.ModuleList([
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
        ])
        # self.batch_size = self.config.rec_bs
        self.in_dim = self.config.in_channels
        self.out_dim = self.config.out_channels
        self.resrf = nn.Parameter(torch.rand((self.out_dim, self.in_dim)))
        self.cls_pred_conv_1 = nn.Conv2d(inner_dim, self.config.num_classes, 1)
        self.cls_pred_conv_2 = nn.Conv2d(inner_dim, self.config.num_classes, 1)
        self.srf = LyotFilter(self.config.in_channels, self.config.out_channels,  self.config.minBand, self.config.maxBand, self.config.nBandDataset, self.config.dataset)
        # self.resrf = reLyotFilter(self.config.in_channels, self.config.out_channels,  self.config.minBand, self.config.maxBand, self.config.nBandDataset, self.config.dataset)
        self.dynamic1 = S3KAIResNet(self.config.out_channels, self.config.num_classes, 2)# nn.Conv2d(self.config.out_channels, self.config.out_channels * self.config.out_channels, 1, 1, 0)#
        self.dynamic2 = nn.Conv2d(self.config.out_channels, self.config.in_channels, 1, 1, 0)#  S3KAIResNet(self.config.out_channels, self.config.num_classes, 2)# 
        self.dynamic3 = RSSAN5(self.config.num_classes, self.config.out_channels, 3, self.config.out_channels, 1, 1, 1, self.config.hidden_channels)
        self.dynamic4 = SEBlock(self.config.out_channels, 2)

        self.reconstruction1 = MST_Plus_Plus(self.config.out_channels, self.config.hidden_channels,  self.config.hidden_channels, 3)
        self.reconstruction2 = BiSRNet(self.config.out_channels, self.config.hidden_channels, self.config.hidden_channels)

        self.reconstruction4 = nn.Conv2d(self.config.out_channels, self.config.hidden_channels, 3, 1, 1)#  S3KAIResNet(self.config.out_channels, self.config.num_classes, 2)# 

        self.herosnet = HerosNet(Ch=self.config.out_channels, stages=8)
        self.GST = GST_MODEL(in_ch=self.config.out_channels,
                           out_ch=self.config.out_channels,
                           noise_mean=0.1,
                           noise_std=0.1,
                           init="normal",
                        #    conv = default_conv,
                           noise_act=nn.Softplus(),
                           inter_channels=10,
                           spatial_scale=4)
        self.time_ = 0
        self.BS = PCA(n_components=self.config.hidden_channels)
        self.start_cls = 0
        self.changed_f1base = 10
        self.turn = False
        self.pre_rec_x = None
        self.pre_x_middle = None
        self.num = 1
        self.num_to_stop = 30
        self.indexes = []
        self.BandSelection1 = MyLinearLayer(self.config.in_channels, self.config.hidden_channels)
        self.BandSelection2 = E2E_BS(self.config.in_channels, self.config.hidden_channels, self.config.dataset)

        # self.BandSelection3 = BandSelectionLayer(self.config.in_channels, self.config.hidden_channels)
        self.BandSelection4 = SpectralAttentionNetwork(self.config.in_channels, self.config.hidden_channels)
        # self.BandSelection5 = SpectralAttentionNetwork(self.config.in_channels, self.config.hidden_channels)
        self.BandSelection6 = BAM2(self.config.in_channels, self.config.hidden_channels)

        self.BandSelection_turn = False
        self.BandSelection_l = []
        self.band_selected = False
        self.start_cls_time = 0
        self.start_rec = False
        self.start_rec_time = 0
        self.att_reg = 0
        self.X_full = None
        self.Y_full = None

    def top_down(self, top, lateral):
        top2x = F.interpolate(top, scale_factor=2.0, mode='nearest')
        return lateral + top2x # 
    
    def forward(self, x, y=None, w=None, **kwargs):

        x_hsi = x
        # seleceted_indexes = torch.rand((self.config.hidden_channels, ))
        self.time_ += 1

        x_hsi = x
        b_, c_, h_, w_ = x_hsi.shape

        x, self.band_selected, spectral_attention, num21 = self.BandSelection4(x, self.time_)

        feat_list = []
        if not self.band_selected:
            for op in self.feature_ops_1:
                x = op(x)
                if isinstance(op, nn.Identity):
                    feat_list.append(x)

            inner_feat_list = [self.reduce_1x1convs_1[i](feat) for i, feat in enumerate(feat_list)]
            inner_feat_list.reverse()

            out_feat_list_1 = [self.fuse_3x3convs_1[0](inner_feat_list[0])]
            for i in range(len(feat_list) - 1):
                inner = self.top_down(out_feat_list_1[i], inner_feat_list[i+1])
                out = self.fuse_3x3convs_1[i + 1](inner)
                out_feat_list_1.append(out)

            final_feat = out_feat_list_1[-1]

            logit = self.cls_pred_conv_1(final_feat)
            # if (self.time_ // 2) > 0:
            #     self.att_reg = (spectral_attention.sum() - self.config.hidden_channels).abs()
            
            if self.training:
                if (self.time_ // 2) > 0 :
                    loss_dict = {
                    'cls_loss': self.loss(logit, y, w),
                    'spectral_attention_loss':  (spectral_attention.sum() - self.config.hidden_channels).abs() * 10  # (spectral_attention.sum() - self.config.hidden_channels).abs() * 10 l1loss1(spectral_attention, torch.zeros_like(spectral_attention))
                    }
                else:
                    loss_dict = {
                    'cls_loss': self.loss(logit, y, w),
                    # 'spectral_attention_loss': (spectral_attention.sum() - self.config.hidden_channels).abs()
                    }
                return loss_dict
            
        else:
            print('num21')
            print(num21)
            if  not self.start_rec:
                self.start_rec_time = int(self.time_/2)
                self.start_rec = True

            # if self.start_cls and self.training:
            #     self.pre_rec_x = x_rec

            for op in self.feature_ops_2:
                x = op(x)
                if isinstance(op, nn.Identity):
                    feat_list.append(x)

            inner_feat_list = [self.reduce_1x1convs_2[i](feat) for i, feat in enumerate(feat_list)]
            inner_feat_list.reverse()

            out_feat_list_2 = [self.fuse_3x3convs_2[0](inner_feat_list[0])]
            for i in range(len(feat_list) - 1):
                inner = self.top_down(out_feat_list_2[i], inner_feat_list[i+1])
                out = self.fuse_3x3convs_2[i + 1](inner)
                out_feat_list_2.append(out)

            final_feat = out_feat_list_2[-1]

            logit = self.cls_pred_conv_2(final_feat)

            if self.training:
                cls_loss = self.loss(logit, y, w)
                loss_dict = {
                    'cls_loss': cls_loss,
                }
                
                return loss_dict

        return torch.softmax(logit, dim=1)

    def weights_reg_loss(self, weights):
        reg_loss = weights.sum() - weights.shape[0]
            # reg_loss += weights[i].sum() - 1
        return reg_loss

    def loss(self, x, y, weight):
        losses = F.cross_entropy(x, y.long() - 1, weight=None,
                                 ignore_index=-1, reduction='none')
        
        # assert losses.mul_(weight).sum() == F.cross_entropy(x, (y - 1).mul_(weight).long(), weight=None, ignore_index=-1, reduction='none').mul_(weight).sum()

        v = losses.mul_(weight).sum() / weight.sum()
        return v

    def set_defalut_config(self):
        self.config.update(dict(
            in_channels=204,
            num_classes=16,
            block_channels=(96, 128, 192, 256),
            num_blocks=(1, 1, 1, 1),
            inner_dim=128,
            reduction_ratio=1.0,
        ))


@registry.MODEL.register('FreeNetE2E_BA_SS')
class FreeNetE2E_BA_SS(CVModule):
    def __init__(self, config):
        super(FreeNetE2E_BA_SS, self).__init__(config)
        r = int(16 * self.config.reduction_ratio)
        block1_channels = int(self.config.block_channels[0] * self.config.reduction_ratio / r) * r
        block2_channels = int(self.config.block_channels[1] * self.config.reduction_ratio / r) * r
        block3_channels = int(self.config.block_channels[2] * self.config.reduction_ratio / r) * r
        block4_channels = int(self.config.block_channels[3] * self.config.reduction_ratio / r) * r
        # print(self.config.in_channels)
        self.feature_ops_1 = nn.ModuleList([
            conv3x3_gn_relu(self.config.in_channels, block1_channels, r),

            repeat_block(block1_channels, r, self.config.num_blocks[0]),
            nn.Identity(),
            downsample2x(block1_channels, block2_channels),

            repeat_block(block2_channels, r, self.config.num_blocks[1]),
            nn.Identity(),
            downsample2x(block2_channels, block3_channels),

            repeat_block(block3_channels, r, self.config.num_blocks[2]),
            nn.Identity(),
            downsample2x(block3_channels, block4_channels),

            repeat_block(block4_channels, r, self.config.num_blocks[3]),
            nn.Identity(),
        ])
        self.feature_ops_2 = nn.ModuleList([
            conv3x3_gn_relu(self.config.hidden_channels, block1_channels, r),

            repeat_block(block1_channels, r, self.config.num_blocks[0]),
            nn.Identity(),
            downsample2x(block1_channels, block2_channels),

            repeat_block(block2_channels, r, self.config.num_blocks[1]),
            nn.Identity(),
            downsample2x(block2_channels, block3_channels),

            repeat_block(block3_channels, r, self.config.num_blocks[2]),
            nn.Identity(),
            downsample2x(block3_channels, block4_channels),

            repeat_block(block4_channels, r, self.config.num_blocks[3]),
            nn.Identity(),
        ])
        inner_dim = int(self.config.inner_dim * self.config.reduction_ratio)
        self.reduce_1x1convs_1 = nn.ModuleList([
            nn.Conv2d(block1_channels, inner_dim, 1),
            nn.Conv2d(block2_channels, inner_dim, 1),
            nn.Conv2d(block3_channels, inner_dim, 1),
            nn.Conv2d(block4_channels, inner_dim, 1),
        ])
        self.reduce_1x1convs_2 = nn.ModuleList([
            nn.Conv2d(block1_channels, inner_dim, 1),
            nn.Conv2d(block2_channels, inner_dim, 1),
            nn.Conv2d(block3_channels, inner_dim, 1),
            nn.Conv2d(block4_channels, inner_dim, 1),
        ])
        self.fuse_3x3convs_1 = nn.ModuleList([
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
        ])
        self.fuse_3x3convs_2 = nn.ModuleList([
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
        ])
        # self.batch_size = self.config.rec_bs
        self.in_dim = self.config.in_channels
        self.out_dim = self.config.out_channels
        self.resrf = nn.Parameter(torch.rand((self.out_dim, self.in_dim)))
        self.cls_pred_conv_1 = nn.Conv2d(inner_dim, self.config.num_classes, 1)
        self.cls_pred_conv_2 = nn.Conv2d(inner_dim, self.config.num_classes, 1)
        self.srf = LyotFilter(self.config.in_channels, self.config.out_channels,  self.config.minBand, self.config.maxBand, self.config.nBandDataset, self.config.dataset)
        # self.resrf = reLyotFilter(self.config.in_channels, self.config.out_channels,  self.config.minBand, self.config.maxBand, self.config.nBandDataset, self.config.dataset)
        self.dynamic1 = S3KAIResNet(self.config.out_channels, self.config.num_classes, 2)# nn.Conv2d(self.config.out_channels, self.config.out_channels * self.config.out_channels, 1, 1, 0)#
        self.dynamic2 = nn.Conv2d(self.config.out_channels, self.config.in_channels, 1, 1, 0)#  S3KAIResNet(self.config.out_channels, self.config.num_classes, 2)# 
        self.dynamic3 = RSSAN5(self.config.num_classes, self.config.out_channels, 3, self.config.out_channels, 1, 1, 1, self.config.hidden_channels)
        self.dynamic4 = SEBlock(self.config.out_channels, 2)

        self.reconstruction1 = MST_Plus_Plus(self.config.out_channels, self.config.hidden_channels,  self.config.hidden_channels, 3)
        self.reconstruction2 = BiSRNet(self.config.out_channels, self.config.hidden_channels, self.config.hidden_channels)

        self.reconstruction4 = nn.Conv2d(self.config.out_channels, self.config.hidden_channels, 3, 1, 1)#  S3KAIResNet(self.config.out_channels, self.config.num_classes, 2)# 

        self.herosnet = HerosNet(Ch=self.config.out_channels, stages=8)
        self.GST = GST_MODEL(in_ch=self.config.out_channels,
                           out_ch=self.config.out_channels,
                           noise_mean=0.1,
                           noise_std=0.1,
                           init="normal",
                        #    conv = default_conv,
                           noise_act=nn.Softplus(),
                           inter_channels=10,
                           spatial_scale=4)
        self.time_ = 0
        self.BS = PCA(n_components=self.config.hidden_channels)
        self.start_cls = 0
        self.changed_f1base = 10
        self.turn = False
        self.pre_rec_x = None
        self.pre_x_middle = None
        self.num = 1
        self.num_to_stop = 30
        self.indexes = []
        self.BandSelection1 = MyLinearLayer(self.config.in_channels, self.config.hidden_channels)
        self.BandSelection2 = E2E_BS(self.config.in_channels, self.config.hidden_channels, self.config.dataset)

        # self.BandSelection3 = BandSelectionLayer(self.config.in_channels, self.config.hidden_channels)
        self.BandSelection4 = SpectralAttentionNetwork(self.config.in_channels, self.config.hidden_channels)
        # self.BandSelection5 = SpectralAttentionNetwork(self.config.in_channels, self.config.hidden_channels)
        self.BandSelection6 = BAM2(self.config.in_channels, self.config.hidden_channels)

        self.BandSelection_turn = False
        self.BandSelection_l = []
        self.band_selected = False
        self.start_cls_time = 0
        self.start_rec = False
        self.start_rec_time = 0
        self.att_reg = 0
        self.X_full = None
        self.Y_full = None

    def top_down(self, top, lateral):
        top2x = F.interpolate(top, scale_factor=2.0, mode='nearest')
        return lateral + top2x # 
    
    def forward(self, x, y=None, w=None, segments=None, **kwargs):

        x_hsi = x
        # seleceted_indexes = torch.rand((self.config.hidden_channels, ))
        self.time_ += 1

        x_hsi = x
        b_, c_, h_, w_ = x_hsi.shape

        x, self.band_selected, spectral_attention, num21 = self.BandSelection4(x, self.time_)

        feat_list = []
        if not self.band_selected:
            for op in self.feature_ops_1:
                x = op(x)
                if isinstance(op, nn.Identity):
                    feat_list.append(x)

            inner_feat_list = [self.reduce_1x1convs_1[i](feat) for i, feat in enumerate(feat_list)]
            inner_feat_list.reverse()

            out_feat_list_1 = [self.fuse_3x3convs_1[0](inner_feat_list[0])]
            for i in range(len(feat_list) - 1):
                inner = self.top_down(out_feat_list_1[i], inner_feat_list[i+1])
                out = self.fuse_3x3convs_1[i + 1](inner)
                out_feat_list_1.append(out)

            final_feat = out_feat_list_1[-1]

            logit = self.cls_pred_conv_1(final_feat)
            # if (self.time_ // 2) > 0:
            #     self.att_reg = (spectral_attention.sum() - self.config.hidden_channels).abs()
            
            if self.training:
                total_var_superpixel = calculate_superpixel_variance(torch.softmax(logit, dim=1).squeeze(0).argmax(dim=0) + 1, segments)
                if (self.time_ // 2) > 0:
                    loss_dict = {
                    'cls_loss': self.loss(logit, y, w),
                    'total_var_superpixel': total_var_superpixel,
                    'spectral_attention_loss':  (spectral_attention.sum() - self.config.hidden_channels).abs() * 10  # (spectral_attention.sum() - self.config.hidden_channels).abs() * 10 l1loss1(spectral_attention, torch.zeros_like(spectral_attention))
                    }
                else:
                    loss_dict = {
                    'total_var_superpixel': total_var_superpixel,
                    'cls_loss': self.loss(logit, y, w),
                    # 'spectral_attention_loss': (spectral_attention.sum() - self.config.hidden_channels).abs()
                    }
                return loss_dict
            
        else:
            print('num21')
            print(num21)
            if  not self.start_rec:
                self.start_rec_time = int(self.time_/2)
                self.start_rec = True

            # if self.start_cls and self.training:
            #     self.pre_rec_x = x_rec

            for op in self.feature_ops_2:
                x = op(x)
                if isinstance(op, nn.Identity):
                    feat_list.append(x)

            inner_feat_list = [self.reduce_1x1convs_2[i](feat) for i, feat in enumerate(feat_list)]
            inner_feat_list.reverse()

            out_feat_list_2 = [self.fuse_3x3convs_2[0](inner_feat_list[0])]
            for i in range(len(feat_list) - 1):
                inner = self.top_down(out_feat_list_2[i], inner_feat_list[i+1])
                out = self.fuse_3x3convs_2[i + 1](inner)
                out_feat_list_2.append(out)

            final_feat = out_feat_list_2[-1]

            logit = self.cls_pred_conv_2(final_feat)

            if self.training:
                cls_loss = self.loss(logit, y, w)
                loss_dict = {
                    'cls_loss': cls_loss,
                }
                
                return loss_dict

        return torch.softmax(logit, dim=1)

    def weights_reg_loss(self, weights):
        reg_loss = weights.sum() - weights.shape[0]
            # reg_loss += weights[i].sum() - 1
        return reg_loss

    def loss(self, x, y, weight):
        losses = F.cross_entropy(x, y.long() - 1, weight=None,
                                 ignore_index=-1, reduction='none')
        
        # assert losses.mul_(weight).sum() == F.cross_entropy(x, (y - 1).mul_(weight).long(), weight=None, ignore_index=-1, reduction='none').mul_(weight).sum()

        v = losses.mul_(weight).sum() / weight.sum()
        return v

    def set_defalut_config(self):
        self.config.update(dict(
            in_channels=204,
            num_classes=16,
            block_channels=(96, 128, 192, 256),
            num_blocks=(1, 1, 1, 1),
            inner_dim=128,
            reduction_ratio=1.0,
        ))

@registry.MODEL.register('FreeNetE2E_BA_Rec')
class FreeNetE2E_BA_Rec(CVModule):
    def __init__(self, config):
        super(FreeNetE2E_BA_Rec, self).__init__(config)
        r = int(16 * self.config.reduction_ratio)
        block1_channels = int(self.config.block_channels[0] * self.config.reduction_ratio / r) * r
        block2_channels = int(self.config.block_channels[1] * self.config.reduction_ratio / r) * r
        block3_channels = int(self.config.block_channels[2] * self.config.reduction_ratio / r) * r
        block4_channels = int(self.config.block_channels[3] * self.config.reduction_ratio / r) * r
        # print(self.config.in_channels)

        self.feature_ops = nn.ModuleList([
            conv3x3_gn_relu(self.config.hidden_channels, block1_channels, r),

            repeat_block(block1_channels, r, self.config.num_blocks[0]),
            nn.Identity(),
            downsample2x(block1_channels, block2_channels),

            repeat_block(block2_channels, r, self.config.num_blocks[1]),
            nn.Identity(),
            downsample2x(block2_channels, block3_channels),

            repeat_block(block3_channels, r, self.config.num_blocks[2]),
            nn.Identity(),
            downsample2x(block3_channels, block4_channels),

            repeat_block(block4_channels, r, self.config.num_blocks[3]),
            nn.Identity(),
        ])
        inner_dim = int(self.config.inner_dim * self.config.reduction_ratio)

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
        # self.batch_size = self.config.rec_bs
        self.in_dim = self.config.in_channels
        self.out_dim = self.config.out_channels
        self.resrf = nn.Parameter(torch.rand((self.out_dim, self.in_dim)))
        self.cls_pred_conv = nn.Conv2d(inner_dim, self.config.num_classes, 1)
        self.srf = LyotFilter(self.config.in_channels, self.config.out_channels,  self.config.minBand, self.config.maxBand, self.config.nBandDataset, self.config.dataset)
        # self.resrf = reLyotFilter(self.config.in_channels, self.config.out_channels,  self.config.minBand, self.config.maxBand, self.config.nBandDataset, self.config.dataset)
        self.dynamic1 = S3KAIResNet(self.config.out_channels, self.config.num_classes, 2)# nn.Conv2d(self.config.out_channels, self.config.out_channels * self.config.out_channels, 1, 1, 0)#
        self.dynamic2 = nn.Conv2d(self.config.out_channels, self.config.in_channels, 1, 1, 0)#  S3KAIResNet(self.config.out_channels, self.config.num_classes, 2)# 
        self.dynamic3 = RSSAN5(self.config.num_classes, self.config.out_channels, 3, self.config.out_channels, 1, 1, 1, self.config.hidden_channels)
        self.dynamic4 = SEBlock(self.config.out_channels, 2)

        self.reconstruction1 = MST_Plus_Plus(self.config.out_channels, self.config.hidden_channels,  self.config.hidden_channels, 3)
        self.reconstruction2 = BiSRNet(self.config.out_channels, self.config.hidden_channels, self.config.hidden_channels)

        self.reconstruction4 = nn.Conv2d(self.config.out_channels, self.config.hidden_channels, 3, 1, 1)#  S3KAIResNet(self.config.out_channels, self.config.num_classes, 2)# 

        self.herosnet = HerosNet(Ch=self.config.out_channels, stages=8)
        self.GST = GST_MODEL(in_ch=self.config.out_channels,
                           out_ch=self.config.out_channels,
                           noise_mean=0.1,
                           noise_std=0.1,
                           init="normal",
                        #    conv = default_conv,
                           noise_act=nn.Softplus(),
                           inter_channels=10,
                           spatial_scale=4)
        self.time_ = 0
        self.BS = PCA(n_components=self.config.hidden_channels)
        self.start_cls = False
        self.changed_f1base = 10
        self.turn = False
        self.pre_rec_x = None
        self.pre_x_middle = None
        self.num = 1
        self.num_to_stop = 10
        self.indexes = []
        self.BandSelection1 = MyLinearLayer(self.config.in_channels, self.config.hidden_channels)
        self.BandSelection2 = E2E_BS(self.config.in_channels, self.config.hidden_channels, self.config.dataset)

        # self.BandSelection3 = BandSelectionLayer(self.config.in_channels, self.config.hidden_channels)
        self.BandSelection4 = SpectralAttentionNetwork(self.config.in_channels, self.config.hidden_channels)
        # self.BandSelection5 = SpectralAttentionNetwork(self.config.in_channels, self.config.hidden_channels)
        self.BandSelection6 = BAM2(self.config.in_channels, self.config.hidden_channels)

        self.BandSelection_turn = False
        self.BandSelection_l = []
        self.band_selected = False
        self.start_cls_time = 0
        self.start_rec = False
        self.start_rec_time = 0
        self.att_reg = 0
        self.rec_loss_threshold = 0.08

    def top_down(self, top, lateral):
        top2x = F.interpolate(top, scale_factor=2.0, mode='nearest')
        return lateral + top2x # 
    
    def forward(self, x, y=None, w=None, **kwargs):
        if self.start_cls:
            # self.srf.weight_.requires_grad  = False
            # MST++
            for name, param in self.reconstruction1.named_parameters():
                if name == 'conv_out.weight' or name == 'conv_in.weight':
                    continue
                else:
                    param.requires_grad = False

        x_hsi = x
        if self.training:
            self.time_ += 1

        x_hsi_bs = x_hsi[:,[  2,  14,  19,  23,  37,  45,  55,  71,  73,  77, 102, 118, 119, 128, 129, 131, 132, 144, 158, 163], :, :]
        feat_list = []

        x = x_hsi.permute(0,2,3,1)
        x = self.srf(x) 
        x = x.permute(0,3,1,2)
        few_shot = x
        start_time = time.time()
        x = self.reconstruction1(x)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Reconstruction time: {execution_time:.6f} seconds")
        x_rec = x

        for op in self.feature_ops:
            x = op(x)
            if isinstance(op, nn.Identity):
                feat_list.append(x)

        inner_feat_list = [self.reduce_1x1convs[i](feat) for i, feat in enumerate(feat_list)]
        inner_feat_list.reverse()

        out_feat_list = [self.fuse_3x3convs[0](inner_feat_list[0])]
        for i in range(len(feat_list) - 1):
            inner = self.top_down(out_feat_list[i], inner_feat_list[i+1])
            out = self.fuse_3x3convs[i + 1](inner)
            out_feat_list.append(out)

        final_feat = out_feat_list[-1]

        logit = self.cls_pred_conv(final_feat)

        if self.training:
            # 0.013 for SV
            # 0.73 for train1 whlk 0.08 for train2 whlk 0.2 for train3 whlk
            if (self.turn or self.time_ > 2800): # self.time_> 1600 or  0.019 for sv 0.16 for whlk5 0.26 for whlk3 in train3 0.029/0.03 for whhc l1loss1(x_rec, x_hsi_bs) < self.rec_loss_threshold or 
                if not self.turn:
                    # print('ok')
                    self.start_cls_time = self.time_
                    self.turn = True
                    self.rec_loss_threshold = l1loss1(x_rec, x_hsi_bs)

                l1_loss = l1loss1(x_hsi_bs, x_rec)
                cls_loss = self.loss(logit, y, w)

                l1_loss_l.append(l1_loss.item())
                cls_loss_l.append(cls_loss.item())
                
                loss_dict = {
                    'cls_loss': cls_loss / 5.0 ,
                    }
                if l1_loss > self.rec_loss_threshold:
                    loss_dict = {
                        'cls_loss': cls_loss / 5.0,
                        'l1_loss': l1_loss / 10.0,
                    }

                # if cls_loss < l1_loss and self.num_to_stop > 0:

                #     self.num_to_stop = self.num_to_stop - 1

                #     sys.exit()
                # self.turn = True


                # MST++
                for name, param in self.reconstruction1.named_parameters():
                    if name == 'conv_out.weight' or name == 'conv_in.weight':
                        continue
                    else:
                        param.requires_grad = False


                
            else:
                        
                loss_dict = {
                    'l1_loss': l1loss1(x_rec, x_hsi_bs),# + self.loss(logit, y, w) , # + abs(20 - PSNR(x_hsi, x_rec)), # 
                # 'cls_loss': self.loss(logit, y, w)
                }
                
                
            return loss_dict
    
        if self.time_ == 2:
            x_hsi_bs_np = x_hsi_bs[0].cpu().numpy()
            few_shot_np = few_shot[0].cpu().numpy() 
            x_rec_np = x_rec[0].cpu().numpy()

            # print('ok')
            # plt.figure()
            # loss_show = np.vstack((np.array(l1_loss_l), np.array(cls_loss_l)))
            # plt.axhline(y=0.013, color='r', linestyle='--')
            # plt.plot(np.arange(self.start_cls_time - 1, 2998, 1), loss_show.T)
            # plt.savefig('2losses.jpg')
            # plt.show()
            
            # 可视化 few_shot 的RGB图像
            # visualize_hsi_rgb(few_shot_np, 'Three_Snapshots')

            # # 可视化 x_hsi_bs 的所有20个波段
            # visualize_and_save_all_bands(x_hsi_bs_np, 'Reconstruction_Reference')

            # # 可视化 x_rec 的所有20个波段
            # visualize_and_save_all_bands(x_rec_np, 'Reconstructed_HSI_of_desired_bands')
        # if self.time_ == 1998:
        #     print('ok')
        #     # loss_show = np.vstack((np.array(l1_loss_l), np.array(cls_loss_l)))
        #     # loss_show = np.array(l1_loss_l)
        #     plt.axhline(y=0.019, color='r', linestyle='--')
        #     plt.plot(np.arange(self.start_rec_time, 999, 1), (np.array(l1_loss_l_1).T))
        #     plt.savefig('loss.jpg')
        #     plt.show()
        
        if not self.training:
            print('PSNR:' + str(PSNR(x_hsi_bs, x_rec)))
            print('SSIM:' + str(SSIM(x_hsi_bs, x_rec)))

        return torch.softmax(logit, dim=1)

    def loss(self, x, y, weight):
        losses = F.cross_entropy(x, y.long() - 1, weight=None,
                                 ignore_index=-1, reduction='none')

        v = losses.mul_(weight).sum() / weight.sum()
        return v

    def set_defalut_config(self):
        self.config.update(dict(
            in_channels=204,
            num_classes=16,
            block_channels=(96, 128, 192, 256),
            num_blocks=(1, 1, 1, 1),
            inner_dim=128,
            reduction_ratio=1.0,
        ))


@registry.MODEL.register('FreeNetE2E_RGB_BA_Rec')
class FreeNetE2E_RGB_BA_Rec(CVModule):
    def __init__(self, config):
        super(FreeNetE2E_RGB_BA_Rec, self).__init__(config)
        r = int(16 * self.config.reduction_ratio)
        block1_channels = int(self.config.block_channels[0] * self.config.reduction_ratio / r) * r
        block2_channels = int(self.config.block_channels[1] * self.config.reduction_ratio / r) * r
        block3_channels = int(self.config.block_channels[2] * self.config.reduction_ratio / r) * r
        block4_channels = int(self.config.block_channels[3] * self.config.reduction_ratio / r) * r
        # print(self.config.in_channels)

        self.feature_ops = nn.ModuleList([
            conv3x3_gn_relu(self.config.hidden_channels, block1_channels, r),

            repeat_block(block1_channels, r, self.config.num_blocks[0]),
            nn.Identity(),
            downsample2x(block1_channels, block2_channels),

            repeat_block(block2_channels, r, self.config.num_blocks[1]),
            nn.Identity(),
            downsample2x(block2_channels, block3_channels),

            repeat_block(block3_channels, r, self.config.num_blocks[2]),
            nn.Identity(),
            downsample2x(block3_channels, block4_channels),

            repeat_block(block4_channels, r, self.config.num_blocks[3]),
            nn.Identity(),
        ])
        inner_dim = int(self.config.inner_dim * self.config.reduction_ratio)

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
        # self.batch_size = self.config.rec_bs
        self.in_dim = self.config.in_channels
        self.out_dim = self.config.out_channels
        self.resrf = nn.Parameter(torch.rand((self.out_dim, self.in_dim)))
        self.cls_pred_conv = nn.Conv2d(inner_dim, self.config.num_classes, 1)
        self.srf = LyotFilter(self.config.in_channels, self.config.out_channels,  self.config.minBand, self.config.maxBand, self.config.nBandDataset, self.config.dataset)
        # self.resrf = reLyotFilter(self.config.in_channels, self.config.out_channels,  self.config.minBand, self.config.maxBand, self.config.nBandDataset, self.config.dataset)
        self.dynamic1 = S3KAIResNet(self.config.out_channels, self.config.num_classes, 2)# nn.Conv2d(self.config.out_channels, self.config.out_channels * self.config.out_channels, 1, 1, 0)#
        self.dynamic2 = nn.Conv2d(self.config.out_channels, self.config.in_channels, 1, 1, 0)#  S3KAIResNet(self.config.out_channels, self.config.num_classes, 2)# 
        self.dynamic3 = RSSAN5(self.config.num_classes, self.config.out_channels, 3, self.config.out_channels, 1, 1, 1, self.config.hidden_channels)
        self.dynamic4 = SEBlock(self.config.out_channels, 2)

        self.reconstruction1 = MST_Plus_Plus(self.config.out_channels, self.config.hidden_channels,  self.config.hidden_channels, 3)
        self.reconstruction2 = BiSRNet(self.config.out_channels, self.config.hidden_channels, self.config.hidden_channels)

        self.reconstruction4 = nn.Conv2d(self.config.out_channels, self.config.hidden_channels, 3, 1, 1)#  S3KAIResNet(self.config.out_channels, self.config.num_classes, 2)# 
        self.rgb = FixedPosLinear(self.config.in_channels, self.config.out_channels,  self.config.minBand, self.config.maxBand, self.config.nBandDataset, self.config.dataset)
        self.herosnet = HerosNet(Ch=self.config.out_channels, stages=8)
        self.GST = GST_MODEL(in_ch=self.config.out_channels,
                           out_ch=self.config.out_channels,
                           noise_mean=0.1,
                           noise_std=0.1,
                           init="normal",
                        #    conv = default_conv,
                           noise_act=nn.Softplus(),
                           inter_channels=10,
                           spatial_scale=4)
        self.time_ = 0
        self.BS = PCA(n_components=self.config.hidden_channels)
        self.start_cls = False
        self.changed_f1base = 10
        self.turn = False
        self.pre_rec_x = None
        self.pre_x_middle = None
        self.num = 1
        self.num_to_stop = 10
        self.indexes = []
        self.BandSelection1 = MyLinearLayer(self.config.in_channels, self.config.hidden_channels)
        self.BandSelection2 = E2E_BS(self.config.in_channels, self.config.hidden_channels, self.config.dataset)

        # self.BandSelection3 = BandSelectionLayer(self.config.in_channels, self.config.hidden_channels)
        self.BandSelection4 = SpectralAttentionNetwork(self.config.in_channels, self.config.hidden_channels)
        # self.BandSelection5 = SpectralAttentionNetwork(self.config.in_channels, self.config.hidden_channels)
        self.BandSelection6 = BAM2(self.config.in_channels, self.config.hidden_channels)

        self.BandSelection_turn = False
        self.BandSelection_l = []
        self.band_selected = False
        self.start_cls_time = 0
        self.start_rec = False
        self.start_rec_time = 0
        self.att_reg = 0
        self.rec_loss_threshold = 0.08

    def top_down(self, top, lateral):
        top2x = F.interpolate(top, scale_factor=2.0, mode='nearest')
        return lateral + top2x # 
    
    def forward(self, x, y=None, w=None, **kwargs):
        if self.start_cls:
            # self.srf.weight_.requires_grad  = False
            # MST++
            for name, param in self.reconstruction1.named_parameters():
                if name == 'conv_out.weight' or name == 'conv_in.weight':
                    continue
                else:
                    param.requires_grad = False

        x_hsi = x
        if self.training:
            self.time_ += 1

        x_hsi_bs = x_hsi[:,[  2,  14,  19,  23,  37,  45,  55,  71,  73,  77, 102, 108, 119, 128, 129, 131, 132, 144, 158, 163], :, :]
        feat_list = []

        x = x_hsi.permute(0,2,3,1)
        x = self.rgb(x) 
        x = x.permute(0,3,1,2)
        image_tensor = x[0]

        # 将 (C, H, W) 转换为 (H, W, C)
        image_numpy = image_tensor.cpu().permute(1, 2, 0).numpy()

        # 归一化到 [0, 1] 范围
        image_numpy = np.clip(image_numpy, 0, None)  # 将负值设为0
        image_numpy = image_numpy / image_numpy.max()  # 归一化到 [0, 1]

        # 可视化图像
        plt.imshow(image_numpy)
        plt.axis('off')  # 隐藏坐标轴

        # 保存图像
        plt.savefig('output_normalized_image.png', bbox_inches='tight', pad_inches=0)       

        # 显示图像
        plt.show()
        x = self.reconstruction1(x)
        x_rec = x

        for op in self.feature_ops:
            x = op(x)
            if isinstance(op, nn.Identity):
                feat_list.append(x)

        inner_feat_list = [self.reduce_1x1convs[i](feat) for i, feat in enumerate(feat_list)]
        inner_feat_list.reverse()

        out_feat_list = [self.fuse_3x3convs[0](inner_feat_list[0])]
        for i in range(len(feat_list) - 1):
            inner = self.top_down(out_feat_list[i], inner_feat_list[i+1])
            out = self.fuse_3x3convs[i + 1](inner)
            out_feat_list.append(out)

        final_feat = out_feat_list[-1]

        logit = self.cls_pred_conv(final_feat)

        if self.training:
            # 0.013 for SV
            # 0.73 for train1 whlk 0.08 for train2 whlk 0.2 for train3 whlk
            if (self.turn or self.time_ > 2800): # self.time_> 1600 or  0.019 for sv 0.16 for whlk5 0.26 for whlk3 in train3 0.029/0.03 for whhc l1loss1(x_rec, x_hsi_bs) < self.rec_loss_threshold or 
                if not self.turn:
                    # print('ok')
                    self.start_cls_time = self.time_
                    self.turn = True
                    self.rec_loss_threshold = l1loss1(x_rec, x_hsi_bs)

                l1_loss = l1loss1(x_hsi_bs, x_rec)
                cls_loss = self.loss(logit, y, w)

                l1_loss_l.append(l1_loss.item())
                cls_loss_l.append(cls_loss.item())
                
                loss_dict = {
                    'cls_loss': cls_loss / 5.0 ,
                    }
                if l1_loss > self.rec_loss_threshold:
                    loss_dict = {
                        'cls_loss': cls_loss / 5.0,
                        'l1_loss': l1_loss / 10.0,
                    }

                # if cls_loss < l1_loss and self.num_to_stop > 0:

                #     self.num_to_stop = self.num_to_stop - 1

                #     sys.exit()
                # self.turn = True


                # MST++
                for name, param in self.reconstruction1.named_parameters():
                    if name == 'conv_out.weight' or name == 'conv_in.weight':
                        continue
                    else:
                        param.requires_grad = False


                
            else:
                        
                loss_dict = {
                    'l1_loss': l1loss1(x_rec, x_hsi_bs),# + self.loss(logit, y, w) , # + abs(20 - PSNR(x_hsi, x_rec)), # 
                # 'cls_loss': self.loss(logit, y, w)
                }
                
                
            return loss_dict
    
        if self.time_ == 2998:
            print('ok')
            plt.figure()
            loss_show = np.vstack((np.array(l1_loss_l), np.array(cls_loss_l)))
            plt.axhline(y=0.013, color='r', linestyle='--')
            plt.plot(np.arange(self.start_cls_time - 1, 2998, 1), loss_show.T)
            plt.savefig('2losses.jpg')
            plt.show()
        
        # if self.time_ == 1998:
        #     print('ok')
        #     # loss_show = np.vstack((np.array(l1_loss_l), np.array(cls_loss_l)))
        #     # loss_show = np.array(l1_loss_l)
        #     plt.axhline(y=0.019, color='r', linestyle='--')
        #     plt.plot(np.arange(self.start_rec_time, 999, 1), (np.array(l1_loss_l_1).T))
        #     plt.savefig('loss.jpg')
        #     plt.show()
        
        if not self.training:
            print('PSNR:' + str(PSNR(x_hsi_bs, x_rec)))

        return torch.softmax(logit, dim=1)

    def loss(self, x, y, weight):
        losses = F.cross_entropy(x, y.long() - 1, weight=None,
                                 ignore_index=-1, reduction='none')

        v = losses.mul_(weight).sum() / weight.sum()
        return v

    def set_defalut_config(self):
        self.config.update(dict(
            in_channels=204,
            num_classes=16,
            block_channels=(96, 128, 192, 256),
            num_blocks=(1, 1, 1, 1),
            inner_dim=128,
            reduction_ratio=1.0,
        ))

@registry.MODEL.register('FreeNetE2E_BA_Rec_SS')
class FreeNetE2E_BA_Rec_SS(CVModule):
    def __init__(self, config):
        super(FreeNetE2E_BA_Rec_SS, self).__init__(config)
        r = int(16 * self.config.reduction_ratio)
        block1_channels = int(self.config.block_channels[0] * self.config.reduction_ratio / r) * r
        block2_channels = int(self.config.block_channels[1] * self.config.reduction_ratio / r) * r
        block3_channels = int(self.config.block_channels[2] * self.config.reduction_ratio / r) * r
        block4_channels = int(self.config.block_channels[3] * self.config.reduction_ratio / r) * r
        # print(self.config.in_channels)

        self.feature_ops = nn.ModuleList([
            conv3x3_gn_relu(self.config.hidden_channels, block1_channels, r),

            repeat_block(block1_channels, r, self.config.num_blocks[0]),
            nn.Identity(),
            downsample2x(block1_channels, block2_channels),

            repeat_block(block2_channels, r, self.config.num_blocks[1]),
            nn.Identity(),
            downsample2x(block2_channels, block3_channels),

            repeat_block(block3_channels, r, self.config.num_blocks[2]),
            nn.Identity(),
            downsample2x(block3_channels, block4_channels),

            repeat_block(block4_channels, r, self.config.num_blocks[3]),
            nn.Identity(),
        ])
        inner_dim = int(self.config.inner_dim * self.config.reduction_ratio)

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
        # self.batch_size = self.config.rec_bs
        self.in_dim = self.config.in_channels
        self.out_dim = self.config.out_channels
        self.resrf = nn.Parameter(torch.rand((self.out_dim, self.in_dim)))
        self.cls_pred_conv = nn.Conv2d(inner_dim, self.config.num_classes, 1)
        self.srf = LyotFilter(self.config.in_channels, self.config.out_channels,  self.config.minBand, self.config.maxBand, self.config.nBandDataset, self.config.dataset)
        # self.resrf = reLyotFilter(self.config.in_channels, self.config.out_channels,  self.config.minBand, self.config.maxBand, self.config.nBandDataset, self.config.dataset)
        self.dynamic1 = S3KAIResNet(self.config.out_channels, self.config.num_classes, 2)# nn.Conv2d(self.config.out_channels, self.config.out_channels * self.config.out_channels, 1, 1, 0)#
        self.dynamic2 = nn.Conv2d(self.config.out_channels, self.config.in_channels, 1, 1, 0)#  S3KAIResNet(self.config.out_channels, self.config.num_classes, 2)# 
        self.dynamic3 = RSSAN5(self.config.num_classes, self.config.out_channels, 3, self.config.out_channels, 1, 1, 1, self.config.hidden_channels)
        self.dynamic4 = SEBlock(self.config.out_channels, 2)

        self.reconstruction1 = MST_Plus_Plus(self.config.out_channels, self.config.hidden_channels,  self.config.hidden_channels, 3)
        self.reconstruction2 = BiSRNet(self.config.out_channels, self.config.hidden_channels, self.config.hidden_channels)

        self.reconstruction4 = nn.Conv2d(self.config.out_channels, self.config.hidden_channels, 3, 1, 1)#  S3KAIResNet(self.config.out_channels, self.config.num_classes, 2)# 

        self.herosnet = HerosNet(Ch=self.config.out_channels, stages=8)
        self.GST = GST_MODEL(in_ch=self.config.out_channels,
                           out_ch=self.config.out_channels,
                           noise_mean=0.1,
                           noise_std=0.1,
                           init="normal",
                        #    conv = default_conv,
                           noise_act=nn.Softplus(),
                           inter_channels=10,
                           spatial_scale=4)
        self.time_ = 0
        self.BS = PCA(n_components=self.config.hidden_channels)
        self.start_cls = False
        self.changed_f1base = 10
        self.turn = False
        self.pre_rec_x = None
        self.pre_x_middle = None
        self.num = 1
        self.num_to_stop = 10
        self.indexes = []
        self.BandSelection1 = MyLinearLayer(self.config.in_channels, self.config.hidden_channels)
        self.BandSelection2 = E2E_BS(self.config.in_channels, self.config.hidden_channels, self.config.dataset)

        # self.BandSelection3 = BandSelectionLayer(self.config.in_channels, self.config.hidden_channels)
        self.BandSelection4 = SpectralAttentionNetwork(self.config.in_channels, self.config.hidden_channels)
        # self.BandSelection5 = SpectralAttentionNetwork(self.config.in_channels, self.config.hidden_channels)
        self.BandSelection6 = BAM2(self.config.in_channels, self.config.hidden_channels)

        self.BandSelection_turn = False
        self.BandSelection_l = []
        self.band_selected = False
        self.start_cls_time = 0
        self.start_rec = False
        self.start_rec_time = 0
        self.att_reg = 0
        self.rec_loss_threshold = 0.08
        self.seed = None

    def top_down(self, top, lateral):
        top2x = F.interpolate(top, scale_factor=2.0, mode='nearest')
        return lateral + top2x # 
    
    def forward(self, x, y=None, w=None, segments=None, **kwargs):
        if self.start_cls:
            # self.srf.weight_.requires_grad  = False
            # MST++
            for name, param in self.reconstruction1.named_parameters():
                if name == 'conv_out.weight' or name == 'conv_in.weight':
                    continue
                else:
                    param.requires_grad = False

        x_hsi = x
        if self.training:
            self.time_ += 1

        x_hsi_bs = x_hsi[:,[  2,   7,  12,  26,  29,  30,  31,  41,  60,  61,  66,  85,  94,  95, 96, 100, 120, 128, 129, 130], :, :]
        feat_list = []

        x = x_hsi.permute(0,2,3,1)
        x = self.srf(x) 
        x = x.permute(0,3,1,2)

        x = self.reconstruction1(x)
        x_rec = x

        for op in self.feature_ops:
            x = op(x)
            if isinstance(op, nn.Identity):
                feat_list.append(x)

        inner_feat_list = [self.reduce_1x1convs[i](feat) for i, feat in enumerate(feat_list)]
        inner_feat_list.reverse()

        out_feat_list = [self.fuse_3x3convs[0](inner_feat_list[0])]
        for i in range(len(feat_list) - 1):
            inner = self.top_down(out_feat_list[i], inner_feat_list[i+1])
            out = self.fuse_3x3convs[i + 1](inner)
            out_feat_list.append(out)

        final_feat = out_feat_list[-1]

        logit = self.cls_pred_conv(final_feat)
        # if self.time_== 3:
        #     print('ok')
        #     final_feat = torch.masked_select(final_feat.cpu().detach().reshape(final_feat.shape[1], -1), w.bool().view(-1).cpu().detach())
        #     X = final_feat.squeeze(0).view(final_feat.squeeze(0).shape[0],-1).transpose(1,0).cpu().detach().numpy()
        #     Y = TSNE(n_components=2, perplexity=1).fit_transform(X)
        #     color = y.squeeze(0).view(-1).cpu().detach().numpy()
        #     plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
        # # # plt.show(fig)
        # # # plt.imshow(final_feat_embedded, cmap='hot',interpolation='nearest')
        # # # plt.colorbar()
        #     plt.show()
        if self.training:
            # logit_cpu = logit.to('cpu')
            # segments_cpu = segments.to('cpu')
            entropy_minimization_loss = calculate_entropy_minimization_loss(logit, segments)
            entropy_minimization_loss_l.append(entropy_minimization_loss.data.item())
            # total_var_superpixel = calculate_superpixel_variance(torch.softmax(logit, dim=1).squeeze(0).argmax(dim=0) + 1, segments)
            # total_var_superpixel_l.append(total_var_superpixel.data.item())

            # 0.013 for SV
            # 0.73 for train1 whlk 0.08 for train2 whlk 0.2 for train3 whlk
            if (self.turn or self.time_ > 2800): # self.time_> 1600 or  0.019 for sv 0.16 for whlk5 0.26 for whlk3 in train3 0.029/0.03 for whhc l1loss1(x_rec, x_hsi_bs) < self.rec_loss_threshold or 
                if not self.turn:
                    # print('ok')
                    self.start_cls_time = self.time_
                    self.turn = True
                    self.rec_loss_threshold = l1loss1(x_rec, x_hsi_bs)
                
                # if total_var_superpixel < 500:
                #     total_var_superpixel = torch.tensor(0.0).cuda()

                l1_loss = l1loss1(x_hsi_bs, x_rec)
                cls_loss = self.loss(logit, y, w)

                l1_loss_l.append(l1_loss.item())
                cls_loss_l.append(cls_loss.item())
                
                loss_dict = {
                    'cls_loss': cls_loss / 5.0 ,
                    'entropy_minimization_loss' : entropy_minimization_loss / 100.0,
                    }
                if l1_loss > self.rec_loss_threshold:
                    loss_dict = {
                        'cls_loss': cls_loss / 5.0,
                        'l1_loss': l1_loss / 10.0,
                        'entropy_minimization_loss' : entropy_minimization_loss / 100.0,
                    }

                # if cls_loss < l1_loss and self.num_to_stop > 0:

                #     self.num_to_stop = self.num_to_stop - 1
                #     # if self.num_to_stop < 0:
                #     sys.exit()
                # self.turn = True


                # MST++
                for name, param in self.reconstruction1.named_parameters():
                    if name == 'conv_out.weight' or name == 'conv_in.weight':
                        continue
                    else:
                        param.requires_grad = False


                
            else:
                        
                loss_dict = {
                    'l1_loss': l1loss1(x_rec, x_hsi_bs),# + self.loss(logit, y, w) , # + abs(20 - PSNR(x_hsi, x_rec)), # 
                # 'cls_loss': self.loss(logit, y, w)
                }
                
                
            return loss_dict
    
        if self.time_ == 4998:
            print('ok')
            plt.figure()
            loss_show = np.vstack((np.array(l1_loss_l), np.array(cls_loss_l)))
            plt.axhline(y=0.013, color='r', linestyle='--')
            plt.plot(np.arange(self.start_cls_time - 1, 4998, 1), loss_show.T)
            plt.savefig(str(self.seed) + '_2losses_' + str(self.mode) + '.jpg')
            # plt.show()

            # plt.figure()
            # plt.plot(np.arange(0, 4998, 1), np.array(entropy_minimization_loss_l))
            # plt.savefig('entropy_minimization_loss.jpg')
            # plt.show()
        
        # if self.time_ == 1998:
        #     print('ok')
        #     # loss_show = np.vstack((np.array(l1_loss_l), np.array(cls_loss_l)))
        #     # loss_show = np.array(l1_loss_l)
        #     plt.axhline(y=0.019, color='r', linestyle='--')
        #     plt.plot(np.arange(self.start_rec_time, 999, 1), (np.array(l1_loss_l_1).T))
        #     plt.savefig('loss.jpg')
        #     plt.show()
        
        if not self.training:
            print('PSNR:' + str(PSNR(x_hsi_bs, x_rec)))

        return torch.softmax(logit, dim=1)

    def loss(self, x, y, weight):
        losses = F.cross_entropy(x, y.long() - 1, weight=None,
                                 ignore_index=-1, reduction='none')

        v = losses.mul_(weight).sum() / weight.sum()
        return v

    def set_defalut_config(self):
        self.config.update(dict(
            in_channels=204,
            num_classes=16,
            block_channels=(96, 128, 192, 256),
            num_blocks=(1, 1, 1, 1),
            inner_dim=128,
            reduction_ratio=1.0,
        ))

@registry.MODEL.register('FreeNetRecRGB')
class FreeNetRecRGB(CVModule):
    def __init__(self, config):
        super(FreeNetRecRGB, self).__init__(config)
        r = int(16 * self.config.reduction_ratio)
        block1_channels = int(self.config.block_channels[0] * self.config.reduction_ratio / r) * r
        block2_channels = int(self.config.block_channels[1] * self.config.reduction_ratio / r) * r
        block3_channels = int(self.config.block_channels[2] * self.config.reduction_ratio / r) * r
        block4_channels = int(self.config.block_channels[3] * self.config.reduction_ratio / r) * r
        # print(self.config.in_channels)
        self.feature_ops = nn.ModuleList([
            conv3x3_gn_relu(self.config.hidden_channels, block1_channels, r),

            repeat_block(block1_channels, r, self.config.num_blocks[0]),
            nn.Identity(),
            downsample2x(block1_channels, block2_channels),

            repeat_block(block2_channels, r, self.config.num_blocks[1]),
            nn.Identity(),
            downsample2x(block2_channels, block3_channels),

            repeat_block(block3_channels, r, self.config.num_blocks[2]),
            nn.Identity(),
            downsample2x(block3_channels, block4_channels),

            repeat_block(block4_channels, r, self.config.num_blocks[3]),
            nn.Identity(),
        ])
        inner_dim = int(self.config.inner_dim * self.config.reduction_ratio)
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
        # self.batch_size = self.config.rec_bs
        self.in_dim = self.config.in_channels
        self.out_dim = self.config.out_channels
        self.resrf = nn.Parameter(torch.rand((self.out_dim, self.in_dim)))
        self.cls_pred_conv = nn.Conv2d(inner_dim, self.config.num_classes, 1)
        
        self.srf = LyotFilter(self.config.in_channels, self.config.out_channels,  self.config.minBand, self.config.maxBand, self.config.nBandDataset, self.config.dataset)
        # self.resrf = reLyotFilter(self.config.in_channels, self.config.out_channels,  self.config.minBand, self.config.maxBand, self.config.nBandDataset, self.config.dataset)
        self.dynamic1 = S3KAIResNet(self.config.out_channels, self.config.num_classes, 2)# nn.Conv2d(self.config.out_channels, self.config.out_channels * self.config.out_channels, 1, 1, 0)#
        self.dynamic2 = nn.Conv2d(self.config.out_channels, self.config.in_channels, 1, 1, 0)#  S3KAIResNet(self.config.out_channels, self.config.num_classes, 2)# 
        self.dynamic3 = RSSAN5(self.config.num_classes, self.config.out_channels, 3, self.config.out_channels, 1, 1, 1, self.config.hidden_channels)
        self.dynamic4 = SEBlock(self.config.out_channels, 2)

        self.reconstruction1 = MST_Plus_Plus(self.config.out_channels, self.config.hidden_channels,  self.config.hidden_channels, 3)
        self.reconstruction2 = BiSRNet(self.config.out_channels, self.config.hidden_channels, self.config.hidden_channels)
        # self.reconstruction3 = nn.Conv3d(
        #             in_channels=self.config.out_channels,
        #             out_channels=self.config.hidden_channels,
        #             kernel_size=(3, 1, 1),
        #             stride=(1, 1, 1),
        #             padding=(1, 0, 0))
        self.reconstruction4 = nn.Conv2d(self.config.out_channels, self.config.hidden_channels, 3, 1, 1)#  S3KAIResNet(self.config.out_channels, self.config.num_classes, 2)# 

        self.herosnet = HerosNet(Ch=self.config.out_channels, stages=8)
        self.GST = GST_MODEL(in_ch=self.config.out_channels,
                           out_ch=self.config.out_channels,
                           noise_mean=0.1,
                           noise_std=0.1,
                           init="normal",
                        #    conv = default_conv,
                           noise_act=nn.Softplus(),
                           inter_channels=10,
                           spatial_scale=4)
        self.time_= 0
        self.BS = PCA(n_components=self.config.hidden_channels)
        self.start_cls = 0
    
        self.turn = False
        self.pre_rec_x = None
        self.pre_x_middle = None
        self.num = 30
        self.indexes = []
        self.rgb = FixedPosLinear(self.config.in_channels, self.config.out_channels,  self.config.minBand, self.config.maxBand, self.config.nBandDataset, self.config.dataset)

    # def entropy(labels, base=2):
    #     value, counts = np.unique(labels, return_counts=True)
    #     norm_counts = counts / counts.sum()
    #     return -(norm_counts * np.log(norm_counts) / np.log(base)).sum()

    def top_down(self, top, lateral):
        top2x = F.interpolate(top, scale_factor=2.0, mode='nearest')
        return lateral + top2x # 
    
    def forward(self, x, y=None, w=None, **kwargs):

        self.indexes = [i for i in range(x.shape[1])]

        if self.turn:

            # self.srf.weight_.requires_grad  = False

            # self.reconstruction4.weight.requires_grad = False
            # self.reconstruction4.bias.requires_grad = False

            # MST++
            for name, param in self.reconstruction1.named_parameters():
                if name == 'conv_out.weight' or name == 'conv_in.weight':
                    continue
                else:
                    param.requires_grad = False

        self.time_+= 1
        x_hsi = x
        b_, c_, h_, w_ = x_hsi.shape

        x_hsi = x_hsi.permute(0,2,3,1)
        x_rgb = self.rgb(x_hsi) 
        x_rgb = x_rgb.permute(0,3,1,2)
        x_hsi_bs = x_rgb

        x = x.permute(0,2,3,1)
        x = self.srf(x) 
        x = x.permute(0,3,1,2)
        # x_middle = x
        # for i in range(self.config.hidden_channels):
        # sv: [23, 24, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 55, 63, 64, 66, 77, 78, 94]
        # wh: [0, 74, 75, 98, 101, 109, 114, 116, 125, 128, 131, 132, 136, 241, 242, 243, 244, 247, 248, 249]

        
        
        # x_hsi_bs = x_hsi[:, 0:self.config.in_channels - 10:int((self.config.in_channels - 10)/self.config.hidden_channels),:,:]
        # x_hsi_bs = torch.tensor(self.BS.fit_transform(x_hsi.reshape(x_hsi.shape[0],\
        #                                     x_hsi.shape[1], x_hsi.shape[2] \
        #                                     * x_hsi.shape[3]).squeeze(0).T\
        #                                     .cpu().detach()).T\
        #                                     .reshape(b_, self.config.hidden_channels, h_, w_))\
        #                                     .cuda()
        # x = x.reshape(self.batch_size * self.batch_size, x.shape[1], int(x.shape[2]/self.batch_size), int(x.shape[3]/self.batch_size))

        x = self.reconstruction1(x) # / 2.0
        # x = x.reshape(b_, x.shape[1], int(x.shape[2] * self.batch_size), int(x.shape[3] * self.batch_size))
        x_rec = x
        # if self.turn and self.training:
        #     assert (self.pre_x_middle == x_middle).all()
        #     assert (self.pre_rec_x == x_rec).all()

        feat_list = []
        for op in self.feature_ops:
            x = op(x)
            if isinstance(op, nn.Identity):
                feat_list.append(x)

        inner_feat_list = [self.reduce_1x1convs[i](feat) for i, feat in enumerate(feat_list)]
        inner_feat_list.reverse()

        out_feat_list = [self.fuse_3x3convs[0](inner_feat_list[0])]
        for i in range(len(feat_list) - 1):
            inner = self.top_down(out_feat_list[i], inner_feat_list[i+1])
            out = self.fuse_3x3convs[i + 1](inner)
            out_feat_list.append(out)

        final_feat = out_feat_list[-1]

        logit = self.cls_pred_conv(final_feat)

        if self.time_== 1998:
            print('ok')
            loss_show = np.vstack((np.array(l1_loss_l),np.array(cls_loss_l)))
            # loss_show = np.array(l1_loss_l)
            plt.axhline(y=0.16, color='r', linestyle='--')
            plt.plot(np.arange(self.start_cls, 999, 1), loss_show.T)
            plt.savefig('loss.jpg')
            # plt.show()

        # if self.time_== 1996:
        #     print('ok')
            
        #     plt.plot(psnr)
        #     plt.savefig('psnr.jpg')
        
        if self.training:
            # l1_loss = l1loss1(x_hsi_bs, x_rec)
            # # cls_loss = self.loss(logit, y, w)
           
            # l1_loss_l.append(l1_loss.item())
            # cls_loss_l.append(cls_loss.item())
            # self.time_> 1200 or 
            if self.time_> 1600 or (l1loss1(x_rec, x_hsi_bs) < 0.16 or self.turn): # 0.019 for sv 0.16 for whlk5 0.26 for whlk3 in train3 0.029/0.03 for whhc
                if not self.turn:
                    # print('ok')
                    self.start_cls = int(self.time/2)
                l1_loss = l1loss1(x_hsi_bs, x_rec)
                cls_loss = self.loss(logit, y, w)

                l1_loss_l.append(l1_loss.item())
                cls_loss_l.append(cls_loss.item())
                # if not self.turn:
                #     rec = x_rec.detach().cpu().numpy()
                
                #     result = {"WHU_Hi_LongKou_train": rec}
                
                #     scipy.io.savemat("WHU_Hi_LongKou_train.mat", result)
                
                loss_dict = {
                    'cls_loss': cls_loss / 5.0 ,
                    # 'l1_loss': l1_loss/ 10.0,
                    }
                if l1_loss > 0.16:
                    loss_dict = {
                    'cls_loss': cls_loss/ 5.0,
                    'l1_loss': l1_loss/ 10.0,
                    }
                # if cls_loss < l1_loss and self.num > 0:
                #     # loss_dict = {
                #     # 'cls_loss': cls_loss/ 5.0,
                #     # # 'l1_loss': l1_loss/ 10.0,
                #     # }
                #     print('ok')
                #     self.num = self.num - 1
                #     # loss_show = np.vstack((np.array(l1_loss_l),np.array(cls_loss_l)))
                #     # # loss_show = np.array(l1_loss_l)
                #     # plt.axhline(y=0.16, color='r', linestyle='--')
                #     # plt.plot(loss_show.T)
                #     # plt.savefig('loss.jpg')
                #     # plt.show()
                #     sys.exit()
                self.turn = True
                # self.pre_rec_x = x_rec
                # self.pre_x_middle = x_middle

                    # # MST++
                for name, param in self.reconstruction1.named_parameters():
                    if name == 'conv_out.weight' or name == 'conv_in.weight':
                        continue
                    else:
                        param.requires_grad = False

                # for name, param in self.reconstruction4.named_parameters():
                    
                #     param.requires_grad = False

                # self.reconstruction4.weight.requires_grad = False
                # self.reconstruction4.bias.requires_grad = False

                
            else:
                        
                loss_dict = {
                    'l1_loss': l1loss1(x_rec, x_hsi_bs),# + self.loss(logit, y, w) , # + abs(20 - PSNR(x_hsi, x_rec)), # 
                # 'cls_loss': self.loss(logit, y, w)
                }
                
            
            return loss_dict
        
        
          

        if not self.training:
            # psnr.append(PSNR(x_hsi_bs, x_rec))
            print('PSNR:' + str(PSNR(x_hsi_bs, x_rec)))

        return torch.softmax(logit, dim=1)
    
    def show_avg_var(self, image, mask, x_rec, title):
        srf = np.zeros([len(np.unique(mask).tolist()) - 1, x_rec.shape[1]])
        srf_var = np.zeros([len(np.unique(mask).tolist()) - 1, x_rec.shape[1]])

        image_by_class = []

                    # 每个类的位置
        for b in range(len(np.unique(mask).tolist()) - 1):
            image_by_class.append(np.where(mask[:,:]==b+1)[0])
            image_by_class.append(np.where(mask[:,:]==b+1)[1])
                        # print('ok')

        for j in range(len(np.unique(mask).tolist()) - 1):
            srf_one_class = np.zeros([len(image_by_class[2*j]), x_rec.shape[1]])
            for k in range(len(image_by_class[2*j])):
                srf_one_class[k] = image[image_by_class[2*j][k], image_by_class[2*j + 1][k]]
            srf[j] = srf_one_class.mean(axis=0)
            srf_var[j] = srf_one_class.var(axis=0)

                    # plt.plot(srf.T)
                    # plt.show()
                    # plt.xticks()
        plt.title('var', fontsize=24)
        
                    # plt.axvline(700)
                    # plt.plot(srf_var.T)
                    # plt.savefig('var.png')
        plt.subplot(1, 2, 1)
        plt.plot( srf.T)
        # plt.plot(srf.T)
        plt.subplot(1, 2, 2)
        plt.plot( srf_var.T)
        # ax = plt.gca()
        # ax.yaxis.get_offset_text().set(size=20)  # 左上角
        plt.yticks(size = 20)
        plt.xticks(size = 20)
        plt.savefig(str(title) + '.png')
                    # plt.show()

    def loss(self, x, y, weight):
        losses = F.cross_entropy(x, y.long() - 1, weight=None,
                                 ignore_index=-1, reduction='none')

        v = losses.mul_(weight).sum() / weight.sum()
        return v

    def set_defalut_config(self):
        self.config.update(dict(
            in_channels=204,
            num_classes=16,
            block_channels=(96, 128, 192, 256),
            num_blocks=(1, 1, 1, 1),
            inner_dim=128,
            reduction_ratio=1.0,
        ))

@registry.MODEL.register('FreeNetFewDynamicChannelsVoltage2stageHC')
class FreeNetFewDynamicChannelsVoltage2stageHC(CVModule):
    def __init__(self, config):
        super(FreeNetFewDynamicChannelsVoltage2stageHC, self).__init__(config)
        r = int(16 * self.config.reduction_ratio)
        block1_channels = int(self.config.block_channels[0] * self.config.reduction_ratio / r) * r
        block2_channels = int(self.config.block_channels[1] * self.config.reduction_ratio / r) * r
        block3_channels = int(self.config.block_channels[2] * self.config.reduction_ratio / r) * r
        block4_channels = int(self.config.block_channels[3] * self.config.reduction_ratio / r) * r
        # print(self.config.in_channels)
        self.feature_ops = nn.ModuleList([
            conv3x3_gn_relu(self.config.hidden_channels, block1_channels, r),

            repeat_block(block1_channels, r, self.config.num_blocks[0]),
            nn.Identity(),
            downsample2x(block1_channels, block2_channels),

            repeat_block(block2_channels, r, self.config.num_blocks[1]),
            nn.Identity(),
            downsample2x(block2_channels, block3_channels),

            repeat_block(block3_channels, r, self.config.num_blocks[2]),
            nn.Identity(),
            downsample2x(block3_channels, block4_channels),

            repeat_block(block4_channels, r, self.config.num_blocks[3]),
            nn.Identity(),
        ])
        inner_dim = int(self.config.inner_dim * self.config.reduction_ratio)
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
        # self.batch_size = self.config.rec_bs
        self.in_dim = self.config.in_channels
        self.out_dim = self.config.out_channels
        self.resrf = nn.Parameter(torch.rand((self.out_dim, self.in_dim)))
        self.cls_pred_conv = nn.Conv2d(inner_dim, self.config.num_classes, 1)
        
        self.srf = LyotFilter(self.config.in_channels, self.config.out_channels,  self.config.minBand, self.config.maxBand, self.config.nBandDataset, self.config.dataset)
        # self.resrf = reLyotFilter(self.config.in_channels, self.config.out_channels,  self.config.minBand, self.config.maxBand, self.config.nBandDataset, self.config.dataset)
        self.dynamic1 = S3KAIResNet(self.config.out_channels, self.config.num_classes, 2)# nn.Conv2d(self.config.out_channels, self.config.out_channels * self.config.out_channels, 1, 1, 0)#
        self.dynamic2 = nn.Conv2d(self.config.out_channels, self.config.in_channels, 1, 1, 0)#  S3KAIResNet(self.config.out_channels, self.config.num_classes, 2)# 
        self.dynamic3 = RSSAN5(self.config.num_classes, self.config.out_channels, 3, self.config.out_channels, 1, 1, 1, self.config.hidden_channels)
        self.dynamic4 = SEBlock(self.config.out_channels, 2)

        self.reconstruction1 = MST_Plus_Plus(self.config.out_channels, self.config.hidden_channels,  self.config.hidden_channels, 3)
        self.reconstruction2 = BiSRNet(self.config.out_channels, self.config.hidden_channels, self.config.hidden_channels)
        # self.reconstruction3 = nn.Conv3d(
        #             in_channels=self.config.out_channels,
        #             out_channels=self.config.hidden_channels,
        #             kernel_size=(3, 1, 1),
        #             stride=(1, 1, 1),
        #             padding=(1, 0, 0))
        self.reconstruction4 = nn.Conv2d(self.config.out_channels, self.config.hidden_channels, 3, 1, 1)#  S3KAIResNet(self.config.out_channels, self.config.num_classes, 2)# 

        self.herosnet = HerosNet(Ch=self.config.out_channels, stages=8)
        self.GST = GST_MODEL(in_ch=self.config.out_channels,
                           out_ch=self.config.out_channels,
                           noise_mean=0.1,
                           noise_std=0.1,
                           init="normal",
                        #    conv = default_conv,
                           noise_act=nn.Softplus(),
                           inter_channels=10,
                           spatial_scale=4)
        self.time_ = 0
        self.BS = PCA(n_components=self.config.hidden_channels)
        self.start_cls = 0
        self.changed_f1base = 10
        self.turn = False
        self.pre_rec_x = None
        self.pre_x_middle = None
        self.num = 1
        self.num_to_stop = 30
        self.indexes = []
        

    # def entropy(labels, base=2):
    #     value, counts = np.unique(labels, return_counts=True)
    #     norm_counts = counts / counts.sum()
    #     return -(norm_counts * np.log(norm_counts) / np.log(base)).sum()
    def checkMulticollinearity(self, s=None, trainx=None):
        """Calculate the VIF value of each selected band in s"""
        vifV = []
        nbands = len(s)
        trainx, _, _ = normalize(trainx)
        for n, i in enumerate(s):
            y = trainx[:, 0, np.where(np.array(self.indexes) == i)[0][0], :, :]
            x = np.zeros((trainx.shape[0], trainx.shape[3], trainx.shape[4], nbands - 1))
            c = 0
            for nb in s:
                if nb != i:
                    x[:, :, :, c] = trainx[:, 0, np.where(np.array(self.indexes) == nb)[0][0], :, :]
                    c += 1
            x = x.reshape((x.shape[0] * x.shape[1] * x.shape[2], nbands - 1))
            y = y.reshape((y.shape[0] * y.shape[1] * y.shape[2], 1))
            model = sm.OLS(y, x)
            results = model.fit()
            rsq = results.rsquared
            vifV.append(round(1 / (1 - rsq), 2))
            # print("R Square value of {} band is {} keeping all other bands as features".format(s[n],
            #                                                                                    (round(rsq, 4))))
            # print("\t\t\tMulticolinearity analysis. Variance Inflation Factor of band {} is {}".format(s[n], vifV[n]))

        return vifV
    
    def tune_DFS(self):
        """Get the mean F1 validation score using a set of selected bands"""
        # X, Y = loadata(name='WHLK')
        # print('Initial image shape: ' + str(X.shape))
        # # X, Y = createImageCubes(X, Y, window=5)
        # # print('Processed dataset shape: ' + str(X.shape))

        # dataset = Dataset(train_x=X, train_y=Y, ind=False, name='WHLK')
        # selector = SelectBands(dataset=dataset,  method='GSS', nbands=20)
        # VIF_best, IBRA_best, GSS_best, stats_best = selector.run_selection(init_vf=11, final_vf=11, preselected_bands=X.shape[2], config = self.config_)
        
        # torch.backends.cudnn.benchmark = True
        print('Start testing candidate selection: ' + str(self.selection))
        avg = 0
        SEED = [1, 2, 3, 4, 5]
        # torch.manual_seed(SEED)
        # torch.cuda.manual_seed(SEED)
        for i in range(len(SEED)):
            res = train.run(config_path='freenet.whlk.freenet_1_0_whlk_test',
                model_dir='./log/freenet_1_0_whlk_test/freenet/1.0_poly',
                cpu_mode=False,
                after_construct_launcher_callbacks=[register_evaluate_fn],
                opts=None,
                np_seed=SEED[i],
                selection=self.selection) # freenet_1_0_salinas_test
            avg = avg + res['launcher'].avg
            # if res['launcher'].avg < 0.8:
            #     break
        # print('ok')
        return avg / 5.0

    def top_down(self, top, lateral):
        top2x = F.interpolate(top, scale_factor=2.0, mode='nearest')
        return lateral + top2x # 
    
    def forward(self, x, y=None, w=None, **kwargs):
        x_hsi = x
        
        self.time_ += 1

        if self.time_ == 1:
            self.X_full = x.permute(0, 2, 3, 1).squeeze(0)
            self.Y_full = y.squeeze(0)
            # self.X_full = torch.load('/mnt/sdd/niuyuanzhuo/FreeNet-master/HC_X_full.pth').cuda()
            # self.Y_full = torch.load('/mnt/sdd/niuyuanzhuo/FreeNet-master/HC_Y_full.pth').cuda()
            # self.w_full = torch.load('/mnt/c/Users/17735/OneDrive/桌面/FreeNet-master/full_w.pth').cuda()
            trainx_hsi, _ = createImageCubes(self.X_full.cpu(), self.Y_full.cpu())
            trainx_hsi = trainx_hsi.reshape(trainx_hsi.shape[0], 1, trainx_hsi.shape[3], trainx_hsi.shape[1], trainx_hsi.shape[2])
            trainx_hsi, _, _ = normalize(trainx_hsi)
            self.indexes = [i for i in range(x.shape[1])]
            entropies =  [entropy(trainx_hsi[:, :, i, :, :]) for i in range(len(self.indexes))]

            # Sort the pre-selected bands according to their entropy (in decreasing order)
            self.preselected = self.indexes.copy()
            pairs = list(tuple(zip(self.preselected, entropies)))
            pairs.sort(key=lambda x: x[1], reverse=True)
            self.preselected, _ = zip(*pairs)

            # Select the first "select" bands
            self.preselected = list(self.preselected)
            self.selection = self.preselected[:self.config.hidden_channels]
            self.selection.sort()
            self.preselected = self.preselected[self.config.hidden_channels:]
            # self.selection = [0, 74, 75, 98, 101, 109, 114, 116, 125, 128, 131, 132, 136, 241, 242, 243, 244, 247, 248, 249]
            ct = 1
            print("\tAnalyzing candidate combination " + str(ct) + ". 5x2 CV using bands: " + str(self.selection))
            self.f1base = self.tune_DFS()
            print("\tMean F1: " + str(self.f1base))
            
            self.bestselection = self.selection.copy()
            ct += 1
        # # if self.training:
        # #     # print('Current F1:' + str(self.tune(self.x_full, self.mask_full, self.w_full)))
        # #     if self.tune_DFS() < 0.6:
        # #         self.turn = True
        # #     else:
        # #         self.turn = False
    
        # # self.selection = self.tune_DFS()
        # x = x_hsi[:, self.selection, :, :]
        
        if self.training and self.num > 0: #  and self.turn and self.num > 0 and len(self.preselected) > 0
            self.num = self.num - 1
            
            # Try new bands until there is no more elements in the list
            while len(self.preselected) > 0:
                if self.f1base > 0.4 or self.changed_f1base > 0:
                    break
                # Calculate the maximum VIF of all the band in "selection"

                VIF = self.checkMulticollinearity(s=self.selection, trainx=trainx_hsi) # (1, 94890, 1, 1, 20)
                # Remove the band with the highest VIF of "selection"
                self.selection.remove(self.selection[VIF.index(max(VIF))])
                # Pop the next available band from "preselected"
                self.selection.append(self.preselected[0])
                self.selection.sort()
                self.preselected = self.preselected[1:]

                # Train using the bands in "selection"
                print("\tAnalyzing candidate combination " + str(ct) + ". 5x2 CV using bands: " + str(self.selection))
                self.f1 = self.tune_DFS()
                # with open("/mnt/sdd/niuyuanzhuo/FreeNet-master/log1_LK.txt", encoding="utf-8",mode="a") as file:  
                #     file.write(str(self.selection) + '  ' + str(self.f1) +'\n')
                
                print("\tMean F1: " + str(self.f1))
                # Check if the new selection has better performance than the previous one. If not, break
                if self.f1 > self.f1base:
                    self.bestselection = self.selection.copy()
                    self.f1base = self.f1
                    self.changed_f1base = 10

                elif self.f1base > 0.4 or self.changed_f1base > 0:
                    break
                else:
                    self.changed_f1base = self.changed_f1base - 1
                ct += 1
                print("\tBest selection so far: " + str(self.bestselection) + "with an F1 score of " + str(self.f1base))
        
        # # if not self.turn:
        # # x = x_hsi[:, [0, 74, 75, 98, 101, 109, 114, 116, 125, 128, 131, 132, 136, 241, 242, 243, 244, 247, 248, 249], :, :]
        # print("\tBest selection so far: " + str(self.bestselection) + "with an F1 score of " + str(self.f1base))
        

        if self.turn:

            # self.srf.weight_.requires_grad  = False

            # self.reconstruction4.weight.requires_grad = False
            # self.reconstruction4.bias.requires_grad = False

            # MST++
            for name, param in self.reconstruction1.named_parameters():
                if name == 'conv_out.weight' or name == 'conv_in.weight':
                    continue
                else:
                    param.requires_grad = False

        # self.time_ += 1
        x_hsi = x
        b_, c_, h_, w_ = x_hsi.shape
        x = x.permute(0,2,3,1)
        x = self.srf(x) 
        x = x.permute(0,3,1,2)
        # x_middle = x
        # for i in range(self.config.hidden_channels):
        # sv: [23, 24, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 55, 63, 64, 66, 77, 78, 94]
        # wh: [0, 74, 75, 98, 101, 109, 114, 116, 125, 128, 131, 132, 136, 241, 242, 243, 244, 247, 248, 249]
        # entropies = [entropy(x_hsi[:, i, :, :]) for i in range(len(self.indexes))]

        # # Sort the pre-selected bands according to their entropy (in decreasing order)
        # preselected = self.indexes.copy()
        # pairs = list(tuple(zip(preselected, entropies)))
        # pairs.sort(key=lambda x: x[1], reverse=True)
        # preselected, _ = zip(*pairs)

        # # Select the first "select" bands
        # preselected = list(preselected)
        # selection = preselected[:self.config.hidden_channels]
        # selection.sort()
        # preselected = preselected[self.config.hidden_channels:]
        
        x_hsi_bs = x_hsi[:, self.bestselection,:,:]
        # x_hsi_bs = x_hsi[:, 0:self.config.in_channels - 10:int((self.config.in_channels - 10)/self.config.hidden_channels),:,:]
        # x_hsi_bs = torch.tensor(self.BS.fit_transform(x_hsi.reshape(x_hsi.shape[0],\
        #                                     x_hsi.shape[1], x_hsi.shape[2] \
        #                                     * x_hsi.shape[3]).squeeze(0).T\
        #                                     .cpu().detach()).T\
        #                                     .reshape(b_, self.config.hidden_channels, h_, w_))\
        #                                     .cuda()
        # x = x.reshape(self.batch_size * self.batch_size, x.shape[1], int(x.shape[2]/self.batch_size), int(x.shape[3]/self.batch_size))

        x = self.reconstruction1(x) # / 2.0
        # x = x.reshape(b_, x.shape[1], int(x.shape[2] * self.batch_size), int(x.shape[3] * self.batch_size))
        x_rec = x

        feat_list = []
        for op in self.feature_ops:
            x = op(x)
            if isinstance(op, nn.Identity):
                feat_list.append(x)

        inner_feat_list = [self.reduce_1x1convs[i](feat) for i, feat in enumerate(feat_list)]
        inner_feat_list.reverse()

        out_feat_list = [self.fuse_3x3convs[0](inner_feat_list[0])]
        for i in range(len(feat_list) - 1):
            inner = self.top_down(out_feat_list[i], inner_feat_list[i+1])
            out = self.fuse_3x3convs[i + 1](inner)
            out_feat_list.append(out)

        final_feat = out_feat_list[-1]

        logit = self.cls_pred_conv(final_feat)

        if self.time_ == 1998 and self.turn:
            print('ok')
            loss_show = np.vstack((np.array(l1_loss_l),np.array(cls_loss_l)))
            # loss_show = np.array(l1_loss_l)
            plt.axhline(y=0.16, color='r', linestyle='--')
            plt.plot(np.arange(self.start_cls, 999, 1), loss_show.T)
            plt.savefig('loss.jpg')
            plt.show()

        # if self.time_== 1996:
        #     print('ok')
            
        #     plt.plot(psnr)
        #     plt.savefig('psnr.jpg')
        
        if self.training:
            # l1_loss = l1loss1(x_hsi_bs, x_rec)
            # cls_loss = self.loss(logit, y, w)
           
            # l1_loss_l.append(l1_loss.item())
            # cls_loss_l.append(cls_loss.item())
            # self.time_> 1200 or 
            if (l1loss1(x_rec, x_hsi_bs) < 0.03 or self.turn): # self.time_> 1600 or  0.019 for sv 0.16 for whlk5 0.26 for whlk3 in train3 0.029/0.03 for whhc
                if not self.turn:
                    # print('ok')
                    self.start_cls = int(self.time_/2)
                l1_loss = l1loss1(x_hsi_bs, x_rec)
                cls_loss = self.loss(logit, y, w)

                l1_loss_l.append(l1_loss.item())
                cls_loss_l.append(cls_loss.item())
                # if not self.turn:
                #     rec = x_rec.detach().cpu().numpy()
                
                #     result = {"WHU_Hi_LongKou_train": rec}
                
                #     scipy.io.savemat("WHU_Hi_LongKou_train.mat", result)
                
                loss_dict = {
                    'cls_loss': cls_loss / 5.0 ,
                    # 'l1_loss': l1_loss/ 10.0,
                    }
                if l1_loss > 0.03:
                    loss_dict = {
                    'cls_loss': cls_loss/ 5.0,
                    'l1_loss': l1_loss/ 10.0,
                    }
                if cls_loss < l1_loss + 0.5 and self.num_to_stop > 0:
                    # loss_dict = {
                    # 'cls_loss': cls_loss/ 5.0,
                    # # 'l1_loss': l1_loss/ 10.0,
                    # }
                    # print('ok')
                    self.num_to_stop = self.num_to_stop - 1
                    # loss_show = np.vstack((np.array(l1_loss_l),np.array(cls_loss_l)))
                    # # loss_show = np.array(l1_loss_l)
                    # plt.axhline(y=0.16, color='r', linestyle='--')
                    # plt.plot(loss_show.T)
                    # plt.savefig('loss.jpg')
                    # plt.show()
                    sys.exit()

                self.turn = True
                # self.pre_rec_x = x_rec
                # self.pre_x_middle = x_middle

                # MST++
                for name, param in self.reconstruction1.named_parameters():
                    if name == 'conv_out.weight' or name == 'conv_in.weight':
                        continue
                    else:
                        param.requires_grad = False

                # for name, param in self.reconstruction4.named_parameters():
                    
                #     param.requires_grad = False

                # self.reconstruction4.weight.requires_grad = False
                # self.reconstruction4.bias.requires_grad = False

                
            else:
                        
                loss_dict = {
                    'l1_loss': l1loss1(x_rec, x_hsi_bs),# + self.loss(logit, y, w) , # + abs(20 - PSNR(x_hsi, x_rec)), # 
                # 'cls_loss': self.loss(logit, y, w)
                }
                
            
            return loss_dict
        
        
          

        if not self.training:
            # psnr.append(PSNR(x_hsi_bs, x_rec))
            print('PSNR:' + str(PSNR(x_hsi_bs, x_rec)))

        return torch.softmax(logit, dim=1)
    
    def show_avg_var(self, image, mask, x_rec, title):
        srf = np.zeros([len(np.unique(mask).tolist()) - 1, x_rec.shape[1]])
        srf_var = np.zeros([len(np.unique(mask).tolist()) - 1, x_rec.shape[1]])

        image_by_class = []

                    # 每个类的位置
        for b in range(len(np.unique(mask).tolist()) - 1):
            image_by_class.append(np.where(mask[:,:]==b+1)[0])
            image_by_class.append(np.where(mask[:,:]==b+1)[1])
                        # print('ok')

        for j in range(len(np.unique(mask).tolist()) - 1):
            srf_one_class = np.zeros([len(image_by_class[2*j]), x_rec.shape[1]])
            for k in range(len(image_by_class[2*j])):
                srf_one_class[k] = image[image_by_class[2*j][k], image_by_class[2*j + 1][k]]
            srf[j] = srf_one_class.mean(axis=0)
            srf_var[j] = srf_one_class.var(axis=0)

                    # plt.plot(srf.T)
                    # plt.show()
                    # plt.xticks()
        plt.title('var', fontsize=24)
        
                    # plt.axvline(700)
                    # plt.plot(srf_var.T)
                    # plt.savefig('var.png')
        plt.subplot(1, 2, 1)
        plt.plot( srf.T)
        # plt.plot(srf.T)
        plt.subplot(1, 2, 2)
        plt.plot( srf_var.T)
        # ax = plt.gca()
        # ax.yaxis.get_offset_text().set(size=20)  # 左上角
        plt.yticks(size = 20)
        plt.xticks(size = 20)
        plt.savefig(str(title) + '.png')
                    # plt.show()

    def loss(self, x, y, weight):
        losses = F.cross_entropy(x, y.long() - 1, weight=None,
                                 ignore_index=-1, reduction='none')

        v = losses.mul_(weight).sum() / weight.sum()
        return v

    def set_defalut_config(self):
        self.config.update(dict(
            in_channels=204,
            num_classes=16,
            block_channels=(96, 128, 192, 256),
            num_blocks=(1, 1, 1, 1),
            inner_dim=128,
            reduction_ratio=1.0,
        ))

@registry.MODEL.register('FreeNetFewDynamicChannelsVoltage2stageSV')
class FreeNetFewDynamicChannelsVoltage2stageSV(CVModule):
    def __init__(self, config):
        super(FreeNetFewDynamicChannelsVoltage2stageSV, self).__init__(config)
        r = int(16 * self.config.reduction_ratio)
        block1_channels = int(self.config.block_channels[0] * self.config.reduction_ratio / r) * r
        block2_channels = int(self.config.block_channels[1] * self.config.reduction_ratio / r) * r
        block3_channels = int(self.config.block_channels[2] * self.config.reduction_ratio / r) * r
        block4_channels = int(self.config.block_channels[3] * self.config.reduction_ratio / r) * r
        # print(self.config.in_channels)
        self.feature_ops = nn.ModuleList([
            conv3x3_gn_relu(self.config.hidden_channels, block1_channels, r),

            repeat_block(block1_channels, r, self.config.num_blocks[0]),
            nn.Identity(),
            downsample2x(block1_channels, block2_channels),

            repeat_block(block2_channels, r, self.config.num_blocks[1]),
            nn.Identity(),
            downsample2x(block2_channels, block3_channels),

            repeat_block(block3_channels, r, self.config.num_blocks[2]),
            nn.Identity(),
            downsample2x(block3_channels, block4_channels),

            repeat_block(block4_channels, r, self.config.num_blocks[3]),
            nn.Identity(),
        ])
        inner_dim = int(self.config.inner_dim * self.config.reduction_ratio)
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
        # self.batch_size = self.config.rec_bs
        self.in_dim = self.config.in_channels
        self.out_dim = self.config.out_channels
        self.resrf = nn.Parameter(torch.rand((self.out_dim, self.in_dim)))
        self.cls_pred_conv = nn.Conv2d(inner_dim, self.config.num_classes, 1)
        
        self.srf = LyotFilter(self.config.in_channels, self.config.out_channels,  self.config.minBand, self.config.maxBand, self.config.nBandDataset, self.config.dataset)
        # self.resrf = reLyotFilter(self.config.in_channels, self.config.out_channels,  self.config.minBand, self.config.maxBand, self.config.nBandDataset, self.config.dataset)
        self.dynamic1 = S3KAIResNet(self.config.out_channels, self.config.num_classes, 2)# nn.Conv2d(self.config.out_channels, self.config.out_channels * self.config.out_channels, 1, 1, 0)#
        self.dynamic2 = nn.Conv2d(self.config.out_channels, self.config.in_channels, 1, 1, 0)#  S3KAIResNet(self.config.out_channels, self.config.num_classes, 2)# 
        self.dynamic3 = RSSAN5(self.config.num_classes, self.config.out_channels, 3, self.config.out_channels, 1, 1, 1, self.config.hidden_channels)
        self.dynamic4 = SEBlock(self.config.out_channels, 2)

        self.reconstruction1 = MST_Plus_Plus(self.config.out_channels, self.config.hidden_channels,  self.config.hidden_channels, 3)
        self.reconstruction2 = BiSRNet(self.config.out_channels, self.config.hidden_channels, self.config.hidden_channels)
        # self.reconstruction3 = nn.Conv3d(
        #             in_channels=self.config.out_channels,
        #             out_channels=self.config.hidden_channels,
        #             kernel_size=(3, 1, 1),
        #             stride=(1, 1, 1),
        #             padding=(1, 0, 0))
        self.reconstruction4 = nn.Conv2d(self.config.out_channels, self.config.hidden_channels, 3, 1, 1)#  S3KAIResNet(self.config.out_channels, self.config.num_classes, 2)# 

        self.herosnet = HerosNet(Ch=self.config.out_channels, stages=8)
        self.GST = GST_MODEL(in_ch=self.config.out_channels,
                           out_ch=self.config.out_channels,
                           noise_mean=0.1,
                           noise_std=0.1,
                           init="normal",
                        #    conv = default_conv,
                           noise_act=nn.Softplus(),
                           inter_channels=10,
                           spatial_scale=4)
        self.time_ = 0
        self.BS = PCA(n_components=self.config.hidden_channels)
        self.start_cls = 0
        self.changed_f1base = 10
        self.turn = False
        self.pre_rec_x = None
        self.pre_x_middle = None
        self.num = 1
        self.num_to_stop = 30
        self.indexes = []
        

    # def entropy(labels, base=2):
    #     value, counts = np.unique(labels, return_counts=True)
    #     norm_counts = counts / counts.sum()
    #     return -(norm_counts * np.log(norm_counts) / np.log(base)).sum()
    def checkMulticollinearity(self, s=None, trainx=None):
        """Calculate the VIF value of each selected band in s"""
        vifV = []
        nbands = len(s)
        trainx, _, _ = normalize(trainx)
        for n, i in enumerate(s):
            y = trainx[:, 0, np.where(np.array(self.indexes) == i)[0][0], :, :]
            x = np.zeros((trainx.shape[0], trainx.shape[3], trainx.shape[4], nbands - 1))
            c = 0
            for nb in s:
                if nb != i:
                    x[:, :, :, c] = trainx[:, 0, np.where(np.array(self.indexes) == nb)[0][0], :, :]
                    c += 1
            x = x.reshape((x.shape[0] * x.shape[1] * x.shape[2], nbands - 1))
            y = y.reshape((y.shape[0] * y.shape[1] * y.shape[2], 1))
            model = sm.OLS(y, x)
            results = model.fit()
            rsq = results.rsquared
            vifV.append(round(1 / (1 - rsq), 2))
            # print("R Square value of {} band is {} keeping all other bands as features".format(s[n],
            #                                                                                    (round(rsq, 4))))
            # print("\t\t\tMulticolinearity analysis. Variance Inflation Factor of band {} is {}".format(s[n], vifV[n]))

        return vifV
    
    def tune_DFS(self):
        """Get the mean F1 validation score using a set of selected bands"""
        # X, Y = loadata(name='WHLK')
        # print('Initial image shape: ' + str(X.shape))
        # # X, Y = createImageCubes(X, Y, window=5)
        # # print('Processed dataset shape: ' + str(X.shape))

        # dataset = Dataset(train_x=X, train_y=Y, ind=False, name='WHLK')
        # selector = SelectBands(dataset=dataset,  method='GSS', nbands=20)
        # VIF_best, IBRA_best, GSS_best, stats_best = selector.run_selection(init_vf=11, final_vf=11, preselected_bands=X.shape[2], config = self.config_)
        
        # torch.backends.cudnn.benchmark = True
        print('Start testing candidate selection: ' + str(self.selection))
        avg = 0
        SEED = [1, 2, 3, 4, 5]
        # torch.manual_seed(SEED)
        # torch.cuda.manual_seed(SEED)
        for i in range(len(SEED)):
            res = train.run(config_path='freenet.whlk.freenet_1_0_whlk_test',
                model_dir='./log/freenet_1_0_whlk_test/freenet/1.0_poly',
                cpu_mode=False,
                after_construct_launcher_callbacks=[register_evaluate_fn],
                opts=None,
                np_seed=SEED[i],
                selection=self.selection) # freenet_1_0_salinas_test
            avg = avg + res['launcher'].avg
            # if res['launcher'].avg < 0.8:
            #     break
        # print('ok')
        return avg / 5.0

    def top_down(self, top, lateral):
        top2x = F.interpolate(top, scale_factor=2.0, mode='nearest')
        return lateral + top2x # 
    
    def forward(self, x, y=None, w=None, **kwargs):
        x_hsi = x
        
        self.time_ += 1

        if self.time_ == 1:
            self.X_full = x.permute(0, 2, 3, 1).squeeze(0)
            self.Y_full = y.squeeze(0)
            # self.X_full = torch.load('/mnt/sdd/niuyuanzhuo/FreeNet-master/SV_X_full.pth').cuda()
            # self.Y_full = torch.load('/mnt/sdd/niuyuanzhuo/FreeNet-master/SV_Y_full.pth').cuda()
            # self.w_full = torch.load('/mnt/c/Users/17735/OneDrive/桌面/FreeNet-master/full_w.pth').cuda()
            trainx_hsi, _ = createImageCubes(self.X_full.cpu(), self.Y_full.cpu())
            trainx_hsi = trainx_hsi.reshape(trainx_hsi.shape[0], 1, trainx_hsi.shape[3], trainx_hsi.shape[1], trainx_hsi.shape[2])
            trainx_hsi, _, _ = normalize(trainx_hsi)
            self.indexes = [i for i in range(x.shape[1])]
            entropies =  [entropy(trainx_hsi[:, :, i, :, :]) for i in range(len(self.indexes))]

            # Sort the pre-selected bands according to their entropy (in decreasing order)
            self.preselected = self.indexes.copy()
            pairs = list(tuple(zip(self.preselected, entropies)))
            pairs.sort(key=lambda x: x[1], reverse=True)
            self.preselected, _ = zip(*pairs)

            # Select the first "select" bands
            self.preselected = list(self.preselected)
            self.selection = self.preselected[:self.config.hidden_channels]
            self.selection.sort()
            self.preselected = self.preselected[self.config.hidden_channels:]
            # self.selection = [0, 74, 75, 98, 101, 109, 114, 116, 125, 128, 131, 132, 136, 241, 242, 243, 244, 247, 248, 249]
            ct = 1
            print("\tAnalyzing candidate combination " + str(ct) + ". 5x2 CV using bands: " + str(self.selection))
            self.f1base = self.tune_DFS()
            print("\tMean F1: " + str(self.f1base))
            
            self.bestselection = self.selection.copy()
            ct += 1
        # # if self.training:
        # #     # print('Current F1:' + str(self.tune(self.x_full, self.mask_full, self.w_full)))
        # #     if self.tune_DFS() < 0.6:
        # #         self.turn = True
        # #     else:
        # #         self.turn = False
    
        # # self.selection = self.tune_DFS()
        # x = x_hsi[:, self.selection, :, :]
        
        if self.training and self.num > 0: #  and self.turn and self.num > 0 and len(self.preselected) > 0
            self.num = self.num - 1
            
            # Try new bands until there is no more elements in the list
            while len(self.preselected) > 0:
                if self.f1base > 0.89 or self.changed_f1base > 0:
                    break
                # Calculate the maximum VIF of all the band in "selection"

                VIF = self.checkMulticollinearity(s=self.selection, trainx=trainx_hsi) # (1, 94890, 1, 1, 20)
                # Remove the band with the highest VIF of "selection"
                self.selection.remove(self.selection[VIF.index(max(VIF))])
                # Pop the next available band from "preselected"
                self.selection.append(self.preselected[0])
                self.selection.sort()
                self.preselected = self.preselected[1:]

                # Train using the bands in "selection"
                print("\tAnalyzing candidate combination " + str(ct) + ". 5x2 CV using bands: " + str(self.selection))
                self.f1 = self.tune_DFS()
                # with open("/mnt/sdd/niuyuanzhuo/FreeNet-master/log1_LK.txt", encoding="utf-8",mode="a") as file:  
                #     file.write(str(self.selection) + '  ' + str(self.f1) +'\n')
                
                print("\tMean F1: " + str(self.f1))
                # Check if the new selection has better performance than the previous one. If not, break
                if self.f1 > self.f1base:
                    self.bestselection = self.selection.copy()
                    self.f1base = self.f1
                    self.changed_f1base = 10

                elif self.f1base > 0.85 or self.changed_f1base > 0:
                    break
                else:
                    self.changed_f1base = self.changed_f1base - 1
                ct += 1
                print("\tBest selection so far: " + str(self.bestselection) + "with an F1 score of " + str(self.f1base))
        
        # # if not self.turn:
        # # x = x_hsi[:, [0, 74, 75, 98, 101, 109, 114, 116, 125, 128, 131, 132, 136, 241, 242, 243, 244, 247, 248, 249], :, :]
        # print("\tBest selection so far: " + str(self.bestselection) + "with an F1 score of " + str(self.f1base))
        

        if self.turn:

            # self.srf.weight_.requires_grad  = False

            # self.reconstruction4.weight.requires_grad = False
            # self.reconstruction4.bias.requires_grad = False

            # MST++
            for name, param in self.reconstruction1.named_parameters():
                if name == 'conv_out.weight' or name == 'conv_in.weight':
                    continue
                else:
                    param.requires_grad = False

        # self.time_ += 1
        x_hsi = x
        b_, c_, h_, w_ = x_hsi.shape
        x = x.permute(0,2,3,1)
        x = self.srf(x) 
        x = x.permute(0,3,1,2)
        # x_middle = x
        # for i in range(self.config.hidden_channels):
        # sv: [23, 24, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 55, 63, 64, 66, 77, 78, 94]
        # wh: [0, 74, 75, 98, 101, 109, 114, 116, 125, 128, 131, 132, 136, 241, 242, 243, 244, 247, 248, 249]
        # entropies = [entropy(x_hsi[:, i, :, :]) for i in range(len(self.indexes))]

        # # Sort the pre-selected bands according to their entropy (in decreasing order)
        # preselected = self.indexes.copy()
        # pairs = list(tuple(zip(preselected, entropies)))
        # pairs.sort(key=lambda x: x[1], reverse=True)
        # preselected, _ = zip(*pairs)

        # # Select the first "select" bands
        # preselected = list(preselected)
        # selection = preselected[:self.config.hidden_channels]
        # selection.sort()
        # preselected = preselected[self.config.hidden_channels:]
        
        x_hsi_bs = x_hsi[:, self.bestselection,:,:]
        # x_hsi_bs = x_hsi[:, 0:self.config.in_channels - 10:int((self.config.in_channels - 10)/self.config.hidden_channels),:,:]
        # x_hsi_bs = torch.tensor(self.BS.fit_transform(x_hsi.reshape(x_hsi.shape[0],\
        #                                     x_hsi.shape[1], x_hsi.shape[2] \
        #                                     * x_hsi.shape[3]).squeeze(0).T\
        #                                     .cpu().detach()).T\
        #                                     .reshape(b_, self.config.hidden_channels, h_, w_))\
        #                                     .cuda()
        # x = x.reshape(self.batch_size * self.batch_size, x.shape[1], int(x.shape[2]/self.batch_size), int(x.shape[3]/self.batch_size))

        x = self.reconstruction1(x) # / 2.0
        # x = x.reshape(b_, x.shape[1], int(x.shape[2] * self.batch_size), int(x.shape[3] * self.batch_size))
        x_rec = x

        feat_list = []
        for op in self.feature_ops:
            x = op(x)
            if isinstance(op, nn.Identity):
                feat_list.append(x)

        inner_feat_list = [self.reduce_1x1convs[i](feat) for i, feat in enumerate(feat_list)]
        inner_feat_list.reverse()

        out_feat_list = [self.fuse_3x3convs[0](inner_feat_list[0])]
        for i in range(len(feat_list) - 1):
            inner = self.top_down(out_feat_list[i], inner_feat_list[i+1])
            out = self.fuse_3x3convs[i + 1](inner)
            out_feat_list.append(out)

        final_feat = out_feat_list[-1]

        logit = self.cls_pred_conv(final_feat)

        if self.time_ == 1998 and self.turn:
            print('ok')
            loss_show = np.vstack((np.array(l1_loss_l),np.array(cls_loss_l)))
            # loss_show = np.array(l1_loss_l)
            plt.axhline(y=0.16, color='r', linestyle='--')
            plt.plot(np.arange(self.start_cls, 999, 1), loss_show.T)
            plt.savefig('loss.jpg')
            plt.show()

        # if self.time_== 1996:
        #     print('ok')
            
        #     plt.plot(psnr)
        #     plt.savefig('psnr.jpg')
        
        if self.training:
            # l1_loss = l1loss1(x_hsi_bs, x_rec)
            # cls_loss = self.loss(logit, y, w)
           
            # l1_loss_l.append(l1_loss.item())
            # cls_loss_l.append(cls_loss.item())
            # self.time_> 1200 or 
            if (l1loss1(x_rec, x_hsi_bs) < 0.019 or self.turn): # self.time_> 1600 or  0.019 for sv 0.16 for whlk5 0.26 for whlk3 in train3 0.029/0.03 for whhc
                if not self.turn:
                    # print('ok')
                    self.start_cls = int(self.time_/2)
                l1_loss = l1loss1(x_hsi_bs, x_rec)
                cls_loss = self.loss(logit, y, w)

                l1_loss_l.append(l1_loss.item())
                cls_loss_l.append(cls_loss.item())
                # if not self.turn:
                #     rec = x_rec.detach().cpu().numpy()
                
                #     result = {"WHU_Hi_LongKou_train": rec}
                
                #     scipy.io.savemat("WHU_Hi_LongKou_train.mat", result)
                
                loss_dict = {
                    'cls_loss': cls_loss / 5.0 ,
                    # 'l1_loss': l1_loss/ 10.0,
                    }
                if l1_loss > 0.019:
                    loss_dict = {
                    'cls_loss': cls_loss/ 5.0,
                    'l1_loss': l1_loss/ 10.0,
                    }
                if cls_loss < l1_loss and self.num_to_stop > 0:
                    # loss_dict = {
                    # 'cls_loss': cls_loss/ 5.0,
                    # # 'l1_loss': l1_loss/ 10.0,
                    # }
                    # print('ok')
                    self.num_to_stop = self.num_to_stop - 1
                    # loss_show = np.vstack((np.array(l1_loss_l),np.array(cls_loss_l)))
                    # # loss_show = np.array(l1_loss_l)
                    # plt.axhline(y=0.16, color='r', linestyle='--')
                    # plt.plot(loss_show.T)
                    # plt.savefig('loss.jpg')
                    # plt.show()
                    sys.exit()
                self.turn = True
                # self.pre_rec_x = x_rec
                # self.pre_x_middle = x_middle

                # MST++
                for name, param in self.reconstruction1.named_parameters():
                    if name == 'conv_out.weight' or name == 'conv_in.weight':
                        continue
                    else:
                        param.requires_grad = False

                # for name, param in self.reconstruction4.named_parameters():
                    
                #     param.requires_grad = False

                # self.reconstruction4.weight.requires_grad = False
                # self.reconstruction4.bias.requires_grad = False

                
            else:
                        
                loss_dict = {
                    'l1_loss': l1loss1(x_rec, x_hsi_bs),# + self.loss(logit, y, w) , # + abs(20 - PSNR(x_hsi, x_rec)), # 
                # 'cls_loss': self.loss(logit, y, w)
                }
                
            
            return loss_dict
        
        
          

        if not self.training:
            # psnr.append(PSNR(x_hsi_bs, x_rec))
            print('PSNR:' + str(PSNR(x_hsi_bs, x_rec)))

        return torch.softmax(logit, dim=1)
    
    def show_avg_var(self, image, mask, x_rec, title):
        srf = np.zeros([len(np.unique(mask).tolist()) - 1, x_rec.shape[1]])
        srf_var = np.zeros([len(np.unique(mask).tolist()) - 1, x_rec.shape[1]])

        image_by_class = []

                    # 每个类的位置
        for b in range(len(np.unique(mask).tolist()) - 1):
            image_by_class.append(np.where(mask[:,:]==b+1)[0])
            image_by_class.append(np.where(mask[:,:]==b+1)[1])
                        # print('ok')

        for j in range(len(np.unique(mask).tolist()) - 1):
            srf_one_class = np.zeros([len(image_by_class[2*j]), x_rec.shape[1]])
            for k in range(len(image_by_class[2*j])):
                srf_one_class[k] = image[image_by_class[2*j][k], image_by_class[2*j + 1][k]]
            srf[j] = srf_one_class.mean(axis=0)
            srf_var[j] = srf_one_class.var(axis=0)

                    # plt.plot(srf.T)
                    # plt.show()
                    # plt.xticks()
        plt.title('var', fontsize=24)
        
                    # plt.axvline(700)
                    # plt.plot(srf_var.T)
                    # plt.savefig('var.png')
        plt.subplot(1, 2, 1)
        plt.plot( srf.T)
        # plt.plot(srf.T)
        plt.subplot(1, 2, 2)
        plt.plot( srf_var.T)
        # ax = plt.gca()
        # ax.yaxis.get_offset_text().set(size=20)  # 左上角
        plt.yticks(size = 20)
        plt.xticks(size = 20)
        plt.savefig(str(title) + '.png')
                    # plt.show()

    def loss(self, x, y, weight):
        losses = F.cross_entropy(x, y.long() - 1, weight=None,
                                 ignore_index=-1, reduction='none')

        v = losses.mul_(weight).sum() / weight.sum()
        return v

    def set_defalut_config(self):
        self.config.update(dict(
            in_channels=204,
            num_classes=16,
            block_channels=(96, 128, 192, 256),
            num_blocks=(1, 1, 1, 1),
            inner_dim=128,
            reduction_ratio=1.0,
        ))

@registry.MODEL.register('FreeNetFewDynamicChannelsVoltage')
class FreeNetFewDynamicChannelsVoltage(CVModule):
    def __init__(self, config):
        super(FreeNetFewDynamicChannelsVoltage, self).__init__(config)
        r = int(16 * self.config.reduction_ratio)
        block1_channels = int(self.config.block_channels[0] * self.config.reduction_ratio / r) * r
        block2_channels = int(self.config.block_channels[1] * self.config.reduction_ratio / r) * r
        block3_channels = int(self.config.block_channels[2] * self.config.reduction_ratio / r) * r
        block4_channels = int(self.config.block_channels[3] * self.config.reduction_ratio / r) * r
        # print(self.config.in_channels)
        self.feature_ops = nn.ModuleList([
            conv3x3_gn_relu(self.config.hidden_channels, block1_channels, r),

            repeat_block(block1_channels, r, self.config.num_blocks[0]),
            nn.Identity(),
            downsample2x(block1_channels, block2_channels),

            repeat_block(block2_channels, r, self.config.num_blocks[1]),
            nn.Identity(),
            downsample2x(block2_channels, block3_channels),

            repeat_block(block3_channels, r, self.config.num_blocks[2]),
            nn.Identity(),
            downsample2x(block3_channels, block4_channels),

            repeat_block(block4_channels, r, self.config.num_blocks[3]),
            nn.Identity(),
        ])
        inner_dim = int(self.config.inner_dim * self.config.reduction_ratio)
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
        # self.batch_size = self.config.rec_bs
        self.in_dim = self.config.in_channels
        self.out_dim = self.config.out_channels
        self.resrf = nn.Parameter(torch.rand((self.out_dim, self.in_dim)))
        self.cls_pred_conv = nn.Conv2d(inner_dim, self.config.num_classes, 1)
        
        self.srf = LyotFilter(self.config.in_channels, self.config.out_channels,  self.config.minBand, self.config.maxBand, self.config.nBandDataset, self.config.dataset)
        # self.resrf = reLyotFilter(self.config.in_channels, self.config.out_channels,  self.config.minBand, self.config.maxBand, self.config.nBandDataset, self.config.dataset)
        self.dynamic1 = S3KAIResNet(self.config.out_channels, self.config.num_classes, 2)# nn.Conv2d(self.config.out_channels, self.config.out_channels * self.config.out_channels, 1, 1, 0)#
        self.dynamic2 = nn.Conv2d(self.config.out_channels, self.config.in_channels, 1, 1, 0)#  S3KAIResNet(self.config.out_channels, self.config.num_classes, 2)# 
        self.dynamic3 = RSSAN5(self.config.num_classes, self.config.out_channels, 3, self.config.out_channels, 1, 1, 1, self.config.hidden_channels)
        self.dynamic4 = SEBlock(self.config.out_channels, 2)

        self.reconstruction1 = MST_Plus_Plus(self.config.out_channels, self.config.hidden_channels,  self.config.hidden_channels, 3)
        self.reconstruction2 = BiSRNet(self.config.out_channels, self.config.hidden_channels, self.config.hidden_channels)
        # self.reconstruction3 = nn.Conv3d(
        #             in_channels=self.config.out_channels,
        #             out_channels=self.config.hidden_channels,
        #             kernel_size=(3, 1, 1),
        #             stride=(1, 1, 1),
        #             padding=(1, 0, 0))
        self.reconstruction4 = nn.Conv2d(self.config.out_channels, self.config.hidden_channels, 3, 1, 1)#  S3KAIResNet(self.config.out_channels, self.config.num_classes, 2)# 

        self.dropout = nn.Dropout(p=0.3, inplace=False)
        self.herosnet = HerosNet(Ch=self.config.out_channels, stages=8)
        self.GST = GST_MODEL(in_ch=self.config.out_channels,
                           out_ch=self.config.out_channels,
                           noise_mean=0.1,
                           noise_std=0.1,
                           init="normal",
                        #    conv = default_conv,
                           noise_act=nn.Softplus(),
                           inter_channels=10,
                           spatial_scale=4)
        self.SSAN = SSAN(self.config.out_channels, self.config.num_classes)
        self.SSRN = SSRN(self.config.out_channels, self.config.num_classes)
        self.time_= 0
        self.BS = PCA(n_components=self.config.hidden_channels)
        self.one_step = nn.Conv2d(self.config.hidden_channels, self.config.num_classes, 1)
        self.freeze_rec = False
        self.reach5 = False
        self.reach1 = False
        self.turn = False
        self.pre_rec_x = None
        self.pre_x_middle = None

    def top_down(self, top, lateral):
        top2x = F.interpolate(top, scale_factor=2.0, mode='nearest')
        return lateral + top2x # 
    
    def forward(self, x, y=None, w=None, **kwargs):

        # if self.turn:

        #     self.srf.weight_.requires_grad  = False

        #     self.reconstruction4.weight.requires_grad = False
        #     self.reconstruction4.bias.requires_grad = False

        #     # MST++

        #     # MST++
        #     for param in self.reconstruction1.parameters():
        #         param.requires_grad = False


        self.time_+= 1
        x_hsi = x
        b_, c_, h_, w_ = x_hsi.shape
        x = x.permute(0,2,3,1)
        x = self.srf(x) 
        x = x.permute(0,3,1,2)
        x_middle = x
        # for i in range(self.config.hidden_channels):
        x_hsi_bs = x_hsi[:, 0:self.config.in_channels - 4:int((self.config.in_channels - 4)/self.config.hidden_channels),:,:]
        # x_hsi_bs = torch.tensor(self.BS.fit_transform(x_hsi.reshape(x_hsi.shape[0],\
        #                                     x_hsi.shape[1], x_hsi.shape[2] \
        #                                     * x_hsi.shape[3]).squeeze(0).T\
        #                                     .cpu().detach()).T\
        #                                     .reshape(b_, self.config.hidden_channels, h_, w_))\
        #                                     .cuda()
        # x = x.reshape(self.batch_size * self.batch_size, x.shape[1], int(x.shape[2]/self.batch_size), int(x.shape[3]/self.batch_size))
        # if self.freeze_rec :
        #     print('ok')
        x = self.reconstruction1(x) # / 2.0
        # x = x.reshape(b_, x.shape[1], int(x.shape[2] * self.batch_size), int(x.shape[3] * self.batch_size))
        x_rec = x
        # if self.turn and self.training:
        #     assert (self.pre_x_middle == x_middle).all()
        #     assert (self.pre_rec_x == x_rec).all()

        # if y is not None:
        #     image = x_rec.squeeze(0).reshape(-1, mask.shape[0], mask.shape[1]).permute(1, 2, 0).detach().cpu()
        # F.relu(weight.transpose(0,1))/weight.transpose(0,1).sum(0)
        # x = torch.matmul(x, F.relu(self.resrf)/self.resrf.sum(0))#.reshape(x.shape[1], x.shape[1], self.dynamic2(x).shape[2], self.dynamic2(x).shape[3])
        # identity = torch.matmul(srf, self.resrf)
        # x = x.permute(0,3,1,2)
        #  = x

        # x_rec = normalization_for_RSNR(x_hsi, x_rec)
        # x = x_rec
        # x_rec = normalization_for_RSNR(x_rec)
        # x = self.herosnet(x)
        # x = x + torch.matmul(x.permute(2,3,0,1), dynamic.permute(2,3,0,1)).permute(2,3,0,1)
        # x = x # torch.Size([1, 5, 288, 400])
        # x_residual = x
        # if 10 * l1loss(x_hsi_bs, x_rec) < 3:
        #     logit = self.one_step(x)
        #     self.freeze_rec = True
        # else:
        # x = x_hsi_bs
        feat_list = []
        for op in self.feature_ops:
            x = op(x)
            if isinstance(op, nn.Identity):
                feat_list.append(x)

        inner_feat_list = [self.reduce_1x1convs[i](feat) for i, feat in enumerate(feat_list)]
        inner_feat_list.reverse()

        out_feat_list = [self.fuse_3x3convs[0](inner_feat_list[0])]
        for i in range(len(feat_list) - 1):
            inner = self.top_down(out_feat_list[i], inner_feat_list[i+1])
            out = self.fuse_3x3convs[i + 1](inner)
            out_feat_list.append(out)

        final_feat = out_feat_list[-1]

        logit = self.cls_pred_conv(final_feat)

        

        # if self.time_== 1999 :
        #     print('ok')
        #     plt.plot(mrae)
        #     plt.savefig('mrae.jpg')
            # plt.show()
        # ssim_score = SSIM(x_hsi, x_rec)
        # 
        
            # loss_dict = {
            #     'l1_loss': l1loss1(x_hsi_bs, x_rec),# + self.loss(logit, y, w) , # + abs(20 - PSNR(x_hsi, x_rec)), # 
            #     'cls_loss': self.loss(logit, y, w) 
            #     }
        #     # print('l1_loss:' + str(l1loss(x_hsi, x_rec)) )
            
        #     # if 10 * l1loss(x_hsi_bs, x_rec) < 5 and not self.reach5:
        #     #     self.show_avg_var(image, mask, x_rec, 5)
        #     #     self.reach5 = True
        #     # loss_dict = {
        #     #     'l1_loss': l1loss(x_hsi_bs, x_rec),# + self.loss(logit, y, w) , # + abs(20 - PSNR(x_hsi, x_rec)), # 
        #     #     'cls_loss': self.loss(logit, y, w)
        #     #     } 
        if self.time_== 1998:
            print('ok')
            # plt.subplot(1,2,1)
            loss_show = np.vstack((np.array(l1_loss_l),np.array(cls_loss_l)))
            plt.plot(loss_show.T)
            # plt.subplot(1,2,2)
            # plt.plot()
            plt.savefig('loss.jpg')
        if self.time_== 1996:
            print('ok')
            # plt.subplot(1,2,1)
            # loss_show = np.vstack((np.array(l1_loss_l),np.array(cls_loss_l)))
            plt.plot(psnr)
            # plt.subplot(1,2,2)
            # plt.plot()
            plt.savefig('psnr.jpg')
        
        if self.training:
            l1_loss = l1loss1(x_hsi_bs, x_rec)
            cls_loss = self.loss(logit, y, w)
            # if self.turn:
            if l1_loss < cls_loss - 0.1:
                loss_dict = {
                    'l1_loss': l1_loss ,# + self.loss(logit, y, w) , # + abs(20 - PSNR(x_hsi, x_rec)), # 
                    'cls_loss': cls_loss
                    }
            else:
                loss_dict = {
                    # 'l1_loss': l1_loss ,# + self.loss(logit, y, w) , # + abs(20 - PSNR(x_hsi, x_rec)), # 
                    'cls_loss': cls_loss
                    }
            l1_loss_l.append(l1_loss.item())
            cls_loss_l.append(cls_loss.item())
            # else:
            #     loss_dict = {
            #         'l1_loss': l1_loss / 5.0 ,# + self.loss(logit, y, w) , # + abs(20 - PSNR(x_hsi, x_rec)), # 
            #         'cls_loss': cls_loss
            #         }
                
            # else:
            #     if l1loss1(x_rec, x_hsi_bs) < 0.017:
            #         self.turn = True
            #         self.pre_rec_x = x_rec
            #         self.pre_x_middle = x_middle
            #         # for param in self.srf.parameters():
            #         #     param.requires_grad = False
            #         # 

            #         self.reconstruction4.weight.requires_grad = False
            #         self.reconstruction4.bias.requires_grad = False

            #         # # MST++
            #         for param in self.reconstruction1.parameters():
            #             param.requires_grad = False
                    
            #         self.srf.weight_.requires_grad  = False
                        
                # loss_dict = {
                #     'l1_loss': l1loss1(x_rec, x_hsi_bs),# + self.loss(logit, y, w) , # + abs(20 - PSNR(x_hsi, x_rec)), # 
                # # 'cls_loss': self.loss(logit, y, w)
                #     }
                
            
            mrae.append(l1loss1(x_rec, x_hsi_bs))
            return loss_dict
        
        
            # plt.show()

        if not self.training:
            psnr.append(PSNR(x_hsi_bs, x_rec))
            print('PSNR:' + str(PSNR(x_hsi_bs, x_rec)))

        return torch.softmax(logit, dim=1)
    
    def show_avg_var(self, image, mask, x_rec, title):
        srf = np.zeros([len(np.unique(mask).tolist()) - 1, x_rec.shape[1]])
        srf_var = np.zeros([len(np.unique(mask).tolist()) - 1, x_rec.shape[1]])

        image_by_class = []

                    # 每个类的位置
        for b in range(len(np.unique(mask).tolist()) - 1):
            image_by_class.append(np.where(mask[:,:]==b+1)[0])
            image_by_class.append(np.where(mask[:,:]==b+1)[1])
                        # print('ok')

        for j in range(len(np.unique(mask).tolist()) - 1):
            srf_one_class = np.zeros([len(image_by_class[2*j]), x_rec.shape[1]])
            for k in range(len(image_by_class[2*j])):
                srf_one_class[k] = image[image_by_class[2*j][k], image_by_class[2*j + 1][k]]
            srf[j] = srf_one_class.mean(axis=0)
            srf_var[j] = srf_one_class.var(axis=0)

                    # plt.plot(srf.T)
                    # plt.show()
                    # plt.xticks()
        plt.title('var', fontsize=24)
        
                    # plt.axvline(700)
                    # plt.plot(srf_var.T)
                    # plt.savefig('var.png')
        plt.subplot(1, 2, 1)
        plt.plot( srf.T)
        # plt.plot(srf.T)
        plt.subplot(1, 2, 2)
        plt.plot( srf_var.T)
        # ax = plt.gca()
        # ax.yaxis.get_offset_text().set(size=20)  # 左上角
        plt.yticks(size = 20)
        plt.xticks(size = 20)
        plt.savefig(str(title) + '.png')
                    # plt.show()

    def loss(self, x, y, weight):
        losses = F.cross_entropy(x, y.long() - 1, weight=None,
                                 ignore_index=-1, reduction='none')

        v = losses.mul_(weight).sum() / weight.sum()
        return v

    def set_defalut_config(self):
        self.config.update(dict(
            in_channels=204,
            num_classes=16,
            block_channels=(96, 128, 192, 256),
            num_blocks=(1, 1, 1, 1),
            inner_dim=128,
            reduction_ratio=1.0,
        ))
        

@registry.MODEL.register('FreeNetReconstructAfterEncoderFewChannelsVoltage')
class FreeNetReconstructAfterEncoderFewChannelsVoltage(CVModule):
    def __init__(self, config):
        super(FreeNetReconstructAfterEncoderFewChannelsVoltage, self).__init__(config)
        r = int(16 * self.config.reduction_ratio)
        block1_channels = int(self.config.block_channels[0] * self.config.reduction_ratio / r) * r # 96
        block2_channels = int(self.config.block_channels[1] * self.config.reduction_ratio / r) * r # 128
        block3_channels = int(self.config.block_channels[2] * self.config.reduction_ratio / r) * r # 192
        block4_channels = int(self.config.block_channels[3] * self.config.reduction_ratio / r) * r # 256
        # print(self.config.in_channels)
        self.feature_ops = nn.ModuleList([
            conv3x3_gn_relu(self.config.out_channels, block1_channels, r),

            repeat_block(block1_channels, r, self.config.num_blocks[0]),
            nn.Identity(),
            downsample2x(block1_channels, block2_channels),

            repeat_block(block2_channels, r, self.config.num_blocks[1]),
            nn.Identity(),
            downsample2x(block2_channels, block3_channels),

            repeat_block(block3_channels, r, self.config.num_blocks[2]),
            nn.Identity(),
            downsample2x(block3_channels, block4_channels),

            repeat_block(block4_channels, r, self.config.num_blocks[3]),
            nn.Identity(),
        ])
        inner_dim = int(self.config.inner_dim * self.config.reduction_ratio) # 128
        self.reduce_1x1convs = nn.ModuleList([
            nn.Conv2d(block4_channels, inner_dim, 1),
        ])
        self.fuse_3x3convs = nn.ModuleList([
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
        ])
        self.fuse_3x3convs_reconstruct = nn.ModuleList([
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
        ])
        self.cls_pred_conv = nn.Conv2d(inner_dim, self.config.num_classes, 1)
        self.reconstruct = nn.Conv2d(inner_dim, self.config.in_channels, 1)
        self.srf = LyotVisualFilter(self.config.in_channels, self.config.out_channels,  self.config.minBand, self.config.maxBand, self.config.nBandDataset, self.config.dataset)
        # self.reconstruct = nn.Conv2d(self.config.out_channels, self.config.in_channels, 3,1,1)

    def top_down(self, top, lateral): # residual
        top2x = F.interpolate(top, scale_factor=2.0, mode='nearest')
        return lateral + top2x # 

    def forward(self, x, y=None, w=None, **kwargs):
        # print(x.shape) # torch.Size([1, 103, 624, 352])
        
        # torch.save(x.to(torch.device('cpu')), "myTensor.pth")
        x_target = x # ([1, 103, 624, 352])
        x = x.permute(0,2,3,1)
        x = self.srf(x) 
        x = x.permute(0,3,1,2) # ([1, 5, 624, 352])
        # x_reconstruct = self.reconstruct(x)
        feat_list = []
        for op in self.feature_ops:
            x = op(x)
            if isinstance(op, nn.Identity):
                feat_list.append(x) # torch.Size([1, 96, 624, 352]) 
                                    # torch.Size([1, 128, 312, 176]) 
                                    # torch.Size([1, 192, 156, 88]) 
                                    # torch.Size([1, 256, 78, 44])

        inner_feat_list = [self.reduce_1x1convs[i](feat) for i, feat in enumerate(feat_list)]

        # feat_list_reconstruct = feat_list[-1]

        inner_feat_list.reverse()

        out_feat_list = [self.fuse_3x3convs[0](inner_feat_list[0])] # ([1, 128, 78, 44])
        for i in range(len(feat_list) - 1):
            inner = self.top_down(out_feat_list[i], inner_feat_list[i+1])
            out = self.fuse_3x3convs[i + 1](inner)
            out_feat_list.append(out)

        final_feat = out_feat_list[-1] # ([1, 128, 624, 352])

        out_feat_list_reconstruct = [self.fuse_3x3convs_reconstruct[0](inner_feat_list[0])]

        for i in range(len(feat_list) - 1):
            inner = self.top_down(out_feat_list_reconstruct[i], 0)
            out = self.fuse_3x3convs_reconstruct[i + 1](inner)
            out_feat_list_reconstruct.append(out)

        final_feat_reconstruct = out_feat_list_reconstruct[-1]

        x_reconstruct = self.reconstruct(final_feat_reconstruct)

        logit = self.cls_pred_conv(final_feat)
        if self.training:
            loss_dict = {
                'cls_loss': self.loss(logit, y, w) + PSNR(x_reconstruct, x_target)
            }
            return loss_dict

        return torch.softmax(logit, dim=1)

    def loss(self, x, y, weight):
        losses = F.cross_entropy(x, y.long() - 1, weight=None,
                                 ignore_index=-1, reduction='none')

        v = losses.mul_(weight).sum() / weight.sum()
        return v

    def set_defalut_config(self):
        self.config.update(dict(
            in_channels=204,
            num_classes=16,
            block_channels=(96, 128, 192, 256),
            num_blocks=(1, 1, 1, 1),
            inner_dim=128,
            reduction_ratio=1.0,
        ))




@registry.MODEL.register('FreeNetNoResidual')
class FreeNetNoResidual(CVModule):
    def __init__(self, config):
        super(FreeNetNoResidual, self).__init__(config)
        r = int(16 * self.config.reduction_ratio)
        block1_channels = int(self.config.block_channels[0] * self.config.reduction_ratio / r) * r
        block2_channels = int(self.config.block_channels[1] * self.config.reduction_ratio / r) * r
        block3_channels = int(self.config.block_channels[2] * self.config.reduction_ratio / r) * r
        block4_channels = int(self.config.block_channels[3] * self.config.reduction_ratio / r) * r
        # print(self.config.in_channels)
        self.feature_ops = nn.ModuleList([
            conv3x3_gn_relu(self.config.in_channels, block1_channels, r),

            repeat_block(block1_channels, r, self.config.num_blocks[0]),
            nn.Identity(),
            downsample2x(block1_channels, block2_channels),

            repeat_block(block2_channels, r, self.config.num_blocks[1]),
            nn.Identity(),
            downsample2x(block2_channels, block3_channels),

            repeat_block(block3_channels, r, self.config.num_blocks[2]),
            nn.Identity(),
            downsample2x(block3_channels, block4_channels),

            repeat_block(block4_channels, r, self.config.num_blocks[3]),
            nn.Identity(),
        ])
        inner_dim = int(self.config.inner_dim * self.config.reduction_ratio)
        self.reduce_1x1convs = nn.ModuleList([
       #     nn.Conv2d(block1_channels, inner_dim, 1),
        #    nn.Conv2d(block2_channels, inner_dim, 1),
        #    nn.Conv2d(block3_channels, inner_dim, 1),
            nn.Conv2d(block4_channels, inner_dim, 1),
        ])
        self.fuse_3x3convs = nn.ModuleList([
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
        ])
        self.cls_pred_conv = nn.Conv2d(inner_dim, self.config.num_classes, 1)

    def top_down(self, top, lateral):
        top2x = F.interpolate(top, scale_factor=2.0, mode='nearest')
        return top2x # lateral +

    def forward(self, x, y=None, w=None, **kwargs):
        # print('okokokokokokokokokokokokokokokokokok')
        # torch.save(x.to(torch.device('cpu')), "myTensor.pth")
        # if  self.training is False:
        #     print('now')
        
        feat_list = []
        for op in self.feature_ops:
            x = op(x)
            if isinstance(op, nn.Identity):
                feat_list.append(x)

        inner_feat_list = [self.reduce_1x1convs[0](feat_list[-1])] # [#self.reduce_1x1convs[i](feat) for i, feat in enumerate(feat_list)]
        # inner_feat_list.reverse()

        out_feat_list = [self.fuse_3x3convs[0](inner_feat_list[0])]
        for i in range(len(feat_list) - 1):
            inner = self.top_down(out_feat_list[i],0)
            out = self.fuse_3x3convs[i + 1](inner)
            out_feat_list.append(out)

        final_feat = out_feat_list[-1]

        logit = self.cls_pred_conv(final_feat)
        if self.training:
            loss_dict = {
                'cls_loss': self.loss(logit, y, w)
            }
            return loss_dict

        return torch.softmax(logit, dim=1)

    def loss(self, x, y, weight):
        losses = F.cross_entropy(x, y.long() - 1, weight=None,
                                 ignore_index=-1, reduction='none')

        v = losses.mul_(weight).sum() / weight.sum()
        return v

    def set_defalut_config(self):
        self.config.update(dict(
            in_channels=204,
            num_classes=16,
            block_channels=(96, 128, 192, 256),
            num_blocks=(1, 1, 1, 1),
            inner_dim=128,
            reduction_ratio=1.0,
        ))

@registry.MODEL.register('FreeNetNoResidualLinearSRF')
class FreeNetNoResidualLinearSRF(CVModule):
    def __init__(self, config):
        super(FreeNetNoResidualLinearSRF, self).__init__(config)
        r = int(16 * self.config.reduction_ratio)
        block1_channels = int(self.config.block_channels[0] * self.config.reduction_ratio / r) * r
        block2_channels = int(self.config.block_channels[1] * self.config.reduction_ratio / r) * r
        block3_channels = int(self.config.block_channels[2] * self.config.reduction_ratio / r) * r
        block4_channels = int(self.config.block_channels[3] * self.config.reduction_ratio / r) * r
        # print(self.config.in_channels)
        self.feature_ops = nn.ModuleList([
            conv3x3_gn_relu(self.config.out_channels, block1_channels, r),

            repeat_block(block1_channels, r, self.config.num_blocks[0]),
            nn.Identity(),
            downsample2x(block1_channels, block2_channels),

            repeat_block(block2_channels, r, self.config.num_blocks[1]),
            nn.Identity(),
            downsample2x(block2_channels, block3_channels),

            repeat_block(block3_channels, r, self.config.num_blocks[2]),
            nn.Identity(),
            downsample2x(block3_channels, block4_channels),

            repeat_block(block4_channels, r, self.config.num_blocks[3]),
            nn.Identity(),
        ])
        inner_dim = int(self.config.inner_dim * self.config.reduction_ratio)
        self.reduce_1x1convs = nn.ModuleList([
       #     nn.Conv2d(block1_channels, inner_dim, 1),
        #    nn.Conv2d(block2_channels, inner_dim, 1),
        #    nn.Conv2d(block3_channels, inner_dim, 1),
            nn.Conv2d(block4_channels, inner_dim, 1),
        ])
        self.fuse_3x3convs = nn.ModuleList([
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
        ])
        self.cls_pred_conv = nn.Conv2d(inner_dim, self.config.num_classes, 1)
        self.srf = PosLinear(self.config.in_channels, self.config.out_channels,  self.config.minBand, self.config.maxBand, self.config.nBandDataset, self.config.dataset)

    def top_down(self, top, lateral):
        top2x = F.interpolate(top, scale_factor=2.0, mode='nearest')
        return top2x # lateral +

    def forward(self, x, y=None, w=None, **kwargs):
        # print(x.shape) # torch.Size([1, 103, 624, 352])
        x = x.permute(0,2,3,1)
        x = self.srf(x)
        x = x.permute(0,3,1,2)
        feat_list = []
        for op in self.feature_ops:
            x = op(x)
            if isinstance(op, nn.Identity):
                feat_list.append(x)

        inner_feat_list = [self.reduce_1x1convs[0](feat_list[-1])]
        # inner_feat_list.reverse()

        out_feat_list = [self.fuse_3x3convs[0](inner_feat_list[0])]
        for i in range(len(feat_list) - 1):
            inner = self.top_down(out_feat_list[i],0)
            out = self.fuse_3x3convs[i + 1](inner)
            out_feat_list.append(out)

        final_feat = out_feat_list[-1]

        logit = self.cls_pred_conv(final_feat)
        if self.training:
            loss_dict = {
                'cls_loss': self.loss(logit, y, w)
            }
            return loss_dict

        return torch.softmax(logit, dim=1)

    def loss(self, x, y, weight):
        losses = F.cross_entropy(x, y.long() - 1, weight=None,
                                 ignore_index=-1, reduction='none')

        v = losses.mul_(weight).sum() / weight.sum()
        return v

    def set_defalut_config(self):
        self.config.update(dict(
            in_channels=204,
            num_classes=16,
            block_channels=(96, 128, 192, 256),
            num_blocks=(1, 1, 1, 1),
            inner_dim=128,
            reduction_ratio=1.0,
        ))

@registry.MODEL.register('FreeNetNoResidualFewChannelsVoltageNoResidual')
class FreeNetNoResidualFewChannelsVoltageNoResidual(CVModule):
    def __init__(self, config):
        super(FreeNetNoResidualFewChannelsVoltageNoResidual, self).__init__(config)
        r = int(16 * self.config.reduction_ratio)
        block1_channels = int(self.config.block_channels[0] * self.config.reduction_ratio / r) * r
        block2_channels = int(self.config.block_channels[1] * self.config.reduction_ratio / r) * r
        block3_channels = int(self.config.block_channels[2] * self.config.reduction_ratio / r) * r
        block4_channels = int(self.config.block_channels[3] * self.config.reduction_ratio / r) * r
        # print(self.config.in_channels)
        self.feature_ops = nn.ModuleList([
            conv3x3_gn_relu(self.config.out_channels, block1_channels, r),

            repeat_block(block1_channels, r, self.config.num_blocks[0]),
            nn.Identity(),
            downsample2x(block1_channels, block2_channels),

            repeat_block(block2_channels, r, self.config.num_blocks[1]),
            nn.Identity(),
            downsample2x(block2_channels, block3_channels),

            repeat_block(block3_channels, r, self.config.num_blocks[2]),
            nn.Identity(),
            downsample2x(block3_channels, block4_channels),

            repeat_block(block4_channels, r, self.config.num_blocks[3]),
            nn.Identity(),
        ])
        inner_dim = int(self.config.inner_dim * self.config.reduction_ratio)
        self.reduce_1x1convs = nn.ModuleList([
       #     nn.Conv2d(block1_channels, inner_dim, 1),
        #    nn.Conv2d(block2_channels, inner_dim, 1),
        #    nn.Conv2d(block3_channels, inner_dim, 1),
            nn.Conv2d(block4_channels, inner_dim, 1),
        ])
        self.fuse_3x3convs = nn.ModuleList([
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
        ])
        self.cls_pred_conv = nn.Conv2d(inner_dim, self.config.num_classes, 1)
        self.srf = LyotVisualFilter(self.config.in_channels, self.config.out_channels,  self.config.minBand, self.config.maxBand, self.config.nBandDataset, self.config.dataset)

    def top_down(self, top, lateral):
        top2x = F.interpolate(top, scale_factor=2.0, mode='nearest')
        return top2x # lateral +

    def forward(self, x, y=None, w=None, **kwargs):
        # print(x.shape) # torch.Size([1, 103, 624, 352])
        
        # torch.save(x.to(torch.device('cpu')), "myTensor.pth")
        x = x.permute(0,2,3,1)
        x = self.srf(x) 
        x = x.permute(0,3,1,2)
        feat_list = []
        for op in self.feature_ops:
            x = op(x)
            if isinstance(op, nn.Identity):
                feat_list.append(x)

        inner_feat_list = [self.reduce_1x1convs[0](feat_list[-1])]
        # inner_feat_list.reverse()

        out_feat_list = [self.fuse_3x3convs[0](inner_feat_list[0])]
        for i in range(len(feat_list) - 1):
            inner = self.top_down(out_feat_list[i],0)
            out = self.fuse_3x3convs[i + 1](inner)
            out_feat_list.append(out)

        final_feat = out_feat_list[-1]

        logit = self.cls_pred_conv(final_feat)
        if self.training:
            loss_dict = {
                'cls_loss': self.loss(logit, y, w)
            }
            return loss_dict

        return torch.softmax(logit, dim=1)

    def loss(self, x, y, weight):
        losses = F.cross_entropy(x, y.long() - 1, weight=None,
                                 ignore_index=-1, reduction='none')

        v = losses.mul_(weight).sum() / weight.sum()
        return v

    def set_defalut_config(self):
        self.config.update(dict(
            in_channels=204,
            num_classes=16,
            block_channels=(96, 128, 192, 256),
            num_blocks=(1, 1, 1, 1),
            inner_dim=128,
            reduction_ratio=1.0,
        ))

@registry.MODEL.register('FreeNetNoResidualFewDynamicChannelsVoltageNoResidual')
class FreeNetNoResidualFewDynamicChannelsVoltageNoResidual(CVModule):
    def __init__(self, config):
        super(FreeNetNoResidualFewDynamicChannelsVoltageNoResidual, self).__init__(config)
        r = int(16 * self.config.reduction_ratio)
        block1_channels = int(self.config.block_channels[0] * self.config.reduction_ratio / r) * r
        block2_channels = int(self.config.block_channels[1] * self.config.reduction_ratio / r) * r
        block3_channels = int(self.config.block_channels[2] * self.config.reduction_ratio / r) * r
        block4_channels = int(self.config.block_channels[3] * self.config.reduction_ratio / r) * r
        # print(self.config.in_channels)
        self.feature_ops = nn.ModuleList([
            conv3x3_gn_relu(self.config.out_channels, block1_channels, r),

            repeat_block(block1_channels, r, self.config.num_blocks[0]),
            nn.Identity(),
            downsample2x(block1_channels, block2_channels),

            repeat_block(block2_channels, r, self.config.num_blocks[1]),
            nn.Identity(),
            downsample2x(block2_channels, block3_channels),

            repeat_block(block3_channels, r, self.config.num_blocks[2]),
            nn.Identity(),
            downsample2x(block3_channels, block4_channels),

            repeat_block(block4_channels, r, self.config.num_blocks[3]),
            nn.Identity(),
        ])
        inner_dim = int(self.config.inner_dim * self.config.reduction_ratio)
        self.reduce_1x1convs = nn.ModuleList([
       #     nn.Conv2d(block1_channels, inner_dim, 1),
        #    nn.Conv2d(block2_channels, inner_dim, 1),
        #    nn.Conv2d(block3_channels, inner_dim, 1),
            nn.Conv2d(block4_channels, inner_dim, 1),
        ])
        self.fuse_3x3convs = nn.ModuleList([
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
        ])
        self.cls_pred_conv = nn.Conv2d(inner_dim, self.config.num_classes, 1)
        self.srf = LyotVisualFilter(self.config.in_channels, self.config.out_channels,  self.config.minBand, self.config.maxBand, self.config.nBandDataset, self.config.dataset)
        self.dynamic = nn.Conv2d(self.config.out_channels, self.config.out_channels * self.config.out_channels, 1, 1, 0)

    def top_down(self, top, lateral):
        top2x = F.interpolate(top, scale_factor=2.0, mode='nearest')
        return top2x # lateral +

    def forward(self, x, y=None, w=None, **kwargs):
        # print(x.shape) # torch.Size([1, 103, 624, 352])
        
        # torch.save(x.to(torch.device('cpu')), "myTensor.pth")
        print(self.dynamic.weight[0][0][0])
        x = x.permute(0,2,3,1)
        x = self.srf(x) 
        x = x.permute(0,3,1,2)
        dynamic = self.dynamic(x).reshape(x.shape[1], x.shape[1], self.dynamic(x).shape[2], self.dynamic(x).shape[3])
        x = torch.matmul(x.permute(2,3,0,1), dynamic.permute(2,3,0,1))
        x = x.permute(2,3,0,1) # torch.Size([1, 3, 512, 224])
        x_residual = x
        feat_list = []
        for op in self.feature_ops:
            x = op(x)
            if isinstance(op, nn.Identity):
                feat_list.append(x)

        inner_feat_list = [self.reduce_1x1convs[0](feat_list[-1])]
        # inner_feat_list.reverse()

        out_feat_list = [self.fuse_3x3convs[0](inner_feat_list[0])]
        for i in range(len(feat_list) - 1):
            inner = self.top_down(out_feat_list[i],0)
            out = self.fuse_3x3convs[i + 1](inner)
            out_feat_list.append(out)

        final_feat = out_feat_list[-1]

        logit = self.cls_pred_conv(final_feat)
        if self.training:
            loss_dict = {
                'cls_loss': self.loss(logit, y, w)
            }
            return loss_dict

        return torch.softmax(logit, dim=1)

    def loss(self, x, y, weight):
        losses = F.cross_entropy(x, y.long() - 1, weight=None,
                                 ignore_index=-1, reduction='none')

        v = losses.mul_(weight).sum() / weight.sum()
        return v

    def set_defalut_config(self):
        self.config.update(dict(
            in_channels=204,
            num_classes=16,
            block_channels=(96, 128, 192, 256),
            num_blocks=(1, 1, 1, 1),
            inner_dim=128,
            reduction_ratio=1.0,
        ))
        

@registry.MODEL.register('FreeNetNoResidualReconstructAfterEncoderFewChannelsVoltage')
class FreeNetNoResidualReconstructAfterEncoderFewChannelsVoltage(CVModule):
    def __init__(self, config):
        super(FreeNetNoResidualReconstructAfterEncoderFewChannelsVoltage, self).__init__(config)
        r = int(16 * self.config.reduction_ratio)
        block1_channels = int(self.config.block_channels[0] * self.config.reduction_ratio / r) * r # 96
        block2_channels = int(self.config.block_channels[1] * self.config.reduction_ratio / r) * r # 128
        block3_channels = int(self.config.block_channels[2] * self.config.reduction_ratio / r) * r # 192
        block4_channels = int(self.config.block_channels[3] * self.config.reduction_ratio / r) * r # 256
        # print(self.config.in_channels)
        self.feature_ops = nn.ModuleList([
            conv3x3_gn_relu(self.config.out_channels, block1_channels, r),

            repeat_block(block1_channels, r, self.config.num_blocks[0]),
            nn.Identity(),
            downsample2x(block1_channels, block2_channels),

            repeat_block(block2_channels, r, self.config.num_blocks[1]),
            nn.Identity(),
            downsample2x(block2_channels, block3_channels),

            repeat_block(block3_channels, r, self.config.num_blocks[2]),
            nn.Identity(),
            downsample2x(block3_channels, block4_channels),

            repeat_block(block4_channels, r, self.config.num_blocks[3]),
            nn.Identity(),
        ])
        inner_dim = int(self.config.inner_dim * self.config.reduction_ratio) # 128
        self.reduce_1x1convs = nn.ModuleList([
            nn.Conv2d(block4_channels, inner_dim, 1),
        ])
        self.fuse_3x3convs = nn.ModuleList([
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
        ])
        self.fuse_3x3convs_reconstruct = nn.ModuleList([
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
        ])
        self.cls_pred_conv = nn.Conv2d(inner_dim, self.config.num_classes, 1)
        self.reconstruct = nn.Conv2d(inner_dim, self.config.in_channels, 1)
        self.srf = LyotVisualFilter(self.config.in_channels, self.config.out_channels,  self.config.minBand, self.config.maxBand, self.config.nBandDataset, self.config.dataset)
        # self.reconstruct = nn.Conv2d(self.config.out_channels, self.config.in_channels, 3,1,1)

    def top_down(self, top, lateral): # residual
        top2x = F.interpolate(top, scale_factor=2.0, mode='nearest')
        return top2x # lateral +

    def forward(self, x, y=None, w=None, **kwargs):
        # print(x.shape) # torch.Size([1, 103, 624, 352])
        
        # torch.save(x.to(torch.device('cpu')), "myTensor.pth")
        x_target = x # ([1, 103, 624, 352])
        x = x.permute(0,2,3,1)
        x = self.srf(x) 
        x = x.permute(0,3,1,2) # ([1, 5, 624, 352])
        # x_reconstruct = self.reconstruct(x)
        feat_list = []
        for op in self.feature_ops:
            x = op(x)
            if isinstance(op, nn.Identity):
                feat_list.append(x) # torch.Size([1, 96, 624, 352]) 
                                    # torch.Size([1, 128, 312, 176]) 
                                    # torch.Size([1, 192, 156, 88]) 
                                    # torch.Size([1, 256, 78, 44])

        inner_feat_list = [self.reduce_1x1convs[0](feat_list[-1])]

        # feat_list_reconstruct = feat_list[-1]

        # # inner_feat_list.reverse()

        out_feat_list = [self.fuse_3x3convs[0](inner_feat_list[0])] # ([1, 128, 78, 44])
        for i in range(len(feat_list) - 1):
            inner = self.top_down(out_feat_list[i], 0)
            out = self.fuse_3x3convs[i + 1](inner)
            out_feat_list.append(out)

        final_feat = out_feat_list[-1] # ([1, 128, 624, 352])

        out_feat_list_reconstruct = [self.fuse_3x3convs_reconstruct[0](inner_feat_list[0])]

        for i in range(len(feat_list) - 1):
            inner = self.top_down(out_feat_list_reconstruct[i], 0)
            out = self.fuse_3x3convs_reconstruct[i + 1](inner)
            out_feat_list_reconstruct.append(out)

        final_feat_reconstruct = out_feat_list_reconstruct[-1]

        x_reconstruct = self.reconstruct(final_feat_reconstruct)

        logit = self.cls_pred_conv(final_feat)
        if self.training:
            loss_dict = {
                'cls_loss': self.loss(logit, y, w) + PSNR(x_reconstruct, x_target)
            }
            return loss_dict

        return torch.softmax(logit, dim=1)

    def loss(self, x, y, weight):
        losses = F.cross_entropy(x, y.long() - 1, weight=None,
                                 ignore_index=-1, reduction='none')

        v = losses.mul_(weight).sum() / weight.sum()
        return v

    def set_defalut_config(self):
        self.config.update(dict(
            in_channels=204,
            num_classes=16,
            block_channels=(96, 128, 192, 256),
            num_blocks=(1, 1, 1, 1),
            inner_dim=128,
            reduction_ratio=1.0,
        ))



@registry.MODEL.register('FreeNetNoResidualReconstructAfterDecoderFewChannelsVoltage')
class FreeNetNoResidualReconstructAfterDecoderFewChannelsVoltage(CVModule):
    def __init__(self, config):
        super(FreeNetNoResidualReconstructAfterDecoderFewChannelsVoltage, self).__init__(config)
        r = int(16 * self.config.reduction_ratio)
        block1_channels = int(self.config.block_channels[0] * self.config.reduction_ratio / r) * r # 96
        block2_channels = int(self.config.block_channels[1] * self.config.reduction_ratio / r) * r # 128
        block3_channels = int(self.config.block_channels[2] * self.config.reduction_ratio / r) * r # 192
        block4_channels = int(self.config.block_channels[3] * self.config.reduction_ratio / r) * r # 256
        # print(self.config.in_channels)
        self.feature_ops = nn.ModuleList([
            conv3x3_gn_relu(self.config.out_channels, block1_channels, r),

            repeat_block(block1_channels, r, self.config.num_blocks[0]),
            nn.Identity(),
            downsample2x(block1_channels, block2_channels),

            repeat_block(block2_channels, r, self.config.num_blocks[1]),
            nn.Identity(),
            downsample2x(block2_channels, block3_channels),

            repeat_block(block3_channels, r, self.config.num_blocks[2]),
            nn.Identity(),
            downsample2x(block3_channels, block4_channels),

            repeat_block(block4_channels, r, self.config.num_blocks[3]),
            nn.Identity(),
        ])
        inner_dim = int(self.config.inner_dim * self.config.reduction_ratio) # 128
        self.reduce_1x1convs = nn.ModuleList([
            nn.Conv2d(block4_channels, inner_dim, 1),
        ])
        self.fuse_3x3convs = nn.ModuleList([
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
        ])
        self.cls_pred_conv = nn.Conv2d(inner_dim, self.config.num_classes, 1)
        self.reconstruct = nn.Conv2d(inner_dim, self.config.in_channels, 1)
        self.srf = LyotVisualFilter(self.config.in_channels, self.config.out_channels,  self.config.minBand, self.config.maxBand, self.config.nBandDataset, self.config.dataset)
        # self.reconstruct = nn.Conv2d(self.config.out_channels, self.config.in_channels, 3,1,1)

    def top_down(self, top, lateral): # residual
        top2x = F.interpolate(top, scale_factor=2.0, mode='nearest')
        return top2x # lateral +

    def forward(self, x, y=None, w=None, **kwargs):
        # print(x.shape) # torch.Size([1, 103, 624, 352])
        
        # torch.save(x.to(torch.device('cpu')), "myTensor.pth")
        x_target = x # ([1, 103, 624, 352])
        x = x.permute(0,2,3,1)
        x = self.srf(x) 
        x = x.permute(0,3,1,2) # ([1, 5, 624, 352])
        # x_reconstruct = self.reconstruct(x)
        feat_list = []
        for op in self.feature_ops:
            x = op(x)
            if isinstance(op, nn.Identity):
                feat_list.append(x) # torch.Size([1, 96, 624, 352]) 
                                    # torch.Size([1, 128, 312, 176]) 
                                    # torch.Size([1, 192, 156, 88]) 
                                    # torch.Size([1, 256, 78, 44])

        inner_feat_list = [self.reduce_1x1convs[0](feat_list[-1])]

        # feat_list_reconstruct = feat_list[-1]

        # # inner_feat_list.reverse()

        out_feat_list = [self.fuse_3x3convs[0](inner_feat_list[0])] # ([1, 128, 78, 44])
        for i in range(len(feat_list) - 1):
            inner = self.top_down(out_feat_list[i], 0)
            out = self.fuse_3x3convs[i + 1](inner)
            out_feat_list.append(out)

        final_feat = out_feat_list[-1] # ([1, 128, 624, 352])

        x_reconstruct = self.reconstruct(final_feat)

        logit = self.cls_pred_conv(final_feat)
        if self.training:
            loss_dict = {
                'cls_loss': self.loss(logit, y, w) + PSNR(x_reconstruct, x_target)
            }
            return loss_dict

        return torch.softmax(logit, dim=1)

    def loss(self, x, y, weight):
        losses = F.cross_entropy(x, y.long() - 1, weight=None,
                                 ignore_index=-1, reduction='none')

        v = losses.mul_(weight).sum() / weight.sum()
        return v

    def set_defalut_config(self):
        self.config.update(dict(
            in_channels=204,
            num_classes=16,
            block_channels=(96, 128, 192, 256),
            num_blocks=(1, 1, 1, 1),
            inner_dim=128,
            reduction_ratio=1.0,
        ))


@registry.MODEL.register('FreeNetReconstructAfterFirstResidualFewChannelsVoltage')
class FreeNetReconstructAfterFirstResidualFewChannelsVoltage(CVModule):
    def __init__(self, config):
        super(FreeNetReconstructAfterFirstResidualFewChannelsVoltage, self).__init__(config)
        r = int(16 * self.config.reduction_ratio)
        block1_channels = int(self.config.block_channels[0] * self.config.reduction_ratio / r) * r # 96
        block2_channels = int(self.config.block_channels[1] * self.config.reduction_ratio / r) * r # 128
        block3_channels = int(self.config.block_channels[2] * self.config.reduction_ratio / r) * r # 192
        block4_channels = int(self.config.block_channels[3] * self.config.reduction_ratio / r) * r # 256
        # print(self.config.in_channels)
        self.feature_ops = nn.ModuleList([
            conv3x3_gn_relu(self.config.out_channels, block1_channels, r),

            repeat_block(block1_channels, r, self.config.num_blocks[0]),
            nn.Identity(),
            downsample2x(block1_channels, block2_channels),

            repeat_block(block2_channels, r, self.config.num_blocks[1]),
            nn.Identity(),
            downsample2x(block2_channels, block3_channels),

            repeat_block(block3_channels, r, self.config.num_blocks[2]),
            nn.Identity(),
            downsample2x(block3_channels, block4_channels),

            repeat_block(block4_channels, r, self.config.num_blocks[3]),
            nn.Identity(),
        ])
        inner_dim = int(self.config.inner_dim * self.config.reduction_ratio) # 128
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
        self.fuse_3x3convs_reconstruct = nn.ModuleList([
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            # nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            # nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
        ])
        self.cls_pred_conv = nn.Conv2d(inner_dim, self.config.num_classes, 1)
        self.reconstruct = nn.Conv2d(inner_dim, self.config.in_channels, 1)
        self.srf = LyotVisualFilter(self.config.in_channels, self.config.out_channels,  self.config.minBand, self.config.maxBand, self.config.nBandDataset, self.config.dataset)
        # self.reconstruct = nn.Conv2d(self.config.out_channels, self.config.in_channels, 3,1,1)

    def top_down(self, top, lateral): # residual
        top2x = F.interpolate(top, scale_factor=2.0, mode='nearest')
        return top2x + lateral # 

    def forward(self, x, y=None, w=None, **kwargs):
        # print(x.shape) # torch.Size([1, 103, 624, 352])
        
        # torch.save(x.to(torch.device('cpu')), "myTensor.pth")
        train_length = (int(int(x.shape[2]/2)/16) + 1) * 16
        test_length = x.shape[2] - train_length

        x_target_train = x[:,:,:train_length,:] # ([1, 103, 624, 352])
        x_target_test = x[:,:,train_length:,:]

        # classification
        x = x.permute(0,2,3,1)
        x = self.srf(x) 
        x = x.permute(0,3,1,2) # ([1, 5, 624, 352])
        x_source_train_rec = x[:,:,:train_length,:] # ([1, 103, 624, 352])
        x_source_test_rec = x[:,:,train_length:,:]
        # x_reconstruct = self.reconstruct(x)
        feat_list = []
        for op in self.feature_ops:
            x = op(x)
            if isinstance(op, nn.Identity):
                feat_list.append(x) # torch.Size([1, 96, 624, 352]) 
                                    # torch.Size([1, 128, 312, 176]) 
                                    # torch.Size([1, 192, 156, 88]) 
                                    # torch.Size([1, 256, 78, 44])

        inner_feat_list = [self.reduce_1x1convs[i](feat) for i, feat in enumerate(feat_list)]
        inner_feat_list.reverse()

        out_feat_list = [self.fuse_3x3convs[0](inner_feat_list[0])] # ([1, 128, 78, 44])
        for i in range(len(feat_list) - 1):
            inner = self.top_down(out_feat_list[i], inner_feat_list[i+1])
            out = self.fuse_3x3convs[i + 1](inner)
            out_feat_list.append(out)

        final_feat = out_feat_list[-1] # ([1, 128, 624, 352])

        logit = self.cls_pred_conv(final_feat)

        if self.training:

            feat_list_reconstruct_train = []
            for op in self.feature_ops:
                x_source_train_rec = op(x_source_train_rec)
                if isinstance(op, nn.Identity):
                    feat_list_reconstruct_train.append(x_source_train_rec) # torch.Size([1, 96, 624, 352]) 
                    break
                                    # torch.Size([1, 128, 312, 176]) 
                                    # torch.Size([1, 192, 156, 88]) 
                                    # torch.Size([1, 256, 78, 44])

            inner_feat_list_reconstrcut = [self.reduce_1x1convs[i](feat) for i, feat in enumerate(feat_list_reconstruct_train)]
            inner_feat_list.reverse()
            out_feat_list_reconstruct = [self.fuse_3x3convs_reconstruct[0](inner_feat_list_reconstrcut[-1])]

            for i in range(len(self.fuse_3x3convs_reconstruct) - 1):
                out = self.fuse_3x3convs_reconstruct[i + 1](out_feat_list_reconstruct[i])
                out_feat_list_reconstruct.append(out)

            final_feat_reconstruct = out_feat_list_reconstruct[-1]

            x_reconstruct_train_pred = self.reconstruct(final_feat_reconstruct)
            loss_dict = {
                'cls_loss': self.loss(logit, y, w) + 1.0 / PSNR(x_reconstruct_train_pred, x_target_train)
            }
            print('Reconstruction Loss:' + str(PSNR(x_reconstruct_train_pred, x_target_train)))
            return loss_dict
        # reconstruct test
        if not self.training:
            
            # # reconstruct test
            feat_list_reconstruct_test = []
            for op in self.feature_ops:
                x_source_test_rec = op(x_source_test_rec)
                if isinstance(op, nn.Identity):
                    feat_list_reconstruct_test.append(x_source_test_rec) # torch.Size([1, 96, 624, 352]) 
                    break
                                    # torch.Size([1, 128, 312, 176]) 
                                    # torch.Size([1, 192, 156, 88]) 
                                    # torch.Size([1, 256, 78, 44])

            inner_feat_list_reconstrcut = [self.reduce_1x1convs[i](feat) for i, feat in enumerate(feat_list_reconstruct_test)]

            out_feat_list_reconstruct = [self.fuse_3x3convs_reconstruct[0](inner_feat_list_reconstrcut[-1])]

            for i in range(len(self.fuse_3x3convs_reconstruct) - 1):
                out = self.fuse_3x3convs_reconstruct[i + 1](out_feat_list_reconstruct[i])
                out_feat_list_reconstruct.append(out)

            final_feat_reconstruct = out_feat_list_reconstruct[-1]

            x_reconstruct_test_pred = self.reconstruct(final_feat_reconstruct)
            return torch.softmax(logit, dim=1) + 1.0 / PSNR(x_reconstruct_test_pred, x_target_test)

    def loss(self, x, y, weight):
        losses = F.cross_entropy(x, y.long() - 1, weight=None,
                                 ignore_index=-1, reduction='none')

        v = losses.mul_(weight).sum() / weight.sum()
        return v

    def set_defalut_config(self):
        self.config.update(dict(
            in_channels=204,
            num_classes=16,
            block_channels=(96, 128, 192, 256),
            num_blocks=(1, 1, 1, 1),
            inner_dim=128,
            reduction_ratio=1.0,
        ))



