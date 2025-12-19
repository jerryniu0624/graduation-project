import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
from torch.autograd import Variable
import numpy as np
import h5py
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


def conv3x3_gn_relu(in_channel, out_channel, num_group):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 3, 1, 1),
        nn.GroupNorm(num_group, out_channel),
        nn.ReLU(inplace=True),
    )

def gn_relu(in_channel, num_group):
    return nn.Sequential(
        nn.GroupNorm(num_group, in_channel),
        nn.ReLU(inplace=True),
    )


def downsample2x(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 3, 2, 1),
        nn.ReLU(inplace=True)
    )


def repeat_block(block_channel1, r, n,conv_size):
    cl_channel = block_channel1 / 8
    cl_channel = int(cl_channel)
    cl2_channel = int(cl_channel / 2)
    gn_a = int(block_channel1 / 2)
    layers = [
        nn.Sequential(
            ConvLSTM(input_channels=cl_channel, hidden_channels=[cl_channel, cl2_channel], kernel_size=conv_size, step=8,
                     effective_step=[7]).cuda(),
            BasicBlock(gn_a), gn_relu(block_channel1, r), )]
    return nn.Sequential(*layers)

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
    return minBand, maxBand, nBandDataset, dataset
    
class FixedPosLinear(nn.Module):
    def __init__(self, in_dim, out_dim, minBand, maxBand, nBandDataset, dataset):
        super(FixedPosLinear, self).__init__()
        P = scipy.io.loadmat(r'/mnt/sde/niuyuanzhuo/FreeNet-master/P.mat')['P']
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
    
class MetricLearningSSDGLNet(nn.Module):
    def __init__(self, num_bands, num_dim, num_classes):
        super(MetricLearningSSDGLNet, self).__init__()
        # if cfg.data.train.params.select_type=='sample_percent':
        # r = int(8* 1.0)
        # kernel_size=3
        # else:
        r = int(4* 1.0)
        kernel_size=5        
        block1_channels = int(96 * 1.0 / r) * r
        block2_channels = int(128 * 1.0 / r) * r
        block3_channels = int(192 * 1.0 / r) * r
        block4_channels = int(256 * 1.0 / r) * r

        self.feature_ops = nn.ModuleList([
        
            conv3x3_gn_relu(num_bands, block1_channels, r),

            repeat_block(block1_channels, r, 1,kernel_size),  # num_blocks=(1, 1, 1, 1)
            nn.Identity(),


            downsample2x(block1_channels, block2_channels),

            repeat_block(block2_channels, r, 1,kernel_size),
            nn.Identity(),

                               
         
            downsample2x(block2_channels, block3_channels),
            repeat_block(block3_channels, r, 1,kernel_size),
            nn.Identity(),

            downsample2x(block3_channels, block4_channels),

            repeat_block(block4_channels, r, 1,kernel_size),
            nn.Identity(),                                
        ])
        inner_dim = int(num_dim * 1.0)

        self.reduce_1x1convs = nn.ModuleList([
            nn.Conv2d(block1_channels, inner_dim, 1),
            nn.Conv2d(block2_channels, inner_dim, 1),
            nn.Conv2d(block3_channels, inner_dim, 1),
            nn.Conv2d(block4_channels, inner_dim, 1),
        ])
        # if cfg.data.train.params.select_type=='sample_percent':
        # self.SA = nn.ModuleList([
        #     SpatialAttention(),
        #     SpatialAttention(),
        #     SpatialAttention(),
        #     SpatialAttention(),
        #     ])
        self.fuse_3x3convs = nn.ModuleList([
            conv3x3_gn_relu(inner_dim, inner_dim, r),
            conv3x3_gn_relu(inner_dim, inner_dim, r),
            conv3x3_gn_relu(inner_dim, inner_dim, r),
            nn.Conv2d(inner_dim, num_bands, 3, 1, 1),
        ])

        self.cls_pred_conv = nn.Conv2d(num_bands, num_classes, 1)

    def top_down(self, top, lateral):

        top2x = F.interpolate(top, scale_factor=2.0, mode='bilinear')


        return lateral+top2x

    def forward(self, x, y=None, w=None, mode='train', **kwargs):
        self.mode = mode
        feat_list = []

        for op in self.feature_ops:
            x = op(x)

            if isinstance(op, nn.Identity):

                feat_list.append(x)

        inner_feat_list = [self.reduce_1x1convs[i](feat) for i, feat in enumerate(feat_list)]

        inner_feat_list.reverse()
        out_feat_list = [self.fuse_3x3convs[0](inner_feat_list[0])]
        for i in range(len(inner_feat_list) - 1):
            inner = self.top_down(out_feat_list[i], inner_feat_list[i + 1])

            out = self.fuse_3x3convs[i + 1](inner)

            out_feat_list.append(out)
        final_feat = out_feat_list[-1]


        logit = self.cls_pred_conv(final_feat)
        # 在测试阶段输出 logits，训练阶段输出 final_feat
        if self.mode == 'test':
            logit = self.cls_pred_conv(final_feat)  # 测试时输出 logits
            return logit
        else:
            return final_feat  # 训练时输出特征
        
class MetricLearningRGBSSDGLNet(nn.Module):
    def __init__(self, num_bands, num_dim, num_classes, minBand, maxBand, nBandDataset, dataset):
        super(MetricLearningRGBSSDGLNet, self).__init__()
        # if cfg.data.train.params.select_type=='sample_percent':
        # r = int(8* 1.0)
        # kernel_size=3
        # else:
        r = int(4* 1.0)
        kernel_size=5        
        block1_channels = int(96 * 1.0 / r) * r
        block2_channels = int(128 * 1.0 / r) * r
        block3_channels = int(192 * 1.0 / r) * r
        block4_channels = int(256 * 1.0 / r) * r

        self.feature_ops = nn.ModuleList([
        
            conv3x3_gn_relu(num_bands, block1_channels, r),

            repeat_block(block1_channels, r, 1,kernel_size),  # num_blocks=(1, 1, 1, 1)
            nn.Identity(),


            downsample2x(block1_channels, block2_channels),

            repeat_block(block2_channels, r, 1,kernel_size),
            nn.Identity(),

                               
         
            downsample2x(block2_channels, block3_channels),
            repeat_block(block3_channels, r, 1,kernel_size),
            nn.Identity(),

            downsample2x(block3_channels, block4_channels),

            repeat_block(block4_channels, r, 1,kernel_size),
            nn.Identity(),                                
        ])
        inner_dim = int(num_dim * 1.0)

        self.reduce_1x1convs = nn.ModuleList([
            nn.Conv2d(block1_channels, inner_dim, 1),
            nn.Conv2d(block2_channels, inner_dim, 1),
            nn.Conv2d(block3_channels, inner_dim, 1),
            nn.Conv2d(block4_channels, inner_dim, 1),
        ])
        # if cfg.data.train.params.select_type=='sample_percent':
        # self.SA = nn.ModuleList([
        #     SpatialAttention(),
        #     SpatialAttention(),
        #     SpatialAttention(),
        #     SpatialAttention(),
        #     ])
        self.fuse_3x3convs = nn.ModuleList([
            conv3x3_gn_relu(inner_dim, inner_dim, r),
            conv3x3_gn_relu(inner_dim, inner_dim, r),
            conv3x3_gn_relu(inner_dim, inner_dim, r),
            nn.Conv2d(inner_dim, num_bands, 3, 1, 1),
        ])

        self.cls_pred_conv = nn.Conv2d(num_bands, num_classes, 1)
        self.rgb_srf = FixedPosLinear(num_bands, 3, minBand, maxBand, nBandDataset, dataset)

    def top_down(self, top, lateral):

        top2x = F.interpolate(top, scale_factor=2.0, mode='bilinear')


        return lateral+top2x

    def forward(self, x, y=None, w=None, mode='train', **kwargs):
        x = x.permute(0,2,3,1)
        x = self.rgb_srf(x) 
        x = x.permute(0,3,1,2)
        self.mode = mode
        feat_list = []

        for op in self.feature_ops:
            x = op(x)

            if isinstance(op, nn.Identity):

                feat_list.append(x)

        inner_feat_list = [self.reduce_1x1convs[i](feat) for i, feat in enumerate(feat_list)]

        inner_feat_list.reverse()
        out_feat_list = [self.fuse_3x3convs[0](inner_feat_list[0])]
        for i in range(len(inner_feat_list) - 1):
            inner = self.top_down(out_feat_list[i], inner_feat_list[i + 1])

            out = self.fuse_3x3convs[i + 1](inner)

            out_feat_list.append(out)
        final_feat = out_feat_list[-1]


        logit = self.cls_pred_conv(final_feat)
        # 在测试阶段输出 logits，训练阶段输出 final_feat
        if self.mode == 'test':
            logit = self.cls_pred_conv(final_feat)  # 测试时输出 logits
            return logit
        else:
            return final_feat  # 训练时输出特征

class MetricLearningLCTFSSDGLNet(nn.Module):
    def __init__(self, num_bands, num_dim, num_classes, num_hsi_bands):
        super(MetricLearningLCTFSSDGLNet, self).__init__()
        # if cfg.data.train.params.select_type=='sample_percent':
        # r = int(8* 1.0)
        # kernel_size=3
        # else:
        r = int(4* 1.0)
        kernel_size=5        
        block1_channels = int(96 * 1.0 / r) * r
        block2_channels = int(128 * 1.0 / r) * r
        block3_channels = int(192 * 1.0 / r) * r
        block4_channels = int(256 * 1.0 / r) * r

        self.feature_ops = nn.ModuleList([
        
            conv3x3_gn_relu(num_bands, block1_channels, r),

            repeat_block(block1_channels, r, 1,kernel_size),  # num_blocks=(1, 1, 1, 1)
            nn.Identity(),


            downsample2x(block1_channels, block2_channels),

            repeat_block(block2_channels, r, 1,kernel_size),
            nn.Identity(),

                               
         
            downsample2x(block2_channels, block3_channels),
            repeat_block(block3_channels, r, 1,kernel_size),
            nn.Identity(),

            downsample2x(block3_channels, block4_channels),

            repeat_block(block4_channels, r, 1,kernel_size),
            nn.Identity(),                                
        ])
        inner_dim = int(num_dim * 1.0)

        self.reduce_1x1convs = nn.ModuleList([
            nn.Conv2d(block1_channels, inner_dim, 1),
            nn.Conv2d(block2_channels, inner_dim, 1),
            nn.Conv2d(block3_channels, inner_dim, 1),
            nn.Conv2d(block4_channels, inner_dim, 1),
        ])
        # if cfg.data.train.params.select_type=='sample_percent':
        # self.SA = nn.ModuleList([
        #     SpatialAttention(),
        #     SpatialAttention(),
        #     SpatialAttention(),
        #     SpatialAttention(),
        #     ])
        self.fuse_3x3convs = nn.ModuleList([
            conv3x3_gn_relu(inner_dim, inner_dim, r),
            conv3x3_gn_relu(inner_dim, inner_dim, r),
            conv3x3_gn_relu(inner_dim, inner_dim, r),
            nn.Conv2d(inner_dim, num_bands, 3, 1, 1),
        ])

        self.cls_pred_conv = nn.Conv2d(num_bands, num_classes, 1)
        self.srf = LyotFilter(num_hsi_bands, num_bands, 400, 1000, 100, 'LK')

    def top_down(self, top, lateral):

        top2x = F.interpolate(top, scale_factor=2.0, mode='bilinear')


        return lateral+top2x

    def forward(self, x, y=None, w=None, mode='train', **kwargs):
        x = x.permute(0,2,3,1)
        x = self.rgb_srf(x) 
        x = x.permute(0,3,1,2)
        self.mode = mode
        feat_list = []

        for op in self.feature_ops:
            x = op(x)

            if isinstance(op, nn.Identity):

                feat_list.append(x)

        inner_feat_list = [self.reduce_1x1convs[i](feat) for i, feat in enumerate(feat_list)]

        inner_feat_list.reverse()
        out_feat_list = [self.fuse_3x3convs[0](inner_feat_list[0])]
        for i in range(len(inner_feat_list) - 1):
            inner = self.top_down(out_feat_list[i], inner_feat_list[i + 1])

            out = self.fuse_3x3convs[i + 1](inner)

            out_feat_list.append(out)
        final_feat = out_feat_list[-1]


        logit = self.cls_pred_conv(final_feat)
        # 在测试阶段输出 logits，训练阶段输出 final_feat
        if self.mode == 'test':
            logit = self.cls_pred_conv(final_feat)  # 测试时输出 logits
            return logit
        else:
            return final_feat  # 训练时输出特征
        
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        ratio = 24 # if dataset_path == "SSDGL.SSDGL_1_0_Indianpine" else 16            
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
     
        self.relu1 = nn.GELU()
        self.relu2 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

    def forward(self, x):
        residual = x
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))

        #avg_out = self.relu2(avg_out)            
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))

        #max_out = self.relu2(max_out)          
        out = self.relu2 (avg_out + max_out)

        y = x * out.view(out.size(0), out.size(1), 1, 1)

        y = y + residual
        return y


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3,7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

        #self.relu1 = nn.GELU()
        self.relu1 = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x
        avg_out = torch.mean(x, dim=1, keepdim=True)

        max_out, _ = torch.max(x, dim=1, keepdim=True)

        out = torch.cat([avg_out, max_out], dim=1)
        out1 = self.conv1(out)
      
        out2 = self.relu1(out1)
        
        out = self.sigmoid(out2)

        y = x * out.view(out.size(0), 1, out.size(-2), out.size(-1))
        y =y + residual
        return y


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, planes):
        super(BasicBlock, self).__init__()

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        #out = self.ca(x) + self.sa(x)
        out = torch.cat([self.ca(x), self.sa(x)], dim=1)
        #min -

        return out


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None
        #self.relu = nn.ReLU(inplace=True)        
        #self.relu1 = nn.GELU()
    def forward(self, x, h, c):
    
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
       
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
            self.Wcf = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
            self.Wco = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda(),
                Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda())


class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size, step=1, effective_step=[1]):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input):
        internal_state = []
        outputs = []

        for step in range(self.step):
            a = input.squeeze()

            b =  int(len(a) / 8)

            x = input[:, step * b:(step + 1) * b, :, :]

            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=(height, width))
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
            # only record effective steps
            deng = self.effective_step[0]
            if step <= deng:
                outputs.append(x)
        result = outputs[0]

        for i in range(self.step - 1):
            result = torch.cat([result, outputs[i + 1]], dim=1)
        return result
    
__all__ = ['MetricLearningSSDGLNet']