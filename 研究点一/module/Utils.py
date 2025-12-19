import numpy as np
import scipy.io as sio
import os
import glob
import re
import torch
import torch.nn as nn
import math
import random

import torch
import torch.nn as nn
import numpy as np
# import torchsnooper
import yaml
import argparse
import torch.nn.functional as F


def get_device():
    parser = argparse.ArgumentParser(description='PyTorch DCNNs Training')
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/config.yml",
        help="Configuration file to use",
    )
    args = parser.parse_args()
    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
    if cfg['GPU'] is not None:
        if torch.cuda.is_available():
            device = torch.device("cuda", cfg["GPU"])
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    return device


def paddingX(x, padding=0, padding_mode='zero'):
    if padding_mode == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    elif padding_mode == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    else:
        layer = nn.ZeroPad2d(padding)
    x = layer(x)
    return x

# ************************************** code of Attention module in SSAN:*****************************************


# @torchsnooper.snoop()
class Attention_gate(nn.Module):
    """
    Although paper said it used 3D conv to change s*s*c1 to s*s*o, but the code Author sent to me is using Conv2d. They have the same shape result shape.
    """
    def __init__(self, gate_channels, gate_depth):
        """
        input shape: NCHW-->C means the spectral channel of input
        :param gate_channels: C
        :param gate_depth: o in paper, set to be 64
        """
        super(Attention_gate, self).__init__()
        self.gate_channels = gate_channels
        self.gate_depth = gate_depth
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=gate_channels, out_channels=gate_depth, kernel_size=(1, 1), stride=(2, 2)),
            nn.BatchNorm2d(gate_depth),
            nn.ReLU(),
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=gate_channels, out_channels=gate_depth, kernel_size=(1, 1), stride=(2, 2)),
            nn.BatchNorm2d(gate_depth),
            nn.ReLU(),
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(in_channels=gate_channels, out_channels=gate_depth, kernel_size=(1, 1), stride=(2, 2)),
            nn.BatchNorm2d(gate_depth),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=gate_depth, out_channels=gate_channels, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(gate_channels),
        )

    def forward(self, x):
        conv1_1 = self.conv1_1(x)  # [N, C=64, H=5, W=5]
        conv1_2 = self.conv1_2(x)  # [N, C=64, H=5, W=5]
        conv1_3 = self.conv1_3(x)  # [N, C=64, H=5, W=5]
        # reshape: [N,C=o=64,H=s,W=s]->[N,C=o,s*s]
        conv1_1 = torch.reshape(conv1_1, (conv1_1.shape[0], conv1_1.shape[1], -1)).transpose(1, 2)  # [N,64,5*5]-->[N,5*5, 64]
        conv1_2 = torch.reshape(conv1_2, (conv1_2.shape[0], conv1_2.shape[1], -1))  # [N,64,5*5]
        conv1_3 = torch.reshape(conv1_3, (conv1_3.shape[0], conv1_3.shape[1], -1)).transpose(1, 2)  # [N,64,5*5]-->[N,5*5, 64]

        matmul1 = torch.matmul(conv1_1, conv1_2)  # [N, s*s, s*s]
        matmul1 = torch.softmax(matmul1, dim=2)  # ss*ss, softmax is utilized to normalize the R by row
        Att = torch.matmul(matmul1, conv1_3).transpose(1, 2)  # [N, s*s, s*s] x [N, s*s, C=o]->[N, s*s, C=o]->[N, C=o, s*s]
        Att = torch.reshape(Att, (Att.shape[0], Att.shape[1], x.shape[-1], -1))  # [N, C=o, s*s]->[N, C=o, H=s, W=s]
        Att_ = self.conv2(Att)  # [N, C=o, H=s, W=s]->[N,C=gate_channels,H=s,W=s]
        out = x + Att_  # [N,C=gate_channels,H=s,W=s]
        return out


# ************************************** code of Attention module in SSAN-TPPI:*************************************


class GeometryPrior(torch.nn.Module):
    def __init__(self, k, channels, multiplier=0.5):
        super(GeometryPrior, self).__init__()
        self.channels = channels
        self.k = k
        self.l1 = torch.nn.Conv2d(2, int(multiplier * channels), 1)
        self.l2 = torch.nn.Conv2d(int(multiplier * channels), channels, 1)

    def forward(self):
        # as the paper does not infer how to construct a 2xkxk position matrix
        # we assume that it's a kxk matrix for deltax,and a kxk matric for deltay.
        # that is, [[[-1,0,1],[-1,0,1],[-1,0,1]],[[1,1,1],[0,0,0],[-1,-1,-1]]] for kernel = 3
        a_range = torch.arange(-1 * (self.k // 2), (self.k // 2) + 1).view(1, -1)
        x_position = a_range.expand(self.k, a_range.shape[1])
        b_range = torch.arange((self.k // 2), -1 * (self.k // 2) - 1, -1).view(-1, 1)

        y_position = b_range.expand(b_range.shape[0], self.k)
        position = torch.cat((x_position.unsqueeze(0), y_position.unsqueeze(0)), 0).unsqueeze(0).float()
        position = position.to(get_device())
        out = self.l2(torch.nn.functional.relu(self.l1(position)))
        return out


class KeyQueryMap(torch.nn.Module):
    def __init__(self, channels, m):
        super(KeyQueryMap, self).__init__()
        self.l = torch.nn.Conv2d(channels, channels // m, 1)

    def forward(self, x):
        return self.l(x)


class AppearanceComposability(torch.nn.Module):
    def __init__(self, k, stride):
        super(AppearanceComposability, self).__init__()
        self.k = k
        self.unfold = torch.nn.Unfold(kernel_size=k, dilation=1, padding=0, stride=stride)

    def forward(self, x):
        key_map, query_map = x
        k = self.k
        key_map_unfold = self.unfold(key_map).transpose(2, 1).contiguous()  # [N batch , H_out*Wout, C channel * k*k]
        query_map_unfold = self.unfold(query_map).transpose(2,
                                                            1).contiguous()  # [N batch , H_out*Wout, C channel * k*k]
        key_map_unfold = key_map_unfold.view(key_map.shape[0], -1, key_map.shape[1],
                                             key_map_unfold.shape[-1] // key_map.shape[1])
        query_map_unfold = query_map_unfold.view(query_map.shape[0], -1, query_map.shape[1],
                                                 query_map_unfold.shape[-1] // query_map.shape[1])
        key_map_unfold = key_map_unfold.transpose(2, 1).contiguous()
        query_map_unfold = query_map_unfold.transpose(2, 1).contiguous()
        shape = (key_map_unfold.shape[0], key_map_unfold.shape[1], key_map_unfold.shape[2], k, k)
        return (key_map_unfold * query_map_unfold[:, :, :, k ** 2 // 2:k ** 2 // 2 + 1]).view(shape)  # [N batch, C channel, (H-k+1)*(W-k+1), k*k]


def combine_prior(appearance_kernel, geometry_kernel):
    return F.softmax(appearance_kernel + geometry_kernel, dim=-1)


# @torchsnooper.snoop()
class LocalRelationalLayer(torch.nn.Module):
    """
    Based on paper: Hu, H., Zhang, Z., Xie, Z., & Lin, S. (2019). Local relation networks for image recognition. https://doi.org/10.1109/ICCV.2019.00356
    """
    def __init__(self, channels, k, stride=1, padding=0, padding_mode='zero', m=None):
        super(LocalRelationalLayer, self).__init__()
        self.channels = channels
        self.k = k
        self.stride = stride
        self.m = m or 8
        self.padding = padding
        self.padding_mode = padding_mode
        self.kmap = KeyQueryMap(channels, self.m)
        self.qmap = KeyQueryMap(channels, self.m)
        self.ac = AppearanceComposability(k, self.stride)
        self.gp = GeometryPrior(k, channels // m)
        self.unfold = torch.nn.Unfold(kernel_size=k, dilation=1, padding=0, stride=self.stride)
        self.final1x1 = torch.nn.Conv2d(channels, channels, 1)

    def forward(self, x):  # x = [N,C,H,W]
        h_out = (x.shape[2] + 2 * self.padding - 1 * self.k) // self.stride + 1
        w_out = (x.shape[3] + 2 * self.padding - 1 * self.k) // self.stride + 1
        x = paddingX(x, padding=self.padding, padding_mode=self.padding_mode)
        km = self.kmap(x)  # [N,C/m,h,w]
        qm = self.qmap(x)  # [N,C/m,h,w]
        ak = self.ac((km, qm))  # [N,C/m,H_out*W_out, k,k]
        gpk = self.gp()  # [1, C/m,k,k]
        ck = combine_prior(ak, gpk.unsqueeze(2))[:, None, :, :, :]  # [N,1,C/m,H_out*W_out, k,k]
        x_unfold = self.unfold(x).transpose(2, 1).contiguous().view(x.shape[0], -1, x.shape[1],
                                                                    self.k * self.k).transpose(2, 1).contiguous()
        x_unfold = x_unfold.view(x.shape[0], self.m, x.shape[1] // self.m, -1, self.k,
                                 self.k)  # [N, m, C/m, H_out*W_out, k,k]
        pre_output = (ck * x_unfold).view(x.shape[0], x.shape[1], -1, self.k * self.k)  # [N, C,HOUT*WOUT, k*k]
        pre_output = torch.sum(pre_output, 3).view(x.shape[0], x.shape[1], h_out, w_out)  # [N, C, H_out*W_out]
        return self.final1x1(pre_output)


# ************************************** code of BottleNeck in pResNet:*************************************
# @torchsnooper.snoop()
class Bottleneck_TPPP(nn.Module):
    outchannel_ratio = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_TPPP, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(planes)

        if stride == 2:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=4, stride=stride, padding=1)
        else:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * Bottleneck_TPPP.outchannel_ratio, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(planes * Bottleneck_TPPP.outchannel_ratio)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)
            featuremap_size = shortcut.size()[2:4]
        else:
            shortcut = x
            featuremap_size = out.size()[2:4]

        batch_size = out.size()[0]
        residual_channel = out.size()[1]
        shortcut_channel = shortcut.size()[1]
        if residual_channel != shortcut_channel:
            padding = torch.autograd.Variable(
                torch.zeros(batch_size, residual_channel - shortcut_channel, featuremap_size[0],
                            featuremap_size[1]).to(get_device()))
            try:
                out += torch.cat((shortcut, padding), 1)
            except:
                print("ERROR", out.shape, shortcut.shape, padding.shape)
                exit()
        else:
            out += shortcut
        return out


# ************************************** code of BottleNeck in pResNet-TPPI:*************************************
# @torchsnooper.snoop()
class Bottleneck_TPPI(nn.Module):
    outchannel_ratio = 4

    def __init__(self, inplanes, planes, reduce=False, downsample=None):
        super(Bottleneck_TPPI, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(planes)
        if reduce:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=0)
        else:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, padding_mode="replicate")
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * Bottleneck_TPPP.outchannel_ratio, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(planes * Bottleneck_TPPP.outchannel_ratio)
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        conv2 = self.conv2(bn1)
        bn2 = self.bn2(conv2)
        conv3 = self.conv3(bn2)
        bn3 = self.bn3(conv3)
        out = self.relu(bn3)

        if self.downsample is not None:
            shortcut = self.downsample(x)
            featuremap_size = shortcut.size()[2:4]
        else:
            shortcut = x
            featuremap_size = out.size()[2:4]

        batch_size = out.size()[0]
        residual_channel = out.size()[1]
        shortcut_channel = shortcut.size()[1]
        if residual_channel != shortcut_channel:
            padding = torch.autograd.Variable(
                torch.zeros(batch_size, residual_channel - shortcut_channel, featuremap_size[0],
                            featuremap_size[1]).to(get_device()))
            try:
                out += torch.cat((shortcut, padding), 1)
            except:
                print("ERROR", out.shape, shortcut.shape, padding.shape)
                exit()
        else:
            out += shortcut
        return out

# ***************************************Others*********************************************************
def get_SSRN_channel(dataset):
    if dataset == 'IP':
        return 97
    elif dataset == 'PU':
        return 49
    elif dataset == 'SV':
        return 99


def get_SSAN_gate_channel(dataset):
    if dataset == 'IP':
        return 128
    elif dataset == 'PU':
        return 128
    elif dataset == 'SV':
        return 128


def get_in_channel(dataset):
    """
    get each dataset spectral channels
    :param dataset: dataset
    :return: the channel of the HSI patch
    """
    if dataset == 'IP':
        return 200
    elif dataset == 'PU':
        return 103
    elif dataset == 'SV':
        return 204


def get_class_num(dataset):
    """
    get each dataset number of class
    :param dataset: dataset
    :return: class number
    """
    if dataset == 'IP':
        return 16
    elif dataset == 'PU':
        return 9
    elif dataset == 'SV':
        return 16


def get_in_planes(dataset):
    """
    get the suitable channel number after feature extraction(PCA or deep learning).
    :param dataset: dataset
    :return: the suitable channel number after feature extraction(PCA or deep learning).
    """
    if dataset == 'IP':
        return 32
    elif dataset == 'PU':
        return 32
    elif dataset == 'SV':
        return 48


def get_fc_in(dataset, model):
    """
    get flatten feature length before FC layer
    :param model:
    :param dataset:
    :return: flatten result before FC layer
    """
    if dataset == 'IP':
        if model == 'CNN_1D':
            return 552
        elif model == 'CNN_2D_new':
            return 128
        elif model == 'CNN_3D_old':
            return 100
        elif model == 'CNN_3D_new':
            return 128
        elif model == 'HybridSN':
            return 1600
        elif model == 'SSAN':
            return 1600
    elif dataset == 'PU':
        if model == 'CNN_1D':
            return 192
        elif model == 'CNN_2D_new':
            return 128
        elif model == 'CNN_3D_old':
            return 32
        elif model == 'CNN_3D_new':
            return 128
        elif model == 'HybridSN':
            return 1600
        elif model == 'SSAN':
            return 1600
    elif dataset == 'SV':
        if model == 'CNN_1D':
            return 480
        elif model == 'CNN_2D_new':
            return 128
        elif model == 'CNN_3D_old':
            return 56
        elif model == 'CNN_3D_new':
            return 128
        elif model == 'HybridSN':
            return 1600
        elif model == 'SSAN':
            return 1600


def get_avgpoosize(dataset):
    """
    only used in pResNet
    :param dataset:
    :return:
    if args.spatialsize < 9: avgpoosize = 1
    elif args.spatialsize <= 11: avgpoosize = 2
    elif args.spatialsize == 15: avgpoosize = 3
    elif args.spatialsize == 19: avgpoosize = 4
    elif args.spatialsize == 21: avgpoosize = 5
    elif args.spatialsize == 27: avgpoosize = 6
    elif args.spatialsize == 29: avgpoosize = 7
    else: print("spatialsize no tested")
    """
    return 1


def get_CNN3D_new_layer3_channel(dataset):
    if dataset == 'IP':
        return 138
    elif dataset == 'PU':
        return 41
    elif dataset == 'SV':
        return 142


# if __name__ == "__main__":
#     LR_layer = LocalRelationalLayer(channels=96, k=5, stride=1, padding=2, m=8)
#     LR_layer = LR_layer.to('cuda')
#     a = np.random.random((2, 96, 5, 5))
#     a = torch.from_numpy(a).float()
#     a = a.to('cuda')
#     b = LR_layer(a)

def _as_floats(im1, im2):
    float_type = np.result_type(im1.dtype, im2.dtype, np.float32)
    im1 = np.asarray(im1, dtype=float_type)
    im2 = np.asarray(im2, dtype=float_type)
    return im1, im2


def compare_mse(im1, im2):
    im1, im2 = _as_floats(im1, im2)
    return np.mean(np.square(im1 - im2), dtype=np.float64)


def compare_psnr(im_true, im_test, data_range=None):
    im_true, im_test = _as_floats(im_true, im_test)

    err = compare_mse(im_true, im_test)
    return 10 * np.log10((data_range ** 2) / err)


def psnr(img1, img2):
   mse = np.mean((img1/255. - img2/255.) ** 2)
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def PSNR_GPU(im_true, im_fake):
    im_true *= 255
    im_fake *= 255
    im_true = im_true.round()
    im_fake = im_fake.round()
    data_range = 255
    esp = 1e-12
    C = im_true.size()[0]
    H = im_true.size()[1]
    W = im_true.size()[2]
    Itrue = im_true.clone()
    Ifake = im_fake.clone()
    mse = nn.MSELoss(reduce=False)
    err = mse(Itrue, Ifake).sum() / (C*H*W)
    psnr = 10. * np.log((data_range**2)/(err.data + esp)) / np.log(10.)
    return psnr


def PSNR_Nssr(im_true, im_fake):
    mse = ((im_true - im_fake)**2).mean()
    psnr = 10. * np.log10(1/mse)
    return psnr


def dataparallel(model, ngpus, gpu0=0):
    if ngpus==0:
        assert False, "only support gpu mode"
    gpu_list = list(range(gpu0, gpu0+ngpus))
    assert torch.cuda.device_count() >= gpu0 + ngpus
    if ngpus > 1:
        if not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model, gpu_list).cuda()
        else:

            model = model.cuda()
    elif ngpus == 1:
        model = model.cuda()
    return model


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch

# load HSIs
def prepare_data(path, file_list, file_num):
    HR_HSI = np.zeros((((512,512,28,file_num))))
    for idx in range(file_num):
        #  read HrHSI
        HR_code = file_list[idx]
        path1 = os.path.join(path) + HR_code + '.mat'
        data = sio.loadmat(path1)
        HR_HSI[:,:,:,idx] = data['data_slice'] / 65535.0
    HR_HSI[HR_HSI < 0.] = 0.
    HR_HSI[HR_HSI > 1.] = 1.
    return HR_HSI


def loadpath(pathlistfile):
    fp = open(pathlistfile)
    pathlist = fp.read().splitlines()
    fp.close()
    random.shuffle(pathlist)
    return pathlist

def y2x(y, Ch):
    ##  Spilt operator
    sz = y.size()
    if len(sz) == 3:
        y = y.unsqueeze(0)
        bs = 1
    else:
        bs = sz[0]
    sz = y.size()
    x = torch.zeros([bs, Ch, sz[2], sz[3] - (Ch - 1) * 2]).cuda()
    for t in range(Ch):
        temp = y[:, :, :, 0 + 2 * t : sz[3] + 2 * t - (Ch - 1) * 2]
        x[:, t, :, :] = temp
    return x

def x2y(x):
    ##  Shift and Sum operator
    sz = x.size()
    if len(sz) == 3:
        x = x.unsqueeze(0).unsqueeze(0)
        bs = 1
    else:
        bs = sz[0]
    sz = x.size()
    y = torch.zeros([bs, 1, sz[2], sz[2]+2*27]).cuda()
    for t in range(28):
        y[:, :, :, 0 + 2 * t : sz[2] + 2 * t] = x[:, t, :, :].unsqueeze(1) + y[:, :, :, 0 + 2 * t : sz[2] + 2 * t]
    return y