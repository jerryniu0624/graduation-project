import os
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,6,7"  # Using GPUs 1 and 2
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import h5py
import argparse
import scipy.io
from simplecv.module import SEBlock
import torch.nn.functional as F
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
import seaborn as sns
from mlmetric import calculate_metrics
import sys
sys.path.append('/mnt/nas/nyz/FreeNet-master/')
from ML.mlfreenet import MetricLearningNet, MetricLearningLCTFNet, MetricLearningDynamicLCTFNet, set_srf_args, MetricLearningDynamicRecLCTFNet, MetricLearningRecLCTFNet, MetricLearningRGBNet, \
cross_entropy_loss, metric_learning_loss, plot_tsne_with_mask, generate_metric_features, nearest_neighbor_classification, nearest_neighbor_classification_with_svm

from mldataset import HyperspectralDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
# import string
import sys

# 设置随机种子函数
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for hyperspectral image classification")

    # Add arguments with default values
    parser.add_argument('--lr_train', type=float, default=0.01, help='Learning rate for training optimizer')
    parser.add_argument('--momentum_train', type=float, default=0.9, help='Momentum for SGD training optimizer')
    parser.add_argument('--lr_finetune', type=float, default=0.01, help='Learning rate for fintuning optimizer')
    parser.add_argument('--momentum_finetune', type=float, default=0.9, help='Momentum for SGD fintuning optimizer')
    parser.add_argument('--total_finetune', type=bool, default=False, help='Fintune the metric learning and classification FC')
    parser.add_argument('--epochs_train', type=int, default=200, help='Number of epochs for training')
    parser.add_argument('--epochs_finetune', type=int, default=500, help='Number of epochs for fintuning')
    parser.add_argument('--batch_size_train', type=int, default=1, help='Batch size for training')
    parser.add_argument('--batch_size_test', type=int, default=1, help='Batch size for testing')
    parser.add_argument('--num_samples_train', type=int, default=10, help='Number of samples per class for training')
    parser.add_argument('--num_samples_test', type=int, default=5, help='Number of samples per class for testing')
    parser.add_argument('--num_first_shots', type=int, default=5, help='Number of spectral bands in the first shot')
    parser.add_argument('--num_second_shots', type=int, default=0, help='Number of spectral bands in the first shot')
    parser.add_argument('--num_bands', type=int, default=6, help='Number of spectral bands in the input')
    parser.add_argument('--num_dim', type=int, default=64, help='Dimension of the embedding space')
    parser.add_argument('--train_data_path', type=str, default='/mnt/nas/nyz/FreeNet-master/merged_dataset1/train/band_100_patch_size_[512, 512]_total_class_num_86_10_shot_0_edge.h5', help='Path to training data')
    parser.add_argument('--test_data_path', type=str, default='/mnt/nas/nyz/FreeNet-master/merged_dataset1/test/AB100_total_class_num_0_10_shot_10_edge.mat', help='Path to testing data')
    parser.add_argument('--network', type=str, default='MetricLearningNet', help='Network Type')

    return parser.parse_args()

# 设置 5 个不同的随机种子
seeds = [42, 123, 456, 789, 987]

# 用于存储 5 次实验的结果
oa_list = []
aa_list = []
kappa_list = []

args = parse_args()

for seed in seeds:
    print(f"Running experiment with seed {seed}")
    set_seed(seed)
    args = parse_args()

    # 加载数据
    train_dataset = HyperspectralDataset(args.train_data_path, args.train_data_path, mode='train', num_samples_train=args.num_samples_train, seed=seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True, num_workers=0)
    test_dataset = HyperspectralDataset(args.test_data_path, args.test_data_path, mode='test', num_samples_test=args.num_samples_test, seed=seed)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size_test, shuffle=False, num_workers=0)
    images, labels, mask = next(iter(test_loader))
    num_classes = len(torch.unique(labels)) - 1
    # 初始化网络
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_bands_train = args.num_bands
    num_dim = args.num_dim
    num_first_shots = args.num_first_shots
    num_second_shots = args.num_second_shots
    minBand, maxBand, nBandDataset, dataset =  set_srf_args(args.test_data_path)
        
    if args.network == 'MetricLearningNet':
        net = MetricLearningNet(num_bands=num_bands_train, num_dim=num_dim, num_classes=num_classes).to(device)
    elif args.network == 'MetricLearningLCTFNet': 
        net = MetricLearningLCTFNet(num_bands=num_bands_train, num_dim=num_dim, num_classes=num_classes).to(device)
    elif args.network == 'MetricLearningRGBNet': 
        net = MetricLearningRGBNet(num_bands=num_bands_train, num_dim=num_dim, num_classes=num_classes).to(device)
    elif args.network == 'MetricLearningRecLCTFNet': 
        net = MetricLearningRecLCTFNet(num_bands=num_bands_train, num_dim=num_dim, num_classes=num_classes).to(device)
    elif args.network == 'MetricLearningDynamicLCTFNet': 
        net = MetricLearningDynamicLCTFNet(num_first_shots=num_first_shots, num_bands=num_bands_train, num_dim=num_dim, num_classes=num_classes,\
                                            minBand=minBand, maxBand=maxBand, nBandDataset=nBandDataset, dataset=dataset).to(device)
    elif args.network == 'MetricLearningDynamicRecLCTFNet': 
        net = MetricLearningDynamicRecLCTFNet(num_first_shots=num_first_shots, num_second_shots=num_second_shots, num_bands=num_bands_train, num_dim=num_dim, num_classes=num_classes,\
                                            minBand=minBand, maxBand=maxBand, nBandDataset=nBandDataset, dataset=dataset).to(device)
    else:
        print('network not found')
        sys.exit(0)

    # Wrap the model for multi-GPU usage
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        net = nn.DataParallel(net)

    # 使用带动量的 SGD 优化器
    optimizer = optim.SGD(net.parameters(), lr=args.lr_train, momentum=args.momentum_train)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

    # 训练网络
    epochs_train = args.epochs_train
    for epoch in range(epochs_train):
        running_loss = 0.0
        net.train()
        for i, data in enumerate(train_loader, 0):
            inputs, labels, masks = data
            inputs, labels, masks = inputs.to(device), labels.to(device), masks.to(device)
            optimizer.zero_grad()
            if isinstance(net, nn.DataParallel):
                net.module.set_mode('train')
            else:
                net.set_mode('train')
            outputs = net(inputs)
            loss = metric_learning_loss(outputs, labels, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs_train}], Loss: {running_loss/len(train_loader)}, LR: {current_lr}')

    print('Finished Training for seed', seed)

    # 测试网络并进行最近邻分类
    # net.eval()
    all_true_labels = []
    all_predicted_labels = []

    # 假设 test_loader 中只有一张图像，直接获取图像和标签
    
    images, labels, mask = next(iter(test_loader))
    # 将图像发送到 GPU（如果有设备）
    images, labels, mask = images.to(device), labels.to(device), mask.to(device)

    # 冻结网络的所有参数，除了最后一层
    if not args.total_finetune:
        if isinstance(net, nn.DataParallel):
            for name, param in net.module.named_parameters():
                if name != 'cls_pred_conv.weight' and name != 'cls_pred_conv.bias':
                    param.requires_grad = False
        else:
            for name, param in net.named_parameters():
                if name != 'cls_pred_conv.weight' and name != 'cls_pred_conv.bias':
                    param.requires_grad = False

    # 创建优化器，并且只优化最后一层的参数
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.module.parameters()), lr=args.lr_finetune)
    # 微调网络，假设微调 5 个 epoch
    for epoch in range(args.epochs_finetune):
        net.train() 
        optimizer.zero_grad()

        # 测试模式下，网络前向传播
        if isinstance(net, nn.DataParallel):
            net.module.set_mode('test')
        else:
            net.set_mode('test')
        logits = net(images)  # 获得度量空间特

        # 计算交叉熵损失
        loss_value = cross_entropy_loss(logits, labels, mask)  # 使用你提供的 loss 函数

        # 反向传播和更新网络参数
        loss_value.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss_value.item()}")

    net.eval()
    if isinstance(net, nn.DataParallel):
        net.module.set_mode('test')
    else:
        net.set_mode('test')
    outputs = net(images).squeeze(0)
    predicted_labels = outputs.argmax(dim=0).cpu() + 1

    valid_mask = (mask == 0) & (labels != 0)
    valid_mask = valid_mask.cpu()
    # valid_mask = valid_mask.unsqueeze(0)
    # 使用 mask 筛选出 mask == 0 的点
    valid_predicted_labels = predicted_labels[valid_mask.squeeze(0)]
    valid_labels = labels[valid_mask]

    # 如果没有 mask == 0 的点，跳过此批次

        # 将真实标签和预测标签存储，以便后续评估
    all_true_labels.append(valid_labels.cpu())
    all_predicted_labels.append(valid_predicted_labels.cpu())

    # 将所有批次的标签拼接起来

    all_true_labels = torch.cat(all_true_labels)
    all_predicted_labels = torch.cat(all_predicted_labels)

    # 假设有 num_classes 个类别
    num_classes = len(torch.unique(labels))  # 假设标签中类别从 1 开始

    # 计算 OA, AA, Kappa
    oa, aa, kappa = calculate_metrics(all_true_labels, all_predicted_labels, num_classes)
    print(f"Seed {seed} results:")
    print(f"Overall Accuracy (OA): {oa * 100:.2f}%")
    print(f"Average Accuracy (AA): {aa * 100:.2f}%")
    print(f"Kappa: {kappa:.4f}")

    # 保存当前实验的结果
    oa_list.append(oa)
    aa_list.append(aa)
    kappa_list.append(kappa)

# 计算 5 次实验的平均结果
oa_mean = np.mean(oa_list)
aa_mean = np.mean(aa_list)
kappa_mean = np.mean(kappa_list)

oa_var = np.var(oa_list)
aa_var = np.var(aa_list)
kappa_var = np.var(kappa_list)

print("\nFinal average results after 5 runs:")
print(f"Overall Accuracy (OA): 平均值 = {oa_mean * 100:.2f}%, 方差 = {oa_var * 100:.2f}%")
print(f"Average Accuracy (AA): 平均值 = {aa_mean * 100:.2f}%, 方差 = {aa_var * 100:.2f}%")
print(f"Kappa: 平均值 = {kappa_mean:.4f}, 方差 = {kappa_var:.4f}")