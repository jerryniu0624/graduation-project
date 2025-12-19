import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_USE_CUDA_DSA"] = "TRUE"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
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
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
import seaborn as sns
from mlmetric import calculate_metrics
from ML.mlfreenet import MetricLearningNet, MetricLearningLCTFNet, MetricLearningDynamicLCTFNet, set_srf_args, MetricLearningDynamicRecLCTFNet, MetricLearningRecLCTFNet, MetricLearningRGBNet,\
cross_entropy_loss, metric_learning_loss, plot_tsne_with_mask, generate_metric_features, nearest_neighbor_classification, nearest_neighbor_classification_with_svm, MetricLearningRGBDynamicLCTFNet, MetricLearning1DynamicLCTFNet

from ML.mlssdgl import MetricLearningSSDGLNet, MetricLearningRGBSSDGLNet, MetricLearningLCTFSSDGLNet

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
    parser.add_argument('--lr_train', type=float, default=0.001, help='Learning rate for training optimizer')
    parser.add_argument('--momentum_train', type=float, default=0.9, help='Momentum for SGD training optimizer')
    parser.add_argument('--lr_finetune', type=float, default=0.001, help='Learning rate for fintuning optimizer')
    parser.add_argument('--momentum_finetune', type=float, default=0.9, help='Momentum for SGD fintuning optimizer')
    parser.add_argument('--total_finetune', type=bool, default=True, help='Fintune the metric learning and classification FC')
    parser.add_argument('--epochs_train', type=int, default=0, help='Number of epochs for training')
    parser.add_argument('--epochs_finetune', type=int, default=100, help='Number of epochs for fintuning')
    parser.add_argument('--batch_size_train', type=int, default=1, help='Batch size for training')
    parser.add_argument('--batch_size_test', type=int, default=1, help='Batch size for testing')
    parser.add_argument('--num_samples_train', type=int, default=1, help='Number of samples per class for training')
    parser.add_argument('--num_samples_test', type=int, default=10, help='Number of samples per class for testing')
    parser.add_argument('--num_first_shots', type=int, default=3, help='Number of spectral bands in the first shot')
    parser.add_argument('--num_second_shots', type=int, default=3, help='Number of spectral bands in the first shot')
    parser.add_argument('--num_third_shots', type=int, default=0, help='Number of spectral bands in the first shot')
    parser.add_argument('--num_bands', type=int, default=100, help='Number of spectral bands in the input')
    parser.add_argument('--num_dim', type=int, default=192, help='Dimension of the embedding space') # rec_bands
    parser.add_argument('--rec_bands', type=int, default=20, help='Number of the reconstruction bands')
    parser.add_argument('--train_data_path', type=str, default='/mnt/nas/xinjiang/code/nyz/FreeNet-master/merged_dataset1/train/band_100_patch_size_[512, 512]_total_class_num_86_10_shot_0_edge.h5', help='Path to training data')
    parser.add_argument('--test_data_path', type=str, default='/mnt/nas/xinjiang/code/nyz/FreeNet-master/merged_dataset1/test/Houston18100_total_class_num_0_10_shot_10_edge.mat', help='Path to testing data')
    parser.add_argument('--network', type=str, default='MetricLearningDynamicRecLCTFNet', help='Network Type')
    args = parser.parse_args()

    # 打印所有参数及其值
    print("Parsed Arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    return parser.parse_args()

# 设置 5 个不同的随机种子
seeds = [789] #  , 987, 42, 123, 456

# 用于存储 5 次实验的结果
oa_list = []
aa_list = []
kappa_list = []

args = parse_args()
index = 0
for seed in seeds:
    print(f"Running experiment with seed {seed}")
    set_seed(seed)
    # args = parse_args()

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
    num_third_shots = args.num_third_shots
    rec_bands = args.rec_bands
    minBand, maxBand, nBandDataset, dataset = set_srf_args(args.test_data_path)
        
    if args.network == 'MetricLearningNet':
        net = MetricLearningNet(num_bands=num_bands_train, num_dim=num_dim, num_classes=num_classes).to(device)
    elif args.network == 'MetricLearningLCTFNet': 
        net = MetricLearningLCTFNet(num_bands=num_bands_train, num_dim=num_dim, num_classes=num_classes).to(device)
    elif args.network == 'MetricLearningRGBNet': 
        net = MetricLearningRGBNet(num_bands=num_bands_train, num_dim=num_dim, num_classes=num_classes, minBand=minBand, maxBand=maxBand, nBandDataset=nBandDataset, dataset=dataset).to(device)
    elif args.network == 'MetricLearningRecLCTFNet': 
        net = MetricLearningRecLCTFNet(num_bands=num_bands_train, num_dim=num_dim, num_classes=num_classes).to(device)
    elif args.network == 'MetricLearningDynamicLCTFNet': 
        net = MetricLearningDynamicLCTFNet(num_first_shots=num_first_shots, num_bands=num_bands_train, num_dim=num_dim, num_classes=num_classes,\
                                            minBand=minBand, maxBand=maxBand, nBandDataset=nBandDataset, dataset=dataset).to(device)
    elif args.network == 'MetricLearning1DynamicLCTFNet': 
        net = MetricLearning1DynamicLCTFNet(num_bands=num_bands_train, num_dim=num_dim, num_classes=num_classes,\
                                            minBand=minBand, maxBand=maxBand, nBandDataset=nBandDataset, dataset=dataset).to(device)
    elif args.network == 'MetricLearningDynamicRecLCTFNet': 
        net = MetricLearningDynamicRecLCTFNet(num_first_shots=num_first_shots, num_second_shots=num_second_shots, num_bands=num_bands_train, num_dim=num_dim, num_classes=num_classes,\
                                            minBand=minBand, maxBand=maxBand, nBandDataset=nBandDataset, dataset=dataset, rec_bands=rec_bands).to(device)
    elif args.network == 'MetricLearningRGBDynamicLCTFNet': 
        net = MetricLearningRGBDynamicLCTFNet(num_first_shots=num_first_shots, num_second_shots=num_second_shots, num_third_shots=num_third_shots, num_bands=num_bands_train, num_dim=num_dim, num_classes=num_classes,\
                                            minBand=minBand, maxBand=maxBand, nBandDataset=nBandDataset, dataset=dataset).to(device)
    elif args.network == 'MetricLearningSSDGLNet':
        net = MetricLearningSSDGLNet(num_bands=num_bands_train, num_dim=num_dim, num_classes=num_classes).to(device)
    elif args.network == 'MetricLearningRGBSSDGLNet':
        net = MetricLearningRGBSSDGLNet(num_bands=num_bands_train, num_dim=num_dim, num_classes=num_classes, minBand=minBand, maxBand=maxBand, nBandDataset=nBandDataset, dataset=dataset).to(device)

    else:
        print('network not found')
        sys.exit(0)
    # 使用带动量的 SGD 优化器
    optimizer = optim.SGD(net.parameters(), lr=args.lr_train, momentum=args.momentum_train)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

    # 训练网络
    epochs_train = args.epochs_train
    net.time_ = 0
    net.seed = seed
    net.l1_loss_l = []
    net.cls_loss_l = []
    for epoch in range(epochs_train):
        running_loss = 0.0
        net.train()
        for i, data in enumerate(train_loader, 0):
            inputs, labels, masks = data
            inputs, labels, masks = inputs.to(device), labels.to(device), masks.to(device)
            optimizer.zero_grad()
            output = net(inputs, y=labels, w=masks, mode='train') # loss, _
            # loss = metric_learning_loss(outputs, labels, masks)
            loss = metric_learning_loss(output, labels, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs_train}], Loss: {running_loss/len(train_loader)}, LR: {current_lr}')


    # 测试网络并进行最近邻分类
    # net.eval()
    all_true_labels = []
    all_predicted_labels = []

    # 假设 test_loader 中只有一张图像，直接获取图像和标签
    
    images, labels, mask = next(iter(test_loader))
    # 将图像发送到 GPU（如果有设备）
    images, labels, mask = images.to(device), labels.to(device), mask.to(device)

    # 冻结网络的所有参数，除了最后一层
    # if not args.total_finetune:
    #     for name, param in net.named_parameters():
    #         if name != 'cls_pred_conv.weight' and name != 'cls_pred_conv.bias':
    #             param.requires_grad = False

    # 创建优化器，并且只优化最后一层的参数
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr_finetune)
    optimizer = optim.SGD(net.parameters(), lr=args.lr_train, momentum=args.momentum_train)
    # 微调网络，假设微调 5 个 epoch
    net.time_ = 0
    net.seed = seed
    net.l1_loss_l = []
    net.cls_loss_l = []
    for epoch in range(args.epochs_finetune):
        net.train() 
        optimizer.zero_grad()

        # 测试模式下，网络前向传播
        output = net(images, y=labels, w=mask, mode='test')  # 获得度量空间特 loss, _
        
        loss = cross_entropy_loss(output, labels, mask)

        # 反向传播和更新网络参数
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    net.eval()
    outputs = net(images, y=labels, w=mask, mode='test')
    outputs = outputs.squeeze(0)
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
    index += 1

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
print(f"Kappa: 平均值 = {kappa_mean:.4f}, 方差 = {kappa_var:.4f}")

