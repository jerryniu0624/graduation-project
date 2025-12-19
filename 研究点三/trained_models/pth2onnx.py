import torch
import torch.onnx
import onnx

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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

from ML.mlfreenet import MetricLearningNet, MetricLearningLCTFNet, MetricLearningDynamicLCTFNet, set_srf_args, MetricLearningDynamicRecLCTFNet, MetricLearningRecLCTFNet, MetricLearningRGBNet,\
cross_entropy_loss, metric_learning_loss, plot_tsne_with_mask, generate_metric_features, nearest_neighbor_classification, nearest_neighbor_classification_with_svm, MetricLearningRGBDynamicLCTFNet, MetricLearning1DynamicLCTFNet

from ML.mlssdgl import MetricLearningSSDGLNet, MetricLearningRGBSSDGLNet, MetricLearningLCTFSSDGLNet


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
    parser.add_argument('--epochs_train', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--epochs_finetune', type=int, default=10, help='Number of epochs for fintuning')
    parser.add_argument('--batch_size_train', type=int, default=1, help='Batch size for training')
    parser.add_argument('--batch_size_test', type=int, default=1, help='Batch size for testing')
    parser.add_argument('--num_samples_train', type=int, default=100, help='Number of samples per class for training')
    parser.add_argument('--num_samples_test', type=int, default=10, help='Number of samples per class for testing')
    parser.add_argument('--num_first_shots', type=int, default=3, help='Number of spectral bands in the first shot')
    parser.add_argument('--num_second_shots', type=int, default=3, help='Number of spectral bands in the first shot')
    parser.add_argument('--num_third_shots', type=int, default=0, help='Number of spectral bands in the first shot')
    parser.add_argument('--num_bands', type=int, default=6, help='Number of spectral bands in the input')
    parser.add_argument('--num_dim', type=int, default=256, help='Dimension of the embedding space') # rec_bands
    parser.add_argument('--rec_bands', type=int, default=20, help='Number of the reconstruction bands')
    parser.add_argument('--train_data_path', type=str, default='/mnt/nas/xinjiang/code/nyz/FreeNet-master/merged_dataset1/train/band_100_patch_size_[512, 512]_total_class_num_86_10_shot_0_edge.h5', help='Path to training data')
    parser.add_argument('--test_data_path', type=str, default='/mnt/nas/xinjiang/code/nyz/FreeNet-master/merged_dataset1/test/Houston18100_total_class_num_0_10_shot_10_edge.mat', help='Path to testing data')
    parser.add_argument('--network', type=str, default='MetricLearningDynamicLCTFNet', help='Network Type')
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

    num_classes = 20
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

# 假设你有一个训练好的模型
# 在您的代码中找到导出ONNX的部分，修改为：

# 假设你有一个训练好的模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = net.to(device=device)
model.load_state_dict(torch.load('trained_models/model_weights.pth'))
model.eval()  # 设置为评估模式

# 定义推理包装器
class InferenceWrapper(nn.Module):
    """包装模型以适应ONNX导出"""
    def __init__(self, model):
        super(InferenceWrapper, self).__init__()
        self.model = model
        
    def forward(self, x):
        # 创建伪标签和掩码
        batch_size, channels, height, width = x.shape
        dummy_label = torch.zeros(batch_size, height, width, dtype=torch.long, device=x.device)
        dummy_mask = torch.zeros(batch_size, height, width, dtype=torch.long, device=x.device)
        
        # 调用原始模型
        output = self.model(x, dummy_label, dummy_mask, mode='test')
        return output

# 使用包装器
wrapped_model = InferenceWrapper(model).to(device)
wrapped_model.eval()

# 创建一个示例输入
batch_size = 1
dummy_input = torch.randn(batch_size, 100, 224, 224).to(device=device)

# 设置输入和输出节点的名称
input_names = ["input"]
output_names = ["output"]

# 导出为 ONNX - 修改这里，添加动态H和W
torch.onnx.export(
    wrapped_model,                    # 使用包装后的模型
    dummy_input,                      # 示例输入
    "trained_models/model.onnx",      # 输出文件路径
    export_params=True,               # 是否导出模型参数
    opset_version=13,                 # ONNX算子集版本
    do_constant_folding=True,         # 是否执行常量折叠优化
    input_names=input_names,          # 输入节点名称
    output_names=output_names,        # 输出节点名称
    dynamic_axes={                    # 动态维度设置
        'input': {
            0: 'batch_size',  # 批次大小动态
            2: 'height',      # 高度动态
            3: 'width'        # 宽度动态
        },
        'output': {
            0: 'batch_size',  # 批次大小动态
            2: 'height',      # 高度动态
            3: 'width'        # 宽度动态
        }
    }
)

print("模型已成功导出为 ONNX 格式，支持动态H和W")