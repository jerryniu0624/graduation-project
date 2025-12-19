import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

def visualize_comparison(images, labels, predicted_labels, batch_idx=None, save_path=None):
    """
    可视化原图（高光谱图像，选择三个波段作为伪彩色图像）、真值标签和预测标签。
    
    Args:
        images (torch.Tensor): 输入的高光谱图像，形状为 [batch_size, num_bands, height, width]。
        labels (torch.Tensor): 真值标签，形状为 [batch_size, height, width]。
        predicted_labels (torch.Tensor): 预测的标签，形状为 [batch_size, height, width]。
        batch_idx (int): 如果指定，将可视化该 batch 内的图像和标签。
        save_path (str): 如果指定，将可视化结果保存到该路径。
    """
    
    # 将数据从 GPU 移动到 CPU 并转换为 numpy 格式
    images_np = images.cpu().numpy()  # [batch_size, num_bands, height, width]
    labels_np = labels.cpu().numpy()  # [batch_size, height, width]
    predicted_labels_np = predicted_labels.cpu().numpy()  # [batch_size, height, width]
    
    # 如果没有指定 batch_idx，默认显示第一个 batch
    if batch_idx is None:
        batch_idx = 0
    
    # 提取当前 batch 的图像、真值标签和预测标签
    image = images_np[batch_idx]  # [num_bands, height, width]
    label = labels_np[batch_idx]  # [height, width]
    predicted_label = predicted_labels_np  # [height, width]
    
    # 选择三个波段进行可视化（伪彩色图像），例如第 30、50、70 波段
    # 注意：波段选择可以根据你的数据进行调整
    band1, band2, band3 = 30, 50, 70
    image_rgb = np.stack([image[band1], image[band2], image[band3]], axis=-1)  # [height, width, 3]
    
    # 归一化图像到 [0, 1] 范围
    image_rgb = (image_rgb - image_rgb.min()) / (image_rgb.max() - image_rgb.min())
    
    # 创建一个 1x3 的子图，分别显示伪彩色原图、真值标签和预测标签
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # 显示原图（伪彩色）
    ax[0].imshow(image_rgb)
    ax[0].set_title("Original Image (Pseudo-RGB)")
    ax[0].axis("off")
    
    # 显示真值标签
    ax[1].imshow(label, cmap="jet")
    ax[1].set_title("Ground Truth")
    ax[1].axis("off")
    
    # 显示预测标签
    ax[2].imshow(predicted_label, cmap="jet")
    ax[2].set_title("Predicted Labels")
    ax[2].axis("off")
    
    # 如果指定了保存路径，保存图片
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"可视化结果已保存到 {save_path}")
    
    # 显示图像
    plt.show()