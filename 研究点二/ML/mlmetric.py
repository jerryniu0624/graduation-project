
import numpy as np
from sklearn.metrics import confusion_matrix

def calculate_metrics(true_labels, predicted_labels, num_classes):
    # 将 true_labels 和 predicted_labels 展平
    true_labels = true_labels.cpu().numpy().flatten()
    predicted_labels = predicted_labels.cpu().numpy().flatten()

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=np.arange(num_classes))

    # Overall Accuracy (OA)
    oa = np.trace(conf_matrix) / np.sum(conf_matrix)

    # Average Accuracy (AA)
    class_accuracy = np.diag(conf_matrix) / (np.sum(conf_matrix, axis=1) + 1e-8)  # 对每个类别的准确率
    aa = np.mean(class_accuracy)

    # Kappa 系数
    total = np.sum(conf_matrix)
    p0 = np.trace(conf_matrix) / total
    pe = np.sum(np.sum(conf_matrix, axis=0) * np.sum(conf_matrix, axis=1)) / (total ** 2)
    kappa = (p0 - pe) / (1 - pe + 1e-8)

    return oa, aa, kappa