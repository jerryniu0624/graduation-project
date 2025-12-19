from torch.utils.data import dataset
import numpy as np
from simplecv.data.preprocess import divisible_pad, intact_divisible_pad
import torch
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float
from scipy.io import loadmat
import os
SEED = 2333
def count_total_foreground_pixels_in_multi_class_superpixels(superpixels, labels):
    """
    计算所有包含多个前景类别的超像素中的前景像素总数

    参数:
    - superpixels: numpy array of shape [h, w], 超像素标签 (0 to n)
    - labels: numpy array of shape [h, w], 类别标签 (0-class_num)，其中0是背景，1-class_num是前景

    返回:
    - total_foreground_pixel_count: int, 所有包含多个前景类别的超像素中的前景像素总数
    """
    unique_superpixels = np.unique(superpixels)
    total_foreground_pixel_count = 0

    for sp in unique_superpixels:
        # 获取当前超像素的掩码
        mask = (superpixels == sp)
        
        # 获取当前超像素内的类别标签
        sp_labels = labels[mask]
        
        # 统计前景类别的数量（不包括背景0）
        foreground_classes = np.unique(sp_labels[sp_labels > 0])
        
        if len(foreground_classes) > 1:
            # 计算前景像素个数
            foreground_pixel_count = np.sum(sp_labels > 0)
            total_foreground_pixel_count += foreground_pixel_count
            
    return total_foreground_pixel_count

def check_unique_class_per_superpixel(segments, train_indicator, mask):
    """
    check_unique_class_per_superpixel

    Parameters:
    - segments: 2D array with superpixel labels
    - train_indicator: 2D array with values 0 (background/test set) or 1 (training set)
    - mask: 2D array with class labels starting from 0

    Returns:
    - updated_mask: 2D array with updated class labels
    """
    unique_segments = np.unique(segments)
    updated_mask = np.copy(mask)

    for segment_id in unique_segments:
        segment_mask = (segments == segment_id)
        train_mask = train_indicator * segment_mask
        
        if np.any(train_mask):
            # Extract class labels within the current superpixel
            class_labels = mask * train_mask
            unique_labels, counts = np.unique(class_labels, return_counts=True)
            
            if 0 in unique_labels:
                if len(unique_labels) - 1 > 1:
                    return False
            else:
                if len(unique_labels) > 1:
                    return False

    return True

def update_indicators_and_mask(segments, train_indicator, test_indicator, mask):
    """
    Update train_indicator, test_indicator, and mask based on segments.

    Parameters:
    - segments: 2D numpy array with superpixel segments
    - train_indicator: 2D numpy array with shape [H, W], indicating training samples
    - test_indicator: 2D numpy array with shape [H, W], indicating testing samples
    - mask: 2D numpy array with shape [H, W], containing pixel-wise labels

    Returns:
    - updated_train_indicator: Updated train_indicator
    - updated_test_indicator: Updated test_indicator (inverse of train_indicator)
    - updated_mask: Updated mask
    """
    updated_train_indicator = train_indicator.copy()
    updated_test_indicator = test_indicator.copy()
    updated_mask = mask.copy()

    # Iterate through each unique segment
    for seg_val in np.unique(segments):
        segment_mask = (segments == seg_val)
        
        # Check if there are any training points within this segment
        if (train_indicator * segment_mask).any():
            # Find the training point within this segment
            train_points = train_indicator * segment_mask
            class_value = mask[train_points.astype(bool)][0]  # Get the class value of the training point
            
            # Only update if class_value is not 0
            if class_value != 0:
                # Set all values in this segment to 1 in the updated_train_indicator
                updated_train_indicator[segment_mask] = 1
                
                # Set all values in this segment to the class_value in the updated_mask
                updated_mask[segment_mask] = class_value
                
                # Set all values in this segment to 0 in the updated_test_indicator
                updated_test_indicator[segment_mask] = 0
            
    # Ensure mask positions that are 0 remain 0 in both train_indicator and test_indicator
    updated_train_indicator[mask == 0] = 0
    updated_test_indicator[mask == 0] = 0

    return updated_train_indicator, updated_test_indicator, updated_mask

class IntactFullImageDataset(dataset.Dataset):
    # 一分为二
    def __init__(self,
                 image,
                 mask,
                 training,
                 np_seed=2333,
                 num_train_samples_per_class=200,
                 sub_minibatch=200,
                 slic=False,
                 n_segments = 100,
                 compactness = 1):

        self.training = training

        self.image = image
        self.mask = mask
                
        self.slic = slic
        self.num_train_samples_per_class = num_train_samples_per_class
        self.sub_minibatch = sub_minibatch
        self._seed = np_seed
        self._rs = np.random.RandomState(np_seed)
        # set list lenght = 9999 to make sure seeds enough
        self.seeds_for_minibatchsample = [e for e in self._rs.randint(low=2 << 31 - 1, size=9999)]
        self.n_segments = n_segments
        self.compactness = compactness
        self.preset()

    def preset(self):
        train_indicator, test_indicator = intact_fixed_num_sample(self.mask, self.num_train_samples_per_class,
                                                           self.num_classes, self._seed)
        segments = np.zeros_like(self.mask)
        # 用slic超像素扩充训练集，并删除测试集对应位置
        
        if not self.training:

            if self.slic:
                n_segments = self.n_segments
                # segments = np.zeros_like(self.mask)
                segments = slic(self.image, n_segments=n_segments, compactness=self.compactness, sigma=1, start_label=1)
                while not check_unique_class_per_superpixel(segments, train_indicator, self.mask):
                    n_segments += self.n_segments//10
                    segments = slic(self.image, n_segments=n_segments, compactness=1, sigma=1, start_label=1)
                    
                print('count_total_foreground_pixels_in_multi_class_superpixels:')
                print(count_total_foreground_pixels_in_multi_class_superpixels(segments, self.mask))
                print('total pixels:')
                print(self.mask.shape[0] * self.mask.shape[1])
                print(n_segments)
                # Visualize the results
                # Create a copy of the labels to highlight training samples
                # Create a light version of the labels for background
                unique_labels = np.unique(self.mask)
                light_colors = [
                    (1.0, 0.8, 0.8),  # Light red
                    (0.8, 1.0, 0.8),  # Light green
                    (0.8, 0.8, 1.0),  # Light blue
                    (1.0, 1.0, 0.8),  # Light yellow
                    (0.8, 1.0, 1.0),  # Light cyan
                    (1.0, 0.8, 1.0),  # Light magenta
                    (1.0, 0.9, 0.7),  # Light orange
                    (0.9, 1.0, 0.7),  # Light lime
                    (0.7, 0.9, 1.0),  # Light sky blue
                    (1.0, 0.7, 0.9),  # Light pink
                    (1.0, 0.9, 0.9),  # Very light red
                    (0.9, 1.0, 0.9),  # Very light green
                    (0.9, 0.9, 1.0),  # Very light blue
                    (1.0, 1.0, 0.9),  # Very light yellow
                    (0.9, 1.0, 1.0),  # Very light cyan
                    (1.0, 0.9, 1.0),  # Very light magenta
                    (1.0, 0.95, 0.85), # Very light orange
                    (0.95, 1.0, 0.85), # Very light lime
                    (0.85, 0.95, 1.0), # Very light sky blue
                    (1.0, 0.85, 0.95)  # Very light pink
                ]

                # Ensure we have enough light colors for all unique labels
                if len(unique_labels) > len(light_colors):
                    raise ValueError("Not enough light colors provided for the number of unique labels.")

                # Create an RGB image with light labels
                label_rgb = np.zeros((*self.mask.shape, 3), dtype=float)
                for i, label in enumerate(unique_labels):
                    label_rgb[self.mask == label] = light_colors[i % len(light_colors)]

                # Convert the label image to float
                label_rgb = img_as_float(label_rgb)
                label_rgb[self.mask == 0] = [0, 0, 0]
                # Visualize the result
                fig, ax = plt.subplots(figsize=(12, 12))

                # Display the segmented image with light labels
                ax.imshow(mark_boundaries(label_rgb, segments, color=(0, 0, 1), mode='inner', outline_color=None))
                ax.set_title('Ground Truth with Training Samples Highlighted')

                # Overlay red dots for training samples
                train_y, train_x = np.where(train_indicator > 0)
                ax.scatter(train_x, train_y, c='red', s=10)  # Adjust 's' for dot size

                ax.axis('off')

                plt.tight_layout()
                file_path = 'slic_show.jpg'  # 替换为你的文件路径

                # 检查文件是否存在
                if os.path.exists(file_path):
                    # 文件存在，删除文件
                    os.remove(file_path)
                    print(f"文件 '{file_path}' 已删除。")
                else:
                    # 文件不存在
                    print(f"文件 '{file_path}' 不存在。")
                plt.savefig('slic_show.jpg')
                # plt.show()

                # Update train_indicator, test_indicator, and labels based on segments
                updated_train_indicator, updated_test_indicator, updated_labels = update_indicators_and_mask(segments, train_indicator, test_indicator, self.mask)

                # Display original and updated train_indicator
                plt.figure(figsize=(15, 10))

                plt.subplot(3, 2, 1)
                plt.imshow(train_indicator, cmap='gray')
                plt.title('Original Train Indicator')
                plt.axis('off')

                plt.subplot(3, 2, 2)
                plt.imshow(updated_train_indicator, cmap='gray')
                plt.title('Updated Train Indicator')
                plt.axis('off')

                plt.subplot(3, 2, 3)
                plt.imshow(test_indicator, cmap='gray')
                plt.title('Original Test Indicator')
                plt.axis('off')

                plt.subplot(3, 2, 4)
                plt.imshow(updated_test_indicator, cmap='gray')
                plt.title('Updated Test Indicator')
                plt.axis('off')

                # Display original and updated labels
                plt.subplot(3, 2, 5)
                plt.imshow(self.mask, cmap='nipy_spectral')
                plt.title('Original Labels')
                plt.axis('off')

                plt.subplot(3, 2, 6)
                plt.imshow(updated_labels, cmap='nipy_spectral')
                plt.title('Updated Labels')
                plt.axis('off')
                file_path = 'slic_show1.jpg'  # 替换为你的文件路径

                # 检查文件是否存在
                if os.path.exists(file_path):
                    # 文件存在，删除文件
                    os.remove(file_path)
                    print(f"文件 '{file_path}' 已删除。")
                else:
                    # 文件不存在
                    print(f"文件 '{file_path}' 不存在。")
                plt.savefig('slic_show1.jpg')
                plt.tight_layout()
                # plt.show()
                plt.close()

                blob = divisible_pad([np.concatenate([self.image.transpose(2, 0, 1),
                                                updated_labels[None, :, :],
                                                updated_train_indicator[None, :, :],
                                                updated_test_indicator[None, :, :],
                                                segments[None, :, :]], axis=0)], 16, False)
                
            else:
                # file_path = 'slic_show2.jpg'
                # # 检查文件是否存在
                # if os.path.exists(file_path):
                #     # 文件存在，删除文件
                #     os.remove(file_path)
                #     print(f"文件 '{file_path}' 已删除。")
                # else:
                #     # 文件不存在
                #     print(f"文件 '{file_path}' 不存在。")
                # plt.figure(figsize=(15, 10))
                # plt.imshow(self.mask, cmap='nipy_spectral')
                # plt.savefig('slic_show2.jpg')
                # plt.show()
                blob = divisible_pad([np.concatenate([self.image.transpose(2, 0, 1),
                                              self.mask[None, :, :],
                                              train_indicator[None, :, :],
                                              test_indicator[None, :, :],
                                              segments[None, :, :]], axis=0)], 16, False)
        
            im = blob[0, :self.image.shape[-1], :, :]

            mask = blob[0, -4, :, :]
            self.train_indicator = blob[0, -3, :, :]
            self.test_indicator = blob[0, -2, :, :]
            self.segments = blob[0, -1, :, :]

        else:
            im = self.image.transpose(0, 3, 1, 2)
            mask = self.mask
            self.train_indicator = train_indicator
            self.test_indicator = test_indicator
            self.segments = segments

        if self.training:
            self.train_inds_list = minibatch_sample(mask, self.train_indicator, self.sub_minibatch,
                                                    seed=self.seeds_for_minibatchsample.pop())

        self.pad_im = im
        self.pad_mask = mask

    def resample_minibatch(self):
        self.train_inds_list = minibatch_sample(self.pad_mask, self.train_indicator, self.sub_minibatch,
                                                seed=self.seeds_for_minibatchsample.pop())

    @property
    def num_classes(self):
        return 9

    def __getitem__(self, idx):

        if self.training:
            return self.pad_im, self.pad_mask, self.train_inds_list[idx]
        else:
            return self.pad_im, self.pad_mask, self.test_indicator

    def __len__(self):
        if self.training:
            return len(self.train_inds_list)
        else:
            return 1
        
class FullImageDataset(dataset.Dataset):
    # 一分为二
    def __init__(self,
                 image,
                 mask,
                 training,
                 np_seed=2333,
                 num_train_samples_per_class=200,
                 sub_minibatch=200,
                 slic=False,
                 n_segments = 100,
                 compactness = 1):

        self.training = training
        flag1 = []
        flag2 = []
        for i in range(len(np.unique(mask[:,:]).tolist()) - 1):
            if np.where(mask[:int(image.shape[0]/2),:].ravel() == (i + 1))[0].size == 0 \
                or np.where(mask[int(image.shape[0]/2):,:].ravel() == (i + 1))[0].size == 0:
                flag1.append(i + 1)
            if np.where(mask[:,:int(image.shape[1]/2)].ravel() == (i + 1))[0].size == 0 \
                or np.where(mask[:,int(image.shape[1]/2):].ravel() == (i + 1))[0].size == 0:
                flag2.append(i + 1)
        
        if len(flag1) <= len(flag2):
            flag = 0 # 使用flag1
            # print('ok')
            for j in range(len(flag1)):
                mask[np.where(mask[:,:] == flag1[j])] = 0
        else:
            flag = 1
            # print('ok')
            for j in range(len(flag2)):
                mask[np.where(mask[:,:] == flag2[j])] = 0
        
        for k in range(len(np.unique(mask).tolist())): # 整理mask
            mask[np.where(mask[:,:] == np.sort(np.unique(mask)).tolist()[k])] = \
                np.argsort(np.unique(mask)).tolist()[k]
            
        if training:
            if flag == 0:
                self.image = image[:int(image.shape[0]/2),:]
                self.mask = mask[:int(image.shape[0]/2),:]
            else:
                self.image = image[:,:int(image.shape[1]/2)]
                self.mask = mask[:,:int(image.shape[1]/2)]
        else:
            if flag == 0:
                self.image = image[int(image.shape[0]/2):,:]
                self.mask = mask[int(image.shape[0]/2):,:]
            else:
                self.image = image[:,int(image.shape[1]/2):]
                self.mask = mask[:,int(image.shape[1]/2):]
        # self.image = image
        # self.mask = mask
                
        self.slic = slic
        self.num_train_samples_per_class = num_train_samples_per_class
        self.sub_minibatch = sub_minibatch
        self._seed = np_seed
        self._rs = np.random.RandomState(np_seed)
        # set list lenght = 9999 to make sure seeds enough
        self.seeds_for_minibatchsample = [e for e in self._rs.randint(low=2 << 31 - 1, size=9999)]
        self.n_segments = n_segments
        self.compactness = compactness
        self.preset()

    def preset(self):
        train_indicator, test_indicator = fixed_num_sample(self.mask, self.num_train_samples_per_class,
                                                           self.num_classes, self._seed)
        segments = np.zeros_like(self.mask)
        # 用slic超像素扩充训练集，并删除测试集对应位置
        if self.slic:
            n_segments = self.n_segments
            # segments = np.zeros_like(self.mask)
            segments = slic(self.image, n_segments=n_segments, compactness=self.compactness, sigma=1, start_label=1)
            while not check_unique_class_per_superpixel(segments, train_indicator, self.mask):
                n_segments += self.n_segments//10
                segments = slic(self.image, n_segments=n_segments, compactness=1, sigma=1, start_label=1)
                
            print('count_total_foreground_pixels_in_multi_class_superpixels:')
            print(count_total_foreground_pixels_in_multi_class_superpixels(segments, self.mask))
            print('total pixels:')
            print(self.mask.shape[0] * self.mask.shape[1])
            print(n_segments)
            # Visualize the results
            # Create a copy of the labels to highlight training samples
            # Create a light version of the labels for background
            unique_labels = np.unique(self.mask)
            light_colors = [
                (1.0, 0.8, 0.8),  # Light red
                (0.8, 1.0, 0.8),  # Light green
                (0.8, 0.8, 1.0),  # Light blue
                (1.0, 1.0, 0.8),  # Light yellow
                (0.8, 1.0, 1.0),  # Light cyan
                (1.0, 0.8, 1.0),  # Light magenta
                (1.0, 0.9, 0.7),  # Light orange
                (0.9, 1.0, 0.7),  # Light lime
                (0.7, 0.9, 1.0),  # Light sky blue
                (1.0, 0.7, 0.9),  # Light pink
                (1.0, 0.9, 0.9),  # Very light red
                (0.9, 1.0, 0.9),  # Very light green
                (0.9, 0.9, 1.0),  # Very light blue
                (1.0, 1.0, 0.9),  # Very light yellow
                (0.9, 1.0, 1.0),  # Very light cyan
                (1.0, 0.9, 1.0),  # Very light magenta
                (1.0, 0.95, 0.85), # Very light orange
                (0.95, 1.0, 0.85), # Very light lime
                (0.85, 0.95, 1.0), # Very light sky blue
                (1.0, 0.85, 0.95)  # Very light pink
            ]

            # Ensure we have enough light colors for all unique labels
            if len(unique_labels) > len(light_colors):
                raise ValueError("Not enough light colors provided for the number of unique labels.")

            # Create an RGB image with light labels
            label_rgb = np.zeros((*self.mask.shape, 3), dtype=float)
            for i, label in enumerate(unique_labels):
                label_rgb[self.mask == label] = light_colors[i % len(light_colors)]

            # Convert the label image to float
            label_rgb = img_as_float(label_rgb)
            label_rgb[self.mask == 0] = [0, 0, 0]
            # Visualize the result
            fig, ax = plt.subplots(figsize=(12, 12))

            # Display the segmented image with light labels
            ax.imshow(mark_boundaries(label_rgb, segments, color=(0, 0, 1), mode='inner', outline_color=None))
            ax.set_title('Ground Truth with Training Samples Highlighted')

            # Overlay red dots for training samples
            train_y, train_x = np.where(train_indicator > 0)
            ax.scatter(train_x, train_y, c='red', s=10)  # Adjust 's' for dot size

            ax.axis('off')

            plt.tight_layout()
            file_path = 'slic_show.jpg'  # 替换为你的文件路径

            # 检查文件是否存在
            if os.path.exists(file_path):
                # 文件存在，删除文件
                os.remove(file_path)
                print(f"文件 '{file_path}' 已删除。")
            else:
                # 文件不存在
                print(f"文件 '{file_path}' 不存在。")
            plt.savefig('slic_show.jpg')
            # plt.show()

            # Update train_indicator, test_indicator, and labels based on segments
            updated_train_indicator, updated_test_indicator, updated_labels = update_indicators_and_mask(segments, train_indicator, test_indicator, self.mask)

            # Display original and updated train_indicator
            plt.figure(figsize=(15, 10))

            plt.subplot(3, 2, 1)
            plt.imshow(train_indicator, cmap='gray')
            plt.title('Original Train Indicator')
            plt.axis('off')

            plt.subplot(3, 2, 2)
            plt.imshow(updated_train_indicator, cmap='gray')
            plt.title('Updated Train Indicator')
            plt.axis('off')

            plt.subplot(3, 2, 3)
            plt.imshow(test_indicator, cmap='gray')
            plt.title('Original Test Indicator')
            plt.axis('off')

            plt.subplot(3, 2, 4)
            plt.imshow(updated_test_indicator, cmap='gray')
            plt.title('Updated Test Indicator')
            plt.axis('off')

            # Display original and updated labels
            plt.subplot(3, 2, 5)
            plt.imshow(self.mask, cmap='nipy_spectral')
            plt.title('Original Labels')
            plt.axis('off')

            plt.subplot(3, 2, 6)
            plt.imshow(updated_labels, cmap='nipy_spectral')
            plt.title('Updated Labels')
            plt.axis('off')
            file_path = 'slic_show1.jpg'  # 替换为你的文件路径

            # 检查文件是否存在
            if os.path.exists(file_path):
                # 文件存在，删除文件
                os.remove(file_path)
                print(f"文件 '{file_path}' 已删除。")
            else:
                # 文件不存在
                print(f"文件 '{file_path}' 不存在。")
            plt.savefig('slic_show1.jpg')
            plt.tight_layout()
            # plt.show()
            plt.close()

            blob = divisible_pad([np.concatenate([self.image.transpose(2, 0, 1),
                                              updated_labels[None, :, :],
                                              updated_train_indicator[None, :, :],
                                              updated_test_indicator[None, :, :],
                                              segments[None, :, :]], axis=0)], 16, False)
            
        else:
            # file_path = 'slic_show2.jpg'
            # # 检查文件是否存在
            # if os.path.exists(file_path):
            #     # 文件存在，删除文件
            #     os.remove(file_path)
            #     print(f"文件 '{file_path}' 已删除。")
            # else:
            #     # 文件不存在
            #     print(f"文件 '{file_path}' 不存在。")
            # plt.figure(figsize=(15, 10))
            # plt.imshow(self.mask, cmap='nipy_spectral')
            # plt.savefig('slic_show2.jpg')
            # plt.show()
            blob = divisible_pad([np.concatenate([self.image.transpose(2, 0, 1),
                                              self.mask[None, :, :],
                                              train_indicator[None, :, :],
                                              test_indicator[None, :, :],
                                              segments[None, :, :]], axis=0)], 16, False)
        
        im = blob[0, :self.image.shape[-1], :, :]

        mask = blob[0, -4, :, :]
        self.train_indicator = blob[0, -3, :, :]
        self.test_indicator = blob[0, -2, :, :]
        self.segments = blob[0, -1, :, :]

        if self.training:
            self.train_inds_list = minibatch_sample(mask, self.train_indicator, self.sub_minibatch,
                                                    seed=self.seeds_for_minibatchsample.pop())

        self.pad_im = im
        self.pad_mask = mask

    def resample_minibatch(self):
        self.train_inds_list = minibatch_sample(self.pad_mask, self.train_indicator, self.sub_minibatch,
                                                seed=self.seeds_for_minibatchsample.pop())

    @property
    def num_classes(self):
        return 9

    def __getitem__(self, idx):

        if self.training:
            return self.pad_im, self.pad_mask, self.train_inds_list[idx]
        else:
            return self.pad_im, self.pad_mask, self.test_indicator

    def __len__(self):
        if self.training:
            return len(self.train_inds_list)
        else:
            return 1


class MinibatchSampler(data.Sampler):
    def __init__(self, dataset: FullImageDataset):
        super(MinibatchSampler, self).__init__(None)
        self.dataset = dataset
        self.g = torch.Generator()
        self.g.manual_seed(SEED)

    def __iter__(self):
        self.dataset.resample_minibatch()
        n = len(self.dataset)
        return iter(torch.randperm(n, generator=self.g).tolist())

    def __len__(self):
        return len(self.dataset)

def intact_fixed_num_sample(gt_mask: np.ndarray,  num_train_samples, num_classes, seed=2333):
    """

    Args:
        gt_mask: 2-D array of shape [height, width]
        num_train_samples: int
        num_classes: scalar
        seed: int

    Returns:
        train_indicator, test_indicator
    """
    rs = np.random.RandomState(seed)

    gt_mask_flatten = gt_mask.ravel()
    train_indicator = np.zeros_like(gt_mask_flatten)
    test_indicator = np.zeros_like(gt_mask_flatten)
    # 如果上半部分没有下半部分的类怎么办？
    for i in range(1, num_classes + 1):

            # exit()
        inds = np.where(gt_mask_flatten == i)[0]
        rs.shuffle(inds)

        train_inds = inds[:num_train_samples] # 上半部分每类n样本
        test_inds = inds[num_train_samples:]

        train_indicator[train_inds] = 1
        test_indicator[test_inds] = 1

    train_indicator = train_indicator.reshape(gt_mask.shape)
    test_indicator = test_indicator.reshape(gt_mask.shape)

    return train_indicator, test_indicator

def fixed_num_sample(gt_mask: np.ndarray,  num_train_samples, num_classes, seed=2333):
    """

    Args:
        gt_mask: 2-D array of shape [height, width]
        num_train_samples: int
        num_classes: scalar
        seed: int

    Returns:
        train_indicator, test_indicator
    """
    rs = np.random.RandomState(seed)

    gt_mask_flatten = gt_mask.ravel()
    train_indicator = np.zeros_like(gt_mask_flatten)
    test_indicator = np.zeros_like(gt_mask_flatten)
    # 如果上半部分没有下半部分的类怎么办？
    for i in range(1, num_classes + 1):
        if np.where(gt_mask_flatten == i)[0].shape[0] == 0:
            print('一分为二的问题')
            # exit()
        inds = np.where(gt_mask_flatten == i)[0]
        rs.shuffle(inds)

        train_inds = inds[:num_train_samples] # 上半部分每类n样本
        test_inds = inds[num_train_samples:]

        train_indicator[train_inds] = 1
        test_indicator[test_inds] = 1

    train_indicator = train_indicator.reshape(gt_mask.shape)
    test_indicator = test_indicator.reshape(gt_mask.shape)

    return train_indicator, test_indicator


def minibatch_sample(gt_mask: np.ndarray, train_indicator: np.ndarray, minibatch_size, seed):
    """

    Args:
        gt_mask: 2-D array of shape [height, width]
        train_indicator: 2-D array of shape [height, width]
        minibatch_size:

    Returns:

    """
    rs = np.random.RandomState(seed) # 1434242557
    # split into N classes
    cls_list = np.unique(gt_mask)
    inds_dict_per_class = dict()
    for cls in cls_list:
        train_inds_per_class = np.where(gt_mask == cls, train_indicator, np.zeros_like(train_indicator))
        inds = np.where(train_inds_per_class.ravel() == 1)[0]
        rs.shuffle(inds)

        inds_dict_per_class[cls] = inds

    train_inds_list = []
    cnt = 0
    while True:
        train_inds = np.zeros_like(train_indicator).ravel()
        for cls, inds in inds_dict_per_class.items():
            left = cnt * minibatch_size
            if left >= len(inds):
                continue
            # remain last batch though the real size is smaller than minibatch_size
            right = min((cnt + 1) * minibatch_size, len(inds))
            fetch_inds = inds[left:right]
            train_inds[fetch_inds] = 1
        cnt += 1
        if train_inds.sum() == 0:
            return train_inds_list
        train_inds_list.append(train_inds.reshape(train_indicator.shape))
