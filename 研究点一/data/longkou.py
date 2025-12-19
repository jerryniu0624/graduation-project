from scipy.io import loadmat
from simplecv.data import preprocess
from sklearn.preprocessing import MinMaxScaler
from data.base import FullImageDataset
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

SEED = 2333


class NewLongKouDataset(FullImageDataset):
    def __init__(self,
                 image_mat_path,
                 gt_mat_path,
                 training=True,
                 num_train_samples_per_class=200,
                 sub_minibatch=200,
                 np_seed=None,
                 slic=False,
                 n_segments = 100,
                 compactness = 0):
                 
        self.im_mat_path = image_mat_path
        self.gt_mat_path = gt_mat_path

        im_mat = loadmat(image_mat_path)
        image = im_mat['WHU_Hi_LongKou']
        gt_mat = loadmat(gt_mat_path)
        mask = gt_mat['WHU_Hi_LongKou_gt']

        self.vanilla_image = image
        shapeor = image.shape # (550, 400, 270)
        image = image.reshape(-1, image.shape[-1])
        # image = MinMaxScaler().fit_transform(image)
        # im_cmean = image.reshape((-1, image.shape[-1])).mean(axis=0)
        # im_cstd = image.reshape((-1, image.shape[-1])).std(axis=0)
        image = image.reshape(shapeor)
        # image = preprocess.mean_std_normalize(image, im_cmean, im_cstd)
        self.training = training
        self.num_train_samples_per_class = num_train_samples_per_class
        self.sub_minibatch = sub_minibatch
        super(NewLongKouDataset, self).__init__(image, mask, training,np_seed=np_seed,
                                                num_train_samples_per_class=num_train_samples_per_class,
                                                sub_minibatch=sub_minibatch, slic=slic, n_segments=n_segments, compactness=compactness)
        
    @property
    def num_classes(self):
        return len(np.unique(self.mask).tolist()) - 1
    
class NewLongKouCaveDataset(FullImageDataset):
    def __init__(self,
                 image_mat_path,
                 gt_mat_path,
                 training=True,
                 num_train_samples_per_class=200,
                 sub_minibatch=200,
                 np_seed=None,
                 slic=False,
                 n_segments = 100):
        self.im_mat_path = image_mat_path
        self.gt_mat_path = gt_mat_path

        im_mat = loadmat(image_mat_path)
        image = im_mat['WHU_Hi_LongKou_gt']
        gt_mat = loadmat(gt_mat_path)
        mask = gt_mat['WHU_Hi_LongKou_gt']

        self.vanilla_image = image
        shapeor = image.shape
        
    #     for i  in range(image.shape[2]):
    # # image = image.reshape(-1, image.shape[-1])
    # # image = MinMaxScaler().fit_transform(image)

    #         tmp = Image.fromarray(image[:,:,i])
    #         if tmp.mode == "F":
    #             tmp = tmp.convert('I')
    # # plt.imshow(kernel1[:,:,199])
    #         res = '/mnt/c/Users/17735/OneDrive/桌面/FreeNet-master/indian_show/' + str(i) + '.png'
    #         tmp.save(res)
            # plt.imshow(np.uint8(image[:,:,i]))
        image = image.reshape(-1, image.shape[-1])
        image = MinMaxScaler().fit_transform(image)
        # im_cmean = image.reshape((-1, image.shape[-1])).mean(axis=0)
        # im_cstd = image.reshape((-1, image.shape[-1])).std(axis=0)
        image = image.reshape(shapeor)

        # image = preprocess.mean_std_normalize(image, im_cmean, im_cstd)
        self.training = training
        self.num_train_samples_per_class = num_train_samples_per_class
        self.sub_minibatch = sub_minibatch
        super(NewLongKouCaveDataset, self).__init__(image, mask, training,np_seed=np_seed,
                                                num_train_samples_per_class=num_train_samples_per_class,
                                                sub_minibatch=sub_minibatch, slic=slic, n_segments=n_segments)
        
    @property
    def num_classes(self):
        return len(np.unique(self.mask).tolist()) - 1

