import numpy as np
import glob
import pathlib
from scipy.io import loadmat
from scipy.io import savemat
import scipy.io as sio
import hdf5storage
import h5py
# import openpyxl
import collections
# from utils import same_seeds
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os
from jiancai_pingjie import SpiltHSI,PatchStack

#设置随机种子
# same_seeds()

dir = r'C:\Users\17735\Desktop\FreeNet-master'

#读取高光谱数据和标签
def read_raw_data(data_path, data_name):
    #path = pathlib.Path(f'./data_set/{data_path}/').absolute()
    path = dir 
    #print(path)
    #先保存为了.npy文件,在这里再读出来
    if data_name in ['Chikusei2131','Xiong_an2131','Houston11231']:
        temp_data = path + rf'{data_name}/{data_name}.mat'
        temp_gt = path + rf'{data_name}/{data_name}_gt.mat'
        #print(temp)
        data_dict = hdf5storage.loadmat(fr'{temp_data}')
        data_gt_dict = hdf5storage.loadmat(fr'{temp_gt}')
        #data_dict =  h5py.File(path / rf'{data_name}/{data_name}.mat')
        #data_gt_dict = h5py.File(path / rf'{data_name}/{data_name}_gt.mat')
        data_name = [t for t in list(data_dict.keys()) if not t.startswith('__')][0]
        data_gt_name = [t for t in list(data_gt_dict.keys()) if not t.startswith('__')][0]
        data = data_dict[data_name]
        data_gt = np.array(data_gt_dict[data_gt_name]).astype(np.int64)
          #D:\特征库项目\FSL_Benchmark\FSL_Benchmark\data_set\2_Dioni\Dioni\Dioni_gt
    else: #D:\特征库项目\FSL_Benchmark\FSL_Benchmark\data_set\2_Dioni\Dioni\Dioin_gt.mat
        print(path + rf'\{data_path}\{data_name}.mat')
        data_dict = loadmat(path + rf'\{data_path}\{data_name}.mat')
        if 'corrected' in data_name:
            data_name = data_name.split('_corrected')[0]
        data_gt_dict = loadmat(path + rf'\{data_path}\{data_name}_gt.mat')
        data_name = [t for t in list(data_dict.keys()) if not t.startswith('__')][0]
        data_gt_name = [t for t in list(data_gt_dict.keys()) if not t.startswith('__')][0]
        data = data_dict[data_name]
        data_gt = data_gt_dict[data_gt_name].astype(np.int64)
    print(data.shape)
    print(data_gt.shape)
    return data, data_gt


#函数拟合(在后面补)。用来上采样
def Function_Fitting(data,threshold=100):
    width, height, channel = data.shape
    zero_data = np.zeros((width, height, threshold))
    # 将HSI图像填进去
    zero_data[:, :, :channel] = data
    for i in range(width):
        for j in range(height):
            y = data[i, j, :]
            x = np.arange(0, channel)
            # 拟合。使用三次多项式拟合
            params = np.polyfit(x, y, 3)
            # print('查看参数',params)
            # 根据得到的参数构建拟合得到的函数
            param_func = np.poly1d(params)
            # print('查看得到的拟合函数',param_func)
            # 根据函数得到的想要的值。输入余下的像素生成像素值
            x_ = np.arange(channel, threshold)
            y_ = param_func(x_)
            zero_data[i, j, channel:threshold] = y_
            #print('插入后的数据',y_)
    #print('插值后数据的形状', zero_data.shape)
    return zero_data

#实现波长范围统一。功能：根据波长范围选择波段。仅限制波长，不会限制波段数量
def unif_wave_legth(length,end_length,data):
    '''
    :param legth:  length scale of input data
    :param end_length: top bound of length that you want
    :param data: HSI data_set
    :return: processed data
    '''
    #print(length)
    width, height, channel = data.shape
    legth_scale_0 = length[1] - length[0]
    legth_scale_1 = end_length - length[0]
    X = int((legth_scale_1  / legth_scale_0) * data.shape[2])
    print('X',X)
    #courrent_data = np.zeros((width, height,X))
    #remain_data = np.zeros((width, height,channel-X))
    if X <= data.shape[2]:
    #波长在1000nm内的波段
        courrent_data = data[:,:,:X]
        #波长超出1000的数据
        remain_data = data[:,:,X:]
        print(f'当前数据集在{end_length}纳米波长范围内的波段数量',X)
    else:
        '''
       # 波长在1000nm内的波段
        courrent_data = Function_Fitting(data,X)
        print('shape od courrent_data',courrent_data.shape)
        # 波长超出1000的数据
        remain_data = data[:,:,:2]
        # = courrent_data[:,:,100:]
        print(f'当前数据集在{end_length}纳米波长范围内的波段数量', X)
        #'''
        courrent_data = data
        remain_data = data[:,:,:2]
    return courrent_data,remain_data #波长在1000nm之前和1000之后的数据

#主成分分析，用来降维
def PCA_transform_img(img=None, n_principle=100):

    height = img.shape[0]
    width = img.shape[1]
    dim = img.shape[2]   #channel
    # reshape img, HORIZONTALLY strench the img, without changing the spectral dim.
    reshaped_img = img.reshape(height * width, dim)
    #进行主成分分析
    pca = PCA(n_components=n_principle) # 保留下来的特征个数n
    pca_img = pca.fit_transform(reshaped_img) # 用reshaped_img来训练PCA模型，同时返回降维后的数据
                  # shape (n_samples, n_features)

    # Regularization: Think about energy of each principles here.
    reg_img = pca_img * 1.0 / pca_img.max()
    # print(reg_img.shape)  (207400, 3)
    reg_img = reg_img.reshape(height,width,-1)
    print('reg_img.shape',reg_img.shape)
    return reg_img

#均匀采样。用来降维
def uniform_sampling(data=None,threshold=100):
    width, height, channel = data.shape
    index = np.linspace(0, channel - 1, threshold).astype(np.int64)
    # print(index)
    data = data[:, :, index]
    return data



#三次样条插值(在中间插)。用来上采样
def interpolate_1d(data,threshold=100):
    width, height, channel = data.shape
    zero_data = np.zeros((width, height, threshold))

    for i in range(width):
        for j in range(height):
            y = data[i, j, :]
            x = np.linspace(0,1,num=channel)
            # 拟合
            #线性插值
            f1 = interp1d(x , y , kind='cubic')
            #三次多项式插值
            #f1 = interp1d(x , y , kind= 'cubic' )
            # print('查看参数',params)
            #均匀的取点
            x_pred = np.linspace(0, 1, threshold)
            #用拟合的函数求出对应的函数值
            y1=f1(x_pred)
            # print('查看得到的拟合函数',param_func)
            # 将HSI图像填进去
            zero_data[i, j,:] = y1
            #print('插入后的数据',y1)
    #print('插值后数据的形状',zero_data.shape)
    return zero_data





#实现波段数量统一。设置一个波段数量的阈值，小于阈值补齐，多于阈值的采样
def unif_channel_num(data,threshold):

    #根据路径读取数据
    print('查看数据形状', data.shape)
    width,height,channel = data.shape
    #print(width,height,channel)
    #判断波段数量,根据波段数量操作
    #波段数量大于阈值
    if channel >= threshold:
        #采用均匀采样降维
        data = uniform_sampling(data, threshold)
        #print('波段数量对齐的方式为均匀采样')
        #采用PCA降维
        #data = PCA_transform_img(data,threshold)
        print('波段数量对齐的方式为PCA')
    #波段数量小于阈值
    else:
        #在中间插值
        data = interpolate_1d(data,threshold)
        print('波段数量对齐的方式为插值')
        #在末尾补齐
        #data = Function_Fitting(data,threshold)
    print('查看数据形状',data.shape)
    return data
"/home/zxb2/code/FSCF_SSL_2023_main/FSCF_SSL_2023_main/datasets/dataset/1_Washington_DC_Mall/1_Washington_DC_Mall/Washington_DC_Mall/Washington_DC_Mall.mat"
'/home/zxb2/code/FSCF_SSL_2023_main/FSCF_SSL_2023_main/datasets/data_set/1_Washington_DC_Mall/1_Washington_DC_Mall/Washington_DC_Mall/Washington_DC_Mall.mat'
######################################### 处理训练数据 #########################################################

# train_data_set = {'1_Washington_DC_Mall':'Washington_DC_Mall','4_Xiong_an':'Xiong_an','5_Xuzhou':'Xuzhou',
#             '7_WHU_Hi_HongHu':'WHU_Hi_HongHu','8_WHU_Hi_LongKou':'WHU_Hi_LongKou', '9_Shanghai':'Shanghai',
#                  '11_Botswana':'Botswana','12_Chikusei':'Chikusei','13_KSC':'KSC','14_Houston_2013':'Houston'
#                  ,'Salinas_corrected':'Salinas_corrected','PaviaU':'PaviaU','Trento':'Trento'} #
# train_data_set = {
#                   '11_Botswana':'Botswana','12_Chikusei':'Chikusei','13_KSC':'KSC','14_Houston_2013':'Houston'
#                   ,'Salinas_corrected':'Salinas_corrected','PaviaU':'PaviaU'} #
# train_data_set = {
#                 '7_WHU_Hi_HongHu':'WHU_Hi_HongHu'} #
train_data_set =  {'Houston13':'Houston13'}
{'indian':'indian_pines_corrected','pavia':'PaviaU','paviac':'Pavia',}
{'indian':'indian_pines_corrected','pavia':'PaviaU','paviac':'Pavia',
            'salinas':'Salinas_corrected','washington':'Washington_DC_Mall', 'botwana':'Botswana',
                 'AB':'Augsburg','BL':'Berlin','houston2018':'Houston'}
{
            'salinas':'Salinas_corrected','washington':'Washington_DC_Mall', 'botwana':'Botswana',
                 'houston2018':'Houston'}





                 #'3_Loukia':'Loukia','11_Botswana':'Botswana','12_Chikusei':'Chikusei','13_KSC':'KSC','14_Houston_2013':'Houston','2_Dioni':'Dioni','3_Loukia':'Loukia',
            #'14_Houston_2013':'Houston','15_Houston_2018':'Houston'}#'9_Shanghai':'Shanghai', '10_Hangzhou':'Hangzhou','2_Dioni':'Dioni','3_Loukia':'Loukia',,'15_Houston_2018':'Houston'
                                                        #,'9_Shanghai':'Shanghai','11_Botswana':'Botswana''13_KSC':'KSC',

#记录波长范围的字典。键为数据集的名字，键对应的列表中记录的是每个数据集数据的波长范围，单位为纳米
length_scale_train = {'indian':[400, 2500],'pavia':[430, 860],'paviac':[430, 860],
            'salinas':[400, 2500],'washington':[400, 2400], 'botwana':[400, 2500],
                 'AB':[400, 2500],'BL':[400, 2500],'houston2018':[380, 1050]}
            #'14_Houston_2013':[380,1050],'15_Houston_2018':[380,1000]}#'9_Shanghai':[356,2577], '10_Hangzhou':[356,2577] #'2_Dioni':[400,2500],'3_Loukia':[400,2500],,'15_Houston_2018':[380,1050]
                                                                #,'9_Shanghai':[356,257 7],'11_Botswana':[400,2500],'13_KSC':[400,2500],
#与目标域类别相同的源域数据的类别(为了保证测试数据没有见过训练数据)
classes_same_with_target = {'indian':[4],'pavia':[4,6],'paviac':[1],
            'salinas':[],'washington':[1,3,5], 'botwana':[1],
                 'AB':[7],'BL':[8],'houston2018':[]}

{'1_Washington_DC_Mall':[3,5,6],'4_Xiong_an':[],'5_Xuzhou':[6],
            '7_WHU_Hi_HongHu':[1,2,22],'8_WHU_Hi_LongKou':[1,7],'9_Shanghai':[1],
            '11_Botswana':[1],'12_Chikusei':[1,8],'13_KSC':[13], '14_Houston_2013':[5,6,9],
            'Salinas_corrected':[],'PaviaU':[4,6],'Pavia':[1,2,3,4,5,6,7,8,9],'Trento':[4,6],
            'MUUFL':[1],'AeroRIT':[1,5]}#'3_Loukia':[13],'15_Houston_2018':[8],'9_Shanghai',:[1],'11_Botswana':[1],'13_KSC':[13],'''

#源域数据中有很多重复的类别，这有可能会影响模型训练，测试之前删除这些重复的类别，只保留一个
classes_same_in_source = {'1_Washington_DC_Mall':[5,6,7],'2_Dioni':[1,2,3,4,5,7,8,9,10,11,12,13,14],
                          '3_Loukia':[],
         '4_Xiong_an':[],'5_Xuzhou':[ ],'6_WHU_Hi_HanChuan':[9,14],
            '7_WHU_Hi_HongHu':[2],'8_WHU_Hi_LongKou':[2],'9_Shanghai':[1],'9_Shanghai':[2,3],'10_Hangzhou':[],
                          '11_Botswana':[1],'12_Chikusei':[],'13_KSC':[],
            '14_Houston_2013':[4,5,6],'15_Houston_2018':[1,2,3,9,11,14,15]} #,'15_Houston_2018':[1,2,3,9,11,14,15],'9_Shanghai':[1],'11_Botswana':[1],'13_KSC':[13],


#每个数据集中的近似类别
jinsi_same_dataset = {'15_Houston_2018':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}
'''{'1_Washington_DC_Mall':[],'2_Dioni':[],
                          '3_Loukia':[],
         '4_Xiong_an':[],'5_Xuzhou':[1,7,5,8 ],'6_WHU_Hi_HanChuan':[],
            '7_WHU_Hi_HongHu':[8,11,12,13],'8_WHU_Hi_LongKou':[],'9_Shanghai':[],'9_Shanghai':[],'10_Hangzhou':[],
                          '11_Botswana':[3,4],'12_Chikusei':[2,3],'13_KSC':[],
            '14_Houston_2013':[],'15_Houston_2018':[]}'''

#相同传感器，相同类别
same_sensor_same_class = {'Dense urban fabric':{'Dioni':[1],'Loukia':[1]},
                    'Water':{'Shanghai':[1],'Hangzhou':[1]},
                    'Cotton':{'WHU_Hi_HongHu':[4],'WHU_Hi_LongKou':[2]},
                          'Road': {'WHU_Hi_HanChuan': [14],'WHU_Hi_HongHu': [2]}}

#不同传感器，相同类别
diff_sensor_same_class = {'Grass':{'Washington_DC_Mall':[4],'Chikusei':[8]},
                    'Water':{'Shanghai':[1],'Hangzhou':[1],'Botswana':[1],'KSC':[13]},
                    'Rice':{'Xiong_an':[10],'Chikusei':[9]},
                          'Road': {'Washington_DC_Mall':[3],'WHU_Hi_HanChuan': [14],'WHU_Hi_HongHu': [2],'Houston':[9]}}



''''Dense urban fabric':{'Dioni':[1],'Loukia':[1]},
                    'Water':{'Shanghai':[1],'Hangzhou':[1]},
                    'Cotton':{'WHU_Hi_HongHu':[4],'WHU_Hi_LongKou':[2]},'''



def train_data_process(data_set,Threshold=100,shot = 5,edge = 10):
    print('the train data_set is processing ...............................')
    #记录每个数据集的类别的数量
    total_class = [0] * len(data_set.keys())
    total_class_num = 0
    data_all = []
    label_all = []
    label_merge_all = []
    # train_mask_all = []
    # test_mask_all = []
    for index,keys in enumerate(data_set.keys()):
        print(f'the {index+1}_th data_set is processing ............................................')
        data_set_name = keys
        print(keys,data_set[keys])
        #读取数据
        data_0, gt_0 = read_raw_data(keys,data_set[keys])
        #统一波长范围
        #print('查看列表',length_scale[key])。1000nm之前和之后的数据
        current_data,remain_data = unif_wave_legth(length_scale_train[keys], 1000, data_0)
        print(current_data.shape,remain_data.shape)
        #统一波段数量

        #处理1000nm之前的数据
        #current_data = data_0
        print('shape of current_data',current_data.shape)
        threshold = Threshold
        #img = data
        img = unif_channel_num(current_data,threshold)
        img = (img * 1.0 - img.min()) / (img.max() - img.min())

        # 第一种标签.合并所有类别之后的标签，在切片之前的整图上做
        h, w = gt_0.shape
        new_label_all = np.zeros_like(gt_0)
        new_label_all = new_label_all.reshape(-1)
        gt_0_ = gt_0.reshape(-1)

        for i in np.unique(gt_0_):
            if i == 0 or i in classes_same_with_target[keys]:
                continue
            total_class_num += 1  # 统计出现过的类别
            index = np.where(gt_0_== i)[0]
            new_label_all[index] =  total_class_num
        new_label_all = new_label_all.reshape(h, w,1)
        print('合并之后的标签',np.unique(new_label_all))
        #--------------切片
        target_size = [512, 512]#没有计算边缘重复像素时的patch_size
        EDGE = edge#边缘的重复像素
        #对图像切片
        patchfied_data, padded_data_size = SpiltHSI(img, target_size, EDGE,gt = None) #返回切片的列表，填充之后图像的真实尺寸（拼接图像的时候有用）
        #对标签切片（合并所有类别之后的）
        patchfied_gt_all, padded_gt_size_all = SpiltHSI(new_label_all, target_size, EDGE,gt = True)  # 返回切片的列表，填充之后图像的真实尺寸（拼接图像的时候有用）
        # 对标签切片（原始标签）
        gt_0 =  np.expand_dims(gt_0,axis=-1)
        patchfied_gt, padded_gt_size = SpiltHSI(gt_0, target_size, EDGE,gt = True)  # 返回切片的列表，填充之后图像的真实尺寸（拼接图像的时候有用）

        ##----第二种标签，将每个子图看做独立图像；在切片之后的图像上做
        patchfied_gt_new = []
        train_mask_list = []
        test_mask_list = []
        for i in range(len(patchfied_gt)):
            label = patchfied_gt[i]

            h, w = label.shape
            new_label = np.zeros_like(label)
            train_mask = np.zeros_like(label)
            test_mask = np.zeros_like(label)
            train_mask = train_mask.reshape(-1)
            test_mask = test_mask.reshape(-1)

            new_label = new_label.reshape(-1)
            label = label.reshape(-1)
            for ind, i in enumerate(np.unique(label)):
                if i == 0:
                    continue

                index = np.where(label == i)[0]
                new_label[index] = ind

                #制作标签掩码，标记训练样本和测试样本
                # np.random.shuffle(index)
                # train_index =  index[:shot]# 5_shot,每类5个小样本
                # test_index = index[shot:]
                # train_mask[train_index] = 1
                # test_mask[test_index] = 1

            new_label = new_label.reshape(h, w)
            # train_mask = train_mask.reshape(h, w)
            # test_mask = test_mask.reshape(h, w)

            patchfied_gt_new.append(new_label)
            # train_mask_list.append(train_mask)
            # test_mask_list.append(test_mask)

        # #解析数据
        for i in range(len(patchfied_data)):
            if len(np.unique(patchfied_gt_new[i])) <= 5:#过滤掉美术数据的子图
                print('okokokok')
                continue
            
            data_all.append(patchfied_data[i])
            # plt.imshow(patchfied_data[i][:,:,20])
            # plt.title('data ' + keys + ' ' + str(i))
            # plt.savefig(r'C:\Users\17735\OneDrive\桌面\FreeNet-master\show'+'\data ' + keys + ' ' + str(i) + '.jpg')
            # # plt.show()
            # print('ok')
            label_all.append(patchfied_gt_new[i])
                # plt.imshow(patchfied_gt[i])
                # plt.title('label')
                # plt.show()
            label_merge_all.append(patchfied_gt_all[i])
            # plt.imshow(patchfied_gt_all[i])
            # plt.title('label_all ' + keys + ' ' + str(i))
            # plt.savefig(r'C:\Users\17735\OneDrive\桌面\FreeNet-master\show'+ '\label_all ' + keys + ' ' + str(i) + '.jpg')
            # # plt.show()
            # print('ok')
                # train_mask_all.append(train_mask_list[i])
                # test_mask_all.append(test_mask_list[i])
                # plt.show()

    #保存数据
    save_path = r'C:\Users\17735\OneDrive\桌面\FreeNet-master\merged_dataset1\train'
    # print('查看文件保存路径',save_path )
    '''
    注意：为了以后使用方便，希望保存处理好的数据，然而将每个处理好数据集单独保存为.mat文件没有意义，因为每个数据集的样本数量、类别数量都可能发生了改变，
    所以保存每个文件为.h5文件。使用的时候读取.h5文件即可。
    '''
    f = h5py.File(
        save_path + '\\'+ f'band_{Threshold}_patch_size_{target_size}_total_class_num_{total_class_num}_{shot}_shot_{edge}_edge.h5',
        'w')
    f['data'] = data_all
    f['gt'] = label_all
    # f['gt'] = label_merge_all
    # f['train_mask'] = train_mask_all
    # f['test_mask'] = test_mask_all
    f.close()


train_data_process(train_data_set,Threshold=100,shot = 20,edge = 0)#第三个参数为波段数量的阈值,#第二个参数为每个类别样本的数量阈值，
######################################### 处理训练数据 #########################################################





######################################### 处理测试数据 #########################################################
'''
处理的方式和训练数据的处理方式大致相同，不同之处在于不对样本数量小于 200的类别进行处理;目标域需要保留所有样本用于测试，但是源域只需要保留部分样本。
xiong_an数据集因为太大，用作测试集时无法处理 
'''

#test_data_set = {'Indian_pines_corrected':'Indian_pines_corrected'}
# test_data_set = {'Indian_pines_corrected':'Indian_pines_corrected','PaviaU':'PaviaU','Salinas_corrected':'Salinas_corrected'}
test_data_set = {'WHU-Hi-HanChuan':'WHU_Hi_HanChuan',
                 'WHU-Hi-HongHu':'WHU_Hi_HongHu',
                 'WHU-Hi-LongKou':'WHU_Hi_LongKou'}
# test_data_set = {'Indian_pines_corrected':'Indian_pines_corrected','3_Loukia':'Loukia','6_WHU_Hi_HanChuan':'WHU_Hi_HanChuan','17_Berlin':'Berlin','15_Houston_2018':'Houston'}
#'Salinas_corrected':[],'Pavia':[],'PaviaU':[],'Indian_pines_corrected':[], ,'Salinas_corrected':'Salinas_corrected','PaviaU':'PaviaU','Indian_pines_corrected':'Indian_pines_corrected'}
length_scale_test = {'WHU-Hi-HanChuan':[400, 1000],
                 'WHU-Hi-HongHu':[400, 1000],
                 'WHU-Hi-LongKou':[400, 1000]}

#目标数据切片
# def test_data_process(data_set,Threshold=100,shot = 5,edge = 10):
#     print('the train data_set is processing ...............................')
#     #记录每个数据集的类别的数量
#     total_class = [0] * len(data_set.keys())
#     total_class_num = 0
#     data_all = []
#     label_all = []
#     label_merge_all = []
#     train_mask_all = []
#     test_mask_all = []
#     for index,keys in enumerate(data_set.keys()):
#         print(f'the {index+1}_th data_set is processing ............................................')
#         data_set_name = keys
#         print(keys,data_set[keys])
#         #读取数据
#         data_0, gt_0 = read_raw_data(keys,data_set[keys])
#         #统一波长范围
#         #print('查看列表',length_scale[key])。1000nm之前和之后的数据
#         current_data,remain_data = unif_wave_legth(length_scale_test[keys],1000,data_0)
#         print(current_data.shape,remain_data.shape)
#         #统一波段数量
#
#         #处理1000nm之前的数据
#         #current_data = data_0
#         print('shape of current_data',current_data.shape)
#         threshold = Threshold
#         #img = data
#         img = unif_channel_num(current_data,threshold)
#         img = (img * 1.0 - img.min()) / (img.max() - img.min())
#
#         # 第一种标签.合并所有类别之后的标签，在切片之前的整图上做
#         h, w = gt_0.shape
#         new_label_all = np.zeros_like(gt_0)
#         new_label_all = new_label_all.reshape(-1)
#         gt_0_ = gt_0.reshape(-1)
#
#         for i in np.unique(gt_0_):
#             if i == 0:
#                 continue
#             total_class_num += 1  # 统计出现过的类别
#             index = np.where(gt_0_== i)[0]
#             new_label_all[index] =  total_class_num
#         new_label_all = new_label_all.reshape(h, w,1)
#         print('合并之后的标签',np.unique(new_label_all))
#         #--------------切片
#         target_size = [128, 128]#没有计算边缘重复像素时的patch_size
#         EDGE = edge#边缘的重复像素
#         #对图像切片
#         patchfied_data, padded_data_size = SpiltHSI(img, target_size, EDGE,gt = None) #返回切片的列表，填充之后图像的真实尺寸（拼接图像的时候有用）
#         #对标签切片（合并所有类别之后的）
#         patchfied_gt_all, padded_gt_size_all = SpiltHSI(new_label_all, target_size, EDGE,gt = True)  # 返回切片的列表，填充之后图像的真实尺寸（拼接图像的时候有用）
#         # 对标签切片（原始标签）
#         gt_0 =  np.expand_dims(gt_0,axis=-1)
#         patchfied_gt, padded_gt_size = SpiltHSI(gt_0, target_size, EDGE,gt = True)  # 返回切片的列表，填充之后图像的真实尺寸（拼接图像的时候有用）
#
#         #----第二种标签，将每个子图看做独立图像；在切片之后的图像上做
#         patchfied_gt_new = []
#         train_mask_list = []
#         test_mask_list = []
#         reference = {}
#         for i in range(len(patchfied_gt)):
#             print('i',i)
#             label = patchfied_gt[i]
#
#             h, w = label.shape
#             new_label = np.zeros_like(label)
#             train_mask = np.zeros_like(label)
#             test_mask = np.zeros_like(label)
#             train_mask = train_mask.reshape(-1)
#             test_mask = test_mask.reshape(-1)
#
#             new_label = new_label.reshape(-1)
#             label = label.reshape(-1)
#
#             reference_patch = []
#             for ind,i in enumerate(np.unique(label)):
#                 if i == 0:
#                     continue
#
#                 index = np.where(label == i)[0]
#                 new_label[index] = ind  #新赋予的类别
#                 reference_patch.append(np.float16(i)) #在原始GT中的类别
#                 #制作标签掩码，标记训练样本和测试样本
#                 np.random.shuffle(index)
#                 train_index =  index[:shot]# 5_shot,每类5个小样本
#                 test_index = index[shot:]
#                 train_mask[train_index] = 1
#                 test_mask[test_index] = 1
#
#             new_label = new_label.reshape(h, w)
#             train_mask = train_mask.reshape(h, w)
#             test_mask = test_mask.reshape(h, w)
#
#             patchfied_gt_new.append(new_label)
#             train_mask_list.append(train_mask)
#             test_mask_list.append(test_mask)
#             reference[f'patch_{i}'] = reference_patch
#
#         #解析数据
#         for i in range(len(patchfied_data)):
#             data_all.append(patchfied_data[i])
#             # plt.imshow(patchfied_data[i][:,:,20])
#             # plt.title('data')
#             # plt.show()
#             label_all.append(patchfied_gt_new[i])
#             # plt.imshow(patchfied_gt[i])
#             # plt.title('label')
#             # plt.show()
#             label_merge_all.append(patchfied_gt_all[i])
#             # plt.imshow(patchfied_gt_all[i])
#             # plt.title('label_all')
#             train_mask_all.append(train_mask_list[i])
#             test_mask_all.append(test_mask_list[i])
#             # plt.show()
#
#         #保存数据
#         save_path = r'/home/zxb2/code/MAML_HSI/data/'
#         # print('查看文件保存路径',save_path )
#         '''
#         注意：为了以后使用方便，希望保存处理好的数据，然而将每个处理好数据集单独保存为.mat文件没有意义，因为每个数据集的样本数量、类别数量都可能发生了改变，
#         所以保存每个文件为.h5文件。使用的时候读取.h5文件即可。
#         '''
#         f = h5py.File(
#             save_path + '/'+ 'target_domain' +  '/'  + f'{keys}' + f'total_class_num_{total_class_num}_{shot}_shot_{edge}_edge.h5',
#             'w')
#         f['data'] = data_all
#         f['label'] = label_all
#         f['label_merge'] = label_merge_all
#         f['train_mask'] = train_mask_all
#         f['test_mask'] = test_mask_all
#         # print(reference)
#         # #f['reference'] = reference
#         # # 创建一个组
#         # group = f.create_group("reference")
#         #
#         # # 创建一个字典数据集
#         # #dictionary = {"key1": "value1", "key2": "value2", "key3": "value3"}
#         # group.create_dataset("dictionary", data=reference)
#
#         f.close()
# #test_data_process(test_data_set,Threshold=100,shot = 5,edge = 10)#第三个参数为波段数量的阈值,#第二个参数为每个类别样本的数量阈值，
######################################### 处理训练数据 #########################################################

def test_data_process(data_set,Threshold=100,shot = 5,edge = 10):
    print('the train data_set is processing ...............................')
    #记录每个数据集的类别的数量
    total_class = [0] * len(data_set.keys())
    total_class_num = 0
    data_all = []
    label_all = []
    label_merge_all = []
    train_mask_all = []
    test_mask_all = []
    for index,keys in enumerate(data_set.keys()):
        print(f'the {index+1}_th data_set is processing ............................................')
        data_set_name = keys
        print(keys,data_set[keys])
        #读取数据
        data_0, gt_0 = read_raw_data(keys,data_set[keys])

        if keys in ['17_Berlin','15_Houston_2018']: #截取部分图像，并重新制作标签
            data_0 = data_0[:, :800, :]
            gt_0 = gt_0[:, :800]
            print(data_0.shape, gt_0.shape)
            print(np.unique(gt_0))
            h, w = gt_0.shape
            print(np.unique(gt_0))
            new_gt = np.zeros(gt_0.shape)
            gt_0 = gt_0.reshape(-1)
            new_gt = new_gt.reshape(-1)
            for ind, i in enumerate(np.unique(gt_0)):
                if i == 0:
                    continue
                index = np.where(gt_0 == i)[0]
                new_gt[index] = ind  # 新赋予的类别
            gt_0 = new_gt.reshape(h, w)
            print(np.unique(gt_0))


        #统一波长范围
        #print('查看列表',length_scale[key])。1000nm之前和之后的数据
        current_data,remain_data = unif_wave_legth(length_scale_test[keys],1000,data_0)
        print(current_data.shape,remain_data.shape)
        #统一波段数量

        #处理1000nm之前的数据
        #current_data = data_0
        print('shape of current_data',current_data.shape)
        threshold = Threshold
        #img = data
        img = unif_channel_num(current_data,threshold)
        img = (img * 1.0 - img.min()) / (img.max() - img.min())

        # 第一种标签.合并所有类别之后的标签，在切片之前的整图上做
        h, w = gt_0.shape
        gt_0_ = gt_0.reshape(-1)

        train_mask = np.zeros_like(gt_0)
        test_mask = np.zeros_like(gt_0)
        train_mask = train_mask.reshape(-1)
        test_mask = test_mask.reshape(-1)

        for i in np.unique(gt_0_):
            if i == 0:
                continue
            index = np.where(gt_0_== i)[0]
            np.random.shuffle(index)
            train_index = index[:shot]  # 5_shot,每类5个小样本
            test_index = index[shot:]
            train_mask[train_index] = 1
            test_mask[test_index] = 1

        train_mask = train_mask.reshape(h, w)
        test_mask = test_mask.reshape(h, w)

        print('合并之后的标签')


        #解析数据
        #保存为三维数组有问题，转化为2维数组保存
        #img = img.reshape(h * w, Threshold)
        print(img.shape)
        data_all.append(np.array(img))
        # plt.imshow(patchfied_data[i][:,:,20])
        # plt.title('data')
        # plt.show()
        label_all.append(gt_0)
        plt.imshow(gt_0)
        plt.title('label')
        plt.show()
        train_mask_all.append(train_mask)
        plt.imshow(train_mask)
        plt.title('train_mask')
        plt.show()
        test_mask_all.append(test_mask)
        plt.imshow(test_mask)
        plt.title('test_mask')
        plt.show()
            # plt.show()

        #保存数据
        save_path = r'C:\Users\17735\OneDrive\桌面\FreeNet-master\merged_dataset\test'# /home/zxb2/code/MAML_HSI/data/target_domain/
        # print('查看文件保存路径',save_path )
        '''
        注意：为了以后使用方便，希望保存处理好的数据，然而将每个处理好数据集单独保存为.mat文件没有意义，因为每个数据集的样本数量、类别数量都可能发生了改变，
        所以保存每个文件为.h5文件。使用的时候读取.h5文件即可。
        '''

        file_name = save_path +  '\\'  + f'{keys}'+ f'{Threshold}'+ f'_total_class_num_{total_class_num}_{shot}_shot_{edge}_edge.mat'
        savemat(file_name, {'data':img, 'gt': gt_0})

# test_data_process(test_data_set,Threshold=100,shot = 5,edge = 10)#第三个参数为波段数量的阈值,#第二个参数为每个类别样本的数量阈值，