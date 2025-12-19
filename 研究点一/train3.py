import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from simplecv import dp_train as train
import torch
from simplecv.util.logger import eval_progress, speed
import time  
from module import freenet
from simplecv.util import metric
from data import dataloader
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sys

# SV freenet_1_0_salinas_3_E2E_BS 10 [ 13,  22,  25,  30,  32,  37,  65,  93, 104, 116, 120, 125, 129, 132, 144, 152, 157, 184, 186, 196]

OA_list = []
AA_list = []
KAPPA_list = []

def fcn_evaluate_fn(self, test_dataloader, config):
    if self.checkpoint.global_step < 0:
        return
    self._model.eval()
    total_time = 0.
    with torch.no_grad():
        for idx, (im, mask, w) in enumerate(test_dataloader):
            start = time.time()
            y_pred = self._model(im).squeeze(0)
            torch.cuda.synchronize()
            time_cost = round(time.time() - start, 3)
            y_pred = y_pred.argmax(dim=0).cpu() + 1
            y_res = y_pred
            w.unsqueeze_(dim=0)

            w = w.byte()
            mask = torch.masked_select(mask.view(-1), w.bool().view(-1))
            y_pred = torch.masked_select(y_pred.view(-1), w.bool().view(-1))

            oa = metric.th_overall_accuracy_score(mask.view(-1), y_pred.view(-1))
            aa, acc_per_class =metric.th_average_accuracy_score(mask.view(-1), y_pred.view(-1),
                                                                 len(np.unique(mask).tolist()),
                                                                 return_accuracys=True)
            kappa = metric.th_cohen_kappa_score(mask.view(-1), y_pred.view(-1), self._model.module.config.num_classes)
            total_time += time_cost
            speed(self._logger, time_cost, 'im')

            eval_progress(self._logger, idx + 1, len(test_dataloader))
            # print(oa.item())
            # print(aa.item())
            # print(kappa.item())

    speed(self._logger, round(total_time / len(test_dataloader), 3), 'batched im (avg)')

    metric_dict = {
        'OA': oa.item(),
        'AA': aa.item(),
        'Kappa': kappa.item()
    }
    if self._ckpt.global_step > 0 and self.max_all < oa.item() + aa.item() + kappa.item():
        self.max_all = oa.item() + aa.item() + kappa.item()
        self.max_oa = oa.item()
        self.max_aa = aa.item()
        self.max_kappa = kappa.item()
        self.max_idx = self._ckpt.global_step

    # if self._ckpt.global_step == 60:
    #     sys.exit()

    OA_list.append(oa.item())
    AA_list.append(aa.item())
    KAPPA_list.append(kappa.item())

    for i, acc in enumerate(acc_per_class):
        metric_dict['acc_{}'.format(i + 1)] = acc.item()
    self._logger.eval_log(metric_dict=metric_dict, step=self.checkpoint.global_step)
    if self.checkpoint.global_step == self._lr_schedule.max_iters:
        # print('ok')
        
        # plt.show(fig)
        # # plt.imshow(final_feat_embedded, cmap='hot',interpolation='nearest')
        # # plt.colorbar()
        # plt.show()
        # w_res = w_res.squeeze(0)
        # for i in range(y_res.shape[0]):
        #     for j in range(y_res.shape[1]):
        #         if w_res[0][i][j] == 0.0:
        #             y_res[i][j] = 0
        plt.subplot(1, 3, 1)
        plt.plot(OA_list)
        plt.subplot(1, 3, 2)
        plt.plot(AA_list)
        plt.subplot(1, 3, 3)
        plt.plot(KAPPA_list)
        plt.savefig('result3.jpg')
        plt.show()
        print('best_idx:' + str(self.max_idx) + '\n')
        print('best_oa:' + str(self.max_oa) + '\n')
        print('best_aa:' + str(self.max_aa) + '\n')
        print('best_kappa:' + str(self.max_kappa) + '\n')

        # plt.imshow(y_res, cmap='nipy_spectral')
        # # plt.colorbar()
        # # fig = plt.figure()
        # plt.axis('off')
        # # plt.savefig('./whlk_rfg-10_1.jpg',bbox_inches='tight', dpi=300, pad_inches=0.0)
        # plt.show()
        # print('ok')

def register_evaluate_fn(launcher):
    launcher.override_evaluate(fcn_evaluate_fn)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='PyTorch DCNNs Training')
    # parser.add_argument('--config_path', default='FreeNetNoResidual.FreeNetNoResidual_1_0_pavia', type=str, help='number of total epochs to run')
    # parser.add_argument('--model_dir', default='./log/pavia/FreeNetNoResidual/1.0_poly', type=str, help='dimensionality reduction')
    # parser.add_argument('--cpu', action='store_false', help='dataset (options: IP, UP, SV, KSC)')
    # parser.add_argument('--opts', default=1, type=float, help='samples of train set') # if tr_percent <1 : samples are %; else: samples are patches

    
    torch.backends.cudnn.benchmark = True
    args = train.parser.parse_args()
    SEED = 2222
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    train.run(config_path=args.config_path,
              model_dir=args.model_dir,
              cpu_mode=args.cpu,
              after_construct_launcher_callbacks=[register_evaluate_fn],
              opts=args.opts,
              np_seed=SEED)
