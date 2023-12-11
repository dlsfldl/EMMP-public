# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np
from utils.LieGroup_torch import *
from tqdm import tqdm
from models.groups import PouringGroup


class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "Overall Acc: \t": acc,
                "Mean Acc : \t": acc_cls,
                "FreqW Acc : \t": fwavacc,
                "Mean IoU : \t": mean_iu,
            },
            cls_iu
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class runningScore_cls(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)


        return {"Overall Acc : \t": acc,
                "Mean Acc : \t": acc_cls,}


    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class averageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def SE3_trajectory_distance(traj1, traj2):
    assert traj1.size()[0] == traj2.size()[0], f"Batch sizes are {traj1.size()[0]} and {traj2.size()[0]} which are not compatible."
    assert traj1.size()[1] == traj2.size()[1], f"Time steps of two trajectories are {traj1.size()[1]} and {traj2.size()[1]} which are not compatible."
    
    batch_size = traj1.size()[0]
    time_step = traj1.size()[1]
    
    R1 = traj1[:, :, :3, :3]; R2 = traj2[:, :, :3, :3]
    p1 = traj1[:, :, :3, 3]; p2 = traj2[:, :, :3, 3]
    
    R1 = R1.reshape(batch_size*time_step, 3, 3)
    R2 = R2.reshape(batch_size*time_step, 3, 3)
    p1 = p1.reshape(batch_size*time_step, 3)
    p2 = p2.reshape(batch_size*time_step, 3)
    
    SO3_dist = (skew_so3((logSO3(invSO3(R1)@R2)))**2).sum(dim=1)
    p_dist = ((p1 - p2)**2).sum(dim=1)
    
    dist = (SO3_dist + p_dist).reshape(batch_size, time_step).mean(dim=1)
    
    return dist

def Encoder_invariance(model, x, w, data_name, group_augment_size=100):
    '''Recommendation: x, w are unaugmented-data '''
    if data_name == 'Pouring':
        group = PouringGroup()
    
    batch_size = len(x)
    g = group._random_h(batch_size=group_augment_size)
    invariance = []
    
    for idx in tqdm(range(batch_size)):
        traj = x[idx].unsqueeze(0).repeat(group_augment_size, 1, 1, 1)
        task = w[idx].unsqueeze(0).repeat(group_augment_size, 1)
        task_aug, traj_aug = group.action_traj(g, task, traj)

        h = model.encode(traj, task)
        h_aug = model.encode(traj_aug, task_aug)
        invariance.append((((h - h_aug)**2).sum(dim=1).mean()).detach().item())

    invariance = torch.tensor(invariance).mean()

    return invariance

# def Decoder_equivariance(model, x, w, data_name, group_augment_size=100):
#         '''Recommendation: x, w are unaugmented-data '''
#     if data_name == 'Pouring':
#         group = PouringGroup()
        
#     batch_size = len9x)
    