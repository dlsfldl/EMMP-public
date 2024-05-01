import torch
import numpy as np
from sklearn.mixture import GaussianMixture
from loader.latent_dataset import LatentDataset


def split_data(data, split_ratio):
    len_data = len(data)
    val_idx = torch.randperm(int(split_ratio * len_data))
    train_idx = torch.ones(len_data)
    train_idx[val_idx] -= 1
    train_idx = train_idx.to(torch.bool)
    data_val = data[val_idx]
    data_train = data[train_idx]
    return data_train, data_val
    

def get_optimized_GMM(
    data_train, data_val=None, 
    split_ratio=0.2, 
    random_state=12, 
    max_n_comp=30, 
    print_prob=True,
    **kwargs):
    if data_val is None:
        data_train, data_val = split_data(data_train, split_ratio)
    else:
        pass
    n_comp_best = 0
    log_prob_max = -np.inf
    gmm_best = None
    for n_comp in range(1, max_n_comp):
        gmm = GaussianMixture(n_components=n_comp, random_state=random_state)
        gmm_labels = gmm.fit_predict(data_train)
        log_prob = gmm._estimate_log_prob_resp(data_val)[0].sum()
        if log_prob > log_prob_max:
            n_comp_best = n_comp
            log_prob_max = log_prob
            gmm_best = gmm
    if print_prob:
        print(f'n_c = {n_comp_best:2d}, l_p = {int(log_prob_max):5d}, ', end='')
    return gmm_best, n_comp_best

def get_optimized_GMM_latent(
    model, data_loader, random_state=12,
    max_n_comp=30, n_aug=None, val_ratio=0.2, 
    val_loader=None, print_prob=True, **kwargs):
    latent_dataset_train = LatentDataset(data_loader.dataset, model, n_aug=n_aug)
    z_train = latent_dataset_train.z_data
    if val_loader is None:
        z_val = None
    else:
        latent_dataset_val = LatentDataset(val_loader.dataset, model, n_aug=n_aug)
        z_val = latent_dataset_val.z_data
    gmm_best, n_comp_best = get_optimized_GMM(
        data_train=z_train, data_val=z_val,
        split_ratio=val_ratio, 
        random_state=random_state, 
        max_n_comp=max_n_comp, print_prob=print_prob, **kwargs)
    return gmm_best, n_comp_best

def get_GMM(
    model, dataloader, 
    val_loader=None, n_components=None, 
    random_state=12, n_aug=10, **kwargs):
    if 'Equi' in str(type(model)):
        n_aug = 0
    w_aug, traj_aug = model.group._random_data_aug(
        dataloader.dataset.w_data,
        dataloader.dataset.traj_data, 
        n_aug=n_aug, )
    if n_components is not None:
        gmm = GaussianMixture(n_components=n_components, random_state=random_state)
        z = model.encode(
            traj_aug.to(model.device),
            w_aug.to(model.device)).detach().cpu()
        gmm_labels = gmm.fit_predict(z)
    else:
        gmm, n_comp_best = get_optimized_GMM_latent(
            model, dataloader, random_state=random_state, 
            n_aug=n_aug, val_loader=val_loader, **kwargs)
    return gmm