import torch
import numpy as np
from sklearn.mixture import GaussianMixture
from loader.latent_dataset import LatentDataset

class RandpermInterpolation():
    def __init__(self, *args, **kwargs) -> None:
        pass
    def sample(z, *args, **kwargs):
        batch_size = len(z)
        perm_index = torch.randperm(batch_size)
        z_perm = z[perm_index]
        alpha = torch.rand(batch_size, 1).to(z)
        z_sampled = alpha * z_perm + (1-alpha) * z
        return z_sampled
    
def sample_randperm_interpolation(z, *args, **kwargs):
    batch_size = len(z)
    perm_index = torch.randperm(batch_size)
    z_perm = z[perm_index]
    alpha = torch.rand(batch_size, 1).to(z)
    z_sampled = alpha * z_perm + (1-alpha) * z
    return z_sampled

def get_optimized_GMM_latent_loaders(
    train_loader, val_loader, random_state=12,
    max_n_comp=40, **kwargs):
    z_train = train_loader.dataset.z_data
    z_val = val_loader.dataset.z_data
    n_comp_best = 0
    log_prob_max = -np.inf
    gmm_best = None
    for n_comp in range(1, max_n_comp):
        gmm = GaussianMixture(n_components=n_comp, random_state=random_state)
        gmm_labels = gmm.fit_predict(z_train)
        # log_prob_0 = gmm._estimate_log_prob_resp(z_train)[0].sum()
        log_prob = gmm._estimate_log_prob_resp(z_val)[0].sum()
        # print('log_prob')
        # print(log_prob)
        if log_prob > log_prob_max:
            n_comp_best = n_comp
            log_prob_max = log_prob
            gmm_best = gmm
        # print(f'n_comp = {n_comp}, log_prob = {log_prob}, {log_prob_0}')
    print(f'n_c= {n_comp_best}, l_p= {log_prob_max:.0f}, ', end='')
    return gmm_best, n_comp_best


## Old version(230315)
# def get_optimized_GMM(
#     model, data_loader, random_state=12,
#     max_n_comp=40, n_aug=2, val_ratio=0.2, 
#     val_loader=None, **kwargs):
#     w_data = data_loader.dataset.w_data
#     traj_data = data_loader.dataset.traj_data
    
#     w_data, traj_data = model.group._random_data_aug(
#         w_data, 
#         traj_data, 
#         n_aug=n_aug)
#     len_dataset = len(w_data)
#     val_idx = torch.randperm(int(val_ratio * len_dataset))
#     train_idx = torch.ones(len_dataset)
#     train_idx[val_idx] -= 1
#     train_idx = train_idx.to(torch.bool)
#     w_data_train = w_data[train_idx]
#     w_data_val = w_data[val_idx]
#     traj_data_train = traj_data[train_idx]
#     traj_data_val = traj_data[val_idx]
#     z_train = model.encode(
#             traj_data_train.to(model.device), 
#             w_data_train.to(model.device)).detach().cpu()
#     z_val = model.encode(
#             traj_data_val.to(model.device), 
#             w_data_val.to(model.device)).detach().cpu()
    
#     n_comp_best = 0
#     log_prob_max = -np.inf
#     gmm_best = None
#     for n_comp in range(1, max_n_comp):
#         gmm = GaussianMixture(n_components=n_comp, random_state=random_state)
#         gmm_labels = gmm.fit_predict(z_train)
#         log_prob = gmm._estimate_log_prob_resp(z_val)[0].sum()
#         # print('log_prob')
#         # print(log_prob)
#         if log_prob > log_prob_max:
#             n_comp_best = n_comp
#             log_prob_max = log_prob
#             gmm_best = gmm
#     print(f'n_c = {n_comp_best}, l_p = {log_prob_max:.0f}, ', end='')
#     return gmm_best, n_comp_best


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
    max_n_comp=30, n_aug=10, val_ratio=0.2, 
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

# def get_optimized_GMM(
#     model, train_loader, random_state=12,
#     val_loader=None, max_n_comp=30, n_aug=2, **kwargs):
    
#     w_data_train, traj_data_train = model.group._random_data_aug(
#         train_loader.dataset.w_data, 
#         train_loader.dataset.traj_data, 
#         n_aug=n_aug)
#     # print('w_data_train[:10]')
#     # print(w_data_train[:10])
    
#     w_data_val = val_loader.dataset.w_data
#     traj_data_val = val_loader.dataset.traj_data
    
    
#     z_train = model.encode(
#             traj_data_train.to(model.device), 
#             w_data_train.to(model.device)).detach().cpu()
#     z_val = model.encode(
#             traj_data_val.to(model.device), 
#             w_data_val.to(model.device)).detach().cpu()
    
#     # z_train = model.encode(
#     #         train_loader.dataset.traj_data.to(model.device), 
#     #         train_loader.dataset.w_data.to(model.device)).detach().cpu()
#     # z_val = model.encode(
#     #         val_loader.dataset.traj_data.to(model.device), 
#     #         val_loader.dataset.w_data.to(model.device)).detach().cpu()
    
#     n_comp_best = 0
#     log_prob_max = -np.inf
#     gmm_best = None
#     for n_comp in range(1, max_n_comp):
#         gmm = GaussianMixture(n_components=n_comp, random_state=random_state)
#         gmm_labels = gmm.fit_predict(z_train)
#         log_prob = gmm._estimate_log_prob_resp(z_val)[0].sum()
#         # print('log_prob')
#         # print(log_prob)
#         if log_prob > log_prob_max:
#             n_comp_best = n_comp
#             log_prob_max = log_prob
#             gmm_best = gmm
#     print('n_comp_best')
#     print(n_comp_best)
#     return gmm_best, n_comp_best

# def get_GMM(model, dataloader, n_components=None, random_state=12, n_aug=5, **kwargs):
#     if 'Equi' in str(type(model)):
#         n_aug = 0
#     w_aug, traj_aug = model.group._random_data_aug(
#         dataloader.dataset.w_data,
#         dataloader.dataset.traj_data, 
#         n_aug=n_aug, )
#     if n_components is not None:
#         gmm = GaussianMixture(n_components=n_components, random_state=random_state)
#         z = model.encode(
#             traj_aug.to(model.device),
#             w_aug.to(model.device)).detach().cpu()
#         gmm_labels = gmm.fit_predict(z)
#     else:
#         gmm, n_comp_best = get_optimized_GMM(
#             model, dataloader, random_state=random_state, 
#             n_aug=n_aug, **kwargs)
#     return gmm