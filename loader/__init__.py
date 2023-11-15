import torch
from torch.utils import data
import loader
from loader.toy_2d_dataset import ToySpline2d
from loader.Pouring_dataset import Pouring, simple_Pouring
from loader.latent_dataset import LatentDataset

def get_dataloader(data_dict, **kwargs):
    dataset = get_dataset(data_dict)
    loader = data.DataLoader(
        dataset,
        batch_size=data_dict["batch_size"],
        shuffle=data_dict.get("shuffle", True)
    )
    return loader

def get_latent_dataloader(data_dict, model, val_ratio=0., **kwargs):
    dataset_pre = get_dataset(data_dict, init_print=False)
    if val_ratio == 0:
        dataset_final = LatentDataset(dataset_pre, model, **data_dict)
        loader = data.DataLoader(
            dataset_final,
            batch_size=data_dict["batch_size"],
            shuffle=data_dict.get("shuffle", True)
        )
        return loader
    elif val_ratio > 0:
        dataset_val = get_dataset(data_dict, init_print=False)
        dataset_val.split = 'validation'
        len_dataset = dataset_pre.__len__()
        val_idx = torch.randperm(int(val_ratio * len_dataset))
        train_idx = torch.ones(len_dataset)
        train_idx[val_idx] -= 1
        train_idx = train_idx.to(torch.bool)
        dataset_pre.w_data = dataset_pre.w_data[train_idx]
        dataset_pre.traj_data = dataset_pre.traj_data[train_idx]
        dataset_val.w_data = dataset_val.w_data[val_idx]
        dataset_val.traj_data = dataset_val.traj_data[val_idx]
        dataset_final = LatentDataset(dataset_pre, model)
        loader = data.DataLoader(
            dataset_final,
            batch_size=data_dict["batch_size"],
            shuffle=data_dict.get("shuffle", True)
        )
        dataset_final_val = LatentDataset(dataset_val, model)
        loader_val = data.DataLoader(
            dataset_final_val,
            batch_size=data_dict["batch_size"],
            shuffle=data_dict.get("shuffle", True)
        )
        return loader, loader_val
        

def get_dataset(data_dict, **kwargs):
    name = data_dict["dataset"]
    dataset = globals()[name](**data_dict, **kwargs)
    return dataset