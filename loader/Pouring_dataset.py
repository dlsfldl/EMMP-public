import numpy as np
import torch
from torch.utils.data import Dataset
import os
from tqdm import tqdm

import sys
sys.path.append('../')
# from models.groups import PouringGroup
from models.groups import PouringGroup
group = PouringGroup()


class Pouring(Dataset):
    def __init__(self, 
                split='training', 
                root='datasets',
                augmentation=False,
                skip_size=1,
                **kwargs):
        self.group = PouringGroup()
        self.split = split
        self.augmentation = augmentation
        self.traj_data = []
        self.w_data = [] 
        self.m = []
        self.num_timestep = 480
        self.skip_size = skip_size
        # np.random.seed(0)
        # torch.manual_seed(0)
        self.base = os.path.join(root, 'Pouring')
        folder_path = os.path.join(self.base, split)
        file_list = os.listdir(folder_path)
        file_list.sort()
        for file_name in file_list:
            file_path = folder_path + '/' + file_name
            tau, traj, m = np.load(file_path, allow_pickle=True)
            if len(traj.shape) == 2:
                assert(int(traj.shape[-1]/12) == self.num_timestep)
            else:
                assert(traj.shape[1] == self.num_timestep)
                traj = traj[:, ::self.skip_size]
            self.traj_data.append(traj)
            self.w_data.append(tau)
            self.m.extend(m)
        self.traj_data = torch.cat(self.traj_data, dim=0)
        if len(self.traj_data.shape)>2: 
            self.traj_data = self.traj_data[:, :, :3, :].flatten(start_dim=1)
        self.w_data = torch.cat(self.w_data, dim=0)
        print(f'{split} dataset for Pouring is ready. {list(self.traj_data.size())}')
        
    
    def __getitem__(self, idx):
        x = self.traj_data[idx]
        w = self.w_data[idx]
        return x, w

    def __len__(self) -> int:
        return len(self.traj_data)

    def rand_aug(self, x, w, n_aug=1):
        w, x = group._random_data_aug(w, x, n_aug=n_aug)
        return x, w