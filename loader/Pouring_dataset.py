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
        
class Pouring_old(Dataset):
    def __init__(self, split='training', position_scale=0.5, aug_size=None, path='datasets', **kwargs):
        self.split = split
        self.position_scale = position_scale
        self.aug_size = aug_size
        self.x = []
        self.w = [] 
        np.random.seed(0)
        torch.manual_seed(0)
        self.base = path
        
        if self.split == 'training':
            # for manner in tqdm(['val_test_cupleft', 'val_test_cupright']):
            for manner in tqdm([ 'manner3_original']):
                folder_path = self.base +'/'+ 'Pouring/' + manner + '/'
                file_list = os.listdir(folder_path)
                file_list.sort()
                for file_name in file_list:
                    file_path = folder_path + '/' + file_name
                    r_idx = file_name.index('r'); h_idx = file_name.index('h')
                    r = int(file_name[r_idx+1:r_idx+3])/100; h = int(file_name[h_idx+1:h_idx+3])/100
                    if r==0.6 and h==0.5:
                        continue
                    traj = torch.tensor(np.load(file_path), dtype=torch.float32).unsqueeze(0)
                    tau = torch.tensor([0, 0, r, 0, h, 0], dtype=torch.float32).unsqueeze(0)
                    if self.aug_size is not None:
                        traj = traj.repeat(self.aug_size, 1, 1, 1)
                        tau = tau.repeat(self.aug_size, 1)
                        
                        p_c = np.random.rand(self.aug_size, 2) * self.position_scale - 0.5 * self.position_scale
                        theta = np.random.rand(self.aug_size, 2) * (2 * np.pi)
                        p_c = torch.tensor(p_c, dtype=torch.float32); theta = torch.tensor(theta, dtype=torch.float32)
                        g = torch.cat((p_c, theta), dim=1)
                        gtau, gtraj = group.action_traj(g, tau, traj)
                        
                        self.x.append(gtraj)
                        self.w.append(gtau)
                        
                    else:
                        self.x.append(traj)
                        self.w.append(tau)
                    
            self.x = torch.cat(self.x, dim=0)#[:, :, :3, :]
            self.w = torch.cat(self.w, dim=0)

            
            print(f'Trianing dataset for Pouring is ready. {list(self.x.size())}')
        
        elif self.split == 'validation':
            for manner in tqdm(['val_test_cupleft', 'val_test_cupright']):
            # for manner in tqdm(['manner1', 'manner2', 'manner3', 'manner4', 'manner5']):
                folder_path = self.base +'/'+ 'Pouring/' + manner + '/'
                file_list = os.listdir(folder_path)
                num_file = len(file_list)
                file_list = file_list[:int(num_file/2)]
                for file_name in file_list:
                    file_path = folder_path + '/' + file_name
                    r_idx = file_name.index('r'); h_idx = file_name.index('h')
                    r = int(file_name[r_idx+1:r_idx+3])/100; h = int(file_name[h_idx+1:h_idx+3])/100
                    
                    traj = torch.tensor(np.load(file_path), dtype=torch.float32).unsqueeze(0)
                    tau = torch.tensor([0, 0, r, 0, h, 0], dtype=torch.float32).unsqueeze(0)

                    if self.aug_size is not None:
                        traj = traj.repeat(self.aug_size, 1, 1, 1)
                        tau = tau.repeat(self.aug_size, 1)
                        
                        p_c = np.random.rand(self.aug_size, 2) * self.position_scale - 0.5 * self.position_scale
                        theta = np.random.rand(self.aug_size, 2) * (2 * np.pi)
                        p_c = torch.tensor(p_c, dtype=torch.float32); theta = torch.tensor(theta, dtype=torch.float32)
                        g = torch.cat((p_c, theta), dim=1)
                        gtau, gtraj = group.action_traj(g, tau, traj)
                        
                        self.x.append(gtraj)
                        self.w.append(gtau)
                        
                    else:
                        self.x.append(traj)
                        self.w.append(tau)
                    
            self.x = torch.cat(self.x, dim=0)[:, :, :3, :]
            self.w = torch.cat(self.w, dim=0)
        
            print(f'Validation dataset for Pouring is ready. {list(self.x.size())}')
            
        elif self.split == 'test':
            for manner in tqdm(['val_test_cupleft', 'val_test_cupright']):
            # for manner in tqdm(['manner1', 'manner2', 'manner3', 'manner4', 'manner5']):   
                folder_path = self.base +'/'+ 'Pouring/' + manner + '/'
                file_list = os.listdir(folder_path)
                num_file = len(file_list)
                file_list = file_list[int(num_file/2):]
                for file_name in file_list:
                    file_path = folder_path + '/' + file_name
                    r_idx = file_name.index('r'); h_idx = file_name.index('h')
                    r = int(file_name[r_idx+1:r_idx+3])/100; h = int(file_name[h_idx+1:h_idx+3])/100
                    
                    traj = torch.tensor(np.load(file_path), dtype=torch.float32).unsqueeze(0)
                    tau = torch.tensor([0, 0, r, 0, h, 0], dtype=torch.float32).unsqueeze(0)
                    
                    if self.aug_size is not None:
                        traj = traj.repeat(self.aug_size, 1, 1, 1)
                        tau = tau.repeat(self.aug_size, 1)
                        
                        p_c = np.random.rand(self.aug_size, 2) * self.position_scale - 0.5 * self.position_scale
                        theta = np.random.rand(self.aug_size, 2) * (2 * np.pi)
                        p_c = torch.tensor(p_c, dtype=torch.float32); theta = torch.tensor(theta, dtype=torch.float32)
                        g = torch.cat((p_c, theta), dim=1)
                        gtau, gtraj = PouringGroup().action_traj(g, tau, traj)
                        
                        self.x.append(gtraj)
                        self.w.append(gtau)
                        
                    else:
                        self.x.append(traj)
                        self.w.append(tau)
                        
            self.x = torch.cat(self.x, dim=0)[:, :, :3, :]
            self.w = torch.cat(self.w, dim=0)
            print(f'Test dataset for Pouring is ready. {list(self.x.size())}')
            
    def __getitem__(self, idx):
        x = self.x[idx]
        w = self.w[idx]

        return x, w

    def __len__(self) -> int:
        return len(self.x)

    
class simple_Pouring(Dataset):
    def __init__(self, split='training', position_scale=0.5, aug_size=None, data_path='datasets', noise=0, **kwargs):
        self.split = split
        self.position_scale = position_scale
        self.aug_size = aug_size
        self.x = []
        self.w = [] 
        np.random.seed(0)
        torch.manual_seed(0)
        self.base = data_path

        if self.split == 'training':
            folder_path = self.base +'/'+ 'Pouring/r40h40s/'
            file_list = os.listdir(folder_path)
            file_list.sort()
            for file_name in file_list:
                file_path = folder_path + '/' + file_name
                r_idx = file_name.index('r'); h_idx = file_name.index('h')
                r = int(file_name[r_idx+1:r_idx+3])/100; h = int(file_name[h_idx+1:h_idx+3])/100
                
                traj = torch.tensor(np.load(file_path), dtype=torch.float32).unsqueeze(0)
                tau = torch.tensor([0, 0, r, 0, h, 0], dtype=torch.float32).unsqueeze(0)

                
                if self.aug_size is not None:
                    traj = traj.repeat(self.aug_size, 1, 1, 1)
                    tau = tau.repeat(self.aug_size, 1)
                    
                    p_c = np.random.rand(self.aug_size, 2) * self.position_scale - 0.5 * self.position_scale
                    theta = np.random.rand(self.aug_size, 2) * (2 * np.pi)
                    p_c = torch.tensor(p_c, dtype=torch.float32); theta = torch.tensor(theta, dtype=torch.float32)
                    g = torch.cat((p_c, theta), dim=1)
                    gtau, gtraj = PouringGroup().action_traj(g, tau, traj)
                    
                    self.x.append(gtraj)
                    self.w.append(gtau)
                    
                else:
                    self.x.append(traj)
                    self.w.append(tau)
                
                
            self.x = torch.cat(self.x, dim=0)[:, :, :3, :]
            self.w = torch.cat(self.w, dim=0)
        

        # noise = torch.randn(self.x.size()) * noise
        # self.x = self.x + noise
        
            print(f'Trianing dataset for simple Pouring is ready. {list(self.x.size())}')
        
        elif self.split == 'validation': 
            folder_path = self.base +'/'+ 'Pouring/r40h40s/'
            file_list = os.listdir(folder_path)
            num_file = len(file_list)
            file_list = file_list[:int(num_file/2)]
            for file_name in file_list:
                file_path = folder_path + '/' + file_name
                r_idx = file_name.index('r'); h_idx = file_name.index('h')
                r = int(file_name[r_idx+1:r_idx+3])/100; h = int(file_name[h_idx+1:h_idx+3])/100
                
                traj = torch.tensor(np.load(file_path), dtype=torch.float32).unsqueeze(0)
                tau = torch.tensor([0, 0, r, 0, h, 0], dtype=torch.float32).unsqueeze(0)

                if aug_size is not None:
                    traj = traj.repeat(self.aug_size, 1, 1, 1)
                    tau = tau.repeat(self.aug_size, 1)
                    
                    p_c = np.random.rand(self.aug_size, 2) * self.position_scale - 0.5 * self.position_scale
                    theta = np.random.rand(self.aug_size, 2) * (2 * np.pi)
                    p_c = torch.tensor(p_c, dtype=torch.float32); theta = torch.tensor(theta, dtype=torch.float32)
                    g = torch.cat((p_c, theta), dim=1)
                    gtau, gtraj = PouringGroup().action_traj(g, tau, traj)
                    
                    self.x.append(gtraj)
                    self.w.append(gtau)
                    
                else:
                    self.x.append(traj)
                    self.w.append(tau)
                        
            self.x = torch.cat(self.x, dim=0)[:, :, :3, :]
            self.w = torch.cat(self.w, dim=0)
            print(f'Validation dataset for simple Pouring is ready. {list(self.x.size())}')
            
        elif self.split == 'test':
            folder_path = self.base +'/'+ 'Pouring/r40h40s/'
            file_list = os.listdir(folder_path)
            num_file = len(file_list)
            file_list = file_list[int(num_file/2):]
            for file_name in file_list:
                file_path = folder_path + '/' + file_name
                r_idx = file_name.index('r'); h_idx = file_name.index('h')
                r = int(file_name[r_idx+1:r_idx+3])/100; h = int(file_name[h_idx+1:h_idx+3])/100

                
                traj = torch.tensor(np.load(file_path), dtype=torch.float32).unsqueeze(0)
                tau = torch.tensor([0, 0, r, 0, h, 0], dtype=torch.float32).unsqueeze(0)

                
                if aug_size is not None:
                    traj = traj.repeat(self.aug_size, 1, 1, 1)
                    tau = tau.repeat(self.aug_size, 1)
                    
                    p_c = np.random.rand(self.aug_size, 2) * self.position_scale - 0.5 * self.position_scale
                    theta = np.random.rand(self.aug_size, 2) * (2 * np.pi)
                    p_c = torch.tensor(p_c, dtype=torch.float32); theta = torch.tensor(theta, dtype=torch.float32)
                    g = torch.cat((p_c, theta), dim=1)
                    gtau, gtraj = PouringGroup().action_traj(g, tau, traj)
                    
                    self.x.append(gtraj)
                    self.w.append(gtau)
                    
                else:
                    self.x.append(traj)
                    self.w.append(tau)
                        
            self.x = torch.cat(self.x, dim=0)[:, :, :3, :]
            self.w = torch.cat(self.w, dim=0)
            print(f'Test dataset for simple Pouring is ready. {list(self.x.size())}')
            
    def __getitem__(self, idx):
        x = self.x[idx]
        w = self.w[idx]

        return x, w

    def __len__(self) -> int:
        return len(self.x)