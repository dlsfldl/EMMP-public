import torch
import numpy as np
import os
from torch.utils.data import Dataset
# from torchvision.datasets.mnist import MNIST
from models.groups import PlanarMobileRobot
class ToySpline2d(Dataset):
    def __init__(self,
        split='training',
        flatten=True,
        # return_vel=False,
        augmentation=False,
        root='datasets/toy_2d/',
        init_print=True,
        data_ratio=1.0,
        **kwargs):
        super().__init__()
        self.split = split
        folder_path = os.path.join(root + self.split)
        file_list = os.listdir(folder_path)
        self.augmentation = augmentation
        self.group = PlanarMobileRobot()
        
        traj_list = []
        w_list = []
        
        seed  = kwargs.get('seed', 12535)
        np.random.seed(seed)
        
        for file_name in file_list:
            traj, w = np.load(os.path.join(folder_path, file_name), allow_pickle=True)
            if type(traj) == np.ndarray:
                traj = torch.from_numpy(traj).to(torch.float)
                w = torch.from_numpy(w).to(torch.float)
            
            if len(w.shape) == 1 or w.shape == traj.shape:
                traj = traj.unsqueeze(0)
                w = w.unsqueeze(0)
        
            
            if w.shape == traj.shape:
                w = torch.zeros(len(traj), 3)
                w[:, :2] = traj[:, 0]
            traj_list.append(traj)
            w_list.append(w)
        
        self.traj_data = torch.cat(traj_list, dim=0)
        self.w_data = torch.cat(w_list, dim=0)
        # self.traj_data = torch.from_numpy(np.array(traj_list)).to(torch.float32)
        # self.w_data = torch.from_numpy(np.array(w_list)).to(torch.float32)
        if data_ratio < 1:
            data_lenth_original = len(self.w_data)
            data_length_new = int(np.clip(data_ratio, 0, 1) * data_lenth_original)
            random_idx = np.random.permutation(data_length_new)
            self.w_data = self.w_data[random_idx]
            self.traj_data = self.traj_data[random_idx]
        
        if flatten:
            self.traj_data = self.traj_data.view(len(self.traj_data), -1)
        if init_print:
            print(f"2D toy spline split {split} | {self.traj_data.size()}")
        np.random.seed()

    def __len__(self):
        return len(self.traj_data)

    def __getitem__(self, idx):
        x = self.traj_data[idx]
        w = self.w_data[idx]
        return x, w
    
    def rand_aug(self, x, w):
        w, x = self.group._random_data_aug(w, x)
        return x, w

class LatentDataset(Dataset):
    def __init__(self, dataset, model) -> None:
        super().__init__()
        if hasattr(dataset, 'dataset'):
            dataset = dataset.dataset
        model = model.to(torch.device('cpu'))
        z = model.encode(dataset.traj_data, dataset.w_data)
        self.z_data = z.detach()
        self.w_data = dataset.w_data

    def __len__(self):
        return len(self.z_data)
    
    def __getitem__(self, idx):
        z = self.z_data[idx]
        w = self.w_data[idx]
        return z, w