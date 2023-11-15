
from torch.utils.data import Dataset
import torch

class LatentDataset(Dataset):
    def __init__(self, dataset_pre, model, n_aug=None, same_in_tp=False, g=None, num_manner=6, **kwargs) -> None:
        super().__init__()
        if n_aug is None:
            if 'Equi' in str(type(model)):
                n_aug = 0
            else:
                n_aug = 10
        if hasattr(dataset_pre, 'dataset'):
            dataset_pre = dataset_pre.dataset
        original_device = model.device
        model = model.to(torch.device('cpu'))
        self.group = dataset_pre.group
        traj_data_original = dataset_pre.traj_data
        w_data_original = dataset_pre.w_data
        n_data_original = len(traj_data_original)
        data_idx = [w_data_original.norm(dim=1).argsort()]
        w_data_original = w_data_original[data_idx]
        traj_data_original = traj_data_original[data_idx]
        
        if g is not None:
            w_data_aug, traj_data_aug = self.group.action_traj(
                    g, w_data_original, traj_data_original,
                )
        elif same_in_tp == False:
            w_data_aug, traj_data_aug = self.group._random_data_aug(
                w_data_original, traj_data_original, n_aug=n_aug)
        else:
            w_data_aug = []
            traj_data_aug = []
            num_tp = int(len(w_data_original)/num_manner)
            for n in range(n_aug):
                g = self.group._random_g(num_tp).repeat_interleave(num_manner, dim=0)
                w_data_aug_n, traj_data_aug_n = self.group.action_traj(
                    g, w_data_original, traj_data_original,
                )
                
                w_data_aug.append(w_data_aug_n)
                traj_data_aug.append(traj_data_aug_n)
            w_data_aug = torch.cat(w_data_aug, dim=0)
            traj_data_aug = torch.cat(traj_data_aug, dim=0)
    
        z_data_aug = model.encode(traj_data_aug, w_data_aug).detach()
        self.z_data = z_data_aug
        self.w_data = w_data_aug
        model = model.to(original_device)
        
        if hasattr(dataset_pre, 'augmentation'):
            self.augmentation = True
        else:
            self.augmentation = False

    def __len__(self):
        return len(self.z_data)
    
    def __getitem__(self, idx):
        z = self.z_data[idx]
        w = self.w_data[idx]
        return z, w
    
    def rand_aug(self, z, w):
        return z, w
        