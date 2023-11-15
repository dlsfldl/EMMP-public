import torch
from utils.line_intersection import collision_check_wall_traj, orientation_batch
from models.groups import PlanarMobileRobot
# from loader.latent_dataset import LatentDataset
# from models.Latent_sampler import get_GMM
from utils.visualization import plot_2d_spline_modulation_axis
import matplotlib.pyplot as plt
from utils.utils import jacobian_conditional_decoder_jvp_parallel
import numpy as np
from functools import partial
from utils import LieGroup_torch as lie
# from utils import Lie_for_debug as lie

def MSE_loss(traj_batch1, traj_batch2, *args, **kwargs):
    return ((traj_batch1 - traj_batch2) ** 2).view(len(traj_batch1), -1).mean(dim=1).mean()

def RMSE_loss(traj_batch1, traj_batch2, *args, **kwargs):
    return ((traj_batch1 - traj_batch2) ** 2).view(len(traj_batch1), -1).mean(dim=1).mean().sqrt()

def MSE_loss_SE3(traj_batch1, traj_batch2, gamma=5):
    batch_size = len(traj_batch1)
    traj_batch1 = traj_batch1.reshape(batch_size, -1, 3, 4)
    num_timestep = traj_batch1.shape[1]
    traj_batch2 = traj_batch2.reshape(batch_size, -1, 3, 4)
    p_recon = traj_batch1[..., -1]
    R_recon = traj_batch1[..., :3]
    p_original = traj_batch2[..., -1]
    R_original = traj_batch2[..., :3]
    p_error = ((p_original - p_recon)**2).sum(dim=-1).mean()
    SO3_diff = (R_original.transpose(2, 3) @ R_recon).reshape(batch_size * num_timestep, 3, 3)
    eye = torch.zeros_like(SO3_diff)
    eye[..., range(3), range(3)] = 1
    so3_error = ((SO3_diff - eye).norm(p='fro', dim=(1, 2))**2).mean()
    return gamma * p_error + so3_error

def MSE_loss_SE3_temp(traj_batch1, traj_batch2, gamma=5):
    batch_size = len(traj_batch1)
    traj_batch1 = traj_batch1.reshape(batch_size, -1, 3, 4)
    num_timestep = traj_batch1.shape[1]
    traj_batch2 = traj_batch2.reshape(batch_size, -1, 3, 4)
    p_recon = traj_batch1[..., -1]
    R_recon = traj_batch1[..., :3]
    p_original = traj_batch2[..., -1]
    R_original = traj_batch2[..., :3]
    p_error = ((p_original - p_recon)**2).sum(dim=-1).mean()
    SO3_diff = (R_original.transpose(2, 3) @ R_recon).reshape(batch_size * num_timestep, 3, 3)
    so3_error = (SO3_diff**2).sum(dim=-1).sum(dim=-1).mean()
    so3diff2 = SO3_diff.detach().clone()
    so3diff2.requires_grad=False
    return gamma * p_error + so3_error

def RMSE_loss_SE3(recon_traj, original_traj, gamma=5):
    mseloss = MSE_loss_SE3(recon_traj, original_traj, gamma=gamma)
    return torch.sqrt(mseloss)