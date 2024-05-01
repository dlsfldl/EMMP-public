import yaml
import numpy as np
import torch
from torch import Tensor
from matplotlib.patches import Ellipse, Rectangle, Polygon
from functools import partial
import os
import glob
import re
import errno
from.LieGroup_torch import exp_so3, log_SO3, skew
from scipy import signal

def save_yaml(filename, text):
    """parse string as yaml then dump as a file"""
    with open(filename, "w") as f:
        yaml.dump(yaml.safe_load(text), f, default_flow_style=False)

def label_to_color(label):
    
    n_points = label.shape[0]
    color = np.zeros((n_points, 3))

    # color template (2021 pantone color: orbital)
    rgb = np.zeros((11, 3))
    rgb[0, :] = [253, 134, 18]
    rgb[1, :] = [106, 194, 217]
    rgb[2, :] = [111, 146, 110]
    rgb[3, :] = [153, 0, 17]
    rgb[4, :] = [179, 173, 151]
    rgb[5, :] = [245, 228, 0]
    rgb[6, :] = [255, 0, 0]
    rgb[7, :] = [0, 255, 0]
    rgb[8, :] = [0, 0, 255]
    rgb[9, :] = [18, 134, 253]
    rgb[10, :] = [155, 155, 155] # grey

    for idx_color in range(10):
        color[label == idx_color, :] = rgb[idx_color, :]
    return color

def figure_to_array(fig):
    fig.canvas.draw()
    return np.array(fig.canvas.renderer._renderer)

def PD_metric_to_ellipse(G, center, scale, **kwargs):
    
    # eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(G)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # find angle of ellipse
    vx, vy = eigvecs[:,0][0], eigvecs[:,0][1]
    theta = np.arctan2(vy, vx)

    # draw ellipse
    width, height = 2 * scale * np.sqrt(eigvals)
    return Ellipse(xy=center, width=width, height=height, angle=np.degrees(theta), **kwargs)

def rectangle_scatter(size, center, color):

    return Rectangle(xy=(center[0]-size[0]/2, center[1]-size[1]/2) ,width=size[0], height=size[1], facecolor=color)

def triangle_scatter(size, center, color):
    
    return Polygon(((center[0], center[1] + size[1]/2), (center[0] - size[0]/2, center[1] - size[1]/2), (center[0] + size[0]/2, center[1] - size[1]/2)), fc=color)


def batch_linspace(start: Tensor, stop: Tensor, num: int):
    """
    Creates a tensor of shape [num, *start.shape] whose values are evenly spaced from start to end, inclusive.
    Replicates but the multi-dimensional bahaviour of numpy.linspace in PyTorch.
    """
    # create a tensor of 'num' steps from 0 to 1
    steps = torch.arange(num, dtype=torch.float32, device=start.device) / (num - 1)
    
    # reshape the 'steps' tensor to [-1, *([1]*start.ndim)] to allow for broadcastings
    # - using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here but torchscript
    #   "cannot statically infer the expected size of a list in this contex", hence the code below
    for i in range(start.ndim):
        steps = steps.unsqueeze(-1)
    
    # the output starts at 'start' and increments until 'stop' in each dimension
    out = start[None] + steps*(stop - start)[None]
    
    return out

# Recursive mkdir
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
        
def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def get_yml_paths(main_root='results/Toy2D'):
    files = glob.glob(main_root + '/**/*.yml', recursive=True)
    return files

def get_pretrained_model_cfg(main_root=None, yml_path=None, ckpt_file='model_best.pkl'):
    if yml_path is not None:
        file = yml_path
        file = file.replace('\\', '/')
        idx_list = []
        for idx in re.finditer('/', file, flags=0):
            idx_list.append(idx.span(0)[0])
        idx_list.sort()
        root = file[:idx_list[-2] + 1]
        identifier = file[idx_list[-2] + 1:idx_list[-1]]
        config_file = file[idx_list[-1] + 1 :]
        ckpt_file = ckpt_file
        return {
            'root': root,
            'identifier': identifier,
            'config_file':config_file,
            'ckpt_file': ckpt_file,
        }
    elif main_root is not None:
        files = get_yml_paths(main_root)
        cfg_list = []
        for idx, file in enumerate(files):
            file = file.replace('\\', '/')
            idx_list = []
            for idx in re.finditer('/', file, flags=0):
                idx_list.append(idx.span(0)[0])
            idx_list.sort()
            root = file[:idx_list[-2] + 1]
            identifier = file[idx_list[-2] + 1:idx_list[-1]]
            run = (root + identifier)[len(main_root):]
            config_file = file[idx_list[-1] + 1 :]
            ckpt_file = 'model_best.pkl'
            cfg = {
            'run': run,
            'root': root,
            'identifier': identifier,
            'config_file':config_file,
            'ckpt_file': ckpt_file,
            }
            cfg_list.append(cfg)
        return cfg_list
    else:
        return None
    
def jacobian_decoder_jvp_parallel(func, inputs, v=None, create_graph=True):
    batch_size, z_dim = inputs.size()
    if v is None:
        v = torch.eye(z_dim).unsqueeze(0).repeat(batch_size, 1, 1).view(-1, z_dim).to(inputs)
    inputs = inputs.repeat(1, z_dim).view(-1, z_dim)
    jac = (
        torch.autograd.functional.jvp(
            func, inputs, v=v, create_graph=create_graph
        )[1].view(batch_size, z_dim, -1).permute(0, 2, 1)
    )
    return jac

def jacobian_conditional_decoder_jvp_parallel(func, inputs, cond, v=None, create_graph=True):
    batch_size, z_dim = inputs.size()
    w_dim = cond.shape[-1]
    if v is None:
        v = torch.eye(z_dim).unsqueeze(0).repeat(batch_size, 1, 1).view(-1, z_dim).to(inputs)
    inputs = inputs.repeat(1, z_dim).view(-1, z_dim)
    cond = cond.repeat(1, z_dim).view(-1, w_dim)
    func_condition = partial(func, w=cond)
    jac = (
        torch.autograd.functional.jvp(
            func_condition, inputs, v=v, create_graph=create_graph
        )[1].view(batch_size, z_dim, -1).permute(0, 2, 1)
    )
    return jac

def load_pouring_traj(root='datasets/Pouring/new', r=None, w=None, m=None, return_path=False, reverse=False):
    condition_r = []
    condition_w = []
    condition_m = []
    if r is not None:
        if type(r) == list:
            for r_ in r:
                condition_r.append(f'r{int(r_)}')
        else:
            condition_r.append(f'r{int(r)}')
    if w is not None:
        if type(w) == list:
            for w_ in w:
                condition_w.append(f'w{int(w_)}')
        else:
            condition_w.append(f'w{int(w)}')
    if m is not None:
        if type(m) == list:
            for m_ in m:
                condition_m.append(f'm{int(m_)}')
        else:
            condition_m.append(f'm{int(m)}')
    condition_total = [condition_r, condition_w, condition_m]
    files = glob.glob(root + '/**/*.npy', recursive=True)
    file_selected = []
    traj_selected = []
    for file in files:
        appnd = True
        for cond in condition_total:
            sat_cond = True
            for cond_single in cond:
                sat_cond = False
                if cond_single in file:
                    sat_cond = True
                    break
            if sat_cond == False:
                appnd = False
                break
        if reverse == True:
            appnd = -appnd + 1
        if appnd == True:
            file_selected.append(file)
    file_selected = np.sort(file_selected)
    for file in file_selected:
        traj = np.load(file, allow_pickle=True)
        traj_selected.append(traj)
    if len(traj_selected) == 1:
        traj_selected = traj_selected[0]
    if return_path == True:
        if len(file_selected) == 1:
            file_selected = file_selected[0]
        return np.array(traj_selected), file_selected
    else:
        return np.array(traj_selected)

def SE3smoothing(traj, mode='moving_average'):
    # input size = (bs, n, 4, 4)
    bs = len(traj)
    n = traj.shape[1]
    if mode == 'moving_average':
        R1 = traj[:, :-1, :3, :3].reshape(-1, 3, 3)
        R2 = traj[:, 1:, :3, :3].reshape(-1, 3, 3)
        p1 = traj[:, :-1, :3, 3:]
        p2 = traj[:, 1:, :3, 3:]
        
        R = R1@exp_so3(0.5*log_SO3(R1.permute(0,2,1)@R2))
        R = R.view(bs, -1, 3, 3)
        p = (p1+p2)/2
        
        traj = torch.cat([
                torch.cat([
                    traj[:, 0:1, :3, :],
                    torch.cat([R, p], dim=-1)  
                ], dim=1),
            traj[:, :, 3:4, :]
            ], dim=2)
    elif mode == 'savgol':
        traj_device = traj.device
        traj = traj.detach().cpu()
        window_length = 50
        polyorder = 3
        R = (traj[:, :, :3, :3]) # size = (bs, n, 3, 3)
        w = skew(log_SO3(R.reshape(-1, 3, 3))).reshape(bs, n, 3)
        w = signal.savgol_filter(w, window_length=window_length, polyorder=polyorder, mode="nearest", axis=1)
        w = torch.from_numpy(w).to(traj)
        R = exp_so3(w.reshape(-1, 3)).reshape(bs, n, 3, 3)
        p = (traj[:, :, :3, 3:]) # size = (bs, n, 3, 1)
        p = signal.savgol_filter(p, window_length=window_length, polyorder=polyorder, mode="nearest", axis=1)
        p = torch.from_numpy(p).to(traj)
        traj = torch.cat(
            [torch.cat([R, p], dim=-1), 
            torch.zeros(bs, n, 1, 4).to(traj)]
            , dim=2)
        traj[..., -1, -1] = 1
        traj = traj.to(traj_device)
    return traj