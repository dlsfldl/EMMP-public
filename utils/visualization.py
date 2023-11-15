from bdb import Breakpoint
from builtins import ValueError
import numpy as np
import torch
import copy 

from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse
from models.groups import PlanarMobileRobot

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


# from sklearn.mixture import GaussianMixture

pallete = ['#377eb8', '#ff7f00', '#4daf4a',
            '#8d5daf', '#9c7c7c', '#23a696', 
            '#df5956', '#f6b45e', '#f5d206',
            '#69767c', '#e5e58b', '#e98074',
            '#6897bb', 
            ]



################################################################################################################################################################
#######################################################  Visualization tools for 2d spline   ###################################################################
################################################################################################################################################################
def Spline_2d_plot_recon(model, train_loader, val_loader, device, fig_dict, *args, **kwargs):
    x_train, w_train = next(iter(train_loader))
    train_aug = train_loader.dataset.augmentation
    if train_aug:
        x_train, w_train = train_loader.dataset.rand_aug(x_train, w_train)
    x_train = x_train.to(device)
    w_train = w_train.to(device)
    z_train = model.encode(x_train, w_train)
    x_train_recon = model.decode(z_train, w_train)
    x_train = x_train.cpu()
    x_train_recon = x_train_recon.detach().cpu()

    x_val, w_val = next(iter(val_loader))
    x_val = x_val.to(device)
    w_val = w_val.to(device)
    z_val = model.encode(x_val, w_val)
    x_val_recon = model.decode(z_val, w_val)
    x_val = x_val.cpu()
    x_val_recon = x_val_recon.detach().cpu()

    x_train_fig = plot_2d_spline_flatten(x_train, **kwargs, alpha=0.5)
    x_train_recon_fig = plot_2d_spline_flatten(x_train_recon, alpha=1, fig=copy.deepcopy(x_train_fig), **kwargs)
    x_val_fig = plot_2d_spline_flatten(x_val, **kwargs, alpha=0.5)
    x_val_recon_fig = plot_2d_spline_flatten(x_val_recon, alpha=1, fig=copy.deepcopy(x_val_fig), **kwargs)
    fig_dict['x_train_recon#'] = x_train_recon_fig
    fig_dict['x_val_recon#'] =  x_val_recon_fig
    return fig_dict


def plot_2d_spline_flatten(flatten_data, grid=True, locater=1., fig=None, 
                        alpha=1, max_n=20, wall=True, **kwargs):
    n_spline = len(flatten_data)
    expanded_data = flatten_data.reshape(n_spline, -1, 2)
    if fig == None:
        fig = plt.figure(figsize=[7, 7])
        ax = fig.add_subplot(111)
        ax.set_xlim(-10.5, 10.5)
        ax.set_ylim(-10.5, 10.5)
        if grid:
            ax.grid(zorder=0)
            ax.xaxis.set_major_locator(plt.MultipleLocator(locater))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(locater * 0.1))
            ax.yaxis.set_major_locator(plt.MultipleLocator(locater))
            ax.yaxis.set_minor_locator(plt.MultipleLocator(locater * 0.1))
    else:
        ax = fig.axes[0]
    for i in range(min(n_spline, max_n)):
        spline = expanded_data[i]
        ax.plot(spline[:, 0], spline[:, 1], color=pallete[i % len(pallete)], zorder=i, alpha=alpha, linewidth=2)
    return fig

def Scatter_latent_2d(model, train_loader, val_loader, device, fig_dict, *args, **kwargs):
    x_train_total = train_loader.dataset.traj_data.to(device)
    labels_total = train_loader.dataset.label_targets
    z_train_total = model.encode(x_train_total).detach().cpu()
    fig_scatter = plt.figure(figsize=[7, 7])
    ax_scatter = fig_scatter.add_subplot(111)
    ax_scatter.scatter(z_train_total[:, 0], z_train_total[:, 1], color=np.array(pallete)[labels_total.tolist()])
    fig_dict['latent_z_scatter#'] = fig_scatter
    return fig_dict

def plot_2d_spline_modulation_fig(model, x, w, gmm=None,
                                    grid=True, idx=0, locater=1, 
                                    n_task=3, n_mod=5, title=None, wall=True):
    batch_size = x.size()[0]
    if len(x.shape) == 2:
        x_flat = x
        x = x.reshape(batch_size, -1, 2)
    elif len(x.shape) == 3:
        x_flat = x.reshape(batch_size, -1)
    time_step = x.size()[1]
    
    z = model.encode(x_flat, w)
    z_dim = z.size()[1]   
    
    x = x[idx : idx + n_task]
    z = z[idx : idx + n_task]
    w = w[idx : idx + n_task]

    if wall:
        walls1, walls2, walls3, walls4 = model.group.get_wall(w)
    if gmm is not None:
        z_modified = torch.from_numpy(gmm.sample(n_mod)[0]).to(x)
    else:
        z_modified = torch.randn(n_mod, z_dim).to(x)
    
    fig = plt.figure(figsize=(6.5 * n_task, 6))
    for task_num in range(n_task):
        ax = fig.add_subplot(1, n_task, task_num + 1)
        ax = plot_2d_spline_modulation_axis(model, w[task_num], ax, gmm,
                                    grid, locater, 
                                    n_mod, wall)
        if wall:
            ax.plot(walls1[0, 0], walls1[0, 1], color='black', linewidth=5)
            ax.plot(walls2[0, 0], walls2[0, 1], color='black', linewidth=5)
            ax.plot(walls3[0, 0], walls3[0, 1], color='black', linewidth=5)
            ax.plot(walls4[0, 0], walls4[0, 1], color='black', linewidth=5)
    return fig

def plot_2d_wall(w, ax, grid=True, locater=1,):
    group = PlanarMobileRobot()
    walls1, walls2, walls3, walls4 = group.get_wall(w)
    ax.set_xlim(-10.5, 10.5)
    ax.set_ylim(-10.5, 10.5)
    ax.axis('equal')
    if grid:
        ax.grid(zorder=0)
        ax.xaxis.set_major_locator(plt.MultipleLocator(locater))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(locater * 0.1))
        ax.yaxis.set_major_locator(plt.MultipleLocator(locater))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(locater * 0.1))
    
    ax.plot(walls1[0, 0], walls1[0, 1], color='black', linewidth=5)
    ax.plot(walls2[0, 0], walls2[0, 1], color='black', linewidth=5)
    ax.plot(walls3[0, 0], walls3[0, 1], color='black', linewidth=5)
    ax.plot(walls4[0, 0], walls4[0, 1], color='black', linewidth=5)
    

def plot_2d_spline_modulation_axis(model, w, ax, gmm=None,
                                    grid=True, locater=1, 
                                    n_mod=5, wall=True):
    ax.set_xlim(-10.5, 10.5)
    ax.set_ylim(-10.5, 10.5)
    ax.axis('equal')
    if len(w.shape) == 1:
        w = w.view(1, -1)
    if wall:
        walls1, walls2, walls3, walls4 = model.group.get_wall(w)
    z_modified = torch.from_numpy(gmm.sample(n_mod)[0]).to(w)
    ax.set_xlim(-10.5, 10.5)
    ax.set_ylim(-10.5, 10.5)
    if grid:
        ax.grid(zorder=0)
        ax.xaxis.set_major_locator(plt.MultipleLocator(locater))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(locater * 0.1))
        ax.yaxis.set_major_locator(plt.MultipleLocator(locater))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(locater * 0.1))
    if wall:
        ax.plot(walls1[0, 0], walls1[0, 1], color='black', linewidth=5)
        ax.plot(walls2[0, 0], walls2[0, 1], color='black', linewidth=5)
        ax.plot(walls3[0, 0], walls3[0, 1], color='black', linewidth=5)
        ax.plot(walls4[0, 0], walls4[0, 1], color='black', linewidth=5)
    for fig_num in range(n_mod):
        mod_traj = model.decode(z_modified[fig_num].unsqueeze(0), w).reshape(-1, 2).detach().cpu()
        ax.plot(mod_traj[:, 0], mod_traj[:, 1], color='tab:green', zorder=3, linewidth=2, alpha=1)
    return ax

################################################################################################################################################################
#####################################################  Visualization tools for Pouring dataset   #####################################################
################################################################################################################################################################

def Pouring_recon_train(model, train_loader, val_loader, device, fig_dict, *args, **kwargs):
    train_loader.shuffle = False
    
    X_train = []
    W_train = []
    for x, w in train_loader:
        X_train.append(x.to(device))
        W_train.append(w.to(device))
    X_train = torch.cat(X_train, dim=0)
    W_train = torch.cat(W_train, dim=0)
    
    train_recon_fig = plot_pouring_recon(model, X_train, W_train)
    fig_dict['train_recon#'] = train_recon_fig
    return fig_dict

def Pouring_recon_train_val(model, train_loader, val_loader, device, fig_dict, *args, **kwargs):
    train_loader.shuffle = False
    val_loader.shuffle = False
    
    X_train = []
    W_train = []
    for x, w in train_loader:
        X_train.append(x.to(device))
        W_train.append(w.to(device))
    X_train = torch.cat(X_train, dim=0)
    W_train = torch.cat(W_train, dim=0)
    
    X_val = []
    W_val = []
    for x, w in val_loader:
        X_val.append(x.to(device))
        W_val.append(w.to(device))
    X_val = torch.cat(X_val, dim=0)
    W_val = torch.cat(W_val, dim=0)
    
    train_recon_fig = plot_pouring_recon(model, X_train, W_train)
    val_recon_fig = plot_pouring_recon(model, X_val, W_val)
    fig_dict['train_recon#'] = train_recon_fig
    fig_dict['val_recon_fig#'] = val_recon_fig
    return fig_dict

def Pouring_mod_task(model, train_loader, val_loader, device, fig_dict, *args, **kwargs):
    train_loader.shuffle = False
    val_loader.shuffle = False
    
    X_train = []
    W_train = []
    for x, w in train_loader:
        X_train.append(x.to(device))
        W_train.append(w.to(device))
    X_train = torch.cat(X_train, dim=0)
    W_train = torch.cat(W_train, dim=0)
    
    X_val = []
    W_val = []
    for x, w in val_loader:
        X_val.append(x.to(device))
        W_val.append(w.to(device))
    X_val = torch.cat(X_val, dim=0)
    W_val = torch.cat(W_val, dim=0)
    task_mod_fig = plot_pouring_task_modification(model, X_train, W_train)
    
    fig_dict['task_mod_fig#'] = task_mod_fig
    return fig_dict

def plot_pouring_recon(model, x, w, view_point=None, offset=0.7, idx=0, title=None):
    in_shape = x.shape
    batch_size = in_shape[0]
    num_col = 5
    num_row = 2
    num_traj = num_col*num_row
    if len(in_shape) == 2:
        time_step = int(in_shape[1]/12)
    elif len(in_shape) == 4:
        time_step = in_shape[1]
    else:
        print(f'in_shape = {in_shape}')
        raise(ValueError, 'input shape is wrong.')

    random_idx = torch.randperm(num_traj)
    x = x[random_idx]
    w = w[random_idx]
    h = model.encode(x, w)
    x = x.reshape(num_traj, time_step, 3, 4)
    hat_x = model.decode(h, w).reshape(num_traj, time_step, 3, 4)
    
    fig = plt.figure(figsize=(15, 6))
    
    for cols in range(num_col):
        for rows in range(num_row):
            ax = fig.add_subplot(num_row, num_col, rows*num_col+cols+1, projection='3d')
            traj = x[rows*num_col+cols]
            recon_traj = hat_x[rows*num_col+cols]
            
            x_coord = traj[:, 0, 3].detach().cpu().numpy()
            y_coord = traj[:, 1, 3].detach().cpu().numpy()
            z_coord = traj[:, 2, 3].detach().cpu().numpy()
            ax.scatter(x_coord, y_coord, z_coord, s=3)
            
            delta_x = x_coord.max() - x_coord.min()
            delta_y = y_coord.max() - y_coord.min()
            delta_z = z_coord.max() - z_coord.min()
            delta = torch.max(torch.tensor([delta_x, delta_y, delta_z]))
            x_mean = (x_coord.max() + x_coord.min()) / 2
            y_mean = (y_coord.max() + y_coord.min()) / 2
            z_mean = (z_coord.max() + z_coord.min()) / 2
            scale = delta * 0.1
            
            x_coord = recon_traj[:, 0, 3].detach().cpu().numpy()
            y_coord = recon_traj[:, 1, 3].detach().cpu().numpy()
            z_coord = recon_traj[:, 2, 3].detach().cpu().numpy()
            ax.scatter(x_coord, y_coord, z_coord, color='tab:orange', s=3)
            
            for idx in np.arange(0, time_step, 20):
                SE3_Visualization(ax, traj[idx], scale=scale)
                SE3_Visualization(ax, recon_traj[idx], scale=scale)
            
            x_c = w[rows*5+cols][0].detach().cpu().numpy(); y_c = w[rows*5+cols][1].detach().cpu().numpy()
            x_b = w[rows*5+cols][2].detach().cpu().numpy(); y_b = w[rows*5+cols][3].detach().cpu().numpy(); z_b = w[rows*5+cols][4].detach().cpu().numpy();
            ax.scatter(x_c, y_c, 0, color='black')
            ax.scatter(x_b, y_b, z_b, color='black', marker='x')
            
            if view_point is None: pass
            else: ax.view_init(view_point[0], view_point[1])
            
            ax.set_box_aspect([1,1,1])
            ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
            ax.set_xlim([x_mean - offset * delta, x_mean + offset * delta])
            ax.set_ylim([y_mean - offset * delta, y_mean + offset * delta])
            ax.set_zlim([z_mean - offset * delta, z_mean + offset * delta])
    
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])        
    # plt.show()
    
    return fig

def plot_pouring_task_modification(model, x, w, view_point=None, offset=0.7, idx=0, title=None):

    in_shape = x.shape
    batch_size = in_shape[0]
    if len(in_shape) == 2:
        time_step = int(in_shape[1]/12)
    elif len(in_shape) == 4:
        time_step = in_shape[1]

    list = np.random.randint(batch_size, size=(5))
    w_modified = w[list].clone()
    
    h = model.encode(x, w)
    x = x.reshape(batch_size, time_step, 3, 4)
    
    x = x[idx:idx+5]
    h = h[idx:idx+5]
    w = w[idx:idx+5]
    
    fig = plt.figure(figsize=(18, 15))
    
    for cols in range(6):
        for rows in range(5):
            if cols == 0:
                ax = fig.add_subplot(5, 6, cols+6*rows+1, projection='3d')
                traj = x[rows]
                recon_traj = model.decode(h[rows].unsqueeze(0), w[rows].unsqueeze(0)).reshape(1, time_step, 3, 4).squeeze(0)

                x_coord = traj[:, 0, 3].detach().cpu().numpy()
                y_coord = traj[:, 1, 3].detach().cpu().numpy()
                z_coord = traj[:, 2, 3].detach().cpu().numpy()
                ax.scatter(x_coord, y_coord, z_coord, s=3)
                
                delta_x = x_coord.max() - x_coord.min()
                delta_y = y_coord.max() - y_coord.min()
                delta_z = z_coord.max() - z_coord.min()
                delta = torch.max(torch.tensor([delta_x, delta_y, delta_z]))
                x_mean = (x_coord.max() + x_coord.min()) / 2
                y_mean = (y_coord.max() + y_coord.min()) / 2
                z_mean = (z_coord.max() + z_coord.min()) / 2
                scale = delta * 0.1
                
                x_coord = recon_traj[:, 0, 3].detach().cpu().numpy()
                y_coord = recon_traj[:, 1, 3].detach().cpu().numpy()
                z_coord = recon_traj[:, 2, 3].detach().cpu().numpy()
                ax.scatter(x_coord, y_coord, z_coord, color='tab:orange', s=3)
                
                x_c = w[rows][0].detach().cpu().numpy(); y_c = w[rows][1].detach().cpu().numpy()
                x_b = w[rows][2].detach().cpu().numpy(); y_b = w[rows][3].detach().cpu().numpy(); z_b = w[rows][4].detach().cpu().numpy()
                ax.scatter(x_c, y_c, 0, color='black')
                ax.scatter(x_b, y_b, z_b, color='black', marker='x')

                
                for idx in np.arange(0, time_step, 10):
                    SE3_Visualization(ax, traj[idx], scale=scale)
                    SE3_Visualization(ax, recon_traj[idx], scale=scale)

                
                if view_point is None: pass
                else: ax.view_init(view_point[0], view_point[1])
                
                ax.set_box_aspect([1,1,1])
                ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
                ax.set_xlim([x_mean - offset * delta, x_mean + offset * delta])
                ax.set_ylim([y_mean - offset * delta, y_mean + offset * delta])
                ax.set_zlim([z_mean - offset * delta, z_mean + offset * delta])
                
                if rows == 0:
                    ax.set_title("Original Task")

            else:
                ax = fig.add_subplot(5, 6, cols+6*rows+1, projection='3d')
                recon_traj = model.decode(h[rows].unsqueeze(0), w_modified[cols-1].unsqueeze(0)).reshape(1, time_step, 3, 4).squeeze(0)
                
                x_coord = recon_traj[:, 0, 3].detach().cpu().numpy()
                y_coord = recon_traj[:, 1, 3].detach().cpu().numpy()
                z_coord = recon_traj[:, 2, 3].detach().cpu().numpy()
                ax.scatter(x_coord, y_coord, z_coord, color='tab:orange', s=3)
                
                delta_x = x_coord.max() - x_coord.min()
                delta_y = y_coord.max() - y_coord.min()
                delta_z = z_coord.max() - z_coord.min()
                delta = torch.max(torch.tensor([delta_x, delta_y, delta_z]))
                x_mean = (x_coord.max() + x_coord.min()) / 2
                y_mean = (y_coord.max() + y_coord.min()) / 2
                z_mean = (z_coord.max() + z_coord.min()) / 2
                scale = delta * 0.1
                
                for idx in np.arange(0, time_step, 10):
                    SE3_Visualization(ax, recon_traj[idx], scale=scale)
                
                x_c = w_modified[cols-1][0].detach().cpu().numpy(); y_c = w_modified[cols-1][1].detach().cpu().numpy()
                x_b = w_modified[cols-1][2].detach().cpu().numpy(); y_b = w_modified[cols-1][3].detach().cpu().numpy(); z_b = w_modified[cols-1][4].detach().cpu().numpy()
                ax.scatter(x_c, y_c, 0, color='black')
                ax.scatter(x_b, y_b, z_b, color='black', marker='x')
                
                if view_point is None: pass
                else: ax.view_init(view_point[0], view_point[1])
                
                ax.set_box_aspect([1,1,1])
                ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
                ax.set_xlim([x_mean - offset * delta, x_mean + offset * delta])
                ax.set_ylim([y_mean - offset * delta, y_mean + offset * delta])
                ax.set_zlim([z_mean - offset * delta, z_mean + offset * delta])
                
                if rows == 0:
                    ax.set_title(f"Modified Task {cols}")
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    # plt.show()
    
    return fig

def plot_pouring_manner_modulation(model, x, w, view_point=None, offset=0.7, idx=0, title=None):

    in_shape = x.shape
    batch_size = in_shape[0]
    if len(in_shape) == 2:
        time_step = int(in_shape[1]/12)
    elif len(in_shape) == 4:
        time_step = in_shape[1]

    
    h = model.encode(x, w)
    h_dim = h.size()[1]   
    x = x.reshape(batch_size, time_step, 3, 4)
    
    x = x[idx:idx+5]
    h = h[idx:idx+5]
    w = w[idx:idx+5]

    h_modified = torch.randn(5, h_dim).to(x)

    fig = plt.figure(figsize=(18, 15))
    
    for cols in range(6):
        for rows in range(5):
            if cols == 0:
                ax = fig.add_subplot(5, 6, cols+6*rows+1, projection='3d')
                traj = x[rows]
                recon_traj = model.decode(h[rows].unsqueeze(0), w[rows].unsqueeze(0)).reshape(1, time_step, 3, 4).squeeze(0)

                x_coord = traj[:, 0, 3].detach().cpu().numpy()
                y_coord = traj[:, 1, 3].detach().cpu().numpy()
                z_coord = traj[:, 2, 3].detach().cpu().numpy()
                ax.scatter(x_coord, y_coord, z_coord, s=3)
                
                delta_x = x_coord.max() - x_coord.min()
                delta_y = y_coord.max() - y_coord.min()
                delta_z = z_coord.max() - z_coord.min()
                delta = torch.max(torch.tensor([delta_x, delta_y, delta_z]))
                x_mean = (x_coord.max() + x_coord.min()) / 2
                y_mean = (y_coord.max() + y_coord.min()) / 2
                z_mean = (z_coord.max() + z_coord.min()) / 2
                scale = delta * 0.1
                
                x_coord = recon_traj[:, 0, 3].detach().cpu().numpy()
                y_coord = recon_traj[:, 1, 3].detach().cpu().numpy()
                z_coord = recon_traj[:, 2, 3].detach().cpu().numpy()
                ax.scatter(x_coord, y_coord, z_coord, color='tab:orange', s=3)
                
                for idx in np.arange(0, time_step, 10):
                    SE3_Visualization(ax, traj[idx], scale=scale)
                    SE3_Visualization(ax, recon_traj[idx], scale=scale)
                
                x_c = w[rows][0].detach().cpu().numpy(); y_c = w[rows][1].detach().cpu().numpy()
                x_b = w[rows][2].detach().cpu().numpy(); y_b = w[rows][3].detach().cpu().numpy(); z_b = w[rows][4].detach().cpu().numpy()
                ax.scatter(x_c, y_c, 0, color='black')
                ax.scatter(x_b, y_b, z_b, color='black', marker='x')
                
                if view_point is None: pass
                else: ax.view_init(view_point[0], view_point[1])
                
                ax.set_box_aspect([1,1,1])
                ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
                ax.set_xlim([x_mean - offset * delta, x_mean + offset * delta])
                ax.set_ylim([y_mean - offset * delta, y_mean + offset * delta])
                ax.set_zlim([z_mean - offset * delta, z_mean + offset * delta])    
                
                if rows == 0:
                    ax.set_title("Original Manner")

            else:
                ax = fig.add_subplot(5, 6, cols+6*rows+1, projection='3d')
                recon_traj = model.decode(h_modified[cols-1].unsqueeze(0), w[rows].unsqueeze(0)).reshape(1, time_step, 3, 4).squeeze(0)
            
                x_coord = recon_traj[:, 0, 3].detach().cpu().numpy()
                y_coord = recon_traj[:, 1, 3].detach().cpu().numpy()
                z_coord = recon_traj[:, 2, 3].detach().cpu().numpy()
                ax.scatter(x_coord, y_coord, z_coord, color='tab:orange', s=3)
                
                delta_x = x_coord.max() - x_coord.min()
                delta_y = y_coord.max() - y_coord.min()
                delta_z = z_coord.max() - z_coord.min()
                delta = torch.max(torch.tensor([delta_x, delta_y, delta_z]))
                x_mean = (x_coord.max() + x_coord.min()) / 2
                y_mean = (y_coord.max() + y_coord.min()) / 2
                z_mean = (z_coord.max() + z_coord.min()) / 2
                scale = delta * 0.1
                
                for idx in np.arange(0, time_step, 10):
                    SE3_Visualization(ax, recon_traj[idx], scale=scale)
                
                
                x_c = w[rows][0].detach().cpu().numpy(); y_c = w[rows][1].detach().cpu().numpy()
                x_b = w[rows][2].detach().cpu().numpy(); y_b = w[rows][3].detach().cpu().numpy(); z_b = w[rows][4].detach().cpu().numpy()
                ax.scatter(x_c, y_c, 0, color='black')
                ax.scatter(x_b, y_b, z_b, color='black', marker='x')
                
                if view_point is None: pass
                else: ax.view_init(view_point[0], view_point[1])
                
                ax.set_box_aspect([1,1,1])
                ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
                ax.set_xlim([x_mean - offset * delta, x_mean + offset * delta])
                ax.set_ylim([y_mean - offset * delta, y_mean + offset * delta])
                ax.set_zlim([z_mean - offset * delta, z_mean + offset * delta])
                
                if rows == 0:
                    ax.set_title(f"Modified Manner {cols}")
                
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])     
    # plt.show()
    
    return fig

def plot_pouring_manner_modulation_linspace(model, x, w, view_point=None, offset=0.7, idx=0, title=None):

    in_shape = x.shape
    batch_size = in_shape[0]
    if len(in_shape) == 2:
        time_step = int(in_shape[1]/12)
    elif len(in_shape) == 4:
        time_step = in_shape[1]

    
    h = model.encode(x, w)
    x = x.reshape(batch_size, time_step, 3, 4)
    h_dim = h.size()[1]   
    
    x = x[idx:idx+5]
    h = h[idx:idx+5]
    w = w[idx:idx+5]

    h_modified = torch.linspace(start=-1, end=1, steps=5).to(x).unsqueeze(1)


    fig = plt.figure(figsize=(18, 15))
    
    for cols in range(6):
        for rows in range(5):
            if cols == 0:
                ax = fig.add_subplot(5, 6, cols+6*rows+1, projection='3d')
                traj = x[rows]
                recon_traj = model.decode(h[rows].unsqueeze(0), w[rows].unsqueeze(0)).reshape(1, time_step, 3, 4).squeeze(0)

                x_coord = traj[:, 0, 3].detach().cpu().numpy()
                y_coord = traj[:, 1, 3].detach().cpu().numpy()
                z_coord = traj[:, 2, 3].detach().cpu().numpy()
                ax.scatter(x_coord, y_coord, z_coord, s=3)
                
                delta_x = x_coord.max() - x_coord.min()
                delta_y = y_coord.max() - y_coord.min()
                delta_z = z_coord.max() - z_coord.min()
                delta = torch.max(torch.tensor([delta_x, delta_y, delta_z]))
                x_mean = (x_coord.max() + x_coord.min()) / 2
                y_mean = (y_coord.max() + y_coord.min()) / 2
                z_mean = (z_coord.max() + z_coord.min()) / 2
                scale = delta * 0.1
                
                x_coord = recon_traj[:, 0, 3].detach().cpu().numpy()
                y_coord = recon_traj[:, 1, 3].detach().cpu().numpy()
                z_coord = recon_traj[:, 2, 3].detach().cpu().numpy()
                ax.scatter(x_coord, y_coord, z_coord, color='tab:orange', s=3)
                
                for idx in np.arange(0, time_step, 10):
                    SE3_Visualization(ax, traj[idx], scale=scale)
                    SE3_Visualization(ax, recon_traj[idx], scale=scale)
                
                x_c = w[rows][0].detach().cpu().numpy(); y_c = w[rows][1].detach().cpu().numpy()
                x_b = w[rows][2].detach().cpu().numpy(); y_b = w[rows][3].detach().cpu().numpy(); z_b = w[rows][4].detach().cpu().numpy()
                ax.scatter(x_c, y_c, 0, color='black')
                ax.scatter(x_b, y_b, z_b, color='black', marker='x')
                
                if view_point is None: pass
                else: ax.view_init(view_point[0], view_point[1])
                
                ax.set_box_aspect([1,1,1])
                ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
                ax.set_xlim([x_mean - offset * delta, x_mean + offset * delta])
                ax.set_ylim([y_mean - offset * delta, y_mean + offset * delta])
                ax.set_zlim([z_mean - offset * delta, z_mean + offset * delta])    
                
                if rows == 0:
                    ax.set_title("Original Manner")

            else:
                ax = fig.add_subplot(5, 6, cols+6*rows+1, projection='3d')
                recon_traj = model.decode(h_modified[cols-1].unsqueeze(0), w[rows].unsqueeze(0)).reshape(1, time_step, 3, 4).squeeze(0)
            
                x_coord = recon_traj[:, 0, 3].detach().cpu().numpy()
                y_coord = recon_traj[:, 1, 3].detach().cpu().numpy()
                z_coord = recon_traj[:, 2, 3].detach().cpu().numpy()
                ax.scatter(x_coord, y_coord, z_coord, color='tab:orange', s=3)
                
                delta_x = x_coord.max() - x_coord.min()
                delta_y = y_coord.max() - y_coord.min()
                delta_z = z_coord.max() - z_coord.min()
                delta = torch.max(torch.tensor([delta_x, delta_y, delta_z]))
                x_mean = (x_coord.max() + x_coord.min()) / 2
                y_mean = (y_coord.max() + y_coord.min()) / 2
                z_mean = (z_coord.max() + z_coord.min()) / 2
                scale = delta * 0.1
                
                for idx in np.arange(0, time_step, 10):
                    SE3_Visualization(ax, recon_traj[idx], scale=scale)
                
                
                x_c = w[rows][0].detach().cpu().numpy(); y_c = w[rows][1].detach().cpu().numpy()
                x_b = w[rows][2].detach().cpu().numpy(); y_b = w[rows][3].detach().cpu().numpy(); z_b = w[rows][4].detach().cpu().numpy()
                ax.scatter(x_c, y_c, 0, color='black')
                ax.scatter(x_b, y_b, z_b, color='black', marker='x')
                
                if view_point is None: pass
                else: ax.view_init(view_point[0], view_point[1])
                
                ax.set_box_aspect([1,1,1])
                ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
                ax.set_xlim([x_mean - offset * delta, x_mean + offset * delta])
                ax.set_ylim([y_mean - offset * delta, y_mean + offset * delta])
                ax.set_zlim([z_mean - offset * delta, z_mean + offset * delta])
                
                if rows == 0:
                    ax.set_title(f"Modified Manner {cols}")
                
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    #plt.show()     
    return fig

def plot_pouring_manner_modulation_fig(model, x, w, gmm=None, 
                                    view_point=None, offset=0.7, n_mod=5,
                                    idx=0, title=None):

    in_shape = x.shape
    batch_size = in_shape[0]
    if len(in_shape) == 2:
        time_step = int(in_shape[1]/12)
    elif len(in_shape) == 4:
        time_step = in_shape[1]

    
    h = model.encode(x, w)
    x = x.reshape(batch_size, time_step, 3, 4)
    h_dim = h.size()[1]   
    
    x = x[idx:idx+5]
    h = h[idx:idx+5]
    w = w[idx:idx+5]

    h_modified = torch.from_numpy(gmm.sample(n_mod)[0]).to(x)


    fig = plt.figure(figsize=(18, 15))
    
    for cols in range(6):
        for rows in range(5):
            if cols == 0:
                ax = fig.add_subplot(5, 6, cols+6*rows+1, projection='3d')
                traj = x[rows]
                recon_traj = model.decode(h[rows].unsqueeze(0), w[rows].unsqueeze(0)).reshape(1, time_step, 3, 4).squeeze(0)
                x_coord = recon_traj[:, 0, 3].detach().cpu().numpy()
                y_coord = recon_traj[:, 1, 3].detach().cpu().numpy()
                z_coord = recon_traj[:, 2, 3].detach().cpu().numpy()
                ax.scatter(x_coord, y_coord, z_coord, color='tab:orange', s=3)
                
                delta_x = x_coord.max() - x_coord.min()
                delta_y = y_coord.max() - y_coord.min()
                delta_z = z_coord.max() - z_coord.min()
                delta = torch.max(torch.tensor([delta_x, delta_y, delta_z]))
                x_mean = (x_coord.max() + x_coord.min()) / 2
                y_mean = (y_coord.max() + y_coord.min()) / 2
                z_mean = (z_coord.max() + z_coord.min()) / 2
                scale = delta * 0.1
                
                
                for idx in np.arange(0, time_step, 10):
                    SE3_Visualization(ax, recon_traj[idx], scale=scale)
                
                x_c = w[rows][0].detach().cpu().numpy(); y_c = w[rows][1].detach().cpu().numpy()
                x_b = w[rows][2].detach().cpu().numpy(); y_b = w[rows][3].detach().cpu().numpy(); z_b = w[rows][4].detach().cpu().numpy()
                ax.scatter(x_c, y_c, 0, color='black')
                ax.scatter(x_b, y_b, 0, color='black', marker='x')
                
                if view_point is None: pass
                else: ax.view_init(view_point[0], view_point[1])
                
                ax.set_box_aspect([1,1,1])
                ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
                ax.set_xlim([x_mean - offset * delta, x_mean + offset * delta])
                ax.set_ylim([y_mean - offset * delta, y_mean + offset * delta])
                ax.set_zlim([z_mean - offset * delta, z_mean + offset * delta])    
                
                if rows == 0:
                    ax.set_title("Original Manner")

            else:
                ax = fig.add_subplot(5, 6, cols+6*rows+1, projection='3d')
                recon_traj = model.decode(h_modified[cols-1].unsqueeze(0), w[rows].unsqueeze(0)).reshape(1, time_step, 3, 4).squeeze(0)
            
                x_coord = recon_traj[:, 0, 3].detach().cpu().numpy()
                y_coord = recon_traj[:, 1, 3].detach().cpu().numpy()
                z_coord = recon_traj[:, 2, 3].detach().cpu().numpy()
                ax.scatter(x_coord, y_coord, z_coord, color='tab:orange', s=3)
                
                delta_x = x_coord.max() - x_coord.min()
                delta_y = y_coord.max() - y_coord.min()
                delta_z = z_coord.max() - z_coord.min()
                delta = torch.max(torch.tensor([delta_x, delta_y, delta_z]))
                x_mean = (x_coord.max() + x_coord.min()) / 2
                y_mean = (y_coord.max() + y_coord.min()) / 2
                z_mean = (z_coord.max() + z_coord.min()) / 2
                scale = delta * 0.1
                
                for idx in np.arange(0, time_step, 10):
                    SE3_Visualization(ax, recon_traj[idx], scale=scale)
                
                
                x_c = w[rows][0].detach().cpu().numpy(); y_c = w[rows][1].detach().cpu().numpy()
                x_b = w[rows][2].detach().cpu().numpy(); y_b = w[rows][3].detach().cpu().numpy(); z_b = w[rows][4].detach().cpu().numpy()
                ax.scatter(x_c, y_c, 0, color='black')
                ax.scatter(x_b, y_b, z_b, color='black', marker='x')
                
                if view_point is None: pass
                else: ax.view_init(view_point[0], view_point[1])
                
                ax.set_box_aspect([1,1,1])
                ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
                ax.set_xlim([x_mean - offset * delta, x_mean + offset * delta])
                ax.set_ylim([y_mean - offset * delta, y_mean + offset * delta])
                ax.set_zlim([z_mean - offset * delta, z_mean + offset * delta])
                
                if rows == 0:
                    ax.set_title(f"Modified Manner {cols}")
                
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    #plt.show()     
    return fig

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)

def SE3_Trajectory_Visualization(traj, task=None, step=20, offset=0.7, view_point=None):
    '''Visualize one batch of SE(3)-trajectory'''
    time_step = traj.size()[0]
    traj = traj.reshape(time_step, 3, 4)
    
    x_coord = traj[:, 0, 3].detach().numpy()
    y_coord = traj[:, 1, 3].detach().numpy()
    z_coord = traj[:, 2, 3].detach().numpy()
    delta_x = x_coord.max() - x_coord.min()
    delta_y = y_coord.max() - y_coord.min()
    delta_z = z_coord.max() - z_coord.min()
    delta = torch.max(torch.tensor([delta_x, delta_y, delta_z]))
    x_mean = (x_coord.max() + x_coord.min()) / 2
    y_mean = (y_coord.max() + y_coord.min()) / 2
    z_mean = (z_coord.max() + z_coord.min()) / 2
    scale = delta * 0.1
    c = np.arange(time_step)
    
    fig = plt.figure(figsize = (8, 8))
    ax = Axes3D(fig)
    # ax = fig.add_subplot(projection='3d')
    
    for idx in np.arange(0, time_step, step):
        SE3_Visualization(ax, traj[idx], scale=scale)

    ax.scatter(x_coord, y_coord, z_coord, c=c, cmap='viridis_r')
    
    if task is not None:
        x_c = task[0].detach().numpy(); y_c = task[1].detach().numpy()
        x_b = task[2].detach().numpy(); y_b = task[3].detach().numpy(); z_b = task[4].detach().numpy();
        ax.scatter(x_c, y_c, 0, c='black')
        ax.scatter(x_b, y_b, z_b, color='black', marker='x')
    if view_point is None: pass
    else: ax.view_init(view_point[0], view_point[1])

    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    ax.set_xlim([x_mean - offset * delta, x_mean + offset * delta])
    ax.set_ylim([y_mean - offset * delta, y_mean + offset * delta])
    ax.set_zlim([z_mean - offset * delta, z_mean + offset * delta])

    # ax.autoscale()


    plt.show()
    
    return fig
    
def SE3_Visualization(ax, T, scale = 0.1):
    origin = T[:3, 3]
    hat_x = (T[:3, 0]*scale + origin).detach().cpu().numpy()
    hat_y = (T[:3, 1]*scale + origin).detach().cpu().numpy()
    hat_z = (T[:3, 2]*scale + origin).detach().cpu().numpy()
    origin = origin.detach().cpu().numpy()
    arrow_prop_dict = dict(mutation_scale=5, arrowstyle='->', shrinkA=0, shrinkB=0)
    
    a = Arrow3D([origin[0], hat_x[0]], [origin[1], hat_x[1]], [origin[2], hat_x[2]], **arrow_prop_dict, color='r')
    ax.add_artist(a)
    a = Arrow3D([origin[0], hat_y[0]], [origin[1], hat_y[1]], [origin[2], hat_y[2]], **arrow_prop_dict, color='b')
    ax.add_artist(a)
    a = Arrow3D([origin[0], hat_z[0]], [origin[1], hat_z[1]], [origin[2], hat_z[2]], **arrow_prop_dict, color='g')
    ax.add_artist(a)
    return ax