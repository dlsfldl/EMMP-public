import torch
from utils.line_intersection import collision_check_wall_traj, orientation_batch, check_entrance_traj
from models.groups import PlanarMobileRobot
from loader.latent_dataset import LatentDataset
from models.Latent_sampler import get_GMM
from utils.visualization import plot_2d_spline_modulation_axis
import matplotlib.pyplot as plt
from utils.utils import jacobian_conditional_decoder_jvp_parallel
from models.Latent_sampler import get_optimized_GMM, get_optimized_GMM_latent
import numpy as np
from functools import partial


def get_w_sample(
    unseen=True, group=PlanarMobileRobot(), 
    w_data=None, sample_size=1000, aug=True,
    **kwargs):
    if unseen:
        w_sample = group._random_task(sample_size).to(w_data)
        if w_data is not None:
            w_sample = w_sample.to(w_data)
    else:
        w_idx_rd = torch.randint(0, len(w_data), [sample_size])
        w_sample = (w_data[w_idx_rd])
        if aug:
            w_sample = group._random_data_aug(w_sample, n_aug=1)
    return w_sample

def get_traj_sample_split(
    model, w_data, sample_size=1000, split_size=200, random_seed=1,
    unseen=True, group=PlanarMobileRobot(), gmm=None, nf=None):
    traj_sample_total_list = []
    w_sample_total = get_w_sample(
        unseen=unseen, group=group, w_data=w_data, sample_size=sample_size).to(model.device)
    if nf is not None:
        z_sample_total = nf.sample(sample_size).to(model.device)
    elif gmm is not None:
        gmm.random_state = random_seed
        z_sample_total = torch.from_numpy(gmm.sample(sample_size)[0]).to(torch.float).to(model.device)
        # print('z_sample_total[:2]')
        # print(z_sample_total[:2])
    else:
        print('No sampler exists..')
        
    for w_sample, z_sample in zip(
        w_sample_total.split(split_size), z_sample_total.split(split_size)):
        # print(model.device, z_sample.device, w_sample.device)
        # print(z_sample.dtype, w_sample.dtype)
        traj_sample = model.decode(z_sample, w_sample).detach().cpu()
        traj_sample_total_list.append(traj_sample)
    traj_sample_total = torch.cat(traj_sample_total_list, dim=0)
    return traj_sample_total, w_sample_total, z_sample_total

def get_successful_traj_sample(model, w_data, sample_size=1000, split_size=200,
                            nf=None,
                            unseen=True, group=PlanarMobileRobot(), threshold=0.1,
                            gmm=None, random_seed=1):
    traj_sample, w_sample, z_sample = get_traj_sample_split(
        model, w_data, sample_size=sample_size, split_size=split_size,
        unseen=unseen, group=group, gmm=gmm, nf=nf, random_seed=random_seed,
        )
    collision_free = check_collision(traj_sample, w_sample, group=group)
    init_dist, final_dist = get_init_final_dist(traj_sample, w_sample)
    success, bc = check_success(collision_free, init_dist, final_dist, threshold=threshold)
    traj_sample_successful = traj_sample[success]
    w_sample_successful = w_sample[success]
    z_sample_successful = z_sample[success]
    len_success = len(traj_sample_successful)
    return traj_sample_successful, w_sample_successful, z_sample_successful, len_success
    
def get_traj_sample(model, dataloader, sample_size=1000, split_size=200,
                    unseen=True, group=PlanarMobileRobot(), 
                    nf=None, #normalizing flow,
                    gmm=None,
                    only_successful=False, **kwargs):
    w_data = dataloader.dataset.w_data.to(model.device)
    if nf is None:
        if gmm == None:
            gmm = get_GMM(model, dataloader, **kwargs)
    else:
        gmm = None
    traj_sample, w_sample, z_sample = get_traj_sample_split(
        model, w_data, sample_size=sample_size, split_size=split_size,
        unseen=unseen, group=group, gmm=gmm, nf=nf)
    # print('w_sample[:10]')
    # print(w_sample[:10])
    if only_successful:
        threshold = kwargs['threshold']
        success_index_prev = 0
        traj_sample_final = torch.zeros_like(traj_sample).to(w_data)
        w_sample_final = torch.zeros_like(w_sample).to(w_data)
        z_sample_final = torch.zeros_like(z_sample).to(w_data)
        num_sample_total = 0
        len_success_total = 0
        counter = 0
        while True:
            torch.cuda.empty_cache()
            (traj_sample_successful, w_sample_successful, 
             z_sample_successful, len_success) = get_successful_traj_sample(
                            model, w_data, sample_size, split_size=split_size,
                            unseen=unseen, group=group, threshold=threshold,
                            random_seed=len_success_total,
                            gmm=gmm, nf=nf)
            num_sample_total += sample_size
            len_success_total += len_success
            success_rate = len_success_total/num_sample_total
            if success_rate < 0.1:
                print('Succes rate is less then 10%. Aborting..')
                return traj_sample, w_sample, z_sample, 0.001
            if len_success + success_index_prev >= sample_size:
                len_success = sample_size - success_index_prev
                traj_sample_final[success_index_prev:success_index_prev+len_success] \
                    = traj_sample_successful[:len_success]
                w_sample_final[success_index_prev:success_index_prev+len_success]  \
                    = w_sample_successful[:len_success]
                z_sample_final[success_index_prev:success_index_prev+len_success]  \
                    = z_sample_successful[:len_success]
                break
            else:
                traj_sample_final[success_index_prev:success_index_prev+len_success]  \
                    = traj_sample_successful
                w_sample_final[success_index_prev:success_index_prev+len_success]  \
                    = w_sample_successful
                z_sample_final[success_index_prev:success_index_prev+len_success]  \
                    = z_sample_successful
                success_index_prev += len_success
                del traj_sample_successful, w_sample_successful, z_sample_successful
        return traj_sample_final, w_sample_final, z_sample_final, success_rate
    else:
        divider = 1
        return traj_sample, w_sample, z_sample, divider

def cal_success_rate(collision_free, init_dist, final_dist, threshold=0.1):
    batch_size = len(collision_free)
    success, boundary_condition = check_success(
        collision_free, init_dist, final_dist, threshold=threshold)
    success_count = torch.sum(success).item()
    collision_free_count = torch.sum(collision_free).item()
    boundary_condition_count = torch.sum(boundary_condition).item()
    return success_count/batch_size, collision_free_count/batch_size, boundary_condition_count/batch_size

def get_collision_bc_model(
    model, dataloader, sample_size=1000, 
    unseen=True, group=PlanarMobileRobot(),
    nf=None,
    gmm=None,
    # n_components=None, random_state=12, 
    **kwargs):
    traj_sample, w_sample, z_sample_, _ = get_traj_sample(
        model, dataloader, sample_size, 
        nf=nf, gmm=gmm,
        unseen=unseen, group=group, **kwargs)
    collision_free = check_collision(traj_sample, w_sample, group=group)
    init_dist, final_dist = get_init_final_dist(traj_sample, w_sample)
    return collision_free, init_dist, final_dist
    
def check_collision(traj_batch, w, group=PlanarMobileRobot()):
    batch_size = len(traj_batch)
    traj_batch = traj_batch.reshape(batch_size, -1, 2)
    walls = group.get_wall(w)
    entrances = group.get_wrong_entrance(w)
    collision_free_idx = 1 - collision_check_wall_traj(walls, traj_batch)
    right_entrance_idx = 1 - check_entrance_traj(entrances, traj_batch)
    return (collision_free_idx * right_entrance_idx).to(torch.bool)

def get_init_final_dist(traj_batch, w):
    batch_size = len(traj_batch)
    traj_batch = traj_batch.reshape(batch_size, -1, 2)
    init_point = traj_batch[:, 0]
    init_cond = w[:, :2].to(traj_batch)
    init_point_batch_dist = torch.norm(init_point - init_cond, dim=1).cpu()
    last_point_batch_dist = torch.norm(traj_batch[:, -1], dim=1).cpu()
    return init_point_batch_dist, last_point_batch_dist

def check_boundary_condition(init_point_batch_dist, last_point_batch_dist, threshold=0.1):
    initial = init_point_batch_dist < threshold
    arrival = last_point_batch_dist < threshold
    boundary_condition = arrival * initial
    return boundary_condition

def check_success(collision_free, init_dist, final_dist, threshold=0.1):
    boundary_condition = check_boundary_condition(init_dist, final_dist, threshold=threshold)
    success = (collision_free * boundary_condition).to(torch.bool)
    return success, boundary_condition
    

def diversity_measure(model, dataloader, sample_size=1000,
                      unseen=True, group=PlanarMobileRobot(),
                      nf=None,
                      n_components=10, random_state=12, modewise=False,
                      only_successful=True, threshold=0.1):
    traj_sample, w_sample_, z_sample_, success_rate_ = get_traj_sample(
        model, dataloader, sample_size, 
        unseen, group, n_components, random_state,
        nf=nf,
        only_successful=only_successful, threshold=threshold)
    if modewise:
        return diversity_measure_modewise_given_traj(traj_sample)
    else:
        return diversity_measure_total_given_traj(traj_sample)

def diversity_measure_total_given_traj(traj_batch):
    max_batch_size = 100000
    batch_size = len(traj_batch)
    if len(traj_batch.shape) == 2:
        traj_batch = traj_batch.reshape(batch_size, -1, 2)
    traj_repeat = traj_batch.repeat(batch_size, 1, 1)
    traj_interleave = traj_batch.repeat_interleave(batch_size, dim=0)
    sum_distance = 0
    for traj_repeat_batch, traj_interleave_batch in zip(
        traj_repeat.split(max_batch_size), 
        traj_interleave.split(max_batch_size)):
        distance = torch.norm((traj_repeat_batch - traj_interleave_batch), dim=[1, 2])
        sum_distance += distance.sum()
    return sum_distance / (batch_size * (batch_size - 1))

def get_mode_traj(traj_batch):
    batch_size = len(traj_batch)
    if len(traj_batch.shape) == 2:
        traj_batch = traj_batch.reshape(batch_size, -1, 2)
    p_batch = traj_batch[:, 0]
    q_batch = traj_batch[:, -1]
    len_traj = traj_batch.shape[1]
    r2_batch = traj_batch[:, int(-len_traj / 2)]
    r_batch = traj_batch.mean(dim=1)
    orientation = orientation_batch(p_batch, q_batch, r_batch)
    cos_threshold = 0.92
    line1 = q_batch - p_batch
    line2 = q_batch - r2_batch
    dotp = torch.sum(line1 * line2, dim=1)
    cos_theta = dotp / (line1.norm(dim=1) * line2.norm(dim=1))
    mode = torch.zeros(len(p_batch)).to(traj_batch)
    mode[(cos_theta <= cos_threshold)] = 1
    mode[orientation < 0 ] = 0
    traj_mode0 = traj_batch[mode == 0]
    traj_mode1 = traj_batch[mode == 1]
    return traj_mode0, traj_mode1

def diversity_measure_modewise_given_traj(traj_batch):
    traj_mode0, traj_mode1 = get_mode_traj(traj_batch)
    diversity_mode0 = diversity_measure_total_given_traj(traj_mode0)
    diversity_mode1 = diversity_measure_total_given_traj(traj_mode1)
    return diversity_mode0, diversity_mode1

def cal_log_p_ambient(model, z_sample, w_sample, divider=1,
                          gmm=None, nf=None, split=200):
    # print(f'success rate = {divider:.2f}', end='')
    if len(w_sample) == 1:
        w_sample = w_sample.repeat(len(z_sample), 1)
    z_sample = z_sample.detach().to(model.device)
    w_sample = w_sample.detach().to(model.device)
    J_g = []
    for w_sample_split, z_sample_split in zip(w_sample.split(split), z_sample.split(split)):
        J_g_split = jacobian_conditional_decoder_jvp_parallel(model.decode, z_sample_split, w_sample_split).detach()
        J_g.append(J_g_split)
    J_g = torch.cat(J_g, dim=0)
    Jt_g = J_g.transpose(1, 2)
    JtJ_g = Jt_g @ J_g
    norm_det_JtJ_g = torch.abs(torch.det(JtJ_g))
    log_multiplier = -1/2 * torch.log(norm_det_JtJ_g)
    if nf is not None:
        log_prob_latent = nf.log_prob(z_sample.to(nf.device)).to(log_multiplier) - np.log(divider)
    elif gmm is not None:
        log_prob_latent = torch.tensor(gmm._estimate_log_prob_resp(z_sample.cpu())[0]).to(log_multiplier) - np.log(divider)
    return log_prob_latent + log_multiplier

def eval_log_likelihood(model, loader, split=200, n_aug=10):
    original_device = model.device
    lat_dataset = LatentDataset(loader.dataset, model, n_aug=n_aug)
    # model = model.to(torch.device('cpu'))
    z = lat_dataset.z_data
    w = lat_dataset.w_data
    gmm, _  = get_optimized_GMM_latent(model, loader, n_aug=n_aug, print_prob=False)
    log_likelihood = cal_log_p_ambient(model, z, w, divider=1, gmm=gmm, split=split)
    # model = model.to(original_device)
    return log_likelihood

def cal_entropy(
    model, train_loader, val_loader=None, group=PlanarMobileRobot(),
    nf=None,
    gmm=None,
    unseen=True, sample_size=1000, only_successful=False, **kwargs):
    
    if nf is not None:
        gmm=None
    elif gmm is None:
        gmm = get_GMM(model, train_loader, val_loader=val_loader, **kwargs)
    else:
        pass
    if only_successful:
        traj_sample, w_sample, z_sample, success_rate = get_traj_sample(
            model, train_loader, sample_size=sample_size,
            nf=nf,
            gmm=gmm,
            unseen=unseen, group=group, #val_loader=val_loader,
            only_successful=only_successful, **kwargs,)
        divider = success_rate
    else:
        w_data = train_loader.dataset.w_data.to(model.device)
        w_sample = get_w_sample(
            unseen=unseen, group=group, 
            sample_size=sample_size, w_data=w_data
            ).to(model.device)
        z_sample = torch.from_numpy(gmm.sample(sample_size)[0]).to(w_sample)
        divider = 1.
    if success_rate < 0.1:
        return 0
    log_p_ambient = cal_log_p_ambient(
        model, z_sample, w_sample, divider,
        gmm=gmm, nf=nf)
    entropy = -torch.mean(log_p_ambient)
    return entropy

def multimodel_modulation_fig_w(model_list, w, dataloader, n_mod=5,
                                grid=True, locater=1, wall=True,):
    num_model = len(model_list)
    fig = plt.figure(figsize=(6 * num_model, 6))
    for i, model in enumerate(model_list):
        model = model.cpu()
        gmm = get_GMM(model, dataloader)
        ax = fig.add_subplot(1, num_model, i + 1)
        ax = plot_2d_spline_modulation_axis(model, w, ax, gmm,
                                            grid, locater,
                                            n_mod, wall)
    return fig

def multimodel_modulation_fig(model_list, dataloader, unseen=True, n_mod=5,
                                grid=True, locater=1, wall=True, **kwargs):
    w_sample = get_w_sample(unseen=unseen, sample_size=1, **kwargs)
    fig = multimodel_modulation_fig_w(model_list, w_sample, dataloader, n_mod,
                                        grid, locater, wall,)
    return fig
