from re import X
import numpy as np
import torch
import torch.nn as nn
import copy
from optimizers import get_optimizer
from functools import partial

import matplotlib.pyplot as plt
from utils import visualization
from utils.utils import label_to_color, figure_to_array, PD_metric_to_ellipse
from torch.distributions.multivariate_normal import MultivariateNormal
from models.base import BaseModel
from evaluation import loss_fn


pallete = ['#377eb8', '#ff7f00', '#4daf4a',
            '#8d5daf', '#9c7c7c', '#23a696', 
            '#df5956', '#f6b45e', '#f5d206',
            '#69767c', '#e5e58b', '#e98074',
            '#6897bb', 
            ]

class AE(BaseModel):
    def __init__(self, encoder, decoder, recon_loss_fn_tr='MSE_loss', recon_loss_fn_val='RMSE_loss'):
        super(AE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.recon_loss_fn_tr=getattr(loss_fn, recon_loss_fn_tr)
        self.recon_loss_fn_val=getattr(loss_fn, recon_loss_fn_val)
        
    def encode(self, x, w=None):
        batch_size = len(x)
        x = x.reshape(batch_size, -1)
        return self.encoder(x)

    def decode(self, z, w=None):
        return self.decoder(z)

    def forward(self, x, w=None):
        z = self.encode(x, w)
        recon = self.decode(z, w)
        return recon
    
    def loss_train(self, x, w=None):
        batch_size = len(x)
        x = x.reshape(batch_size, -1)
        recon = self(x, w)
        recon_loss = self.recon_loss_fn_tr(recon, x)
        #((recon - x) ** 2).view(len(x), -1).mean(dim=1).mean()
        loss_dict = {}
        loss_dict["recon_loss_MSE"] = recon_loss
        return loss_dict
    
    def loss_val(self, x, w=None):
        batch_size = len(x)
        x = x.reshape(batch_size, -1)
        recon = self(x, w)
        recon_loss = self.recon_loss_fn_val(recon, x)
        #((recon - x) ** 2).view(len(x), -1).mean(dim=1).mean()
        loss_dict = {}
        loss_dict["recon_loss_RMSE"] = recon_loss
        return loss_dict

    def train_step(self, x, w=None, optimizer=None, **kwargs):
        loss_dict = self.loss_train(x, w)
        loss = loss_dict["recon_loss_MSE"]
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return {"loss": loss.item()}
    
    def validation_step(self, x, w=None, **kwargs):
        loss_dict = self.loss_val(x, w)
        recon_loss = loss_dict["recon_loss_RMSE"]
        loss = recon_loss
        
        return {"loss": loss.item()}

    def visualization_step(self, train_loader, val_loader, device, *args, **kwargs):
        fig_dict = {}
        type_list = kwargs.get('type', [])
        for plot_type in type_list:
            fig_dict = getattr(
                visualization, plot_type)(
                    self, train_loader, val_loader, device, fig_dict)
            
        if len(fig_dict) > 0:
            return fig_dict
        pass


class VAE(AE):
    def __init__(
        self, encoder, decoder, recon_loss_fn_tr='MSE_loss', recon_loss_fn_val='RMSE_loss'
    ):
        super().__init__(encoder, decoder, recon_loss_fn_tr=loss_fn, recon_loss_fn_val=recon_loss_fn_val)

    def encode(self, x, w=None):
        z = self.encoder(x)
        half_chan = int(z.shape[1] / 2)
        return z[:, :half_chan]

    def decode(self, z, w=None):
        return self.decoder(z)

    def sample_latent(self, z):
        half_chan = int(z.shape[1] / 2)
        mu, log_sig = z[:, :half_chan], z[:, half_chan:]
        eps = torch.randn(*mu.shape, dtype=torch.float32)
        eps = eps.to(z.device)
        return mu + torch.exp(log_sig) * eps

    def kl_loss(self, z):
        """analytic (positive) KL divergence between gaussians
        KL(q(z|x) | p(z))"""
        half_chan = int(z.shape[1] / 2)
        mu, log_sig = z[:, :half_chan], z[:, half_chan:]
        mu_sq = mu ** 2
        sig_sq = torch.exp(log_sig) ** 2
        kl = mu_sq + sig_sq - torch.log(sig_sq) - 1
        return 0.5 * torch.sum(kl.view(len(kl), -1), dim=1)
    
    def loss(self, x, w=None):
        batch_size = len(x)
        x = x.reshape(batch_size, -1)
        z = self.encoder(x)
        z_sample = self.sample_latent(z)
        recon = self.decode(z_sample, w)
        recon_loss = self.recon_loss_fn_tr(recon, x)
        
        ((recon - x) ** 2).view(len(x), -1).mean(dim=1).mean()
        nll = - self.decoder.log_likelihood(x, z_sample).mean()
        kl_loss = self.kl_loss(z).mean()
        
        loss_dict = {}
        loss_dict["recon_loss"] = recon_loss
        loss_dict["nll"] = nll
        loss_dict["kl_loss"] = kl_loss
        
        return loss_dict
    
    def train_step(self, x, w=None, optimizer=None, **kwargs):
        loss_dict = self.loss(x, w)      
        nll = loss_dict["nll"]; kl_loss = loss_dict["kl_loss"]  
        loss = nll + kl_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {
            "loss": loss.item(),
            # "nll_": nll.item(),
            # "kl_loss_": kl_loss.mean(),
            # "sigma_": self.decoder.sigma.item(),
        }
        
    def validation_step(self, x, w=None, **kwargs):
        loss_dict = self.loss(x, w)
        recon_loss = loss_dict["recon_loss"]
        nll = loss_dict["nll"]; kl_loss = loss_dict["kl_loss"]
        
        loss = recon_loss

        return {
            "loss": loss.item(),
            "nll_": nll.item(),
            "kl_loss_": kl_loss.mean(),
            # "sigma_": self.decoder.sigma.item(),
        }

class MMP_VAE(VAE):
    def __init__(
        self, encoder, decoder, group, beta=1.0, 
        recon_loss_fn_tr='MSE_loss', recon_loss_fn_val='RMSE_loss', 
        **regularization_dict
    ):
        super(VAE, self).__init__(encoder, decoder, recon_loss_fn_tr=recon_loss_fn_tr, recon_loss_fn_val=recon_loss_fn_val)
        self.beta = beta
        self.group = group
        if 'reg_type' in regularization_dict.keys():
            self.reg_type = regularization_dict['reg_type']
            self.alpha = regularization_dict['alpha']
            if self.reg_type == 'auxillary':
                self.reg_net = regularization_dict['reg_net']
                self.reg_optimizer = get_optimizer(regularization_dict['reg_optimizer'], self.reg_net.parameters())
            if self.reg_type == 'independence':
                pass
        else:
            self.reg_type = 'None'
    @staticmethod
    def list2gen(x):
        for element in x:
            yield element

    def parameters(self):
        return self.list2gen(list(self.encoder.parameters()) + list(self.decoder.parameters()))
    
    def decode(self, z, w):
        '''outputs the mean of p(x|h, w)'''
        # # For extended task parameter learning. 
        extended_w = self.group.extend_w(w)
        return self.decoder(torch.cat((z, extended_w), 1))

    def auxilary_loss(self, z, w):
        MSELoss = torch.nn.MSELoss()
        aux_output = self.reg_net(z)
        extended_w = self.group.extend_w(w)
        return MSELoss(aux_output, extended_w)
    
    def independence_loss(self, z):
        MSELoss = torch.nn.MSELoss()
        w_random = self.group._random_task(len(z)).to(z)
        z_pred = self.encode(self.decode(z, w_random), w_random)
        znorm = z.norm(dim=1).mean()
        indep_loss = MSELoss(z, z_pred)/znorm
        return indep_loss
    
    def loss(self, x, w=None, val=False):
        batch_size = len(x)
        x = x.reshape(batch_size, -1)
        z_mean_var = self.encoder(x)
        z_sample = self.sample_latent(z_mean_var)
        extended_w = self.group.extend_w(w)
        recon = self.decode(z_sample, w)
        
        recon_loss = self.recon_loss_fn_tr(recon, x)
        D = torch.prod(torch.tensor(x.shape[1:]))
        sig = self.decoder.sigma
        const = -D * 0.5 * torch.log(
            2 * torch.tensor(np.pi, dtype=torch.float32)
        ) - D * torch.log(sig)
        loglik = const - 0.5 * ((x - recon) ** 2).view((x.shape[0], -1)).sum(
            dim=1
        ) / (sig ** 2)
        nll = -loglik.mean()
        kl_loss = self.kl_loss(z_mean_var).mean()
        
        loss_dict = {}
        loss_dict["recon_loss"] = recon_loss
        loss_dict["nll"] = nll
        loss_dict["kl_loss"] = kl_loss
        
        if self.reg_type == 'auxillary':
            half_chan = int(z_mean_var.shape[1] / 2)
            z_mean = z_mean_var[:, :half_chan]
            aux_loss = self.auxilary_loss(z_mean, w)
            loss_dict["aux_loss"] = aux_loss
        if val == False:
            if self.reg_type == 'independence':
                half_chan = int(z_mean_var.shape[1] / 2)
                z_mean = z_mean_var[:, :half_chan]
                indep_loss = self.independence_loss(z_mean)
                loss_dict["indep_loss"] = indep_loss
        else:
            half_chan = int(z_mean_var.shape[1] / 2)
            z_mean = z_mean_var[:, :half_chan]
            indep_loss = self.independence_loss(z_mean)
            loss_dict["indep_loss"] = indep_loss
            
            
        return loss_dict
    
    def train_step(self, x, w=None, optimizer=None, **kwargs):
        optimizer.zero_grad()
        loss_dict = self.loss(x, w)      
        nll = loss_dict["nll"]; kl_loss = loss_dict["kl_loss"]
        recon_loss = loss_dict["recon_loss"]
        max_grad_norm = 1
        norm_type = 2
        base_loss = nll + self.beta * kl_loss
        
        if self.reg_type == 'auxillary':
            aux_loss = loss_dict["aux_loss"]
            loss = base_loss - self.alpha * aux_loss
            loss.backward()
            if max_grad_norm > 0:
                for group in optimizer.param_groups:
                    torch.nn.utils.clip_grad_norm_(group['params'], max_grad_norm, norm_type)
            optimizer.step()
            self.reg_optimizer.zero_grad()
            z_mean = self.encode(x, w)
            aux_loss = self.auxilary_loss(z_mean, w)
            aux_loss.backward()
            if max_grad_norm > 0:
                for group in self.reg_optimizer.param_groups:
                    torch.nn.utils.clip_grad_norm_(group['params'], max_grad_norm, norm_type)
            self.reg_optimizer.step()
        elif self.reg_type == 'independence':
            indep_loss = loss_dict["indep_loss"]
            loss = base_loss + self.alpha * indep_loss
            loss.backward()
            if max_grad_norm > 0:
                for group in optimizer.param_groups:
                    torch.nn.utils.clip_grad_norm_(group['params'], max_grad_norm, norm_type)
            optimizer.step()
        else:
            loss = base_loss
            loss.backward()
            if max_grad_norm > 0:
                for group in optimizer.param_groups:
                    torch.nn.utils.clip_grad_norm_(group['params'], max_grad_norm, norm_type)
            optimizer.step()
        
        sigma_threshold = 0.09
        if self.decoder.sigma.data < sigma_threshold:
            self.decoder.sigma.data = torch.tensor(sigma_threshold).to(x)
        
        d_train = {
            "loss": loss.item(),
            "recon_loss_": recon_loss.item(),
            "nll_": nll.item(),
            "kl_loss_": kl_loss.mean(),
            "sigma_": self.decoder.sigma.item(),
        }
        if self.reg_type == 'auxillary':
            d_train['aux_loss_'] = aux_loss.item()
        elif self.reg_type == 'independence':
            d_train['indep_loss_'] = indep_loss.item()
        return d_train
    
    def validation_step(self, x, w=None, **kwargs):
        loss_dict = self.loss(x, w, val=True)
        recon_loss = loss_dict["recon_loss"]
        nll = loss_dict["nll"]; kl_loss = loss_dict["kl_loss"]
        loss = recon_loss
        d_val = {
            "loss": loss.item(),
            "val_recon_loss_": recon_loss.item(),
            "nll_": nll.item(),
            "kl_loss_": kl_loss.mean(),
        }
        if self.reg_type == 'auxillary':
            aux_loss = loss_dict["aux_loss"]
        indep_loss = loss_dict["indep_loss"]
        d_val['indep_loss_'] = indep_loss.item()
        return d_val
        
class MMP_AE(AE):
    def __init__(
        self, encoder, decoder, group, 
        recon_loss_fn_tr='MSE_loss', recon_loss_fn_val='RMSE_loss',
        **regularization_dict,
        # aux_net=None, alpha=0, aux_optimizer=None
    ):
        super().__init__(encoder, decoder, recon_loss_fn_tr=recon_loss_fn_tr, recon_loss_fn_val=recon_loss_fn_val)
        self.group = group
            
        
        if 'reg_type' in regularization_dict.keys():
            self.reg_type = regularization_dict['reg_type']
            self.alpha = regularization_dict['alpha']
            if self.reg_type == 'auxillary':
                self.reg_net = regularization_dict['reg_net']
                self.reg_optimizer = get_optimizer(regularization_dict['reg_optimizer'], self.reg_net.parameters())
            if self.reg_type == 'independence':
                pass
        else:
            self.reg_type = 'None'
    @staticmethod
    def list2gen(x):
        for element in x:
            yield element

    def parameters(self):
        return self.list2gen(list(self.encoder.parameters()) + list(self.decoder.parameters()))
    
    def decode(self, z, w):
        '''outputs the mean of p(x|h, w)'''
        # # For extended task parameter learning. 
        extended_w = self.group.extend_w(w)
        
        return self.decoder(torch.cat((z, extended_w), dim=1))
        # return self.decoder(torch.cat((z, w), 1))

    def auxilary_loss(self, z, w):
        # L1loss = torch.nn.L1Loss()
        MSELoss = torch.nn.MSELoss()
        aux_output = self.reg_net(z)
        extended_w = self.group.extend_w(w)
        return MSELoss(aux_output, extended_w)
    
    def independence_loss(self, z):
        MSELoss = torch.nn.MSELoss()
        w_random = self.group._random_task(len(z)).to(z)
        z_pred = self.encode(self.decode(z, w_random), w_random)
        # znorm = z.norm(dim=1).mean()
        # indep_loss = MSELoss(z, z_pred)/znorm
        znorm = z.norm(dim=1).mean()
        indep_loss = MSELoss(z, z_pred)/znorm
        return indep_loss
        
    
    def loss(self, x, w=None, val=False):
        z = self.encode(x)
        recon = self.decode(z, w)
        recon_loss = self.recon_loss_fn_tr(recon, x)
        # extended_w = self.group.extend_w(w)
        
        loss_dict = {}
        loss_dict["recon_loss"] = recon_loss
        
        if self.reg_type == 'auxillary':
            aux_loss = self.auxilary_loss(z, w)
            loss_dict["aux_loss"] = aux_loss
        if val == False:    
            if self.reg_type == 'independence':
                indep_loss = self.independence_loss(z)
                loss_dict["indep_loss"] = indep_loss
        else:
            indep_loss = self.independence_loss(z)
            loss_dict["indep_loss"] = indep_loss
        return loss_dict
    
    def train_step(self, x, w=None, optimizer=None, **kwargs):
        optimizer.zero_grad()
        loss_dict = self.loss(x, w)
        recon_loss = loss_dict["recon_loss"]
        max_grad_norm = 1
        norm_type = 2
        base_loss = recon_loss
        
        if self.reg_type == 'auxillary':
            aux_loss = loss_dict["aux_loss"]
            loss = base_loss - self.alpha * aux_loss
            loss.backward()
            if max_grad_norm > 0:
                for group in optimizer.param_groups:
                    torch.nn.utils.clip_grad_norm_(group['params'], max_grad_norm, norm_type)
            optimizer.step()
            self.reg_optimizer.zero_grad()
            z = self.encode(x, w)
            aux_loss = self.auxilary_loss(z, w)
            aux_loss.backward()
            if max_grad_norm > 0:
                for group in self.reg_optimizer.param_groups:
                    torch.nn.utils.clip_grad_norm_(group['params'], max_grad_norm, norm_type)
            self.reg_optimizer.step()
        elif self.reg_type == 'independence':
            indep_loss = loss_dict["indep_loss"]
            loss = base_loss + self.alpha * indep_loss
            loss.backward()
            if max_grad_norm > 0:
                for group in optimizer.param_groups:
                    torch.nn.utils.clip_grad_norm_(group['params'], max_grad_norm, norm_type)
            optimizer.step()
        else:
            loss = base_loss
            loss.backward()
            if max_grad_norm > 0:
                for group in optimizer.param_groups:
                    torch.nn.utils.clip_grad_norm_(group['params'], max_grad_norm, norm_type)
            optimizer.step()
        
        d_train = {
            "loss": loss.item(),
            "recon_loss_": recon_loss.item(),
        }
        if self.reg_type == 'auxillary':
            d_train['aux_loss_'] = aux_loss.item()
        elif self.reg_type == 'independence':
            d_train['indep_loss_'] = indep_loss.item()
        return d_train
        
    def validation_step(self, x, w=None, **kwargs):
        loss_dict = self.loss(x, w, val=True)
        recon_loss = loss_dict["recon_loss"] 
        
        loss = recon_loss
        d_val = {
            "loss": loss.item(),
            "recon_loss_": recon_loss.item(),
        }
        indep_loss = loss_dict["indep_loss"]
        d_val['indep_loss_'] = indep_loss.item()
        if self.reg_type == 'auxillary':
            aux_loss = loss_dict["aux_loss"]
            d_val['aux_loss_'] = aux_loss.item()
        elif self.reg_type == 'independence':
            indep_loss = loss_dict["indep_loss"]
            d_val['indep_loss_'] = indep_loss.item()

        return d_val

class EMMP_VAE(MMP_VAE):
    def __init__(self, encoder, decoder, group, beta=1.0, 
                recon_loss_fn_tr='MSE_loss', recon_loss_fn_val='RMSE_loss',
                **regularization_dict) -> None:
        super().__init__(encoder, decoder, group, beta=1.0, 
                        recon_loss_fn_tr=recon_loss_fn_tr, recon_loss_fn_val=recon_loss_fn_val,
                        **regularization_dict)

    def encode(self, x, w):
        hat_g = self.group.project_group(w)
        _, x_transf = self.group.action_traj(hat_g, w, x)
        return super().encode(x_transf)

    def decode(self, z, w):
        hat_g = self.group.project_group(w)
        hat_g_inv = self.group.get_inv(hat_g)
        hat_w = self.group.action_task(hat_g, w)
        
        # # For reduced task parameter learning.
        squeezed_hat_w = self.group.squeeze_hat_w(hat_w)
        hat_x = self.decoder(torch.cat((z, squeezed_hat_w), 1))
        
        batch_size = hat_x.size()[0]
        _, hat_x= self.group.action_traj(hat_g_inv, hat_w, hat_x)
        hat_x = hat_x.reshape(batch_size, -1)
        
        return hat_x
    
    def auxilary_loss(self, z, w):
        hat_g = self.group.project_group(w)
        w_0 = self.group.action_task(hat_g, w)
        MSELoss = torch.nn.MSELoss()
        aux_output = self.reg_net(z)
        squeezed_w_0 = self.group.squeeze_hat_w(w_0)
        return MSELoss(aux_output, squeezed_w_0)

    def loss(self, x, w, val=False):
            
        hat_g = self.group.project_group(w)
        _, x_transf = self.group.action_traj(hat_g, w, x)
        
        batch_size = len(x_transf)
        x = x.reshape(batch_size, -1)
        x_transf = x_transf.reshape(batch_size, -1)
            
        z_mean_var = self.encoder(x_transf)
        z_sample = self.sample_latent(z_mean_var)
        recon = self.decode(z_sample, w)
        
        D = torch.prod(torch.tensor(x.shape[1:]))
        sig = self.decoder.sigma
        const = -D * 0.5 * torch.log(
            2 * torch.tensor(np.pi, dtype=torch.float32)
        ) - D * torch.log(sig)
        loglik = const - 0.5 * ((x - recon) ** 2).view((x.shape[0], -1)).sum(
            dim=1
        ) / (sig ** 2)
        nll = -loglik.mean()
        
        recon_loss = self.recon_loss_fn_tr(recon, x)
        D = torch.prod(torch.tensor(x.shape[1:]))
        sig = self.decoder.sigma
        const = -D * 0.5 * torch.log(
            2 * torch.tensor(np.pi, dtype=torch.float32)
        ) - D * torch.log(sig)
        loglik = const - 0.5 * ((x - recon) ** 2).view((x.shape[0], -1)).sum(
            dim=1
        ) / (sig ** 2)
        nll = -loglik.mean()
        kl_loss = self.kl_loss(z_mean_var).mean()
        
        loss_dict = {}
        loss_dict["recon_loss"] = recon_loss
        loss_dict["nll"] = nll
        loss_dict["kl_loss"] = kl_loss
        
        if self.reg_type == 'auxillary':
            half_chan = int(z_mean_var.shape[1] / 2)
            z_mean = z_mean_var[:, :half_chan]
            aux_loss = self.auxilary_loss(z_mean, w)
            loss_dict["aux_loss"] = aux_loss
        if val == False:
            if self.reg_type == 'independence':
                half_chan = int(z_mean_var.shape[1] / 2)
                z_mean = z_mean_var[:, :half_chan]
                indep_loss = self.independence_loss(z_mean)
                loss_dict["indep_loss"] = indep_loss
        else:
            half_chan = int(z_mean_var.shape[1] / 2)
            z_mean = z_mean_var[:, :half_chan]
            indep_loss = self.independence_loss(z_mean)
            loss_dict["indep_loss"] = indep_loss
        return loss_dict

class EMMP_AE(MMP_AE):
    def __init__(self, encoder, decoder, group, 
        recon_loss_fn_tr='MSE_loss', recon_loss_fn_val='RMSE_loss',
        **regularization_dict) -> None:
        super().__init__(encoder, decoder, group, 
                        recon_loss_fn_tr=recon_loss_fn_tr, recon_loss_fn_val=recon_loss_fn_val,
                        **regularization_dict)

    def encode(self, x, w):
        hat_g = self.group.project_group(w)
        _, x_transf = self.group.action_traj(hat_g, w, x)
        
        # 221209 testing!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # batch_size = len(x_transf)
        # x_transf = x_transf.reshape(batch_size, -1)
        
        return super().encode(x_transf)

    def decode(self, z, w):
        hat_g = self.group.project_group(w)
        hat_g_inv = self.group.get_inv(hat_g)
        hat_w = self.group.action_task(hat_g, w)
        
        # # For reduced task parameter learning.
        squeezed_hat_w = self.group.squeeze_hat_w(hat_w)
        hat_x = self.decoder(torch.cat((z, squeezed_hat_w), 1))
        # hat_x = super().decode(z, hat_w)
        
        batch_size = hat_x.size()[0]
        time_step = int(hat_x.size()[1]/12)
        # hat_x = hat_x.reshape(batch_size, time_step, 3, 4)
        _, hat_x= self.group.action_traj(hat_g_inv, hat_w, hat_x)
        hat_x = hat_x.reshape(batch_size, -1)
        
        return hat_x
    
    def auxilary_loss(self, z, w):
        # L1loss = torch.nn.L1Loss()
        L1loss = torch.nn.MSELoss()
        hat_g = self.group.project_group(w)
        hat_g_inv = self.group.get_inv(hat_g)
        hat_w = self.group.action_task(hat_g, w)
        squeezed_hat_w = self.group.squeeze_hat_w(hat_w)
        aux_output = self.reg_net(z)
        return L1loss(aux_output, squeezed_hat_w)
    
    def loss(self, x, w, val=False):
        z = self.encode(x, w)
        recon = self.decode(z, w)
        recon_loss = self.recon_loss_fn_tr(recon, x)
        
        loss_dict = {}
        loss_dict["recon_loss"] = recon_loss
        
        
        if self.reg_type == 'auxillary':
            aux_loss = self.auxilary_loss(z, w)
            loss_dict["aux_loss"] = aux_loss
        
        if val == False:    
            if self.reg_type == 'independence':
                indep_loss = self.independence_loss(z)
                loss_dict["indep_loss"] = indep_loss
        else:
            indep_loss = self.independence_loss(z)
            loss_dict["indep_loss"] = indep_loss
        return loss_dict