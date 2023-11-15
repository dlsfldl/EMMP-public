import os
from omegaconf import OmegaConf
from functools import partial
import torch
import torch.nn as nn

from models.ae import (
    AE,
    VAE,
    MMP_AE,
    MMP_VAE,
    EMMP_VAE,
    EMMP_AE,
)

from models.modules import (
    FC_SE3,
    FC_vec,
    FC_vec_rescale,
    IsotropicGaussian,
    TCVAE_encoder,
    TCVAE_decoder,
    TCVAE_decoder_SE3,
    Conv1d_encoder,
)

from models.groups import (
    PouringGroup,
    PlanarMobileRobot,
)

# from models import Latent_sampler

def get_net(in_dim, out_dim, **kwargs):
    if kwargs["arch"] == "fc_vec":
        l_hidden = kwargs["l_hidden"]
        activation = kwargs["activation"]
        out_activation = kwargs["out_activation"]
        net = FC_vec(
            in_chan=in_dim,
            out_chan=out_dim,
            l_hidden=l_hidden,
            activation=activation,
            out_activation=out_activation,
        )
    elif kwargs["arch"] == "fc_se3":
        l_hidden = kwargs["l_hidden"]
        activation = kwargs["activation"]
        out_activation = kwargs["out_activation"]
        net = FC_SE3(
            in_chan=in_dim,
            out_chan=out_dim,
            l_hidden=l_hidden,
            activation=activation,
            out_activation=out_activation,
        )
    elif kwargs["arch"] == "fc_vec_rescale":
        rescale = kwargs["rescale"]
        l_hidden = kwargs["l_hidden"]
        activation = kwargs["activation"]
        out_activation = kwargs["out_activation"]
        net = FC_vec_rescale(
            in_chan=in_dim,
            out_chan=out_dim,
            l_hidden=l_hidden,
            activation=activation,
            out_activation=out_activation,
            rescale=rescale
        )
    elif kwargs["arch"] == "tcvae_tcn_encoder":
        x_dim = in_dim
        z_dim = out_dim
        FC_l_hidden = kwargs['FC_l_hidden']
        FC_activation = kwargs['FC_activation']
        FC_out_activation = kwargs['FC_out_activation']
        TCN_kernel_size = kwargs['TCN_kernel_size']
        TCN_num_channel_list = kwargs['TCN_num_channel_list']
        time_step = kwargs['time_step']
        net = TCVAE_encoder(
            x_dim=x_dim,
            z_dim=z_dim,
            FC_l_hidden=FC_l_hidden,
            FC_activation=FC_activation,
            FC_out_activation=FC_out_activation,
            TCN_kernel_size=TCN_kernel_size,
            TCN_num_channel_list=TCN_num_channel_list,
            time_step=time_step
        )
    elif kwargs["arch"] == "tcvae_tcn_decoder":
        x_dim = in_dim
        z_dim = out_dim
        w_dim = kwargs['w_dim']
        FCz_l_hidden = kwargs['FCz_l_hidden']
        FCz_activation = kwargs['FCz_activation']
        FCz_out_dim = kwargs['FCz_out_dim']
        FCz_out_activation = kwargs['FCz_out_activation']
        FCw_l_hidden = kwargs['FCw_l_hidden']
        FCw_activation = kwargs['FCw_activation']
        FCw_out_dim = kwargs['FCw_out_dim']
        FCw_out_activation = kwargs['FCw_out_activation']
        FC_l_hidden = kwargs['FC_l_hidden']
        FC_activation = kwargs['FC_activation']
        FC_out_activation = kwargs['FC_out_activation']
        TCN_kernel_size = kwargs['TCN_kernel_size']
        TCN_num_channel_list = kwargs['TCN_num_channel_list']
        time_step = kwargs['time_step']
        net = TCVAE_decoder(
            x_dim=x_dim,
            z_dim=z_dim,
            w_dim=w_dim,
            FCz_l_hidden=FCz_l_hidden,
            FCz_activation=FCz_activation,
            FCz_out_dim=FCz_out_dim,
            FCz_out_activation=FCz_out_activation,
            FCw_l_hidden=FCw_l_hidden,
            FCw_activation=FCw_activation,
            FCw_out_dim=FCw_out_dim,
            FCw_out_activation=FCw_out_activation,
            FC_l_hidden=FC_l_hidden,
            FC_activation=FC_activation,
            FC_out_activation=FC_out_activation,
            TCN_kernel_size=TCN_kernel_size,
            TCN_num_channel_list=TCN_num_channel_list,
            time_step=time_step
        )
    elif kwargs["arch"] == 'tcvae_tcn_decoder_se3':
        x_dim = in_dim
        z_dim = out_dim
        w_dim = kwargs['w_dim']
        FCz_l_hidden = kwargs['FCz_l_hidden']
        FCz_activation = kwargs['FCz_activation']
        FCz_out_dim = kwargs['FCz_out_dim']
        FCz_out_activation = kwargs['FCz_out_activation']
        FCw_l_hidden = kwargs['FCw_l_hidden']
        FCw_activation = kwargs['FCw_activation']
        FCw_out_dim = kwargs['FCw_out_dim']
        FCw_out_activation = kwargs['FCw_out_activation']
        FC_l_hidden = kwargs['FC_l_hidden']
        FC_activation = kwargs['FC_activation']
        FC_out_activation = kwargs['FC_out_activation']
        TCN_kernel_size = kwargs['TCN_kernel_size']
        TCN_num_channel_list = kwargs['TCN_num_channel_list']
        time_step = kwargs['time_step']
        net = TCVAE_decoder_SE3(
            x_dim=x_dim,
            z_dim=z_dim,
            w_dim=w_dim,
            FCz_l_hidden=FCz_l_hidden,
            FCz_activation=FCz_activation,
            FCz_out_dim=FCz_out_dim,
            FCz_out_activation=FCz_out_activation,
            FCw_l_hidden=FCw_l_hidden,
            FCw_activation=FCw_activation,
            FCw_out_dim=FCw_out_dim,
            FCw_out_activation=FCw_out_activation,
            FC_l_hidden=FC_l_hidden,
            FC_activation=FC_activation,
            FC_out_activation=FC_out_activation,
            TCN_kernel_size=TCN_kernel_size,
            TCN_num_channel_list=TCN_num_channel_list,
            time_step=time_step
        )
        
    elif kwargs["arch"] == 'conv1d':
        x_dim = in_dim
        z_dim = out_dim
        time_step = kwargs['time_step']
        fc_hidden = kwargs['fc_hidden']
        fc_activation = kwargs['fc_activation']
        out_activation = kwargs['out_activation']
        conv_hidden = kwargs['conv_hidden']
        conv_act = kwargs['conv_act']
        kernel_size = kwargs['kernel_size']
        stride = kwargs['stride']
        net = Conv1d_encoder(
            x_dim=x_dim,
            z_dim=z_dim,
            time_step=time_step,
            fc_hidden=fc_hidden,
            fc_activation=fc_activation,
            out_activation=out_activation,
            conv_hidden=conv_hidden,
            conv_act=conv_act,
            kernel_size=kernel_size,
            stride=stride,
        )
    return net

def get_ae(data, **model_cfg):
    x_dim = model_cfg['x_dim']
    z_dim = model_cfg['z_dim']
    arch = model_cfg["arch"]
    recon_loss_fn_tr = model_cfg.get('recon_loss_fn_tr', 'MSE_loss')
    input_dict = {}
    if 'tc' in arch:
        w_dim = model_cfg["w_dim"]
        if 'iso_reg' in model_cfg.keys():
            if model_cfg['iso_reg'] > 0:
                iso_reg = model_cfg['iso_reg']
                input_dict['iso_reg'] = iso_reg
                # print(f'input_dict = {input_dict}')
        if 'reg_type' in model_cfg.keys():     
            reg_type = model_cfg['reg_type']
            alpha = model_cfg.get("alpha", 1.)
            reg_optimizer = model_cfg.get('reg_optimizer', None)
            input_dict['alpha'] = alpha
            input_dict['reg_type'] = reg_type
            input_dict['reg_optimizer'] = reg_optimizer
            if reg_type == 'auxillary':
                reg_net = get_net(in_dim=z_dim, out_dim=w_dim, **model_cfg["reg_net"])
                input_dict['reg_net'] = reg_net
            elif reg_type == 'independence':
                pass
            else:
                print("reg_type is not in the list. Running with no regulerization..")
    if arch == "ae":
        encoder = get_net(in_dim=x_dim, out_dim=z_dim, **model_cfg["encoder"])
        decoder = get_net(in_dim=z_dim, out_dim=x_dim, **model_cfg["decoder"])
        model = AE(encoder, decoder, recon_loss_fn_tr=recon_loss_fn_tr)
    elif arch == "vae":
        encoder = get_net(in_dim=x_dim, out_dim=z_dim * 2, **model_cfg["encoder"])
        decoder = get_net(in_dim=z_dim, out_dim=x_dim, **model_cfg["decoder"])
        model = VAE(encoder, IsotropicGaussian(decoder), recon_loss_fn_tr=recon_loss_fn_tr)
    elif arch == "mmp_ae":
        w_dim = model_cfg["w_dim"]
        encoder = get_net(in_dim=x_dim, out_dim=z_dim, **model_cfg["encoder"])
        decoder = get_net(in_dim=z_dim+w_dim, out_dim=x_dim, **model_cfg['decoder'])
        group = get_group(data)
        model = MMP_AE(
            encoder, decoder, group,
                recon_loss_fn_tr=recon_loss_fn_tr, **input_dict,
        )
    elif arch == "mmp_vae":
        w_dim = model_cfg["w_dim"]
        encoder = get_net(in_dim=x_dim, out_dim=z_dim * 2, **model_cfg["encoder"])
        decoder = get_net(in_dim=z_dim+w_dim, out_dim=x_dim, **model_cfg['decoder'])
        beta = model_cfg.get("beta", 1.0)
        group = get_group(data)
        model = MMP_VAE(
            encoder, IsotropicGaussian(decoder), group, beta=beta,
                recon_loss_fn_tr=recon_loss_fn_tr, **input_dict
        )
    elif arch =="tcvae":
        w_dim = model_cfg["w_dim"]
        encoder = get_net(in_dim=x_dim, out_dim=z_dim, **model_cfg["encoder"])
        decoder = get_net(in_dim=x_dim, out_dim=z_dim, w_dim=w_dim, **model_cfg['decoder'])
        beta = model_cfg.get("beta", 1.0)
        group = get_group(data)
        model = MMP_VAE(
            encoder, IsotropicGaussian(decoder), group, beta=beta,
                recon_loss_fn_tr=recon_loss_fn_tr, **input_dict,
        )
    elif arch == "emmp_ae":
        w_dim = model_cfg["w_dim"]
        encoder = get_net(in_dim=x_dim, out_dim=z_dim, **model_cfg["encoder"])
        decoder = get_net(in_dim=z_dim+w_dim, out_dim=x_dim, **model_cfg['decoder'])
        group = get_group(data)
        model = EMMP_AE(
            encoder, decoder, group,
                recon_loss_fn_tr=recon_loss_fn_tr, **input_dict,
        )
    elif arch == "emmp_vae":
        w_dim = model_cfg["w_dim"]
        encoder = get_net(in_dim=x_dim, out_dim=z_dim * 2, **model_cfg["encoder"])
        decoder = get_net(in_dim=z_dim+w_dim, out_dim=x_dim, **model_cfg['decoder'])
        beta = model_cfg.get("beta", 1.0)
        group = get_group(data)
        model = EMMP_VAE(
            encoder, IsotropicGaussian(decoder), group, beta=beta,
                recon_loss_fn_tr=recon_loss_fn_tr, **input_dict
        )
    return model

def get_model(cfg, *args, version=None, **kwargs):
    # cfg can be a whole config dictionary or a value of a key 'model' in the config dictionary (cfg['model']).
    if "model" in cfg:
        model_dict = cfg["model"]
        training_dict = cfg['training']
    elif "arch" in cfg:
        model_dict = cfg
    else:
        raise ValueError(f"Invalid model configuration dictionary: {cfg}")
    name = model_dict["arch"]
    data = cfg["data"]["training"]["dataset"]
    if "reg_optimizer" in training_dict.keys():
        model_dict['reg_optimizer'] = training_dict['reg_optimizer'] 
    model = _get_model_instance(name)
    model = model(data, **model_dict)
    return model

def get_group(data):
    if data == 'Pouring':
        group = PouringGroup()
    elif data == 'simple_Pouring':
        group = PouringGroup()
    elif data == 'ToySpline2d':
        group = PlanarMobileRobot()
    return group


def _get_model_instance(name):
    try:
        return {
            "ae": get_ae,
            "vae": get_ae,
            "irvae": get_ae,
            "tcvae": get_ae,
            "tcae":get_ae,
            "equivariant_tcvae": get_ae,
            "equivariant_tcae": get_ae,
            "tcvae_tcn":get_ae,
            "equivariant_tcvae_tcn": get_ae,
        }[name]
    except:
        raise ("Model {} not available".format(name))

def load_pretrained(identifier, config_file, ckpt_file, root='pretrained', **kwargs):
    """
    load pre-trained model.
    identifier: '<model name>/<run name>'. e.g. 'ae_mnist/z16'
    config_file: name of a config file. e.g. 'ae.yml'
    ckpt_file: name of a model checkpoint file. e.g. 'model_best.pth'
    root: path to pretrained directory
    """
    config_path = os.path.join(root, identifier, config_file)
    ckpt_path = os.path.join(root, identifier, ckpt_file)
    cfg = OmegaConf.load(config_path)
    model = get_model(cfg)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'model_state' in ckpt:
        ckpt = ckpt['model_state']
    model.load_state_dict(ckpt)
    
    return model, cfg