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
    IsotropicGaussian,
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
    return net

def get_ae(data, **model_cfg):
    x_dim = model_cfg['x_dim']
    z_dim = model_cfg['z_dim']
    arch = model_cfg["arch"]
    recon_loss_fn_tr = model_cfg.get('recon_loss_fn_tr', 'MSE_loss')
    input_dict = {}
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
    elif data == 'ToySpline2d':
        group = PlanarMobileRobot()
    return group


def _get_model_instance(name):
    try:
        return {
            "ae": get_ae,
            "vae": get_ae,
            "mmp_vae": get_ae,
            "mmp_ae":get_ae,
            "emmp_vae": get_ae,
            "emmp_ae": get_ae,
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