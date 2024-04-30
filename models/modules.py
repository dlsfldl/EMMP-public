import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import numpy as np
from utils import LieGroup_torch as lie

def get_activation(s_act):
    if s_act == "relu":
        return nn.ReLU(inplace=True)
    elif s_act == "sigmoid":
        return nn.Sigmoid()
    elif s_act == "softplus":
        return nn.Softplus()
    elif s_act == "linear":
        return None
    elif s_act == "tanh":
        return nn.Tanh()
    elif s_act == "leakyrelu":
        return nn.LeakyReLU(0.2, inplace=True)
    elif s_act == "softmax":
        return nn.Softmax(dim=1)
    elif s_act == "selu":
        return nn.SELU()
    elif s_act == "elu":
        return nn.ELU()
    else:
        raise ValueError(f"Unexpected activation: {s_act}")

class FC_SE3(nn.Module):
    def __init__(
        self,
        in_chan=2,
        out_chan=480*12,
        l_hidden=None,
        activation=None,
        out_activation=None,
        **kwargs,
    ):
        super().__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.len_traj = int(out_chan / 12)
        l_neurons = l_hidden + [6 * self.len_traj]
        if len(activation) > len(l_hidden):
            activation = activation[:len(l_hidden)]
        elif len(activation) < len(l_hidden):
            while len(activation) < len(l_hidden):
                activation.append(activation[-1])
        activation = activation + [out_activation]
        # print(activation, l_neurons)
        
        l_layer = []
        prev_dim = in_chan
        for [n_hidden, act] in (zip(l_neurons, activation)):
            l_layer.append(nn.Linear(prev_dim, n_hidden))
            act_fn = get_activation(act)
            if act_fn is not None:
                l_layer.append(act_fn)
            prev_dim = n_hidden

        self.net = nn.Sequential(*l_layer)

    def forward(self, x):
        batch_size = len(x)
        y_local = self.net(x).view(-1, 6)
        R = lie.exp_so3(y_local[:, :3]).reshape(-1, 3, 3)
        p = (y_local[:, 3:]).reshape(-1, 3, 1)
        y_global = torch.cat([R, p], dim=2).reshape(batch_size, -1)
        # y_global = lie.exp_se3(y_local, dim12=True).reshape(batch_size, -1)
        return y_global

class FC_vec(nn.Module):
    def __init__(
        self,
        in_chan=2,
        out_chan=1,
        l_hidden=None,
        activation=None,
        out_activation=None,
    ):
        super().__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        l_neurons = l_hidden + [out_chan]
        if len(activation) > len(l_hidden):
            activation = activation[:len(l_hidden)]
        elif len(activation) < len(l_hidden):
            while len(activation) < len(l_hidden):
                activation.append(activation[-1])
        activation = activation + [out_activation]

        l_layer = []
        prev_dim = in_chan
        for [n_hidden, act] in (zip(l_neurons, activation)):
            l_layer.append(nn.Linear(prev_dim, n_hidden))
            act_fn = get_activation(act)
            if act_fn is not None:
                l_layer.append(act_fn)
            prev_dim = n_hidden

        self.net = nn.Sequential(*l_layer)

    def forward(self, x):
        # 221209 testing!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if len(x.shape) > 2:
            batch_size = len(x)
            x = x.reshape(batch_size, -1)
        return self.net(x)

class IsotropicGaussian(nn.Module):
    """Isotripic Gaussian density function paramerized by a neural net.
    standard deviation is a free scalar parameter"""

    def __init__(self, net, sigma=1):
        super().__init__()
        self.net = net
        self.in_chan = self.net.in_chan
        self.out_chan = self.net.out_chan
        sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float))
        self.register_parameter("sigma", sigma)

    def log_likelihood(self, x, z, *args):
        decoder_out = self.net(z, *args)
        D = torch.prod(torch.tensor(x.shape[1:]))
        sig = self.sigma
        const = -D * 0.5 * torch.log(
            2 * torch.tensor(np.pi, dtype=torch.float32)
        ) - D * torch.log(sig)
        loglik = const - 0.5 * ((x - decoder_out) ** 2).view((x.shape[0], -1)).sum(
            dim=1
        ) / (sig ** 2)
        return loglik

    def forward(self, z, *args):
        return self.net(z, *args)

    def sample(self, z):
        x_hat = self.net(z)
        return x_hat + torch.randn_like(x_hat) * self.sigma