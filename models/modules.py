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
    
class FC_vec_rescale(nn.Module):
    def __init__(
        self,
        in_chan=2,
        out_chan=2,
        l_hidden=None,
        activation=None,
        out_activation=None,
        rescale=False
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
        self.rescale_bool = rescale
        if self.rescale_bool:
            self.rescale = nn.utils.weight_norm(Rescale(in_chan))

    def forward(self, x):
        if len(x.shape) > 2:
            batch_size = len(x)
            x = x.reshape(batch_size, -1)
        if self.rescale_bool:
            return self.rescale(torch.tanh(self.net(x)))
        else:
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

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # input shape : (batch * timestep * dim)
        x = x.permute(0, 2, 1)
        x = self.network(x)
        x = x.permute(0, 2, 1)
        return x

class TCVAE_encoder(nn.Module):
    '''
    x is batch * time_step * x_dim
    o=TCN(x) is batch * time_step * x_dim
    o_bar = o.mean(dim=2) is batch * x_dim
    FC(o_bar) is batch * (2*h_dim) where the first h_dim columns are mu(x) and the last h_dim columns are Sigma(x)

    TCVAE_encoder(x) outputs the mean mu_phi(x) and the variance Sigma_phi(x) of manner h
    '''

    def __init__(self, x_dim, z_dim,
    FC_l_hidden, FC_activation, FC_out_activation, 
    TCN_kernel_size, TCN_num_channel_list, time_step
    ):
        super(TCVAE_encoder, self).__init__()

        # TCN_num_channels = [x_dim] * TCN_block_num
        self.x_dim = x_dim
        self.time_step = time_step
        self.TCN = TemporalConvNet(num_inputs=x_dim, num_channels=TCN_num_channel_list, kernel_size=TCN_kernel_size)
        self.FC = FC_vec(in_chan=TCN_num_channel_list[-1], out_chan=z_dim * 2, l_hidden=FC_l_hidden, activation=FC_activation, out_activation=FC_out_activation)

        # For post_training
        self.in_chan = x_dim
        self.out_chan = z_dim * 2
    
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.reshape(-1, self.time_step, self.x_dim)
        o = self.TCN(x)
        o_bar = o.mean(dim=1)
        return self.FC(o_bar)

class TCVAE_decoder(nn.Module):
    '''
    h is batch * h_dim
    w is batch * w_dim
    Encode manner h and task w by FCN
    h_enc is batch * h_dim
    w_enc is batch * w_dim
    phase represents time step, 0, ..., t/T, ..., 1
    Concatenation of h_enc, w_enc and phase is enc_latent, batch * time_step * (h_dim+w_dim+1) 
    '''

    def __init__(self, x_dim, z_dim, w_dim, 
    FCz_l_hidden, FCz_activation, FCz_out_dim, FCz_out_activation, 
    FCw_l_hidden, FCw_activation, FCw_out_dim, FCw_out_activation,
    FC_l_hidden, FC_activation, FC_out_activation,
    TCN_kernel_size, TCN_num_channel_list, time_step
    ):
        super(TCVAE_decoder, self).__init__()
        self.time_step = time_step
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.TCN_num_channel_list = TCN_num_channel_list

        self.TCN = TemporalConvNet(num_inputs=FCz_out_dim + FCw_out_dim + 1, num_channels=TCN_num_channel_list, kernel_size=TCN_kernel_size)
        self.FCz = FC_vec(in_chan=z_dim, out_chan=FCz_out_dim, l_hidden=FCz_l_hidden, activation=FCz_activation, out_activation=FCz_out_activation)
        self.FCw = FC_vec(in_chan=w_dim, out_chan=FCw_out_dim, l_hidden=FCw_l_hidden, activation=FCw_activation, out_activation=FCw_out_activation)
        self.FC = FC_vec(in_chan=TCN_num_channel_list[-1], out_chan=x_dim, l_hidden=FC_l_hidden, activation=FC_activation, out_activation=FC_out_activation)
        
        # For post_training
        self.in_chan = z_dim + w_dim
        self.out_chan = x_dim
    
    

    def forward(self, zw):
        '''
        forward function output is batch * time_step * x_dim
        '''
        batch_size = len(zw)
        z = zw[:, :self.z_dim]
        w = zw[:, self.z_dim:]

        z_enc = self.FCz(z)  # dim: batch * z_dim
        w_enc = self.FCw(w)  # dim: batch * w_dim
        
        phase = torch.arange(1, self.time_step + 1)/self.time_step # dim: time_step
        phase = phase.unsqueeze(0).repeat(batch_size, 1).unsqueeze(-1) # dim: batch * time_step * 1
        
        enc_latent = torch.cat([z_enc, w_enc], dim=1).unsqueeze(1).repeat(1, self.time_step, 1)  # dim: batch * time_step * (h_dim + w_dim)
        phase = phase.to(z_enc.device)
        i = torch.cat([enc_latent, phase], dim=2)  # dim: batch * time_step * (h_dim + w_dim + 1)
        o = self.TCN(i)  # dim: batch * time_step * TCN_num_channel_list[-1]
        # Fix from here!!!
        o = o.reshape(-1, self.TCN_num_channel_list[-1])  # dim: (batch*time_step) * TCN_num_channel_list[-1]

        output = self.FC(o)  # dim: (batch*time_step) * x_dim
        output = output.reshape(batch_size, self.time_step, self.x_dim).flatten(start_dim=1)
        # print(f'self.x_dim = {self.x_dim}')
        # print(f'output.shape = {output.shape}')

        return output
    
    
class TCVAE_decoder_SE3(nn.Module):
    '''
    h is batch * h_dim
    w is batch * w_dim
    Encode manner h and task w by FCN
    h_enc is batch * h_dim
    w_enc is batch * w_dim
    phase represents time step, 0, ..., t/T, ..., 1
    Concatenation of h_enc, w_enc and phase is enc_latent, batch * time_step * (h_dim+w_dim+1) 
    '''

    def __init__(self, x_dim, z_dim, w_dim, 
    FCz_l_hidden, FCz_activation, FCz_out_dim, FCz_out_activation, 
    FCw_l_hidden, FCw_activation, FCw_out_dim, FCw_out_activation,
    FC_l_hidden, FC_activation, FC_out_activation,
    TCN_kernel_size, TCN_num_channel_list, time_step
    ):
        super(TCVAE_decoder_SE3, self).__init__()
        self.time_step = time_step
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.TCN_num_channel_list = TCN_num_channel_list

        self.TCN = TemporalConvNet(num_inputs=FCz_out_dim + FCw_out_dim + 1, num_channels=TCN_num_channel_list, kernel_size=TCN_kernel_size)
        self.FCz = FC_vec(in_chan=z_dim, out_chan=FCz_out_dim, l_hidden=FCz_l_hidden, activation=FCz_activation, out_activation=FCz_out_activation)
        self.FCw = FC_vec(in_chan=w_dim, out_chan=FCw_out_dim, l_hidden=FCw_l_hidden, activation=FCw_activation, out_activation=FCw_out_activation)
        self.FC = FC_SE3(in_chan=TCN_num_channel_list[-1], out_chan=x_dim, l_hidden=FC_l_hidden, activation=FC_activation, out_activation=FC_out_activation)
        
        # For post_training
        self.in_chan = z_dim + w_dim
        self.out_chan = x_dim
    
    

    def forward(self, zw):
        '''
        forward function output is batch * time_step * x_dim
        '''
        batch_size = len(zw)
        z = zw[:, :self.z_dim]
        w = zw[:, self.z_dim:]

        z_enc = self.FCz(z)  # dim: batch * z_dim
        w_enc = self.FCw(w)  # dim: batch * w_dim
        
        phase = torch.arange(1, self.time_step + 1)/self.time_step # dim: time_step
        phase = phase.unsqueeze(0).repeat(batch_size, 1).unsqueeze(-1) # dim: batch * time_step * 1
        
        enc_latent = torch.cat([z_enc, w_enc], dim=1).unsqueeze(1).repeat(1, self.time_step, 1)  # dim: batch * time_step * (h_dim + w_dim)
        phase = phase.to(z_enc.device)
        i = torch.cat([enc_latent, phase], dim=2)  # dim: batch * time_step * (h_dim + w_dim + 1)
        o = self.TCN(i)  # dim: batch * time_step * TCN_num_channel_list[-1]
        # Fix from here!!!
        o = o.reshape(-1, self.TCN_num_channel_list[-1])  # dim: (batch*time_step) * TCN_num_channel_list[-1]

        output = self.FC(o)  # dim: (batch*time_step) * x_dim
        output = output.reshape(batch_size, self.time_step, self.x_dim).flatten(start_dim=1)
        # print(f'self.x_dim = {self.x_dim}')
        # print(f'output.shape = {output.shape}')

        return output

class Conv1d_encoder(nn.Module):
    def __init__(
        self,
        x_dim=402,
        z_dim=1,
        time_step=201,
        fc_hidden=None,
        fc_activation=None,
        out_activation=None,
        conv_hidden=[8, 16, 32],
        conv_act=['relu'],
        kernel_size=[5, 5, 4],
        stride=[2, 2, 2],
    ):
        super().__init__()
        self.x_dim = x_dim
        self.in_chan = int(x_dim/time_step)
        self.z_dim = z_dim
        self.time_step = time_step
        
        l_neurons = fc_hidden + [z_dim]
        if len(fc_activation) > len(fc_hidden):
            fc_activation = fc_activation[:len(fc_hidden)]
        elif len(fc_activation) < len(fc_hidden):
            while len(fc_activation) < len(fc_hidden):
                fc_activation.append(fc_activation[-1])
        fc_activation = fc_activation + [out_activation]
        
        if len(conv_act) > len(conv_hidden):
            conv_act = conv_act[:len(conv_hidden)]
        elif len(conv_act) < len(conv_hidden):
            while len(conv_act) < len(conv_hidden):
                conv_act.append(conv_act[-1])
        # (n + 2p - f)/s + 1
        # 201 -> 99 -> 48 -> 23
        prev_chan = self.in_chan
        time_step_conv = self.time_step
        l_layer = []
        for i in range(len(conv_hidden)):
            l_layer.append(nn.Conv1d(prev_chan, conv_hidden[i], 
                                    kernel_size=kernel_size[i], 
                                    bias=True, stride=stride[i]))
            l_layer.append(get_activation(conv_act[i]))
            prev_chan = conv_hidden[i]
            time_step_conv = int((time_step_conv - kernel_size[i])/2) + 1
        l_layer.append(nn.Flatten())
        dim_flatten = time_step_conv * conv_hidden[-1]
        
        
        prev_dim = dim_flatten 
        for [n_hidden, act] in (zip(l_neurons, fc_activation)):
            l_layer.append(nn.Linear(prev_dim, n_hidden))
            act_fn = get_activation(act)
            if act_fn is not None:
                l_layer.append(act_fn)
            prev_dim = n_hidden

        self.net = nn.Sequential(*l_layer)
        
    def forward(self, x):
        bs = len(x)
        x = x.reshape(bs, self.time_step, self.in_chan).transpose(1, 2)
        x = self.net(x)
        return x