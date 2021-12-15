import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import PyTorchModule
from rlkit.torch.modules import LayerNorm
from rlkit.torch.networks import Mlp
from rlkit.torch.sac.policies import LOG_SIG_MAX,LOG_SIG_MIN
from rlkit.policies.base import ExplorationPolicy
from rlkit.torch.distributions import TanhNormal
from rlkit.torch.core import np_ify
import debug_config

def identity(x):
    return x

def generator(l):
    for e in l:
        yield e
        

class FiLM(nn.Module):
    def __init__(self, gamma_offset=0):
        super().__init__()
        self.gamma_offset=gamma_offset
        
    def forward(self, x, gammas, betas):
        if debug_config.IGNOREFILM:            
            return ptu.zeros(x.size())
        #gammas.size() = (batch, num_feat_tensors) 
        gammas = gammas.unsqueeze(2) + self.gamma_offset
        betas = betas.unsqueeze(2)
        return  (gammas * x) + betas 

class FiLMGenNet(Mlp):
    def forward(self, inp):
        ret = super().forward(inp)
        return ret

class FiLMedMlp(PyTorchModule):
    def __init__(
            self,
            hidden_sizes, #
            filmed, # [False or True]
            feat_sizes, # list of size of flatten feat tensor 
            num_feat_tensors,
            gennet_hidden_sizes,
            output_size,
            input_sizes, # for [FiLMGenNet,FiLMedMlp]
            with_connection_layer=False,
            with_activation_before_film=True,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            **kwargs,
    ):
        self.algo_params = kwargs
        self.with_connection_layer=with_connection_layer
        self.with_activation_before_film = with_activation_before_film
        self.hidden_init = hidden_init
        self.b_init_value = b_init_value
        
        assert(len(filmed) == len(hidden_sizes) + len(feat_sizes)) 

        self.save_init_params(locals())
        super().__init__()
        self.filmed = filmed

        self.input_size = input_sizes[1]
        self.output_size = output_size

        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm 
        self.layers = []
        #self.layer_norms = []
        in_size = input_sizes[1]

        #
        self.num_feat_tensors = num_feat_tensors
        genoutput_size = sum(num_feat_tensors) * 2 # gamma and beta for each feat tensor 
        self.film_gennet = FiLMGenNet(hidden_sizes=gennet_hidden_sizes,
                                output_size=genoutput_size,
                                input_size=input_sizes[0],
        )
        #self.film_vars = torch.randn(genoutput_size)        # used only if debug_config.MODIFYFILM
        if debug_config.MODIFYFILM:
            self.film_vars = nn.Parameter(torch.randn(genoutput_size), requires_grad=True)        # used only if debug_config.MODIFYFILM
        else:
            self.film_vars = torch.Tensor()
        self.ground_params_list = []
        if with_connection_layer:
            fid = 0
            for i in range(len(filmed)):
                if filmed[i]:
                    f_size = feat_sizes[fid] * num_feat_tensors[fid]
                    idx = fid*3+i if self.layer_norm else fid*2+i
                    self.add_linear_layer(in_size, f_size, idx) #connection layer before resnet branch
                    self.add_linear_layer(f_size, f_size, idx+1, self.layer_norm) #connection layer after resnet branch
                    layer = FiLM()
                    self.layers.append(layer)
                    fid += 1
                    in_size = f_size
                else:
                    next_size = hidden_sizes[i - fid]
                    idx = fid*3+i if self.layer_norm else fid*2+i
                    self.add_linear_layer(in_size, next_size, idx)
                    in_size = next_size
        else:
            fid = 0
            for i in range(len(filmed)):
                if filmed[i]:
                    layer = FiLM()
                    next_size = feat_sizes[fid] * num_feat_tensors[fid] 
                    assert(in_size == next_size)
                    fid += 1
                    assert i == len(filmed) -1 or not filmed[i+1], "Adjoined FiLM layers is not considered"
                    self.layers.append(layer)
                else:
                    if i == len(filmed) -1 or not filmed[i+1]:
                        next_size = hidden_sizes[i - fid]
                    else:
                        next_size = feat_sizes[fid] * num_feat_tensors[fid] 
                        assert next_size == hidden_sizes[i-fid] #Before FiLM layer, size of MLP is regularized
                        self.add_linear_layer(in_size, next_size, i)
                in_size = next_size
            

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)
        self.ground_params_list += list(self.last_fc.parameters())
    def add_linear_layer(self, in_size, out_size, idx, with_layer_norm=False):
        layer = nn.Linear(in_size, out_size)
        self.hidden_init(layer.weight)
        layer.bias.data.fill_(self.b_init_value)
        self.__setattr__("fc{}".format(idx), layer)        
        self.ground_params_list += list(layer.parameters())
        self.layers.append(layer)
        if with_layer_norm:
            ln = LayerNorm(out_size)
            self.__setattr__("layer_norm{}".format(idx), ln) #TODO: fix idx
            self.layers.append(ln)
        
    def forward(self, inps, return_preactivations=False, without_last=False):
        if self.film_vars.numel() > 0: #debug_config.MODIFYFILM or self.algo_params["learning_type_at_eval"] == 4:
            size = list(inps[0].size()[:-1]) + [-1] # 
            filmvars = self.film_vars.expand(size)
        else:          
            filmvars = self.film_gennet(inps[0])
        itr = 0
        filmid = 0
        h = inps[1]
        b, _ = h.size()
        if self.with_connection_layer:
            layerid = 0
            for i, f in enumerate(self.filmed):
                if f:
                    # connection layer before branch
                    h = self.layers[layerid](h)
                    h = self.hidden_activation(h)
                    layerid+=1
                    # connection layer after branch
                    _h = self.layers[layerid](h)
                    if self.with_activation_before_film:
                        _h = self.hidden_activation(_h)
                    layerid += 1
                    if self.layer_norm:
                        _h = self.layers[layerid](_h)
                        layerid += 1
                    # film
                    size = self.num_feat_tensors[filmid] 
                    gammas = filmvars.narrow(1, itr, size)
                    betas = filmvars.narrow(1, itr+size, size)
                    _h = _h.view(b, size, -1)
                    _h = self.layers[layerid](_h, gammas, betas)
                    _h = _h.view(b, -1)
                    _h = F.relu(_h)
                    layerid += 1
                    itr = itr+size*2
                    filmid += 1
                    h = _h  + h #add input (ResNet)

                else:
                    h = self.layers[layerid](h)
                    h = self.hidden_activation(h)
                    layerid+=1
        else:
            for i, f in enumerate(self.filmed):
                if f:                
                    size = self.num_feat_tensors[filmid] 
                    gammas = filmvars.narrow(1, itr, size)
                    betas = filmvars.narrow(1, itr+size, size)
                    _h = h.view(b, size, -1)
                    _h = self.layers[i](_h, gammas, betas)
                    _h = F.relu(_h)
                    h = _h.view(b, -1) # + h #add input (ResNet)
                    itr = itr+size*2
                    filmid += 1
                else:
                    h = self.layers[i](h)
                    #if self.layer_norm and i < len(self.layers) - 1:
                    #    h = self.layer_norms[i](h)
                    h = self.hidden_activation(h)

        if without_last:
            return h
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output

    def parameters_for_adaptation(self):
        if self.algo_params['learning_type_at_eval'] == 0:
            params = self.film_gennet.parameters()
        elif self.algo_params['learning_type_at_eval'] == 1:
            params = self.ground_params_list            
        elif self.algo_params['learning_type_at_eval'] == 2: 
            params = self.ground_params_list + list(self.film_gennet.parameters())
        elif self.algo_params['learning_type_at_eval'] == 3:
            # without param. update ... this func won't be called
            pass
        elif self.algo_params['learning_type_at_eval'] == 4:
            params = self.ground_params_list + [self.film_vars]
        elif self.algo_params['learning_type_at_eval'] == 5:
            params = [self.film_vars]
        else:
            raise NotImplementedError
        return params
    def to(self, device):
        super().to(device)

    def set_film_vars(self, inps):
        filmvars = self.film_gennet(inps).detach()
        self.film_vars = nn.Parameter(filmvars, requires_grad=True)        # used only if debug_config.MODIFYFILM
        
class FiLMedTanhGaussianPolicy(FiLMedMlp, ExplorationPolicy):
    """
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```
    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """
    def __init__(
            self,
            hidden_sizes, #
            filmed, # [False or True]
            feat_sizes, # list of size of flatten feat tensor 
            num_feat_tensors,
            gennet_hidden_sizes,
            action_dim,
            input_sizes, # for [FiLMGenNet,FiLMedMlp]
            with_connection_layer=False,
            std=None,
            init_w=1e-3,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes, #
            filmed, # [False or True]
            feat_sizes, # list of size of flatten feat tensor 
            num_feat_tensors,
            gennet_hidden_sizes,
            output_size=action_dim,
            input_sizes=input_sizes, # for [FiLMGenNet,FiLMedMlp] 
            init_w=init_w,
            with_connection_layer=with_connection_layer,
            **kwargs
        )#
        self.log_std = None
        self.std = std
        if std is None:
            assert(len(filmed) > 0)
            if filmed[-1]:
                last_hidden_size = feat_sizes[-1]*num_feat_tensors[-1]                
            else:
                assert(len(hidden_sizes) > 0)
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
            self.ground_params_list += list(self.last_fc_log_std.parameters())
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def get_action(self, inps, deterministic=False):
        actions = self.get_actions(inps, deterministic=deterministic)
        return actions[0, :], {}

    @torch.no_grad()
    def get_actions(self, inps, deterministic=False):
        outputs = self.forward(inps, deterministic=deterministic)[0]
        return np_ify(outputs)

    def forward(
            self,
            inps,
            reparameterize=False,
            deterministic=False,
            return_log_prob=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        """
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        """
        h = super().forward(inps, without_last=True)
        # remain unchanged
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        log_prob = None
        expected_log_prob = None
        mean_action_log_prob = None
        pre_tanh_value = None
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = TanhNormal(mean, std)
            if return_log_prob:
                if reparameterize:
                    action, pre_tanh_value = tanh_normal.rsample(
                        return_pretanh_value=True
                    )
                else:
                    action, pre_tanh_value = tanh_normal.sample(
                        return_pretanh_value=True
                    )
                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                if reparameterize:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()

        return (
            action, mean, log_std, log_prob, expected_log_prob, std,
            mean_action_log_prob, pre_tanh_value,
        )
