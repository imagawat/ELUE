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


def identity(x):
    return x
class BeliefVariables(nn.Module):
    def __init__(self, tensor):
        super().__init__()
        self.variable = nn.Parameter(tensor.to(ptu.device).detach(), requires_grad=True)
    def set_size(self, size):
        self.variable.view(1, -1).repeat(size, 1)
class BeliefMlp(PyTorchModule):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            bottleneck_layerid, # make bottleneck after the layer 
            bottleneck_size, #
            naive_bottleneck, #True or False.  False means INFOBOT like bottleneck
            belief_input_size,
            mlp, # may be this is not used
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            layer_norm_kwargs=None,
            std_range = [1e-8, 1e+8],
            policy=False,
            **kwargs,            
    ):
        self.algo_params = kwargs
        self.b_init_value = b_init_value
        self.save_init_params(locals())
        super().__init__()
        self.min_logstd = np.log(std_range[0])
        self.max_logstd = np.log(std_range[1])

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()
        self.mlp = mlp
        self.input_size = input_size
        self.belief_input_size = belief_input_size
        self.belief_module = None
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        self.hidden_init = hidden_init
        in_size = input_size + belief_input_size
        if not policy and not self.algo_params['q_v_bottleneck']:
            self.bottleneck_layerid = bottleneck_layerid = -1
        else:
            self.bottleneck_layerid = bottleneck_layerid
        self.naive_bottleneck = naive_bottleneck
        self.hidden_sizes = hidden_sizes
        std = self.algo_params['std_for_variational_dist_for_bottleneck'] * ptu.ones(bottleneck_size)
        self.default_dist = torch.distributions.Normal(ptu.zeros(bottleneck_size), std)         
        add = 0
        for i, next_size in enumerate(self.hidden_sizes):
            if i == bottleneck_layerid:
                self.set_layer(in_size, bottleneck_size, i+add) # for mean
                add += 1
                self.set_layer(in_size, bottleneck_size, i+add) # for logstd
                add += 1
                if naive_bottleneck:
                    in_size = bottleneck_size
                else:
                    in_size = bottleneck_size + input_size
                self.set_layer(in_size, next_size, i+add)                
            else:
                self.set_layer(in_size, next_size, i+add)
            in_size = next_size
        if len(self.hidden_sizes) == bottleneck_layerid:
            self.set_layer(in_size, bottleneck_size, len(self.hidden_sizes)+add) # for mean
            add += 1
            self.set_layer(in_size, bottleneck_size, len(self.hidden_sizes)+add) # for logstd
            add += 1
            if naive_bottleneck:
                in_size = bottleneck_size
            else:
                in_size = bottleneck_size + input_size
        self.last_hidden_size = in_size
        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)
        self.is_set_belief = False
        self.ordinary_params = list(self.parameters())
    def set_layer(self, in_size, next_size, i):
        fc = nn.Linear(in_size, next_size)
        self.hidden_init(fc.weight)
        fc.bias.data.fill_(self.b_init_value)
        self.__setattr__("fc{}".format(i), fc)
        self.fcs.append(fc)
        if self.layer_norm:
            ln = LayerNorm(next_size)
            self.__setattr__("layer_norm{}".format(i), ln)
            self.layer_norms.append(ln)

    def forward(self, inps, reparameterize, dim=1, without_last=False, return_preactivations=False, denoise=False, middle_output=False, both=False):#
        h = inps[1]
        add = inps[0] #belief  
        hin = self.with_belief([h, add], layerid=0, dim=dim)
        i = 0
        (mid, mid_diff_log_prob, mid2, mid2_diff_log_prob) = (None, None, None, None)
        while i < len(self.fcs):
            hout = self.fcs[i](hin)
            if self.layer_norm and i < len(self.fcs) - 1:
                hout = self.layer_norms[i](hout)
            hout = self.hidden_activation(hout)
            hin = hout
            i = i + 1
            if i == self.bottleneck_layerid:
                hout, mid_diff_log_prob, mid2, mid2_diff_log_prob = self.bottleneck_output(i, hin, dim, reparameterize, denoise, middle_output, both)
                mid = hout
                if self.naive_bottleneck:
                    hin = hout
                else:
                    hin = torch.cat([hout, inps[1]], dim)
                i = i + 2 
        if without_last:
            return hin, (mid, mid_diff_log_prob, mid2, mid2_diff_log_prob)
        preactivation = self.last_fc(hin)
        output = self.output_activation(preactivation)
        if return_preactivations:
            assert(not middle_output)
            return output, preactivation
        return output, (mid, mid_diff_log_prob, mid2, mid2_diff_log_prob)
    def bottleneck_output(self, layerid, inp, dim, reparameterize, denoise, middle_output, both):
        mu = self.fcs[layerid](inp)
        if denoise:
            return mu, None, None, None
        logstd = self.fcs[layerid+1](inp)        
        std = torch.exp(torch.clamp(logstd, min=self.min_logstd, max=self.max_logstd))
        dist = torch.distributions.Normal(mu, std)
        if both:            
            val = dist.rsample()            
            return val, self.diff_log_prob(dist, val, dim), mu, self.diff_log_prob(dist, val, dim)

        if reparameterize:
            val = dist.rsample()         
        else:
            val = dist.sample()  
        if middle_output:
            return val, self.diff_log_prob(dist, val, dim), None, None
        
        return val, None, None, None

    def diff_log_prob(self, dist, val, dim):
        if self.algo_params['uniform_variational_dist_for_bottleneck']:
            # omit because the variational distribution is constant
            if self.algo_params['mean_bottleneck_loss']:
                return torch.mean(dist.log_prob(val), dim=dim, keepdim=True)
            return torch.sum(dist.log_prob(val), dim=dim, keepdim=True)        
        
        if self.algo_params['mean_bottleneck_loss']:
            return torch.mean(dist.log_prob(val) - self.default_dist.log_prob(val), dim=dim, keepdim=True)
            
        return torch.sum(dist.log_prob(val) - self.default_dist.log_prob(val), dim=dim, keepdim=True)

    def parameters_for_adaptation(self):
        if self.algo_params['learning_type_at_eval'] == 0:
            raise NotImplementedError
        elif self.algo_params['learning_type_at_eval'] == 1:
            raise NotImplementedError
        elif self.algo_params['learning_type_at_eval'] == 2: 
            params = self.ordinary_params
        elif self.algo_params['learning_type_at_eval'] == 3:
            # without param. update ... this func won't be called
            pass
        elif self.algo_params['learning_type_at_eval'] == 4:
            params = [{'params': self.ordinary_params}, {'params': [self.belief]}]
            #params = [{'params': self.ordinary_params, 'lr': 1e-4}, {'params': [self.belief]}]
        elif self.algo_params['learning_type_at_eval'] == 5:
            params = self.belief_module.parameters()
        else:
            raise NotImplementedError
        return params

    def set_belief(self, belief):
        # only used at eval phase
        if self.algo_params['learning_type_at_eval'] == 5:
            # grads may be used in other type of learning
            for p in self.parameters():
                p.requires_grad = False
        self.belief_module = BeliefVariables(belief) #BeliefVariables(self.belief, belief)
        self.is_set_belief = True

    def with_belief(self, inps, layerid, dim=1):
        if True:
            if self.is_set_belief == False:
                ret = torch.cat(inps,dim=dim)
            else:
                #pid = self.belief_module.layerid2paramid[layerid]
                size = inps[0].size(0)
                assert(dim == 1) 
                add = self.belief_module.variable.view(1, -1).repeat(size, 1)
                ret = torch.cat([inps[0], add], dim=dim)
            return ret
        return inps[0]
        

    
class BeliefTanhGaussianPolicy(BeliefMlp, ExplorationPolicy):
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
            output_size,
            input_size,
            std=None,
            init_w=1e-3,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes, #
            output_size=output_size,
            input_size=input_size,
            init_w=init_w,
            policy=True,
            **kwargs
        )#　ここでlayerの設定は完了している
        self.log_std = None
        self.std = std
        if std is None:
            self.last_fc_log_std = nn.Linear(self.last_hidden_size, output_size)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
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
    def get_action_log_prob(self, inps, action, reparameterize=False, dim=1):
        h, mid = super().forward(inps, dim=dim, without_last=True, reparameterize=reparameterize)
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        log_prob = None
        tanh_normal = TanhNormal(mean, std)
        log_prob = tanh_normal.log_prob(action)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return log_prob, mid
        

        
    def forward(
            self,
            inps,
            dim=1,
            reparameterize=False,
            deterministic=False,
            return_log_prob=False,
            denoise=False,
            middle_output=False,
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
        
        #h, middle = super.forward(inps, dim=dim, without_last=True, reparameterize=reparameterize, both=True)
        denoise = (deterministic or denoise)
        h, middle = super().forward(inps, dim=dim, without_last=True, reparameterize=reparameterize, middle_output=middle_output, denoise=denoise)
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
            mean_action_log_prob, pre_tanh_value, #
            *middle #8,9,10,11
        )
