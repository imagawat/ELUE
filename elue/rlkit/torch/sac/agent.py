import numpy as np

import torch
from torch import nn as nn
import torch.nn.functional as F
import rlkit.torch.pytorch_util as ptu
import gtimer as gt
import debug_config
from rlkit.torch.core import np_ify

def _product_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, sigma_squared


def _mean_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of mean of gaussians
    '''
    mu = torch.mean(mus, dim=0)
    sigma_squared = torch.mean(sigmas_squared, dim=0)
    return mu, sigma_squared


def _natural_to_canonical(n1, n2):
    ''' convert from natural to canonical gaussian parameters '''
    mu = -0.5 * n1 / n2
    sigma_squared = -0.5 * 1 / n2
    return mu, sigma_squared


def _canonical_to_natural(mu, sigma_squared):
    ''' convert from canonical to natural gaussian parameters '''
    n1 = mu / sigma_squared
    n2 = -0.5 * 1 / sigma_squared
    return n1, n2



class PEARLAgent(nn.Module):

    def __init__(self,
                 latent_dim,
                 context_encoder,
                 policy,
                 **kwargs
    ):
        super().__init__()
        self.algo_params = kwargs
        self.latent_dim = latent_dim

        self.context_encoder = context_encoder
        self.policy = policy

        self.recurrent = kwargs['recurrent']
        self.use_ib = kwargs['use_information_bottleneck']
        self.sparse_rewards = kwargs['sparse_rewards']
        self.use_next_obs_in_context = kwargs['use_next_obs_in_context']

        # initialize buffers for z dist and z
        # use buffers so latent context can be saved along with model weights
        self.register_buffer('z', torch.zeros(1, latent_dim))
        self.register_buffer('z_means', torch.zeros(1, latent_dim))
        self.register_buffer('z_vars', torch.zeros(1, latent_dim))
        self.clear_z()
        self.context = None
    def clear_z(self, num_tasks=1):
        '''
        reset q(z|c) to the prior
        sample a new z from the prior
        '''
        # reset distribution over z to the prior
        mu = ptu.zeros(num_tasks, self.latent_dim)
        if self.use_ib:
            var = ptu.ones(num_tasks, self.latent_dim)
        else:
            var = ptu.zeros(num_tasks, self.latent_dim)
        self.z_means = mu
        self.z_vars = var
        # sample a new z from the prior
        self.sample_z()
        # reset the context collected so far
        self.context = None
        # reset any hidden state in the encoder network (relevant for RNN)
        self.context_encoder.reset(num_tasks)

    def detach_z(self):
        ''' disable backprop through z '''
        self.z = self.z.detach()
        if self.recurrent:
            self.context_encoder.hidden = self.context_encoder.hidden.detach()

    def update_context(self, inputs):
        ''' append single transition to the current context '''
        o, a, r, no, d, info = inputs
        if self.sparse_rewards:
            r = info['sparse_reward']
        o = ptu.from_numpy(o[None, None, ...])
        a = ptu.from_numpy(a[None, None, ...])
        r = ptu.from_numpy(np.array([r])[None, None, ...])
        no = ptu.from_numpy(no[None, None, ...])

        if self.use_next_obs_in_context:
            data = torch.cat([o, a, r, no], dim=2)
        else:
            data = torch.cat([o, a, r], dim=2)
        if self.context is None:
            self.context = data
        else:
            self.context = torch.cat([self.context, data], dim=1)

    def compute_kl_div(self):
        ''' compute KL( q(z|c) || r(z) ) '''
        prior = torch.distributions.Normal(ptu.zeros(self.latent_dim), ptu.ones(self.latent_dim))
        posteriors = [torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
        kl_divs = [torch.distributions.kl.kl_divergence(post, prior) for post in posteriors]
        kl_div_sum = torch.sum(torch.stack(kl_divs))
        return kl_div_sum

    def infer_posterior(self, context, with_context_reset=False):
        ''' compute q(z|c) as a function of input context and sample new z from it'''
        params = self.context_encoder(context)
        params = params.view(context.size(0), -1, self.context_encoder.output_size)
        # with probabilistic z, predict mean and variance of q(z | c)
        if self.use_ib:
            mu = params[..., :self.latent_dim]
            sigma_squared = F.softplus(params[..., self.latent_dim:])
            z_params = [_product_of_gaussians(m, s) for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]
            self.z_means = torch.stack([p[0] for p in z_params])
            self.z_vars = torch.stack([p[1] for p in z_params])
        # sum rather than product of gaussians structure
        else:
            self.z_means = torch.mean(params, dim=1)
        self.sample_z()
        if with_context_reset:
            self.context = None
    def infer_posterior_by_preserved_z_and_context(self):
        ''' compute q(z|c) as a function of input context and sample new z from it'''
        params = self.context_encoder(self.context)
        params = params.view(self.context.size(0), -1, self.context_encoder.output_size)
        # with probabilistic z, predict mean and variance of q(z | c)
        if self.use_ib:
            mu =torch.cat([torch.squeeze(params[..., :self.latent_dim], dim=0), self.z_means], dim=0)
            sigma_squared = torch.cat([torch.squeeze(F.softplus(params[..., self.latent_dim:]), dim=0), self.z_vars], dim=0)
            z_params = [_product_of_gaussians(mu, sigma_squared)]
            self.z_means = torch.stack([p[0] for p in z_params])
            self.z_vars = torch.stack([p[1] for p in z_params])
        # sum rather than product of gaussians structure
        else:
            self.z_means = torch.mean(params, dim=1)
        self.sample_z()
        self.context = None
  
    def sample_z(self):
        if self.use_ib:
            posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
            z = [d.rsample() for d in posteriors]
            self.z = torch.stack(z)
        else:
            self.z = self.z_means

    def get_action(self, obs, deterministic=False):
        ''' sample action from the policy, conditioned on the task embedding '''
        z = self.z
        if debug_config.IGNOREZ:
            z = ptu.zeros(1, self.latent_dim)
        obs = ptu.from_numpy(obs[None])
        in_ = torch.cat([obs, z], dim=1)
        return self.policy.get_action(in_, deterministic=deterministic)

    def set_num_steps_total(self, n):
        self.policy.set_num_steps_total(n)

    def forward(self, obs, context):
        ''' given context, get statistics under the current policy of a set of observations '''
        self.infer_posterior(context)
        self.sample_z()

        task_z = self.z

        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        task_z = [z.repeat(b, 1) for z in task_z]
        task_z = torch.cat(task_z, dim=0)

        if debug_config.IGNOREZ:
            task_z = ptu.zeros(t* b, self.latent_dim)

        # run policy, get log probs and new actions
        in_ = torch.cat([obs, task_z.detach()], dim=1)
        policy_outputs = self.policy(in_, reparameterize=True, return_log_prob=True)

        return policy_outputs, task_z

    def log_diagnostics(self, eval_statistics):
        '''
        adds logging data about encodings to eval_statistics
        '''
        z_mean = np.mean(np.abs(ptu.get_numpy(self.z_means[0])))
        z_sig = np.mean(ptu.get_numpy(self.z_vars[0]))
        eval_statistics['Z mean eval'] = z_mean
        eval_statistics['Z variance eval'] = z_sig

    @property
    def networks(self):
        return [self.context_encoder, self.policy]



"""    
from tensorboardX import SummaryWriter    
with SummaryWriter(comment='RNN') as w:
    w.add_graph(rnn, (cat, dummy_input, hidden), verbose=False)
"""
    
class ELUEAgent(nn.Module):
    def __init__(self,
                 latent_dim,
                 encoder, #
                 integrator,
                 rews_decoder,
                 nobs_decoder,
                 predictor,
                 policy,
                 eval_mode,
                 **kwargs
    ):
        super().__init__()
        self.algo_params = kwargs
        self.latent_dim = latent_dim
        self.eval_mode = eval_mode
        self.policy = policy
        self.integrator = integrator
        self.rews_decoder = rews_decoder  
        self.nobs_decoder = nobs_decoder  
        self.encoder = encoder
        self.predictor = predictor
        self.max_context_length = kwargs['max_context_length']
        self.representation_buffer= ptu.zeros(self.max_context_length, self.encoder.output_size) 
        self.representation_count = 0
        self.max_elem_loss = 1e+6 #TODO:FIX this temp value
        self.recurrent = kwargs['recurrent']
        self.sparse_rewards = kwargs['sparse_rewards']
        self.use_nobs_decoder = kwargs['use_nobs_decoder']
        self.use_next_obs_in_context = kwargs['use_next_obs_in_context']
        self.num_sampled_z = kwargs['num_sampled_z']
        self.num_tasks = 1
        self.noise = [[] for i in range(self.num_tasks)] 
        # initialize buffers for z dist and z
        # use buffers so latent context can be saved along with model weights
        self.register_buffer('z_means', torch.zeros(1, latent_dim))
        self.register_buffer('z_stds', torch.zeros(1, latent_dim))
        #self.cvae_params = torch.nn.ParameterList([self.encoder.parameters(), self.integrator.parameters(), self.decoder.parameters()])
        self.cvae_params = list(self.encoder.parameters()) + list(self.integrator.parameters()) + list(self.rews_decoder.parameters())
        if self.use_nobs_decoder:
            self.cvae_params += list(self.nobs_decoder.parameters())
        self.mlp = kwargs['mlp']
#list(self.decoder.parameters())


    @property
    def networks(self):
        #return [self.encoder, self.integrator, self.decoder, self.policy]
        if self.use_nobs_decoder:
            return [self.encoder, self.integrator, self.rews_decoder, self.nobs_decoder, self.policy]
        return [self.encoder, self.integrator, self.rews_decoder, self.policy]

    def update_context(self, inputs):
        ''' append single transition to the current context '''
        if self.num_tasks > 1:
            obss, acts, rews, nobss, terms, infos = inputs
            if self.sparse_rewards:
                rews = infos['sparse_reward']
            o = torch.stack([ptu.from_numpy(o) for o in obss], dim=0)
            a = torch.stack([ptu.from_numpy(a) for a in acts], dim=0)
            r = torch.stack([ptu.from_numpy(np.array([r])) for r in rews], dim=0)
            no = torch.stack([ptu.from_numpy(no) for no in nobss], dim=0)

            if self.use_next_obs_in_context:
                self.new_data = torch.cat([o, a, r, no], dim=1)
            else:
                self.new_data = torch.cat([o, a, r], dim=1)
            return 0
        o, a, r, no, d, info = inputs
        if self.sparse_rewards:
            r = info['sparse_reward']
        o = ptu.from_numpy(o[None, None, ...])
        a = ptu.from_numpy(a[None, None, ...])
        r = ptu.from_numpy(np.array([r])[None, None, ...])
        no = ptu.from_numpy(no[None, None, ...])
        if self.use_next_obs_in_context:
            self.new_data = torch.cat([o, a, r, no], dim=2)
        else:
            self.new_data = torch.cat([o, a, r], dim=2)

  
    
    def cvae_reconstruction_loss_batch(self, context, indices):
        gt.subdivide('sub')
        obs, acts, rews, nobs = context  #obs shape is (tasks, num_context, context_length, ?)
        t, num_context, context_length, _ = obs.size()
        obs = obs.view(t*num_context, context_length,  -1)
        acts = acts.view(t*num_context, context_length,  -1)
        rews = rews.view(t*num_context, context_length,  -1)
        nobs = nobs.view(t*num_context, context_length,  -1)
        if self.use_next_obs_in_context:
            context = torch.cat([obs,acts,rews,nobs], dim=2)
        else:
            context = torch.cat([obs,acts,rews], dim=2)
            assert(not self.algo_params["rew_decoder_with_nobs"])
            assert(not self.use_nobs_decoder)

        enc = self.encoder(context)
        enc = torch.sum(enc, dim=1)
        gt.stamp('enc')
        means, stds = self.integrator(enc,dim=1) #means shape is (tasks*num_context, latent_dim)
        
        latent_dim = means.size(1)
        posteriors = [torch.distributions.Normal(m, s) for m, s in zip(torch.unbind(means), torch.unbind(stds))]
        gt.stamp('integrate')
        z = torch.stack([torch.stack([d.rsample() for i in range(self.num_sampled_z)], dim=0) for d in posteriors], dim=0) 
        #(tasks*num_context, num_z, latent_dim)
        z = z.view(t*num_context, 1, self.num_sampled_z, latent_dim).repeat(1, context_length, 1, 1)
        #(tasks*num_context, context_length,  num_z, latent_dim)
        concated = torch.cat([obs, acts], dim=2)
        concated = concated.view(t*num_context, context_length, 1 ,-1).repeat(1, 1, self.num_sampled_z, 1)
        concated = torch.cat([z, concated], dim=3)
        concated = ptu.flatten(concated)
        #recrew_means, recrew_stds, recnob_means, recnob_stds = self.decoder(concated, dim=1) #(num_z* tasks*context_length, )
        if self.algo_params["rew_decoder_with_nobs"]:
            rconcated = torch.cat([obs, acts, nobs], dim=2)
            rconcated = rconcated.view(t*num_context, context_length, 1 ,-1).repeat(1, 1, self.num_sampled_z, 1)
            rconcated = torch.cat([z, rconcated], dim=3)
            rconcated = ptu.flatten(rconcated)
        else:
            rconcated = concated
        recrew_means, recrew_stds = self.rews_decoder(rconcated, dim=1) #(num_z* tasks*context_length, )        
        rrews = rews.view(t*num_context, context_length, 1 ,-1).repeat(1, 1, self.num_sampled_z, 1)
        rrews = ptu.flatten(rrews)
        elem_loss = -torch.distributions.Normal(recrew_means, recrew_stds).log_prob(rrews) # loss of each element of context
        elem_loss = torch.clamp(elem_loss, max=self.max_elem_loss)
        #loss = torch.sum(elem_loss)
        loss = torch.mean(elem_loss)
        if self.use_nobs_decoder:
            recnob_means, recnob_stds = self.nobs_decoder(concated, dim=1) #(num_z* tasks*context_length, )
            rnobs = nobs.view(t*num_context, context_length, 1 ,-1).repeat(1, 1, self.num_sampled_z, 1)
            rnobs = ptu.flatten(rnobs)
            elem_loss = -torch.distributions.Normal(recnob_means, recnob_stds).log_prob(rnobs)
            elem_loss = torch.clamp(elem_loss, max=self.max_elem_loss)
            #loss += torch.sum(elem_loss)
            loss += torch.mean(elem_loss)
        gt.stamp('loss_z')
        # this is not needed because loss is changed from sum -> mean
        #loss /= self.num_sampled_z 
        assert(not torch.isnan(loss))
        # KL loss    
        prior = torch.distributions.Normal(ptu.zeros(self.latent_dim), ptu.ones(self.latent_dim))
        elem_kl_loss = [torch.distributions.kl.kl_divergence(post, prior) for post in posteriors]
        elem_kl_loss = [torch.clamp(elem_loss, max=self.max_elem_loss) for elem_loss in elem_kl_loss]
        #kl_loss = torch.sum(sum(elem_kl_loss))
        kl_loss = torch.mean(sum(elem_kl_loss)/len(elem_kl_loss))
        assert(not torch.isnan(kl_loss))

        gt.end_subdivision()
        return loss, kl_loss, {"elem_loss": elem_loss, "elem_kl_loss": elem_kl_loss, "means": means, "stds": stds}
    
    def get_action(self, obs, deterministic=False):
        # used at sampling from env
        if self.num_tasks == 1:
            obs = obs[None]        
            obs = ptu.from_numpy(obs)
        else:
            #0dim is for taskID
            obs = torch.stack([ptu.from_numpy(o) for o in obs], dim=0)
        belief = torch.cat([self.z_means, self.z_stds], dim=1)
        assert(not any(torch.isnan(belief[0])))
        #if debug_config.IGNOREZ:
        # nothing is needed

            
        if self.algo_params['film_gen_with_full_inputs']:
            inps = [torch.cat([belief, obs], dim=1), obs]
        else:
            inps = [belief, obs]

        if self.algo_params['noisy_input'] == 1:
            inps[0] = torch.cat([inps[0], self.noise], dim=1)
        elif self.algo_params['noisy_input'] == 2:
            inps[1] = torch.cat([inps[1], self.noise], dim=1)
            
        if self.num_tasks == 1:
            action = self.policy.get_action(inps, deterministic=deterministic)
        else:
            action = [np_ify(self.policy(inps, deterministic=deterministic)[0]), [{} for i in range(self.num_tasks)]]
        assert(not np.isnan(action[0]).any())
        return action

    def forward(self, obs, context): 
        # used at sac training
        
        ''' given context, get statistics under the current policy of a set of observations '''
        belief, rep = self.infer_posterior(context)
        # shape is (num_tasks, ...)
        
        t,s,b, _ = obs.size()
        obs = ptu.flatten(obs.contiguous())
        assert(belief.size(0) == t)
        belief = belief.view(t,1,1,-1)
        assert(len(belief.size()) == 3)        
        belief = ptu.flatten(belief.repeat(1,1,b,1))
        #if debug_config.IGNOREZ: そのままでOK
        rep = rep.view(t,1,1,-1)
        rep = ptu.flatten(rep.repeat(1,1,b,1))

        if self.algo_params['noisy_input'] == 1 or\
           self.algo_params['noisy_input'] == 2:
            noise = ptu.flatten(self.noise.view(t,s,1,-1).repeat(1,1,b, 1))

           
        if self.algo_params['film_gen_with_full_inputs']:
            inps = [torch.cat([belief, obs], dim=1), obs]
        else:
            inps = [belief, obs]
        if self.algo_params['noisy_input'] == 1 or\
           self.algo_params['noisy_input'] == 2:
            inps[self.algo_params['noisy_input'] - 1] = torch.cat([inps[self.algo_params['noisy_input'] - 1], noise], dim=1)

        policy_outputs = self.policy(inps, reparameterize=True, return_log_prob=True, middle_output=True)

        return policy_outputs, belief, rep

    def set_noise_for_sac_training(self, num_tasks, num_sac_set):
        dist = torch.distributions.Normal(0,1)
        self.noise = dist.sample((num_tasks, num_sac_set)).to(ptu.device)

    def set_noise(self, num_tasks=None):
        dist = torch.distributions.Normal(0,1)
        if num_tasks != None:
            num_t = num_tasks
        else:
            num_t = self.num_tasks

        if self.algo_params['noisy_input'] == 0:
            pass
        elif self.algo_params['noisy_input'] == 1  or\
             self.algo_params['noisy_input'] == 2:
            self.noise = dist.sample((num_t, 1)).to(ptu.device) 
        else:
            raise NotImplementedError

    def reset_noise(self, taskid):
        dist = torch.distributions.Normal(0,1)
        if self.algo_params['noisy_input'] == 1  or\
           self.algo_params['noisy_input'] == 2:
            self.noise[taskid][0] = dist.sample()


    def reset(self, num_tasks):# 
        # reset distribution over z to the prior
        mean = ptu.zeros(num_tasks, self.latent_dim)
        std = ptu.ones(num_tasks, self.latent_dim)
        self.z = self.z_means = mean
        self.z_stds = std
        #self.integrator.reset(num_tasks)
        # reset the context collected so far
        self.context = None

        #self.representation_buffer = []
 
        self.representation_buffer = ptu.zeros(num_tasks, self.max_context_length, self.encoder.output_size)
        #self.representation_buffer = ptu.zeros(self.max_context_length, self.encoder.output_size)
        self.representation_count = 0
        self.current_representation = ptu.zeros(num_tasks,  self.encoder.output_size)
        self.num_tasks = num_tasks
        #self.noise = ptu.zeros(num_tasks,  1)
        
    def update_representation_buffer_batch(self, reps): 
        if self.representation_count > self.max_context_length - 1:
            newbuf = self.representation_buffer[:, 1:]
            self.representation_buffer = torch.cat([newbuf, reps[:,None,:]], dim=1) 
            self.current_representation = self.representation_buffer.sum(1)
        else:
            self.representation_buffer[:, self.representation_count] = reps
            self.representation_count += 1
            self.current_representation += reps
        return self.current_representation
    
    def update_representation_buffer(self, rep):
        if self.representation_count > self.max_context_length -1 :
            self.representation_buffer[0] = torch.cat([self.representation_buffer[0][1:], rep], dim=0)
            self.current_representation = self.representation_buffer.sum(1) 
        else:
            self.representation_buffer[0][self.representation_count] = rep
            self.representation_count += 1
            self.current_representation += rep
        return self.current_representation
    def empty_belief(self, size):
        rep = ptu.zeros(size+[self.encoder.output_size])
        self.z_means = ptu.zeros(size+[self.latent_dim]) 
        self.z_stds = ptu.ones(size+[self.latent_dim])
        dim = len(size)
        
        return torch.cat([self.z_means, self.z_stds],dim=dim).detach(), rep.detach()
        
# sampling from env, sac training, cvae training
    def infer_posterior_batch(self, new_data=None, incremental=True):
        # None and torch.Tensor can not be used as "tensor == None"
        if incremental: #for sampling from env
            # called after update_context
            assert(new_data == None)
            with torch.no_grad():
                enc = self.encoder(self.new_data)
                rep = self.update_representation_buffer_batch(enc)
                mean, std = self.integrator(rep)
                
            if debug_config.IGNOREZ:
                s = mean.size()
                m = ptu.zeros(s)
                v = ptu.ones(s)
                return torch.cat([m,v], dim=1), rep
            self.z_means = mean
            self.z_stds = std
            return torch.cat([mean, std], dim=1), rep
    def infer_posterior(self, new_data=None, incremental=False):#
        # None and torch.Tensor can not be used as "tensor == None"
        if incremental: #for sampling from env
            # called after update_context
            assert(new_data == None)
            with torch.no_grad():
                enc = self.encoder(torch.squeeze(self.new_data,dim=1)) 
                rep = self.update_representation_buffer(enc)
                mean, std = self.integrator(rep)
            if debug_config.IGNOREZ:
                s = mean.size()
                m = ptu.zeros(s)
                v = ptu.ones(s)
                return torch.cat([m,v], dim=1), rep
            self.z_means = mean
            self.z_stds = std
            return torch.cat([mean, std], dim=1), rep

        # sac training
        rep = torch.sum(self.encoder(new_data), dim=2) 
        self.z_means, self.z_stds = self.integrator(rep, dim=2)
        if debug_config.IGNOREZ:
            s = self.z_means.size()
            m = ptu.zeros(s)
            v = ptu.ones(s)
            return torch.cat([m,v], dim=2).detach(), rep.detach()
        return torch.cat([self.z_means, self.z_stds],dim=2).detach(), rep.detach()

    def infer_next_posterior(self, new_data, representation):
        # for sac training
        rep = self.encoder(new_data) + representation
        self.z_means, self.z_stds = self.integrator(rep)
        if debug_config.IGNOREZ:
            s = self.z_means.size()
            m = ptu.zeros(s)
            v = ptu.ones(s)
            return torch.cat([m,v],dim=1), rep
        return torch.cat([self.z_means, self.z_stds],dim=1).detach(), rep.detach()

    def get_action_batch(self, obs, beliefs):
        if self.algo_params['film_gen_with_full_inputs']:
            inps = [torch.cat([beliefs, obs], dim=1), obs]
        else:
            inps = [beliefs, obs]
        return self.policy(inps, reparameterize=True, return_log_prob=True, middle_output=True)

    # TODO: fix if use these functions
    # for no_vf
    def get_next_action(self, next_obs, next_beliefs):
        inps = [next_beliefs, next_obs]
        #
        if self.algo_params['film_gen_with_full_inputs']:
            inps = [torch.cat([next_beliefs, next_obs], dim=1), next_obs]

        out = self.policy(inps, reparameterize=False, return_log_prob=True)
        return out[0], out[3] # = action, logpi

    ######### for sac sequential training ##########

    def init_belief(self, contexts):
        init_rep = torch.sum(self.encoder(contexts), dim=2).detach() # (task, sac_set, ...)
        init_belief = torch.cat(self.integrator(init_rep, dim=2), dim=2).detach()
        assert(len(init_rep.size()) == 3)
        t,s,r = init_rep.size()
        self.representation_buffer = ptu.zeros(self.max_context_length,t,s,r)
        self.representation_count = 0
        self.representation_buffer[self.representation_count] = init_rep 
        self.representation_count += 1
        return init_belief
    
    def update_belief(self, new_data):#
        with torch.no_grad():
            enc = self.encoder(new_data)  
            rep = self.representation_buffer[self.representation_count-1]
            rep = enc + rep
            mean, std = self.integrator(rep, dim=2) #(task, sac_set, ...)

            #self.update_representation_buffer(rep[None, ...])
            if self.representation_count > self.max_context_length -1 :
                if self.eval_mode and self.algo_params['fix_context_at_eval']:
                # do nothing
                    pass
                else:
                    self.representation_buffer -= self.representation_buffer[0]
                    self.representation_buffer = torch.cat([self.representation_buffer[1:], rep[None,...]], dim=0)
            else:
                self.representation_buffer[self.representation_count] = rep[None,...]
                self.representation_count += 1

            self.z_means = mean
            self.z_stds = std
            return torch.cat([mean, std], dim=2)
    def sequencial_updated_belief(self, contexts, additional_data):
        beliefs = []
        bel = self.init_belief(contexts)
        beliefs.append(bel)
        t, s, b, _ = additional_data.size()

        if debug_config.IGNOREZ:
            size = bel.size()
            belief = ptu.zeros(size)
            return torch.stack([belief for i in range(b+1)], dim=2)

        for i in range(b):
            bel = self.update_belief(additional_data[:, :, i])
            beliefs.append(bel)            
        return torch.stack(beliefs, dim=2)
    ######### for sac sequential training ##########
