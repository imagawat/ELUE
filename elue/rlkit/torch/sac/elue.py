from collections import OrderedDict
import numpy as np
import random

import torch
import torch.optim as optim
from torch import nn as nn
import gtimer as gt
from rlkit.core import logger 
from rlkit.data_management.path_builder import PathBuilder
import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.core.rl_algorithm import MetaRLAlgorithm
from rlkit.samplers.parallel import ParallelEnvExecutor
import debug_config
import rlkit.torch.sac.misc
from rlkit.core import additional_csv_writer

class ELUESoftActorCritic(MetaRLAlgorithm):
    def __init__(
            self,
            ml,
            env,
            latent_dim,
            nets,
            tasks,
            sampler,
            eval_env,
            num_eval_tasks,
            num_processes,
            policy_lr=1e-3,
            qf_lr=1e-3,
            vf_lr=1e-3,
            context_lr=1e-3,
            kl_lambda=1.,
            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,
            policy_pre_activation_weight=0.,
            optimizer_class=optim.Adam,
            loss_function=nn.MSELoss,
            use_information_bottleneck=True,
            use_next_obs_in_context=False,
            sparse_rewards=False,
            soft_target_tau=1e-2,
            plotter=None,
            render_eval_paths=False,
            reset_belief_rate= 1, # the agent resets belief every episode when it samples from env
            eval_mode=True,
            **kwargs
    ):
        super().__init__(
            ml=ml,
            env=env,
            tasks=tasks,
            agent=nets[0],
            sampler=sampler,
            eval_env=eval_env,
            eval_tasks=tasks, #temp TODO: remove....?
            num_processes=num_processes,
            **kwargs
        )
        self.num_eval_tasks = num_eval_tasks        
        self.reset_belief_rate = reset_belief_rate
        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.latent_dim = latent_dim
        """
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()
        self.vib_criterion = nn.MSELoss()
        self.l2_reg_criterion = nn.MSELoss()
        """
        self.loss_function = loss_function() #reduction='sum' ????

        #
        self.kl_lambda = kl_lambda

        self.use_information_bottleneck = use_information_bottleneck
        self.sparse_rewards = sparse_rewards
        self.use_next_obs_in_context = use_next_obs_in_context

        self.qf1, self.qf2, self.vf = nets[1:]
        if kwargs['no_vf']:
            self.target_qf1 = self.qf1.copy()
            self.target_qf2 = self.qf2.copy()
        else:
            self.target_vf = self.vf.copy()
        self.initial_policy = None
        self.num_steps = kwargs['num_steps']
        self.num_context_per_train_step = kwargs['num_context_per_train_step']
        self.num_sac_set_per_train_step = kwargs['num_sac_set_per_train_step']
        self.rand_length_train_rate = kwargs['rand_length_train_rate']
        self.max_context_length = kwargs['max_context_length']
        self.num_init_embedding_train = kwargs['num_init_embedding_train']
        self.num_embedding_train_steps_per_itr = kwargs['num_embedding_train_steps_per_itr']
        self.entropy_coeff = kwargs['entropy_coeff']
        self.grad_clip = kwargs['grad_clip'] #TODO:FIX temp value
        self.optimizer_class = optimizer_class
        self.eval_mode = eval_mode
        self.policy_lr = policy_lr
        self.qf_lr = qf_lr
        self.vf_lr = vf_lr
        self.context_lr = context_lr
        self.two_step_sac = kwargs['two_step_sac']
        if kwargs['max_rand_length'] < 0:
            self.max_rand_length = self.max_context_length - 1
            if self.two_step_sac:
                self.max_rand_length -= 1
        else:
            self.max_rand_length = kwargs['max_rand_length']
        if eval_mode:
            if debug_config.IGNOREZ:
                self.burn_in_iteration_size = 0
                self.burn_in = False
            else:
                self.burn_in_iteration_size = max(kwargs['burn_in_iteration_size'], 0)
                self.burn_in = True
            self.num_embedding_train_steps_per_itr = self.num_train_steps_per_itr
        else:
            self.burn_in_iteration_size = 0
            self.burn_in = False
        self.reset_belief_at_eval =  kwargs['reset_belief_at_eval']
        self.train_steps_gamma = self.algo_params['train_steps_gamma']
        self.bottleneck = True if kwargs['bottleneck_layerid'] >= 0 else False      
        self.q_bottleneck_coeff = self.algo_params['q_bottleneck_coeff']
        self.v_bottleneck_coeff = self.algo_params['v_bottleneck_coeff']
        self.pi_bottleneck_coeff = self.algo_params['pi_bottleneck_coeff']
  
        self.num_write = 10 
        self.num_episodes = 0
    def set_initial_policy(self):
        self.initial_policy = self.agent.policy.copy()
    def set_optimizers(self):
        if True: ########################not self.eval_mode:
            self.policy_optimizer = self.optimizer_class(
                self.get_optimizers_inputs(self.agent.policy),
                lr=self.policy_lr,
            )

            self.qf1_optimizer = self.optimizer_class(
                self.get_optimizers_inputs(self.qf1),
                lr=self.qf_lr,
            )
            self.qf2_optimizer = self.optimizer_class(
                self.get_optimizers_inputs(self.qf2),
                lr=self.qf_lr,
            )
            self.vf_optimizer = self.optimizer_class(
                self.get_optimizers_inputs(self.vf),
                lr=self.vf_lr,
            )
            self.cvae_optimizer = self.optimizer_class(
                self.agent.cvae_params,
                lr=self.context_lr,
            )
        else: #####################################
            self.qf1_optimizer = self.optimizer_class(
                self.qf1.parameters_for_adaptation(),
                lr=self.qf_lr,
            )
            self.qf2_optimizer = self.optimizer_class(
                self.qf2.parameters_for_adaptation(),
                lr=self.qf_lr,
            )
            self.vf_optimizer = self.optimizer_class(
                self.vf.parameters_for_adaptation(),
                lr=self.vf_lr,
            )
            self.policy_optimizer = self.optimizer_class(
                self.agent.policy.parameters_for_adaptation(),
                lr=self.policy_lr,
            )
            self.agent.encoder.train(False)
            self.agent.integrator.train(False)
            #self.agent.decoder.train(False)
            self.agent.rews_decoder.train(False)
            if self.agent.use_nobs_decoder:
                self.agent.nobs_decoder.train(False)
            # don't train belief net at eval_mode
        
    def get_optimizers_inputs(self, net):
        if self.agent.mlp > 0:
            return net.parameters()
        inps = [{'params': params} for params in net.ground_params_list]
        if debug_config.MODIFYFILM:
            inps = net.parameters()
        else:
            inps.append({'params': net.film_gennet.parameters(), 'lr': self.algo_params['film_gennet_lr']})
        return inps
    ###### Torch stuff #####
    @property
    def networks(self):
        if self.algo_params['no_vf']:
            return self.agent.networks + [self.agent] + [self.qf1, self.qf2, self.target_qf1, self.target_qf2]
        n = self.agent.networks + [self.agent] + [self.qf1, self.qf2, self.vf, self.target_vf]
        if self.eval_mode and self.algo_params['policy_kl_loss_coeff'] > 0 and not debug_config.IGNOREZ:
            n = n + [self.initial_policy]
        return n


    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device=None):
        if device == None:
            device = ptu.device
        for net in self.networks:
            net.to(device)

    ##### Data handling #####
    
    def unpack_batch(self, batch, sparse_reward=False):
        ''' unpack a batch and return individual elements '''
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        if sparse_reward:
            r = batch['sparse_rewards'][None, ...]
        else:
            r = batch['rewards'][None, ...]
        no = batch['next_observations'][None, ...]
        t = batch['terminals'][None, ...]
        return [o, a, r, no, t]

    def sample_context_set(self, indices, context_length, num_context = 1): #これを消して，sample_batchesに統合
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        bss = [context_length for i in range(num_context)]
        batches = [ptu.np_to_pytorch_batch(self.replay_buffer.random_batches(idx, batch_sizes=bss)) for idx in indices]
        context = [self.unpack_batch(batch, sparse_reward=self.sparse_rewards) for batch in batches]
        # group like elements together
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [torch.cat(x, dim=0) for x in context]
        # full context consists of [obs, act, rewards, next_obs, terms]
        context = context[:-1]
        return context

    def sample_sequences(self, indices, lengths, with_term=False):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        batches = [ptu.np_to_pytorch_batch(self.replay_buffer.random_batches(idx, batch_sizes=lengths, sequence=True)) for idx in indices]
        context = [self.unpack_batch(batch, sparse_reward=self.sparse_rewards) for batch in batches]
        # group like elements together
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [torch.cat(x, dim=0) for x in context]

        if with_term:
            return context
        # full context consists of [obs, act, rewards, next_obs, terms]
        context = context[:-1]
        return context
    def sample_batches(self, indices, lengths, with_term=False):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        batches = [ptu.np_to_pytorch_batch(self.replay_buffer.random_batches(idx, batch_sizes=lengths, sequence=False)) for idx in indices]
        context = [self.unpack_batch(batch, sparse_reward=self.sparse_rewards) for batch in batches]
        # group like elements together
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [torch.cat(x, dim=0) for x in context]

        if with_term:
            return context
        # full context consists of [obs, act, rewards, next_obs, terms]
        context = context[:-1]
        return context
    def two_step_sample(self, indices, lengths):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        # 
        batches = [ptu.np_to_pytorch_batch(self.replay_buffer.random_two_step_batches(idx, batch_sizes=lengths)) for idx in indices]
        context = [self.unpack_batch(batch, sparse_reward=self.sparse_rewards) for batch in batches]
        # group like elements together
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [torch.cat(x, dim=0) for x in context]

        # full context consists of [obs*2, act*2, rewards*2, terms*2]
        ret = []
        for x in context:        #(task, sac_set, 2, batchsize ...)
            ts = torch.chunk(x, 2, dim=2)
            for t in ts:
                ret.append(t.squeeze(dim=2))
        return ret


    

    ##### Training #####
    def _do_training(self, indices, itr):
        gt.subdivide('sub')
        if not self.eval_mode: 
            self._train_embedding(indices, itr)
            gt.stamp('train_embedding')
        if not self.eval_mode and self.two_step_sac:
            self._train_two_step_sac(indices, itr)
        else:
            self._train_sac(indices, itr)
        gt.stamp('train_sac')
        gt.end_subdivision()

    #TODO: remove
    def _training_embedding_with_sac(self, indices, itr, rate):
        gt.subdivide('sub')
        if not self.eval_mode: 
            self._train_embedding(indices, itr)
        gt.stamp('train_embedding')
        if itr % rate == 0:
            if not self.eval_mode and self.two_step_sac:
                self._train_two_step_sac(indices, itr//rate)
            else:
                self._train_sac(indices, itr//rate)
        gt.stamp('train_sac')
        gt.end_subdivision()
    #TODO: remove
    def _training_sac_with_embedding(self, indices, itr, rate):
        gt.subdivide('sub')
        if not self.eval_mode and itr%rate == 0:
            self._train_embedding(indices, itr//rate)
        gt.stamp('train_embedding')
        if not self.eval_mode and self.two_step_sac:
            self._train_two_step_sac(indices, itr)
        else:
            self._train_sac(indices, itr)
        gt.stamp('train_sac')
        gt.end_subdivision()


    def _min_q(self, obs, actions, belief): # output q value for policy and v target
        if self.algo_params['film_gen_with_full_inputs']:
            inps = [torch.cat([belief.detach(), obs, actions], dim=1), torch.cat([obs, actions], dim=1)]
        else:
            inps = [belief.detach(), torch.cat([obs, actions], dim=1)]
                
        q1, _ = self.qf1(inps, reparameterize=False, denoise=True) #
        q2, _ = self.qf2(inps, reparameterize=False, denoise=True) #
        if self.algo_params['qf_ave_as_target']:
            q = torch.mean(torch.stack([q1,q2],dim=1), dim=1)
        else:
            q = torch.min(q1, q2)
        return q

    def _update_target_network(self, itr):
        if self.algo_params['drastic_target_update']:
            if itr == 0:
                ptu.soft_update_from_to(self.vf, self.target_vf, 1.0)
            else:
                ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)
        else:
            ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)


    def _train_embedding(self, indices, itr, init_train=False):
        if debug_config.IGNOREZ:
            return 0
        gt.subdivide('train_embedding')
        num_context = self.num_context_per_train_step
        if not self.algo_params['max_length_embedding_train'] and itr+1 % self.rand_length_train_rate == 0:
            context_length = random.randint(1, self.max_context_length)
        else:
            context_length = self.max_context_length
        lengths = [context_length for i in range(num_context)] 
        if self.algo_params['sequential']:
            obs, actions, rewards, next_obs= self.sample_sequences(indices, lengths)
        else:
            obs, actions, rewards, next_obs= self.sample_batches(indices, lengths)
        gt.stamp('sampling_context')
        loss, kl_loss, info_debug = self.agent.cvae_reconstruction_loss_batch([obs, actions, rewards, next_obs], indices)

        
        if init_train:
            pass#print(ptu.get_numpy(loss), ptu.get_numpy(kl_loss))
        else:
            if itr == 0:
                logger.record_tabular('Rec Loss', ptu.get_numpy(loss))
                logger.record_tabular('KL Loss', ptu.get_numpy(kl_loss))
            elif itr + 1 == self.num_embedding_train_steps_per_itr:
                logger.record_tabular('Rec Loss Last', ptu.get_numpy(loss))
                logger.record_tabular('KL Loss Last', ptu.get_numpy(kl_loss))
        loss += self.kl_lambda * kl_loss
        gt.stamp('calc_loss')
        self.cvae_optimizer.zero_grad()
        loss.backward()                
        torch.nn.utils.clip_grad_norm_(self.agent.cvae_params, self.grad_clip)
        self.cvae_optimizer.step()
        gt.stamp('update')
        gt.end_subdivision()

        return 0


    def get_epoch_snapshot(self, epoch):
        # NOTE: overriding parent method which also optionally saves the env
        if self.algo_params['no_vf']:
            if self.agent.use_nobs_decoder:
                snapshot = OrderedDict(
                qf1=self.qf1.state_dict(),
                qf2=self.qf2.state_dict(),
                policy=self.agent.policy.state_dict(),
                target_qf1=self.target_qf1.state_dict(),
                target_qf2=self.target_qf2.state_dict(),
                encoder=self.agent.encoder.state_dict(),
                integrator=self.agent.integrator.state_dict(),
                rews_decoder=self.agent.rews_decoder.state_dict(),
                nobs_decoder=self.agent.nobs_decoder.state_dict(),
                )
            else:
                snapshot = OrderedDict(
                qf1=self.qf1.state_dict(),
                qf2=self.qf2.state_dict(),
                policy=self.agent.policy.state_dict(),
                target_qf1=self.target_qf1.state_dict(),
                target_qf2=self.target_qf2.state_dict(),
                encoder=self.agent.encoder.state_dict(),
                integrator=self.agent.integrator.state_dict(),
                rews_decoder=self.agent.rews_decoder.state_dict(),
                )

            return snapshot

        if self.agent.use_nobs_decoder:
            snapshot = OrderedDict(
            qf1=self.qf1.state_dict(),
            qf2=self.qf2.state_dict(),
            policy=self.agent.policy.state_dict(),
            vf=self.vf.state_dict(),
            target_vf=self.target_vf.state_dict(),
            encoder=self.agent.encoder.state_dict(),
            integrator=self.agent.integrator.state_dict(),
            rews_decoder=self.agent.rews_decoder.state_dict(),
            nobs_decoder=self.agent.nobs_decoder.state_dict(),
        )
        else:
            snapshot = OrderedDict(
            qf1=self.qf1.state_dict(),
            qf2=self.qf2.state_dict(),
            policy=self.agent.policy.state_dict(),
            vf=self.vf.state_dict(),
            target_vf=self.target_vf.state_dict(),
            encoder=self.agent.encoder.state_dict(),
            integrator=self.agent.integrator.state_dict(),
            rews_decoder=self.agent.rews_decoder.state_dict(),
        )

        return snapshot
    def collect_data(self, num_samples, update_belief=True, add_to_enc_buffer=True):
        '''
        get trajectories from current env in batch mode with given policy
        collect complete trajectories until the number of collected transitions >= num_samples

        :param agent: policy to rollout
        :param num_samples: total number of transitions to sample
        :param add_to_enc_buffer: whether to add collected data to encoder replay buffer
        '''
            
        num_transitions = 0
        epirew_sum = 0
        success_sum = 0
        num_episodes = 0
        if self.eval_mode and not self.reset_belief_at_eval: #in eval, keep belief
            max_trajs = np.inf
        elif self.reset_belief_rate < 1:
            max_trajs = np.inf
        else:
            max_trajs = self.reset_belief_rate
            
        while num_transitions < num_samples:
            if not (self.eval_mode and not self.reset_belief_at_eval):
                self.agent.reset(num_tasks = 1)

            paths, n_samples = self.sampler.obtain_samples(deterministic=self.algo_params['deterministic_sampling'],
                                                           max_samples=num_samples - num_transitions,
                                                           max_trajs=max_trajs,
                                                           accum_context=True, update_belief=update_belief)      
            num_transitions += n_samples
            self.replay_buffer.add_paths(self.task_idx, paths)

            for path in paths:
                if path['terminals'][-1]:
                    epirew = path['rewards'].sum()
                    epirew_sum += epirew
                    num_episodes += 1
                    if 'env_infos' in path and 'success' in path['env_infos'][-1]:
                        success_sum += path['env_infos'][-1]['success']

                    
                    if self.eval_mode and self.num_write > self.num_episodes:
                        self.num_episodes += 1
                        additional_csv_writer.csv_write({'n_env_steps':self._n_env_steps_total, 'epinum': self.num_episodes, 'epirew': epirew}, (self.num_episodes == 1))
                        if 'env_infos' in path and 'fingerXY' in path['env_infos'][-1]:      
                            additional_csv_writer.trajectory_write(path['env_infos'], self.num_episodes)



                    
        self._n_env_steps_total += num_transitions
        gt.stamp('sample')
        return epirew_sum, num_episodes, success_sum    
    def collect_data_parallelly(self, taskset, num_samples, update_belief=True, add_to_enc_buffer=True, initial=False):
        # initial data collection

        # 
        if self.eval_mode:          
            raise NotImplementedError
        epirew_sum = 0
        success_sum = 0
        num_episodes = 0
        ### reset beliefはobtain_samplesの中でやる．（一部のタスクのみ終わるとかあり得るので）
        #基本的には，obtain_sample一回よんだら，tsについては必要なデータは全て得られるようにする
        num_loops = len(taskset['tasks']) // self.num_processes
        res = len(taskset['tasks']) % self.num_processes
        if res > 0:
            # in this case #tasks is res, num_process, num_process,...
            # additional loop
            num_loops += 1
            num_tasks = res

        else:
            num_tasks= self.num_processes
        for loopid in range(num_loops):
            if loopid > 1:
                num_tasks = self.num_processes 
            self.sampler.vecenv.set_tasks(taskset['tasks'][loopid*self.num_processes: loopid*self.num_processes + num_tasks])
            #if not self.eval_mode:
            self.agent.reset(num_tasks)
            #TODO: reset belief like collect_data()
            paths, n_samples = self.sampler.obtain_samples(max_samples=num_samples, accum_context=True, update_belief=update_belief)
            for i, taskid in enumerate(taskset['ids'][loopid*self.num_processes: loopid*self.num_processes + num_tasks]):
                self.replay_buffer.add_paths(taskid, paths[i])
                if not initial:
                    for path in paths[i]:
                        if path['terminals'][-1]:
                            epirew_sum += path['rewards'].sum()
                            num_episodes += 1
                            if 'env_infos' in path and 'success' in path['env_infos'][-1]:
                                success_sum += path['env_infos'][-1]['success']
            self._n_env_steps_total += n_samples

        #eval_epirew_sum = 0
        #eval_num_episodes = 0
        gt.stamp('sample')
        if not initial:
            ave = (1.0*epirew_sum)/num_episodes if num_episodes>0 else 0        
            ave2 = (1.0*success_sum)/num_episodes if num_episodes>0 else 0        
            logger.record_tabular('Average Episode Reward', ave) 
            logger.record_tabular('Success Rate', ave2) 
            logger.record_tabular('#Episodes', num_episodes) 


    def target_q_values(self, obs, acts, beliefs):
        with torch.no_grad():
            inps = [beliefs, torch.cat([obs, acts], dim=1)]
            if self.algo_params['film_gen_with_full_inputs']:
                inps = [torch.cat([beliefs, obs, acts], dim=1), torch.cat([obs, acts], dim=1)]            
            q1, _ = self.target_qf1(inps,reparameterize=False, denoise=True)
            q2, _ = self.target_qf2(inps,reparameterize=False, denoise=True)
            if self.algo_params['qf_ave_as_target']:
                q = torch.mean(torch.stack([q1,q2],dim=1), dim=1)
            else:
                q = torch.min(q1, q2)
        return q 
    def update_target_qf(self, itr):
        if self.algo_params['drastic_target_update']:
            if itr == 0:
                ptu.soft_update_from_to(self.qf1, self.target_qf1, 1.0)
                ptu.soft_update_from_to(self.qf2, self.target_qf2, 1.0)
            else:
                ptu.soft_update_from_to(self.qf1, self.target_qf1, self.soft_target_tau)
                ptu.soft_update_from_to(self.qf2, self.target_qf2, self.soft_target_tau)
        else:
            ptu.soft_update_from_to(self.qf1, self.target_qf1, self.soft_target_tau)
            ptu.soft_update_from_to(self.qf2, self.target_qf2, self.soft_target_tau)

    def _train_two_step_sac(self, indices, itr): # TODO: remove
        gt.subdivide('train_sac')
        if itr+1 % self.rand_length_train_rate == 0:

            context_length = random.randint(1, self.max_rand_length-2)
        else:
            context_length = self.max_context_length-2         #

        if context_length > 0:
            lengths = [context_length for i in range(self.num_sac_set_per_train_step)]
            cobs, cacts, crews, cnobs = self.sample_batches(indices, lengths, with_term=False) #(task, sac_set, context_length, ...)
        lengths = [self.batch_size for i in range(self.num_sac_set_per_train_step)]
        obs, nobs, acts, nacts, rews, nrews, _, nnobs, _, nterms = self.two_step_sample(indices, lengths) #(task, sac_set, batchsize, ...)        
        gt.stamp('sample_sac')
        if self.use_next_obs_in_context:
            if context_length > 0:
                context = torch.cat([cobs, cacts, crews, cnobs], dim=3)
            additional1 = torch.cat([obs, acts, rews, nobs], dim=3)
            additional2= torch.cat([nobs, nacts, nrews, nnobs], dim=3)
        else:
            if context_length > 0:
                context = torch.cat([cobs, cacts, crews], dim=3)
            additional1 = torch.cat([obs, acts, rews], dim=3)
            additional2 = torch.cat([nobs, nacts, nrews], dim=3)

        t, s, b, _ = obs.size()
        if context_length > 0:
            init_task_belief, init_task_representation = self.agent.infer_posterior(context) #(task, sac_set, ...)
        else:
            init_task_belief, init_task_representation = self.agent.empty_belief([t,s,1]) #(task, sac_set, ...)
        nobs, nacts, nrews, nnobs, nterms, additional1, additional2 = ptu.flatten_all(
            [nobs, nacts, nrews, nnobs, nterms, additional1, additional2]
        )
        init_task_belief = ptu.flatten(init_task_belief.view(t,s,1,-1).repeat(1,1,b,1))
        init_task_representation = ptu.flatten(init_task_representation.view(t,s,1,-1).repeat(1,1,b,1))
            
        # belief update for inner update
        task_belief, task_representation = self.agent.infer_next_posterior(representation=init_task_representation, new_data=additional1)

        # belief update for SAC loss calculation and outer update
        task_next_belief, _ = self.agent.infer_next_posterior(representation=task_representation, new_data=additional2)

        data = [nobs, nacts, nrews, nnobs, nterms, task_belief, task_next_belief]
        self.update_sac(data, itr)

    def update_sac(self, data, itr):
        obs, actions, rewards, next_obs, terms, task_belief, task_next_belief = data 
        policy_outputs = self.agent.get_action_batch(obs, task_belief)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
        if self.bottleneck:
            mid, mid_diff_log_prob = policy_outputs[8:10]
        mu_diff, std_diff = torch.chunk(task_next_belief - task_belief, 2, dim=1)
        if self.algo_params['no_vf']: # TODO: remove
            next_new_actions, next_log_pi = self.agent.get_next_action(next_obs,task_next_belief)
            target_v_values = self.target_q_values(next_obs, next_new_actions, task_next_belief) - self.entropy_coeff * next_log_pi #FIX if use
        else:
            if self.algo_params['film_gen_with_full_inputs']:
                vinps = [torch.cat([task_belief.detach(), obs], dim=1), obs]
                tvinps = [torch.cat([task_next_belief.detach(), next_obs], dim=1), next_obs]
            else:
                vinps = [task_belief.detach(), obs]
                tvinps = [task_next_belief.detach(), next_obs]            
            v_pred, v_middle = self.vf(vinps, reparameterize=True, middle_output=True)                                                         
            with torch.no_grad():
                target_v_values, _ = self.target_vf(tvinps, reparameterize=False, denoise=True)
            # compute min Q on the new actions
            q_new_actions = self._min_q(obs, new_actions, task_belief)                                                                     
            # vf_loss                                            
            if self.bottleneck:
                v_target = q_new_actions - self.pi_bottleneck_coeff * mid_diff_log_prob - self.entropy_coeff * log_pi
                if self.algo_params["q_v_bottleneck"]:
                    vf_loss = self.loss_function(v_pred, v_target.detach()) - torch.mean(self.v_bottleneck_coeff * v_middle[1])
                else:
                    vf_loss = self.loss_function(v_pred, v_target.detach())
            else:
                v_target = q_new_actions - self.entropy_coeff * log_pi                                
                vf_loss = self.loss_function(v_pred, v_target.detach())                    
            self.vf_optimizer.zero_grad()
            vf_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.vf.parameters(), self.grad_clip)
            self.vf_optimizer.step()
            self._update_target_network(itr)
            
        if self.algo_params['film_gen_with_full_inputs']:
            qinps = [torch.cat([task_belief.detach(), obs, actions], dim=1), torch.cat([obs, actions], dim=1)]
        else:
            qinps = [task_belief.detach(), torch.cat([obs, actions], dim=1)]

        q1_pred, q1_middle = self.qf1(qinps, reparameterize=True, middle_output=True)                                         
        q2_pred, q2_middle = self.qf2(qinps, reparameterize=True, middle_output=True)
                                                         
        # qf_loss                                                     
        rewards = rewards * self.reward_scale                                 
        q_target = rewards + (1. - terms) * self.discount * target_v_values
        """
        qf1_diff = torch.mean((q1_pred - q_target) ** 2)
        qf2_diff = torch.mean((q2_pred - q_target) ** 2)             
        """
        qf1_diff = self.loss_function(q1_pred, q_target)
        qf2_diff = self.loss_function(q2_pred, q_target)             
        qf_diff = torch.mean((q1_pred - q2_pred) ** 2)
        if self.bottleneck and self.algo_params['q_v_bottleneck']:
            qf_loss = qf1_diff + qf2_diff - self.q_bottleneck_coeff * torch.mean(q1_middle[1] + q2_middle[1])
        else:
            qf_loss = qf1_diff + qf2_diff
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        qf_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qf1.parameters(), self.grad_clip)
        #print("before update: ", self.qf1.film_vars)
        #print("    ref before update: ", self.qf1.layers[0].weight[0])
        self.qf1_optimizer.step()
        torch.nn.utils.clip_grad_norm_(self.qf2.parameters(), self.grad_clip)
        #print("after update: ", self.qf1.film_vars)
        #print("    ref after update: ", self.qf1.layers[0].weight[0])
        self.qf2_optimizer.step()
        if self.algo_params['no_vf']:
            self.update_target_qf(itr)
            q_new_actions = self._min_q(obs, new_actions, task_belief) # used in policy_loss calc

    
        # policy_loss                                             
        log_policy_target = q_new_actions                                     
        if self.bottleneck:
            policy_loss = (                                                 
                self.entropy_coeff * log_pi + self.pi_bottleneck_coeff * mid_diff_log_prob - log_policy_target                                         
            ).mean()
        else:                                          
            policy_loss = (                                                 
                self.entropy_coeff * log_pi - log_policy_target                                         
            ).mean()
        
        if self.algo_params['additional_policy_regs']:
            mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()                     
            std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()                     
            pre_tanh_value = policy_outputs[7]                                         
            pre_activation_reg_loss = self.policy_pre_activation_weight * (                         
                (pre_tanh_value**2).sum(dim=1).mean()                                     
            )                                                         
            policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss                     
            policy_loss = policy_loss + policy_reg_loss
        if self.eval_mode and self.algo_params['policy_kl_loss_coeff'] > 0 and not debug_config.IGNOREZ and self.entropy_coeff > 0:

            if self.algo_params['film_gen_with_full_inputs']:# remove this case
                inps = [torch.cat([task_belief, obs], dim=1), obs]
            else:
                inps = [task_belief, obs]
            log_init_pi, init_middle = self.initial_policy.get_action_log_prob(inps, new_actions) 
            policy_loss -= self.algo_params['policy_kl_loss_coeff'] * log_init_pi.mean() 
        #if not self.eval_mode or not self.burn_in:
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.policy.parameters(), self.grad_clip)
        self.policy_optimizer.step()
            
        gt.stamp('update')
        gt.end_subdivision()
        if itr == 0:
            if not self.algo_params['no_vf']:
                logger.record_tabular('VF Loss', ptu.get_numpy(vf_loss))
            logger.record_tabular('QF Loss', ptu.get_numpy(qf_loss))
            logger.record_tabular('Policy Loss', ptu.get_numpy(policy_loss))
            logger.record_tabular('QF Diff', ptu.get_numpy(qf_diff))
            mu_diff = ptu.get_numpy(mu_diff)
            std_diff = ptu.get_numpy(std_diff)
            logger.record_tabular('Mu Diff Ave', np.mean(mu_diff))
            logger.record_tabular('Mu Diff Std', np.std(mu_diff))
            logger.record_tabular('Std Diff Ave', np.mean(std_diff))
            logger.record_tabular('Std Diff Std', np.std(std_diff))
            if self.bottleneck:
                logger.record_tabular('Policy Bt Loss', ptu.get_numpy(torch.mean(mid_diff_log_prob)))
                if self.algo_params["q_v_bottleneck"]:
                    logger.record_tabular('QF Bt Loss', ptu.get_numpy(torch.mean(q1_middle[1]+q2_middle[1])))
                    logger.record_tabular('VF Bt Loss', ptu.get_numpy(torch.mean(v_middle[1])))
        elif itr + 1 == self.num_train_steps_per_itr:
            if not self.algo_params['no_vf']:
                logger.record_tabular('VF Loss Last', ptu.get_numpy(vf_loss))
            logger.record_tabular('QF Loss Last', ptu.get_numpy(qf_loss))
            logger.record_tabular('Policy Loss Last', ptu.get_numpy(policy_loss))
            logger.record_tabular('QF Diff Last', ptu.get_numpy(qf_diff))           
            mu_diff = ptu.get_numpy(mu_diff)
            std_diff = ptu.get_numpy(std_diff)
            logger.record_tabular('Mu Diff Ave Last', np.mean(mu_diff))
            logger.record_tabular('Mu Diff Std Last', np.std(mu_diff))
            logger.record_tabular('Std Diff Ave Last', np.mean(std_diff))
            logger.record_tabular('Std Diff Std Last', np.std(std_diff))
        elif itr + 1 == self.rand_length_train_rate and not self.eval_mode:
            # at rand_length_train
            mu_diff = ptu.get_numpy(mu_diff)
            std_diff = ptu.get_numpy(std_diff)
            logger.record_tabular('Mu Diff Ave Rand', np.mean(mu_diff))
            logger.record_tabular('Mu Diff Std Rand', np.std(mu_diff))
            logger.record_tabular('Std Diff Ave Rand', np.mean(std_diff))
            logger.record_tabular('Std Diff Std Rand', np.std(std_diff))

    def _train_sac(self, indices, itr):
        if self.burn_in: #modified burn_in
            if itr == 0:
                if not self.algo_params['no_vf']:
                    logger.record_tabular('VF Loss',None)
                logger.record_tabular('QF Loss',None)
                logger.record_tabular('Policy Loss',None)
                logger.record_tabular('QF Diff',None)
                logger.record_tabular('Mu Diff Ave', None)
                logger.record_tabular('Mu Diff Std', None)
                logger.record_tabular('Std Diff Ave', None)
                logger.record_tabular('Std Diff Std', None)
                if self.bottleneck:
                    logger.record_tabular('Policy Bt Loss',None)
                    if self.algo_params["q_v_bottleneck"]:
                        logger.record_tabular('QF Bt Loss', None)
                        logger.record_tabular('VF Bt Loss', None)
            elif itr + 1 == self.num_train_steps_per_itr:
                if not self.algo_params['no_vf']:
                    logger.record_tabular('VF Loss Last', None)
                logger.record_tabular('QF Loss Last', None)
                logger.record_tabular('Policy Loss Last', None)
                logger.record_tabular('QF Diff Last', None)
                logger.record_tabular('Mu Diff Ave Last', None)
                logger.record_tabular('Mu Diff Std Last', None)
                logger.record_tabular('Std Diff Ave Last', None)
                logger.record_tabular('Std Diff Std Last', None)
            return 0
        gt.subdivide('train_sac')
        self.agent.set_noise_for_sac_training(len(indices), self.num_sac_set_per_train_step)
        if self.eval_mode:
            lengths = [self.batch_size for i in range(self.num_sac_set_per_train_step)]
            #TODO: sample batch or sequence
            if self.algo_params['sequential']:
                obs, actions, rewards, next_obs, terms = self.sample_sequences(indices, lengths, with_term=True) #(task, sac_set, batch_size, ...)
            else:
                obs, actions, rewards, next_obs, terms = self.sample_batches(indices, lengths, with_term=True) #(task, sac_set, batch_size, ...)
        
            gt.stamp('sample_sac')
            if self.num_processes > 1:
                task_belief = torch.cat([self.agent.z_means[0], self.agent.z_stds[0]], dim = 0) 
            else:
                task_belief = torch.cat([self.agent.z_means, self.agent.z_stds], dim = 1)            # use belief at sampling from env
            task_belief = task_belief.view(1,1,1,-1).repeat(1,self.num_sac_set_per_train_step,self.batch_size, 1)

            #flatten
            task_belief = ptu.flatten(task_belief)
            obs = ptu.flatten(obs)
            actions = ptu.flatten(actions)
            next_obs = ptu.flatten(next_obs)
            rewards = ptu.flatten(rewards)
            terms = ptu.flatten(terms)

            if self.algo_params['noisy_input'] == 1 or\
               self.algo_params['noisy_input'] == 2:
                noise = ptu.flatten(self.agent.noise.view(len(indices),self.num_sac_set_per_train_step,1,-1).repeat(1,1,self.batch_size, 1))
                if self.algo_params['noisy_input'] == 1:
                    task_belief = torch.cat([task_belief, noise], dim=1)
                elif  self.algo_params['noisy_input'] == 2:
                    obs = torch.cat([obs, noise], dim=1)

            task_next_belief = task_belief #

        elif self.algo_params['sequential']:
            #context_length = self.max_context_length-1#random.randint(1, self.max_context_length-1)
            if itr+1 % self.rand_length_train_rate == 0:
                context_length = random.randint(1, self.max_context_length-1)
            else:
                context_length = self.max_context_length-1         #
            lengths = [context_length + self.batch_size for i in range(self.num_sac_set_per_train_step)]

            obs, actions, rewards, next_obs, terms = self.sample_sequences(indices, lengths, with_term=True) #(task, sac_set, context_length+batch_size, ...)
        
            gt.stamp('sample_sac')
            cobs, obs = torch.split(obs, [context_length, self.batch_size], dim=2)
            cacts, actions = torch.split(actions, [context_length, self.batch_size], dim=2)
            crews, rewards = torch.split(rewards, [context_length, self.batch_size], dim=2)
            cnobs, next_obs = torch.split(next_obs, [context_length, self.batch_size], dim=2)
            _, terms = torch.split(terms, [context_length, self.batch_size], dim=2)
            if self.use_next_obs_in_context:
                context = torch.cat([cobs, cacts, crews, cnobs], dim=3)
                additional = torch.cat([obs, actions, rewards, next_obs], dim=3)
            else:
                context = torch.cat([cobs, cacts, crews], dim=3)
                additional = torch.cat([obs, actions, rewards], dim=3)

            beliefs = self.agent.sequencial_updated_belief(context, additional)
            task_belief = beliefs[:, :, :-1]
            task_next_belief = beliefs[:, :, 1:]

            #flatten
            task_belief = ptu.flatten(task_belief.contiguous())
            task_next_belief = ptu.flatten(task_next_belief.contiguous())
            obs = ptu.flatten(obs.contiguous())
            actions = ptu.flatten(actions.contiguous())
            next_obs = ptu.flatten(next_obs.contiguous())
            rewards = ptu.flatten(rewards.contiguous())
            terms = ptu.flatten(terms.contiguous())

            if self.algo_params['noisy_input'] == 1 or\
               self.algo_params['noisy_input'] == 2:
                noise = ptu.flatten(self.agent.noise.view(len(indices),self.num_sac_set_per_train_step,1,-1).repeat(1,1,self.batch_size, 1))
                if self.algo_params['noisy_input'] == 1:
                    task_belief = torch.cat([task_belief, noise], dim=1)
                elif  self.algo_params['noisy_input'] == 2:
                    obs = torch.cat([obs, noise], dim=1)

            gt.stamp('output_actions')
        else:

            if itr+1 % self.rand_length_train_rate == 0:
                context_length = random.randint(1, self.max_context_length-1)
            else:
                context_length = self.max_context_length-1         #
            lengths = [context_length + self.batch_size for i in range(self.num_sac_set_per_train_step)]
            obs, actions, rewards, next_obs, terms = self.sample_batches(indices, lengths, with_term=True) #(task, sac_set, context_length+batch_size, ...)
            gt.stamp('sample_sac')
            cobs, obs = torch.split(obs, [context_length, self.batch_size], dim=2)
            cacts, actions = torch.split(actions, [context_length, self.batch_size], dim=2)
            crews, rewards = torch.split(rewards, [context_length, self.batch_size], dim=2)
            cnobs, next_obs = torch.split(next_obs, [context_length, self.batch_size], dim=2)
            _, terms = torch.split(terms, [context_length, self.batch_size], dim=2)
            if self.use_next_obs_in_context:
                context = torch.cat([cobs, cacts, crews, cnobs], dim=3)
                additional = torch.cat([obs, actions, rewards, next_obs], dim=3)
            else:
                context = torch.cat([cobs, cacts, crews], dim=3)
                additional = torch.cat([obs, actions, rewards], dim=3)

            t, s, b, _ = obs.size()
            task_belief, task_representation = self.agent.infer_posterior(context) #(task, sac_set, ...)

            obs = ptu.flatten(obs.contiguous())
            actions = ptu.flatten(actions.contiguous())
            next_obs = ptu.flatten(next_obs.contiguous())
            additional =  ptu.flatten(additional.contiguous())
            rewards =  ptu.flatten(rewards.contiguous())
            terms =  ptu.flatten(terms.contiguous())
            task_belief = ptu.flatten(task_belief.view(t,s,1,-1).repeat(1,1,b,1))
            #if debug_config.IGNOREZ: nothing to do

            task_representation = ptu.flatten(task_representation.view(t,s,1,-1).repeat(1,1,b,1))


            if self.algo_params['noisy_input'] == 1 or\
               self.algo_params['noisy_input'] == 2:
                noise = ptu.flatten(self.agent.noise.view(len(indices),self.num_sac_set_per_train_step,1,-1).repeat(1,1,self.batch_size, 1))

            
            if self.algo_params['film_gen_with_full_inputs']:
                inps = [torch.cat([task_belief, obs], dim=1), obs]
            else:
                inps = [task_belief, obs]
                
            if self.algo_params['noisy_input'] == 1 or\
               self.algo_params['noisy_input'] == 2:
                inps[self.algo_params['noisy_input'] - 1] = torch.cat([inps[self.algo_params['noisy_input'] - 1], noise], dim=1)

            task_next_belief, _ = self.agent.infer_next_posterior(representation=task_representation, new_data=additional)


            gt.stamp('output_actions')
        data = [obs, actions, rewards, next_obs, terms, task_belief, task_next_belief]
        self.update_sac(data, itr)
        

    def init_collect_data_from_tasks(self):
        """
        init collection of data from envs
        """
        if self.num_processes == 1:
            if self.eval_mode:
                assert(len(self.tasks) == 1)
                self.agent.reset(num_tasks = 1) # and do not reset again...
                update = True
            else:
                update = False
            for i, task in enumerate(self.tasks):
                self.task_idx = i #self.train_task_idx(i)
                self.env.set_task(task)   #
                self.collect_data(self.num_initial_steps, update_belief=update)
        else:
            if self.eval_mode:
                self.agent.reset(self.num_processes)
                taskset = {'tasks': [self.tasks[0] for i in range(self.num_processes)], 'ids': [0 for i in range(self.num_processes)]}
                self.collect_data_parallelly(taskset, self.num_initial_steps//self.num_processes, update_belief=True, initial=True)
                print("m: ", self.agent.z_means)
                print("s: ", self.agent.z_stds)
            else:
                taskset = {'tasks': self.tasks, 'ids': [i for i in range(len(self.tasks))]}
                self.collect_data_parallelly(taskset, self.num_initial_steps, update_belief=False, initial=True)

    def validation(self, n_iter):
        #
        outputs = self.algo_params['validation_outputs'] # 
        max_n_episodes = len(outputs)
        n_collection_in_same_task = 1        
        if (n_iter + 1) % self.algo_params['eval_interval'] == 0:
            epirews = np.zeros(max_n_episodes) #
            successes = np.zeros(max_n_episodes) #
            lengths = np.zeros(max_n_episodes) #
            tasks = self.eval_env.sample_tasks(self.num_eval_tasks)
            for task in tasks:
                for i in range(n_collection_in_same_task):
                    self.eval_env.set_task(task)
                    self.sampler.set_env(self.eval_env)            
                    self.agent.reset(num_tasks = 1) 
                    paths, n_samples = self.sampler.obtain_samples(max_trajs=max_n_episodes, accum_context=True, update_belief=True)
                    n_steps = 0
                    for j, path in enumerate(paths):
                        if path['terminals'][-1]:
                            epirew = path['rewards'].sum()
                            epirews[j] += epirew
                            if 'env_infos' in path and 'success' in path['env_infos'][-1]:
                                successes[j] += path['env_infos'][-1]['success']
                            length = len(path['rewards'])
                            n_steps += length
                            #lengths[j] += length
                            lengths[j] += n_steps

            epirews /= 1.0*n_collection_in_same_task * self.num_eval_tasks
            successes /= 1.0*n_collection_in_same_task * self.num_eval_tasks
            lengths /= 1.0* n_collection_in_same_task * self.num_eval_tasks                            
        else:
            # 
            epirews = np.empty(max_n_episodes) #
            successes = np.empty(max_n_episodes) #
            lengths = np.empty(max_n_episodes) 
            epirews[:] = np.nan
            successes[:] = np.nan
            lengths[:] = np.nan
        for i, out in enumerate(outputs):
            if out:
                logger.record_tabular('Validation Episode {} Reward'.format(i), epirews[i]) 
                logger.record_tabular('Validation Episode {} Success'.format(i), successes[i]) 
                logger.record_tabular('#Time Steps at the End of Episode {}'.format(i), lengths[i])

        self.sampler.set_env(self.env)
    def collect_data_from_tasks(self):
        update_belief = False if self.eval_mode and not self.algo_params['fix_context_at_eval'] else True
        if self.num_processes == 1:
            epirew_sum = 0
            success_sum = 0
            num_episodes = 0
            buf = "types of sampled tasks "
            for i in range(self.num_tasks_sample):
                idx = np.random.randint(len(self.tasks))
                self.task_idx = idx
                if self.metaworld:
                    buf += str(self.tasks[idx]['task']) + " "
                self.env.set_task(self.tasks[idx])
                r, e, s = self.collect_data(self.num_steps, update_belief=update_belief)
                epirew_sum += r
                success_sum += s
                num_episodes += e
            ave = (1.0*epirew_sum)/num_episodes if num_episodes>0 else 0        
            ave2 = (1.0*success_sum)/num_episodes if num_episodes>0 else 0        
            logger.record_tabular('Average Episode Reward', ave) 
            logger.record_tabular('Success Rate', ave2) 
            logger.record_tabular('#Episodes', num_episodes)
            if self.metaworld:
                print(buf)
        else:
            if self.eval_mode:
                ids = [0 for i in range(self.num_processes)]#np.random.randint(0, len(self.tasks), self.num_tasks_sample)
                taskset = {'tasks':[self.tasks[0] for i in ids], 'ids': ids}
                self.collect_data_parallelly(taskset, self.num_steps//self.num_processes, update_belief=False)
                print("m: ", self.agent.z_means)
                print("s: ", self.agent.z_stds)
            else:
                ids = np.random.randint(0, len(self.tasks), self.num_tasks_sample)
                taskset = {'tasks':[self.tasks[i] for i in ids], 'ids': ids}
                self.collect_data_parallelly(taskset, self.num_steps, update_belief=True)
        assert self._n_env_steps_total > 0
    def preparation_for_learning(self):
        if self.algo_params["learning_type_at_eval"]  == 4 or \
           self.algo_params["learning_type_at_eval"]  == 5: 
            if self.num_processes > 1:
                task_belief = torch.cat([self.agent.z_means[0], self.agent.z_stds[0]], dim = 0)
            else:
                task_belief = torch.cat([self.agent.z_means, self.agent.z_stds], dim = 1)
                task_belief = task_belief.view(-1)
            for net in [self.agent.policy, self.qf1, self.qf2, self.vf, self.target_vf]:
                if self.agent.mlp > 0:
                    net.set_belief(task_belief)
                else:
                    net.set_film_vars(task_belief)
                
        if not self.algo_params["learning_type_at_eval"]  == 3:
            self.set_optimizers()
    def train(self):
        '''
        meta-training loop
        '''
        self.pretrain()
        params = self.get_epoch_snapshot(-1)
        logger.save_itr_params(-1, params)
        gt.reset()
        gt.set_def_unique(False)
        self._current_path_builder = PathBuilder()

        for it_ in gt.timed_for(
                range(self.num_iterations),
                save_itrs=True,
        ):
            self._start_epoch(it_)
            if it_ == 0:
                print('collecting initial pool of data for train and eval')
                self.init_collect_data_from_tasks()
                self.preparation_for_learning()
                if not self.eval_mode:
                    print('initial cvae training')
                    indices = self.task_ids
                    for train_step in range(self.num_init_embedding_train):
                        self._train_embedding(indices, itr=train_step, init_train=True)
                print('training loop')

            # Sample data from train tasks.
            self.collect_data_from_tasks()
            if self.eval_mode and it_ + 1 > self.burn_in_iteration_size:
                self.burn_in = False
                logger.log("End of burn in phase")
            if not self.algo_params['learning_type_at_eval'] == 3: 
                if self.num_train_steps_per_itr >= self.num_embedding_train_steps_per_itr:
                    rate = int(self.num_train_steps_per_itr) // int(self.num_embedding_train_steps_per_itr)
                    train_steps = self.num_train_steps_per_itr
                    for train_step in range(train_steps):
                        if self.meta_batch < 1:
                            indices = self.task_ids
                        else:
                            indices = np.random.choice(range(len(self.tasks)), self.meta_batch)
                        gt.subdivide('sub')
                        if not self.eval_mode and train_step%rate == 0:
                            if self.algo_params['embedding_batch_with_all_tasks']:
                                e_indices = self.task_ids
                            else:
                                e_indices = indices
                            self._train_embedding(e_indices, train_step//rate)
                        gt.stamp('train_embedding')
                        if not self.eval_mode and self.two_step_sac:
                            self._train_two_step_sac(indices, train_step)
                        else:
                            self._train_sac(indices, train_step)
                        gt.stamp('train_sac')
                        gt.end_subdivision()
                        self._n_train_steps_total += 1

                else:
                    raise NotImplementedError
                """
                if self.num_train_steps_per_itr >= self.num_embedding_train_steps_per_itr:
                    rate = int(self.num_train_steps_per_itr) // int(self.num_embedding_train_steps_per_itr)
                    train_steps = self.num_train_steps_per_itr
                    for train_step in range(train_steps):
                        if self.meta_batch < 1:
                            indices = self.task_ids
                        else:
                            indices = np.random.choice(range(len(self.tasks)), self.meta_batch)
                        self._training_sac_with_embedding(indices, itr=train_step, rate=rate)
                        self._n_train_steps_total += 1
                else:
                    raise NotImplementedError

                """
            gt.stamp('train')
            #print(gt.report())
            #return 0


            # output etc.                
            if not self.eval_mode:
                self.validation(it_)

            self._try_to_eval(it_)
            gt.stamp('eval')

            self._end_epoch()
