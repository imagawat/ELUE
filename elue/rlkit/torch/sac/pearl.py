from collections import OrderedDict
import numpy as np

import torch
import torch.optim as optim
from torch import nn as nn
import gtimer as gt

from rlkit.data_management.path_builder import PathBuilder
from rlkit.core import logger
import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.core.rl_algorithm import MetaRLAlgorithm
import debug_config
from rlkit.core import additional_csv_writer

class PEARLSoftActorCritic(MetaRLAlgorithm):
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
            recurrent=False,
            use_information_bottleneck=True,
            use_next_obs_in_context=False,
            sparse_rewards=False,
            soft_target_tau=1e-2,
            plotter=None,
            render_eval_paths=False,
            training_mode=True,
            **kwargs
    ):
        super().__init__(
            ml=ml,
            env=env,
            tasks=tasks,            
            agent=nets[0],
            sampler=sampler,
            eval_env=eval_env,
            eval_tasks=tasks, #temp TODO: remove...?
            num_processes=num_processes,
            **kwargs
        )
        self.algo_params = kwargs
        self.num_eval_tasks = num_eval_tasks
        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.recurrent = recurrent
        self.latent_dim = latent_dim
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()
        self.vib_criterion = nn.MSELoss()
        self.l2_reg_criterion = nn.MSELoss()
        self.kl_lambda = kl_lambda

        self.use_information_bottleneck = use_information_bottleneck
        self.sparse_rewards = sparse_rewards
        self.use_next_obs_in_context = use_next_obs_in_context

        self.qf1, self.qf2, self.vf = nets[1:]
        self.target_vf = self.vf.copy()

        self.first_update =True # for outputting log
        self.eval_mode = not training_mode
        self.num_episodes = 0
        self.num_write = 10
        self.prior = True # used at eval_mode for checking whether self.agent.z_means and vars are updated
        if training_mode:        

            self.policy_optimizer = optimizer_class(
                self.agent.policy.parameters(),
                lr=policy_lr,
            )

            self.qf1_optimizer = optimizer_class(
                self.qf1.parameters(),
                lr=qf_lr,
            )

            self.qf2_optimizer = optimizer_class(
                self.qf2.parameters(),
                lr=qf_lr,

            )

            self.vf_optimizer = optimizer_class(
                self.vf.parameters(),
                lr=vf_lr,
            )
            self.context_optimizer = optimizer_class(
                self.agent.context_encoder.parameters(),
                lr=context_lr,
            )

    ###### Torch stuff #####
    @property
    def networks(self):
        return self.agent.networks + [self.agent] + [self.qf1, self.qf2, self.vf, self.target_vf]

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

    def sample_sac(self, indices):
        ''' sample batch of training data from a list of tasks for training the actor-critic '''
        # this batch consists of transitions sampled randomly from replay buffer
        # rewards are always dense
        batches = [ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=self.batch_size)) for idx in indices]
        unpacked = [self.unpack_batch(batch) for batch in batches]
        # group like elements together
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]
        return unpacked

    def sample_context(self, indices):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]
        batches = [ptu.np_to_pytorch_batch(self.enc_replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size, sequence=self.recurrent)) for idx in indices]
        context = [self.unpack_batch(batch, sparse_reward=self.sparse_rewards) for batch in batches]
        # group like elements together
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [torch.cat(x, dim=0) for x in context]
        # full context consists of [obs, act, rewards, next_obs, terms]
        # if dynamics don't change across tasks, don't include next_obs
        # don't include terminals in context
        if self.use_next_obs_in_context:
            context = torch.cat(context[:-1], dim=2)
        else:
            context = torch.cat(context[:-2], dim=2)
        return context

    ##### Training #####
    def _do_training(self, indices):
        mb_size = self.embedding_mini_batch_size
        num_updates = self.embedding_batch_size // mb_size

        # sample context batch
        context_batch = self.sample_context(indices)

        # zero out context and hidden encoder state
        self.agent.clear_z(num_tasks=len(indices))

        # do this in a loop so we can truncate backprop in the recurrent encoder
        for i in range(num_updates):
            context = context_batch[:, i * mb_size: i * mb_size + mb_size, :]
            self._take_step(indices, context)

            # stop backprop
            self.agent.detach_z()

    def _min_q(self, obs, actions, task_z):
        q1 = self.qf1(obs, actions, task_z.detach())
        q2 = self.qf2(obs, actions, task_z.detach())
        min_q = torch.min(q1, q2)
        return min_q

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)

    def _take_step(self, indices, context):

        num_tasks = len(indices)

        # data is (task, batch, feat)
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices)

        # run inference in networks ######################################
        policy_outputs, task_z = self.agent(obs, context)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # flattens out the task dimension

        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)

        # Q and V networks
        # encoder will only get gradients from Q nets
        q1_pred = self.qf1(obs, actions, task_z)
        q2_pred = self.qf2(obs, actions, task_z)
        v_pred = self.vf(obs, task_z.detach())
        # get targets for use in V and Q updates
        with torch.no_grad():
            target_v_values = self.target_vf(next_obs, task_z)

        # KL constraint on z if probabilistic
        self.context_optimizer.zero_grad()
        if self.use_information_bottleneck:
            kl_div = self.agent.compute_kl_div()
            kl_loss = self.kl_lambda * kl_div
            kl_loss.backward(retain_graph=True)

        # qf and encoder update (note encoder does not get grads from policy or vf)
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        rewards_flat = rewards.view(self.batch_size * num_tasks, -1)
        # scale rewards for Bellman update
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(self.batch_size * num_tasks, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        qf_loss.backward()
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()
        self.context_optimizer.step()

        # compute min Q on the new actions
        min_q_new_actions = self._min_q(obs, new_actions, task_z)

        ent_coeff = self.algo_params['entropy_coeff']
        # vf update
        v_target = min_q_new_actions - log_pi * ent_coeff
        vf_loss = self.vf_criterion(v_pred, v_target.detach())
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()
        self._update_target_network()

        # policy update
        # n.b. policy update includes dQ/da
        log_policy_target = min_q_new_actions

        policy_loss = (
                log_pi * ent_coeff - log_policy_target
        ).mean()

        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value**2).sum(dim=1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        if self.first_update:
            logger.record_tabular('VF Loss', ptu.get_numpy(vf_loss))
            logger.record_tabular('QF Loss', ptu.get_numpy(qf_loss))
            logger.record_tabular('Policy Loss', ptu.get_numpy(policy_loss))
            qf_diff = torch.mean((q1_pred - q2_pred) ** 2)
            logger.record_tabular('QF Diff', ptu.get_numpy(qf_diff))
            self.first_update = False

        """
        # save some statistics for eval
        if self.eval_statistics is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            self.eval_statistics = OrderedDict()
            if self.use_information_bottleneck:
                z_mean = np.mean(np.abs(ptu.get_numpy(self.agent.z_means[0])))
                z_sig = np.mean(ptu.get_numpy(self.agent.z_vars[0]))
                self.eval_statistics['Z mean train'] = z_mean
                self.eval_statistics['Z variance train'] = z_sig
                self.eval_statistics['KL Divergence'] = ptu.get_numpy(kl_div)
                self.eval_statistics['KL Loss'] = ptu.get_numpy(kl_loss)

            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'V Predictions',
                ptu.get_numpy(v_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))
        """
    def get_epoch_snapshot(self, epoch):
        # NOTE: overriding parent method which also optionally saves the env
        snapshot = OrderedDict(
            qf1=self.qf1.state_dict(),
            qf2=self.qf2.state_dict(),
            policy=self.agent.policy.state_dict(),
            vf=self.vf.state_dict(),
            target_vf=self.target_vf.state_dict(),
            context_encoder=self.agent.context_encoder.state_dict(),
        )
        return snapshot
    def collect_data_for_eval(self, num_samples): #TODO: FIX
        """
        for evaluation by executing eval task in a training loop 
        evalのためのbufferを用意しないといけない．
        """

    def collect_data(self, num_samples, resample_z_rate, update_posterior_rate, add_to_enc_buffer=True):
        '''
        get trajectories from current env in batch mode with given policy
        collect complete trajectories until the number of collected transitions >= num_samples

        :param agent: policy to rollout
        :param num_samples: total number of transitions to sample
        :param resample_z_rate: how often to resample latent context z (in units of trajectories)
        :param update_posterior_rate: how often to update q(z | c) from which z is sampled (in units of trajectories)
        :param add_to_enc_buffer: whether to add collected data to encoder replay buffer
        '''
        if not self.eval_mode:
            self.agent.clear_z()
            accum_context = False
        else:
            accum_context = True

        epirew_sum = 0
        success_sum = 0
        num_episodes = 0
        num_transitions = 0
        while num_transitions < num_samples:
            paths, n_samples = self.sampler.obtain_samples(deterministic=self.algo_params['deterministic_sampling'],
                                                           max_samples=num_samples - num_transitions,
                                                           max_trajs=update_posterior_rate,
                                                           accum_context=accum_context,
                                                           resample=resample_z_rate)
            num_transitions += n_samples
            self.replay_buffer.add_paths(self.task_idx, paths)
            if add_to_enc_buffer:
                self.enc_replay_buffer.add_paths(self.task_idx, paths)
            if update_posterior_rate != np.inf:
                if self.eval_mode:
                    if self.prior:
                        self.prior = False
                        self.agent.infer_posterior(self.agent.context, with_context_reset=True)
                    else:
                        self.agent.infer_posterior_by_preserved_z_and_context()
                else:
                    context = self.sample_context(self.task_idx)
                    self.agent.infer_posterior(context)
            for path in paths:
                if path['terminals'][-1]:
                    epirew = path['rewards'].sum()
                    epirew_sum += epirew
                    if 'env_infos' in path and 'success' in path['env_infos'][-1]:
                        success_sum += path['env_infos'][-1]['success']
                    num_episodes += 1
                    if self.eval_mode and self.num_write > self.num_episodes:
                        self.num_episodes += 1
                        additional_csv_writer.csv_write({'n_env_steps':self._n_env_steps_total, 'epinum': self.num_episodes, 'epirew': epirew}, (self.num_episodes == 1))
                        if 'env_infos' in path and 'fingerXY' in path['env_infos'][-1]:      
                            additional_csv_writer.trajectory_write(path['env_infos'], self.num_episodes)

        self._n_env_steps_total += num_transitions
        gt.stamp('sample')
        return epirew_sum, num_episodes, success_sum
    #logger.record_tabular('Episode Average Reward', (1.0*epirew_sum)/num_episodes) 
    def validation(self, n_iter):
        #特に変わるところがなければ統合を試みても良いが優先度は低い
        ##とりあえず別のコードにする
        outputs = self.algo_params['validation_outputs'] # 
        max_n_episodes = len(outputs)
        n_collection_in_same_task = 1        
        if (n_iter + 1) % self.algo_params['eval_interval'] == 0:
            epirews = np.zeros(max_n_episodes) #
            successes = np.zeros(max_n_episodes) #
            lengths = np.zeros(max_n_episodes) # episode lengthでなく，累積的なものの方が良い？ ML10とかだと．．，でも無理
            tasks = self.eval_env.sample_tasks(self.num_eval_tasks)
            for task in tasks:
                for i in range(n_collection_in_same_task):
                    self.eval_env.set_task(task)
                    self.sampler.set_env(self.eval_env)            
                    self.agent.clear_z(num_tasks = 1)
                    paths =[]
                    n_samples = 0
                    #paths, n_samples = self.sampler.obtain_samples(max_trajs=max_n_episodes, accum_context=True)
                    #まとめてとるとたぶんZが更新されないのでこうする
                    for i in range(max_n_episodes):
                        path, n_sample = self.sampler.obtain_samples(max_trajs=1, accum_context=True)
                        self.agent.infer_posterior_by_preserved_z_and_context()
                        paths += path
                        n_samples += n_sample
                        
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
            # 辻褄合わせのためのNan出力
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

        # at each iteration, we first collect data from tasks, perform meta-updates, then try to evaluate
        for it_ in gt.timed_for(
                range(self.num_iterations),
                save_itrs=True,
        ):
            self._start_epoch(it_)
            #self.training_mode(True)
            if it_ == 0:
                print('collecting initial pool of data for train and eval')
                #self.init_collect_data_from_tasks()
                if self.eval_mode:
                    self.agent.clear_z()
                    update_post_train = self.update_post_train
                else:
                    update_post_train = np.inf

                for i, task in enumerate(self.tasks):
                    self.task_idx = i #self.train_task_idx(i)
                    #self.env.reset_task(idx)
                    self.env.set_task(task)
                    self.collect_data(self.num_initial_steps, 1, update_post_train)

            epirew_sum = 0
            success_sum = 0
            num_episodes = 0
            buf = "types of sampled tasks "
            # Sample data from train tasks.
            for i in range(self.num_tasks_sample):
                idx = np.random.randint(len(self.tasks))
                self.task_idx = idx
                self.env.set_task(self.tasks[idx])
                if self.metaworld:
                    buf += str(self.tasks[idx]['task']) + " "
                self.enc_replay_buffer.task_buffers[idx].clear()

                # collect some trajectories with z ~ prior
                if self.num_steps_prior > 0:
                    r, e, s = self.collect_data(self.num_steps_prior, 1, np.inf)
                    epirew_sum += r
                    success_sum += s
                    num_episodes += e
                # collect some trajectories with z ~ posterior
                if self.num_steps_posterior > 0:
                    r, e, s = self.collect_data(self.num_steps_posterior, 1, self.update_post_train)
                    epirew_sum += r
                    success_sum += s
                    num_episodes += e
                # even if encoder is trained only on samples from the prior, the policy needs to learn to handle z ~ posterior
                if self.num_extra_rl_steps_posterior > 0:
                    r, e, s= self.collect_data(self.num_extra_rl_steps_posterior, 1, self.update_post_train, add_to_enc_buffer=False)
                    epirew_sum += r
                    success_sum += s
                    num_episodes += e
            ave = (1.0*epirew_sum) / num_episodes if num_episodes > 0 else 0
            ave2 = (1.0*success_sum) / num_episodes if num_episodes > 0 else 0
            logger.record_tabular('Average Episode Reward', ave)
            logger.record_tabular('Success Rate', ave2)
            logger.record_tabular('#Episodes', num_episodes)
            if self.metaworld:
                print(buf)
            if not self.eval_mode:
            # Sample train tasks and compute gradient updates on parameters.
                self.first_update = True                
                for train_step in range(self.num_train_steps_per_itr):
                    indices = np.random.choice(range(len(self.tasks)), self.meta_batch)
                    self._do_training(indices)
                    self._n_train_steps_total += 1
            gt.stamp('train')
            
            #いらないのでは？
            #self.training_mode(False)

            
            if self.training_mode:
                self.validation(it_)

            self._try_to_eval(it_)
            gt.stamp('eval')

            self._end_epoch()



#### parallelization (under construction)####
    def collect_data_parallelly(self, taskset, num_samples, update_belief=True, add_to_enc_buffer=True, initial=False):
        assert not self.eval_mode
        # eval時にはとりあえず使わない方針
        epirew_sum = 0
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
            if self.sampler.vecenv == None :            
                vecenv = ParallelEnvExecutor(env=self.env, n_envs=self.num_processes)
                self.sampler.set_vecenv(vecenv)
            else:
                #端数があるときにvecenvの調整（n_envsとかいじる）とかしなくてもOK？
                pass
            
            self.sampler.vecenv.set_tasks(taskset['tasks'][loopid*self.num_processes: loopid*self.num_processes + num_tasks])
            self.agent.reset(num_tasks)
            paths, n_samples = self.sampler.obtain_samples(max_samples=num_samples, accum_context=True, update_belief=update_belief)      
            for i, taskid in enumerate(taskset['ids'][loopid*self.num_processes: loopid*self.num_processes + num_tasks]):
                self.replay_buffer.add_paths(taskid, paths[i])
                if not initial:
                    for path in paths[i]:
                        if path['terminals'][-1]:
                            epirew_sum += path['rewards'].sum()
                            num_episodes += 1
            self._n_env_steps_total += n_samples

        #eval_epirew_sum = 0
        #eval_num_episodes = 0
        gt.stamp('sample')
        if not initial:
            ave = (1.0*epirew_sum)/num_episodes if num_episodes>0 else 0        
            logger.record_tabular('Average Episode Reward', ave) 
            logger.record_tabular('#Episodes', num_episodes) 

    def init_collect_data_from_tasks(self):
        """
        init collection of data from envs
        """
        if self.num_processes == 1:
            for i, task in enumerate(self.tasks):
                self.task_idx = i
                self.env.set_task(task)  
                self.collect_data(self.num_initial_steps, update_belief=False)
        else:
            taskset = {'tasks': self.tasks, 'ids': [i for i in range(len(self.tasks))]}
            self.collect_data_parallelly(taskset, self.num_initial_steps, update_belief=False, initial=True)

    def collect_data_from_tasks(self):
        if self.num_processes == 1:
            epirew_sum = 0
            num_episodes = 0
            for i in range(self.num_tasks_sample):
                idx = np.random.randint(len(self.tasks))
                self.task_idx = idx
                self.env.set_task(self.tasks[idx])
                r, e, r2, e2 = self.collect_data(self.num_steps)
                epirew_sum += r
                num_episodes += e
            ave = (1.0*epirew_sum)/num_episodes if num_episodes>0 else 0        
            logger.record_tabular('Average Episode Reward', ave) 
            logger.record_tabular('#Episodes', num_episodes) 
            if self.eval_mode:
                assert(self.num_tasks_sample == 1)
                logger.record_tabular('Average Episode Reward (Eval)', r2/e2) 
                logger.record_tabular('#Episodes (Eval)', e2)
        else:
            ids = np.random.randint(0, len(self.tasks), self.num_tasks_sample)
            taskset = {'tasks':[self.tasks[i] for i in ids], 'ids': ids}
            self.collect_data_parallelly(taskset, self.num_steps)
