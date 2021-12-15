import numpy as np
import pickle as pickle
from multiprocessing import Process, Pipe
import copy
import itertools

from rlkit.samplers.util import rollout
from rlkit.torch.sac.policies import MakeDeterministic

class ParallelSampler(object):
    """
    pathを集める．．．他の処理は他で
    PEARLへの対応はあとで？

    """
    def __init__(self, policy):
        self.vecenv = None
        self.policy = policy

    def set_vecenv(self, vecenv):
        self.vecenv = vecenv
    def start_worker(self):
        #いらない
        pass

    def shutdown_worker(self):
        #いらない
        pass

    def obtain_samples(self, deterministic=False, max_samples=np.inf, max_trajs=np.inf, accum_context=True, update_belief=True):
        """
        とりあえず．ELUEしか対応しないので，
        trajは使う？
        期待される挙動は？入出力
        今のvecenvに対して，それぞれmax_samples分のデータを得て，pathとしてつないで，返す        
        """
        assert (self.vecenv != None)
        assert max_samples < np.inf or max_trajs < np.inf, "either max_samples or max_trajs must be finite"
        policy = MakeDeterministic(self.policy) if deterministic else self.policy
        paths = []
        n_steps_total = 0
        #n_trajs = 0
        #max trajは使う？　
        paths, n_steps_total = self.rollout(policy, max_samples = max_samples, accum_context=accum_context, update_belief=update_belief) #max_path_length = inf
            # save the latent context that generated this trajectory
            #path['context'] = policy.z.detach().cpu().numpy()

            # don't we also want the option to resample z ever transition?
            #if n_trajs % resample == 0:
            #    policy.sample_z()
        return paths, n_steps_total
    def rollout(self, policy,  max_samples, accum_context=True, update_belief=False):
        observations = [[] for i in range(self.vecenv.num_active_envs)]
        actions = [[] for i in range(self.vecenv.num_active_envs)]
        rewards = [[] for i in range(self.vecenv.num_active_envs)]
        terminals = [[] for i in range(self.vecenv.num_active_envs)]
        agent_infos = [[] for i in range(self.vecenv.num_active_envs)]
        env_infos = [[] for i in range(self.vecenv.num_active_envs)]
        obss = self.vecenv.reset()
        next_obss = None
        paths = [[] for i in range(self.vecenv.num_active_envs)]
        n_steps_total = 0
        policy.set_noise()
        for timestep in range(max_samples):
            acts, agent_infs = self.policy.get_action(obss)
            next_obss, rews, dones, env_infs = self.vecenv.step(acts)
            
            # update the agent's current context
            if accum_context:
                policy.update_context([obss, acts, rews, next_obss, dones, env_infs])
                if update_belief:
                    policy.infer_posterior_batch(incremental=True)

            ## reset env if it is done 
            #initobss = self.vecenv.reset(dones)
            # save results
            for idx, obs, act, rew, env_inf, agent_inf, done in zip(itertools.count(), obss, acts, rews, env_infs, agent_infs, dones):
                observations[idx].append(obs)
                actions[idx].append(act)
                rewards[idx].append(rew)
                terminals[idx].append(done)
                agent_infos[idx].append(agent_inf)
                env_infos[idx].append(env_inf)

                # if running path is done, add it to paths and empty the running path
                if done:
                    o = np.asarray(observations[idx])
                    no = np.vstack((o[1:], np.expand_dims(next_obss[idx], 0)))
                    paths[idx].append(dict(
                        observations=o,
                        actions=np.asarray(actions[idx]),
                        rewards=np.asarray(rewards[idx]),
                        next_observations=no,
                        terminals=np.asarray(terminals[idx]),
                        env_infos=env_infos[idx],
                        agent_infos=agent_infos[idx]
                    ))
                    n_steps_total += len(observations[idx])
                    observations[idx] = []
                    actions[idx] = []
                    rewards[idx] = []
                    terminals[idx] = []
                    env_infos[idx] = []
                    agent_infos[idx] = []
                    next_obss[idx] = env_inf['init_obs']
                    policy.reset_noise(idx)
                    # set noise
                    #next_obss[idx] = self.vecenv.reset([idx])[0]#initobss[idx]
                    #new_samples += len(running_paths[idx]["rewards"])
                    #running_paths[idx] = _get_empty_running_paths_dict()
            #　こんなかたちで，resetをする
            # oのうちtermの部分だけ，initobsに入れ替える
                    # bufferも消してOK                

            obss = next_obss
        #ここらへんの処理もよく確認する
        """
        actions = np.array(actions)
        if len(actions.shape) == 1:
            actions = np.expand_dims(actions, 1)
        observations = np.array(observations)
        if len(observations.shape) == 1:
            observations = np.expand_dims(observations, 1)
            next_o = np.array([next_o])
        next_observations = np.vstack(
            (
                observations[1:, :],
                np.expand_dims(next_o, 0)
            )
        )dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        )
        """
        return paths, n_steps_total
class BeliefUpdateParallelSampler(ParallelSampler):
    pass    



#これは基本的にいじらない
#インターフェイスをいじる
class ParallelEnvExecutor(object):
    """
    Wraps multiple environments of the same kind and provides functionality to reset / step the environments
    in a vectorized manner. Thereby the environments are distributed among meta_batch_size processes and
    executed in parallel.

    Args:
        env (maml_zoo.envs.base.MetaEnv): meta environment object
        meta_batch_size (int): number of meta tasks
        envs_per_task (int): number of environments per meta task
        max_path_length (int): maximum length of sampled environment paths - if the max_path_length is reached,
                             the respective environment is reset
    """
    #envはdeep copyされる？さすがにリモートなのでそうなるはず
    def __init__(self, env, n_envs, max_path_length=np.inf):
        #init だと，全部のタスクでサンプルするので，こことは違う．．．．initはnot vec encでやる
        self.n_envs = n_envs
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(n_envs)])
        seeds = np.random.choice(range(10**6), size=n_envs, replace=False)
        self.envs_per_task = 1 # each worker has only one env
        self.num_active_envs = 0

        self.ps = [
            Process(target=worker, args=(work_remote, remote, pickle.dumps(env),  max_path_length, seed))
            for (work_remote, remote, seed) in zip(self.work_remotes, self.remotes, seeds)]  # Why pass work remotes?

        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()
        
    def step(self, actions):
        """
        Executes actions on each env

        Args:
            actions (list): lists of actions, of length num_active_envs    

        Returns
            (tuple): a length 4 tuple of lists, containing obs (np.array), rewards (float), dones (bool), env_infos (dict)
                      each list is of length meta_batch_size x envs_per_task (assumes that every task has same number of envs)
        """
        #assert len(actions) == self.num_envs
        assert len(actions) == self.num_active_envs

        # split list of actions in list of list of actions per meta tasks
        chunks = lambda l, n: [l[x: x + n] for x in range(0, len(l), n)]
        actions_per_meta_task = chunks(actions, self.envs_per_task)

        # step remote environments
        for remote, action_list in zip(self.remotes, actions_per_meta_task):
            remote.send(('step', action_list))

        results = [self.remotes[i].recv() for i in range(self.num_active_envs)]

        obs, rewards, dones, env_infos = map(lambda x: sum(x, []), zip(*results))

        return obs, rewards, dones, env_infos

    def reset(self, targets=None):
        """
        Resets the environments of each worker

        Returns:
            (list): list of (np.ndarray) with the new initial observations.
        """
        assert self.num_active_envs > 0 #set tasks before reset
        for i in range(self.num_active_envs):
            self.remotes[i].send(('reset', None))
        return sum([self.remotes[i].recv() for i in range(self.num_active_envs)], [])


        
    def set_tasks(self, tasks=None):
        """
        Sets a list of tasks to each worker

        Args:
            tasks (list): list of the tasks for each worker
        """
        self.num_active_envs = len(tasks)
        for remote, task in zip(self.remotes, tasks):
            remote.send(('set_task', task))
        for i in range(len(tasks)):
            self.remotes[i].recv()
        """
        for remote in self.remotes:
            remote.recv()
        """
    @property
    def num_envs(self):
        """
        Number of environments

        Returns:
            (int): number of environments
        """
        return self.n_envs


def worker(remote, parent_remote, env_pickle, max_path_length, seed, n_envs=1):
    """
    Instantiation of a parallel worker for collecting samples. It loops continually checking the task that the remote
    sends to it.

    Args:
        remote (multiprocessing.Connection):
        parent_remote (multiprocessing.Connection):
        env_pickle (pkl): pickled environment
        n_envs (int): number of environments per worker
        max_path_length (int): maximum path length of the task
        seed (int): random seed for the worker
    """
    parent_remote.close()

    envs = [pickle.loads(env_pickle) for _ in range(n_envs)]
    np.random.seed(seed)

    ts = np.zeros(n_envs, dtype='int')

    while True:
        # receive command and data from the remote
        cmd, data = remote.recv()

        # do a step in each of the environment of the worker
        if cmd == 'step':
            all_results = [env.step(a) for (a, env) in zip(data, envs)]
            obs, rewards, dones, infos = map(list, zip(*all_results))
            ts += 1
            for i in range(n_envs):
                if dones[i] or (ts[i] >= max_path_length):
                    dones[i] = True
                    infos[i]['init_obs'] = envs[i].reset()
                    #obs[i] = envs[i].reset()
                    ts[i] = 0
            remote.send((obs, rewards, dones, infos))

        # reset all the environments of the worker
        elif cmd == 'reset':
            obs = [env.reset() for env in envs]
            ts[:] = 0
            remote.send(obs)
            """
        elif cmd == 'reset':
            obs = []
            for i, d in enumerate(data):
                if d:
                    obs.append(envs[i].reset()) 
                    ts[i] = 0
            remote.send(obs)
            """            
        # set the specified task for each of the environments of the worker
        elif cmd == 'set_task':
            for env in envs:
                env.set_task(data)
            remote.send(None)

        # close the remote and stop the worker
        elif cmd == 'close':
            remote.close()
            break

        else:
            raise NotImplementedError

###

#対応を変えないと　whileを抜けず，　はじめから
"""
def _get_empty_running_paths_dict():
    return dict(observations=[], actions=[], rewards=[], dones=[], env_infos=[], agent_infos=[])
        
class MAMLSampler(Sampler):

    Sampler for Meta-RL

    Args:
        env (maml_zoo.envs.base.MetaEnv) : environment object
        policy (maml_zoo.policies.base.Policy) : policy object
        batch_size (int) : number of trajectories per task
        meta_batch_size (int) : number of meta tasks
        max_path_length (int) : max number of steps per trajectory
        envs_per_task (int) : number of envs to run vectorized for each task (influences the memory usage)


    def __init__(
            self,
            env,
            policy,
            rollouts_per_meta_task,
            meta_batch_size,
            max_path_length,
            envs_per_task=None,
            parallel=False
            ):
        super(MAMLSampler, self).__init__(env, policy, rollouts_per_meta_task, max_path_length)
        assert hasattr(env, 'set_task')

        self.envs_per_task = rollouts_per_meta_task if envs_per_task is None else envs_per_task
        self.meta_batch_size = meta_batch_size
        self.total_samples = meta_batch_size * rollouts_per_meta_task * max_path_length
        self.parallel = parallel
        self.total_timesteps_sampled = 0

        # setup vectorized environment

        if self.parallel:
            self.vec_env = MAMLParallelEnvExecutor(env, self.meta_batch_size, self.envs_per_task, self.max_path_length)
        else:
            self.vec_env = MAMLIterativeEnvExecutor(env, self.meta_batch_size, self.envs_per_task, self.max_path_length)


    def set_tasks(self, tasks):
        assert len(tasks) == self.meta_batch_size
        self.vec_env.set_tasks(tasks)

    def obtain_samples(self, log=False, log_prefix=''):
"""
"""        Collect batch_size trajectories from each task

        Args:
            log (boolean): whether to log sampling times
            log_prefix (str) : prefix for logger

        Returns: 
            (dict) : A dict of paths of size [meta_batch_size] x (batch_size) x [5] x (max_path_length)
"""
"""
        # initial setup / preparation
        paths = OrderedDict()
        for i in range(self.meta_batch_size):
            paths[i] = []

        n_samples = 0
        running_paths = [_get_empty_running_paths_dict() for _ in range(self.vec_env.num_envs)]

        pbar = ProgBar(self.total_samples)
        policy_time, env_time = 0, 0

        policy = self.policy
        policy.reset(dones=[True] * self.meta_batch_size)

        # initial reset of envs
        obses = self.vec_env.reset()
        
        while n_samples < self.total_samples:
            
            # execute policy
            t = time.time()
            obs_per_task = np.split(np.asarray(obses), self.meta_batch_size)
            actions, agent_infos = policy.get_actions(obs_per_task)
            policy_time += time.time() - t

            # step environments
            t = time.time()
            actions = np.concatenate(actions) # stack meta batch
            next_obses, rewards, dones, env_infos = self.vec_env.step(actions)
            env_time += time.time() - t

            #  stack agent_infos and if no infos were provided (--> None) create empty dicts
            agent_infos, env_infos = self._handle_info_dicts(agent_infos, env_infos)

            new_samples = 0
            for idx, observation, action, reward, env_info, agent_info, done in zip(itertools.count(), obses, actions,
                                                                                    rewards, env_infos, agent_infos,
                                                                                    dones):
                # append new samples to running paths
                if isinstance(reward, np.ndarray):
                    reward = reward[0]
                running_paths[idx]["observations"].append(observation)
                running_paths[idx]["actions"].append(action)
                running_paths[idx]["rewards"].append(reward)
                running_paths[idx]["dones"].append(done)
                running_paths[idx]["env_infos"].append(env_info)
                running_paths[idx]["agent_infos"].append(agent_info)

                # if running path is done, add it to paths and empty the running path
                if done:
                    paths[idx // self.envs_per_task].append(dict(
                        observations=np.asarray(running_paths[idx]["observations"]),
                        actions=np.asarray(running_paths[idx]["actions"]),
                        rewards=np.asarray(running_paths[idx]["rewards"]),
                        dones=np.asarray(running_paths[idx]["dones"]),
                        env_infos=utils.stack_tensor_dict_list(running_paths[idx]["env_infos"]),
                        agent_infos=utils.stack_tensor_dict_list(running_paths[idx]["agent_infos"]),
                    ))
                    new_samples += len(running_paths[idx]["rewards"])
                    running_paths[idx] = _get_empty_running_paths_dict()

            pbar.update(new_samples)
            n_samples += new_samples
            obses = next_obses
        pbar.stop()

        self.total_timesteps_sampled += self.total_samples
        if log:
            logger.logkv(log_prefix + "PolicyExecTime", policy_time)
            logger.logkv(log_prefix + "EnvExecTime", env_time)

        return paths

    def _handle_info_dicts(self, agent_infos, env_infos):
        if not env_infos:
            env_infos = [dict() for _ in range(self.vec_env.num_envs)]
        if not agent_infos:
            agent_infos = [dict() for _ in range(self.vec_env.num_envs)]
        else:
            assert len(agent_infos) == self.meta_batch_size
            assert len(agent_infos[0]) == self.envs_per_task
            agent_infos = sum(agent_infos, [])  # stack agent_infos

        assert len(agent_infos) == self.meta_batch_size * self.envs_per_task == len(env_infos)
        return agent_infos, env_infos

"""

###
