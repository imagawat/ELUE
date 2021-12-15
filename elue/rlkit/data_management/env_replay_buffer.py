import numpy as np

from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer
from gym.spaces import Box, Discrete, Tuple


class MultiTaskReplayBuffer(object):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            task_ids, # 1D ID of tasks
            task_pairs,#[(class, goalpos)]
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        :param tasks: for multi-task setting
        """
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space
        self.task_buffers = dict(
            [(idx, SimpleReplayBuffer(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            )) for idx in task_ids]
        )


    def set_env(self, env):
        self.env = env
        
    def add_sample(self, task_id, observation, action, reward, terminal,
            next_observation, **kwargs):

        if isinstance(self._action_space, Discrete):
            action = np.eye(self._action_space.n)[action]
        self.task_buffers[task_id].add_sample(
                observation, action, reward, terminal,
                next_observation, **kwargs)

    def terminate_episode(self, task_id):
        self.task_buffers[task_id].terminate_episode()

    def random_batch(self, task_id, batch_size, sequence=False):
        if sequence:
            batch = self.task_buffers[task_id].random_sequence(batch_size)
        else:
            batch = self.task_buffers[task_id].random_batch(batch_size)
        return batch

    def random_batches(self, task_id, batch_sizes, sequence=False):
        if sequence:
            batch = self.task_buffers[task_id].random_sequences(batch_sizes)
        else:
            batch = self.task_buffers[task_id].random_batches(batch_sizes)
        return batch
    def random_two_step_batches(self, task_id, batch_sizes):        
        return self.task_buffers[task_id].random_two_step_batches(batch_sizes)

    def num_steps_can_sample(self, task_id):
        return self.task_buffers[task_id].num_steps_can_sample()

    def add_path(self, task_id, path):
        self.task_buffers[task_id].add_path(path)

    def add_paths(self, task_id, paths):
        for path in paths:
            self.task_buffers[task_id].add_path(path)

    def clear_buffer(self, task_id):
        self.task_buffers[task_id].clear()


def get_dim(space):
    if isinstance(space, Box):
        return space.low.size
    elif isinstance(space, Discrete):
        return space.n
    elif isinstance(space, Tuple):
        return sum(get_dim(subspace) for subspace in space.spaces)
    elif hasattr(space, 'flat_dim'):
        return space.flat_dim
    else:
        # import OldBox here so it is not necessary to have rand_param_envs 
        # installed if not running the rand_param envs
        from rand_param_envs.gym.spaces.box import Box as OldBox
        if isinstance(space, OldBox):
            return space.low.size
        else:
            raise TypeError("Unknown space: {}".format(space))
