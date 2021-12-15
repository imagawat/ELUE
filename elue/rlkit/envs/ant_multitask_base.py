import numpy as np

from rlkit.envs.ant import AntEnv
# from gym.envs.mujoco.ant import AntEnv

class MultitaskAntEnv(AntEnv):
    def __init__(self, task={}, n_tasks=2, **kwargs):
        self._task = task
        self.tasks = self.sample_tasks(n_tasks)
        self._goal = self.tasks[0]['goal']
        super(MultitaskAntEnv, self).__init__(**kwargs)
        self.time_steps = 0        
        self.max_episode_steps = 150        

    """
    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = -1.0 * abs(forward_vel - self._goal_vel)
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        infos = dict(reward_forward=forward_reward,
                     reward_ctrl=-ctrl_cost, task=self._task)
        return (observation, reward, done, infos)
    """


    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = self._task['goal'] # assume parameterization of task by single vector
        self.time_steps = 0
        self.reset()

    def get_all_tasks(self):
        return self.tasks
        
    def set_task(self, task):
        # get taskとかの返り値を
        self._task = task #self.tasks[]
        self._goal = self._task['goal'] # assume parameterization of task by single vector
        self.time_steps = 0
        self.reset()

    def set_train_tasks(self, train_tasks):
        self.train_mode = True
        self.tasks = train_tasks
        return self
    def set_test_tasks(self, test_tasks):
        self.train_mode = False
        self.tasks = test_tasks
        return self
