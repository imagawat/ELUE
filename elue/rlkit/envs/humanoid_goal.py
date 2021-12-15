import numpy as np

from gym.envs.mujoco import HumanoidEnv as HumanoidEnv
from . import register_env

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))


@register_env('sparse-humanoid-goal')
class SparseHumanoidGoalEnv(HumanoidEnv):

    def __init__(self, task={}, n_tasks=2, randomize_tasks=True):
        self.tasks = self.sample_tasks(n_tasks)
        self.reset_task(0)
        super(SparseHumanoidGoalEnv, self).__init__()

    def step(self, action):
        pos_before = np.copy(mass_center(self.model, self.sim)[:2])
        self.do_simulation(action, self.frame_skip)
        pos_after = mass_center(self.model, self.sim)[:2]

        alive_bonus = 5.0
        data = self.sim.data
        max_reward = 1000
        radius =  3.0
        goal_reward = max_reward * max(0, 1 - np.linalg.norm(pos_after - self._goal, ord=2) / radius) # if dist < radius, goal_reward > 0 
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = goal_reward - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.sim.data.qpos
        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))

        return self._get_obs(), reward, done, dict(reward_goal=goal_reward,
                                                   reward_quadctrl=-quad_ctrl_cost,
                                                   reward_alive=alive_bonus,
                                                   reward_impact=-quad_impact_cost)

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat,
                               data.cinert.flat,
                               data.cvel.flat,
                               data.qfrc_actuator.flat,
                               data.cfrc_ext.flat])

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = self._task['goal'] # assume parameterization of task by single vector

    def sample_tasks(self, num_tasks):
        # velocities = np.random.uniform(0., 1.0 * np.pi, size=(num_tasks,))
        r = 10
        a = np.random.random(num_tasks)  * np.pi
        goals = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)
        tasks = [{'goal': g} for g in goals]
        return tasks
