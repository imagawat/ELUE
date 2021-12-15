import numpy as np
from rlkit.envs.ant_multitask_base import MultitaskAntEnv
from . import register_env


#
@register_env('ant-goal')
class AntGoalEnv(MultitaskAntEnv):
    def __init__(self, task={}, n_tasks=2, randomize_tasks=True, **kwargs):
        self.time_steps = 0        
        self.max_episode_steps = 150        
        super(AntGoalEnv, self).__init__(task, n_tasks, **kwargs)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        xposafter = np.array(self.get_body_com("torso"))

        goal_reward = -np.sum(np.abs(xposafter[:2] - self._goal)) # make it happy, not suicidal

        ctrl_cost = .1 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.0 #??? imediate suicide is the best strategy????
        reward = goal_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        self.time_steps += 1
        done = (self.time_steps >= self.max_episode_steps)
        #done = False
        ob = self._get_obs()
        return ob, reward, done, dict(
            goal_forward=goal_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
        )

    def sample_tasks(self, num_tasks):
        a = np.random.random(num_tasks) * 2 * np.pi
        r = 3 * np.random.random(num_tasks) ** 0.5
        goals = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)
        tasks = [{'goal': goal} for goal in goals]
        return tasks

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])
    def reset(self):
        self.time_steps = 0
        return super().reset()


@register_env('sparse-ant-goal')
class SparseAntGoalEnv(MultitaskAntEnv):
    def __init__(self, task={}, n_tasks=2, randomize_tasks=True, **kwargs):
        self.time_steps = 0        
        self.max_episode_steps = 150        
        super(SparseAntGoalEnv, self).__init__(task, n_tasks, **kwargs)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        xposafter = np.array(self.get_body_com("torso"))
        self.time_steps += 1
        max_reward = 1000
        radius =  3.0
        goal_reward = max_reward * max(0, 1 - np.linalg.norm(xposafter[:2] - self._goal, ord=2) / radius) # if dist < radius, goal_reward > 0 
        #goal_reward = -np.sum(np.abs(xposafter[:2] - self._goal)) # make it happy, not suicidal

        ctrl_cost = .1 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0 
        reward = goal_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        done = (self.time_steps >= self.max_episode_steps)
        ob = self._get_obs()
        return ob, reward, done, dict(
            goal_forward=goal_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
        )

    def sample_tasks(self, num_tasks):
        #a = np.random.random(num_tasks) * 2 * np.pi
        #r = 3 * np.random.random(num_tasks) ** 0.5
        a = np.random.random(num_tasks)  * np.pi
        r = 10 
        goals = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)
        tasks = [{'goal': goal} for goal in goals]
        return tasks

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])
    
    def reset(self):
        self.time_steps = 0
        return super().reset()

