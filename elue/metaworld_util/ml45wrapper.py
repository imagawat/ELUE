from metaworld.benchmarks.base import Benchmark
from metaworld.core.serializable import Serializable
from metaworld.envs.mujoco.multitask_env import MultiClassMultiTaskEnv
from metaworld.envs.mujoco.env_dict import HARD_MODE_ARGS_KWARGS, HARD_MODE_CLS_DICT
#from metaworld.benchmarks import ML45
from gym.spaces import Box




class HackyML45(MultiClassMultiTaskEnv, Benchmark, Serializable):
    """
    to avoid inconsistency between train and test in ML45, I add little trick.
    """

    def __init__(self, env_type='train', sample_all=False):
        assert env_type == 'train' or env_type == 'test'
        Serializable.quick_init(self, locals())

        cls_dict = HARD_MODE_CLS_DICT[env_type]
        args_kwargs = HARD_MODE_ARGS_KWARGS[env_type]

        super().__init__(
            task_env_cls_dict=cls_dict,
            task_args_kwargs=args_kwargs,
            sample_goals=True,
            obs_type='plain',
            sample_all=sample_all)
        self._eval_mode = (env_type=='test')

    def hack(self):
        if self._eval_mode:
            self._max_plain_dim = 9
    @property
    def observation_space(self):
        if self._eval_mode:
            #dummy 
            return Box(low=-1e8, high=1e8, shape=(9,))
        return super().observation_space
"""
class HackyML45(ML45):
    def __init__(self):
        super().__init__()
        self.eval_mode = False
    def hack(self, eval_mode):        
        if eval_mode:
            self._max_plain_dim = 9            
        self.eval_mode = eval_mode
    def observation_space(self):
        if self.eval_mode:
            #dummy 
            return Box(low=-1e8, high=1e8, shape=(9,))
        return super().observation_space()
"""
