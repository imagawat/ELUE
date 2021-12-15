import os
import pathlib
import numpy as np
import click
import json
import torch
import random
import copy
from datetime import datetime
import torch.optim as optim
import torch.nn as nn
from metaworld.benchmarks import ML1, ML10, ML45
"""
we use metaworld 2019/12/20(2957703f095ff5ca9c44c0295b80b4bb46aeca12) with a little modification for trajectory analysis.
note that recent version may not have benchmarks folder
"""
from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder, MlpIntegrator, MlpDecoder, MlpFlattenEncoder, Mlp, MlpWithAdditionalInputs
from rlkit.torch.extended_mlp import BeliefMlp, BeliefTanhGaussianPolicy
from rlkit.torch.sac.pearl import PEARLSoftActorCritic
from rlkit.torch.sac.elue import ELUESoftActorCritic
from rlkit.torch.sac.agent import PEARLAgent, ELUEAgent
from rlkit.launchers.launcher_util import setup_logger
import rlkit.torch.pytorch_util as ptu
from rlkit.samplers.in_place import InPlacePathSampler, BeliefUpdateInPlacePathSampler
from rlkit.samplers.parallel import ParallelSampler, ParallelEnvExecutor
from rlkit.torch.filmednets import FiLMedMlp, FiLMedTanhGaussianPolicy
from configs.default import default_config
from metaworld_util.ml45wrapper import HackyML45
import debug_config
from rlkit.core.additional_csv_writer import set_dir
from gym import wrappers


def experiment(variant, jobid):
    set_seed(variant['env_seed'])
    algo_params = variant['algo_params']
    
    exp_id = jobid
    if variant['path_to_weights'] != None:
        if variant['log_dir'] == None:
            exp_tag = 'itr_%d' % variant['num_itr']
        else:
            exp_tag = variant['log_dir']
        experiment_log_dir = setup_logger(exp_tag, variant=variant, exp_id=exp_id, base_log_dir=variant['path_to_weights'], snapshot_mode="last")
        set_dir(experiment_log_dir)
    else:
        if variant['log_dir'] == None:
            if variant['ml'] < 0:
                exp_tag = variant['env_name']
            elif variant['ml'] == 1:
                exp_tag = 'ML1_' + variant['env_name']
            else:
                exp_tag = 'ML%d' % variant['ml']
        else:
            exp_tag = variant['log_dir']
        experiment_log_dir = setup_logger(exp_tag, variant=variant, exp_id=exp_id, base_log_dir=variant['util_params']['base_log_dir'], snapshot_mode="gap", snapshot_gap=100)

    
    # optionally save eval trajectories as pkl files
    if algo_params['dump_eval_paths']:
        pickle_dir = experiment_log_dir + '/eval_trajectories'
        pathlib.Path(pickle_dir).mkdir(parents=True, exist_ok=True)



    
    # create multi-task environment and sample tasks
    if variant['ml'] == 1:
        env = ML1
    elif variant['ml'] == 10:
        env = ML10
    elif variant['ml'] == 45:
        env = HackyML45#ML45
    elif variant['ml'] == -1: # for mujoco 
        pass #do nothing
    
    else:
        raise NotImplementedError
    
    if variant['path_to_weights'] != None:
        if variant['ml'] == 1:
            env = env.get_test_tasks(variant['env_name'])
            tasks = env.sample_tasks(1)
        elif variant['ml'] == 10:
            env = env.get_test_tasks()
            tasks = env.sample_tasks(1)
        elif variant['ml'] == 45:
            env = env.get_test_tasks()
            # make obs_dim 9 for consistency between training and testing
            # TODO: check whether there is a better way
            #env._max_plain_dim = 9
            env.hack()
            tasks = env.sample_tasks(1)

        elif variant['ml'] == -1:
            env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
            alltasks = env.get_all_tasks()
            tasks = [alltasks[np.random.randint(0, variant['env_params']['n_tasks'])]]
            """
            eval_tasks = alltasks[variant['n_train_tasks']:]
            env.set_train_tasks(tasks)
            eval_env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
            eval_env.set_test_tasks(eval_tasks)
            """
        else:
            raise NotImplementedError
        """
        if variant['monitor']:
            #env = wrappers.Monitor(env, experiment_log_dir, video_callable=(lambda it: it % 100 == 0))
            # TODO: fix (this is not working)
            env = wrappers.Monitor(env, experiment_log_dir) 
        """
        eval_env = env
        eval_tasks = tasks
        eval_mode = True        
    else:
        if variant['ml'] == 1:
            if debug_config.EVALFROMSCRATCH:
                env = env.get_test_tasks(variant['env_name'])
                tasks = env.sample_tasks(1) 
            else:
                env = env.get_train_tasks(variant['env_name'])
            eval_env = env.get_test_tasks(variant['env_name'])
        elif variant['ml'] == -1:
            env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
            alltasks = env.get_all_tasks()
            tasks = alltasks[:variant['n_train_tasks']]
            eval_tasks = alltasks[variant['n_train_tasks']:]
            env.set_train_tasks(tasks)
            eval_env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
            eval_env.set_test_tasks(eval_tasks)
            if debug_config.EVALFROMSCRATCH:
                raise NotImplementedError
        else:
            if debug_config.EVALFROMSCRATCH:
                env = env.get_test_tasks()
                tasks = env.sample_tasks(1) 
            env = env.get_train_tasks()
            eval_env = env.get_test_tasks()

        if variant['ml'] != -1 and not debug_config.EVALFROMSCRATCH:
            tasks = env.sample_tasks(variant['n_train_tasks'])
        eval_mode = False
        variant["learning_type_at_eval"] = -1 

    if variant['ml'] > 0:#
        print_tasks(tasks)
    set_seed(variant['alg_seed'], True) 




    
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    reward_dim = 1
    algo_params = variant['algo_params']

    # instantiate networks
    latent_dim = algo_params['latent_size']
    net_size = algo_params['net_size']
    
    if variant['algo_name'] == "PEARL" or variant['algo_name'] == "PEARL_with_relearning": 
        recurrent = algo_params['recurrent']
        encoder_model = RecurrentEncoder if recurrent else MlpFlattenEncoder
        context_encoder_input_dim = 2 * obs_dim + action_dim + reward_dim if algo_params['use_next_obs_in_context'] else obs_dim + action_dim + reward_dim
        context_encoder_output_dim = latent_dim * 2 if algo_params['use_information_bottleneck'] else latent_dim
        training_mode = not eval_mode if variant['algo_name'] == "PEARL" else True        
        context_encoder = encoder_model(
            hidden_sizes=[200, 200, 200],
            input_size=context_encoder_input_dim,
            output_size=context_encoder_output_dim,
        )
        qf1 = FlattenMlp(
            hidden_sizes=[net_size, net_size, net_size],
            input_size= obs_dim + action_dim + latent_dim,
            output_size=1,
        )
        qf2 = FlattenMlp(
            hidden_sizes=[net_size, net_size, net_size],
            input_size= obs_dim + action_dim + latent_dim,
            output_size=1,
        )
        vf = FlattenMlp(
            hidden_sizes=[net_size, net_size, net_size],
            input_size= obs_dim + latent_dim,
            output_size=1,
        )    
        policy = TanhGaussianPolicy(
            hidden_sizes=[net_size, net_size, net_size],
            obs_dim= obs_dim + latent_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
        )
        agent = PEARLAgent(
            latent_dim,
            context_encoder,
            policy,
            **algo_params
        )
        sampler = InPlacePathSampler(
            env=env,
            policy=agent,
            max_path_length=algo_params['max_path_length']
        )
        algorithm = PEARLSoftActorCritic(
            ml=variant['ml'],
            env=env,
            tasks=tasks,
            nets=[agent, qf1, qf2, vf],
            latent_dim=latent_dim,
            sampler=sampler,
            training_mode=training_mode,
            eval_env=eval_env,
            num_eval_tasks=variant['n_eval_tasks'],
            num_processes=variant['num_processes'],
            **algo_params
        )

    elif variant['algo_name'] == "ELUE": 
    # encoder, integrator, decoder(predictor)
        belief_dim = latent_dim * 2 # latent variables are independent
        representation_dim = algo_params['representation_dim']
        encoder = MlpEncoder(
            hidden_sizes=[net_size],
            input_size=2*obs_dim + action_dim + reward_dim if algo_params['use_next_obs_in_context'] else obs_dim + action_dim + reward_dim,
            output_size=representation_dim
        )
        integrator = MlpIntegrator(
            hidden_sizes=[net_size],
            input_size=representation_dim,
            output_size=belief_dim,
            std_range=[1e-6, 1e+8],
        )
        assert(reward_dim == 1)
        rews_decoder = MlpDecoder(
            obs_dim=obs_dim,
            hidden_sizes=[net_size, net_size],
            input_size=latent_dim + obs_dim * 2 + action_dim if algo_params['rew_decoder_with_nobs'] else latent_dim + obs_dim + action_dim,            
            output_size=(reward_dim)*2,
            std_range=[1e-6, 1e+8],
        )
        if algo_params['use_nobs_decoder']: 
            nobs_decoder = MlpDecoder(
                obs_dim=obs_dim,
                hidden_sizes=[net_size, net_size],
                input_size=latent_dim + obs_dim + action_dim,#sampled_z
                output_size=(obs_dim)*2,
                std_range=[1e-6, 1e+8],
            )
        else:
            nobs_decoder = None
        if algo_params['noisy_input'] == 0:
            additional_input_dim = [0, 0]
        elif algo_params['noisy_input'] == 1:
            additional_input_dim = [1, 0]
        elif algo_params['noisy_input'] == 2:
            additional_input_dim = [0, 1]
        else:
            raise NotImplementedError
        if algo_params['mlp'] == 0:
            qf1 = FiLMedMlp(
            input_sizes=[belief_dim + obs_dim + action_dim, obs_dim + action_dim] if algo_params['film_gen_with_full_inputs'] else [belief_dim,obs_dim + action_dim],
            output_size=1,
            **algo_params,
            )
            qf2 = FiLMedMlp(
            input_sizes=[belief_dim + obs_dim + action_dim, obs_dim + action_dim] if algo_params['film_gen_with_full_inputs'] else [belief_dim,obs_dim + action_dim],
            output_size=1,
            **algo_params,
            )

            vf = FiLMedMlp(
            input_sizes=[belief_dim + obs_dim, obs_dim] if algo_params['film_gen_with_full_inputs'] else [belief_dim,obs_dim],
            output_size=1,
            **algo_params,
            )

            
            policy = FiLMedTanhGaussianPolicy(
            input_sizes=[belief_dim + obs_dim + additional_input_dim[0], obs_dim + additional_input_dim[1]] if algo_params['film_gen_with_full_inputs'] else [belief_dim + additional_input_dim[0], obs_dim + additional_input_dim[1]],
            action_dim=action_dim,
            **algo_params,
            )

        else:
            qf1 = BeliefMlp(
                input_size=obs_dim + action_dim,
                belief_input_size=belief_dim,
                output_size=1,
                **algo_params,
            )
            qf2 = BeliefMlp(
                input_size=obs_dim + action_dim,
                belief_input_size=belief_dim,
                output_size=1,
                **algo_params,
            )
            vf = BeliefMlp(
                input_size=obs_dim,
                belief_input_size=belief_dim,
                output_size=1,
                **algo_params,
            )
            
            policy = BeliefTanhGaussianPolicy(
                input_size=obs_dim + additional_input_dim[1],
                belief_input_size=belief_dim + additional_input_dim[0],
                output_size=action_dim,
                **algo_params,
            )
            
        if debug_config.PREDTEST:
            predictor = Mlp(
                hidden_sizes = [net_size],
                input_size = belief_dim,
                output_size = len(tasks),
            )
        else:
            predictor = None

        agent = ELUEAgent(
            latent_dim=latent_dim,
            encoder=encoder,
            integrator=integrator,
            rews_decoder=rews_decoder,
            nobs_decoder=nobs_decoder,
            predictor=predictor,
            policy=policy,
            eval_mode=eval_mode,
            **algo_params
        )
        if variant['num_processes'] == 1:
            sampler = BeliefUpdateInPlacePathSampler(
            env=env,
            policy=agent,
            max_path_length=algo_params['max_path_length'] #max_path_length is not used
            )
        else:            
            sampler = ParallelSampler(policy=agent)
            vecenv = ParallelEnvExecutor(env=env, n_envs=variant['num_processes'])
            sampler.set_vecenv(vecenv)
            # at eval, sampling by this is different from sampling when num_processes == 1, because beliefs are not shared among envs. 
            
        algorithm = ELUESoftActorCritic(
            ml=variant['ml'],
            env=env,
            latent_dim=latent_dim,
            nets=[agent, qf1, qf2, vf],
            tasks=tasks,
            sampler=sampler,
            eval_env=eval_env,
            num_eval_tasks=variant['n_eval_tasks'],
            num_processes=variant['num_processes'],
            loss_function=eval(algo_params['loss_metric']),
            eval_mode=eval_mode,
            optimizer_class=eval(algo_params['optimizer']),
            **algo_params
        )

    else:
       raise NotImplementedError
        
    #    if eval_mode and not debug_config.IGNOREZ:
    if eval_mode:
        path = variant['path_to_weights']
        st = '%s' + '_itr_%d.pth' % variant['num_itr']
        qf1.load_state_dict(torch.load(os.path.join(path, st % 'qf1')))
        qf2.load_state_dict(torch.load(os.path.join(path, st % 'qf2')))
        vf.load_state_dict(torch.load(os.path.join(path, st % 'vf')))
        if algo_params['not_load_target']:        
            algorithm.target_vf = vf.copy()
        else:
            algorithm.target_vf.load_state_dict(torch.load(os.path.join(path, st % 'target_vf')))

        
        policy.load_state_dict(torch.load(os.path.join(path, st % 'policy')))
        

        if variant['algo_name'] == "ELUE":
            if algo_params['policy_kl_loss_coeff'] > 0:
                algorithm.set_initial_policy()
            encoder.load_state_dict(torch.load(os.path.join(path, st % 'encoder')))
            integrator.load_state_dict(torch.load(os.path.join(path, st % 'integrator')))
            rews_decoder.load_state_dict(torch.load(os.path.join(path, st % 'rews_decoder')))
            if algo_params['use_nobs_decoder']:
                nobs_decoder.load_state_dict(torch.load(os.path.join(path, st % 'nobs_decoder')))

        else:
            context_encoder.load_state_dict(torch.load(os.path.join(path, st % 'context_encoder')))


    


    # optional GPU mode
    if ptu.gpu_enabled():
        algorithm.to()
        set_gpu_seed(variant['alg_seed'])
        
    """
    print(qf1.film_vars)        
    qf1.film_vars.to(ptu.device)
    print(qf1.film_vars)
    qf1.film_vars.to(torch.device("cuda"))
    print(qf1.film_vars)
    """
    # debugging triggers a lot of printing and logs to a debug directory
    #DEBUG = variant['util_params']['debug']
    #os.environ['DEBUG'] = str(int(DEBUG))


    # run the algorithm
    algorithm.train()

    

def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to

def expand_path(dirname):    
    if dirname[0] == '~':
        return os.path.expanduser(dirname)
    elif dirname[0] == '/':
        return dirname
    else:
        return os.getcwd() + '/' + dirname


##@click.option('--docker', is_flag=True, default=False)
#@click.option('--debug', is_flag=True, default=False)

def set_seed(seed, with_torch=False):
    np.random.seed(seed)
    random.seed(seed)
    if with_torch:
        torch.manual_seed(seed)

def set_gpu_seed(seed):
    torch.cuda.manual_seed_all(seed)

def print_tasks(tasks):
    buf = [[] for i in range(45)]
    for t in tasks:
        buf[t['task']].append(t['goal'])

    for i in range(45):
        for g in buf[i]:
            print(i, g)

    
@click.command()
@click.option('--config', default=None)
@click.option('--gpu', default=0)
@click.option('--logdir', default=None)
@click.option('--saveddir', default=None) # None means training mode, (eval mode needs pre-trained networks)
@click.option('--num_itr', default=0) # this value is used for identification of pre-trained networks. this is ignored if saveddir is none
@click.option('--jobid', default=None) # 
@click.option('--num_processes', default=None) # 
@click.option('--env_seed', default=None) # meta world's goal positions can not be controlled by this seed ? 
@click.option('--alg_seed', default=None) # results may not be the same even if you set seed 
@click.option('--use_debug_config', is_flag=True, default=False) # to avoid unintentionally using it
@click.option('--monitor', is_flag=True, default=False) # used only in meta-testing ...may not work?
def main(config, gpu, logdir, saveddir, num_itr, jobid, num_processes, env_seed, alg_seed, use_debug_config, monitor):
    print(debug_config.variables)
    if not use_debug_config:
        for k, v in debug_config.variables.items():
            if v:
                print(k, " is not expected to be True")
                raise RuntimeError            

    variant = default_config
    if saveddir != None:
        # this means eval_mode
        # use same config when it was trained
        saveddir = expand_path(saveddir)
        with open(os.path.join(saveddir, 'variant.json')) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)

    
    if config != None:
        config = expand_path(config)
        with open(config) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    variant['path_to_weights'] = saveddir
    variant['log_dir'] = logdir
    variant['util_params']['gpu_id'] = gpu
    variant['num_itr'] = num_itr
    variant['util_params']['base_log_dir'] = variant['algo_name']
    if num_processes != None:
        variant['num_processes'] = int(num_processes)
    if variant['num_processes'] > 1 and saveddir != None:
        print("parallel sampling at eval may be different from single process sampling")
        raise NotImplementedError
    for seed, name in zip([env_seed, alg_seed], ['env_seed', 'alg_seed']):
        if seed == None:
            seed = datetime.now().microsecond
        variant[name] = seed            
    variant['monitor'] = monitor
    # optional GPU mode
    ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])
    experiment(variant, jobid)

if __name__ == "__main__":
    main()
