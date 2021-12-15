default_config = dict(
    ml = None,
    algo_name="ELUE",
    env_name='cheetah-dir',
    env_params={
    },
    n_train_tasks=2,
    n_eval_tasks=10, 
    path_to_weights=None, # path to pre-trained weights to load into networks
    num_processes=1, # 
    #env_params=dict(
    #    n_tasks=2, # number of distinct tasks in this domain, shoudl equal sum of train and eval tasks
    #    randomize_tasks=True, # shuffle the tasks after creating them
    #),
    algo_params=dict(

        #TODO: remove (original PEARL's unused params)
        max_path_length=200, # max path length for this environment
        dump_eval_paths=False, # whether to save evaluation trajectories
        num_evals=2, # number of independent evals for each eval task in training loop 
        num_steps_per_eval=400,  # nunmber of transitions for eval in training loop 

        # shared
        latent_size=5, # dimension of the latent context vector
        net_size=300, # 
        meta_batch=16, # number of tasks to average the gradient across 
        num_iterations=500, # number of data sampling / training iterates
        num_initial_steps=2000, # number of transitions collected per task before training
        num_tasks_sample=16, # number of randomly sampled tasks to collect data for each iteration
        num_train_steps_per_itr=2000, # number of meta-gradient steps taken per iteration
        batch_size=256, # number of transitions in the RL batch
        discount=0.99, # RL discount factor
        soft_target_tau=0.005, # for SAC target network update
        policy_lr=3E-4,
        qf_lr=3E-4,
        vf_lr=3E-4,
        context_lr=3e-4,
        reward_scale=5., # scale rewards before constructing Bellman update, effectively controls weight on the entropy of the policy
        sparse_rewards=False, # whether to sparsify rewards as determined in env
        kl_lambda=.1, # weight on KL divergence term 
        validation_outputs=[True, False, True],# whether n-th episode stats are logged or not
        eval_interval = 10,
        deterministic_sampling=False, # sampling data with deterministic policy (for analysis at meta-test phase)
        use_next_obs_in_context=True, # use next obs if it is useful in distinguishing tasks

        # PEARL
        embedding_batch_size=64, # number of transitions in the context batch
        embedding_mini_batch_size=64, # number of context transitions to backprop through (should equal the arg above except in the recurrent encoder case)
        num_steps_prior=400, # number of transitions to collect per task with z ~ prior
        num_steps_posterior=0, # number of transitions to collect per task with z ~ posterior
        num_extra_rl_steps_posterior=400, # number of additional transitions to collect per task with z ~ posterior that are only used to train the policy and NOT the encoder
        use_information_bottleneck=True, # False makes latent context deterministic
        recurrent=False, # recurrent or permutation-invariant encoder
        update_post_train=1, # how often to resample the context when collecting data during training (in trajectories)
        
        # ELUE
        optimizer="optim.Adam",
        loss_metric="nn.MSELoss",
        entropy_coeff=1,
        num_sampled_z=1, # number of sampled z when belief net is trained like CVAE 
        num_context_per_train_step=2, # number of context when embeddings are trained
        num_sac_set_per_train_step=2, # number of RL batch when actor and critic are trained 
        num_steps=800, # number of transitions to collect per task 
        reset_belief_rate=2, # rate of reseting belief per episode when collecting data from env (this value is inverse number which is >= 1 because int is preferable)
        rand_length_train_rate=5, # (inverse number) rate of random length training of CVAE (otherwise the length is max_context_length)
        max_rand_length=5, # max length when random length training is executed
        max_context_length=64,#
        representation_dim=32,
        num_init_embedding_train=500,
        num_embedding_train_steps_per_itr=1000, #
        embedding_batch_with_all_tasks=True, # True: each batch data for embedding training contains all tasks' data, False: random sampled tasks' data (same as meta_batch)
        grad_clip=1e+4,
        hidden_sizes=[256, 256, 256], #hidden sizes of layers in policy, Q, and V networks
        mlp=1, # mlp > 1 means mlp, 0 means film
        rew_decoder_with_nobs=True, # use next observation for reward decoder or not
        use_nobs_decoder=True,
        # params about bottleneck
        naive_bottleneck=True,
        bottleneck_layerid=2,# negative means no bottleneck
        q_v_bottleneck=True, # True means uses bottleneck in q and v net
        bottleneck_size=100,
        q_bottleneck_coeff=0.1,
        v_bottleneck_coeff=0.1,
        pi_bottleneck_coeff=0.1,
        mean_bottleneck_loss = True,
        uniform_variational_dist_for_bottleneck = True,
        std_for_variational_dist_for_bottleneck = 1,
        additional_policy_regs=True, # PEARL's addtional policy regularization loss
        max_length_embedding_train=False, #train embedding only with max_context_length or with random length (rand_length_train_rate)

        # params for meta-testing
        reset_belief_at_eval = False, # for ELUE 
        learning_type_at_eval=0, # types of params which are updated by the optimizer 0,1: NotImpremented, 2: all params of networks , 3:nothing, 4: all params and belief, 5: belief (when it is at meta-training, this value is set to be -1) # for ELUE (PEARL)
        fix_context_at_eval=True, #


        # using FiLM TODO: remove (unused)       
        film_gennet_lr=0.0003,
        with_connection_layer=True,
        with_activation_before_film=True,
        gennet_hidden_sizes=[16,16],
        filmed=[True, True], #filmed=[False, False, True, False], #
        feat_sizes=[16,16],  #len is the number of FiLM layers
        num_feat_tensors=[4,4],
        #hidden_sizes=[64, 64], #size must be feat_size*num_feat_tensor before and after FiLM layer if with_connection_layer is False
        layer_norm=False,
        #learning_type_at_eval=0, #0 is filmgen, 1 is the other parts, 2 is all (except task embedding) and 3 is nothing   4 is directly change film_var (film_var and other parts),...  when it is at meta-training, the value set to be -1

        #TODO: remove (unused)     
        burn_in_iteration_size = 0,# at burn_in phase, nothing is updated
        two_step_sac=False,
        drastic_target_update=False,
        no_vf=False,        
        sequential=False,#rue, # training based on sequential data
        qf_ave_as_target=False,
        consistency_loss_coeff=0, #
        film_gen_with_full_inputs=False,
        noisy_input=0, # add 1-d Gaussian noise to policy input (0 means without noise, 1 with belief, 2 with state, and 3 both). The noise is consistent from the beggining of episode to the end. This may help exploration by adding diversity like MAESN ().

        # eval unused params TODO: remove
        policy_kl_loss_coeff = 1, #TODO: rename (actually this is a coef. of cross entropy)
        train_steps_gamma = 0,
        not_load_target = False, # for debug 

    ),
    util_params=dict(
        #base_log_dir='output', # put this value by @click.option
        use_gpu=True,
        gpu_id=0,
        debug=False, # debugging triggers printing and writes logs to debug directory
        docker=False, # TODO docker is not yet supported
    ),
    #debug_params=dict(
    #    train_test=False,
    #    ignore_film=False,
    #),
)



"""
# original params in PEARL code.
#ant goal
{
    "env_name": "ant-goal",
    "n_train_tasks": 150,
    "n_eval_tasks": 30,
    "env_params": {
        "n_tasks": 180
    }
    ,
    "algo_params": {
        "meta_batch": 10,
        "num_initial_steps": 2000,
        "num_steps_prior": 400,
        "num_steps_posterior": 0,
        "num_extra_rl_steps_posterior": 600,
        "num_train_steps_per_itr": 4000,
        "num_evals": 1,
        "num_steps_per_eval": 600,
        "num_exp_traj_eval": 2,
        "embedding_batch_size": 256,
        "embedding_mini_batch_size": 256,
        "kl_lambda": 1.0
    }
}
#default
default_config = dict(
    env_name='cheetah-dir',
    n_train_tasks=2,
    n_eval_tasks=2,
    latent_size=5, # dimension of the latent context vector
    net_size=300, # number of units per FC layer in each network
    path_to_weights=None, # path to pre-trained weights to load into networks
    env_params=dict(
        n_tasks=2, # number of distinct tasks in this domain, shoudl equal sum of train and eval tasks
        randomize_tasks=True, # shuffle the tasks after creating them
    ),
    algo_params=dict(
        meta_batch=16, # number of tasks to average the gradient across
        num_iterations=500, # number of data sampling / training iterates
        num_initial_steps=2000, # number of transitions collected per task before training
        num_tasks_sample=5, # number of randomly sampled tasks to collect data for each iteration
        num_steps_prior=400, # number of transitions to collect per task with z ~ prior
        num_steps_posterior=0, # number of transitions to collect per task with z ~ posterior
        num_extra_rl_steps_posterior=400, # number of additional transitions to collect per task with z ~ posterior that are only used to train the policy and NOT the encoder
        num_train_steps_per_itr=2000, # number of meta-gradient steps taken per iteration
        num_evals=2, # number of independent evals
        num_steps_per_eval=600,  # nuumber of transitions to eval on
        batch_size=256, # number of transitions in the RL batch
        embedding_batch_size=64, # number of transitions in the context batch
        embedding_mini_batch_size=64, # number of context transitions to backprop through (should equal the arg above except in the recurrent encoder case)
        max_path_length=200, # max path length for this environment
        discount=0.99, # RL discount factor
        soft_target_tau=0.005, # for SAC target network update
        policy_lr=3E-4,
        qf_lr=3E-4,
        vf_lr=3E-4,
        context_lr=3e-4,
        reward_scale=5., # scale rewards before constructing Bellman update, effectively controls weight on the entropy of the policy
        sparse_rewards=False, # whether to sparsify rewards as determined in env
        kl_lambda=.1, # weight on KL divergence term in encoder loss
        use_information_bottleneck=True, # False makes latent context deterministic
        use_next_obs_in_context=False, # use next obs if it is useful in distinguishing tasks
        update_post_train=1, # how often to resample the context when collecting data during training (in trajectories)
        num_exp_traj_eval=1, # how many exploration trajs to collect before beginning posterior sampling at test time
        recurrent=False, # recurrent or permutation-invariant encoder
        dump_eval_paths=False, # whether to save evaluation trajectories
    ),
    util_params=dict(
        base_log_dir='output',
        use_gpu=True,
        gpu_id=0,
        debug=False, # debugging triggers printing and writes logs to debug directory
        docker=False, # TODO docker is not yet supported
    )
)
"""
