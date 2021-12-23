import os
import pickle

import ray
import torch as th
import numpy as np
from datetime import datetime
from functools import partial
from scipy import io

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from imitation.data import rollout
from imitation.util import logger
from stable_baselines3.common.vec_env import DummyVecEnv
from algos.torch.MaxEntIRL import *
from algos.tabular.qlearning import *
from algos.tabular.viter import *
from common.util import make_env
from common.verification import verify_policy
from common.wrappers import ActionRewardWrapper
from common.rollouts import generate_trajectories_without_shuffle


# ray.init(local_mode=True)


def trial_name_string(trial):
    trialname = f"{trial.config['expt']}_{trial.config['actuation']}_{trial.config['trial']}_" + trial.trial_id
    return trialname


def try_train(config, demo_dir):
    logger.configure(tune.get_trial_dir(), format_strs=["tensorboard"])

    if config['rew_arch'] == 'linear':
        rew_arch = []
    elif config['rew_arch'] == 'one':
        rew_arch = [8]
    elif config['rew_arch'] == 'two':
        rew_arch = [8, 8]
    else:
        raise NotImplementedError
    if config['feature'] == 'ext':
        def feature_fn(x):
            return th.cat([x, x ** 2], dim=1)
    elif config['feature'] == 'cross':
        def feature_fn(x):
            x1, x2, dx1, dx2 = th.split(x, 1, dim=-1)
            return th.cat([x ** 2, x1 * x2, dx1 * dx2, x1 * dx1, x2 * dx2], dim=1)
    elif config['feature'] == 'no':
        def feature_fn(x):
            return x
    elif config['feature'] == 'sq':
        def feature_fn(x):
            return x ** 2
    # elif config['feature'] == '1hot':
    #     def feature_fn(x):
    #         if len(x.shape) == 1:
    #             x = x.reshape(1, -1)
    #         ft = th.zeros([x.shape[0], config['map_size'] ** 2], dtype=th.float32)
    #         for i, row in enumerate(x):
    #             idx = int((row[0] + row[1] * config['map_size']).item())
    #             ft[i, idx] = 1
    #         return ft
    else:
        raise NotImplementedError

    subpath = os.path.join(demo_dir, "..", "HPC", config['expt'], config['expt'])
    with open(demo_dir + f"/09191927/{config['expt']}_{config['actuation']}_{config['trial']}.pkl", "rb") as f:
        expert_trajs = pickle.load(f)
    init_states = []
    for traj in expert_trajs:
        init_states += [traj.obs[0]]
    bsp = io.loadmat(subpath + f"i1.mat")['bsp']
    env = make_env(f"{config['env_id']}-v2", N=[9, 19, 19, 27], bsp=bsp)
    eval_env = make_env(f"{config['env_id']}-v0", N=[9, 19, 19, 27], bsp=bsp, init_states=init_states)

    agent = FiniteSoftQiter(env, gamma=config['gamma'], alpha=config['alpha'], device='cpu')
    eval_agent = SoftQiter(env, gamma=config['gamma'], alpha=config['alpha'], device='cpu')

    sample_until = rollout.make_sample_until(n_timesteps=None, n_episodes=len(expert_trajs))

    algo = MaxEntIRL(
        env,
        eval_env=eval_env,
        feature_fn=feature_fn,
        agent=agent,
        expert_trajectories=expert_trajs,
        use_action_as_input=config['use_action'],
        rew_arch=rew_arch,
        device='cpu',
        env_kwargs={'vec_normalizer': None, 'reward_wrapper': ActionRewardWrapper},
        rew_kwargs={'type': 'ann', 'scale': 1, 'norm_coeff': config['norm_coeff'], 'lr': config['lr']},
    )
    trial_dir = tune.get_trial_dir()
    eval_env = DummyVecEnv([lambda: eval_env])
    expt_obs = rollout.flatten_trajectories(expert_trajs).obs
    for epoch in range(1):
        os.makedirs(trial_dir + f"/model/{epoch:03d}", exist_ok=False)

        """ Learning """
        algo.learn(
            total_iter=200,
            agent_learning_steps=0,
            n_episodes=len(expert_trajs),
            max_agent_iter=1,
            min_agent_iter=1,
            max_gradient_steps=1,
            min_gradient_steps=1,
        )

        """ Testing """
        eval_agent.policy.policy_table = algo.agent.policy.policy_table[0]
        trajectories = generate_trajectories_without_shuffle(
            eval_agent, eval_env, sample_until, deterministic_policy=False)

        agent_obs = rollout.flatten_trajectories(trajectories).obs
        mean_obs_differ = np.abs((expt_obs - agent_obs)).mean()
        # algo.agent.save(trial_dir + f"/model/{epoch:03d}/agent")
        algo.reward_net.save(trial_dir + f"/model/{epoch:03d}/reward_net")

        algo.agent.set_env(env)
        algo.reward_net.feature_fn = feature_fn
        tune.report(mean_obs_differ=mean_obs_differ)


def main(target):
    metric = "mean_obs_differ"
    demo_dir = os.path.abspath(os.path.join("..", "demos", target))
    config = {
        'env_id': target,
        'gamma': tune.grid_search([1]),
        'alpha': tune.grid_search([0.005]),
        'use_action': tune.grid_search([True]),
        'actuation': tune.grid_search([1, 2, 3, 4, 5, 6, 7]),
        'trial': tune.grid_search([1, 2, 3, 4, 5]),
        'expt': tune.grid_search([f"sub{i:02d}" for i in [1, 3, 4, 5, 6, 7, 9, 10]]),
        'rew_arch': tune.grid_search(['linear']),
        'feature': tune.grid_search(['sq']),
        'lr': tune.grid_search([1e-2]),
        'norm_coeff': tune.grid_search([0.0]),
    }

    scheduler = ASHAScheduler(
        metric=metric,
        mode="min",
        max_t=1,
        grace_period=1,
        reduction_factor=2,
    )
    reporter = CLIReporter(metric_columns=[metric, "training_iteration"])
    irl_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    result = tune.run(
        partial(try_train, demo_dir=demo_dir),
        name=target + '_sq_act',
        resources_per_trial={"cpu": 2},
        config=config,
        num_samples=1,
        scheduler=scheduler,
        progress_reporter=reporter,
        checkpoint_at_end=True,
        trial_name_creator=trial_name_string,
        local_dir=f"{irl_path}/tmp/log/ray_result"
    )

    best_trial = result.get_best_trial(metric, "min", "all")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result[metric]:.4f}")

    best_logdir = result.get_best_logdir(metric=metric, mode='min')
    print(best_logdir)


if __name__ == "__main__":
    main('DiscretizedHuman')
