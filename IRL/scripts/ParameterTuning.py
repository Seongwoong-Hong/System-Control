import os
import pickle

import ray
import torch as th
import numpy as np
from datetime import datetime
from functools import partial

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
from common.wrappers import RewardWrapper
from common.rollouts import generate_trajectories_without_shuffle


# ray.init(local_mode=True)


def trial_str_creator(trial):
    trialname = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + trial.trial_id
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
    elif config['feature'] == 'no':
        def feature_fn(x):
            return x
    elif config['feature'] == '1hot':
        def feature_fn(x):
            if len(x.shape) == 1:
                x = x.reshape(1, -1)
            ft = th.zeros([x.shape[0], 100], dtype=th.float32)
            for i, row in enumerate(x):
                idx = int((row[0] + row[1] * config['map_size']).item())
                ft[i, idx] = 1
            return ft
    else:
        raise NotImplementedError

    env = make_env(f"{config['env_id']}_disc-v2", map_size=config['map_size'])
    eval_env = make_env(f"{config['env_id']}_disc-v0", map_size=config['map_size'])

    agent = FiniteSoftQiter(env, gamma=config['gamma'], alpha=config['alpha'], device='cpu')

    with open(demo_dir + f"/{config['expt']}_disc_{config['map_size']}.pkl", "rb") as f:
        expert_trajs = pickle.load(f)

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
        env_kwargs={'vec_normalizer': None, 'reward_wrapper': RewardWrapper},
        rew_kwargs={'type': 'ann', 'scale': 1, 'alpha': config['norm_coeff'], 'lr': config['lr']},
    )
    trial_dir = tune.get_trial_dir()
    eval_env = DummyVecEnv([lambda: eval_env])
    for epoch in range(1000):
        import time
        t1 = time.time()
        os.makedirs(trial_dir + f"/model/{epoch:03d}", exist_ok=False)

        """ Learning """
        algo.learn(
            total_iter=10,
            agent_learning_steps=5e3,
            n_episodes=len(expert_trajs),
            max_agent_iter=1,
            min_agent_iter=1,
            max_gradient_steps=1,
            min_gradient_steps=1,
        )

        """ Testing """
        # trajectories = generate_trajectories_without_shuffle(
        #     algo.agent, eval_env, sample_until, deterministic_policy=False)
        trajectories = []
        t2 = time.time()
        mean_obs_differ = 0.0
        for expt_traj in expert_trajs:
            init_ob = expt_traj.obs[0, :][None, :]
            agent_obs, _ = algo.agent.predict(init_ob)
            mean_obs_differ += np.abs(expt_traj.obs - agent_obs).mean()
        mean_obs_differ /= len(expert_trajs)
        # expt_obs = rollout.flatten_trajectories(expert_trajs).obs
        # agent_obs = rollout.flatten_trajectories(trajectories).obs
        # mean_obs_differ = np.abs((expt_obs - agent_obs)).mean()
        if epoch % 20 == 0:
            algo.agent.save(trial_dir + f"/model/{epoch:03d}/agent")
            algo.reward_net.save(trial_dir + f"/model/{epoch:03d}/reward_net")

        algo.agent.set_env(env)
        algo.reward_net.feature_fn = feature_fn
        t3 = time.time()
        tune.report(mean_obs_differ=mean_obs_differ)


def main(target):
    metric = "mean_obs_differ"
    demo_dir = os.path.abspath(os.path.join("..", "demos", target))
    config = {
        'env_id': target,
        'gamma': tune.choice([0.8]),
        'alpha': tune.uniform(0.05, 4),
        'use_action': tune.choice([False]),
        'expt': tune.choice(['softqiter']),
        'map_size': tune.choice([10]),
        'rew_arch': tune.choice(['linear']),
        'feature': tune.choice(['ext', '1hot']),
        'lr': tune.uniform(0.01, 0.1),
        'norm_coeff': tune.choice([0.5]),
    }

    scheduler = ASHAScheduler(
        metric=metric,
        mode="min",
        max_t=100,
        grace_period=20,
        reduction_factor=2,
    )
    reporter = CLIReporter(metric_columns=[metric, "training_iteration"])
    irl_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    result = tune.run(
        partial(try_train, demo_dir=demo_dir),
        name=target + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
        resources_per_trial={"cpu": 1},
        config=config,
        num_samples=500,
        scheduler=scheduler,
        progress_reporter=reporter,
        checkpoint_at_end=True,
        trial_name_creator=trial_str_creator,
        local_dir=f"{irl_path}/tmp/log/ray_result"
    )

    best_trial = result.get_best_trial(metric, "min", "all")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result[metric]:.4f}")

    best_logdir = result.get_best_logdir(metric=metric, mode='min')
    print(best_logdir)


if __name__ == "__main__":
    main('2DTarget')
