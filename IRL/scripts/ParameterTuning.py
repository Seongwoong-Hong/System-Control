import os
import pickle
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
from algos.torch.MaxEntIRL import MaxEntIRL
from algos.tabular.qlearning import SoftQLearning
from common.util import make_env
from common.wrappers import RewardWrapper
from common.rollouts import generate_trajectories_without_shuffle


def trial_str_creator(trial):
    trialname = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + trial.trial_id
    return trialname


def try_train(config, demo_dir):
    logger.configure(tune.get_trial_dir(), format_strs=['tensorboard'])

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
            return th.cat([x, x**2], dim=1)
    elif config['feature'] == 'no':
        def feature_fn(x):
            return x
    else:
        raise NotImplementedError

    env = make_env(f"{config['env_id']}_disc-v2", map_size=config['map_size'])
    eval_env = make_env(f"{config['env_id']}_disc-v0", map_size=config['map_size'])

    agent = SoftQLearning(env, gamma=config['gamma'], epsilon=0.4, alpha=config['alpha'], device='cpu')

    with open(demo_dir + f"/viter_disc_{config['map_size']}.pkl", "rb") as f:
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
        rew_kwargs={'type': 'ann', 'scale': 1, 'alpha': 0.05},
    )

    for epoch in range(50):
        trial_dir = tune.get_trial_dir()
        """ Learning """
        algo.learn(
            total_iter=1,
            agent_learning_steps=5e3,
            n_episodes=len(expert_trajs),
            max_agent_iter=12,
            min_agent_iter=2,
            max_gradient_steps=6000,
            min_gradient_steps=1000,
        )

        """ Testing """
        trajectories = generate_trajectories_without_shuffle(
            algo.agent, DummyVecEnv([lambda: eval_env]), sample_until, deterministic_policy=False)

        expt_obs = rollout.flatten_trajectories(expert_trajs).obs
        agent_obs = rollout.flatten_trajectories(trajectories).obs
        mean_obs_differ = np.abs((expt_obs - agent_obs).mean())
        algo.agent.save(trial_dir + "/agent")
        algo.reward_net.save(trial_dir + "/reward_net")

        tune.report(mean_obs_differ=mean_obs_differ)


def main(target):

    demo_dir = os.path.abspath(os.path.join("..", "demos", target))
    config = {
        'env_id': target,
        'gamma': tune.choice([0.9, 0.8]),
        'alpha': tune.choice([0.1, 0.2, 0.3]),
        'use_action': tune.choice([True, False]),
        'expt': tune.choice(['viter']),
        'map_size': tune.choice([4, 5, 6, 7]),
        'rew_arch': tune.choice(['linear', 'one', 'two']),
        'feature': tune.choice(['ext', 'no'])
    }

    scheduler = ASHAScheduler(
        metric="mean_obs_differ",
        mode="min",
        max_t=50,
        grace_period=10,
        reduction_factor=2,
    )
    reporter = CLIReporter(metric_columns=["mean_obs_differ", "training_iteration"])
    result = tune.run(
        partial(try_train, demo_dir=demo_dir),
        name=target + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
        resources_per_trial={"cpu": 1},
        config=config,
        num_samples=200,
        scheduler=scheduler,
        progress_reporter=reporter,
        checkpoint_at_end=True,
        trial_name_creator=trial_str_creator,
        local_dir="/opt/project/IRL/tmp/log/ray_result"
    )

    best_trial = result.get_best_trial("mean_reward", "max", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['mean_reward']:.4f}")

    best_logdir = result.get_best_logdir(metric='mean_obs_differ', mode='min')
    print(best_logdir)


if __name__ == "__main__":
    main('2DTarget')
