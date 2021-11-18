import os
import numpy as np
from datetime import datetime
from functools import partial

import click

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from algos.torch.sac import SAC, MlpPolicy
from common.util import make_env


def try_train(config):
    algo = SAC(config['policy'],
               env=config['env'],
               batch_size=config['batch_size'],
               learning_starts=100,
               learning_rate=config['lr'],
               train_freq=(config['accum_steps'], 'step'),
               gradient_steps=config['gradient_steps'],
               gamma=config['gamma'],
               ent_coef=config['ent_coef'],
               target_update_interval=1,
               device='cuda')

    for epoch in range(50):
        """ Training """
        algo.learn(total_timesteps=int(1e4), reset_num_timesteps=False)

        """ Validation """
        rewards = []
        for _ in range(10):
            obs = algo.env.reset()
            done = False
            traj_r = 0
            while not done:
                act, _ = algo.predict(obs, deterministic=False)
                ns, r, done, info = algo.env.step(act)
                traj_r += r
            rewards.append(traj_r)
        mean_reward = np.mean(rewards)

        """ Temporally save a model"""
        trial_dir = tune.get_trial_dir()
        path = os.path.join(trial_dir, "agent")
        algo.save(path)
        # tune.report()
        tune.report(mean_reward=mean_reward)


@click.command()
@click.option('--target', default='IDP')
def main(target):
    env = make_env(f"{target}_custom-v2")
    config = {
        'policy': MlpPolicy,
        'env': env,
        'batch_size': tune.choice([64, 128, 256]),
        'accum_steps': tune.choice([1000, 2000, 3000, 4000, 5000]),
        'gradient_steps': tune.choice([1000, 2000, 3000]),
        'gamma': tune.choice([0.99, 0.975, 0.95]),
        'ent_coef': tune.choice([0.1, 0.2, 0.3, 0.4, 0.5]),
        'lr': tune.loguniform(1e-4, 5e-4),
    }

    scheduler = ASHAScheduler(
        metric="mean_reward",
        mode="max",
        max_t=50,
        grace_period=20,
        reduction_factor=2,
    )
    reporter = CLIReporter(metric_columns=["mean_reward", "training_iteration"])
    result = tune.run(
        partial(try_train),
        name=target + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
        resources_per_trial={"cpu": 1, "gpu": 0.33},
        config=config,
        num_samples=500,
        scheduler=scheduler,
        progress_reporter=reporter,
        checkpoint_at_end=True,
    )

    best_trial = result.get_best_trial("mean_reward", "max", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['mean_reward']:.4f}")

    best_logdir = result.get_best_logdir(metric='mean_reward', mode='max')
    print(best_logdir)


if __name__ == "__main__":
    main()
