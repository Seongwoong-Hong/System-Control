import os
import numpy as np
from datetime import datetime
from functools import partial

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from algos.torch.sac import SAC, MlpPolicy
from common.util import make_env


def trial_str_creator(trial):
    trialname = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + trial.trial_id
    return trialname


def try_train(config):
    algo = SAC(config['policy'],
               env=config['env'],
               batch_size=config['batch_size'],
               learning_starts=config['batch_size'] * 4,
               learning_rate=config['lr'],
               train_freq=(config['accum_steps'], 'step'),
               gradient_steps=config['gradient_steps'],
               gamma=config['gamma'],
               ent_coef=config['ent_coef'],
               target_update_interval=1,
               device='cuda')

    for epoch in range(1000):
        """ Learning """
        algo.learn(total_timesteps=int(8.192e3), reset_num_timesteps=True)

        """ Testing """
        mean_reward = np.mean([ep_info["r"] for ep_info in algo.ep_info_buffer])

        """ Temporally save a model"""
        trial_dir = tune.get_trial_dir()
        path = os.path.join(trial_dir, "agent")
        algo.save(path)
        # tune.report()
        tune.report(mean_reward=mean_reward)


def main(target):
    env = make_env(f"{target}_custom-v2")
    config = {
        'policy': MlpPolicy,
        'env': env,
        'batch_size': tune.choice([64, 128, 256]),
        'accum_steps': tune.choice([512, 1024, 2048]),
        'gradient_steps': tune.choice([1000, 1500, 2000]),
        'gamma': tune.choice([0.99, 0.98, 0.97, 0.95]),
        'ent_coef': tune.choice([0.1, 0.15, 0.2, 0.25]),
        'lr': tune.loguniform(1e-4, 5e-4),
    }

    scheduler = ASHAScheduler(
        metric="mean_reward",
        mode="max",
        max_t=100,
        grace_period=10,
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
        trial_name_creator=trial_str_creator,
    )

    best_trial = result.get_best_trial("mean_reward", "max", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['mean_reward']:.4f}")

    best_logdir = result.get_best_logdir(metric='mean_reward', mode='max')
    print(best_logdir)


if __name__ == "__main__":
    main("IDP")
