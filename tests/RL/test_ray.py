import copy

import ray
from ray import tune
from RL.scripts.ParameterTuning import main
from algos.torch.ppo import MlpPolicy


def test_main():
    ray.init()
    scheduler = tune.schedulers.ASHAScheduler(
        grace_period=1,
        max_t=1,
    )
    config = {
        'total_steps': 1e3,
        'policy': MlpPolicy,
        'n_steps': 2**10,
        'batch_size': 256,
        'n_epochs': 10,
        'lr': 3e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'ent_coef': 1e-3,
        'tune_config': {
            'metric': "mean_reward",
            'mode': "max",
            'scheduler': scheduler,
            'num_samples': 1,
        }
    }
    main("IP", config)


