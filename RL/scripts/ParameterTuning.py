import ray
import numpy as np
from datetime import datetime
from pathlib import Path
from scipy import io
from ray import tune, train

from algos.torch.ppo import PPO, MlpPolicy
from common.util import make_env
from common.wrappers import ActionWrapper


def try_train(config):
    algo = PPO(config['policy'],
               env=config['env'],
               n_steps=config['n_steps'],
               batch_size=config['batch_size'],
               n_epochs=config['n_epochs'],
               learning_rate=config['lr'],
               gamma=config['gamma'],
               gae_lambda=config['gae_lambda'],
               ent_coef=config['ent_coef'],
               policy_kwargs={'net_arch': [dict(pi=[16, 16], vf=[32, 32])]},
               device='cpu')

    """ Learning """
    algo.learn(total_timesteps=int(config['total_steps']), reset_num_timesteps=True)

    """ Testing """
    mean_reward = np.mean([ep_info["r"] for ep_info in algo.ep_info_buffer])

    return {'mean_reward': mean_reward}


def main(env_id, config):
    proj_path = Path(__file__).parent.parent.parent
    if env_id == "HPC":
        subpath = (proj_path / "demos" / "HPC" / "sub01" / "sub01")
        env = make_env(f"{env_id}_custom-v2", wrapper=ActionWrapper, subpath=str(subpath))
    elif env_id == "IP":
        subpath = (proj_path / "demos" / "IP" / "sub04" / "sub04")
        states = [None for _ in range(35)]
        for i in range(6, 11):
            humanData = io.loadmat(str(subpath) + f"i{i}.mat")
            bsp = humanData['bsp']
            states[i - 1] = humanData['state']
        env = make_env(f"{env_id}_custom-v2", bsp=bsp, humanStates=states, use_norm=True)
    else:
        env = make_env(f"{env_id}_custom-v2")

    config['env'] = env

    train_model = tune.with_resources(try_train, {"cpu": 1})
    tune_config = tune.TuneConfig(**config['tune_config'])

    reporter = tune.CLIReporter(metric_columns=["mean_reward", "training_iteration"])
    storage_path = (Path(__file__).parent / "tmp" / "log" / (f"ray_{env_id}" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
    storage_path.mkdir(parents=True, exist_ok=False)
    run_config = train.RunConfig(progress_reporter=reporter, storage_path=str(storage_path), name="tuning")

    tuner = tune.Tuner(
        train_model,
        param_space=config,
        tune_config=tune_config,
        run_config=run_config,
    )

    results = tuner.fit()

    best_trial = results.get_best_result("mean_reward", "max", "last")
    print(f"Best trial config: {best_trial.config}")


if __name__ == "__main__":
    scheduler = tune.schedulers.ASHAScheduler(
        max_t=100,
        grace_period=1,
    )
    config = {
        'total_steps': 5e6,
        'policy': MlpPolicy,
        'n_steps': tune.choice([2**7, 2**9, 2**11, 2**13, 2**15]),
        'batch_size': tune.choice([256, 512, 1024]),
        'n_epochs': tune.choice([10, 20]),
        'lr': tune.choice([3e-4, 1e-3]),
        'gamma': tune.choice([0.99, 0.995]),
        'gae_lambda': 0.95,
        'ent_coef': tune.choice([1e-3, 1e-4, 1e-2]),
        'tune_config': {
            'metric': "mean_reward",
            'mode': "max",
            'scheduler': scheduler,
            'num_samples': 200,
        }
    }
    ray.init(address='auto')
    main("IP", config)
