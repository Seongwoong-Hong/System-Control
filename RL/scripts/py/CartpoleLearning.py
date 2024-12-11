import json
from datetime import datetime

from common.path_config import LOG_DIR
from algos.torch.sb3.ppo import PPO, MlpPolicy
from common.sb3.util import make_env

if __name__ == "__main__":
    log_dir = LOG_DIR / ("Cartpole" + datetime.now().strftime("_%d-%H-%M-%S"))
    log_dir.mkdir(exist_ok=True)
    config_dict = {
        "use_norm": True,
        "algo_kwargs": {
            "n_steps": 128,
            "batch_size": 8192,
            "learning_rate": 3e-4,
            "n_epochs": 8,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "vf_coef": 1,
            "ent_coef": 0.0,
            "policy_kwargs": {
                'net_arch': [{"pi": [32, 32], "vf": [32, 32]}],
                'log_std_range': [-10, None]
            },
        }
    }

    config = json.dumps(config_dict, indent=4)
    with open(str(log_dir / "config.json"), "w") as f:
        f.write(config)

    env = make_env("Cartpole-v2", num_envs=64, use_norm=True)

    algo = PPO(
        MlpPolicy,
        env=env,
        verbose=1,
        tensorboard_log=str(log_dir),
        device="cuda",
        **config_dict['algo_kwargs'],
    )

    for i in range(100):
        algo.learn(total_timesteps=int(8192), tb_log_name=f"tensorboard_log", reset_num_timesteps=False)
        algo.save(algo.tensorboard_log + f"/agent_{i + 1}")
        if config_dict['use_norm']:
            algo.env.save(algo.tensorboard_log + f"/normalization_{i + 1}.pkl")
        print(f"{i+1}th epoch")
    print(f"Policy saved in {algo.tensorboard_log}")
