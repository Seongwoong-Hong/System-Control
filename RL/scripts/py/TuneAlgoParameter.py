import os
import shutil
import time
from datetime import datetime

import numpy as np
import ray
from ray import tune, train

from RL.scripts.PolicyLearning import define_algo, TimeTrajCallback
from common.path_config import LOG_DIR

# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
ray.init(address='auto')


def custom_trial_name(trial):
    """트라이얼 ID만을 사용하여 폴더 이름을 생성합니다."""
    return f"trial_{trial.trial_id}"


def try_train(config):
    name_tail = f"_{config['env_id']}_ptb{config['stptb']}to{config['edptb']}"
    if config['env_kwargs']['delay']:
        name_tail += "/delay_passive"
        if 'stiffness_set' in config.keys():
            if config['stiffness_set'] == 1:
                config['env_kwargs']['stiffness'] = [300, 30]
                config['env_kwargs']['damping'] = [20, 10]
            elif config['stiffness_set'] == 2:
                config['env_kwargs']['stiffness'] = [300, 100]
                config['env_kwargs']['damping'] = [30, 30]
            elif config['stiffness_set'] == 3:
                config['env_kwargs']['stiffness'] = [100, 50]
                config['env_kwargs']['damping'] = [20, 20]
            else:
                raise Exception("Stiffness set must be 1 or 2 or 3")
        else:
            config['env_kwargs']['stiffness'] = [config['stiffness1'], config['stiffness2']]
            config['env_kwargs']['damping'] = [config['damping1'], config['damping2']]
    else:
        name_tail += "/no_delay"
    if config['env_kwargs']['ankle_limit'] == 'hard':
        name_tail += "/limHard"
    elif config['env_kwargs']['ankle_limit'] == 'soft':
        name_tail += f"/limLevel_{round(config['env_kwargs']['limLevel']*100)}"
    else:
        name_tail += "/limSatu"
    config["name_tail"] = name_tail
    algo = define_algo(input_argu_dict=config, **config)

    callback = TimeTrajCallback(trials=[1, 2, 11, 12, 21, 22, 33, 34], **config)
    for i in range(config['totaltimesteps']):
        algo.learn(total_timesteps=int(1e6), tb_log_name=f"tensorboard_log", reset_num_timesteps=False, callback=callback)
        mean_reward = np.mean([ep_info["r"] for ep_info in algo.ep_info_buffer])
        mean_epi_len = np.mean([ep_info["l"] for ep_info in algo.ep_info_buffer])
        mean_rsq = algo.rsq
        train.report({'mean_reward': mean_reward, 'mean_rsq': mean_rsq, 'mean_epi_len': mean_epi_len})


def tune_algo_parameter():
    crt_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    config = {
        "env_type": "IDP",
        "algo_type": "ppo",
        "env_id": "HeadTrack",
        "device": "cuda",
        "subj": "sub10",
        "use_norm": "True",
        "isPseudo": "False",
        "log_dir": LOG_DIR / "algo_tune" / crt_time,
        "tags": [],
        'stiffness1': 300,
        'stiffness2': 100,
        'damping1': 30,
        'damping2': 30,
        "env_kwargs": {
            # 'ptb_range': [0.03 + 0.015 * i for i in range(10)],
            'use_seg_ang': False,
            'ptb_act_time': 1/3,
            'ankle_limit': 'satu',
            'ankle_torque_max': 120,
            'delay': True,
            'delayed_time': 0.1,
            'const_ratio': 1.0,
            'tq_ratio': 0.5,
            'tqcost_ratio': 1.0,
            'ank_ratio': 0.5,
            'vel_ratio': 0.01,
            'limLevel': 0.1,  # 0(soft) ~ 1(hard)
            'torque_rate_limit': False,
        },
        "algo_kwargs": {
            "n_steps": 1024,
            "batch_size": tune.choice([512, 1024, 2048, 4096,]),
            "learning_rate": tune.choice([1e-3, 7e-4, 3e-4, 1e-4]),
            "n_epochs": 5,
            "gamma": tune.choice([0.999, 0.995, 0.99, 0.975]),
            "gae_lambda": tune.choice([0.995, 0.99, 0.975, 0.95, 0.9]),
            "vf_coef": 0.5,
            "ent_coef": tune.choice([1e-4, 7e-5, 3e-4]),
            "policy_kwargs": {
                'net_arch': [{"pi": [128, 128], "vf": [128, 128]}],
                'log_std_range': [-10, None]
            },
        },
        "verbose": 0,
        "stptb": 1,
        "edptb": 7,
        "totaltimesteps": 60,  # millions
        "num_envs": 32,
    }

    metric = "mean_reward"

    scheduler = tune.schedulers.ASHAScheduler(
        metric=metric,
        mode="max",
        max_t=config['totaltimesteps'],
        grace_period=25,
        reduction_factor=2,
    )

    train_model = tune.with_resources(try_train, {"gpu": 0.1})
    tune_config = tune.TuneConfig(
        scheduler=scheduler,
        num_samples=500,
        trial_dirname_creator=custom_trial_name,
    )

    reporter = tune.CLIReporter(metric_columns=["mean_reward", "mean_rsq", "mean_epi_len", "training_iteration"])
    run_config = train.RunConfig(
        progress_reporter=reporter,
        name="TuneAlgoParameter",
        storage_path=f"/home/hsw/ray_results/{crt_time}",
        local_dir=f"/home/hsw/ray_results/{crt_time}",
    )

    tuner = tune.Tuner(
        train_model,
        param_space=config,
        tune_config=tune_config,
        run_config=run_config,
    )

    results = tuner.fit()

    best_trial = results.get_best_result(metric, "max", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial metrics: {best_trial.metrics}")

    shutil.rmtree(f"/home/hsw/ray_results/{crt_time}/TuneAlgoParameter")
    df = results.get_dataframe()
    df.to_csv(str(config['log_dir'] / "all_trial_results.csv"))


if __name__ == "__main__":
    tune_algo_parameter()
