import os
import shutil
from datetime import datetime

import ray
from ray import tune, train

from RL.scripts.PolicyLearning import define_algo, TimeTrajCallback
from common.path_config import LOG_DIR

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ray.init(address='auto')
# ray.init()


def try_train(config):
    name_tail = f"_{config['env_id']}_ptb{config['stptb']}to{config['edptb']}"
    if config['env_kwargs']['delay']:
        name_tail += "/delay_passive"
        if 'stiffness_set' in config.keys():
            if config['stiffness_set'] == 1:
                config['env_kwargs']['stiffness'] = [300, 50]
                config['env_kwargs']['damping'] = [30, 20]
            elif config['stiffness_set'] == 2:
                config['env_kwargs']['stiffness'] = [300, 70]
                config['env_kwargs']['damping'] = [30, 30]
            elif config['stiffness_set'] == 3:
                config['env_kwargs']['stiffness'] = [300, 50]
                config['env_kwargs']['damping'] = [30, 10]
            elif config['stiffness_set'] == 4:
                config['env_kwargs']['stiffness'] = [300, 70]
                config['env_kwargs']['damping'] = [30, 20]
            elif config['stiffness_set'] == 5:
                config['env_kwargs']['stiffness'] = [300, 100]
                config['env_kwargs']['damping'] = [30, 10]
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
    # config["name_tail"] = name_tail + f"/stfset{config['stiffness_set']}"
    config["name_tail"] = name_tail + f"/tqcost{config['env_kwargs']['tqcost_ratio']}_vel{config['env_kwargs']['vel_ratio']}_atm{config['env_kwargs']['ankle_torque_max']}"
    algo = define_algo(input_argu_dict=config, **config)

    callback = TimeTrajCallback(trials=[1, 2, 11, 12, 21, 22, 26, 27, 33, 34], **config)
    for i in range(config['totaltimesteps']):
        algo.learn(total_timesteps=int(1e6), tb_log_name=f"tensorboard_log", reset_num_timesteps=False, callback=callback)
        algo.save(algo.tensorboard_log + f"/agent_{i + 1}")
        if config['use_norm']:
            algo.env.save(algo.tensorboard_log + f"/normalization_{i + 1}.pkl")


def tune_algo_parameter():
    crt_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    config = {
        "env_type": "IDP",
        "algo_type": "ppo",
        "env_id": "MinEffort",
        "device": "cuda",
        "subj": "sub10",
        "use_norm": "True",
        "isPseudo": "False",
        "log_dir": LOG_DIR / "erectCost"/ "constNotEarly1104",
        "tags": ["withDone"],
        "stiffness_set": 1,
        "env_kwargs": {
            'use_seg_ang': False,
            'ptb_act_time': 1/3,
            'ankle_limit': 'soft',
            'ankle_torque_max': 100,
            'delay': True,
            'delayed_time': 0.1,
            'const_ratio': 1.0,
            'tq_ratio': 0.5,
            'tqcost_ratio': 0.1,
            'ank_ratio': 0.5,
            'vel_ratio': 0.1,
            'limLevel': 0.5,
            'torque_rate_limit': False,
            'ptb_range': [0.0 + 0.015*i for i in range(12)],
        },
        "algo_kwargs": {
            "n_steps": 8192,
            "batch_size": 8192,
            "learning_rate": 1e-4,
            "n_epochs": 5,
            "gamma": 0.999,
            "gae_lambda": 0.99,
            "vf_coef": 0.5,
            "ent_coef": 3e-4,
            "policy_kwargs": {
                'net_arch': [{"pi": [128, 128], "vf": [128, 128]}],
                'log_std_range': [-10, None]
            },
        },
        "verbose": 0,
        "stptb": 1,
        "edptb": 7,
        "totaltimesteps": 300,
        "num_envs": 32,
    }

    train_model = tune.with_resources(try_train, resources={'gpu': 0.2, 'cpu': 1})
    tune_config = tune.TuneConfig(num_samples=2)

    reporter = tune.CLIReporter()
    run_config = train.RunConfig(
        progress_reporter=None,
        verbose=0,
        name="LearnMultiParameters",
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
    shutil.rmtree(f"/home/hsw/ray_results/{crt_time}")


if __name__ == "__main__":
    tune_algo_parameter()
