import json
from datetime import datetime
from pathlib import Path
from typing import List

from scipy import io
from stable_baselines3.common.utils import logger, configure_logger

from algos.torch.sb3.ppo import PPO, MlpPolicy
from common.path_config import MAIN_DIR
from common.sb3.util import make_env, str2bool


def define_algo(
        env_type,
        env_id,
        algo_type,
        subj,
        use_norm,
        isPseudo,
        name_tail,
        stptb,
        edptb,
        num_envs: int = 8,
        device='cpu',
        verbose=1,
        log_dir: Path = None,
        algo_kwargs: dict = None,
        env_kwargs: dict = None,
        except_trials: List = None,
        input_argu_dict: dict = None,
        tags: List = None,
        *args,
        **kwargs,
):
    env_id = f"{env_type}_{env_id}"
    isPseudo = str2bool(isPseudo)
    use_norm = str2bool(use_norm)
    if except_trials is None:
        except_trials = []
    if algo_kwargs is None:
        algo_kwargs = {}
    if log_dir is None:
        from common.path_config import LOG_DIR
        log_dir = LOG_DIR
    if tags is None:
        tags = []

    if isPseudo:
        env_type = "Pseudo" + env_type

    subpath = (MAIN_DIR / "demos" / env_type / subj / subj)
    if env_type == "IDPPD":
        subpath = (MAIN_DIR / "demos" / "IDP" / subj / subj)
    states = [None for _ in range(35)]
    for i in range(5*(stptb - 1) + 1, 5*edptb + 1):
        humanData = io.loadmat(str(subpath) + f"i{i}.mat")
        bsp = humanData['bsp']
        states[i - 1] = humanData['state']
    for trial in except_trials:
        states[trial - 1] = None

    ptb_range = [0.03, 0.045, 0.06, 0.075, 0.09, 0.12, 0.15]
    if 'ptb_range' not in env_kwargs:
        env_kwargs['ptb_range'] = ptb_range[stptb-1:edptb]
    elif len(env_kwargs['ptb_range']) == 0:
        env_kwargs['ptb_range'] = ptb_range[stptb-1:edptb]
    env = make_env(f"{env_id}-v2", num_envs=num_envs, bsp=bsp, humanStates=states, use_norm=use_norm, **env_kwargs)

    log_dir = log_dir / env_type / (algo_type + name_tail)
    n = 1
    while (log_dir / f"policies_{n}").is_dir():
        n += 1
    while True:
        try:
            (log_dir / f"policies_{n}").mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            while (log_dir / f"policies_{n}").is_dir():
                n += 1
        else:
            break
    log_dir = (log_dir / f"policies_{n}")
    if "log_dir" in input_argu_dict and isinstance(input_argu_dict['log_dir'], Path):
        input_argu_dict['log_dir'] = str(input_argu_dict['log_dir'])
    if isinstance(input_argu_dict, dict):
        config = json.dumps(input_argu_dict, indent=4)
        with open(str(log_dir / "config.json"), "w") as f:
            f.write(config)

    configure_logger(1, str(log_dir), "tensorboard_log", reset_num_timesteps=False)
    for key, value in input_argu_dict.items():
        logger.record("kwargs/" + key, str(value))
    for tag in tags:
        logger.record(tag, "on")
    logger.dump()

    algo = PPO(
        MlpPolicy,
        env=env,
        verbose=verbose,
        tensorboard_log=str(log_dir),
        device=device,
        **algo_kwargs,
    )

    return algo


if __name__ == "__main__":
    crt_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    # 환경 설정
    config_dict = {
        "env_type": "IDP",
        "algo_type": "ppo",
        "env_id": "MinEffort",
        "device": "cuda",
        "subj": "sub10",
        "use_norm": "True",
        "isPseudo": "False",
        "log_dir": MAIN_DIR / "RL" / "examples" / "runs" / "mujoco_example_result" / crt_time,
        "tags": [],
        'stiffness1': 300,
        'stiffness2': 100,
        'damping1': 30,
        'damping2': 30,
        "env_kwargs": {
            'ptb_range': [0.03 + 0.015 * i for i in range(10)],
            'use_seg_ang': False,
            'ptb_act_time': 1 / 3,
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
            "batch_size": 2048,
            "learning_rate": 1e-4,
            "n_epochs": 5,
            "gamma": 0.999,
            "gae_lambda": 0.995,
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
        "totaltimesteps": 60,  # millions
        "num_envs": 32,
    }

    if config_dict['env_kwargs']['ankle_limit'] == 'hard':
        name_tail = "/limHard"
    elif config_dict['env_kwargs']['ankle_limit'] == 'soft':
        name_tail = f"/limLevel_{round(config_dict['env_kwargs']['limLevel']*100)}"
    else:
        name_tail = "/limSatu"
    config_dict["name_tail"] = name_tail

    algo = define_algo(input_argu_dict=config_dict, **config_dict)

    for i in range(config_dict['totaltimesteps']):
        algo.learn(total_timesteps=int(1e6), tb_log_name=f"tensorboard_log", reset_num_timesteps=False)
        algo.save(algo.tensorboard_log + f"/agent_{i + 1}")
        if config_dict['use_norm']:
            algo.env.save(algo.tensorboard_log + f"/normalization_{i + 1}.pkl")
    print(f"Policy saved in {algo.tensorboard_log}")
