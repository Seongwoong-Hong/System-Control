import json
import shutil
from datetime import datetime
from typing import List

import numpy as np
from pathlib import Path
from scipy import io
from stable_baselines3.common.utils import configure_logger, logger

from RL.scripts.PolicyLearning import TimeTrajCallback
from algos.torch.ppo import PPO
from common.path_config import LOG_DIR, MAIN_DIR
from common.sb3.util import make_env, str2bool


def ptb_curriculum_learning(
        env_type,
        env_id,
        algo_type,
        subj,
        use_norm,
        isPseudo,
        stptb,
        edptb,
        policy_num,
        tmp_num,
        load_name_tail,
        num_envs: int = 8,
        log_dir_tail: str = None,
        device: str = 'cpu',
        log_dir: Path = LOG_DIR,
        env_kwargs: dict = None,
        except_trials: List = None,
        input_args: dict = None,
        algo_kwargs: dict = None,
        tags=None,
        *args,
        **kwargs,
):
    env_id = f"{env_type}_{env_id}"
    isPseudo = str2bool(isPseudo)
    use_norm = str2bool(use_norm)
    if "PDgain" in env_kwargs:
        env_kwargs["PDgain"] = np.array(env_kwargs["PDgain"])
    if except_trials is None:
        except_trials = []
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

    load_dir = (log_dir / env_type / (algo_type + load_name_tail))
    if use_norm:
        use_norm = str((load_dir / f"policies_{policy_num}" / f"normalization_{tmp_num}.pkl"))
    env = make_env(f"{env_id}-v2", num_envs=num_envs, bsp=bsp, humanStates=states, use_norm=use_norm, **env_kwargs)
    algo = PPO.load(str(load_dir / f"policies_{policy_num}" / f"agent_{tmp_num}"), device=device)

    log_dir = (load_dir / f"curriculum_ptb{stptb}to{edptb}")
    if log_dir_tail is not None:
        log_dir = (log_dir / log_dir_tail)

    n = 1
    while (log_dir / f"policies_{n}").is_dir():
        n += 1
    log_dir = (log_dir / f"policies_{n}")
    log_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy(str(Path(__file__)), str(log_dir))
    if "log_dir" in input_args and isinstance(input_args['log_dir'], Path):
        input_args['log_dir'] = str(input_args['log_dir'])

    if isinstance(input_args, dict):
        config = json.dumps(input_args, indent=4)
        with open(str(log_dir / "config.json"), "w") as f:
            f.write(config)

    configure_logger(1, str(log_dir), "tensorboard_log", reset_num_timesteps=False)
    for key, value in input_args.items():
        logger.record("kwargs/" + key, str(value))
    for tag in tags:
        logger.record(tag, "on")
    logger.dump()

    algo.init_kwargs['tensorboard_log'] = str(log_dir)
    algo.reset_except_policy_param(env, **algo_kwargs)
    return algo


if __name__ == "__main__":
    # 환경 설정
    algo_type = "ppo"
    env_type = "IDP"
    env_id = "MinEffort"
    log_dir = LOG_DIR / "longerLearning" / "tqConstEarly"
    name_tail = "_MinEffort_ptb1to7/delay_passive/limSatu/tm120_tq1.0"
    policy_num = 3
    tmp_num = "50"

    env_id = f"{env_type}_{env_id}"

    model_dir = log_dir / env_type / (algo_type + name_tail) / f"policies_{policy_num}"
    if not model_dir.is_dir():
        raise Exception("Model directory doesn't exist")

    with open(f"{str(model_dir)}/config.json", 'r') as f:
        config = json.load(f)

    curri_config = {
        "device": "cuda",
        "tags": [datetime.now().strftime('%Y-%m-%d-%H-%M-%S')],
        "policy_num": policy_num,
        "tmp_num": tmp_num,
        "log_dir": log_dir,
        "load_name_tail": name_tail,
        "stptb": 1,
        "edptb": 7,
        "totaltimesteps": 15,  # millions
        "num_envs": 8,
        "algo_kwargs":{
            "n_steps": 2048,
            "batch_size": 1024,
        }
    }

    for key, value in config.items():
        if key not in curri_config:
            curri_config[key] = value

    curri_config['env_kwargs']['ankle_limit'] = 'satu'
    curri_config['env_kwargs']['limLevel'] = 0.0

    #########################################################################
    # !!!!! You Need to Change this Name before you start the LEARNING!!!!! #
    #########################################################################
    algo = ptb_curriculum_learning(input_args=curri_config, **curri_config)

    callback = TimeTrajCallback([1, 2, 11, 12, 21, 22, 33, 34], **curri_config)
    for i in range(curri_config["totaltimesteps"]):
        algo.learn(total_timesteps=int(1e6), tb_log_name="tensorboard_log", reset_num_timesteps=False, callback=callback)
        algo.save(algo.tensorboard_log + f"/agent_{i + 1}")
        if curri_config["use_norm"]:
            algo.env.save(algo.tensorboard_log + f"/normalization_{i + 1}.pkl")
