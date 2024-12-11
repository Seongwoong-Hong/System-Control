import argparse
import copy
import json
from pathlib import Path
from typing import List

from scipy import io
from stable_baselines3.common.logger import Figure
from stable_baselines3.common.utils import logger, configure_logger

from RL.analysis.sb3.drawTimeTraj import draw_time_trajs_with_humandata
from algos.torch.sb3.ppo import PPO, MlpPolicy
from common.analyzer import exec_policy
from common.sb3.callbacks import PolicyEvalCallback
from common.path_config import MAIN_DIR
from common.sb3.util import make_env, str2bool


class TimeTrajCallback(PolicyEvalCallback):
    def _on_training_end(self) -> None:
        eval_env = copy.deepcopy(self.eval_env)
        if self.use_norm:
            tmp_env = copy.deepcopy(self.model.env)
            tmp_env.venv = None
            tmp_env.set_venv(eval_env)
        else:
            tmp_env = eval_env
        obs, _, _, _, ifs = exec_policy(tmp_env, self.model, render=None, deterministic=True, repeat_num=len(self.trials), infos = ['torque'])
        tqs = ifs['torque']
        fig = draw_time_trajs_with_humandata(self.trials, obs, tqs, self.states, self.torques, showfig=False)
        rsq = 0.0
        for ob, tq, hob, htq in zip(obs, tqs, self.states, self.torques):
            ob = ob[:-1]
            rsq += 1 - ((2500 * (hob[:len(ob)] - ob[:, :4]) ** 2).sum() + ((htq[:len(tq)] - tq) ** 2).sum()) / ((2500 * hob[:len(ob)] ** 2).sum() + (htq[:len(tq)] ** 2).sum())
        rsq /= len(self.trials)
        self.model.rsq = rsq
        self.logger.record("train/rsq", rsq)
        self.logger.record("fig/time_trajs", Figure(fig, close=True), exclude='stdout')
        self.logger.dump(self.num_timesteps)


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, default="MinEffort")
    parser.add_argument('--env_type', type=str, default="IDP")
    parser.add_argument('--cost_ratio', type=int, default=1)
    parser.add_argument('--state_ratio', type=int, default=5)
    parser.add_argument('--stiffness', type=int, default=0)
    args = parser.parse_args()
    # 환경 설정
    config_dict = {
        "env_type": args.env_type,
        "env_id": args.env_id,
        "algo_type": "ppo",
        "device": "cpu",
        "subj": "sub04",
        "use_norm": "True",
        "isPseudo": "False",
        "env_kwargs": {
            'use_seg_ang': False,
            'ankle_max': 100,
            'soft_act': True,
            'cost_ratio': 0.1,
            'stiffness': 30,
        },
        "algo_kwargs": {
            "n_steps": 512,
            "batch_size": 2048,
            "learning_rate": 0.0003,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "vf_coef": 0.2,
            "ent_coef": 0.001,
            "policy_kwargs": {
                'net_arch': [dict(pi=[256, 256], vf=[256, 256])],
                'log_std_range': [-10, None]
            },
        },
        "stptb": 1,
        "edptb": 7,
        "except_trials": [13, 16],
        "totaltimesteps": 30,  # millions
    }

    #########################################################################
    # !!!!! You Need to Change this Name before you start the LEARNING!!!!! #
    #########################################################################
    if config_dict["env_type"] == "IDPPD":
        config_dict["env_kwargs"]["PDgain"] = [500, 100]
        torque_type = "PD500100"
        if config_dict["env_kwargs"]["soft_act"]:
            torque_type = "softTq" + torque_type
    elif config_dict["env_type"] == "IDP":
        torque_type = "Tq"
        if config_dict["env_kwargs"]["soft_act"]:
            torque_type = "soft" + torque_type
    else:
        raise Exception(f"{config_dict['env_type']}은 정의되지 않은 환경 타입입니다.")

    config_dict["env_kwargs"]["cost_ratio"] = 0.1*args.cost_ratio

    name_tail = f"_{config_dict['env_id']}_ptb{config_dict['stptb']}to{config_dict['edptb']}/{torque_type}_{args.cost_ratio}vs{10-args.cost_ratio}_hardLim"
    config_dict["env_kwargs"]["stiffness"] = args.stiffness
    if args.stiffness != 0:
        name_tail += "/passive_spring"

    config_dict["name_tail"] = name_tail
    algo = define_algo(input_argu_dict=config_dict, **config_dict)

    for i in range(config_dict['totaltimesteps']):
        algo.learn(total_timesteps=int(1e6), tb_log_name=f"tensorboard_log", reset_num_timesteps=False)
        algo.save(algo.tensorboard_log + f"/agent_{i + 1}")
        if config_dict['use_norm']:
            algo.env.save(algo.tensorboard_log + f"/normalization_{i + 1}.pkl")
    print(f"Policy saved in {algo.tensorboard_log}")