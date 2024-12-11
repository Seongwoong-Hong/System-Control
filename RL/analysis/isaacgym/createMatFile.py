import time

import hydra

import os

from datetime import datetime

import isaacgym
from isaacgym import gymtorch
from isaacgymenvs.pbt.pbt import PbtAlgoObserver, initial_pbt_check
from isaacgymenvs.utils.rlgames_utils import multi_gpu_get_rank
from hydra.utils import to_absolute_path
from isaacgymenvs.tasks import Ant, isaacgym_task_map
import gym
from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
from isaacgymenvs.utils.utils import set_np_formatting, set_seed

from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, MultiObserver, ComplexObsRLGPUEnv
from isaacgymenvs.utils.wandb_utils import WandbAlgoObserver
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import AlgoObserver
from rl_games.torch_runner import Runner
from rl_games.algos_torch import model_builder
from rl_games.algos_torch.players import PpoPlayerContinuous
from isaacgymenvs.learning import amp_continuous
from isaacgymenvs.learning import amp_players
from isaacgymenvs.learning import amp_models
from isaacgymenvs.learning import amp_network_builder
import isaacgymenvs

from omegaconf import DictConfig, OmegaConf

import torch as th
import matplotlib.pyplot as plt
import matplotlib
from common.path_config import MAIN_DIR
from gym_envs.envs.isaacgym import *

isaacgym_task_map["IDPMinEffort"] = IDPMinEffort
isaacgym_task_map["IDPMinEffortDet"] = IDPMinEffortDet
isaacgym_task_map["IDPMinEffortLean"] = IDPMinEffortLean
isaacgym_task_map["IDPMinEffortLeanDet"] = IDPMinEffortLeanDet
target_dir = MAIN_DIR / "RL" / "scripts" / "rlgames"
play_games = 6
# matplotlib.use('module://backend_interagg')

from pathlib import Path

import numpy as np
from scipy import io

from common.analyzer import exec_policy
from common.path_config import LOG_DIR, MAIN_DIR
from common.sb3.util import load_result


@hydra.main(version_base="1.1", config_name="config", config_path=str(target_dir / "cfg"))
def launch_rlg_hydra(cfg: DictConfig):

    if cfg.pbt.enabled:
        initial_pbt_check(cfg)

    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{cfg.wandb_name}_{time_str}"

    if not (cfg.load_checkpoint or cfg.test):
        cfg.checkpoint = ''

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(str(target_dir / cfg.checkpoint))

    # set numpy formatting for printing only
    set_np_formatting()

    # global rank of the GPU
    global_rank = int(os.getenv("RANK", "0"))

    # sets seed. if seed is -1 will pick a random one
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=global_rank)
    cfg_task_dict = omegaconf_to_dict(cfg.task)
    if "IDPMinEffort" in cfg.task.name:
        cfg_task_dict['env']['bsp_path'] = (MAIN_DIR / "demos" / cfg.task.env_type / cfg.task.subj / f"{cfg.task.subj}i1.mat")

    if cfg.test:
        cfg_task_dict['env']['numEnvs'] = 1
        cfg.task.name += "Det"

    def create_isaacgym_env(**kwargs):
        envs = isaacgym_task_map[cfg.task.name](
            cfg=cfg_task_dict,
            rl_device=cfg.rl_device,
            sim_device=cfg.sim_device,
            graphics_device_id=cfg.graphics_device_id,
            headless=cfg.headless,
            virtual_screen_capture=cfg.capture_video,
            force_render=cfg.force_render,
        )
        if cfg.capture_video:
            envs.is_vector_env = True
            envs = gym.wrappers.RecordVideo(
                envs,
                str(target_dir / f"videos/{run_name}"),
                step_trigger=lambda step: step % cfg.capture_video_freq == 0,
                video_length=cfg.capture_video_len,
            )
        return envs

    env_configurations.register('rlgpu', {
        'vecenv_type': 'RLGPU',
        'env_creator': lambda **kwargs: create_isaacgym_env(**kwargs),
    })

    vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))

    rlg_config_dict = omegaconf_to_dict(cfg.train)
    rlg_config_dict = preprocess_train_config(cfg, rlg_config_dict)

    observers = [RLGPUAlgoObserver()]

    if cfg.pbt.enabled:
        pbt_observer = PbtAlgoObserver(cfg)
        observers.append(pbt_observer)

    if cfg.wandb_activate:
        cfg.seed += global_rank
        if global_rank == 0:
            # initialize wandb only once per multi-gpu run
            wandb_observer = WandbAlgoObserver(cfg)
            observers.append(wandb_observer)

    # register new AMP network builder and agent
    def build_runner(algo_observer):
        runner = Runner(algo_observer)
        runner.algo_factory.register_builder('amp_continuous', lambda **kwargs: amp_continuous.AMPAgent(**kwargs))
        runner.player_factory.register_builder('amp_continuous',
                                               lambda **kwargs: amp_players.AMPPlayerContinuous(**kwargs))
        runner.player_factory.register_builder('a2c_continuous', lambda **kwargs : PpoPlayerCustom(**kwargs))
        model_builder.register_model('continuous_amp', lambda network, **kwargs: amp_models.ModelAMPContinuous(network))
        model_builder.register_network('amp', lambda **kwargs: amp_network_builder.AMPBuilder())

        return runner

    # convert CLI arguments into dictionary
    # create runner and set the settings
    runner = build_runner(MultiObserver(observers))
    rlg_config_dict['params']['config']['device_name'] = rlg_config_dict['params']['config']['device']
    runner.load(rlg_config_dict)
    runner.reset()

    # dump config dict
    if not cfg.test:
        experiment_dir = os.path.join(str(target_dir / 'runs'), cfg.train.params.config.name +
                                      '_{date:%d-%H-%M-%S}'.format(date=datetime.now()))

        os.makedirs(experiment_dir, exist_ok=True)
        with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
            f.write(OmegaConf.to_yaml(cfg))

    runner.run({
        'train': not cfg.test,
        'play': cfg.test,
        'checkpoint': cfg.checkpoint,
        'sigma': cfg.sigma if cfg.sigma != '' else None
    })

    target = "delay_passive/limLevel_50"
    load_kwargs = {
        "algo_type": "ppo",
        "env_type": "IDP",
        "env_id": "MinEffort",
        "trials": list(range(1, 36)),
        "policy_num": 3,
        "tmp_num": 80,
        "log_dir": LOG_DIR / "erectCost" / "constNotEarly0823",
        "name_tail": f"_MinEffort_ptb1to7/{target}/const1.0_vel0.1_atm100",
        "env_kwargs": {"ankle_torque_max": 120},
    }
    save_video = None

    env, agent, loaded_result = load_result(**load_kwargs)
    bsp = loaded_result['bsp']
    obs, _, _, _, ifs = exec_policy(env, agent, render=None, deterministic=True, repeat_num=len(load_kwargs['trials']))
    tqs = ifs['torque']

    bsps = []
    pltdds = [[] for _ in range(35)]
    pltqs = [[] for _ in range(35)]
    for trial in load_kwargs['trials']:
        humanData = io.loadmat(str(MAIN_DIR / "demos" / "IDP" / "sub10" / f"sub10i{trial}.mat"))
        bsps += [bsp]
        pltdd = env.get_attr("ptb_acc", 0)[0][:360]
        pltdds[env.get_attr("ptb_idx", 0)[0]] = pltdd
        m1 = 2 * sum(bsp[2:4, 0])
        m2 = bsp[6, 0]
        l1c = (bsp[2, 0] * bsp[2, 2] + bsp[3, 0] * (bsp[2, 1] + bsp[3, 2])) / sum(bsp[2:4, 0])
        l1 = sum(bsp[2:4, 1])
        l2c = bsp[6, 2]
        pltq1 = (m1*l1c*np.cos(obs[trial-1][:-1, 0]) + m2*l1*np.cos(obs[trial-1][:-1, 0]) + m2*l2c*np.cos(obs[trial-1][:-1, 1]))*pltdd
        pltq2 = m2*l2c*np.cos(obs[trial-1][:-1, 1])*pltdd
        pltqs[env.get_attr("ptb_idx", 0)[0]] = np.array([pltq1, pltq2]).T
        env.reset()

    tg_dir = Path(f"MATLAB/{load_kwargs['env_id']}/{target}_lean3/sub10")
    tg_dir.mkdir(parents=True, exist_ok=True)
    for i, (ob, tq, bsp, pltdd, pltq) in enumerate(zip(obs, tqs, bsps, pltdds, pltqs)):
        data = {
            'bsp': bsp,
            'pltdd': pltdd,
            'pltq': pltq,
            'state': ob[:-1, :],
            'tq': tq,
        }
        io.savemat(str(tg_dir / f"sub10i{i+1}.mat"), data)


if __name__ == "__main__":
    launch_rlg_hydra()