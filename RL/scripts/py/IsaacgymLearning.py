import time

import hydra

import os

from datetime import datetime

import isaacgym
from isaacgymenvs.pbt.pbt import PbtAlgoObserver, initial_pbt_check
from hydra.utils import to_absolute_path
from isaacgymenvs.tasks import Ant, isaacgym_task_map, Cartpole
import gym
from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
from isaacgymenvs.utils.utils import set_np_formatting, set_seed

from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, MultiObserver, ComplexObsRLGPUEnv
from isaacgymenvs.utils.wandb_utils import WandbAlgoObserver
from matplotlib import pyplot as plt
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import AlgoObserver
from rl_games.torch_runner import Runner, _restore, _override_sigma
from rl_games.algos_torch import model_builder
from isaacgymenvs.learning import amp_continuous
from isaacgymenvs.learning import amp_players
from isaacgymenvs.learning import amp_models
from isaacgymenvs.learning import amp_network_builder
import isaacgymenvs

from omegaconf import DictConfig, OmegaConf

from algos.torch.rlgames.players import PpoPlayerCustom
from common.path_config import MAIN_DIR
from common.rlgames.observer import DrawTimeTrajObserver
from gym_envs.envs.isaacgym import *

isaacgym_task_map["IDPMinEffort"] = IDPMinEffort
isaacgym_task_map["IDPMinEffortDet"] = IDPMinEffortDet
isaacgym_task_map["CartpoleDet"] = Cartpole
isaacgym_task_map["AntDet"] = Ant
load_dir = MAIN_DIR / "RL" / "scripts" / "rlgames" / "cfg"
target_dir = MAIN_DIR / "RL" / "scripts" / "rlgames"


class PosturalControlObserver(DrawTimeTrajObserver):
    def after_init(self, algo):
        super().after_init(algo)
        self.save_mat = self.algo.player_config.get("save_mat", False)
        if self.save_mat:
            self.trial_idx = 0
        self.fig = plt.figure(figsize=[8.0, 10.0])
        for i in range(8):
            self.fig.add_subplot(4, 2, i + 1)

    def after_play(self):
        tspan = np.linspace(0, (self.obs.shape[0]-1)*self.algo.env.dt, self.obs.shape[0])
        for i in range(4):
            self.fig.axes[i].plot(tspan[:-1], np.rad2deg(-self.obs[1:, 0, i].numpy()), linewidth=2)
        for i in range(2):
            self.fig.axes[i + 4].plot(tspan[:-1], -self.acts[1:, 0, i] * self.algo.env.joint_gears[i].cpu(), linewidth=2)
        # self.fig.axes[-1].plot(tspan[:-1], self.infos['torque_rate'][1:, 0, 0], linewidth=2)
        # self.fig.axes[-2].plot(tspan[:-1], self.infos['torque_rate'][1:, 0, 1], linewidth=2)
        com = ((self.algo.env.mass[1] * self.algo.env.com[1] * torch.sin(self.obs[:, 0, 0]) +
               self.algo.env.mass[2] * (self.algo.env.len[1] * torch.sin(self.obs[:, 0, 0]) +
                                        self.algo.env.com[2] * torch.sin(self.obs[:, 0, :2].sum(dim=1)))
               ) / self.algo.env.mass[1:].sum()).numpy()
        cop = (self.infos['foot_forces'][:, 0, 4] + 0.08 * self.infos['foot_forces'][:, 0, 0]) / -self.infos['foot_forces'][:, 0, 2]
        self.fig.axes[-2].plot(tspan[:-1], cop[1:])
        self.fig.axes[-1].plot(tspan[:-1], com[1:])
        # self.fig.axes[-2].axhline(0.16, linestyle='dashed', color='k')
        # self.fig.axes[-2].plot(tspan[:-1], self.infos['ptb_forces'][1:, 0, 1, 0])
        # self.fig.axes[-1].plot(tspan[:-1], self.infos['ptb_forces'][1:, 0, 2, 0])

        high = [5, 10, 50, 100, 120, 85, 500, 500]
        low = [-10, -30, -50, -100, -50, -25, -500, -500]
        for i in range(8):
            if i < 6:
                self.fig.axes[i].set_ylim(low[i], high[i])
            self.fig.axes[i].axvline(1 / 3, linestyle='--', color='k', linewidth=0.75)
            self.fig.axes[i].axvline(1/3 + 0.1, linestyle='--', color='k', linewidth=0.75)
            self.fig.axes[i].axvline(2 / 3, linestyle='--', color='k', linewidth=0.75)

        if self.save_mat:
            self.trial_idx += 5
            # up_name = self.algo.config["full_experiment_name"][:self.algo.config["full_experiment_name"].find('/')]
            tg_dir = Path(f"../MATLAB/MinEffort/delay_passive_randomize/limLevel_0/sub10")
            tg_dir.mkdir(parents=True, exist_ok=True)
            humanData = io.loadmat(str(MAIN_DIR / "demos" / "IDP" / "sub10" / f"sub10i{self.trial_idx}.mat"))
            bsp = humanData['bsp']
            pltdd = self.algo.env._ptb_acc[0, :-1].cpu().numpy()
            m1 = 2 * sum(bsp[2:4, 0])
            m2 = bsp[6, 0]
            l1c = (bsp[2, 0] * bsp[2, 2] + bsp[3, 0] * (bsp[2, 1] + bsp[3, 2])) / sum(bsp[2:4, 0])
            l1 = sum(bsp[2:4, 1])
            l2c = bsp[6, 2]
            state = np.zeros([len(pltdd), 4])
            state[:] = np.nan
            state[:self.obs.shape[0] - 1] = -self.obs[1:, 0, :]
            torque = np.zeros([len(pltdd), 2])
            torque[:] = np.nan
            torque[:self.acts.shape[0] - 1] = (-self.acts[1:, 0, :] * self.algo.env.joint_gears.cpu()).numpy()
            pltq1 = (m1*l1c*np.cos(state[:, 0]) + m2*l1*np.cos(state[:, 0]) + m2*l2c*np.cos(state[:, 1]))*pltdd
            pltq2 = m2 * l2c * np.cos(state[:, 1]) * pltdd
            pltq = np.array([pltq1, pltq2]).T
            com = ((m1*l1c*np.sin(state[:, 0]) + m2*(l1*np.sin(state[:, 0]) + l2c*np.sin(state[:, :2].sum(axis=1))))
                   / (m1 + m2))
            cop = ((self.infos['foot_forces'][1:, 0, 4] + 0.08 * self.infos['foot_forces'][1:, 0, 0])
                   / -self.infos['foot_forces'][1:, 0, 2])

            data = {
                'bsp': humanData['bsp'],
                'pltdd': pltdd,
                'pltq': pltq,
                'state': state.astype(np.float64).reshape(-1, 4),
                'tq': torque,
                'com': com.astype(np.float64),
                'cop': cop.numpy().astype(np.float64),
            }
            io.savemat(str(tg_dir / f"sub10i{self.trial_idx}.mat"), data)

def preprocess_train_config(cfg, config_dict):
    """
    Adding common configuration parameters to the rl_games train config.
    An alternative to this is inferring them in task-specific .yaml files, but that requires repeating the same
    variable interpolations in each config.
    """

    train_cfg = config_dict['params']['config']

    train_cfg['device'] = cfg.rl_device

    train_cfg['population_based_training'] = cfg.pbt.enabled
    train_cfg['pbt_idx'] = cfg.pbt.policy_idx if cfg.pbt.enabled else None

    train_cfg['full_experiment_name'] = cfg.get('full_experiment_name')
    train_cfg['train_dir'] = str(target_dir / "runs")

    print(f'Using rl_device: {cfg.rl_device}')
    print(f'Using sim_device: {cfg.sim_device}')
    print(train_cfg)

    try:
        model_size_multiplier = config_dict['params']['network']['mlp']['model_size_multiplier']
        if model_size_multiplier != 1:
            units = config_dict['params']['network']['mlp']['units']
            for i, u in enumerate(units):
                units[i] = u * model_size_multiplier
            print(
                f'Modified MLP units by x{model_size_multiplier} to {config_dict["params"]["network"]["mlp"]["units"]}')
    except KeyError:
        pass

    return config_dict


@hydra.main(version_base="1.1", config_name="config", config_path=str(load_dir))
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
    cfg.full_experiment_name = cfg.experiment_name
    if "IDPMinEffort" in cfg.task.name:
        cfg_task_dict['env']['bsp_path'] = (MAIN_DIR / "demos" / cfg.task.env_type / cfg.task.subj / f"{cfg.task.subj}i1.mat")
        if cfg.task.env.ankle_limit == "satu":
            name_head = "limSatu"
        elif cfg.task.env.ankle_limit == "hard":
            name_head = "limHard"
        elif cfg.task.env.ankle_limit == "soft":
            name_head = f"limLevel{int(100*cfg.task.env.limLevel)}"
        else:
            raise ValueError(f"Unknown ankle limit {cfg.task.env.ankle_limit}")
        name_head += f"/atm{cfg.task.env.ankle_torque_max}_as{cfg.task.env.stiff_ank}"
        cfg.full_experiment_name = cfg.full_experiment_name + "/" + name_head
    cfg.full_experiment_name += f"/{run_name}"

    if cfg.test:
        cfg_task_dict['env']['numEnvs'] = 1
        cfg.task.name += "Det"
        observers = [PosturalControlObserver()]
        from common.rlgames.observer import MultiObserverPlayer
        multi_observer = MultiObserverPlayer(observers)
    else:
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
        multi_observer = MultiObserver(observers)

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
    runner = build_runner(multi_observer)
    rlg_config_dict['params']['config']['device_name'] = rlg_config_dict['params']['config']['device']
    runner.load(rlg_config_dict)
    runner.reset()

    # dump config dict
    if not cfg.test:
        experiment_dir = str(target_dir / 'runs' / cfg.full_experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
            f.write(OmegaConf.to_yaml(cfg))

    runner.run({
        'train': not cfg.test,
        'play': cfg.test,
        'checkpoint': cfg.checkpoint,
        'sigma': cfg.sigma if cfg.sigma != '' else None
    })


if __name__ == "__main__":
    launch_rlg_hydra()
