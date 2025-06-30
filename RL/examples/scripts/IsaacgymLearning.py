import hydra

from datetime import datetime

import isaacgym
from isaacgymenvs.pbt.pbt import PbtAlgoObserver, initial_pbt_check
from hydra.utils import to_absolute_path
import gym
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from isaacgymenvs.utils.utils import set_np_formatting, set_seed

from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, MultiObserver
from isaacgymenvs.utils.wandb_utils import WandbAlgoObserver
from matplotlib import pyplot as plt
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import AlgoObserver
from rl_games.torch_runner import Runner
from rl_games.algos_torch import model_builder
from isaacgymenvs.learning import amp_continuous
from isaacgymenvs.learning import amp_players
from isaacgymenvs.learning import amp_models
from isaacgymenvs.learning import amp_network_builder

from omegaconf import DictConfig, OmegaConf

from algos.torch.rlgames.players import PpoPlayerCustom
from common.path_config import MAIN_DIR
from common.rlgames.observer import DrawTimeTrajObserver
from gym_envs.envs.isaacgym import *

load_dir = MAIN_DIR / "RL" / "examples" / "scripts" / "cfg"
target_dir = MAIN_DIR / "RL" / "examples" / "scripts"

class CurriculumUpdator(AlgoObserver):
    def after_init(self, algo):
        self.algo = algo
        self.env = algo.vec_env.env
        self.param = self.env.cfg['env']['curriculum']
        self.update_freq = 10
        self.ac_reduced_rate = np.exp(np.log(self.param['min_avg_coeff'] / self.env.avg_coeff) / (self.param['end_epoch'] / self.update_freq))
        self.tqr_reduced_rate = np.exp(np.log(self.param['min_tqr_limit'] / self.env.tqr_limit.item()) / (self.param['end_epoch'] / self.update_freq))
        self.la_increment = np.deg2rad(self.param['max_lean_angle'] / (self.param['end_epoch'] / self.update_freq))

    def after_print_stats(self, frame, epoch_num, total_time):
        if epoch_num % self.update_freq == 0 and self.env.use_curriculum:
            crr_params = {}
            if self.algo.game_lengths.get_mean() > 0.8*self.env.max_episode_length.cpu().item():
                self.env.cfg['env']['edptb'] = min(self.param['max_edptb'], self.env.cfg['env']['edptb'] + 0.03)
                ptb_range = np.arange(0, round((self.env.cfg['env']['edptb'] - self.env.cfg['env']['stptb']) / self.env.cfg['env']['ptb_step']) + 1) * self.env.cfg['env']['ptb_step'] + self.env.cfg['env']['stptb']
                if self.env.cfg['env']['edptb'] >= self.param["max_edptb"]:
                    lean_angle = self.env.lean_angle + np.deg2rad(0.5)
                else:
                    lean_angle = self.env.lean_angle
                crr_params["lean_angle"] = min(np.deg2rad(self.param['max_lean_angle']), lean_angle)
                crr_params["_ptb_range"] = ptb_range
            crr_params["tqr_limit"] = max(self.param['min_tqr_limit'], self.env.tqr_limit * self.tqr_reduced_rate)
            crr_params["avg_coeff"] = max(self.param['min_avg_coeff'], self.env.avg_coeff * self.ac_reduced_rate)

            self.env.update_curriculum(**crr_params)

class RunnerTrajectoryObserver(AlgoObserver):
    def __init__(self, num_trajs=10):
        super().__init__()
        self.num_trajs = num_trajs

    def after_init(self, algo):
        self.algo = algo
        self.fig = plt.figure(figsize=[10.0, 8.0])
        for i in range(8):
            self.fig.add_subplot(4, 2, i + 1)

    def after_print_stats(self, frame, epoch_num, total_time):
        if (self.algo.save_freq > 0) and ((epoch_num + 1) % self.algo.save_freq == 0):
            done_traj_idx = torch.where(self.algo.vec_env.env.reset_buf)[0]
            num_sample = min(self.num_trajs, len(done_traj_idx))
            traj_idx = random.sample(done_traj_idx.tolist(), num_sample)

            obs_traj = self.algo.vec_env.env.obs_traj[traj_idx]
            act_traj = self.algo.vec_env.env.act_traj[traj_idx]
            tqr_traj = self.algo.vec_env.env.tqr_traj[traj_idx]
            ptb_acc = self.algo.vec_env.env._ptb[traj_idx]
            tspan = np.linspace(0, (obs_traj.shape[1] - 1) * self.algo.vec_env.env.dt, obs_traj.shape[1])
            for i in range(8):
                self.fig.axes[i].cla()
            for i in range(4):
                self.fig.axes[i].plot(tspan, np.rad2deg(-obs_traj[..., i].T.cpu().numpy()), linewidth=2)
            for i in range(2):
                self.fig.axes[i + 4].plot(tspan, (-act_traj[..., i].T * self.algo.vec_env.env.joint_gears[i]).cpu(), linewidth=2)
            for i in range(2):
                self.fig.axes[i + 6].plot(tspan, tqr_traj[..., i].T.cpu(), linewidth=2)
            high = [5, 10, 50, 100, 120, 85, 500, 500]
            low = [-10, -30, -50, -100, -50, -25, -300, -300]
            for i in range(8):
                if i < 6:
                    self.fig.axes[i].set_ylim(low[i], high[i])
                self.fig.axes[i].set_xlim(0, 3)
            self.fig.tight_layout()
            self.algo.writer.add_figure('performance/trajectories', self.fig, frame)


class PosturalControlObserver(DrawTimeTrajObserver):
    def __init__(self, fig_path=None, mat_path=None):
        super().__init__()
        if fig_path is not None:
            self.fig_path = fig_path
        if mat_path is not None:
            self.mat_path = mat_path

    def after_init(self, algo):
        super().after_init(algo)
        self.drawn_plt_num = 0
        self.save_mat = self.algo.player_config.get("save_mat", False)
        if self.save_mat:
            self.trial_idx = 0
        self.fig = plt.figure(figsize=[14.0, 10.0])
        for i in range(12):
            self.fig.add_subplot(4, 3, i + 1)
        if self.save_fig and self.fig_path is None:
            (target_dir / "fig").parent.mkdir(exist_ok=True)
            fig_idx = 0
            self.fig_path = (target_dir / "fig") / f"result{fig_idx}.png"
            while self.fig_path.exists():
                fig_idx += 1
                self.fig_path = (target_dir / "fig") / f"result{fig_idx}.png"

    def after_play(self):
        tspan = np.linspace(0, self.obs.shape[0], self.obs.shape[0]) * self.algo.env.dt
        n_plots = min(self.algo.env.num_envs, self.algo.games_num)

        # 마스크를 생성하여 end_idx에 맞는 데이터를 추출
        time_indices = torch.arange(self.obs.shape[0]).unsqueeze(1)  # 시간 인덱스 텐서 (T, 1)
        mask = (time_indices <= self.algo.end_idx.unsqueeze(0))[:, :n_plots].to(self.algo.env.device).unsqueeze(-1)  # (T, n_plots) 형태의 마스크 생성

        masked_obs = torch.where(mask, self.obs[:, :n_plots, :], torch.tensor(float('nan'), device=self.algo.env.device))
        masked_acts = torch.where(mask, self.acts[:, :n_plots, :], torch.tensor(float('nan'), device=self.algo.env.device))
        masked_tqr = torch.where(mask, self.infos['torque_rate'][:, :n_plots, :], torch.tensor(float('nan'), device=self.algo.env.device))
        masked_ffc = torch.where(mask, self.infos['foot_forces'][:, :n_plots, :], torch.tensor(float('nan'), device=self.algo.env.device))

        state_cost = torch.nansum(masked_obs ** 2, dim=0)
        action_cost = torch.nansum(masked_acts ** 2, dim=0)
        tqrate_cost = torch.nansum(torch.max((masked_tqr / 3) ** 2 - 1, torch.tensor(0.0)), dim=0)

        vals = torch.concat([
            torch.rad2deg(-masked_obs),
            -masked_acts*self.algo.env.joint_gears.reshape(1, 1, -1),
            -masked_tqr*self.algo.env.joint_gears.reshape(1, 1, -1),
            ], dim=-1).cpu().numpy()
        rews = torch.concat([state_cost, action_cost, tqrate_cost], dim=-1).cpu().numpy()
        pert_x = np.arange(self.drawn_plt_num, self.drawn_plt_num + n_plots)
        bar_width = 0.35
        bar_colors = ['b', 'r']
        traj_colors = np.tile(0.1*np.linspace(7, 1, 7)[:, None] * np.ones([7, 3]), (5, 1))
        for pi in range(self.drawn_plt_num, n_plots):
            for i in range(vals.shape[-1]):
                self.fig.axes[3*(i // 2) + i % 2].plot(tspan, vals[:, pi, i], linewidth=1.5, color=traj_colors[pi])
                self.fig.axes[3*(i // 2) + 2].bar(pert_x + ((i % 2) - 1/2)*bar_width, rews[:, i], bar_width, color=bar_colors[i % 2])
        high = [7.5, 10, 15,
                100, 100, 150,
                120, 150, 45,
                600, 600, 1.2]
        low = [-15, -50, 0,
               -50, -200, 0,
               -50, -25, 0,
               -400, -400, 0]
        not_adjust_lim = [11]
        for i in range(12):
            if not i in not_adjust_lim:
                self.fig.axes[i].set_ylim(low[i], high[i])
            if i % 3 != 2:
                self.fig.axes[i].set_xlim(0, 3)
                self.fig.axes[i].axvline(0.33333, linestyle='--', color='k', linewidth=0.75)
                self.fig.axes[i].axvline(0.60833, linestyle='--', color='k', linewidth=0.75)

        if self.save_mat:
            if self.mat_path is None:
                self.mat_path = MAIN_DIR / "RL" / "analysis" / "MATLAB"
            for i in range(self.drawn_plt_num, n_plots):
                self.trial_idx = i//7 + 1 + 5*(i%7)
                humanData = io.loadmat(str(MAIN_DIR / "demos" / "IDP" / "sub10" / f"sub10i{self.trial_idx}.mat"))
                bsp = humanData['bsp']
                pltdd = self.algo.env._ptb[0, :-1].cpu().numpy()
                m1 = 2 * sum(bsp[2:4, 0])
                m2 = bsp[6, 0]
                l1c = (bsp[2, 0] * bsp[2, 2] + bsp[3, 0] * (bsp[2, 1] + bsp[3, 2])) / sum(bsp[2:4, 0])
                l1 = sum(bsp[2:4, 1])
                l2c = bsp[6, 2]
                state = np.ones([360, 4]) * np.nan
                torque = np.ones([360, 2]) * np.nan
                len_t = min(360, masked_obs.shape[0])
                state[:len_t, :] = -masked_obs[:len_t, i, :].cpu().numpy()
                torque[:len_t, :] = -(masked_acts[:len_t, i, :] * self.algo.env.joint_gears).cpu().numpy()
                pltq1 = (m1*l1c*np.cos(state[:, 0]) + m2*l1*np.cos(state[:, 0]) + m2*l2c*np.cos(state[:, 1]))*pltdd
                pltq2 = m2 * l2c * np.cos(state[:, 1]) * pltdd
                pltq = np.array([pltq1, pltq2]).T
                com = ((m1*l1c*np.sin(state[:, 0]) + m2*(l1*np.sin(state[:, 0]) + l2c*np.sin(state[:, :2].sum(axis=1))))
                       / (m1 + m2))
                cop = np.ones([360,]) * np.nan
                cop[:len_t] = ((masked_ffc[:len_t, i, 4] + 0.08 * masked_ffc[:len_t, i, 0]) / -masked_ffc[:len_t, i, 2]).cpu().numpy()

                data = {
                    'bsp': humanData['bsp'],
                    'pltdd': pltdd,
                    'pltq': pltq,
                    'state': state.astype(np.float64).reshape(-1, 4),
                    'tq': torque.astype(np.float64),
                    'com': com.astype(np.float64),
                    'cop': cop.astype(np.float64),
                }
                io.savemat(f"{self.mat_path}/sub10i{self.trial_idx}.mat", data)

        self.drawn_plt_num += n_plots


    def after_run(self):
        super().after_run()
        print("\nAttempting to clean up Isaac Gym resources...")
        if hasattr(self.algo, 'env'):
            try:
                self.algo.env.close()
                print("Isaac Gym environment closed successfully via runner.vec_env.close().")
            except Exception as cleanup_e:
                print(f"!!!!!!!! Error during environment cleanup: {cleanup_e} !!!!!!!!")
                try:
                    if hasattr(self.algo, 'env') and hasattr(self.algo.env, 'gym') and hasattr(self.algo.env, 'sim') and 'cuda' in self.algo.env.device:
                        print("Attempting direct sim destruction...")
                        # Viewer가 있다면 먼저 종료
                        if hasattr(self.algo.env, 'viewer') and self.algo.env.viewer is not None:
                            self.algo.env.gym.destroy_viewer(self.algo.env.viewer)
                            print("Viewer destroyed.")
                        self.algo.env.gym.destroy_sim(self.algo.env.sim)
                        print("Simulation destroyed directly.")
                except Exception as direct_cleanup_e:
                    print(f"!!!!!!!! Error during direct cleanup: {direct_cleanup_e} !!!!!!!!")
        else:
            print("Runner object or vec_env not found, cleanup skipped.")

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

    # sets seed. if seed is -1 will pick a random one
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=global_rank)
    cfg_task_dict = omegaconf_to_dict(cfg.task)
    cfg.full_experiment_name = cfg.experiment_name
    if "IDP" in cfg.task.name:
        cfg_task_dict['env']['bsp_path'] = (MAIN_DIR / "demos" / cfg.task.env_type / cfg.task.subj / f"{cfg.task.subj}i1.mat")
        if cfg.task.env.ankle_limit == "satu":
            name_head = f"limSatuLevel{int(100*cfg.task.env.limLevel)}"
        elif cfg.task.env.ankle_limit == "hard":
            name_head = f"limHardLevel{int(100*cfg.task.env.limLevel)}"
        else:
            raise ValueError(f"Unknown ankle limit {cfg.task.env.ankle_limit}")
        name_head += f"_upright{cfg.task.env.curriculum.max_lean_angle}/atm{cfg.task.env.ankle_torque_max}_as{cfg.task.env.stiff_ank}"
        cfg.full_experiment_name = cfg.full_experiment_name + "/" + name_head
    cfg.full_experiment_name += f"/{run_name}"

    rlg_config_dict = omegaconf_to_dict(cfg.train)
    rlg_config_dict = preprocess_train_config(cfg, rlg_config_dict)
    rlg_config_dict['params']['config']['device_name'] = rlg_config_dict['params']['config']['device']

    if cfg.test:
        cfg_task_dict['env']['numEnvs'] = cfg.train.params.config.player.games_num
        cfg.task.name += "Det"
        fig_path = None if cfg.fig_path == "" else cfg.fig_path
        mat_path = None if cfg.mat_path == "" else cfg.mat_path
        observers = [PosturalControlObserver(fig_path=fig_path, mat_path=mat_path)]
        from common.rlgames.observer import MultiObserverPlayer
        multi_observer = MultiObserverPlayer(observers)
    else:
        observers = [RLGPUAlgoObserver(), CurriculumUpdator(), RunnerTrajectoryObserver()]

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

    # dump config dict
    if not cfg.test:
        experiment_dir = str(target_dir / 'runs' / cfg.full_experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
            f.write(OmegaConf.to_yaml(cfg))

    runner = build_runner(multi_observer)
    runner.load(rlg_config_dict)
    runner.reset()

    runner.run({
        'train': not cfg.test,
        'play': cfg.test,
        'checkpoint': cfg.checkpoint,
        'sigma': cfg.sigma if cfg.sigma != '' else None
    })


if __name__ == "__main__":
    launch_rlg_hydra()
