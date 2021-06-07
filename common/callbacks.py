import os
import pickle
from copy import deepcopy
from typing import Any, Dict, Sequence

import numpy as np
import torch
from matplotlib import cm
from matplotlib import pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video, Figure


class VFCustomCallback(BaseCallback):
    def __init__(self,
                 eval_env,
                 render_freq: int,
                 n_eval_episodes: int = 1,
                 deterministic: bool = True,
                 costfn: torch.nn.Module = None,
                 draw_dim: Sequence[int] = (0, 1, 0)):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and logs it to TensorBoard

        :param eval_env: A gym environment from which the trajectory is recorded
        :param render_freq: Render the agent's trajectory every eval_freq call of the callback.
        :param n_eval_episodes: Number of episodes to render
        :param deterministic: Whether to use deterministic or stochastic policy
        :param draw_dim: First two arguments are for observation dimension that are input for drawing a figure
                         and third argument is for action dimension that is a output for figure.
        """
        super().__init__()
        self._eval_env = eval_env
        self._render_freq = render_freq
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic
        self.num = 0
        self.fps = int(1/eval_env.dt)
        self.costfn = costfn
        self.draw_dim = draw_dim

    def _on_step(self) -> bool:
        if self.n_calls % self._render_freq == 0:
            screens = []

            def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                """
                Renders the environment in its current state, recording the screen in the captured `screens` list

                :param _locals: A dictionary containing all local variables of the callback's scope
                :param _globals: A dictionary containing all global variables of the callback's scope
                """
                screen = self._eval_env.render(mode="rgb_array")
                # Mujoco uses HxWxC image convention, cv2 need HxWxC image convention
                screens.append(screen.transpose(2, 0, 1))

            evaluate_policy(
                self.model,
                self._eval_env,
                callback=grab_screens,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic,
            )
            with torch.no_grad():
                fig = self._draw_figure(self._eval_env, self.draw_dim)

            self.logger.record(
                "trajectory/video",
                Video(torch.ByteTensor([screens]), fps=self.fps),
                exclude=("stdout", "log", "json", "csv")
            )
            self.logger.record(
                "trajectory/cost",
                Figure(fig, close=True),
                exclude=("stdout", "log", "json", "csv")
            )
            plt.close()
        return True

    def _draw_figure(self, env, draw_dim) -> plt.figure:
        ndim, nact = env.observation_space.shape[0], env.action_space.shape[0]
        d1, d2 = np.meshgrid(np.linspace(-0.25, 0.25, 100), np.linspace(-0.25, 0.25, 100))
        pact = np.zeros((100, 100), dtype=np.float64)
        cost = np.zeros(d1.shape)
        for i in range(d1.shape[0]):
            for j in range(d1.shape[1]):
                iobs = np.zeros(ndim)
                iobs[draw_dim[0]], iobs[draw_dim[1]] = deepcopy(d1[i][j]), deepcopy(d2[i][j])
                iacts, _ = self.model.predict(np.array(iobs), deterministic=True)
                pact[i][j] = iacts[draw_dim[2]]
                inp = torch.from_numpy(np.append(iobs, iacts)).double().to(self.model.device).reshape(1, -1)
                cost[i][j] = self.costfn(inp).item()
        cost /= np.amax(cost)
        title_list = ["norm_cost", "abs_action"]
        yval_list = [cost, np.abs(pact)]
        xlabel, ylabel = "d1", "d2"
        max_list = [1.0, 0.5]
        min_list = [0.0, 0.0]
        fig = plt.figure()
        for i in range(2):
            ax = fig.add_subplot(1, 2, (i + 1))
            surf = ax.pcolor(d1, d2, yval_list[i], cmap=cm.coolwarm, shading='auto', vmax=max_list[i],
                             vmin=min_list[i])
            clb = fig.colorbar(surf, ax=ax)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            clb.ax.set_title(title_list[i])

        return fig

    def set_costfn(self, costfn: torch.nn.Module = None):
        self.costfn = costfn


class VideoCallback(BaseCallback):
    def __init__(self,
                 eval_env,
                 render_freq: int,
                 n_eval_episodes: int = 1,
                 deterministic: bool = True,
                 costfn: torch.nn.Module = None):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and logs it to TensorBoard

        :param eval_env: A gym environment from which the trajectory is recorded
        :param render_freq: Render the agent's trajectory every eval_freq call of the callback.
        :param n_eval_episodes: Number of episodes to render
        :param deterministic: Whether to use deterministic or stochastic policy
        """
        super().__init__()
        self._eval_env = eval_env
        self._render_freq = render_freq
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic
        self.num = 0
        self.fps = int(1/eval_env.dt)
        self.costfn = costfn

    def _on_step(self) -> bool:
        if self.n_calls % self._render_freq == 0:
            screens = []

            def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                """
                Renders the environment in its current state, recording the screen in the captured `screens` list

                :param _locals: A dictionary containing all local variables of the callback's scope
                :param _globals: A dictionary containing all global variables of the callback's scope
                """
                screen = self._eval_env.render(mode="rgb_array")
                # Mujoco uses HxWxC image convention, cv2 need HxWxC image convention
                screens.append(screen.transpose(2, 0, 1))

            evaluate_policy(
                self.model,
                self._eval_env,
                callback=grab_screens,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic,
            )
            self.logger.record(
                "trajectory/video",
                Video(torch.ByteTensor([screens]), fps=self.fps),
                exclude=("stdout", "log", "json", "csv")
            )
        return True


class SaveCallback:
    def __init__(self, cycle: int, dirpath: str):
        self.cycle = cycle
        os.makedirs(dirpath, exist_ok=True)
        self.path = dirpath

    def net_save(self, network, itr):
        if itr % self.cycle == 0:
            log_dir = self.path + f"/{itr:03d}"
            os.makedirs(log_dir, exist_ok=True)
            if network.agent.get_vec_normalize_env():
                network.wrap_env.save(log_dir + "/normalization.pkl")
            with open(log_dir + "/reward_net.pkl.tmp", "wb") as f:
                pickle.dump(network.reward_net, f)
            network.agent.save(log_dir + "/agent")
            os.replace(log_dir + "/reward_net.pkl.tmp", log_dir+"/reward_net.pkl")

    def rew_save(self, network, itr):
        if itr % self.cycle == 0:
            log_dir = self.path + f"{itr:03d}"
            with open(log_dir + "/reward_net.pkl.tmp", "wb") as f:
                pickle.dump(network.reward_net, f)
            os.replace(log_dir + "/reward_net.pkl.tmp", log_dir + "/reward_net.pkl")
            if network.agent.get_vecnormalize_env():
                network.wrap_env.save(log_dir + "/normalization.pkl")

    def agent_save(self, network, itr):
        if itr % self.cycle == 0:
            log_dir = self.path + f"/{itr:03d}"
            network.agent.save(log_dir + "/agent")
            if network.agent.get_vecnormalize_env():
                network.wrap_env.save(log_dir + "/normalization.pkl")
