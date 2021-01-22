from typing import Any, Dict, Union
import numpy as np
import gym, cv2
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video, Figure
from matplotlib import pyplot as plt
from matplotlib import cm

class VFCustomCallback(BaseCallback):
    def __init__(self, path: str,
                 eval_env: gym.Env,
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
        self.path = path
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

            fig = self.draw_figure()

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

    def draw_figure(self) -> plt.figure:
        th, dth = np.meshgrid(np.linspace(-0.15, 0.15, 100), np.linspace(-0.15, 0.15, 100))
        pact = np.zeros((100, 100), dtype=np.float64)
        cost = np.zeros(th.shape)
        for i in range(th.shape[0]):
            for j in range(th.shape[1]):
                pact[i][j], _ = self.model.predict(np.append(th[i][j], dth[i][j]), deterministic=True)
                inp = torch.Tensor((th[i][j], dth[i][j], pact[i][j])).double()
                cost[i][j] = self.costfn(inp).item()

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        surf1 = ax1.plot_surface(th, dth, cost, cmap=cm.coolwarm)
        surf2 = ax2.plot_surface(th, dth, pact, cmap=cm.coolwarm)
        fig.colorbar(surf1, ax=ax1)
        fig.colorbar(surf2, ax=ax2)
        ax1.view_init(azim=0, elev=90)
        ax2.view_init(azim=0, elev=90)
        return fig

    def _set_costfn(self, costfn: torch.nn.Module = None):
        self.costfn = costfn