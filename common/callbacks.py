from typing import Any, Dict, Union

import gym, cv2
import torch as th
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video

class VideoRecorderCallback(BaseCallback):
    def __init__(self, path: str, eval_env: gym.Env, render_freq: int, n_eval_episodes: int = 1, deterministic: bool = True):
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
                Video(th.ByteTensor([screens]), fps=self.fps),
                exclude=("stdout", "log", "json", "csv")
            )
            # filename = self.path + "/video_" + str(self.num) + ".avi"

            # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            # width, height, _ = screens[0].shape
            # writer = cv2.VideoWriter(filename, fourcc, 1/self._eval_env.dt, (width, height))
            # for render in screens:
            #     img = cv2.cvtColor(render, cv2.COLOR_RGB2BGR)
            #     writer.write(img)
            # writer.release()

            # self.num += 1
        return True