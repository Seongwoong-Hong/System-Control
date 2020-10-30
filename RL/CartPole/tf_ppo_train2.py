import os, cv2, warnings
import tensorflow as tf
from datetime import datetime
import numpy as np
import gym, gym_envs
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.policies import ActorCriticPolicy, RecurrentActorCriticPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines.common import make_vec_env, SetVerbosity, tf_util
from stable_baselines import PPO2

class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        action *= 0.125 * (self.action_space.high - self.action_space.low)
        return np.clip(action, self.action_space.low, self.action_space.high)

class SaveGifCallback(BaseCallback):
    """
    Callback for saving a model's gif every `save_freq` steps

    :param save_freq: (int)
    :param save_path: (str) Path to the folder where the model will be saved.
    :param fps: (int) frame per second
    :param verbose: (boolean)
    """
    def __init__(self, save_freq: int, save_path: str, fps: int, verbose=False):
        super(SaveGifCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.fps = fps
        self.fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        self.writer = None
        self.write_flag = False
        self.c_count = 0

    def _init_callback(self) -> None:
        self.n_sample = 5 * self.fps
        self.img_size = self.model.env.render(mode='rgb_array').shape[1::-1]
        assert (self.save_freq / self.model.n_envs) > self.n_sample, \
            "Try to save video too frequently. Increase save_freq. "
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % int(self.save_freq / self.model.n_envs) == 0:
            path = os.path.join(self.save_path, '{}_steps'.format(self.num_timesteps))
            self.writer = cv2.VideoWriter(path + '.avi',
                                          self.fourcc,
                                          self.fps,
                                          self.img_size)
            self.write_flag = True
            self.c_count = 0
            if self.verbose:
                print("Saving video of model to {}".format(path))

        if self.write_flag:
            rendered = self.model.env.render(mode='rgb_array')
            cv2.putText(rendered,
                        '#step: ' + str(self.num_timesteps),
                        (10, self.img_size[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 0), 2, cv2.LINE_AA)
            self.writer.write(rendered)
            if self.n_sample == self.c_count:
                self.writer.release()
                self.write_flag = False
            else:
                self.c_count += 1
        return True

def make_env(env_id, rank, Wrapper_class = None, seed=0, env_kwargs=None):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    env_kwargs = {} if env_kwargs is None else env_kwargs
    def _init():
        if isinstance(env_id, str):
            env = gym.make(env_id)
            if len(env_kwargs) > 0:
                warnings.warn("No environment class was passed (only an env ID) so `env_kwargs` will be ignored")
        else:
            env = env_id(**env_kwargs)
        env.seed(seed + rank)
        env.action_space.seed(seed + rank)
        if Wrapper_class is not None:
            env = Wrapper_class(env)
        return env
    return _init


if __name__=="__main__":
    name = "ppo_ctl_5"
    log_dir = "tmp/IP_ctl/" + name
    stats_dir = "tmp/IP_ctl/" + name + ".pkl"
    tensorboard_dir = os.path.join(os.path.dirname(__file__), "tmp", "log", "tf")
    env_name = "CartPoleCont-v0"
    env = SubprocVecEnv([make_env(env_name, i, NormalizedActions) for i in range(5)])
    policy_kwargs = dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])])

    # callback = SaveGifCallback(save_freq=5e+6, save_path=log_dir, fps=50)

    model = PPO2("MlpPolicy",
                 tensorboard_log=tensorboard_dir,
                 verbose=1,
                 noptepochs=10,
                 env=env,
                 gamma=1,
                 n_steps=16000,
                 lam=1,
                 policy_kwargs=policy_kwargs)

    model.learn(total_timesteps=8000000, tb_log_name=name)

    model.save(log_dir)
    stats_path = os.path.join(stats_dir)
    # env.save(stats_path)

    now = datetime.now()
    print("%s.%s.%s., %s:%s" %(now.year, now.month, now.day, now.hour, now.minute))