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

def make_env(env_id, rank, Wrapper_class = None, seed=0):
    def _init():
        env = gym.make(env_id, max_ep=1000, limit=1.3)
        env.seed(seed + rank)
        if Wrapper_class is not None:
            env = Wrapper_class(env)
        env.seed(seed + rank)
        env.action_space.seed(seed + rank)
        return env
    return _init

if __name__ == "__main__":
    name = "ppo_ctl_try"
    log_dir = "tmp/IP_ctl/tf/" + name
    stats_dir = "tmp/IP_ctl/tf/" + name + ".pkl"
    tensorboard_dir = os.path.join(os.path.dirname(__file__), "tmp", "log", "tf")
    env_name = "CartPoleCont-v0"
    # s_env = gym.make(env_name, max_ep=1000)
    limit = 0.2
    env = SubprocVecEnv([make_env(env_name, i, NormalizedActions) for i in range(10)])
    test_env = NormalizedActions(gym.make(id=env_name, max_ep=1000, limit=limit))
    # Automatically normalize the input features and reward
    env = VecNormalize(env, norm_obs=False, norm_reward=False, clip_obs=10., clip_reward=10.,)

    policy_kwargs = dict(net_arch=[dict(pi=[256, 128], vf=[256, 128])])

    # callback = SaveGifCallback(save_freq=5e+6, save_path=log_dir, fps=50)
    # model = PPO2.load(load_path="tmp/IP_ctl/tf/ppo_ctl_Comp.zip", env=env, tensorboard_log=tensorboard_dir, n_steps=6400)
    model = PPO2("MlpPolicy",
                 tensorboard_log=tensorboard_dir,
                 verbose=1,
                 noptepochs=12,
                 env=env,
                 gamma=1,
                 n_steps=6400,
                 lam=1,
                 policy_kwargs=policy_kwargs)
    prev_cost, curr_cost = np.inf, 0
    while (True):
        model.learn(total_timesteps=1600000, tb_log_name=name+'_'+str(limit))
        _ = test_env.reset()
        test_env.set_state(np.array([0, 0, 0, limit]))
        obs = test_env.__getattr__('state')
        done = False
        step = 0
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, cost, done, info = test_env.step(act)
            curr_cost += cost
            step += 1
        if step >= 1000:
            limit += 0.1
            break
        if curr_cost > prev_cost:
            model = PPO2.load(load_path="tmp/IP_ctl/ppo_ctl_try_p.zip", env=env, tensorboard_log=tensorboard_dir, n_steps=6400)
            curr_cost = 0
        else:
            model.save("tmp/IP_ctl/ppo_ctl_try_p.zip")
            prev_cost = curr_cost
            curr_cost = 0

    model.save(log_dir)

    now = datetime.now()
    print("%s.%s.%s., %s:%s" %(now.year, now.month, now.day, now.hour, now.minute))
    env.close()
    del model