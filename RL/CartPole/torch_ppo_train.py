import gym, gym_envs, os
import numpy as np
from datetime import datetime
from RL.algo.torch.ppo import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        action *= 0.125 * (self.action_space.high - self.action_space.low)
        return np.clip(action, self.action_space.low, self.action_space.high)

def make_env(env_id, rank, Wrapper_class = None, seed=0, limit=0.2):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id, max_ep=1000, limit=limit)
        env.seed(seed + rank)
        if Wrapper_class is not None:
            env = Wrapper_class(env)
        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':
    name = "ppo_ctl_try"
    log_dir = "tmp/IP_ctl/torch/" + name
    stats_dir = "tmp/IP_ctl/torch/" + name + ".pkl"
    tensorboard_dir = os.path.join(os.path.dirname(__file__), "tmp", "log", "torch")
    env_id = "CartPoleCont-v0"
    num_cpu = 10  # Number of processes to use
    limit = 0.2
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(env_id, i, NormalizedActions, limit=limit) for i in range(num_cpu)])
    policy_kwargs = dict(net_arch=[dict(pi=[256, 128], vf=[256, 128])])
    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you:
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0)
    model = PPO('MlpPolicy',
               tensorboard_log=tensorboard_dir,
               verbose=1,
               env=env,
               gamma=1,
               n_steps=6400,
               ent_coef=0.01,
               gae_lambda=1,
               device='cpu',
               policy_kwargs=policy_kwargs)
    
    # model = PPO.load(path="tmp/IP_ctl/ppo_ctl_try_p.zip", env=env, tensorboard_log=tensorboard_dir)
    model.learn(total_timesteps=3200000, tb_log_name=name+"_"+str(limit))
    prev_rew, curr_rew = -np.inf, 0
    for _ in range(10):
        test_env = NormalizedActions(gym.make(id=env_id, max_ep=1000, limit=limit))
        while(True):
            model.learn(total_timesteps=3200000, tb_log_name=name+"_"+str(limit))
            _ = test_env.reset()
            test_env.set_state(np.array([0, 0, 0, limit]))
            obs = test_env.__getattr__('state')
            done = False
            step = 0
            while not done:
                act, _ = model.predict(obs, deterministic=True)
                obs, rew, done, info = test_env.step(act)
                curr_rew += rew
                step += 1
            if curr_rew < prev_rew:
                model = PPO.load(load_path="tmp/IP_ctl/ppo_ctl_try_p.zip", env=env, tensorboard_log=tensorboard_dir)
                print("Previous model is better")
                curr_rew = 0
            else:
                model.save("tmp/IP_ctl/ppo_ctl_try_p.zip")
                prev_rew = curr_rew
                curr_rew = 0
            if step >= 1000:
                break
        limit += 0.2
        env = SubprocVecEnv([make_env(env_id, i, NormalizedActions, limit=limit) for i in range(num_cpu)])

    model.save(log_dir)
    now = datetime.now()
    print("%s.%s.%s., %s:%s" % (now.year, now.month, now.day, now.hour, now.minute))
    env.close()
    test_env.close()
    del model
