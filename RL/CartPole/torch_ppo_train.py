import gym, gym_envs, os, shutil
import numpy as np
from datetime import datetime
from RL.algo.torch.ppo import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        action *= 0.125 * (self.action_space.high - self.action_space.low)
        return np.clip(action, self.action_space.low, self.action_space.high)

def make_env(env_id, rank, Wrapper_class = None, seed=0, high=0, low=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id, max_ep=1000, high=high, low=low)
        env.seed(seed + rank)
        if Wrapper_class is not None:
            env = Wrapper_class(env)
        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':
    name = "ppo_ctl_try1"
    log_dir = "tmp/IP_ctl/torch/" + name
    stats_dir = "tmp/IP_ctl/torch/" + name + ".pkl"
    tensorboard_dir = os.path.join(os.path.dirname(__file__), "tmp", "log", "torch")
    env_id = "CartPoleCont-v0"
    num_cpu = 10  # Number of processes to use
    high = 0.2  # Initial Limit
    low = 0.0
    seed = 0
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(env_id, i, NormalizedActions, seed=seed, high=high, low=low) for i in range(num_cpu)])
    policy_kwargs = dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])])

    model = PPO('MlpPolicy',
               tensorboard_log=tensorboard_dir,
               verbose=0,
               env=env,
               gamma=1,
               n_steps=6400,
               ent_coef=0.01,
               gae_lambda=1,
               device='cpu',
               policy_kwargs=policy_kwargs)

    for _ in range(10):
        test_env = NormalizedActions(gym.make(id=env_id, max_ep=1000, high=high, low=low))
        for i in range(10):
            if i > 2:
                shutil.rmtree(os.path.join(tensorboard_dir, name) + "_%.2f_%d" %(high, i-2))
            # model.learn(total_timesteps=3200000, tb_log_name=name+"_%.2f" %(high))
            test_env.reset()
            test_env.set_state(np.array([0, 0, 0, high]))
            obs = test_env.__getattr__('state')
            done = False
            step = 0
            while not done:
                act, _ = model.predict(obs, deterministic=True)
                obs, rew, done, info = test_env.step(act)
                step += 1
            print("theta: %.2f, x: %.2f" % (obs[1], obs[3]))
            if abs(obs[1]) < 0.05 and abs(obs[3]) < 0.05:
                fail = False
                break
            fail = True
        model.save(log_dir + "_%.2f.zip" %(high))
        if fail:
            print("Can't Learn the Current Curriculum. Last limit value is %.2f" %(high))
            break
        high += 0.10
        low += 0.10
        seed += num_cpu
        env = SubprocVecEnv([make_env(env_id, i, NormalizedActions, seed=seed, high=high, low=low) for i in range(num_cpu)])
        model.set_env(env)

    model.save(log_dir)
    now = datetime.now()
    print("%s.%s.%s., %s:%s" % (now.year, now.month, now.day, now.hour, now.minute))
    env.close()
    test_env.close()
    del model
