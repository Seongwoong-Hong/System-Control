import os, torch, gym, gym_envs
from common.modules import NNCost
from stable_baselines3.common.vec_env import DummyVecEnv
from algo.torch.ppo import PPO
from common.wrappers import CostWrapper

if __name__ == "__main__":
    model_dir = os.path.join("..", "tmp", "model")
    costfn = torch.load(model_dir + "/bar10_costfn.pt")
    env_id = "IP_custom-v2"
    n_steps = 100
    device = "cuda:1"
    name = "bar10_extra"
    current_path = os.path.dirname(__file__)
    log_dir = os.path.join(current_path, "tmp", "log", name)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    env = DummyVecEnv([lambda: CostWrapper(gym.make(env_id, n_steps=n_steps), costfn)])
    algo = PPO("MlpPolicy",
               env=env,
               n_steps=4096,
               batch_size=128,
               gamma=0.99,
               gae_lambda=0.95,
               ent_coef=0.01,
               verbose=0,
               device=device,
               tensorboard_log=log_dir)
    algo.learn(total_timesteps=1024000, tb_log_name=name)