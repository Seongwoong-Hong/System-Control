import os, torch, gym, gym_envs
from common.modules import NNCost
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from algo.torch.ppo import PPO
from common.wrappers import CostWrapper

class LocalCW(CostWrapper):
    def reward(self, obs):
        cost_inp = torch.from_numpy(obs).to(self.costfn.device)
        return -1*self.costfn.forward(cost_inp)

if __name__ == "__main__":
    name = "2021-1-24-0-23-35"
    num = 4
    model_dir = os.path.join("tmp", "log", name, "model")
    costfn = torch.load(model_dir + "/costfn{}.pt".format(num))
    env_id = "IP_custom-v1"
    n_steps = 100
    device = "cuda:1"
    current_path = os.path.dirname(__file__)
    log_dir = os.path.join(current_path, "tmp", "log", name)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    env = DummyVecEnv([lambda: gym.make(env_id, n_steps=n_steps)])
    # nenv = VecNormalize(env, norm_obs=False, norm_reward=True)
    algo = PPO("MlpPolicy",
               env=env,
               n_steps=4096,
               batch_size=128,
               gamma=0.99,
               gae_lambda=0.95,
               ent_coef=0.01,
               verbose=1,
               device=device,
               tensorboard_log=log_dir)
    algo.learn(total_timesteps=409600, tb_log_name=name+"_extra")
    algo.save(model_dir+"/extra_ppo")