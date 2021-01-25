import gym, gym_envs, os, torch
from stable_baselines3.common.vec_env import DummyVecEnv
from common.callbacks import VFCustomCallback
from common.modules import NNCost
from common.wrappers import CostWrapper
from algo.torch.ppo import PPO
from mujoco_py import GlfwContext

if __name__ == "__main__":
    n_steps, n_episodes = 40, 10
    env_id = "IP_custom-v1"
    log_dir = os.path.join("..", "tmp", "log")
    env = gym.make(env_id, n_steps=n_steps)
    num_obs = env.observation_space.shape[0]
    num_act = env.action_space.shape[0]
    name = "2021-1-22-15-20-34"
    model_dir = os.path.join("..", "tmp", "log", name, "model")
    algo = PPO.load(model_dir + "/ppo50.zip")
    costfn = torch.load(model_dir + "/costfn50.pt").to(algo.device)
    env = DummyVecEnv([lambda: CostWrapper(env, costfn)])
    GlfwContext(offscreen=True)

    video_recorder = VFCustomCallback(log_dir+"/video/bar",
                                      gym.make(env_id, n_steps=n_steps),
                                      n_eval_episodes=5,
                                      render_freq=4096,
                                      costfn=costfn)
    for _ in range(2):
        video_recorder._set_costfn(costfn=costfn)
        env = DummyVecEnv([lambda: gym.make(env_id, n_steps=n_steps)])
        algo.set_env(env)
        algo.learn(total_timesteps=4096, callback=video_recorder, tb_log_name="test_callback")