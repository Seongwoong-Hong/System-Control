import os
import pickle

from stable_baselines3.common import callbacks
from stable_baselines3.common.vec_env import VecNormalize
from imitation.policies import serialize

from common.callbacks import VideoCallback
from common.util import make_env
from common.wrappers import RewardWrapper
from RL.project_policies import def_policy

from mujoco_py import GlfwContext


if __name__ == "__main__":
    env_type = "HPC"
    algo_type = "ppo"
    name = "HPC_custom"
    device = "cpu"
    env_id = f"{name}-v1"
    env = make_env(env_id, use_vec_env=True, num_envs=8, n_steps=600, subpath="../IRL/demos/HPC/sub01/sub01")
    name += ""
    current_path = os.path.dirname(__file__)
    log_dir = os.path.join(current_path, env_type, "tmp", "log", name, algo_type)
    os.makedirs(log_dir, exist_ok=True)
    GlfwContext(offscreen=True)
    algo = def_policy(algo_type, env, device=device, log_dir=log_dir, verbose=1)
    n = 1
    while os.path.isdir(log_dir + f"/extra_{n}"):
        n += 1
    os.makedirs(log_dir + f"/policies_{n}", exist_ok=False)
    # video_recorder = VideoCallback(make_env(env_id, use_vec_env=False, n_steps=600),
    #                                n_eval_episodes=5,
    #                                render_freq=int(5e5))
    # save_policy_callback = serialize.SavePolicyCallback(log_dir + f"/policies_{n}", None)
    # save_policy_callback = callbacks.EveryNTimesteps(int(5e5), save_policy_callback)
    # callback_list = callbacks.CallbackList([video_recorder, save_policy_callback])
    algo.learn(total_timesteps=int(1e6), tb_log_name="extra")
    algo.save(log_dir+f"/policies_{n}/{algo_type}0")
    print(f"saved as policies_{n}/{algo_type}0")
