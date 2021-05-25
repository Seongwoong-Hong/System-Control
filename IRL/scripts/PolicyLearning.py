import gym_envs
import os

from stable_baselines3.common import callbacks
from imitation.policies import serialize

from IRL.scripts.project_policies import def_policy
from common.callbacks import VideoCallback
from common.util import make_env, create_path
from mujoco_py import GlfwContext


if __name__ == "__main__":
    env_type = "IDP"
    algo_type = "sac"
    name = f"{env_type}/{algo_type}"
    env_id = f"{env_type}_custom-v1"
    n_steps = 600
    env = make_env(env_id, num_envs=8, n_steps=600, subpath="sub01")
    device = "cpu"
    proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    log_dir = os.path.join(proj_path, "tmp", "log", name, "test")
    model_dir = os.path.join(log_dir, "model")
    create_path(model_dir)
    GlfwContext(offscreen=True)
    algo = def_policy(algo_type, env, device=device, log_dir=log_dir)
    n = 1
    while os.path.isdir(log_dir + f"/extra_{n}"):
        n += 1
    create_path(log_dir + f"/policies_{n}")
    video_recorder = VideoCallback(gym_envs.make(env_id, n_steps=n_steps),
                                   n_eval_episodes=5,
                                   render_freq=62500)
    save_policy_callback = serialize.SavePolicyCallback(log_dir + f"/policies_{n}", None)
    save_policy_callback = callbacks.EveryNTimesteps(125000, save_policy_callback)
    callback_list = callbacks.CallbackList([video_recorder, save_policy_callback])
    algo.learn(total_timesteps=5e6, tb_log_name="extra", callback=callback_list)
    algo.save(model_dir + f"/policies_{n}/{algo_type}0")
    print(f"saved as policies_{n}/{algo_type}0")
