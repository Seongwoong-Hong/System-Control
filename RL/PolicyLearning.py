import os

from stable_baselines3.common import callbacks
from imitation.policies import serialize

from common.callbacks import VideoCallback
from common.util import create_path, make_env
from RL.project_policies import def_policy

from mujoco_py import GlfwContext


if __name__ == "__main__":
    env_type = "IP"
    algo_type = "sac"
    name = "IP_custom"
    device = "cpu"
    env_id = f"{name}-v2"
    env = make_env(env_id, use_vec_env=False, num_envs=8, n_steps=600)
    current_path = os.path.dirname(__file__)
    log_dir = os.path.join(current_path, env_type, "tmp", "log", name, algo_type)
    create_path(dirname=log_dir)
    GlfwContext(offscreen=True)
    algo = def_policy(algo_type, env, device=device, log_dir=log_dir, verbose=1)
    n = 1
    while os.path.isdir(log_dir + f"/extra_{n}"):
        n += 1
    create_path(log_dir + f"/policies_{n}")
    video_recorder = VideoCallback(make_env(env_id, use_vec_env=False, n_steps=600),
                                   n_eval_episodes=5,
                                   render_freq=int(5e5))
    save_policy_callback = serialize.SavePolicyCallback(log_dir + f"/policies_{n}", None)
    save_policy_callback = callbacks.EveryNTimesteps(int(5e5), save_policy_callback)
    callback_list = callbacks.CallbackList([video_recorder, save_policy_callback])
    algo.learn(total_timesteps=int(5e6), tb_log_name="extra", callback=callback_list)
    algo.save(log_dir+f"/policies_{n}/{algo_type}0")
    print(f"saved as policies_{n}/{algo_type}0")
