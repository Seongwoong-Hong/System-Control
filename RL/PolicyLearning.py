import os
import shutil

from stable_baselines3.common import callbacks
from imitation.policies import serialize

from common.callbacks import VideoCallback
from common.util import make_env
from common.wrappers import ActionWrapper
from RL.project_policies import def_policy


if __name__ == "__main__":
    env_type = "2DTarget"
    algo_type = "viter"
    map_size = 10
    name = f"{env_type}_disc"
    device = "cpu"
    env_id = f"{name}-v2"
    subpath = os.path.abspath(os.path.join("..", "IRL", "demos", env_type, "sub01", "sub01"))
    # env = make_env(env_id, use_vec_env=False, num_envs=1, subpath=subpath, wrapper=ActionWrapper, use_norm=False)
    env = make_env(env_id, use_vec_env=True, num_envs=1, map_size=map_size)
    # env = make_env(env_id, use_vec_env=False)
    name += f"_{map_size}"
    current_path = os.path.dirname(__file__)
    log_dir = os.path.join(current_path, env_type, "tmp", "log", name, algo_type)
    os.makedirs(log_dir, exist_ok=True)
    algo = def_policy(algo_type, env, device=device, log_dir=log_dir, verbose=1)
    n = 1
    while os.path.isdir(log_dir + f"/extra_{n}"):
        n += 1
    os.makedirs(log_dir + f"/policies_{n}", exist_ok=False)
    shutil.copy(os.path.abspath(__file__), log_dir + f"/policies_{n}")
    # video_recorder = VideoCallback(make_env(env_id, subpath=subpath, wrapper=ActionWrapper),
    #                                n_eval_episodes=5,
    #                                render_freq=int(0.5e5))
    save_policy_callback = serialize.SavePolicyCallback(log_dir + f"/policies_{n}", None)
    save_policy_callback = callbacks.EveryNTimesteps(int(5e10), save_policy_callback)
    # callback_list = callbacks.CallbackList([video_recorder, save_policy_callback])
    # for i in range(20):
    import time

    start = time.time()
    algo.learn(total_timesteps=int(5e4), tb_log_name="extra", callback=save_policy_callback, reset_num_timesteps=False)
    end = time.time()
    print(f"time: {end - start}")
    algo.save(log_dir + f"/policies_{n}/agent")
    if algo.get_vec_normalize_env():
        algo.env.save(log_dir + f"/policies_{n}/normalization.pkl")
    print(f"saved as policies_{n}/agent.pkl")
