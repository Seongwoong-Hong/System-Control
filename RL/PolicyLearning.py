import os
from scipy import io

from stable_baselines3.common import callbacks
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from imitation.policies import serialize

from common.callbacks import VideoCallback
from common.util import make_env
from RL.project_policies import def_policy


if __name__ == "__main__":
    env_type = "HPC"
    algo_type = "sac"
    name = f"{env_type}_custom"
    device = "cuda:2"
    env_id = f"{name}-v1"
    subpath = os.path.abspath(os.path.join("..", "IRL", "demos", env_type, "sub01", "sub01"))
    env = make_env(env_id, use_vec_env=True, num_envs=1, subpath=subpath)
    name += ""
    current_path = os.path.dirname(__file__)
    log_dir = os.path.join(current_path, env_type, "tmp", "log", name, algo_type)
    os.makedirs(log_dir, exist_ok=True)
    algo = def_policy(algo_type, env, device=device, log_dir=log_dir, verbose=1)
    n = 1
    while os.path.isdir(log_dir + f"/extra_{n}"):
        n += 1
    os.makedirs(log_dir + f"/policies_{n}", exist_ok=False)
    video_recorder = VideoCallback(make_env(env_id, use_vec_env=False, subpath=subpath),
                                   n_eval_episodes=5,
                                   render_freq=int(0.5e5))
    save_policy_callback = serialize.SavePolicyCallback(log_dir + f"/policies_{n}", None)
    save_policy_callback = callbacks.EveryNTimesteps(int(0.5e5), save_policy_callback)
    callback_list = callbacks.CallbackList([video_recorder, save_policy_callback])
    algo.learn(total_timesteps=int(1.5e5), tb_log_name="extra")
    algo.save(log_dir+f"/policies_{n}/agent")
    if algo.get_vec_normalize_env():
        algo.env.save(log_dir+f"/policies_{n}/normalization.pkl")
    print(f"saved as policies_{n}/agent.zip")
