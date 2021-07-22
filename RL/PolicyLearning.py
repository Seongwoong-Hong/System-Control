import os
from scipy import io

from stable_baselines3.common import callbacks
from stable_baselines3.common.vec_env import SubprocVecEnv
from imitation.policies import serialize

from common.callbacks import VideoCallback
from common.util import make_env
from RL.project_policies import def_policy


if __name__ == "__main__":
    env_type = "HPC"
    algo_type = "sac"
    name = "HPC_custom"
    device = "cpu"
    env_id = f"{name}-v1"
    pltqs = []
    for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        file = "../IRL/demos/HPC/sub01/sub01" + f"i{i + 1}.mat"
        pltqs += [io.loadmat(file)['pltq']]
    env = make_env(env_id, use_vec_env=False, num_envs=16, use_norm=False, pltqs=pltqs)
    name += ""
    current_path = os.path.dirname(__file__)
    log_dir = os.path.join(current_path, env_type, "tmp", "log", name, algo_type)
    os.makedirs(log_dir, exist_ok=True)
    algo = def_policy(algo_type, env, device=device, log_dir=log_dir, verbose=1)
    n = 1
    while os.path.isdir(log_dir + f"/extra_{n}"):
        n += 1
    os.makedirs(log_dir + f"/policies_{n}", exist_ok=False)
    video_recorder = VideoCallback(make_env(env_id, use_vec_env=False, subpath="../IRL/demos/HPC/sub01/sub01"),
                                   n_eval_episodes=5,
                                   render_freq=int(2.5e6))
    save_policy_callback = serialize.SavePolicyCallback(log_dir + f"/policies_{n}", None)
    save_policy_callback = callbacks.EveryNTimesteps(int(2.5e6), save_policy_callback)
    callback_list = callbacks.CallbackList([video_recorder, save_policy_callback])
    algo.learn(total_timesteps=int(0.25e7), tb_log_name="extra", callback=callback_list)
    algo.save(log_dir+f"/policies_{n}/{algo_type}0")
    if algo.get_vec_normalize_env():
        algo.env.save(log_dir+f"/policies_{n}/normalization.pkl")
    print(f"saved as policies_{n}/{algo_type}0")
