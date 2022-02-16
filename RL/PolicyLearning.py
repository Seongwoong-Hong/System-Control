import os
import shutil
from scipy import io

from stable_baselines3.common import callbacks
from imitation.policies import serialize

from algos.tabular.viter import SoftQiter
from common.util import make_env


if __name__ == "__main__":
    env_type = "DiscretizedHuman"
    algo_type = "softqiter"
    # env_op = 0.1
    name = f"{env_type}"
    device = "cuda:0"
    env_id = f"{name}-v2"
    # subpath = os.path.abspath(os.path.join("..", "IRL", "demos", env_type, "sub01", "sub01"))
    # env = make_env(env_id, num_envs=1, map_size=50)
    subj = "sub06"
    irl_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "IRL"))
    subpath = os.path.join(irl_dir, "demos", "HPC", subj, subj)
    bsp = io.loadmat(subpath + f"i1.mat")['bsp']
    env = make_env(env_id, num_envs=1, N=[17, 17, 17, 19], NT=[11, 11], bsp=bsp)
    # env = make_env(env_id, use_vec_env=False)
    name += f"_{subj}_17171719"
    current_path = os.path.dirname(__file__)
    log_dir = os.path.join(current_path, env_type, "tmp", "log", name, algo_type)
    os.makedirs(log_dir, exist_ok=True)
    algo = SoftQiter(env, gamma=0.995, alpha=0.001, device=device)
    n = 1
    while os.path.isfile(log_dir + f"/policies_{n}/agent.pkl"):
        n += 1
    os.makedirs(log_dir + f"/policies_{n}", exist_ok=False)
    shutil.copy(os.path.abspath(__file__), log_dir + f"/policies_{n}")
    shutil.copy(os.path.dirname(__file__) + "/project_policies.py", log_dir + f"/policies_{n}")
    # video_recorder = VideoCallback(make_env(env_id, subpath=subpath, wrapper=ActionWrapper),
    #                                n_eval_episodes=5,
    #                                render_freq=int(0.5e5))
    save_policy_callback = serialize.SavePolicyCallback(log_dir + f"/policies_{n}", None)
    save_policy_callback = callbacks.EveryNTimesteps(int(5e10), save_policy_callback)
    # callback_list = callbacks.CallbackList([video_recorder, save_policy_callback])
    # for i in range(20):
    algo.learn(total_timesteps=int(2000), tb_log_name="extra", callback=save_policy_callback, reset_num_timesteps=False)
    algo.save(log_dir + f"/policies_{n}/agent")
    if algo.get_vec_normalize_env():
        algo.env.save(log_dir + f"/policies_{n}/normalization.pkl")
    print(f"saved as policies_{n}/agent.pkl")
