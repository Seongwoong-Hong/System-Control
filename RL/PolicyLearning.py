import os
import shutil
from scipy import io

from stable_baselines3.common import callbacks
from imitation.policies import serialize

from common.callbacks import VideoCallback
from common.util import make_env
from common.wrappers import ActionWrapper
from RL.project_policies import def_policy


if __name__ == "__main__":
    env_type = "DiscretizedDoublePendulum"
    algo_type = "softqiter"
    # env_op = 0.1
    name = f"{env_type}"
    device = "cpu"
    env_id = f"{name}-v2"
    # subpath = os.path.abspath(os.path.join("..", "IRL", "demos", env_type, "sub01", "sub01"))
    # env = make_env(env_id, use_vec_env=False, num_envs=1, subpath=subpath, wrapper=ActionWrapper, use_norm=False)
    subj = "sub07"
    expt = f"{subj}"
    proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "IRL"))
    subpath = os.path.join(proj_path, "demos", "HPC", f"{subj}_cropped", subj)
    init_states = []
    for i in range(5):
        for j in range(6):
            bsp = io.loadmat(subpath + f"i{i + 1}_{j}.mat")['bsp']
            init_states += [io.loadmat(subpath + f"i{i + 1}_{j}.mat")['state'][0, :4]]
    env = make_env(env_id, num_envs=1, N=[11, 17, 17, 17])
    # env = make_env(env_id, use_vec_env=False)
    name += f""
    current_path = os.path.dirname(__file__)
    log_dir = os.path.join(current_path, env_type, "tmp", "log", name, algo_type)
    os.makedirs(log_dir, exist_ok=True)
    algo = def_policy(algo_type, env, device=device, log_dir=log_dir, verbose=1)
    n = 1
    while os.path.isdir(log_dir + f"/extra_{n}"):
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
    import time

    start = time.time()
    algo.learn(total_timesteps=int(5e4), tb_log_name="extra", callback=save_policy_callback, reset_num_timesteps=False)
    end = time.time()
    print(f"time: {end - start}")
    algo.save(log_dir + f"/policies_{n}/agent")
    if algo.get_vec_normalize_env():
        algo.env.save(log_dir + f"/policies_{n}/normalization.pkl")
    print(f"saved as policies_{n}/agent.pkl")
